"""
RunPod serverless worker for FluxGym training.

Receives a training job from the local FluxGym UI (via runpod_client.py),
recreates the dataset + scripts on the worker, runs Kohya sd-scripts on the
GPU while streaming logs back, and pushes the resulting LoRA to HuggingFace.

This is a generator handler: each yielded {"log": "..."} chunk is streamed to
the client through the endpoint's /stream interface.
"""
import os
import sys
import time
import base64
import io
import zipfile
import subprocess

# Repo root is the parent of this runpod/ directory (e.g. /app/fluxgym).
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, "sd-scripts"))

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import runpod
from slugify import slugify
from argparse import Namespace

from fluxgym_core import models, download, readme, resolve_path_without_quotes
from library import huggingface_util


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _extract_dataset(dataset_zip_b64, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    raw = base64.b64decode(dataset_zip_b64)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(dataset_dir)


def handler(job):
    inp = job.get("input", {}) or {}

    base_model = inp.get("base_model")
    lora_name = inp.get("lora_name")
    train_script = inp.get("train_script")
    train_config = inp.get("train_config")
    sample_prompts = inp.get("sample_prompts", "")
    local_root = inp.get("local_root")
    dataset_zip_b64 = inp.get("dataset_zip_b64")

    missing = [k for k in ("base_model", "lora_name", "train_script", "train_config",
                           "local_root", "dataset_zip_b64") if not inp.get(k)]
    if missing:
        yield {"error": f"Missing required input fields: {missing}"}
        return
    if base_model not in models:
        yield {"error": f"Unknown base_model '{base_model}'. Known: {list(models.keys())}"}
        return

    output_name = slugify(lora_name)

    # 1. Rewrite the client's absolute repo-root paths to the worker's root.
    #    train.sh / dataset.toml only contain host-specific paths under the repo
    #    root (model/clip/t5/ae/dataset_config/output_dir/image_dir).
    train_script = train_script.replace(local_root, REPO_DIR)
    train_config = train_config.replace(local_root, REPO_DIR)

    yield {"log": f"[worker] Rewrote paths {local_root} -> {REPO_DIR}"}

    # 2. Recreate the dataset under datasets/<slug> (matches dataset.toml image_dir).
    dataset_dir = resolve_path_without_quotes(f"datasets/{output_name}")
    _extract_dataset(dataset_zip_b64, dataset_dir)
    n_files = len(os.listdir(dataset_dir))
    yield {"log": f"[worker] Dataset extracted to {dataset_dir} ({n_files} files)"}

    # 3. Write the training scripts (same layout as the local start_training).
    output_dir = resolve_path_without_quotes(f"outputs/{output_name}")
    os.makedirs(output_dir, exist_ok=True)
    sh_path = os.path.join(output_dir, "train.sh")
    _write(sh_path, train_script)
    _write(os.path.join(output_dir, "dataset.toml"), train_config)
    _write(os.path.join(output_dir, "sample_prompts.txt"), sample_prompts)
    yield {"log": f"[worker] Wrote train.sh / dataset.toml / sample_prompts.txt"}

    # 4. Download base model + shared encoders (cached on the network volume).
    yield {"log": f"[worker] Ensuring models for '{base_model}' (cached on volume)..."}
    download(base_model)
    yield {"log": "[worker] Models ready."}

    # 5. Run training, streaming stdout line by line.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["LOG_LEVEL"] = "DEBUG"
    yield {"log": "[worker] Starting training..."}
    proc = subprocess.Popen(
        ["bash", sh_path],
        cwd=REPO_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(proc.stdout.readline, ""):
        yield {"log": line.rstrip("\n")}
    proc.stdout.close()
    rc = proc.wait()
    yield {"log": f"[worker] Training process exited with code {rc}"}
    if rc != 0:
        yield {"error": f"Training failed (exit code {rc})"}
        return

    # 6. Generate README and push the LoRA folder to HuggingFace (if configured).
    prompts = [p.strip() for p in sample_prompts.splitlines()
               if p.strip() and not p.strip().startswith("#")]
    config_class_tokens = None
    try:
        import toml as _toml
        config_class_tokens = _toml.loads(train_config)["datasets"][0]["subsets"][0]["class_tokens"]
    except Exception:
        pass
    md = readme(base_model, lora_name, config_class_tokens, prompts)
    _write(os.path.join(output_dir, "README.md"), md)

    # Sanity: confirm training actually produced a LoRA before we claim success.
    produced = [os.path.join(r, f) for r, _, fs in os.walk(output_dir)
                for f in fs if f.endswith(".safetensors")]
    yield {"log": f"[worker] Output dir: {sorted(os.listdir(output_dir))}"}
    if not produced:
        yield {"error": f"No .safetensors produced in {output_dir} - training did not save a LoRA."}
        return

    hf_repo = inp.get("hf_repo")
    hf_token = inp.get("hf_token") or os.environ.get("HF_TOKEN", "")
    # There is no network volume in this deployment, so HF is the ONLY sink for
    # the LoRA. If we can't deliver it, FAIL loudly (don't silently 'complete').
    if not (hf_repo and hf_token):
        yield {"error": "No hf_repo/hf_token provided and no persistent volume - "
                        "nowhere to deliver the LoRA. Set HF_REPO_OWNER + HF_TOKEN."}
        return

    # Robust upload, bypassing kohya's error-swallowing wrapper:
    #  - disable hf_transfer (reliable for downloads, flaky for uploads)
    #  - create repo, upload folder, then VERIFY the files actually landed
    #  - retry, and raise (-> job FAILED) if it can't be confirmed.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    private = inp.get("hf_visibility", "public") != "public"
    last_err = None
    for attempt in range(1, 4):
        try:
            yield {"log": f"[worker] Uploading to https://huggingface.co/{hf_repo} (attempt {attempt}/3)..."}
            api.create_repo(repo_id=hf_repo, repo_type="model", private=private, exist_ok=True)
            api.upload_folder(repo_id=hf_repo, repo_type="model", folder_path=output_dir)
            repo_files = api.list_repo_files(repo_id=hf_repo, repo_type="model")
            if any(f.endswith(".safetensors") for f in repo_files):
                yield {"log": f"[worker] Upload VERIFIED: {len(repo_files)} files at https://huggingface.co/{hf_repo}"}
                last_err = None
                break
            last_err = "upload returned but no .safetensors present in repo afterwards"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        yield {"log": f"[worker] upload attempt {attempt} problem: {last_err}"}
        time.sleep(5)
    if last_err:
        yield {"error": f"HF upload failed after 3 attempts: {last_err}"}
        return

    yield {"log": "[worker] Done."}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
