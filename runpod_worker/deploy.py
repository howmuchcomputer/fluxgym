"""
Provision the FluxGym RunPod serverless backend from .env.

Steps:
  1. Ensure a network volume exists (caches base models across jobs).
  2. Create a serverless template referencing your pushed Docker image.
  3. Create a serverless endpoint from that template, attaching the volume.
  4. Print the endpoint id to paste back into .env as RUNPOD_ENDPOINT_ID.

Prereqs:
  - Build & push the image first:  ./runpod_worker/build_and_push.sh
  - Fill in .env (copy from .env.example), at minimum RUNPOD_API_KEY + DOCKER_IMAGE.

Run:  python runpod_worker/deploy.py
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:
    pass

import runpod


def env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        sys.exit(f"Missing required env var: {name} (set it in .env)")
    return val


def ensure_network_volume(name, size_gb, data_center_id):
    """Return a network volume id, creating one if RUNPOD_NETWORK_VOLUME_ID unset."""
    existing = os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
    if existing:
        print(f"Using existing network volume: {existing}")
        return existing

    # The SDK (this version) lacks volume creation, so use the REST API v1.
    import requests
    resp = requests.post(
        "https://rest.runpod.io/v1/networkvolumes",
        headers={"Authorization": f"Bearer {runpod.api_key}",
                 "Content-Type": "application/json"},
        json={"name": name, "dataCenterId": data_center_id, "size": int(size_gb)},
        timeout=60,
    )
    if resp.status_code not in (200, 201):
        sys.exit(f"Failed to create network volume ({resp.status_code}): {resp.text}\n"
                 f"Create one in the RunPod dashboard (data center {data_center_id}) "
                 f"and set RUNPOD_NETWORK_VOLUME_ID in .env.")
    vol_id = resp.json().get("id")
    print(f"Created network volume: {vol_id} ({size_gb}GB in {data_center_id})")
    return vol_id


def main():
    runpod.api_key = env("RUNPOD_API_KEY", required=True)
    image = env("DOCKER_IMAGE", required=True)
    name = env("RUNPOD_NAME", "fluxgym-krea")
    # Must be a data center with network-volume (storage) support.
    data_center_id = env("RUNPOD_DATA_CENTER_ID", "US-KS-2")
    volume_size = env("RUNPOD_VOLUME_GB", "100")
    container_disk = int(env("RUNPOD_CONTAINER_DISK_GB", "30"))
    gpu_ids = env("RUNPOD_GPU_IDS", "AMPERE_48")  # 48GB Ampere pool (A40/A6000)
    workers_max = int(env("RUNPOD_WORKERS_MAX", "1"))
    idle_timeout = int(env("RUNPOD_IDLE_TIMEOUT", "60"))

    # 1. Network volume (mounted at /runpod-volume by RunPod).
    volume_id = ensure_network_volume(f"{name}-vol", volume_size, data_center_id)

    # 2. Serverless template.
    hf_token = env("HF_TOKEN", "")
    template = runpod.create_template(
        name=f"{name}-template",
        image_name=image,
        container_disk_in_gb=container_disk,
        is_serverless=True,
        env={"HF_TOKEN": hf_token, "HF_HUB_ENABLE_HF_TRANSFER": "1"},
    )
    template_id = template["id"] if isinstance(template, dict) else template
    print(f"Created template: {template_id}")

    # 3. Serverless endpoint (attaches the volume, scales to zero when idle).
    endpoint = runpod.create_endpoint(
        name=f"{name}-endpoint",
        template_id=template_id,
        gpu_ids=gpu_ids,
        network_volume_id=volume_id,
        workers_min=0,
        workers_max=workers_max,
        idle_timeout=idle_timeout,
    )
    endpoint_id = endpoint["id"] if isinstance(endpoint, dict) else endpoint

    print("\n=== Done ===")
    print(f"Endpoint id: {endpoint_id}")
    print("\nNext: add this to your .env so the local UI routes training here:")
    print(f"  RUNPOD_ENDPOINT_ID={endpoint_id}")
    print(f"  RUNPOD_NETWORK_VOLUME_ID={volume_id}")
    print("\nNOTE: training runs can be long. In the RunPod dashboard, raise the")
    print("endpoint's Execution Timeout to cover your longest run (e.g. several hours).")


if __name__ == "__main__":
    main()
