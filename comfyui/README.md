# Krea LoRA inference on a RunPod ComfyUI Pod (no standing storage cost)

Run the **full interactive ComfyUI** on a cloud GPU and use your trained
FLUX.1 Krea LoRA. Models are pulled from HuggingFace on startup by
[`provision_krea.sh`](provision_krea.sh), so you can **terminate the pod when
done and pay $0 while it's off** — just re-deploy next time (re-downloads ~18GB,
a few minutes).

> ⚠️ **Use a CURRENT ComfyUI (2025+).** FLUX.1 Krea (July 2025) ships as a
> "scaled fp8" checkpoint; an older ComfyUI (e.g. the Sept-2024 build in
> `ai-dock/comfyui:latest`) loads it without applying the scale tensors and
> produces **pure noise**. Pick a template/image whose ComfyUI is recent, or have
> the provisioning `git pull` ComfyUI to latest on startup.

## Cost model
- **Storage: $0** if you terminate the pod between sessions (no network volume).
- **GPU: only while running.** A 24GB card (RTX 4090) is plenty for fp8 Krea —
  roughly **$0.34–0.69/hr**. Stop/terminate when you're done.
- Optional: add a small (~25–40GB) network volume (~$2–3/mo) if you'd rather skip
  the ~18GB re-download each session. Not required.

## One-time: make sure the provisioning script URL is reachable
The script lives in this repo. Use its raw URL:

```
https://raw.githubusercontent.com/howmuchcomputer/fluxgym/feat/krea-runpod-training/comfyui/provision_krea.sh
```

(Once the PR is merged to `main`, switch `feat/krea-runpod-training` → `main`.)

## Deploy the pod (RunPod dashboard)
1. **Pods → Deploy**.
2. **GPU:** a 24GB card (RTX 4090 recommended; A40/L40 48GB also fine).
3. **Image / template:** the **ai-dock ComfyUI** image
   `ghcr.io/ai-dock/comfyui:latest` (or pick a "ComfyUI (ai-dock)" template from
   the RunPod hub). This image supports the `PROVISIONING_SCRIPT` hook.
4. **Environment variables:**
   - `WEB_ENABLE_AUTH` = `false`  ← **required**, or ai-dock redirects ComfyUI to a
     login portal and the URL errors. With it off, ComfyUI serves directly on 8188
     (still behind RunPod's proxy auth, so only your account can reach it).
   - `PROVISIONING_SCRIPT` = the raw URL above
   - `HF_TOKEN` = your HuggingFace token (for the private LoRA repo)
   - `LORA_REPO` = `dhurks/<your-lora>` (e.g. `dhurks/ohcpx-v2`)
   - `LORA_FILE` = `<your-lora>.safetensors` (e.g. `ohcpx-v2.safetensors`)
5. **Expose HTTP port `8188`** (ComfyUI).
6. **Deploy.** First boot pulls the image + runs provisioning (~10–15 min). Watch
   the pod logs for `[provision] done.`
7. Click **Connect → HTTP 8188** to open ComfyUI in your browser.

## Build the workflow (most reliable path)
Rather than import a hand-made graph, start from ComfyUI's built-in Krea template
(guaranteed to match these filenames), then add one LoRA node:

1. In ComfyUI: **Workflow → Browse Templates → Flux → "Flux Krea dev"** (loads the
   standard Krea text-to-image graph).
2. Confirm the loaders point at the downloaded files:
   - **Load Diffusion Model:** `flux1-krea-dev_fp8_scaled.safetensors`
   - **DualCLIPLoader:** `t5xxl_fp8_e4m3fn.safetensors` + `clip_l.safetensors`, type `flux`
   - **Load VAE:** `ae.safetensors`
3. **Add your LoRA:** right-click → Add Node → **loaders → LoraLoaderModelOnly**.
   Wire `Load Diffusion Model → LoraLoaderModelOnly → (model input of the sampler/guider)`.
   Set `lora_name` = your LoRA, `strength_model` ≈ **0.8–1.0**.
4. **Prompt:** include your **trigger word** (e.g. `ohcpx`) in the positive prompt.
   Keep `FluxGuidance` ≈ 3.5, resolution 1024×1024, ~20 steps.
5. **Queue Prompt** → generate.

## When you're done
**Terminate** the pod (not just stop) to avoid any storage charge. Next session,
deploy again with the same env vars — provisioning re-downloads everything.

> Note: the `fp8_scaled` Krea checkpoint is the *correct* choice for inference
> (it's ComfyUI-optimized). That's different from training, where the full bf16
> checkpoint is required.
