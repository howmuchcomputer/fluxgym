#!/usr/bin/env bash
# Provisioning script for a RunPod ComfyUI Pod (works with the ai-dock/comfyui
# image's PROVISIONING_SCRIPT hook, or run manually in any ComfyUI pod).
#
# Downloads FLUX.1 Krea (fp8, the correct variant for INFERENCE) + text encoders
# + VAE + your trained LoRA from HuggingFace into ComfyUI's model folders. This
# means you can TERMINATE the pod when done (no persistent volume to pay for) and
# just re-deploy next time -- provisioning re-downloads everything (~18GB, a few
# minutes). For private LoRA repos, pass HF_TOKEN.
#
# Env vars (set on the pod):
#   HF_TOKEN   HuggingFace token (required for a private LoRA repo)
#   LORA_REPO  e.g. dhurks/ohcpx-v1            (your LoRA's HF repo)
#   LORA_FILE  e.g. ohcpx-v1.safetensors       (the LoRA filename in that repo)
set -eu

# Locate the ComfyUI install (ai-dock uses /workspace/ComfyUI).
COMFY=""
for c in "/workspace/runpod-slim/ComfyUI" "${WORKSPACE:-/workspace}/ComfyUI" \
         "/workspace/ComfyUI" "/opt/ComfyUI" "/ComfyUI"; do
  [ -d "$c" ] && COMFY="$c" && break
done
# last resort: locate any ComfyUI install
[ -z "$COMFY" ] && COMFY="$(dirname "$(find /workspace /opt / -maxdepth 4 -type d -name ComfyUI 2>/dev/null | head -1)/x")"
echo "[provision] ComfyUI dir: $COMFY"
[ -d "$COMFY" ] || { echo "[provision] ERROR: ComfyUI dir not found"; exit 1; }
# Note: download text encoders into BOTH text_encoders/ and clip/ so DualCLIPLoader
# finds them regardless of ComfyUI version (newer reads text_encoders, older clip).
mkdir -p "$COMFY/models/diffusion_models" "$COMFY/models/text_encoders" \
         "$COMFY/models/clip" "$COMFY/models/vae" "$COMFY/models/loras"

HF="https://huggingface.co"

dl() {  # dl <url> <dest> [bearer-token]
  local url="$1" dest="$2" token="${3:-}"
  if [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" -gt 1000000 ]; then
    echo "[provision] exists: $(basename "$dest")"; return 0
  fi
  echo "[provision] downloading $(basename "$dest") ..."
  # curl -L --location-trusted RE-SENDS the auth header across HF's CDN redirect
  # (wget drops it, which silently breaks private-repo downloads).
  # --retry handles transient HF failures (one of these silently failed before).
  local retry=(--retry 5 --retry-delay 5 --retry-connrefused -fL)
  if [ -n "$token" ]; then
    curl "${retry[@]}" --location-trusted -H "Authorization: Bearer ${token}" -o "$dest" "$url"
  else
    curl "${retry[@]}" -o "$dest" "$url"
  fi
}

# --- Base model (fp8 scaled: correct for ComfyUI inference) ---
dl "$HF/Comfy-Org/FLUX.1-Krea-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-krea-dev_fp8_scaled.safetensors" \
   "$COMFY/models/diffusion_models/flux1-krea-dev_fp8_scaled.safetensors"

# --- Text encoders -> text_encoders/, then mirror into clip/ ---
dl "$HF/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
   "$COMFY/models/text_encoders/clip_l.safetensors"
dl "$HF/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors" \
   "$COMFY/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors"
cp -n "$COMFY/models/text_encoders/"*.safetensors "$COMFY/models/clip/" 2>/dev/null || true

# --- VAE (FLUX ae). NOTE: black-forest-labs repos are now GATED (401 without a
# license-accepted token), so pull the byte-identical ae from the public
# cocktailpeanut/xulf-dev repo (same file FluxGym uses for training). ---
dl "$HF/cocktailpeanut/xulf-dev/resolve/main/ae.sft" \
   "$COMFY/models/vae/ae.safetensors"

# --- Your trained LoRA (private repo -> needs HF_TOKEN via --location-trusted) ---
LORA_REPO="${LORA_REPO:-dhurks/ohcpx-v2}"
LORA_FILE="${LORA_FILE:-ohcpx-v2.safetensors}"
if [ -n "${HF_TOKEN:-}" ]; then
  dl "$HF/${LORA_REPO}/resolve/main/${LORA_FILE}" \
     "$COMFY/models/loras/${LORA_FILE}" "${HF_TOKEN}" \
     || echo "[provision] WARNING: LoRA download failed ($LORA_REPO/$LORA_FILE)"
else
  echo "[provision] HF_TOKEN not set -> skipping private LoRA download ($LORA_REPO)"
fi

echo "[provision] done. diffusion_models/text_encoders/clip/vae/loras populated under $COMFY/models"
