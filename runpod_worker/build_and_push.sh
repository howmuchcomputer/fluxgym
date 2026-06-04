#!/usr/bin/env bash
# Build the FluxGym RunPod serverless image for linux/amd64 and push it to your
# registry. RunPod pulls the image from there when starting workers.
#
# Usage:
#   DOCKER_IMAGE=docker.io/<user>/fluxgym-runpod:latest ./runpod_worker/build_and_push.sh
# or set DOCKER_IMAGE in .env (this script will source it).
set -euo pipefail

cd "$(dirname "$0")/.."

# Load .env if present so DOCKER_IMAGE can live there.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

: "${DOCKER_IMAGE:?Set DOCKER_IMAGE (e.g. docker.io/<user>/fluxgym-runpod:latest) in .env or the environment}"

# Non-interactive login if a Docker access token is provided.
if [ -n "${DOCKER_API_KEY:-}" ] && [ -n "${DOCKER_USERNAME:-}" ]; then
    echo "Logging in to Docker Hub as ${DOCKER_USERNAME}..."
    echo "${DOCKER_API_KEY}" | docker login docker.io -u "${DOCKER_USERNAME}" --password-stdin
fi

echo "Building ${DOCKER_IMAGE} for linux/amd64..."
docker buildx build \
    --platform linux/amd64 \
    -f Dockerfile.runpod \
    -t "${DOCKER_IMAGE}" \
    --push \
    .

echo "Pushed ${DOCKER_IMAGE}"
echo "Set DOCKER_IMAGE=${DOCKER_IMAGE} in .env, then run: python runpod_worker/deploy.py"
