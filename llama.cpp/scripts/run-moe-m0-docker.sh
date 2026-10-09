#!/usr/bin/env bash
set -euo pipefail

CONTAINER=moe-llama-dev
if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/run-moe-m0-docker.sh {env-check|setup|build|data|self-test|run|audit|fixture} [args...]"
    exit 1
fi

[[ $(docker inspect --format '{{.State.Running}}' "$CONTAINER") == true ]] || {
    echo "Start the existing moe-llama-dev container first." >&2; exit 1;
}
exec docker exec --user "$(id -u):$(id -g)" -w /workspace/llama.cpp \
    -e PYTHONDONTWRITEBYTECODE=1 -e XDG_CACHE_HOME=/ssd/test/m0/cache \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace/llama.cpp \
    "$CONTAINER" bash /workspace/llama.cpp/scripts/moe-m0-container.sh "$@"
