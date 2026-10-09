#!/usr/bin/env bash
set -euo pipefail
[[ -f /.dockerenv ]] || { echo "M0 experiments must run in Docker" >&2; exit 1; }
M0_ROOT=/ssd/test/m0
mkdir -p "$M0_ROOT/tmp" "$M0_ROOT/logs"
export TMPDIR="$M0_ROOT/tmp"
export PYTHONPATH="$M0_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
M0_ACTION=$1
shift
M0_LOG="$M0_ROOT/logs/${M0_ACTION}-$(date -u +%Y%m%dT%H%M%S)-$$.log"
exec > >(tee "$M0_LOG") 2>&1
echo "M0 Docker action: $M0_ACTION; log: $M0_LOG"
case "$M0_ACTION" in
    env-check)
        uname -a
        nvidia-smi
        nvcc --version
        cmake --version
        python3 --version
        command -v nsys || true
        free -h
        ;;
    setup)
        if ! python3 -c 'import numpy, pyarrow'; then
            python3 -m pip install --no-cache-dir --target "$M0_ROOT/python" numpy pyarrow
        fi
        if ! python3 -c 'import yaml'; then
            python3 -m pip install --no-cache-dir --target "$M0_ROOT/python" PyYAML
        fi
        ;;
    build)
        cmake -S /workspace/llama.cpp -B "$M0_ROOT/build" \
            -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release \
            -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
        cmake --build "$M0_ROOT/build" --target llama-moe-m0 -j 4
        ;;
    data)
        python3 scripts/moe-m0-data.py "$@"
        ;;
    run|audit|self-test)
        python3 scripts/moe-m0.py "$M0_ACTION" "$@"
        ;;
    fixture)
        PYTHONPATH="/workspace/llama.cpp/gguf-py:$PYTHONPATH" python3 tools/moe-m0/make-fixture.py "$@"
        ;;
    *) echo "Unknown M0 action: $M0_ACTION" >&2; exit 1 ;;
esac
