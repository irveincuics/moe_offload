#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -d /ssd ]]; then
    default_ssd_root=/ssd
else
    default_ssd_root=/home/cuichengshang/moe_ssd
fi
ssd_root=${SSD_ROOT:-${default_ssd_root}}
result_dir=${RESULT_DIR:-${ssd_root}/test/ssd_v2}
model_path=${MODEL_PATH:-${ssd_root}/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf}
bench_bin=${BENCH_BIN:-${repo_dir}/build/bin/llama-bench}
cache_mib=${CACHE_MIB:-8192}
cache_workers=${CACHE_WORKERS:-4}
n_gen=${N_GEN:-1}
repetitions=${REPETITIONS:-1}
run_name=${RUN_NAME:-ssd-v2-cache${cache_mib}-worker${cache_workers}-n${n_gen}}

mkdir -p "${result_dir}"

"${bench_bin}" \
    -m "${model_path}" \
    -ngl 999 \
    --ssd-moe-offload \
    --ssd-moe-cache-mib "${cache_mib}" \
    --ssd-moe-cache-workers "${cache_workers}" \
    --ssd-moe-stats 1 \
    -p 0 \
    -n "${n_gen}" \
    -r "${repetitions}" \
    -o jsonl \
    > "${result_dir}/${run_name}.jsonl" \
    2> "${result_dir}/${run_name}.log"

printf 'result: %s\n' "${result_dir}/${run_name}.jsonl"
printf 'log:    %s\n' "${result_dir}/${run_name}.log"
