# SSD MoE offload V1 to V2.2

## Goal and scope

V1 validates the minimum SSD-offload data path for Qwen3 MoE on DGX01:

1. Keep routed-expert weights in the GGUF file instead of preloading all of them into accelerator memory.
2. Read only the experts selected by the router for each `MUL_MAT_ID` node.
3. Copy the selected expert slices to the compute device and keep the matrix operation on that device.
4. Preserve the default llama.cpp behavior unless `--ssd-moe-offload` is explicitly enabled.

V1 uses the Linux page cache as the host-side cache. It does not yet implement an application-managed expert cache, direct I/O, asynchronous prefetch, or CPU-assisted expert execution.

V2.0 to V2.2 replace the page-fault source path with an instrumented, bounded cache. The implementation is designed for DGX01's unified physical memory. It does not describe host and device allocations as separate physical memory tiers.

## Data path

```text
GGUF on SSD
    |
    | mmap; expert ranges are excluded from eager prefetch
    v
Host virtual address / Linux page cache
    |
    | router produces Top-K expert IDs
    | scheduler copies only the selected expert slices
    v
CUDA device temporary tensor
    |
    v
MUL_MAT_ID on the accelerator
```

On DGX01's unified-memory architecture, the host mapping is still the source of truth for V1. The implementation does not assume that the whole mapped expert tensor is resident: first access to an uncached page can fault data in from the SSD, while warm access can be served by the operating-system cache.

## Step 1: Add an explicit feature switch

The public model parameters and common command parameters contain `ssd_moe_offload`, which defaults to `false`.

- `include/llama.h` adds the public `llama_model_params` field.
- `common/common.h` adds the command parameter.
- `common/common.cpp` passes the value to the model loader and forces lazy mode on when SSD offload is enabled.
- `common/arg.cpp` adds `--ssd-moe-offload` and `LLAMA_ARG_SSD_MOE_OFFLOAD`.

This keeps V1 opt-in and avoids changing existing inference behavior.

## Step 2: Mark Qwen3 MoE routed experts for lazy reading

`src/models/qwen3moe.cpp` applies `TENSOR_READ_LAZY` to `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` when the feature is enabled. Attention tensors, router tensors, normalization tensors, embeddings, and other non-routed-expert tensors retain the normal placement policy.

The existing lazy loader then:

- allocates the routed-expert tensor in a host buffer;
- maps its storage directly from the GGUF file;
- excludes its file ranges from eager `WILLNEED` prefetch;
- applies random-access advice to those ranges;
- avoids locking the whole lazy tensor into memory.

## Step 3: Reuse the Top-K selective-copy path

No new compute kernel is required for V1. The ggml backend scheduler already special-cases host-backed `GGML_OP_MUL_MAT_ID` weights:

1. Read the router's expert-ID tensor.
2. Build the set of experts used by the current batch.
3. Merge adjacent expert IDs into contiguous ranges.
4. Copy only those ranges into the device-side input tensor.
5. Run `MUL_MAT_ID` on the selected backend.

Because the source is an mmap-backed host tensor, touching a cold expert range causes the operating system to read the corresponding GGUF pages from SSD.

## Step 4: Add validation and benchmark support

`src/llama-model.cpp` rejects unsupported model architectures when the switch is enabled. V1 currently supports only `LLM_ARCH_QWEN3MOE`. The loader also prints an enablement message and warns that `--check-tensors` reads every expert.

`tools/llama-bench/llama-bench.cpp` and its README accept and report `--ssd-moe-offload`, including CSV, JSON, and Markdown result fields. This makes baseline and V1 runs distinguishable in experiment records.

## Build and run

CPU-only compile check:

```sh
cmake -S . -B /tmp/moe-offload-build-make \
    -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
CCACHE_DISABLE=1 cmake --build /tmp/moe-offload-build-make \
    --target llama-cli llama-bench -j2
```

DGX01 inference example:

```sh
./build/bin/llama-cli \
    -m /ssd/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf \
    -ngl all \
    --ssd-moe-offload \
    -p "Hello"
```

Benchmark example:

```sh
./build/bin/llama-bench \
    -m /ssd/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf \
    -ngl 999 \
    --ssd-moe-offload \
    -p 512 -n 128 -r 5 -o jsonl
```

Device operation offload must remain enabled. If MoE operations are assigned to the CPU, the selective host-to-device copy path is not exercised.

## V2.0: telemetry

Each graph evaluation reports cumulative cache statistics when `--ssd-moe-stats` is enabled:

- `hit_rate`, `hits`, and `misses`: expert-slice cache accesses;
- `resident`: bytes owned by the bounded application cache;
- `disk`: physical bytes requested by `pread`, including direct-I/O alignment;
- `logical`: useful expert bytes filled on cache misses;
- `backend_copy`: bytes submitted from cache slots to the compute backend;
- `io`: cumulative wall time spent by I/O workers;
- `evictions`: LRU entries removed after the cache reached its limit;
- `direct_reads`: successful `O_DIRECT` reads;
- `fallbacks`: reads that fell back to the original mmap address.

For token-generation benchmarks with batch size one, one evaluated decode graph corresponds to one generated token. For prompt processing or batched decode, the counters are per graph evaluation rather than per token.

## V2.1: bounded unified-memory expert cache

The model loader registers the file path and exact GGUF offset of every routed gate, up, and down expert tensor. Each inference context creates its own cache, so different contexts and models do not share entries.

The cache key is `(tensor, expert_id)`. Entries are loaded lazily, retained across tokens, and evicted with LRU when the configured byte limit is reached. Cache storage is allocated only for entries that are used. An 8 GiB limit does not allocate 8 GiB at startup.

Before reusing an evicted slot, the scheduler synchronizes the target backend. This ensures an asynchronous CUDA copy cannot retain a pointer to released cache memory.

The implementation reads directly from the original GGUF. It does not require a second packed model file. On Linux it opens the model again with `O_DIRECT`; unaligned GGUF offsets are handled with aligned reads and a small per-read bounce buffer. If direct I/O is unavailable, the same path uses buffered `pread`. If both explicit paths fail, it falls back to the original mmap data for correctness and increments `fallbacks`.

### Current expert layout

V2.1 does not rewrite the GGUF and does not combine an expert's gate, up, and down weights. The three source tensors remain independent:

```text
blk.N.ffn_gate_exps -> cache key (gate tensor, expert ID)
blk.N.ffn_up_exps   -> cache key (up tensor, expert ID)
blk.N.ffn_down_exps -> cache key (down tensor, expert ID)
```

For the BF16 test model, one logical expert therefore uses three cache entries of about 3 MiB each, rather than one contiguous 9 MiB entry. A cold logical expert can issue up to three reads. The common router IDs ensure that all three entries refer to the same logical expert, but this is selection consistency, not storage merging.

A packed expert sidecar is a separate future change. Its intended record layout is `[gate][up][down]`, aligned to 4 KiB, with one index entry per `(layer, expert)`. It must not overwrite the original GGUF.

## V2.2: concurrent load and cache-slot submission

A fixed worker pool loads all cache misses in the current Top-K set concurrently. The scheduler waits only for the selected entries, then submits each ready cache slot directly to the existing `MUL_MAT_ID` backend input. Consecutive mmap ranges are no longer touched on the V2 cache-hit path.

The CUDA operation still uses its normal transient input tensor. This is compute workspace in the same DGX01 unified physical memory, not a persistent second expert cache. The current version deliberately does not change the ggml tensor ABI or CUDA kernel argument layout. A future zero-copy experiment may add compact expert-ID remapping or an expert pointer table, but that change must be benchmarked separately because it affects all CUDA `MUL_MAT_ID` kernels.

## V2 options

```text
--ssd-moe-offload
--ssd-moe-cache-mib N
--ssd-moe-cache-workers N
--ssd-moe-stats / --no-ssd-moe-stats
```

`llama-bench` uses `--ssd-moe-stats 0|1` because it has a separate argument parser.

Recommended first DGX01 run:

```sh
result_dir=/home/cuichengshang/moe_ssd/test/v2
mkdir -p "$result_dir"

./build/bin/llama-bench \
    -m /path/to/Qwen3-30B-A3B-BF16.gguf \
    -ngl 999 \
    --ssd-moe-offload \
    --ssd-moe-cache-mib 8192 \
    --ssd-moe-cache-workers 4 \
    --ssd-moe-stats 1 \
    -p 0 -n 1 -r 1 -o jsonl \
    > "$result_dir/ssd-v2-cache8192-worker4.jsonl" \
    2> "$result_dir/ssd-v2-cache8192-worker4.log"
```

Cache sweep:

```sh
result_dir=/home/cuichengshang/moe_ssd/test/v2
mkdir -p "$result_dir"

for cache in 4096 8192 16384; do
    ./build/bin/llama-bench \
        -m /path/to/Qwen3-30B-A3B-BF16.gguf \
        -ngl 999 --ssd-moe-offload \
        --ssd-moe-cache-mib "$cache" \
        --ssd-moe-cache-workers 4 \
        --ssd-moe-stats 1 \
        -p 0 -n 32 -r 1 -o jsonl \
        > "$result_dir/cache-${cache}.jsonl" \
        2> "$result_dir/cache-${cache}.log"
done
```

All benchmark results and stderr logs must be written under `/home/cuichengshang/moe_ssd/test`. Do not use the repository-local `test` directory or the old `results` directory for this experiment.

Worker sweep should use a fixed cache size and test 1, 2, 4, and 8 workers. Do not drop the Linux page cache between repetitions when evaluating V2 cache-hit behavior. Use a separately labeled cold run when measuring physical SSD behavior.

## Verification completed

- The working branch is `ssd_off`.
- `git diff --check` passes.
- `llama-cli` and `llama-bench` compile in a clean CPU-only build.
- `llama-bench` compiles in a clean CUDA 13.0 build.
- Both help pages expose the SSD offload, cache-size, worker-count, and statistics options.
- A one-token real-model smoke test with Qwen3-30B-A3B BF16 completes successfully in CPU mode.
- Benchmark JSON records `"ssd_moe_offload": true` and the effective lazy mode as `"on"`.

The current development sandbox does not expose an NVIDIA device, so the real GPU runtime benchmark must still be executed in the DGX01 host or GPU-enabled Docker environment before performance conclusions are drawn.

## V1 acceptance criteria

1. The model loads with `--ssd-moe-offload` without eagerly reading every routed expert.
2. Qwen3 MoE generates valid output with GPU layer offload enabled.
3. Benchmark output identifies SSD mode explicitly.
4. Peak accelerator memory is lower than the normal all-expert placement.
5. Cold and warm runs are recorded separately because the Linux page cache materially changes observed SSD traffic.

## V2 acceptance criteria

1. Startup reports 144 registered expert tensors for the 48-layer Qwen3 MoE model.
2. The cache source line reports `O_DIRECT` on the DGX01 model filesystem, or clearly reports buffered `pread` as a fallback.
3. A one-token cold run reports nonzero misses, disk bytes, logical bytes, and backend-copy bytes.
4. A multi-token run increases hits and reaches a measurable nonzero hit rate.
5. `resident` never exceeds the configured cache limit except for MiB display rounding.
6. Generated tokens match V1 for deterministic sampling within the model's normal numeric tolerance.

## Current limitations and next step

- Qwen3 MoE is the only enabled architecture in V1.
- The explicit I/O implementation uses a fixed worker pool and `pread`, not `io_uring`.
- Unaligned GGUF offsets require an aligned bounce buffer for `O_DIRECT`; a 4 KiB-aligned packed expert file can remove this copy in a later experiment.
- The compute backend still receives selected experts through its standard transient tensor.
- CPU-assisted expert computation is intentionally deferred.

The next version should first compare direct-I/O and buffered-pread results, then add cross-layer prediction and low-priority prefetch. CPU-assisted expert computation should be introduced only after single-expert CPU and GPU kernel costs have been measured on the shared-memory system.
