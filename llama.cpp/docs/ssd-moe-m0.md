# M0: GPU / SSD path validation

## Environment contract

Use the user's **running `moe-llama-dev` container** for all future experiments.
The launcher uses `docker exec`, never `docker run`, `docker start`, or an image build.
It runs as the host user's UID/GID to avoid root-owned result directories.

- Source: `/workspace/llama.cpp`, mounted from `/home/cuichengshang/moe_offload/llama.cpp`.
- All datasets, logs, binaries and results: `/ssd/test/m0`, mounted from `/home/cuichengshang/moe_ssd/test/m0`.
- Existing image: `moe-llama-dev:cuda12.9`. GB10 build target: `121` (CMake resolves this to `121a`).
- The old host-side build logs are retained for provenance, not used as Docker experimental results.
- No global cache dropping, service stopping, model rewriting, or GGUF expert packing.

The implementation branch is `ssd_off_m0`, created from `ssd_off` while preserving all previous uncommitted V1/V2 changes. No commit or push was performed.

## What M0 implements

| Baseline | Expert source | Demand handling |
| --- | --- | --- |
| `resident` | CUDA-backend weights loaded at model initialization | No explicit expert cache |
| `mmap` | Same lazy/file-backed weights as V2 | Copy the selected ranges from mmap; no explicit read cache |
| `v2` | Bounded host-accessible expert cache | Parallel worker reads; wait for all selected reads, then submit copies |
| `async` | Same cache, capacity, workers and LRU policy | Enqueue all selected misses; wait/copy each selected expert in order while other reads may still run |

The async baseline pipelines demand I/O and copy submission. It is **not** a predictive prefetcher, io_uring implementation, readiness-order queue, or CPU expert computation scheme. Pageable buffers and backend synchronization can limit the actual overlap; M0 does not assume a speedup.

GB10 uses unified physical memory. CUDA/CPU here describe access/backend placement, not two independent physical memory pools. Cache capacity means payload bytes only. Full-shape CUDA temporary tensors, KV state, model allocations, direct-I/O bounce buffers, diagnostic readback buffers and OS page cache must not be described as free memory or included in an invented small cache-only footprint.

`GGML_MOE_FORCE_GPU=1` is an explicit experiment policy for `MUL_MAT_ID` on supported GPU/IGPU backends. The tool applies it consistently to diagnostics and timing. It does not depend on `GGML_OP_OFFLOAD_MIN_BATCH=1`; the driver removes that variable. Actual consumer records must say CUDA or the audit fails. Outside the M0 tool, the new policy and diagnostics are off by default.

## Dataset protocol

| Domain | Repository / configuration / source split | Group exclusion |
| --- | --- | --- |
| QA | `rajpurkar/squad`, `plain_text`, validation | Wikipedia title |
| Code | `google-research-datasets/mbpp`, `sanitized`, test | Task ID |
| Math | `openai/gsm8k`, `main`, test | Question SHA-256 |
| Summarization | `abisee/cnn_dailymail`, `3.0.0`, test | Article ID |

Sources: [SQuAD](https://huggingface.co/datasets/rajpurkar/squad), [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail).

Default development split: 3 requests/domain, 12 total. Confirmation split: 12/domain, 48 total. A seed fixes hash-ranked selection with one request per source group; duplicate prompt content is excluded. Repository commits, downloaded file hashes, selected IDs, group IDs and prompt contents are saved. Mirror transport, when explicitly selected, is recorded rather than reported as an official-site download.

These are workload samples for a systems experiment, not full benchmark accuracy evaluations. Reference solutions are never put in the prompts, and generated Python is never executed. A known Qwen3 ChatML no-thinking template is frozen. Passage bodies are shortened to target about 256 model tokens while questions/instructions are retained. Short math/code requests are not padded. Actual lengths and rendered prompts are stored.

`freeze` uses the resident baseline and greedy natural generation, up to 128 output tokens or EOG. `replay` performs real model forward passes on the exact stored prompt and continuation token IDs; it does not simulate routing. Early EOG is recorded, not replaced by random tokens. Prefill uses identical batches (default 32). Each request creates a new context/cache. OS page-cache state is uncontrolled; hashing the model itself can warm it.

## Running in the existing container

From the host repository directory `/home/cuichengshang/moe_offload/llama.cpp`:

```bash
bash scripts/run-moe-m0-docker.sh env-check
bash scripts/run-moe-m0-docker.sh setup
bash scripts/run-moe-m0-docker.sh build
bash scripts/run-moe-m0-docker.sh self-test
bash scripts/run-moe-m0-docker.sh data --name public-v1
```

`setup` only supplies missing Python libraries under `/ssd/test/m0/python`. It does not create another container, virtual environment, or install into system Python. Dependencies are NumPy/PyArrow for Parquet and logits, plus PyYAML required by the existing GGUF package for the optional small fixture.

If the official site is unreachable, use an explicitly named new snapshot:

```bash
bash scripts/run-moe-m0-docker.sh data --name public-mirror-v1 --endpoint https://hf-mirror.com
```

Output directories are not overwritten. Choose a new name after a partial failed download/run; failed artifacts are retained. Use the dataset path that actually completed.

First run a small **real-model** experiment when enough shared UMA memory is available:

```bash
bash scripts/run-moe-m0-docker.sh run --experiment m0-smoke --phase freeze \
  --dataset /ssd/test/m0/datasets/public-v1/dev.json --limit 1 --n-predict 4
bash scripts/run-moe-m0-docker.sh run --experiment m0-smoke --phase diagnostic
bash scripts/run-moe-m0-docker.sh run --experiment m0-smoke --phase correctness
```

The freeze preflight refuses to load a model when available UMA memory is below model file size + configured cache + a 4 GiB workspace guard. This is a conservative safety check, not a full peak-memory guarantee. It writes `blocked.json` without stopping other workloads or changing system memory settings. Defaults use the existing Qwen3-30B-A3B BF16 GGUF under `/ssd/models/`.

For the 12-request development protocol:

```bash
bash scripts/run-moe-m0-docker.sh run --experiment m0-dev-8g --phase freeze \
  --dataset /ssd/test/m0/datasets/public-v1/dev.json --cache-mib 8192 --n-predict 128
bash scripts/run-moe-m0-docker.sh run --experiment m0-dev-8g --phase diagnostic
bash scripts/run-moe-m0-docker.sh run --experiment m0-dev-8g --phase correctness
bash scripts/run-moe-m0-docker.sh run --experiment m0-dev-8g --phase timing --repetitions 3
bash scripts/run-moe-m0-docker.sh run --experiment m0-dev-8g --phase generate
```

Parameters are frozen at the first phase. Later phases reuse them, and reject changed source, executable or token snapshots. A model full SHA-256 is recorded on freeze, with size/mtime checked later. Do not edit/rebuild between phases; create a new experiment after a change. The driver randomizes baseline order per repetition and writes that order. Main comparison is paired per request, not independent token samples. Each process has a timeout and records its container PID/process-group in `*.status.json` while running; failure/timeout logs remain.

To stop an identified run, inspect its status file and verify that PID inside `moe-llama-dev` before sending a signal to that exact process group. Do not stop the shared container or use broad `pkill` commands.

For a low-memory **functional fixture only**:

```bash
bash scripts/run-moe-m0-docker.sh fixture
bash scripts/run-moe-m0-docker.sh run --experiment fixture-smoke --phase freeze \
  --model /ssd/test/m0/fixture/random-qwen3moe.gguf \
  --dataset /ssd/test/m0/datasets/public-v1/dev.json \
  --limit 1 --n-predict 4 --cache-mib 1 --batch 8
bash scripts/run-moe-m0-docker.sh run --experiment fixture-smoke --phase diagnostic \
  --model /ssd/test/m0/fixture/random-qwen3moe.gguf
bash scripts/run-moe-m0-docker.sh run --experiment fixture-smoke --phase correctness \
  --model /ssd/test/m0/fixture/random-qwen3moe.gguf
```

The fixture copies only the real tokenizer metadata and generates small seeded random FP32 tensors. Its cache is deliberately smaller than all expert weights to exercise eviction. It cannot establish BF16 numerical correctness, model quality, realistic route locality, or actual-model performance.

## Trace and correctness checks

Each request gets a JSONL trace and metrics; diagnostic runs also write complete vocabulary logits per output step (`logits.f32`). Events contain request/eval/absolute input-token position, input batch length, tensor (encoding layer/projection), expert ID, unique cache allocation ID and host monotonic time.

Recorded events include reserve, enqueue, read submit/complete, hit/in-flight join, weight verification, copy submit/complete/readback verification, actual selected IDs per input token, consumer begin/end, eviction, and request boundaries. M0 has no predictor, so request metadata explicitly declares `no_prediction`; it does not invent prediction events.

CUDA event spans cover selected-expert copies and individual `MUL_MAT_ID` consumers in diagnostics. Host events correlate device spans by eval/tensor/offset. Device elapsed milliseconds are **not** timestamps on the host clock; never subtract one from the other. Event spans include any idle time inside their stream interval and are not asserted to be kernel-only execution time.

Diagnostic mode executes one graph node at a time and synchronizes, disables cross-node fusion, compares source bytes to mmap, and reads back selected copied bytes. These checks warm the OS page cache and strongly perturb performance. The normal timing mode omits trace/logit dumps and those diagnostic synchronizations. Diagnostic/timing differences are instrumentation-plus-execution-mode overhead, not merely JSON logging cost. Worker I/O duration sums are not exposed GPU wait.

The separate `correctness` phase captures logits on the normal, non-diagnostic execution path to test the code that timing actually uses. A combined audit requires both phases. The timing phase writes `diagnostic-overhead.json` using paired request/baseline medians across repetitions. TPOT uses intervals between output-token readiness, including greedy token selection; `decode_ms` separately records forward-only time. TTFT excludes model/context initialization.

Audit failures include CPU expert consumers, incomplete layer/projection coverage, invalid Top-K IDs, duplicate reads, unsafe eviction, copy/consumer before readiness, cache overflow/fallback, missing CUDA completion events, source/readback mismatch, non-finite logits and logit/top-1 mismatch. Logits default to `atol=1e-3`, `rtol=1e-4`; freeze any alternative thresholds before inspecting confirmation results.

`pread_completed_request_bytes` counts completed explicit read request lengths including alignment, not measured SSD physical traffic. Buffered reads can hit the page cache; failed direct attempts and block-device amplification are not inferred from this counter. `peak_process_rss_kib` is a process-lifetime high-water mark and is not total unified physical memory consumption, nor a per-request GPU-memory counter.

`audit.json` deliberately separates host/device path-and-logits checks from full M0 acceptance. Passing the random fixture does not accept the real model. Real-model validation, total memory accounting and the diagnostic/normal-path comparison remain necessary before making optimization claims. `profile` optionally runs Nsight Systems if installed in the existing container; it is not auto-installed.

## Changed implementation

- `tools/moe-m0/`: explicit JSON-configured model runner, frozen token replay, full-logit capture and optional tiny fixture.
- `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-moe-trace.h`, public trace declarations: bounded-cache lifecycle instrumentation and demand pipelining; short-read and source-lifetime fixes.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: opt-in CUDA event spans; no event creation when tracing is off.
- `src/llama-model.cpp`: opt-in mmap-only control, preserving the same lazy expert tensor placement.
- `scripts/moe-m0*.py`: frozen datasets, orchestration, provenance, request metrics and trace/logit audit.
- `scripts/run-moe-m0-docker.sh`, `scripts/moe-m0-container.sh`: reuse the exact running experiment container and output tree.

No CPU-assisted expert compute, predictive cache replacement, up/down/gate packing, or GGUF weight-layout rewriting is added by M0.

## Validation artifacts from this implementation

The successfully downloaded snapshot is `/ssd/test/m0/datasets/public-mirror-v4/`: 12 development requests and 48 confirmation requests across all four domains. Earlier incomplete download directories are retained; do not use them as datasets. An interrupted snapshot can be continued with the same `--name`, `--endpoint` and selection parameters plus `--resume`; saved metadata keeps the repository revision fixed.

The initial `fixture-smoke-v1` and `fixture-dev-v1` runs both passed diagnostic and normal-path correctness on the small random-weight fixture. The former also exercised three interleaved timing repetitions and natural generation. These names identify the exact source snapshot recorded in their manifests; later source changes must be tested under a new experiment name.

In the initial one-request smoke run, each baseline recorded 66 CUDA expert consumers. Each cached baseline performed 264 reads, 333 checked copies and 264 evictions (including cache teardown). Peak cache payload was 1,031,680 bytes under the 1 MiB cap. Diagnostic logits matched exactly; normal-path maximum absolute difference was about 0.000164, with all four greedy tokens matching. These are functional-test observations only, not real-model performance measurements.

The final delivered code was tested under `/ssd/test/m0/fixture-release/`:

- All 12 development requests passed all four diagnostic paths and normal-path logit checks.
- Maximum normal-path absolute logit difference: `0.00020296499133110046`; all greedy token sequences matched the resident baseline.
- Three randomized timing repetitions completed for each baseline (12 runs total).
- Separate natural generation completed for all 12 requests; mmap, V2 and async token sequences all matched resident.
- Timing logs contain no per-graph CUDA debug output. `diagnostic-overhead.json` records paired instrumentation/execution-mode overhead.
- `audit.json` still has `m0_accepted=false`: the tiny random-weight model is not the real BF16 acceptance experiment.

The real 48-layer BF16 model also completed a resident reference freeze with 64 prompt tokens and 4 generated tokens under `/ssd/test/m0/m0-real-preflight-v1/`. That was before the final quiet-log change and is not a completed four-baseline BF16 comparison. Do not mix it with the final executable; start a new experiment:

```bash
bash scripts/run-moe-m0-docker.sh run --experiment m0-bf16-v1 --phase freeze \
  --dataset /ssd/test/m0/datasets/public-mirror-v4/dev.json --limit 1 --n-predict 4 --prompt-tokens 64
bash scripts/run-moe-m0-docker.sh run --experiment m0-bf16-v1 --phase diagnostic
bash scripts/run-moe-m0-docker.sh run --experiment m0-bf16-v1 --phase correctness
```

After that passes, increase to the full development protocol before using the disjoint confirmation set. No real-model optimization speedup is claimed by the current validation artifacts.
