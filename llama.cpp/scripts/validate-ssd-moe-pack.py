#!/usr/bin/env python3
"""Run isolated GPU validation of packed experts using the existing M0 runner."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    if (root / "validation-plan.json").exists():
        raise RuntimeError("plan already exists")
    source = Path("/ssd/test/m0/bf16-accept-dev-20260929/freeze/sequences.json")
    requests, domains = [], set()
    for request in read(source)["requests"]:
        if request["domain"] not in domains:
            domains.add(request["domain"])
            request["continuation_tokens"] = request["continuation_tokens"][:16]
            requests.append(request)
    assert len(requests) == 4
    write(root / "input.json", {"requests": requests})
    diagnostic = dict(next(r for r in requests if r["domain"] == "code"))
    diagnostic["continuation_tokens"] = diagnostic["continuation_tokens"][:4]
    write(root / "diagnostic-input.json", {"requests": [diagnostic]})
    order = []
    rng = random.Random(20261008)
    for repetition in range(3):
        modes = ["packed", "zerocopy"]
        rng.shuffle(modes)
        order.extend(f"perf-{repetition}-{mode}" for mode in modes)
    src = Path(__file__).resolve().parents[1]
    sources = ["scripts/pack-ssd-moe.py", "scripts/validate-ssd-moe-pack.py",
               "tools/moe-m0/moe-m0.cpp", "ggml/src/ggml-backend.cpp",
               "ggml/include/ggml-backend.h", "ggml/src/ggml-cuda/ggml-cuda.cu",
               "ggml/src/ggml-moe-trace.h", "ggml/src/ggml-cuda/common.cuh",
               "ggml/src/ggml-cuda/mmf.cu", "ggml/src/ggml-cuda/mmf.cuh", "src/llama-model.cpp"]
    model = "/ssd/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf"
    st = Path(model).stat()
    write(root / "validation-plan.json", {
        "model": model, "source_size": st.st_size, "source_mtime_ns": st.st_mtime_ns,
        "pack": "/ssd/test/ssd-pack-20261008/weights/Qwen3-30B-A3B-BF16.experts.gguf",
        "source_sequences": str(source), "source_sequences_sha256": sha(source),
        "input_sha256": sha(root / "input.json"), "diagnostic_input_sha256": sha(root / "diagnostic-input.json"),
        "sources_sha256": {name: sha(src / name) for name in sources},
        "normal_logit_atol": 1e-3, "normal_logit_rtol": 1e-4,
        "cache_mib": 8192, "workers": 4, "batch": 32, "threads": 4,
        "performance_order": order, "performance_repetitions": 3,
        "correctness_modes": ["packed", "zerocopy"], "performance_modes": ["packed", "zerocopy"],
        "scope": "first request per development domain; full prompt; first 16 continuation tokens at most",
        "limitations": ["no equal total physical memory quota", "OS page cache not dropped",
                        "clocks not locked", "short workload, not full thesis acceptance",
                        "same process binary; new model per run, new cache per request"]})


def run(root, label):
    plan = read(root / "validation-plan.json")
    mode = label.split("-")[-1]
    if mode not in ("resident", "unpacked", "packed", "zerocopy"):
        raise ValueError(mode)
    diagnostic = label.startswith("diagnostic-")
    correctness = label.startswith("correctness-")
    smoke = label.startswith("smoke-")
    output = root / label
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    config = {key: plan[key] for key in ("model", "cache_mib", "workers", "batch", "threads")}
    config.update(output=str(output), input=str(root / ("diagnostic-input.json" if diagnostic or smoke else "input.json")),
                  baseline="resident" if mode == "resident" else "async", action="replay", gpu_layers=99,
                  gpu_experts=True, diagnostic=diagnostic, verify_weights=diagnostic,
                  save_logits=correctness or smoke, ssd_stats=True, pack=plan["pack"] if mode in ("packed", "zerocopy") else "",
                  zero_copy=mode == "zerocopy")
    env = dict(os.environ)
    env.pop("LLAMA_SSD_MOE_PACK", None)
    env["GGML_MOE_ZERO_COPY"] = "1" if mode == "zerocopy" else "0"
    if mode in ("packed", "zerocopy"):
        env["LLAMA_SSD_MOE_PACK"] = plan["pack"]
    src = Path(__file__).resolve().parents[1]
    for name, expected in plan["sources_sha256"].items():
        if sha(src / name) != expected:
            raise RuntimeError(f"source changed after plan freeze: {name}")
    binary = root / "build/bin/llama-moe-m0"
    config_path = root / f"{label}.json"
    write(config_path, config)
    started = time.time()
    with (root / f"{label}.log").open("x") as log:
        result = subprocess.run([str(binary), str(config_path)], env=env, stdout=log, stderr=subprocess.STDOUT)
    write(root / f"{label}.status.json", {"exit_code": result.returncode, "started": started,
          "finished": time.time(), "binary_sha256": sha(binary), "config_sha256": sha(config_path),
          "pack_environment": env.get("LLAMA_SSD_MOE_PACK", "")})
    if result.returncode:
        raise RuntimeError(f"{label} failed: {result.returncode}; see log")
    print(f"finished {label} in {time.time() - started:.1f}s", flush=True)


def compare(root):
    plan = read(root / "validation-plan.json")
    reference = read(root / "correctness-resident/metrics.json")
    results = []
    for mode in plan.get("correctness_modes", ["unpacked", "packed"]):
        metrics = read(root / f"correctness-{mode}/metrics.json")
        assert [m["id"] for m in metrics] == [m["id"] for m in reference]
        for expected, actual in zip(reference, metrics):
            name = actual["id"]
            a = np.fromfile(root / "correctness-resident" / name / "logits.f32", dtype=np.float32)
            b = np.fromfile(root / f"correctness-{mode}" / name / "logits.f32", dtype=np.float32)
            assert a.size == b.size == expected["generated_tokens"] * expected["n_vocab"]
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            close = bool(np.allclose(a, b, atol=plan["normal_logit_atol"], rtol=plan["normal_logit_rtol"]))
            tokens = expected["greedy_tokens"] == actual["greedy_tokens"]
            results.append(dict(mode=mode, request=name, values=int(a.size), finite=finite,
                                allclose=close, tokens_equal=tokens, max_abs_diff=float(np.max(np.abs(a - b)))))
    write(root / "correctness-audit.json", results)
    assert all(r["finite"] and r["allclose"] and r["tokens_equal"] for r in results), results
    return results


def audit_trace(root):
    plan = read(root / "validation-plan.json")
    diagnostic_mode = "zerocopy" if (root / "diagnostic-zerocopy").exists() else "packed"
    traces = list((root / f"diagnostic-{diagnostic_mode}").glob("*/trace.jsonl"))
    assert len(traces) == 1
    live, ready, copying, checked, copies, consumers = {}, set(), set(), set(), {}, Counter()
    groups, events = defaultdict(set), Counter()
    peak = resident = 0
    pattern = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight")
    previous_seq = 0
    with traces[0].open() as stream:
        for line in stream:
            e = json.loads(line)
            assert e["seq"] == previous_seq + 1
            previous_seq = e["seq"]
            event, slot = e["event"], e["slot"]
            events[event] += 1
            match = pattern.search(e["tensor"])
            layer, projection = (int(match[1]), match[2]) if match else (-1, "")
            key = (e["eval"], layer, projection, e["expert"])
            if event == "reserve":
                assert slot not in live and e["detail"] == "packed_record"
                live[slot] = (layer, e["expert"], e["bytes"])
                resident += e["bytes"]
                peak = max(peak, resident)
            elif event == "io_complete":
                assert slot in live and slot not in ready
                ready.add(slot)
            elif event in ("io_error", "cache_fallback"):
                raise AssertionError(e)
            elif event == "weight_check":
                assert e["detail"] == "pass" and slot in ready
                checked.add((key, slot))
            elif event in ("copy_submit", "zero_copy_bind"):
                assert slot in ready and live[slot][:2] == (layer, e["expert"])
                assert (key, slot) in checked and (key, slot) not in copying
                assert e["detail"].startswith("CUDA")
                if event == "copy_submit":
                    copying.add((key, slot))
                else:
                    copies[key] = slot
                groups[(e["eval"], layer, e["expert"])].add(slot)
            elif event == "copy_complete":
                copying.remove((key, slot))
            elif event == "copy_check":
                assert e["detail"] == "pass" and (key, slot) not in copying
                copies[key] = slot
            elif event == "consumer_begin":
                assert e["detail"].startswith("CUDA") and e["top_k"] == 8 and e["n_experts"] == 128
                ids = {expert for row in e["ids"] for expert in row}
                for expert in ids:
                    assert (e["eval"], layer, projection, expert) in copies
                consumers[(e["eval"], layer, projection)] += 1
            elif event == "evict":
                assert slot in live and slot in ready and not any(s == slot for _, s in copying)
                resident -= live.pop(slot)[2]
                ready.remove(slot)
    assert not live and not ready and not copying and resident == 0
    assert peak <= plan["cache_mib"] * 1024**2
    request = read(root / "diagnostic-input.json")["requests"][0]
    evals = math.ceil(len(request["prompt_tokens"]) / plan["batch"]) + len(request["continuation_tokens"]) - 1
    assert len(consumers) == evals * 48 * 3 and set(consumers.values()) == {1}
    assert events["consumer_begin"] == events["consumer_end"] == evals * 48 * 3
    assert events["reserve"] == events["io_submit"] == events["io_complete"] == events["evict"]
    if diagnostic_mode == "zerocopy":
        assert events["weight_check"] == events["zero_copy_bind"]
        assert events["copy_submit"] == events["copy_complete"] == events["copy_check"] == 0
    else:
        assert events["weight_check"] == events["copy_submit"] == events["copy_complete"] == events["copy_check"]
    assert groups and all(len(slots) == 1 for slots in groups.values())
    for evaluation, layer, expert in groups:
        assert all((evaluation, layer, p, expert) in copies for p in ("gate", "up", "down"))
    result = dict(passed=True, events=dict(events), peak_cache_bytes=peak, evaluations=evals,
                  shared_expert_groups=len(groups), trace_sha256=sha(traces[0]))
    write(root / "trace-audit.json", result)
    return result


def report(root):
    plan = read(root / "validation-plan.json")
    result = dict(correctness=compare(root), trace=audit_trace(root), performance={}, counters={})
    performance_modes = plan.get("performance_modes", ["unpacked", "packed"])
    for mode in performance_modes:
        rows = [read(root / f"perf-{r}-{mode}/metrics.json") for r in range(3)]
        assert all(len(row) == 4 for row in rows)
        result["performance"][mode] = {
            "mean_request_tpot_ms": statistics.mean(statistics.mean(m["decode_ms"]) for row in rows for m in row),
            "mean_request_ttft_ms": statistics.mean(m["ttft_ms"] for row in rows for m in row),
            "repetition_mean_tpot_ms": [statistics.mean(statistics.mean(m["decode_ms"]) for m in row) for row in rows],
            "per_request": {m["id"]: {"mean_tpot_ms": statistics.mean(statistics.mean(row[i]["decode_ms"]) for row in rows),
                              "mean_ttft_ms": statistics.mean(row[i]["ttft_ms"] for row in rows)} for i, m in enumerate(rows[0])}}
        for r in range(3):
            log = (root / f"perf-{r}-{mode}.log").read_text()
            counters = [dict(re.findall(r"(\w+)=([\d.]+)", line)) for line in log.splitlines()
                        if line.startswith("ssd_moe_cache: final_evals=")]
            assert len(counters) == 4
            assert all(int(c["fallbacks"]) == 0 for c in counters)
            if mode in ("packed", "zerocopy"):
                assert all(int(c["host_bounce_bytes"]) == 0 and int(c["packed_reads"]) == int(c["direct_reads"]) > 0 for c in counters)
            if mode == "zerocopy":
                assert all(float(c["backend_copy"]) == 0 and float(c["backend_copy_saved"]) > 0 and
                           int(c["zero_copy_binds"]) > 0 and int(c["zero_copy_fallbacks"]) == 0 for c in counters)
            result["counters"][f"{r}-{mode}"] = counters
    baseline, optimized = performance_modes
    b, o = (result["performance"][m]["mean_request_tpot_ms"] for m in (baseline, optimized))
    result["performance_baseline"] = baseline
    result["performance_optimized"] = optimized
    result["decode_speed_ratio"] = b / o
    result["limitations"] = plan["limitations"]
    write(root / "validation-report.json", result)
    print(json.dumps(result["performance"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run", "compare", "audit-trace", "performance", "report"])
    parser.add_argument("--root", type=Path, default=Path("/ssd/test/ssd-pack-20261008"))
    parser.add_argument("--label")
    args = parser.parse_args()
    if args.phase == "run":
        run(args.root, args.label)
    elif args.phase == "performance":
        for label in read(args.root / "validation-plan.json")["performance_order"]:
            run(args.root, label)
    else:
        {"prepare": prepare, "compare": compare, "audit-trace": audit_trace, "report": report}[args.phase](args.root)


if __name__ == "__main__":
    main()
