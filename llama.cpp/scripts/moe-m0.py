#!/usr/bin/env python3
"""Run and audit the M0 GPU/SSD path experiment. No system cache eviction."""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

ROOT = Path("/ssd/test/m0")
REPO = Path(__file__).resolve().parents[1]
BASELINES = ("resident", "mmap", "v2", "async")
MODEL = "/ssd/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def command_output(command):
    try:
        p = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        return {"returncode": p.returncode, "output": p.stdout}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def source_hash():
    # Include untracked M0 sources; HEAD alone does not identify this dirty worktree.
    paths = subprocess.check_output(["git", "-C", str(REPO), "ls-files", "-co", "--exclude-standard", "-z"]).decode().split("\0")
    result = hashlib.sha256()
    for name in sorted(set(paths) - {""}):
        path = REPO / name
        if path.is_file() and path.suffix in (".cpp", ".c", ".cu", ".h", ".hpp", ".py", ".txt", ".cmake"):
            result.update(name.encode())
            result.update(bytes.fromhex(sha(path)))
    return result.hexdigest()


def canonical(name):
    match = re.search(r"blk\.\d+\.ffn_(?:up|gate|down)_exps\.weight", name)
    return match.group(0) if match else name


def audit_events(events, baseline, layers, capacity, verify=True):
    errors, slots, live, copies, pending, consumers = [], {}, {}, {}, {}, {}
    counts, backends, evals = collections.Counter(), collections.Counter(), collections.Counter()
    device_spans, projections = {}, collections.defaultdict(set)
    device_ms = collections.Counter()
    resident = peak = read_bytes = 0
    previous_ns = previous_seq = -1
    began = ended = False
    for e in events:
        event, slot = e["event"], e["slot"]
        tensor = canonical(e["tensor"])
        key = (tensor, e["expert"])
        ek = (e["eval"], *key)
        ck = (e["eval"], tensor)
        counts[event] += 1
        def require(condition, message):
            if not condition:
                errors.append(f"seq={e['seq']} {event}: {message}")
        require(e["host_ns"] >= previous_ns and e["seq"] > previous_seq, "nonmonotonic trace")
        previous_ns, previous_seq = e["host_ns"], e["seq"]
        if event == "request_begin":
            require(not began, "duplicate request begin")
            began = True
        elif event == "request_end":
            ended = True
        elif event == "reserve":
            require(slot not in slots and key not in live, "duplicate live read/slot")
            slots[slot] = {"key": key, "state": "reserved", "bytes": e["bytes"], "copying": False, "checked": False}
            live[key] = slot
            resident += e["bytes"]
            peak = max(peak, resident)
            require(resident <= capacity, "cache payload capacity exceeded")
        elif event in ("enqueue", "io_submit", "io_complete"):
            expected = {"enqueue": "reserved", "io_submit": "queued", "io_complete": "reading"}[event]
            require(slot in slots and slots[slot]["state"] == expected, "missing/duplicate/out-of-order read")
            if slot in slots:
                slots[slot]["state"] = {"enqueue": "queued", "io_submit": "reading", "io_complete": "ready"}[event]
            if event == "io_complete":
                read_bytes += e["bytes"]
                require(e["detail"] != "mmap_fallback", "explicit I/O failed and fell back to mmap")
        elif event == "hit_or_join":
            require(live.get(key) == slot and slots.get(slot, {}).get("state") in ("queued", "reading", "ready"), "join is not a live task")
        elif event in ("weight_check", "copy_check"):
            require(e["detail"] == "pass", "weight/copy mismatch")
            if event == "weight_check" and slot in slots:
                slots[slot]["checked"] = e["detail"] == "pass"
        elif event == "copy_submit":
            require(live.get(key) == slot and slots.get(slot, {}).get("state") == "ready", "copy before data ready")
            require(ek not in pending, "duplicate pending copy")
            if slot in slots:
                slots[slot]["copying"] = True
                require(not verify or slots[slot]["checked"], "source weight was not checked")
            pending[ek] = slot
        elif event == "copy_complete":
            require(pending.get(ek) == slot, "copy completion without matching submit")
            pending.pop(ek, None)
            copies[ek] = slot
            if slot in slots:
                slots[slot]["copying"] = False
        elif event == "consumer_begin":
            require(ck not in consumers, "overlapping consumer")
            require(e["detail"].startswith("CUDA"), "expert operator is not on CUDA")
            require(bool(e.get("ids")), "missing actual router IDs")
            for selected in e.get("ids", []):
                require(len(selected) == e.get("top_k") and len(selected) == len(set(selected)) and
                        all(0 <= i < e.get("n_experts", 0) for i in selected), "invalid Top-K IDs")
                if baseline in ("v2", "async"):
                    for expert in selected:
                        require((e["eval"], tensor, expert) in copies, "consumer before selected weight copy completed")
            consumers[ck] = e["host_ns"]
            backends[e["detail"]] += 1
            evals[e["eval"]] += 1
            projections[e["eval"]].add(tensor)
        elif event == "consumer_end":
            require(ck in consumers, "consumer end without begin")
            consumers.pop(ck, None)
        elif event == "evict":
            entry = slots.get(slot, {})
            require(live.get(key) == slot and entry.get("state") == "ready" and not entry.get("copying"), "slot evicted during I/O/copy")
            resident -= entry.get("bytes", 0)
            live.pop(key, None)
            if entry:
                entry["state"] = "evicted"
        elif event == "cache_fallback":
            require(False, "bounded cache path fell back to mmap")
        elif event in ("cuda_copy_begin", "cuda_consumer_begin"):
            dk = (e["eval"], tensor, e.get("offset", 0), event)
            require(dk not in device_spans, "duplicate CUDA event span")
            device_spans[dk] = e["host_ns"]
        elif event in ("cuda_copy_end", "cuda_consumer_end"):
            dk = (e["eval"], tensor, e.get("offset", 0), event.replace("_end", "_begin"))
            require(dk in device_spans and e.get("device_elapsed_ms", -1) >= 0, "invalid CUDA completion event")
            device_spans.pop(dk, None)
            device_ms[event] += max(0, e.get("device_elapsed_ms", 0))
    if not began or not ended or not evals:
        errors.append("incomplete request or empty MoE trace")
    if pending or consumers:
        errors.append("unfinished copies/consumers")
    if any(n != 3 * layers for n in evals.values()):
        errors.append("not all layer/projection consumers were traced in every eval")
    expected_projections = {f"blk.{i}.ffn_{p}_exps.weight" for i in range(layers) for p in ("up", "gate", "down")}
    if any(p != expected_projections for p in projections.values()):
        errors.append("missing/duplicate layer or projection")
    device_ok = not device_spans and counts["cuda_consumer_end"] == counts["consumer_begin"] and bool(evals)
    if not device_ok:
        errors.append("CUDA completion events do not cover every expert consumer")
    if baseline in ("v2", "async") and counts["cuda_copy_end"] < counts["copy_complete"]:
        errors.append("CUDA events do not cover all expert copies")
    if baseline in ("v2", "async") and not counts["io_complete"]:
        errors.append("no explicit selected-expert reads")
    if verify and baseline in ("v2", "async") and counts["copy_check"] != counts["copy_complete"]:
        errors.append("not every copied source was read back and checked")
    return {"pass": not errors, "errors": errors[:100], "error_count": len(errors), "events": dict(counts),
            "backends": dict(backends), "cache_payload_peak_bytes": peak, "pread_completed_request_bytes": read_bytes,
            "physical_device_bytes": None, "device_timeline_verified": device_ok,
            "device_elapsed_ms_sum": dict(device_ms)}


def compare_logits(reference, candidate, atol, rtol):
    import numpy as np
    a_meta, b_meta = read(reference / "metrics.json"), read(candidate / "metrics.json")
    shape = (a_meta["generated_tokens"], a_meta["n_vocab"])
    expected = math.prod(shape) * 4
    if (a_meta["generated_tokens"], a_meta["n_vocab"]) != (b_meta["generated_tokens"], b_meta["n_vocab"]):
        return {"pass": False, "reason": "logits shape mismatch"}
    if any((d / "logits.f32").stat().st_size != expected for d in (reference, candidate)):
        return {"pass": False, "reason": "logits file size mismatch"}
    a = np.memmap(reference / "logits.f32", dtype=np.float32, mode="r", shape=shape)
    b = np.memmap(candidate / "logits.f32", dtype=np.float32, mode="r", shape=shape)
    maximum, wrong, top1, finite = 0.0, 0, 0, True
    for x, y in zip(a, b):
        finite &= bool(np.isfinite(x).all() and np.isfinite(y).all())
        if finite:
            maximum = max(maximum, float(np.abs(x - y).max()))
        wrong += int(np.count_nonzero(~np.isclose(x, y, atol=atol, rtol=rtol)))
        top1 += int(x.argmax() == y.argmax())
    return {"pass": finite and wrong == 0 and top1 == shape[0], "finite": finite,
            "max_abs_error": maximum if finite else None, "outside_tolerance": wrong,
            "top1_agree": top1, "steps": shape[0], "atol": atol, "rtol": rtol}


def audit(experiment, atol, rtol):
    results = {}
    frozen = read(experiment / "freeze" / "sequences.json")["requests"]
    for baseline in BASELINES:
        entries = []
        for request in frozen:
            directory = experiment / "diagnostic" / baseline / request["id"]
            try:
                metadata = read(directory / "metrics.json")
                with open(directory / "trace.jsonl") as stream:
                    checked = audit_events((json.loads(line) for line in stream), baseline, metadata["layers"],
                                           read(experiment / "diagnostic" / baseline / "config.json")["cache_mib"] * 1024**2)
                reference = experiment / "diagnostic" / "resident" / request["id"]
                checked["logits"] = compare_logits(reference, directory, atol, rtol)
                checked["pass"] &= checked["logits"]["pass"]
                checked["id"] = request["id"]
            except (OSError, ValueError, KeyError) as exc:
                checked = {"id": request["id"], "pass": False, "error": str(exc)}
            entries.append(checked)
        results[baseline] = entries
    host_pass = all(x["pass"] for rows in results.values() for x in rows)
    normal_results = {}
    for baseline in BASELINES:
        checked = []
        for request in frozen:
            directory = experiment / "correctness" / baseline / request["id"]
            reference = experiment / "correctness" / "resident" / request["id"]
            try:
                checked.append({"id": request["id"], **compare_logits(reference, directory, atol, rtol)})
            except (OSError, ValueError, KeyError) as exc:
                checked.append({"id": request["id"], "pass": False, "error": str(exc)})
        normal_results[baseline] = checked
    normal_pass = all(x["pass"] for rows in normal_results.values() for x in rows)
    report = {"host_path_and_logits_pass": host_pass, "m0_accepted": False,
              "normal_path_correctness_pass": normal_pass, "normal_path_results": normal_results,
              "pending": ["Validate on the real BF16 model; a random-weight fixture is only a functional test.",
                          "Account for backend workspaces, direct-I/O bounce buffers and total UMA memory, not only cache payload.",
                          "Diagnostic synchronization changes execution/fusion; measure its overhead separately."],
              "results": results}
    write(experiment / "audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "normal_path_results")}, indent=2))
    return 0 if host_pass and normal_pass else 2


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(fraction * len(values)) - 1)]


def launch(binary, config, config_path, log_path, timeout, profile=False):
    write(config_path, config)
    command = [str(binary), str(config_path)]
    if profile:
        command = ["nsys", "profile", "--trace=cuda,osrt", "--sample=none", "--cpuctxsw=none",
                   "--force-overwrite=false", "--output=" + str(log_path.with_suffix(".nsys")), *command]
    environment = os.environ.copy()
    environment.pop("GGML_OP_OFFLOAD_MIN_BATCH", None)
    environment["TMPDIR"] = str(config_path.parent / "tmp")
    Path(environment["TMPDIR"]).mkdir(exist_ok=True)
    started = time.time()
    before = command_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,temperature.gpu,power.draw", "--format=csv"])
    with open(log_path, "x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        write(log_path.with_suffix(".status.json"), {"command": command, "state": "running", "container_pid": process.pid,
                                                   "container_process_group": process.pid, "started_unix_s": started})
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    write(log_path.with_suffix(".status.json"), {"command": command, "returncode": code, "elapsed_s": time.time() - started,
                                              "gpu_before": before, "gpu_after": command_output(["nvidia-smi"]),
                                              "meminfo_after": Path("/proc/meminfo").read_text()})
    if code:
        raise RuntimeError(f"run failed ({code}); see {log_path}")


def run(args):
    experiment = ROOT / args.experiment
    binary, model = Path(args.binary).resolve(), Path(args.model).resolve()
    if not binary.is_file() or not model.is_file():
        raise ValueError("binary/model does not exist")
    parameters = {"cache_mib": args.cache_mib, "workers": args.workers, "batch": args.batch,
                  "threads": args.threads, "gpu_layers": args.gpu_layers, "gpu_experts": True,
                  "prompt_tokens": args.prompt_tokens, "n_predict": args.n_predict}
    if args.phase == "freeze":
        experiment.mkdir(parents=True, exist_ok=False)
        available = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)) * 1024
        minimum = model.stat().st_size + args.cache_mib * 1024**2 + 4 * 1024**3
        if available < minimum:
            write(experiment / "blocked.json", {"reason": "insufficient available UMA memory for full-resident reference",
                                               "available_bytes": available, "conservative_minimum_bytes": minimum})
            raise RuntimeError("not enough available UMA memory for resident reference; see blocked.json; no model loaded")
        data = read(args.dataset)
        if args.limit:
            data["requests"] = data["requests"][:args.limit]
        write(experiment / "input.json", data)
        print("Hashing model and executable; this reads the GGUF and can warm the OS page cache.", flush=True)
        manifest = {"model": str(model), "model_sha256": sha(model), "model_stat": [model.stat().st_size, model.stat().st_mtime_ns],
                    "binary": str(binary), "binary_sha256": sha(binary), "source_sha256": source_hash(),
                    "git": command_output(["git", "-C", str(REPO), "status", "--short", "--branch"]),
                    "head": command_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
                    "dataset": str(Path(args.dataset).resolve()), "dataset_sha256": sha(args.dataset),
                    "parameters": parameters, "seed": args.seed,
                    "platform": command_output(["uname", "-a"]), "os_page_cache": "uncontrolled; no global drop_caches",
                    "memory_budget": "cache payload only; total physical UMA budget not enforced"}
        write(experiment / "manifest.json", manifest)
        configuration = {**parameters, "model": str(model), "input": str(experiment / "input.json"),
                         "output": str(experiment / "freeze"), "baseline": "resident", "action": "freeze", "save_logits": False}
        launch(binary, configuration, experiment / "freeze-config.json", experiment / "freeze.log", args.timeout)
        manifest["frozen_sha256"] = sha(experiment / "freeze" / "sequences.json")
        write(experiment / "manifest.json", manifest)
        return 0
    manifest = read(experiment / "manifest.json")
    for name, actual, expected in (("binary", sha(binary), manifest["binary_sha256"]),
                                  ("source", source_hash(), manifest["source_sha256"]),
                                  ("frozen sequence", sha(experiment / "freeze" / "sequences.json"), manifest["frozen_sha256"])):
        if actual != expected:
            raise ValueError(name + " changed after freeze; create a new experiment")
    if str(model) != manifest["model"] or [model.stat().st_size, model.stat().st_mtime_ns] != manifest["model_stat"]:
        raise ValueError("model identity/stat changed after freeze")
    parameters = manifest["parameters"]
    directory = experiment / args.phase
    directory.mkdir(exist_ok=False)
    baselines = list(BASELINES)
    rng = random.Random(manifest["seed"])
    repetitions = args.repetitions if args.phase == "timing" else 1
    order, rows = [], []
    for repetition in range(repetitions):
        rng.shuffle(baselines)
        for baseline in baselines:
            name = f"{baseline}-r{repetition}" if args.phase == "timing" else baseline
            order.append(name)
            write(directory / "order.json", order)
            print(args.phase, name, flush=True)
            config = {**parameters, "model": str(model), "input": str(experiment / "freeze" / "sequences.json"),
                      "output": str(directory / name), "baseline": baseline,
                      "action": "generate" if args.phase == "generate" else "replay",
                      "diagnostic": args.phase == "diagnostic", "verify_weights": args.phase == "diagnostic",
                      "save_logits": args.phase in ("diagnostic", "correctness")}
            launch(binary, config, directory / (name + "-config.json"), directory / (name + ".log"),
                   args.timeout, profile=args.phase == "profile")
            for metrics in read(directory / name / "metrics.json"):
                latency = metrics["token_ms"]
                rows.append({"run": name, "id": metrics["id"], "baseline": baseline, "repetition": repetition,
                             "ttft_ms": metrics["ttft_ms"], "tpot_ms": sum(latency) / len(latency) if latency else None,
                             "decode_p50_ms": percentile(latency, .5), "decode_p95_ms": percentile(latency, .95),
                             "peak_process_rss_kib": metrics["peak_process_rss_kib"]})
            write(directory / "summary.json", rows)
    if args.phase in ("diagnostic", "correctness"):
        if (experiment / "diagnostic" / "summary.json").exists() and (experiment / "correctness" / "summary.json").exists():
            return audit(experiment, args.atol, args.rtol)
        print("Run both diagnostic and correctness phases before the combined audit.")
    if args.phase == "timing" and (experiment / "diagnostic" / "summary.json").exists():
        overhead = []
        for diagnostic in read(experiment / "diagnostic" / "summary.json"):
            paired = [r["tpot_ms"] for r in rows if r["baseline"] == diagnostic["baseline"] and r["id"] == diagnostic["id"] and r["tpot_ms"]]
            median = percentile(paired, .5)
            overhead.append({"baseline": diagnostic["baseline"], "id": diagnostic["id"],
                             "diagnostic_tpot_ms": diagnostic["tpot_ms"], "normal_median_tpot_ms": median,
                             "instrumentation_and_execution_mode_ratio": diagnostic["tpot_ms"] / median if median and diagnostic["tpot_ms"] else None})
        write(experiment / "diagnostic-overhead.json", overhead)
    return 0


def self_test():
    events = []
    def event(name, **kw):
        value = {"seq": len(events), "host_ns": len(events), "event": name, "slot": 1, "eval": 0,
                 "tensor": "blk.0.ffn_up_exps.weight", "expert": 0, "bytes": 16, "detail": ""}
        value.update(kw)
        events.append(value)
    event("request_begin")
    for projection in ("up", "gate", "down"):
        tensor = f"blk.0.ffn_{projection}_exps.weight"
        slot = {"up": 1, "gate": 2, "down": 3}[projection]
        for name in ("reserve", "enqueue", "io_submit", "io_complete", "weight_check", "copy_submit", "copy_complete", "copy_check"):
            if name == "copy_complete":
                event("cuda_copy_begin", tensor=tensor, slot=0, offset=0)
                event("cuda_copy_end", tensor=tensor, slot=0, offset=0, device_elapsed_ms=.1)
            event(name, tensor=tensor, slot=slot, detail="pass" if name.endswith("check") else "direct_pread")
        event("consumer_begin", tensor=tensor, slot=slot, detail="CUDA0", ids=[[0]], top_k=1, n_experts=8)
        event("cuda_consumer_begin", tensor=tensor, slot=0, offset=0)
        event("cuda_consumer_end", tensor=tensor, slot=0, offset=0, device_elapsed_ms=.1)
        event("consumer_end", tensor=tensor, slot=slot, detail="CUDA0")
    event("request_end")
    assert audit_events(events, "v2", 1, 48)["pass"]
    assert not audit_events(events, "v2", 1, 32)["pass"]
    assert not audit_events(events[:-1], "v2", 1, 48)["pass"]
    for removed in ("io_submit", "copy_complete", "weight_check", "copy_check"):
        assert not audit_events([e for e in events if e["event"] != removed], "v2", 1, 48)["pass"]
    wrong = [dict(e, detail="CPU") if e["event"] == "consumer_begin" else e for e in events]
    assert not audit_events(wrong, "v2", 1, 48)["pass"]
    spec = importlib.util.spec_from_file_location("m0_data", REPO / "scripts" / "moe-m0-data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = [{"id": str(i), "group": str(i // 2), "content_sha256": str(i)} for i in range(40)]
    dev, confirm = module.select(data, 42, 3, 12)
    assert not {r["group"] for r in dev} & {r["group"] for r in confirm}
    assert (dev, confirm) == module.select(list(reversed(data)), 42, 3, 12)
    print("PASS: trace ordering, missing events, capacity, backend, weight checks, grouped deterministic split")


def main():
    if not Path("/.dockerenv").exists():
        raise RuntimeError("M0 must run through scripts/run-moe-m0-docker.sh")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    check = sub.add_parser("audit")
    check.add_argument("--experiment", required=True)
    check.add_argument("--atol", type=float, default=1e-3)
    check.add_argument("--rtol", type=float, default=1e-4)
    execute = sub.add_parser("run")
    execute.add_argument("--experiment", required=True)
    execute.add_argument("--phase", choices=("freeze", "diagnostic", "correctness", "timing", "profile", "generate"), required=True)
    execute.add_argument("--binary", default="/ssd/test/m0/build/bin/llama-moe-m0")
    execute.add_argument("--model", default=MODEL)
    execute.add_argument("--dataset", default=str(ROOT / "datasets/public-v1/dev.json"))
    for option, default in (("cache-mib", 8192), ("workers", 4), ("batch", 32), ("threads", 4), ("gpu-layers", 99),
                            ("prompt-tokens", 256), ("n-predict", 128), ("repetitions", 3), ("seed", 20260928),
                            ("limit", 0), ("timeout", 3600)):
        execute.add_argument("--" + option, type=int, default=default)
    execute.add_argument("--atol", type=float, default=1e-3)
    execute.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return 0
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.experiment):
        parser.error("experiment name must contain only letters, digits, hyphens and underscores")
    if args.command == "audit":
        return audit(ROOT / args.experiment, args.atol, args.rtol)
    if args.limit < 0 or min(args.cache_mib, args.workers, args.batch, args.threads, args.prompt_tokens,
                             args.n_predict, args.repetitions, args.timeout) <= 0:
        parser.error("counts must be positive (limit may be zero)")
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print("M0 error:", exc, file=sys.stderr)
        sys.exit(1)
