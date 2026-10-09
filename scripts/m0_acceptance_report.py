#!/usr/bin/env python3
"""Summarize frozen M0 acceptance artifacts without changing runtime or thresholds."""
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

BASELINES = ("resident", "mmap", "v2", "async")
ROOT = Path("/ssd/test/m0")


def read(path, default=None):
    if not path.exists() and default is not None:
        return default
    return json.loads(path.read_text())


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def mean(values):
    return statistics.mean(values) if values else None


def report(experiment):
    root = ROOT / experiment
    manifest = read(root / "manifest.json")
    audit = read(root / "audit.json", {})
    sequences = read(root / "freeze/sequences.json")["requests"]
    ids = {r["id"] for r in sequences}
    domains = dict(collections.Counter(r["domain"] for r in sequences))
    parameters = manifest["parameters"]
    metrics = {m["id"]: m for m in read(root / "freeze/metrics.json")}
    checks = {}
    checks["real_bf16_model"] = manifest["model"].endswith("/Qwen3-30B-A3B-BF16.gguf") and all(m["layers"] == 48 for m in metrics.values())
    checks["development_workload"] = len(sequences) == len(ids) == 12 and domains == {d: 3 for d in ("qa", "code", "math", "summary")}
    checks["frozen_inputs_intact"] = sha(root / "freeze/sequences.json") == manifest["frozen_sha256"]
    checks["diagnostic_path_and_logits"] = audit.get("host_path_and_logits_pass", False)
    checks["normal_path_logits"] = audit.get("normal_path_correctness_pass", False)
    checks["full_eval_coverage"] = True
    diagnostics, memory_ledger = {}, {}
    for baseline in BASELINES:
        entries = audit.get("results", {}).get(baseline, [])
        if len(entries) != len(ids) or {e["id"] for e in entries} != ids:
            checks["full_eval_coverage"] = False
        totals = collections.Counter()
        max_error, peak_payload = None, 0
        for entry in entries:
            m = metrics[entry["id"]]
            expected = (math.ceil(m["prompt_tokens"] / parameters["batch"]) + m["generated_tokens"] - 1) * 48 * 3
            checks["full_eval_coverage"] &= entry.get("events", {}).get("consumer_begin") == expected
            checks["full_eval_coverage"] &= entry.get("events", {}).get("cuda_consumer_end") == expected
            totals.update(entry.get("events", {}))
            error = entry.get("logits", {}).get("max_abs_error")
            if error is not None:
                max_error = max(max_error or 0, error)
            peak_payload = max(peak_payload, entry.get("cache_payload_peak_bytes", 0))
        diagnostics[baseline] = {"request_count": len(entries), "passed_requests": sum(bool(e.get("pass")) for e in entries),
                                 "failures": [e for e in entries if not e.get("pass")],
                                 "events": dict(totals), "max_abs_logit_error": max_error,
                                 "peak_cache_payload_bytes": peak_payload,
                                 "pread_completed_request_bytes": sum(e.get("pread_completed_request_bytes", 0) for e in entries)}
        log_path = root / "diagnostic" / (baseline + ".log")
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        memory_ledger[baseline] = {
            "model_buffers_MiB": re.findall(r"load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB", log),
            "max_compute_buffer_MiB": {device: max([float(size) for found, size in re.findall(
                r"sched_reserve:\s+(\S+) compute buffer size =\s+([\d.]+) MiB", log) if found == device], default=0)
                for device in ("CUDA0", "CUDA_Host")},
            "max_KV_buffer_MiB": max([float(x) for x in re.findall(r"CUDA0 KV buffer size =\s+([\d.]+) MiB", log)], default=0),
            "direct_io_bounce_bound_bytes": parameters["workers"] * (3 * 1024**2 + 8192) if baseline in ("v2", "async") else 0,
            "bounce_bound_assumption": "BF16 expert projection 3 MiB, <=512-byte copy padding, 4096-byte alignment; excludes allocator metadata",
        }
    timing = read(root / "timing/summary.json", [])
    expected_runs = {(b, repetition, request) for b in BASELINES for repetition in range(3) for request in ids}
    actual_runs = {(r["baseline"], r["repetition"], r["id"]) for r in timing}
    checks["three_timing_repetitions"] = actual_runs == expected_runs and len(timing) == len(expected_runs)
    checks["diagnostic_overhead_measured"] = len(read(root / "diagnostic-overhead.json", [])) == 4 * len(ids)
    reference = {m["id"]: m for m in read(root / "generate/resident/metrics.json", [])}
    generation = {}
    for baseline in BASELINES:
        rows = read(root / "generate" / baseline / "metrics.json", [])
        same = len(rows) == len(ids) and {r["id"] for r in rows} == ids and set(reference) == ids
        differences = []
        for row in rows:
            equal = row["greedy_tokens"] == reference.get(row["id"], {}).get("greedy_tokens")
            same &= equal
            if not equal:
                differences.append(row["id"])
        generation[baseline] = {"tokens_match": same, "different_requests": differences}
    checks["natural_generation_matches"] = all(r["tokens_match"] for r in generation.values())
    performance = {}
    for baseline in BASELINES:
        per_request = []
        for request in sorted(ids):
            paired = [r for r in timing if r["baseline"] == baseline and r["id"] == request]
            per_request.append({"id": request, "ttft_ms": mean([r["ttft_ms"] for r in paired]),
                                "tpot_ms": mean([r["tpot_ms"] for r in paired if r["tpot_ms"] is not None])})
        performance[baseline] = {"mean_request_ttft_ms": mean([r["ttft_ms"] for r in per_request if r["ttft_ms"] is not None]),
                                 "mean_request_tpot_ms": mean([r["tpot_ms"] for r in per_request if r["tpot_ms"] is not None]),
                                 "requests": per_request}
    resources = read(root / "resource-observation-summary.json", {})
    sampled = collections.defaultdict(lambda: {"peak_rss_kib": 0, "last_sampled_read_bytes": 0})
    gpu_samples = []
    resource_file = root / "resource-observation.jsonl"
    if resource_file.exists():
        for line in resource_file.open():
            sample = json.loads(line)
            if sample.get("gpu_csv"):
                gpu_samples.append(sample["gpu_csv"])
            for process in sample["processes"]:
                if str(root) not in process["command"]:
                    continue
                value = sampled[str(process["pid"])]
                value["command"] = process["command"]
                value["peak_rss_kib"] = max(value["peak_rss_kib"], process["memory_kib"].get("VmRSS") or 0)
                value["last_sampled_read_bytes"] = max(value["last_sampled_read_bytes"], process["io"].get("read_bytes", 0))
    checks["resource_observation_present"] = bool(resources and sampled and gpu_samples)
    statuses = {str(p.relative_to(root)): read(p) for p in sorted(root.glob("**/*.status.json"))}
    failed_processes = {name: {key: value.get(key) for key in ("command", "returncode", "elapsed_s")}
                        for name, value in statuses.items() if value.get("returncode") not in (None, 0)}
    checks["no_failed_processes"] = not failed_processes
    unstarted_phases = [phase for phase in ("correctness", "timing", "generate") if not (root / phase).exists()]
    partial_audit = read(root / "partial-audit.json", {})
    limitations = [
        "No equal total-physical-memory quota: the 8 GiB setting bounds cache payload, not model/workspaces/page cache together.",
        "Diagnostic mode synchronizes each node and disables fusion; measured overhead is not pure logging overhead.",
        "No directly measured exposed GPU I/O wait; worker I/O sums and CUDA spans must not be relabeled as that metric.",
        "Shared partition counters include other workloads; process sampling can miss final counter values and peaks.",
        "The OS page cache was not dropped, and GPU clocks/power were observed but not pinned.",
        "This acceptance covers the 12-request development workload at 8 GiB; 48 confirmation requests, 16 GiB and long-context stress are not covered.",
        "audit.json has a hard-coded m0_accepted=false placeholder; this report uses the actual checks, not that placeholder.",
    ]
    result = {"experiment": experiment, "m0_core_accepted": all(checks.values()), "full_research_protocol_accepted": False,
              "checks": checks, "limitations": limitations, "parameters": parameters, "domains": domains,
              "prompt_token_lengths": [metrics[r["id"]]["prompt_tokens"] for r in sequences],
              "continuation_token_lengths": [metrics[r["id"]]["generated_tokens"] for r in sequences],
              "total_generated_tokens": sum(m["generated_tokens"] for m in metrics.values()),
              "diagnostics": diagnostics, "generation": generation, "performance": performance,
              "memory_ledger": memory_ledger, "resource_observation": resources,
              "sampled_processes": dict(sampled), "gpu_sample_count": len(gpu_samples),
              "failed_processes": failed_processes, "unstarted_phases": unstarted_phases,
              "supplemental_partial_audit": partial_audit,
              "source_sha256": manifest["source_sha256"], "model_sha256": manifest["model_sha256"]}
    (root / "acceptance-report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = ["# M0 正式 BF16 验收报告", "", f"实验：`{experiment}`。", "",
             "结论：" + ("M0 核心运行与正确性验收通过；完整研究协议尚有缺项。" if result["m0_core_accepted"] else "M0 验收未通过或尚未完成，见下列检查项。"), "",
             "本次不修改运行时代码、不调整数值误差阈值，不覆盖旧实验结果。", "",
             "| 检查项 | 结果 |", "| --- | --- |"]
    lines += [f"| {name} | {'PASS' if passed else 'FAIL / INCOMPLETE'} |" for name, passed in checks.items()]
    lines += ["", "## 执行状态", ""]
    for name, failure in failed_processes.items():
        lines.append(f"- `{name}`：退出码 {failure['returncode']}，运行 {failure['elapsed_s']:.2f} 秒。退出码 124 表示本次启动器超时或中断，不能算作通过。")
    if unstarted_phases:
        lines.append("- 未启动阶段：" + ", ".join(unstarted_phases) + "。")
    lines += ["", "| 诊断基线 | 已通过请求 / 总请求 | 最大绝对 logits 误差 |", "| --- | ---: | ---: |"]
    for baseline, values in diagnostics.items():
        error = values['max_abs_logit_error'] if values['max_abs_logit_error'] is not None else '未获得'
        lines.append(f"| {baseline} | {values['passed_requests']} / {len(ids)} | {error} |")
    lines += ["", "上表按原始完整审计统计；缺失文件也会导致未通过，不等于已经发现数值错误。"]
    if partial_audit:
        lines += ["", f"async 补充审计：{partial_audit['passed_requests']} 条完整请求通过；未完成请求不计为通过。",
                  "补充审计使用保留的启动配置，未补造超时进程的最终 config.json，也未改写原始 audit.json。详见 `partial-audit.json`。"]
    lines += ["", f"开发请求：{len(ids)}；领域分布：{domains}；冻结续写总 token：{result['total_generated_tokens']}。",
              "每条最多生成 128 token；遇 EOG 自然停止，不补随机 token。", "", "## 重复测时", "",
              "以下为每请求三次测量取均值、再对请求等权平均；不同请求不按 token 数加权。", "",
              "| 基线 | TTFT (ms) | TPOT (ms) |", "| --- | ---: | ---: |"]
    for baseline, values in performance.items():
        ttft = values['mean_request_ttft_ms'] if values['mean_request_ttft_ms'] is not None else '未运行'
        tpot = values['mean_request_tpot_ms'] if values['mean_request_tpot_ms'] is not None else '未运行'
        lines.append(f"| {baseline} | {ttft} | {tpot} |")
    lines += ["", "以上不是同一总物理内存预算下的优化收益结论。", "", "## 限制与未验收范围", ""]
    lines += ["- " + item for item in limitations]
    lines += ["", "完整数值、逐项统计、内存分配记录和采样范围见同目录 `acceptance-report.json`。",
              "原始检查结果见 `audit.json`。只有对应阶段完成后才会产生 `timing/summary.json` 和 `diagnostic-overhead.json`。"]
    (root / "acceptance-report.zh.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"m0_core_accepted": result["m0_core_accepted"], "checks": checks, "report": str(root / "acceptance-report.zh.md")}, indent=2))


if __name__ == "__main__":
    if not Path("/.dockerenv").exists() or len(sys.argv) != 2 or not re.fullmatch(r"[A-Za-z0-9_-]+", sys.argv[1]):
        raise SystemExit("Run inside moe-llama-dev with one experiment name")
    report(sys.argv[1])
