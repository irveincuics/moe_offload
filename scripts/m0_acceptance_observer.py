#!/usr/bin/env python3
"""Read-only resource observation for an existing M0 experiment in Docker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def fields(path):
    result = {}
    try:
        for line in Path(path).read_text().splitlines():
            parts = line.replace(":", "").split()
            if len(parts) >= 2 and parts[1].isdigit():
                result[parts[0]] = int(parts[1])
    except (OSError, PermissionError):
        pass
    return result


def observe(root):
    result = {"monotonic_ns": time.monotonic_ns(), "unix_s": time.time(), "processes": []}
    memory = fields("/proc/meminfo")
    result["host_memory_kib"] = {k: memory.get(k) for k in ("MemTotal", "MemAvailable", "Cached", "SwapFree", "Dirty", "Writeback")}
    for name in ("memory.current", "memory.max", "memory.swap.current", "memory.events", "memory.stat"):
        path = Path("/sys/fs/cgroup") / name
        try:
            value = path.read_text().strip()
            result["container_" + name] = int(value) if value.isdigit() else value
        except OSError:
            result["container_" + name] = None
    device = os.stat("/ssd").st_dev
    for line in Path("/proc/diskstats").read_text().splitlines():
        p = line.split()
        if (int(p[0]), int(p[1])) == (os.major(device), os.minor(device)):
            result["shared_partition_io"] = {"device": p[2], "read_bytes": int(p[5]) * 512,
                                             "write_bytes": int(p[9]) * 512, "busy_ms": int(p[12])}
    active = False
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            comm = (proc / "comm").read_text().strip()
            if "moe-m0.py" in command and root.name in command:
                active = True
            if comm != "llama-moe-m0":
                continue
            status = fields(proc / "status")
            result["processes"].append({"pid": int(proc.name), "command": command,
                "memory_kib": {k: status.get(k) for k in ("VmRSS", "VmHWM", "VmSwap")},
                "smaps_rollup_kib": fields(proc / "smaps_rollup"), "io": fields(proc / "io")})
        except (OSError, PermissionError):
            continue
    return result, active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--seconds", type=int, default=3600)
    args = parser.parse_args()
    if not Path("/.dockerenv").exists() or Path(args.experiment).name != args.experiment:
        raise RuntimeError("use an experiment name inside the existing Docker container")
    root = Path("/ssd/test/m0") / args.experiment
    for _ in range(60):
        if root.exists():
            break
        time.sleep(1)
    if not root.exists():
        raise RuntimeError("experiment was not created by its runner")
    start, last_active = time.monotonic(), time.monotonic()
    peak_rss = peak_cgroup = max_process_read = 0
    first_disk = last_disk = None
    samples = 0
    with (root / "resource-observation.jsonl").open("x") as log:
        while time.monotonic() - start < args.seconds:
            sample, active = observe(root)
            if samples % 10 == 0:
                try:
                    gpu = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
                    sample["gpu_csv"] = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
                except (OSError, subprocess.TimeoutExpired) as exc:
                    sample["gpu_csv"] = str(exc)
            log.write(json.dumps(sample) + "\n")
            log.flush()
            samples += 1
            if active:
                last_active = time.monotonic()
            peak_cgroup = max(peak_cgroup, sample.get("container_memory.current") or 0)
            for process in sample["processes"]:
                peak_rss = max(peak_rss, process["memory_kib"].get("VmRSS") or 0)
                max_process_read = max(max_process_read, process["io"].get("read_bytes", 0))
            disk = sample.get("shared_partition_io")
            if disk:
                first_disk = first_disk or disk
                last_disk = disk
            if time.monotonic() - last_active > 15:
                break
            time.sleep(1)
    report = {"samples": samples, "observed_peak_process_rss_kib": peak_rss,
              "observed_peak_container_memory_bytes": peak_cgroup,
              "largest_sampled_process_read_bytes": max_process_read,
              "shared_partition_read_bytes_delta": last_disk["read_bytes"] - first_disk["read_bytes"] if first_disk else None,
              "limitations": ["Sampling can miss peaks and process-final counters.",
                              "Container accounting includes all container processes; partition counters include other workloads.",
                              "Do not add RSS, cgroup and page-cache counts: they overlap.",
                              "No total-memory quota is imposed. Observed usage does not prove a bounded total UMA budget.",
                              "Observation starts when this helper starts; earlier model hashing/loading may be outside its window."]}
    (root / "resource-observation-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
