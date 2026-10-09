#!/usr/bin/env python3
"""Freeze small, document-disjoint M0 workloads from existing public datasets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.parse

ROOT = Path("/ssd/test/m0")
SOURCES = {
    "qa": ("rajpurkar/squad", "plain_text", "validation"),
    "code": ("google-research-datasets/mbpp", "sanitized", "test"),
    "math": ("openai/gsm8k", "main", "test"),
    "summary": ("abisee/cnn_dailymail", "3.0.0", "test"),
}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(url, path, endpoint):
    if path.exists():
        return
    partial = path.with_suffix(path.suffix + ".part")
    for _ in range(10):
        result = subprocess.check_output(["curl", "--fail", "--silent", "--show-error", "--proto", "=https",
            "--connect-timeout", "10", "--max-time", "180", "--retry", "2", "--retry-all-errors", "--retry-delay", "1",
            "--user-agent", "Mozilla/5.0", "--output", str(partial), "--write-out", "%{http_code}\n%{redirect_url}", url], text=True)
        status, redirect = result.split("\n", 1)
        if status == "200":
            partial.rename(path)
            return
        if not redirect:
            raise RuntimeError("unexpected HTTP status " + status)
        parts = urllib.parse.urlsplit(redirect)
        if endpoint != "https://huggingface.co" and parts.netloc == "huggingface.co":
            redirect = endpoint + urllib.parse.urlunsplit(("", "", parts.path, parts.query, parts.fragment))
        url = redirect
    raise RuntimeError("too many dataset redirects")


def normalize(domain, row):
    if domain == "qa":
        group = row["title"]
        prefix, body, suffix = ("Answer the question using the passage.\nPassage:\n", row["context"],
                                "\nQuestion: " + row["question"] + "\nAnswer:")
        original_id = row["id"]
    elif domain == "code":
        group = str(row["task_id"])
        prefix, body, suffix = "Write a Python function for this task.\n" + row["prompt"], "", "\nReturn Python code."
        original_id = group
    elif domain == "math":
        group = digest(row["question"].strip())
        prefix, body, suffix = "Solve this problem and explain the steps.\n" + row["question"], "", "\nSolution:"
        original_id = group
    else:
        group = row["id"]
        prefix, body, suffix = "Summarize the following news passage.\n", row["article"], "\nSummary:"
        original_id = group
    return {"id": domain + "_" + digest(str(original_id))[:16], "domain": domain,
            "source_id": str(original_id), "group": group, "prefix": prefix, "body": body,
            "suffix": suffix, "content_sha256": digest(prefix + body + suffix)}


def select(rows, seed, dev, confirm):
    # One request per document/task, including within each split.
    groups = {}
    for row in sorted(rows, key=lambda r: digest(str(seed) + r["id"])):
        groups.setdefault(row["group"], row)
    ranked = sorted(groups.values(), key=lambda r: digest(str(seed) + r["group"]))
    chosen, seen = [], set()
    for row in ranked:
        if row["content_sha256"] not in seen:
            chosen.append(row)
            seen.add(row["content_sha256"])
        if len(chosen) == dev + confirm:
            return chosen[:dev], chosen[dev:]
    raise ValueError("not enough distinct source groups")


def main():
    if not Path("/.dockerenv").exists():
        raise RuntimeError("M0 must run through scripts/run-moe-m0-docker.sh")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="public-v1")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--endpoint", choices=("https://huggingface.co", "https://hf-mirror.com"), default="https://huggingface.co")
    parser.add_argument("--resume", action="store_true", help="resume an incomplete snapshot using saved repository metadata")
    parser.add_argument("--dev-per-domain", type=int, default=3)
    parser.add_argument("--confirm-per-domain", type=int, default=12)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be a simple directory name")
    if min(args.dev_per_domain, args.confirm_per_domain) <= 0:
        parser.error("sample counts must be positive")
    # Keep dependency caches in the experiment tree too.
    os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache")
    import pyarrow.parquet as pq
    destination = ROOT / "datasets" / args.name
    destination.mkdir(parents=True, exist_ok=args.resume)
    if (destination / "manifest.json").exists():
        raise RuntimeError("snapshot already complete; refusing to overwrite")
    manifest = {"seed": args.seed, "endpoint": args.endpoint, "sources": {}, "selection": "sha256-ranked unique document/task groups"}
    partitions = {"dev": [], "confirm": []}
    for domain, (repo, config, split) in SOURCES.items():
        print("Downloading", domain, repo, config, split, flush=True)
        metadata_path = destination / (domain + "-metadata.json")
        download(args.endpoint + "/api/datasets/" + repo, metadata_path, args.endpoint)
        info = json.loads(metadata_path.read_text())
        revision = info["sha"]
        paths = sorted(x["rfilename"] for x in info["siblings"]
                       if x["rfilename"].startswith(config + "/" + split + "-") and x["rfilename"].endswith(".parquet"))
        if not paths:
            raise RuntimeError("no parquet files found for " + repo + "/" + config + "/" + split)
        rows, files = [], []
        for index, remote in enumerate(paths):
            local = destination / f"{domain}-{index}.parquet"
            url = f"{args.endpoint}/datasets/{repo}/resolve/{revision}/{remote}"
            download(url, local, args.endpoint)
            metadata = next(x for x in info["siblings"] if x["rfilename"] == remote)
            expected_hash = metadata.get("lfs", {}).get("sha256")
            if expected_hash and file_hash(local) != expected_hash:
                raise RuntimeError("download does not match repository LFS SHA-256")
            files.append({"url": url, "sha256": file_hash(local), "file": local.name})
            for batch in pq.ParquetFile(local).iter_batches(batch_size=512):
                rows.extend(normalize(domain, row) for row in batch.to_pylist())
        dev, confirm = select(rows, args.seed, args.dev_per_domain, args.confirm_per_domain)
        for name, selected in (("dev", dev), ("confirm", confirm)):
            for row in selected:
                row["dataset"] = repo
                row["revision"] = revision
                row["source_split"] = split
            partitions[name].extend(selected)
        manifest["sources"][domain] = {"repository": repo, "config": config, "split": split,
                                        "revision": revision, "files": files, "rows": len(rows)}
    for name, requests in partitions.items():
        path = destination / (name + ".json")
        path.write_text(json.dumps({"requests": requests}, ensure_ascii=False, indent=2) + "\n")
        manifest[name] = {"requests": len(requests), "sha256": file_hash(path)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
