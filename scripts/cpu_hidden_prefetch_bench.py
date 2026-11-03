#!/usr/bin/env python3
"""Run matched SSD/UMA prefetch experiments in the existing container."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time

ROOT = Path('/ssd/test/prefetch-cpu-hidden-20261102')
WORK = Path('/workspace/results/prefetch-cpu-hidden-20261102')
BINARY = Path('/ssd/test/prefetch-cpu/build/bin/llama-moe-m0')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def config(mode, name, correctness):
    result = dict(
        model='/ssd/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf',
        pack='/ssd/test/ssd-pack-20261008/weights/Qwen3-30B-A3B-BF16.experts.gguf',
        output=str(ROOT/name),
        input='/workspace/results/ac-prefetch-20261012/data/input-%d.json' % (4 if correctness else 32),
        baseline='async', action='replay', cache_mib=8192, history_reserve_mib=64,
        workers=4, batch=32, threads=4, gpu_layers=99, gpu_experts=True,
        diagnostic=False, verify_weights=False, save_logits=correctness, ssd_stats=True,
        zero_copy=True, prefetch=mode, prefetch_count=2, prefetch_distance=4,
        prefill_prefetch_count=16, prefill_prefetch_slots=64, cpu_decode_prefetch_slots=4,
        cpu_decode_temporal_weight=1.0, cpu_async=True, history_bank='', b_adapter='', b_margin=0.25)
    if mode == 'cpu_sync':
        result.update(prefetch='cpu', cpu_async=False)
    if mode == 'cpu_history':
        result.update(prefill_prefetch_count=8, prefill_prefetch_slots=32)
    if mode == 'none':
        result.update(prefetch_count=0, prefill_prefetch_count=0, prefill_prefetch_slots=0)
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(mode, name, correctness):
    cfg = config(mode, name, correctness)
    write(WORK/(name+'.json'), cfg)
    print('START', name, flush=True)
    binary_sha256 = digest(BINARY)
    library_sha256 = {p.name: digest(p) for p in sorted(BINARY.parent.glob('*.so'))}
    start = time.time()
    with (ROOT/(name+'.log')).open('x') as log:
        result = subprocess.run([str(BINARY), str(WORK/(name+'.json'))], stdout=log, stderr=subprocess.STDOUT)
    write(ROOT/(name+'.status.json'), dict(returncode=result.returncode, elapsed_s=time.time()-start,
          binary_sha256=binary_sha256, library_sha256=library_sha256,
          mode=mode, correctness=correctness))
    if result.returncode:
        print((ROOT/(name+'.log')).read_text()[-6000:], flush=True)
        raise SystemExit(result.returncode)
    rows = json.loads((ROOT/name/'metrics.json').read_text())
    print('DONE', name, json.dumps(summarize(rows)), flush=True)


def percentile(values, p):
    return sorted(values)[math.ceil(len(values)*p)-1]


def summarize(rows):
    ttft = [r['ttft_ms'] for r in rows]
    intervals = [t for r in rows for t in r['token_ms']]
    seconds = sum(r['wall_ms'] for r in rows)/1000
    return dict(requests=len(rows), ttft_mean_ms=sum(ttft)/len(ttft), ttft_p99_ms=percentile(ttft, .99),
        prefill_tok_s=sum(r['prompt_tokens'] for r in rows)/(sum(r['prefill_ms'] for r in rows)/1000),
        decode_tpot_ms=sum(intervals)/len(intervals), decode_p99_ms=percentile(intervals, .99),
        decode_tok_s=len(intervals)/(sum(intervals)/1000),
        output_tok_s=sum(r['generated_tokens'] for r in rows)/seconds)


def report():
    results = {}
    for status_file in sorted(ROOT.glob('*.status.json')):
        name = status_file.name.removesuffix('.status.json')
        status = json.loads(status_file.read_text())
        if status['returncode']:
            continue
        rows = json.loads((ROOT/name/'metrics.json').read_text())
        stats = {}
        for line in (ROOT/(name+'.log')).read_text().splitlines():
            prefix = line.split(':', 1)[0]
            if prefix not in ('ssd_moe_cache', 'moe_prefetch', 'moe_prefetch_prefill', 'moe_prefetch_cpu'):
                continue
            target = stats.setdefault(prefix, {})
            for key, value in re.findall(r'(\w+)=([0-9.]+)', line):
                if key in ('resident', 'peak_bytes', 'router_bytes', 'async'):
                    target[key] = max(target.get(key, 0), float(value))
                else:
                    target[key] = target.get(key, 0) + float(value)
        cache = stats['ssd_moe_cache']
        cache['hit_rate'] = 100 * cache['hits'] / (cache['hits'] + cache['misses'])
        results[name] = dict(status=status, metrics=summarize(rows), counters=stats, requests=rows)
    checks = {}
    token_checks = {}
    timing_baseline = next((v for k, v in results.items() if k.startswith('timing-none-')), None)
    if timing_baseline:
        reference = {r['id']: r['greedy_tokens'] for r in timing_baseline['requests']}
        for name, data in results.items():
            if not data['status']['correctness']:
                token_checks[name] = {r['id']: r['greedy_tokens'] == reference[r['id']]
                                     for r in data['requests']}
    baseline = ROOT/'correctness-none'
    if baseline.exists():
        for name, data in results.items():
            if not data['status']['correctness']:
                continue
            checks[name] = {p.parent.name: digest(p) == digest(ROOT/name/p.parent.name/'logits.f32')
                for p in baseline.glob('*/logits.f32')}
    invariants = {}
    for name, data in results.items():
        counters = data['counters']
        cache = counters['ssd_moe_cache']
        check = dict(zero_copy_active=cache['zero_copy_binds'] > 0,
                     no_payload_copy=cache['backend_copy'] == 0,
                     no_zero_copy_fallback=cache['zero_copy_fallbacks'] == 0)
        if data['status']['mode'] in ('cpu', 'cpu_sync'):
            cfg = json.loads((WORK/(name+'.json')).read_text())
            expected_jobs = sum((48-cfg['prefetch_distance']) *
                ((math.ceil(r['prompt_tokens']/cfg['batch']) if cfg['prefill_prefetch_count'] else 0) +
                 (r['generated_tokens']-1 if cfg['prefetch_count'] else 0)) for r in data['requests'])
            check['one_prediction_per_layer'] = counters['moe_prefetch_cpu']['jobs'] == expected_jobs
            check['no_pending_jobs'] = counters['moe_prefetch_cpu']['pending'] == 0
        invariants[name] = check
    output = dict(runs=results, logits_identical=checks, timing_tokens_identical=token_checks,
        invariants=invariants,
        limitations=['Four independent requests; exploratory results, TTFT P99 is sample maximum.',
                     'Correctness and performance are separate. No global page-cache drop.',
                     'All runs use P2 zero-copy, 8128 MiB expert cache and 64 MiB predictor reserve.'])
    write(WORK/'report.json', output)
    print(json.dumps(dict(logits_identical=checks, runs={k:v['metrics'] for k,v in results.items()}), indent=2))
    if any(not passed for group in (checks, token_checks, invariants)
           for check in group.values() for passed in check.values()):
        raise SystemExit('Correctness or runtime invariant check failed; see report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['correctness', 'timing', 'report'])
    parser.add_argument('--modes', nargs='+', default=['none', 'bguf', 'cpu'])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--series', default='')
    args = parser.parse_args()
    if args.series:
        ROOT /= args.series
        WORK /= args.series
    ROOT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.phase == 'correctness':
        for mode in args.modes:
            run(mode, 'correctness-'+mode, True)
    elif args.phase == 'timing':
        rng = random.Random(20261102)
        for round_id in range(args.rounds):
            order = list(args.modes)
            rng.shuffle(order)
            for mode in order:
                run(mode, 'timing-%s-r%d' % (mode, round_id), False)
    report()
