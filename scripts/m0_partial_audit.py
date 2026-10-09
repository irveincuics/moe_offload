#!/usr/bin/env python3
"""Audit completed requests from an interrupted run, without repairing its artifacts."""
import importlib.util
import json
from pathlib import Path
import re
import sys


def main(name):
    root = Path('/ssd/test/m0') / name
    spec = importlib.util.spec_from_file_location('m0_driver', '/workspace/llama.cpp/scripts/moe-m0.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    manifest = driver.read(root / 'manifest.json')
    if driver.source_hash() != manifest['source_sha256']:
        raise RuntimeError('Source changed; partial audit refused')
    config = driver.read(root / 'diagnostic/async-config.json')
    requests = driver.read(root / 'freeze/sequences.json')['requests']
    rows = []
    for request in requests:
        directory = root / 'diagnostic/async' / request['id']
        if not (directory / 'metrics.json').exists():
            rows.append({'id': request['id'], 'complete': False, 'pass': False})
            continue
        metadata = driver.read(directory / 'metrics.json')
        with (directory / 'trace.jsonl').open() as stream:
            checked = driver.audit_events((json.loads(line) for line in stream), 'async', metadata['layers'],
                                          config['cache_mib'] * 1024**2)
        checked['logits'] = driver.compare_logits(root / 'diagnostic/resident' / request['id'], directory, 1e-3, 1e-4)
        checked['pass'] &= checked['logits']['pass']
        checked.update({'id': request['id'], 'complete': True})
        rows.append(checked)
        print(request['id'], checked['pass'], flush=True)
    result = {'baseline': 'async', 'full_run_accepted': False,
              'scope': 'Completed requests only; uses preserved launch config because final config.json was not written on timeout.',
              'atol': 1e-3, 'rtol': 1e-4, 'passed_requests': sum(r['pass'] for r in rows), 'results': rows}
    driver.write(root / 'partial-audit.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'results'}, indent=2))


if __name__ == '__main__':
    if not Path('/.dockerenv').exists() or len(sys.argv) != 2 or not re.fullmatch(r'[A-Za-z0-9_-]+', sys.argv[1]):
        raise SystemExit('Run inside moe-llama-dev with one experiment name')
    main(sys.argv[1])
