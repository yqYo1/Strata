#!/usr/bin/env python3
"""Closed-run host analysis: recover an old timestamp tag from both body hashes.

Run before changing the pinned benchmark client. No GPU, HTTP or clock change.
This identifies input provenance; it does not qualify a performance comparison.
"""
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = A / 'selective-iq-clean-bench-full-0-v1.json'
        original = json.loads(receipt.read_text())
        require(not original['active'], 'original job must be closed')
        for command in original['commands']:
            require(command.get('direct_child_reaped') and command.get('session_empty'),
                    'original owned process/session not retired')
            for identity in command.get('owners', []):
                stat = Path('/proc', str(identity['pid']), 'stat')
                if stat.exists():
                    current = stat.read_text().rsplit(')', 1)[1].split()
                    require(int(current[19]) != identity['start_ticks'],
                            'original owned identity remains, including zombie')
        for relative, pin in original['benchmark_source'].items():
            require(sha(P / relative) == pin, 'pinned benchmark source changed: ' + relative)
        native_job = [c for c in original['commands'] if c['label'] == 'bench-full-0']
        require(len(native_job) == 1, 'unique original benchmark command required')
        command = native_job[0]
        lo = int(datetime.datetime.fromisoformat(command['started_utc']).timestamp()) - 1
        hi = int(datetime.datetime.fromisoformat(command['finished_utc']).timestamp()) + 1
        require(0 < hi - lo <= 30000, 'finite original timestamp search bound')
        root = B / 'main-selective-iq-clean-bench-full-0-root-v1'
        journals = list((root / 'stock-results').glob('*/requests.jsonl'))
        require(len(journals) == 1, 'unique request journal required')
        journal = journals[0]
        require(journal.stat().st_size <= 8 << 20, 'bounded request journal')
        sys.path.insert(0, str(P / 'tools'))
        spec = importlib.util.spec_from_file_location('original_benchmark_client', P / 'tools/strata_bench.py')
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        verification = client.verify_request_journal(journal.parent)
        require(verification['valid'], 'original request journal invalid')
        starts = [r for r in map(json.loads, journal.read_text().splitlines())
                  if r['kind'] == 'start' and r['phase'] == 'calibration']
        require(len(starts) == 2 and {r['run'] for r in starts} == {200, 800},
                'both original calibration body hashes required')
        require(all(r['max_tokens'] == 1 and r['thinking'] is False for r in starts),
                'unexpected original calibration contract')
        hashes = {r['run']: r['request_sha256'] for r in starts}
        fillers = {n: client.long_prompt(n) for n in (200, 800)}

        def body_hash(n, nonce):
            messages = [{'role': 'user', 'content': f'[cal-{n}-{nonce}]\n' + fillers[n]}]
            payload = json.dumps(client.chat_body(messages, 1, False, None, None)).encode()
            return hashlib.sha256(payload).hexdigest()

        candidates = [str(n) for n in range(lo, hi + 1)
                      if body_hash(200, str(n)) == hashes[200]]
        matches = [n for n in candidates if body_hash(800, n) == hashes[800]]
        require(len(matches) == 1, 'timestamp nonce not uniquely verified by both actual requests')
        output = A / 'selective-iq-full0-workload-provenance-v1.json'
        result = dict(schema='closed-benchmark-workload-provenance-v1', passed=True,
                      complete=True, active=False, GPU_executed=False,
                      original_receipt_sha256=sha(receipt), original_passed=original['passed'],
                      journal_sha256=sha(journal), journal_verification=verification,
                      benchmark_source=original['benchmark_source'], recipe_sha256=sha(Path(__file__)),
                      search_first_epoch=lo, search_last_epoch=hi, recovered_nonce=matches[0],
                      actual_calibration_request_sha256=hashes,
                      performance_qualified=False, quality_qualified=False,
                      scope='Two exact historical calibration body hashes identify one timestamp tag. '
                            'All later actual request hashes and the reviewed client delta must still '
                            'match before reusing this run in a paired comparison.')
        with output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(dict(passed=True, recovered_nonce=matches[0], GPU_executed=False)))


if __name__ == '__main__':
    main()
