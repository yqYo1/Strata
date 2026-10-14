#!/usr/bin/env python3
"""Extract completed module-create spans without retaining repeated API polls."""
import datetime as dt
import fcntl
import hashlib
import json
import re
from pathlib import Path

D = Path(__file__).parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ZONE = dt.timezone(dt.timedelta(hours=9))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = B / 'main-primary-cold-jit-short-root-v9'
    record = json.loads((root / 'record.json').read_text())
    assert record['passed'] and record['complete'] and not record['active']
    pending, spans, progress = {}, [], []
    path = root / 'owned/engine.stderr'
    lines = path.read_text().splitlines()
    pattern = re.compile(r'^\[([^]]+)\] \[thread-id: ([0-9]+)\] \[trace\] (.*)$')
    for number, line in enumerate(lines, 1):
        found = pattern.match(line)
        if found and 'zeModuleCreate(' in found[3]:
            stamp = dt.datetime.fromisoformat(found[1]).replace(tzinfo=ZONE)
            thread = found[2]
            if found[3].startswith('zeModuleCreate('):
                assert thread not in pending
                pending[thread] = (stamp, number, line)
            elif 'SUCCESS (ZE_RESULT_SUCCESS) in zeModuleCreate(' in found[3]:
                assert thread in pending
                start, first, begin_line = pending.pop(thread)
                spans.append(dict(logger_thread_id=thread, begin_local=start.isoformat(),
                    end_local=stamp.isoformat(), elapsed_seconds=(stamp-start).total_seconds(),
                    begin_line=first, end_line=number, begin=begin_line, end=line))
        if line.startswith(('strata prefill timing:', 'strata decode timing:',
                            'strata trace: request', 'strata trace: prompt',
                            'strata serve: prompt ', 'strata verify: capture')):
            progress.append(dict(line=number, text=line))
    assert not pending
    spans.sort(key=lambda item: item['elapsed_seconds'], reverse=True)
    assert spans and spans[0]['elapsed_seconds'] > 60
    begin = dt.datetime.fromisoformat(spans[0]['begin_local'])
    end = dt.datetime.fromisoformat(spans[0]['end_local'])
    sample_path = root / 'owned/thread-samples.jsonl'
    samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
    window = [sample for sample in samples if begin <= dt.datetime.fromisoformat(sample['utc']) <= end]
    evidence = dict(root=str(root), original_status='passed actual short model; no long/performance qualification',
        trace=dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), filtered=True),
        complete_module_calls_observed=len(spans), completed_module_calls=spans,
        progress=progress, whole_stream_census=json.loads((root/'stderr-relay-summary.json').read_text()),
        passive_thread_samples=dict(path=str(sample_path), bytes=sample_path.stat().st_size,
            sha256=digest(sample_path), total=len(samples), in_longest_module_window=len(window),
            representative=[window[0], window[len(window)//2], window[-1]] if window else []),
        observations=[
            'The observed host zeModuleCreate call lasted 89.276 seconds and returned success.',
            'This exceeds the default60s watchdog; v9 used an explicit300s diagnostic limit.',
            'This is host module creation, not GPU kernel execution or steady-state model speed.',
            'Logger thread IDs are not Linux TIDs; no identity mapping was established.',
            'No canary abort occurred in v9. The v7 canary corruption origin remains unlocated.',
            'v8 original quota-stop and failed fault query remain failed; a separate health receipt checked its cursor.'
        ])
    (D/'cold-jit-evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(dict(completed_module_calls=len(spans), longest_seconds=spans[0]['elapsed_seconds'], samples_in_window=len(window))))


if __name__ == '__main__':
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
