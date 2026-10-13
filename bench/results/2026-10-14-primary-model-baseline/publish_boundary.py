#!/usr/bin/env python3
"""Preserve exact closed receipts and compact diagnostics; no log deletion."""
import fcntl
import gzip
import hashlib
import json
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
R = B / 'research-20261009'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    index = []
    roots = [f'main-primary-model-diagnostic-root-v{x}' for x in (1, 2, 4)]
    roots += [f'main-primary-size-t-range-build-root-v{x}' for x in (1, 2)]
    roots += ['main-primary-throw-observer-root-v1']
    for optional in ['main-primary-size-t-range-gpu-root-v1', 'main-primary-model-diagnostic-root-v5']:
        if (B / optional / 'record.json').is_file():
            roots.append(optional)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for root in roots:
            output = B / root
            record = json.loads((output / 'record.json').read_text())
            assert not record['active'], root
            for command in record['commands']:
                assert (command.get('exit_code') is not None and command.get('observation_complete')
                    and command.get('direct_child_reaped') and command.get('session_empty')
                    and not command.get('survivors')), (root, command['label'])
            for name in ['record.json', 'result.json', 'engine-exit.json']:
                source = output / name
                if not source.exists():
                    continue
                data = source.read_bytes()
                target = D / 'receipts' / root / (name + '.gz')
                target.parent.mkdir(parents=True, exist_ok=True)
                compressed = gzip.compress(data, mtime=0)
                assert gzip.decompress(compressed) == data
                if target.exists():
                    assert gzip.decompress(target.read_bytes()) == data
                else:
                    target.write_bytes(compressed)
                index.append(dict(original_path=str(source), path=str(target.relative_to(D)),
                    decoded_bytes=len(data), decoded_sha256=sha(data), archive_sha256=sha(target.read_bytes())))
        reports = []
        for round_number, name in [
            (459, 'research-primary-profiler-attribution-round459-v1.txt'),
            (460, 'research-low-vram-moe-scheduling-round460-v1.txt'),
            (461, 'research-primary-origin-reaudit-round461-v1.txt'),
            (462, 'research-primary-component-reaudit-round462-v1.txt'),
            (463, 'research-primary-large-range-contract-round463-v1.txt'),
        ]:
            data = (R / name).read_bytes()
            target = D / 'research' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == data
            else:
                target.write_bytes(data)
            reports.append(dict(round=round_number, path=str(target.relative_to(D)),
                original_path=str(R / name), sha256=sha(data), model='gpt-6-luna', read_only=True))
        (D / 'research-index.json').write_text(json.dumps(dict(
            prior_indexes=['../2026-10-14-primary-native-runtime/research-index.json',
                           '../2026-10-14-niko-main-service-port/research-index.json'], reports=reports,
            no_intervention_until_final=True, main_work_in_parallel='primary source/ref audit and native range-contract build',
            main_corrections=['R461/R462 describe committed HEAD; the new uncommitted range flag is a separate Main change.',
                'R463 identifies the SYCL host guard, not the exact failing operation. Flag/build/host test alone do not prove the model fix.',
                'Real primary short HTTP succeeded with heavy diagnostics; it is not a qualified speed baseline.',
                'Niko shared non-inference obligations remain even where main Arc lacks the root CUDA batch protocol.']), indent=2) + '\n')
        windows = []
        for version in (1, 2, 4, 5):
            source = B / f'main-primary-model-diagnostic-root-v{version}/owned/engine.stderr'
            if not source.exists():
                continue
            lines = source.read_text().splitlines()
            indices = set()
            for i, line in enumerate(lines):
                if any(marker in line for marker in ['strata trace: request', 'strata trace: prompt start',
                    'strata trace: prompt chunk', 'strata serve: prompt ', 'strata prefill timing:',
                    'strata decode timing:', 'needs --spec', 'unknown argument', 'unknown option',
                    'terminate called', 'what():', 'strata diagnostic SYCL throw:']):
                    indices.update(range(max(0, i-1), min(len(lines), i+6)))
            # A remaining throw needs its whole backtrace, not just its signal.
            for i, line in enumerate(lines):
                if line.startswith('strata diagnostic SYCL throw:'):
                    indices.update(range(i, min(len(lines), i+55)))
            data = source.read_bytes()
            windows.append(dict(version=version, original_path=str(source), original_bytes=len(data),
                original_sha256=sha(data), original_lines=len(lines),
                selected_lines=[dict(line=i+1, text=lines[i]) for i in sorted(indices)],
                reason='distinct CLI/startup/range mechanism and relevant request/phase evidence'))
        (D / 'model-diagnostic-windows.json').write_text(json.dumps(windows, indent=2) + '\n')
        (D / 'receipts-index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(dict(closed_roots=len(roots), receipts=len(index), research=len(reports), raw_deleted=False)), flush=True)


if __name__ == '__main__':
    main()
