#!/usr/bin/env python3
"""Retire successful verbose logs only after their closed-run evidence is preserved."""
import fcntl
import hashlib
import importlib.util
import json
import stat
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
LD = P / 'bench/results/2026-10-14-long-generation-benchmark'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    names = ['main-primary-long-generation-suite-root-v4-upload-fix-v1',
             'main-primary-qsa-layout-build-root-v1', 'main-primary-qsa-layout-validation-root-v1',
             'main-primary-qsa-layout-microbench-root-v1']
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = {}
        for name in names:
            path = B / name / 'record.json'
            r = json.loads(path.read_text())
            assert not r['active'] and r['passed'] and r['complete']
            for c in r['commands']:
                assert owner.closed(c) and c['exit_code'] == 0
                now = owner.identity(c['pid'])
                assert now is None or now['start_ticks'] != c['direct_identity']['start_ticks']
            records[name] = dict(path=str(path), sha256=sha(path))
        suite = json.loads((LD / 'suite-v4-upload-fix-final-summary.json').read_text())
        assert suite['benchmark_protocol_qualified'] and all(suite['gates'].values())
        screen = json.loads((D / 'qsa-layout-screen-summary.json').read_text())
        assert screen['passed'] and screen['all_fresh_fault_checks_zero']
        for name, pin in screen['receipt_sha256'].items():
            assert sha(D / name) == pin
        raw = B / names[2] / 'owned/qsa-layout-validation.stderr'
        receipt = json.loads((D / 'qsa-layout-validation-v1.json').read_text())
        assert sha(raw) == receipt['stderr_sha256']
        lines = raw.read_text().splitlines()
        selected = set()
        for api in ['urEnqueueKernelLaunch', 'urEnqueueUSMMemcpy', 'urQueueFinish', 'urContextRelease']:
            indices = [i for i, line in enumerate(lines) if api in line]
            for i in indices[:2]:
                selected.update(range(max(0, i - 1), min(len(lines), i + 2)))
        excerpt = dict(source_path=str(raw), source_bytes=raw.stat().st_size, source_sha256=sha(raw),
                       trace_summary=receipt['api_trace_summary'],
                       lines=[dict(line=i + 1, text=lines[i]) for i in sorted(selected)],
                       scope='Representative successful API entries/results; full trace question closed by20case parity/oracle/exit/fault validation. No device timing claim.')
        assert len(json.dumps(excerpt).encode()) <= 256 << 10
        (D / 'qsa-layout-validation-api-excerpt.json').write_text(json.dumps(excerpt, indent=2) + '\n')
        evidence = [LD / 'suite-v4-upload-fix-final-summary.json', LD / 'suite-v4-upload-fix-engine-summary.json',
                    LD / 'suite-v4-upload-fix-kernel-summary.json', LD / 'receipts/main-primary-long-generation-suite-root-v4-upload-fix-v1/index.json',
                    D / 'qsa-layout-screen-summary.json', D / 'qsa-layout-validation-api-excerpt.json']
        for path in evidence:
            assert path.is_file()
        remove = []
        for name in names:
            remove.extend(p for p in (B / name / 'owned').iterdir() if p.suffix in ['.stdout', '.stderr'])
        remove.extend((B / names[0] / 'stock-results').glob('*/README.md'))
        manifest = dict(scope='Closed successful text logs only, exclusive measurement lock. Structured originals, all samples, sources, binaries, current fixtures and statuses untouched.',
                        closed_records=records, retained_evidence={str(p): sha(p) for p in evidence},
                        files=[], logical_bytes_removed=0, allocation_bytes_removed=0,
                        available_space_reclaimed_measured=False,
                        raw_structured_retained_consumer='Long-generation finalizer CPU mutation fixture and original reference comparison; next review after consumer archive adaptation. Not a progress backup.')
        output = D / 'closed-trace-retention-v1.json'
        assert not output.exists()
        for path in sorted(remove):
            info = path.lstat()
            assert stat.S_ISREG(info.st_mode) and not path.is_symlink()
            row = dict(path=str(path), bytes=info.st_size, sha256=sha(path),
                       allocated_bytes=info.st_blocks * 512,
                       reason='Successful full trace/console/journal or duplicate formatted README; individual results, original closure and selected validation evidence preserved',
                       removed=False)
            manifest['files'].append(row)
        output.write_text(json.dumps(manifest, indent=2) + '\n')
        for row in manifest['files']:
            path = Path(row['path'])
            assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
            path.unlink()
            row['removed'] = True
            manifest['logical_bytes_removed'] += row['bytes']
            manifest['allocation_bytes_removed'] += row['allocated_bytes']
            output.write_text(json.dumps(manifest, indent=2) + '\n')
        for r in records.values():
            assert sha(Path(r['path'])) == r['sha256']
        print(json.dumps(dict(removed_files=len(manifest['files']), logical_bytes=manifest['logical_bytes_removed'],
                              allocation_bytes=manifest['allocation_bytes_removed'], original_statuses_unchanged=True)), flush=True)


if __name__ == '__main__':
    main()
