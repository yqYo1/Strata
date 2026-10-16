#!/usr/bin/env python3
"""Main owned immutable official-source copy and exact audit-patch application.
Repository LICENSE. No compiler/configure/build/provider/device operations.
"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
PTI = Path('/home/yayoi/ghq/github.com/intel/pti-gpu')
D = A / 'own-metric-query-tiny-v2'
REVISION = '6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033'


def pin(p):
    with p.open('rb') as stream:
        h = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(p), realpath=str(p.resolve()), bytes=p.stat().st_size, sha256=h)


def main():
    root = B / 'main-metric-query-collector-source-v1'
    target = A / 'own-metric-collector-source-preparation-v1.json'
    assert not root.exists() and not target.exists()
    package = pin(D / 'SHA256SUMS')
    assert package['sha256'] == '959340640f7704acce3c052ed1e12f888639133e141f8712188023d6e3426651'
    for line in (D / 'SHA256SUMS').read_text().splitlines():
        h, name = line.split('  ', 1)
        assert pin(D / name)['sha256'] == h
    owner_source = A.parents[2] / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert pin(owner_source)['sha256'] == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('metric_source_prep_owner', owner_source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        source = root / 'src'
        source.mkdir()
        record = dict(schema='strata-metric-collector-source-preparation-v1', status='INCOMPLETE',
            complete=False, active=False, commands=[], GPU_executed=False, compiler_executed=False,
            official_pti_commit=REVISION, package=package, patch=pin(D / 'pti-query-status.patch'),
            private_source=str(source), original_checkout_mutated=False)
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = helper.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')

        def run(label, argv, cwd=PTI):
            c, out, err = owner.run(label, argv, env, cwd, wall=30, cpu=20,
                rss_cap=1 << 30, text_cap=8 << 20, total_cap=128 << 20, file_cap=8 << 20)
            assert helper.closed(c) and c['exit_code'] == 0
            return out.read_text(), err.read_text()

        try:
            head, _ = run('official-source-revision', ['/usr/bin/git', 'rev-parse', 'HEAD'])
            assert head.strip() == REVISION
            clean, _ = run('official-source-clean', ['/usr/bin/git', 'status', '--porcelain=v1'])
            assert not clean
            listing, _ = run('official-tracked-files', ['/usr/bin/git', 'ls-files', '--stage', '-z'])
            hashes = {}
            modes = {}
            for row in listing.split('\0'):
                if not row:
                    continue
                info, name = row.split('\t', 1)
                mode, object_id, stage = info.split()
                assert mode in ('100644', '100755') and stage == '0'
                relative = Path(name)
                assert not relative.is_absolute() and '..' not in relative.parts
                original = PTI / relative
                assert original.is_file() and not original.is_symlink()
                data = original.read_bytes()
                assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == object_id
                hashes[name] = hashlib.sha256(data).hexdigest()
                modes[name] = mode
                copy = source / relative
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_bytes(data)
                os.chmod(copy, 0o755 if mode == '100755' else 0o644)
            before = A / 'own-metric-collector-baseline-inventory-v1.json'
            assert not before.exists()
            before.write_text(json.dumps(dict(official_pti_commit=REVISION, complete=True, files=hashes,
                modes=modes, submodule_gitlinks=0, source_snapshot='Every tracked Git blob verified from its content; clean official checkout copied without .git or build objects.'), indent=2) + '\n')
            stdout, stderr = run('apply-reviewed-status-patch',
                ['/usr/bin/patch', '--batch', '--fuzz=0', '-p1', '-i', str(D / 'pti-query-status.patch')], source)
            assert not re.search(r'offset|fuzz|reject|failed', stdout + stderr, re.I)
            expected = json.loads((D / 'patched-source-sha256.json').read_text())
            actual = {str(p.relative_to(source)): pin(p)['sha256'] for p in source.rglob('*') if p.is_file()}
            assert actual == {**hashes, **expected}
            for name, h in hashes.items():
                assert pin(PTI / name)['sha256'] == h
            after = A / 'own-metric-collector-patched-inventory-v1.json'
            assert not after.exists()
            after.write_text(json.dumps(dict(official_pti_commit=REVISION, complete=True, files=actual,
                patch=record['patch'], only_declared_changes=True), indent=2) + '\n')
            record.update(status='PREPARED_SOURCE_ONLY', complete=True, baseline_inventory=pin(before),
                patched_inventory=pin(after), patch_apply_fuzz=0, patch_apply_offset=0,
                original_source_files=len(hashes), patched_source_files=len(actual),
                source_bytes=sum(p.stat().st_size for p in source.rglob('*') if p.is_file()))
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(helper.closed(c) and c['exit_code'] == 0 for c in owner.commands)
            for c in owner.commands:
                for old in c['owners'] + [c['direct_identity']]:
                    now = helper.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
            persist()
            target.write_bytes((root / 'record.json').read_bytes())
        print(json.dumps(dict(receipt=pin(target), status=record['status'], complete=record['complete'], GPU_executed=False)))
        return 0 if record['complete'] and record['ownership_closed'] else 1


if __name__ == '__main__':
    sys.exit(main())
