#!/usr/bin/env python3
"""Delete closed duplicated evidence only after exact receipts are in Git."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = P / 'bench/results/2026-10-14-primary-native-runtime'
BUILD = B / 'main-primary-native-build-root-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
        def committed(path):
            relative = str(path.relative_to(P))
            data = subprocess.check_output(['git', 'show', revision + ':' + relative], cwd=P)
            assert path.read_bytes() == data, relative
            return data
        receipts = json.loads(committed(D / 'receipts-index.json'))
        for item in receipts:
            encoded = committed(D / item['path'])
            decoded = gzip.decompress(encoded)
            assert sha(encoded) == item['gzip_sha256'] and sha(decoded) == item['sha256']
            assert len(decoded) == item['bytes']
        spec = importlib.util.spec_from_file_location('publication', D / 'publish_evidence.py')
        publication = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(publication)
        roots = []
        for root in publication.ROOTS:
            raw = B / root / 'record.json'
            saved = gzip.decompress(committed(D / 'receipts' / root / 'record.json.gz'))
            assert raw.read_bytes() == saved
            record = json.loads(saved)
            assert not record['active']
            assert all(command['direct_child_reaped'] and command['session_empty'] and
                       not command.get('survivors') for command in record['commands'])
            if root != BUILD.name:
                roots.append(B / root)
        roots.append(BUILD / 'owned')
        roots += [BUILD / ('engine-build-owned' + ('' if i == 1 else '-v' + str(i)))
                  for i in range(1, 14)]
        # Individual closed research reports now have one durable copy in Git.
        research = json.loads(committed(D / 'research-index.json'))
        reports = []
        for item in research['reports']:
            original = Path(item['original_path'])
            saved = committed(D / item['path'])
            assert original.read_bytes() == saved and sha(saved) == item['sha256']
            reports.append(original)
        manifest = []
        def add(path):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                manifest.append(dict(path=str(path), type='symlink', target=target,
                                     bytes=0, sha256=sha(os.fsencode(target))))
            else:
                assert stat.S_ISREG(info.st_mode), path
                manifest.append(dict(path=str(path), type='file', bytes=info.st_size,
                                     sha256=sha(path.read_bytes())))
        for root in roots:
            assert root.exists() and root.is_relative_to(B) and not root.is_symlink(), root
            for parent, directories, names in os.walk(root, followlinks=False):
                for name in list(directories):
                    path = Path(parent) / name
                    if path.is_symlink():
                        directories.remove(name)
                        add(path)
                for name in names:
                    add(Path(parent) / name)
        for report in reports:
            add(report)
        for item in manifest:
            item.update(evidence_revision=revision, replacement='bench/results/2026-10-14-primary-native-runtime/',
                        reason='Closed run/research; exact receipts, source, commands, env, all individual results and representative diagnostics verified in Git. No current raw consumer.')
        data = b''.join((json.dumps(item) + '\n').encode() for item in manifest)
        encoded = gzip.compress(data, mtime=0)
        (D / 'log-deletion-manifest.jsonl.gz').write_bytes(encoded)
        assert gzip.decompress(encoded) == data
        record = dict(complete=False, evidence_revision=revision, owner='Main',
                      manifest=dict(path='log-deletion-manifest.jsonl.gz', records=len(manifest),
                                    sha256=sha(encoded), decoded_sha256=sha(data),
                                    logical_bytes=sum(item['bytes'] for item in manifest)),
                      original_result_status_unchanged=True, serialized_with_GPU_build_test=True,
                      remaining_build=dict(path=str(BUILD / 'build'), budget_bytes=4 << 30,
                          consumer='next primary model/HTTP baseline and native feature integration',
                          review_point='after baseline qualification or binary/toolchain supersession'),
                      source_toolchain_retained=True, models_and_service_GPU_state_untouched=True)
        def persist():
            (D / 'log-retirement.json').write_text(json.dumps(record, indent=2) + '\n')
        persist()
        for item in manifest:
            path = Path(item['path'])
            actual = os.fsencode(os.readlink(path)) if path.is_symlink() else path.read_bytes()
            assert sha(actual) == item['sha256'], path
        for root in roots:
            shutil.rmtree(root)
        for report in reports:
            report.unlink()
        assert (BUILD / 'build/strata').is_file()
        record['complete'] = True
        persist()
    print(json.dumps(dict(retired_files=len(manifest), logical_bytes=record['manifest']['logical_bytes'],
                          primary_build_retained=True)), flush=True)


if __name__ == '__main__':
    main()
