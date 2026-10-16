#!/usr/bin/env python3
"""Close mirror retirement after the separate original-license copy remained."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = A / 'integrated-nll-source-mirrors-retention-v1.json'
        r = json.loads(target.read_text())
        require(r['status'] == 'planned' and len(r['deletions']) == 339, 'exact interrupted directory cleanup')
        require(sha(A / 'retire_integrated_nll_source_mirrors_v1.py') == r['recipe_sha256'], 'original recipe retained unchanged')
        for name in ('own-dense-prefill-nll-build-v3.json', 'own-dense-prefill-nll-smoke-v3.json', 'own-dense-prefill-nll-smoke-v4.json'):
            receipt = json.loads((A / name).read_text())
            require(not receipt['active'], 'no active integration job')
            for c in receipt['commands']:
                require(module.closed(c), 'terminal command')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'owner absent')
        require(all(not Path(v['path']).exists() for v in r['deletions']), 'all preverified339 files already retired')
        s = A / 'own-dense-prefill-nll-v3'
        remaining = [p for root in (s / 'base', s / 'files') for p in root.rglob('*') if p.is_file()]
        license_path = s / 'base/LICENSE'
        require(remaining == [license_path], 'only separate original license copy remains')
        data = license_path.read_bytes()
        expected = '949b9b0aada7476006469e58ee4ffb0e0f68c2d9137e90a356757b61a09d37f0'
        require(sha(license_path) == expected and sha(P / 'LICENSE') == expected, 'root retains identical complete license')
        require(subprocess.check_output(['git','show',r['integrated_commit'] + ':LICENSE'], cwd=P) == data, 'license preserved in pushed Git source')
        oid = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        r['deletions'].append(dict(path=str(license_path), bytes=len(data), sha256=expected,
                                  recovery_commit=r['integrated_commit'], recovery_path='LICENSE',
                                  git_blob=oid, status='planned'))
        r.update(directory_cleanup_interruption='All339 preverified mirrors had been unlinked; root.rmdir stopped with OSError39 because separately listed base/LICENSE was outside production_source_snapshot. No inference result changed.',
                 completion_recipe_sha256=sha(Path(__file__)))
        target.write_text(json.dumps(r, indent=2) + '\n')
        require(sha(license_path) == expected, 'license identity before unlink')
        license_path.unlink()
        for root in (s / 'base', s / 'files'):
            for p in sorted((p for p in root.rglob('*') if p.is_dir()), key=lambda p:len(p.parts), reverse=True):
                p.rmdir()
            root.rmdir()
        for item in r['deletions']:
            item['status'] = 'removed'
        r.update(status='complete', files=len(r['deletions']), removed_logical_bytes=sum(v['bytes'] for v in r['deletions']))
        target.write_text(json.dumps(r, indent=2) + '\n')
        print(json.dumps({k:r[k] for k in ('status','files','removed_logical_bytes')}))


if __name__ == '__main__':
    main()
