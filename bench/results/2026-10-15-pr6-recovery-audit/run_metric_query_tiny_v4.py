#!/usr/bin/env python3
"""Main externally bounds one reviewed tiny controller. No automatic next run."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    root = B / 'main-metric-query-tiny-v4-supervisor-v1'
    assert not root.exists()
    root.mkdir(mode=0o700)
    source = A.parents[2] / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert sha(source) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('metric_query_supervisor_owner', source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    record = dict(active=False, commands=[], schema='strata-metric-query-controller-supervision-v1')
    owner = None

    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

    owner = m.Owner(root / 'owned', persist)
    record['commands'] = owner.commands
    package = '1816ff99b174030b6586283fde10f28f49afb3eca008a72de7a5b3e118167cff'
    seal = A / 'own-metric-query-tiny-main-seal-v1.json'
    assert sha(seal) == '5ba737d125623305cd68b859dc534ab67aa6446f52c3eb62f72061ee701424d9'
    env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
    c, out, err = owner.run('tiny-controller', ['/usr/bin/python3', '-I', '-B',
        str(A / 'own-metric-query-tiny-v4/main_controller.py'), '--seal', str(seal),
        '--seal-sha256', sha(seal), '--reviewed-package-sha256', package, '--run-id', 'v1'],
        env, A.parents[2], wall=315, cpu=360, rss_cap=12 << 30, text_cap=4 << 20,
        total_cap=16 << 20, file_cap=4 << 20)
    record.update(ownership_closed=m.closed(c), exit_code=c['exit_code'],
        stdout=out.read_text(), stderr=err.read_text())
    persist()
    target = A / 'own-metric-query-tiny-supervision-v1.json'
    assert not target.exists()
    target.write_bytes((root / 'record.json').read_bytes())
    print(json.dumps(dict(controller_exit=c['exit_code'], closed=m.closed(c),
        stdout=record['stdout'], stderr=record['stderr'][-4000:], receipt=str(target))))
    return c['exit_code'] if m.closed(c) else 99


if __name__ == '__main__':
    sys.exit(main())
