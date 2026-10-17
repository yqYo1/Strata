#!/usr/bin/env python3
"""Relink host-only failure diagnostics against the identical qualified GPU object."""
import datetime
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location('observer_build_helpers', A / 'build_qsa_prefix16_v1.py')
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    root = h.B / 'main-qsa16-oracle-observer-build-v1'
    out = A / 'qsa16-oracle-observer-build-v1.json'
    assert not root.exists() and not out.exists()
    base_path = A / 'qsa-prefix16-build-v1.json'
    base = json.loads(base_path.read_text())
    assert base['passed'] and base['complete'] and base['ownership_closed'] and not base['active']
    closure = A / 'qsa-prefix16-262144-failure-closure-v1.json'
    assert json.loads(closure.read_text())['physical_process_closure_verified']
    fixture = A / 'qsa16-oracle-observer-v1/fixture.cpp'
    frozen = dict(base['frozen_files'])
    for path in (base_path, closure, fixture, A / 'qsa16-oracle-observer-v1/manifest.json', Path(__file__), Path(base['object']['path'])):
        frozen[str(path)] = h.pin(path)
    owned = h.module('qsa16_observer_owner', A / 'direct_owner_transient_artifacts_v1.py')
    with (h.B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        (root / 'tmp').mkdir()
        env = dict(base['environment'], TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
        record = dict(schema='strata-qsa16-oracle-observer-build-v1', passed=False, complete=False,
                      active=False, GPU_executed=False, commands=[], frozen_files=frozen,
                      identical_GPU_object=base['object'], products={}, environment=env,
                      scope='Only host observer/explicit numerical-failure cleanup footer added. Original math, input, assertion and candidate GPU object preserved; compile/link only.')
        owner = None

        def save():
            record['active'] = bool(owner and owner.active is not None)
            out.write_text(json.dumps(record, indent=2) + '\n')

        owner = owned.Owner(root / 'owned', save)
        record['commands'] = owner.commands

        def verify():
            for path, expected in frozen.items():
                assert h.pin(path) == expected, path

        def run(label, argv):
            command, stdout, stderr = owner.run(label, argv, env, h.P, wall=300, cpu=900,
                rss_cap=8 << 30, text_cap=64 << 20, file_cap=128 << 20, total_cap=1 << 30)
            assert owned.closed(command) and command['exit_code'] == 0, stderr.read_text()[-4000:]
            return stdout.read_text()

        save()
        try:
            verify()
            original = next(c for c in base['commands'] if c['label'] == 'link-timing')['argv']
            argv = list(original)
            original_fixture = base['products']['timing']['fixture']['path']
            assert argv.count(original_fixture) == 1
            argv[argv.index(original_fixture)] = str(fixture)
            exe = root / 'sycl_qsa16_oracle_observer'
            argv[argv.index('-o') + 1] = str(exe)
            run('link-observer', argv)
            dynamic = run('readelf', ['/usr/bin/readelf', '-d', str(exe)])
            verify()
            record['products']['timing'] = dict(binary=h.pin(exe), dynamic=dynamic, fixture=h.pin(fixture))
            record.update(binary=h.pin(exe), passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(owned.closed(c) for c in owner.commands)
            record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save()
        print(json.dumps(dict(passed=record['passed'], receipt=str(out), error=record.get('error'))), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
