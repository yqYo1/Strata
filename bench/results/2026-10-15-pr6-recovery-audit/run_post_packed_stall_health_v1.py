#!/usr/bin/env python3
"""Main-owned small traced GPU checks after the stopped packed diagnostic."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEV = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    out = A / 'post-packed-stall-health-v1.json'
    root = B / 'main-post-packed-stall-health-v1'
    assert not out.exists() and not root.exists()
    owner_path = A / 'direct_owner_transient_artifacts_v1.py'
    assert sha(owner_path) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owned = module('post_stall_owner', owner_path)
    build = json.loads((A / 'own-native16-prefill-build-v2.json').read_text())
    good_path = A / 'own-qsa-bf16-prefix-tilepack-profile-census-bf16-prefix-optimized-r0-v2.json'
    failed_path = A / 'own-qsa-bf16-prefix-tilepack-profile-census-bf16-prefix-tilepack-r0-v2.json'
    good = json.loads(good_path.read_text())
    failed = json.loads(failed_path.read_text())
    for r in (good, failed):
        assert r['ownership_closed'] and not r['active'] and r['fresh_faults'] == [] and not r['driver_dump_exists']
        for command in r['commands']:
            assert owned.closed(command)
            for identity in command['owners'] + [command['direct_identity']]:
                now = owned.identity(identity['pid'])
                assert now is None or now['start_ticks'] != identity['start_ticks']
    assert good['passed'] and good['complete'] and not failed['passed']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == good['boot_id'] == build['boot_id']
    assert all(sha(Path(f['path'])) == f['sha256'] for f in good['runtime_files'])
    package = A / 'own-vllm-xe2-qualification-v1'
    meta = json.loads((package / 'manifest.json').read_text())
    assert sha(package / 'manifest.json') == '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    helper = package / 'main_controller.py'
    assert sha(helper) == meta['package_files']['main_controller.py']['sha256']
    health = module('post_stall_guard', helper)
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        health.no_model_owners(meta['known_model_files'])
        health.driver_guard()
        root.mkdir(mode=0o700)
        keep = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'MKLROOT', 'OMP_NUM_THREADS',
                'ONEAPI_DEVICE_SELECTOR', 'UR_ADAPTERS_FORCE_LOAD', 'SYCL_CACHE_PERSISTENT',
                'NEOReadDebugKeys', 'EnableDirectSubmission', 'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD')
        env = {key: good['environment'][key] for key in keep}
        env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                   UR_LOG_TRACING='level:info;flush:info;output:stderr',
                   UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                   UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                   UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
                   STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0', STRATA_SYCL_PREFILL_SAME_QUEUE='1',
                   ZES_ENABLE_SYSMAN='0', XDG_CACHE_HOME=str(root / 'cache'))
        record = dict(schema='strata-post-packed-stall-health-v1', passed=False, complete=False,
                      GPU_executed=False, commands=[], cases=[], environment=env, boot_id=good['boot_id'],
                      recipe_sha256=sha(__file__), previous_good_sha256=sha(good_path),
                      stopped_profile_sha256=sha(failed_path), runtime_files=good['runtime_files'],
                      scope='Small runtime/native16 GPU checks only, after stopping the packed diagnostic. No model, speed, full-context or RESTORE qualification.',
                      retention=dict(root=str(root), budget_bytes=32 << 20, consumer='Main GPU recovery without reset',
                                     review_point='At closure: retain compact PASS/fault and API counts; retire successful full API logs.'))

        def save():
            out.write_text(json.dumps(record, indent=2) + '\n')

        owner = owned.Owner(root / 'owned', save)
        record['commands'] = owner.commands

        def run(label, argv, child_env=env):
            c, so, se = owner.run(label, argv, child_env, P, wall=90, cpu=180,
                                  rss_cap=3 << 30, text_cap=8 << 20,
                                  total_cap=32 << 20, file_cap=8 << 20)
            assert owned.closed(c) and c['exit_code'] == 0, (label, c.get('exit_code'), str(se))
            return so.read_text(), se.read_text()

        cursor = None
        save()
        try:
            raw, _ = run('journal-before', ['/usr/bin/journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', raw)[1]
            record['journal_before_cursor'] = cursor
            binary_root = Path(build['binary']).parent
            for name, args, proof in (
                ('sycl_runtime_test', [], '64 changing cross-queue handoffs'),
                ('sycl_native16_prefill_test', ['--smoke'], 'native16_prefill PASS')):
                binary = binary_root / name
                assert sha(binary) == build['target_pins'][name]
                record['GPU_executed'] = True
                so, se = run(name, [str(binary), *args])
                assert proof in so
                record['cases'].append(dict(name=name, binary=str(binary), binary_sha256=sha(binary),
                                             stdout=so, stderr_sha256=hashlib.sha256(se.encode()).hexdigest(),
                                             stderr_bytes=len(se.encode()), passed=True))
                health.driver_guard()
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            try:
                assert owner.active is None
                if cursor:
                    raw, _ = run('journal-final', ['/usr/bin/journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    rows = [json.loads(s) for s in raw.splitlines() if s.strip()]
                    record['fresh_faults'] = [r for r in rows if FAULT.search(str(r.get('MESSAGE', '')))]
                record['ownership_closed'] = all(owned.closed(c) for c in owner.commands)
                record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
                record['driver_dump_exists'] = (DEV / 'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*')))
                assert record['ownership_closed'] and record['boot_unchanged'] and not record['driver_dump_exists'] and record['fresh_faults'] == []
            except BaseException as error:
                record.update(closure_error=repr(error), passed=False, complete=False)
            record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save()
        print(json.dumps(dict(passed=record['passed'], complete=record['complete'], error=record.get('error'), receipt=str(out))), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
