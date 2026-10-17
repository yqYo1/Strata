#!/usr/bin/env python3
"""Run the built routing fixture serially; first small process is API-traced."""
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
MAIN = Path('/home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEV = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    out = A / 'moe-window-fixture-GPU-v1.json'
    root = B / 'main-moe-window-fixture-GPU-v1'
    assert not out.exists() and not root.exists()
    owner_path = A / 'direct_owner_transient_artifacts_v1.py'
    assert sha(owner_path) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owned = module('moe_fixture_owner', owner_path)
    build_path = A / 'moe-window-engine-build-v1.json'
    build = json.loads(build_path.read_text())
    assert build['passed'] and build['complete'] and not build['active'] and build['ownership_closed']
    assert build['fresh_faults'] == [] and not build['driver_dump_exists'] and build['boot_unchanged']
    for command in build['commands']:
        assert owned.closed(command) and command['exit_code'] == 0
    health_path = MAIN / 'post-packed-stall-health-v1.json'
    health = json.loads(health_path.read_text())
    assert health['passed'] and health['complete'] and health['ownership_closed'] and health['fresh_faults'] == []
    assert health['boot_id'] == build['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert all(sha(Path(v['path'])) == v['sha256'] for v in health['runtime_files'])
    package = A / 'own-vllm-xe2-qualification-v1'
    meta = json.loads((package / 'manifest.json').read_text())
    assert sha(package / 'manifest.json') == '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    helper = package / 'main_controller.py'
    assert sha(helper) == meta['package_files']['main_controller.py']['sha256']
    guard = module('moe_fixture_guard', helper)
    binary = build['products']['sycl_moe_window_test']
    assert sha(binary['path']) == binary['sha256']
    assert all(sha(path) == expected for path, expected in build['source_pins'].items())
    cases = [('smoke', 3), ('single', 1), ('unwind', 3), ('negative-source', 3),
             ('negative-slot', 3), ('negative-expert', 3), ('negative-down', 3),
             ('hot8193', 8193), ('hot16384', 16384), ('balanced16384', 16384), ('tail16383', 16383)]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.no_model_owners(meta['known_model_files'])
        guard.driver_guard()
        root.mkdir(mode=0o700)
        keep = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'MKLROOT', 'OMP_NUM_THREADS',
                'ONEAPI_DEVICE_SELECTOR', 'UR_ADAPTERS_FORCE_LOAD', 'SYCL_CACHE_PERSISTENT',
                'NEOReadDebugKeys', 'EnableDirectSubmission', 'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD')
        base = {key: health['environment'][key] for key in keep}
        base.update(STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0', STRATA_SYCL_PREFILL_SAME_QUEUE='1',
                    ZES_ENABLE_SYSMAN='0', XDG_CACHE_HOME=str(root / 'cache'))
        record = dict(schema='strata-moe-window-production-routing-GPU-v1', passed=False, complete=False,
                      active=False, GPU_executed=False, GPU_launch_attempted=False,
                      commands=[], cases=[], expected_cases=cases, binary=binary,
                      build_receipt=dict(path=str(build_path), sha256=sha(build_path)),
                      health_receipt=dict(path=str(health_path), sha256=sha(health_path)),
                      recipe_sha256=sha(__file__), boot_id=build['boot_id'],
                      scope='Synthetic production route/gather/combine and independent pre-combine oracle, guards and lease unwind only. No real expert GEMM/dequant/attention/model/state/context/RESTORE/speed qualification.',
                      retention=dict(root=str(root), budget_bytes=64 << 20,
                                     consumer='Main MoE GPU indexing/owner qualification',
                                     review_point='After closure: keep all case/counter/fault results; retire successful full API traces after extracting proof.'))

        owner = None

        def save():
            record['active'] = bool(owner and owner.active is not None)
            out.write_text(json.dumps(record, indent=2) + '\n')

        owner = owned.Owner(root / 'owned', save)
        record['commands'] = owner.commands

        def run(label, argv, env):
            c, so, se = owner.run(label, argv, env, P, wall=120, cpu=240,
                                  rss_cap=4 << 30, text_cap=8 << 20,
                                  total_cap=64 << 20, file_cap=8 << 20)
            assert owned.closed(c) and c['exit_code'] == 0, (label, c.get('exit_code'), str(se))
            return so.read_text(), se.read_text()

        def journal(label, cursor=None):
            argv = ['/usr/bin/journalctl', '-k', '-b', '--no-pager']
            argv += ['--after-cursor', cursor, '-o', 'json'] if cursor else ['-n', '0', '--show-cursor']
            so, _ = run(label, argv, base)
            return so

        cursor = None
        save()
        try:
            cursor = re.search(r'-- cursor: (\S+)', journal('journal-before'))[1]
            record['journal_before_cursor'] = cursor
            for name, rows in cases:
                assert sha(binary['path']) == binary['sha256']
                env = dict(base, UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                           UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                           UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                           UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
                if name == 'smoke':
                    env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                               UR_LOG_TRACING='level:info;flush:info;output:stderr')
                record['GPU_launch_attempted'] = True
                if record['GPU_executed'] is not True:
                    record['GPU_executed'] = None
                so, se = run(name, [binary['path'], '--case', name], env)
                lines = [line for line in so.splitlines() if line.startswith('moe_window case ')]
                assert len(lines) == 2 and 'moe_window PASS (synthetic route/gather/combine only;' in so
                samples = [{k: int(v) for k, v in re.findall(r'([a-z_]+)=([0-9]+)', line)} for line in lines]
                assert [sample['salt'] for sample in samples] == [1, 37]
                for sample in samples:
                    assert sample['rows'] == rows and sample['concentrated'] == int(name.startswith('hot'))
                    assert sample['stopped'] == int(name == 'unwind') and sample['checks_completed'] == int(name != 'unwind')
                counters = [{k: int(v) for k, v in re.findall(r'([a-z_]+)=([0-9]+)', line)}
                            for line in so.splitlines() if line.startswith('moe_window counters ')]
                assert len(counters) == (0 if name == 'unwind' else 2)
                record['cases'].append(dict(name=name, environment=env, samples=samples, counters=counters,
                                             stdout=so, stderr_sha256=hashlib.sha256(se.encode()).hexdigest(),
                                             stderr_bytes=len(se.encode()), passed=True))
                record['GPU_executed'] = True
                guard.driver_guard()
                raw = journal('journal-after-' + name, cursor)
                faults = [json.loads(line) for line in raw.splitlines() if line.strip() and FAULT.search(str(json.loads(line).get('MESSAGE', '')))]
                assert not faults, faults
                save()
                print(json.dumps(dict(case=name, passed=True, samples=samples, counters=counters)), flush=True)
            assert all(sha(path) == expected for path, expected in build['source_pins'].items())
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            try:
                assert owner.active is None
                if cursor:
                    raw = journal('journal-final', cursor)
                    record['fresh_faults'] = [json.loads(line) for line in raw.splitlines() if line.strip() and FAULT.search(str(json.loads(line).get('MESSAGE', '')))]
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
