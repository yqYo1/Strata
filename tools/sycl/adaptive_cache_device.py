#!/usr/bin/env python3
"""Bounded fresh-device qualification for the fixed model build; repository MIT.

No model, performance or positive-swap qualification follows from these cases.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import adaptive_cache_model as built
import proc_observer
import runtime_probe as runtime
import ur_validation as validation

base, P, B, RESULT = built.base, built.P, built.B, built.RESULT
require, digest, write = base.require, base.digest, base.write
ROOT, KEY = built.ROOT / 'device-checks-v1', 'adaptive_cache_device_v1'
CONTROLLER = Path(__file__).resolve()


def compact(row):
    result = {k: v for k, v in row.items() if k not in ('controller_inputs', 'commands_executed')}
    result['controller_inputs'] = dict(files=len(row['controller_inputs']), aggregate_sha256=
        hashlib.sha256(json.dumps(row['controller_inputs'], sort_keys=True).encode()).hexdigest())
    return result


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def current(row):
    require(base.committed(RESULT)[KEY] == compact(row) and
            base.controls(row['source_commit'], controller=CONTROLLER) == row['controller_inputs'],
            'frozen device plan/controller changed')
    native = built.verify()
    require(row['build_record'] == dict(path=str(built.ROOT / 'record.json'),
            sha256=digest(built.ROOT / 'record.json')) and row['native_source_commit'] == native['source_commit'],
            'actual model build changed')
    reuse = row['fixture_reuse']['qualified_component_evidence']
    previous = json.loads(base.basis.build.git(P, 'show', reuse['commit'] + ':' + reuse['path']))
    require(all(hashlib.sha256(json.dumps(previous[key], sort_keys=True).encode()).hexdigest() == sha
            for key, sha in reuse['entries'].items()), 'original component reuse evidence changed')
    return native, built.fallback.retained.runtime.inventory(row['runtime_manifest'])


def unchanged_fixtures(native):
    old = built.fallback.verify()
    roots = (built.fallback.ROOT, built.ROOT)

    def normalized(value, root):
        return json.loads(json.dumps(value).replace(str(root), '<generation>'))

    dbs = [normalized(json.loads((root / 'build/compile_commands.json').read_text()), root) for root in roots]
    maps = [normalized(json.loads((root / 'compiler-inputs.json').read_text()), root) for root in roots]
    require(set(old['native_sources']) == set(native['native_sources']) and
            [n for n in old['native_sources'] if old['native_sources'][n] != native['native_sources'][n]] == [built.PROGRAM],
            'unplanned native change outside the program fix')
    require(dbs[0] == dbs[1] and set(maps[0]) == set(maps[1]) and
            [n for n in maps[0] if maps[0][n] != maps[1][n]] == ['<generation>/source/' + built.PROGRAM] and
            old['actual_link_inputs']['frozen_inputs'] == native['actual_link_inputs']['frozen_inputs'] and
            all(normalized(old[k], roots[0]) == normalized(native[k], roots[1]) for k in ('configure_argv','environment')),
            'fixture compiler/link/source recipe differs')
    names = [n for n in old['binaries'] if n != 'strata']
    require(len(names) == 8 and all(old['binaries'][n]['sha256'] == native['binaries'][n]['sha256'] and
            old['binaries'][n]['bytes'] == native['binaries'][n]['bytes'] for n in names), 'fixture ELF differs')
    data = base.committed(RESULT)
    for key in ('gdn16_model_component_device_v1', 'gdn16_model_component_speed_v1'):
        gate = data[key]
        require(gate['passed'] and gate['complete'] and gate['ownership_closed'] and not gate['active'] and
                not gate['fresh_faults'] and gate['native_source_commit'] == old['source_commit'] and
                gate['binary']['sha256'] == native['binaries'][built.fallback.COMPONENT]['sha256'],
                'original qualified production wrapper differs')
    evidence = dict(commit=base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip(),
        path=str(RESULT.relative_to(P)), entries={key: hashlib.sha256(json.dumps(data[key], sort_keys=True).encode()).hexdigest()
            for key in ('gdn16_model_component_device_v1','gdn16_model_component_speed_v1')})
    return dict(original_native_commit=old['source_commit'], current_native_commit=native['source_commit'],
        original_inputs_recovered=True, unchanged_fixture_count=8, unchanged_nonprogram_source_files=556,
        compiled_units=len(dbs[0]), normalized_compile_recipe_sha256=
            hashlib.sha256(json.dumps(dbs[1], sort_keys=True).encode()).hexdigest(),
        actual_compiler_content_changes=[built.PROGRAM], actual_link_inputs_equal=True,
        configure_and_build_environment_equal=True, all_fixture_ELFs_byte_identical=True,
        qualified_component_evidence=evidence,
        scope='Reuse unchanged component evidence only;new program/model/prefill-to-decode/lifecycle and performance checks remain.')


def prepare():
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'one fresh device cohort')
    native = built.verify(); reuse = unchanged_fixtures(native); probe = runtime.components()
    gate = base.committed(RESULT)['gdn16_model_component_device_v1']
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    ROOT.mkdir(mode=0o700)
    row = dict(stage='prepared', source_commit=revision,
        controller_inputs=base.controls(revision, controller=CONTROLLER),
        native_source_commit=native['source_commit'],
        build_record=dict(path=str(built.ROOT / 'record.json'), sha256=digest(built.ROOT / 'record.json')),
        runtime_manifest=native['runtime_manifest'], native_validation_probe=native['native_validation_probe'],
        fixture_reuse=reuse,
        boot=native['boot'], driver=native['driver'],
        environment=dict(gate['environment'], OMP_NUM_THREADS='1'),
        cases=[dict(name='host-canary', binary=probe['canary'], args=[], trace=False, gated=False),
            dict(name='submission', binary=native['binaries']['sycl_submission_test'],
                args=['--submission'], trace=True, gated=True)],
        processes=[], commands_executed=[], active=False, ownership_closed=True,
        passed=False, complete=False, GPU_executed=False, model_executed=False, inference_adopted=False,
        ordinary_runtime_teardown_qualified=False,
        budgets=dict(wall=600, cpu=2400, rss_cap=8 << 30, text_cap=8 << 20,
            file_cap=64 << 20, total_cap=64 << 20), whole_bytes=64 << 20, whole_wall_seconds=1800,
        plan=dict(order='Host-only preload canary,then small traced explicit/default/cross-queue copy and numerical/lifecycle submission fixture. Stop on first failure;no retries.',
            contract='All eight fixture ELFs and relevant native/compiler/link inputs match the qualified generation;reuse the unchanged component evidence. Fresh runtime submission check precedes the changed engine program. One B570 only;no multi-GPU-stage retirement claim.',
            safety='Immutable qualified copy-offload/direct-submission avoidance and queue settings;private activation preload held until OS exit. No model load,reset or deliberate device fault.',
            retention='Keep case outcomes,numerical coverage/maxima,source/ELF identity and fault/closure evidence. Retire successful API trace,logs and duplicate kernel rows after verification. Current nine-ELF fallback remains until model/lifecycle checks.'))
    save(row); print('Prepared two bounded fresh-device cases plus exact unchanged-fixture reuse;commit before execution.', flush=True)


def run():
    row = json.loads((ROOT / 'record.json').read_text()); _, providers = current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'one fresh ordered cohort')
    frozen, owner = copy.deepcopy(row), None
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True,
        timeout=15).splitlines() if v.startswith('-- cursor: '))
    deadline = time.monotonic() + row['whole_wall_seconds']

    def persist():
        write(ROOT / 'execution.json', dict(active=bool(owner and owner.active),
            commands=owner.commands if owner else [], completed_cases=len(row['processes'])))

    try:
        owner = base.supervision.Owner(ROOT / 'owned', persist)
        for case in row['cases']:
            folder = ROOT / case['name']; folder.mkdir(mode=0o700)
            for part in ('tmp', 'cache'): (folder / part).mkdir()
            asset = case['binary']; require(digest(asset['path']) == asset['sha256'], 'case ELF changed')
            env = dict(row['environment'], TMPDIR=str(folder / 'tmp'), XDG_CACHE_HOME=str(folder / 'cache'),
                STRATA_TEST_MAIN_GATES='1' if case['gated'] else '0',
                UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if case['trace'] else 'UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'ur.trace'),
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(folder / 'validation.log'))
            mapped, gates, observed = set(), [], proc_observer.ProcObserver()

            def observe(pid):
                raw = Path('/proc', str(pid), 'environ').read_bytes()
                if not raw: raise ProcessLookupError('empty retiring environment')
                actual = dict(v.split(b'=', 1) for v in raw.split(b'\0') if b'=' in v)
                require(all(actual.get(k.encode()) == v.encode() for k, v in env.items()) and
                        set(actual)-{k.encode() for k in env} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
                        'native environment changed')
                seen = runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
                if not seen: raise ProcessLookupError('empty retiring maps')
                require(seen <= set(providers) | {str(Path(asset['path']).resolve())}, 'unplanned native provider')
                mapped.update(seen)

            def monitor(io, entry):
                if case['gated']:
                    for expected in ('before-runtime', 'runtime-ready', 'before-exit'):
                        while True:
                            line = io.line(180)
                            if line.startswith('COMPONENT_READY '): break
                        require(line == 'COMPONENT_READY ' + expected, 'component gate differs')
                        observe(io.proc.pid); gates.append(expected); io.send(b'X')
                io.proc.stdin.close(); next_sample = 0.
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll(); require(time.monotonic() < deadline, 'whole device deadline')
                    if time.monotonic() >= next_sample:
                        if case['name'] != 'host-canary':
                            observed.sample(lambda: observe(io.proc.pid), lambda: bool(io.exit_watch.poll(0)))
                        require(base.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole device output budget')
                        next_sample = time.monotonic() + .1
                    if select.select([io.proc.stdout], [], [], .01)[0]: io.read(); io.pending.clear()
                io.exit(1)

            print('Device check:', case['name'], flush=True)
            entry, out, err = owner.run(case['name'], [asset['path'], *case['args']], env, P,
                interaction=monitor, **row['budgets'])
            text, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
            require(base.supervision.closed(entry) and entry['exit_code'] == 0,
                    'native case failed: ' + case['name'] + ': ' + stderr[-3000:])
            activation = validation.preload_evidence(stderr, entry['pid'])
            require(activation['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER'] and
                    (not (folder / 'validation.log').exists() or (folder / 'validation.log').stat().st_size == 0),
                    'activation/parameter validation failed')
            details = {}
            if case['name'] == 'host-canary':
                require(json.loads(text) == dict(pid=entry['pid'], initialized=0, checked=46,
                    expected=46, version=(1 << 32)-1, closed=0), 'host canary differs')
                details['host_only'] = True
            else:
                required = {env['UR_ADAPTERS_FORCE_LOAD'], env['ZE_ENABLE_ALT_DRIVERS'],
                    '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0',
                    env['LD_PRELOAD'], env['STRATA_UR_PROBE_EXPECTED_LOADER']}
                require(required <= mapped, 'actual provider evidence incomplete')
                if case['name'] == 'submission':
                    require('fixture_status=PASS mode=submission' in text and
                        [int(v) for v in re.findall(r'^case queued_kernels=(\d+) numerical=PASS guards=PASS event_timing=PASS kernel_profiling=PASS$', text, re.M)] == [1,119,120,121,241] and
                        'case ordinary_host_roundtrips=3 status=PASS' in text, 'submission coverage incomplete')
                    details = dict(queued_kernels=[1,119,120,121,241], host_roundtrips=3, gates=gates)
                row['GPU_executed'] = True
            case_check = dict(host_probe_accepted=True); validation.finish(case_check, owner, cursor, row['boot'])
            require(case_check['passed'] and case_check['driver_after'] == row['driver'], 'case fault/closure gate failed')
            row['processes'].append(dict(name=case['name'], passed=True, native_exit_code=0,
                binary=asset, validation=activation, observed_provider_count=len(mapped),
                proc_observation=observed.summary(), details=details))
            persist(); print('Passed', case['name'], flush=True)
        current(frozen)
        row.update(stage='closed', complete=True, host_probe_accepted=True)
    except BaseException as error:
        row.update(stage='failed', complete=False, host_probe_accepted=False, error=repr(error))
    finally:
        if owner: validation.finish(row, owner, cursor, row['boot'])
        if row.get('driver_after') != row['driver']: row['passed'] = False
        if not row.get('passed'): row['stage'] = 'failed'
        row.update(active=bool(owner and owner.active), commands_executed=owner.commands if owner else [])
        require(owner is None or owner.active is None, 'device ownership unresolved'); save(row)
    print(json.dumps({k: row.get(k) for k in ('stage','passed','complete','GPU_executed','fresh_faults','error')}), flush=True)
    require(row['passed'] and row['complete'], 'fresh-device checks failed')


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare','run'), 'use prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else run()
