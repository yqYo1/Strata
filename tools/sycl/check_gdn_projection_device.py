#!/usr/bin/env python3
"""First bounded GPU check of the private FP32 GDN producer; MIT.

Main owns this two-process progression. Host descriptor ownership is a
prerequisite, not a substitute for the guarded device oracle. No speed,
production dispatch, sustained decode or physical-context qualification.
"""
import copy
import fcntl
import json
from pathlib import Path
import re
import select
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_gdn_projection_device as built
import check_gdn_projection_ownership as host
import runtime_probe as runtime

image, base = built.image, built.base
RESULT = built.RESULT
ROOT = image.ROOT / 'device-smoke-v1'
KEY = 'official_gdn_fp32_projection_device_smoke_v1'
require, digest, write = base.require, base.digest, base.write
SHAPES = [(1, 6144, 6144), (129, 10240, 10256)]
CASES = [dict(name='host-canary', args=[], trace=False),
         dict(name='fp32-smoke', args=['--smoke'], trace=True)]


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = base.compact(row)
    write(RESULT, data)


def prerequisites():
    """Recover and check the actual objects, inputs, owners and runtime pins."""
    data = base.committed(RESULT)
    refs, records = {}, {}
    for key, path in ((image.KEY, image.RECORD),
                      (host.KEY, host.ROOT / 'record.json'),
                      (built.KEY, built.ROOT / 'record.json')):
        record = json.loads(path.read_text())
        require(data[key] == base.compact(record) and record['passed'] and
                record['ownership_closed'] and not record['active'],
                'closed committed prerequisite required: ' + key)
        refs[key] = dict(path=str(path), sha256=digest(path))
        records[key] = record
    prior, proof, fixture = (records[k] for k in (image.KEY, host.KEY, built.KEY))
    require(prior['compiler_inputs_attested'] and
            prior['codegen_member']['manual_FP32_store_review_passed'] and
            proof['stage'] == 'host-ownership-passed' and
            proof['result']['checked_shapes'] == 65536 and
            proof['result']['GPU_submissions'] == 0 and
            not proof['result']['physical262144_qualified'] and
            fixture['stage'] == 'built-not-executed' and
            proof['compiled_native_implementation'] == fixture['native_implementation_commit'] ==
            prior['implementation_commit'], 'image/host/fixture scopes differ')
    original = image.admission(prior)
    allowed = image.build.admit(original)
    full, _ = image.build.source.sources()
    for group in full['dependencies']:
        for name, sha in group['source_sha256'].items():
            path = str((Path(group['root']) / name).resolve())
            require(digest(path) == sha, 'dependency changed: ' + path)
            allowed[path] = sha
    for record, controller in ((proof, Path(host.__file__).resolve()),
                               (fixture, Path(built.__file__).resolve())):
        require(base.controls(record['source_commit'], controller=controller) ==
                record['controller_inputs'], 'prerequisite controller changed')
        for name, sha in record['native_sources'].items():
            require(digest(base.P / name) == sha, 'fixture source changed: ' + name)
            allowed[str((base.P / name).resolve())] = sha
        asset = record['binary']
        require(digest(asset['path']) == asset['sha256'] and
                Path(asset['path']).stat().st_size == asset['bytes'], 'fixture ELF changed')
        require(all(base.supervision.closed(v) for v in record['commands_executed']),
                'prerequisite process ownership incomplete')
    for name in ('image.o', 'host.o'):
        asset = prior['outputs'][name]
        allowed[asset['path']] = asset['sha256']
    for record, field in ((prior, 'compiled_inputs'), (proof, 'actual_compiler_inputs'),
                          (fixture, 'actual_compiler_inputs')):
        asset = record[field]
        require(digest(asset['path']) == asset['sha256'], 'compiler-input manifest changed')
        actual = json.loads(Path(asset['path']).read_text())
        require(len(actual) == asset['files'] and all('/MistVVK/' not in p and
                allowed.get(p) == sha == digest(p) for p, sha in actual.items()),
                'actual compiler input not pinned')
    for record, field in ((proof, 'link_inputs'), (fixture, 'actual_link_inputs')):
        require(record[field]['uncovered_external_inputs'] == 0 and all(
                allowed.get(p) == sha == digest(p)
                for p, sha in record[field]['frozen_inputs'].items()), 'actual linker input changed')
    require(proof['image_proof'] == fixture['original_image_record'] == refs[image.KEY],
            'original image receipt changed')
    gate = data['official_prefill_strategies_device_v2']
    providers = image.build.retained.runtime.inventory(gate['runtime_manifest'])
    require(runtime.components()['library'] == gate['native_validation_probe']['library'],
            'native pre-main validation component changed')
    return dict(prerequisites=refs, binary=fixture['binary'], canary=gate['canary'],
                native_validation_probe=gate['native_validation_probe'],
                environment=gate['environment'], runtime_manifest=gate['runtime_manifest'],
                boot=gate['boot'], driver=gate['driver']), providers


def current(row):
    require(base.committed(RESULT)[KEY] == base.compact(row), 'commit exact device-smoke plan')
    require(base.controls(row['source_commit'], controller=Path(__file__).resolve()) ==
            row['controller_inputs'], 'device-smoke controller changed')
    snapshot, providers = prerequisites()
    require(all(row[k] == v for k, v in snapshot.items()) and row['cases'] == CASES,
            'frozen device-smoke inputs changed')
    require(base.guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'],
            'qualified boot/driver changed')
    return providers


def prepare():
    require(KEY not in base.committed(RESULT) and not ROOT.exists(), 'fresh first-device plan only')
    snapshot, _ = prerequisites()
    revision = base.basis.build.git(base.P, 'rev-parse', 'HEAD').decode().strip()
    ROOT.mkdir(mode=0o700)
    row = dict(snapshot, stage='prepared', source_commit=revision,
        controller_inputs=base.controls(revision, controller=Path(__file__).resolve()),
        native_sources={}, cases=copy.deepcopy(CASES), processes=[], active=False,
        ownership_closed=True, passed=False, complete=False, GPU_executed=False,
        GPU_execution_requested=False,
        model_executed=False, inference_adopted=False,
        budgets=dict(wall=180, cpu=600, rss_cap=4 << 30, text_cap=8 << 20,
                     file_cap=64 << 20, total_cap=64 << 20),
        whole_wall_seconds=420, whole_bytes=64 << 20,
        plan=dict(scope='Host canary followed once by two small traced GPU shapes;stop first failure,no retries.',
            acceptance='Pinned image/fixture/host proof/providers;genuine pre-main parameter validation;actual environment/maps;two native kernel launches;full dyadic FP32 and oneMKL output checks including guards;normal checked ownership,no fresh fault.',
            numerical='Exact dyadic inputs/products/sums;zero ULP FP32 oracle and unchanged oneMKL control;outputs must include values not representable in FP16.',
            shapes=SHAPES, limits='No full T1..8192 device coverage,non-dyadic/task/performance/decode/262144/RESTORE qualification.'),
        retention='Keep compact two-case results,trace kernel census and source/runtime identity. Retire successful raw trace/logs after verification;keep useful failure evidence only.')
    save(row)
    print('Prepared first bounded FP32 device smoke;commit before execution.', flush=True)


def launch_census(trace):
    created, native, total = {}, 0, 0
    for line in trace.splitlines():
        if '<--- urKernelCreate(' in line and 'UR_RESULT_SUCCESS' in line:
            match = re.search(r'\((_ZTS\S+)\).*?\.phKernel = .*?\((0x[0-9a-f]+)\)', line)
            if match:
                created[match[2]] = match[1]
        if '<--- urEnqueueKernelLaunch' in line and 'UR_RESULT_SUCCESS' in line:
            total += 1
            handle = re.search(r'\.hKernel = (0x[0-9a-f]+)', line)
            if handle and 'GdnProjectionFp32Kernel' in created.get(handle[1], ''):
                native += 1
    return dict(native_FP32_launches=native, all_successful_kernel_launches=total)


def analyze(case, stdout, stderr, pid, folder):
    if case['name'] == 'host-canary':
        require(json.loads(stdout) == dict(pid=pid, initialized=0, checked=46,
                expected=46, version=(1 << 32) - 1, closed=0), 'host-only canary differs')
        return dict(host_only=True)
    require(stdout.count('fixture_status=PASS;no model/speed/physical262144 qualification') == 1,
            'device fixture did not complete')
    rows = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    require([(v['T'], v['N'], v['ldy']) for v in rows] == SHAPES and all(
            v['dyadic_ULP'] == 0 and v['guards'] == v['oneMKL'] == v['owner_retirement'] == 'passed'
            and v['not_FP16_representable_outputs'] > 0 for v in rows), 'device oracle/shape coverage differs')
    for marker in ('case-start T=', 'native-complete', 'reference-complete'):
        require(stderr.count(marker) == 2, 'device progress evidence missing: ' + marker)
    census = launch_census((folder / 'ur.trace').read_text())
    require(census['native_FP32_launches'] == 2, 'actual new-kernel dispatch missing')
    return dict(cases=rows, kernel_census=census, full_device_domain_qualified=False)


def run():
    row = json.loads((ROOT / 'record.json').read_text())
    providers = current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'one fresh cohort only')
    frozen, owner, active_case, observed = copy.deepcopy(row), None, None, set()
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
    deadline = time.monotonic() + row['whole_wall_seconds']
    def persist():
        write(ROOT / 'execution.json', dict(processes=row['processes'], active_case=active_case,
            commands=owner.commands if owner else [], active=bool(owner and owner.active)))
    try:
        owner = base.supervision.Owner(ROOT / 'owned', persist)
        for case in row['cases']:
            active_case = case['name']
            folder = ROOT / active_case
            folder.mkdir(mode=0o700)
            for part in ('tmp', 'cache'):
                (folder / part).mkdir()
            env = dict(row['environment'], TMPDIR=str(folder / 'tmp'), XDG_CACHE_HOME=str(folder / 'cache'),
                UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if case['trace'] else 'UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'ur.trace'),
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(folder / 'validation.log'))
            asset = row['canary'] if active_case == 'host-canary' else row['binary']
            mapped, env_seen = set(), False
            def observe(pid):
                nonlocal env_seen
                actual = dict(v.split(b'=', 1) for v in
                    Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
                require(all(actual.get(k.encode()) == v.encode() for k, v in env.items()) and
                    set(actual) - {k.encode() for k in env} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
                    'actual native environment differs')
                seen = runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
                require(seen <= set(providers) | {str(Path(asset['path']).resolve())},
                        'unplanned executable map: ' + repr(seen - set(providers)))
                mapped.update(seen)
                env_seen = True
            def monitor(io, entry):
                io.proc.stdin.close()
                next_size = 0.
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    require(time.monotonic() < deadline, 'whole smoke wall budget')
                    try:
                        observe(io.proc.pid)
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                    except PermissionError:
                        require(io.exit_watch.poll(20), 'live native /proc access denied')
                    if time.monotonic() >= next_size:
                        require(base.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole smoke byte budget')
                        next_size = time.monotonic() + .25
                    if select.select([io.proc.stdout], [], [], .01)[0]:
                        io.read()
                        io.pending.clear()
                io.exit(1)
            row['GPU_execution_requested'] = bool(case['trace'])
            entry, out, err = owner.run(active_case, [asset['path'], *case['args']],
                env, base.P, interaction=monitor, **row['budgets'])
            stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
            require(base.supervision.closed(entry) and entry['exit_code'] == 0,
                    'native process failed: ' + active_case + ': ' + stderr[-3000:])
            activation = runtime.validation.preload_evidence(stderr, entry['pid'])
            require(activation['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER'], 'wrong pre-main loader')
            require(not (folder / 'validation.log').exists() or (folder / 'validation.log').stat().st_size == 0,
                    'final parameter validation is not quiet')
            details = analyze(case, stdout, stderr, entry['pid'], folder)
            if case['trace']:
                require(env_seen and {env['LD_PRELOAD'], env['STRATA_UR_PROBE_EXPECTED_LOADER'],
                    env['ZE_ENABLE_ALT_DRIVERS'], '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0'} <= mapped,
                    'actual device provider observation incomplete')
                row['GPU_executed'] = True
            check = dict(host_probe_accepted=True)
            runtime.validation.finish(check, owner, cursor, row['boot'])
            require(check['passed'] and check['driver_after'] == row['driver'], 'fault/ownership gate failed')
            row['processes'].append(dict(name=active_case, passed=True, exit_code=entry['exit_code'],
                pre_main_parameter_validation=True, quiet_final_validation=True,
                environment_observed=env_seen, ownership_closed=True, details=details))
            observed.update(mapped)
            persist()
            print('Passed ' + active_case, flush=True)
        current(frozen)
        row.update(stage='smoke-passed', complete=True, host_probe_accepted=True)
    except BaseException as error:
        row.update(stage='failed', complete=False, host_probe_accepted=False,
                   failed_case=active_case, error=repr(error))
    finally:
        if owner:
            runtime.validation.finish(row, owner, cursor, row['boot'])
        if row.get('driver_after') != row['driver']:
            row['passed'] = False
        if not row.get('passed'):
            row['stage'] = 'failed'
        trace = ROOT / 'fp32-smoke/ur.trace'
        if trace.exists():
            row['GPU_executed'] = launch_census(trace.read_text(errors='replace'))['native_FP32_launches'] > 0
        row.update(active=bool(owner and owner.active), observed_provider_count=len(observed),
                   ordinary_runtime_teardown_qualified=False)
        require(owner is None or owner.active is None, 'device smoke owner unresolved')
        save(row)
    print(json.dumps({k:row.get(k) for k in ('stage', 'passed', 'GPU_executed', 'fresh_faults', 'error')}), flush=True)
    require(row['passed'], 'first FP32 GPU smoke failed')


if __name__ == '__main__':
    with (base.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']:
            prepare()
        elif sys.argv[1:] == ['run']:
            run()
        else:
            raise RuntimeError('check_gdn_projection_device.py prepare|run')
