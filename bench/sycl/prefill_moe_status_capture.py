"""Qualify a sparse host clock for the V2 MoE status boundary."""

import fcntl
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_kv_upload_pti as fixture
import ur_call_subscriber as subscriber
from engine_once import native_environment

bank, basis, require = fixture.bank, fixture.basis, fixture.require
owner_module = fixture.owner_module
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'prefill_moe_status_capture_v2'
ROOT = bank.B / KEY.replace('_', '-')
TIMELINE = '4bc26aaa9cbdd8d4084336826361aae7bb886449'
CLOCK_COMPARISON = 'prefill_kv_upload_model_pti_v1'
BUDGET = dict(wall=100, cpu=100, rss_cap=4 << 30, text_cap=8 << 20,
              file_cap=16 << 20, total_cap=64 << 20)
SOURCES = (THIS, subscriber.SOURCE, subscriber.DRIVER_SOURCE)
ACTIONS = ('prepare', 'build', 'host', 'control', 'observed')
LAUNCHES = ('urEnqueueKernelLaunch', 'urEnqueueKernelLaunchWithArgsExp')
LINK_INPUTS = ('Scrt1.o', 'crt1.o', 'crti.o', 'crtbegin.o', 'crtbeginS.o', 'crtend.o', 'crtendS.o',
    'crtn.o', 'libstdc++.so', 'libgcc.a', 'libgcc_s.so', 'libgcc_s.so.1', 'libm.so', 'libm.so.6',
    'libmvec.so.1', 'libc.so', 'libc.so.6', 'libc_nonshared.a', 'libpthread.a', 'ld-linux-x86-64.so.2')
PLAN = dict(order=list(ACTIONS), fixture_arg='--smoke',
    scope='Use unchanged native V2 device fixture and Intel 2026.2 XPTI. Compare the default '
          'observer on CPU, exercise sparse filtering on synthetic CPU callbacks, then run the '
          'same small device workload with and without sparse host observation.',
    acceptance='Exact CPU filter and application output, actual paired 4-byte copy and wait/launch '
               'coverage on the native device route, parameter validation, owned exit zero and '
               'no fresh driver fault. No GPU timing, model, or diagnosis of the original PTI gap.',
    sdk_selection='Use the pinned SDK launch-with-arguments API. Keep the first compile refusal at 1dd8688fd.')


def identity(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=bank.sha(path))


def committed(key):
    return basis.validation.committed(bank.RESULT, (key,))


def device_basis():
    native, timing, _, proof = fixture.capacity.prerequisites(fixture.capacity.upload.experiment('store-time'))
    trace = committed(CLOCK_COMPARISON)
    require(trace == bank.previous.historic(TIMELINE, CLOCK_COMPARISON) and trace['stage'] == 'closed'
            and trace['decision']['normal_route_attribution_qualified']
            and trace['binary'] == native['binaries']['strata']
            and trace['native_source_commit'] == native['native_source_commit'], 'closed V2 timeline required')
    return dict(native_source_commit=native['native_source_commit'], native_build_sha256=proof['build_sha256'],
                engine=native['binaries']['strata'], binary=native['binaries']['sycl_expert_bank_test'],
                native_validation_probe=timing['native_validation_probe'], driver=timing['driver'],
                boot=timing['boot'], environment=timing['environment_template'], timeline_commit=TIMELINE)


def pins(revision):
    comparison = fixture.capacity.upload.experiment('store-time')
    local = [*SOURCES, 'tools/sycl/ur_call_subscriber.py', 'tools/sycl/measurement_owner.py',
             fixture.THIS, fixture.capacity.THIS]
    return comparison.pins(revision) | bank.stage.source.pin(bank.P, local)['source_sha256']


def compact(row):
    inputs = dict(row['device'])
    inputs['environment_sha256'] = bank.signature(inputs.pop('environment'))
    return dict(row, device=inputs, sources=dict(count=len(row['sources']), sha256=bank.signature(row['sources'])),
                compiler_files=dict(count=len(row['compiler_files']), sha256=bank.signature(row['compiler_files'])))


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    data = bank.previous.read(bank.RESULT)
    data[KEY] = compact(row)
    bank.stage.write(bank.RESULT, data)


def named_deps(path):
    values = path.read_text().replace('\\\n', ' ').split(':', 1)[1]
    require('\\' not in values, 'unrecognised dependency-file escaping')
    return {str(Path(name).resolve()) for name in values.split()}


def checked_command(owner, name, command, env, *, output_error_ok=False):
    entry, out, err = owner.run(name, command, env, bank.P, **BUDGET)
    require(owner_module.closed(entry) and entry['exit_code'] == 0,
            name + ' failed' + (': ' + err.read_text()[:2048] if err.exists() else ''))
    if not output_error_ok:
        require(not err.stat().st_size, 'unexpected diagnostics in ' + name)
    return entry, out, err


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'fresh capture only')
    revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
    source = pins(revision)
    device = device_basis()
    loader = Path(device['native_validation_probe']['loader']['path']).resolve()
    sdk = loader.parent.parent
    dispatcher = identity((loader.parent / 'libxptifw.so.1.0.1').resolve())
    historical = subscriber.r.validation.committed(subscriber.RESULT, subscriber.BUFFER_KEY)
    tools = historical['compiler_tools']
    require(all(bank.sha(p) == sha for p, sha in tools.items()), 'pinned system compiler tools changed')
    compiler = historical['command'][0]
    env = dict(historical['environment'], TMPDIR=str(ROOT / 'tmp'))
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'host-default', 'host-sparse', 'control', 'observed'):
        path = ROOT / name
        path.mkdir(mode=0o700)
        if name in ('control','observed'):
            (path / 'tmp').mkdir(); (path / 'cache').mkdir()
    owner = owner_module.Owner(ROOT / 'prepare-owned', lambda: None)
    common = [compiler, '-std=c++17', '-O2', '-pthread', '-DSTRATA_UR_WITH_ARGS_ONLY',
              '-I' + str(sdk / 'include')]
    selected = ((subscriber.SOURCE, 'libstrata_ur_status.so'), (subscriber.DRIVER_SOURCE, 'ur_call_guard_test'))
    commands, compiler_files = {}, dict(tools)
    # Enumerate the current SDK and system dependencies for this recipe.
    compiler_files[dispatcher['path']] = dispatcher['sha256']
    for name in LINK_INPUTS:
        path = Path(subprocess.check_output([compiler, '-print-file-name=' + name],
                                            env=env, text=True, timeout=5).strip())
        if path.is_absolute():
            compiler_files[str(path.resolve())] = bank.sha(path)
    for name, output in selected:
        gcc = common + (['-fPIC', '-shared', '-DXPTI_CALLBACK_API_EXPORTS'] if output.endswith('.so') else [])
        src = str(bank.P / name)
        scan, dep = ROOT / (output + '.scan.d'), ROOT / (output + '.d')
        checked_command(owner, 'deps-' + output, [p for p in gcc if p != '-shared'] +
                        ['-M', '-MT', 'probe', '-MF', str(scan), src], env)
        compiler_files.update({p: bank.sha(p) for p in named_deps(scan)})
        compiler_files[str(scan)] = bank.sha(scan)
        link = ['-Wl,-rpath,' + str(loader.parent), '-Wl,--no-undefined',
                '-Wl,--dependency-file=' + str(ROOT / (output + '.link.d'))]
        commands[output] = gcc + ['-MD', '-MF', str(dep), src, dispatcher['path'],
                                 *link, '-o', str(ROOT / output)]
    save(dict(stage='prepared', source_commit=revision, sources=source, plan=PLAN, budgets=BUDGET,
              device=device, commands=commands, compiler_files=compiler_files, dispatcher=dispatcher,
              compiler_environment=env, outcomes={}, active=False, ownership_closed=True, passed=False))
    print('Prepared exact 2026.2 compiler and fixture inputs; commit before build.', flush=True)


def current(row):
    require(compact(row) == committed(KEY) and pins(row['source_commit']) == row['sources']
            and row['plan'] == PLAN and row['budgets'] == BUDGET
            and all(bank.sha(path) == sha for path, sha in row['compiler_files'].items()),
            'commit exact observed source, compiler inputs and planned stage')
    require(all(identity(x['path']) == x for x in row.get('artifacts', {}).values()),
            'built observer or CPU peer changed')


def selected_calls(text, pid):
    summary = subscriber.parse_calls(text, pid, intervals=True)
    spans = summary['intervals']
    begins = {e['instance']: e['metadata']['thread_ns'] for e in map(json.loads, text.splitlines()) if e['kind'] == 'begin'}
    for call in spans:
        call['cpu_begin_ns'] = begins[call['instance']]
        call['cpu_end_ns'] = call['metadata']['thread_ns']
        require(0 <= call['cpu_begin_ns'] <= call['cpu_end_ns'], 'invalid thread-clock span')
    return spans


def observer_environment(row, root):
    library = row['artifacts']['libstrata_ur_status.so']['path']
    return dict(XPTI_TRACE_ENABLE='1', XPTI_SUBSCRIBERS=library,
                XPTI_FRAMEWORK_DISPATCHER=row['dispatcher']['path'], STRATA_UR_CALL_STATUS_ONLY='1',
                STRATA_UR_CALL_RECORD_LIMIT='8192', STRATA_UR_CALL_LOG=str(root / 'calls.jsonl'))


def host_check(row, owner):
    exe = row['artifacts']['ur_call_guard_test']['path']
    for name, case in (('host-default','metadata'), ('host-sparse','status')):
        root = ROOT / name
        env = dict(row['compiler_environment'], **observer_environment(row, root))
        env.update(LD_PRELOAD=env['XPTI_SUBSCRIBERS'], STRATA_UR_CALL_STATUS_ONLY=str(int(case == 'status')))
        entry, out, _ = checked_command(owner, name, [exe, case], env)
        capture = (root / 'calls.jsonl').read_text()
        if case == 'metadata':
            result = subscriber.metadata_fixture(capture, entry['pid'])
        else:
            calls = selected_calls(capture, entry['pid'])
            require(json.loads(out.read_text()) == {'status_fixture':'PASS','simulated_calls':8}
                    and [p['name'] for p in calls] == ['urKernelCreate', 'urEnqueueUSMMemcpy',
                        'urQueueFinish','urEnqueueKernelLaunchWithArgsExp']
                    and all(p['status'] == 0 for p in calls)
                    and calls[0]['metadata']['half_dequant'] == 1
                    and calls[1]['metadata']['bytes'] == 4
                    and all(p['metadata']['queue'] == 0x101 for p in calls[1:])
                    and calls[-1]['metadata']['kernel'] == calls[0]['metadata']['created_kernel'],
                    'sparse CPU callback selection differs')
            result = dict(simulated_calls=8, selected_calls=4, unrelated_queue_ignored=True)
        row['outcomes'][name] = dict(result, ownership_closed=True)


def device_launch(row, action):
    root = ROOT / action
    v2 = row['device']
    template = v2['environment']
    env = {k: v.replace(template['STRATA_MEASUREMENT_ROOT'], str(root)) for k,v in template.items()}
    env.update(TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'), LD_DEBUG='libs',
               UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_TRACING='level:info;flush:info;output:stderr',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr')
    require(env['ZES_ENABLE_SYSMAN'] == env['EnableDirectSubmission'] == '0'
            and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == env['NEOReadDebugKeys'] == '1'
            and env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1'
            and not any(k.startswith(('UNITRACE_', 'ZET_', 'XPTI_')) for k in env), 'V2 safety envelope changed')
    env = native_environment(v2, env)
    if action == 'observed':
        env.update(observer_environment(row, root))
    return [v2['binary']['path'], PLAN['fixture_arg']], env


def device_check(row, owner, action):
    root = ROOT / action
    v2 = row['device']
    require(device_basis() == v2 and identity(v2['binary']['path']) == v2['binary'], 'active fixture basis changed')
    command, env = device_launch(row, action)
    require(row['device_cases'][action] == dict(command=command, env_sha256=bank.signature(env)),
            'fixture command or environment changed')
    entry, out, err = checked_command(owner, action, command, env, output_error_ok=True)
    validation = basis.validation.preload_evidence(err.read_text(), entry['pid'])
    require(validation['loader'] == v2['native_validation_probe']['loader']['path']
            and 'expert_bank_fixture PASS' in out.read_text(), 'actual loader or fixture success differs')
    fixture.validation_log(root)
    result = dict(command=command, env_sha256=bank.signature(env), stdout_sha256=bank.sha(out), validation=validation)
    if action == 'observed':
        require(result['stdout_sha256'] == row['outcomes']['control']['stdout_sha256'], 'device numerical fixture changed')
        calls = selected_calls((root/'calls.jsonl').read_text(), entry['pid'])
        copies = [p for p in calls if p['name'] == 'urEnqueueUSMMemcpy']
        waits = [p for p in calls if p['name'] in ('urQueueFinish','urEventWait')]
        launches = [p for p in calls if p['name'] in LAUNCHES]
        require(copies and waits and launches and all(not p['status'] for p in (*copies, *waits, *launches))
                and all(p['metadata']['bytes'] == 4 for p in copies), 'incomplete native sparse call coverage')
        result['captured'] = dict(copies=len(copies), waits=len(waits), next_launches=len(launches))
    return result


def run(action):
    row = bank.previous.read(ROOT / 'record.json')
    order = ('prepared','built','host-checked','control')
    require(row['stage'] == order[ACTIONS.index(action)-1], 'select only the next frozen stage')
    current(row)
    result = {}
    case = ROOT / ('host-sparse' if action == 'host' else action)
    case.mkdir(exist_ok=True)
    cursor = next(t.removeprefix('-- cursor: ') for t in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines()
        if t.startswith('-- cursor: '))
    owner = owner_module.Owner(case/'owned', lambda: bank.stage.write(case/'execution.json',
            dict(action=action, active=bool(owner.active), commands=owner.commands)))
    try:
        if action == 'build':
            for name, command in row['commands'].items():
                checked_command(owner, 'compile-' + name, command, row['compiler_environment'])
                actual = named_deps(ROOT/(name+'.d'))
                require(actual == named_deps(ROOT/(name+'.scan.d'))
                        and actual <= row['compiler_files'].keys(), 'actual compiler input census changed')
                links = named_deps(ROOT/(name+'.link.d'))
                external = {path for path in links if not (Path(path).suffix == '.o' and
                            Path(path).is_relative_to(ROOT/'tmp'))}
                require(external <= row['compiler_files'].keys(),
                        'unplanned link inputs: ' + ', '.join(sorted(external - row['compiler_files'].keys())))
            row['artifacts'] = {name:identity(ROOT/name) for name in row['commands']}
            row['device_cases'] = {}
            for role in ('control', 'observed'):
                command, env = device_launch(row, role)
                row['device_cases'][role] = dict(command=command, env_sha256=bank.signature(env))
        elif action == 'host':
            host_check(row, owner)
        else:
            result = device_check(row, owner, action)
        result.update(complete=True, host_probe_accepted=True)
    except BaseException as error:
        result.update(error=repr(error), complete=False)
    finally:
        basis.validation.finish(result, owner, cursor, row['device']['boot'])
        passed = result['passed'] = bool(result['passed'] and result['complete']
                                           and result['driver_after'] == row['device']['driver'])
        stage = dict(build='built', host='host-checked', control='control', observed='closed')[action]
        row['outcomes'][action] = result
        row.update(stage=stage if passed else 'failed', active=bool(owner.active),
                   ownership_closed=result['ownership_closed'], passed=passed and action == 'observed')
        owner.persist()
        save(row)
    print({k:result[k] for k in ('passed','fresh_faults','captured','error') if k in result}, flush=True)
    require(passed, 'stop after the recorded stage outcome')


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ACTIONS, 'select one frozen stage')
    with (bank.B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        prepare() if action == 'prepare' else run(action)
