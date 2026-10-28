#!/usr/bin/env python3
"""Own same-production-source gates for the isolated direct-K model build.

This controller never executes the model or accepts a performance claim.
The old component's vetted parsers/device checks are reused, not its retired
build admission, plans or selection decisions. Production is arm0 here.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_qsa_k_model as build
import check_qsa_k_loop as checks

P, B, ROOT, RESULT = build.P, build.B, build.ROOT, build.RESULT
require, digest, write = build.require, build.digest, build.write
ORDER = ['properties', 'id-smoke', 'id-boundary', 'oracle-smoke', 'oracle-component']


def output_folder(mode):
    # The first property attempt remains a failed, separately recorded run.
    return ROOT / ('properties-v2' if mode == 'properties' else mode)


def inputs(revision):
    # These are the only two local imports of this controller. Each producer
    # recursively pins its own imports; also pin this entrypoint itself.
    pins = dict(build.sources(revision), **checks.inputs(revision))
    path = Path(__file__).resolve()
    name = str(path.relative_to(P))
    raw = build.component.gen.git('show', revision + ':' + name)
    require(raw == path.read_bytes(), 'commit exact gate entrypoint first')
    pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def build_record(scope):
    row = json.loads((ROOT / 'record.json').read_text())
    commands = row.pop('commands_executed')
    require(build.compact(row) == scope['build'] and row['stage'] == 'built-not-executed' and
            row['passed'] and row['model_built'] and row['ownership_closed'] and not row['active'] and
            not row['GPU_executed'] and not row['model_selected'] and
            len(commands) == len(row['commands']) and all(build.runtime.supervision.closed(c) for c in commands),
            'closed exact committed model build required')
    require(digest(ROOT / 'build/compile_commands.json') == row['compile_database_sha256'],
            'recorded effective compile database changed')
    return row


def admit(scope, plan):
    row = build_record(scope)
    build.admit(row)
    require(inputs(plan['source_commit']) == plan['controller_inputs'], 'gate controller changed')
    artifact = row['actual_compiler_inputs']
    require(artifact['uncovered_files'] == 0 and digest(artifact['path']) == artifact['sha256'],
            'actual compiler manifest changed')
    actual = json.loads(Path(artifact['path']).read_text())
    require(len(actual) == artifact['files'] and all(digest(p) == sha for p,sha in actual.items()),
            'actual compiled source changed')
    for asset in [*row['binaries'].values(), *row['fixtures'].values()]:
        require(Path(asset['path']).stat().st_size == asset['bytes'] and digest(asset['path']) == asset['sha256'],
                'same-build executable changed')
    require(build.component.kernel_ids(row['device_IR']) == row['attention_kernel_ids'] == plan['kernel_ids'],
            'compiled kernel identities changed')
    rt = build.runtime.validation.committed(build.runtime.RESULT)
    require(rt['passed'] and rt['ownership_closed'] and not rt['active'] and rt['fresh_faults'] == [] and
            rt['runtime_files'] == plan['runtime_files'] and rt['preload'] == plan['preload'] and
            checks.device_inventory(rt) == plan['physical_device'], 'runtime/device qualification changed')
    if plan['mode'].startswith('id-'):
        asset = plan['selected_ID_fixture']
        require(Path(asset['path']).stat().st_size == asset['bytes'] and digest(asset['path']) == asset['sha256'],
                'selected-ID input changed')
    return rt


def prepare(mode):
    require(mode in ORDER, 'unknown qualification stage')
    record = build.component.gen.committed(RESULT)
    scope = record[build.KEY]
    history = scope.get('same_build_checks', [])
    require([r['mode'] for r in history] == ORDER[:ORDER.index(mode)] and
            all(r['passed'] and r['complete'] and r['ownership_closed'] and not r['fresh_faults'] for r in history),
            'ordered successful same-build gates required;no retry')
    require('check_plan' not in scope or scope['check_plan']['stage'] == 'closed', 'prior check still open/failed')
    row = build_record(scope)
    rt = build.runtime.validation.committed(build.runtime.RESULT)
    revision = build.component.gen.git('rev-parse', 'HEAD').decode().strip()
    folder = output_folder(mode)
    require(not folder.exists(), 'fresh check directory required')
    loader = [p for p in rt['runtime_files'] if Path(p).name.startswith('libze_loader.so.')]
    require(len(loader) == 1, 'one qualified Level Zero loader required')
    env = dict(build.runtime.validation.committed(build.runtime.SETTINGS)['environment'],
        TMPDIR=str(folder / 'tmp'), XDG_CACHE_HOME=str(folder / 'cache'),
        LD_PRELOAD=rt['preload']['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=rt['preload']['loader'],
        ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
        STRATA_SYCL_PREFILL_COMMAND_PROFILE='0',
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'api.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(folder / 'validation.log'))
    plan = dict(stage='prepared', mode=mode, source_commit=revision, controller_inputs=inputs(revision),
        kernel_ids=row['attention_kernel_ids'], runtime_files=rt['runtime_files'], preload=rt['preload'],
        physical_device=checks.device_inventory(rt), environment=env, output_root=str(folder),
        budgets=dict(wall=300,cpu=600,rss_cap=8 << 30,text_cap=32 << 20,file_cap=64 << 20,total_cap=128 << 20),
        criteria='Same-production-source qualification only;no model execution,performance acceptance or full262144context claim. Exact source/dependency/executable/runtime/device/env/maps;pre-main validation,normal owned closure and no fresh fault. First property-only queries enqueue no kernel;then production/private exact3query/final262143address checks and2/11case independent production-entrypoint oracle. All traced actual local arguments must match18argSG16/WG128contract.',
        retention='Keep compact properties/numerical coverage/fault/closure results;retire successful full traces/loader logs after verification. Extra generation remains only for the imminent bounded model comparison.')
    if mode == 'properties':
        env['LD_DEBUG'] = 'files,bindings'
        plan['command'] = [row['fixtures']['fixture']['path'], '--admit', *plan['kernel_ids'], loader[0]]
    elif mode.startswith('id-'):
        asset = record[build.component.KEY]['selected_fixture']
        plan['selected_ID_fixture'] = {k:asset[k] for k in ('path','bytes','sha256','header')}
        plan['command'] = [row['fixtures']['fixture']['path'], '--' + mode.removeprefix('id-'), asset['path']]
    else:
        env['STRATA_TEST_MAIN_GATES'] = '1'
        plan['command'] = [row['fixtures']['numerical']['path'], '--' + mode.removeprefix('oracle-')]
    admit(scope, plan)
    folder.mkdir(mode=0o700)
    for name in ('tmp','cache'):
        (folder / name).mkdir()
    write(folder / 'plan.json', plan)
    scope['check_plan'] = plan
    write(RESULT, record)
    print('Prepared same-production-source ' + mode + ';commit before GPU initialization.', flush=True)


def live(io, plan, rt, stage, device):
    native = Path('/proc', str(io.proc.pid))
    require(str((native / 'exe').resolve()) == str(Path(plan['command'][0]).resolve()), 'wrong native executable')
    actual = dict(v.split(b'=',1) for v in (native / 'environ').read_bytes().split(b'\0') if b'=' in v)
    require(all(actual.get(k.encode()) == v.encode() for k,v in plan['environment'].items()), 'native environment changed')
    text = (native / 'maps').read_text()
    mapped = build.runtime.executable_mappings(text)
    require(mapped <= set(rt['runtime_files']) | {str(Path(plan['command'][0]).resolve())}, 'unqualified code mapping')
    if stage == 'runtime-ready':
        require({rt['preload']['loader'],rt['preload']['library']['path'],plan['environment']['ZE_ENABLE_ALT_DRIVERS']} <= mapped,
                'qualified provider missing')
        identity = checks.live_device(io.proc.pid, device, plan['physical_device'], rt)
        return identity, text
    return None, text


def run(mode):
    record = build.component.gen.committed(RESULT)
    scope, folder = record[build.KEY], output_folder(mode)
    plan = scope['check_plan']
    require(mode in ORDER and plan['mode'] == mode and plan['stage'] == 'prepared' and
            plan == json.loads((folder / 'plan.json').read_text()) and not (folder / 'owned.json').exists(),
            'fresh committed check required')
    rt = admit(scope, plan)
    build.runtime.guards.no_model_owners([])
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursor = re.findall(r'^-- cursor: (.+)$', subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'], text=True, timeout=15), re.M)
    require(len(cursor) == 1, 'fresh fault cursor required')
    outcome = dict(mode=mode, passed=False, complete=False, active=False, ownership_closed=False,
        GPU_kernels_executed=None, model_qualified=False, model_executed=False,
        controller_commit=plan['source_commit'], plan_commit=build.component.gen.git('rev-parse','HEAD').decode().strip())
    owner = None
    def persist():
        outcome['active'] = bool(owner and owner.active is not None)
        write(folder / 'owned.json', dict(outcome, commands=owner.commands if owner else []))
    owner = build.runtime.supervision.Owner(folder / 'owned', persist)
    loader_object = {}
    started = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    try:
        def interact(io, entry):
            if mode == 'properties':
                require(io.line(30) == 'admission_stage=before-runtime', 'unexpected property start')
                name = plan['physical_device']['name']
                require(io.line(90) == 'admission_stage=runtime-ready device=' + name, 'property device differs')
                outcome['physical_device'], text = live(io, plan, rt, 'runtime-ready', name)
                rows = [line.split(maxsplit=5) for line in text.splitlines()]
                ze_loader = plan['command'][-1]
                require(Path(ze_loader).name.startswith('libze_loader.so.') and
                        ze_loader != rt['preload']['loader'], 'ZE and UR loader identities must be distinct')
                rows = [r for r in rows if len(r) == 6 and 'x' in r[1] and r[5].startswith('/') and
                        str(Path(r[5]).resolve()) == ze_loader]
                require(len(rows) == 1, 'one mapped executable loader object required')
                r = rows[0]
                loader_object.update(path=ze_loader, sha256=rt['runtime_files'][ze_loader],
                    device=r[3], inode=int(r[4]), executable_start=r[0].split('-')[0], file_offset=r[2])
                while True:
                    line = io.line(90)
                    if line.startswith('admission_status='):
                        require(line.startswith('admission_status=PASS '), 'native property refusal')
                        break
            else:
                name = None
                prefix = 'QSA_DIRECT_K_GATE ' if mode.startswith('id-') else 'QSA_BF16_READY '
                for stage in ('before-runtime','runtime-ready','after-cleanup'):
                    while True:
                        line = io.line(250)
                        if line.startswith('device='):
                            require(name is None, 'duplicate native device name')
                            name = line.removeprefix('device=').split(' driver=',1)[0]
                        if line.startswith(prefix):
                            break
                    require(line == prefix + stage, 'native gate order differs')
                    delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()), started)]
                    require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
                    identity, _ = live(io, plan, rt, stage, name)
                    if identity:
                        outcome['physical_device'] = identity
                    io.send(b'X')
                footer = io.line(20)
                require(json.loads(footer) == {'kind':'PASS'} if mode.startswith('id-') else
                        footer.startswith('fixture_status=PASS '), 'native footer missing')
            io.proc.stdin.close(); io.exit(20)
        print('Running same-production-source ' + mode, flush=True)
        entry, out, err = owner.run(mode, plan['command'], plan['environment'], P,
            interaction=interact, **plan['budgets'])
        require(build.runtime.supervision.closed(entry) and entry['exit_code'] == 0, 'native process failed/ownership open')
        stderr, stdout = err.read_text(), out.read_text()
        proof = build.runtime.validation.preload_evidence(stderr, entry['pid'])
        require(proof['loader'] == rt['preload']['loader'], 'UR loader mismatch')
        body = stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
        require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:', body),
                'native runtime error')
        warnings = folder / 'validation.log'
        require(not warnings.exists() or warnings.stat().st_size == 0, 'validation warning/error')
        trace = (folder / 'api.trace').read_text()
        if mode == 'properties':
            require('urKernelGet' in trace and not re.search(r'urEnqueue\w*Kernel',trace), 'missing queries/unexpected kernel enqueue')
            rows = []
            for line in stdout.splitlines():
                if line.startswith('arm='):
                    fields = dict(part.split('=',1) for part in line.split())
                    require(fields.pop('status') == 'PASS', 'property refusal')
                    rows.append({key:int(value) for key,value in fields.items()})
            require([r['arm'] for r in rows] == [0,1,2] and
                    stdout.count('admission_status=PASS kernel_submissions=0 ') == 1, 'incomplete property coverage')
            require(all(r['subgroup'] == 16 and r['arguments'] == 18 and r['max_workgroup'] >= 128 and
                r['reported_local_bytes'] >= 0 and r['declared_dynamic_local_bytes'] == 20036 and
                r['reserve_local_bytes'] == 4096 and r['conservative_local_budget'] ==
                r['reported_local_bytes'] + r['declared_dynamic_local_bytes'] + r['reserve_local_bytes'] and
                r['conservative_local_budget'] <= r['per_group_local_max'] and r['spill_descriptor'] >= 0
                for r in rows), 'property values disagree with the compiled native contract')
            outcome.update(properties=rows, loader_object=checks.loader_bindings(stderr,loader_object), GPU_kernels_executed=False)
        elif mode.startswith('id-'):
            launches = checks.local_launches(trace, plan['kernel_ids'], [0,1,2])
            require(all(v['calls'] == 2 for v in launches.values()), 'actual-ID dispatch count differs')
            outcome.update(output=checks.parsed_output(stdout,mode.removeprefix('id-'),0),
                           local_launches=launches, GPU_kernels_executed=True)
        else:
            launches = checks.local_launches(trace,plan['kernel_ids'],[0])
            # Two normal calls/case plus19extended diagnostic calls in case1.
            # Rejected/no-op calls must not enqueue the selected attention kernel.
            expected = 23 if mode == 'oracle-smoke' else 41
            require(launches['0']['calls'] == expected, 'production oracle dispatch count differs')
            outcome.update(output=checks.parsed_oracle(stdout,mode.removeprefix('oracle-')),
                local_launches=launches, GPU_kernels_executed=True)
        admit(scope, plan)
        outcome.update(passed=True, complete=True, validation=proof, exit_code=0)
    except BaseException as error:
        outcome.update(passed=False, complete=False, error=repr(error))
    finally:
        outcome['ownership_closed'] = owner.active is None and all(build.runtime.supervision.closed(c) for c in owner.commands)
        if not outcome['ownership_closed']:
            outcome.update(passed=False, complete=False, error='native ownership remains open')
        checks.postchecks(outcome, boot, cursor[0]); persist()
        plan['stage'] = 'closed' if outcome['passed'] else 'failed'
        scope.setdefault('same_build_checks', []).append(outcome)
        write(RESULT, record)
    print(json.dumps({k:outcome.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}), flush=True)
    require(outcome['passed'], 'same-build gate did not qualify')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare':prepare, 'run':run}[sys.argv[1]](sys.argv[2])
