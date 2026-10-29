#!/usr/bin/env python3
"""Finite, Main-owned Down row-address correctness and resident timing gates.

Main-authored, repository LICENSE. Reuses only this project's existing source,
device, ownership and fault checks; no external implementation is imported.
This component comparison never adopts an engine or qualifies physical context.
"""
import ast
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_down_row_address as n
import check_qsa_k_loop as device

r, P, B, RESULT, KEY = n.r, n.P, n.B, n.RESULT, n.KEY
ROOT = n.ROOT / 'device-v1'
require, digest, write = n.require, n.digest, n.write
SHAPES = [(rows, irregular) for rows in (2334, 5963, 8192) for irregular in (0, 1)]
COLUMNS = ['process', 'rows', 'irregular', 'round', 'order', 'arm', 'repeat',
           'kernel_service_ms', 'queue_batch_ms', 'wall_ms', 'submit_cpu_ms', 'complete_cpu_ms']


def pins(revision):
    pending, seen, result = [Path(__file__).resolve()], set(), {}
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        name = str(path.relative_to(P))
        raw = n.build.git(P, 'show', revision + ':' + name)
        require(raw == path.read_bytes(), 'commit exact controller input: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            modules = [v.name for v in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                local = path.parent / ((module or '').split('.')[0] + '.py')
                if local.is_file():
                    pending.append(local)
    return result


def native(scope):
    row = json.loads((n.ROOT / 'record.json').read_text())
    commands = row.pop('commands')
    require(n.compact(row) == scope['build'] and row['stage'] == 'built-not-executed' and
            row['passed'] and row['ownership_closed'] and not row['active'] and
            not row['engine_built'] and len(commands) == 3 and
            all(r.supervision.closed(c) and c['exit_code'] == 0 for c in commands),
            'closed exact-source component build required')
    n.admit(row)
    actual = row['compiled_inputs']
    require(actual['uncovered_files'] == 0 and
            digest(actual['manifest']['path']) == actual['manifest']['sha256'], 'compiler manifest changed')
    values = json.loads(Path(actual['manifest']['path']).read_text())
    require(len(values) == actual['files'] and all(digest(p) == sha for p, sha in values.items()),
            'actual compiled input changed')
    for value in row['fixtures'].values():
        require(digest(value['path']) == value['sha256'] and
                Path(value['path']).stat().st_size == value['bytes'], 'component executable changed')
    return row


def admit(scope, plan):
    row = native(scope)
    require(pins(plan['source_commit']) == plan['controller_inputs'] and
            row['fixtures'][n.TARGETS[0]] == plan['binary'], 'controller/executable changed')
    for path, sha in plan['generated_build_inputs'].items():
        require(digest(path) == sha, 'effective build description changed')
    rt = r.validation.committed(r.RESULT)
    require(digest(r.RESULT) == plan['runtime_sha256'] and rt['passed'] and
            rt['ownership_closed'] and not rt['active'] and rt['fresh_faults'] == [] and
            device.device_inventory(rt) == plan['physical_device'], 'runtime/device qualification changed')
    return rt


def environment(plan, case):
    folder = ROOT / case['name']
    return dict(plan['environment'], TMPDIR=str(folder / 'tmp'), XDG_CACHE_HOME=str(folder / 'cache'),
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if case['trace'] else 'UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'api.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(folder / 'validation.log'))


def prepare():
    record = r.validation.committed(RESULT)
    scope = record[KEY]
    require('device' not in scope and not ROOT.exists(), 'fresh finite comparison only; no retry')
    built = native(scope)
    rt = r.validation.committed(r.RESULT)
    source = n.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    env = dict(r.validation.committed(r.SETTINGS)['environment'],
        LD_PRELOAD=rt['preload']['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=rt['preload']['loader'],
        ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
        STRATA_TEST_MAIN_GATES='1', STRATA_SYCL_PREFILL_NATIVE16='bf16',
        STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm', STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',
        STRATA_SYCL_PREFILL_COMMAND_PROFILE='0', STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE='0',
        STRATA_SYCL_PREFILL_GU_SWIGLU='0', STRATA_SYCL_PREFILL_DOWN_SCATTER='0', ZES_ENABLE_SYSMAN='0')
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and
            env['NEOReadDebugKeys'] == '1' and env['EnableDirectSubmission'] == '0' and
            env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1', 'qualified immutable safety required')
    cases = []
    for name, args, trace in [('tiny', ['--split-row-address-tiny'], True),
                              ('smoke', ['--split-row-address-smoke'], False)] + [
            ('timing-' + str(p), ['--split-row-address-bench', str(p)], False) for p in range(3)]:
        cases.append(dict(name=name, command=[built['fixtures'][n.TARGETS[0]]['path'], *args], trace=trace))
    plan = dict(stage='prepared', source_commit=source, controller_inputs=pins(source),
        native_source_commit=built['source_commit'], binary=built['fixtures'][n.TARGETS[0]],
        generated_build_inputs={str(n.ROOT / 'build' / p): digest(n.ROOT / 'build' / p)
                                for p in ('compile_commands.json', 'build.ninja')},
        runtime_sha256=digest(r.RESULT), environment=env,
        physical_device=device.device_inventory(rt), cases=cases,
        budgets=dict(wall=900, cpu=3600, rss_cap=16 << 30, text_cap=32 << 20,
                     file_cap=64 << 20, total_cap=192 << 20), whole_output_cap=768 << 20,
        correctness='Tiny:two17-token edge-input split cases. Then six17/81921-token job checks '
            'cover regular/distributed maps,primary/tail/final address,full post-BF16 FP32 host FMA '
            'oracle,captured/no-capture outputs and guards. Timing checks both original/candidate '
            'oracles and final full FP32 bits,all inputs/status/guards. Invalid-map device tests '
            'are not added by these modes;no inference adoption or full-context qualification.',
        timing='Three fresh processes0,1,2;each tests M2334/5963/8192,regular/distributed,window117488 '
            'and primary65536,N2560,K640. One warmup and three rounds,three rotating arms per shape;'
            '64calls/batch,>=5ms marker interval,all actual kernel events checked. Keep every '
            'measured timing and separate warmups. No retry,exclusion,adaptive repeat or early speed stop.',
        acceptance='For each of six shape/maps,geomean across the three paired process median '
            'candidate/original service and wall ratios <=1.02. Overall service geomean <=0.99 '
            'and overall wall geomean <=1.00. All correctness,clock,batch,source,provider,validation,'
            'ownership and fresh-fault checks must pass. This permits only a later production plan.',
        stopping='Stop first correctness,clock,budget,source,provider,validation,ownership or fresh-fault '
            'failure;do not replace a failed case. Finish all three timing processes before deciding speed.',
        limits='Resident/repeated X,W and mapping control,with large resets and CPU oracles outside '
            'timing. No PCIe copy in timed batches. GEMM-only writes BF16 dense output;fused arms '
            'update FP32 scatter output. Different output traffic and caches prevent interpreting '
            'their difference as an additive epilogue time or any hardware peak/saturation. '
            'No model PP,TG,decode preservation,task quality or physical262144 claim.',
        retention='Retain compact independent samples,coverage,decision and source/runtime identity. '
            'Retire successful traces,logs,JIT and intermediates after verified committed evidence. '
            'Keep one component executable only until this decision;retained model generations unchanged.')
    admit(scope, plan)
    ROOT.mkdir(mode=0o700)
    for case in cases:
        folder = ROOT / case['name']; folder.mkdir()
        for name in ('tmp', 'cache'):
            (folder / name).mkdir()
    write(ROOT / 'plan.json', plan)
    scope['device'] = plan
    write(RESULT, record)
    print('Prepared five ordered Down component processes;commit before GPU execution.', flush=True)


def live(io, case, plan, rt, stage):
    proc = Path('/proc', str(io.proc.pid))
    require(str((proc / 'exe').resolve()) == str(Path(plan['binary']['path']).resolve()), 'wrong executable')
    actual = dict(v.split(b'=', 1) for v in (proc / 'environ').read_bytes().split(b'\0') if b'=' in v)
    child_env = environment(plan,case)
    require(all(actual.get(k.encode()) == v.encode() for k, v in child_env.items()), 'child environment changed')
    mapped = r.executable_mappings((proc / 'maps').read_text())
    require(mapped <= set(rt['runtime_files']) | {str(Path(plan['binary']['path']).resolve())}, 'unqualified native code map')
    if stage != 'before-runtime':
        require({rt['preload']['loader'], rt['preload']['library']['path'],
                 child_env['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'qualified provider missing')
        return device.live_device(io.proc.pid, plan['physical_device']['name'], plan['physical_device'], rt)


def parse(text, name):
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    require(rows and all(v['status'] == 'PASS' for v in rows), 'native component result failed')
    oracles = [v for v in rows if v['kind'] == 'split_correctness']
    require(all(v['expected_mask'] == 0 and v['full_FP32_host_oracle'] and
                v['full_BF16_capture'] and v['guards'] for v in oracles), 'incomplete numerical/guard proof')
    if name in ('tiny', 'smoke'):
        expected = [17, 17] if name == 'tiny' else [17, 81921, 81921, 17, 81921, 81921]
        require(len(rows) == len(oracles) and [v['tokens'] for v in oracles] == expected and
                all(v['primary_rows'] == (8 if v['tokens'] == 17 else 65536) for v in oracles), 'wrong boundary cases')
        return dict(oracle_cases=[{k:v[k] for k in ('tokens', 'primary_rows', 'job', 'job_rows')} for v in oracles],
                    full_FP32_host_oracles=len(oracles), full_BF16_capture=True, guards=True)
    process = int(name.rsplit('-', 1)[1])
    samples = [v for v in rows if v['kind'] == 'row_address_sample']
    cases = [v for v in rows if v['kind'] == 'row_address_case']
    expected = [(m, i, round_, order) for m, i in SHAPES for round_ in (-1, 0, 1, 2) for order in range(3)]
    require([(v['rows'],v['irregular'],v['round'],v['order']) for v in samples] == expected and
            [(v['rows'],v['irregular']) for v in cases] == SHAPES and len(oracles) == 12 and
            len(rows) == 90 and all(v['final_bits_equal'] and v['guards'] and v['inputs_immutable'] for v in cases),
            'incomplete timing/correctness population')
    for v in samples:
        require(v['process'] == process and v['arm'] == (process + v['round'] + 3 + v['order']) % 3 and
                v['repeat'] == 64 and v['queue_batch_ms'] >= 5 and
                all(math.isfinite(v[k]) and v[k] >= 0 for k in COLUMNS[7:]) and
                v['kernel_service_ms'] > 0 and v['wall_ms'] > 0 and
                v['kernel_service_ms'] * 64 <= v['queue_batch_ms'] + 1e-6, 'ineligible timing sample')
    return dict(process=process, full_FP32_host_oracles=12, final_full_bits_equal=True,
                guards=True, inputs_immutable=True,
                samples=[[v[k] for k in COLUMNS] for v in samples if v['round'] >= 0],
                warmups=[[v[k] for k in COLUMNS] for v in samples if v['round'] == -1])


def decision(results):
    timing = [x for x in results if x['name'].startswith('timing-')]
    require(len(timing) == 3 and [v['coverage']['process'] for v in timing] == [0,1,2], 'three fresh timing processes required')
    rows = [dict(zip(COLUMNS, v)) for x in timing for v in x['coverage']['samples']]
    out = []
    for m, irregular in SHAPES:
        service, wall, processes = [], [], []
        for process in range(3):
            values = [v for v in rows if (v['rows'],v['irregular'],v['process']) == (m,irregular,process)]
            medians = [{k:statistics.median(v[k] for v in values if v['arm'] == arm)
                        for k in ('kernel_service_ms','wall_ms')} for arm in range(3)]
            service.append(medians[1]['kernel_service_ms'] / medians[0]['kernel_service_ms'])
            wall.append(medians[1]['wall_ms'] / medians[0]['wall_ms'])
            processes.append(dict(process=process, arm_medians=medians))
        out.append(dict(rows=m, irregular=irregular, process_ratios=dict(service=service,wall=wall),
                        service_ratio=statistics.geometric_mean(service), wall_ratio=statistics.geometric_mean(wall),
                        processes=processes))
    service = statistics.geometric_mean(v['service_ratio'] for v in out)
    wall = statistics.geometric_mean(v['wall_ratio'] for v in out)
    passed = service <= .99 and wall <= 1 and all(v['service_ratio'] <= 1.02 and v['wall_ratio'] <= 1.02 for v in out)
    return dict(cases=out, service_geomean=service, wall_geomean=wall,
                component_speed_gate_passed=passed, model_qualified=False, inference_adopted=False)


def run():
    record = r.validation.committed(RESULT); scope = record[KEY]; plan = scope['device']
    require(plan['stage'] == 'prepared' and plan == json.loads((ROOT / 'plan.json').read_text()) and
            not (ROOT / 'results.json').exists(), 'fresh committed component plan only;no retry')
    rt = admit(scope, plan); r.guards.no_model_owners([])
    results = []
    write(ROOT / 'results.json', dict(stage='running', cases=results))
    try:
        for case in plan['cases']:
            folder = ROOT / case['name']
            require(not (folder / 'owned.json').exists(), 'existing owner;do not duplicate')
            admit(scope, plan)
            cursor = re.findall(r'^-- cursor: (.+)$', subprocess.check_output(
                ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'], text=True, timeout=15), re.M)
            require(len(cursor) == 1, 'fresh journal cursor required')
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            outcome = dict(name=case['name'], passed=False, complete=False, active=False, ownership_closed=False)
            owner = None
            def persist():
                outcome['active'] = bool(owner and owner.active is not None)
                write(folder / 'owned.json', dict(outcome, commands=owner.commands if owner else []))
            owner = r.supervision.Owner(folder / 'owned', persist)
            started = (time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
            try:
                def interact(io, entry):
                    for stage in ('before-runtime','runtime-ready','before-exit'):
                        while True:
                            line = io.line(880)
                            if line.startswith('SPLIT_READY '):
                                break
                        require(line == 'SPLIT_READY ' + stage, 'native gate order differs')
                        delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),started)]
                        require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
                        identity = live(io,case,plan,rt,stage)
                        if identity:
                            outcome['physical_device'] = identity
                        require(r.supervision.artifact_bytes(ROOT) <= plan['whole_output_cap'], 'whole comparison size budget')
                        io.send(b'X')
                    require(io.line(20) == 'fixture_status=PASS', 'missing native completion marker')
                    io.proc.stdin.close(); io.exit(30)
                print('Running Down row-address ' + case['name'], flush=True)
                entry,out,err = owner.run(case['name'],case['command'],environment(plan,case),P,
                                           interaction=interact,**plan['budgets'])
                require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'native exit/ownership failed')
                text = err.read_text(); proof = r.validation.preload_evidence(text,entry['pid'])
                require(proof['loader'] == rt['preload']['loader'], 'wrong UR loader')
                body = text.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
                require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:',body), 'native runtime error')
                warning = folder / 'validation.log'
                require(not warning.exists() or warning.stat().st_size == 0, 'parameter validation warning/error')
                if case['trace']:
                    trace = folder / 'api.trace'
                    require(trace.is_file() and 'urEnqueueKernelLaunch' in trace.read_text(), 'missing small kernel trace')
                outcome.update(coverage=parse(out.read_text(),case['name']), validation=proof,
                               passed=True,complete=True,exit_code=0,native_pid=entry['pid'])
                admit(scope,plan)
            except BaseException as error:
                outcome.update(passed=False,complete=False,error=repr(error))
            finally:
                outcome['ownership_closed'] = owner.active is None and all(r.supervision.closed(c) for c in owner.commands)
                if not outcome['ownership_closed']:
                    outcome.update(passed=False,complete=False,error='native owner remains open')
                device.postchecks(outcome,boot,cursor[0]); persist(); results.append(outcome)
                write(ROOT / 'results.json',dict(stage='running',cases=results))
            print(json.dumps({k:outcome.get(k) for k in ('name','passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
            require(outcome['passed'], 'component gate failed;no later process')
        plan['decision'] = decision(results)
        plan['stage'] = 'closed'
    except BaseException as error:
        plan.update(stage='failed',error=repr(error))
    finally:
        plan.update(results=results,sample_columns=COLUMNS,model_qualified=False,inference_adopted=False)
        write(ROOT / 'results.json',dict(stage=plan['stage'],cases=results,decision=plan.get('decision'),error=plan.get('error')))
        write(RESULT,record)
    print(json.dumps(dict(stage=plan['stage'],decision=plan.get('decision'),error=plan.get('error'))),flush=True)
    return int(plan['stage'] != 'closed')


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'],['run']), 'check_down_row_address.py prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.exit(prepare() if sys.argv[1] == 'prepare' else run())
