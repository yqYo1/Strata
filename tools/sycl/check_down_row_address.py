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


def native(scope, build_root=n.ROOT):
    row = json.loads((build_root / 'record.json').read_text())
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
    row = native(scope, Path(plan.get('build_root', n.ROOT)))
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
    folder = Path(plan.get('output_root', ROOT)) / case['name']
    if case.get('host'):
        return dict(plan['host_environment'],TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'))
    return dict(plan['environment'], TMPDIR=str(folder / 'tmp'), XDG_CACHE_HOME=str(folder / 'cache'),
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if case['trace'] else 'UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'api.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(folder / 'validation.log'))



def context(row64):
    build_root = B/'moe-down-row64-v1' if row64 else n.ROOT
    return build_root, build_root/'device-v1', 'moe_down_row64' if row64 else KEY

def prepare(row64=False):
    build_root, root, key = context(row64)
    record = r.validation.committed(RESULT)
    scope = record[key]
    require('device' not in scope and not root.exists(), 'fresh finite comparison only; no retry')
    built = native(scope,build_root)
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
    for name, args, trace in [('tiny', ['--split-row64-tiny' if row64 else '--split-row-address-tiny'], True),
                              ('smoke', ['--split-row64-smoke' if row64 else '--split-row-address-smoke'], False)] + [
            ('timing-' + str(p), ['--split-row64-bench' if row64 else '--split-row-address-bench', str(p)], False) for p in range(3)]:
        cases.append(dict(name=name, command=[built['fixtures'][n.TARGETS[0]]['path'], *args], trace=trace))
    if row64:
        cases.insert(0,dict(name='ownership',command=[built['fixtures'][n.TARGETS[0]]['path'],'--row64-ownership'],trace=False,host=True))
    plan = dict(stage='prepared', source_commit=source, controller_inputs=pins(source),
        native_source_commit=built['source_commit'], binary=built['fixtures'][n.TARGETS[0]],
        generated_build_inputs={str(build_root / 'build' / p): digest(build_root / 'build' / p)
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
    if row64:
        plan.update(variant='row64',build_root=str(build_root),output_root=str(root),
            host_environment={k:v for k,v in built['environment'].items() if k not in
                ('LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER','UR_ENABLE_LAYERS')},
            correctness='First host coordinate/row-metadata ownership at14 M sizes;no runtime/device request '
                'in that explicit source branch. Then traced tiny M3 including BF16 edge values and '
                'invalid-map masks1/2/3. Extended M63/64/65 and117488split-window first/final jobs '
                'in regular/distributed maps. Compare full BF16 dense64 output and split FP32 '
                'host-FMA/capture/no-capture outputs,guards,inputs,status and actual event order.',
            timing='Three fresh processes0,1,2;M2334/5963/8192,regular/distributed,window117488,primary65536,'
                'N2560,K640. Four rotating arms:qualified row-address128,independent row64 scatter,'
                'dense128,dense64. One warmup plus four measured rounds balances every arm/order '
                'within each process.128events/batch,>=5ms floor;no adaptive repeat,retry or exclusion.',
            limits=plan['limits']+' Both scatter arms cache row addresses;dense arms differ only '
                'in the explicit Down geometry. No shared Tile/GU/decode selector is changed.')
    admit(scope, plan)
    root.mkdir(mode=0o700)
    for case in cases:
        folder = root / case['name']; folder.mkdir()
        for name in ('tmp', 'cache'):
            (folder / name).mkdir()
    write(root / 'plan.json', plan)
    scope['device'] = plan
    write(RESULT, record)
    print(f'Prepared {len(cases)} ordered Down component processes;commit before native execution.', flush=True)


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


def parse(text, name, row64=False):
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    require(rows and all(v['status'] == 'PASS' for v in rows), 'native component result failed')
    if row64 and name == 'ownership':
        require([v['rows'] for v in rows] == [1,3,17,63,64,65,127,128,129,2333,2334,5963,8191,8192] and
                all(v['kind']=='row64_ownership' and v['cells']==v['rows']*2560 and v['metadata_slots']>0
                    for v in rows), 'incomplete host coordinate ownership proof')
        return dict(host_coordinate_cases=rows,model_qualified=False)
    dense = [v for v in rows if v['kind']=='row64_dense']
    require(all(v['full_BF16_bits_equal'] and v['guards'] for v in dense), 'dense64 output/guard failure')
    oracles = [v for v in rows if v['kind'] == 'split_correctness']
    require(all((row64 and name=='tiny' or v['expected_mask'] == 0) and v['full_FP32_host_oracle'] and
                v['full_BF16_capture'] and v['guards'] for v in oracles), 'incomplete numerical/guard proof')
    if name in ('tiny', 'smoke'):
        if row64:
            expected = [17]*5 if name=='tiny' else [17,143,145,147,117488,117488]*2
            masks = [0,0,1,2,3] if name=='tiny' else [0]*12
            require([v['tokens'] for v in oracles]==expected and [v['expected_mask'] for v in oracles]==masks and
                    len(dense)==(2 if name=='tiny' else 10) and len(rows)==len(oracles)+len(dense),
                    'incomplete row64 boundary/negative/dense checks')
            return dict(oracle_cases=[{k:v[k] for k in ('tokens','primary_rows','job','job_rows','expected_mask')}
                                      for v in oracles],full_FP32_host_oracles=len(oracles),
                        dense_BF16_oracles=len(dense),guards=True)
        expected = [17, 17] if name == 'tiny' else [17, 81921, 81921, 17, 81921, 81921]
        require(len(rows) == len(oracles) and [v['tokens'] for v in oracles] == expected and
                all(v['primary_rows'] == (8 if v['tokens'] == 17 else 65536) for v in oracles), 'wrong boundary cases')
        return dict(oracle_cases=[{k:v[k] for k in ('tokens', 'primary_rows', 'job', 'job_rows')} for v in oracles],
                    full_FP32_host_oracles=len(oracles), full_BF16_capture=True, guards=True)
    process = int(name.rsplit('-', 1)[1])
    arms, rounds, repeat = (4,4,128) if row64 else (3,3,64)
    prefix = 'row64_' if row64 else 'row_address_'
    samples = [v for v in rows if v['kind'] == prefix+'sample']
    cases = [v for v in rows if v['kind'] == prefix+'case']
    expected = [(m, i, round_, order) for m, i in SHAPES for round_ in range(-1,rounds) for order in range(arms)]
    require([(v['rows'],v['irregular'],v['round'],v['order']) for v in samples] == expected and
            [(v['rows'],v['irregular']) for v in cases] == SHAPES and len(oracles) == 12 and
            len(rows) == (144 if row64 else 90) and (not row64 or len(dense)==6) and all(v['final_bits_equal'] and v['guards'] and v['inputs_immutable'] for v in cases),
            'incomplete timing/correctness population')
    for v in samples:
        require(v['process'] == process and v['arm'] == (process + v['round'] + arms + v['order']) % arms and
                v['repeat'] == repeat and v['queue_batch_ms'] >= 5 and
                all(math.isfinite(v[k]) and v[k] >= 0 for k in COLUMNS[7:]) and
                v['kernel_service_ms'] > 0 and v['wall_ms'] > 0 and
                v['kernel_service_ms'] * repeat <= v['queue_batch_ms'] + 1e-6, 'ineligible timing sample')
    result = dict(process=process, full_FP32_host_oracles=12, final_full_bits_equal=True,
                guards=True, inputs_immutable=True,
                samples=[[v[k] for k in COLUMNS] for v in samples if v['round'] >= 0],
                warmups=[[v[k] for k in COLUMNS] for v in samples if v['round'] == -1])
    if row64:
        result['dense_BF16_oracles'] = len(dense)
    return result


def decision(results, row64=False):
    timing = [x for x in results if x['name'].startswith('timing-')]
    require(len(timing) == 3 and [v['coverage']['process'] for v in timing] == [0,1,2], 'three fresh timing processes required')
    rows = [dict(zip(COLUMNS, v)) for x in timing for v in x['coverage']['samples']]
    out = []
    for m, irregular in SHAPES:
        service, wall, processes = [], [], []
        for process in range(3):
            values = [v for v in rows if (v['rows'],v['irregular'],v['process']) == (m,irregular,process)]
            medians = [{k:statistics.median(v[k] for v in values if v['arm'] == arm)
                        for k in ('kernel_service_ms','wall_ms')} for arm in range(4 if row64 else 3)]
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


def run(row64=False):
    build_root, root, key = context(row64)
    record = r.validation.committed(RESULT); scope = record[key]; plan = scope['device']
    require(plan['stage'] == 'prepared' and plan == json.loads((root / 'plan.json').read_text()) and
            not (root / 'results.json').exists(), 'fresh committed component plan only;no retry')
    rt = admit(scope, plan); r.guards.no_model_owners([])
    results = []
    write(root / 'results.json', dict(stage='running', cases=results))
    try:
        for case in plan['cases']:
            folder = root / case['name']
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
                        require(r.supervision.artifact_bytes(root) <= plan['whole_output_cap'], 'whole comparison size budget')
                        io.send(b'X')
                    require(io.line(20) == 'fixture_status=PASS', 'missing native completion marker')
                    io.proc.stdin.close(); io.exit(30)
                print('Running Down ' + ('row64 ' if row64 else 'row-address ') + case['name'], flush=True)
                entry,out,err = owner.run(case['name'],case['command'],environment(plan,case),P,
                                           interaction=None if case.get('host') else interact,**plan['budgets'])
                require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'native exit/ownership failed')
                if case.get('host'):
                    require('fixture_status=PASS' in out.read_text(), 'host ownership completion missing')
                    outcome.update(coverage=parse(out.read_text(),case['name'],True),passed=True,
                                   complete=True,exit_code=0,native_pid=entry['pid'],host_only=True)
                else:
                    text = err.read_text(); proof = r.validation.preload_evidence(text,entry['pid'])
                    require(proof['loader'] == rt['preload']['loader'], 'wrong UR loader')
                    body = text.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
                    require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:',body), 'native runtime error')
                    warning = folder / 'validation.log'
                    require(not warning.exists() or warning.stat().st_size == 0, 'parameter validation warning/error')
                    if case['trace']:
                        trace = folder / 'api.trace'
                        require(trace.is_file() and 'urEnqueueKernelLaunch' in trace.read_text(), 'missing small kernel trace')
                    outcome.update(coverage=parse(out.read_text(),case['name'],row64), validation=proof,
                                   passed=True,complete=True,exit_code=0,native_pid=entry['pid'])
                admit(scope,plan)
            except BaseException as error:
                outcome.update(passed=False,complete=False,error=repr(error))
            finally:
                outcome['ownership_closed'] = owner.active is None and all(r.supervision.closed(c) for c in owner.commands)
                if not outcome['ownership_closed']:
                    outcome.update(passed=False,complete=False,error='native owner remains open')
                device.postchecks(outcome,boot,cursor[0]); persist(); results.append(outcome)
                write(root / 'results.json',dict(stage='running',cases=results))
            print(json.dumps({k:outcome.get(k) for k in ('name','passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
            require(outcome['passed'], 'component gate failed;no later process')
        plan['decision'] = decision(results,row64)
        plan['stage'] = 'closed'
    except BaseException as error:
        plan.update(stage='failed',error=repr(error))
    finally:
        plan.update(results=results,sample_columns=COLUMNS,model_qualified=False,inference_adopted=False)
        write(root / 'results.json',dict(stage=plan['stage'],cases=results,decision=plan.get('decision'),error=plan.get('error')))
        write(RESULT,record)
    print(json.dumps(dict(stage=plan['stage'],decision=plan.get('decision'),error=plan.get('error'))),flush=True)
    return int(plan['stage'] != 'closed')


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'],['run'],['prepare-row64'],['run-row64']),
            'check_down_row_address.py prepare|run|prepare-row64|run-row64')
    row64 = sys.argv[1].endswith('-row64')
    with (B / 'owned-v0141-measurement.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.exit(prepare(row64) if sys.argv[1].startswith('prepare') else run(row64))
