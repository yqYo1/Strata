#!/usr/bin/env python3
"""Ordered, bounded split-MoE device gates; repository LICENSE applies.

No model run, performance inference or adoption. Main owns all execution.
"""
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_moe_split as build
import check_qsa_k_loop as device
import check_down_row_address as row_control

r, q, P, B, RESULT, KEY = build.r, build.q, build.P, build.B, build.RESULT, build.KEY
ROOT = build.ROOT.parent / 'device'
ORDER = ['smoke', 'validate', 'route-off', 'route-on']
DOWN_ORDER = ['tiny-row64', 'tiny-row128', 'large-row64', 'large-row128',
              'route-off', 'route-original', 'route-row128', 'route-row64']
require, digest, write = build.require, build.digest, build.write


def pins(revision):
    return q.sources(revision, Path(__file__).resolve(),
                     [*build.model.DIRECT, *build.model.ORCHESTRATION, build.model.GEMM, build.model.FIXTURE])


def context(variant):
    native_root, key = build.context(variant)
    return native_root, ROOT if variant == 'split' else native_root/'device', key, ORDER if variant == 'split' else DOWN_ORDER


def native(scope, native_root=build.ROOT):
    row = json.loads((native_root / 'record.json').read_text())
    commands = row.pop('commands_executed')
    require(row['stage'] == 'built-not-executed' and row['passed'] and row['ownership_closed'] and
            not row['active'] and build.compact(row) == scope['build'] and
            len(commands) == len(row['commands']) and all(r.supervision.closed(v) for v in commands),
            'closed exact-source build required')
    build.admit(row, native_root)
    actual = row['actual_compiler_inputs']
    require(actual['uncovered'] == 0 and digest(actual['path']) == actual['sha256'],
            'actual compiler manifest changed')
    values = json.loads(Path(actual['path']).read_text())
    require(len(values) == actual['files'] and all(digest(p) == sha for p,sha in values.items()),
            'actual compiled inputs changed')
    require(digest(native_root/'build/compile_commands.json') == row['compile_database_sha256'],
            'effective compile database changed')
    return row


def admit(scope, plan):
    native_root, _, _, _ = context(plan.get('variant', 'split'))
    row = native(scope, native_root)
    require(pins(plan['source_commit']) == plan['inputs'], 'component gate source changed')
    require(row['binaries']['sycl_moe_down_scatter_bench'] == plan['binary'], 'component executable changed')
    rt = r.validation.committed(r.RESULT)
    require(rt['passed'] and rt['ownership_closed'] and not rt['active'] and not rt['fresh_faults'] and
            rt['runtime_files'] == plan['runtime_files'] and rt['preload'] == plan['preload'] and
            device.device_inventory(rt) == plan['physical_device'], 'qualified runtime/device changed')
    return rt


def prepare(mode, variant='split'):
    native_root, root, key, order = context(variant)
    require(mode in order, 'unknown split device gate')
    record = r.validation.committed(RESULT)
    scope = record[key]
    history = scope.get('device_checks', [])
    require([v['mode'] for v in history] == order[:order.index(mode)] and
            all(v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults'] for v in history),
            'ordered successful split device gates required; do not retry a failed gate')
    require('device_plan' not in scope or scope['device_plan']['stage'] == 'closed', 'prior device gate open/failed')
    row = native(scope, native_root)
    rt = r.validation.committed(r.RESULT)
    revision = r.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    folder = root / mode
    require(not folder.exists(), 'fresh owned device output directory required')
    enabled = mode in ('route-on', 'route-original', 'route-row128', 'route-row64')
    kernel = mode[6:] if mode in ('route-original', 'route-row128', 'route-row64') else 'original'
    env = dict(r.validation.committed(r.SETTINGS)['environment'],
        TMPDIR=str(folder/'tmp'), XDG_CACHE_HOME=str(folder/'cache'),
        LD_PRELOAD=rt['preload']['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=rt['preload']['loader'],
        ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
        STRATA_SYCL_PREFILL_COMMAND_PROFILE='0', STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE='0',
        STRATA_SYCL_PREFILL_NATIVE16='bf16', STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',
        STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto', STRATA_SYCL_PREFILL_GU_SWIGLU='0',
        STRATA_TEST_MAIN_GATES='1', STRATA_SYCL_PREFILL_DOWN_SCATTER='large' if enabled else '0',
        UR_LOG_TRACING='level:info;flush:info;output:file,'+str(folder/'api.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
    require(env['UR_ENABLE_LAYERS'] == 'UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' and
            env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['EnableDirectSubmission'] == '0',
            'mandatory tracing/validation/safety environment required')
    binary = row['binaries']['sycl_moe_down_scatter_bench']
    args = ['--split-'+mode] if mode in ('smoke','validate') else ['--split-route-smoke',str(int(enabled))]
    if variant == 'down':
        env['STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL'] = kernel
        if mode.startswith(('tiny-', 'large-')):
            name = 'row64' if mode.endswith('row64') else 'row-address'
            args = ['--split-' + name + ('-tiny' if mode.startswith('tiny-') else '-smoke')]
    plan = dict(stage='prepared',mode=mode,source_commit=revision,inputs=pins(revision),binary=binary,
        runtime_files=rt['runtime_files'],preload=rt['preload'],physical_device=device.device_inventory(rt),
        command=[binary['path'],*args],environment=env,
        budgets=dict(wall=900,cpu=3600,rss_cap=16 << 30,text_cap=32 << 20,file_cap=64 << 20,total_cap=128 << 20),
        criteria='Same-build component only;normal owned closure,no fresh fault,immutable safety/provider/code maps '
            'and pre-main parameter validation. Smoke:ordinary3-row oracle,split3/17,16admission refusals,3invalid-map '
            'masks and exception-unwind drain. Extended:6cases65537/81921/117488 regular/irregular;selectedjobs '
            'reach65535/65536/finalrow;allFP32 outputs/post-BF16 CPUFMA oracle,BF16capture,guards,alljobweightmaps '
            'and complete mixed output for selectedgathers. Routes:8cases2333/2334/2335/8192 regular/irregular '
            'per immutable startup arm;dispatch/false-no-write/fallback. No fullmodel,performance,quality,decode '
            'or physical262144qualification.',
        retention='Keep compact coverage/counts and fault/ownership outcome;retire successful trace/poll logs '
            'after verification. Candidate generation remains only for imminent model comparison.')
    if variant == 'down':
        expected_launches = dict(original=0, row128=0, row64=0)
        if mode.startswith('route-'):
            if enabled:
                expected_launches[kernel] = 6
        else:
            direct = 'row64' if mode.endswith('row64') else 'row128'
            expected_launches[direct] = {'tiny-row64':10, 'tiny-row128':4,
                                        'large-row64':24, 'large-row128':12}[mode]
        plan.update(variant='down', selected_split_kernel=kernel, native_source_commit=row['source_commit'],
            expected_split_kernel_launches=expected_launches,
            criteria='Fresh same-build numerical and production-route gates only;not model/performance adoption. '
                'Tiny row64 includes edge values,invalid-map masks1/2/3 and dense64 bits;tiny row128 preserves '
                'the existing address contract. Large modes exercise63/64/65 and split primary/tail/final rows. '
                'Each production arm checks2333/2334/2335/8192 in regular/distributed maps,all FP32 output bits, '
                'immutable inputs,false-return no-write and fallback. Immutable startup choice plus actual '
                'UR create-to-launch identities must give exactly six chosen split kernels and no other '
                'split policy;route-off launches none. Qualified providers/pre-main validation,source/owner '
                'closure,no fresh fault and stop-first-failure remain mandatory.')
    admit(scope,plan)
    folder.mkdir(parents=True,mode=0o700)
    for name in ('tmp','cache'):(folder/name).mkdir()
    write(folder/'plan.json',plan)
    scope['device_plan'] = plan
    write(RESULT,record)
    print('Prepared split '+mode+';commit before GPU initialization.',flush=True)


def live(io, plan, rt, stage):
    proc = Path('/proc',str(io.proc.pid))
    require(str((proc/'exe').resolve()) == str(Path(plan['binary']['path']).resolve()), 'wrong native executable')
    env = dict(v.split(b'=',1) for v in (proc/'environ').read_bytes().split(b'\0') if b'=' in v)
    require(all(env.get(k.encode()) == v.encode() for k,v in plan['environment'].items()), 'child environment changed')
    mapped = r.executable_mappings((proc/'maps').read_text())
    require(mapped <= set(rt['runtime_files']) | {str(Path(plan['binary']['path']).resolve())}, 'unqualified native code map')
    if stage == 'runtime-ready':
        require({rt['preload']['loader'],rt['preload']['library']['path'],plan['environment']['ZE_ENABLE_ALT_DRIVERS']} <= mapped,
                'qualified provider missing')
        return device.live_device(io.proc.pid,plan['physical_device']['name'],plan['physical_device'],rt)
    return None


def parse(stdout, mode):
    if mode in ('tiny-row64', 'tiny-row128', 'large-row64', 'large-row128'):
        return row_control.parse(stdout, 'tiny' if mode.startswith('tiny-') else 'smoke', mode.endswith('row64'))
    rows = [json.loads(s) for s in stdout.splitlines() if s.startswith('{')]
    require(rows and all(v['status'] == 'PASS' for v in rows), 'incomplete/failed component result')
    kinds = {k:[v for v in rows if v['kind'] == k] for k in {v['kind'] for v in rows}}
    oracles = kinds.get('split_correctness',[])
    require(all(v['full_FP32_host_oracle'] and v['full_BF16_capture'] and v['guards'] for v in oracles),
            'incomplete split arithmetic/guard proof')
    if mode == 'smoke':
        require(len(kinds.get('correctness',[])) == 1 and
                [v['tokens'] for v in kinds.get('split_case',[])] == [3,17] and
                kinds.get('split_admissions') == [dict(kind='split_admissions',rejected=16,invalid_map_cases=3,status='PASS')] and
                kinds.get('split_exception_unwind') == [dict(kind='split_exception_unwind',queue_complete=True,status='PASS')],
                'smoke coverage missing')
        require({v['expected_mask'] for v in oracles} == {0,1,2,3}, 'invalid-map mask coverage missing')
    elif mode == 'validate':
        cases = kinds.get('split_case',[])
        require([(v['tokens'],v['irregular']) for v in cases] ==
                [(n,i) for i in (0,1) for n in (65537,81921,117488)] and
                all(v['primary_rows'] == 65536 and v['physical_final_row'] for v in cases), 'extended coverage missing')
        require(oracles and all(v['expected_mask'] == 0 for v in oracles), 'extended oracle status failure')
    else:
        routes = kinds.get('selected_route',[])
        enabled = mode in ('route-on', 'route-original', 'route-row128', 'route-row64')
        require([(v['rows'],v['enabled'],v['selected']) for v in routes] ==
                [(n,int(enabled),int(enabled and n>=2334)) for n in (2333,2334,2335,8192) for _ in range(2)] and
                all(v['inputs_immutable'] and v['false_route_no_write'] and v['expected_mask'] == 0 for v in routes),
                'production route/fallback coverage missing')
    if mode in ('smoke','validate'):
        flow = kinds.get('split_dataflow',[])
        require(len(flow)==1 and flow[0]['tokens'] == (17 if mode=='smoke' else 117488) and
                all(flow[0][k] for k in ('full_weight_map','full_gather','inputs_immutable','guards')),
                'production weight/mixed dataflow coverage missing')
    return dict(result_counts={k:len(v) for k,v in kinds.items()},
        cases=kinds.get('split_case',[]),post_BF16_FP32_oracles=len(kinds.get('split_correctness',[])),
        production_routes=[{k:v[k] for k in ('rows','enabled','selected','split_rows')} for v in kinds.get('selected_route',[])],
        limitation='Fixture post-BF16 consumer contract only;selected-job gathers,not everyjob. No model/performance qualification.')


def split_kernel_launches(trace):
    """Associate successful UR launches with their created C++ kernel names."""
    created = {}
    counts = dict(original=0, row128=0, row64=0)
    stems = dict(original='SplitDownScatterKernel', row128='SplitRowAddressDownKernel', row64='SplitDown64Kernel')
    for line in trace.splitlines():
        if '<--- urKernelCreate(' in line and 'UR_RESULT_SUCCESS' in line:
            match = re.search(r'\((_ZTS\S+)\).*?\.phKernel = .*?\((0x[0-9a-f]+)\)', line)
            if match:
                created[match[2]] = match[1]
        if '<--- urEnqueueKernelLaunch' in line and 'UR_RESULT_SUCCESS' in line:
            handle = re.search(r'\.hKernel = (0x[0-9a-f]+)', line)
            if handle:
                name = created.get(handle[1], '')
                for policy, stem in stems.items():
                    if stem in name:
                        counts[policy] += 1
    return counts


def run(mode, variant='split'):
    _, root, key, order = context(variant)
    record = r.validation.committed(RESULT);scope = record[key];plan = scope['device_plan'];folder = root/mode
    require(mode in order and plan['mode'] == mode and plan['stage'] == 'prepared' and
            json.loads((folder/'plan.json').read_text()) == plan and not (folder/'owned.json').exists(),
            'fresh committed split component plan required')
    rt = admit(scope,plan);r.guards.no_model_owners([])
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursors = re.findall(r'^-- cursor: (.+)$',subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15),re.M)
    require(len(cursors) == 1,'fresh fault cursor required')
    outcome = dict(mode=mode,passed=False,complete=False,active=False,ownership_closed=False,
        GPU_kernels_executed=None,model_executed=False,model_qualified=False,
        controller_commit=plan['source_commit'],plan_commit=r.build.git(P,'rev-parse','HEAD').decode().strip())
    owner = None
    def persist():
        outcome['active'] = bool(owner and owner.active is not None)
        write(folder/'owned.json',dict(outcome,commands=owner.commands if owner else []))
    owner = r.supervision.Owner(folder/'owned',persist)
    started = (time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
    try:
        def interact(io,entry):
            for stage in ('before-runtime','runtime-ready','before-exit'):
                while True:
                    line = io.line(800)
                    if line.startswith('SPLIT_READY '):break
                require(line == 'SPLIT_READY '+stage,'native gate order differs')
                delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),started)]
                require(min(delta)>=0 and abs(delta[0]-delta[1])<=50000000,'suspend/clock discontinuity')
                identity = live(io,plan,rt,stage)
                if identity:outcome['physical_device'] = identity
                io.send(b'X')
            require(io.line(20) == 'fixture_status=PASS','native final marker missing')
            io.proc.stdin.close();io.exit(30)
        print('Running split device '+mode,flush=True)
        entry,out,err = owner.run(mode,plan['command'],plan['environment'],P,interaction=interact,**plan['budgets'])
        require(r.supervision.closed(entry) and entry['exit_code']==0,'native exit/ownership failed')
        text = err.read_text();proof = r.validation.preload_evidence(text,entry['pid'])
        require(proof['loader']==rt['preload']['loader'],'UR loader mismatch')
        body = text.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
        require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:',body),
                'native runtime error')
        warnings = folder/'validation.log'
        require(not warnings.exists() or warnings.stat().st_size==0,'parameter validation warning/error')
        trace = folder/'api.trace'
        require(trace.is_file() and 'urEnqueueKernelLaunch' in trace.read_text(),'missing actual kernel trace')
        coverage = parse(out.read_text(),mode)
        if variant == 'down':
            kernel = plan['selected_split_kernel']
            enabled = mode in ('route-original','route-row128','route-row64')
            if mode.startswith('route-'):
                message = '[native16-down] fusion=' + ('large' if enabled else '0') + ' min_rows=2334 split_kernel=' + kernel
                require(text.count(message) == 1, 'actual native startup selection differs')
            launches = split_kernel_launches(trace.read_text())
            require(launches == plan['expected_split_kernel_launches'], 'actual split kernel launches differ')
            coverage.update(actual_split_kernel_launches=launches,
                            production_selector_exercised=mode.startswith('route-'))
        outcome.update(coverage=coverage,passed=True,complete=True,
            GPU_kernels_executed=True,validation=proof,exit_code=0)
        admit(scope,plan)
    except BaseException as error:
        outcome.update(passed=False,complete=False,error=repr(error))
    finally:
        outcome['ownership_closed'] = owner.active is None and all(r.supervision.closed(c) for c in owner.commands)
        if not outcome['ownership_closed']:outcome.update(passed=False,complete=False,error='native ownership remains open')
        device.postchecks(outcome,boot,cursors[0]);persist()
        plan['stage'] = 'closed' if outcome['passed'] else 'failed'
        scope.setdefault('device_checks',[]).append(outcome);write(RESULT,record)
    print(json.dumps({k:outcome.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    require(outcome['passed'],'split component did not qualify')


if __name__ == '__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(len(sys.argv) == 3 and sys.argv[1] in ('prepare','run','prepare-down','run-down'),
                'check_moe_split.py prepare|run|prepare-down|run-down MODE')
        variant = 'down' if sys.argv[1].endswith('-down') else 'split'
        (prepare if sys.argv[1].startswith('prepare') else run)(sys.argv[2], variant)
