#!/usr/bin/env python3
"""Own serial property, exact-ID and selector-service gates; no model adoption."""
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
import build_qsa_radix as b
import check_qsa_k_loop as native
from measure_down_scatter import postchecks

P, B, ROOT, RESULT = b.P, b.B, b.ROOT, b.RESULT
require, digest, write = b.require, b.digest, b.write
ORDER = ['properties', 'correctness', 'event-smoke', 'performance']
ARMS = ['original', 'control', 'batched']


def inputs(revision):
    pins = b.inputs(revision) | native.inputs(revision)
    name = str(Path(__file__).resolve().relative_to(P))
    raw = b.base.component.gen.git('show', revision + ':' + name)
    require(raw == (P / name).read_bytes(), 'commit exact check controller first')
    pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def build_record(scope):
    row = scope['build']['outcome']
    b.admit(json.loads((ROOT / 'plan.json').read_text()))
    require(scope['build']['stage'] == 'closed' and row['passed'] and row['complete'] and
            row['ownership_closed'] and not row.get('active'), 'closed selector build required')
    require(row['device_IR_identical_across_observation'], 'observation changed device code')
    for field in ('binary', 'observed_binary', 'actual_compiler_inputs', 'kernel_inventory'):
        asset = row[field]
        require(Path(asset['path']).stat().st_size == asset['bytes'] and digest(asset['path']) == asset['sha256'],
                'build asset changed: ' + field)
    actual = json.loads(Path(row['actual_compiler_inputs']['path']).read_text())
    require(row['actual_compiler_inputs']['uncovered'] == 0 and all(digest(p) == s for p,s in actual.items()),
            'compiled source closure changed')
    return row


def kernel_names(build, observed=False):
    inventory = json.loads(Path(build['kernel_inventory']['path']).read_text())
    names = []
    for unit, marker in [('original', 'submit_topkILi1E'),
                         ('proposal', 'radix4ILb0E'), ('proposal', 'radix4ILb1E')]:
        chosen = [r['name'] for r in inventory[('observed-' if observed else '')+unit] if marker in r['name']]
        require(len(chosen) == 1, 'one exact selector definition required for ' + marker)
        names.extend(chosen)
    require(len(set(names)) == 3, 'selector identities overlap')
    return names


def admit(scope, plan):
    build = build_record(scope)
    require(inputs(plan['source_commit']) == plan['input_sha256'] and
            {'clean':build['binary'],'events':build['observed_binary']} == plan['binaries'] and
            kernel_names(build) == kernel_names(build,True) == plan['kernel_ids'],
            'check/native inputs changed')
    rt = b.base.runtime.validation.committed(b.base.runtime.RESULT)
    b.base.runtime.current(rt)
    require(rt['passed'] and rt['ownership_closed'] and not rt['active'] and not rt['fresh_faults'] and
            rt['runtime_files'] == plan['runtime_files'] and rt['preload'] == plan['preload'] and
            native.device_inventory(rt) == plan['physical_device'], 'runtime/device qualification changed')
    return rt


def prepare(mode):
    require(mode in ORDER, 'unknown selector check stage')
    record = b.base.component.gen.committed(RESULT); scope = record[b.KEY]
    previous = scope.get('checks', [])
    require([r['mode'] for r in previous] == ORDER[:ORDER.index(mode)] and all(
        r['passed'] and r['complete'] and r['ownership_closed'] and not r['fresh_faults'] for r in previous),
        'ordered successful gates required; no automatic retry')
    if mode == 'performance':
        require(previous[-1]['query_gate_passed'], 'small actual-event query budget required')
    require('check_plan' not in scope or scope['check_plan']['stage'] == 'closed', 'prior plan still open')
    build = build_record(scope)
    rt = b.base.runtime.validation.committed(b.base.runtime.RESULT)
    folder = ROOT / mode
    require(not folder.exists(), 'fresh check directory required')
    revision = b.base.component.gen.git('rev-parse', 'HEAD').decode().strip()
    loaders = [p for p in rt['runtime_files'] if Path(p).name.startswith('libze_loader.so.')]
    require(len(loaders) == 1, 'one qualified Level Zero property provider required')
    environment = dict(b.base.runtime.validation.committed(b.base.runtime.SETTINGS)['environment'],
        LD_PRELOAD=rt['preload']['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=rt['preload']['loader'],
        ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
        STRATA_SYCL_SELECT_RADIX4='1', STRATA_SYCL_SELECT_CYCLIC='0', STRATA_QSA_FIXTURE_HOLD='1',
        STRATA_SYCL_PREFILL_COMMAND_PROFILE='0')
    names = kernel_names(build)
    jobs = []
    if mode == 'properties':
        jobs = [dict(label=kind+'-properties',arm='original',binary_kind=kind,
                     args=['--properties',loaders[0],*names]) for kind in ('clean','events')]
        environment['LD_DEBUG'] = 'files,bindings'
    elif mode == 'correctness':
        jobs = [dict(label=kind+'-'+arm,arm=arm,binary_kind=kind,
                     args=['--dump',str(folder/(kind+'-'+arm+'.ids'))])
                for kind in ('clean','events') for arm in ARMS]
    elif mode == 'event-smoke':
        jobs = [dict(label=arm, arm=arm, binary_kind='events', args=['--event-smoke','2','events','--dump',
                                              str(folder / (arm + '.ids'))]) for arm in ARMS]
    else:
        environment.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                           UR_LOG_TRACING='level:warning;flush:warning;output:stderr')
        for block in range(3):
            order = ARMS[block:] + ARMS[:block]
            for observer in (['events', 'clean'] if block == 1 else ['clean', 'events']):
                for arm in order:
                    label = f'{block}-{observer}-{arm}'
                    jobs.append(dict(label=label, arm=arm, block=block, observer=observer,binary_kind=observer,
                        args=['--timing', '128', observer, '--dump', str(folder / (label + '.ids'))]))
    plan = dict(stage='prepared', mode=mode, source_commit=revision, input_sha256=inputs(revision),
        binaries={'clean':build['binary'],'events':build['observed_binary']}, kernel_ids=names, property_loader=loaders[0],
        runtime_files=rt['runtime_files'], preload=rt['preload'], physical_device=native.device_inventory(rt),
        environment=environment, jobs=jobs,
        budgets=dict(wall=360, cpu=900, rss_cap=4 << 30, text_cap=32 << 20,
                     file_cap=64 << 20, total_cap=768 << 20), whole_wall_seconds=7200,
        active=False, ownership_closed=True,
        performance_criteria=dict(batch_floor_ms=20, maximum_clock_fraction=.002,
            maximum_observer_wall_ratio=1.20, minimum_service_timer_ticks_per_event=1000,
            maximum_query_ms_per_batch=1000, small_probe_query_cap_ms=1000,
            maximum_geomean_time_ratio=.99, every_process_ratio_below_one=True,
            lengths=[32768,78453,117489,262144], patterns=[0,1], rounds=3, processes_per_arm_mode=3),
        resource_model=dict(queries=256,score_key_passes=10,id_output_bytes=256*2051*4,
            nominal_score_bytes_per_call={str(n):10*256*(n//4+1)*4 for n in (32768,78453,117489,262144)},
            scope='Logical source score reads only;compiler/cache traffic unknown. No scoring,transfers or CPU math inside selector timing. Not physical VRAM/PCIe traffic or a saturation ceiling.'),
        criteria='Property-only first with no kernel enqueue for both builds;identical device IR and native resource properties across the host event seam. Full independent weighted-cell oracle,canaries,immutable inputs,invalid/tail/full-component cases for all3arms/both builds. Next a small traced actual-event smoke:3queries/20cells,2retained actual parallel_for events,without a last-event proxy,and7successful kernel-identified launches per arm;correct IDs/timestamps and a finite separate query budget. Only then synthetic selector-only timed batches:3counterbalanced independent processes/arm/mode,3rounds and8length/pattern cases. Clean uses unchanged production-source objects;events uses the host-only finish seam,requires exactly one actual submitted event per call,and retains distinct events. Batched versus BOTH original and SG16-control must improve clean public-entry wall and observed GPU service by>=1percent geometric mean for every case,with every process ratio<1. No scoring/oracle/transfers inside timing. New guard/capability CPU cost stays charged;event/query observation is separate and bounded. Query-cap breaches make attribution ineligible without erasing clean samples. Native/runtime/maps/immutable environment/physical device,pre-main retained validation,normal owned exit and no fresh fault mandatory. Component256K is not physical full-model qualification;no production/decode change or adoption.',
        retention='Keep individual eligible/ineligible timing samples,first-use separately,exact-ID hashes,queried native resource facts and failure/decision limits. Retire successful API/loader output and dumps after verification;two same-generation clean/captured fixtures only through the imminent mechanism/model decision.')
    admit(scope, plan)
    folder.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (folder / name).mkdir()
    write(folder / 'plan.json', plan)
    scope['check_plan'] = b.compact(plan)
    write(RESULT, record)
    print('Prepared', mode, ';commit frozen plan before execution.', flush=True)


def sample_rows(text, job, plan):
    lines = [s for s in text.splitlines() if s.startswith('qsa_timing ')]
    rows = []
    calibration = re.findall(r'^qsa_clock maximum_pair_ns=([\d.e+-]+)$', text, re.M)
    require(len(calibration) == 1, 'missing timer calibration')
    clock_ns = float(calibration[0]); require(math.isfinite(clock_ns) and clock_ns > 0, 'invalid timer calibration')
    for line in lines:
        row = dict(part.split('=', 1) for part in line.split()[1:])
        mode = row.pop('mode')
        row = {k:float(v) for k,v in row.items()}
        require(all(math.isfinite(v) and v >= 0 for v in row.values()) and mode == job['observer'] and
                row['nq'] == 256 and row['cap'] == 2051 and row['repeats'] == 128,
                'timing sample geometry/values')
        criteria = plan['performance_criteria']
        reasons = []
        if row['total_ms'] < criteria['batch_floor_ms']: reasons.append('batch_duration_floor')
        if 2 * clock_ns > criteria['maximum_clock_fraction'] * row['total_ms'] * 1e6:
            reasons.append('timer_overhead')
        if mode == 'events':
            require(row['events'] == row['repeats'] and 0 < row['service_ns'] <= row['extent_ns'],
                    'missing or invalid actual event service')
            if row['service_ns'] / row['repeats'] < criteria['minimum_service_timer_ticks_per_event'] * row['timer_resolution_ns']:
                reasons.append('service_timer_resolution')
            if row['query_ms'] > criteria['maximum_query_ms_per_batch']:
                reasons.append('query_work_cap')
            named = row['invoke_ms'] + row['retention_ms'] + row['drain_ms']
            named_cpu = row['invoke_cpu_ms'] + row['retention_cpu_ms'] + row['drain_cpu_ms']
            require(named <= row['total_ms'] + .01 and named_cpu <= row['cpu_ms'] + .01,
                    'overlapping host accounting')
            row.update(unclassified_ms=max(0,row['total_ms']-named),
                       unclassified_cpu_ms=max(0,row['cpu_ms']-named_cpu))
        else:
            require(row['events'] == row['service_ns'] == row['extent_ns'] == 0,
                    'observer present in clean batch')
        rows.append(dict(row, observer=mode, eligible=not reasons, ineligible_reasons=reasons))
    expected = {(n,p,r) for n in plan['performance_criteria']['lengths']
                for p in plan['performance_criteria']['patterns'] for r in range(3)}
    require(len(rows) == 24 and {(r['n'],r['pattern'],r['round']) for r in rows} == expected,
            'missing/extra timing batches')
    return rows


def decision(processes, plan):
    criteria = plan['performance_criteria']; groups = []
    for n in criteria['lengths']:
        for pattern in criteria['patterns']:
            def median(block, arm, mode, key):
                p = next(p for p in processes if p['block'] == block and p['arm'] == arm and p['observer'] == mode)
                return statistics.median(r[key] for r in p['samples'] if r['n'] == n and r['pattern'] == pattern)
            overhead = {arm:[median(i,arm,'events','total_ms')/median(i,arm,'clean','total_ms')
                             for i in range(3)] for arm in ARMS}
            observed = all(v <= criteria['maximum_observer_wall_ratio'] for values in overhead.values() for v in values)
            for reference in ('original', 'control'):
                ratios = {}
                for metric, mode, key in [('wall','clean','total_ms'),('service','events','service_ns')]:
                    ratios[metric] = [median(i,'batched',mode,key)/median(i,reference,mode,key) for i in range(3)]
                geomeans = {k:math.exp(statistics.mean(math.log(v) for v in values)) for k,values in ratios.items()}
                passed = observed and all(v <= criteria['maximum_geomean_time_ratio'] for v in geomeans.values()) and \
                    all(v < 1 for values in ratios.values() for v in values)
                groups.append(dict(n=n,pattern=pattern,reference=reference,process_time_ratios=ratios,
                    geomean_time_ratios=geomeans,observer_wall_ratios=overhead,passed=passed))
    eligible = all(s['eligible'] for p in processes for s in p['samples'])
    return dict(eligible=eligible, passed=eligible and all(g['passed'] for g in groups), groups=groups,
                model_adopted=False, hardware_saturation='UNMEASURED')


def compact_outcome(outcome):
    """Keep individual speeds; omit repeated geometry/zero fields from Git."""
    if outcome['mode'] != 'performance': return outcome
    result = dict(outcome, processes=[])
    for process in outcome['processes']:
        row = {k:v for k,v in process.items() if k not in ('samples','device_bindings')}
        row['physical_device_verified_before_and_after'] = len(process['device_bindings']) == 2
        row['samples'] = []
        for n in (32768,78453,117489,262144):
            for pattern in (0,1):
                chosen = [s for s in process['samples'] if s['n'] == n and s['pattern'] == pattern]
                chosen.sort(key=lambda s:s['round'])
                entry = dict(n=n,pattern=pattern,rounds=[s['round'] for s in chosen],
                             eligible=[s['eligible'] for s in chosen])
                for key in ('total_ms','cpu_ms','process_cpu_ms'):
                    entry[key] = [s[key] for s in chosen]
                if process['observer'] == 'events':
                    for key in ('service_ns','extent_ns'):
                        entry[key] = [s[key] for s in chosen]
                    entry['host_partition_medians_ms'] = {k:statistics.median(s[k] for s in chosen) for k in (
                        'invoke_ms','invoke_cpu_ms','retention_ms','retention_cpu_ms','drain_ms','drain_cpu_ms',
                        'unclassified_ms','unclassified_cpu_ms','query_ms','query_cpu_ms','query_process_cpu_ms')}
                if any(s['ineligible_reasons'] for s in chosen):
                    entry['ineligible_reasons'] = [s['ineligible_reasons'] for s in chosen]
                row['samples'].append(entry)
        result['processes'].append(row)
    return result


def smoke_launches(trace, expected_name, arm):
    """Resolve successful actual API returns, never count call entry as work."""
    created, launches = {}, []
    for line in trace.splitlines():
        if '<--- urKernelCreate(' in line and 'UR_RESULT_SUCCESS' in line:
            match = re.search(r'\((_ZTS\S+)\).*?\.phKernel = .*?\((0x[0-9a-f]+)\)',line)
            if match: created[match[2]] = match[1]
        call = re.search(r'<--- (urEnqueueKernelLaunch\w*)\(',line)
        if not call: continue
        require('UR_RESULT_SUCCESS' in line and call[1] in
                ('urEnqueueKernelLaunch','urEnqueueKernelLaunchWithArgsExp'),
                'failed or unsupported native kernel launch')
        handle = re.search(r'\.hKernel = (0x[0-9a-f]+)',line)
        require(handle and created.get(handle[1]) == expected_name,
                'native launch is not the admitted selector')
        require('.workDim = 1,' in line and re.search(r'\.pLocalWorkSize = 0x[0-9a-f]+ \(256\)',line) and
                re.search(r'\.pGlobalWorkSize = 0x[0-9a-f]+ \(768\)',line),
                'small selector launch shape differs')
        local = [(int(i),int(n)) for i,n in re.findall(
            r'UR_EXP_KERNEL_ARG_TYPE_LOCAL, \.index = (\d+), \.size = (\d+)',line)]
        if arm == 'batched':
            require(call[1] == 'urEnqueueKernelLaunchWithArgsExp' and sorted(n for _,n in local) == [64,1024],
                    'batched actual local argument payload differs')
        launches.append(dict(api=call[1],kernel=expected_name,local_arguments=local))
    require(len(launches) == 7, 'missing or extra successful selector launch')
    require(all(r == launches[0] for r in launches), 'small launch contract changed')
    return dict(calls=7,**launches[0])


def run(mode):
    record = b.base.component.gen.committed(RESULT); scope = record[b.KEY]
    plan = json.loads((ROOT / mode / 'plan.json').read_text()); folder = ROOT / mode
    require(plan['mode'] == mode and plan['stage'] == 'prepared' and scope['check_plan'] == b.compact(plan) and
            not (folder / 'owned').exists(), 'fresh committed check required')
    rt = admit(scope, plan)
    b.base.runtime.guards.no_model_owners([])
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    text = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor = re.findall(r'^-- cursor: (.+)$',text,re.M); require(len(cursor) == 1, 'fresh journal cursor required')
    outcome = dict(mode=mode, passed=False, complete=False, active=False, ownership_closed=False, processes=[])
    owner = None
    def persist():
        write(folder / 'record.json',dict(outcome,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    owner = b.base.runtime.supervision.Owner(folder / 'owned', persist)
    deadline = time.monotonic() + plan['whole_wall_seconds']
    try:
        hashes = set()
        for job in plan['jobs']:
            admit(scope, plan)
            require(time.monotonic() < deadline, 'whole selector check deadline')
            label = job['label']; trace, warnings = folder / (label + '.trace'), folder / (label + '.validation')
            env = dict(plan['environment'], TMPDIR=str(folder/'tmp'), XDG_CACHE_HOME=str(folder/'cache'),
                       UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(warnings))
            if mode != 'performance': env['UR_LOG_TRACING'] = 'level:info;flush:info;output:file,' + str(trace)
            binary = plan['binaries'][job['binary_kind']]
            args = [binary['path'], '--arm', job['arm'], *job['args']]
            devices, observed = [], {}
            def snapshot(io):
                proc = Path('/proc',str(io.proc.pid))
                actual = dict(v.split(b'=',1) for v in (proc/'environ').read_bytes().split(b'\0') if b'=' in v)
                require(all(actual.get(k.encode()) == v.encode() for k,v in env.items()), 'native environment changed')
                maps = (proc/'maps').read_text(); mapped = b.base.runtime.executable_mappings(maps)
                require(mapped <= set(rt['runtime_files']) | {str(Path(binary['path']).resolve())},
                        'unqualified executable mapping')
                require({rt['preload']['loader'],rt['preload']['library']['path'],env['ZE_ENABLE_ALT_DRIVERS'],
                         str(Path(env['UR_ADAPTERS_FORCE_LOAD']).resolve())} <= mapped, 'missing qualified providers')
                devices.append(native.live_device(io.proc.pid,plan['physical_device']['name'],plan['physical_device'],rt))
                if mode == 'properties':
                    loader = plan['property_loader']
                    rows = [line.split(maxsplit=5) for line in maps.splitlines()]
                    rows = [r for r in rows if len(r)==6 and 'x' in r[1] and r[5].startswith('/') and str(Path(r[5]).resolve()) == loader]
                    require(len(rows) == 1, 'one executable property-loader object required')
                    observed.update(path=loader,sha256=rt['runtime_files'][loader],device=rows[0][3],inode=int(rows[0][4]))
            def interact(io, entry):
                poll = io.poll
                next_size = 0
                def guarded_poll():
                    nonlocal next_size
                    poll()
                    if time.monotonic() >= next_size:
                        require(b.base.runtime.supervision.artifact_bytes(folder) <= plan['budgets']['total_cap'],
                                'whole check artifact budget')
                        require(not trace.exists() or trace.stat().st_size <= plan['budgets']['file_cap'],
                                'API trace file budget')
                        next_size = time.monotonic() + .5
                io.poll = guarded_poll
                require(io.line(60) == 'qsa_histogram_fixture_arm=' + job['arm'], 'wrong native arm')
                require(io.line(60) == 'QSA_READY name=' + plan['physical_device']['name'], 'wrong native device')
                snapshot(io); io.send(b'GO\n')
                while io.line(plan['budgets']['wall']) != 'QSA_DONE': pass
                snapshot(io); io.send(b'DONE\n'); io.exit(30)
            print('Selector check:',mode,label,flush=True)
            entry, out, err = owner.run(label,args,env,P,interaction=interact,
                **(plan['budgets'] | {'wall':min(plan['budgets']['wall'],deadline-time.monotonic())}))
            require(b.base.runtime.supervision.closed(entry) and entry['exit_code'] == 0, 'selector child failed')
            stdout, stderr = out.read_text(), err.read_text()
            proof = b.base.runtime.validation.preload_evidence(stderr,entry['pid'])
            require(proof['loader'] == rt['preload']['loader'], 'UR provider mismatch')
            body = stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
            require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:',body),
                    'native runtime error')
            require(not warnings.exists() or not warnings.stat().st_size, 'parameter validation warning/error')
            result = {k:v for k,v in job.items() if k != 'args'} | {'passed':True,'device_bindings':devices}
            if mode == 'properties':
                result['loader_binding'] = native.loader_bindings(stderr,observed)
                api = trace.read_text()
                require('PASS properties_no_kernel_enqueue' in stdout and 'urEnqueueKernelLaunch' not in api,
                        'property stage submitted a kernel or did not finish')
                result['properties'] = [dict(p.split('=',1) for p in line.split()[1:])
                                        for line in stdout.splitlines() if line.startswith('qsa_property ')]
                require(len(result['properties']) == 3, 'missing native properties')
            else:
                payload = folder / (label + '.ids'); hashes.add(digest(payload))
                result['output_sha256'] = digest(payload)
                markers = {'correctness':'PASS cases=17', 'event-smoke':'PASS event_smoke_cases=1',
                           'performance':'PASS timing_cases=8'}
                require(markers[mode] in stdout,
                        'incomplete independent fixture coverage')
                if mode == 'event-smoke':
                    rows = [dict(p.split('=',1) for p in line.split()[1:])
                            for line in stdout.splitlines() if line.startswith('qsa_timing ')]
                    require(len(rows) == 1 and all(rows[0][k] == v for k,v in
                        dict(n='20',nq='3',cap='7',repeats='2',events='2',mode='events').items()) and
                        float(rows[0]['service_ns']) > 0, 'small actual-event probe failed')
                    result['actual_launches'] = smoke_launches(trace.read_text(),
                        plan['kernel_ids'][ARMS.index(job['arm'])],job['arm'])
                    result['event_probe'] = rows[0]
                    result['query_budget_eligible'] = float(rows[0]['query_ms']) <= \
                        plan['performance_criteria']['small_probe_query_cap_ms']
                if mode == 'performance':
                    result['samples'] = sample_rows(stdout,job,plan)
                    result['first_shape_use'] = [dict(part.split('=',1) for part in line.split()[1:])
                        for line in stdout.splitlines() if line.startswith('qsa_first ')]
            outcome['processes'].append(result); persist()
        require(mode == 'properties' or len(hashes) == 1, 'exact output IDs differ across arms/processes')
        if mode == 'properties':
            require(len(outcome['processes']) == 2 and
                    outcome['processes'][0]['properties'] == outcome['processes'][1]['properties'],
                    'event observation changed native resource properties')
        if mode == 'event-smoke':
            outcome['query_gate_passed'] = all(p['query_budget_eligible'] for p in outcome['processes'])
        admit(scope,plan)
        if mode == 'performance': outcome['decision'] = decision(outcome['processes'],plan)
        outcome.update(passed=True,complete=True)
    except BaseException as error:
        outcome['error'] = repr(error)
        raise
    finally:
        outcome['ownership_closed'] = owner.active is None and all(b.base.runtime.supervision.closed(c) for c in owner.commands)
        postchecks(outcome,boot,cursor[0])
        outcome['passed'] = bool(outcome['passed'] and 'error' not in outcome and outcome['complete'])
        persist(); scope.setdefault('checks',[]).append(compact_outcome(outcome))
        if mode == 'performance':
            scope['component_advance_eligible'] = bool(outcome['passed'] and outcome.get('decision',{}).get('passed'))
        if mode == 'event-smoke':
            scope['event_service_eligible'] = bool(outcome['passed'] and outcome.get('query_gate_passed'))
        scope['check_plan']['stage'] = 'closed'; write(RESULT,record)
    print(json.dumps({k:outcome.get(k) for k in ('mode','passed','complete','fresh_faults')}),flush=True)


def advance():
    """Main-owned serial gates, with a committed plan/result at each boundary.

    No retries, source edits, branch creation, model launch or adoption. A
    refusal commits its original result and stops the pipeline.
    """
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=P,text=True,timeout=90).strip()
    require(git('branch','--show-current') == 'feature/sycl' and
            not git('diff','--cached','--name-only') and not git('diff','--name-only'),
            'clean primary worktree required before serial advancement')
    source = git('rev-parse','HEAD')
    require(inputs(source), 'committed complete gate source required')
    def save(label):
        relative = str(RESULT.relative_to(P))
        require(git('branch','--show-current') == 'feature/sycl' and
                git('diff','--name-only').splitlines() == [relative] and
                not git('diff','--cached','--name-only'),
                'only the current compact result may change')
        git('diff','--check')
        print('Committing selector boundary:',label,flush=True)
        print(git('diff','--stat','--',relative),flush=True)
        git('add','--',relative)
        git('commit','-m','test(sycl): '+label)
        git('push','origin','feature/sycl')
        require(git('ls-remote','origin','refs/heads/feature/sycl').split()[0] ==
                git('rev-parse','HEAD'), 'remote boundary not verified')
    record = b.base.component.gen.committed(RESULT); scope = record[b.KEY]
    previous = scope.get('checks',[])
    require([r['mode'] for r in previous] == ORDER[:len(previous)] and
            all(r['passed'] and r['complete'] and r['ownership_closed'] and
                not r['fresh_faults'] for r in previous),
            'resume only the exact successful prefix;no retry or skip')
    for mode in ORDER[len(previous):]:
        prepare(mode)
        save('freeze '+mode+' selector plan')
        try:
            run(mode)
        except BaseException:
            if git('diff','--name-only'):
                save('preserve refused '+mode+' selector gate')
            raise
        save('record closed '+mode+' selector gate')
    final = b.base.component.gen.committed(RESULT)[b.KEY]
    print(json.dumps(dict(component_advance_eligible=final.get('component_advance_eligible',False),
                          model_selected=False)),flush=True)


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if sys.argv[1] == 'advance':
            advance()
        else:
            {'prepare':prepare,'run':run}[sys.argv[1]](sys.argv[2])
