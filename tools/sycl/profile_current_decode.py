#!/usr/bin/env python3
"""Current cache0 MTP device/CPU attribution with matched observer controls.

Main-authored from this fork's owned benchmark interfaces; repository LICENSE.
No engine rebuild, XeStrata input, counter request or clean-speed claim.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import select
import shutil
import socket
import statistics
import sys
import time

sys.dont_write_bytecode = True
import check_cache_loan_long as long
import profile_compat_exec as collector
import current_profile_control as control
import cpu_mixed_cache_bench as baseline
import cpu_service_evidence as leaves
import decode_service_evidence as caller
import decode_window_trace as device
from profile_cache_map_decode import client_decode_window

P, B, RESULT = collector.P, collector.B, collector.RESULT
ROOT, KEY = B / 'current-decode-device-service-v1', 'current_decode_device_service_v1'
SELF = Path(__file__).resolve()
READ, RUNS, ORDER = 32769, 2, ('reference-before', 'profile', 'reference-after')
require, digest, write, signature = collector.require, collector.digest, collector.write, collector.signature
HEADER, READER = 'include/strata/sycl/decode_service_profile.hpp', 'tools/sycl/decode_service_evidence.py'
BUDGET = dict(wall=2400, cpu=14400, rss_cap=104 << 30, text_cap=64 << 20,
              file_cap=1024 << 20, total_cap=1536 << 20)


def inputs(revision):
    pins = long.bench.sources(revision)
    for name, raw in long.small.loan.source.blobs(P, revision, (
            'tests/core/test_decode_window_trace.py', 'tools/sycl/measurement_environment.json')):
        require((P / name).read_bytes() == raw, 'commit invoked host test/settings: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def observer_projection(proof, sources):
    value, after = copy.deepcopy(proof), dict(sources)
    require(value['observer_source_files'] == len(after) and value['observer_source_sha256'] == control.signature(after),
        'new observer source census differs')
    for name in ('tools/sycl/profile_current_decode.py', 'tools/sycl/test_profile_current_decode.py'):
        require(value['declared_observers'].get(name) == after.get(name) and name in after
            and name in value['additional_observers'], 'declare only exact uncompiled profile observers')
        del value['declared_observers'][name]; del after[name]
        value['additional_observers'].remove(name); value['observer_source_files'] -= 1
    value['observer_source_sha256'] = control.signature(after)
    return value


def basis():
    extra = {'tools/sycl/profile_current_decode.py', 'tools/sycl/test_profile_current_decode.py'}
    native = control.fixture_build.model.native(observers=control.fixture_build.OBSERVERS | control.EXTRA | extra)
    prior = collector.adapter.validation.committed(RESULT, (control.KEY,))
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete'] and prior['ownership_closed']
        and not prior['active'] and prior['host']['tests'] == 9 and len(prior['cases']) == 2
        and all(c['passed'] and c['native_exit_code'] == 0 and not c['fresh_faults'] for c in prior['cases'])
        and prior['cases'][1]['trace']['collection_active_from_exec'], 'closed current-runtime controls first')
    source = control.fixture_build.model.base.original.device.common.source
    after = source.pin(P, control.fixture_build.model.base.original.device.common.PREFIXES)['source_sha256']
    require(observer_projection(native['model_observer_admission'], after) == prior['native_proof'],
        'recover exact current control admission before adding observers')
    pin = json.loads((control.ROOT / 'record.json').read_text())['source']
    require(all(digest(P / n) == h for n, h in pin['source_sha256'].items())
        and {n: hashlib.sha256(raw).hexdigest() for n, raw in source.blobs(P, pin['revision'], pin['source_sha256'])}
        == pin['source_sha256'], 'tested control source changed')
    require(digest(prior['binary']['path']) == prior['binary']['sha256'] and
        collector.adapter.profile.collector_assets(collector.adapter.validation.committed(RESULT)) == prior['assets'],
        'control fixture or collector changed')
    old = collector.adapter.validation.committed(RESULT, ('decode_service_reader_contract_v1',))
    require(old['passed'] and old['complete'] and old['ownership_closed'] and not old['active']
        and old['native_header_binding']['header_sha256'] == native['project']['source_sha256'][HEADER]
        and old['sources'][READER] == digest(P / READER) and old['reader_controls']['corrupt_cases_rejected'] == 28
        and old['reader_controls']['exact_integer_clock_conservation']
        and '-DSTRATA_CPU_SERVICE_PROFILE:BOOL=ON' in native['recipe']['configure'], 'same compiled caller contract required')
    return native, dict(native_source_commit=native['native_source_commit'], binary=native['binaries']['strata'],
        native_observer_admission=native['model_observer_admission'], control_key=control.KEY,
        control_source=pin['revision'], control_sha256=signature(prior), fixture=prior['binary'],
        collector_trace_sha256=prior['cases'][1]['trace']['sha256'], assets=prior['assets'],
        caller_header_sha256=old['native_header_binding']['header_sha256'], caller_reader_sha256=old['sources'][READER])


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('sources', 'models', 'runtime_files')}
    out['source_inputs'] = dict(files=len(row['sources']), sha256=signature(row['sources']))
    out['plans'] = [{k: v for k, v in plan.items() if k != 'environment'} |
        dict(environment_sha256=signature(plan['environment'])) for plan in row['plans']]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def current(row, full=False):
    require(collector.adapter.validation.committed(RESULT, (KEY,)) == compact(row), 'commit exact model-profile plan/result')
    require(inputs(row['source_commit']) == row['sources'], 'model-profile source changed')
    _, proof = basis(); require(proof == row['qualification'], 'selected native/collector/caller proof changed')
    probe, reuse = long.small.loan.current_probe()
    require(probe == row['native_validation_probe'] and reuse == row['runtime_reuse'], 'current runtime proof changed')
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
        json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'] and
        all(digest(n) == sha for n, sha in row['runtime_files'].items()), 'runtime ELF inventory changed')
    require(long.small.loan.guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')
    baseline.previous.shared.models.model_pins(row['configuration'], row['models'], full=full)
    manifest = long.bench.python_runtime.read_manifest(row['python_contract'])
    long.bench.python_runtime.validate_manifest(manifest)
    for plan in row['plans']:
        long.bench.python_runtime.require_environment(plan['environment'])
        require(long.bench.python_runtime.selection(plan['environment']) == row['python_selected'] and
            all(digest(Path(plan['root']) / n) == sha for n, sha in plan['generated'].items()), 'Python/config/native plan changed')
    return manifest


def prepare():
    require(not ROOT.exists() and KEY not in collector.adapter.validation.committed(RESULT), 'fresh bounded current profile only')
    native, proof = basis()
    seed = json.loads((baseline.ROOT / 'record.json').read_text())
    public = collector.adapter.validation.committed(RESULT, (baseline.KEY,))
    require(public == baseline.core.compact(seed) and seed['stage'] == 'closed' and seed['passed'] and seed['complete']
        and seed['ownership_closed'] and not seed['active'] and len(seed['processes']) == 6
        and seed['binary'] == proof['binary'] and seed['native_source_commit'] == proof['native_source_commit']
        and not any(v['reproducible_TG_gain'] for v in seed['decision']['workloads'].values()),
        'preserve closed rejected cache comparison;profile cache0 independently')
    reference = seed['plans'][0]
    seed_env = baseline.environment(seed, reference)
    require(baseline.runner.comparison.fingerprint(seed_env) == reference['environment_sha256'], 'original reference environment changed')
    probe, reuse = collector.adapter.feature.current_probe()
    require(probe == seed['native_validation_probe'], 'current probe changed')
    contract, manifest = baseline.runner.model.python_manifest()
    files = json.loads(Path(seed['runtime_manifest']['path']).read_text())
    row = dict(stage='prepared', source_commit=collector.adapter.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
        sources=inputs(collector.adapter.source.git(P, 'rev-parse', 'HEAD').decode().strip()),
        qualification=proof, native_source_commit=proof['native_source_commit'], binary=proof['binary'],
        configuration=seed['configuration'], models=seed['model_assets'], template=seed['template'],
        python_contract=contract, python_selected=manifest['selected'], native_validation_probe=probe, runtime_reuse=reuse,
        runtime_manifest=seed['runtime_manifest'], runtime_files=files, boot=seed['boot'], driver=seed['driver'],
        plans=[], cases=[], host=None, complete=False, passed=False, active=False, ownership_closed=True,
        budget=BUDGET, whole_bytes=2 << 30,
        plan=dict(question='Which current cache0 MTP GPU services,conserved caller waits and CPU leaves dominate near32769context?',
            order='Three fresh immutable reference-before,profile,reference-after processes;one24output thinking warmup and two32769input/24output requests each. Commit each closure;stop first failure,no retries.',
            constants='Same393/e8,13ELFs,static1256slots,pcie0,adapt0,spec4,MTP16,5workers+host,262144logical/20480resident INT8,8192throughput BF16,vLLM/native16/GDN16,head2/dense1 and qualified immutable safety.',
            observers='All arms charge parameter validation,stage trace,legacy timing and caller schema2. Profile additionally enables CPU leaf/census and active-from-exec unitrace host/device/call/kernel logging. This is not clean speed.',
            acceptance='Current source/build/fixture/collector/runtime/Python/15model asset identities;frontend probes,actual route and24CPUcontrols,no skips. Exact fresh32769bodies/counts/24outputs,quiet pre-main validation,normal native+benchmark0,closed owners,no fresh fault. Caller integer partitions conserve;profile CPU leaves bind its own census. Included GPU IDs join unique native flows in guarded RAW windows. No pause/resume or environment mutation.',
            observer='Per request prompt/decode/clientTTFT profile divided by median of bracketing references must each be<=1.5 with identical request hashes and exactly24outputs. This is a prospective local combined-observer diagnostic screen,not the old5percent incremental-caller gate,which remains failed.',
            clocks='Sampled RAW/MONOTONIC brackets<=1ms,offset envelope<=50ms,BOOTTIME continuity;interior client decode clips100ms at each end,>=100ms. Sampled offsets do not prove unsampled clock error.',
            retention='1GiB/file,1.5GiB/job,2GiBwhole,2400s/104GiBRSS;one complete native trace is needed because paused graph construction loses replays. Reduce after native exit under240s/16GiBCPU owner;retain compact samples/clock/conservation/topservices and retire normal timelines/API/JIT.',
            limits='24output requests are resource diagnostics,not sustainedTG,quality,agent/session,length scaling,physical262144/RESTORE,default adoption or1000PP/70TG proof. Service/API sums overlap;CPU leaves include memory stalls;requested bytes and observed gaps are not bus/VRAM/DRAM traffic or hardware saturation. No engine rebuild.'))
    require(shutil.disk_usage(B).free >= 4 << 30, 'bounded profile disk headroom')
    baseline.previous.shared.models.model_pins(row['configuration'], row['models'], full=False)
    ROOT.mkdir(mode=0o700)
    for label in ORDER:
        root = ROOT / label; root.mkdir(mode=0o700)
        for name in ('tmp', 'cache', 'trace'): (root / name).mkdir(mode=0o700)
        env = {k: v.replace(reference['root'], str(root)) for k, v in seed_env.items()}
        require(not any(k.startswith('UNITRACE_') for k in env) and not env.get('LD_PRELOAD'), 'fresh observer environment')
        env.update(STRATA_MEASUREMENT_ROOT=str(root),STRATA_SYCL_DECODE_SERVICE_PROFILE='1',STRATA_DECODE_TIMING='1',
            STRATA_CPU_SERVICE='1' if label=='profile' else '0',STRATA_CPU_WORK_CENSUS='1' if label=='profile' else '0',
            STRATA_CPU_COMPACT_GU_MIB='0')
        prof = None
        if label == 'profile':
            prof = dict(collector.adapter.profile.profiling(root, proof['assets']), mode='aggregate')
            env.update(UNITRACE_HostTiming='1',UNITRACE_ChromeKernelLogging='1',UNITRACE_ChromeCallLogging='1',UNITRACE_ChromeEventBufferSize='128')
        cfg = copy.deepcopy(seed['configuration']); cfg['log'] = str(root / 'engine.stderr')
        write(root / 'config.json', cfg)
        plan = dict(binary=row['binary'], native_validation_probe=probe)
        if prof: plan['native_profile'] = prof
        write(root / 'plan.json', plan)
        with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        argv=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
            '--prefill-exact-tokens',str(READ),'--think','--runs',str(RUNS),'--profile-request-gates','--profile-startup-gate',
            '--workload-id','current_decode_resource_20261114','--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]
        row['plans'].append(dict(label=label,root=str(root),environment=env,configuration=cfg,profile=prof,
            generated={n:digest(root/n) for n in ('config.json','plan.json')},command=argv))
    save(row)
    print('Prepared current cache0 reference/profile/reference32769input diagnostics;commit before host controls.',flush=True)


def route_config(path):
    sys.path.insert(0,str(P));from serve import server
    cfg=json.loads(Path(path).read_text());args=server.engine_args(cfg,exe=cfg['exe'],announce=False)
    def value(flag):return args[args.index(flag)+1] if flag in args else None
    wanted={'--expert-cache':'1256','--prefill':'8192','--max-context':'262144','--kv-resident':'20480',
        '--spec':'4','--pool-workers':'5','--pcie-frac':'0','--adapt-swaps':'0','--prefill-strategy':'throughput'}
    require(all(value(k)==v for k,v in wanted.items()) and '--mtp' in args and '--draft-model' not in args
        and value('--draft') in (None,'mtp') and cfg['backend']=='sycl','actual current MTP route differs')
    print(json.dumps(dict(args=args,backend=server.drafting.detect_hardware())),flush=True)


def host():
    row = json.loads((ROOT / 'record.json').read_text()); manifest = current(row)
    require(row['stage'] == 'prepared' and row['host'] is None, 'fresh host controls only')
    owner = collector.adapter.ownership.Owner(ROOT / 'host-owned'); outcome = dict(passed=False, GPU_executed=False, results=[])
    try:
        env = row['plans'][0]['environment']; allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
        for script in ('tools/strata_bench.py','serve/server.py'):
            entry,out,err = owner.run(Path(script).stem,[row['python_selected']['path'],'-B',str(long.bench.python_runtime.SOURCE),'probe',str(P / script),row['template']['path']],env,P,
                wall=60,cpu=60,rss_cap=1<<30,text_cap=1<<20,file_cap=2<<20,total_cap=8<<20)
            require(collector.adapter.ownership.closed(entry) and entry['exit_code']==0 and not err.stat().st_size,'frontend Python probe failed')
            meta=json.loads(out.read_text());require(meta['executable']==row['python_selected']['path'],'host interpreter differs')
            for name in [*meta['modules'],*meta['code_maps']]:
                path=Path(name).resolve();require(row['sources'].get(str(path.relative_to(P)))==digest(path) if path.is_relative_to(P) else str(path) in allowed,'unfrozen frontend provider')
            outcome['results'].append(dict(role=script,passed=True))
        for label,args,count in [('profile-reader',['-m','unittest','-v','tools.sycl.test_profile_dflash2','tools.sycl.test_profile_current_decode'],8),
            ('guarded-window',['-m','unittest','discover','-v','-s','tests/core','-p','test_decode_window_trace.py'],16)]:
            entry,out,err=owner.run(label,[row['python_selected']['path'],'-B',*args],env,P,wall=60,cpu=60,rss_cap=1<<30,text_cap=1<<20,file_cap=2<<20,total_cap=8<<20)
            text=out.read_text()+err.read_text();require(collector.adapter.ownership.closed(entry) and entry['exit_code']==0 and f'Ran {count} tests' in text and '\nOK\n' in text and 'skipped' not in text,'CPU controls failed: '+label)
            outcome['results'].append(dict(role=label,passed=True,tests=count,skipped=0))
        for plan in row['plans']:
            entry,out,_=owner.run(plan['label'],[row['python_selected']['path'],'-B',str(SELF),'route-config',str(Path(plan['root'])/'config.json')],plan['environment'],P,
                wall=60,cpu=60,rss_cap=1<<30,text_cap=1<<20,file_cap=2<<20,total_cap=8<<20)
            require(collector.adapter.ownership.closed(entry) and entry['exit_code']==0,'actual route probe failed')
            actual=json.loads(out.read_text());require(actual['backend'][0]=='sycl','actual backend differs')
            plan['native_args']=actual['args']
        require(all(p['native_args']==row['plans'][0]['native_args'] for p in row['plans']),'matched native args differ')
        outcome.update(passed=True,tests=24,skipped=0)
    except BaseException as error: outcome['error']=repr(error)
    finally:
        outcome['ownership_closed']=owner.active is None and all(collector.adapter.ownership.closed(c) for c in owner.commands)
        outcome['passed']=bool(outcome['passed'] and outcome['ownership_closed'])
        row.update(host=outcome,stage='ready' if outcome['passed'] else 'failed',ownership_closed=outcome['ownership_closed'],active=bool(owner.active));save(row)
    require(outcome['passed'],outcome.get('error','host admission failed'))
    print('Frontend/route/24CPU controls pass;commit before native execution.',flush=True)


def journal(entries):
    events=[x for x in entries if x['kind']!='identity'];expected=[('warm-up/chat',None),('prefill-only',0),('prefill-only',1)]
    require(len(events)==6,'exact warmup/two-request journal required')
    hashes=[]
    for i,(phase,run) in enumerate(expected):
        start,end=events[2*i:2*i+2]
        require(start['kind']=='start' and end['kind']=='result' and all(x['phase']==phase and x.get('run')==run and x.get('turn') is None for x in (start,end)) and
            start['max_tokens']==24 and start['thinking'] is True and end['request_index']==start['index'] and end['sample']['ok'] is True and
            end['sample']['request_sha256']==start['request_sha256'] and re.fullmatch('[0-9a-f]{64}',start['request_sha256']) is not None,'journal request identity differs')
        hashes.append(start['request_sha256'])
    return dict(warmups=1,long_requests=2,request_sha256=hashes)


def families(events, begin, end):
    rows={};begin/=1000;end/=1000
    for e in events:
        if e.get('ph')!='X' or e.get('cat')!='gpu_op':continue
        a,z=max(begin,e['ts']),min(end,e['ts']+e['dur'])
        if a>=z:continue
        kind=device.kind(e['name'])
        name=kind
        r=rows.setdefault(name,dict(count=0,clipped_sum_us=0.,intervals=[]));r['count']+=1;r['clipped_sum_us']+=z-a;r['intervals'].append((a,z))
    return {name:dict(count=r['count'],clipped_sum_us=r['clipped_sum_us'],interval_union_us=device.union_us(r['intervals'])) for name,r in rows.items()}


def observer_decision(cases):
    require(len(cases)==3 and [c['label'] for c in cases]==list(ORDER),'complete reference/profile/reference required')
    result=[]
    for index in range(RUNS):
        samples=[c['samples'][index] for c in cases]
        require(len({s['request_sha256'] for s in samples})==1,'observer request bodies differ')
        values={}
        for metric in ('prompt_ms','decode_ms','ttft_s'):
            numbers=[s[metric] for s in samples];require(all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in numbers),'invalid observer clock')
            ratio=numbers[1]/statistics.median((numbers[0],numbers[2]))
            values[metric]=dict(reference_samples=[numbers[0],numbers[2]],profile=numbers[1],ratio=ratio,screen_passed=ratio<=1.5)
        fixed_work=all(s['generated_tokens']==24 for s in samples)
        result.append(dict(request=index,fixed_24_output_work=fixed_work,phases=values,observer_screen_passed=fixed_work and all(v['screen_passed'] for v in values.values())))
    return dict(requests=result,observer_screen_passed=all(v['observer_screen_passed'] for v in result),clean_speed_qualified=False,sustained_TG_qualified=False)


def reduce_case(index):
    row=json.loads((ROOT/'record.json').read_text());plan=row['plans'][index];root=Path(plan['root'])
    require(plan['profile'] is not None and row['sources'][str(SELF.relative_to(P))]==digest(SELF),'profile reducer source/arm differs')
    observation=json.loads((root/'execution.json').read_text());require(observation['native_execution_complete'] and observation['native_exit_code']==0,'closed native run required before reduction')
    traces=list((root/'trace').glob('*.json'));require(len(traces)==1 and traces[0].stat().st_size<=BUDGET['file_cap'],'one finite flushed native trace required')
    events=json.loads(traces[0].read_text())['traceEvents'];outputs=[]
    for sample,bounds in zip(observation['samples'],observation['windows']):
        clocks=[observation['initial_clock'],bounds['begin_clock'],bounds['end_clock'],observation['final_clock']]
        inner=client_decode_window(sample['client_interval'],clocks)
        summary=device.summarize(events,native_pid=observation['native_identity']['pid'],begin_ns=inner['begin_ns'],end_ns=inner['end_ns'],
            capture_begin_ns=clocks[0]['raw_before_ns'],capture_end_ns=clocks[3]['raw_after_ns'])
        grouped=families(events,inner['begin_ns'],inner['end_ns'])
        require(sum(v['count'] for k,v in grouped.items() if k == 'kernel')==summary['operation_counts'].get('kernel',0),'kernel family count does not conserve observed inventory')
        outputs.append(dict(window=inner,services=summary,named_families=grouped))
    write(root/'reduced.json',dict(trace_sha256=digest(traces[0]),trace_bytes=traces[0].stat().st_size,requests=outputs,
        family_limits='Command-kind sums overlap;top services retain actual names. Not hardware utilization,complete device coverage or physical traffic.'))
    print('Reduced two guarded decode windows;complete coverage/physical traffic remain unqualified.',flush=True)


def run(index):
    row=json.loads((ROOT/'record.json').read_text());manifest=current(row,full=index==0)
    require(row['stage'] in ('ready','running') and len(row['cases'])==index<len(ORDER) and all(c['passed'] for c in row['cases']),'fresh frozen process order only')
    plan=row['plans'][index];root=Path(plan['root']);env=plan['environment'];prof=plan['profile']
    require(not (root/'execution.json').exists(),'no duplicate process')
    cursor=collector.adapter.profile.cursor();owner=controls=reducer=None
    initial=(time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
    observed=dict(label=plan['label'],passed=False,complete=False,model_executed=False,native_execution_complete=False,
        initial_clock=collector.adapter.profile.clock_pair(),windows=[])
    def persist():write(root/'execution.json',dict(observed,active=bool((owner and owner.active) or (controls and controls.active) or (reducer and reducer.active)),commands=owner.commands if owner else []))
    def live(entry):
        claim=json.loads((root/'engine-start.json').read_text());pid=claim['native']['pid']
        require(long.identity.identity_matches(collector.adapter.ownership.identity(pid),claim['native'],entry['pid']) and
            str(Path('/proc',str(pid),'exe').resolve())==row['binary']['path'] and claim['args']==['--serve',*plan['native_args']],'native identity/arguments differ')
        expected=collector.expected_environment(dict(row,environment=env,profile=prof)) if prof else collector.adapter.engine_once.native_environment(row,env)
        actual=dict(v.split('=',1) for v in Path('/proc',str(pid),'environ').read_text().split('\0') if '=' in v)
        collector.adapter.profile.same_environment(actual,expected)
        files=dict(row['runtime_files'])
        if prof:files[prof['collector']['path']]=prof['collector']['sha256'];collector.adapter.profile.session_state(prof['session'],0)
        meta=long.small.model.runtime_evidence(dict(row,probe=row['native_validation_probe'],environment=env,runtime_files=files,
            native_build=dict(binaries=dict(strata=row['binary']))),pid,(root/'engine.stderr').read_text(errors='replace'))
        observed.update(meta,model_executed=True,native_identity=claim['native'])
        long.bench.python_runtime.check_running(entry['pid'],P/'tools/strata_bench.py',manifest['selected'],env,{v['realpath'] for v in manifest['inventory']['files'].values()},plan['command'][2:])
    def gate(io,entry,stage,phase,request):
        while True:
            line=io.line(1000)
            if line.startswith('STRATA_PROFILE_GATE '):break
        require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==dict(stage=stage,phase=phase,run=request),'benchmark gate differs')
        clock=collector.adapter.profile.clock_pair()
        delta=[a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),initial)]
        require(min(delta)>=0 and abs(delta[0]-delta[1])<=50000000,'clock/suspend discontinuity')
        require(collector.adapter.ownership.artifact_bytes(ROOT)<=row['whole_bytes'],'whole profile output budget')
        live(entry);warning=root/'validation.log';require(not warning.exists() or not warning.stat().st_size,'validation warning/error')
        return clock
    def interact(io,entry):
        gate(io,entry,'ready','startup',-1)
        text=(root/'engine.stderr').read_text(errors='replace')
        chunks=re.findall(r'^strata prefill: FP16 MoE compact .*?chunk=(\d+),',text,re.M)
        free=re.findall(r'^strata serve: (\d+) MiB of VRAM free with everything loaded$',text,re.M)
        require(chunks==['8192'] and len(free)==1 and int(free[0])>0 and "chunk's prompt buffers need" not in text,'actual8192 startup differs')
        observed['startup']=dict(chunk=8192,free_MiB=int(free[0]));print(plan['label']+' startup8192 admitted',flush=True);io.send(b'continue\n')
        for request in range(RUNS):
            bounds={}
            for stage in ('begin','end'):
                bounds[stage+'_clock']=gate(io,entry,stage,'prefill-only',request)
                bounds[stage]=(root/'engine.stderr').stat().st_size
                print(f'{plan["label"]} request{request} {stage}',flush=True);io.send(b'continue\n')
            observed['windows'].append(bounds);persist()
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(time.monotonic()<deadline,'native/benchmark teardown deadline')
            if select.select([io.proc.stdout],[],[],.02)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        require(not prof or not collector.adapter.profile.session_path(prof['session']).exists(),'preexisting profile session')
        owner=collector.adapter.ownership.Owner(root/'owned',persist)
        entry,_,_=owner.run('bench',plan['command'],env,P,interaction=interact,**row['budget'])
        require(collector.adapter.ownership.closed(entry) and entry['exit_code']==0,'benchmark closure failed')
        native_record=dict(row,native_profile=prof) if prof else row
        end=long.bench.evidence.native_end(root,native_record,dict(config=dict(args=plan['native_args'])),row['boot'],entry['pid'])
        observed.update(native_execution_complete=True,native_exit_code=end['exit_code'],benchmark_exit_code=entry['exit_code'],final_clock=collector.adapter.profile.clock_pair())
        outputs=list((root/'bench').glob('*/results.json'));require(len(outputs)==1,'one benchmark result required')
        result=json.loads(outputs[0].read_text());require(result['complete'] and not result['results']['failures'],'incomplete benchmark')
        sys.path.insert(0,str(P/'tools'));import strata_bench as client
        require(client.verify_request_journal(outputs[0].parent)['valid'],'invalid complete request journal')
        entries=[json.loads(v) for v in (outputs[0].parent/'requests.jsonl').read_text().splitlines()];observed['journal']=journal(entries)
        cohort=result['results']['prefill_only'];samples=cohort['requests'];counts=cohort['count_preflight']
        require(len(samples)==len(counts)==len(observed['windows'])==RUNS,'two completed requests required')
        raw=(root/'engine.stderr').read_bytes();text=raw.decode()
        require(not any(s in text for s in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called','teardown uncertain','[WARNING]','[ERROR]')) and
            re.search(r'(?i)(lost events|dropped events|buffer.{0,20}overflow|trace.{0,20}lost)',text) is None,'runtime/collector error or reported trace loss')
        ledger=caller.extract(text,[1,2,3]);observed['caller']=[caller.compact(r) for r in ledger]
        routes=[]
        for i,(sample,count,bounds) in enumerate(zip(samples,counts,observed['windows'])):
            require(all(sample[k] is True for k in ('ok','eligible','stream_complete','thinking_requested','reasoning_observed')) and
                sample['actual_prompt_tokens']==sample['actual_read_tokens']==count['counted_input_tokens']==READ and sample['actual_reused_tokens']==0 and
                sample['request_sha256']==count['body_sha256'] and sample['generated_tokens']==24 and sample['prompt_tps_source']=='engine' and
                all(type(sample[k]) in (int,float) and math.isfinite(sample[k]) and sample[k]>0 for k in ('prompt_ms','decode_ms','ttft_s','total_s')) and
                sample['ttft_s']<=sample['total_s'],'exact thinking/body/count/clock differs')
            if row['cases']:require(sample['request_sha256']==row['cases'][0]['samples'][i]['request_sha256'],'matched request body differs')
            tail=observed['windows'][i+1]['begin'] if i+1<RUNS else len(raw);part=raw[bounds['begin']:tail].decode()
            route=baseline.route(part,READ)
            route['rounded_decode_timing']=long.bench.evidence.decode_timing_evidence(part,[dict(sample,prompt_tokens=sample['actual_prompt_tokens'],read_tokens=sample['actual_read_tokens'],reused_tokens=0)])[0]
            routes.append(route)
        observed.update(samples=samples,routes=routes)
        if prof:
            headers=re.findall(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*?, (\d+) generated',text,re.M)
            require(len(headers)==3,'three native request headers')
            all_samples=[dict(prompt_tokens=int(n),reused_tokens=int(r),read_tokens=int(t),generated_tokens=int(g)) for n,r,t,g in headers]
            work=baseline.observe(text,all_samples,'reference')
            bindings=[dict(prompt_tokens=r['read_tokens'],windows=r['windows'],rows=r['rows'],cpu_jobs=r['CPU_jobs'],
                cpu_entries=r['CPU_entries'],pool_ms=r['pool_ms']) for r in work['requests']]
            formats=baseline.census.manifest_formats(baseline.model.device.candidate.cpu.inputs.METADATA)
            membership=baseline.census.expected_membership(P/'data/expert-profile.bin',1256)
            leaf=leaves.extract(text,bindings,formats,membership)
            observed['CPU_work']=work
            observed['CPU_leaf']=[{k:v for k,v in r.items() if k!='format_bins'} | dict(format_operation_totals=[dict(gu=gu,down=down,op=op,
                calls=sum(b['calls'] for b in r['format_bins'] if (b['gu'],b['down'],b['op'])==(gu,down,op)),
                wall_ns=sum(b['wall_ns'] for b in r['format_bins'] if (b['gu'],b['down'],b['op'])==(gu,down,op)),
                cpu_ns=sum(b['cpu_ns'] for b in r['format_bins'] if (b['gu'],b['down'],b['op'])==(gu,down,op)))
                for gu,down,op in sorted({(b['gu'],b['down'],b['op']) for b in r['format_bins']})]) for r in leaf]
        else:
            require(not leaves.records(text) and baseline.census.PREFIX not in text,'reference observers must stay off')
        persist()
        if prof:
            reducer=collector.adapter.ownership.Owner(root/'reduce-owned',persist)
            command=[row['python_selected']['path'],'-B',str(SELF),'reduce',str(index)]
            reduced,_,_=reducer.run('reduce',command,env,P,wall=240,cpu=240,rss_cap=16<<30,text_cap=1<<20,file_cap=2<<20,total_cap=row['budget']['total_cap'])
            require(collector.adapter.ownership.closed(reduced) and reduced['exit_code']==0,'guarded profile reduction failed')
            observed['device']=json.loads((root/'reduced.json').read_text())
        current(row,full=index==2)
        observed.update(passed=True,complete=True)
    except BaseException as error:observed['error']=repr(error)
    finally:
        if prof and owner and owner.active is None and collector.adapter.profile.session_path(prof['session']).exists():
            controls=collector.adapter.ownership.Owner(root/'controls',persist)
            try:
                stopped,_,_=controls.run('stop',[prof['launcher']['path'],'--stop',prof['session']],env,P,wall=15,cpu=15,rss_cap=2<<30,text_cap=1<<20,file_cap=32<<20,total_cap=row['budget']['total_cap'])
                require(collector.adapter.ownership.closed(stopped) and stopped['exit_code']==0,'profile session stop failed')
            except BaseException as error:observed['control_error']=repr(error)
        observed['ownership_closed']=all(o is None or o.active is None and all(collector.adapter.ownership.closed(c) for c in o.commands) for o in (owner,controls,reducer))
        try:
            observed['fresh_faults']=long.small.state.faults(cursor);require(not observed['fresh_faults'],'fresh GPU fault')
            require(long.small.loan.guards.driver_guard()==row['driver'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'],'boot/driver changed')
        except BaseException as error:observed['finalization_error']=repr(error)
        observed['passed']=bool(observed['passed'] and observed['ownership_closed'] and 'control_error' not in observed and 'finalization_error' not in observed and
            (not prof or not collector.adapter.profile.session_path(prof['session']).exists()))
        observed['commands']=[dict(label=c['label'],exit_code=c.get('exit_code'),ownership_closed=collector.adapter.ownership.closed(c),error=c.get('error'),errors=c.get('errors',[])) for o in (owner,controls,reducer) if o for c in o.commands]
        row['cases'].append(observed);row.update(stage='running' if observed['passed'] else 'failed',ownership_closed=observed['ownership_closed'],active=any(o and o.active for o in (owner,controls,reducer)))
        row['active']=bool(row['active'])
        if index==2 and all(c['passed'] for c in row['cases']):
            row['observer_decision']=observer_decision(row['cases'])
            row.update(stage='closed',complete=True,passed=row['observer_decision']['observer_screen_passed'])
        save(row)
    require(observed['passed'],observed.get('error',observed.get('finalization_error','profile process failed')))
    print(plan['label']+' closed;commit before next process. No clean-speed claim.',flush=True)


def execute(action, index=None):
    """A source/provider refusal is a closed attempt, never a silent retry."""
    root = ROOT if index is None else ROOT / ORDER[index]
    attempt = root / ('host-attempt.json' if index is None else 'attempt.json')
    with attempt.open('x') as stream:
        json.dump(dict(action=action,index=index,key=KEY),stream)
    try:
        host() if index is None else run(index)
    except BaseException as error:
        row = json.loads((ROOT/'record.json').read_text())
        if row['stage'] != 'failed':
            row.update(stage='failed',passed=False,complete=False,failed_action=action,
                failed_index=index,error=repr(error))
            save(row)
        raise


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='route-config':route_config(sys.argv[2])
    elif len(sys.argv)==3 and sys.argv[1]=='reduce':reduce_case(int(sys.argv[2]))
    else:
        with (B/'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);long.small.loan.guards.no_model_owners([])
            if sys.argv[1:]==['prepare']:prepare()
            elif sys.argv[1:]==['host']:execute('host')
            elif len(sys.argv)==3 and sys.argv[1]=='run':execute('run',int(sys.argv[2]))
            else:raise RuntimeError('profile_current_decode.py prepare|host|run INDEX|route-config FILE')
