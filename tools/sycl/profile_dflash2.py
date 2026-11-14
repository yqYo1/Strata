#!/usr/bin/env python3
"""Bounded current DFlash2 GPU/caller attribution and matched observer screen.

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
import decode_service_evidence as caller
import decode_window_trace as device
from profile_cache_map_decode import client_decode_window

P, B, RESULT = collector.P, collector.B, collector.RESULT
ROOT, KEY = B / 'dflash2-device-service-v2', 'dflash2_device_service_v2'
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


def basis():
    native, proof = long.small.loan.native()
    control = json.loads((collector.ROOT / 'record.json').read_text()); collector.current(control)
    require(control['stage'] == 'closed' and control['passed'] and control['ownership_closed'] and
        not control['active'] and control['host']['tests'] == 3 and control['host']['skipped'] == 0 and
        control['outcome']['trace']['collection_active_from_exec'], 'current active-from-exec control required')
    old = collector.adapter.validation.committed(RESULT, ('decode_service_reader_contract_v1',))
    require(old['passed'] and old['complete'] and old['ownership_closed'] and not old['active'] and
        old['native_header_binding']['header_sha256'] == native['project']['source_sha256'][HEADER] and
        old['sources'][READER] == digest(P / READER) and old['reader_controls']['corrupt_cases_rejected'] == 28 and
        old['reader_controls']['exact_integer_clock_conservation'] and
        '-DSTRATA_CPU_SERVICE_PROFILE:BOOL=ON' in native['recipe']['configure'], 'same compiled caller contract required')
    return proof, dict(native_record=proof['record'], native_source_commit=proof['source_commit'],
        binary=proof['binaries']['strata'], collector_source=control['source']['revision'],
        fixture=control['binary'], collector_trace_sha256=control['outcome']['trace']['sha256'],
        assets=control['assets'], caller_header_sha256=old['native_header_binding']['header_sha256'],
        caller_reader_sha256=old['sources'][READER])


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
    long.small.model.model_pins(row, full=full)
    manifest = long.bench.python_runtime.read_manifest(row['python_contract'])
    long.bench.python_runtime.validate_manifest(manifest)
    for plan in row['plans']:
        long.bench.python_runtime.require_environment(plan['environment'])
        require(long.bench.python_runtime.selection(plan['environment']) == row['python_selected'] and
            all(digest(Path(plan['root']) / n) == sha for n, sha in plan['generated'].items()), 'Python/config/native plan changed')
    return manifest


def prepare():
    require(not ROOT.exists() and KEY not in collector.adapter.validation.committed(RESULT), 'fresh bounded profile only')
    _, proof = basis()
    status = collector.adapter.validation.committed(long.STATUS)
    admitted = status['current_sync']['dflash2_cache_loan_long_v1']
    require(admitted['passed'] and admitted['complete'] and admitted['ownership_closed'] and
        admitted['native_build']['record'] == proof['native_record'], 'same-build exact32K admission required')
    commit = admitted['original_plan_commit']
    old = json.loads(long.small.loan.source.git(P, 'show', commit + ':' + str(long.STATUS.relative_to(P))))['current_sync']['dflash2_cache_loan_long_v1']
    seed = old['plans'][0]
    require(old['stage'] == 'ready' and seed['slots'] == 256 and not seed['borrowed'], 'original owned256 recipe required')
    reference = json.loads((long.small.model.REFERENCE / 'record.json').read_text())
    probe, reuse = long.small.loan.current_probe()
    revision = long.small.loan.source.git(P, 'rev-parse', 'HEAD').decode().strip()
    row = dict(stage='prepared', source_commit=revision, sources=inputs(revision), qualification=proof,
        previous_attempt=dict(commit='d99f1f21471e46c99014771bd9fffc563e8e7243', key='dflash2_device_service_v1',
            reason='V1 source inventory omitted the invoked guarded-window test. Closed before CPU tests/model/GPU;preserve its refusal. V2 pins that file and the measurement settings;no hash refresh.'),
        binary=proof['binary'], native_source_commit=proof['native_source_commit'], native_validation_probe=probe, runtime_reuse=reuse,
        runtime_manifest=reference['runtime_manifest'], runtime_files=json.loads(Path(reference['runtime_manifest']['path']).read_text()),
        models=reference['model_assets'], models_sha256=long.small.model.fingerprint(reference['model_assets']),
        python_contract=reference['python_contract'], python_selected=reference['python_selected'], template=reference['template'],
        boot=reference['boot'], driver=reference['driver'], recipe=dict(commit=commit,path=str(long.STATUS.relative_to(P)),key='current_sync.dflash2_cache_loan_long_v1.plans[0]'),
        plans=[], host=None, cases=[], passed=False, complete=False, active=False, ownership_closed=True,
        budget=BUDGET, whole_bytes=2 << 30,
        plan=dict(question='Which observed device families and conserved caller/wait regions dominate the current DFlash2 route,and how much does whole-process collection perturb it?',
            order='Three fresh immutable processes:reference-before,profile,reference-after. Each uses official bench.sh,one short thinking warmup and two exact32769-input/thinking-on requests,each capped24outputs. Commit each result before next process. No pause/resume or arm switching.',
            constants='Same591c/90f,owned256(no loan),65536INT8resident context,8192chunk,384packed bank,5workers,pcie0,adapt0,spec4,DFlash2 confidence0,absent MTP,head retirement1,throughput BF16/native16/GDN16. Caller schema2 and rounded decode timing enabled in all three processes;parameter validation and stage progress remain charged.',
            acceptance='Current source/native/fixture/census/collector/runtime/Python/15asset identities. Both frontend probes,actual route probe and20CPU controls with no skips. Exact bodies/counts/reuse/route/journal,quiet before-main validation,normal native AND benchmark0,reap/empty,no fresh fault. Complete schema2 records for warmup/request2/request3 with exact integer conservation. Profile stays active from exec through exit/flush;all included GPU IDs join unique native flows with finite guarded RAW clocks and correct PCI lane. Stop first failure;no retry or changed pins.',
            observer='Per request position compare profile prompt/decode intervals and clientTTFT separately with the median of the two reference processes. Each phase ratio must be<=1.5 and generated work must be24in every cell. This is a local observer-admission screen with two bracketing reference processes,not a variance-free estimate,clean speed or sustained TG. Preserve mixed/failing phase results separately.',
            clocks='Four RAW/MONOTONIC pairs per request:pre-spawn,begin,end,post-exit. Brackets<=1ms,observed offset envelope<=50ms,BOOTTIME continuity. Interior decode is first client output+maxOffset+100ms through client end+minOffset-100ms,at least100ms. A sampled envelope is not an unsampled clock-error bound.',
            retention='One profile needs complete history because pausing during graph construction loses later kernels. Budget1GiB/file,1.5GiB/job,2GiBwhole,2400s/104GiBRSS. Reduce after native exit under a separate240s/16GiB CPU owner. Keep compact individual samples,counters,clocks,services,decisions and faults;retire successful full traces/logs/JIT after guarded reduction and observer decision.',
            limits='Diagnostic24-output requests do not qualify sustained TG,strong/weak draft classes,agent/task quality,length scaling,physical262144/RESTORE or adoption. Caller CPU excludes worker CPU. Device command services/API wall are not FLOP utilization,physical RAM/VRAM/PCIe traffic,complete device coverage or hardware saturation. No engine rebuild.'))
    require(shutil.disk_usage(B).free >= 4 << 30, 'finite profile output headroom required')
    # Full model hashing is required immediately before the first native run
    # and after the last; preparation only checks the immutable inventory.
    long.small.model.model_pins(row, full=False)
    ROOT.mkdir(mode=0o700)
    for label in ORDER:
        root = ROOT / label; root.mkdir(mode=0o700)
        for n in ('tmp', 'cache', 'trace'): (root / n).mkdir(mode=0o700)
        env = {k: v.replace(seed['root'], str(root)) for k, v in seed['environment'].items()}
        require(not any(k.startswith('UNITRACE_') for k in env) and not env.get('LD_PRELOAD'), 'no inherited profiler/preload')
        env.update(STRATA_SYCL_DECODE_SERVICE_PROFILE='1', STRATA_DECODE_TIMING='1', STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0')
        prof = None
        if label == 'profile':
            prof = dict(collector.adapter.profile.profiling(root, proof['assets']), mode='aggregate')
            env.update(UNITRACE_HostTiming='1', UNITRACE_ChromeKernelLogging='1', UNITRACE_ChromeCallLogging='1', UNITRACE_ChromeEventBufferSize='128')
        cfg = copy.deepcopy(seed['configuration']); cfg['log'] = str(root / 'engine.stderr')
        write(root / 'config.json', cfg)
        native_plan = dict(binary=row['binary'], native_validation_probe=probe)
        if prof: native_plan['native_profile'] = prof
        write(root / 'plan.json', native_plan)
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        command = ['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
            '--prefill-exact-tokens', str(READ), '--think', '--runs', str(RUNS), '--profile-request-gates',
            '--profile-startup-gate', '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error',
            '--ready-timeout', '600', '--out', str(root / 'bench')]
        row['plans'].append(dict(label=label, root=str(root), environment=env, configuration=cfg, profile=prof,
            generated={n:digest(root / n) for n in ('config.json','plan.json')}, command=command))
    save(row); print('Prepared reference/profile/reference exact32K diagnostic;commit before host controls.', flush=True)


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
        for label,args,count in [('profile-reader',['-m','unittest','-v','tools.sycl.test_profile_dflash2'],4),
            ('guarded-window',['-m','unittest','discover','-v','-s','tests/core','-p','test_decode_window_trace.py'],16)]:
            entry,out,err=owner.run(label,[row['python_selected']['path'],'-B',*args],env,P,wall=60,cpu=60,rss_cap=1<<30,text_cap=1<<20,file_cap=2<<20,total_cap=8<<20)
            text=out.read_text()+err.read_text();require(collector.adapter.ownership.closed(entry) and entry['exit_code']==0 and f'Ran {count} tests' in text and '\nOK\n' in text and 'skipped' not in text,'CPU controls failed: '+label)
            outcome['results'].append(dict(role=label,passed=True,tests=count,skipped=0))
        for plan in row['plans']:
            entry,out,_=owner.run(plan['label'],[row['python_selected']['path'],'-B',str(long.SELF),'route-config',str(Path(plan['root'])/'config.json')],plan['environment'],P,
                wall=60,cpu=60,rss_cap=1<<30,text_cap=1<<20,file_cap=2<<20,total_cap=8<<20)
            require(collector.adapter.ownership.closed(entry) and entry['exit_code']==0,'actual route probe failed')
            actual=json.loads(out.read_text());require(actual['backend'][0]=='sycl','actual backend differs')
            plan['native_args']=actual['args']
        require(all(p['native_args']==row['plans'][0]['native_args'] for p in row['plans']),'matched native args differ')
        outcome.update(passed=True,tests=20,skipped=0)
    except BaseException as error: outcome['error']=repr(error)
    finally:
        outcome['ownership_closed']=owner.active is None and all(collector.adapter.ownership.closed(c) for c in owner.commands)
        outcome['passed']=bool(outcome['passed'] and outcome['ownership_closed'])
        row.update(host=outcome,stage='ready' if outcome['passed'] else 'failed',ownership_closed=outcome['ownership_closed'],active=bool(owner.active));save(row)
    require(outcome['passed'],outcome.get('error','host admission failed'))
    print('Frontend/route/20CPU controls pass;commit before native execution.',flush=True)


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
        name=('named_dflash2_kernel' if 'df2_' in e['name'] else 'other_kernel') if kind=='kernel' else kind
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
        require(sum(v['count'] for k,v in grouped.items() if k in ('named_dflash2_kernel','other_kernel'))==summary['operation_counts'].get('kernel',0),'kernel family count does not conserve observed inventory')
        outputs.append(dict(window=inner,services=summary,named_families=grouped))
    write(root/'reduced.json',dict(trace_sha256=digest(traces[0]),trace_bytes=traces[0].stat().st_size,requests=outputs,
        family_limits='Only names containing df2_ identify a DFlash2 subset;shared kernels remain other_kernel. Family sums overlap;neither a hardware utilization nor a complete drafter-time decomposition.'))
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
                sample['request_sha256']==count['body_sha256'] and 1<=sample['generated_tokens']<=24 and sample['prompt_tps_source']=='engine' and
                all(type(sample[k]) in (int,float) and math.isfinite(sample[k]) and sample[k]>0 for k in ('prompt_ms','decode_ms','ttft_s','total_s')) and
                sample['ttft_s']<=sample['total_s'],'exact thinking/body/count/clock differs')
            if row['cases']:require(sample['request_sha256']==row['cases'][0]['samples'][i]['request_sha256'],'matched request body differs')
            tail=observed['windows'][i+1]['begin'] if i+1<RUNS else len(raw);part=raw[bounds['begin']:tail].decode()
            route=long.bench.request_route(part,'dflash2',sample);route['loan']=long.loans(part,False)
            route['rounded_decode_timing']=long.bench.evidence.decode_timing_evidence(part,[dict(sample,prompt_tokens=sample['actual_prompt_tokens'],read_tokens=sample['actual_read_tokens'],reused_tokens=0)])[0]
            routes.append(route)
        observed.update(samples=samples,routes=routes);persist()
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


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='reduce':reduce_case(int(sys.argv[2]))
    else:
        with (B/'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);long.small.loan.guards.no_model_owners([])
            if sys.argv[1:]==['prepare']:prepare()
            elif sys.argv[1:]==['host']:host()
            elif len(sys.argv)==3 and sys.argv[1]=='run':run(int(sys.argv[2]))
            else:raise RuntimeError('profile_dflash2.py prepare|host|run INDEX')
