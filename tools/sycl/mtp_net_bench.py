#!/usr/bin/env python3
"""Main-owned official long-generation MTP attribution pilot, not adoption.

Repository LICENSE. Same qualified private binary and original prefill in both
fresh processes; only native --mtp-max-t differs. No XeStrata application code.
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
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import mtp_resident_serve as short
import measurement_clock as clock
import generation_compare as comparison
import generation_engine as gen

P, B = short.P, short.B
ROOT = B / 'mtp-net-bench-v1'
RESULT = short.RESULT.with_name('mtp-net-bench-current.json')
WORKLOADS = ('regular-code', 'varied-prose')
WORKLOAD_ID = comparison.WORKLOAD_ID
READER = P / 'bench/sycl/client_resources.py'
sys.path.insert(0, str(READER.parent)); import client_resources as resources
require, digest, write = short.require, short.digest, short.write


def short_proof():
    row = short.gate.validation.committed(short.RESULT)
    require(row['stage'] == 'closed' and row['passed'] and row['complete'] and row['ownership_closed'] and
        not row['active'] and row['final_full_model_hashes'] == 15 and len(row['processes']) == 2 and
        all(p['passed'] and p['exit_code'] == 0 and p['fresh_faults'] == [] for p in row['processes']),
        'closed same-binary cap0/1 model qualification required')
    keys = ('schema','controller_source_commit','base_native_source','native_controller_source_commit','binary',
        'input_fingerprint','private_inputs_sha256','model_reference','configuration','processes','runtime',
        'passed','ownership_closed','final_full_model_hashes')
    return {k:row[k] for k in keys}


def benchmark_pins(revision):
    pins = gen.benchmark_pins(revision)
    for path in (clock.SOURCE,READER,P/'tests/core/test_client_resources.py'):
        name = str(path.relative_to(P)); raw = gen.git('show',revision+':'+name)
        require(path.read_bytes() == raw,'commit exact measurement input: '+name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def seal(row):
    return short.gate.native.fingerprint({k:row[k] for k in ('environments','model_assets')})


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('environments','model_assets','commands')}
    out['clock'] = clock.compact(row['clock'])
    return out


def prepare():
    require(not ROOT.exists() and not RESULT.exists(),'fresh pilot only;no retry')
    native = short.native_gate(); proof = short_proof(); base = short.model.setup()
    require(b'using Clock = std::chrono::steady_clock;' in
        (short.gate.native.ROOT/'source/src/program/generate.cpp').read_bytes(), 'actual native clock source differs')
    require(native['assets']['strata'] == proof['binary'] and base['reference'] == proof['model_reference'],
        'qualified model/source identity differs')
    short.model.model_pins(base['config'],base['models'])
    speed = short.gate.validation.committed(RESULT.with_name('phase-prefill-speed-current.json'))
    provider = speed['host_clock']['provider']
    require(digest(provider['path']) == provider['sha256'],'qualified native clock provider changed')
    revision = gen.git('rev-parse','HEAD').decode().strip(); pins = benchmark_pins(revision)
    ROOT.mkdir(mode=0o700); (ROOT/'host').mkdir()
    host = clock.prepare(proof['binary'],native['compiler'],native['environment'],ROOT/'host',provider)
    environments = {}; plans = []
    for cap in (0,1):
        root = ROOT/str(cap); root.mkdir(mode=0o700)
        for name in ('cache','tmp'): (root/name).mkdir()
        config = copy.deepcopy(base['config'])
        config['args'][config['args'].index('--expert-profile')+1] = str(root/'expert-profile.bin')
        config['args'] += ['--mtp-max-t',str(cap)]
        config.update(log=str(root/'engine.stderr'),env={})
        comparison.bank.admission.require_batched_warmup(config)
        shutil.copyfile(P/'data/expert-profile.bin',root/'expert-profile.bin')
        env = dict(speed['environment_template'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
            STRATA_MEASUREMENT_ROOT=str(root),STRATA_TRACE='1',STRATA_SNAPSHOT_VERIFY='1',
            STRATA_SYCL_CACHE_LOAN_AUDIT='1',STRATA_DECODE_TIMING='1',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
        require('LD_PRELOAD' not in env and 'STRATA_SYCL_PREFILL_SEPARATE_LAYOUT' not in env,
            'native-only preload and original prefill policy required')
        for name in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission',
                     'STRATA_SYCL_PREFILL_SAME_QUEUE','SYCL_CACHE_PERSISTENT'):
            require(env[name] == native['environment'][name],'runtime safety changed: '+name)
        collection = dict(source=dict(path=str(READER),sha256=digest(READER)),seconds=3300,interval=1)
        write(root/'config.json',config)
        write(root/'plan.json',dict(binary=proof['binary'],native_validation_probe=proof['runtime']['preload'],
            resource_collection=collection))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
        plans.append(dict(cap=cap,root=str(root),config=config,collection=collection,
            generated_inputs={str(root/n):digest(root/n) for n in ('config.json','plan.json','expert-profile.bin')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--long-generation-only','--long-generation-speed-only','--think','--runs','1',
                '--profile-request-gates','--workload-id',WORKLOAD_ID,'--fail-fast-engine-error',
                '--ready-timeout','600','--out',str(root/'bench')]))
        environments[str(cap)] = env
    row = dict(stage='host-prepared',source_commit=revision,benchmark_fingerprint=short.gate.native.fingerprint(pins),
        benchmark_files=len(pins),short_proof=proof,short_proof_sha256=short.gate.native.fingerprint(proof),
        clock=host,model_assets=base['models'],model_reference=base['reference'],environments=environments,
        plans=plans,binary=proof['binary'],runtime_reference_sha256=digest(RESULT.with_name('client-resources-current.json')),
        phase_reference_sha256=digest(RESULT.with_name('phase-prefill-speed-current.json')),
        criteria='Attribution pilot only:two fresh processes cap0 then1,same binary/config/original prefill/1256slots/resident131072,4 standard warmups and both official v4 thinking-on workloads each. Actual fresh input>=32768,generated1024..16384 including thought,answer>=512 chars,valid complete SSE/journal/native exit. Zero drafts allowed only forcap1;MTP4 reference counters>=256. Require actual target windows/draft rounds/prompt mode2 cells,source-confirmed<64 fallback,and GPU BF16/mode10 with cache/dense restoration. One same-library65536-pair clock screen before each native process;prompt and decode durations>=10s,observed endpoint+rounding<=0.01%,no subtraction/retry. Require native-only preexec validation,source/assets/runtime/owner/boot/fault gates,whole-cohort15 hashes. Keep all samples;stop first failure. No speed/adoption or MTP-classification qualification from one process per arm.',
        resources='Passive proc/fdinfo reader,1s interval,3300s deadline,32MiB reader budget. Report request-contained scheduled CPU and associated CCS busy cycles plus logical native copy counts. CPU ticks include polling/stalls,CCS busy includes stalls and shared clients;neither proves arithmetic saturation. Actual PCIe traffic/VRAM bandwidth/CPU instruction saturation remain unknown. Stage timers are nested/nonadditive;observers remain charged. Separate repeated clean speed and long/agent/phase preservation remain required.',
        budgets=dict(wall=3600,cpu=54000,rss_cap=104<<30,text_cap=64<<20,file_cap=64<<20,total_cap=128<<20),
        host_budgets=dict(wall=120,cpu=120,rss_cap=4<<30,text_cap=2<<20,file_cap=4<<20,total_cap=16<<20),
        retention='Keep useful individual samples,clock screens,work/resource aggregates,source/configuration/decisions. Replay then retire normal logs/JIT/full resource timeline. One original selected13 and private13 remain for this comparison;remove private generation after closure/next qualified source recovery. Full short-model receipt can retire once these sealed model/environment inputs admit.',
        processes=[],active=False,passed=False,complete=False,ownership_closed=True,
        production_selected=False,model_adoption_qualified=False,clean_speed_qualified=False,
        hardware_saturation_qualified=False)
    row['private_inputs_sha256'] = seal(row)
    write(ROOT/'record.json',row);write(RESULT,compact(row))
    print('Prepared clock build and two official long-generation attribution processes;commit before host gate.',flush=True)


def current(row):
    require(RESULT.read_bytes() == gen.git('show','HEAD:'+str(RESULT.relative_to(P))) and
        json.loads(RESULT.read_text()) == compact(row) and seal(row) == row['private_inputs_sha256'],
        'pilot differs from committed plan/private metadata')
    pins = benchmark_pins(row['source_commit'])
    require(len(pins) == row['benchmark_files'] and short.gate.native.fingerprint(pins) == row['benchmark_fingerprint'],
        'benchmark/controller inputs changed')
    proof = short_proof()
    require(short.gate.native.fingerprint(proof) == row['short_proof_sha256'] and proof == row['short_proof'] and
        short.native_gate()['assets']['strata'] == row['binary'],'native/model qualification changed')
    clock.check(row['clock'])
    if 'probe' in row['clock']: require(digest(row['clock']['probe']['path']) == row['clock']['probe']['sha256'],'clock probe changed')
    require(short.model.setup()['reference'] == row['model_reference'] and
        digest(RESULT.with_name('phase-prefill-speed-current.json')) == row['phase_reference_sha256'] and
        digest(RESULT.with_name('client-resources-current.json')) == row['runtime_reference_sha256'],
        'qualified model/environment/runtime reference changed')
    short.model.model_pins(short.model.setup()['config'],row['model_assets'])
    gen.runtime_assets_unchanged(gen.committed(RESULT.with_name('client-resources-current.json')))
    require(short.gate.native.guards.driver_guard() == proof['runtime']['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == proof['runtime']['boot'],'platform changed')
    for plan in row['plans']:
        for p,sha in plan['generated_inputs'].items(): require(digest(p) == sha,'generated pilot input changed: '+p)


def host():
    row=json.loads((ROOT/'record.json').read_text());current(row)
    require(row['stage']=='host-prepared' and not (ROOT/'host/owned').exists(),'fresh host gate only')
    owner=None
    def persist():write(ROOT/'host/record.json',dict(commands=owner.commands if owner else []))
    try:
        owner=short.gate.native.supervision.Owner(ROOT/'host/owned',persist)
        for label,args in [('tests',['/usr/bin/python3','-B','tools/sycl/test_mtp_net_bench.py']),
                           ('clock-tests',['/usr/bin/python3','-B','tools/sycl/test_measurement_clock.py']),
                           ('compile',row['clock']['command'])]:
            entry,_,_=owner.run(label,args,row['clock']['environment'],P,**row['host_budgets'])
            require(short.gate.native.supervision.closed(entry) and entry['exit_code']==0,'host gate failed: '+label)
        probe=ROOT/'host/clock_probe';row['clock']['probe']=dict(path=str(probe),sha256=digest(probe),bytes=probe.stat().st_size)
        entry,out,_=owner.run('initial-clock',[str(probe)],row['clock']['environment'],P,**row['host_budgets'])
        require(short.gate.native.supervision.closed(entry) and entry['exit_code']==0,'initial clock failed')
        row['clock']['initial_screen']=clock.evidence(out.read_text(),row['clock']['provider'])
        clock.check(row['clock']);row['stage']='host-checked'
    except BaseException as error:row.update(stage='host-failed',error=repr(error))
    finally:
        row['ownership_closed']=owner is None or owner.active is None and all(short.gate.native.supervision.closed(c) for c in owner.commands)
        write(ROOT/'record.json',row);write(RESULT,compact(row))
    print(json.dumps(dict(stage=row['stage'],ownership_closed=row['ownership_closed'])),flush=True)
    return 0 if row['stage']=='host-checked' else 1


def sample_eligible(sample,cap):
    offered,accepted=sample.get('drafts_offered'),sample.get('drafts_accepted')
    common=(sample.get('speed_eligible') is True and sample.get('ok') is True and sample.get('stream_complete') is True and
        sample.get('finish_reason') in ('stop','length') and sample.get('reused_tokens')==0 and
        type(sample.get('read_tokens')) is int and sample['read_tokens']>=32768 and
        sample.get('prompt_tokens')==sample['read_tokens'] and type(sample.get('generated_tokens')) is int and
        1024<=sample['generated_tokens']<=16384 and sample.get('usage_generated_tokens')==sample['generated_tokens'] and
        sample.get('generated_tokens_source')=='usage' and sample.get('output_chars',0)>=512 and
        sample.get('reasoning_chars',0)>0 and sample.get('completion_includes_thinking') is True and
        sample.get('decode_tps_source')=='engine' and type(offered) is int and type(accepted) is int and 0<=accepted<=offered and
        all(type(sample.get(k)) in (int,float) and math.isfinite(sample[k]) and sample[k]>0 for k in ('decode_ms','decode_tps','prompt_ms','total_s')))
    return bool(common and (offered==accepted==0 and sample.get('acceptance_rate') is None if cap else gen.sample_eligible(sample)))


def request_routes(text,samples,cap):
    headers=list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$',text,re.M))
    marks=re.findall(r'^strata resident MTP control: mtp_max_t=(\d+) cumulative_draft_rounds=(\d+)$',text,re.M)
    require(len(headers)==len(marks)==len(samples),'missing native request/MTP footer')
    previous=rounds=0;out=[]
    for h,mark,s in zip(headers,marks,samples):
        require(tuple(map(int,h.groups()))==tuple(s[k] for k in ('prompt_tokens','reused_tokens','read_tokens','generated_tokens')) and s['reused_tokens']==0 and int(mark[0])==cap,'request extent/cap differs')
        segment=text[previous:h.start()];cells=s['read_tokens']-1
        chunks=[8192]*(cells//8192)+([cells%8192]if cells%8192 else[])
        draft=gen.draft_rows(segment)
        require(draft==[dict(path='token'if n<64 else'batched',mode=2,cells=n)for n in chunks],'prompt draft chunks/modes differ')
        windows=[tuple(map(int,w))for w in re.findall(r'^strata trace: window (\d+) (\d+)$',segment,re.M)]
        require(windows and windows[0]==(cells,1) and all(t==(1 if cap else 4)for _,t in windows[1:]) and
            all(1<=b[0]-a[0]<=a[1]for a,b in zip(windows,windows[1:])) and
            sum(t-1 for _,t in windows)==s['drafts_offered'],'target windows/offers differ')
        before_tail=sum(b[0]-a[0]-1 for a,b in zip(windows,windows[1:]))
        require(0<=s['drafts_accepted']-before_tail<windows[-1][1] and
            0<=len(windows)+s['drafts_accepted']-s['generated_tokens']<windows[-1][1],'accepted/output conservation differs')
        actual=int(mark[1]);require(actual-rounds==(0 if cap else len(windows)-1),'MTP round count differs')
        if cap:require(actual==s['drafts_offered']==s['drafts_accepted']==0,'disabled draft work remains')
        long=None
        if s['read_tokens']>=32768:
            route=re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) residual_host=0 residual_staging=0 ',segment,re.M)
            require(len(route)==1 and tuple(map(int,route[0]))==(cells,cells*20480,cells,cells*20480) and
                len(re.findall(r'^strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows='+str(min(cells,81920))+r' mixed_fp16_bytes=0$',segment,re.M))==1,'GPU BF16/mode10 route differs')
            cycle,pub=comparison.bank.admission.loan_cycle(segment)
            dense=comparison.bank.admission.records(segment,'strata prefill dense',3)
            require([(v['phase'],v['residency'],v['activations'])for _,v in dense]==[('retired','Full','0'),('bank','Bank','0'),('restored','Full','48')] and
                pub[0][0]<dense[0][0]<dense[1][0]<dense[2][0]<pub[1][0],'cache/dense restoration differs')
            long=dict(GPU_BF16_rows=cells,MoE_mode=10,restored_dense_layers=48,cache_payload_and_map_match=True,loan_generation=cycle['generation'])
        out.append(dict(target_windows=len(windows),target_rows=sum(t for _,t in windows),draft_rounds_delta=actual-rounds,
            cumulative_draft_rounds=actual,prompt_chunks=chunks,prompt_paths=[r['path']for r in draft],long_prefill=long))
        rounds=actual;previous=h.end()
    return out


def timing_eligible(sample):
    return all(type(sample.get(k)) in (int,float) and math.isfinite(sample[k]) and
        sample[k]>=clock.MIN_TIME_MS for k in ('prompt_ms','decode_ms'))


def official(root,cap):
    path,=(root/'bench').glob('*/results.json');result=json.loads(path.read_text());suite=result['results']['long_generation']
    rows=gen.collect_samples(suite)
    require(result['complete'] and not result['results']['failures'] and result['params']['long_generation_speed_only'] is True and
        result['params']['runs']==1 and result['params']['think'] is True and result['params']['long_generation_only'] is True and
        suite['speed_only'] is True and suite['workload_version']==4 and suite['thinking'] is True and
        suite['reasoning_budget_tokens']==0 and suite['max_tokens']==16384 and suite['selected_workloads']==list(WORKLOADS) and
        len(rows)==2 and all(sample_eligible(s,cap)for s in rows),'incomplete/ineligible official workload')
    require([(s['workload'],s['run'],s['order'])for s in rows]==[(w,0,i)for i,w in enumerate(WORKLOADS)],'official workload order differs')
    journal,jrows=gen.journal_evidence(root);require(journal['valid'] and journal['complete'],'incomplete official journal')
    require([{k:v for k,v in s.items()if k not in ('controller_eligible','reasoning_tokens')}for s in rows]==jrows,'suite/journal rows differ')
    path,=(root/'bench').glob('*/requests.jsonl');events=[json.loads(l)for l in path.read_text().splitlines()]
    warm=gen.warmup_reconciliation(events,WORKLOADS)
    from strata_bench import client_fingerprint
    identity=events[0];binding=result['comparison_identity'];work=dict(workload_id=WORKLOAD_ID,nonce=WORKLOAD_ID,nonce_policy='fixed-id')
    require(identity['source']==binding['client']==client_fingerprint() and identity['long_generation_speed_only'] is True and
        binding['long_generation_speed_only'] is True and identity['context']==binding['context']==131072 and
        identity['workload']==binding['workload']==result['params']['workload']==work and
        identity['thinking'] is binding['thinking'] is True,'official client/workload identity differs')
    for s in rows:
        require(timing_eligible(s),'request below declared10s timing floor')
    return rows,events,dict(warm,journal_sha256=digest(path))


def native_end(root,row,plan,leader):
    start=json.loads((root/'engine-start.json').read_text());end=json.loads((root/'engine-exit.json').read_text())
    require(set(start)=={'wrapper','binary','args','boot_id','native','native_validation_probe'} and
        start['binary']==row['binary'] and start['args']==['--serve',*plan['config']['args']] and
        start['boot_id']==row['short_proof']['runtime']['boot'] and
        start['native_validation_probe']==row['short_proof']['runtime']['preload'],'native start differs')
    for key in ('native','wrapper'):
        v=start[key];require(set(v)=={'pid','start_ticks','sid'} and v['sid']==leader,'native session ownership differs')
    collector=end['resource_collector'];require(collector['sid']==leader and collector['start_ticks']>=start['native']['start_ticks'] and
        len({collector['pid'],start['native']['pid'],start['wrapper']['pid']})==3 and
        end==dict(start,exit_code=0,resource_collector=collector,resource_collector_exit_code=0),'native/reader exit differs')
    return start


def run(cap):
    row=json.loads((ROOT/'record.json').read_text());current(row)
    require(cap==len(row['processes']) and cap in (0,1) and row['stage'] in ('host-checked','partial') and
        all(p['passed']for p in row['processes']) and not row['active'],'ordered fresh process only')
    plan=row['plans'][cap];root=Path(plan['root']);require(not (root/'owned').exists()and not(root/'launch.json').exists(),'no duplicate/retry')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',int(plan['command'][plan['command'].index('--port')+1])))
    before=copy.deepcopy(row);outcome=dict(cap=cap,passed=False);owner=None
    cursor_text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(l.split(': ',1)[1]for l in cursor_text.splitlines()if l.startswith('-- cursor: '))
    def persist():
        outcome['active']=bool(owner and owner.active is not None)
        write(root/'record.json',dict(outcome,commands=owner.commands if owner else []))
    def interact(io,leader):
        for i,name in enumerate(WORKLOADS):
            for stage in ('begin','end'):
                while True:
                    line=io.line(3300)
                    if line.startswith('STRATA_PROFILE_GATE '):break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==dict(stage=stage,phase='long-generation/'+name,run=0),'request gate differs')
                if stage=='begin':
                    claim=json.loads((root/'engine-start.json').read_text());v=claim['native'];actual=short.gate.native.supervision.identity(v['pid'])
                    require(actual and actual['sid']==leader['pid'] and actual['start_ticks']==v['start_ticks'] and
                        Path('/proc',str(v['pid']),'exe').resolve()==Path(row['binary']['path']).resolve(),'live native owner/executable differs')
                    env=dict(p.split(b'=',1)for p in Path('/proc',str(v['pid']),'environ').read_bytes().split(b'\0')if b'='in p)
                    expected=dict(row['environments'][str(cap)],LD_PRELOAD=row['short_proof']['runtime']['preload']['library']['path'],
                        STRATA_UR_PROBE_EXPECTED_LOADER=str(Path(row['short_proof']['runtime']['preload']['loader']['path']).resolve()))
                    require(all(env.get(k.encode())==v.encode()for k,v in expected.items()),'effective native environment differs')
                    path,=(root/'bench').glob('*/requests.jsonl');events=[json.loads(l)for l in path.read_text().splitlines()]
                    require(len(events)==10+2*i and events[-1]['kind']=='start','prior request journal extent differs')
                    gen.warmup_reconciliation(events[:-1],WORKLOADS[:i])
                    samples=[e['sample']for e in events[:-1]if e['kind']=='result']
                    request_routes((root/'engine.stderr').read_text(),samples,cap)
                    outcome.update(native_identity=v,effective_environment_verified=True);persist()
                io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(time.monotonic()<deadline,'benchmark teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        owner=short.gate.native.supervision.Owner(root/'owned',persist)
        entry,out,_=owner.run('clock',[row['clock']['probe']['path']],row['clock']['environment'],P,**row['host_budgets'])
        require(short.gate.native.supervision.closed(entry)and entry['exit_code']==0,'per-process clock failed')
        outcome['clock']=clock.evidence(out.read_text(),row['clock']['provider'])
        entry,_,_=owner.run('bench',plan['command'],row['environments'][str(cap)],P,interaction=interact,**row['budgets'])
        require(short.gate.native.supervision.closed(entry)and entry['exit_code']==0,'owned official benchmark failed')
        rows,events,journal=official(root,cap);claim=native_end(root,row,plan,entry['pid'])
        text=(root/'engine.stderr').read_text();all_samples=[e['sample']for e in events if e['kind']=='result']
        outcome['routes']=request_routes(text,all_samples,cap)
        outcome['activation']=short.gate.validation.preload_evidence(text,claim['native']['pid'])
        warnings=root/'validation.log';require(not warnings.exists()or warnings.stat().st_size==0,'native validation warnings')
        capture=[json.loads(l)for l in (root/'client-resources.jsonl').read_text().splitlines()]
        outcome['resources']=[resources.summarize_capture(capture,claim['native'],claim['boot_id'],s['client_interval'])for s in rows]
        if cap:require([s['request_sha256']for s in rows]==[s['request_sha256']for s in row['processes'][0]['requests']],'paired input bodies differ')
        current(before)
        if cap:short.model.model_pins(short.model.setup()['config'],row['model_assets'],full=True);outcome['final_full_model_hashes']=15
        outcome.update(passed=True,host_probe_accepted=True,requests=rows,journal=journal,native_exit_code=0,
            native_identity=claim['native'],evidence_sha256={p.name:digest(p)for p in (root/'engine.stderr',root/'client-resources.jsonl',root/'engine-start.json',root/'engine-exit.json')})
    except BaseException as error:outcome['error']=repr(error)
    finally:
        if owner is not None:short.gate.validation.finish(outcome,owner,cursor,row['short_proof']['runtime']['boot'])
        else:outcome.update(passed=False,ownership_closed=True)
        outcome['passed']=bool(outcome.get('passed')and outcome.get('driver_after')==row['short_proof']['runtime']['driver'])
        persist();row['processes'].append(outcome);row['active']=False;row['ownership_closed']=outcome['ownership_closed']
        row['stage']=('closed'if cap else'partial')if outcome['passed']else'failed'
        row['passed']=row['complete']=bool(cap and outcome['passed'])
        write(ROOT/'record.json',row);write(RESULT,compact(row))
    print(json.dumps({k:outcome.get(k)for k in ('cap','passed','ownership_closed','native_exit_code','error')}),flush=True)
    return 0 if outcome['passed']else 1


if __name__=='__main__':
    require(sys.argv[1:]in(['prepare'],['host'],['0'],['1']),'prepare|host|0|1')
    with (B/'owned-v0141-measurement.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);short.gate.native.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='host':sys.exit(host())
        else:sys.exit(run(int(sys.argv[1])))
