#!/usr/bin/env python3
"""Main-owned clean MTP-width comparison; repository LICENSE, no native rebuild.

Selected6e40 runs widths4/2. The qualified private resident control supplies
width1 without decode proposals, retaining MTP weights/KV/prompt precompute.
Native observers are off; source-fixed width plus actual footer work is checked.
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
import check_full_context_source as original
import check_mtp_resident_control as private
import full_context_model as model
import generation_engine as gen
import measurement_clock as clock
import ur_validation as validation

P, B = original.P, original.B
ROOT = B / 'mtp-width-clean-v1'
RESULT = gen.A / 'mtp-width-clean-current.json'
ORDER = (2, 4, 1, 4, 1, 2, 1, 2, 4)
WORKLOADS = ('regular-code', 'varied-prose')
WORKLOAD_ID = 'native16_mtp_width_clean_20261024'
require, digest, write = original.guards.require, original.build.digest, original.write
fingerprint = private.native.fingerprint
REFERENCE_NAMES = ('mtp2-route-current.json', 'mtp-resident-serve-current.json',
    'mtp-resident-check-current.json', 'phase-prefill-speed-current.json', 'client-resources-current.json')
FORBIDDEN = ('STRATA_TRACE', 'STRATA_DECODE_TIMING', 'STRATA_VERIFY_PROFILE',
    'STRATA_CPU_WORK_CENSUS', 'STRATA_IQ2_PAIR_AUDIT', 'STRATA_IQ_TIMING',
    'STRATA_STATE_HASH', 'STRATA_SNAPSHOT_VERIFY', 'STRATA_SYCL_CACHE_LOAN_AUDIT',
    'STRATA_SYCL_PREFILL_SEPARATE_LAYOUT', 'LD_PRELOAD', 'UR_LOG_TRACING')


def qualification():
    native, proof = original.baseline()
    control = private.current_build()
    fixed = validation.committed(RESULT.with_name('mtp2-route-current.json'))
    short = validation.committed(RESULT.with_name('mtp-resident-serve-current.json'))
    require(fixed['passed'] and fixed['complete'] and fixed['ownership_closed'] and not fixed['active'] and
        fixed['binary'] == native['binary'] and fixed['outcomes']['model']['final_full_model_hashes'] == 15 and
        fixed['outcomes']['model']['fresh_faults'] == [] and
        short['passed'] and short['complete'] and short['ownership_closed'] and not short['active'] and
        short['binary'] == control['assets']['strata'] and short['final_full_model_hashes'] == 15 and
        [r['cap'] for r in short['processes']] == [0, 1] and
        all(r['passed'] and r['exit_code'] == 0 and r['fresh_faults'] == [] for r in short['processes']),
        'closed exact-binary width2 and resident-control model gates required')
    return native, control, proof


def pins(revision):
    found = gen.benchmark_pins(revision)
    name = str(clock.SOURCE.relative_to(P)); raw = gen.git('show', revision+':'+name)
    require(clock.SOURCE.read_bytes() == raw, 'commit exact clock source')
    found[name] = hashlib.sha256(raw).hexdigest()
    return found


def seal(row):
    return fingerprint({k: row[k] for k in ('environments', 'model_assets')})


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('environments', 'model_assets', 'commands')}
    out['clock'] = clock.compact(row['clock'])
    return out


def prepare():
    require(not ROOT.exists() and not RESULT.exists(), 'fresh width plan only; no duplicate/retry')
    native, control, proof = qualification(); base = model.setup()
    model.model_pins(base['config'], base['models'])
    speed = validation.committed(RESULT.with_name('phase-prefill-speed-current.json'))
    require(speed['selection']['selected_source'] == native['source_commit'] and
        not speed['selection']['candidate_adopted'], 'original prefill remains selected')
    preload = validation.preload_qualification()
    gen.runtime_assets_unchanged(validation.committed(RESULT.with_name('client-resources-current.json')))
    revision = gen.git('rev-parse', 'HEAD').decode().strip(); inputs = pins(revision)
    ROOT.mkdir(mode=0o700); (ROOT/'host').mkdir()
    host = clock.prepare(native['binary'], native['compiler'], native['environment'], ROOT/'host',
        speed['host_clock']['provider'])
    plans = []; environments = {}
    for index, width in enumerate(ORDER):
        root = ROOT/str(index); root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root/name).mkdir()
        binary = control['assets']['strata'] if width == 1 else native['binary']
        config = copy.deepcopy(base['config']); args = config['args']
        require(args[args.index('--spec')+1] == '4' and '--mtp-max-t' not in args, 'fixed allocation required')
        args[args.index('--expert-profile')+1] = str(root/'expert-profile.bin')
        args += ['--mtp-max-t', str(0 if width == 4 else width)]
        config.update(log=str(root/'engine.stderr'), env={})
        shutil.copyfile(P/'data/expert-profile.bin', root/'expert-profile.bin')
        env = dict(speed['environment_template'], TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'),
            STRATA_MEASUREMENT_ROOT=str(root),
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
        require(not any(k in env for k in FORBIDDEN) and env['STRATA_SYCL_MTP_OBSERVER'] == '0',
            'clean native observers/prefill selection differ')
        for name in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission',
                     'STRATA_SYCL_PREFILL_SAME_QUEUE','SYCL_CACHE_PERSISTENT'):
            require(env[name] == native['environment'][name], 'qualified runtime safety changed: '+name)
        write(root/'config.json', config)
        write(root/'plan.json', dict(binary=binary, native_validation_probe=preload))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
        plans.append(dict(index=index, width=width, block=index//3, root=str(root), binary=binary, config=config,
            generated_inputs={str(root/n):digest(root/n) for n in ('config.json','plan.json','expert-profile.bin')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--long-generation-only','--long-generation-speed-only','--think','--runs','1',
                '--profile-request-gates','--workload-id',WORKLOAD_ID,'--fail-fast-engine-error',
                '--ready-timeout','600','--out',str(root/'bench')]))
        environments[str(index)] = env
    row = dict(stage='host-prepared', source_commit=revision, benchmark_files=len(inputs),
        benchmark_fingerprint=fingerprint(inputs), original=proof, native_validation_probe=preload,
        model_assets=base['models'], model_reference=base['reference'], environments=environments,
        references={str(RESULT.with_name(n)):digest(RESULT.with_name(n)) for n in REFERENCE_NAMES},
        clock=host, plans=plans, boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        driver=original.guards.driver_guard(),
        mechanism='Previously observed draft wrapper costs9.50/15.21ms per emitted code/prose token,with CPU-pool and GPU-reach waits also dominant. Width2 removes chain proposals and reduces target rows versus4;it can lose accepted tokens. Source/lifetime qualified;test actual end-to-end tradeoff. Host scopes do not prove CPU or GPU arithmetic saturation;physical PCIe/VRAM ceilings remain unknown.',
        criteria='Nine fresh processes in balanced blocks2,4,1 /4,1,2 /1,2,4;three independent samples per width/workload. Four standard warmups then official v4 regular-code/varied-prose thinking-on with>=32768 fresh input,1024..16384 generated,>=512 answer chars,reasoning present,complete valid SSE/journal and native exit0. Same original prefill/chunk8192/resident131072/1256slots/caches0. Width4/2 use exact original6e40;1 uses qualified private control retaining MTP weights/KV/prompt precompute. Native observers absent/disabled;only qualified parameter-validation/preexec safety remains. Actual native args/environment and aggregate offered/accepted/emitted work bind the source-fixed widths;no per-window trace or hardware attribution claim. Clock screen before every process;prompt/decode>=10s,endpoint+rounding<=0.01%. Full source/assets/runtime/owner/boot/no-fault gates each process;all15 hashes after the cohort. Stop first execution/eligibility/fault failure,no replacement or changed criteria;slower samples do not stop collection. No adoption before all separate gates.',
        selection_rule='For each candidate1/2 and each workload,all3 block-paired decode ratios>=1.0 and geometric mean>1.02 to advance that workload. Independently require all3 paired prefill ratios>=1.0;any measured impact fails separation/preservation and prevents adoption. Compare native counts/exact unrounded times;keep every sample/spread and original failed gate. Record acceptance per width;MTP4 code>=0.80/prose<=0.40 alone validates intended strong/weak reference classification,not labels imposed on narrower widths.70tok/s target requires repeated actual model throughput,not a component gain. Long/agent-session,task,physical262144/RESTORE and ordinary teardown remain separate pending adoption gates.',
        budgets=dict(wall=3900,cpu=46800,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20),
        host_budgets=dict(wall=120,cpu=120,rss_cap=4<<30,text_cap=2<<20,file_cap=4<<20,total_cap=16<<20),
        retention='Keep individual samples,clock screens,source/configuration,comparison/fault decisions. Replay then retire successful logs/JIT/journals/raw responses and one-shot launcher. Original13/private13 assets serve this imminent comparison only;retire private after closure and exact original-input recovery,or before a new native generation.',
        processes=[], active=False, ownership_closed=True, passed=False, complete=False,
        production_selected=False, model_adoption_qualified=False, clean_comparison_complete=False)
    row['private_inputs_sha256'] = seal(row); write(ROOT/'record.json',row); write(RESULT,compact(row))
    print('Prepared balanced nine-process clean width comparison;commit before host/build/model work.',flush=True)


def current(row):
    require(RESULT.read_bytes() == gen.git('show','HEAD:'+str(RESULT.relative_to(P))) and
        json.loads(RESULT.read_text()) == compact(row) and seal(row) == row['private_inputs_sha256'],
        'committed plan/private metadata differ')
    found = pins(row['source_commit'])
    require(len(found) == row['benchmark_files'] and fingerprint(found) == row['benchmark_fingerprint'], 'frozen inputs differ')
    native, control, proof = qualification(); require(proof == row['original'], 'original source proof differs')
    for plan in row['plans']:
        require(plan['binary'] == (control['assets']['strata'] if plan['width']==1 else native['binary']), 'engine differs')
        for path,sha in plan['generated_inputs'].items(): require(digest(path)==sha,'generated input changed: '+path)
    for path,sha in row['references'].items(): require(digest(path)==sha,'qualification reference changed: '+path)
    require(model.setup()['reference']==row['model_reference'] and
        validation.preload_qualification()==row['native_validation_probe'], 'model/preload reference differs')
    model.model_pins(model.setup()['config'],row['model_assets']); clock.check(row['clock'])
    if 'probe' in row['clock']: require(digest(row['clock']['probe']['path'])==row['clock']['probe']['sha256'],'clock probe changed')
    gen.runtime_assets_unchanged(validation.committed(RESULT.with_name('client-resources-current.json')))
    require(original.guards.driver_guard()==row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'],'platform differs')


def sample_eligible(sample,width):
    if width != 1: return gen.sample_eligible(sample)
    # Check the explicit speed contract without relabeling agreement/counters.
    return bool(sample.get('speed_eligible') is True and sample.get('ok') is True and
        sample.get('stream_complete') is True and sample.get('finish_reason') in ('stop','length') and
        sample.get('reused_tokens') == 0 and type(sample.get('read_tokens')) is int and
        sample['read_tokens'] >= 32768 and sample.get('prompt_tokens') == sample['read_tokens'] and
        type(sample.get('generated_tokens')) is int and 1024 <= sample['generated_tokens'] <= 16384 and
        sample.get('usage_generated_tokens') == sample['generated_tokens'] and
        sample.get('generated_tokens_source') == 'usage' and sample.get('output_chars',0) >= 512 and
        sample.get('reasoning_chars',0) > 0 and sample.get('completion_includes_thinking') is True and
        sample.get('decode_tps_source') == 'engine' and
        type(sample.get('drafts_offered')) is int and type(sample.get('drafts_accepted')) is int and
        sample['drafts_offered'] == sample['drafts_accepted'] == 0 and sample.get('acceptance_rate') is None and
        all(type(sample.get(k)) in (int,float) and math.isfinite(sample[k]) and sample[k] > 0
            for k in ('decode_ms','decode_tps','prompt_ms','total_s')))


def native_work(text,samples,width):
    require(width in (1,2,4),'unknown fixed width')
    pattern = (r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, '
        r'(\d+) generated[^\n]*, drafts accepted (\d+) of (\d+),[^\n]*$')
    headers = re.findall(pattern,text,re.M); require(len(headers)==len(samples),'native request footer count differs')
    require(not any(mark in text for mark in ('strata trace:', 'strata decode timing:', 'strata MTP_OBSERVER ',
        'strata CPU_WORK', 'strata IQ2_GU_AUDIT')), 'native observer was enabled')
    marks = re.findall(r'^strata resident MTP control: mtp_max_t=(\d+) cumulative_draft_rounds=(\d+)$',text,re.M)
    require(marks==[('1','0')]*len(samples) if width==1 else not marks,'resident control route differs')
    out=[]
    for header,sample in zip(headers,samples):
        n,reused,read,emitted,accepted,offered=map(int,header)
        require((n,reused,read,emitted,accepted,offered)==tuple(sample[k]for k in
            ('prompt_tokens','reused_tokens','read_tokens','generated_tokens','drafts_accepted','drafts_offered')) and
            reused==0 and read==n and n>0 and emitted>0 and 0<=accepted<=offered,'native/client work differs')
        if width==1: require(offered==accepted==0,'draft work remains'); windows=emitted
        else:
            require(offered%(width-1)==0,'offered rows do not fit fixed width')
            windows=1+offered//(width-1)
            require(0<=windows+accepted-emitted<width,'output/window conservation differs')
        out.append(dict(source_fixed_width=width,target_windows_inferred_from_offered=windows,
            target_rows_inferred=1+(windows-1)*width,emitted=emitted,accepted=accepted,offered=offered))
    return out


def comparison(processes):
    require(len(processes)==len(ORDER) and [p['width']for p in processes]==list(ORDER) and
        all(p['passed']for p in processes),'complete balanced comparison required')
    out={}
    for width in (1,2):
        out[str(width)]={}
        for workload_index,name in enumerate(WORKLOADS):
            paired=[]
            for block in range(3):
                group={p['width']:p['requests'][workload_index]for p in processes[3*block:3*block+3]}
                ref,candidate=group[4],group[width]
                paired.append(dict(block=block,decode_ratio=(candidate['generated_tokens']/candidate['decode_ms'])/
                    (ref['generated_tokens']/ref['decode_ms']),prefill_ratio=(candidate['read_tokens']/candidate['prompt_ms'])/
                    (ref['read_tokens']/ref['prompt_ms'])))
            ratios=[p['decode_ratio']for p in paired];prefill=[p['prefill_ratio']for p in paired]
            geometric_mean=math.exp(sum(map(math.log,ratios))/len(ratios))
            out[str(width)][name]=dict(pairs=paired,decode_geomean_ratio=geometric_mean,
                decode_gain_gate=all(r>=1 for r in ratios) and geometric_mean>1.02,
                prefill_preservation_gate=all(r>=1 for r in prefill),production_adopted=False)
    refs=[p['requests']for p in processes if p['width']==4]
    return dict(candidates=out,reference_classification=all(r[0]['acceptance_rate']>=.8 and
        r[1]['acceptance_rate']<=.4 for r in refs),production_adopted=False,
        limits='Advancement gates only;long/agent-session,task/full-context/RESTORE and teardown remain pending.')


def official(root,width):
    path,=(root/'bench').glob('*/results.json'); result=json.loads(path.read_text())
    suite=result['results']['long_generation']; rows=gen.collect_samples(suite)
    require(result['complete'] and not result['results']['failures'] and
        result['params']['long_generation_speed_only'] is True and result['params']['runs']==1 and
        result['params']['think'] is True and result['params']['long_generation_only'] is True and
        suite['speed_only'] is True and suite['workload_version']==4 and suite['thinking'] is True and
        suite['reasoning_budget_tokens']==0 and suite['max_tokens']==16384 and
        suite['selected_workloads']==list(WORKLOADS) and len(rows)==2 and
        all(sample_eligible(s,width) and all(type(s.get(k))in(int,float) and math.isfinite(s[k]) and
            s[k]>=clock.MIN_TIME_MS for k in ('prompt_ms','decode_ms')) for s in rows), 'official work/clock eligibility failed')
    require([(s['workload'],s['run'],s['order'])for s in rows]==[(w,0,i)for i,w in enumerate(WORKLOADS)],'workload order differs')
    journal,jrows=gen.journal_evidence(root)
    require(journal['valid'] and journal['complete'] and
        [{k:v for k,v in s.items()if k not in ('controller_eligible','reasoning_tokens')}for s in rows]==jrows,'journal work differs')
    path,=(root/'bench').glob('*/requests.jsonl'); events=[json.loads(l)for l in path.read_text().splitlines()]
    warm=gen.warmup_reconciliation(events,WORKLOADS)
    from strata_bench import client_fingerprint
    identity=events[0]; binding=result['comparison_identity']
    work=dict(workload_id=WORKLOAD_ID,nonce=WORKLOAD_ID,nonce_policy='fixed-id')
    require(identity['source']==binding['client']==client_fingerprint() and
        identity['long_generation_speed_only'] is binding['long_generation_speed_only'] is True and
        identity['context']==binding['context']==131072 and
        identity['workload']==binding['workload']==result['params']['workload']==work and
        identity['thinking'] is binding['thinking'] is True,'official workload identity differs')
    return rows,events,dict(warm,journal_sha256=digest(path))


def retain_client_result(root,outcome):
    """Save returned samples before exit/eligibility gates, without qualifying them."""
    paths=list((root/'bench').glob('*/results.json'))
    if len(paths)!=1:return
    result=json.loads(paths[0].read_text())
    outcome['requests']=gen.collect_samples(result.get('results',{}).get('long_generation',{}))
    outcome['benchmark_complete']=result.get('complete')
    outcome['benchmark_failures']=result.get('results',{}).get('failures',[])


def host(row):
    require(row['stage']=='host-prepared' and not (ROOT/'host/owned').exists(),'fresh host phase only')
    owner=None
    def persist(): write(ROOT/'host/record.json',dict(commands=owner.commands if owner else []))
    try:
        owner=original.supervision.Owner(ROOT/'host/owned',persist)
        for label,args in [('width-tests',['/usr/bin/python3','-B','tools/sycl/test_mtp_width_bench.py']),
                ('clock-tests',['/usr/bin/python3','-B','tools/sycl/test_measurement_clock.py']),
                ('compile',row['clock']['command'])]:
            entry,_,_=owner.run(label,args,row['clock']['environment'],P,**row['host_budgets'])
            require(original.supervision.closed(entry) and entry['exit_code']==0,'host phase failed: '+label)
        probe=ROOT/'host/clock_probe';row['clock']['probe']=dict(path=str(probe),sha256=digest(probe),bytes=probe.stat().st_size)
        entry,out,_=owner.run('clock',[str(probe)],row['clock']['environment'],P,**row['host_budgets'])
        require(original.supervision.closed(entry) and entry['exit_code']==0,'clock screen failed')
        row['clock']['initial_screen']=clock.evidence(out.read_text(),row['clock']['provider']);row['stage']='host-checked'
    except BaseException as error: row.update(stage='host-failed',error=repr(error))
    finally:
        row['ownership_closed']=owner is None or owner.active is None and all(original.supervision.closed(c)for c in owner.commands)
        write(ROOT/'record.json',row);write(RESULT,compact(row))
    print(json.dumps(dict(stage=row['stage'],ownership_closed=row['ownership_closed'])),flush=True)
    return int(row['stage']!='host-checked')


def run(row,index):
    require(index==len(row['processes']) and index<len(ORDER) and not row['active'] and
        row['stage'] in ('host-checked','partial') and all(p['passed']for p in row['processes']),'ordered fresh process only')
    plan=row['plans'][index]; root=Path(plan['root']); require(not (root/'owned').exists(),'no retry')
    with socket.socket() as sock: sock.bind(('127.0.0.1',int(plan['command'][plan['command'].index('--port')+1])))
    before=copy.deepcopy(row);outcome=dict(index=index,width=plan['width'],passed=False);owner=None
    raw=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(l.split(': ',1)[1]for l in raw.splitlines()if l.startswith('-- cursor: '))
    def persist():
        outcome['active']=bool(owner and owner.active is not None)
        write(root/'record.json',dict(outcome,commands=owner.commands if owner else []))
    def interact(io,entry):
        for name in WORKLOADS:
            for stage in ('begin','end'):
                while True:
                    line=io.line(3600)
                    if line.startswith('STRATA_PROFILE_GATE '): break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==dict(stage=stage,phase='long-generation/'+name,run=0),'client request gate differs')
                if stage=='begin':
                    claim=json.loads((root/'engine-start.json').read_text());v=claim['native'];actual=original.supervision.identity(v['pid'])
                    require(actual and actual['sid']==entry['pid'] and actual['start_ticks']==v['start_ticks'] and
                        Path('/proc',str(v['pid']),'exe').resolve()==Path(plan['binary']['path']).resolve(),'live native owner differs')
                    env=dict(v.split(b'=',1)for v in Path('/proc',str(v['pid']),'environ').read_bytes().split(b'\0')if b'='in v)
                    expected=dict(row['environments'][str(index)],LD_PRELOAD=row['native_validation_probe']['library']['path'],
                        STRATA_UR_PROBE_EXPECTED_LOADER=str(Path(row['native_validation_probe']['loader']['path']).resolve()))
                    require(all(env.get(k.encode())==v.encode()for k,v in expected.items()) and
                        not any(k.encode()in env for k in FORBIDDEN if k!='LD_PRELOAD'),'effective native environment differs')
                    outcome['effective_native_environment_verified']=True;persist()
                io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(time.monotonic()<deadline,'benchmark teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        owner=original.supervision.Owner(root/'owned',persist)
        entry,out,_=owner.run('clock',[row['clock']['probe']['path']],row['clock']['environment'],P,**row['host_budgets'])
        require(original.supervision.closed(entry) and entry['exit_code']==0,'per-process clock screen failed')
        outcome['clock']=clock.evidence(out.read_text(),row['clock']['provider'])
        entry,_,_=owner.run('bench',plan['command'],row['environments'][str(index)],P,interaction=interact,**row['budgets'])
        outcome['benchmark_exit_code']=entry['exit_code']
        retain_client_result(root,outcome)
        end=gen.native_end(root,dict(binary=plan['binary'],native_validation_probe=row['native_validation_probe']),plan,row['boot'],entry['pid'])
        outcome['native_exit_code']=end['exit_code']
        require(original.supervision.closed(entry) and entry['exit_code']==0,'owned official benchmark failed')
        rows,events,journal=official(root,plan['width'])
        text=(root/'engine.stderr').read_text()
        outcome['work']=native_work(text,[e['sample']for e in events if e['kind']=='result'],plan['width'])
        outcome['activation']=validation.preload_evidence(text,end['native']['pid'])
        warning=root/'validation.log';require(not warning.exists()or warning.stat().st_size==0,'validation warning')
        if row['processes']: require([s['request_sha256']for s in rows]==[s['request_sha256']for s in row['processes'][0]['requests']],'matched input bodies differ')
        current(before)
        if index==len(ORDER)-1:model.model_pins(model.setup()['config'],row['model_assets'],full=True);outcome['final_full_model_hashes']=15
        outcome.update(passed=True,host_probe_accepted=True,requests=rows,journal=journal,native_exit_code=0,
            evidence_sha256={n:digest(root/n)for n in ('engine.stderr','engine-start.json','engine-exit.json')})
    except BaseException as error:outcome['error']=repr(error)
    finally:
        if owner is not None:validation.finish(outcome,owner,cursor,row['boot'])
        else:outcome['ownership_closed']=True
        outcome['passed']=bool(outcome.get('passed')and outcome.get('driver_after')==row['driver']);persist()
        row['processes'].append(outcome);row['active']=False;row['ownership_closed']=outcome['ownership_closed']
        row['stage']=('closed'if index==len(ORDER)-1 else'partial')if outcome['passed']else'failed'
        row['passed']=row['complete']=row['clean_comparison_complete']=bool(index==len(ORDER)-1 and outcome['passed'])
        if row['clean_comparison_complete']:row['comparison']=comparison(row['processes'])
        write(ROOT/'record.json',row);write(RESULT,compact(row))
    print(json.dumps({k:outcome.get(k)for k in ('index','width','passed','ownership_closed','native_exit_code','error')}),flush=True)
    return int(not outcome['passed'])


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1]in('prepare','host',*[str(i)for i in range(len(ORDER))]),'prepare|host|0..8')
    with (B/'owned-v0141-measurement.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);original.guards.no_model_owners([])
        if sys.argv[1]=='prepare':
            try:prepare()
            except BaseException as error:
                write(RESULT,dict(stage='preparation-failed',source_commit=gen.git('rev-parse','HEAD').decode().strip(),
                    error=repr(error),partial_root=str(ROOT),active=False,ownership_closed=True,GPU_executed=False,
                    next_action='Preserve this failure;inspect partial inputs and commit corrected source and a fresh plan/root before retry.'))
                raise
        else:
            row=json.loads((ROOT/'record.json').read_text());current(row)
            sys.exit(host(row)if sys.argv[1]=='host'else run(row,int(sys.argv[1])))
