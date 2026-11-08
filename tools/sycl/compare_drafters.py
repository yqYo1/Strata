#!/usr/bin/env python3
"""Bounded, matched long-generation screens of the qualified named drafters.

Main-authored from this repository's benchmark/ownership interfaces; repository
LICENSE. Reuses the closed owned-buffer admission without rebuilding an engine.
The older drafter_generation.py is retained only for its still-live read-only
source audit; retire that superseded driver once the auditor formally returns.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import select
import socket
import sys
import time

sys.dont_write_bytecode = True
import drafter_benchmark as admission
import generation_engine as evidence

P, B = admission.P, admission.B
ROOT = B / 'drafter-generation-v2'
KEY = 'drafter_generation_v2'
RESULT = admission.RESULT
SELF = Path(__file__).resolve()
ORDER = ('mtp', 'dflash2')
WORKLOAD = 'regular-code'
MAXIMUM = 16384
require, digest, write = admission.require, admission.digest, admission.write


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('sources', 'admission')}
    out['source_inputs'] = dict(files=len(row['sources']), sha256=admission.fingerprint(row['sources']))
    out['admission_input'] = dict(path=str(admission.ROOT/'record.json'),
        canonical_sha256=admission.fingerprint(row['admission']), evidence='e73050a42f9f07ffcec33f3df2d14d473478fdd9')
    return out


def save(row):
    write(ROOT/'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def current(row):
    require(admission.heads.model.validation.committed(RESULT, (KEY,)) == compact(row),
        'commit the exact prospective generation plan/result')
    require(admission.sources(row['source_commit']) == row['sources'] and
        row['sources'].get(str(SELF.relative_to(P))) == digest(SELF), 'generation controller/source changed')
    require(previous_failure() == row['previous_failure'], 'original startup failure changed')
    require(json.loads((admission.ROOT/'record.json').read_text()) == row['admission'],
        'original closed admission changed')
    manifest = admission.current(row['admission'])
    require(row['admission']['passed'] and row['admission']['ownership_closed'], 'closed owned-buffer route required')
    for plan in row['plans']:
        require(all(digest(Path(plan['root'])/n) == h for n,h in plan['generated'].items()),
            'generation config/native plan changed')
        admission.python_runtime.require_environment(plan['environment'])
        require(admission.python_runtime.selection(plan['environment']) == row['python_selected'],
            'qualified Python selection changed')
    return manifest



def previous_failure():
    old = admission.heads.model.validation.committed(RESULT, ('drafter_generation_v1',))
    require(old['stage'] == 'failed' and old['passed'] is False and old['ownership_closed'] is False and
        old['failure_diagnosis']['GPU_startup_executed'] is True and
        old['failure_diagnosis']['long_generation_started'] is False and
        old['failure_diagnosis']['physical_session_closed'] is True, 'preserve failed V1 startup gate')
    return dict(evidence='c70ef71ae5c6ba2a681ab6a763f725aeec8eaf8a',
        source_commit=old['source_commit'], failure_diagnosis=old['failure_diagnosis'])


def identity_matches(actual, claim, session):
    """Match stable /proc identity; dynamic state/RSS are observations, not identity."""
    keys = ('pid', 'start_ticks', 'sid')
    return bool(type(claim) is dict and set(claim) == set(keys) and actual and
        all(type(claim[k]) is int and claim[k] > 0 and actual.get(k) == claim[k] for k in keys) and
        actual['sid'] == session)


def identity_host_gate():
    actual = admission.heads.ownership.identity(os.getpid())
    require(actual and len(actual) > 3, 'real /proc observation must include nonidentity fields')
    claim = {k: actual[k] for k in ('pid', 'start_ticks', 'sid')}
    require(identity_matches(actual, claim, claim['sid']), 'real live stable identity rejected')
    require(not identity_matches(None, claim, claim['sid']), 'missing process admitted')
    for key in claim:
        wrong = dict(claim); wrong[key] += 1
        require(not identity_matches(actual, wrong, claim['sid']), 'wrong stable field admitted: ' + key)
    require(not identity_matches(actual, claim, claim['sid'] + 1), 'foreign session admitted')
    require(not identity_matches(actual, dict(claim, rss_bytes=actual['rss_bytes']), claim['sid']),
        'unrecognized claim fields admitted')
    return dict(passed=True, cases=7, actual_status_fields=sorted(actual), identity_fields=sorted(claim),
        source='Current real /proc process,plus missing/foreign/stale identity negative cases;no GPU.')


def prepare():
    require(not ROOT.exists(), 'one prospective generation screen; no retry')
    prior = json.loads((admission.ROOT/'record.json').read_text())
    admission.current(prior); admission.heads.model.model_pins(prior['qualification'], full=True)
    revision = evidence.git('rev-parse','HEAD').decode().strip()
    pins = admission.sources(revision)
    require(pins.get(str(SELF.relative_to(P))) == digest(SELF), 'commit this controller before preparation')
    row = dict(stage='prepared', source_commit=revision, sources=pins, admission=prior,
        binary=prior['binary'], native_source_commit=prior['native_source_commit'],
        native_validation_probe=prior['native_validation_probe'], boot=prior['boot'],
        python_contract=prior['python_contract'], python_selected=prior['python_selected'],
        previous_failure=previous_failure(), plans=[], host=[], processes=[], passed=False, complete=False, ownership_closed=True,
        budgets=dict(wall=2400, cpu=28800, rss_cap=104<<30, text_cap=16<<20,
            file_cap=32<<20, total_cap=64<<20),
        plan=dict(question='Compare sustained named MTP and eager DFlash2 on the same owned-buffer engine/configuration.',
            order='One fresh process per arm, MTP then DFlash2; one regular-code v5 request each, official warmups unchanged. No speed-based stopping or within-process arm changes.',
            constants='Same f633 native;65536 INT8 context/resident,256 expert slots,5workers,8192chunk,spec4,suffix0,zero prefix/conversation caches/adaptation,throughput BF16/native16/GDN16,384packed bank,owned buffers,head retirement1. Only named drafting differs. Thinking/xhigh enabled,greedy,16384 total output allowance.',
            acceptance='Official bench.sh generation-only;>=32768 actual read/input tokens,zero reuse,>=1024 generated tokens including thought,>=512 answer characters,normal SSE and native clocks,>=256 valid offered drafts. Verify actual dynamic extent/route,providers,pre-main validation,normal native+benchmark exit0,reaped empty owner,unchanged source/assets,no fresh fault. Fail closed on first input/route/clock/validation/ownership/budget failure;retain failures.',
            decision='Parameter validation and stage progress remain charged. One request per arm is a route/sustained-output screen,not clean speed adoption,repeatability,strong/weak pair,task quality,physical262144/RESTORE or a physical resource ceiling. Preserve independent prefill/decode and all faster original phase references. Advance the measured faster candidate to repeated clean comparison only after eligibility;do not discard a faster arm merely for differing output.',
            retention='Compact individual samples,actual commands/route/counters/limits and hashes;retire normal raw logs,journals and JIT cache after verification. No new engine generation.'))
    ROOT.mkdir(mode=0o700)
    old = prior['plans'][0]
    for index,drafter in enumerate(ORDER):
        root=ROOT/drafter; root.mkdir(mode=0o700)
        for name in ('tmp','cache'): (root/name).mkdir(mode=0o700)
        env={k:v.replace(old['root'],str(root)) for k,v in old['environment'].items()}
        cfg=json.loads((Path(old['root'])/'config.json').read_text())
        cfg.update(drafting=drafter, log=str(root/'engine.stderr'))
        write(root/'config.json',cfg)
        write(root/'plan.json',dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        row['plans'].append(dict(drafter=drafter,root=str(root),environment=env,
            generated={n:digest(root/n) for n in ('config.json','plan.json')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--long-generation-only','--long-generation-workload',WORKLOAD,'--think','--runs','1',
                '--long-generation-max-tokens',str(MAXIMUM),'--profile-request-gates','--profile-startup-gate',
                '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600',
                '--out',str(root/'bench')]))
    save(row); print('Prepared matched sustained-output screen;commit before host/native execution.',flush=True)


def host():
    row=json.loads((ROOT/'record.json').read_text()); current(row)
    require(row['stage']=='prepared' and not row['host'], 'fresh host admission only')
    row['identity_host_gate']=identity_host_gate()
    owner=admission.heads.ownership.Owner(ROOT/'host-owned')
    try:
        for plan in row['plans']:
            entry,out,_=owner.run('args-'+plan['drafter'],[row['python_selected']['path'],'-B',
                str(admission.SELF),'route-config',str(Path(plan['root'])/'config.json')],
                plan['environment'],P,wall=60,cpu=60,rss_cap=1<<30,
                text_cap=1<<20,file_cap=2<<20,total_cap=8<<20)
            require(admission.heads.ownership.closed(entry) and entry['exit_code']==0,'actual named argument admission failed')
            actual=json.loads(out.read_text()); require(actual['backend'][0]=='sycl','hardware backend changed')
            plan['native_args']=actual['args']; row['host'].append(dict(drafter=plan['drafter'],passed=True))
        row['stage']='ready'
    except BaseException as error: row.update(stage='failed',error=repr(error))
    finally:
        row['ownership_closed']=owner.active is None and all(admission.heads.ownership.closed(v) for v in owner.commands)
        row['host_commands']=owner.commands;save(row)
    require(row['stage']=='ready' and row['ownership_closed'],row.get('error','host closure failed'))
    print('Both actual named frontend arguments admitted;commit before GPU.',flush=True)


def route(text,drafter,sample):
    def one(prefix):
        rows=[dict(re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)',s))
            for s in text.splitlines() if s.startswith(prefix)]
        require(len(rows)==1,'missing/extra native marker:'+prefix);return rows[0]
    n=sample['read_tokens']-1
    main=one('strata prefill: layer-major candidate, ')
    require(all(main.get(k)==v for k,v in dict(tokens=str(n),chunk='8192',residual_dtype='bf16',
        residual_gpu_rows=str(n),residual_gpu=str(n*20480),residual_host='0',residual_staging='0').items()),
        'actual native16/GPU extent or chunk differs')
    bank=one('strata MoE expert bank: ');finish=one('strata MoE expert bank complete: ')
    require(bank['requested_slots']==bank['effective_slots']==finish['total_slots']=='384' and
        bank['staged']=='1' and bank['general_map']=='private_scratch_never_published' and
        bank['device_bytes']==main['layer_cache'] and finish['final_drain']=='checked','staged bank route differs')
    window=one('strata MoE window configuration: ');split=one('strata MoE split admission: ')
    require(window['requested_mode']==window['effective_mode']=='10' and
        window['maximum_rows']==str(n) and split['active']==split['decode_changed']=='0','mode10/split route differs')
    head=one('strata prefill heads: target detached ')
    require(head['bytes']==head['D2H']==head['H2D-on-restore']=='521472000','target head retirement differs')
    require(one('strata prefill heads: draft_reader=')['draft_reader']==('mtp4' if drafter=='mtp' else 'absent-dflash2'),
        'optional MTP reader changed')
    callbacks=re.findall(r'^strata serve: DRAFT_PREFILL path=(\w+) mode=(-?\d+) cells=(\d+)$',text,re.M)
    active=re.findall(r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=(\d+) vram_bytes=(\d+)$',text,re.M)
    if drafter=='mtp':
        cells=[8192]*(n//8192)+([n%8192] if n%8192 else [])
        require(callbacks==[('batched','2',str(v)) for v in cells] and not active,'MTP callback extent differs')
    else:
        require(not callbacks and len(active)==1 and active[0][:2]==('2046','0'),'DFlash2 tail/activation differs')
    headers=re.findall(admission.decode_footer.HEADER,text,re.M)
    require(len(headers)==1 and tuple(map(int,headers[0]))==tuple(sample[k] for k in
        ('prompt_tokens','reused_tokens','read_tokens','generated_tokens','drafts_accepted','drafts_offered')),
        'native/client actual extent or generic draft counters differ')
    clocks=re.findall(r'^strata serve: prompt \d+ tokens = \d+ reused \+ \d+ read in (\d+) ms '
        r'\((\d+(?:\.\d+)?) tok/s\), \d+ generated in (\d+) ms \((\d+(?:\.\d+)?) tok/s\), drafts accepted ',text,re.M)
    require(len(clocks)==1,'native phase clock missing')
    pm,pp,dm,tg=map(float,clocks[0]);ap=sample['prompt_ms'];ad=sample['decode_ms']
    require(min(ap,ad)>.05 and abs(pm-ap)<=.550001 and abs(dm-ad)<=.550001 and
        abs(sample['prompt_tps']-1000*sample['read_tokens']/ap)<=.050001 and
        abs(pp-1000*sample['read_tokens']/ap)<=.050001+1000*sample['read_tokens']*.05/(ap*(ap-.05)) and
        abs(tg-sample['decode_tps'])<=.100001,'native/API clocks or rounded rates differ')
    return dict(rows=n,chunk=8192,residual_gpu_bytes=n*20480,prefill_slots=384,
        window_mode=10,split_active=False,target_detached_bytes=521472000,
        reader='mtp4' if drafter=='mtp' else 'absent-dflash2',MTP_callback_cells=n if callbacks else 0,
        DFlash2_tail_rows=2046 if active else 0,native_prompt_ms=pm,native_decode_ms=dm,
        physical_resource_saturation_qualified=False)


def run(index):
    row=json.loads((ROOT/'record.json').read_text());manifest=current(row)
    require(row['stage'] in ('ready','running') and len(row['processes'])==index<len(ORDER) and
        all(p['passed'] for p in row['processes']),'fresh frozen arm order required')
    plan=row['plans'][index];root=Path(plan['root']);env=plan['environment']
    require(not (root/'execution.json').exists(),'do not repeat an existing arm')
    kernel=admission.heads.model.mask.ring.device.kernel_log
    cursor=next(s.split(': ',1)[1] for s in kernel('-n','0','--show-cursor').splitlines() if s.startswith('-- cursor: '))
    outcome=dict(index=index,drafter=plan['drafter'],passed=False,complete=False,model_executed=False)
    owner=None;offsets={};initial=(time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
    def faults():
        log=kernel('--after-cursor',cursor,'-n','2048');require(len(log.splitlines())<2048,'journal overflow')
        return [s for s in log.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)]
    def persist(): write(root/'execution.json',dict(outcome,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    def live(entry):
        claim=json.loads((root/'engine-start.json').read_text());pid=claim['native']['pid']
        now=admission.heads.ownership.identity(pid)
        require(identity_matches(now,claim['native'],entry['pid']) and
            str(Path('/proc',str(pid),'exe').resolve())==row['binary']['path'] and
            claim['args']==['--serve',*plan['native_args']],'owned native arguments/identity differ')
        expected=admission.engine_once.native_environment(dict(native_validation_probe=row['native_validation_probe']),env)
        actual=dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
        require(all(actual.get(k.encode())==v.encode() for k,v in expected.items()) and
            set(actual)-{k.encode() for k in expected}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'native environment changed')
        import runtime_probe
        maps=runtime_probe.executable_mappings(Path('/proc',str(pid),'maps').read_text())
        require(maps<=set(row['admission']['qualification']['runtime_files'])|{row['binary']['path']} and
            {row['native_validation_probe']['loader']['path'],row['native_validation_probe']['library']['path'],
             env['ZE_ENABLE_ALT_DRIVERS']}<=maps,'unqualified provider mapping')
        admission.heads.model.validation.preload_evidence((root/'engine.stderr').read_text(errors='replace'),pid)
        admission.python_runtime.check_running(entry['pid'],P/'tools/strata_bench.py',manifest['selected'],env,
            {v['realpath'] for v in manifest['inventory']['files'].values()},plan['command'][2:])
        outcome.update(model_executed=True,native_identity=claim['native'],provider_mappings=len(maps))
    def gate(io,entry,stage,phase,run_id):
        while True:
            line=io.line(2100)
            if line.startswith('STRATA_PROFILE_GATE '):break
        require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==dict(stage=stage,phase=phase,run=run_id),
            'unexpected generation gate')
        delta=[a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),initial)]
        require(min(delta)>=0 and abs(delta[0]-delta[1])<=50000000,'clock/suspend discontinuity')
        live(entry)
        warnings=root/'validation.log';require(not warnings.exists() or warnings.stat().st_size==0,'validation warning/error')
    def interact(io,entry):
        gate(io,entry,'ready','startup',-1)
        log=(root/'engine.stderr').read_text(errors='replace')
        chunks=re.findall(r'^strata prefill: FP16 MoE compact .*?chunk=(\d+),',log,re.M)
        free=re.findall(r'^strata serve: (\d+) MiB of VRAM free with everything loaded$',log,re.M)
        require(chunks==['8192'] and len(free)==1 and int(free[0])>=(1536 if plan['drafter']=='dflash2' else 512),
            'startup chunk/headroom differs')
        outcome['startup_free_MiB']=int(free[0]);io.send(b'continue\n')
        for stage in ('begin','end'):
            gate(io,entry,stage,'long-generation/'+WORKLOAD,0)
            offsets[stage]=(root/'engine.stderr').stat().st_size
            print(plan['drafter']+' long-generation '+stage,flush=True);io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(time.monotonic()<deadline,'native/benchmark closure deadline')
            if select.select([io.proc.stdout],[],[],.02)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        require(not faults(),'fresh fault before launch')
        owner=admission.heads.ownership.Owner(root/'owned',persist)
        entry,_,_=owner.run('bench',plan['command'],env,P,interaction=interact,**row['budgets'])
        require(admission.heads.ownership.closed(entry) and entry['exit_code']==0,'benchmark/ownership failure')
        paths=list((root/'bench').glob('*/results.json'));require(len(paths)==1,'one benchmark result required')
        result=json.loads(paths[0].read_text());generation=result['results']['long_generation']
        samples=evidence.collect_samples(generation,MAXIMUM)
        outcome['samples']=samples
        require(result['complete'] and not result['results']['failures'] and generation['workload_version']==5 and
            generation['thinking'] is True and generation['reasoning_budget_tokens']==0 and generation['max_tokens']==MAXIMUM and
            generation['selected_workloads']==[WORKLOAD] and generation['diagnostic_subset'] is True and
            len(samples)==1 and samples[0]['controller_eligible'],'incomplete/ineligible sustained generation')
        journal,_=evidence.journal_evidence(root);require(journal['valid'] and journal['complete'],'incomplete request journal')
        entries=[json.loads(s) for s in (paths[0].parent/'requests.jsonl').read_text().splitlines()]
        outcome['warmups']=evidence.warmup_reconciliation(entries,(WORKLOAD,),MAXIMUM)
        end=evidence.native_end(root,row,dict(config=dict(args=plan['native_args'])),row['boot'],entry['pid'])
        raw=(root/'engine.stderr').read_bytes();text=raw.decode()
        require('prompt path borrows' not in text and text.count('strata serve: the prompt path allocates its own buffers ')==1 and
            not any(s in text for s in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called','teardown uncertain')),
            'unexpected borrowed route/runtime failure')
        outcome['route']=route(raw[offsets['begin']:offsets['end']].decode(),plan['drafter'],samples[0])
        if index==len(ORDER)-1:admission.heads.model.model_pins(row['admission']['qualification'],full=True)
        outcome.update(passed=True,complete=True,native_exit_code=end['exit_code'],speed_adoption_qualified=False)
    except BaseException as error:outcome['error']=repr(error)
    finally:
        outcome['ownership_closed']=not owner or owner.active is None and all(admission.heads.ownership.closed(c) for c in owner.commands)
        outcome['commands']=owner.commands if owner else []
        try:
            outcome['fresh_faults']=faults();require(not outcome['fresh_faults'],'fresh GPU fault');current(row)
        except BaseException as error:outcome.update(passed=False,finalization_error=repr(error))
        outcome['passed']=bool(outcome['passed'] and outcome['ownership_closed'])
        row['processes'].append(outcome);row.update(stage='running' if outcome['passed'] else 'failed',ownership_closed=outcome['ownership_closed'])
        if len(row['processes'])==len(ORDER) and outcome['passed']:row.update(stage='closed',passed=True,complete=True)
        save(row)
    require(outcome['passed'],outcome.get('error',outcome.get('finalization_error','generation failed')))
    print(plan['drafter']+' sustained generation closed;commit before next arm.',flush=True)


if __name__=='__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);admission.heads.guards.no_model_owners([])
        if sys.argv[1:]==['prepare']:prepare()
        elif sys.argv[1:]==['host']:host()
        elif len(sys.argv)==3 and sys.argv[1]=='run':run(int(sys.argv[2]))
        else:raise RuntimeError('use prepare|host|run INDEX')
