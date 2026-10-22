#!/usr/bin/env python3
"""Main-owned same-native window-ranked cache pilot; never automatic adoption."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import cpu_work_census as census
import generation_engine as gen
import resource_engine as resources
import routing_workmix as routing

assets,restored,comparison = resources.modules()
guards,supervision = resources.guards,resources.supervision
P,B,A = gen.P,gen.B,gen.A
ROOT,RESULT = B/'cache-window-rank-v1',A/'cache-window-rank-current.json'
NATIVE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
REFERENCE = dict(commit='5c99e051d45a1e66ba916375721b92d64c70b378',
    path='bench/results/2026-10-15-pr6-recovery-audit/routing-workmix-current.json',
    sha256='08111adcacce24a5b7b0d29efea7f1ffb7ecf26cb6f8ddc58cda62d02aae775d')
WORKLOADS = ('regular-code','varied-prose')


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def prospective():
    return dict(order=['original-profile','window-ranked-profile'],
        question='Does cache selection by observed unique verifier-window frequency reduce actual CPU logical weight work and token latency in both strong/weak MTP workloads at unchanged prefill/decode allocation?',
        mechanism='Reduce unique CPU expert jobs and their logical blob visits by selecting experts with frequent verifier-window reuse. The fixed-route20.517% estimate is training evidence,not measured CPU service or physical RAM traffic. Existing pool wall is on the verifier critical path but does not establish hardware saturation. More GPU expert work may offset the host saving;measure the complete token path. No decode expert-weight miss H2D is requested under pcie0.',
        native_audit='Exact original6e40 engine and12 same-build fixtures. Profile generation preserves all24576 unique layer/expert pairs and the complete reverse map;only stable layer-local expert rank changes,all original layer labels remain. Both native per-layer admissions keep26 experts/layer,last34,1256 total,2575206400 allocated bytes and245657600 protected-prefix bytes. Packed slot offsets depend on layer blob size,not expert ID. Native bounds remain48 layers/512 experts/top10/T<=4,<=40 distinct groups,one1256-slot owner and immutable restored table. Cache fill,verifier slot-offset binding and scratch lease retirement/refill/publication use existing qualified code. No new kernel,collective,queue,copy policy,graph code or environment mutation. Check the actual map and payload before long dispatch.',
        constants='Same model,131072 residentINT8 KV,8192 prefill,spec4,5workers,pcie0,adapt0,caches0,IQ2 decode1 and unchanged prefill reference. Both profiles use the same per-layer layout/bytes;original selected defaults stay unchanged. Existing diagnostic/loan/IQ2/census/timing observers,parameter validation and warning logging;MTP graph observer0,no passive collector,no route dump.',
        execution='Two independent processes in frozen baseline/candidate order. Each official bench.sh --long-generation-only --think --runs1 performs4 warmups then regular-code and varied-prose,each>=32768 fresh input/>=1024 generated and substantive answer. Before either long request bind native PID/start/SID/executable/environment,complete prior journal,actual cache map and all previous loan/payload/refill cycles. Strong observed MTP>=.60 and weak<=.40 required.',
        admission='Commit exact controller,profile generator,host tests and generated plans before launch. Recover9853 original native inputs/recipe and all13 artifact bytes,18-group qualification and immutable reference/matrix. Reproduce the archived fixed-route analysis exactly. Full15-model hashes before cohort and after each model process;stat identities elsewhere. Same-native tiny traced bank continuity is archived71d0b6e63;the4 native warmups exercise the new profile before long execution. Stop on any mismatch,fault,budget or missing closure;never retry a used root.',
        decision='Only propose a separate repeated clean comparison if both processes complete/close normally without a fresh fault,matched request hashes/read extents and actual restored slot/byte maps,GPU BF16/mode10/private bank/8192 chunk/48 dense-layer restoration pass. For EACH workload require CPU logical job-blob bytes per actual verifier row<=.95 baseline,increased GPU routed-entry fraction,and exact decode_ms/generated_tokens<=baseline. Generated text need not match;changed work/MTP is recorded. Otherwise reject this pilot;no posthoc threshold change. A pass is not model adoption or prefill speed preservation.',
        retention='Keep4 individual long-request samples,compact work/NT,actual loans/restoration,source/environment/fault/closure and decision. Successful full logs/JIT retire after evidence replay. Source393236B count matrix and generated profiles stay only through this pilot/profile decision;retire on rejection or when the next named consumer ends. No raw routing or tensors collected.',
        remaining='Clean independent repetitions,matched long-prefill preservation,heldout/agent/task quality and full physical262144/session/RESTORE remain open. Training-route prediction and this diagnostic pair do not establish physical traffic,hardware saturation,1000-prefill/70-decode target or adoption.')


def reference():
    raw=gen.git('show',REFERENCE['commit']+':'+REFERENCE['path'])
    guards.require(hashlib.sha256(raw).hexdigest()==REFERENCE['sha256'],'archived routing reference changed')
    value=json.loads(raw)
    guards.require(value['schema']=='strata-framed-routing-workmix-closed-v1' and value['passed'] and
        value['complete'] and value['ownership_closed'] and not value['active'] and
        value['build_source_commit']==NATIVE and value['artifact_retirement']['complete'] and
        value['offline_analysis']['decision']['can_propose_new_model_plan'] and
        not value['offline_analysis']['decision']['adoption_qualified'],'qualified fixed-route prerequisite required')
    return value


def generated_profile(ref):
    profile,manifest = [Path(ref['rank_inputs'][k]['path']) for k in ('profile','manifest')]
    expected={k:v for k,v in ref['offline_analysis'].items() if k not in ('source_commit','host_tests_passed')}
    guards.require(routing.analyze(ref,profile,manifest)==expected,'original ranking analysis changed')
    raw=Path(ref['per_expert_counts']['path']).read_bytes()
    return routing.window_profile(routing.read_counts(raw),profile)


def pins(revision):
    values=gen.benchmark_pins(revision)
    for name in ('data/expert-profile.bin','tests/core/test_routing_workmix.py','tests/core/test_cache_window_rank_engine.py'):
        raw=gen.git('show',revision+':'+name)
        guards.require((P/name).read_bytes()==raw,'uncommitted window-rank input: '+name)
        values[name]=hashlib.sha256(raw).hexdigest()
    return values


def config(setup,root):
    value=restored.recipe.config(setup,root);args=value['args']
    for name,expected in (('--expert-cache','1256'),('--adapt-swaps','0'),('--pcie-frac','0'),('--spec','4')):
        guards.require(args.count(name)==1 and args[args.index(name)+1]==expected,'fixed cache/window config differs')
    guards.require('--expert-cache-per-layer' in args and '--no-spec-split' in args and
        '--dump-routing' not in args,'per-layer/unsplit/no-dump policy required')
    return value


def current(record,full_models=False):
    full,proof,setup=resources.qualification()
    guards.require(full['source_commit']==NATIVE and record['binary']==full['binary'] and
        record['qualification']==proof and record['plan']==prospective() and record['rank_reference']==REFERENCE,
        'native/qualification/ranking/plan changed')
    ref=reference();candidate=generated_profile(ref)
    guards.require(hashlib.sha256(candidate).hexdigest()==record['candidate_profile_sha256'],'profile recipe changed')
    actual=pins(record['source_commit'])
    guards.require(len(actual)==record['source_count'] and assets.fingerprint(actual)==record['source_fingerprint'] and
        gen.common.digest(gen.SETTINGS)==record['settings_sha256'],'controller inputs/settings changed')
    for path,sha in record['generated_inputs'].items():
        guards.require(gen.common.digest(path)==sha,'generated rank input changed: '+path)
    for row in record['process_plans']:
        root=Path(row['root'])
        guards.require(row['config']==config(setup,root) and row['environment']==restored.environment(setup,root,'0'),
            'rank config/environment changed')
    guards.require(assets.machine_identity()==record['machine_identity'] and
        assets.model_pins(setup['config'],record['model_assets'],full=full_models)==record['model_assets'],'model/platform changed')
    gen.runtime_assets_unchanged(record)
    return full,setup


def prepare():
    guards.require(not ROOT.exists(),'single-use window-rank root;never retry')
    ref=reference();candidate=generated_profile(ref);full,proof,setup=resources.qualification()
    guards.require(full['source_commit']==NATIVE,'exact original native required')
    source=gen.git('rev-parse','HEAD').decode().strip();sources=pins(source)
    models=assets.model_pins(setup['config'],setup['model_assets'],full=True)
    ROOT.mkdir(mode=0o700);plans=[];inputs={}
    for i,name in enumerate(('original-profile','window-ranked-profile')):
        root=ROOT/name;root.mkdir(mode=0o700)
        for part in ('tmp','cache'):(root/part).mkdir()
        (root/'expert-profile.bin').write_bytes(candidate if i else (P/'data/expert-profile.bin').read_bytes())
        cfg,env=config(setup,root),restored.environment(setup,root,'0')
        write(root/'config.json',cfg);write(root/'plan.json',dict(binary=full['binary'],native_validation_probe=None))
        masks=census.expected_membership(root/'expert-profile.bin')
        expected=ref['offline_analysis']['rankings']['windows']['membership_sha256'] if i else ref['census']['membership_sha256']
        guards.require(hashlib.sha256(b''.join(m.to_bytes(64,'little') for m in masks)).hexdigest()==expected and
            [m.bit_count() for m in masks]==[26]*47+[34],'generated profile membership/quotas differ')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plans.append(dict(root=str(root),config=cfg,environment=env,membership_sha256=expected,
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--long-generation-only','--think','--runs','1','--profile-request-gates',
                '--workload-id',comparison.WORKLOAD_ID,'--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]))
        inputs.update({str(root/n):gen.common.digest(root/n) for n in ('config.json','plan.json','expert-profile.bin')})
    record=dict(schema='strata-window-ranked-cache-pilot-v1',mode='engine-compare',stage='prepared',source_commit=source,
        build_source_commit=full['source_commit'],binary=full['binary'],qualification=proof,plan=prospective(),rank_reference=REFERENCE,
        candidate_profile_sha256=hashlib.sha256(candidate).hexdigest(),process_plans=plans,generated_inputs=inputs,
        source_count=len(sources),source_fingerprint=assets.fingerprint(sources),settings_sha256=gen.common.digest(gen.SETTINGS),
        model_assets=models,machine_identity=assets.machine_identity(),runtime_asset_delta=setup['runtime_asset_delta'],native_validation_probe=None,
        budgets=dict(wall=2400,cpu=72000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20),
        processes=[],active=False,ownership_closed=True,complete=False,passed=False,clean_speed_qualified=False,adoption_qualified=False)
    current(record);write(ROOT/'frozen-plan.json',record);write(RESULT,record)
    print('Prepared immutable two-workload window-ranked pilot;commit plan before execution.',flush=True)


def work_evidence(text,samples,plan,root):
    guards.require(not re.search(comparison.FAILURE,text),'native error in rank pilot')
    guards.require(re.findall(r'^strata generate: expert cache (\d+) slots,',text,re.M)==['1256'] and
        'R4.2g PER-LAYER:' in text and 'pre-filled 1256 of 1256 slots from the profile; slot 0 verified' in text,
        'actual per-layer cache admission differs')
    assets.verify_pool(text,5);gen.request_evidence(text,samples)
    timing=gen.decode_timing_evidence(text,samples)
    bindings=[dict(prompt_tokens=s['prompt_tokens'],windows=t['windows'],rows=t['work']['target_rows'],
        cpu_jobs=t['work']['cpu_unique_jobs'],cpu_entries=t['work']['cpu_entries'],pool_ms=t['ms_per_window']['CPU_pool_run']*t['windows'])
        for s,t in zip(samples,timing)]
    args=plan['config']['args'];formats=census.manifest_formats(Path(args[args.index('--pack')+1])/'native_experts.txt')
    masks=census.expected_membership(root/'expert-profile.bin')
    guards.require(hashlib.sha256(b''.join(m.to_bytes(64,'little') for m in masks)).hexdigest()==plan['membership_sha256'],
        'rank pilot expected membership changed')
    work=census.extract(text,bindings,formats,masks)
    audits=gen.iq2_pair_evidence(text,samples,1,restored.recipe.component.base.machine()['physical_cpus'])
    for audit,w in zip(audits,work):
        guards.require(audit['row_ops']==640*w['cpu_entries'] and
            audit['nt1_rows']==640*sum(b['jobs_NT'][0] for b in w['format_bins'] if b['gu']==22),'IQ2/work conservation differs')
    headers=list(re.finditer(r'^strata serve: prompt .*$',text,re.M));loans=[];previous=0
    guards.require(len(headers)==len(samples),'rank loan/request framing differs')
    for header in headers:
        cycle,_=comparison.bank.admission.loan_cycle(text[previous:header.start()]);previous=header.end()
        loans.append(cycle)
    guards.require(all(a['generation']<b['generation'] for a,b in zip(loans,loans[1:])), 'rank loan generation repeated')
    return work,audits,timing,loans


def decide(rows):
    guards.require(len(rows)==2 and all(r['passed'] and r['complete'] for r in rows),'complete rank pair required')
    decisions=[]
    for i,name in enumerate(WORKLOADS):
        a,b=[r['requests'][i] for r in rows];ca,cb=[r['work'][i] for r in rows]
        matched=a['request_sha256']==b['request_sha256'] and a['read_tokens']==b['read_tokens']
        byte_ratio=(cb['logical_job_blob_bytes']/cb['rows'])/(ca['logical_job_blob_bytes']/ca['rows'])
        gpu=[c['vram_entries']/(c['cpu_entries']+c['vram_entries']) for c in (ca,cb)]
        latency=(b['decode_ms']/b['generated_tokens'])/(a['decode_ms']/a['generated_tokens'])
        decisions.append(dict(workload=name,matched_request_and_read=matched,logical_bytes_per_row_ratio=byte_ratio,
            vram_entry_fraction=gpu,token_latency_ratio=latency,passed=matched and byte_ratio<=.95 and gpu[1]>gpu[0] and latency<=1))
    return dict(workloads=decisions,advance_to_new_clean_plan=all(d['passed'] for d in decisions),
        repetitions_per_workload_per_arm=1,clean_speed_qualified=False,adoption_qualified=False)


def run():
    record=gen.committed(RESULT);current(record)
    mutable={'stage','processes','active','ownership_closed','complete','passed','decision'}
    frozen=json.loads((ROOT/'frozen-plan.json').read_text())
    guards.require({k:v for k,v in record.items() if k not in mutable}=={k:v for k,v in frozen.items() if k not in mutable} and
        record['stage'] in ('prepared','collecting') and not record['active'] and record['ownership_closed'] and
        all(r['passed'] for r in record['processes']),'closed/changed/failed rank plan')
    arm=len(record['processes']);guards.require(arm<2,'rank pilot complete')
    plan=record['process_plans'][arm];root=Path(plan['root'])
    guards.require(not (root/'record.json').exists() and not (root/'launch.json').exists(),'existing rank attempt;never retry')
    row=dict(arm=arm,passed=False,complete=False,active=False,ownership_closed=True);owner=None
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in before.splitlines() if s.startswith('-- cursor: '))
    def persist():
        row['active']=bool(owner and owner.active is not None)
        row['ownership_closed']=not row['active'] and (owner is None or all(supervision.closed(e) for e in owner.commands))
        write(root/'record.json',dict(row,commands=owner.commands if owner else []))
    def interact(io,leader):
        for i,name in enumerate(WORKLOADS):
            for stage in ('begin','end'):
                while True:
                    line=io.line(2250)
                    if line.startswith('STRATA_PROFILE_GATE '):break
                guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==
                    dict(stage=stage,phase='long-generation/'+name,run=0),'rank request gate differs')
                if stage=='begin':
                    native=json.loads((root/'engine-start.json').read_text())['native'];identity=supervision.identity(native['pid'])
                    guards.require(identity and identity['start_ticks']==native['start_ticks'] and identity['sid']==leader['pid'] and
                        Path('/proc',str(native['pid']),'exe').resolve()==Path(record['binary']['path']).resolve(),'rank native owner/executable differs')
                    env=dict(x.split(b'=',1) for x in Path('/proc',str(native['pid']),'environ').read_bytes().split(b'\0') if b'=' in x)
                    guards.require(all(env.get(k.encode())==v.encode() for k,v in plan['environment'].items()) and
                        b'LD_PRELOAD' not in env,'rank native environment differs')
                    journal,=(root/'bench').glob('*/requests.jsonl');events=[json.loads(s) for s in journal.read_text().splitlines()]
                    guards.require(len(events)==10+2*i and events[-1]['kind']=='start' and
                        events[-1]['phase']=='long-generation/'+name,'rank prior journal extent differs')
                    gen.warmup_reconciliation(events[:-1],WORKLOADS[:i])
                    samples=[e['sample'] for e in events[:-1] if e['kind']=='result']
                    guards.require(all(gen.sample_eligible(s) for s in samples[4:]),'prior long rank request ineligible')
                    deadline=time.monotonic()+5
                    while True:
                        text=(root/'engine.stderr').read_text(errors='replace')
                        count=len(re.findall(r'^strata serve: prompt .*$',text,re.M))
                        guards.require(count<=4+i,'extra rank request footer')
                        if count==4+i:break
                        io.poll();guards.require(time.monotonic()<deadline,'rank prior footer deadline');time.sleep(.01)
                    work_evidence(text,samples,plan,root)
                    row.update(native_identity=native,effective_environment_verified=True,verified_prior_requests=4+i);persist()
                io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();guards.require(time.monotonic()<deadline,'rank teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    owner=supervision.Owner(root/'owned',persist)
    try:
        entry,_,_=owner.run('bench',plan['command'],plan['environment'],P,interaction=interact,**record['budgets'])
        guards.require(supervision.closed(entry) and entry.get('exit_code')==0,'rank official owner failed')
        path,=(root/'bench').glob('*/results.json');samples,warm,events=comparison.official(root,json.loads(path.read_text()))
        guards.require(samples[0]['acceptance_rate']>=.60 and samples[1]['acceptance_rate']<=.40,'observed strong/weak MTP regimes required')
        native=gen.native_end(root,record,plan,boot,entry['pid'])
        guards.require(native['native']==row['native_identity'] and row['effective_environment_verified'],'rank native handshake changed')
        text=(root/'engine.stderr').read_text(errors='replace');all_samples=[e['sample'] for e in events if e['kind']=='result']
        work,audits,timing,loans=work_evidence(text,all_samples,plan,root)
        routes=comparison.native_routes(text,events,1)
        guards.require(len(loans)==6 and all(c['chunk']==8192 for c in loans[4:]),'rank long prefill chunk changed')
        current(record,full_models=True)
        row.update(requests=samples,warmups=warm,work=work[4:],iq2_dispatch=audits[4:],decode_timing=timing[4:],loans=loans,routes=routes,
            full_model_hashes_verified_after_process=15,native_exit_code=native['exit_code'],passed=True,complete=True)
    except BaseException as error:
        row['error']=str(error)
        if (root/'engine.stderr').exists():row['error_excerpt']=(root/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        row.update(fresh_faults=None,fault_probe_successful=False,boot_unchanged=False)
        try:
            after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            row.update(fresh_faults=[s for s in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)],
                fault_probe_successful=True,boot_unchanged=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            guards.driver_guard()
        except Exception as error:row['fault_probe_error']=str(error)
        persist()
        row['passed']=bool(row['passed'] and row['ownership_closed'] and row['fault_probe_successful'] and
            row['fresh_faults']==[] and row['boot_unchanged'] and 'fault_probe_error' not in row)
        persist();record['processes'].append(row)
        record.update(stage='closed' if not row['passed'] or arm==1 else 'collecting',active=row['active'],
            ownership_closed=row['ownership_closed'],complete=bool(arm==1 and all(r['complete'] for r in record['processes'])),
            passed=bool(arm==1 and all(r['passed'] for r in record['processes'])))
        if record['passed']:record['decision']=decide(record['processes'])
        write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ('arm','passed','complete','ownership_closed','error')}),flush=True)
    return 0 if row['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare() if args.action=='prepare' else run()


if __name__=='__main__':sys.exit(main())
