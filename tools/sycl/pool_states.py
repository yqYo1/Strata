#!/usr/bin/env python3
"""Private unchanged-policy CPU pool attribution; no inference adoption."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode=True
import cpu_roofline as base
import measurement_guards as guards
import measurement_owner as owner
import query_profile as q

P,B,A=base.P,base.B,base.A
ROOT=B/'cpu-pool-states-v2'
PLAN=A/'cpu-pool-states-plan.json'
RESULT=A/'cpu-pool-states-current.json'
PREFIXES=(*base.PREFIXES,'tools/sycl','bench/sycl/cpu_iq2','bench/sycl/cpu_pool_state_observer','tests/core/test_pool_states.py')
CASES=[(down,mix,phase,gap) for down,mix in ((20,0),(42,1)) for phase in (0,1) for gap in (0,40)]
ORDER=(0,1,1,0)

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def committed(path):
    raw=path.read_bytes();base.need(raw==base.git(P,'show','HEAD:'+str(path.relative_to(P))),'commit input '+path.name);return json.loads(raw)
def environment():return dict(base.environment(),STRATA_IQ2_PAIR='1',STRATA_IQ2_PAIR_AUDIT='1',STRATA_IQ_SINGLE_EXACT='0',STRATA_IQ_MT_MIN='2')
def text_or_error(path):
    try:return Path(path).read_text().strip()
    except OSError as e:return dict(unavailable=type(e).__name__)

def prepare():
    base.need(not ROOT.exists(),'fresh single-use v2 root required')
    old=committed(RESULT);old_plan=committed(PLAN)
    base.need(old['source_commit']=='4ac05411622d502870558185c49980059bac630f' and old['stage']=='check' and not old['passed'] and not old['complete'] and not old['active'] and old['ownership_closed'] and old['artifact_retirement']['complete'] and old['plan_sha256']==base.digest(PLAN),'preserve and retire failed v1 first')
    previous=committed(A/'cpu-iq2-indices-pool-current.json');base.need(previous['complete'] and previous['ownership_closed'] and not previous['performance_gate_passed'] and previous['artifact_retirement']['complete'],'close failed comparator first')
    native,n=q.current_build();revision=base.git(P,'rev-parse','HEAD').decode().strip();forms=[r for r in base.formats() if r['gu']==22]
    for row in forms:row['experts']=168;row['fixture_sha256']=base.fixture_hash(row)
    fixture=''.join(f"{r['shard']} {r['down']} {r['gate']} {r['up']} {r['down_offset']} {r['blob_bytes']} 168\n" for r in forms)
    base.need(all(not any(c.isspace() for c in r['shard']) for r in forms),'fixture whitespace')
    plan=dict(schema='strata-cpu-pool-states-v2',source=base.source_pin(P,revision,PREFIXES),ggml=base.source_pin(base.GGML,base.GGML_REV,('ggml',)),native_reference=dict(source=native['source_commit'],binary=native['binary'],inputs_recovered=n),tools={str(Path(x).resolve()):base.digest(x) for x in (base.LLVM/'bin/clang',base.LLVM/'bin/clang++','/usr/bin/cmake','/usr/bin/ninja','/usr/bin/ld','/usr/bin/ar','/usr/bin/ranlib','/usr/bin/ldd',sys.executable)},environment=environment(),machine=base.machine(),formats=forms,metadata_sha256=base.digest(base.METADATA),model_identity={r['shard']:base.identity(r['shard']) for r in forms},fixture_text=fixture,
        scheduler_capability=dict(sched_schedstats=text_or_error('/proc/sys/kernel/sched_schedstats'),perf_event_paranoid=text_or_error('/proc/sys/kernel/perf_event_paranoid'),switch_event=text_or_error('/sys/kernel/tracing/events/sched/sched_switch/id'),wakeup_event=text_or_error('/sys/kernel/tracing/events/sched/sched_wakeup/id'),runnable_delay_collected=False,meaning='No usable scheduler event collection in this CPU fixture;never infer blocked/runnable duration from CV wall or CPU ticks.'),
        configure=['/usr/bin/cmake','-S',str(P/'bench/sycl/cpu_pool_state_observer'),'-B',str(ROOT/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_AR=/usr/bin/ar','-DCMAKE_RANLIB=/usr/bin/ranlib','-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=/usr/bin/ld','-DCMAKE_C_COMPILER='+str(base.LLVM/'bin/clang'),'-DCMAKE_CXX_COMPILER='+str(base.LLVM/'bin/clang++'),'-DSTRATA_GGML_DIR='+str(base.GGML)],
        host_test=[sys.executable,'-B',str(P/'tests/core/test_pool_states.py')],
        build=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target','pool_state_raw','pool_state_observed','--parallel','6'],cases=CASES,order=ORDER,
        protocol=dict(question='Where does unchanged-policy CPU pool thread time go under the two prior slow/variable streaming cases?',
            scope='CPU-only diagnostic,not token throughput. Actual22/20 code mixture28jobs/35entries and22/42 prose24jobs/36entries,168 real experts. Stream all168 per pass. Explicit Prefill/Decode calls in separate processes. Controlled0/40ms idle after each batch exercises20ms spin/CV policy;gaps are workload probes,not a model-time distribution or policy treatment.',
            unchanged='Keep original20ms policy,5 pinned workers+host,epoch/sleepers ordering,predicate wait,claims,stop/join,GU/quant/Down math and immutable startup environment. Restore inherited affinity for pool topology/worker creation,then pin host. Keep pre-pin topology for final placement check. Raw library compiles original pool.cpp. Observed library differs only by the committed private observer insertions;other source/flags match. No product source or GPU build modification.',
            accounting='One sole-writer slot per thread;read only after join. CLOCK_MONOTONIC and CLOCK_THREAD_CPUTIME_ID transitions partition full root lifetime into Control,Spin,CV-wait scope,GU-task,Down-task,host WaitParked/WaitDone,Quant and Other-task,by Unknown/Prefill/Decode. Idle and transition phases remainUnknown. Claimed native task scopes close before done release. No per-PAUSE counter. CV scope includes blocking,runnable delay,predicate/reacquisition;it is not pure sleep. Thread CPU also includes stalls/control. Context-switch counts bracket lifetime,not durations.',
            coverage='Per thread:closed,zero errors/depth,nonnegative intervals,exact wall/CPU sum conservation. Known task phases correct;idleUnknown. Across six distinct physical TIDs:18 GU+18Down tasks,1Quant,2WaitDone,4WaitParked per pool call including two warm passes;Other0. Original per-sample GU callback/weighted-row/NT1/pair audits conserve work and every thread does GU math. Full active outputs match independent NT1/direct and unchanged multi-row oracle;canaries,inactive tokens,input hashes.',
            check='All8cases x raw/observed (16 owned check processes),two warm passes only,no samples. Verify CV entered for40ms case. Complete output captures match per shape across phase/gap/arm. Check is not timing evidence.',
            measure='8cases x4 separate owned processes raw/observed/observed/raw (32 processes). Each5 full-pass batches target1.2s,floor1s,cap1000000passes;160 samples. Two warm passes outside samples but included lifetime ledger. All reference/model loading precede pool lifetime;output/ledger emission follows stop/join. CPU fixtures and explicit sleeps are not a model critical-path decomposition.',
            overhead='Clock probe10000 stamp pairs before pool,below0.1% warm pass. Compare observed/raw median batch wall for each shape/phase/gap pair;both ratios<=1.10 required for overhead-qualified attribution. Gap40ms wall is sleep dominated,so only gap0 qualifies overhead cost for computation. Failures remain failed;no retry or altered gate. Timestamp calibration omits bookkeeping,so it is not an overhead proof alone.',
            decision='No optimization/model adoption gate. Report attribution only when numerical,coverage and prospective overhead gates pass;otherwise preserve valid raw counts and mark attribution limited. Runnable delay remainsunknown without usable scheduler events. No CPU/RAM/PCIe saturation claim.',
            stop='First source,ownership,numerical,capture,accounting,phase,timer or sample failure closes stage;no automatic retry.',
            retention='Keep source/recipe/binary identities,useful individual160 samples,per-thread/phase disjoint totals and conservation,format/phase overhead decisions,correctness/hash limits. Retire normal logs,48 captures and extra build after source recovery;archive superseded observer source in Git.'),
        provenance=dict(basis='Main-authored hooks in original6e40 pool.cpp (MIT),Main CPU fixture/oracle. No XeStrata application code. Installed official compiler only.',original_pool_sha256=base.digest(P/'src/kernels/cpu/pool.cpp'),previous_diagnostic=dict(commit='5c47879f0',result_sha256=base.digest(RESULT),plan_sha256=base.digest(PLAN),passed=False,retired=True,correction='Restore inherited affinity before pool topology/worker creation;pin host afterward;use pre-pin topology at profile closure.'),prior_comparator=dict(path='bench/results/2026-10-15-pr6-recovery-audit/cpu-iq2-indices-pool-current.json',sha256=base.digest(A/'cpu-iq2-indices-pool-current.json'))),budgets=dict(build_seconds=600,child_seconds=120,cpu_seconds=2400,text_bytes=64<<20,root_bytes=512<<20,rss_bytes=3<<30),retained_generation='Original qualified6e40 engine+12fixtures;one temporary CPU diagnostic build,retire at closure. Legacy holds remain separate in current build result.')
    ROOT.mkdir(mode=0o700);(ROOT/'inputs.txt').write_text(fixture);write(PLAN,plan);write(RESULT,dict(stage='prepared',source_commit=revision,plan_sha256=base.digest(PLAN),passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False))

def load():
    p=committed(PLAN);native,n=q.current_build();base.need(p['schema']=='strata-cpu-pool-states-v2','schema');base.need(dict(source=native['source_commit'],binary=native['binary'],inputs_recovered=n)==p['native_reference'],'native drift')
    base.need(base.source_pin(P,p['source']['revision'],PREFIXES)==p['source'] and base.source_pin(base.GGML,base.GGML_REV,('ggml',))==p['ggml'],'source drift');base.need(base.machine()==p['machine'] and environment()==p['environment'] and base.digest(base.METADATA)==p['metadata_sha256'],'environment drift')
    base.need((ROOT/'inputs.txt').read_text()==p['fixture_text'],'fixture drift')
    for path,h in p['tools'].items():base.need(base.digest(path)==h,'tool drift')
    for row in p['formats']:base.need(base.identity(row['shard'])==p['model_identity'][row['shard']] and base.fixture_hash(row)==row['fixture_sha256'],'model drift')
    base.need(base.digest(A/'cpu-iq2-indices-pool-current.json')==p['provenance']['prior_comparator']['sha256'],'prior evidence drift')
    return p

def parse(lines,case,observed,mode,cores):
    base.need(mode in ('check','measure') and tuple(case) in CASES and bool(lines),'bounded parser input')
    down,mix,phase,gap=case;last=lines[-1];base.need(last['kind']=='closed' and last['mode']==mode and last['observed'] is bool(observed) and (last['down'],last['mixture'],last['phase'],last['gap_ms'])==tuple(case),'case framing')
    base.need(all(last[k] is True for k in ('passed','bitwise','guards','inputs_unchanged')),'numerical closure');base.need(0<last['clock_probe_s']<last['warm_pass_s']*.001,'clock overhead')
    threads=[r for r in lines[:-1] if r['kind']=='thread'];samples=[r for r in lines[:-1] if r['kind']=='sample'];base.need(len(threads)+len(samples)==len(lines)-1,'unknown records')
    base.need(len(samples)==(0 if mode=='check' else 5) and {s['rep'] for s in samples}==set(range(len(samples))),'complete unique samples')
    jobs,entries,ones=(28,35,22) if mix==0 else (24,36,16);calls=2*(168//jobs)
    for s in samples:
        base.need(type(s['passes']) is int and 0<s['passes']<1000000 and s['batch_calls']==s['passes']*(168//jobs),'batch work');n=s['batch_calls'];calls+=n
        base.need(s['row_ops']==n*640*entries and s['NT1_rows']==n*640*ones and s['pair_rows']==(n*640*ones if phase else 0) and s['min_thread_rows']>0,'native dispatch/rows')
        base.need(all(type(s[k]) in (int,float) and math.isfinite(s[k]) and s[k]>=0 for k in ('wall_s','cpu_s','GU_s','quant_s','Down_s')) and s['wall_s']>=1 and s['GU_s']+s['quant_s']+s['Down_s']<=s['wall_s']+.002,'timing eligibility')
    base.need(last['calls_including_warmups']==calls,'lifetime calls')
    if not observed:base.need(not threads,'raw source emitted observer data');return dict(samples=samples,threads=[],footer=last)
    base.need(len(threads)==6 and {t['slot'] for t in threads}==set(range(6)) and len({t['tid'] for t in threads})==6 and {t['core'] for t in threads}==set(cores),'observer placement')
    counts=[0]*9;cv=0
    for t in threads:
        base.need(all(type(t[k]) is int and t[k]>=0 for k in ('slot','core','life_wall_ns','life_cpu_ns','voluntary_switches','involuntary_switches')) and type(t['tid']) is int and t['tid']>0 and t['life_wall_ns']>0,'thread type/extent')
        states=t['states'];base.need(len({tuple(v[:2]) for v in states})==len(states),'duplicate state');wall=cpu=0
        for p,k,n,w,c in states:
            base.need(all(type(v) is int and v>=0 for v in (p,k,n,w,c)) and p<3 and k<9 and n>0,'state type/extent');wall+=w;cpu+=c
            base.need(k!=8 and (k not in (1,2) or p==0) and (k not in (3,4,5,6,7) or p==phase+1),'state phase contract')
            if p==phase+1:counts[k]+=n
            if k==2:cv+=n
        base.need(wall==t['life_wall_ns'] and cpu==t['life_cpu_ns'] and cpu<=wall+1000000 and min(t['voluntary_switches'],t['involuntary_switches'])>=0,'full-lifetime accounting')
    base.need([counts[k] for k in (3,4,7,6,5)]==[18*calls,18*calls,calls,2*calls,4*calls],'task/wait/quant coverage')
    if gap:base.need(cv>0,'long gap did not exercise CV scope')
    return dict(samples=samples,threads=threads,footer=last)

def reduce(checks):
    decisions=[]
    for case in CASES:
        rows=[x for x in checks if x['case']==list(case)];base.need([x['observed'] for x in rows]==list(ORDER),'complete process order')
        med=[statistics.median(s['wall_s']/s['batch_calls'] for s in r['samples']) for r in rows];ratios=[med[1]/med[0],med[2]/med[3]]
        decisions.append(dict(case=list(case),observed_raw_ratios=ratios,overhead_gate_passed=max(ratios)<=1.10,compute_overhead_case=case[-1]==0))
    return decisions

def execute(stage):
    p=load();previous=committed(RESULT);base.need(previous['stage']=={'build':'prepared','check':'build','measure':'check'}[stage] and not previous['active'] and previous['ownership_closed'],'preceding closed stage')
    if stage!='build':base.need(previous['passed'] and previous['complete'],'preceding stage failed')
    receipt=ROOT/(stage+'.json');base.need(not receipt.exists(),'single-use stage,inspect original owner')
    r=dict(stage=stage,source_commit=p['source']['revision'],plan_sha256=base.digest(PLAN),passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,checks=[]);own=None
    def persist():r.update(active=bool(own and own.active is not None),ownership_closed=not(own and own.active is not None) and all(owner.closed(e) for e in (own.commands if own else [])));write(receipt,dict(r,commands=own.commands if own else []))
    own=owner.Owner(ROOT/'owned',persist)
    def run(label,argv,wall):
        e,out,err=own.run(stage+'-'+label,argv,p['environment'],P,wall=wall,cpu=2400,rss_cap=3<<30,text_cap=64<<20,total_cap=512<<20,file_cap=64<<20);base.need(owner.closed(e) and e['exit_code']==0,'owned child failed '+label);base.need(owner.artifact_bytes(ROOT)<=512<<20,'root budget');return out.read_text()
    try:
        if stage=='build':
            run('host-tests',p['host_test'],30);run('configure',p['configure'],120);run('compile',p['build'],600);r['binaries']={}
            for arm in (0,1):
                binary=ROOT/'build'/('pool_state_observed' if arm else 'pool_state_raw');r['binaries'][str(arm)]=dict(path=str(binary),sha256=base.digest(binary),bytes=binary.stat().st_size);ldd=run('link-'+str(arm),['/usr/bin/ldd',str(binary)],20);base.need(not any(x in ldd.lower() for x in ('sycl','level_zero','libze','cuda','amdhip')),'GPU runtime linked')
            r['compile_commands_sha256']=base.digest(ROOT/'build/compile_commands.json')
        else:
            built=json.loads((ROOT/'build.json').read_text());base.need(built['passed'] and built['complete'] and built['ownership_closed'],'closed build');r['binaries']=built['binaries']
            for a in r['binaries'].values():base.need(base.digest(a['path'])==a['sha256'],'binary drift')
            hashes={}
            if stage=='measure':
                checked=json.loads((ROOT/'check.json').read_text());base.need(checked['passed'] and checked['complete'] and checked['ownership_closed'],'closed numerical stage')
                base.need(len(checked['checks'])==16,'complete numerical matrix')
                for row in checked['checks']:base.need(hashes.setdefault(row['case'][0],row['capture_sha256'])==row['capture_sha256'],'numerical capture agreement')
            for case in CASES:
                for arm in (ORDER if stage=='measure' else (0,1)):
                    index=len(r['checks']);capture=ROOT/(stage+'-'+str(index)+'.bin');stdout=run(str(index),[r['binaries'][str(arm)]['path'],stage,str(ROOT/'inputs.txt'),str(capture),*[str(v) for v in case]],120)
                    parsed=parse([json.loads(s) for s in stdout.splitlines()],case,arm,'check' if stage=='check' else 'measure',p['machine']['physical_cpus']);mix=case[1];base.need(capture.stat().st_size==(2150400 if mix==0 else 2580480),'capture extent');h=base.digest(capture);base.need(hashes.setdefault(case[0],h)==h,'phase/gap/arm capture mismatch')
                    r['checks'].append(dict(index=index,case=list(case),observed=arm,capture_sha256=h,**parsed));persist();print('Closed '+stage+' '+str(index+1),flush=True)
            if stage=='measure':
                r['decisions']=reduce(r['checks']);r['attribution_gate_passed']=all(d['overhead_gate_passed'] for d in r['decisions']);r['model_adoption_qualified']=False;r['runnable_delay_qualified']=False
        load();r.update(passed=True,complete=True)
    except BaseException as e:r['error']=str(e)
    finally:r['ownership_closed']=own.active is None and all(owner.closed(e) for e in own.commands);r['passed']=bool(r['passed'] and r['ownership_closed']);persist();write(RESULT,r)
    print(json.dumps({k:r.get(k) for k in ('stage','passed','complete','ownership_closed','error')}),flush=True);return 0 if r['passed'] else 1

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('prepare','build','check','measure'));args=ap.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);return prepare() if args.stage=='prepare' else execute(args.stage)
if __name__=='__main__':sys.exit(main())
