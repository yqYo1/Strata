#!/usr/bin/env python3
"""Single-use private IQ2_S BMI2 production-pool comparison; no model selection."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import statistics
import sys
sys.dont_write_bytecode=True
import cpu_roofline as base
import measurement_guards as guards
import measurement_owner as owner
import query_profile as q

P,B,A=base.P,base.B,base.A
ROOT=B/'cpu-iq2-indices-pool-v1'
PLAN=A/'cpu-iq2-indices-pool-plan.json'
RESULT=A/'cpu-iq2-indices-pool-current.json'
PREFIXES=(*base.PREFIXES,'tools/sycl','bench/sycl/cpu_iq2_indices','bench/sycl/cpu_iq2_indices_pool','tests/core/test_iq2_indices_pool.py')
ORDER=(0,1,1,0,0,1)
CAPTURE_BYTES=9461760

def environment():
    return dict(base.environment(),STRATA_IQ2_PAIR='1',STRATA_IQ2_PAIR_AUDIT='1',STRATA_IQ_SINGLE_EXACT='0',STRATA_IQ_MT_MIN='2')

def write(path,j):path.write_text(json.dumps(j,indent=2)+'\n')

def committed(path):
    raw=path.read_bytes();base.need(raw==base.git(P,'show','HEAD:'+str(path.relative_to(P))),'commit current input '+path.name)
    return json.loads(raw)

def prepare():
    base.need(not ROOT.exists(),'fresh single-use experiment required')
    base.need(not PLAN.exists() and not RESULT.exists(),'fresh source-pinned index trial required')
    pre=committed(A/'cpu-iq2-indices-current.json');base.need(pre['performance_gate_passed'] and pre['complete'] and pre['ownership_closed'] and pre['artifact_retirement']['complete'],'positive closed kernel prerequisite')
    native,n=q.current_build();revision=base.git(P,'rev-parse','HEAD').decode().strip()
    forms=[r for r in base.formats() if r['gu']==22]
    for row in forms:row['experts']=168;row['fixture_sha256']=base.fixture_hash(row)
    fixture=''.join(f"{r['shard']} {r['down']} {r['gate']} {r['up']} {r['down_offset']} {r['blob_bytes']} 168\n" for r in forms)
    base.need(all(not any(c.isspace() for c in r['shard']) for r in forms),'fixture path whitespace')
    plan=dict(schema='strata-iq2-indices-pool-v1',source=base.source_pin(P,revision,PREFIXES),ggml=base.source_pin(base.GGML,base.GGML_REV,('ggml',)),
        native_reference=dict(source=native['source_commit'],binary=native['binary'],inputs_recovered=n),
        tools={str(Path(x).resolve()):base.digest(x) for x in (base.LLVM/'bin/clang',base.LLVM/'bin/clang++',base.LLVM/'lib/clang/22/include/bmi2intrin.h','/usr/bin/cmake','/usr/bin/ninja','/usr/bin/nm','/usr/bin/objdump','/usr/bin/ldd','/usr/bin/ld','/usr/bin/ar','/usr/bin/ranlib',sys.executable)},
        environment=environment(),machine=base.machine(),formats=forms,metadata_sha256=base.digest(base.METADATA),model_identity={r['shard']:base.identity(r['shard']) for r in forms},fixture_text=fixture,
        configure=['/usr/bin/cmake','-S',str(P/'bench/sycl/cpu_iq2_indices_pool'),'-B',str(ROOT/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_AR=/usr/bin/ar','-DCMAKE_RANLIB=/usr/bin/ranlib','-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=/usr/bin/ld','-DCMAKE_C_COMPILER='+str(base.LLVM/'bin/clang'),'-DCMAKE_CXX_COMPILER='+str(base.LLVM/'bin/clang++'),'-DSTRATA_GGML_DIR='+str(base.GGML)],
        build=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target','cpu_iq2_indices_pool','--parallel','6'],
        protocol=dict(question='Does the exact PDEP IQ2_S NT1 producer improve unchanged production ExpertPool with current model dispatch and NT1-4 mixtures?',
            mechanism='Only privately linked pair producer differs. Both arms use original NativeGuPhase::Decode gate,IQ2_PAIR1,MT_MIN2,SINGLE_EXACT0,unchanged NT2-4/quant/Down,pool placement,scheduling and observer settings. Linker --wrap intercepts the existing external IQ2 pair symbol;identical wrapper/registration/two relaxed atomic counter operations are charged to both arms. Immutable argument arm is published before pool threads are created. Original native source/build stays unchanged.',
            scope='Five physically pinned workers plus computing host. Two actual22/20 and22/42 layers,168 experts each,H2560/FF640. Two rounded code/prose mixtures:28 jobs with NT counts22,5,1,0 (35 entries),and24 jobs with16,5,2,1 (36 entries). These represent census marginals,not the complete model distribution. Repeat28/24 unique experts versus168 streaming experts. Same deterministic seed20261021 and activation quantization in both arms.',
            isolation='Before timing,run complete pool output checks for both mixtures/formats in Prefill and Decode phases. Prefill must perform zero wrapped calls/rows and match its unchanged independent oracle. Decode must wrap all and only NT1 rows;existing production row/callback audits and six wrapper/native TID identities must agree. No phase environment mutation. This component does not qualify model state handoff or prefill speed.',
            codegen_gate='Before execution:both pair_dot functions have zero VPGATHERDQ;reference0 PDEP,candidate32 PDEP,no candidate calls/vector stack accesses. Exact wrapper symbol must be defined and unchanged original pair symbol linked. Static counts do not establish speed or saturation.',
            numerical='Exhaustive packed-index contract;60 producer geometry/span/alias/row/padding cases in two fresh check processes. Complete GU/quant/Down pool outputs match independent direct NT1 GU and existing NT2-4 routes for168 experts x2 formats x2 mixtures. Preserve inactive-token poison,canaries and input hashes. All eight9461760-byte captures match. Repeat complete numerical/phase checks around timed work.',
            order=list(ORDER),samples_per_cell=5,sample_floor_s=1.0,target_s=1.2,max_passes=1000000,timer_fraction_cap=.001,
            timing='Six separate owned processes A0/B0/B1/A1/A2/B2. Eight cells per process (format,mix,access),five full-pass batches>=1s each;240samples total. Warm pass and10000clock probe outside samples;probe<0.1% warm pass. Production dispatch,fallback,GU,hidden quant,Down,barriers and route observers inside. Oracles,full captures and snapshots outside. GU/quant/Down clocks are nested,not hardware service or additive to wall.',
            performance_gate='Per Down format independently:geometric mean of12 process-pair/mix/access candidate/reference median full-pool latencies<=0.97,every ratio<=1.02,and each three-pair geometric mean<=1.0. Decide formats independently;no aggregate hides a failed format. Passing permits a separate native integration/build/safety plan,not model adoption or token-speed claim.',
            safety='CPU-only static GGML,reject GPU runtime linkage. Preserve exact6e40 engine+12fixtures,current prefill/decode settings and1256slots. No original native source modification or environment mutation.',
            stop='Close first source,ownership,lowering,numerical,dispatch,timing or capture failure without automatic retry. Preserve original outcome.',
            retention='Keep source/recipe/input/binary identities,lowering,individual240 wall/CPU samples,phase/route conservation,capture hashes and per-format decisions. Retire successful raw,all captures and auxiliary build after source recovery. Remove unused fixture/controller when comparison closes.'),
        provenance=dict(basis='Main-owned existing production-pool fixture and PDEP candidate8ccdcb9c9;repository MIT/ggml notices retained. No XeStrata application code.',
            kernel_prerequisite=dict(path='bench/results/2026-10-15-pr6-recovery-audit/cpu-iq2-indices-current.json',sha256=base.digest(A/'cpu-iq2-indices-current.json'),latency_geomeans=[0.8693537276310952,0.880802726039441]),
            symbol='_ZN6strata7kernels3cpu16iq2_pair_gu_rowsERKNS1_9NativeFmtEPKhPKvPfii',
            contract='Packed index helper unchanged from checked candidate. Counters use separate64-byte-aligned per-thread slots,atomic registration and relaxed counter increments;Main snapshots only after synchronous pool return. No shared hot counter contention. Two identical successful wrapper counter updates in both arms. At most six caller threads.'),
        budgets=dict(build_seconds=600,check_seconds=120,measure_seconds=180,cpu_seconds=2400,text_bytes=64<<20,root_bytes=512<<20,rss_bytes=3<<30),retained_generation='Only original qualified6e40 engine+12fixtures;one temporary CPU-only component consumer,retire at comparison closure.')
    ROOT.mkdir(mode=0o700);(ROOT/'inputs.txt').write_text(fixture);write(PLAN,plan);write(RESULT,dict(stage='prepared',source_commit=revision,plan_sha256=base.digest(PLAN),passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False))

def load():
    plan=committed(PLAN);native,n=q.current_build()
    pre=committed(A/'cpu-iq2-indices-current.json');base.need(base.digest(A/'cpu-iq2-indices-current.json')==plan['provenance']['kernel_prerequisite']['sha256'] and pre['performance_gate_passed'],'kernel prerequisite drift')
    base.need(plan['schema']=='strata-iq2-indices-pool-v1','wrong prospective plan')
    base.need(dict(source=native['source_commit'],binary=native['binary'],inputs_recovered=n)==plan['native_reference'],'retained native reference drift')
    base.need(base.source_pin(P,plan['source']['revision'],PREFIXES)==plan['source'] and base.source_pin(base.GGML,base.GGML_REV,('ggml',))==plan['ggml'],'source drift')
    base.need(base.machine()==plan['machine'] and environment()==plan['environment'] and base.digest(base.METADATA)==plan['metadata_sha256'],'machine/environment/metadata drift')
    base.need((ROOT/'inputs.txt').read_text()==plan['fixture_text'],'fixture drift')
    for path,h in plan['tools'].items():base.need(base.digest(path)==h,'tool drift '+path)
    for row in plan['formats']:base.need(base.identity(row['shard'])==plan['model_identity'][row['shard']] and base.fixture_hash(row)==row['fixture_sha256'],'model/fixture drift')
    return plan

def parse(lines,arm,mode,cpu,process):
    placement=lines[0];base.need(placement['kind']=='placement' and placement['workers']==5 and placement['host_works'] is True,'production placement')
    threads=placement['threads'];base.need(len(threads)==6 and len({t[0] for t in threads})==6 and {t[1] for t in threads}==set(cpu) and placement['host_tid'] in {t[0] for t in threads},'six distinct physical computing threads')
    base.need(lines[-1]==dict(kind='closed',passed=True),'terminal closure')
    base.need(lines[-2]==dict(kind='validation',contract_cases=60 if mode=='check' else 0,experts_per_format=168,formats=2,mixtures=2,bitwise=True,prefill_isolated=True,index_layout=True,passed=True),'phase/numeric/contract framing')
    samples=lines[1:-2];base.need(all(s['kind']=='sample' for s in samples),'unknown records')
    expected={(down,mix,access,rep) for down in (20,42) for mix in (0,1) for access in (0,1) for rep in range(5)} if mode=='measure' else set()
    actual=[(s['down'],s['mixture'],s['access'],s['rep']) for s in samples];base.need(len(actual)==len(expected) and set(actual)==expected,'complete unique40-sample process matrix')
    for s in samples:
        jobs,entries,ones=(24,36,16) if s['mixture'] else (28,35,22)
        base.need(s['arm']==arm and s['process']==process and type(s['passes']) is int and 0<s['passes']<1000000 and s['batch_calls']==s['passes']*(168//jobs if s['access'] else 1),'full-pass work conservation')
        calls=s['batch_calls'];base.need(s['row_ops']==calls*640*entries and s['NT1_rows']==s['paired_rows']==s['wrapped_rows']==calls*640*ones and s['other_rows']==0 and s['min_thread_rows']>0,'actual production/wrapped route conservation')
        fields=('wall_s','cpu_s','GU_s','quant_s','Down_s','timer_s','warm_pass_s');base.need(all(type(s[k]) in (int,float) and math.isfinite(s[k]) and s[k]>=0 for k in fields) and s['wall_s']>=1 and s['warm_pass_s']>0 and s['timer_s']<s['warm_pass_s']*.001,'timing floor/overhead')
        base.need(s['GU_s']+s['quant_s']+s['Down_s']<=s['wall_s']+.002,'nested phase clocks exceed pool wall')
    return samples

def reduce(checks):
    base.need(len(checks)==6 and [c['arm'] for c in checks]==list(ORDER),'complete independent process order')
    decisions=[]
    for down in (20,42):
        medians={(i,mix,access):statistics.median(s['wall_s']/s['batch_calls'] for s in checks[i]['samples'] if s['down']==down and s['mixture']==mix and s['access']==access) for i in range(6) for mix in (0,1) for access in (0,1)}
        ratios=[dict(pair=i//2,mixture=mix,access=access,ratio=medians[i if ORDER[i] else i+1,mix,access]/medians[i+1 if ORDER[i] else i,mix,access]) for i in (0,2,4) for mix in (0,1) for access in (0,1)]
        pairs=[statistics.geometric_mean(r['ratio'] for r in ratios if r['pair']==i) for i in range(3)];gm=statistics.geometric_mean(r['ratio'] for r in ratios)
        decisions.append(dict(down=down,ratios=ratios,pair_geomeans=pairs,geomean=gm,performance_gate_passed=gm<=.97 and max(r['ratio'] for r in ratios)<=1.02 and max(pairs)<=1))
    return decisions

def execute(stage):
    plan=load();previous=committed(RESULT);base.need(previous['stage']=={'build':'prepared','check':'build','measure':'check'}[stage] and not previous['active'] and previous['ownership_closed'],'preceding closed stage required')
    if stage!='build':base.need(previous['passed'] and previous['complete'],'preceding stage failed')
    path=ROOT/(stage+'.json');base.need(not path.exists(),'single-use stage;inspect original owner,never retry')
    r=dict(stage=stage,source_commit=plan['source']['revision'],plan_sha256=base.digest(PLAN),passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,checks=[])
    own=None
    def persist():
        r['active']=bool(own and own.active is not None);r['ownership_closed']=not r['active'] and all(owner.closed(e) for e in (own.commands if own else []));write(path,dict(r,commands=own.commands if own else []))
    own=owner.Owner(ROOT/'owned',persist)
    def run(label,argv,wall):
        entry,out,err=own.run(stage+'-'+label,argv,plan['environment'],P,wall=wall,cpu=2400,rss_cap=3<<30,text_cap=64<<20,total_cap=512<<20,file_cap=64<<20)
        base.need(owner.closed(entry) and entry['exit_code']==0,'owned child failed '+label)
        base.need(owner.artifact_bytes(ROOT)<=512<<20,'root byte budget');return out.read_text(),err.read_text()
    try:
        if stage=='build':
            run('configure',plan['configure'],120);run('compile',plan['build'],600);binary=ROOT/'build/cpu_iq2_indices_pool';r['binary']=dict(path=str(binary),sha256=base.digest(binary),bytes=binary.stat().st_size);r['compile_commands_sha256']=base.digest(ROOT/'build/compile_commands.json')
            ldd,_=run('linkage',['/usr/bin/ldd',str(binary)],20);base.need(not any(k in ldd.lower() for k in ('sycl','level_zero','libze','cuda','amdhip')),'GPU runtime linked')
            nm,_=run('symbols',['/usr/bin/nm','-S','--defined-only','--demangle',str(binary)],20);r['lowering']={};raw_nm,_=run('raw-symbols',['/usr/bin/nm','--defined-only',str(binary)],20);base.need(all(len(re.findall(r'^[0-9a-f]+\s+T\s+'+re.escape(symbol)+r'$',raw_nm,re.M))==1 for symbol in (plan['provenance']['symbol'],'__wrap_'+plan['provenance']['symbol'])),'private linker wrapper and original pair symbols required')
            for name in ('pair_dot','indices_pair_dot'):
                matches=re.findall(r'^([0-9a-f]+) ([0-9a-f]+) t .*::'+name+r'\(.*$',nm,re.M);base.need(len(matches)==1,'unique function '+name);start,size=(int(x,16) for x in matches[0]);asm,_=run(name,['/usr/bin/objdump','-d','--no-show-raw-insn',f'--start-address={start}',f'--stop-address={start+size}',str(binary)],20)
                instructions=[m.group(1) for m in re.finditer(r'^\s*[0-9a-f]+:\s+(.*)$',asm,re.M)]
                r['lowering'][name]=dict(start=start,bytes=size,gathers=sum(x.startswith('vpgatherdq ') for x in instructions),pdep=sum(x.startswith('pdep ') for x in instructions),calls=sum(x.startswith('call') for x in instructions),vector_stack_accesses=sum(bool(re.match(r'v\w+\s',x) and re.search(r'%[xy]mm\d+',x) and re.search(r'\(%r(?:sp|bp)\)',x)) for x in instructions))
            a,b=r['lowering']['pair_dot'],r['lowering']['indices_pair_dot'];base.need(a['gathers']==b['gathers']==0 and a['pdep']==0 and b['pdep']==32 and b['calls']==0 and b['vector_stack_accesses']==0,'prospective lowering screen failed')
        else:
            built=json.loads((ROOT/'build.json').read_text());r['binary']=built['binary'];base.need(built['passed'] and built['complete'] and built['ownership_closed'] and base.digest(r['binary']['path'])==r['binary']['sha256'],'exact qualified component build required')
            expected_hash=previous['checks'][0]['capture_sha256'] if stage=='measure' else None
            for i,arm in enumerate(ORDER if stage=='measure' else (0,1)):
                capture=ROOT/(stage+'-'+str(i)+'.bin');stdout,_=run(str(i),[r['binary']['path'],'perf' if stage=='measure' else 'check',str(ROOT/'inputs.txt'),str(capture),str(arm),str(i//2)],180 if stage=='measure' else 120)
                samples=parse([json.loads(s) for s in stdout.splitlines()],arm,'measure' if stage=='measure' else 'check',plan['machine']['physical_cpus'],i//2);base.need(capture.stat().st_size==CAPTURE_BYTES,'capture extent');h=base.digest(capture);expected_hash=expected_hash or h;base.need(h==expected_hash,'cross-arm capture mismatch')
                r['checks'].append(dict(index=i,arm=arm,bitwise=True,capture_sha256=h,samples=samples));persist();print('Closed '+stage+' '+str(i+1),flush=True)
            if stage=='measure':r['decisions']=reduce(r['checks']);r['model_adoption_qualified']=False
        load();r.update(passed=True,complete=True)
    except BaseException as e:r['error']=str(e)
    finally:
        r['ownership_closed']=own.active is None and all(owner.closed(e) for e in own.commands);r['passed']=bool(r['passed'] and r['ownership_closed']);persist();write(RESULT,r)
    print(json.dumps({k:r.get(k) for k in ('stage','passed','complete','ownership_closed','error')}),flush=True)
    return 0 if r['passed'] else 1

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','build','check','measure'));args=p.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        return prepare() if args.stage=='prepare' else execute(args.stage)
if __name__=='__main__':sys.exit(main())
