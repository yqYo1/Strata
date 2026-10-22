#!/usr/bin/env python3
"""Single-use private IQ2_S BMI2 lookup-index comparison; no model selection."""
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
ROOT=B/'cpu-iq2-indices-v1'
PLAN=A/'cpu-iq2-indices-plan.json'
RESULT=A/'cpu-iq2-indices-current.json'
PREFIXES=(*base.PREFIXES,'tools/sycl','bench/sycl/cpu_iq2_indices','tests/core/test_cpu_iq2_indices.py')
ORDER=(0,1,1,0,0,1)
CAPTURE_BYTES=2*168*640*4

def write(path,j):path.write_text(json.dumps(j,indent=2)+'\n')

def committed(path):
    raw=path.read_bytes();base.need(raw==base.git(P,'show','HEAD:'+str(path.relative_to(P))),'commit current input '+path.name)
    return json.loads(raw)

def prepare():
    base.need(not ROOT.exists(),'fresh single-use experiment required')
    base.need(not PLAN.exists() and not RESULT.exists(),'fresh source-pinned index trial required')
    native,n=q.current_build();revision=base.git(P,'rev-parse','HEAD').decode().strip()
    forms=[r for r in base.formats() if r['gu']==22]
    for row in forms:row['experts']=168;row['fixture_sha256']=base.fixture_hash(row)
    fixture=''.join(f"{r['shard']} {r['down']} {r['gate']} {r['up']} {r['down_offset']} {r['blob_bytes']}\n" for r in forms)
    base.need(all(not any(c.isspace() for c in r['shard']) for r in forms),'fixture path whitespace')
    plan=dict(schema='strata-iq2-indices-v1',source=base.source_pin(P,revision,PREFIXES),ggml=base.source_pin(base.GGML,base.GGML_REV,('ggml',)),
        native_reference=dict(source=native['source_commit'],binary=native['binary'],inputs_recovered=n),
        tools={str(Path(x).resolve()):base.digest(x) for x in (base.LLVM/'bin/clang',base.LLVM/'bin/clang++',base.LLVM/'lib/clang/22/include/bmi2intrin.h','/usr/bin/cmake','/usr/bin/ninja','/usr/bin/nm','/usr/bin/objdump','/usr/bin/ldd',sys.executable)},
        environment=base.environment(),machine=base.machine(),formats=forms,metadata_sha256=base.digest(base.METADATA),model_identity={r['shard']:base.identity(r['shard']) for r in forms},fixture_text=fixture,
        configure=['/usr/bin/cmake','-S',str(P/'bench/sycl/cpu_iq2_indices'),'-B',str(ROOT/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_C_COMPILER='+str(base.LLVM/'bin/clang'),'-DCMAKE_CXX_COMPILER='+str(base.LLVM/'bin/clang++'),'-DSTRATA_GGML_DIR='+str(base.GGML)],
        build=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target','cpu_iq2_indices','--parallel','6'],
        protocol=dict(question='Does constructing four exact10-bit grid indices with two BMI2 PDEP operations reduce the current IQ2_S NT1 GU producer cost on Zen3?',
            mechanism='Original pair producer is unchanged. Candidate changes lookup-index construction and private symbols only. Pack four QS bytes into16-bit lanes with PDEP mask00ff00ff00ff00ff,and four QH2-bit groups with mask0300030003000300,then OR. Full unrolling,8192-byte codebook,scalar grid reads,signs,scales,integer/FP arithmetic and rounding order remain unchanged. Scalar QS/QH code-field loads fall from5 to2 per half;actual cache-line/DRAM traffic is not established.',
            scope='Kernel-only actual iq2_pair_gu_rows versus private candidate,including identical guards,SiLU and output store. Single pinned physical computing core;no production dispatcher,pool,Down or NT2-4. This isolates lookup lowering before any proposed decode-only integration.',
            inputs='Two actual22/20 and22/42 format layers,168 experts each,H2560/FF640. Same four deterministic quantized activation rows. Reuse first6 shuffled experts versus stream168;GU active footprints6297600/176332800B are logical,not physical RAM traffic. Full fixture loading outside timing.',
            codegen_gate='Before execution:both pair_dot functions have zero VPGATHERDQ instructions;the reference has zero PDEP and the candidate32 PDEP instructions per fully unrolled block body,no calls and no vector stack accesses. Static counts do not prove dynamic uops,cache traffic or speed. A failed screen permits no execution.',
            numerical='Exhaust all256 QH bytes x256 equal QS-byte choices x4 lanes,and all32 independent QS basis bits,against scalar index construction. Both arms independently match existing iq256_single_gu_rows output bits for every row/expert;finite outputs,partial/empty/tail rows,rejected metadata/null/alignment/extent,canaries and immutable inputs. Complete860160-byte final captures must match in two check and all six timing processes.',
            order=list(ORDER),samples_per_cell=5,sample_floor_s=1.0,target_s=1.2,max_passes=1000000,timer_fraction_cap=.001,
            timing='Six separate owned processes A/B/B/A/A/B;each both format layers and reuse/stream cells. Eight warm passes,then one timer-reference pass. Five full-pass batches>=1s per cell;timer probe outside samples and<0.1% warm pass. No added repetitions or retries after results.',
            performance_gate='Per Down format independently:geometric mean of six process-pair/access candidate/reference median latency ratios<=0.97,every ratio<=1.02 and each of three process-pair geomeans<=1.0. No aggregate gain hides a failed format. Passing permits a separate production-pool plan,not model adoption.',
            safety='CPU-only static GGML build;reject GPU runtime linkage. Preserve exact6e40 engine+12fixtures at original paths. No process environment mutation or model dispatch change.',
            stop='Close first source,ownership,lowering,numerical,capture,timing or work failure without automatic retry. Preserve original outcome.',
            retention='Keep exact source/recipe/input/binary identities,static lowering summary,individual120 wall/CPU samples,correctness/capture hashes and decision. Retire normal logs,captures and auxiliary build after source recovery;remove private source if no imminent pool candidate.'),
        provenance=dict(basis='Main-owned original6e40 src/kernels/cpu/iq2_pair_avx2.cpp and Main fixture atf5eaccdd5;repository MIT and existing ggml authors notice retained. No XeStrata application code.',
            grid_contract='Pinned compiler lib/clang/22/include/bmi2intrin.h:_pdep_u64 deposits successive unsigned source bits into set mask positions and zeros other bits. Nonoverlapping masks form four16-bit words with values0..1023. Four-byte QS reads stay within the first32 QS bytes;QH reads stay within8 bytes. CPU BMI2 support checked before execution;no instruction-speed assumption.',
            table_inspection=dict(header_sha256='0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c',entries=1024,magnitudes=[8,25,43],decision='Exact16-bit codebook storage is possible but adds expansion before the existing dot. This trial instead isolates index formation without changing table storage.'),
            baseline_observation=dict(binary_sha256=native['binary']['sha256'],pair_dot_start='0x6f0ed0',pair_dot_bytes=3220,indexed_grid_qword_loads=64,vector_inserts=16,vector_stack_accesses=0)),
        budgets=dict(build_seconds=600,check_seconds=120,measure_seconds=180,cpu_seconds=2400,text_bytes=64<<20,root_bytes=512<<20,rss_bytes=3<<30),retained_generation='Only original qualified6e40 engine+12fixtures;one temporary CPU-only component consumer,retire at comparison closure.')
    ROOT.mkdir(mode=0o700);(ROOT/'inputs.txt').write_text(fixture);write(PLAN,plan);write(RESULT,dict(stage='prepared',source_commit=revision,plan_sha256=base.digest(PLAN),passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False))

def load():
    plan=committed(PLAN);native,n=q.current_build()
    base.need(plan['schema']=='strata-iq2-indices-v1','wrong prospective plan')
    base.need(dict(source=native['source_commit'],binary=native['binary'],inputs_recovered=n)==plan['native_reference'],'retained native reference drift')
    base.need(base.source_pin(P,plan['source']['revision'],PREFIXES)==plan['source'] and base.source_pin(base.GGML,base.GGML_REV,('ggml',))==plan['ggml'],'source drift')
    base.need(base.machine()==plan['machine'] and base.environment()==plan['environment'] and base.digest(base.METADATA)==plan['metadata_sha256'],'machine/environment/metadata drift')
    base.need((ROOT/'inputs.txt').read_text()==plan['fixture_text'],'fixture drift')
    for path,h in plan['tools'].items():base.need(base.digest(path)==h,'tool drift '+path)
    for row in plan['formats']:base.need(base.identity(row['shard'])==plan['model_identity'][row['shard']] and base.fixture_hash(row)==row['fixture_sha256'],'model/fixture drift')
    return plan

def parse(lines,arm,mode,cpu):
    base.need(lines[0]==dict(kind='placement',cpu=cpu,computing_threads=1,arm=arm),'compute placement')
    base.need(lines[-1]==dict(kind='closed',arm=arm,formats=2,experts_per_format=168,index_layout=True,bitwise=True,all_guards=True,inputs_unchanged=True,passed=True),'closure/numerical framing')
    samples=lines[1:-1];base.need(all(s['kind']=='sample' for s in samples),'unknown records')
    expected={(down,access,rep) for down in (20,42) for access in (0,1) for rep in range(5)} if mode=='measure' else set()
    actual=[(s['down'],s['access'],s['rep']) for s in samples]
    base.need(len(actual)==len(expected) and set(actual)==expected,'unique complete samples')
    for s in samples:
        base.need(s['arm']==arm and type(s['passes']) is int and 0<s['passes']<1000000 and s['jobs']==s['passes']*(168 if s['access'] else 6),'work conservation')
        base.need(all(type(s[k]) in (int,float) and math.isfinite(s[k]) and s[k]>=0 for k in ('wall_s','cpu_s','timer_s','warm_s')) and s['wall_s']>=1 and s['warm_s']>0 and s['timer_s']<s['warm_s']*.001,'timing floor/overhead')
    return samples

def reduce(checks):
    base.need(len(checks)==6 and [c['arm'] for c in checks]==list(ORDER),'complete independent process order')
    decisions=[]
    for down in (20,42):
        medians={(i,access):statistics.median(s['wall_s']/s['jobs'] for s in checks[i]['samples'] if s['down']==down and s['access']==access) for i in range(6) for access in (0,1)}
        ratios=[dict(pair=i//2,access=access,ratio=medians[i if ORDER[i] else i+1,access]/medians[i+1 if ORDER[i] else i,access]) for i in (0,2,4) for access in (0,1)]
        pairs=[statistics.geometric_mean(r['ratio'] for r in ratios if r['pair']==i) for i in range(3)];gm=statistics.geometric_mean(r['ratio'] for r in ratios)
        decisions.append(dict(down=down,ratios=ratios,pair_geomeans=pairs,geomean=gm,performance_gate_passed=gm<=.97 and max(r['ratio'] for r in ratios)<=1.02 and max(pairs)<=1))
    return decisions

def execute(stage):
    plan=load();previous=committed(RESULT);base.need(previous['stage']=={'build':'prepared','check':'build','measure':'check'}[stage] and not previous['active'] and previous['ownership_closed'],'preceding closed stage required')
    if stage!='build':base.need(previous['passed'] and previous['complete'],'preceding stage failed')
    path=ROOT/(stage+'.json');base.need(not path.exists(),'single-use stage;inspect original owner,never retry')
    r=dict(stage=stage,source_commit=plan['source']['revision'],plan_sha256=base.digest(PLAN),passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,checks=[])
    own=None
    def persist():r['active']=bool(own and own.active is not None);write(path,dict(r,commands=own.commands if own else []))
    own=owner.Owner(ROOT/'owned',persist)
    def run(label,argv,wall):
        entry,out,err=own.run(stage+'-'+label,argv,plan['environment'],P,wall=wall,cpu=2400,rss_cap=3<<30,text_cap=64<<20,total_cap=512<<20,file_cap=64<<20)
        base.need(owner.closed(entry) and entry['exit_code']==0,'owned child failed '+label)
        base.need(owner.artifact_bytes(ROOT)<=512<<20,'root byte budget');return out.read_text(),err.read_text()
    try:
        if stage=='build':
            run('configure',plan['configure'],120);run('compile',plan['build'],600);binary=ROOT/'build/cpu_iq2_indices';r['binary']=dict(path=str(binary),sha256=base.digest(binary),bytes=binary.stat().st_size);r['compile_commands_sha256']=base.digest(ROOT/'build/compile_commands.json')
            ldd,_=run('linkage',['/usr/bin/ldd',str(binary)],20);base.need(not any(k in ldd.lower() for k in ('sycl','level_zero','libze','cuda','amdhip')),'GPU runtime linked')
            nm,_=run('symbols',['/usr/bin/nm','-S','--defined-only','--demangle',str(binary)],20);r['lowering']={}
            for name in ('pair_dot','indices_pair_dot'):
                matches=re.findall(r'^([0-9a-f]+) ([0-9a-f]+) t .*::'+name+r'\(.*$',nm,re.M);base.need(len(matches)==1,'unique function '+name);start,size=(int(x,16) for x in matches[0]);asm,_=run(name,['/usr/bin/objdump','-d','--no-show-raw-insn',f'--start-address={start}',f'--stop-address={start+size}',str(binary)],20)
                instructions=[m.group(1) for m in re.finditer(r'^\s*[0-9a-f]+:\s+(.*)$',asm,re.M)]
                r['lowering'][name]=dict(start=start,bytes=size,gathers=sum(x.startswith('vpgatherdq ') for x in instructions),pdep=sum(x.startswith('pdep ') for x in instructions),calls=sum(x.startswith('call') for x in instructions),vector_stack_accesses=sum(bool(re.match(r'v\w+\s',x) and re.search(r'%[xy]mm\d+',x) and re.search(r'\(%r(?:sp|bp)\)',x)) for x in instructions))
            a,b=r['lowering']['pair_dot'],r['lowering']['indices_pair_dot'];base.need(a['gathers']==b['gathers']==0 and a['pdep']==0 and b['pdep']==32 and b['calls']==0 and b['vector_stack_accesses']==0,'prospective lowering screen failed')
        else:
            built=json.loads((ROOT/'build.json').read_text());r['binary']=built['binary'];base.need(built['passed'] and built['complete'] and built['ownership_closed'] and base.digest(r['binary']['path'])==r['binary']['sha256'],'exact qualified component build required')
            expected_hash=previous['checks'][0]['capture_sha256'] if stage=='measure' else None
            for i,arm in enumerate(ORDER if stage=='measure' else (0,1)):
                capture=ROOT/(stage+'-'+str(i)+'.bin');stdout,_=run(str(i),[r['binary']['path'],'measure' if stage=='measure' else 'check',str(ROOT/'inputs.txt'),str(capture),str(arm)],180 if stage=='measure' else 120)
                samples=parse([json.loads(s) for s in stdout.splitlines()],arm,'measure' if stage=='measure' else 'check',plan['machine']['physical_cpus'][0]);base.need(capture.stat().st_size==CAPTURE_BYTES,'capture extent');h=base.digest(capture);expected_hash=expected_hash or h;base.need(h==expected_hash,'cross-arm capture mismatch')
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
