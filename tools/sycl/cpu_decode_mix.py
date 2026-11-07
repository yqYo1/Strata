#!/usr/bin/env python3
"""CPU-only exact-source/current-work calibration; Main serializes all stages.

Main-authored, repository LICENSE. No GPU runtime or model-throughput claim.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys
import time

sys.dont_write_bytecode=True
import cpu_roofline as base
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as owner
import build_adaptive_payload_model as engine

P,B,A=gen.P,gen.B,gen.A
ROOT=B/'cpu-compact-iq3xxs-v1'
BASELINE_CALIBRATION='8547cb65d3119e9f86456db979332d82d7e6f870'
PLAN=A/'cpu-roofline-plan.json'
RESULT=A/'cpu-roofline-current.json'
RESOURCE=engine.RESULT
RESOURCE_KEY='adaptive_cpu_leaves_regular_code_v1'
NATIVE_ROOT=engine.ROOT
CPU_SOURCE=NATIVE_ROOT/'source'
COMPILER=Path('/opt/intel/oneapi/compiler/2026.2')
WORKERS=(1,2,3,5)
GROUPS=(6,28)
SCHEMA='strata-cpu-compact-iq3xxs-v1'
LEAF_ORDER=(False,True,True,False,False,True)
CPU_UNITS=('expert.cpp','pool.cpp','expert_layout.cpp','q2_avx2.cpp','native_expert.cpp',
           'iq_avx512.cpp','iq_avx2.cpp','iq_single_avx2.cpp','iq2_pair_avx2.cpp','kq_avx2.cpp')
CXX_FLAGS='-fsycl-id-queries-range=size_t -Wno-error=deprecated-declarations --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 -DSTRATA_CPU_SERVICE_PROFILE=1'

# Pin the controller/helpers here; recover exact native CPU sources separately.
# A later helper edit must not silently change the frozen calibration.
PREFIXES=('bench/sycl/cpu_decode_mix','tools/sycl','tests/core/test_cpu_decode_mix.py',
          'AGENTS.md','docs/ARTIFACT_RETENTION.md','bench/sycl/cpu_roofline/main.cpp')


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def observed_histograms(resource):
    """Counts are useful independently of the charged observer's cost verdict."""
    guards.require(resource['stage']=='closed' and resource['passed'] and resource['complete'] and
        resource['ownership_closed'] and not resource['active'] and len(resource['processes'])==6,
        'close all six current CPU-leaf processes first')
    values=[]
    for index,p in enumerate(resource['processes']):
        guards.require(p['index']==index and p['passed'] and p['complete'] and p['ownership_closed'] and
            p['native_exit_code']==0 and not p['fresh_faults'] and p['resources']['passed'] and
            p['resources']['leaf_enabled'] is LEAF_ORDER[index] and
            p['resources']['completed_requests']==5 and p['resources']['warmup_prefix']==4,
            'successful current native/work binding required')
        if p['resources']['leaf_enabled']:
            cells=p['resources']['long_request']['format_nt_jobs']
            table={}
            for cell in cells:
                pair=cell['gu'],cell['down'];nt=cell['nt'];jobs=cell['jobs']
                guards.require(pair in base.PAIRS and type(nt) is int and 1<=nt<=4 and
                    type(jobs) is int and jobs>=0 and (pair,nt) not in table,'invalid format/NT count')
                table[pair,nt]=jobs
            guards.require(len(table)==28 and sum(table.values())==
                p['resources']['long_request']['cpu_jobs'],'complete current work histogram')
            values.append(table)
    guards.require(len(values)==3 and values[0]==values[1]==values[2],
        'three matching measured CPU-job distributions required')
    guards.require(len({p['resources']['work_signature_sha256'] for p in resource['processes']})==1,
        'matching aggregate work for current calibration')
    return {pair:[values[0][pair,nt] for nt in range(1,5)] for pair in sorted(base.PAIRS)}


def qualification():
    native=json.loads((NATIVE_ROOT/'record.json').read_text())
    guards.require(native['passed'] and native['complete'] and native['ownership_closed'] and
        not native['active'],'closed native source/build required')
    _,values=engine.basis()
    hashes={n:hashlib.sha256(raw).hexdigest() for n,raw in values.items()}
    guards.require(hashes==native['native_sources'] and len(hashes)==568,
        'recover exact composed native source from Git')
    # The immutable snapshot,not today's public CPU files,is the compute source.
    selected={n:h for n,h in hashes.items() if
        n.startswith(('src/kernels/cpu/','include/strata/kernels/cpu/','third_party/ggml/')) or
        n in ('include/strata/core/progress.hpp','bench/sycl/cpu_roofline/CMakeLists.txt',
              'bench/sycl/cpu_roofline/main.cpp')}
    guards.require(len(selected)>=24 and all(base.digest(CPU_SOURCE/n)==h for n,h in selected.items()),
        'actual native CPU source changed')
    for n,h in native['controller_inputs'].items():
        guards.require(base.digest(P/n)==h,'original native construction helper changed: '+n)
    guards.require(base.digest(native['compiled_inputs']['path'])==native['compiled_inputs']['sha256'] and
        base.digest(native['compile_database']['path'])==native['compile_database']['sha256'],
        'original actual compiler evidence changed')
    database=json.loads(Path(native['compile_database']['path']).read_text())
    native_cpu=[u for u in database if Path(u['file']).is_relative_to(CPU_SOURCE/'src/kernels/cpu')]
    native_c=[u for u in database if u['file'].endswith('/ggml-cpu/arch/x86/quants.c')]
    guards.require(native_cpu and len(native_c)==1 and all(shlex.split(u['command'])[0]==
        str(COMPILER/'bin/icpx') for u in native_cpu) and
        shlex.split(native_c[0]['command'])[0]==str(COMPILER/'bin/icx'),
        'match actual native C and C++ compilers')
    for asset in native['binaries'].values():
        guards.require(base.digest(asset['path'])==asset['sha256'],'preserve actual native executable/fixtures')
    resource=gen.committed(RESOURCE)[RESOURCE_KEY]
    observed_histograms(resource)
    cohort=json.loads((B/'adaptive-cpu-leaves-regular-code-v1/record.json').read_text())
    model_signature=hashlib.sha256(json.dumps(cohort['model_assets'],sort_keys=True).encode()).hexdigest()
    guards.require(cohort['processes']==resource['processes'] and len(cohort['model_assets'])==15 and
        model_signature==resource['input_signatures']['model_assets'] and
        cohort['workload']=='regular-code' and cohort['selected_decode_budget']==96,
        'current profile/model binding changed')
    for path in (base.METADATA,*(Path(r['shard']) for r in base.formats())):
        identity=base.identity(path);identity['realpath']=identity.pop('path')
        guards.require(str(path) in cohort['model_assets'] and
            identity==cohort['model_assets'][str(path)]['identity'],'current model identity changed')
    guards.require(base.digest(base.METADATA)==cohort['model_assets'][str(base.METADATA)]['sha256'],
        'profile expert metadata changed')
    guards.require(resource['binary']==native['binaries']['strata'] and
        resource['native_source_commit']==native['source_commit'],
        'current profile differs from the native generation')
    proof=dict(native_source=native['source_commit'],native_record_sha256=base.digest(NATIVE_ROOT/'record.json'),
        recovered_native_files=len(hashes),native_source_sha256=base.fingerprint(hashes),
        CPU_sources=selected,native_binary=native['binaries']['strata'],
        compiled_inputs=native['compiled_inputs'],native_compile_database=native['compile_database'],
        toolchain_image=native['toolchain_image'],external_inputs=native['external_inputs'],
        resource_key=RESOURCE_KEY,resource_source_commit=resource['source_commit'],
        resource_model_assets_sha256=model_signature,
        resource_work_signature=resource['processes'][0]['resources']['work_signature_sha256'],
        resource_histograms_sha256=base.fingerprint(resource['processes'][1]['resources']['long_request']['format_nt_jobs']),
        observer_cost_verdict=resource['decision'])
    return resource,proof


def environment():
    return dict(PATH=str(COMPILER/'bin')+':/usr/bin:/bin',
        LD_LIBRARY_PATH=str(COMPILER/'lib'),HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',
        OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',STRATA_IQ2_PAIR='1',
        STRATA_IQ3_PAIR='1',STRATA_IQ3_PAIR_AUDIT='1',STRATA_IQ_SINGLE_EXACT='0',
        STRATA_IQ2_PAIR_AUDIT='0',STRATA_CPU_SERVICE='0',STRATA_CPU_WORK_CENSUS='0',
        STRATA_POOL_SPIN_US='20000')


def matrix():
    return {(gu,dn,w,g) for gu,dn in base.PAIRS for w in WORKERS for g in GROUPS}


def validate_order(order):
    guards.require(type(order) is list and len(order)==len(matrix()) and
        all(isinstance(row,(list,tuple)) and len(row)==4 and all(type(v) is int for v in row)
            for row in order) and set(map(tuple,order))==matrix(),'full unique typed calibration matrix required')




def calibration_origin():
    data=gen.git('show',BASELINE_CALIBRATION+':'+str(RESULT.relative_to(P)))
    value=json.loads(data)
    guards.require(value['passed'] and value['complete'] and value['ownership_closed'] and
        not value['active'] and value['decision']['calibration_complete'] and
        value['artifact_retirement']['complete'],'closed current-source calibration required')
    return dict(commit=BASELINE_CALIBRATION,path=str(RESULT.relative_to(P)),
        sha256=hashlib.sha256(data).hexdigest())


def prepare():
    guards.require(not ROOT.exists(),'fresh single-use compact-view root required')
    old=gen.committed(RESULT);origin=calibration_origin()
    guards.require(base.digest(RESULT)==origin['sha256'],'archive the closed calibration before this prototype')
    resource,proof=qualification();observed=observed_histograms(resource)
    rows=[r for r in base.formats() if r['gu']==18]
    guards.require({(r['gu'],r['down']) for r in rows}=={(18,20),(18,42)},'two current IQ3_XXS model formats')
    for row in rows:
        row['experts']=28*math.ceil((128<<20)/(28*row['blob_bytes']))
        row['observed_jobs_NT']=observed[row['gu'],row['down']]
        row['fixture_sha256']=base.fixture_hash(row)
    revision=gen.git('rev-parse','HEAD').decode().strip()
    plan=dict(schema=SCHEMA,source=base.source_pin(P,revision,PREFIXES),
        ggml=base.source_pin(base.GGML,base.GGML_REV,('ggml',)),qualification=proof,
        calibration=origin,environment=environment(),machine=base.machine(),formats=rows,
        metadata_sha256=base.digest(base.METADATA),model_identity={r['shard']:base.identity(r['shard']) for r in rows},
        tools={str(p):base.digest(p) for p in (COMPILER/'bin/icx',COMPILER/'bin/icpx','/usr/bin/cmake','/usr/bin/ninja')},
        configure=['/usr/bin/cmake','-S',str(P/'bench/sycl/cpu_decode_mix'),'-B',str(ROOT/'build'),
            '-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_CXX_EXTENSIONS=OFF',
            '-DCMAKE_C_COMPILER='+str(COMPILER/'bin/icx'),'-DCMAKE_CXX_COMPILER='+str(COMPILER/'bin/icpx'),
            '-DCMAKE_CXX_FLAGS='+CXX_FLAGS,'-DCMAKE_C_FLAGS=-DSTRATA_CPU_SERVICE_PROFILE=1',
            '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t','-DSTRATA_GGML_DIR='+str(base.GGML),
            '-DSTRATA_CPU_SOURCE_ROOT='+str(CPU_SOURCE)],
        build=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target','cpu_decode_mix','--parallel','4'],
        protocol=dict(question='Does the exact134B IQ3_XXS compact GU view preserve native integer lanes,FP32/FMA/reduction behavior and bounds before performance comparison?',
            order=['build','controls','check18/20','check18/42'],
            controls='All256grid/128sign/16scale base combinations,134217728 encoded values,524288 actual SIMD integer chunks;16 synthetic full GU cases with zero/subnormal/max-finite/negative-zero half scales and signed/zero activation scales/-128 lanes;64 partial ranges,10 GU and2 encoder refusals.',
            real='Both current IQ3_XXS model-format layers,all distinct loaded experts at four quantized NT1 activations. Compare the actual native paired entry directly to compact GU;this isolates that component,not full pool/model dispatch.',
            numerical='Require exact finite native-pair outputs,identical integer lanes,unaltered inactive rows and guards. Preserve raw98B blocks and allocate full134B views;no changed accumulation or model quantization.',
            gate='Commit each stage before the next. Require all controls and both model children,normal0,closed owners. Stop first failure/budget;no retry/replacement/exclusion.',
            performance='No timing/adoption claim in this correctness stage. Next measure setup/full raw-plus-view footprint,repeated matched reuse/stream and1/6-thread/current NT work before production selection.',
            retention='Retain compact numeric/guard outcomes,source/build/environment identity. Keep this single auxiliary ELF only for the immediately following component timing preparation/qualification,then retire successful raw/build output after source recovery. Preserve the two inference generations.'),
        budgets=dict(build_seconds=600,controls_seconds=180,check_child_seconds=120,stage_seconds=1800,
            rss_bytes=3<<30,text_bytes=8<<20,root_bytes=256<<20))
    ROOT.mkdir(mode=0o700);write(PLAN,plan)
    write(RESULT,dict(schema=SCHEMA,stage='prepared',source_commit=revision,plan_sha256=base.digest(PLAN),
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,
        calibration_origin=origin,hardware_saturation_qualified=False,model_adoption_qualified=False))
    print('Prepared exact compact-view correctness plan;commit before build.',flush=True)


def load():
    plan=gen.committed(PLAN)
    guards.require(plan['schema']==SCHEMA and
        base.source_pin(P,plan['source']['revision'],PREFIXES)==plan['source'] and
        base.source_pin(base.GGML,base.GGML_REV,('ggml',))==plan['ggml'] and
        plan['calibration']==calibration_origin() and environment()==plan['environment'] and
        base.machine()==plan['machine'],'source/environment/platform/calibration drift')
    for path,sha in plan['tools'].items():guards.require(base.digest(path)==sha,'compiler/tool drift')
    guards.require(base.digest(base.METADATA)==plan['metadata_sha256'],'format metadata drift')
    resource,proof=qualification();guards.require(proof==plan['qualification'],'native/resource proof drift')
    observed=observed_histograms(resource)
    original={(r['gu'],r['down']):r for r in base.formats() if r['gu']==18}
    guards.require(len(plan['formats'])==2 and {(r['gu'],r['down']) for r in plan['formats']}==set(original),
        'complete unique compact model cases')
    for row in plan['formats']:
        key=row['gu'],row['down']
        guards.require(all(row[k]==v for k,v in original[key].items() if k!='experts') and
            row['experts']==28*math.ceil((128<<20)/(28*row['blob_bytes'])) and
            row['observed_jobs_NT']==observed[key] and
            base.identity(row['shard'])==plan['model_identity'][row['shard']] and
            base.fixture_hash(row)==row['fixture_sha256'],'compact fixture/work drift')
    return plan


def command(row,workers,group,stage):
    return [str(ROOT/'build/cpu_decode_mix'),row['shard'],str(row['gu']),str(row['down']),
        str(row['gate']),str(row['up']),str(row['down_offset']),str(row['blob_bytes']),str(workers),str(group),
        stage,*map(str,row['observed_jobs_NT']),'20261022']


def expected_histogram(hist,group):
    guards.require(isinstance(hist,(list,tuple)) and len(hist)==4 and
        all(type(v) is int and 0<=v<=1000000000 for v in hist) and sum(hist)>0 and
        type(group) is int and group in GROUPS,'positive integer NT histogram/known group required')
    total=sum(hist);n=100*group
    result=[n*v//total for v in hist]
    order=sorted(range(4),key=lambda i:(-(n*hist[i]%total),i))
    for i in order[:n-sum(result)]:result[i]+=1
    return result


def parse(lines,row,workers,group,stage):
    guards.require(type(workers) is int and workers in WORKERS and type(group) is int and
        group in GROUPS and stage in ('check','measure'),'planned worker/group/stage required')
    guards.require(lines and lines[-1]==dict(kind='closed',passed=True),'missing native completion')
    placement=[r for r in lines if r['kind']=='placement']
    guards.require(placement==[dict(kind='placement',workers=workers,host_cpu=base.machine()['physical_cpus'][0],
        worker_cpus=base.machine()['physical_cpus'][1:workers+1],actual_affinity_verified=True)],'actual pool placement differs')
    work=[r for r in lines if r['kind']=='work'];valid=[r for r in lines if r['kind']=='validation']
    count=group*math.ceil((128<<20)/(group*row['blob_bytes']))
    guards.require(work==[dict(kind='work',workers=workers,host_works=workers==5,group=group,batches=100,
        unique_stream_experts=count,blob_bytes=row['blob_bytes'],jobs_NT=expected_histogram(row['observed_jobs_NT'],group),phase='decode')],
        'actual calibration work/dispatch differs')
    want=dict(kind='validation',distinct_experts=count,NT_mask=30,pool_serial_bits_equal=True,mixed_NT_bits_equal=True,
        inactive_outputs_untouched=True,canaries=True)
    guards.require(valid==[want]*(3 if stage=='measure' else 1),'numerical/guard checks incomplete')
    samples=[r for r in lines if r['kind']=='sample'];timers=[r for r in lines if r['kind']=='timer']
    guards.require(len(lines)==3+len(valid)+len(samples)+len(timers),'unknown/extra records')
    if stage=='check':guards.require(not samples and not timers,'timed work in check stage');return dict(work=work[0],validated=True)
    guards.require(len(timers)==2 and [r['stream'] for r in timers]==[False,True] and
        all(r['fastest_pass_s']>0 and 0<=r['query_s']/r['fastest_pass_s']<.001 for r in timers),'timer gate')
    guards.require([(r['stream'],r['rep']) for r in samples]==[(stream,i) for stream in (False,True) for i in range(5)],'five samples per cell required')
    for s in samples:
        guards.require(type(s['passes']) is int and 0<s['passes']<1000000 and
            all(type(s[k]) in (int,float) and math.isfinite(s[k]) and s[k]>=0 for k in
                ('wall_s','cpu_s','gu_s','quant_s','down_s')) and s['wall_s']>=1 and
            s['gu_s']+s['quant_s']+s['down_s']<=s['wall_s']+.0001,'invalid sample/timing conservation')
    return dict(work=work[0],validated=True,timers=timers,samples=samples)



def parse_compact(lines,row=None):
    value=(dict(kind='compact_controls',codec_cases=524288,integer_chunks=524288,
        encoded_values=134217728,FMA_cases=16,partial_ranges=64,GU_refusals=10,
        encode_refusals=2,finite_canaries=True) if row is None else
        dict(kind='compact_validation',format=18,nt=1,activation_cases=4,experts=row['experts'],
            native_pair_bits_equal=True,finite=True,canaries=True,compact_GU_bytes_per_expert=1715200,
            raw_plus_compact_bytes_per_expert=row['blob_bytes']+1715200))
    expected=[value,dict(kind='closed',passed=True)]
    guards.require(json.dumps(lines,sort_keys=True)==json.dumps(expected,sort_keys=True),
        'complete typed compact numerical/guard evidence required')
    return value


def allowed_inputs(plan):
    """Resolve the original toolchain inventory and exact prospective inputs."""
    proof=plan['qualification'];allowed={}
    for key in ('compiled_inputs','external_inputs'):
        asset=proof[key];guards.require(base.digest(asset['path'])==asset['sha256'],'native input inventory drift')
        for path,sha in json.loads(Path(asset['path']).read_text()).items():
            if isinstance(sha,str):allowed[str(Path(path).resolve())]=sha
    asset=proof['toolchain_image']
    guards.require(base.digest(asset['path'])==asset['sha256'],'original toolchain inventory drift')
    image=json.loads(Path(asset['path']).read_text())
    for item in image['files'].values():
        if item['kind']=='file':allowed[item['realpath']]=item['sha256']
    for name,sha in proof['CPU_sources'].items():allowed[str((CPU_SOURCE/name).resolve())]=sha
    for name in ('bench/sycl/cpu_decode_mix/main.cpp','bench/sycl/cpu_decode_mix/compact_iq3xxs.hpp'):
        path=P/name;allowed[str(path.resolve())]=base.digest(path)
    native=json.loads((NATIVE_ROOT/'record.json').read_text())
    original=NATIVE_ROOT/'build/ggml/src/ggml-version.h'
    guards.require(str(original) in native['generated_expected'] and
        base.digest(original)==native['generated_expected'][str(original)],'native GGML version changed')
    generated=ROOT/'build/reference/ggml/src/ggml-version.h'
    allowed[str(generated)]=native['generated_expected'][str(original)]
    return allowed


def attest_build(plan,output):
    allowed=allowed_inputs(plan);build=ROOT/'build'
    paths={str((Path(line.strip()) if Path(line.strip()).is_absolute() else build/line.strip()).resolve())
        for line in output.splitlines() if line.startswith('    ') and line.strip()}
    required={str((CPU_SOURCE/'src/kernels/cpu'/n).resolve()) for n in CPU_UNITS}
    required.add(str((P/'bench/sycl/cpu_decode_mix/main.cpp').resolve()))
    # Headers need not all compile in this target,but every actual input must be pinned.
    guards.require(required<=paths and len(paths)>100 and
        all(allowed.get(path)==base.digest(path) for path in paths),
        'unplanned or incomplete actual CPU compiler inputs')
    db=build/'compile_commands.json';units=json.loads(db.read_text())
    guards.require({str(Path(u['file']).resolve()) for u in units if str(Path(u['file']).resolve()) in required}==required and
        all(sum(str(Path(u['file']).resolve())==path for u in units)==1 for path in required),
        'one actual compile command per required CPU producer')
    for unit in units:
        argv=shlex.split(unit['command']);file=Path(unit['file']).resolve()
        guards.require(argv[0]==str(COMPILER/'bin'/('icx' if file.suffix=='.c' else 'icpx')) and
            '-O3' in argv and '-DNDEBUG' in argv and
            not any(flag in argv for flag in ('-ffast-math','-Ofast','-ffp-contract=fast')),
            'unexpected CPU compiler or math flags')
        if str(file) in required:
            guards.require('-std=c++20' in argv and '-march=znver3' in argv and
                '-mtune=znver3' in argv and '-DSTRATA_CPU_SERVICE_PROFILE=1' in argv,
                'actual production CPU layout/ISA flags differ')
        if '/ggml-cpu/' in str(file) and file.name=='quants.c':
            guards.require('-march=native' in argv and '-fno-associative-math' in argv,
                'actual GGML C arithmetic flags differ')
    # CMake's -Wl,-t records the actual objects/archives/runtime link inputs.
    external={}
    text=(ROOT/'owned/build-compile.stdout').read_text()
    for line in text.splitlines():
        candidate=Path(line.strip())
        if not candidate.is_absolute() or not candidate.is_file():continue
        path=candidate.resolve()
        if path.is_relative_to(build):continue
        guards.require(allowed.get(str(path))==base.digest(path),'unplanned link input: '+str(path))
        external[str(path)]=allowed[str(path)]
    guards.require(external,'actual external linker input trace missing')
    write(ROOT/'compiler-inputs.json',{path:base.digest(path) for path in sorted(paths)})
    return dict(compile_commands_sha256=base.digest(db),
        compiled_inputs=dict(path=str(ROOT/'compiler-inputs.json'),files=len(paths),
            sha256=base.digest(ROOT/'compiler-inputs.json'),uncovered=0),
        actual_link_inputs=external)


def verify_built(plan,built):
    guards.require(built['passed'] and built['complete'] and built['ownership_closed'] and
        not built['active'] and built['plan_sha256']==base.digest(PLAN),'closed exact build required')
    for asset in built['binaries'].values():guards.require(base.digest(asset['path'])==asset['sha256'],'binary drift')
    guards.require(base.digest(ROOT/'build/compile_commands.json')==built['compile_commands_sha256'],
        'actual compiler commands changed')
    asset=built['compiled_inputs']
    guards.require(base.digest(asset['path'])==asset['sha256'],'actual input list changed')
    paths=json.loads(Path(asset['path']).read_text());allowed=allowed_inputs(plan)
    guards.require(len(paths)==asset['files'] and all(allowed.get(path)==sha==base.digest(path)
        for path,sha in paths.items()),'actual compiler input changed')
    guards.require(all(allowed.get(path)==sha==base.digest(path) for path,sha in
        built['actual_link_inputs'].items()),'actual linker input changed')


def compact_record(record,commands):
    result={k:v for k,v in record.items() if k!='commands'}
    result['owned_commands']=dict(count=len(commands),
        all_closed=all(owner.closed(c) for c in commands),
        all_exit_zero=all(c.get('exit_code')==0 for c in commands))
    result['hardware_saturation_qualified']=False
    result['model_adoption_qualified']=False
    return result



def run(stage):
    started=time.monotonic();plan=load();gen.committed(RESULT)
    path=ROOT/(stage+'.json');guards.require(not path.exists(),'single-use stage;inspect owner,never retry')
    record=dict(schema=SCHEMA,stage=stage,source_commit=plan['source']['revision'],
        plan_sha256=base.digest(PLAN),calibration_origin=plan['calibration'],
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,children=[])
    own=None;deadline=started+plan['budgets']['stage_seconds']
    with path.open('x') as marker:marker.write(json.dumps(record,indent=2)+'\n')
    def persist():
        record['active']=bool(own and own.active is not None)
        write(path,dict(record,commands=own.commands if own else []))
    def execute(label,argv,wall):
        guards.require(time.monotonic()<deadline,'whole-stage budget expired')
        entry,_,_=own.run(stage+'-'+label,argv,plan['environment'],P,
            wall=min(wall,deadline-time.monotonic()),cpu=1200,rss_cap=plan['budgets']['rss_bytes'],
            text_cap=plan['budgets']['text_bytes'],total_cap=plan['budgets']['root_bytes'],
            file_cap=plan['budgets']['root_bytes'])
        guards.require(owner.closed(entry) and entry['exit_code']==0,'owned child failed: '+label)
        return entry
    try:
        own=owner.Owner(ROOT/'owned',persist)
        if stage=='build':
            execute('configure',plan['configure'],120)
            execute('compile',plan['build'],plan['budgets']['build_seconds'])
            execute('dependencies',['/usr/bin/ninja','-C',str(ROOT/'build'),'-t','deps'],40)
            record.update(attest_build(plan,(ROOT/'owned/build-dependencies.stdout').read_text()))
            binary=ROOT/'build/cpu_decode_mix'
            record['binaries']=dict(mix=dict(path=str(binary),sha256=base.digest(binary)))
            execute('link',['/usr/bin/ldd',str(binary)],20)
            text=(ROOT/'owned/build-link.stdout').read_text()
            guards.require(not re.search('sycl|level_zero|cuda|amdhip|libze|not found',text,re.I),
                'GPU runtime linked or missing library')
        else:
            built=json.loads((ROOT/'build.json').read_text());verify_built(plan,built)
            record['binaries']=built['binaries']
            if stage=='controls':cases=[('synthetic',[built['binaries']['mix']['path'],'--compact-controls'],None)]
            else:
                control=json.loads((ROOT/'controls.json').read_text())
                guards.require(control['passed'] and control['complete'] and control['ownership_closed'] and
                    not control['active'] and control['plan_sha256']==base.digest(PLAN) and
                    control['binaries']==built['binaries'] and len(control['children'])==1,
                    'closed compact controls required')
                parse_compact([control['children'][0]['validation'],dict(kind='closed',passed=True)])
                cases=[('format-'+str(row['down']),command(row,1,28,'compact-check'),row) for row in plan['formats']]
            for label,argv,row in cases:
                entry=execute(label,argv,plan['budgets']['controls_seconds' if row is None else 'check_child_seconds'])
                stdout=ROOT/'owned'/(stage+'-'+label+'.stdout')
                guards.require((ROOT/'owned'/(stage+'-'+label+'.stderr')).read_bytes()==b'','unexpected compact diagnostic')
                value=parse_compact([json.loads(s) for s in stdout.read_text().splitlines()],row)
                record['children'].append(dict(validation=value,exit_code=entry['exit_code'],stdout_sha256=base.digest(stdout)))
                persist();print('Closed compact '+label,flush=True)
        load()
        if stage!='build':verify_built(plan,built)
        guards.require(time.monotonic()<=deadline,'whole-stage budget expired')
        record.update(passed=True,complete=True)
    except BaseException as error:
        record['error']=repr(error)
    finally:
        record['ownership_closed']=own is None or (own.active is None and all(owner.closed(c) for c in own.commands))
        record['passed']=bool(record['passed'] and record['ownership_closed'])
        persist();write(RESULT,compact_record(record,own.commands if own else []))
    print(json.dumps({k:record.get(k) for k in ('stage','passed','complete','ownership_closed','error')}),flush=True)
    return 0 if record['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','build','controls','check'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        # This bounded CPU build never overwrites the retained engine generation.
        return prepare() if args.stage=='prepare' else run(args.stage)


if __name__=='__main__':sys.exit(main())
