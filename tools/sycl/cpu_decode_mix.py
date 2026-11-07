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
import random
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
ROOT=B/'cpu-decode-mix-v3'
PLAN=A/'cpu-roofline-plan.json'
RESULT=A/'cpu-roofline-current.json'
RESOURCE=engine.RESULT
RESOURCE_KEY='adaptive_cpu_leaves_regular_code_v1'
NATIVE_ROOT=engine.ROOT
CPU_SOURCE=NATIVE_ROOT/'source'
COMPILER=Path('/opt/intel/oneapi/compiler/2026.2')
WORKERS=(1,2,3,5)
GROUPS=(6,28)
SCANS=(1,2,3,6)
SCHEMA='strata-cpu-decode-mix-v3'
LEAF_ORDER=(False,True,True,False,False,True)
CPU_UNITS=('expert.cpp','pool.cpp','expert_layout.cpp','q2_avx2.cpp','native_expert.cpp',
           'iq_avx512.cpp','iq_avx2.cpp','iq_single_avx2.cpp','iq2_pair_avx2.cpp','kq_avx2.cpp')
CXX_FLAGS='-fsycl-id-queries-range=size_t -Wno-error=deprecated-declarations --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 -DSTRATA_CPU_SERVICE_PROFILE=1'

# Pin the controller/helpers here; recover exact native CPU sources separately.
# A later helper edit must not silently change the frozen calibration.
PREFIXES=('bench/sycl/cpu_decode_mix','tools/sycl','tests/core/test_cpu_decode_mix.py',
          'AGENTS.md','docs/ARTIFACT_RETENTION.md')


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


def previous_fixture_retired(old):
    # Legacy single-binary receipts predate the `complete` retention key.
    # Verify every executable against the actual manifest,never rewrite receipts.
    retired=old['artifact_retirement']
    guards.require(retired['original_recipes_recovered'] and
        base.digest(retired['manifest'])==retired['manifest_sha256'],'original calibration retirement evidence differs')
    rows=[json.loads(s) for s in Path(retired['manifest']).read_text().splitlines()]
    binaries=list(old['binaries'].values()) if 'binaries' in old else [old['binary']]
    guards.require(binaries and len({b['path'] for b in binaries})==len(binaries),
        'original calibration executable identities missing or duplicated')
    for binary in binaries:
        matched=[r for r in rows if r['path']==binary['path']]
        guards.require(len(matched)==1 and matched[0]['sha256']==binary['sha256'] and
            not Path(binary['path']).exists(),'original calibration executable not retired')


def matrix():
    return {(gu,dn,w,g) for gu,dn in base.PAIRS for w in WORKERS for g in GROUPS}


def validate_order(order):
    guards.require(type(order) is list and len(order)==len(matrix()) and
        all(isinstance(row,(list,tuple)) and len(row)==4 and all(type(v) is int for v in row)
            for row in order) and set(map(tuple,order))==matrix(),'full unique typed calibration matrix required')


def prepare():
    guards.require(not ROOT.exists(),'fresh single-use calibration root required')
    old=gen.committed(RESULT)
    guards.require(old['complete'] and old['passed'] and old['ownership_closed'] and not old['active'],
        'close earlier calibration first')
    previous_fixture_retired(old)
    resource,proof=qualification();observed=observed_histograms(resource);rows=base.formats()
    for row in rows:
        row['observed_jobs_NT']=observed[row['gu'],row['down']]
        row['experts']=max(g*math.ceil((128<<20)/(g*row['blob_bytes'])) for g in GROUPS)
        row['fixture_sha256']=base.fixture_hash(row)
    revision=gen.git('rev-parse','HEAD').decode().strip()
    order=sorted(matrix());random.Random(20261107).shuffle(order)
    configure=['/usr/bin/cmake','-S',str(P/'bench/sycl/cpu_decode_mix'),'-B',str(ROOT/'build'),
        '-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
        '-DCMAKE_CXX_EXTENSIONS=OFF',
        '-DCMAKE_C_COMPILER='+str(COMPILER/'bin/icx'),
        '-DCMAKE_CXX_COMPILER='+str(COMPILER/'bin/icpx'),
        '-DCMAKE_CXX_FLAGS='+CXX_FLAGS,'-DCMAKE_C_FLAGS=-DSTRATA_CPU_SERVICE_PROFILE=1',
        '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t',
        '-DSTRATA_GGML_DIR='+str(base.GGML),'-DSTRATA_CPU_SOURCE_ROOT='+str(CPU_SOURCE)]
    plan=dict(schema=SCHEMA,source=base.source_pin(P,revision,PREFIXES),
        ggml=base.source_pin(base.GGML,base.GGML_REV,('ggml',)),qualification=proof,
        environment=environment(),machine=base.machine(),formats=rows,order=order,
        tools={str(p):base.digest(p) for p in
            (COMPILER/'bin/icx',COMPILER/'bin/icpx','/usr/bin/cmake','/usr/bin/ninja')},
        metadata_sha256=base.digest(base.METADATA),model_identity={r['shard']:base.identity(r['shard']) for r in rows},
        previous_result=dict(commit=gen.git('log','-1','--format=%H','--',str(RESULT.relative_to(P))).decode().strip(),
            path=str(RESULT.relative_to(P)),sha256=base.digest(RESULT)),
        configure=configure,build=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target',
            'cpu_decode_mix','cpu_roofline','--parallel','4'],
        protocol=dict(question='How does actual e1b011 CPU expert work respond to compute-thread count and working-set reuse at the current adaptive96 NT/format distribution?',
            groups=list(GROUPS),workers=list(WORKERS),computed_threads=[1,2,3,6],scan_threads=list(SCANS),
            batches_per_pass=100,repetitions=5,sample_floor_seconds=1.0,target_seconds=1.3,timer_fraction_cap=.001,
            NT='Three matching leaf samples provide per-format NT1..4 job counts. Largest-remainder rounding to100*group jobs and a fixed shuffled schedule approximate those marginals,not the full layer/activation/access trace.',
            reuse='Same distinct group of6 or28 real expert blobs;28 weights can exceed32MiB L3.',
            stream='Rotate distinct groups through at least128MiB.100 batches/pass can revisit the extent. Requested blob visits are not measured physical RAM traffic.',
            scan='Read-only256MiB persistent1/2/3/6-thread checksum reference;five>=1s samples. Achieved reference,not a DRAM or arithmetic ceiling.',
            validation='All loaded distinct experts at NT1..4 and heterogeneous-NT timed-size groups:production pool bit-exact to serial production GU/quant/down,finite outputs,untouched inactive lanes and external canaries. This is route consistency,not independent task quality.',
            observers='Same native IQ2/IQ3 pairs and IQ3 audit,CPU service/census off,20ms pool spin. Profile-enabled CPU layout and actual icx/icpx2026.2 sources;no GPU runtime.',
            gates='All56 check children before timing;all112 expert cells and4 read references with five individual eligible samples,actual affinity/work/histogram conservation and normal closed owners. Stop first failure/budget;no retries,replacement or exclusions.',
            interpretation='Thread and reuse sensitivity guide CPU optimization. Pool waiting,quantization,scheduling and memory effects remain mixed. Neither concurrent CPU time nor requested bytes/read reference proves physical traffic,saturation,critical path,model speed or70tok/s.',
            retention='Keep compact individual timing/work/validation samples and exact provenance. Retire successful raw/compiler output and this auxiliary CPU build after the calibration decision and original-source recovery. Preserve the two current engine generations.'),
        retained_generation='Auxiliary CPU-only calibration;does not overwrite either of the two current engine/fixture generations. Retire it at calibration closure.',
        budgets=dict(build_seconds=600,check_child_seconds=120,measure_child_seconds=180,scan_seconds=40,
                     stage_seconds=3600,rss_bytes=3<<30,text_bytes=8<<20,root_bytes=256<<20))
    ROOT.mkdir(mode=0o700);write(PLAN,plan)
    write(RESULT,dict(schema=SCHEMA,stage='prepared',source_commit=revision,
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,
        plan_sha256=base.digest(PLAN),hardware_saturation_qualified=False,model_adoption_qualified=False))
    print('Prepared56-child current CPU calibration;commit exact plan before build.',flush=True)


def load():
    plan=gen.committed(PLAN)
    guards.require(plan['schema']==SCHEMA and
        base.source_pin(P,plan['source']['revision'],PREFIXES)==plan['source'] and
        base.source_pin(base.GGML,base.GGML_REV,('ggml',))==plan['ggml'] and
        environment()==plan['environment'] and base.machine()==plan['machine'],'source/environment/platform drift')
    for p,h in plan['tools'].items():guards.require(base.digest(p)==h,'compiler/tool drift')
    guards.require(base.digest(base.METADATA)==plan['metadata_sha256'],'format metadata drift')
    resource,proof=qualification();guards.require(proof==plan['qualification'],'original native/resource proof changed')
    validate_order(plan['order'])
    observed=observed_histograms(resource)
    original={(r['gu'],r['down']):r for r in base.formats()}
    guards.require(len(plan['formats'])==len(original) and
        {(r['gu'],r['down']) for r in plan['formats']}==set(original),
        'complete unique current format inventory required')
    for row in plan['formats']:
        key=row['gu'],row['down']
        guards.require(all(row[k]==v for k,v in original[key].items() if k!='experts') and
            row['observed_jobs_NT']==observed[key] and
            row['experts']==max(g*math.ceil((128<<20)/(g*row['blob_bytes'])) for g in GROUPS),
            'format/offset/observed histogram differs')
        guards.require(base.identity(row['shard'])==plan['model_identity'][row['shard']] and
            base.fixture_hash(row)==row['fixture_sha256'],'selected expert input drift')
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
    for name in ('CMakeLists.txt','main.cpp'):
        path=P/'bench/sycl/cpu_decode_mix'/name;allowed[str(path.resolve())]=base.digest(path)
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
    required|={str((P/'bench/sycl/cpu_decode_mix/main.cpp').resolve()),
        str((CPU_SOURCE/'bench/sycl/cpu_roofline/main.cpp').resolve())}
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
    path=ROOT/(stage+'.json');guards.require(not path.exists(),'single-use stage;inspect existing owner,never retry')
    record=dict(schema=SCHEMA,stage=stage,source_commit=plan['source']['revision'],
        plan_sha256=base.digest(PLAN),passed=False,complete=False,active=False,
        ownership_closed=True,GPU_executed=False,children=[])
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
            record['binaries']={n:dict(path=str(ROOT/'build'/p),sha256=base.digest(ROOT/'build'/p))
                for n,p in (('mix','cpu_decode_mix'),('scan','reference/cpu_roofline'))}
            for n,asset in record['binaries'].items():
                execute('link-'+n,['/usr/bin/ldd',asset['path']],20)
                text=(ROOT/'owned'/('build-link-'+n+'.stdout')).read_text()
                guards.require(not re.search('sycl|level_zero|cuda|amdhip|libze',text,re.I),'GPU library linked')
        else:
            built=json.loads((ROOT/'build.json').read_text());verify_built(plan,built)
            record['binaries']=built['binaries']
            if stage=='measure':
                checked=json.loads((ROOT/'check.json').read_text())
                guards.require(checked['passed'] and checked['complete'] and checked['ownership_closed'] and
                    not checked['active'] and checked['binaries']==built['binaries'] and
                    checked['plan_sha256']==base.digest(PLAN) and len(checked['children'])==len(matrix()) and
                    {(tuple(c['pair']),c['workers'],c['group']) for c in checked['children']}==
                        {(pair,w,g) for pair in base.PAIRS for w in WORKERS for g in GROUPS},
                    'all56 numerical cases before timing')
            lookup={(r['gu'],r['down']):r for r in plan['formats']}
            for i,(gu,dn,w,g) in enumerate(plan['order']):
                row=lookup[gu,dn];label='cell-'+str(i)
                entry=execute(label,command(row,w,g,stage),plan['budgets'][stage+'_child_seconds'])
                stdout=ROOT/'owned'/(stage+'-'+label+'.stdout')
                values=parse([json.loads(s) for s in stdout.read_text().splitlines()],row,w,g,stage)
                record['children'].append(dict(pair=[gu,dn],workers=w,group=g,**values,
                    exit_code=entry['exit_code'],stdout_sha256=base.digest(stdout)))
                persist();print('Closed '+stage+' '+str(i+1)+'/'+str(len(plan['order'])),flush=True)
            if stage=='measure':
                for n in SCANS:
                    label='scan-'+str(n)
                    execute(label,[built['binaries']['scan']['path'],'scan',
                        ','.join(map(str,plan['machine']['physical_cpus'][:n]))],
                        plan['budgets']['scan_seconds'])
                    lines=[json.loads(s) for s in (ROOT/'owned'/('measure-'+label+'.stdout')).read_text().splitlines()]
                    guards.require(lines[-1]==dict(kind='closed',passed=True),'read reference incomplete')
                    samples=[s for s in lines if s['kind']=='sample']
                    guards.require(len(samples)==5 and [s['rep'] for s in samples]==list(range(5)) and
                        all(s['threads']==n and math.isfinite(s['wall_s']) and s['wall_s']>=1 for s in samples),
                        'read reference samples')
                    record.setdefault('read_reference',[]).append(dict(threads=n,samples=samples))
        # Recheck the whole frozen plan,actual input bytes and owners outside timing.
        load()
        if stage!='build':verify_built(plan,built)
        guards.require(time.monotonic()<=deadline,'whole-stage budget expired during final checks')
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','build','check','measure'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        # This bounded CPU build never overwrites the retained engine generation.
        return prepare() if args.stage=='prepare' else run(args.stage)


if __name__=='__main__':sys.exit(main())
