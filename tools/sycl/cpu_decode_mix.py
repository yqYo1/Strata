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
import statistics
import sys
import time

sys.dont_write_bytecode=True
import cpu_roofline as base
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as owner
import build_adaptive_payload_model as engine

P,B,A=gen.P,gen.B,gen.A
ROOT=B/'cpu-packed-iq3xxs-gather-v1'
BASELINE_CALIBRATION='8547cb65d3119e9f86456db979332d82d7e6f870'
NUMERICAL_ORIGIN='286b4a132b960a3a633511173846b57ebef32cee'
PLAN=A/'cpu-roofline-plan.json'
RESULT=A/'cpu-roofline-current.json'
RESOURCE=engine.RESULT
RESOURCE_KEY='adaptive_cpu_leaves_regular_code_v1'
NATIVE_ROOT=engine.ROOT
CPU_SOURCE=NATIVE_ROOT/'source'
COMPILER=Path('/opt/intel/oneapi/compiler/2026.2')
WORKERS=(1,2,3,5)
GROUPS=(6,28)
SCHEMA='strata-cpu-packed-iq3xxs-gather-v1'
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


def numerical_origin():
    data=gen.git('show',NUMERICAL_ORIGIN+':'+str(RESULT.relative_to(P)))
    value=json.loads(data)
    guards.require(value['passed'] and value['complete'] and value['ownership_closed'] and
        not value['active'] and value['decision']['screen_complete'] and
        value['verified_conservation']['timing_samples']==240 and value['artifact_retirement']['complete'],
        'closed compact timing/numerical predecessor required')
    return dict(commit=NUMERICAL_ORIGIN,path=str(RESULT.relative_to(P)),
        sha256=hashlib.sha256(data).hexdigest())


def check_order(rows):
    return [[r['gu'],r['down'],w,g] for r in rows for w in (1,5) for g in GROUPS]


def screen_order():
    # One representative format/current NT mixture first; other formats and
    # pure NT1 timing remain future questions,not implicit model qualification.
    return [[18,20,w,g,pair,arm] for w in (1,5) for g in GROUPS for pair in range(3)
            for arm in (('raw','packed') if pair!=1 else ('packed','raw'))]


def prepare():
    guards.require(not ROOT.exists(),'fresh single-use compact-view root required')
    gen.committed(RESULT);origin=calibration_origin();numeric=numerical_origin()
    guards.require(base.digest(RESULT)==numeric['sha256'],'archive the closed numerical predecessor first')
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
        calibration=origin,numerical_origin=numeric,environment=environment(),machine=base.machine(),formats=rows,
        metadata_sha256=base.digest(base.METADATA),model_identity={r['shard']:base.identity(r['shard']) for r in rows},
        tools={str(p):base.digest(p) for p in (COMPILER/'bin/icx',COMPILER/'bin/icpx','/usr/bin/cmake','/usr/bin/ninja')},
        configure=['/usr/bin/cmake','-S',str(P/'bench/sycl/cpu_decode_mix'),'-B',str(ROOT/'build'),
            '-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_CXX_EXTENSIONS=OFF',
            '-DCMAKE_C_COMPILER='+str(COMPILER/'bin/icx'),'-DCMAKE_CXX_COMPILER='+str(COMPILER/'bin/icpx'),
            '-DCMAKE_CXX_FLAGS='+CXX_FLAGS,'-DCMAKE_C_FLAGS=-DSTRATA_CPU_SERVICE_PROFILE=1',
            '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t','-DSTRATA_GGML_DIR='+str(base.GGML),
            '-DSTRATA_CPU_SOURCE_ROOT='+str(CPU_SOURCE)],
        build=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target','cpu_decode_mix','--parallel','4'],
        protocol=dict(question='Does direct AVX2 gather from the original98B IQ3_XXS format reduce current-mixture GU time without per-expert expansion?',
            check_order=check_order(rows),screen_order=screen_order(),
            controls='Repeat compact controls,and independently check packed gather/sign integer lanes on524288mixed grid/sign/scale chunks;16full FMA/GU cases,64partial ranges and11packed GU refusals.',
            real='Both formats,groups6/28,1/6compute threads;four NT1 activation vectors per real expert plus NT1..4/mixed threaded jobs. Validate all3raw/compact/packed variants,finite/native bits,inactive rows,guards and actual affinity.',
            scope='Component isolation:both arms use the same bounded GuTeam barrier scheduler. NT1 calls the compiled native pair or packed-gather entry directly;compact is an additional numerical control. NT2+uses unchanged native_gu_rows. This is not the production ExpertPool dispatch/scheduling or model latency.',
            memory='Both arms keep the original raw blobs and all compact views. Report allocation+encode wall/CPU time,logical and guarded view storage,raw bytes and owned peak RSS. Packed needs no per-expert expansion;the unused compact views intentionally coexist in both arms to match the prior component envelope. Report this diagnostic setup separately,not as required packed conversion. No measured production RSS claim.',
            samples='Per child,reuse then stream;one initial pass,three pilot/warmup passes,10K timer queries with query/fastest pass<0.001. Five whole-pass samples per mode,target1.3s,floor1s,passes<1000000. Validate actual total/NT1/compact/packed/raw row counts. All samples retained;no replacement.',
            comparison='Representative18/20/current measured NT mixture screen:1/6compute threads,groups6/28,three opposite-order process pairs per shape. Each child supplies both reuse/stream. Per-process median seconds/pass;paired raw/packed ratio;geometric mean and all individual ratios/spread retained.',
            acceptance='A shape/stream is eligible for a separately committed follow-up only when all3paired ratios>1 and geometric mean>1;no blanket minimum speedup. Mixed/negative results stay recorded. No per-expert setup is required by packed;shared unused-view setup remains separately recorded. No conversion to model TG.',
            gate='Commit each stage before next. Require all8checks and synthetic controls before screen;normal0,quiet stderr,closed owners,exact source/build/work/env. Stop first execution/numerical/budget failure;no retry/exclusion. No model trial from this screen alone.',
            pending='Before production selection,validate other-format/pureNT1 performance and actual production entry/lifecycle,then model task quality,prefill independence and physical262144/RESTORE. Keep the qualified prefill route.',
            retention='Keep individual useful samples,setup/footprint,source/build identity and compact decisions. The30e45b85timing ELF is the immediate numerical fallback until new checks pass;recover its original313e5source/recipe/636inputs before retirement. Retire successful raw timing/build output after closing the comparison. Preserve both inference generations.'),
        budgets=dict(build_seconds=600,controls_seconds=180,check_child_seconds=120,screen_child_seconds=180,stage_seconds=1800,
            rss_bytes=3<<30,text_bytes=8<<20,root_bytes=256<<20))
    ROOT.mkdir(mode=0o700);write(PLAN,plan)
    write(RESULT,dict(schema=SCHEMA,stage='prepared',source_commit=revision,plan_sha256=base.digest(PLAN),
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,
        calibration_origin=origin,hardware_saturation_qualified=False,model_adoption_qualified=False))
    print('Prepared packed98B gather timing plan;commit before build.',flush=True)


def load():
    plan=gen.committed(PLAN)
    guards.require(plan['schema']==SCHEMA and
        base.source_pin(P,plan['source']['revision'],PREFIXES)==plan['source'] and
        base.source_pin(base.GGML,base.GGML_REV,('ggml',))==plan['ggml'] and
        plan['calibration']==calibration_origin() and plan['numerical_origin']==numerical_origin() and environment()==plan['environment'] and
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
    guards.require(plan['protocol']['check_order']==check_order(plan['formats']) and
        plan['protocol']['screen_order']==screen_order(),'frozen complete comparison order differs')
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
        encode_refusals=2,packed_integer_chunks=524288,packed_FMA_cases=16,
        packed_partial_ranges=64,packed_GU_refusals=11,finite_canaries=True) if row is None else
        dict(kind='compact_validation',format=18,nt=1,activation_cases=4,experts=row['experts'],
            native_pair_bits_equal=True,packed_native_pair_bits_equal=True,finite=True,canaries=True,compact_GU_bytes_per_expert=1715200,
            raw_plus_compact_bytes_per_expert=row['blob_bytes']+1715200))
    expected=[value,dict(kind='closed',passed=True)]
    guards.require(json.dumps(lines,sort_keys=True)==json.dumps(expected,sort_keys=True),
        'complete typed compact numerical/guard evidence required')
    return value


def parse_gu(lines,row,workers,group,arm):
    guards.require(type(workers) is int and workers in (1,5) and type(group) is int and
        group in GROUPS and arm in ('gu-check','gu-raw','gu-compact','gu-packed'),'planned GU shape/arm required')
    timed=arm!='gu-check';count=group*math.ceil((128<<20)/(group*row['blob_bytes']))
    if not timed:
        parse_compact(lines[:1]+[dict(kind='closed',passed=True)],dict(row,experts=count));lines=lines[1:]
    kinds=['gu_setup','gu_validation','gu_work']
    if timed:
        for _ in range(2):kinds+=['gu_timer']+['gu_sample']*5+['gu_result','gu_validation']
    else:kinds+=['gu_result']
    kinds+=['closed']
    guards.require(isinstance(lines,list) and all(isinstance(r,dict) for r in lines) and
        [r.get('kind') for r in lines]==kinds,'complete ordered GU evidence required')
    def equal(actual,expected):
        guards.require(json.dumps(actual,sort_keys=True)==json.dumps(expected,sort_keys=True),
            'complete typed GU dispatch/numerical/guard evidence required')
    def positive(value,zero=False):
        return type(value) in (int,float) and math.isfinite(value) and (value>=0 if zero else value>0)
    setup=lines[0]
    equal({k:v for k,v in setup.items() if k not in ('allocation_encode_wall_s','allocation_encode_cpu_s')},
        dict(kind='gu_setup',experts=count,logical_view_bytes=count*1715200,
            view_storage_bytes=count*12802*134,raw_blob_bytes=count*row['blob_bytes']))
    guards.require(all(positive(setup.get(k)) for k in ('allocation_encode_wall_s','allocation_encode_cpu_s')),
        'finite positive setup costs required')
    histogram=expected_histogram(row['observed_jobs_NT'],group)
    valid=dict(kind='gu_validation',compute_threads=6 if workers==5 else 1,group=group,experts=count,
        NT_mask=30,variants=3,mixed_NT=True,bits=True,inactive=True,canaries=True)
    work=dict(kind='gu_work',arm=arm,compute_threads=6 if workers==5 else 1,group=group,experts=count,
        batches=100,jobs_NT=histogram,host_cpu=base.machine()['physical_cpus'][0],affinity_verified=True)
    result=dict(kind='gu_result',experts=count,active_values_equal=True,finite_guards=True)
    for r in lines:
        if r['kind']=='gu_validation':equal(r,valid)
        elif r['kind']=='gu_work':equal(r,work)
        elif r['kind']=='gu_result':equal(r,result)
    equal(lines[-1],dict(kind='closed',passed=True))
    samples=[r for r in lines if r['kind']=='gu_sample'];timers=[r for r in lines if r['kind']=='gu_timer']
    if timed:
        equal([(r.get('stream'),r.get('rep')) for r in samples],
            [(stream,i) for stream in (False,True) for i in range(5)])
        for r,stream in zip(timers,(False,True)):
            guards.require(set(r)=={'kind','stream','query_s','fastest_pass_s'} and r['stream'] is stream and
                positive(r['query_s'],True) and positive(r['fastest_pass_s']) and
                r['query_s']/r['fastest_pass_s']<.001,'GU timer overhead gate')
        for s in samples:
            guards.require(set(s)=={'kind','stream','rep','passes','wall_s','cpu_s',
                'rows','NT1_rows','compact_rows','raw_rows','packed_rows'} and type(s['stream']) is bool and
                type(s['rep']) is int and type(s['passes']) is int and 0<s['passes']<1000000 and
                positive(s['wall_s']) and s['wall_s']>=1 and positive(s['cpu_s'],True),
                'eligible typed GU timing batch required')
            total=s['passes']*100*group*640;one=s['passes']*histogram[0]*640
            compact=one if arm=='gu-compact' else 0;packed=one if arm=='gu-packed' else 0
            equal({k:s[k] for k in ('rows','NT1_rows','compact_rows','raw_rows','packed_rows')},
                dict(rows=total,NT1_rows=one,compact_rows=compact,raw_rows=total-compact-packed,packed_rows=packed))
    return dict(setup=setup,work=work,validated=True,timers=timers,samples=samples)


def screen_decision(children):
    """All declared samples/pairs,including negative shapes; no model claim."""
    guards.require([c.get('case') for c in children]==screen_order(),
        'all declared ordered screen processes required')
    cells=[]
    for workers in (1,5):
        for group in GROUPS:
            for stream in (False,True):
                arms={}
                for arm in ('raw','packed'):
                    values=[]
                    for pair in range(3):
                        selected=[c for c in children if c['case']==[18,20,workers,group,pair,arm]]
                        guards.require(len(selected)==1,'complete unique screen process required')
                        child=selected[0];s=[r for r in child['validation']['samples'] if r['stream'] is stream]
                        guards.require(len(s)==5,'all five timing samples required')
                        rates=[r['wall_s']/r['passes'] for r in s]
                        values.append(dict(pair=pair,seconds_per_pass=statistics.median(rates),
                            minimum_s=min(rates),maximum_s=max(rates),
                            CV=statistics.stdev(rates)/statistics.mean(rates),
                            setup_s=child['validation']['setup']['allocation_encode_wall_s']))
                    arms[arm]=values
                ratios=[a['seconds_per_pass']/b['seconds_per_pass'] for a,b in zip(arms['raw'],arms['packed'])]
                saved=statistics.median([a['seconds_per_pass']-b['seconds_per_pass']
                    for a,b in zip(arms['raw'],arms['packed'])])
                gm=math.exp(sum(math.log(r) for r in ratios)/3)
                cells.append(dict(compute_threads=6 if workers==5 else 1,group=group,stream=stream,
                    processes=arms,paired_raw_over_packed=ratios,geomean=gm,
                    median_saved_seconds_per_pass=saved,
                    per_expert_expansion_required=False,followup_eligible=all(r>1 for r in ratios) and gm>1))
    return dict(screen_complete=True,cells=cells,
        performance_qualified=any(c['followup_eligible'] for c in cells),model_adoption_qualified=False,
        next_action='Use eligible packed-gather shapes for other-format/NT1 and actual production comparisons;preserve negative cells. No model trial or throughput claim from this component screen.')


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
    for name in ('bench/sycl/cpu_decode_mix/main.cpp','bench/sycl/cpu_decode_mix/compact_iq3xxs.hpp',
                 'bench/sycl/cpu_decode_mix/packed_iq3xxs.hpp'):
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
            if stage=='controls':cases=[('synthetic',[built['binaries']['mix']['path'],'--compact-controls'],None,None)]
            else:
                control=json.loads((ROOT/'controls.json').read_text())
                guards.require(control['passed'] and control['complete'] and control['ownership_closed'] and
                    not control['active'] and control['plan_sha256']==base.digest(PLAN) and
                    control['binaries']==built['binaries'] and len(control['children'])==1,
                    'closed compact controls required')
                parse_compact([control['children'][0]['validation'],dict(kind='closed',passed=True)])
                rows={(r['gu'],r['down']):r for r in plan['formats']}
                if stage=='check':
                    cases=[('-'.join(map(str,c)),command(rows[tuple(c[:2])],c[2],c[3],'gu-check'),
                        rows[tuple(c[:2])],c) for c in plan['protocol']['check_order']]
                else:
                    guards.require(stage=='screen','known compact GU stage')
                    checked=json.loads((ROOT/'check.json').read_text())
                    guards.require(checked['passed'] and checked['complete'] and checked['ownership_closed'] and
                        not checked['active'] and checked['plan_sha256']==base.digest(PLAN) and
                        checked['binaries']==built['binaries'] and
                        [c['case'] for c in checked['children']]==plan['protocol']['check_order'] and
                        gen.committed(RESULT)==compact_record(checked,checked['commands']),
                        'commit all8numerical checks before screen')
                    cases=[('-'.join(map(str,c)),command(rows[tuple(c[:2])],c[2],c[3],'gu-'+c[5]),
                        rows[tuple(c[:2])],c) for c in plan['protocol']['screen_order']]
            for label,argv,row,case in cases:
                seconds=plan['budgets']['controls_seconds' if row is None else
                    'screen_child_seconds' if stage=='screen' else 'check_child_seconds']
                entry=execute(label,argv,seconds)
                stdout=ROOT/'owned'/(stage+'-'+label+'.stdout')
                guards.require((ROOT/'owned'/(stage+'-'+label+'.stderr')).read_bytes()==b'','unexpected compact diagnostic')
                lines=[json.loads(s) for s in stdout.read_text().splitlines()]
                value=parse_compact(lines) if row is None else parse_gu(lines,row,case[2],case[3],argv[10])
                record['children'].append(dict(case=case,validation=value,exit_code=entry['exit_code'],
                    peak_session_rss_bytes=entry.get('peak_session_rss_bytes'),stdout_sha256=base.digest(stdout)))
                persist();print('Closed compact '+label,flush=True)
            if stage=='screen':record['decision']=screen_decision(record['children'])
            elif stage=='check':record['decision']=dict(threaded_GU_numerical_qualified=True,
                cases=len(cases),performance_qualified=False,model_adoption_qualified=False,
                next_action='Committed representative-format/current-mixture screen;no timing measured by this check.')
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','build','controls','check','screen'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        # This bounded CPU build never overwrites the retained engine generation.
        return prepare() if args.stage=='prepare' else run(args.stage)


if __name__=='__main__':sys.exit(main())
