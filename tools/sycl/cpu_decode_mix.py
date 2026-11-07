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
import benchmark_runtime as runtime
import cpu_roofline as base
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as owner
import build_adaptive_payload_model as engine

P,B,A=gen.P,gen.B,gen.A
ROOT=B/'cpu-activation-lut-v1'
BASELINE_CALIBRATION='8547cb65d3119e9f86456db979332d82d7e6f870'
NUMERICAL_ORIGIN='e4f6f064449c89b97f3f52527d4acad2baf4af37'
PLAN=A/'cpu-roofline-plan.json'
RESULT=A/'cpu-roofline-current.json'
RESOURCE=engine.RESULT
RESOURCE_KEY='adaptive_cpu_leaves_regular_code_v1'
NATIVE_ROOT=engine.ROOT
CPU_SOURCE=NATIVE_ROOT/'source'
COMPILER=Path('/opt/intel/oneapi/compiler/2026.2')
GROUPS=(1,10)
SCHEMA='strata-cpu-activation-lut-v1'
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
    return [[r['gu'],r['down'],g] for r in rows for g in GROUPS]


def screen_order():
    return [[18,20,g,pair,arm] for g in GROUPS for pair in range(3)
            for arm in (('raw','lut') if pair!=1 else ('lut','raw'))]


def stream_count(row):return 10*math.ceil((128<<20)/(10*row['blob_bytes']))


def prepare():
    guards.require(not ROOT.exists(),'fresh single-use LUT root required')
    previous=gen.committed(RESULT);origin=calibration_origin();numeric=numerical_origin()
    guards.require(base.digest(RESULT)==numeric['sha256'] and
        previous['retained_auxiliary']['roots']==[] and
        all(not Path(r['root']).exists() for r in previous['artifact_retirement']['recovered']),
        'closed/recovered/retired predecessors required before another auxiliary build')
    resource,proof=qualification();observed=observed_histograms(resource)
    rows=[r for r in base.formats() if r['gu']==18]
    guards.require({(r['gu'],r['down']) for r in rows}=={(18,20),(18,42)},'both IQ3_XXS formats required')
    for r in rows:
        r['experts']=stream_count(r);r['observed_jobs_NT']=observed[r['gu'],r['down']];r['fixture_sha256']=base.fixture_hash(r)
    host=gen.committed(RESOURCE)['upstream_nonblocking_console_entry_host_v1']
    host_env=dict(host['environment']);host_env['TMPDIR']=str(ROOT/'tmp');host_env['XDG_CACHE_HOME']=str(ROOT/'cache')
    runtime.require_environment(host_env);runtime.validate_manifest(runtime.read_manifest(host['python_contract']))
    guards.require(runtime.selection(host_env)==host['python_selected'],'qualified host Python changed')
    revision=gen.git('rev-parse','HEAD').decode().strip()
    plan=dict(schema=SCHEMA,source=base.source_pin(P,revision,PREFIXES),
        ggml=base.source_pin(base.GGML,base.GGML_REV,('ggml',)),qualification=proof,
        calibration=origin,numerical_origin=numeric,environment=environment(),machine=base.machine(),formats=rows,
        metadata_sha256=base.digest(base.METADATA),model_identity={r['shard']:base.identity(r['shard']) for r in rows},
        tools={str(p):base.digest(p) for p in (COMPILER/'bin/icx',COMPILER/'bin/icpx','/usr/bin/cmake','/usr/bin/ninja','/usr/bin/python3')},
        configure=['/usr/bin/cmake','-S',str(P/'bench/sycl/cpu_decode_mix'),'-B',str(ROOT/'build'),
            '-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_CXX_EXTENSIONS=OFF',
            '-DCMAKE_C_COMPILER='+str(COMPILER/'bin/icx'),'-DCMAKE_CXX_COMPILER='+str(COMPILER/'bin/icpx'),
            '-DCMAKE_CXX_FLAGS='+CXX_FLAGS,'-DCMAKE_C_FLAGS=-DSTRATA_CPU_SERVICE_PROFILE=1',
            '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t','-DSTRATA_GGML_DIR='+str(base.GGML),
            '-DSTRATA_CPU_SOURCE_ROOT='+str(CPU_SOURCE)],
        build=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target','cpu_decode_mix','--parallel','4'],
        host_environment=host_env,python_contract=host['python_contract'],python_selected=host['python_selected'],
        host_command=[host['python_selected']['path'],'-X','faulthandler','-B',str(P/'tests/core/test_cpu_decode_mix.py'),'-v'],
        protocol=dict(question='Does a512KiB exact block-local activation-dependent partial-dot LUT repay generation cost within the actual top_k10reuse limit?',
            check_order=check_order(rows),screen_order=screen_order(),
            controls='All2097152table entries for8edge activations,4194304scaled native-sign chunks,14shape/alias refusals;all guards and no writes on refusal.',
            real='Both real model IQ3_XXS formats,E1/E10,NT1,four activation vectors per expert;all rows bit-equal to current production native_gu_rows and finite with guards.',
            scope='Serial pinned host thread,component GU only. Raw arm calls actual native_gu_rows Decode entry with audited IQ3 paired dispatch. LUT arm has its own bounded checked entry. No production ExpertPool,phase dispatch,model TG or saturation claim. NT2-4and threading remain pending.',
            sharing='Same quantized input across E1orE10experts only. Actual source limits per-token routes to10;28jobs across tokens is not28way activation reuse. Candidate rebuilds10slabs per call with no cross-request reuse.',
            memory='512KiB slab plus64B/FF-row accumulators,preallocated equally in both arms. Clear accumulators,build tables,raw row loads,FMA,SwiGLU and output writes are timed. No per-expert expansion. Both arms retain identical >=128MiB raw fixture set;reuse and cyclic stream measured separately.',
            samples='One initial call,three pilots,10K timer queries/query<0.001fastest call,five>=1s whole-call batches,target1.3s. All samples retained. Actual native pair rows and LUT build/block/output counts must conserve;validate final timed output before resetting it.',
            comparison='Three opposite-order independent process pairs per E;both reuse/stream in every child. Per-process median seconds/call;raw/LUT paired ratios and geometric means,with all samples/spread.',
            acceptance='A cell is eligible for a separately committed next screen only when all3paired ratios>1and GM>1. No blanket minimum gain. E1is negative control;only E10gain warrants actual reuse/NT/thread tests,never model adoption. Keep unsuccessful results;no retry or posthoc exclusions.',
            gate='Commit source,host tests and plan before build. Commit build,controls and all4real checks before next stage. Stop first execution,numerical,budget or ownership failure. No model trial from this discriminator alone.',
            pending='If favorable,verify actual per-token CPU-miss/NT reuse,production entry and critical-path benefit;preserve prefill/decode handoff and repeated model performance,agent/task and physical262144/RESTORE.',
            retention='One fresh auxiliary generation;baseline and LUT compile together from the frozen native CPU source. Prior compact/gather roots are retired. Keep useful samples and source/recipe identities;retire successful raw output after closure. Preserve two inference generations.'),
        budgets=dict(build_seconds=600,controls_seconds=180,check_child_seconds=180,screen_child_seconds=180,stage_seconds=1800,
            rss_bytes=3<<30,text_bytes=8<<20,root_bytes=256<<20))
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir();(ROOT/'cache').mkdir();write(PLAN,plan)
    write(RESULT,dict(schema=SCHEMA,stage='prepared',source_commit=revision,plan_sha256=base.digest(PLAN),
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,
        calibration_origin=origin,hardware_saturation_qualified=False,model_adoption_qualified=False))
    print('Prepared exact activation-LUT discriminator;commit before build.',flush=True)


def load():
    plan=gen.committed(PLAN)
    guards.require(plan['schema']==SCHEMA and base.source_pin(P,plan['source']['revision'],PREFIXES)==plan['source'] and
        base.source_pin(base.GGML,base.GGML_REV,('ggml',))==plan['ggml'] and plan['calibration']==calibration_origin() and
        plan['numerical_origin']==numerical_origin() and environment()==plan['environment'] and base.machine()==plan['machine'],
        'source/environment/platform/origin drift')
    for p,s in plan['tools'].items():guards.require(base.digest(p)==s,'compiler/tool drift')
    runtime.require_environment(plan['host_environment']);runtime.validate_manifest(runtime.read_manifest(plan['python_contract']))
    guards.require(runtime.selection(plan['host_environment'])==plan['python_selected'],'host interpreter changed')
    guards.require(base.digest(base.METADATA)==plan['metadata_sha256'],'metadata drift')
    resource,proof=qualification();guards.require(proof==plan['qualification'],'native/resource proof drift')
    observed=observed_histograms(resource);original={(r['gu'],r['down']):r for r in base.formats() if r['gu']==18}
    guards.require(len(plan['formats'])==2 and {(r['gu'],r['down']) for r in plan['formats']}==set(original),'complete formats')
    for row in plan['formats']:
        key=row['gu'],row['down']
        guards.require(all(row[k]==v for k,v in original[key].items() if k!='experts') and
            row['experts']==stream_count(row) and row['observed_jobs_NT']==observed[key] and
            base.identity(row['shard'])==plan['model_identity'][row['shard']] and base.fixture_hash(row)==row['fixture_sha256'],
            'fixture/work drift')
    guards.require(plan['protocol']['check_order']==check_order(plan['formats']) and
        plan['protocol']['screen_order']==screen_order(),'frozen order drift')
    return plan


def command(row,group,arm):
    return [str(ROOT/'build/cpu_decode_mix'),row['shard'],str(row['gu']),str(row['down']),str(row['gate']),str(row['up']),
        str(row['down_offset']),str(row['blob_bytes']),str(group),arm,'20261108']


def typed_equal(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(typed_equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(typed_equal(x,y) for x,y in zip(a,b))
    return a==b


def parse_controls(lines):
    expected=dict(kind='lut_controls',table_entries=2097152,scaled_chunks=4194304,activation_edges=8,refusals=14,guards=True)
    guards.require(typed_equal(lines,[expected,dict(kind='closed',passed=True)]) and
        all(type(lines[0][k]) is int for k in ('table_entries','scaled_chunks','activation_edges','refusals')) and
        type(lines[0]['guards']) is bool,'incomplete/invalid LUT controls')
    return expected


def parse_lut(lines,row,group,arm):
    guards.require(type(group) is int and group in GROUPS and arm in ('check','raw','lut'),'planned case required')
    count=stream_count(row)
    validation=dict(kind='lut_validation',experts=count,activation_cases=4,comparisons=4*count,native_bits_equal=True,finite_guards=True)
    work=dict(kind='lut_work',arm=arm,group=group,NT=1,experts=count,host_cpu=base.machine()['physical_cpus'][0],
        affinity_verified=True,slab_bytes=524288,accumulator_bytes=group*640*64,raw_blob_bytes=count*row['blob_bytes'])
    guards.require(len(lines)>2 and typed_equal(lines[:2],[validation,work]) and typed_equal(lines[-1],dict(kind='closed',passed=True)),
        'work/validation/completion drift')
    for r in lines[:2]:
        guards.require(all(type(v) is type((validation if r['kind']=='lut_validation' else work)[k]) for k,v in r.items()),'typed work required')
    if arm=='check':
        guards.require(len(lines)==3,'unexpected timed check');return dict(work=work,numerical_qualified=True)
    guards.require(len(lines)==17,'incomplete or extra timing records')
    samples=[];timers=[]
    for at,stream in [(2,False),(9,True)]:
        t=lines[at];guards.require(set(t)=={'kind','stream','query_s','fastest_call_s'} and t['kind']=='lut_timer' and
            type(t['stream']) is bool and t['stream']==stream and all(type(t[k]) in (int,float) and math.isfinite(t[k]) and t[k]>0
                for k in ('query_s','fastest_call_s')) and t['query_s']/t['fastest_call_s']<.001,'timer gate')
        timers.append(t)
        for rep,s in enumerate(lines[at+1:at+6]):
            guards.require(set(s)=={'kind','stream','rep','calls','wall_s','cpu_s','output_rows','slabs','block_rows','native_pair_rows'} and
                s['kind']=='lut_sample' and type(s['stream']) is bool and s['stream']==stream and s['rep']==rep and
                all(type(s[k]) is int for k in ('rep','calls','output_rows','slabs','block_rows','native_pair_rows')) and
                0<s['calls']<1000000 and all(type(s[k]) in (int,float) and math.isfinite(s[k]) and s[k]>=0 for k in ('wall_s','cpu_s')) and
                s['wall_s']>=1 and s['cpu_s']<=s['wall_s']+.05,'invalid sample/floor')
            rows=s['calls']*group*640
            guards.require(s['output_rows']==rows and s['slabs']==(s['calls']*10 if arm=='lut' else 0) and
                s['block_rows']==(rows*10 if arm=='lut' else 0) and s['native_pair_rows']==(rows if arm=='raw' else 0),
                'actual build/dispatch/output counts differ')
            samples.append(s)
        guards.require(typed_equal(lines[at+6],dict(kind='lut_result',stream=stream,native_bits_equal=True,finite_guards=True)),
            'final timed output/guards not checked')
    return dict(work=work,numerical_qualified=True,timers=timers,samples=samples)


def screen_decision(children):
    guards.require(typed_equal([c['case'] for c in children],screen_order()),'full unique paired order required')
    cells=[]
    for group in GROUPS:
        for stream in (False,True):
            ratios=[];raw_times=[];lut_times=[]
            for pair in range(3):
                values={}
                for arm in ('raw','lut'):
                    c=next(c for c in children if c['case']==[18,20,group,pair,arm])
                    guards.require(c['exit_code']==0,'failed child cannot qualify')
                    s=[s for s in c['validation']['samples'] if s['stream']==stream]
                    guards.require(len(s)==5,'five individual batches required')
                    values[arm]=statistics.median(x['wall_s']/x['calls'] for x in s)
                raw_times.append(values['raw']);lut_times.append(values['lut']);ratios.append(values['raw']/values['lut'])
            gm=math.exp(statistics.mean(math.log(r) for r in ratios));eligible=gm>1 and all(r>1 for r in ratios)
            cells.append(dict(group=group,NT=1,stream=stream,paired_raw_over_lut=ratios,geomean=gm,
                raw_median_seconds_per_call=statistics.median(raw_times),lut_median_seconds_per_call=statistics.median(lut_times),
                repeated_gain_qualified=eligible,followup_eligible=eligible and group==10))
    return dict(screen_complete=True,cells=cells,performance_qualified=any(c['followup_eligible'] for c in cells),
        model_adoption_qualified=False,interpretation='Setup-inclusive NT1 same-activation E1negative/E10optimistic discriminator only;no model TG,actual CPU-miss reuse,production critical path or physical saturation qualification.')

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
    for name in ('bench/sycl/cpu_decode_mix/main.cpp','bench/sycl/cpu_decode_mix/activation_lut.hpp'):
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
    result['owned_commands']=dict(count=len(commands),all_closed=all(owner.closed(c) for c in commands),all_exit_zero=all(c.get('exit_code')==0 for c in commands))
    result['hardware_saturation_qualified']=False;result['model_adoption_qualified']=False
    return result


def run(stage):
    started=time.monotonic();plan=load();gen.committed(RESULT);path=ROOT/(stage+'.json')
    guards.require(not path.exists(),'single-use stage;inspect existing owner rather than retry')
    record=dict(schema=SCHEMA,stage=stage,source_commit=plan['source']['revision'],plan_sha256=base.digest(PLAN),
        calibration_origin=plan['calibration'],passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,children=[])
    own=None;deadline=started+plan['budgets']['stage_seconds']
    with path.open('x') as marker:marker.write(json.dumps(record,indent=2)+'\n')
    def persist():
        record['active']=bool(own and own.active is not None);write(path,dict(record,commands=own.commands if own else []))
    def execute(label,argv,wall,environment=None):
        guards.require(time.monotonic()<deadline,'whole-stage budget expired')
        entry,_,_=own.run(stage+'-'+label,argv,environment if environment is not None else plan['environment'],P,wall=min(wall,deadline-time.monotonic()),cpu=1200,
            rss_cap=plan['budgets']['rss_bytes'],text_cap=plan['budgets']['text_bytes'],total_cap=plan['budgets']['root_bytes'],file_cap=plan['budgets']['root_bytes'])
        guards.require(owner.closed(entry) and entry['exit_code']==0,'owned child failed: '+label);return entry
    try:
        own=owner.Owner(ROOT/'owned',persist)
        if stage=='host':
            entry=execute('unittest',plan['host_command'],120,plan['host_environment'])
            text=(ROOT/'owned/host-unittest.stderr').read_text()
            count=re.search(r'^Ran (\d+) tests? in [0-9.]+s$',text,re.M)
            guards.require(count and re.search(r'^OK$',text,re.M) and int(count[1])>0,'host controller tests incomplete/failed/skipped')
            record['host_controller_validation']=dict(tests=int(count[1]),skipped=0,exit_code=entry['exit_code'],ownership_closed=True)
        elif stage=='build':
            hosted=json.loads((ROOT/'host.json').read_text())
            guards.require(hosted['passed'] and hosted['complete'] and hosted['ownership_closed'] and not hosted['active'] and
                hosted['plan_sha256']==base.digest(PLAN) and gen.committed(RESULT)==compact_record(hosted,hosted['commands']),
                'commit passed controller tests before build')
            execute('configure',plan['configure'],120);execute('compile',plan['build'],plan['budgets']['build_seconds'])
            execute('dependencies',['/usr/bin/ninja','-C',str(ROOT/'build'),'-t','deps'],40)
            record.update(attest_build(plan,(ROOT/'owned/build-dependencies.stdout').read_text()))
            binary=ROOT/'build/cpu_decode_mix';record['binaries']=dict(mix=dict(path=str(binary),sha256=base.digest(binary)))
            execute('link',['/usr/bin/ldd',str(binary)],20)
            guards.require(not re.search('sycl|level_zero|cuda|amdhip|libze|not found',(ROOT/'owned/build-link.stdout').read_text(),re.I),'GPU runtime linked or missing dependency')
        else:
            built=json.loads((ROOT/'build.json').read_text());verify_built(plan,built);record['binaries']=built['binaries']
            if stage=='controls':cases=[('synthetic',[built['binaries']['mix']['path'],'--lut-controls'],None,None)]
            else:
                control=json.loads((ROOT/'controls.json').read_text())
                guards.require(control['passed'] and control['complete'] and control['ownership_closed'] and not control['active'] and
                    control['plan_sha256']==base.digest(PLAN) and control['binaries']==built['binaries'] and len(control['children'])==1,'closed controls required')
                parse_controls([control['children'][0]['validation'],dict(kind='closed',passed=True)])
                rows={(r['gu'],r['down']):r for r in plan['formats']}
                if stage=='check':cases=[('-'.join(map(str,c)),command(rows[tuple(c[:2])],c[2],'check'),rows[tuple(c[:2])],c) for c in plan['protocol']['check_order']]
                else:
                    guards.require(stage=='screen','known stage')
                    checked=json.loads((ROOT/'check.json').read_text())
                    guards.require(checked['passed'] and checked['complete'] and checked['ownership_closed'] and not checked['active'] and
                        checked['plan_sha256']==base.digest(PLAN) and checked['binaries']==built['binaries'] and
                        [c['case'] for c in checked['children']]==plan['protocol']['check_order'] and
                        gen.committed(RESULT)==compact_record(checked,checked['commands']),'commit all4real checks before screen')
                    cases=[('-'.join(map(str,c)),command(rows[tuple(c[:2])],c[2],c[4]),rows[tuple(c[:2])],c) for c in plan['protocol']['screen_order']]
            for label,argv,row,case in cases:
                seconds=plan['budgets']['controls_seconds' if row is None else 'screen_child_seconds' if stage=='screen' else 'check_child_seconds']
                entry=execute(label,argv,seconds);stdout=ROOT/'owned'/(stage+'-'+label+'.stdout')
                guards.require((ROOT/'owned'/(stage+'-'+label+'.stderr')).read_bytes()==b'','unexpected diagnostics')
                lines=[json.loads(s) for s in stdout.read_text().splitlines()]
                value=parse_controls(lines) if row is None else parse_lut(lines,row,case[2],argv[9])
                record['children'].append(dict(case=case,validation=value,exit_code=entry['exit_code'],peak_session_rss_bytes=entry.get('peak_session_rss_bytes'),stdout_sha256=base.digest(stdout)))
                persist();print('Closed LUT '+label,flush=True)
            if stage=='screen':record['decision']=screen_decision(record['children'])
            elif stage=='check':record['decision']=dict(numerical_qualified=True,cases=len(cases),performance_qualified=False,model_adoption_qualified=False,next_action='Committed setup-inclusive E1/E10 screen;no speed measured here.')
        load()
        if stage not in ('host','build'):verify_built(plan,built)
        guards.require(time.monotonic()<=deadline,'whole-stage budget expired');record.update(passed=True,complete=True)
    except BaseException as error:record['error']=repr(error)
    finally:
        record['ownership_closed']=own is None or (own.active is None and all(owner.closed(c) for c in own.commands))
        record['passed']=bool(record['passed'] and record['ownership_closed']);persist();write(RESULT,compact_record(record,own.commands if own else []))
    print(json.dumps({k:record.get(k) for k in ('stage','passed','complete','ownership_closed','error')}),flush=True)
    return 0 if record['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','host','build','controls','check','screen'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        return prepare() if args.stage=='prepare' else run(args.stage)


if __name__=='__main__':sys.exit(main())
