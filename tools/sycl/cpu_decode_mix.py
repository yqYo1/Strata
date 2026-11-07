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
ROOT=B/'cpu-decode-mix-v4'
ORIGINAL_ROOT=B/'cpu-decode-mix-v3'
ORIGINAL_RESULT='49b2d4821077e4aa51425c0450a8414e013cd79a'
ORIGINAL_STAGES=dict(build='ce9f3aa0f9ebc7e6c6dabcb185b009d52b887f71',
    check='6f453cc31eaca7e22687e2757bd37b5a82ac6dbe',measure=ORIGINAL_RESULT)
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
SCHEMA='strata-cpu-decode-mix-v4'
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



def original_calibration():
    """Reuse closed expert evidence, never retry or relabel the failed stage."""
    relative=str(RESULT.relative_to(P))
    plan_bytes=gen.git('show',ORIGINAL_RESULT+':'+str(PLAN.relative_to(P)))
    plan=json.loads(plan_bytes);stages={}
    for stage,revision in ORIGINAL_STAGES.items():
        gen.git('merge-base','--is-ancestor',revision,'HEAD')
        historical=json.loads(gen.git('show',revision+':'+relative))
        path=ORIGINAL_ROOT/(stage+'.json');record=json.loads(path.read_text())
        guards.require(compact_record(record,record['commands'])==historical and
            record['schema']=='strata-cpu-decode-mix-v3' and record['stage']==stage and
            record['plan_sha256']==hashlib.sha256(plan_bytes).hexdigest() and
            record['source_commit']==plan['source']['revision'] and not record['GPU_executed'] and
            record['ownership_closed'] and not record['active'],
            'original closed evidence differs: '+stage)
        guards.require(all(owner.closed(c) and c['environment']==plan['environment'] for c in record['commands']),
            'original owner/environment evidence differs')
        for command_entry in record['commands']:
            for identity in command_entry['owners']:
                live=owner.identity(identity['pid'])
                guards.require(not live or live['start_ticks']!=identity['start_ticks'],
                    'original calibration owner still live')
        stages[stage]=record
    built,checked,measured=(stages[s] for s in ('build','check','measure'))
    guards.require(built['passed'] and built['complete'] and checked['passed'] and checked['complete'] and
        measured['passed'] is False and measured['complete'] is False and
        measured['error']=="RuntimeError('owned child failed: scan-2')" and
        len(built['commands'])==5 and len(checked['commands'])==56 and len(measured['commands'])==58 and
        all(c['exit_code']==0 for c in built['commands']+checked['commands']+measured['commands'][:-1]) and
        measured['commands'][-1]['exit_code']==1 and
        measured['commands'][-1]['label']=='measure-scan-2',
        'only the exact closed scanner-2 failure is eligible for this continuation')
    guards.require(checked['binaries']==measured['binaries']==built['binaries'] and
        environment()==plan['environment'] and base.machine()==plan['machine'],
        'original expert executable/environment/platform differs')
    # Recover the original controller and recipe even though this continuation changes them.
    revision=plan['source']['revision']
    names=gen.git('ls-tree','-r','--name-only',revision,'--',*PREFIXES[:-1]).decode().splitlines()
    hashes={n:hashlib.sha256(raw).hexdigest() for n,raw in gen.blobs(P,revision,names)}
    guards.require(len(hashes)==plan['source']['count'] and
        base.fingerprint(hashes)==plan['source']['aggregate_sha256'],
        'original controller/recipe is not recoverable')
    for asset in built['binaries'].values():
        guards.require(base.digest(asset['path'])==asset['sha256'],'original executable bytes changed')
    asset=built['compiled_inputs']
    guards.require(asset['files']==634 and asset['uncovered']==0 and
        base.digest(asset['path'])==asset['sha256'],'original compiler inventory differs')
    actual=json.loads(Path(asset['path']).read_text())
    guards.require(len(actual)==634 and all(base.digest(p)==h for p,h in actual.items()) and
        all(base.digest(p)==h for p,h in built['actual_link_inputs'].items()) and
        base.digest(ORIGINAL_ROOT/'build/compile_commands.json')==built['compile_commands_sha256'],
        'original actual compiler/link inputs differ')
    for tool,sha in plan['tools'].items():
        guards.require(base.digest(tool)==sha,'original compiler/tool changed')
    validate_order(plan['order'])
    lookup={(r['gu'],r['down']):r for r in plan['formats']}
    for stage in ('check','measure'):
        record=stages[stage]
        guards.require(len(record['children'])==56,'complete original expert matrix required')
        for i,(gu,dn,w,g) in enumerate(plan['order']):
            child=record['children'][i];entry=record['commands'][i]
            stdout=ORIGINAL_ROOT/'owned'/(stage+'-cell-'+str(i)+'.stdout')
            guards.require(entry['label']==stage+'-cell-'+str(i) and
                entry['argv']==[built['binaries']['mix']['path'],*command(lookup[gu,dn],w,g,stage)[1:]] and
                base.digest(stdout)==child['stdout_sha256'],'original command/output binding differs')
            parsed=parse([json.loads(s) for s in stdout.read_text().splitlines()],lookup[gu,dn],w,g,stage)
            guards.require(child==dict(pair=[gu,dn],workers=w,group=g,**parsed,
                exit_code=0,stdout_sha256=base.digest(stdout)),
                'original numerical/timing samples differ')
    failed_stderr=ORIGINAL_ROOT/'owned/measure-scan-2.stderr'
    guards.require(failed_stderr.read_text()=='CPU calibration failed: scan thread count\n',
        'original scanner failure mechanism differs')
    proof=dict(result_commit=ORIGINAL_RESULT,stage_commits=ORIGINAL_STAGES,
        original_passed=False,original_complete=False,original_error=measured['error'],
        original_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        original_source=plan['source'],expert_binary=built['binaries']['mix'],
        receipts={s:dict(path=str(ORIGINAL_ROOT/(s+'.json')),
            sha256=base.digest(ORIGINAL_ROOT/(s+'.json'))) for s in ORIGINAL_STAGES},
        compiled_inputs=asset,compile_commands_sha256=built['compile_commands_sha256'],
        link_inputs_sha256=base.fingerprint(built['actual_link_inputs']),
        numerical_cases=56,expert_cells=112,expert_samples=560,
        scope='Original expert checks and samples are reused unchanged,not executed by V4. All four read references are newly measured. V3 remains failed in its pinned commit.')
    return plan,measured,proof


def prepare():
    guards.require(not ROOT.exists(),'fresh single-use calibration root required')
    old=gen.committed(RESULT)
    previous,_,reuse=original_calibration()
    guards.require(old==json.loads(gen.git('show',ORIGINAL_RESULT+':'+str(RESULT.relative_to(P)))),
        'preserve the original failed result before preparing this continuation')
    resource,proof=qualification()
    guards.require(proof==previous['qualification'],'original native/resource qualification differs')
    observed=observed_histograms(resource)
    for row in previous['formats']:
        guards.require(row['observed_jobs_NT']==observed[row['gu'],row['down']] and
            base.identity(row['shard'])==previous['model_identity'][row['shard']] and
            base.fixture_hash(row)==row['fixture_sha256'],'original expert/model fixture differs')
    revision=gen.git('rev-parse','HEAD').decode().strip()
    plan=dict(previous)
    plan.update(schema=SCHEMA,source=base.source_pin(P,revision,PREFIXES),reuse=reuse,
        previous_result=dict(commit=ORIGINAL_RESULT,path=str(RESULT.relative_to(P)),sha256=base.digest(RESULT)))
    plan['configure']=[arg.replace(str(ORIGINAL_ROOT/'build'),str(ROOT/'build')) for arg in previous['configure']]
    plan['configure'].append('-DSTRATA_CPU_SCAN_SOURCE='+str(P/'bench/sycl/cpu_roofline/main.cpp'))
    plan['build']=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target','cpu_roofline','--parallel','4']
    plan['protocol']=dict(previous['protocol'],
        question='Complete the 1/2/3/6-thread read-reference curve alongside explicitly reused exact-source V3 expert calibration.',
        gates='Reuse all original56 checks/112 cells/560 samples with full source/build/fixture/environment proof. Build only the new scanner. Require four bounded negative CLI checks,then all four read references with five eligible samples each. Stop first failure;no retries or sample replacement.',
        analysis='Per format/worker/group/mode use the median wall/pass sample and its matched GU/quant/down/cpu values. Divide by passes*100*group for per-job costs,weight formats by observed job counts;compare 1/2/3/6 threads and stream/reuse. Read rates are requested_bytes/wall with median/range/CV. Synthetic marginals and achieved read rates are not model latency,physical DRAM traffic,hardware saturation or a70TG ceiling.',
        continuation=reuse['scope'])
    ROOT.mkdir(mode=0o700);write(PLAN,plan)
    write(RESULT,dict(schema=SCHEMA,stage='prepared',source_commit=revision,reuse_origin=reuse,
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,
        plan_sha256=base.digest(PLAN),hardware_saturation_qualified=False,model_adoption_qualified=False))
    print('Prepared scanner-only continuation with560 original expert samples;commit plan before build.',flush=True)


def load():
    plan=gen.committed(PLAN)
    guards.require(plan['schema']==SCHEMA and
        base.source_pin(P,plan['source']['revision'],PREFIXES)==plan['source'] and
        base.source_pin(base.GGML,base.GGML_REV,('ggml',))==plan['ggml'] and
        environment()==plan['environment'] and base.machine()==plan['machine'],'source/environment/platform drift')
    for p,h in plan['tools'].items():guards.require(base.digest(p)==h,'compiler/tool drift')
    _,_,reuse=original_calibration()
    guards.require(reuse==plan['reuse'],'original expert reuse evidence differs')
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


def parse_scan(lines,threads):
    guards.require(type(threads) is int and threads in SCANS and len(lines)==7 and
        lines[-1]==dict(kind='closed',passed=True),'complete read-reference records required')
    timer=lines[0];samples=lines[1:-1]
    guards.require(timer['kind']=='timer_check' and timer['mode']=='scan' and timer['threads']==threads and
        all(type(timer[k]) in (int,float) and math.isfinite(timer[k]) for k in ('query_s','fastest_pass_s')) and
        timer['query_s']>=0 and timer['fastest_pass_s']>0 and
        timer['query_s']/timer['fastest_pass_s']<.001,'read-reference timer gate')
    for i,s in enumerate(samples):
        guards.require(s['kind']=='sample' and s['mode']=='scan' and s['threads']==threads and
            type(s['rep']) is int and s['rep']==i and s['attempt']==0 and
            type(s['iterations']) is int and 0<s['iterations']<1000000 and
            s['visits']==s['iterations'] and s['requested_bytes']==(256<<20)*s['iterations'] and
            all(type(s[k]) in (int,float) and math.isfinite(s[k]) and s[k]>=0 for k in ('wall_s','cpu_s')) and
            s['wall_s']>=1,'read-reference work/timing gate')
    return dict(threads=threads,timer=timer,samples=samples)


def scan_refusals(plan):
    first=plan['machine']['physical_cpus'][0]
    return [('duplicate',[first,first],'duplicate scan CPU'),
            ('negative',[-1],'scan CPU outside allowed affinity'),
            ('range',[1024],'scan CPU outside allowed affinity'),
            ('empty',[],'scan thread count')]


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
    for name in ('bench/sycl/cpu_decode_mix/main.cpp','bench/sycl/cpu_roofline/main.cpp'):
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
    required.add(str((P/'bench/sycl/cpu_roofline/main.cpp').resolve()))
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
    def execute(label,argv,wall,expected_exit=0):
        guards.require(time.monotonic()<deadline,'whole-stage budget expired')
        entry,_,_=own.run(stage+'-'+label,argv,plan['environment'],P,
            wall=min(wall,deadline-time.monotonic()),cpu=1200,rss_cap=plan['budgets']['rss_bytes'],
            text_cap=plan['budgets']['text_bytes'],total_cap=plan['budgets']['root_bytes'],
            file_cap=plan['budgets']['root_bytes'])
        guards.require(owner.closed(entry) and entry['exit_code']==expected_exit,'owned child failed: '+label)
        return entry
    try:
        own=owner.Owner(ROOT/'owned',persist)
        record['reuse_origin']=plan['reuse']
        if stage=='build':
            execute('configure',plan['configure'],120)
            execute('compile',plan['build'],plan['budgets']['build_seconds'])
            execute('dependencies',['/usr/bin/ninja','-C',str(ROOT/'build'),'-t','deps'],40)
            record.update(attest_build(plan,(ROOT/'owned/build-dependencies.stdout').read_text()))
            scanner=ROOT/'build/reference/cpu_roofline'
            record['binaries']=dict(mix=plan['reuse']['expert_binary'],
                scan=dict(path=str(scanner),sha256=base.digest(scanner)))
            for n,asset in [('scan',record['binaries']['scan'])]:
                execute('link-'+n,['/usr/bin/ldd',asset['path']],20)
                text=(ROOT/'owned'/('build-link-'+n+'.stdout')).read_text()
                guards.require(not re.search('sycl|level_zero|cuda|amdhip|libze|not found',text,re.I),
                    'GPU library linked or dynamic library missing')
        else:
            built=json.loads((ROOT/'build.json').read_text());verify_built(plan,built)
            record['binaries']=built['binaries']
            if stage=='check':
                for label,cpus,error in scan_refusals(plan):
                    execute(label,[built['binaries']['scan']['path'],'scan',','.join(map(str,cpus))],10,1)
                    prefix=ROOT/'owned'/('check-'+label)
                    guards.require(prefix.with_suffix('.stdout').read_bytes()==b'' and
                        prefix.with_suffix('.stderr').read_text()=='CPU calibration failed: '+error+'\n',
                        'scanner refusal mechanism differs')
                    record['children'].append(dict(label=label,cpus=cpus,error=error,exit_code=1))
                    persist()
            elif stage=='measure':
                checked=json.loads((ROOT/'check.json').read_text())
                guards.require(checked['passed'] and checked['complete'] and checked['ownership_closed'] and
                    not checked['active'] and checked['binaries']==built['binaries'] and
                    checked['plan_sha256']==base.digest(PLAN) and
                    checked['children']==[dict(label=l,cpus=c,error=e,exit_code=1) for l,c,e in scan_refusals(plan)],
                    'all four scanner refusal checks before timing')
                _,original,_=original_calibration()
                record['children']=original['children']
                for n in SCANS:
                    label='scan-'+str(n)
                    execute(label,[built['binaries']['scan']['path'],'scan',
                        ','.join(map(str,plan['machine']['physical_cpus'][:n]))],
                        plan['budgets']['scan_seconds'])
                    stdout=ROOT/'owned'/('measure-'+label+'.stdout')
                    guards.require((ROOT/'owned'/('measure-'+label+'.stderr')).read_bytes()==b'',
                        'unexpected scanner diagnostic')
                    lines=[json.loads(s) for s in stdout.read_text().splitlines()]
                    record.setdefault('read_reference',[]).append(parse_scan(lines,n))
                    persist();print('Closed read reference '+str(n)+' threads',flush=True)
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
