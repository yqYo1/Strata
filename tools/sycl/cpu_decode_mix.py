#!/usr/bin/env python3
"""CPU-only current-route batch/cache/NT calibration; Main serializes all stages."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import random
import re
import sys

sys.dont_write_bytecode=True
import cpu_roofline as base
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as owner
import query_profile as q

P,B,A=gen.P,gen.B,gen.A
ROOT=B/'cpu-decode-mix-v1'
PLAN=A/'cpu-roofline-plan.json'
RESULT=A/'cpu-roofline-current.json'
RESOURCE=A/'client-resources-current.json'
PREFIXES=(*base.PREFIXES,'bench/sycl/cpu_decode_mix','tools/sycl/cpu_decode_mix.py','tests/core/test_cpu_decode_mix.py')


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def qualification():
    full,count=q.current_build()
    resource=gen.committed(RESOURCE)
    guards.require(resource['passed'] and resource['complete'] and resource['ownership_closed'] and
        not resource['active'] and resource['stage']=='closed' and resource['fresh_faults']==[] and
        resource['binary']==full['binary'] and resource['build_source_commit']==full['source_commit'] and
        resource['artifact_retirement']['complete'] and resource['iq2_dispatch']['pair_rows']>0 and
        resource['census']['request']==5 and resource['requests'][0]['workload']=='regular-code' and
        gen.sample_eligible(resource['requests'][0]),'closed current post-IQ2 resource evidence required')
    return resource,dict(native_source=full['source_commit'],native_inputs_recovered=count,
        native_binary=full['binary'],native_fixture_count=len(full['fixtures']),
        component_sha256=base.digest(q.COMPONENT),resource_sha256=base.digest(RESOURCE))


def environment():
    return dict(base.environment(),STRATA_IQ2_PAIR='1',STRATA_IQ_SINGLE_EXACT='0',
        STRATA_IQ2_PAIR_AUDIT='0')


def prepare():
    guards.require(not ROOT.exists(),'fresh single-use calibration root required')
    old=gen.committed(RESULT)
    guards.require(old['complete'] and old['passed'] and old['ownership_closed'] and
        old['artifact_retirement']['complete'],'close/retire earlier calibration first')
    resource,proof=qualification();rows=base.formats()
    lookup={(r['gu'],r['down']):r for r in resource['census']['format_bins']}
    for row in rows:
        observed=lookup[row['gu'],row['down']]
        row['observed_jobs_NT']=observed['jobs_NT'][:4]
        row['experts']=max(g*math.ceil((128<<20)/(g*row['blob_bytes'])) for g in (6,28))
        # Each group uses its own extent. Pin the union of all read slices.
        row['fixture_sha256']=base.fixture_hash(row)
    revision=gen.git('rev-parse','HEAD').decode().strip()
    order=[(r['gu'],r['down'],w,g) for r in rows for w in (1,5) for g in (6,28)]
    random.Random(20261022).shuffle(order)
    root_recipe=['/usr/bin/cmake','-S',str(P/'bench/sycl/cpu_decode_mix'),'-B',str(ROOT/'build'),
        '-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
        '-DCMAKE_C_COMPILER='+str(base.LLVM/'bin/clang'),
        '-DCMAKE_CXX_COMPILER='+str(base.LLVM/'bin/clang++'),'-DSTRATA_GGML_DIR='+str(base.GGML)]
    plan=dict(schema='strata-cpu-decode-mix-v1',source=base.source_pin(P,revision,PREFIXES),
        ggml=base.source_pin(base.GGML,base.GGML_REV,('ggml',)),qualification=proof,
        environment=environment(),machine=base.machine(),formats=rows,order=order,
        tools={str(Path(p).resolve()):base.digest(p) for p in
            (base.LLVM/'bin/clang',base.LLVM/'bin/clang++','/usr/bin/cmake','/usr/bin/ninja')},
        metadata_sha256=base.digest(base.METADATA),model_identity={r['shard']:base.identity(r['shard']) for r in rows},
        previous_result=dict(commit=gen.git('log','-1','--format=%H','--',str(RESULT.relative_to(P))).decode().strip(),
            path=str(RESULT.relative_to(P)),sha256=base.digest(RESULT)),
        configure=root_recipe,build=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target',
            'cpu_decode_mix','cpu_roofline','--parallel','6'],
        protocol=dict(question='Does current decode CPU expert work show batch/cache-footprint or compute-thread sensitivity at the actual post-IQ2 NT/format mix?',
            groups=[6,28],workers=[1,5],computed_threads=[1,6],batches_per_pass=100,repetitions=5,
            sample_floor_seconds=1.0,target_seconds=1.3,timer_fraction_cap=.001,
            NT='Per-format measured NT1..4 histogram rounded by largest remainder to100*group jobs; deterministic shuffled schedule. This approximates that marginal distribution,not the complete layer/activation/working-set trace.',
            reuse='Reuse the same distinct group of6 or28 real expert blobs.28-expert weights may exceed32MiB L3;do not label it cache-resident.',
            stream='Rotate distinct groups through at least128MiB. A pass has100 batches and may revisit the extent;requested blob visits are not physical RAM traffic.',
            scan='Existing read-only256MiB checksum fixture,persistent1/6 threads,five>=1s samples. An achieved read reference,not an arithmetic or DRAM upper bound.',
            validation='Each loaded distinct expert at NT1..4:production pool bit-exact to serial production GU/quant/down,finite outputs,untouched inactive NT lanes,external canaries. Real distinct blobs detect job permutations. This is route consistency,not independent model/task validation.',
            observers='Native IQ2 pairing1,SINGLE_EXACT0,IQ2 audit0;fixed before process startup. Existing production pool subphase clocks only. No extra per-callback/thread clocks,no GPU runtime.',
            gates='All28 check children before measure. All56 expert cells and2 read cells with five individual eligible samples,actual placement/work/histogram conservation,normal ownership closure. Stop at first failure or budget,no sample replacement/retry.',
            interpretation='6/28 batches and reuse/stream change scheduling,cache and work together;thread scaling includes main quantization and waiting. Sensitivity may guide a later optimization;it does not establish arithmetic/RAM saturation,physical traffic,clean model speed or70tok/s.',
            retention='Keep compact per-cell individual timing/work samples,source/recipe/fixture hashes,numerical and ownership outcomes. Retire full output and this extra CPU build after source recovery and calibration decision;retain original6e40 engine+12fixtures.'),
        retained_generation='Keep original6e40 engine and12 fixtures at their original paths. Extra CPU-only build is solely for this imminent calibration and retires when it closes.',
        budgets=dict(build_seconds=600,check_child_seconds=120,measure_child_seconds=180,scan_seconds=40))
    ROOT.mkdir(mode=0o700);write(PLAN,plan)
    write(RESULT,dict(schema=plan['schema'],stage='prepared',source_commit=revision,
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,
        plan_sha256=base.digest(PLAN),hardware_saturation_qualified=False,model_adoption_qualified=False))
    print('Prepared28-child CPU mix calibration;commit exact plan before build.',flush=True)


def load():
    plan=gen.committed(PLAN)
    guards.require(plan['schema']=='strata-cpu-decode-mix-v1' and
        base.source_pin(P,plan['source']['revision'],PREFIXES)==plan['source'] and
        base.source_pin(base.GGML,base.GGML_REV,('ggml',))==plan['ggml'] and
        environment()==plan['environment'] and base.machine()==plan['machine'],'source/environment/platform drift')
    for p,h in plan['tools'].items():guards.require(base.digest(p)==h,'compiler/tool drift')
    guards.require(base.digest(base.METADATA)==plan['metadata_sha256'],'format metadata drift')
    resource,proof=qualification();guards.require(proof==plan['qualification'],'original native/resource proof changed')
    guards.require(len(plan['order'])==28 and set(map(tuple,plan['order']))==
        {(r['gu'],r['down'],w,g) for r in base.formats() for w in (1,5) for g in (6,28)},'full unique matrix required')
    observed={(b['gu'],b['down']):b['jobs_NT'][:4] for b in resource['census']['format_bins']}
    original={(r['gu'],r['down']):r for r in base.formats()}
    for row in plan['formats']:
        key=row['gu'],row['down']
        guards.require(all(row[k]==v for k,v in original[key].items() if k!='experts') and
            row['observed_jobs_NT']==observed[key] and
            row['experts']==max(g*math.ceil((128<<20)/(g*row['blob_bytes'])) for g in (6,28)),
            'format/offset/observed histogram differs')
        guards.require(base.identity(row['shard'])==plan['model_identity'][row['shard']] and
            base.fixture_hash(row)==row['fixture_sha256'],'selected expert input drift')
    return plan


def command(row,workers,group,stage):
    return [str(ROOT/'build/cpu_decode_mix'),row['shard'],str(row['gu']),str(row['down']),
        str(row['gate']),str(row['up']),str(row['down_offset']),str(row['blob_bytes']),str(workers),str(group),
        stage,*map(str,row['observed_jobs_NT']),'20261022']


def expected_histogram(hist,group):
    total=sum(hist);n=100*group
    result=[n*v//total for v in hist]
    order=sorted(range(4),key=lambda i:(-(n*hist[i]%total),i))
    for i in order[:n-sum(result)]:result[i]+=1
    return result


def parse(lines,row,workers,group,stage):
    guards.require(lines and lines[-1]==dict(kind='closed',passed=True),'missing native completion')
    placement=[r for r in lines if r['kind']=='placement']
    guards.require(placement==[dict(kind='placement',workers=workers,host_cpu=base.machine()['physical_cpus'][0],
        worker_cpus=base.machine()['physical_cpus'][1:workers+1],actual_affinity_verified=True)],'actual pool placement differs')
    work=[r for r in lines if r['kind']=='work'];valid=[r for r in lines if r['kind']=='validation']
    count=group*math.ceil((128<<20)/(group*row['blob_bytes']))
    guards.require(work==[dict(kind='work',workers=workers,host_works=workers==5,group=group,batches=100,
        unique_stream_experts=count,blob_bytes=row['blob_bytes'],jobs_NT=expected_histogram(row['observed_jobs_NT'],group),phase='decode')],
        'actual calibration work/dispatch differs')
    want=dict(kind='validation',distinct_experts=count,NT_mask=30,pool_serial_bits_equal=True,
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


def run(stage):
    plan=load();gen.committed(RESULT)
    path=ROOT/(stage+'.json');guards.require(not path.exists(),'single-use stage;inspect existing owner,never retry')
    record=dict(stage=stage,source_commit=plan['source']['revision'],plan_sha256=base.digest(PLAN),
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,children=[])
    own=None
    def persist():
        record['active']=bool(own and own.active is not None)
        write(path,dict(record,commands=own.commands if own else []))
    own=owner.Owner(ROOT/'owned',persist)
    def execute(label,argv,wall):
        entry,_,_=own.run(stage+'-'+label,argv,plan['environment'],P,wall=wall,cpu=1200,
            rss_cap=3<<30,text_cap=64<<20,total_cap=64<<20,file_cap=64<<20)
        guards.require(owner.closed(entry) and entry['exit_code']==0,'owned child failed: '+label)
        return entry
    try:
        if stage=='build':
            execute('configure',plan['configure'],120);execute('compile',plan['build'],600)
            record['binaries']={n:dict(path=str(ROOT/'build'/p),sha256=base.digest(ROOT/'build'/p)) for n,p in
                (('mix','cpu_decode_mix'),('scan','reference/cpu_roofline'))}
            for n,v in record['binaries'].items():
                execute('link-'+n,['/usr/bin/ldd',v['path']],20)
                text=(ROOT/'owned'/('build-link-'+n+'.stdout')).read_text()
                guards.require(not re.search('sycl|level_zero|cuda|amdhip|libze',text,re.I),'GPU library linked')
            record['compile_commands_sha256']=base.digest(ROOT/'build/compile_commands.json')
        else:
            built=json.loads((ROOT/'build.json').read_text());guards.require(built['passed'] and built['complete'] and built['ownership_closed'] and built['plan_sha256']==base.digest(PLAN),'closed exact build required')
            record['binaries']=built['binaries']
            for v in record['binaries'].values():guards.require(base.digest(v['path'])==v['sha256'],'binary drift')
            if stage=='measure':
                checked=json.loads((ROOT/'check.json').read_text());guards.require(checked['passed'] and checked['complete'] and checked['ownership_closed'] and checked['binaries']==built['binaries'] and len(checked['children'])==28,'all28 numerical cases before timing')
            lookup={(r['gu'],r['down']):r for r in plan['formats']}
            for i,(gu,dn,w,g) in enumerate(plan['order']):
                row=lookup[gu,dn];label='cell-'+str(i)
                entry=execute(label,command(row,w,g,stage),plan['budgets'][stage+'_child_seconds'])
                stdout=ROOT/'owned'/(stage+'-'+label+'.stdout')
                values=parse([json.loads(s) for s in stdout.read_text().splitlines()],row,w,g,stage)
                record['children'].append(dict(pair=[gu,dn],workers=w,group=g,**values,exit_code=entry['exit_code'],stdout_sha256=base.digest(stdout)))
                persist();print('Closed '+stage+' '+str(i+1)+'/28',flush=True)
            if stage=='measure':
                for n in (1,6):
                    label='scan-'+str(n);execute(label,[built['binaries']['scan']['path'],'scan',','.join(map(str,plan['machine']['physical_cpus'][:n]))],40)
                    lines=[json.loads(s) for s in (ROOT/'owned'/('measure-'+label+'.stdout')).read_text().splitlines()]
                    guards.require(lines[-1]==dict(kind='closed',passed=True),'read reference incomplete')
                    samples=[s for s in lines if s['kind']=='sample'];guards.require(len(samples)==5 and
                        [s['rep'] for s in samples]==list(range(5)) and all(s['threads']==n and s['wall_s']>=1 for s in samples),'read reference samples')
                    record.setdefault('read_reference',[]).append(dict(threads=n,samples=samples))
        # Complete input/owner checks remain charged to the stage, outside timing.
        load();record.update(passed=True,complete=True)
    except BaseException as error:
        record['error']=str(error)
    finally:
        record['ownership_closed']=own.active is None and all(owner.closed(c) for c in own.commands)
        record['passed']=bool(record['passed'] and record['ownership_closed'])
        persist();write(RESULT,record)
    print(json.dumps({k:record.get(k) for k in ('stage','passed','complete','ownership_closed','error')}),flush=True)
    return 0 if record['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','build','check','measure'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        # This bounded CPU build never overwrites the retained engine generation.
        return prepare() if args.stage=='prepare' else run(args.stage)


if __name__=='__main__':sys.exit(main())
