#!/usr/bin/env python3
"""Main-owned isolated production Q6_K subset-head calibration; no engine edits."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

sys.dont_write_bytecode=True
import cpu_roofline as source
import cpu_census_engine as faults
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as owner
import mtp_model_observer as context
import query_profile as q
import ur_validation as validation
sys.path.insert(0,str(gen.P/'tools'))
from gguf_reader import GGUFFile

P,B,A=gen.P,gen.B,gen.A
ROOT=B/'mtp-q6-head-v1'
PLAN=A/'mtp-head-plan.json'
RESULT=A/'mtp-head-current.json'
PREFIXES=('CMakeLists.txt','cmake','src','include','third_party','tools','bench/sycl/mtp_head','tests/core/test_mtp_head_calibration.py')
ORDER=(0,1,1,0,0,1)


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def qualification():
    native,count=q.current_build();setup=context.original_setup()
    settings=gen.committed(gen.SETTINGS);delta=gen.committed(q.COMPONENT)['runtime_asset_delta']
    guards.require(set(delta)<=set(settings['runtime_elfs']),'unknown runtime delta')
    actual={}
    for p,item in settings['runtime_elfs'].items():
        proposed=delta.get(p)
        if proposed:guards.require(proposed['qualified_sha256']==item['sha256'],'changed qualified runtime baseline')
        expected=proposed['candidate_sha256'] if proposed else item['sha256']
        actual[p]=source.digest(p);guards.require(actual[p]==expected,'runtime changed '+p)
    return native,setup,dict(native_source=native['source_commit'],native_binary=native['binary'],
        original_native_inputs_recovered=count,component_sha256=source.digest(q.COMPONENT),
        settings_sha256=source.digest(gen.SETTINGS),runtime_sha256=actual,
        native_validation_preload=validation.preload_qualification())


def head_input(setup):
    args=setup['config']['args'];shard=Path(args[args.index('--native')+1]);vocab=Path(args[args.index('--mtp')+1])/'draft_vocab.bin'
    before=source.identity(shard);gguf=GGUFFile(shard);found=[t for t in gguf.tensors if t.name=='output.weight']
    guards.require(len(found)==1,'unique output head required');t=found[0]
    guards.require(t.type_id==14 and t.shape==[2560,248320] and t.expected_bytes()==521472000,'actual Q6_K head geometry differs')
    raw=vocab.read_bytes();guards.require(len(raw)==425196,'draft map size')
    ids=struct.unpack('<106299i',raw);guards.require(min(ids)>=0 and max(ids)<248320 and len(set(ids))==len(ids),'draft map IDs')
    offset=gguf.data_start+t.offset;guards.require(offset+t.expected_bytes()<=before['bytes'],'head tensor extent')
    h=hashlib.sha256()
    with shard.open('rb') as f:
        header=hashlib.sha256(f.read(gguf.data_start)).hexdigest();f.seek(offset);left=t.expected_bytes()
        while left:
            block=f.read(min(left,8<<20));guards.require(block,'short tensor');h.update(block);left-=len(block)
    guards.require(source.identity(shard)==before,'head input changed during digest')
    return dict(shard=str(shard),identity=before,header_sha256=header,offset=offset,type=14,N=2560,
        full_rows=248320,rows=106299,row_bytes=2100,selected_bytes=223227900,
        full_head_sha256=h.hexdigest(),vocab=str(vocab),vocab_sha256=hashlib.sha256(raw).hexdigest())


def environment(setup,label,traced=False,clean=False):
    old=setup['environment']['STRATA_MEASUREMENT_ROOT'];root=ROOT/'jobs'/label
    env={k:v.replace(old,str(root)) for k,v in setup['environment'].items()}
    for k in ('UR_ENABLE_LAYERS','UR_LOG_TRACING','STRATA_TRACE','STRATA_DECODE_TIMING',
              'STRATA_CPU_WORK_CENSUS','STRATA_IQ2_PAIR_AUDIT','STRATA_SYCL_MTP_OBSERVER','LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER'):
        env.pop(k,None)
    if not clean:
        probe=validation.preload_qualification()
        env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'])
    for k in ('UR_LOG_VALIDATION','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO'):env[k]='level:warning;flush:warning;output:stderr'
    if traced:env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',UR_LOG_TRACING='level:info;flush:info;output:stderr')
    expected=gen.committed(gen.SETTINGS)['environment']
    for k in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[k]==expected[k],'qualified safety setting changed')
    return env


def prepare():
    guards.require(not ROOT.exists() and not PLAN.exists() and not RESULT.exists(),'fresh single-use head calibration required')
    native,setup,proof=qualification();head=head_input(setup);revision=gen.git('rev-parse','HEAD').decode().strip()
    configure=list(native['configure_argv']);configure[configure.index('-S')+1]=str(P/'bench/sycl/mtp_head');configure[configure.index('-B')+1]=str(ROOT/'build')
    plan=dict(schema='strata-mtp-q6-head-v1',source=source.source_pin(P,revision,PREFIXES),qualification=proof,
        head=head,machine=source.machine(),configure=configure,
        build=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target','mtp_head','--parallel','6'],
        tools={str(Path(p).resolve()):source.digest(p) for p in (native['compiler']['path'],'/usr/bin/cmake','/usr/bin/ninja')},
        environment={label:environment(setup,label,traced=label=='tiny',clean=label=='measure') for label in ('build','tiny','full','measure')},
        protocol=dict(question='Can the actual compressed Q6_K subset head explain a dominant fraction of the31.208ms/window MTP graph envelope?',
            route='Unchanged production native_quantize_q8_1 + native_mmvq(type14,N2560,rows106299,NT1),ordinary graph capture and per-replay checked drain. Small17-row check before full shape.',
            numerical='Full selected real GGUF rows in original draft-map order. Independently reproduce Q8_1 bytes;dequantized Q6_K double sum scaled error<=3e-6 for every row. Direct/graph output bits,input/scratch/weight/output guards before and after timing.',
            order=list(ORDER),repetitions_per_process=5,sample_floor_s=1.0,target_s=1.3,timer_fraction_cap=.001,
            profiling='Separate ordinary/profiled graph processes in0/1/1/0/0/1 order. Profiled processes retain64 qualified head envelopes outside wall-timed batches. Envelopes include the two nodes and gaps;not physical bandwidth or arithmetic saturation.',
            interpretation='Compare3*median isolated head envelope with31.208297484375ms/window. Below10% disfavors isolated-head dominance;at least50% merits an in-model discriminator. Neither threshold proves the in-model share:graph context,cache/interference and profiling effects differ. No inference adoption or precision change.',
            retention='Keep exact source/recipe/input and binary identities,individual five wall/CPU samples and64 envelope durations per profiled process,numerical/fault/ownership checks and decision. Retire this auxiliary build and normal logs after source recovery and decision;keep exact6e40 engine+12fixtures.'),
        retained_generation='One original qualified6e40 engine+12fixtures. This extra isolated-head build has only the current calibration consumer and retires at closure.',
        budgets=dict(build_seconds=1800,tiny_seconds=120,full_seconds=180,measure_seconds=180,text_bytes=64<<20,artifact_bytes=2<<30,RSS_bytes=10<<30))
    ROOT.mkdir(mode=0o700);write(PLAN,plan);write(RESULT,dict(stage='prepared',source_commit=revision,passed=False,complete=False,active=False,ownership_closed=True,plan_sha256=source.digest(PLAN)))
    print('Prepared isolated Q6_K head calibration;commit plan before build.',flush=True)


def load():
    plan=gen.committed(PLAN);native,setup,proof=qualification()
    guards.require(plan['schema']=='strata-mtp-q6-head-v1' and proof==plan['qualification'] and
        source.source_pin(P,plan['source']['revision'],PREFIXES)==plan['source'] and
        head_input(setup)==plan['head'] and source.machine()==plan['machine'],'source/model/native/platform drift')
    for p,h in plan['tools'].items():guards.require(source.digest(p)==h,'compiler/tool drift')
    for label in ('build','tiny','full','measure'):
        guards.require(environment(setup,label,traced=label=='tiny',clean=label=='measure')==plan['environment'][label],'environment drift')
    return plan


def parse(lines,rows,profile,mode):
    guards.require(lines and lines[-1]==dict(kind='closed',profile=profile,passed=True,checked_retirement=True),'native completion missing')
    ref=[r for r in lines if r['kind']=='reference'];valid=[r for r in lines if r['kind']=='validation']
    device=[r for r in lines if r['kind']=='device']
    guards.require(len(device)==1 and 'B570' in device[0]['name'] and
        type(device[0]['global_bytes']) is int and 9<<30<=device[0]['global_bytes']<=16<<30,'wrong native GPU')
    guards.require(len(ref)==1 and set(ref[0])=={'kind','rows','type','N','NT','weight_bytes','map_unique','Q8_1_bytes_equal','max_scaled_error'},'reference framing')
    expected=dict(kind='reference',rows=rows,type=14,N=2560,NT=1,weight_bytes=rows*2100,map_unique=True,Q8_1_bytes_equal=True)
    guards.require({k:v for k,v in ref[0].items() if k!='max_scaled_error'}==expected and
        math.isfinite(ref[0]['max_scaled_error']) and 0<=ref[0]['max_scaled_error']<=3e-6,'numerical reference failed')
    guards.require(valid==[dict(kind='validation',graph_direct_bits_equal=True,all_guards=True,inputs_unchanged=True)]*2,'numerical/bounds/owner validation incomplete')
    clocks=[r for r in lines if r['kind']=='envelope'];samples=[r for r in lines if r['kind']=='sample'];timers=[r for r in lines if r['kind']=='timer']
    guards.require(len(lines)==5+len(clocks)+len(samples)+len(timers) and
        len(clocks)==((64 if mode=='measure' else 2) if profile else 0),'unknown records or clocks')
    previous=0
    for i,t in enumerate(clocks):
        guards.require(t['rep']==i and all(type(t[k]) is int and t[k]>=0 for k in ('submit','start','end','resolution')) and
            t['resolution']>0 and t['end']>t['start']>=previous and t['end']-t['start']>=t['resolution'] and
            all(type(t[k]) in (int,float) and math.isfinite(t[k]) and t[k]>=0 for k in ('wall_s','query_s')) and t['query_s']<1,'invalid graph clock')
        previous=t['end']
    if mode=='check':guards.require(not samples and not timers,'timed samples in check');return dict(device=device[0],reference=ref[0],validated=True,graph_envelopes_ns=[t['end']-t['start'] for t in clocks])
    guards.require(len(timers)==1 and timers[0]['fastest_replay_s']>0 and
        0<=timers[0]['query_s']/timers[0]['fastest_replay_s']<.001,'timer overhead gate')
    guards.require(len(samples)==5 and [s['rep'] for s in samples]==list(range(5)),'five samples required')
    for s in samples:guards.require(type(s['replays']) is int and 0<s['replays']<1000000 and
        all(type(s[k]) in (int,float) and math.isfinite(s[k]) and s[k]>=0 for k in ('wall_s','cpu_s')) and s['wall_s']>=1,'invalid/short sample')
    return dict(device=device[0],reference=ref[0],validated=True,graph_envelopes_ns=[t['end']-t['start'] for t in clocks],
        max_query_s=max((t['query_s'] for t in clocks),default=0),timer_fraction=timers[0]['query_s']/timers[0]['fastest_replay_s'],samples=samples)


def run(stage):
    plan=load();gen.committed(RESULT);path=ROOT/(stage+'.json');guards.require(not path.exists(),'single-use stage;inspect original owner,never retry')
    r=dict(stage=stage,source_commit=plan['source']['revision'],plan_sha256=source.digest(PLAN),passed=False,complete=False,active=False,ownership_closed=True,checks=[])
    own=None
    def persist():r['active']=bool(own and own.active is not None);write(path,dict(r,commands=own.commands if own else []))
    own=owner.Owner(ROOT/'owned',persist)
    def execute(label,argv,wall):
        old=ROOT/'jobs'/stage;root=ROOT/'jobs'/(stage+'-'+label);root.mkdir(parents=True,exist_ok=True)
        env={k:v.replace(str(old),str(root)) for k,v in plan['environment'][stage].items()}
        entry,_,_=own.run(stage+'-'+label,argv,env,P,wall=wall,cpu=2400,
            rss_cap=plan['budgets']['RSS_bytes'],text_cap=64<<20,total_cap=2<<30,file_cap=(2<<30 if stage=='build' else 64<<20))
        guards.require(owner.closed(entry) and entry['exit_code']==0,'owned child failed '+label)
        return entry
    try:
        if stage=='build':
            execute('configure',plan['configure'],120);execute('compile',plan['build'],1800)
            binary=ROOT/'build/mtp_head';r.update(binary=dict(path=str(binary),sha256=source.digest(binary)),compile_commands_sha256=source.digest(ROOT/'build/compile_commands.json'))
        else:
            built=json.loads((ROOT/'build.json').read_text());guards.require(built['passed'] and built['complete'] and built['ownership_closed'] and built['plan_sha256']==source.digest(PLAN),'closed exact build required')
            r['binary']=built['binary'];guards.require(source.digest(r['binary']['path'])==r['binary']['sha256'],'binary drift')
            prior={'tiny':'build','full':'tiny','measure':'full'}[stage];gate=json.loads((ROOT/(prior+'.json')).read_text())
            guards.require(gate['passed'] and gate['complete'] and gate['ownership_closed'] and gate.get('binary')==r['binary'],'preceding gate failed')
            boot,cursor=faults.start_fault_scope();r['fresh_faults']=[]
            for i,profile in enumerate(ORDER if stage=='measure' else (0,1)):
                rows=17 if stage=='tiny' else 106299;mode='measure' if stage=='measure' else 'check';h=plan['head']
                argv=[r['binary']['path'],h['shard'],str(h['offset']),h['vocab'],str(rows),mode,str(profile)]
                entry=execute(str(i),argv,plan['budgets'][stage+'_seconds']);log=ROOT/'owned'/(stage+'-'+str(i)+'.stdout')
                parsed=parse([json.loads(s) for s in log.read_text().splitlines()],rows,profile,mode)
                stderr=log.with_suffix('.stderr').read_text(errors='replace')
                if stage!='measure':
                    parsed['native_validation']=validation.preload_evidence(stderr,entry['direct_identity']['pid'])
                    boundary=re.search(r'^strata UR validation: .*phase=before-main[^\n]*\n',stderr,re.M)
                    guards.require(boundary is not None and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|uncertain retirement',stderr[boundary.end():]),'runtime/validation error after native admission')
                evidence=faults.fault_evidence(cursor,boot);guards.require(evidence['boot_unchanged'] and not evidence['fresh_faults'],'fresh device fault/boot change')
                r['checks'].append(dict(index=i,profile=profile,**parsed,**evidence));persist();print('Closed '+stage+' '+str(i+1),flush=True)
        load();r.update(passed=True,complete=True)
    except BaseException as e:r['error']=str(e)
    finally:
        r['ownership_closed']=own.active is None and all(owner.closed(x) for x in own.commands);r['passed']=bool(r['passed'] and r['ownership_closed']);persist();write(RESULT,r)
    print(json.dumps({k:r.get(k) for k in ('stage','passed','complete','ownership_closed','error')}),flush=True)
    return 0 if r['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','build','tiny','full','measure'));args=p.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare() if args.stage=='prepare' else run(args.stage)


if __name__=='__main__':sys.exit(main())
