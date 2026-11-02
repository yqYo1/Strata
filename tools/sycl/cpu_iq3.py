#!/usr/bin/env python3
"""Main-owned, CPU-only IQ3_S production GU qualification; repository LICENSE.

prepare -> commit plan -> build -> commit -> check -> commit -> measure.
Single-use stages; no changed hashes, hidden retries or GPU execution.
"""
import fcntl
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import cpu_roofline as base
import measurement_guards as guards
import measurement_owner as owner

P,B=base.P,base.B
ROOT=B/'cpu-iq3-gu-v1'
RESULT=base.A/'cpu-iq3-current.json'
PREFIXES=(*base.PREFIXES,'bench/sycl/cpu_iq3','tools/sycl/cpu_iq3.py','tools/sycl/measurement_guards.py')
ORDER=[(0,0),(0,1),(1,1),(1,0),(2,0),(2,1)]
COLUMNS=['process','arm','down','access','rep','passes','calls','wall_s','cpu_s','timer_s','warm_s','selected_calls','selected_rows']


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def prepare():
    base.need(not ROOT.exists() and not RESULT.exists(),'single-use IQ3 root/result already exists')
    revision=base.git(P,'rev-parse','HEAD').decode().strip()
    source=base.source_pin(P,revision,PREFIXES)
    forms=[r for r in base.formats() if r['gu']==21]
    for r in forms:
        r['experts']=64;r['fixture_sha256']=base.fixture_hash(r)
    base.need([r['down'] for r in forms]==[20,42] and all(not any(c.isspace() for c in r['shard'])for r in forms),'exact two model forms')
    text=''.join(f"{r['shard']} {r['down']} {r['gate']} {r['up']} {r['down_offset']} {r['blob_bytes']} 64\n"for r in forms)
    ROOT.mkdir();fixture=ROOT/'inputs.txt';fixture.write_text(text)
    machine=base.machine()
    plan=dict(source=source,ggml=base.source_pin(base.GGML,base.GGML_REV,('ggml',)),
        tools={str(Path(x).resolve()):base.digest(x)for x in (base.LLVM/'bin/clang',base.LLVM/'bin/clang++','/usr/bin/cmake','/usr/bin/ninja',sys.executable)},
        environment=dict(base.environment(),STRATA_IQ_SINGLE_EXACT='0',STRATA_IQ_MT_MIN='2',STRATA_IQ3_PAIR_AUDIT='1'),
        machine=machine,core=machine['physical_cpus'][1],formats=forms,metadata_sha256=base.digest(base.METADATA),
        model_identity={r['shard']:base.identity(r['shard'])for r in forms},fixture=dict(path=str(fixture),sha256=base.digest(fixture),text=text),
        configure=['/usr/bin/cmake','-S',str(P/'bench/sycl/cpu_iq3'),'-B',str(ROOT/'build'),'-G','Ninja',
            '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_C_COMPILER='+str(base.LLVM/'bin/clang'),
            '-DCMAKE_CXX_COMPILER='+str(base.LLVM/'bin/clang++'),'-DSTRATA_GGML_DIR='+str(base.GGML)],
        build=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target','cpu_iq3_gu','--parallel','6'],order=ORDER,
        protocol=dict(question='Does sharing activation loads across IQ3_S gate/up improve the actual Decode NT1 GU dispatcher on this Zen3 CPU?',
            scope='GU only,H2560/FF640,64real experts for each21/20and21/42form. Calls native_gu_rows,including selector/fallback,swish and optional selected-route audit. Does not call ExpertPool,quantize hidden states or execute Down. In particular,the newer pool.cpp/pool.hpp/service_profile.hpp are not a measured model baseline.',
            arms='Same binary,immutable pre-exec STRATA_IQ3_PAIR=0/1. Three check processes:reference,candidate,candidate with STRATA_NO_IQ256=1. No setenv/unsetenv after startup.',
            conformance='Both forms,64experts x4actual Q8_K-quantized finite activations x2validations/process. Compare Decode NT1 against explicit Prefill GGML reference bitwise,finite outputs and guard rows;test Prefill,NT2,NT4 leave new dispatch counters unchanged. Direct producer checks empty/partial/padded ranges,invalid type/geometry/alignment/null/span/overlap/uintptr-overflow,and synthetic finite F16/grid/high-bit/sign/scale boundaries. Inputs immutable. Exactness is this candidate design choice,not a general FP32 adoption requirement.',
            dispatch='Audit=1 in both arms;selected calls/rows must match every timed NT1 invocation only in candidate. Candidate counter overhead remains inside its measured entry. Pure producer does not increment selection counters;fallback does not select. Each process is actually pinned to one physical core.',
            access='hot:one full expert per pass;stream:64distinct real expert blobs per pass. First GU footprint1.408MB,stream90.112MB per form;requested/logical footprints are not physical RAM traffic or saturation. No model/cache/MTP mix projection.',
            repetition='A0/B0/B1/A1/A2/B2 independent processes,2forms x2access x6batches/process. Whole passes until0.35s,min0.25s,max1000000passes. No retries or additional samples under this plan.',
            clock='Per-pass clock probe below0.1%of one warm pass. Setup,model IO,quantization,references,input checks and snapshots outside timing. Production GU dispatch and math inside. Keep per-batch wall and process CPU seconds.',
            gate='Decide each Down-form/access cell separately from3paired independent-process medians. Repeatable gain if all3candidate/reference ratios<1;no minimum percentage. Format eligible for model follow-up only if both access cells show repeatable gain. Mixed signs remain inconclusive;no changed criterion or added repeats without a new committed plan. Component success alone is not model speed/adoption.',
            stop='First source,ownership,numerical,dispatch,timing or budget failure closes that stage as failed;no retry. No GPU test on a failed component performance gate.',
            provenance='Main-authored fixture/controller uses only existing Strata and linked ggml references. Official installed LLVM happens to reside under an old external worktree;no XeStrata source,helper,test or build recipe is an input.'),
        budgets=dict(root_bytes=512<<20,owned_total_bytes=64<<20,per_file_bytes=16<<20,rss_bytes=3<<30,
            child_wall_seconds=120,child_cpu_seconds=120,build_wall_seconds=600,build_cpu_seconds=3600),
        retention=dict(current_known_good_model='cache-map-model-v1/strata:552b1616d21ad1b9aa6a317cb940d375c003246b896a255a935e09292804c96a',
            model_fallback='adaptive-cache-model-v1/strata:bdba183e;existing entry-proof consumer must be redirected before another full engine generation.',
            component_consumer='This new CPU-only fixture serves the current GU comparison. Does not overwrite either model generation. Retire successful logs and build objects after committed result/provenance;retain fixture only for a concrete next model qualification consumer.',
            permanent='Source/recipe,pins,individual time samples,correctness/dispatch counts,cell decisions and limitations. No tensor captures,full IO histories or successful compiler output.'))
    write(RESULT,dict(schema='strata-iq3-gu-v1',stage='prepared',active=False,ownership_closed=True,passed=False,complete=False,plan=plan))
    print('Prepared IQ3 GU plan;commit it before build.')


def load():
    base.need(RESULT.read_bytes()==base.git(P,'show','HEAD:'+str(RESULT.relative_to(P))),'commit current component state first')
    r=json.loads(RESULT.read_text());p=r['plan']
    base.need(base.source_pin(P,p['source']['revision'],PREFIXES)==p['source'],'project source changed')
    base.need(base.source_pin(base.GGML,base.GGML_REV,('ggml',))==p['ggml'],'ggml changed')
    base.need(base.machine()==p['machine'] and all(base.digest(f)==h for f,h in p['tools'].items()),'machine/tool changed')
    base.need(base.digest(base.METADATA)==p['metadata_sha256'],'metadata changed')
    base.need(Path(p['fixture']['path']).read_text()==p['fixture']['text'] and base.digest(p['fixture']['path'])==p['fixture']['sha256'],'generated input changed')
    for f in p['formats']:base.need(base.identity(f['shard'])==p['model_identity'][f['shard']] and base.fixture_hash(f)==f['fixture_sha256'],'model bytes changed')
    return r


def decisions(samples):
    expected={(proc,arm,down,access,rep)for proc in range(3)for arm in range(2)for down in (20,42)for access in range(2)for rep in range(6)}
    keys=[tuple(s[k]for k in COLUMNS[:5])for s in samples]
    base.need(len(keys)==len(set(keys))==len(expected) and set(keys)==expected,'complete unique144sample matrix')
    for s in samples:
        calls=s['passes']*(64 if s['access']else 1)
        base.need(s['wall_s']>=.25 and 0<s['passes']<1000000 and s['calls']==calls and 0<s['timer_s']<s['warm_s']*.001,'timing/count gate')
        base.need(s['selected_calls']==(calls if s['arm']else 0) and s['selected_rows']==(calls*640 if s['arm']else 0) and s['cpu_s']>=0,'actual dispatch gate')
    values={tuple(s[k]for k in COLUMNS[:5]):s['wall_s']/s['calls'] for s in samples}
    rows=[]
    for down in (20,42):
        for access in range(2):
            pairs=[]
            for proc in range(3):
                reference=statistics.median(values[proc,0,down,access,rep]for rep in range(6));candidate=statistics.median(values[proc,1,down,access,rep]for rep in range(6))
                pairs.append(dict(process=proc,reference_s=reference,candidate_s=candidate,ratio=candidate/reference))
            ratios=[p['ratio']for p in pairs];rows.append(dict(down=down,access=access,pairs=pairs,geomean_ratio=statistics.geometric_mean(ratios),repeatable_gain=all(v<1 for v in ratios)))
    return rows


def execute(stage):
    record=load();p=record['plan'];base.need(record['stage']=={'build':'prepared','check':'built','measure':'checked'}[stage] and not record['active'],'wrong stage;no retries')
    guards.no_model_owners(set(p['model_identity']))
    receipt=ROOT/(stage+'.json');base.need(not receipt.exists(),'stage already attempted')
    state=dict(stage=stage,passed=False,complete=False,active=False,commands=[])
    def persist():state['active']=own.active is not None;write(receipt,state)
    own=owner.Owner(ROOT/'owned',persist);state['commands']=own.commands
    def run(label,argv,env,wall=120,cpu=120):
        entry,out,err=own.run(stage+'-'+label,argv,env,P,wall=wall,cpu=cpu,rss_cap=p['budgets']['rss_bytes'],text_cap=16<<20,file_cap=16<<20,total_cap=64<<20)
        base.need(owner.closed(entry) and entry['exit_code']==0,'child failed:'+label+':'+err.read_text()[-4000:])
        base.need(owner.artifact_bytes(ROOT)<=p['budgets']['root_bytes'],'root byte budget')
        return out
    try:
        if stage=='build':
            run('configure',p['configure'],p['environment'],120,600);run('compile',p['build'],p['environment'],600,3600)
            binary=ROOT/'build/cpu_iq3_gu';record['binary']=dict(path=str(binary),bytes=binary.stat().st_size,sha256=base.digest(binary))
            output=run('ldd',['/usr/bin/ldd',str(binary)],p['environment'],20,20).read_text();base.need(not any(k in output.lower()for k in ('sycl','level_zero','cuda','amdhip','libze')),'GPU runtime linked')
            database=ROOT/'build/compile_commands.json';entries=json.loads(database.read_text())
            base.need(entries and all(any(Path(e['file']).resolve().is_relative_to(root)for root in (P,base.GGML,ROOT))for e in entries),'unqualified configured source root')
            record['compile_commands_sha256']=base.digest(database);record['configured_compile_entries']=len(entries)
            record['linked_runtime']={line.split('=>')[1].strip().split()[0]:None for line in output.splitlines()if '=> /'in line}
            record['linked_runtime']={name:base.digest(name)for name in record['linked_runtime']}
        else:
            base.need(base.digest(record['binary']['path'])==record['binary']['sha256'] and all(base.digest(f)==h for f,h in record['linked_runtime'].items()),'binary/runtime changed')
            samples=[];proof=[];sequence=[(proc,arm,0)for proc,arm in ORDER]if stage=='measure'else[(0,0,0),(0,1,0),(0,1,1)]
            for index,(proc,arm,blocked)in enumerate(sequence):
                env=dict(p['environment'],STRATA_IQ3_PAIR=str(arm))
                if blocked:env['STRATA_NO_IQ256']='1'
                out=run(str(index),[record['binary']['path'],'perf'if stage=='measure'else'check',p['fixture']['path'],str(arm),str(proc),str(p['core']),str(blocked)],env)
                rows=[json.loads(line)for line in out.read_text().splitlines()];terminal=rows[-1]
                base.need(terminal['kind']=='closed' and terminal['passed'] and terminal['contract_cases']==(74 if stage=='check'else 0),'native terminal/contract')
                validations=[v for v in rows if v['kind']=='validation'];base.need(len(validations)==2 and [v['down']for v in validations]==[20,42] and all(v['bitwise'] and v['phase_isolated'] and v['passed'] and v['experts']==64 and v['activations']==4 and v['core']==p['core']for v in validations),'full conformance/placement')
                proof.append(dict(process=proc,arm=arm,blocked=blocked,validations=validations,contracts=terminal['contract_cases']))
                samples.extend(v for v in rows if v['kind']=='sample');print(f'Closed {stage} process{index+1}/{len(sequence)}',flush=True)
            record[stage+'_evidence']=proof
            if stage=='measure':
                record['decisions']=decisions(samples);record['sample_columns']=COLUMNS;record['samples']=[[v[k]for k in COLUMNS]for v in samples]
                record['eligible_down_formats']=[down for down in (20,42)if all(c['repeatable_gain']for c in record['decisions']if c['down']==down)]
                record['performance_gate_passed']=bool(record['eligible_down_formats']);record['model_qualified']=False;record['inference_adopted']=False
        load();state.update(passed=True,complete=True)
    except BaseException as error:state['error']=str(error)
    finally:
        state.update(active=own.active is not None,ownership_closed=own.active is None and all(owner.closed(c)for c in own.commands));persist()
        record.update(active=state['active'],ownership_closed=state['ownership_closed'],passed=state['passed'],complete=state['complete'],stage={'build':'built','check':'checked','measure':'closed'}[stage]if state['passed']else'failed')
        if 'error'in state:record['error']=state['error']
        write(RESULT,record)
    print(json.dumps({k:record.get(k)for k in ('stage','passed','complete','ownership_closed','performance_gate_passed','error')}));return 0 if state['passed']else 1


if __name__=='__main__':
    base.need(len(sys.argv)==2 and sys.argv[1]in ('prepare','build','check','measure'),'prepare/build/check/measure')
    with (B/'owned-v0141-measurement.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if sys.argv[1]=='prepare':prepare()
        else:sys.exit(execute(sys.argv[1]))
