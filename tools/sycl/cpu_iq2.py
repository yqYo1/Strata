#!/usr/bin/env python3
"""Main-owned, CPU-only IQ2_S NT1 production-pool qualification. Single-use stages."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import cpu_roofline as base
import measurement_guards as guards
import measurement_owner as owner

P, B = base.P, base.B
ROOT = B/'cpu-iq2-pool-v3'
RESULT = base.A/'cpu-iq2-current.json'
PREFIXES = (*base.PREFIXES, 'bench/sycl/cpu_iq2', 'tools/sycl/cpu_iq2.py',
    'tools/sycl/measurement_guards.py', 'tests/core/test_cpu_iq2.py')
ORDER = [(0,0),(0,1),(1,1),(1,0),(2,0),(2,1)]
COLUMNS = ['process','arm','down','mixture','access','rep','passes','batch_calls','wall_s','cpu_s',
    'GU_s','quant_s','Down_s','row_ops','NT1_rows','paired_rows','min_thread_rows','timer_s','warm_pass_s']
CAPTURE_BYTES = 9461760


def dump(path, value):
    raw = json.dumps(value, indent=2)+'\n'
    # Individual samples are useful; avoid expanding each numeric row across19 lines.
    if 'samples' in value:
        start = raw.index('  "samples": [\n'); end = raw.index('\n  ]',start)+4
        raw = raw[:start]+'  "samples": [\n'+',\n'.join('    '+json.dumps(r,separators=(',',':')) for r in value['samples'])+'\n  ]'+raw[end:]
    path.write_text(raw)


def prepare():
    base.need(not ROOT.exists(), 'single-use pool root exists')
    prior = json.loads(RESULT.read_text())
    base.need(RESULT.read_bytes() == base.git(P,'show','HEAD:'+str(RESULT.relative_to(P))) and
        prior.get('ownership_closed') and not prior.get('active'), 'commit closed predecessor before pool plan')
    predecessor_commit = base.git(P,'log','-1','--format=%H','--',str(RESULT.relative_to(P))).decode().strip()
    base.need(prior['schema']=='strata-iq2-pool-v1' and prior['stage']=='closed' and prior['passed'] and
        prior['complete'] and prior['performance_gate_passed'] and prior['artifact_retirement']['complete'] and
        prior['plan']['environment']['STRATA_IQ_SINGLE_EXACT']=='1' and
        prior['advancement_hold']['model_single_exact']=='0', 'close and preserve the differently configured pool screen')
    base.need(Path(prior['plan']['fixture']['path']).parent != ROOT, 'fresh pool root required')
    model_path=base.A/'cpu-work-census-current.json';model=json.loads(model_path.read_text())
    base.need(model_path.read_bytes()==base.git(P,'show','HEAD:'+str(model_path.relative_to(P))) and
        model['passed'] and model['complete'] and model['ownership_closed'] and not model['active'] and
        model['environment']['STRATA_IQ_SINGLE_EXACT']=='0' and
        model['environment'].get('STRATA_IQ_MT_MIN','2')=='2', 'committed actual model baseline0 required')
    model_baseline_reference=dict(commit=base.git(P,'log','-1','--format=%H','--',str(model_path.relative_to(P))).decode().strip(),
        path=str(model_path.relative_to(P)),sha256=base.digest(model_path),single_exact='0',mt_min='2')
    previous_comparison=dict(commit=predecessor_commit,path=str(RESULT.relative_to(P)),source=prior['plan']['source'],
        single_exact='1',latency_geomeans={str(d['down']):d['geomean']for d in prior['decisions']},
        limits='Valid prior configuration-specific pool screen; executable retired. This trial rebuilds and validates independently with the actual model baseline0.')
    revision = base.git(P,'rev-parse','HEAD').decode().strip()
    forms = [r for r in base.formats() if r['gu']==22]
    for row in forms:
        row['experts']=168;row['fixture_sha256']=base.fixture_hash(row)
    text=''.join(f"{r['shard']} {r['down']} {r['gate']} {r['up']} {r['down_offset']} {r['blob_bytes']} 168\n" for r in forms)
    base.need(all(not any(c.isspace() for c in r['shard']) for r in forms),'fixture path whitespace')
    ROOT.mkdir();fixture=ROOT/'inputs.txt';fixture.write_text(text)
    plan = dict(source=base.source_pin(P,revision,PREFIXES),ggml=base.source_pin(base.GGML,base.GGML_REV,('ggml',)),
        tools={str(Path(x).resolve()):base.digest(x) for x in (base.LLVM/'bin/clang',base.LLVM/'bin/clang++','/usr/bin/cmake','/usr/bin/ninja',sys.executable)},
        environment=dict(base.environment(),STRATA_IQ_SINGLE_EXACT='0',STRATA_IQ_MT_MIN='2',STRATA_IQ2_PAIR_AUDIT='1'),
        machine=base.machine(),formats=forms,metadata_sha256=base.digest(base.METADATA),
        model_identity={r['shard']:base.identity(r['shard']) for r in forms},
        fixture=dict(path=str(fixture),text=text,sha256=base.digest(fixture)),
        configure=['/usr/bin/cmake','-S',str(P/'bench/sycl/cpu_iq2'),'-B',str(ROOT/'build'),'-G','Ninja',
            '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_C_COMPILER='+str(base.LLVM/'bin/clang'),
            '-DCMAKE_CXX_COMPILER='+str(base.LLVM/'bin/clang++'),'-DSTRATA_GGML_DIR='+str(base.GGML)],
        build=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target','cpu_iq2_pool','--parallel','6'],
        order=ORDER,
        protocol=dict(question='Does the expanded direct IQ2_S NT1 producer improve the actual five-worker+computing-host pool with model SINGLE_EXACT=0?',
            arms='Same binary, startup STRATA_IQ2_PAIR=0 versus1 in separate owned processes. Other flags fixed before exec; no runtime environment mutation.',
            mixtures=[dict(jobs=28,NT_counts=[22,5,1,0],entries=35),dict(jobs=24,NT_counts=[16,5,2,1],entries=36)],
            scope='Representative rounded job mixtures derived from the exact closed code/prose census, not model MTP measurements. Types22/20 and22/42 are independent decisions. NT2-4,Down,pool scheduling and storage remain unchanged.',
            access='Repeat uses28/24 distinct real experts (does not fit wholly in L3); stream uses168 distinct blobs,6/7 batches with no reuse per pass.168*blob_bytes is a logical footprint,not physical DRAM bytes. Seed20261021 controls finite activations,token assignment,NT permutation and expert order.',
            numerical='Two fresh check processes:60 checked-producer geometry,span,alignment,alias,row/padding cases; complete pool outputs match serial direct NT1 GU and unchanged multi-row/Down routes for168 experts x2 formats x2 mixtures. Poison inactive tokens and canaries; verify input immutability. Repeat before/after every performance process; canonical9,461,760B output payloads must be byte-identical across arms/processes.',
            dispatch='Audit enabled in both arms and charged to component time. Each timed sample conserves GU callbacks,weighted row ops,NT1 rows and selected-pair rows. All six distinct,physically pinned threads,including host,must do positive GU work per sample. Per-thread rows are summarized after conservation; no request identity from TLS.',
            repetition='A0/B0/B1/A1/A2/B2 independent processes. Each process both formats,both mixtures,repeat then stream,six repetitions per cell. Whole passes until0.35s,minimum0.25s,maximum1000000passes. No retries,changed floors or extra samples after observing results.',
            overhead='Per-pass clock probe below0.1% of a warm pass. Source/model loading,oracles,capture and snapshots outside timing. Production dispatch/fallback,GU,hidden quant,Down,barriers and audits inside; phase clocks are nested in full-pool wall.',
            gate='Per Down format: geometric mean of12 process-pair/mixture/access paired medians<=0.97,each median<=1.02,each independent process-pair geomean<=1.0. Decide formats independently; no aggregate gain hides a rejected format. Component execution success is separate from this gate.',
            stop='First source,ownership,numerical,dispatch,timing or byte-capture failure closes the stage; no automatic retry. Failed performance permits no inference trial of that format.',
            limits='CPU component with observers,not model throughput or CPU/RAM saturation. Full inference rebuild/safety,clean sustained generation and physical262144/quality/session/RESTORE still required.'),
        budgets=dict(owned_total_bytes=256<<20,per_file_bytes=64<<20,root_bytes=512<<20,rss_bytes=3<<30,
            child_wall_seconds=180,child_cpu_seconds=1200,build_wall_seconds=600,build_cpu_seconds=3600,
            capture_consumer='Eight bounded9.46MB temporary tensors prove full-output equality across both check and six performance processes; retire all after compact hashes/comparisons are committed.'),
        previous_comparison=previous_comparison,model_baseline_reference=model_baseline_reference,
        input_scope_note='Current fingerprint explicitly includes measurement_guards.py. The earlier32-source screen did not inventory that helper; do not retroactively expand its original receipt. This independent pool trial requalifies the producer and all current controls.',
        retention='Keep useful individual samples,format decisions,output hashes,work conservation and minimal thread placement. Validated63af fallback stays untouched;526e is compile-only. Retire closed tensors/raw/auxiliary build at pool-gate closure after recovering original Git inputs/recipes.')
    dump(RESULT,dict(schema='strata-iq2-pool-v1',plan=plan,stage='prepared',passed=False,complete=False,active=False,
        GPU_executed=False,model_adoption_qualified=False))
    print('Prepared frozen full-pool comparison; commit before build.')


def load():
    base.need(RESULT.read_bytes()==base.git(P,'show','HEAD:'+str(RESULT.relative_to(P))),'commit current stage first')
    value=json.loads(RESULT.read_text());p=value['plan']
    reference=p['model_baseline_reference'];raw=(P/reference['path']).read_bytes()
    base.need(base.digest(P/reference['path'])==reference['sha256'] and
        raw==base.git(P,'show','HEAD:'+reference['path']),'actual model baseline reference drift')
    base.need(base.source_pin(P,p['source']['revision'],PREFIXES)==p['source'],'project source drift')
    base.need(base.source_pin(base.GGML,base.GGML_REV,('ggml',))==p['ggml'],'ggml drift')
    base.need(base.machine()==p['machine'] and all(base.digest(f)==h for f,h in p['tools'].items()),'machine/tool drift')
    base.need(base.digest(base.METADATA)==p['metadata_sha256'],'metadata drift')
    base.need(Path(p['fixture']['path']).read_text()==p['fixture']['text'] and base.digest(p['fixture']['path'])==p['fixture']['sha256'],'generated fixture drift')
    for row in p['formats']:
        base.need(base.identity(row['shard'])==p['model_identity'][row['shard']] and base.fixture_hash(row)==row['fixture_sha256'],'fixture/model drift')
    return value


def validate_samples(samples):
    expected={(process,arm,down,mix,access,rep) for process in range(3) for arm in range(2)
        for down in (20,42) for mix in range(2) for access in range(2) for rep in range(6)}
    actual=[tuple(s[k] for k in COLUMNS[:6]) for s in samples]
    base.need(len(samples)==288 and len(set(actual))==288 and set(actual)==expected,'complete unique288 sample matrix')
    for s in samples:
        n,entries,ones=(24,36,16) if s['mixture'] else (28,35,22)
        calls=s['passes']*(168//n if s['access'] else 1)
        base.need(s['wall_s']>=.25 and 0<s['passes']<1000000 and s['batch_calls']==calls,'timing floor/count')
        base.need(s['row_ops']==calls*640*entries and s['NT1_rows']==calls*640*ones and
            s['paired_rows']==(s['NT1_rows'] if s['arm'] else 0) and s['min_thread_rows']>0,'actual dispatch/work conservation')
        base.need(0<s['timer_s']<s['warm_pass_s']*.001 and all(s[k]>=0 for k in ('cpu_s','GU_s','quant_s','Down_s')),'clock/phase gate')
        base.need(s['GU_s']+s['quant_s']+s['Down_s']<=s['wall_s']+.002,'nested phase clocks exceed wall')


def reduce_samples(samples):
    validate_samples(samples)
    v={tuple(s[k] for k in COLUMNS[:6]):s['wall_s']/s['batch_calls'] for s in samples}
    decisions=[]
    for down in (20,42):
        cells=[];pairs=[]
        for process in range(3):
            selected=[]
            for mix in range(2):
                for access in range(2):
                    ratios=[v[process,1,down,mix,access,r]/v[process,0,down,mix,access,r] for r in range(6)]
                    median=statistics.median(ratios);selected.append(median)
                    cells.append(dict(process=process,mixture=mix,access=access,ratios=ratios,median=median))
            pairs.append(statistics.geometric_mean(selected))
        gm=statistics.geometric_mean(c['median'] for c in cells)
        decisions.append(dict(down=down,cells=cells,pair_geomeans=pairs,geomean=gm,
            performance_gate_passed=gm<=.97 and max(c['median'] for c in cells)<=1.02 and max(pairs)<=1))
    return decisions


def execute(stage):
    record=load();p=record['plan'];prior={'build':'prepared','check':'built','measure':'checked'}[stage]
    base.need(record['stage']==prior and not record['active'],'wrong stage; no retries')
    receipt=ROOT/(stage+'.json');base.need(not receipt.exists(),'stage already attempted')
    state=dict(stage=stage,passed=False,complete=False,active=False,commands=[])
    def persist():
        state['active']=own.active is not None;dump(receipt,state)
    own=owner.Owner(ROOT/'owned',persist);state['commands']=own.commands
    def run(label,argv,env,wall=180,cpu=1200):
        e,out,err=own.run(stage+'-'+label,argv,env,P,wall=wall,cpu=cpu,rss_cap=3<<30,
            text_cap=64<<20,file_cap=64<<20,total_cap=256<<20)
        base.need(owner.closed(e) and e['exit_code']==0,'closed successful child required: '+label)
        base.need(sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())<=512<<20,'root byte budget')
        return e,out,err
    try:
        if stage=='build':
            run('configure',p['configure'],p['environment'],120,600);run('compile',p['build'],p['environment'],600,3600)
            binary=ROOT/'build/cpu_iq2_pool';record['binary']=dict(path=str(binary),sha256=base.digest(binary),bytes=binary.stat().st_size)
            _,out,_=run('ldd',['/usr/bin/ldd',str(binary)],p['environment'],20,20)
            base.need(not any(k in out.read_text().lower() for k in ('sycl','level_zero','cuda','amdhip','libze')),'GPU runtime linked')
            record['compile_commands_sha256']=base.digest(ROOT/'build/compile_commands.json')
        else:
            base.need(base.digest(record['binary']['path'])==record['binary']['sha256'],'binary drift')
            proof=[];samples=[];captures=[]
            sequence=p['order'] if stage=='measure' else [(0,0),(0,1)]
            for index,(process,arm) in enumerate(sequence):
                capture=ROOT/'owned'/f'{stage}-{index}.f32';base.need(not capture.exists(),'capture overwrite')
                env=dict(p['environment'],STRATA_IQ2_PAIR=str(arm))
                _,out,_=run(str(index),[record['binary']['path'],'perf' if stage=='measure' else 'check',
                    p['fixture']['path'],str(capture),str(arm),str(process)],env)
                rows=[json.loads(line) for line in out.read_text().splitlines()]
                base.need(rows[-1]==dict(kind='closed',passed=True),'native terminal marker')
                placement=[r for r in rows if r['kind']=='placement'];validation=[r for r in rows if r['kind']=='validation']
                base.need(len(placement)==len(validation)==1 and placement[0]['workers']==5 and placement[0]['host_works'] and
                    len(placement[0]['threads'])==6 and len({t[0]for t in placement[0]['threads']})==6 and
                    {t[1]for t in placement[0]['threads']}==set(p['machine']['physical_cpus']) and
                    placement[0]['host_tid'] in {t[0]for t in placement[0]['threads']},'actual six-thread placement')
                base.need(validation[0]['passed'] and validation[0]['bitwise'] and validation[0]['experts_per_format']==168 and
                    validation[0]['formats']==2 and validation[0]['mixtures']==2 and
                    validation[0]['contract_cases']==(60 if stage=='check' else 0),'numerical/producer contract')
                base.need(capture.stat().st_size==CAPTURE_BYTES,'canonical capture extent')
                info=dict(process=process,arm=arm,sha256=base.digest(capture),bytes=CAPTURE_BYTES)
                reference=record.get('check_output_sha256',info['sha256'])
                base.need(info['sha256']==reference and (not captures or captures[0]['sha256']==info['sha256']),'cross-arm full-output bits')
                captures.append(info);proof.append(dict(process=process,arm=arm,placement=placement[0],validation=validation[0]))
                samples += [r for r in rows if r['kind']=='sample']
                print(f'Closed {stage} process {index+1}/{len(sequence)}',flush=True)
            record[stage+'_evidence']=proof;record[stage+'_output_hashes']=captures
            if stage=='check':record['check_output_sha256']=captures[0]['sha256']
            else:
                record['decisions']=reduce_samples(samples);record['sample_columns']=COLUMNS
                record['samples']=[[s[k]for k in COLUMNS]for s in samples]
                record['performance_gate_passed']=any(d['performance_gate_passed']for d in record['decisions'])
                record['eligible_down_formats']=[d['down']for d in record['decisions']if d['performance_gate_passed']]
        load();state.update(passed=True,complete=True)
    except BaseException as error:
        state['error']=str(error)
    finally:
        state.update(active=own.active is not None,ownership_closed=own.active is None and all(owner.closed(c)for c in own.commands));persist()
        record.update(active=state['active'],ownership_closed=state['ownership_closed'],passed=state['passed'],complete=state['complete'],
            stage={'build':'built','check':'checked','measure':'closed'}[stage] if state['passed'] else 'failed')
        if 'error' in state:record['error']=state['error']
        dump(RESULT,record)
    print(json.dumps({k:record.get(k)for k in ('stage','passed','complete','ownership_closed','performance_gate_passed','error')}))
    return 0 if state['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','build','check','measure'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        for proc in Path('/proc').iterdir():
            if not proc.name.isdecimal():continue
            try:exe=(proc/'exe').resolve(strict=True)
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            base.need(exe.name not in ('cpu_iq2_pool','cpu_iq2_pair','cpu_iq2_subset','cpu_roofline'),'existing CPU owner '+proc.name)
        return prepare() if args.stage=='prepare' else execute(args.stage)


if __name__=='__main__':sys.exit(main())
