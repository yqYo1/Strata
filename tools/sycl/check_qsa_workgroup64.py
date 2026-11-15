#!/usr/bin/env python3
"""Owned WG64 numerical and isolated geometry checks;repository LICENSE.

Main-authored. Direct production-baseline/component exports isolate geometry;
there is no WG64 model dispatcher and no inference-adoption claim.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode=True
import build_qsa_workgroup64 as build
import current_pti_probe as maps
import engine_once
import ur_validation as validation

P,B,RESULT=build.P,build.B,build.RESULT
ROOT,KEY=B/'qsa-prefill-workgroup64-check-v3','qsa_prefill_workgroup64_check_v3'
THIS='tools/sycl/check_qsa_workgroup64.py'
STAGES=('smoke','oracle4','oracle16','bench-smoke','timing0','timing1','timing2')
require,sha=build.require,build.sha


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def compact(row):return {k:v for k,v in row.items() if k not in ('sources','environment')}


def save(row):
    (ROOT/'record.json').write_text(json.dumps(row,indent=2)+'\n')
    data=json.loads(RESULT.read_text());data[KEY]=compact(row)
    RESULT.write_text(json.dumps(data,indent=2)+'\n')


def environment(row,stage):
    root=ROOT/stage;env={k:v.replace(str(build.ROOT),str(root)) for k,v in row['environment'].items()}
    env.update(STRATA_TEST_MAIN_GATES='1',CUDA_MODULE_LOADING='EAGER',ZES_ENABLE_SYSMAN='0',
        TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if stage=='smoke' else 'UR_LAYER_PARAMETER_VALIDATION')
    if stage=='smoke':env['UR_LOG_TRACING']='level:info;flush:info;output:stderr'
    else:env.pop('UR_LOG_TRACING',None)
    for name in ('UR_LOG_VALIDATION','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO'):env[name]='level:warning;flush:warning;output:stderr'
    require(env.get('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD')=='1' and env.get('NEOReadDebugKeys')=='1'
        and env.get('EnableDirectSubmission')=='0' and not any('xestrata' in str(v).lower() for v in env.values()),
        'qualified independent safety/provider settings required')
    return engine_once.native_environment(dict(native_validation_probe=row['preload']),env)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'one fresh WG64check')
    native=json.loads((build.ROOT/'record.json').read_text());build.current(native)
    require(native['stage']=='built' and native['passed'] and native['complete'] and native['ownership_closed'],
        'closed fresh component build required')
    require(all(sha(v['path'])==v['sha256'] for v in native['binaries'].values()),'fresh fixture identity differs')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    controllers=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,'--','tools/sycl'],text=True).splitlines()
    names=sorted(set([*native['sources'],*controllers,THIS,'tools/sycl/current_pti_probe.py',
        'tools/sycl/current_pti_admission.py','tools/sycl/request_profile_reader.py',
        'tools/sycl/ur_validation.py','tools/sycl/engine_once.py','tools/sycl/measurement_guards.py']))
    pins={n:hashlib.sha256(build.blob(revision,n)).hexdigest() for n in names}
    require(all(sha(P/n)==h for n,h in pins.items()),'commit all numerical/timing controllers')
    base=json.loads((build.BASE_ROOT/'record.json').read_text())
    row=dict(stage='prepared',source_commit=revision,sources=pins,source_sha256=signature(pins),
        build_record=dict(path=str(build.ROOT/'record.json'),sha256=sha(build.ROOT/'record.json')),
        build_evidence_sha256=signature(build.compact(native)),binaries=native['binaries'],
        native_source_commit=native['source_commit'],baseline_native_source=native['baseline_native_source'],
        environment=native['environment'],environment_sha256=signature(native['environment']),
        admission=dict(runtime_manifest=native['baseline_recovery']['runtime_manifest']),preload=base['native_validation_probe'],
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),driver=validation.guards.driver_guard(),
        stages=list(STAGES),results=[],passed=False,complete=False,active=False,ownership_closed=True,
        whole_bytes=768<<20,budgets=dict(wall=300,cpu=600,rss_cap=6<<30,text_cap=16<<20,file_cap=64<<20,total_cap=128<<20),
        plan=dict(unit='Only64versus128threads with required head stepping/PVpartition/eightfragments. Same32cells,pitch33,cachedQ,KVvec4,orderedFP32softmax and BF16 storage;no kernel/model setting combination.',
            numerical='First traced oracle16smoke(two real/padded boundary heads per KVhead) with actual maps/environment/pre-main validation. Then independent oracle4and16full cases:11cases each including262144live cells,masked/tail/invalid-input/alias/guard cases;all output bits match oldprefix33;rawFP32maximum/alpha/normalizer/sum match WG128ordered for all12realheads atpage4andreal/paddedheads atpage16. Then normal-export benchmark smoke. Any failure stops before timing.',
            timing='Three independent normal bench processes0..2,each with3shapes(78453cells,8192queries,cap33gather or2051gather/clustered). Six paired rounds per shape,rotated order. Both direct exports refreshed/drained every call;references,resets and readbacks outside batches. cap33uses64calls,cap2051uses2calls;each timed batch>=50ms,inner clock-pair overhead*64<=1percent of50ms. Ineligible batch stops;no post-hoc size/criteria changes.',
            acceptance='All source/ELF/actual-map/driver/boot/validation/owner/numerical gates pass,then per shape all3process candidate/baseline median ratios<1supports that shape only,without minimum gain. Model-stage planning is eligible only when both2051patterns pass;report33separately and retain128outside qualified shapes. This direct-export experiment explicitly isolates geometry;actual production dispatch/whole-model/handoff/quality/262144/TGneed later new gates.',
            stopping='Stop first input,dependency,protocol,owner,validation,fault,clock,batch-floor or numerical failure;no retry/hashrefresh/speed-based exclusions. Commit each closed stage. All environment fixed before exec.'),
        retention='Keep individual paired timings,conservation/numerical/actual-provider/closure outcomes and source/recipe identities;no modelPP/TG conversion. Retire raw/JITand extra component generation after selection/source recovery. No production route or model adoption.')
    ROOT.mkdir(mode=0o700)
    for stage in STAGES:
        root=ROOT/stage;root.mkdir()
        for name in ('tmp','cache'):(root/name).mkdir()
    row['environment_per_stage_sha256']={s:signature(environment(row,s)) for s in STAGES}
    save(row);print('Prepared WG64numerical/isolated-timing plan;commit before GPU execution.',flush=True)


def current(row):
    require(RESULT.read_bytes()==build.blob('HEAD',str(RESULT.relative_to(P)))
        and json.loads(RESULT.read_text())[KEY]==compact(row),'commit exact previous WG64stage')
    require(all(sha(P/n)==h for n,h in row['sources'].items()) and signature(row['environment'])==row['environment_sha256'],
        'frozen qualification inputs changed')
    require(sha(row['build_record']['path'])==row['build_record']['sha256'],'component build record changed')
    native=json.loads(Path(row['build_record']['path']).read_text());build.current(native)
    require(signature(build.compact(native))==row['build_evidence_sha256']
        and all(sha(x['path'])==x['sha256'] for x in row['binaries'].values()),'source/build/fixture mismatch')
    require(all(signature(environment(row,s))==row['environment_per_stage_sha256'][s] for s in STAGES),
        'pre-exec stage environment changed')
    require(build.owner.artifact_bytes(ROOT)<=row['whole_bytes'],'qualification output budget')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot']
        and validation.guards.driver_guard()==row['driver'],'boot/driver identity changed')


def parse_samples(text,process):
    rows=[]
    for line in text.splitlines():
        if not line.startswith('sample '):continue
        pairs=re.findall(r'([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)',line)
        require(len(dict(pairs))==len(pairs),'duplicate timing field');r=dict(pairs)
        require(int(r['process'])==process and r['exact_bits']==r['guards']==r['status']=='PASS'
            and int(r['cells'])==78453 and int(r['queries'])==8192 and int(r['compared'])==8192*24*256,
            'actual timed shape/guards/process differs')
        cap,variant=int(r['cap']),int(r['variant']);calls=64 if cap==33 else 2
        require((cap,r['pattern']) in ((33,'gather'),(2051,'gather'),(2051,'clustered'))
            and variant in (0,1) and r['implementation']==('prefill_ordered_exp_WG128' if variant==0 else 'prefill_ordered_exp_WG64')
            and int(r['batch_calls'])==calls and math.isfinite(float(r['batch_ms'])) and float(r['batch_ms'])>=50
            and math.isfinite(float(r['milliseconds'])) and float(r['milliseconds'])>0
            and abs(float(r['batch_ms'])/calls-float(r['milliseconds']))<1e-6,'timing eligibility/dispatch differs')
        rows.append(dict(cap=cap,pattern=r['pattern'],round=int(r['round']),order=int(r['order']),variant=variant,
            milliseconds=float(r['milliseconds']),batch_calls=calls,batch_ms=float(r['batch_ms'])))
    require(len(rows)==36,'all36timing batches required')
    for cap,pattern in ((33,'gather'),(2051,'gather'),(2051,'clustered')):
        selected=[r for r in rows if (r['cap'],r['pattern'])==(cap,pattern)]
        require(len(selected)==12 and {(r['round'],r['order']) for r in selected}=={(a,b) for a in range(6) for b in range(2)}
            and all(r['variant']==(process+r['round']+r['order'])%2 for r in selected),'paired/rotated order differs')
    return rows


def run(stage):
    row=json.loads((ROOT/'record.json').read_text());current(row);index=len(row['results'])
    require(index<len(STAGES) and STAGES[index]==stage and all(r['passed'] for r in row['results']),
        'one ordered fresh stage;no retry')
    name='oracle16' if stage=='smoke' else stage if stage in ('oracle4','oracle16') else 'bench'
    args=['--smoke'] if stage in ('smoke','bench-smoke') else ['--component']
    if stage.startswith('timing'):args.append(stage[-1])
    binary=row['binaries'][name];root=ROOT/stage;env=environment(row,stage)
    text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in text.splitlines() if s.startswith('-- cursor: '))
    result=dict(stage=stage,passed=False,complete=False,host_probe_accepted=False,maps=[],binary=binary)
    job=None
    def persist():
        receipt=dict(result,active=bool(job and job.active is not None),commands=job.commands if job else [])
        (root/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    job=build.owner.Owner(root/'owned',persist);native=dict(row,fixture=binary)
    def interact(io,entry):
        for phase in ('before-runtime','runtime-ready','after-cleanup'):
            while True:
                line=io.line(180)
                if line=='QSA_BF16_READY '+phase:break
            observed=dict(v.split(b'=',1) for v in Path('/proc',str(entry['pid']),'environ').read_bytes().split(b'\0') if b'=' in v)
            require(all(observed.get(k.encode())==v.encode() for k,v in env.items())
                and set(observed)-{k.encode() for k in env} <= {b'PWD',b'OLDPWD',b'SHLVL',b'_'},
                'actual native environment differs')
            result['maps'].append(dict(phase=phase,**maps.mapped(entry['pid'],native,'control')))
            io.send(b'X')
        require(io.line(30).startswith('fixture_status=PASS '),'actual fixture completion required');io.exit(90)
    try:
        entry,out,err=job.run(stage,[binary['path'],*args],env,P,interaction=interact,**row['budgets'])
        result['process_observation']={k:entry.get(k) for k in
            ('pid','exit_code','error','direct_child_reaped','session_empty','observation_complete','elapsed_seconds')}
        output,errors=out.read_text(),err.read_text()
        require(build.owner.closed(entry) and entry['exit_code']==0,'normal closed fixture required')
        proof=validation.preload_evidence(errors,entry['pid'])
        start=re.search(r'^strata UR validation: [^\n]*phase=before-main[^\n]*\n',errors,re.M)
        require(start and proof['loader']==row['preload']['loader']['path'] and
            not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain',errors[start.end():]),
            'validation activation/after-main failure')
        require('Intel(R) Arc(TM) B570' in output,'actual B570 runtime device required')
        if name.startswith('oracle'):
            cases=2 if stage=='smoke' else 11
            require(len(re.findall(r'^case cells=.*status=PASS$',output,re.M))==cases
                and output.count('exact_FP32_online_state=PASS')==cases and output.count('exact_prefix33_bits=PASS')==cases,
                'all oracle/output/raw-state/guard cases required')
            result['numerical_cases']=cases;result['component_physical262144_cells_checked']=stage in ('oracle4','oracle16')
        elif stage=='bench-smoke':require('samples=0 full_bits=PASS' in output,'normal-export smoke required')
        else:result['samples']=parse_samples(output,int(stage[-1]))
        current(row)
        result.update(complete=True,host_probe_accepted=True,native_exit_code=0,validation=proof,
            stdout_sha256=sha(out),stderr_sha256=sha(err))
    except BaseException as error:result['error']=repr(error)
    finally:
        validation.finish(result,job,cursor,row['boot'])
        result['passed']=bool(result['passed'] and result['complete'] and result.get('driver_after')==row['driver'])
        row['results'].append(result);row.update(stage='running' if result['passed'] else 'failed',
            ownership_closed=result['ownership_closed'],active=job.active is not None)
        if result['passed'] and len(row['results'])==len(STAGES):
            decisions=[]
            for cap,pattern in ((33,'gather'),(2051,'gather'),(2051,'clustered')):
                ratios=[]
                for item in row['results'][-3:]:
                    samples=[r for r in item['samples'] if (r['cap'],r['pattern'])==(cap,pattern)]
                    ratios.append(statistics.median(r['milliseconds'] for r in samples if r['variant']==1)/
                        statistics.median(r['milliseconds'] for r in samples if r['variant']==0))
                decisions.append(dict(cap=cap,pattern=pattern,process_median_time_ratios=ratios,all_processes_faster=all(v<1 for v in ratios)))
            row.update(stage='closed',complete=True,passed=True,decision=dict(shapes=decisions,
                wide_geometry_screen_passed=all(s['all_processes_faster'] for s in decisions if s['cap']==2051),
                production_dispatch_qualified=False,model_adopted=False,model_PP_qualified=False,
                physical262144_model_qualified=False,sustained_TG_qualified=False))
        save(row)
    print(json.dumps({k:result.get(k) for k in ('stage','passed','complete','native_exit_code','fresh_faults','error')}),flush=True)
    require(result['passed'],'WG64stage failed')


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare',*STAGES),'prepare|'+ '|'.join(STAGES))
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validation.guards.no_model_owners([])
        (prepare if sys.argv[1]=='prepare' else lambda:run(sys.argv[1]))()
