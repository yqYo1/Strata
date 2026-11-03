#!/usr/bin/env python3
"""Owned QSA correctness and matched timing; repository LICENSE.

The selected model engine stays unchanged. Component timings never qualify
model PP/TG,task quality,physical262144 context or RESTORE.
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
import time

sys.dont_write_bytecode = True
import qsa_subgroup_rows_component as build

entry, base, P, B, RESULT = build.entry, build.base, build.P, build.B, build.RESULT
ROOT, KEY = build.ROOT/'device-v1', 'qsa_subgroup_rows_device_v1'
ORDER = ('smoke','numerical','speed0','speed1','speed2')
SELF = Path(__file__).resolve()
require, digest, write = base.require, base.digest, base.write


def compact(row):
    result = {k:v for k,v in row.items() if k != 'sources'}
    result['source_inputs'] = dict(files=len(row['sources']), aggregate_sha256=build.fingerprint(row['sources']))
    return result


def save(row):
    write(ROOT/'record.json',row)
    data=json.loads(RESULT.read_text());data[KEY]=compact(row);write(RESULT,data)


def inputs(revision):
    values=build.sources(revision)
    values.update(base.controls(revision,controller=SELF))
    return values


def admit_build():
    row=json.loads((build.ROOT/'record.json').read_text())
    require(row['stage']=='built-not-executed' and all(row[k] is True for k in
            ('passed','complete','ownership_closed')) and not row['active'] and not row['GPU_executed'],
            'closed CPU-only build required')
    allowed=build.current(row)
    inv=row['compiler_inputs'];require(digest(inv['path'])==inv['sha256'],'compiler inventory changed')
    files=json.loads(Path(inv['path']).read_text())
    require(len(files)==inv['files'] and all(allowed.get(p)==sha==digest(p) for p,sha in files.items()),
            'actual compiled source/dependency changed')
    require(all(allowed.get(p)==sha==digest(p) for p,sha in row['actual_link_inputs']['frozen_inputs'].items())
            and row['actual_link_inputs']['uncovered_external_inputs']==0,'actual external link inputs changed')
    require(all(digest(v['path'])==v['sha256'] and Path(v['path']).stat().st_size==v['bytes']
                for v in row['binaries'].values()),'actual component executable changed')
    # build.current just verified the selected native generation,worker gate
    # and providers. Read those same checked manifests without repeating the
    # complete SDK/Git/package admission a second time.
    native=json.loads((entry.model.ROOT/'record.json').read_text())
    require(native['source_commit']==row['native_source_commit']
            and native['binaries']['strata']==row['selected_engine'],'verified native receipt changed')
    worker=json.loads((entry.device.ROOT/'record.json').read_text())
    providers=entry.model.prior.fallback.retained.runtime.inventory(worker['runtime_manifest'])
    return row,native,providers


def current(row):
    require(base.committed(RESULT)[KEY]==compact(row) and inputs(row['source_commit'])==row['sources'],
            'commit/freeze exact device source and plan')
    compiled,native,providers=admit_build()
    require(row['build_identity']=={k:compiled[k] for k in ('source_commit','compiler_inputs','binaries')}
            and row['boot']==native['boot'] and row['driver']==native['driver'], 'device build/platform differs')
    return compiled,native,providers


def prepare(mode):
    require(mode in ORDER,'unknown device mode')
    data=base.committed(RESULT)
    if KEY in data:
        row=json.loads((ROOT/'record.json').read_text());compiled,native,providers=current(row)
        require(row['stage']=='between-cases' and row['ownership_closed'] and not row['active'],
                'previous case failed/open;no retry')
    else:
        require(mode==ORDER[0] and not ROOT.exists(),'fresh smoke first')
        row=dict(stage='preparing',source_commit=base.basis.build.git(P,'rev-parse','HEAD').decode().strip(),
                 sources={},plans=[],cases=[],passed=False,complete=False,active=False,ownership_closed=True,
                 model_executed=False,inference_adopted=False)
        row['sources']=inputs(row['source_commit'])
        compiled,native,providers=admit_build()
    require(len(row['cases'])==ORDER.index(mode) and all(v['passed'] for v in row['cases']),
            'ordered successful cases required;no retry/replacement')
    root=ROOT/mode;require(not root.exists(),'fresh owned case root')
    root.mkdir(mode=0o700,parents=True)
    for name in ('tmp','cache'):(root/name).mkdir()
    worker=json.loads((entry.device.ROOT/'record.json').read_text())
    old=worker['environment_template']['STRATA_MEASUREMENT_ROOT']
    env={k:v.replace(old,str(root)) for k,v in worker['environment_template'].items()}
    traced=mode=='smoke'
    env.update(STRATA_TEST_MAIN_GATES='1',
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if traced else 'UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'ur.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
    require(all(env.get(k)==v for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',
        EnableDirectSubmission='0',STRATA_SYCL_PREFILL_SAME_QUEUE='1',ZES_ENABLE_SYSMAN='0',
        ONEAPI_DEVICE_SELECTOR='level_zero:gpu').items()),'preserve immutable GPU safety/selection')
    env=entry.device.engine_once.native_environment(worker,env)
    speed=mode.startswith('speed');binary=compiled['binaries']['paired' if speed else 'numerical']
    command=[binary['path'],'--current-prefill',mode[-1]] if speed else \
            [binary['path'],'--smoke' if traced else '--component']
    plan=dict(mode=mode,root=str(root),command=command,environment=env,binary=binary,traced=traced,
        budgets=dict(wall=600 if speed else 180,cpu=1200 if speed else 360,rss_cap=8<<30,
                     text_cap=8<<20,file_cap=32<<20,total_cap=64<<20),
        criteria='Exact source/dependency/ELF/provider/environment;three native gates,normal0,quiet final validation,no freshfault,exact output/guard/input checks. Smoke traces kernel enqueue;full numerical has11cases including262144-cell component addresses,not a full model context.',
        performance='After smoke/numerical,three fresh speed processes in order0,1,2;all six synthetic current shapes with six alternating pairs each. Keep all individual intervals. Require>=10ms and observed clock-pair maximum<1percent;failure stops without replacement. Assess each shape separately by three process ratios and order;no blanket minimum gain. Synthetic noncausal shapes do not establish model speed or saturation.',
        retention='Retain compact coverage,individual timing samples,source/build/provider and fault/closure evidence. Retire successful full trace/JIT/raw output after publication;keep diagnostic only for a distinct unresolved failure.')
    if not ROOT.joinpath('record.json').exists():
        row.update(build_identity={k:compiled[k] for k in ('source_commit','compiler_inputs','binaries')},
                   native_validation_probe=native['native_validation_probe'],boot=native['boot'],driver=native['driver'])
    row['plans'].append(plan);row['stage']='prepared';save(row)
    print('Prepared '+mode+';commit before GPU execution.',flush=True)


def fields(line):
    pairs=re.findall(r'([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)',line)
    require(len(pairs)==len(dict(pairs)),'duplicate output field')
    return dict(pairs)


def parse_output(text,mode):
    if not mode.startswith('speed'):
        cases=[fields(v) for v in text.splitlines() if v.startswith('case cells=')]
        require(len(cases)==(2 if mode=='smoke' else 11) and all(v['status']=='PASS'
            and v['guards']==v['inputs']==v['packed_logical_prefix']==v['packed_guards']=='PASS'
            and math.isfinite(float(v['max_budget_ratio'])) for v in cases),'numerical case coverage differs')
        require(text.count('case exact_prefix33_bits=PASS ')==len(cases)
            and 'case positive_logical_page_overflow=PASS status=PASS' in text,
            'selected-kernel comparison/bounds coverage missing')
        return dict(numerical_cases=cases,exact_prefix_comparisons=len(cases),
                    positive_overflow_IDs_checked=True,model_quality_qualified=False)
    case_lines=[fields(v) for v in text.splitlines() if v.startswith('case cells=')]
    samples=[fields(v) for v in text.splitlines() if v.startswith('sample process=')]
    clocks=[fields(v) for v in text.splitlines() if v.startswith('timing_contract ')]
    require(len(case_lines)==6 and len(samples)==72 and len(clocks)==1,'timing coverage differs')
    clock=float(clocks[0]['maximum_clock_pair_ms']);require(math.isfinite(clock) and clock>0,'clock probe invalid')
    expected=[(n,q,2051,p) for p in ('gather','clustered') for n,q in
              ((65536,8192),(114688,8192),(117488,2800))]
    slim=[];summaries=[]
    for i,(case,shape) in enumerate(zip(case_lines,expected)):
        require((int(case['cells']),int(case['queries']),int(case['cap']),case['pattern'])==shape
            and all(case[k]=='PASS' for k in ('full_bits','guards','inputs','status')),'timing case differs')
        window=samples[i*12:(i+1)*12];times={0:[],1:[]}
        for j,s in enumerate(window):
            variant=(int(mode[-1])+j//2+j%2)%2;ms=float(s['milliseconds'])
            require(int(s['process'])==int(mode[-1]) and int(s['round'])==j//2 and int(s['order'])==j%2
                and int(s['variant'])==variant and s['implementation']==('subgroup_rows' if variant else 'ordered_exp')
                and (int(s['cells']),int(s['queries']),int(s['cap']),s['pattern'])==shape
                and all(s[k]=='PASS' for k in ('exact_bits','guards','status'))
                and math.isfinite(ms) and ms>=10 and clock<.01*ms,'ineligible or unplanned timing sample')
            times[variant].append(ms);slim.append(dict(case=i,round=j//2,order=j%2,variant=variant,milliseconds=ms))
        a,b=statistics.median(times[0]),statistics.median(times[1])
        summaries.append(dict(cells=shape[0],queries=shape[1],cap=shape[2],pattern=shape[3],
                              reference_median_ms=a,candidate_median_ms=b,candidate_over_reference=b/a))
    return dict(clock_probe=clocks[0],cases=summaries,samples=slim,
                model_speed_qualified=False,physical_saturation_qualified=False)


def run(mode):
    row=json.loads((ROOT/'record.json').read_text());compiled,native,providers=current(row)
    require(row['stage']=='prepared' and row['plans'][-1]['mode']==mode
            and len(row['cases'])==ORDER.index(mode),'fresh committed case only')
    plan=row['plans'][-1];root=Path(plan['root']);require(not (root/'execution.json').exists(),'no retry')
    cursor_text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(v.split(': ',1)[1] for v in cursor_text.splitlines() if v.startswith('-- cursor: '))
    outcome=dict(mode=mode,passed=False,complete=False,active=False,ownership_closed=False,
                 host_probe_accepted=False,GPU_kernels_executed=False,model_executed=False)
    owner=None;phases=[];mapped=set();start=(time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
    def persist():
        write(root/'execution.json',dict(outcome,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    try:
        owner=base.supervision.Owner(root/'owned',persist)
        def interact(io,command):
            device_line=None
            for phase in ('before-runtime','runtime-ready','after-cleanup'):
                while True:
                    line=io.line(150)
                    if line.startswith('device='): device_line=line
                    if line.startswith('QSA_BF16_READY '):break
                require(line=='QSA_BF16_READY '+phase,'native checkpoint order differs')
                proc=Path('/proc',str(command['pid']))
                require((proc/'exe').resolve()==Path(plan['binary']['path']),'actual native executable differs')
                actual=dict(v.split('=',1) for v in (proc/'environ').read_text().split('\0') if '=' in v)
                require(all(actual.get(k)==v for k,v in plan['environment'].items())
                    and set(actual)-set(plan['environment'])<={'PWD','OLDPWD','SHLVL','_'},'native environment differs')
                seen=entry.template.shared.runtime.executable_mappings((proc/'maps').read_text())
                require(seen<=set(providers)|{plan['binary']['path']},'unqualified loaded provider')
                if phase=='runtime-ready':
                    require(device_line and 'Intel' in device_line and 'B570' in device_line,'wrong GPU')
                    require({row['native_validation_probe']['loader']['path'],row['native_validation_probe']['library']['path'],
                             plan['environment']['ZE_ENABLE_ALT_DRIVERS']}<=seen,'required GPU provider absent')
                    outcome['device']=device_line
                delta=[a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),start)]
                require(min(delta)>=0 and abs(delta[0]-delta[1])<=50000000,'suspend/clock discontinuity')
                phases.append(phase);mapped.update(seen);print(mode+' '+phase,flush=True);io.send(b'X')
            io.proc.stdin.close();io.exit(30)
        command,out,err=owner.run(mode,plan['command'],plan['environment'],P,interaction=interact,**plan['budgets'])
        require(base.supervision.closed(command) and command['exit_code']==0,'normal owned native exit required')
        text,errors=out.read_text(),err.read_text();validation=entry.device.ur_validation.preload_evidence(errors,command['pid'])
        require(validation['loader']==row['native_validation_probe']['loader']['path']
            and not any(v in errors for v in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called','cleanup failed'))
            and (not (root/'validation.log').exists() or (root/'validation.log').stat().st_size==0),
            'final runtime validation failed')
        require(text.count('fixture_status=PASS ')==1,'native completion marker missing')
        parsed=parse_output(text,mode)
        if plan['traced']:
            trace=(root/'ur.trace').read_text();require('urEnqueueKernelLaunch' in trace,'actual kernel trace missing')
            parsed['trace']=dict(bytes=(root/'ur.trace').stat().st_size,sha256=digest(root/'ur.trace'),kernel_enqueue_observed=True)
        current(row)
        outcome.update(complete=True,host_probe_accepted=True,GPU_kernels_executed=True,native_exit_code=0,
                       validation=validation,phases=phases,results=parsed)
    except BaseException as error:outcome['error']=repr(error)
    finally:
        if owner:entry.device.ur_validation.finish(outcome,owner,cursor,row['boot'])
        outcome['passed']=bool(outcome.get('passed') and outcome['complete'] and outcome.get('driver_after')==row['driver'])
        row['cases'].append(outcome);row.update(stage='between-cases' if outcome['passed'] else 'failed',
            active=bool(owner and owner.active),ownership_closed=bool(outcome.get('ownership_closed')))
        if outcome['passed'] and len(row['cases'])==len(ORDER):row.update(stage='closed',passed=True,complete=True)
        save(row);persist()
    print(json.dumps({k:outcome.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    require(outcome['passed'],'QSA device case failed;stop without retry')


if __name__=='__main__':
    require(len(sys.argv)==3 and sys.argv[1] in ('prepare','run') and sys.argv[2] in ORDER,
            'use prepare|run smoke|numerical|speed0|speed1|speed2')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        {'prepare':prepare,'run':run}[sys.argv[1]](sys.argv[2])
