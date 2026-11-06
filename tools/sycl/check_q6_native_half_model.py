#!/usr/bin/env python3
"""Run the bounded same-build Q6/phase admission; Main, repository LICENSE.

Reuse this fork's qualified runtime/owner/parser contracts. No model, speed,
physical saturation, ordinary teardown or full-context qualification follows.
"""
import fcntl
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
import build_q6_native_half_model as build
import check_verify_window_phase as phase
import measurement_owner as ownership
import verify_window_model_entry as model_template
from iq3_model_entry import dispatches
from run_q6_columns import validation_log

P,B,RESULT,base=build.P,build.B,build.RESULT,build.base
ROOT=B/'q6-native-half-device-v1'
KEY='q6_native_half_model_device_v1'
SELF=Path(__file__).resolve()
require,digest,write=base.require,base.digest,base.write


def compact(row):
    out={n:v for n,v in row.items() if n not in ('controller_inputs','commands')}
    out['controller_inputs']=dict(files=len(row['controller_inputs']),
        aggregate_sha256=build.lineage.signature(row['controller_inputs']))
    return out


def save(row):
    write(ROOT/'record.json',row)
    data=json.loads(RESULT.read_text());data[KEY]=compact(row);write(RESULT,data)


def built():
    native=build.context().verify()
    host=native.get('host_result',{})
    require(host.get('passed') and host.get('complete') and host.get('ownership_closed') and
            not host.get('active') and not host.get('GPU_executed') and
            [(r['mode'],r['exit_code'],r['cases']) for r in host['cases']]==[(0,0,28),(1,0,28)] and
            host['cases'][0]['checksum']==host['cases'][1]['checksum'],
            'commit closed same-build real-pool parity before GPU preparation')
    require(native['native_recipe_checks']['passed'] and not native['GPU_executed'],
            'full verifier/generator and same-build Q6 fixture must compile first')
    return native


def prepare():
    require(KEY not in base.committed(RESULT) and not ROOT.exists(),'one fresh device admission')
    native=built()
    original=build.historical(build.COMPONENT_EVIDENCE,'q6_native_half_component_v3')
    require(original['passed'] and original['complete'] and original['ownership_closed'] and
            original['smoke']['passed_cases']==32,'closed same-kernel component smoke required')
    revision=base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    env=dict(original['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),
        STRATA_MEASUREMENT_ROOT=str(ROOT),PYTHONPYCACHEPREFIX=str(ROOT/'cache/python'),
        LD_PRELOAD=original['native_validation_probe']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=original['native_validation_probe']['loader']['path'],
        STRATA_SYCL_DECODE_Q6_NATIVE_HALF='0',STRATA_SYCL_DECODE_Q6_CENTERING='0',STRATA_SYCL_Q6_ROUTE_AUDIT='0',
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr')
    require(all(env.get(n)==v for n,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',
        NEOReadDebugKeys='1',EnableDirectSubmission='0',ZES_ENABLE_SYSMAN='0',
        STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0',SYCL_CACHE_PERSISTENT='0').items()),'fixed runtime safety')
    row=dict(stage='prepared',source_commit=revision,
        controller_inputs=base.controls(revision,controller=SELF),
        native_source_commit=native['source_commit'],
        build_record=dict(path=str(build.ROOT/'record.json'),sha256=digest(build.ROOT/'record.json')),
        binaries=native['binaries'],environment=env,
        native_validation_probe=original['native_validation_probe'],runtime_manifest=original['native_recipe_origin'].get('runtime_manifest',
            build.historical(build.COMPONENT_EVIDENCE,'decode_iq3_audit_speed_v2')['runtime_manifest']),
        boot=original['boot'],driver=original['driver'],
        cases=[dict(name='submission',target='sycl_submission_test',args=['--submission']),
            dict(name='q6-half',target='sycl_q6_columns_bench',args=['--smoke','--native-half']),
            dict(name='phase-cache',target='sycl_verifier_retirement_test',args=['--phase-cache'])],
        budgets=dict(wall=180,cpu=240,rss_cap=4<<30,text_cap=16<<20,file_cap=16<<20,total_cap=64<<20),
        whole_wall_seconds=900,
        plan='Once each:submission10,Q6-half32 explicit small shapes and scalar/Q81 byte oracles,then16 phase/T windows with12captures. Require normal0,strict same-PID before-main validation,quiet layers,guards,closed owners,unchanged build/source/ELFs/provider/boot/driver,and no fresh fault. Stop on first error,no retries.',
        limits='The Q6 entry is called directly by the same-build numerical fixture. Synthetic phase graphs do not execute full record_window/run or the new model route writer. Model gain,prefill/handoff,physical saturation,ordinary teardown,agent/task,262144/tails/RESTORE remain unqualified.',
        retention='Immediate model-entry prerequisite. Keep compact identities/outcomes;retire normal traces,owner detail and cache after closure. The preload retains one runtime reference until process exit.',
        results=[],commands=[],passed=False,complete=False,active=False,ownership_closed=True,
        GPU_executed=False,GPU_execution_attempted=False,model_executed=False,inference_adopted=False)
    ROOT.mkdir(mode=0o700)
    for n in ['tmp','cache']:(ROOT/n).mkdir(mode=0o700)
    save(row);print('Prepared same-build traced submission/Q6/phase checks;commit before execution.',flush=True)


def unchanged(row):
    require(base.controls(row['source_commit'],controller=SELF)==row['controller_inputs'], 'device controls changed')
    require(digest(row['build_record']['path'])==row['build_record']['sha256'], 'native build record changed')
    for asset in [*row['binaries'].values(),*row['native_validation_probe'].values()]:
        require(digest(asset['path'])==asset['sha256'],'executable or validation provider changed')
    inv=row['runtime_manifest'];require(digest(inv['path'])==inv['sha256'],'runtime manifest changed')
    values=json.loads(Path(inv['path']).read_text())
    require(len(values)==inv['files'] and all(digest(p)==h for p,h in values.items()),'runtime provider changed')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'] and
            base.guards.driver_guard()==row['driver'],'boot/driver changed')


def validate_case(name,text):
    if name=='submission':
        require([s for s in text.splitlines() if s.startswith('case ')]==phase.SUBMISSION and
            text.count('fixture_status=PASS mode=submission')==1,'submission completion differs')
        return 10
    if name=='phase-cache':
        require(text.splitlines()==[phase.PHASE,'phase_fixture_status=PASS cases=1'],'phase completion differs')
        return 16
    require(name=='q6-half','unknown case')
    rows=re.findall(r'^Q6_CASE_PASS ni=(\d+) no=(\d+) nc=(\d+) scalar_oracle=PASS q8_oracle=PASS$',text,re.M)
    expected=[(ni,no,c) for ni,no in [(256,1),(512,3),(1024,13),(2560,129)] for c in range(1,9)]
    require([tuple(map(int,r)) for r in rows]==expected and
        all(text.count(s)==1 for s in ['Q6_COMPARISON=native-half','Q6_TARGET_B570=1',
            'Q6_TARGET_FP16=1','Q6_TARGET_ENDIAN_LITTLE=1','Q6_COLUMNS_PASS mode=smoke']),
        'same-build Q6 target/numerical/normal-completion evidence differs')
    return 32


def run():
    row=json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[KEY]==compact(row) and row['stage']=='prepared' and
            not (ROOT/'run-attempt.json').exists() and not (ROOT/'owned').exists(),'one committed device attempt')
    write(ROOT/'run-attempt.json',dict(source_commit=row['source_commit'],operation='device'))
    owner,start=None,None;deadline=time.monotonic()+row['whole_wall_seconds']
    try:
        unchanged(row);built();start=phase.cursor()
        row.update(active=True,ownership_closed=False);write(ROOT/'record.json',row)
        owner=ownership.Owner(ROOT/'owned')
        owner.persist=lambda:write(ROOT/'execution.json',dict(commands=owner.commands))
        for case in row['cases']:
            unchanged(row);require(time.monotonic()<deadline,'whole device admission deadline')
            print('Q6 same-build device:',case['name'],flush=True)
            row['GPU_execution_attempted']=True
            if row['GPU_executed'] is not True:row['GPU_executed']=None
            write(ROOT/'record.json',row)
            entry,out,err=owner.run(case['name'],[row['binaries'][case['target']]['path'],*case['args']],
                row['environment'],ROOT,**dict(row['budgets'],wall=min(row['budgets']['wall'],deadline-time.monotonic())))
            require(ownership.closed(entry) and entry['exit_code']==0,'native fixture failed: '+(out.read_text()+err.read_text())[-4000:])
            checks=validate_case(case['name'],out.read_text())
            proof=validation_log(err.read_text(),entry['pid'],row['native_validation_probe']['loader']['path'],True)
            require(phase.faults(start)==[],'fresh GPU fault')
            row['results'].append(dict(name=case['name'],pid=entry['pid'],start_ticks=entry['direct_identity']['start_ticks'],
                native_exit=0,checks=checks,ownership_closed=True,validation=proof,
                stdout_sha256=digest(out),stderr_sha256=digest(err)))
            row['GPU_executed']=True;write(ROOT/'record.json',row)
        unchanged(row);built();row.update(stage='closed',passed=True,complete=True)
    except BaseException as e:
        row.update(stage='failed',passed=False,complete=False,error=repr(e));raise
    finally:
        row.update(commands=owner.commands if owner else [],active=owner is not None and owner.active is not None,
            ownership_closed=owner is None or (owner.active is None and all(ownership.closed(e) for e in owner.commands)))
        try:
            row.update(fresh_faults=phase.faults(start) if start is not None else None,
                boot_unchanged=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'])
            require(row['ownership_closed'] and row['boot_unchanged'] and
                (start is None or row['fresh_faults']==[]),'final owner/boot/fault gate')
        except BaseException as e:row.update(stage='failed',passed=False,complete=False,final_probe_error=repr(e))
        save(row)
    require(row['passed'] and row['complete'],'same-build device admission incomplete')
    print('Same-build Q6/phase admission passed;full model routes and speed remain pending.',flush=True)


@lru_cache(maxsize=1)
def measured_shapes():
    def winners(key):
        rows=build.historical(build.COMPONENT_EVIDENCE,key)['decision']['shapes']
        return {tuple(r['shape']) for r in rows if r['all_three_faster']}
    columns=winners('q6_byte_load_component_v2')
    half=columns & winners('q6_native_half_component_v3')
    require(len(columns)==30 and len(half)==26,'independent measured shape sets changed')
    return columns,half


def snapshots(text,arm,request):
    """Validate observed live epochs; no retired-history or request-delta claim."""
    require(arm in (0,1) and type(request) is int and 0<request<2**64,'snapshot identity')
    root=re.compile(r'strata Q6_ROUTE request=(\d+) device=(\d+) phase=(prompt|decode) tokens=([1-8]) '
        r'capture_id=(\d+) scope=live_capture_cumulative native_half_enabled=([01]) complete=1 '
        r'shapes=(\d+) cache_hits=(\d+) replay_started=(\d+) replay_completed=(\d+)')
    shape=re.compile(r'strata Q6_SHAPE request=(\d+) device=(\d+) phase=(prompt|decode) tokens=([1-8]) '
        r'capture_id=(\d+) ni=(\d+) no=(\d+) nc=([1-8]) reference_calls=(\d+) columns_calls=(\d+) '
        r'centered_calls=(\d+) native_half_calls=(\d+)')
    windows={};details={};capture_ids=set();columns,half=measured_shapes()
    for line in text.splitlines():
        if line.startswith('strata Q6_ROUTE '):
            m=root.fullmatch(line);require(m is not None,'malformed/incomplete Q6 snapshot')
            req,device,phase_name,t,capture,enabled,count,hits,started,completed=m.groups()
            numbers=list(map(int,[req,device,t,capture,enabled,count,hits,started,completed]))
            req,device,t,capture,enabled,count,hits,started,completed=numbers
            require(all(0<=v<2**64 for v in numbers) and req==request and device==0 and
                enabled==arm and capture>0 and count<=64 and started==completed,'snapshot scope/counters differ')
            key=(phase_name,t,capture)
            require(key not in windows and capture not in capture_ids,'duplicate live capture identity')
            capture_ids.add(capture)
            windows[key]=dict(phase=phase_name,tokens=t,capture_id=capture,shapes=count,
                cache_hits=hits,replay_started=started,replay_completed=completed,shape_calls=[])
        elif line.startswith('strata Q6_SHAPE '):
            m=shape.fullmatch(line);require(m is not None,'malformed Q6 shape')
            req,device,phase_name,t,capture,ni,no,nc,*calls=m.groups()
            req,device,t,capture,ni,no,nc=map(int,[req,device,t,capture,ni,no,nc]);calls=list(map(int,calls))
            require(req==request and device==0 and 0<ni<2**31 and 0<no<2**31 and
                all(0<=n<2**64 for n in calls) and 0<sum(calls)<2**64,'invalid Q6 shape identity/count')
            key=(phase_name,t,capture);entry=(ni,no,nc)
            require(key in windows and entry not in details.setdefault(key,set()),'unjoined/duplicate Q6 shape')
            details[key].add(entry)
            selected=0
            if phase_name=='decode' and entry in columns:selected=3 if arm and entry in half else 1
            require(calls[selected]>0 and all(c==0 for i,c in enumerate(calls) if i!=selected),
                'actual Q6 route disagrees with the independently measured phase/shape policy')
            windows[key]['shape_calls'].append(dict(shape=list(entry),calls=calls))
    require(windows and all(len(v['shape_calls'])==v['shapes'] for v in windows.values()),
        'missing complete live Q6 snapshot/shape rows')
    head=[]
    for window in windows.values():
        if window['phase']!='decode' or not window['replay_completed']:continue
        for item in window['shape_calls']:
            if item['shape'][:2]==[2560,248320] and tuple(item['shape']) in half:
                require(item['calls'][3 if arm else 1]>0,'Q6 head did not select the declared arm')
                head.append(window['capture_id'])
    require(head,'no completed decode window with the selected Q6 head')
    return dict(request=request,scope='observed_live_capture_cumulative',windows=list(windows.values()),
        completed_head_capture_ids=head,prompt_window_observed=any(k[0]=='prompt' for k in windows))


def parser_controls():
    def text(arm):
        return (f'strata Q6_ROUTE request=2 device=0 phase=decode tokens=4 capture_id=7 scope=live_capture_cumulative native_half_enabled={arm} complete=1 shapes=1 cache_hits=2 replay_started=3 replay_completed=3\n'
            f'strata Q6_SHAPE request=2 device=0 phase=decode tokens=4 capture_id=7 ni=2560 no=248320 nc=4 reference_calls=0 columns_calls={1-arm} centered_calls=0 native_half_calls={arm}\n')
    for arm in (0,1):snapshots(text(arm),arm,2)
    good=text(1)
    bad=[good.replace('complete=1','complete=0'),good.replace('request=2','request=3'),
        good.replace('device=0','device=1'),good.replace('shapes=1','shapes=2'),
        good.replace('replay_completed=3','replay_completed=2'),good.replace('replay_started=3 replay_completed=3','replay_started=0 replay_completed=0'),
        good.replace('no=248320','no=640'),good.replace('phase=decode','phase=prompt'),
        good.replace('native_half_enabled=1','native_half_enabled=0'),
        good.replace('native_half_calls=1','native_half_calls=0'),
        good.replace('columns_calls=0','columns_calls=1'),
        good.replace('native_half_calls=1',f'native_half_calls={2**64}'),
        good.replace('nc=4','nc=9'),good.replace('capture_id=7 ni=','capture_id=8 ni='),
        good.splitlines()[0]+'\n',good.splitlines()[1]+'\n',good+good]
    for value in bad:
        try:snapshots(value,1,2)
        except RuntimeError:continue
        raise RuntimeError('corrupt Q6 snapshot accepted')
    return dict(passed=True,positive=2,negative=len(bad),GPU_executed=False,
        scope='Synthetic parser controls;not execution of the native route writer.')


def model_proof():
    row=json.loads((ROOT/'record.json').read_text());unchanged(row);native=built()
    require(base.committed(RESULT)[KEY]==compact(row) and row['stage']=='closed' and
        row['passed'] and row['complete'] and row['ownership_closed'] and not row['active'] and
        row['GPU_executed'] and row['fresh_faults']==[] and row['boot_unchanged'] and
        [(r['name'],r['native_exit'],r['checks']) for r in row['results']]==
            [('submission',0,10),('q6-half',0,32),('phase-cache',0,16)],'closed same-build device gate required')
    providers=json.loads(Path(row['runtime_manifest']['path']).read_text())
    native=dict(native,environment=native['recipe']['environment'],boot=row['boot'],driver=row['driver'])
    return native,dict(source_commit=row['source_commit'],environment=row['environment']),providers


def model_context(arm):
    # Start from the shared benchmark supervisor, not the legacy Q6 wrapper's
    # fixed spec4 guard. This prospective entry uses the intended spec8 route.
    spec=importlib.util.spec_from_file_location('strata_q6_half_model_entry',model_template.template.__file__)
    ctx=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctx)
    ctx.configure('adaptive-throughput')
    ctx.ROOT=build.ROOT/('model-entry-half'+str(arm)+'-v1')
    ctx.KEY='q6_native_half_model_entry_arm'+str(arm)+'_v1'
    ctx.proof=model_proof
    original_pins=model_template.pinned_sources
    ctx.pinned_sources=lambda revision:dict(original_pins(revision),**base.controls(revision,controller=SELF))
    previous_current,previous_route=ctx.current,ctx.profile_route
    def current(row):
        env=row['environment'];args=row['configuration']['args']
        require(env['STRATA_SYCL_DECODE_Q6_NATIVE_HALF']==str(arm) and
            env['STRATA_SYCL_Q6_ROUTE_AUDIT']==env['STRATA_SYCL_DECODE_Q6_COLUMNS']==env['STRATA_IQ3_PAIR']==env['STRATA_IQ3_PAIR_AUDIT']=='1' and
            env['STRATA_SYCL_DECODE_Q6_CENTERING']==env['STRATA_CPU_SERVICE']==env['STRATA_CPU_WORK_CENSUS']==env['STRATA_DECODE_TIMING']=='0' and
            args[args.index('--spec')+1]=='8' and args[args.index('--short-read')+1]=='0' and
            args[args.index('--adapt-swaps')+1]=='0' and args[args.index('--expert-cache')+1]=='1256' and
            row['read_tokens']==32769 and row['runs']==2,
            'freeze the independently selected half arm and route observer')
        return previous_current(row)
    def route(text):
        value=previous_route(text);ids=dispatches(text,1,positive=True)
        require(len(ids)==1 and ids[0]['request'] in (2,3),'join current native request')
        value['Q6_live_snapshot']=snapshots(text,arm,ids[0]['request'])
        return value
    ctx.current,ctx.profile_route=current,route
    return ctx


def model_prepare(ctx,arm):
    controls=parser_controls();ctx.prepare()
    row=json.loads((ctx.ROOT/'record.json').read_text())
    row['environment'].update(STRATA_SYCL_DECODE_Q6_NATIVE_HALF=str(arm),STRATA_SYCL_Q6_ROUTE_AUDIT='1',
        STRATA_SYCL_DECODE_Q6_COLUMNS='1',STRATA_SYCL_DECODE_Q6_CENTERING='0',STRATA_SYCL_ADAPT_SYNC='1',
        STRATA_IQ3_PAIR='1',STRATA_IQ3_PAIR_AUDIT='1',STRATA_IQ_SINGLE_EXACT='0',
        STRATA_CPU_SERVICE='0',STRATA_CPU_WORK_CENSUS='0',STRATA_DECODE_TIMING='0')
    args=row['configuration']['args'];require(args[args.index('--spec')+1]=='4','original template spec changed')
    args[args.index('--spec')+1]='8'
    write(ctx.ROOT/'config.json',row['configuration']);row['generated']['config.json']=digest(ctx.ROOT/'config.json')
    row['half_arm'],row['Q6_snapshot_parser_controls']=arm,controls
    row['qualification_basis']=dict(build_key=build.KEY,device_key=KEY,
        source_commit=build.POLICY_SOURCE,kernel_source=build.KERNEL_SOURCE,incumbent_shapes=30,selected_intersection=26)
    row['criteria']=('Two pinned frontend probes,one short thinking-on warmup,and two fresh32769-input/1..24-output bench.sh requests.'
        ' Spec8,short-read0,1256decode slots,adapt-swaps0,chunk8192,throughput,GDN16/MTP16,and immutable runtime safety.'
        ' Require complete existing body/token/route/cache/dense/head/MTP/clock/validation/asset/fault/owner gates.'
        ' Require request-bound complete live phase/T/shape snapshots with exact independently measured routes,'
        'no selection in observed prompt snapshots,closed replay counts,and positive completed Q6 output-head use in the declared arm.'
        'Snapshot counters are cumulative for live capture epochs;do not call them request deltas or retired history.'
        'This audit-on short-output model entry is not clean/sustained TG,prompt-verify,task or physical262144/RESTORE qualification.')
    row['retention']=('Keep compact request identities,clocks,observed live shape/replay evidence for the imminent half0/1 model comparison;'
        'review those detail arrays after that decision. Retire successful raw output/JIT after useful outcomes and exact-source recovery are committed.')
    ctx.save(row);print('Prepared immutable half'+str(arm)+' thinking-on32K route entry;commit before probes.',flush=True)


if __name__=='__main__':
    device_mode=len(sys.argv)==2 and sys.argv[1] in ['prepare','run']
    parser_mode=len(sys.argv)==2 and sys.argv[1]=='parser'
    model_mode=len(sys.argv)==3 and sys.argv[1] in ['0','1'] and sys.argv[2] in ['prepare','host','run']
    require(device_mode or parser_mode or model_mode,'use prepare|run|parser,or 0|1 prepare|host|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        if parser_mode:print(json.dumps(parser_controls()))
        elif device_mode:{'prepare':prepare,'run':run}[sys.argv[1]]()
        else:
            arm=int(sys.argv[1]);ctx=model_context(arm)
            {'prepare':lambda:model_prepare(ctx,arm),'host':ctx.host,'run':ctx.run}[sys.argv[2]]()
