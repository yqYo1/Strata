#!/usr/bin/env python3
"""Run the bounded same-build Q6/phase admission; Main, repository LICENSE.

Reuse this fork's qualified runtime/owner/parser contracts. No model, speed,
physical saturation, ordinary teardown or full-context qualification follows.
"""
import fcntl
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
import build_q6_native_half_model as build
import check_verify_window_phase as phase
import measurement_owner as ownership
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


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ['prepare','run'],'use prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        {'prepare':prepare,'run':run}[sys.argv[1]]()
