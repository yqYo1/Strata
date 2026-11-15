#!/usr/bin/env python3
"""Observe pinned launcher startup environment without a model workload.

Main-authored; repository LICENSE. Collector/provider constructors may run,
but the target only reads its own environment/maps and does no GPU arithmetic.
This diagnostic never refreshes a failed model plan or relaxes its checker.
"""
import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
import current_request_profile as campaign
import current_pti_probe as small

P,B=campaign.P,campaign.B
ROOT,KEY=B/'current-pti-environment-probe-v3','unitrace_current_environment_probe_v3'
PRIOR_ROOT,PRIOR_KEY=B/'current-request-profile-v2','unitrace_current_request_profile_v2'
THIS='tools/sycl/profile_environment_probe.py'
SCRIPT=('import os,json;print(json.dumps(dict(pid=os.getpid(),exe=os.readlink("/proc/self/exe"),'
    'environment=dict(v.split("=",1) for v in open("/proc/self/environ","rb").read().decode().split("\\0") if "=" in v))))')
require,sha=campaign.require,campaign.sha


def save(row):
    (ROOT/'record.json').write_text(json.dumps(row,indent=2)+'\n')
    data=json.loads(campaign.RESULT.read_text())
    data[KEY]={k:v for k,v in row.items() if k not in ('sources','environment','expected_environment')}
    data[KEY]['source_inputs_sha256']=campaign.signature(row['sources'])
    campaign.RESULT.write_text(json.dumps(data,indent=2)+'\n')


def prepare():
    require(not ROOT.exists(),'one fresh diagnostic')
    prior=json.loads((PRIOR_ROOT/'record.json').read_text())
    published=campaign.admission.committed_json(campaign.RESULT,P)[PRIOR_KEY]
    # Only the later explanation is outside the original failed run receipt.
    # Every original result/qualification field must still match exactly.
    require({k:v for k,v in published.items() if k!='post_closure_analysis'}==campaign.compact(prior)
            and prior['stage']=='failed' and len(prior['processes'])==2
            and not prior['processes'][1]['model_executed'],'preserve the closed pre-long environment rejection')
    admitted,_,_=campaign.basis()
    contract,python=campaign.runner.entry.python_manifest()
    require(contract==prior['python_contract'] and python['selected']==prior['python_selected'],'same qualified interpreter required')
    plan=dict(prior['plans'][1],root=str(ROOT));native=campaign.native_context(prior,plan)
    expected=campaign.runner.engine_once.native_environment(native,campaign.environment(prior,plan))
    profiling=campaign.profile(prior,plan);env=dict(expected)
    env['STRATA_UNITRACE_TARGET_PRELOAD']=env.pop('LD_PRELOAD');env.update(profiling['environment'])
    require(not any(k.upper().endswith(('_TOKEN','_PASSWORD','_SECRET','_API_KEY')) for k in env),
            'diagnostic accepts only the controlled non-credential benchmark environment')
    revision=campaign.admission.pti.git(P,'rev-parse','HEAD').decode().strip()
    row=dict(stage='prepared',source_commit=revision,sources=campaign.source_pins(revision),admission=admitted,
        prior_failure=PRIOR_KEY,python_contract=contract,python_selected=python['selected'],
        checker_source_sha256=sha(P/campaign.THIS),
        boot=prior['boot'],driver=prior['driver'],environment=env,expected_environment=expected,plan=plan,
        native_profile=profiling,command=campaign.runner.engine_once.profile_command(profiling,[python['selected']['path'],'-I','-c',SCRIPT]),
        budgets=dict(wall=60,cpu=30,rss_cap=1<<30,text_cap=128<<10,file_cap=1<<20,total_cap=4<<20),
        passed=False,complete=False,active=False,ownership_closed=True,
        GPU_workload_executed=False,runtime_constructor_initialization_possible=True,
        question='Which exact pinned launcher additions cause the pre-long native environment checker to reject? '
                 'Run only Python self environment/maps,not model inference;keep the old checker result separate from execution success.',
        retention='Keep added environment names and selected public tracing fields,source/normal-exit/owner/fault results;'
                  'retire normal private environment dump/logs at the correction boundary. No credentials,model or hardware-speed claim.')
    require(THIS in row['sources'],'commit the actual diagnostic controller')
    ROOT.mkdir(mode=0o700)
    for name in ('trace','tmp','cache'): (ROOT/name).mkdir()
    (ROOT/'plan.json').write_text(json.dumps(dict(native_profile=profiling),indent=2)+'\n')
    row['generated_plan_sha256']=sha(ROOT/'plan.json');save(row)
    print('Prepared bounded launcher-environment diagnostic;commit before execution.')


def run():
    row=json.loads((ROOT/'record.json').read_text());require(row['stage']=='prepared','one declared execution')
    public={k:v for k,v in row.items() if k not in ('sources','environment','expected_environment')}
    public['source_inputs_sha256']=campaign.signature(row['sources'])
    require(campaign.admission.committed_json(campaign.RESULT,P)[KEY]==public
            and all(sha(P/n)==h for n,h in row['sources'].items())
            and sha(ROOT/'plan.json')==row['generated_plan_sha256'],'frozen diagnostic inputs changed')
    now,_,_=campaign.basis();require({k:v for k,v in now.items() if k!='controller_commit'}==
        {k:v for k,v in row['admission'].items() if k!='controller_commit'},'frozen source/assets changed')
    journal=campaign.admission.subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(v.split(': ',1)[1] for v in journal.splitlines() if v.startswith('-- cursor: '))
    owner=campaign.runner.build.supervision.Owner(ROOT/'owned')
    try:
        entry,out,err=owner.run('environment',row['command'],row['environment'],P,**row['budgets'])
        require(campaign.runner.build.supervision.closed(entry) and entry['exit_code']==0,'normal diagnostic closure required')
        actual=json.loads(out.read_text());require(actual['pid']==entry['pid'] and
            actual['exe']==row['python_selected']['file']['realpath'],'actual target interpreter identity differs')
        observed=actual['environment'];require(type(observed) is dict,'actual exec environment required')
        try:
            campaign.check_native_environment({k.encode():v.encode() for k,v in observed.items()},row['expected_environment'],row['plan'])
            checker=dict(accepted=True)
        except ValueError as error:checker=dict(accepted=False,error=str(error))
        selected={k:v for k,v in observed.items() if k.startswith('UNITRACE_') or k in
            ('LD_PRELOAD','INTEL_LIBITTNOTIFY64','ZE_ENABLE_TRACING_LAYER','ZES_ENABLE_SYSMAN','ZET_ENABLE_METRICS')}
        row.update(passed=True,complete=True,host_probe_accepted=True,checker_result=checker,native_exit_code=0,
            added_environment_names=sorted(set(observed)-set(row['expected_environment'])),
            selected_public_trace_environment=selected,actual_environment_sha256=sha(out),
            actual_native_pid=actual['pid'])
    except BaseException as error:row['error']=repr(error)
    finally:
        small.validation.finish(row,owner,cursor,row['boot'])
        row['passed']=bool(row['passed'] and row.get('driver_after')==row['driver'])
        row.update(active=False,stage='closed' if row['passed'] else 'failed')
        save(row)
    require(row['passed'],row.get('error','diagnostic failed'))
    print('Launcher environment observed;current checker accepted='+str(row['checker_result']['accepted']))
    print('Added names: '+','.join(row['added_environment_names']))


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);campaign.admission.pti.guards.no_model_owners([])
        (prepare if sys.argv[1]=='prepare' else run)()
