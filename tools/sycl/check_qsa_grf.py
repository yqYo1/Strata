#!/usr/bin/env python3
"""Bounded, first-use QSA GRF validation; does not launch the model engine."""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import build_qsa_grf as build
import measurement_guards as guards
import measurement_owner as supervision
import query_profile as qualified
import ur_validation as validation
from measure_down_scatter import postchecks

P,B,ROOT,RESULT=build.P,build.B,build.ROOT,build.RESULT


def inputs(revision):
    pending=[Path(__file__).resolve()];seen=set();pins={}
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path)
        raw=build.gen.git('show',revision+':'+str(path.relative_to(P)))
        guards.require(raw==path.read_bytes(),'commit exact check input: '+str(path))
        pins[str(path.relative_to(P))]=hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            modules=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                local=path.parent/((module or '').split('.')[0]+'.py')
                if local.is_file():pending.append(local)
    return pins


def runtime(record):
    qualified.runtime_assets_unchanged(dict(mode='engine-compare',runtime_asset_delta=record['runtime_asset_delta']))
    guards.require(validation.preload_qualification()==record['validation_preload'],'validation provider changed')


def admission(record,plan):
    build.admission(record,True)
    guards.require(inputs(plan['source_commit'])==plan['controller_inputs'],'check controller changed')
    runtime(plan)
    guards.require(record['stage']=='built' and record['passed'] and record['complete'] and
        record['ownership_closed'] and not record['active'],'closed isolated build required')


def prepare(mode):
    record=build.gen.committed(RESULT)
    guards.require('check_plan' not in record or record['check_plan']['stage']=='closed','previous check not closed')
    checks=record.get('checks',[])
    guards.require((mode=='smoke' and not checks) or (mode=='component' and len(checks)==1 and
        checks[0]['mode']=='smoke' and checks[0]['passed']),'require one smoke then component; no retries')
    root=ROOT/('check-'+mode);guards.require(not root.exists(),'inspect existing check owner; no replacement')
    _,_,setup=build.resource.qualification()
    revision=build.gen.git('rev-parse','HEAD').decode().strip()
    plan=dict(stage='prepared',mode=mode,source_commit=revision,controller_inputs=inputs(revision),
        runtime_asset_delta=setup['runtime_asset_delta'],validation_preload=validation.preload_qualification(),
        gates='All3explicit process selectors must complete original independent tensor-oracle cases,exact prefix33 bits,guards,input immutability,alias/domain/live-extent rejections. Smoke2cases;component11including physical262144 source addressing. Same-PID preload proves first UR init and parameter validation. Normal native/session closure and no new driver fault required. No model,decode,task-quality or speed qualification.',
        budgets=dict(wall_s=180,cpu_s=180,rss_bytes=4<<30,text_bytes=16<<20,file_bytes=32<<20,total_bytes=192<<20),
        jobs=[])
    admission(record,plan);root.mkdir(mode=0o700)
    base=build.gen.committed(P/'tools/sycl/measurement_environment.json')['environment']
    for arm in ('default','128','256'):
        folder=root/arm;folder.mkdir()
        for name in ('tmp','cache'):(folder/name).mkdir()
        env=dict(base,TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),
            LD_PRELOAD=plan['validation_preload']['library']['path'],
            STRATA_UR_PROBE_EXPECTED_LOADER=plan['validation_preload']['loader']['path'])
        if mode=='component':
            env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION';env.pop('UR_LOG_TRACING',None)
        plan['jobs'].append(dict(arm=arm,environment=env,command=[record['binaries']['numerical']['path'],'--'+mode,arm]))
    record['check_plan']=plan
    build.write(root/'plan.json',plan);build.write(RESULT,record)
    print('Prepared '+mode+'; commit exact controller and plan before execution.',flush=True)


def run(mode):
    record=build.gen.committed(RESULT);plan=record['check_plan'];root=ROOT/('check-'+mode)
    guards.require(plan['mode']==mode and plan['stage']=='prepared' and
        plan==json.loads((root/'plan.json').read_text()) and not (root/'record.json').exists(),
        'matching fresh committed check required; inspect existing receipt')
    admission(record,plan)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
    result=dict(mode=mode,source_commit=plan['source_commit'],numerical_binary=record['binaries']['numerical'],
        passed=False,complete=False,active=False,ownership_closed=False,observations=[])
    owner=None
    def persist():
        result['active']=bool(owner and owner.active is not None)
        build.write(root/'record.json',dict(result,owned_commands=owner.commands if owner else []))
    owner=supervision.Owner(root/'owned',persist);persist()
    try:
        for job in plan['jobs']:
            # No controller/environment changes, even between independent arms.
            guards.require(inputs(plan['source_commit'])==plan['controller_inputs'],'controller changed during comparison')
            runtime(plan);guards.driver_guard();guards.no_model_owners([])
            guards.require(build.digest(record['binaries']['numerical']['path'])==record['binaries']['numerical']['sha256'],'fixture changed')
            print('Validating '+mode+' GRF '+job['arm'],flush=True)
            b=plan['budgets']
            entry,out,err=owner.run(job['arm'],job['command'],job['environment'],P,
                wall=b['wall_s'],cpu=b['cpu_s'],rss_cap=b['rss_bytes'],text_cap=b['text_bytes'],
                file_cap=b['file_bytes'],total_cap=b['total_bytes'])
            stdout,stderr=out.read_text(),err.read_text();lines=stdout.splitlines()
            guards.require(supervision.closed(entry) and entry['exit_code']==0,'numerical fixture did not close normally')
            proof=validation.preload_evidence(stderr,entry['pid'])
            guards.require(proof['loader']==plan['validation_preload']['loader']['path'],'unexpected loader')
            # The pinned preload deliberately tests a null UR platform before
            # main (expected46); reject errors after that verified diagnostic.
            body_stderr=stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
            guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:',body_stderr),
                'runtime/validation diagnostic failed')
            count=2 if mode=='smoke' else 11
            cases=[x for x in lines if x.startswith('case cells=')]
            guards.require(len(cases)==count and all('status=PASS' in x for x in cases) and
                sum(x.startswith('case exact_prefix33_bits=PASS') for x in lines)==count and
                'fixture_status=PASS model_quality=UNRUN model_PP=UNRUN physical262144_model=UNRUN faults=EXTERNAL_REQUIRED' in lines
                and 'grf_control variant='+job['arm']+' arithmetic=unchanged' in lines,'incomplete fixture cases')
            result['observations'].append(dict(arm=job['arm'],pid=entry['pid'],exit_code=entry['exit_code'],
                ownership_closed=True,validation=proof,case_count=count,cases=cases,
                exact_prefix33_bits=True,physical262144_component_addressed=mode=='component',
                stdout_sha256=build.digest(out),stderr_sha256=build.digest(err)))
            persist()
        result.update(passed=True,complete=True)
    except BaseException as error:result['error']=repr(error)
    finally:
        result['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        postchecks(result,boot,cursor);persist()
        plan['stage']='closed' if result['passed'] else 'failed'
        record.setdefault('checks',[]).append(result);record['GPU_executed']=True
        record['model_adoption_qualified']=False;build.write(RESULT,record)
    print(json.dumps({k:result.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    guards.require(len(sys.argv)==3 and sys.argv[1] in ('prepare','run') and sys.argv[2] in ('smoke','component'),
        'use prepare|run smoke|component')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        sys.exit(prepare(sys.argv[2]) if sys.argv[1]=='prepare' else run(sys.argv[2]))
