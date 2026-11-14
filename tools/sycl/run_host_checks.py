#!/usr/bin/env python3
"""Run a committed CPU/mock unittest plan with the existing direct owner.

Main-authored; repository LICENSE. Reusable for required frontend imports.
No native engine,build,profiler or device recovery is launched here.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import benchmark_runtime as runtime
import build_engine as inputs
import measurement_guards as guards
import measurement_owner as owner

P,B = inputs.P,inputs.B
RESULT = P/'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
require = guards.require


def committed():
    raw = inputs.git(P,'show','HEAD:'+str(RESULT.relative_to(P)))
    require(RESULT.read_bytes()==raw,'commit the exact host plan/results first')
    return json.loads(raw)


def current(key,frozen):
    require(committed()[key]==frozen,'host plan changed')
    recovered = {name:hashlib.sha256(raw).hexdigest() for name,raw in
                 inputs.blobs(P,frozen['source_commit'],frozen['sources'])}
    require(recovered==frozen['sources'] and all(inputs.digest(P/n)==h
            for n,h in recovered.items()),'host source/template inputs changed')
    env = frozen['environment']
    runtime.require_environment(env)
    manifest = runtime.read_manifest(frozen['python_contract'])
    runtime.validate_manifest(manifest)
    require(runtime.selection(env)==frozen['python_selected'],'host interpreter changed')
    address_cap = frozen.get('virtual_address_bytes',2<<30)
    require(type(address_cap) is int and 1<<30 <= address_cap <= 16<<30,
            'finite host virtual-address budget required')
    for case in frozen['commands']:
        argv = case['command']
        require(re.fullmatch(r'[a-z0-9-]+',case['role']) and argv==[
            '/usr/bin/timeout','-k','5s','300s','/usr/bin/prlimit',f'--as={address_cap}',
            '--cpu=120','--fsize=8388608',frozen['python_selected']['path'],
            '-X','faulthandler','-B','-m','unittest','-v',*case['modules']] and
            case['modules'] and all(re.fullmatch(r'(serve|tools)\.[A-Za-z0-9_.]+',n)
                for n in case['modules']),'unexpected host unittest command')


def run(key):
    data = committed();frozen = copy.deepcopy(data[key]);root = Path(frozen['root'])
    require(frozen['stage']=='prepared' and not frozen['active'] and
            root.parent==B and not root.exists(),'one fresh private host attempt')
    current(key,frozen)
    root.mkdir(mode=0o700)
    for name in ('tmp','cache'):(root/name).mkdir(mode=0o700)
    result = dict(tests=[],passed=False,complete=False,GPU_executed=False,model_executed=False)
    supervisor = None
    def persist():
        record = dict(result,active=bool(supervisor and supervisor.active),
            commands=supervisor.commands if supervisor else [])
        (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    try:
        supervisor = owner.Owner(root/'owned',persist)
        for case in frozen['commands']:
            current(key,frozen);guards.no_model_owners([])
            entry,stdout,stderr = supervisor.run(case['role'],case['command'],frozen['environment'],
                P,**frozen['budgets'])
            text = stderr.read_text(errors='replace')
            count = re.search(r'^Ran (\d+) tests? in [0-9.]+s$',text,re.M)
            status = re.search(r'^OK(?: \(skipped=(\d+)\))?$',text,re.M)
            test = dict(role=case['role'],exit_code=entry.get('exit_code'),
                ownership_closed=owner.closed(entry),tests=int(count[1]) if count else None,
                skipped=int(status[1] or 0) if status else None,
                skip_reasons=[line for line in text.splitlines() if ' ... skipped ' in line],
                stdout_bytes=stdout.stat().st_size,stderr_bytes=stderr.stat().st_size)
            result['tests'].append(test);persist()
            require(owner.closed(entry) and entry['exit_code']==0 and count and status and
                    int(count[1])>0 and len(test['skip_reasons'])==test['skipped'],
                    'host suite failed: '+case['role']+'\n'+text[-4096:])
            current(key,frozen);guards.no_model_owners([])
            print(json.dumps(test),flush=True)
        result.update(passed=True,complete=True)
    except BaseException as error:
        result['error']=repr(error)
    finally:
        result.update(active=bool(supervisor and supervisor.active),
            ownership_closed=not supervisor or supervisor.active is None and
                all(owner.closed(v) for v in supervisor.commands))
        result['passed']=bool(result['passed'] and result['ownership_closed'])
        persist();data[key].update(result,stage='closed' if result['passed'] else 'failed')
        RESULT.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    require(result['passed'],result.get('error','host owner did not close normally'))


if __name__=='__main__':
    require(len(sys.argv)==2 and re.fullmatch(r'[a-z0-9_]+_host_v\d+',sys.argv[1]),
            'use a committed host-plan key')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        run(sys.argv[1])
