#!/usr/bin/env python3
"""Recover the closed native router samples after the Python summary failed.

Main-authored. Repository LICENSE. This replays earlier evidence without
starting a GPU process or altering the original numerical criteria.
"""
import fcntl
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import prefill_prefix_router as original

KEY = 'prefill_prefix_router_replay_v1'
require, sha, write = original.require, original.sha, original.write


def captured(case):
    record = case/'execution.json'
    outcome=json.loads(record.read_text())
    commands=outcome.pop('commands')
    active=outcome.pop('active')
    require(not active and len(commands)==1
        and original.basis.comparison.build.supervision.closed(commands[0])
        and commands[0]['exit_code']==0 and outcome['complete'] and outcome['passed']
        and outcome['ownership_closed'] and not outcome['fresh_faults']
        and outcome['request_routes_checked_before_continue']==original.RUNS
        and len(outcome['routes'])==len(outcome['samples'])==original.RUNS, 'native observation did not close')
    return outcome


def mtp_restores(case):
    text=(case/'engine.stderr').read_text()
    phases=re.findall(r'^strata mtp: prompt expert phase=(\S+) ',text,re.M)
    values=re.findall(r'^strata mtp: experts restored bytes=(\d+),',text,re.M)
    require(phases==['before-detach','after-detach','after-prompt-before-restore','after-restore']*original.RUNS
        and values==['707788800']*original.RUNS,'actual request handoff evidence differs')
    return list(map(int,values))


def recover_old():
    private=json.loads((original.ROOT/'record.json').read_text())
    require(private['stage']=='running' and len(private['processes'])==2 and not private['active'],
        'preserve the original unfinished summary state')
    ctx=original.model_context()
    old=original.basis.validation.committed(original.RESULT,(original.KEY,))
    require(ctx.compact(private)==old,'exact original committed freeze required')
    processes=[captured(Path(plan['root'])) for plan in private['plans']]
    require(processes[:2]==private['processes'] and processes[-1]['index']==2
        and all(p['binary']==private['binary'] and p['native_source_commit']==private['native_source_commit']
                and p['boot_unchanged'] and p['driver_after']==private['driver'] for p in processes),
        'reconcile the original launched process identities')
    try:
        original.numerical_decision(processes)
    except KeyError as error:
        require(error.args==('MTP_expert_restore_bytes',),'different aggregation failure')
    else:
        raise RuntimeError('original missing field failure did not reproduce')
    return private,processes


def prepare():
    public=json.loads(original.RESULT.read_text())
    require(KEY not in public,'fresh replay only')
    native,processes=recover_old()
    old=public[original.KEY]
    old.update(stage='failed',complete=False,passed=False,active=False,ownership_closed=True,
        error="KeyError('MTP_expert_restore_bytes') in final aggregation after normal owned model exits",
        processes=processes)
    revision=original.source.git(original.P,'rev-parse','HEAD').decode().strip()
    public[KEY]=dict(stage='prepared',source_commit=revision,controller_sha256=sha(__file__),
        original_plan_source=native['source_commit'],original_failure_unchanged=True,
        root=str(original.ROOT),binary=native['binary'],
        receipts=[dict(path=str(Path(p['root'])/'execution.json'),sha256=sha(Path(p['root'])/'execution.json'),
                       engine_log_sha256=sha(Path(p['root'])/'engine.stderr')) for p in native['plans']],
        plan='Re-read both original per-process logs and journals for each arm. Recover the MTP restore count already checked by request_route from actual native stdout, then rerun the original decision. Verify all 15 model asset hashes. No new native execution.',
        passed=False,ownership_closed=True)
    write(original.RESULT,public)


def replay():
    public=json.loads(original.RESULT.read_text())
    source_row=original.basis.validation.committed(original.RESULT,(KEY,))
    old=original.basis.validation.committed(original.RESULT,(original.KEY,))
    require(public[KEY]==source_row and public[original.KEY]==old
        and source_row['stage']=='prepared' and source_row['controller_sha256']==sha(__file__)
        and old['stage']=='failed' and not old['passed'] and old['ownership_closed'],'commit original failure and replay plan')
    path=Path(__file__).resolve().relative_to(original.P)
    require(original.source.git(original.P,'show',source_row['source_commit']+':'+str(path))==Path(__file__).read_bytes(),
        'frozen replay controller differs')
    private=json.loads((original.ROOT/'record.json').read_text())
    require(private['source_commit']==source_row['original_plan_source'],'original model plan changed')
    native=json.loads((original.BUILD_ROOT/'record.json').read_text())
    original.check_build(native)
    require(native['binaries']['strata']==source_row['binary']
        and all(sha(v['path'])==v['sha256'] for v in native['binaries'].values()),'same original build differs')
    processes=[]
    for plan,receipt,outcome in zip(private['plans'],source_row['receipts'],old['processes'],strict=True):
        case=Path(plan['root'])
        require(sha(case/'execution.json')==receipt['sha256'] and sha(case/'engine.stderr')==receipt['engine_log_sha256']
            and captured(case)==outcome, 'original native observation changed')
        result,=list((case/'bench').glob('*/results.json'))
        require(original.basis.comparison.shared.client.verify_request_journal(result.parent)['valid'],
                'original request journal changed')
        samples=json.loads(result.read_text())['results']['prefill_only']['requests']
        require([x['request_sha256'] for x in samples]==[x['request_sha256'] for x in outcome['samples']],
                'request body correspondence changed')
        for route,value in zip(outcome['routes'],mtp_restores(case),strict=True):
            require('MTP_expert_restore_bytes' not in route,'original missing projection differs')
            route['MTP_expert_restore_bytes']=value
        processes.append(outcome)
    decision=original.numerical_decision(processes)
    original.basis.comparison.shared.models.model_pins(private['configuration'],private['model_assets'],full=True)
    source_row.update(stage='closed',passed=True,decision=decision,model_assets_rehashed=15,
        replayed_requests=len(processes)*original.RUNS,new_native_processes=0)
    # The restored fields exist only in the projected inputs to the comparison.
    # Keep the original per-process evidence byte-for-byte in the published row.
    published=json.loads(original.RESULT.read_text())
    require(published==public,'published original evidence changed during replay')
    published[KEY]=source_row
    write(original.RESULT,published)
    print(decision)


if __name__=='__main__':
    require(sys.executable==original.PYTHON and sys.argv[1:] in (['prepare'],['replay']),'prepare|replay')
    with (original.B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original.builder.guards.no_model_owners([])
        (prepare if sys.argv[1]=='prepare' else replay)()
