#!/usr/bin/env python3
"""Bounded target-loader coexistence probe; no GPU command or model workload."""
import ast
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys

sys.dont_write_bytecode = True
import unitrace_launch as launch
import ur_validation as validation

tool = launch.tool
P, B, RESULT = tool.P, tool.B, tool.RESULT
ROOT = B/'unitrace-target-loader-v2'


def inputs(revision):
    pending = [Path(__file__).resolve()]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen: continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                if module:
                    local = path.parent/(module.split('.')[0]+'.py')
                    if local.is_file(): pending.append(local)
    seen.add(P/'tools/sycl/measurement_environment.json')
    result = {}
    for path in sorted(seen):
        name = str(path.relative_to(P));original = tool.git(P,'show',revision+':'+name)
        tool.guards.require(original==path.read_bytes(),'commit exact probe input: '+name)
        result[name]=tool.sha(original)
    return result


def pin_assets(container):
    collector, owner = container['unitrace_build'], container['unitrace_session_owner']
    tool.guards.require(collector['passed'] and owner['passed'] and collector['ownership_closed'] and
        owner['ownership_closed'] and not collector['active'] and not owner['active'], 'closed collector/launcher builds required')
    launch.validate(owner,collector)
    tool.guards.require(tool.sha(Path(owner['binary']['path']).read_bytes())==owner['binary']['sha256'], 'launcher bytes changed')
    probe = validation.preload_qualification()
    # This named current comparison still consumes the unchanged status engine and fixtures.
    import query_profile
    build, count = query_profile.current_build()
    canary=json.loads(validation.INDEX.read_text())['UR_silent_validation_native_child_scope']['preload_component']['canary']
    tool.guards.require(tool.sha(Path(canary['path']).read_bytes())==canary['sha256'],'canary bytes changed')
    return dict(collector=collector['artifacts']['libunitrace_tool.so'],launcher=owner['binary'],preload=probe,canary=canary,
                native_source_commit=build['source_commit'],native_inputs_recovered=count,native_binary=build['binary'])


def main(action):
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tool.guards.no_model_owners([])
        container = json.loads(RESULT.read_text());record=container.get('unitrace_loader_probe')
        assets=pin_assets(container)
        if action=='prepare':
            tool.guards.require(record and record['stage']=='prepared' and not record['active'] and
                record['source_commit'].startswith('c407208b0') and not (B/'unitrace-target-loader-v1/record.json').exists()
                and not ROOT.exists(), 'only the reviewed unexecuted v1 plan permits this new v2 plan')
            revision=tool.git(P,'rev-parse','HEAD').decode().strip();pins=inputs(revision)
            previous=RESULT.read_bytes()
            tool.guards.require(previous==tool.git(P,'show',revision+':'+str(RESULT.relative_to(P))),'commit previous unexecuted plan first')
            session='StrataLoader'+secrets.token_hex(12)
            ROOT.mkdir(mode=0o700)
            for name in ('tmp','cache','trace'): (ROOT/name).mkdir()
            environment=dict(json.loads((P/'tools/sycl/measurement_environment.json').read_text())['environment'],
                TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),CUDA_MODULE_LOADING='EAGER',
                STRATA_UNITRACE_TARGET_PRELOAD=assets['preload']['library']['path'],
                STRATA_UR_PROBE_EXPECTED_LOADER=assets['preload']['loader']['path'],
                LD_DEBUG='libs',
                UNITRACE_FollowChildProcess='1',UNITRACE_TraceOutputDir=str(ROOT/'trace'),
                UR_LOG_TRACING='level:info;flush:info;output:file,'+str(ROOT/'ur.log'))
            tool.guards.require('LD_PRELOAD' not in environment and environment['ZES_ENABLE_SYSMAN']=='0', 'target-only preload/Sysman-off required')
            record=dict(schema='strata-unitrace-loader-probe-v1',stage='prepared',source_commit=revision,input_sha256=pins,assets=assets,
                previous_unexecuted_plan=dict(commit=revision,path=str(RESULT.relative_to(P)),sha256=tool.sha(previous),
                    outcome='V1 superseded before any execution after independent review: add unique session, failure-path cleanup and direct constructor/retention evidence. Original unrun outcome unchanged.'),
                environment=environment,command=[assets['launcher']['path'],'--session',session,'--start-paused',
                    '--host-timing','--device-timing','--chrome-device-logging','--chrome-event-buffer-size','128',
                    assets['canary']['path']],
                stop_command=[assets['launcher']['path'],'--stop',session],session=session,
                gates='Independent review f35cc9ba4 required corrections applied. One launcher creates a unique paused session and execs original CPU-only UR canary. Same-PID preload proves single-thread first UR init and validation before body. LD_DEBUG libs must show validation init/proof before collector init, then canary begin/end before collector fini. Canary init/check/close succeeds. All processes close normally; journal/boot checks find no fresh fault. Finally stop/unlink only this exact session after proven target closure, including failure paths; leave unknown/live owners untouched. No result-dir, device/metric selection, KMD, Sysman, XPTI or ITT option. Collector may enumerate LevelZero devices but no target/GPU commands or model. Loader success does not qualify GPU timing, temporal gating, graph/queue coverage or32K+ production.',
                retention='Retire normal API trace and build/loader output after recording same-PID proof and next GPU smoke plan. Keep one distinct failure excerpt if unresolved.',
                budgets=dict(wall_s=60,cpu_s=60,rss_bytes=2<<30,text_bytes=8<<20,file_bytes=16<<20,total_bytes=64<<20),
                active=False,complete=False,passed=False,gpu_work=False,gpu_qualified=False,model_qualified=False)
            container['unitrace_loader_probe']=record
            RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
            print('Prepared CPU-only loader probe; independent review and committed plan required before run.')
            return 0
        tool.guards.require(action=='run' and record and record['stage']=='prepared','committed prepared loader plan required')
        tool.guards.require(RESULT.read_bytes()==tool.git(P,'show','HEAD:'+str(RESULT.relative_to(P))),'commit loader plan first')
        tool.guards.require(inputs(record['source_commit'])==record['input_sha256'] and assets==record['assets'], 'loader inputs/assets changed')
        session_path=Path('/dev/shm/uctrl'+str(os.getuid())+'_'+record['session'])
        tool.guards.require(not session_path.exists(),'session already exists; inspect, never overwrite')
        driver=tool.guards.driver_guard();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
        cursor=next(line.split(': ',1)[1] for line in journal.splitlines() if line.startswith('-- cursor: '))
        owner=None

        def persist():
            record['active']=bool(owner and owner.active is not None)
            (ROOT/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')

        owner=tool.supervision.Owner(ROOT/'owned',persist);record.update(stage='running',driver_before=driver);persist()
        entry=None
        try:
            entry,out,err=owner.run('target',record['command'],record['environment'],ROOT,wall=60,cpu=60,
                rss_cap=2<<30,text_cap=8<<20,file_cap=16<<20,total_cap=64<<20)
            tool.guards.require(tool.supervision.closed(entry) and entry.get('exit_code')==0,'loader target failed; inspect existing owned output')
            body=json.loads(out.read_text());proof=validation.preload_evidence(err.read_text(),body['pid'])
            tool.guards.require(body['pid']==entry['pid'] and body==dict(pid=body['pid'],initialized=0,checked=46,expected=46,version=4294967295,closed=0)
                and proof['loader']==assets['preload']['loader']['path'],'CPU canary or validation proof failed')
            record.update(preload_evidence=proof,canary=body,target_closed=True)
            text=err.read_text()
            def loader_marker(phase,expected):
                matches=[m for m in re.finditer(r'calling '+phase+r':\s+([^\s]+)',text)
                         if Path(m[1]).resolve()==Path(expected).resolve()]
                tool.guards.require(len(matches)==1,'missing/ambiguous loader '+phase+' marker: '+expected)
                return matches[0].start()
            val=loader_marker('init',assets['preload']['library']['path'])
            start=loader_marker('init',assets['collector']['path']);end=loader_marker('fini',assets['collector']['path'])
            tool.guards.require(val<text.index('phase=before-main')<start<text.index('strata UR canary begin')
                <text.index('strata UR canary end')<end,'constructor order/retention evidence failed')
            record.update(loader_order_verified=True,collector_retained_to_exit=True,stage='closed',complete=True,passed=True)
        except BaseException as error:record.update(stage='failed',error=repr(error),passed=False)
        finally:
            record['target_closed']=bool(entry and tool.supervision.closed(entry))
            if record['target_closed'] and owner.active is None:
                try:
                    if session_path.exists():
                        stop,_,_=owner.run('stop',record['stop_command'],record['environment'],ROOT,wall=15,cpu=15,
                            rss_cap=2<<30,text_cap=1<<20,file_cap=16<<20,total_cap=64<<20)
                        tool.guards.require(tool.supervision.closed(stop) and stop.get('exit_code')==0,'session stop failed')
                    record['session_removed']=not session_path.exists()
                except Exception as error:record.update(cleanup_error=repr(error),passed=False)
            else:
                record.update(session_removed=False,cleanup_pending='Target closure unproven; leave exact session/owners untouched.',passed=False)
            record['ownership_closed']=owner.active is None and all(tool.supervision.closed(c) for c in owner.commands)
            try:
                journal=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
                record['fresh_faults']=[line for line in journal.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',line)]
                record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                record['driver_after']=tool.guards.driver_guard()
            except Exception as error:record['final_probe_error']=str(error)
            record['passed']=bool(record['passed'] and record['ownership_closed'] and record.get('session_removed') and record.get('fresh_faults')==[]
                and record.get('boot_unchanged') and 'final_probe_error' not in record)
            persist();container['unitrace_loader_probe']=record
            RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({k:record.get(k) for k in ('stage','passed','ownership_closed','gpu_qualified','error')}))
        return 0 if record['passed'] else 1


if __name__=='__main__':
    tool.guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'use unitrace_probe.py prepare|run')
    raise SystemExit(main(sys.argv[1]))
