#!/usr/bin/env python3
"""Bounded resident native-event query diagnostic; never model qualification."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import measurement_guards as guards
import measurement_owner as supervision
from check_verifier_retirement import DELTA, P, B, A, SETTINGS, digest

ROOT=B/'native-event-gate-v6'
RESULT=A/'native-event-gate-current.json'
BUILD=A/'native-gdn-engine-build.json'
CASE='copy-query-profiled'
COUNTER_KEY='EnableImplicitConvertionToCounterBasedEvents'
CONTROL_COMMIT='79eb60594a40d4915041d18d53286d3944064aa4'


def committed(path):
    data=path.read_bytes()
    guards.require(data==subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(P))],cwd=P),
        'commit exact input first: '+str(path))
    return json.loads(data)


def main():
    guards.require(len(sys.argv)==1,'no options or automatic retries')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        compact=committed(BUILD)
        full=json.loads((B/'main-native-gdn-engine/record.json').read_text())
        guards.require(full['passed'] and full['complete'] and full['ownership_closed'] and not full['active'] and
            all(full[k]==compact[k] for k in ('source_commit','binary','fixtures','compiler','configure_argv')) and
            'sycl_host_event_gate_test' in full['fixtures'] and len(full.get('fixture_external_headers',{}))==3,
            'closed exact native-event fixture build required')
        def unchanged():
            for source in [full['project'],*full['dependencies']]:
                for name,expected in source['source_sha256'].items():
                    guards.require(digest(Path(source['root'])/name)==expected,'build input changed: '+name)
            for item in [full['binary'],*full['fixtures'].values()]:
                guards.require(digest(item['path'])==item['sha256'],'compiled artifact changed')
            for name,expected in full['fixture_external_headers'].items():
                guards.require(digest(name)==expected,'external header changed: '+name)
        unchanged()
        guards.require('tools/sycl/check_event_gate.py' in full['project']['source_sha256'],'controller not pinned by build')
        guards.require(not ROOT.exists(),'inspect existing qualification; no automatic retry')
        previous=committed(RESULT)
        guards.require(previous['ownership_closed'] and not previous['active'] and previous['passed'] and
            previous['complete'] and previous.get('qualification_eligible') is False and
            previous.get('mode')=='native-event-gate-compute-profiling-diagnostic' and
            previous.get('observed_queue_configuration')==dict(copy_profiling=1,compute_profiling=0,
                probe_profiling=0,all_in_order=1) and
            previous.get('native_query_observation',{}).get('queries')==20 and
            previous['native_query_observation']['native_ready']==0,
            'closed profiling-off diagnostic required; never retry an existing configuration')
        control_bytes=subprocess.check_output(['git','show',CONTROL_COMMIT+':'+str(RESULT.relative_to(P))],cwd=P)
        control=json.loads(control_bytes)
        control_reference=dict(commit=CONTROL_COMMIT,path=str(RESULT.relative_to(P)),
            sha256=hashlib.sha256(control_bytes).hexdigest(),source_commit=control['source_commit'])
        guards.require(not control['passed'] and control['ownership_closed'] and not control['active'] and
            control.get('first_failed_case')=='copy-query' and
            control['queue_configuration']['compute']==dict(in_order=True,profiling=True),
            'preserve the original profiled-compute control failure')
        prior_query=control.get('query_call_observation',{}).get('longest',{})
        guards.require(prior_query.get('name')=='resident-query' and prior_query.get('elapsed_us',0)>200000,
            'committed slow profiled native-query control required before the NEO setting contrast')
        previous_reference=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
            path=str(RESULT.relative_to(P)),sha256=digest(RESULT))
        settings=committed(SETTINGS)
        def runtime_unchanged():
            guards.require(set(DELTA)<=set(settings['runtime_elfs']),'unknown diagnostic runtime delta')
            for name,item in settings['runtime_elfs'].items():
                expected=item['sha256']
                if name in DELTA:
                    old,expected=DELTA[name]
                    guards.require(old==item['sha256'],'original runtime baseline differs')
                guards.require(digest(name)==expected,'runtime asset changed: '+name)
        presence=previous['candidate_key_presence']
        guards.require(presence['present'] and presence['key']==COUNTER_KEY and
            presence['path'] in DELTA and presence['sha256']==DELTA[presence['path']][1] and
            digest(presence['path'])==presence['sha256'] and
            COUNTER_KEY.encode()+b'\0' in Path(presence['path']).read_bytes(),
            'exact installed NEO debug key/DSO evidence required; presence alone is not effectiveness')
        guards.require(COUNTER_KEY not in settings['environment'],'baseline already overrides the candidate setting')
        guards.require(control['environment_sha256']==digest(SETTINGS) and
            control['runtime_asset_delta']==previous['runtime_asset_delta'],
            'profiled control uses a different baseline environment/runtime delta')
        record=dict(mode='native-event-gate-counter-event-config-diagnostic',source_commit=full['source_commit'],binary=full['binary'],
            qualification_eligible=False,previous_result_reference=previous_reference,
            profiled_control_reference=control_reference,candidate_key_presence=presence,
            diagnostic_environment_override={COUNTER_KEY:'0'},
            key_source=dict(repository='intel/compute-runtime',commit='050536ff3b6f830199ec99cff8ffcac3d804c40a',
                registry='shared/source/debug_settings/debug_variables_base.inl:282',
                consumer='level_zero/core/source/cmdlist/cmdlist_hw.inl:5155',
                meaning='0 disables implicit regular-to-counter-event conversion; explicit counter events need not change'),
            fixture=full['fixtures']['sycl_host_event_gate_test'],fixture_external_headers=full['fixture_external_headers'],
            environment_reference=str(SETTINGS.relative_to(P)),environment_sha256=digest(SETTINGS),
            per_case_budgets=dict(wall_seconds=30,cpu_seconds=60,rss_bytes=1<<30,text_bytes=64<<20,total_bytes=64<<20),
            runtime_asset_delta={n:dict(qualified_sha256=v[0],candidate_sha256=v[1]) for n,v in DELTA.items()},
            complete=False,passed=False,active=False,cases=[],model_executed=False,model_quality_qualified=False,
            consumer='Restore the original v4 profiled compute queue and set only the documented original NEO implicit regular-to-counter-event conversion key to0 before native exec. Compare with the committed profiled v4 control, not solely the unprofiled v5 result. Preserve enqueue order, bounds, fallback, calls and runtime safety settings. Installed key presence does not prove effectiveness or exact source mapping; no model or replacement-gate qualification.',
            gate='One fresh copy-query-profiled process. Require actual copy/compute/probe profiling1/1/0 and all queues in-order. Arm a2s fallback before import. Import/dependent submits/native-handle materialization each within100ms. Query the resident native event at most20 times5ms apart while gate is held, with a200ms observation cap. If unready, release gate before SYCL event wait; if ready, wait with gate held. Record exact branch, wait time, gate state and post-wait marker; check payload and all owner drains. Never read the marker before checked completion. Diagnostic cannot qualify a model or57-case replacement gate even if it passes.',
            limits='Test-only runtime progress observation. No qualification of the replacement gate, production projection math, physical wire traffic, model correctness, overlap gain or throughput.')
        runtime_unchanged()
        ROOT.mkdir(mode=0o700)
        for n in ('tmp','cache'):(ROOT/n).mkdir()
        env=dict(settings['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),
            **record['diagnostic_environment_override'])
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
        cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        owner=None
        def persist():
            record['active']=bool(owner and owner.active is not None)
            (ROOT/'record.json').write_text(json.dumps(dict(record,commands=owner.commands if owner else []),indent=2)+'\n')
        owner=supervision.Owner(ROOT/'owned',persist);persist()
        try:
            for mode in (CASE,):
                entry,out,err=owner.run(mode,[record['fixture']['path'],mode],env,P,
                    wall=30,cpu=60,rss_cap=1<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
                stdout,stderr=out.read_text(errors='replace'),err.read_text(errors='replace')
                queue_rows=re.findall(r'^event_gate_queue copy_profiling=(\d+) compute_profiling=(\d+) probe_profiling=(\d+) in_order=(\d+)$',stdout,re.M)
                guards.require(queue_rows==[('1','1','0','1')],'actual profiled diagnostic queue properties differ')
                record['observed_queue_configuration']=dict(zip(('copy_profiling','compute_profiling','probe_profiling','all_in_order'),map(int,queue_rows[0])))
                starts=re.findall(r'^event_gate_call phase=begin name=([a-z-]+) index=(\d+) offset_us=(\d+)$',stdout,re.M)
                ends=re.findall(r'^event_gate_call phase=end name=([a-z-]+) index=(\d+) offset_us=(\d+) elapsed_us=(\d+) result=(\d+)$',stdout,re.M)
                windows=re.findall(r'^event_gate_window query_us=(\d+) queries=(\d+) native_ready=(\d+)$',stdout,re.M)
                if starts or ends or windows:
                    calls=[dict(name=n,index=int(i),offset_us=int(o),elapsed_us=int(e),result=int(r)) for n,i,o,e,r in ends]
                    summary={name:dict(completed=sum(c['name']==name for c in calls),
                        max_us=max(c['elapsed_us'] for c in calls if c['name']==name),
                        last_result=next(c['result'] for c in reversed(calls) if c['name']==name))
                        for name in sorted({c['name'] for c in calls})}
                    record['query_call_observation']=dict(started=len(starts),completed=len(ends),by_call=summary,
                        longest=max(calls,key=lambda c:c['elapsed_us']) if calls else None,
                        window=dict(zip(('query_us','queries','native_ready'),map(int,windows[0]))) if len(windows)==1 else None,
                        stdout_sha256=digest(out),stderr_sha256=digest(err))
                observed=re.findall(r'^event_gate_resident value=(\d+) held=(\d+) wait_us=(\d+) submit_us=(\d+)$',stdout,re.M)
                if observed:
                    guards.require(len(observed)==1,'ambiguous resident diagnostic')
                    record['resident_observation']=dict(zip(('value','held','wait_us','submit_us'),map(int,observed[0])))
                guards.require(supervision.closed(entry) and entry.get('exit_code')==0,
                    mode+' failed: '+stderr[-1024:])
                if mode==CASE:
                    query=re.findall(r'^event_gate_query native_ready=(\d+) queries=(\d+) import_us=(\d+) submit_us=(\d+) handle_us=(\d+) query_us=(\d+) wait_us=(\d+) held_before_wait=(\d+) held_after_wait=(\d+) released_before_wait=(\d+) marker=(\d+) release=(manual|fallback) payload=PASS status=PASS$',stdout,re.M)
                    guards.require(len(query)==1,'missing native query diagnostic evidence')
                    keys=('native_ready','queries','import_us','submit_us','handle_us','query_us','wait_us','held_before_wait','held_after_wait','released_before_wait','marker')
                    observation=dict(zip(keys,map(int,query[0][:-1])),release=query[0][-1])
                    guards.require(observation['native_ready'] in (0,1) and 1<=observation['queries']<=20 and
                        observation['import_us']<=100000 and observation['submit_us']<=100000 and
                        observation['handle_us']<=100000 and observation['query_us']<=200000 and observation['marker']==60 and
                        observation['held_before_wait'] in (0,1) and observation['held_after_wait'] in (0,1) and
                        observation['released_before_wait']==1-observation['native_ready'] and
                        not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed',stderr),
                        'native query diagnostic bounds/branch/marker differ')
                    record['native_query_observation']=observation
                    record['cases'].append(dict(mode=mode,passed=True,stdout_sha256=digest(out),stderr_sha256=digest(err)))
                    persist();continue
                rows=re.findall(r'^event_gate_case=(\w+) import_us=(\d+) submit_us=(\d+) pending_us=(\d+) release=manual payload=PASS status=PASS$',stdout,re.M)
                guards.require(len(rows)==1 and rows[0][0]==mode and
                    int(rows[0][1])<=100000 and int(rows[0][2])<=100000 and int(rows[0][3])>=50000 and
                    not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed',stderr),
                    'native gate result/latency/progress evidence differs')
                record['cases'].append(dict(mode=mode,import_us=int(rows[0][1]),submit_us=int(rows[0][2]),
                    pending_us=int(rows[0][3]),passed=True,stdout_sha256=digest(out),stderr_sha256=digest(err)))
                persist()
            unchanged();runtime_unchanged();record.update(complete=True,passed=True)
        except BaseException as error:record['error']=str(error)
        finally:
            record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
            after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True)
            record['fresh_faults']=[x for x in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
            record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            try:record['driver_after']=guards.driver_guard()
            except Exception as error:record.update(driver_guard_error=str(error),passed=False)
            record['passed']=bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and not record['fresh_faults'])
            persist();RESULT.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:record.get(k) for k in ('complete','passed','ownership_closed','error')}))
        return 0 if record['passed'] else 1


if __name__=='__main__':sys.exit(main())
