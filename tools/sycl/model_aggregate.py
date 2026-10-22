#!/usr/bin/env python3
"""Main-owned whole-process kernel service diagnostic on the unchanged engine."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import select
import shutil
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import resource_engine as resource
import generation_engine as gen
import generation_compare as comparison
import cpu_capacity as assets
import measurement_guards as guards
import measurement_owner as supervision
import unitrace_probe as probe
import production_profile as production
from profile_aggregate import device_summary

P, B, RESULT = probe.P, probe.B, probe.RESULT
ROOT = B/'unitrace-model-aggregate-v1'
KEY = 'native_model_aggregate'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def profile_assets(container):
    qualified = container['unitrace_graph_aggregate']
    guards.require(qualified['passed'] and qualified['complete'] and qualified['ownership_closed']
        and not qualified['active'] and qualified['fresh_faults'] == [] and qualified['boot_unchanged']
        and qualified['service_coverage_qualified']
        and [r['coverage']['program_calls'] for r in qualified['observations'][1:]] == [4,10],
        'closed initial-construction aggregate qualification required')
    archive = qualified['plan_archive']
    raw = gen.git('show', archive['commit']+':'+archive['path'])
    guards.require(hashlib.sha256(raw).hexdigest() == archive['sha256'], 'original collector plan changed')
    pinned = json.loads(raw)['unitrace_graph_aggregate']['assets']
    guards.require(probe.pin_assets(container) == pinned, 'collector/runtime/native assets changed')
    return pinned


def environment(setup):
    return resource.modules()[1].environment(setup, ROOT, '0')


def prospective_plan():
    return dict(question='Which verbose kernel shapes account for device service on the unchanged original6e40 route?',
        execution='One owned official bench.sh process: four warmups and the existing thinking-enabled regular-code request, >=32K fresh input and >=1024 generated tokens with a substantive answer. Original binary,1256-slot layout,8192 prefill,131072 resident INT8 context,5 workers,MTP4 and arithmetic settings unchanged.',
        collection='Qualified target-only validation preload and unchanged PTI collector; device-timing+verbose from exec. Unique session,Sysman off,no counters,no timeline,no pause during graph capture or replay. All environment fixed before native exec.',
        gates='Commit source/generated plan before launch; recover9853 native inputs and13 artifacts,controller closure,model/runtime and collector qualification. Verify actual native PID/start/SID/exe/environment and four completed warmups before dispatch. Require complete official stream,journal,answer,>=.60 draft acceptance,actual phase routes,all loan/refill and48-layer restoration,dispatch/work/clock conservation,validation constructor order,one complete positive device table with exact sum,normal native/owner/session closure,unchanged15 full model hashes and no fresh fault. First failure stops; no retry.',
        separation='No inference implementation,configuration,expert placement or phase selector changes. This diagnostic does not qualify a prefill-only change or decode preservation; repeated clean phase-preserving comparisons remain required.',
        interpretation='Whole-process summed service includes loading,warmups,prefill and decode. Verbose shapes can support source mapping but do not isolate phase time,overlap,critical-path wall,physical bus/memory traffic,hardware saturation or observer overhead. Unmapped/mixed shapes remain unknown.',
        limits='One observed strong-MTP process only. No clean speed,weak-MTP/agent/task quality,full physical262144/session/RESTORE,adoption or70tok/s qualification.',
        retention='Keep compact individual request,conservation checks,top shape service plus source mappings/unknowns and exact identity/decision. Keep the aggregate table and normal logs only through source attribution/replay; retire after that consumer closes. No full API trace or kernel timeline.')


def current(record, full_models=False):
    full, proof, setup = resource.qualification()
    guards.require(record['binary'] == full['binary'] and record['qualification'] == proof
        and record['plan'] == prospective_plan(), 'native qualification or plan changed')
    pins = gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins) == record['source_count'] and assets.fingerprint(pins) == record['source_fingerprint'],
        'committed controller/benchmark input changed')
    for path, sha in record['generated_inputs'].items():
        guards.require(gen.common.digest(path) == sha, 'generated input changed: '+path)
    guards.require(profile_assets(json.loads(RESULT.read_text())) == record['profile_assets']
        and record['environment'] == environment(setup)
        and record['config'] == resource.modules()[1].recipe.config(setup, ROOT)
        and assets.machine_identity() == record['machine_identity']
        and gen.common.digest(gen.SETTINGS) == record['settings_sha256']
        and assets.model_pins(record['config'], record['model_assets'], full=full_models) == record['model_assets'],
        'profile/config/environment/model/platform changed')
    gen.runtime_assets_unchanged(record)
    return setup


def prepare(container):
    guards.require(KEY not in container and not ROOT.exists(), 'fresh single-use plan only; inspect prior owners')
    full, proof, setup = resource.qualification()
    pinned = profile_assets(container)
    revision = gen.git('rev-parse','HEAD').decode().strip()
    pins = gen.benchmark_pins(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('cache','tmp','trace'):
        (ROOT/name).mkdir()
    shutil.copyfile(P/'data/expert-profile.bin',ROOT/'expert-profile.bin')
    guards.require(gen.common.digest(ROOT/'expert-profile.bin') == setup['expert_profile_sha256'], 'profile changed')
    config = resource.modules()[1].recipe.config(setup, ROOT)
    env = environment(setup)
    profile = dict(mode='aggregate',launcher=pinned['launcher'],collector=pinned['collector'],
        session='StrataModelAggregate'+secrets.token_hex(12),
        environment=dict(UNITRACE_SystemTime='1',UNITRACE_FollowChildProcess='1',
            UNITRACE_TraceOutputDir=str(ROOT/'trace'),LD_DEBUG='libs'))
    write(ROOT/'config.json',config)
    write(ROOT/'plan.json',dict(binary=full['binary'],native_validation_probe=pinned['preload'],native_profile=profile))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    record = dict(mode='engine-compare',stage='prepared',source_commit=revision,
        build_source_commit=full['source_commit'],binary=full['binary'],qualification=proof,
        profile_assets=pinned,native_profile=profile,native_validation_probe=pinned['preload'],
        source_count=len(pins),source_fingerprint=assets.fingerprint(pins),plan=prospective_plan(),
        config=config,environment=env,model_assets=assets.model_pins(config,setup['model_assets'],full=True),
        runtime_asset_delta=setup['runtime_asset_delta'],machine_identity=assets.machine_identity(),
        settings_sha256=gen.common.digest(gen.SETTINGS),
        generated_inputs={str(ROOT/name):gen.common.digest(ROOT/name) for name in ('config.json','plan.json','expert-profile.bin')},
        command=['/bin/sh',str(P/'bench.sh'),'--config',str(ROOT/'config.json'),'--port',str(port),
            '--long-generation-only','--long-generation-workload','regular-code','--think','--runs','1',
            '--profile-request-gates','--workload-id',comparison.WORKLOAD_ID,'--fail-fast-engine-error',
            '--ready-timeout','600','--out',str(ROOT/'bench')],
        budgets=dict(wall=2400,cpu=72000,rss_cap=96<<30,text_cap=16<<20,file_cap=64<<20,total_cap=128<<20),
        complete=False,passed=False,active=False,ownership_closed=True,clean_speed_qualified=False,
        hardware_saturation_qualified=False,model_adoption_qualified=False)
    current(record)
    write(ROOT/'diagnostic-plan.json',record)
    container[KEY] = record;write(RESULT,container)
    print('Prepared one whole-process aggregate diagnostic; commit exact plan before run.',flush=True)


def native_identity(record, leader, boot, closed=False):
    start=json.loads((ROOT/'engine-start.json').read_text())
    guards.require(set(start) == {'wrapper','native','binary','args','boot_id','native_validation_probe','native_profile'}
        and start['binary'] == record['binary'] and start['args'] == ['--serve',*record['config']['args']]
        and start['boot_id'] == boot and start['native_profile'] == record['native_profile']
        and start['native_validation_probe'] == record['native_validation_probe'], 'native launch claim differs')
    for name in ('native','wrapper'):
        row=start[name]
        guards.require(set(row) == {'pid','start_ticks','sid'} and all(type(x) is int and x>0 for x in row.values())
            and row['sid'] == leader, 'native session identity differs')
    guards.require(start['native']['pid'] != start['wrapper']['pid'], 'native/wrapper identity collision')
    if closed:
        end=json.loads((ROOT/'engine-exit.json').read_text())
        guards.require(end == dict(start,exit_code=0), 'normal native exit required')
        return end
    native=start['native'];actual=supervision.identity(native['pid'])
    guards.require(actual and actual['start_ticks'] == native['start_ticks'] and actual['sid'] == leader
        and Path('/proc',str(native['pid']),'exe').resolve() == Path(record['binary']['path']).resolve(),
        'actual native identity differs')
    env=dict(p.split(b'=',1) for p in Path('/proc',str(native['pid']),'environ').read_bytes().split(b'\0') if b'=' in p)
    expected=dict(record['environment'],**record['native_profile']['environment'])
    guards.require(all(env.get(k.encode()) == v.encode() for k,v in expected.items()), 'actual native environment differs')
    preloads=[Path(p.decode()).resolve() for p in env.get(b'LD_PRELOAD',b'').split(b':') if p]
    guards.require(preloads == [Path(record['profile_assets']['collector']['path']).resolve(),
        Path(record['native_validation_probe']['library']['path']).resolve()], 'actual target preload/order differs')
    return start


def closed_evidence(record,setup,entry,boot):
    guards.require(supervision.closed(entry) and entry['exit_code']==0, 'official benchmark failed')
    path,=(ROOT/'bench').glob('*/results.json')
    official=json.loads(path.read_text())
    rows,warmups,events=comparison.official(ROOT,official,workloads=('regular-code',))
    guards.require(official['complete'] and not official['results']['failures']
        and rows[0]['acceptance_rate']>=.60 and rows[0]['request_sha256']==setup['request_sha256'],
        'official request completion/regime differs')
    resource.require_journal_binding(rows,gen.journal_evidence(ROOT)[1])
    end=native_identity(record,entry['pid'],boot,closed=True)
    guards.require(record.get('admitted_native') == end['native'], 'native exit identity differs from admission')
    text=(ROOT/'engine.stderr').read_text(errors='replace')
    validation=probe.validation.preload_evidence(text,end['native']['pid'])
    positions=[]
    for phase,artifact in [('init',record['native_validation_probe']['library']),
                           ('init',record['profile_assets']['collector']),('fini',record['profile_assets']['collector'])]:
        matches=[m.start() for m in re.finditer(r'calling '+phase+r':\s+([^\s]+)',text)
            if Path(m[1]).resolve()==Path(artifact['path']).resolve()]
        guards.require(len(matches)==1,'missing/ambiguous loader phase');positions.extend(matches)
    guards.require(positions[0]<text.index('phase=before-main')<positions[1]<positions[2], 'collector/validation lifetime differs')
    guards.require(not list((ROOT/'trace').rglob('*.json')), 'aggregate unexpectedly emitted a timeline')
    lookup=[line for line in text.splitlines() if 'symbol lookup error' in line]
    guards.require(all(re.search(r'undefined symbol: (zelLoaderTranslateHandle|zelSetDriverTeardown) \(fatal\)$',line)
        for line in lookup), 'new loader failure mechanism')
    table=device_summary(text)
    all_rows=[e['sample'] for e in events if e['kind']=='result']
    paired,work,timing=resource.modules()[1].recipe.work_evidence(text,all_rows,record['config'],1,ROOT)
    route=comparison.native_routes(text,events,1,workloads=('regular-code',))[0]
    return dict(requests=rows,warmups=warmups,native_exit_code=0,validation=validation,
        loader_order_verified=True,table=table,route=route,decode_timing=timing[-1],
        iq2_dispatch=paired[-1],census=resource.modules()[1].recipe.compact_work(work)[-1],
        no_timeline=True,evidence_sha256={str(p.relative_to(ROOT)):gen.common.digest(p)
            for p in (path,ROOT/'engine.stderr',ROOT/'engine-start.json',ROOT/'engine-exit.json')})


def run(container):
    record=container[KEY]
    guards.require(record['stage']=='prepared' and record==json.loads((ROOT/'diagnostic-plan.json').read_text())
        and not (ROOT/'record.json').exists() and not (ROOT/'launch.json').exists(), 'inspect existing job; no retry')
    setup=current(record)
    session=Path('/dev/shm/uctrl'+str(os.getuid())+'_'+record['native_profile']['session'])
    guards.require(not session.exists(), 'session already exists')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in before.splitlines() if s.startswith('-- cursor: '))
    owner=None;entry=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(record,owned_commands=owner.commands if owner else []))
    def interact(io,leader):
        for stage in ('begin','end'):
            while True:
                line=io.line(2100)
                if line.startswith('STRATA_PROFILE_GATE '):break
            guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==
                dict(stage=stage,phase='long-generation/regular-code',run=0), 'request gate differs')
            if stage=='begin':
                native=native_identity(record,leader['pid'],boot)
                journal,=(ROOT/'bench').glob('*/requests.jsonl')
                events=[json.loads(line) for line in journal.read_text().splitlines()]
                guards.require(len(events)==10 and events[-1]['kind']=='start', 'four warmups/long start required')
                gen.warmup_reconciliation(events[:-1],workloads=())
                text=(ROOT/'engine.stderr').read_text(errors='replace')
                gen.request_evidence(text,[e['sample'] for e in events[:-1] if e['kind']=='result'])
                assets.verify_pool(text,5)
                record['admitted_native']=native['native'];persist()
                print('Four warmups and exact unchanged native admitted; collecting official32K+ request.',flush=True)
            io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+240
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();guards.require(time.monotonic()<deadline,'benchmark teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    owner=supervision.Owner(ROOT/'owned',persist)
    record.update(stage='running',driver_before=guards.driver_guard());persist()
    try:
        entry,_,_=owner.run('bench',record['command'],record['environment'],P,interaction=interact,**record['budgets'])
        record.update(closed_evidence(record,setup,entry,boot))
        current(record,full_models=True)
        record.update(passed=True,complete=True)
    except BaseException as error:
        record['error']=repr(error)
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        if record['ownership_closed'] and entry and supervision.closed(entry) and session.exists():
            try:
                stop,_,_=owner.run('stop',[record['native_profile']['launcher']['path'],'--stop',record['native_profile']['session']],
                    record['environment'],ROOT,wall=15,cpu=15,rss_cap=2<<30,text_cap=1<<20,file_cap=64<<20,total_cap=128<<20)
                record['session_stop_closed']=supervision.closed(stop) and stop['exit_code']==0
            except Exception as error:
                record['session_stop_error']=repr(error)
        record['session_removed']=not session.exists()
        try:record.update(production.fault_evidence(cursor,boot))
        except Exception as error:record['fault_probe_error']=repr(error)
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        record['passed']=bool(record['passed'] and record['ownership_closed'] and record['session_removed']
            and record.get('fresh_faults')==[] and record.get('boot_unchanged')
            and not any(k in record for k in ('fault_probe_error','session_stop_error')))
        record['stage']='closed';persist();container[KEY]=record;write(RESULT,container)
    print(json.dumps({k:record.get(k) for k in ('passed','complete','ownership_closed','session_removed','error')}),flush=True)
    return 0 if record['passed'] else 1


def main(action):
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guards.no_model_owners([]);guards.driver_guard()
        container=gen.committed(RESULT)
        return prepare(container) if action=='prepare' else run(container)


if __name__=='__main__':
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'use model_aggregate.py prepare|run')
    sys.exit(main(sys.argv[1]))
