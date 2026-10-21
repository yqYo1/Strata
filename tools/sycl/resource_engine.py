#!/usr/bin/env python3
"""Main-owned current-engine decode resource diagnostic; no new native build."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import generation_engine as generation
import measurement_guards as guards
import measurement_owner as supervision
from build_engine import blobs

P, B, A = generation.P, generation.B, generation.A
RESULT = A / 'client-resources-current.json'
ROOT = B / 'decode-current-resources-v1'
READER = P / 'bench/sycl/client_resources.py'
sys.path.insert(0, str(READER.parent))
import client_resources


def require_journal_binding(requests, journal_rows):
    # These two result-only fields do not supply clocks, work or success gates.
    official=[{k:v for k,v in row.items() if k not in ('controller_eligible','reasoning_tokens')}
              for row in requests]
    guards.require(official==journal_rows, 'official work/client clocks differ from the verified journal')


def collection_read_spans(samples, client_interval):
    """Observed reader-call wall spans, not a causal observer-overhead bound."""
    begin,first,end=(client_interval[k] for k in
        ('request_begin_monotonic_ns','first_output_monotonic_ns','request_end_monotonic_ns'))
    result={}
    for name,lo,hi in [('before_first_output',begin,first),('after_first_output',first,end)]:
        spans=sorted((s['end_monotonic_ns']-s['begin_monotonic_ns'])/1e9 for s in samples
            if lo<=s['begin_monotonic_ns']<=s['end_monotonic_ns']<=hi)
        guards.require(spans and hi>lo,'missing contained collector read spans')
        result[name]=dict(contained_calls=len(spans),sum_wall_seconds=sum(spans),
            median_wall_seconds=statistics.median(spans),max_wall_seconds=max(spans),
            sum_fraction_of_client_window=sum(spans)/((hi-lo)/1e9))
    return result


def source_proof(record):
    receipt = record['source_build_record']
    guards.require(generation.common.digest(receipt['path']) == receipt['sha256'], 'original source receipt changed')
    full = json.loads(Path(receipt['path']).read_text())
    reference = record['fallback_reference']
    raw = generation.git('show', reference['commit'] + ':' + reference['path'])
    guards.require(hashlib.sha256(raw).hexdigest()==reference['sha256'], 'original fallback reference changed')
    retained = json.loads(raw)['retained_validation_reference']
    guards.require(full['source_commit'] == record['build_source_commit'] == retained['source_commit'] and
        all(full['binary'][k] == record['binary'][k] for k in ('bytes','sha256')) and
        retained['binary'] == record['binary'] and retained['source_build_record'] == receipt,
        'original build source/binary or committed relocation differs')
    count = 0
    for source in [full['project'], *full['dependencies']]:
        pins = source['source_sha256']
        recovered = {n: hashlib.sha256(data).hexdigest() for n, data in
            blobs(Path(source['root']), source['revision'], list(pins))}
        guards.require(recovered == pins, 'original native/dependency source cannot be recovered')
        count += len(pins)
    recipe = full['recipe_reference']
    guards.require(hashlib.sha256(generation.git('show', full['source_commit'] + ':' + recipe['path'])).hexdigest()
        == recipe['sha256'], 'original build recipe cannot be recovered')
    guards.require(generation.common.digest(record['binary']['path']) == record['binary']['sha256'], 'engine bytes changed')
    return count


def modules():
    # Lazy imports avoid the legacy cpu_capacity -> resource_engine import edge.
    import cpu_capacity as assets
    import packed_bank as bank
    import generation_compare as comparison
    return assets, bank, comparison


def qualification():
    assets, bank, comparison = modules()
    full, proof = bank.qualification()
    closed = generation.committed(comparison.RESULT)
    guards.require(closed['complete'] and closed['passed'] and closed['ownership_closed'] and
        not closed['active'] and closed['stage']=='closed' and len(closed['processes'])==2,
        'close both retained/current generation pilots before resource preparation')
    selected=closed['processes'][1]
    guards.require(selected['arm']==1 and selected['passed'] and selected['complete'] and
        selected['ownership_closed'] and not selected['active'] and selected['native_exit_code']==0 and
        selected['fault_probe_successful'] and selected['fresh_faults']==[] and selected['boot_unchanged'] and
        selected['build_source_commit']==full['source_commit'] and selected['binary_sha256']==full['binary']['sha256'] and
        all(r['passed'] and r['complete'] and r['ownership_closed'] and not r['active'] and
            r['native_exit_code']==0 and r['fault_probe_successful'] and r['fresh_faults']==[] and r['boot_unchanged']
            for r in closed['processes']) and closed['plans'][1]['binary']==full['binary'] and len(selected['requests'])==len(selected['routes'])==2 and
        all(generation.sample_eligible(r) for r in selected['requests']) and
        [r['workload'] for r in selected['requests']]==['regular-code','varied-prose'] and
        selected['requests'][0]['acceptance_rate']>=.60 and
        all(r['actual_MoE_mode']==10 and r['restored_dense_layers']==48 and r['cache_payload_and_map_match'] and
            r['private_bank']['actual_private_layers']==48 for r in selected['routes']),
        'same-build current generation/route/closure proof required')
    return full, dict(native=proof, generation_reference_sha256=generation.common.digest(comparison.RESULT)), closed


def benchmark_pins(revision):
    pins=generation.benchmark_pins(revision)
    name=str(READER.relative_to(P));raw=generation.git('show',revision+':'+name)
    guards.require(READER.read_bytes()==raw,'commit exact passive reader first')
    pins[name]=hashlib.sha256(raw).hexdigest()
    return pins


def prospective_plan():
    return dict(question='On the current qualified strong-MTP route, where does decode host wall time go, how much native CPU time is scheduled and how much request-contained CCS work is observed?',
        process_count=1, workload='regular-code', runs=1, collector_seconds=900, collector_interval=1,
        constants='Currente062 native/ac904 build and nine fixtures; same model/profile,131072 resident INT8 context,8192 prefill,1256 slots,5 workers,pcie0,adapt0,spec4,suffix0,caches0,greedy,no token graph/no coupled draft,Q8 MTP. Official bench.sh thinking-enabled regular-code diagnostic subset with four warmups and >=32K fresh input,>=1024 generated tokens and >=512 answer characters. Same request body as the closed current pilot.',
        observers='Existing STRATA_DECODE_TIMING=1 plus passive client_resources.py proc/fdinfo reader. Keep the qualified parameter-validation,stage-progress,minimal draft/cache-audit and all runtime safety settings. No private preload or native/codegen change. Environment fixed before exec.',
        admission='Recover exact current native/dependencies/recipe/fixtures, original closed current pilot and source/controller/model/runtime inputs. Verify actual native PID/start/SID/executable/environment and four warmups before long dispatch. Require complete official stream/journal/engine clocks,MTP work and GPU BF16/mode10/cache restoration, exact declared passive collector/source/exit, wholly contained CPU/CCS spans before/after first output, recomputed intervals and rounded host-scope/work conservation. Stop after failure or budget; never retry.',
        interpretation='Keep verify/commit/draft outer wall clocks separate from nested host/wait/pool clocks. CPU_pool_run includes scheduling/waits,not useful arithmetic alone. GPU-reach wait is not device compute time. Compare logical CPU jobs/entries/NT1 lower bound and dot equivalents with source-compatible CPU calibration only after checking source/flags/shape limits. CCS activity,scheduled CPU and requested copies do not prove arithmetic,DRAM or PCIe saturation.',
        limits='One observed strong-MTP request. No clean speed,observer-overhead,weak-MTP/agent/task quality,physical262144/session/RESTORE,adoption or70tok/s qualification. Missing hardware counters/ceilings remain unknown.',
        retention='Keep one individual request,compact before/after-first-output CPU/CCS aggregates,rounded host-clock/work checks,actual config/environment/source/native/owner/fault identities and decision. Retire normal logs,full sample timeline and JIT after replay/coverage verification; no full API trace or tensor capture.')


def current(record, full_models=False):
    assets, _, comparison=modules()
    full, proof, closed=qualification()
    guards.require(full['binary']==record['binary'] and proof==record['qualification'] and
        prospective_plan()==record['plan'],'current native/generation qualification or prospective plan changed')
    pins=benchmark_pins(record['source_commit'])
    guards.require(len(pins)==record['source_count'] and assets.fingerprint(pins)==record['benchmark_fingerprint'] and
        generation.common.digest(generation.SETTINGS)==record['settings_sha256'],'source/settings changed')
    for name,sha in record['generated_inputs'].items():
        guards.require(generation.common.digest(name)==sha,'generated diagnostic input changed: '+name)
    expected={k:v.replace(closed['plans'][1]['root'],str(ROOT)) for k,v in closed['plans'][1]['environment'].items()}
    expected['STRATA_DECODE_TIMING']='1'
    guards.require(record['environment']==expected and assets.machine_identity()==record['machine_identity'] and
        assets.model_pins(record['config'],record['model_assets'],full=full_models)==record['model_assets'],
        'immutable environment or model/platform changed')
    generation.runtime_assets_unchanged(record)
    return closed


def prepare():
    assets, _, comparison=modules()
    guards.require(not ROOT.exists(),'single-use current resource root; no retry')
    previous=generation.committed(RESULT)
    guards.require(previous['complete'] and previous['passed'] and previous['ownership_closed'] and not previous['active'] and
        previous.get('artifact_retirement',{}).get('complete') is True,
        'preserve/close original resource observation before replacement')
    full,proof,closed=qualification()
    selected=closed['plans'][1]
    source=generation.git('rev-parse','HEAD').decode().strip();pins=benchmark_pins(source)
    models=assets.model_pins(selected['config'],closed['model_assets'],full=True)
    guards.require(selected['config']['cwd']==str(P),'prepare/run only in the primary worktree after integration')
    ROOT.mkdir(mode=0o700)
    for name in ('cache','tmp'):(ROOT/name).mkdir()
    config=copy.deepcopy(selected['config'])
    config['args'][config['args'].index('--expert-profile')+1]=str(ROOT/'expert-profile.bin')
    config.update(log=str(ROOT/'engine.stderr'),env={})
    env={k:v.replace(selected['root'],str(ROOT)) for k,v in selected['environment'].items()}
    env['STRATA_DECODE_TIMING']='1'
    shutil.copyfile(P/'data/expert-profile.bin',ROOT/'expert-profile.bin')
    collection=dict(source=dict(path=str(READER),sha256=generation.common.digest(READER)),seconds=900,interval=1)
    (ROOT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (ROOT/'plan.json').write_text(json.dumps(dict(binary=full['binary'],native_validation_probe=None,
        resource_collection=collection),indent=2)+'\n')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    record=dict(mode='engine-compare',stage='prepared',source_commit=source,build_source_commit=full['source_commit'],
        binary=full['binary'],qualification=proof,native_validation_probe=None,
        previous_resource_reference=dict(commit=generation.git('log','-1','--format=%H','--',str(RESULT.relative_to(P))).decode().strip(),
            path=str(RESULT.relative_to(P)),sha256=generation.common.digest(RESULT)),
        plan=prospective_plan(),config=config,environment=env,resource_collection=collection,
        generated_inputs={str(ROOT/n):generation.common.digest(ROOT/n) for n in ('config.json','plan.json','expert-profile.bin')},
        command=['/bin/sh',str(P/'bench.sh'),'--config',str(ROOT/'config.json'),'--port',str(port),
            '--long-generation-only','--long-generation-workload','regular-code','--think','--runs','1',
            '--profile-request-gates','--workload-id',comparison.WORKLOAD_ID,'--fail-fast-engine-error',
            '--ready-timeout','600','--out',str(ROOT/'bench')],
        runtime_asset_delta=closed['runtime_asset_delta'],model_assets=models,machine_identity=assets.machine_identity(),
        source_count=len(pins),benchmark_fingerprint=assets.fingerprint(pins),settings_sha256=generation.common.digest(generation.SETTINGS),
        complete=False,passed=False,active=False,ownership_closed=True,clean_speed_qualified=False,
        hardware_saturation_qualified=False,model_adoption_qualified=False,
        budgets=dict(wall=1800,cpu=54000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20))
    current(record)
    (ROOT/'diagnostic-plan.json').write_text(json.dumps(record,indent=2)+'\n')
    RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print('Prepared current-engine resource diagnostic; commit exact plan before run.')


def closed_evidence(record, closed, entry, boot):
    """Replay actual closed files; callers separately prove source/assets/fault gates."""
    _, _, comparison=modules()
    evidence={}
    paths=list((ROOT/'bench').glob('*/results.json'))
    guards.require(supervision.closed(entry) and entry.get('exit_code')==0 and len(paths)==1,'owned official benchmark failed')
    result=json.loads(paths[0].read_text())
    rows,journal,events=comparison.official(ROOT,result,workloads=('regular-code',))
    evidence['requests']=rows
    guards.require(result['complete'] and not result['results']['failures'] and rows[0]['request_sha256']==
        closed['processes'][1]['requests'][0]['request_sha256'],'incomplete suite or changed request body')
    evidence['warmups']=journal
    native=generation.native_end(ROOT,record,dict(config=record['config']),boot,entry['pid'])
    guards.require(record.get('native_identity')==native['native'] and record.get('effective_environment_verified') is True,
        'native identity differs from the admitted long-request handshake')
    evidence.update(native_exit_code=native['exit_code'],resource_collector_exit_code=native['resource_collector_exit_code'])
    path,=(ROOT/'bench').glob('*/requests.jsonl');events=[json.loads(x) for x in path.read_text().splitlines()]
    text=(ROOT/'engine.stderr').read_text(errors='replace')
    evidence['route']=comparison.native_routes(text,events,1,workloads=('regular-code',))[0]
    all_rows=[e['sample'] for e in events if e['kind']=='result']
    evidence['decode_timing']=generation.decode_timing_evidence(text,all_rows)[-1]
    capture=ROOT/'client-resources.jsonl';samples=[json.loads(x) for x in capture.read_text().splitlines()]
    evidence['resources']=client_resources.summarize_capture(samples,native['native'],boot,rows[0]['client_interval'])
    evidence['collector_read_spans']=collection_read_spans(samples[1:-1],rows[0]['client_interval'])
    evidence['evidence_sha256']={p.name:generation.common.digest(p) for p in
        (paths[0],path,capture,ROOT/'engine-start.json',ROOT/'engine-exit.json',ROOT/'engine.stderr')}
    return evidence


def run():
    import select
    import time
    assets, _, comparison=modules()
    record=generation.committed(RESULT);closed=current(record)
    guards.require(record==json.loads((ROOT/'diagnostic-plan.json').read_text()) and record['stage']=='prepared' and
        not record['active'] and not (ROOT/'record.json').exists() and not (ROOT/'launch.json').exists(),
        'fresh exact diagnostic only; inspect an existing owner instead of retrying')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in before.splitlines() if s.startswith('-- cursor: '))
    owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        (ROOT/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    def interact(io,leader):
        for stage in ('begin','end'):
            while True:
                line=io.line(1500)
                if line.startswith('STRATA_PROFILE_GATE '):break
            guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==
                dict(stage=stage,phase='long-generation/regular-code',run=0),'long request gate differs')
            if stage=='begin':
                native=json.loads((ROOT/'engine-start.json').read_text())['native']
                identity=supervision.identity(native['pid'])
                guards.require(identity and identity['start_ticks']==native['start_ticks'] and identity['sid']==leader['pid'] and
                    Path('/proc',str(native['pid']),'exe').resolve()==Path(record['binary']['path']).resolve(), 'native ownership/executable differs')
                actual=dict(p.split(b'=',1) for p in Path('/proc',str(native['pid']),'environ').read_bytes().split(b'\0') if b'=' in p)
                guards.require(all(actual.get(k.encode())==v.encode() for k,v in record['environment'].items()) and
                    b'LD_PRELOAD' not in actual,'effective native environment differs')
                path,=(ROOT/'bench').glob('*/requests.jsonl')
                events=[json.loads(x) for x in path.read_text().splitlines()]
                guards.require(len(events)==10 and events[-1]['kind']=='start' and events[-1]['phase']=='long-generation/regular-code',
                    'warmup/long journal extent differs')
                generation.warmup_reconciliation(events[:-1],workloads=())
                text=(ROOT/'engine.stderr').read_text(errors='replace')
                generation.request_evidence(text,[e['sample'] for e in events[:-1] if e['kind']=='result'])
                assets.verify_pool(text,5)
                record.update(native_identity=native,effective_environment_verified=True)
                persist()
            io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();guards.require(time.monotonic()<deadline,'benchmark teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    owner=supervision.Owner(ROOT/'owned',persist)
    try:
        entry,_,_=owner.run('bench',record['command'],record['environment'],P,interaction=interact,**record['budgets'])
        record.update(closed_evidence(record,closed,entry,boot))
        current(record,full_models=True)
        record.update(passed=True,complete=True)
    except BaseException as error:
        record['error']=str(error)
        if (ROOT/'engine.stderr').exists():record['error_excerpt']=(ROOT/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        record.update(ownership_closed=owner.active is None and all(supervision.closed(e) for e in owner.commands),
            fresh_faults=None,fault_probe_successful=False,boot_unchanged=False)
        try:
            after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            record.update(fresh_faults=[s for s in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)],
                fault_probe_successful=True,boot_unchanged=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            guards.driver_guard()
        except Exception as error:record['fault_probe_error']=str(error)
        record['passed']=bool(record['passed'] and record['ownership_closed'] and record['fault_probe_successful'] and
            record['fresh_faults']==[] and record['boot_unchanged'] and 'fault_probe_error' not in record)
        record['stage']='closed';persist();RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ('passed','complete','ownership_closed','error')}))
    return 0 if record['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare() if args.action=='prepare' else run()


if __name__=='__main__':sys.exit(main())
