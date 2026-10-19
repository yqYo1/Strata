#!/usr/bin/env python3
"""Main-owned passive resource diagnostic on the retained old speed reference."""
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
ROOT = B / 'decode-client-resources-v2'
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


def prepare():
    previous = generation.committed(RESULT)
    guards.require(previous.get('mode') == 'client-resources' and
        previous.get('diagnostic_version', 1) == 1 and previous.get('complete') is True and
        previous.get('passed') is True and previous.get('ownership_closed') is True and
        not previous.get('active') and not previous.get('fresh_faults') and
        previous.get('resource_admission_qualified') is True and
        previous.get('artifact_retirement', {}).get('complete') is True,
        'collector-only diagnostic requires the closed admitted first capture; no automatic retry')
    previous_raw = RESULT.read_bytes()
    previous_commit = generation.git('log','-1','--format=%H','--',str(RESULT.relative_to(P))).decode().strip()
    closed = generation.committed(generation.RESULT)
    summary = closed['engine_comparison_summary']
    guards.require(closed['complete'] and closed['passed'] and closed['ownership_closed'] and
        len(summary['processes']) == 6 and summary['diagnostic_nonregression_gate_passed'] is False,
        'close the six-process comparison and preserve its failed gate first')
    selected = [p for p in summary['processes'] if p['arm'] == 'old'][-1]
    revision, path = selected['receipt_commit'], str(generation.RESULT.relative_to(P))
    raw = generation.git('show', revision + ':' + path)
    old = json.loads(raw)
    guards.require(old['complete'] and old['passed'] and old['ownership_closed'] and not old['fresh_faults'] and
        old['engine_arm'] == 'old' and old['observer_records_absent'], 'invalid selected old-engine qualification')
    receipt_path = B / 'known-good-396d0c91f301/source-build.json'
    record = dict(schema=1, mode='client-resources', diagnostic_version=2,
        source_commit=generation.git('rev-parse', 'HEAD').decode().strip(),
        previous_capture=dict(commit=previous_commit,path=str(RESULT.relative_to(P)),
            sha256=hashlib.sha256(previous_raw).hexdigest()),
        reference=dict(commit=revision, path=path, sha256=hashlib.sha256(raw).hexdigest()),
        binary=old['binary'], build_source_commit=old['build_source_commit'],
        source_build_record=dict(path=str(receipt_path), sha256=old['engine_build_receipts'][str(receipt_path)]),
        fallback_reference=dict(commit=revision,path=str(generation.common.BUILD.relative_to(P)),
            sha256=old['engine_build_receipts'][str(generation.common.BUILD)]),
        runtime_asset_delta=old['runtime_asset_delta'], complete=False, passed=False, active=False,
        clean_speed_qualified=False, hardware_bottleneck_qualified=False, model_quality_qualified=False,
        purpose='Confirm the first admitted resource observation with only the external collector: remove parameter validation/stage progress, use the same selected old engine and official workload. Scheduled CPU and CCS cycles do not isolate useful arithmetic or physical traffic.',
        prospective_gates='One diagnostic process, no automatic retry. Normal official regular-code request after four successful warmups, thinking enabled, at least32768 fresh inputs and1024 generated tokens, answer and engine clocks. Exact source/recipe/runtime/binary binding, immutable collector-only environment preserving qualified safety settings, journal-bound work/client clocks, reader exit0 and explicit terminal, recomputed eligible CCS intervals inside observed request windows. Preserve incomplete/no-data/fault outcomes. No clean-speed/observer-overhead, engine-choice, physical-traffic, saturation or70tok/s qualification. Review the original and new samples separately; elapsed-time differences are not a matched overhead estimate.',
        budgets=dict(wall=1800,cpu=54000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20))
    record['original_native_inputs_recovered'] = source_proof(record)
    generation.runtime_assets_unchanged(record)
    guards.require(not ROOT.exists(), 'never overwrite or retry an existing diagnostic')
    ROOT.mkdir(mode=0o700)
    for name in ('cache', 'tmp'): (ROOT/name).mkdir()
    original_root = old['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(original_root,str(ROOT)) for k,v in old['environment'].items()}
    settings = generation.committed(generation.SETTINGS)
    for key in ('UR_ENABLE_LAYERS','UR_LOG_TRACING','STRATA_TRACE',
                'STRATA_DECODE_CPU_PROFILE','STRATA_DECODE_TIMING'):
        guards.require(key not in env,'collector-only environment contains an internal observer: '+key)
    for key in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission',
                'STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[key] == settings['environment'][key], 'qualified safety setting differs: '+key)
    config = copy.deepcopy(old['config'])
    config['args'][config['args'].index('--expert-profile')+1] = str(ROOT/'expert-profile.bin')
    config.update(log=str(ROOT/'engine.stderr'),env={})
    guards.require(config['exe'] == str(P/'tools/sycl/engine_once.py'), 'one-launch wrapper required')
    shutil.copyfile(P/'data/expert-profile.bin', ROOT/'expert-profile.bin')
    guards.require(generation.common.digest(ROOT/'expert-profile.bin') == old['profile_sha256'], 'expert profile changed')
    (ROOT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    command=['/bin/sh',str(P/'bench.sh'),'--config',str(ROOT/'config.json'),'--port',str(port),
        '--long-generation-only','--think','--runs','1','--workload-id',generation.WORKLOAD_ID,
        '--fail-fast-engine-error','--ready-timeout','600','--out',str(ROOT/'bench'),
        '--long-generation-workload','regular-code']
    pins=generation.benchmark_pins(record['source_commit'])
    pins[str(READER.relative_to(P))]=generation.common.digest(READER)
    guards.require(READER.read_bytes()==generation.git('show',record['source_commit']+':'+str(READER.relative_to(P))),
        'commit exact reader before preparation')
    record.update(environment=env, config=config, command=command,
        config_sha256=generation.common.digest(ROOT/'config.json'),profile_sha256=old['profile_sha256'],
        settings_sha256=generation.common.digest(generation.SETTINGS),source_count=len(pins),
        source_aggregate_sha256=hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest(),
        resource_collection=dict(source=dict(path=str(READER),sha256=pins[str(READER.relative_to(P))]),seconds=900,interval=1))
    (ROOT/'plan.json').write_text(json.dumps(dict(record,source_sha256=pins),indent=2)+'\n')
    RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print('Prepared '+str(RESULT.relative_to(P))+'; commit before run.')


def run():
    record=generation.committed(RESULT)
    plan=json.loads((ROOT/'plan.json').read_text());pins=plan.pop('source_sha256')
    guards.require(not record['complete'] and record == plan and len(pins)==record['source_count'] and
        hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()==record['source_aggregate_sha256'], 'fresh exact plan required')
    recovered=generation.benchmark_pins(record['source_commit'],match_worktree=False)
    recovered[str(READER.relative_to(P))]=hashlib.sha256(generation.git('show',record['source_commit']+':'+str(READER.relative_to(P)))).hexdigest()
    guards.require(recovered==pins,'original diagnostic inputs cannot be recovered')
    for name,sha in pins.items():
        guards.require(generation.common.digest(P/name)==sha,'diagnostic source changed: '+name)
    guards.require(generation.common.digest(generation.SETTINGS)==record['settings_sha256'] and
        generation.common.digest(ROOT/'config.json')==record['config_sha256'] and
        generation.common.digest(ROOT/'expert-profile.bin')==record['profile_sha256'], 'diagnostic inputs changed')
    guards.require(source_proof(record)==record['original_native_inputs_recovered'],'native source proof differs')
    generation.runtime_assets_unchanged(record)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
    cursor=next(x.split(': ',1)[1] for x in before.splitlines() if x.startswith('-- cursor: '))
    owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        (ROOT/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    owner=supervision.Owner(ROOT/'owned',persist)
    try:
        entry,_,_=owner.run('bench',record['command'],record['environment'],P,**record['budgets'])
        record.update(complete=bool(supervision.closed(entry)),exit_code=entry.get('exit_code'))
        result,=(ROOT/'bench').glob('*/results.json')
        generation_result=json.loads(result.read_text())['results']['long_generation']
        rows=generation.collect_samples(generation_result);record['requests']=rows
        generation.require_complete_generation(generation_result,rows,('regular-code',))
        journal,journal_rows=generation.journal_evidence(ROOT)
        require_journal_binding(rows,journal_rows)
        record['warmups']=generation.warmup_evidence(ROOT,journal)
        end=json.loads((ROOT/'engine-exit.json').read_text());record['native_exit']=end
        guards.require(entry['exit_code']==end['exit_code']==end['resource_collector_exit_code']==0,
            'benchmark/native/resource collector failed')
        guards.require(end['boot_id']==boot,'native receipt boot differs')
        log=(ROOT/'engine.stderr').read_text(errors='replace')
        guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:',log), 'runtime failure')
        samples=[json.loads(line) for line in (ROOT/'client-resources.jsonl').read_text().splitlines()]
        record['resources']=client_resources.summarize_capture(samples,end['native'],boot,rows[0]['client_interval'])
        record['resource_evidence_sha256']=dict(official_result=generation.common.digest(result),
            capture=generation.common.digest(ROOT/'client-resources.jsonl'))
        generation.runtime_assets_unchanged(record)
        record['resource_admission_qualified']=True
        record['passed']=True
    except BaseException as error:
        record['error']=str(error)
        if (ROOT/'engine.stderr').exists():record['error_excerpt']=(ROOT/'engine.stderr').read_text(errors='replace')[-8192:]
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


def summarize():
    record=generation.committed(RESULT)
    guards.require(not record.get('post_closure_admission'), 'preserve the original admission; a new analysis requires a distinct plan')
    guards.require(record['complete'] and record['passed'] and record['ownership_closed'] and
        not record['active'] and not record['fresh_faults'] and record['boot_unchanged'],
        'closed successful execution required for strengthened resource admission')
    path,=(ROOT/'bench').glob('*/results.json')
    suite=json.loads(path.read_text())['results']['long_generation']
    rows=generation.collect_samples(suite)
    generation.require_complete_generation(suite,rows,('regular-code',))
    journal,journal_rows=generation.journal_evidence(ROOT)
    require_journal_binding(rows,journal_rows)
    guards.require(rows==record['requests'] and generation.warmup_evidence(ROOT,journal)==record['warmups'],
        'official samples or warmup evidence differ from the original execution receipt')
    native=json.loads((ROOT/'engine-exit.json').read_text())
    guards.require(native==record['native_exit'],'native closure receipt changed')
    capture=ROOT/'client-resources.jsonl'
    captured=[json.loads(line) for line in capture.read_text().splitlines()]
    resources=client_resources.summarize_capture(captured,
        native['native'],native['boot_id'],rows[0]['client_interval'])
    guards.require(resources==record['resources'],'resource replay differs from original admitted values')
    record['collector_read_spans']=dict(windows=collection_read_spans(
        captured[1:-1],
        rows[0]['client_interval']),limits='Inclusive synchronous reader-call wall time, including descheduling. Does not bound asynchronous driver/cache/clock effects or total model slowdown; only a prospectively matched collector-off/on comparison can measure that effect.')
    record['post_closure_admission']=dict(analyzer_source_commit=generation.git('rev-parse','HEAD').decode().strip(),
        original_execution_receipt_commit=generation.git('log','-1','--format=%H','--',str(RESULT.relative_to(P))).decode().strip(),
        official_result_sha256=generation.common.digest(path),capture_sha256=generation.common.digest(capture),
        journal_bound_clocks_and_work=True,original_resources_reproduced=True,
        original_execution_outcome_unchanged=True,GPU_executed=False)
    record['resource_admission_qualified']=True
    RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record['post_closure_admission']))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run','summarize'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if args.action=='summarize':return summarize()
        guards.driver_guard()
        return prepare() if args.action=='prepare' else run()


if __name__=='__main__':sys.exit(main())
