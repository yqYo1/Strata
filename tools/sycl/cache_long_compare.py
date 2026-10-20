#!/usr/bin/env python3
"""Retained-engine117K comparison decisions; execution follows long admission.

The two arms differ in engine revision and scratch/cache placement. This is a
route comparison, not an isolated kernel experiment or sustained decode test.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import sys

sys.dont_write_bytecode=True
import cache_long as admission
import resource_engine as resources

from measurement_guards import require

q, gen, assets, guards, supervision = admission.q, admission.gen, admission.assets, admission.guards, admission.supervision
P, B, A = admission.P, admission.B, admission.A
RESULT = A/'cache-long-comparison-current.json'
SCOPE = 'retained_117K_clean_comparison_scope'
ORDER = (0, 1, 1, 0, 0, 1)
PAIRS = ((0, 1), (3, 2), (4, 5))


def comparison_plan():
    return dict(order=list(ORDER), pairs=[list(p) for p in PAIRS],
        arms=['retained396-owned128', 'current-borrowed1256'],
        requests_per_process=4, actual_fresh_input_tokens=117489, output_cap=24,
        thinking=True, workload_id='native_gdn_20261018',
        classification='Route-level prefill comparison on the current runtime; engine and placement differ.',
        prerequisites='Close1256-slot117K all-GPU/two-bank admission, then a separately checked old-owned117K request. Verify both retained source/recipe/binary/fixture sets before every process. Freeze full benchmark/config/runtime/model inputs before collection.',
        decision='Compare first-use separately. For each pair use the ratio of subsequent-request median throughput (three fresh requests/process). Faster screen requires all three ratios>1 and geomean>1.02; slower screen requires all<1 and geomean<1/1.02. Otherwise inconclusive. Nonregression requires geomean>=0.98 and every pair>=0.95. The1000 milestone requires each of three candidate process medians>=1000tok/s.',
        stopping='Stop on first process, input, route, fault, ownership, budget or sample-eligibility failure. Never retry a root or replace failed samples. No speed-based early stopping.',
        limits='Two old-first pairs and one new-first pair represent both chronology directions without exact balance. Four requests share a process; the three process pairs are the repetition units.24-output checks do not qualify decode, agent-session, quality or physical262144/session/RESTORE. Neither screen adopts a model route.')


def summarize(processes):
    require(len(processes) == 6 and [r['cohort'] for r in processes] == list(range(6)) and
        [r['arm'] for r in processes] == list(ORDER), 'complete frozen six-process order required')
    require(len({(r['native_identity']['pid'],r['native_identity']['start_ticks']) for r in processes}) == 6,
        'independent native processes required')
    for key in ('benchmark_fingerprint', 'machine_identity', 'model_assets_fingerprint'):
        require(all(r[key] == processes[0][key] for r in processes), 'cross-process input changed: '+key)
    for arm in (0,1):
        group=[r for r in processes if r['arm'] == arm]
        require(all(r['binary_sha256'] == group[0]['binary_sha256'] and
            r['build_source_commit'] == group[0]['build_source_commit'] for r in group), 'arm source/binary changed')
    medians=[]
    first=[]
    for process in processes:
        require(process['passed'] and process['ownership_closed'] and not process['fresh_faults'] and
            process['observer_records_absent'] and process['actual_route_checked'] and len(process['requests']) == 4,
            'failed, observed, unqualified or partial process')
        speeds=[]
        for run,sample in enumerate(process['requests']):
            require(sample['run'] == run and all(sample.get(k) is True for k in
                ('ok','eligible','stream_complete','thinking_requested','reasoning_observed')) and
                sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == 117489 and
                sample['actual_reused_tokens'] == 0 and 1 <= sample['generated_tokens'] <= 24 and
                sample['prompt_tps_source'] == 'engine', 'ineligible or changed request')
            require(type(sample['prompt_ms']) in (int,float) and math.isfinite(sample['prompt_ms']) and
                sample['prompt_ms'] > 0, 'invalid prompt clock')
            speeds.append(117489000/sample['prompt_ms'])
        first.append(speeds[0]); medians.append(statistics.median(speeds[1:]))
    for baseline,candidate in PAIRS:
        require(all(processes[baseline]['requests'][i]['request_sha256'] ==
            processes[candidate]['requests'][i]['request_sha256'] for i in range(4)), 'paired input body changed')
    ratios=[medians[b]/medians[a] for a,b in PAIRS]
    geo=math.exp(statistics.mean(math.log(r) for r in ratios))
    decision=('candidate_faster_screen' if all(r>1 for r in ratios) and geo>1.02 else
        'candidate_slower_screen' if all(r<1 for r in ratios) and geo<1/1.02 else 'inconclusive_variability_or_small_effect')
    return dict(prefill_decision=decision,subsequent_pair_ratios=ratios,subsequent_ratio_geomean=geo,
        first_use_tps=first,subsequent_median_tps=medians,
        prefill_nonregression=geo>=0.98 and all(r>=0.95 for r in ratios),
        candidate1000tok_s_all_process_medians=all(medians[i]>=1000 for i,r in enumerate(processes) if r['arm']==1),
        adoption_qualified=False,sustained_decode_qualified=False,limits=comparison_plan()['limits'])


def retained_builds():
    current,count=q.current_build()
    old=gen.committed(assets.RESULT)
    old_count=resources.source_proof(old)
    ref=old['fallback_reference']
    retained=json.loads(gen.git('show',ref['commit']+':'+ref['path']))['retained_validation_reference']
    require(len(retained['fixtures']) == 5, 'old same-build fixtures missing')
    for item in retained['fixtures'].values():
        require(gen.common.digest(item['path']) == item['sha256'], 'old fixture bytes changed')
    return old,current,dict(old_original_inputs_recovered=old_count,current_original_inputs_recovered=count,
        old_fixtures=retained['fixtures'],current_fixtures=current['fixtures'],
        current_receipt_sha256=gen.common.digest(q.BUILD_RECORD),
        current_component_sha256=gen.common.digest(q.COMPONENT),
        old_receipt=old['source_build_record'],old_fallback_reference=old['fallback_reference'])


def qualification():
    result=gen.committed(admission.RESULT)
    require(result['complete'] and result['passed'] and result['ownership_closed'] and not result['active'] and
        not result['fresh_faults'] and result['full_model_hashes_before_and_after_match'] and
        result['long_capacity_evidence']['checked_retirements'] == 96 and
        result['long_capacity_evidence']['slots'] == 1256, 'close1256-slot117K admission first')
    if 'original_plan' in result:
        ref=result['original_plan']; raw=gen.git('show',ref['commit']+':'+ref['path'])
        require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'original long admission plan changed')
        original=json.loads(raw)
    else:
        original=result
    old,current,proof=retained_builds()
    require(original['binary'] == current['binary'] and
        proof['current_receipt_sha256'] == original['source_build_record']['sha256'] and
        proof['current_component_sha256'] == original['component_result_sha256'], 'admitted current build changed')
    gen.runtime_assets_unchanged(dict(mode='engine-compare',runtime_asset_delta=original['runtime_asset_delta']))
    return old,current,proof,original


def admitted_loan():
    result=gen.committed(admission.RESULT)
    proof=result['long_capacity_evidence']; loan=proof['long_loan']
    require(proof['slots'] == 1256 and proof['exact_payload_and_map_cycles'] == 2 and
        all(type(loan[k]) is int for k in ('first','lent','chunk','capacity','need')) and
        loan['chunk'] == 8192 and 128 <= loan['first'] < 1256 and
        loan['first']+loan['lent'] == 1256 and loan['capacity'] >= loan['need'] > 0,
        'exact occupied-suffix admission required')
    return {k:loan[k] for k in ('first','lent','chunk','capacity','need')}


def cache_placement(log,arm,loan):
    slots=1256 if arm else 128
    admission.cap.matches(r'strata generate: expert cache '+str(slots)+r' slots, [^\n]+',log,1)
    counts=re.findall(r'^strata serve: the prompt path borrows (\d+) CUDA0 cache slots [^\n]+$',log,re.M)
    owned='strata serve: the prompt path allocates its own buffers (too few cache slots to borrow)' in log
    if arm:
        require(loan['first']+loan['lent'] == slots and len(counts) == 1 and
            int(counts[0]) == loan['lent'] and slots-int(counts[0]) == loan['first'] and not owned,
            'actual borrowed suffix differs from exact admitted first/lent')
    else:
        require(not counts and owned,'actual owned scratch placement differs')
    return dict(expert_cache_slots=slots,loan_first=loan['first'] if arm else None,
        loan_slots=loan['lent'] if arm else 0,
        suffix_identity_basis='Exact completed payload/map admission, unchanged profile and static per-layer cache; first=slots-lent.')


def reviewed():
    row=gen.committed(admission.cap.borrowed.RESEARCH)[SCOPE]
    require(row.get('worker_closed') is True and row.get('clean_comparison_accepted') is True,
        'curated retained-engine comparison review required')
    for name,expected in row['frozen_inputs'].items():
        require(gen.common.digest(P/name) == expected and
            hashlib.sha256(gen.git('show',row['source_commit']+':'+name)).hexdigest() == expected,
            'reviewed comparison source changed: '+name)
    return {k:row[k] for k in ('source_commit','frozen_inputs','report_sha256','native_review_sha256')}


def official(root,result,count):
    require(result['params']['mode'] == 'prefill-only' and result['params']['runs'] == count and
        result['params']['think'] is True and result['params']['prefill_tokens'] == 98304 and
        result['complete'] and not result['results']['failures'], 'official prefill suite incomplete')
    cohort=result['results']['prefill_only']; samples=cohort['requests']
    require(cohort['complete'] and cohort['eligible_runs'] == count == len(samples) and
        cohort['max_tokens'] == 24 and cohort['thinking_requested'] is True, 'wrong prefill-only scope')
    for i,row in enumerate(samples):
        require(row['run'] == i and all(row.get(k) is True for k in
            ('ok','eligible','stream_complete','thinking_requested','reasoning_observed')) and
            row['actual_prompt_tokens'] == row['actual_read_tokens'] == 117489 and
            row['actual_reused_tokens'] == 0 and type(row['generated_tokens']) is int and
            1 <= row['generated_tokens'] <= 24 and row['prompt_tps_source'] == 'engine' and
            type(row['prompt_ms']) in (int,float) and math.isfinite(row['prompt_ms']) and row['prompt_ms'] > 0,
            'ineligible117K sample')
    journal,_=gen.journal_evidence(root)
    path,=(root/'bench').glob('*/requests.jsonl')
    entries=[json.loads(line) for line in path.read_text().splitlines()]
    require(journal['valid'] and journal['complete'] and len(entries) == 1+2*(count+1) and
        entries[0]['kind'] == 'identity' and entries[0]['phase'] == 'suite' and
        all(r['kind'] != 'identity' for r in entries[1:]), 'incomplete/extra/misplaced journal identity')
    identity,comparison=entries[0],result['comparison_identity']
    from strata_bench import client_fingerprint
    workload=dict(workload_id='native_gdn_20261018',nonce='native_gdn_20261018',nonce_policy='fixed-id')
    require(identity['mode'] == comparison['mode'] == 'prefill-only' and identity['requested_runs'] == count and
        identity['thinking'] is comparison['thinking'] is True and
        identity['context'] == comparison['context'] == result['machine']['context'] == 131072 and
        identity['source'] == comparison['client'] == client_fingerprint() and
        identity['workload'] == comparison['workload'] == result['params']['workload'] == workload and
        identity['prefill_plan'] == comparison['prefill_plan'] and
        all(cohort[k] == value for k,value in identity['prefill_plan'].items()) and
        all(identity[k] == comparison[k] == result['machine'][k] for k in ('model','quant')), 'journal workload identity differs')
    events=entries[1:]
    hashes=[]
    for i,(phase,run) in enumerate([('warm-up/chat',None),*(('prefill-only',i) for i in range(count))]):
        start,end=events[2*i:2*i+2]
        require(start['kind'] == 'start' and end['kind'] == 'result' and
            all(r['phase'] == phase and r.get('run') == run and r.get('turn') is None for r in (start,end)) and
            start['max_tokens'] == 24 and start['thinking'] is True and end['sample']['ok'] is True and
            end['request_index'] == start['index'] and end['sample']['request_sha256'] == start['request_sha256'] and
            re.fullmatch('[0-9a-f]{64}',start['request_sha256']) is not None, 'journal request identity differs')
        if i: require(end['sample'] == {k:v for k,v in samples[i-1].items() if k != 'run'}, 'journal/suite request differs')
        hashes.append(start['request_sha256'])
    return samples,dict(successful_warmups=1,successful_long_requests=count,request_sha256=hashes,
        journal_sha256=gen.common.digest(path),state_sha256=gen.common.digest(path.with_name('journal-state.json')))


def native_evidence(log,samples,arm,check,loan):
    headers=admission.cap.matches(r'strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*',log,len(samples)+1)
    require(0<int(headers[0][1])<32768 and int(headers[0][2]) == 0 and
        int(headers[0][1]) == int(headers[0][3]) and 1<=int(headers[0][4])<=24, 'native warmup differs')
    for header,sample in zip(headers[1:],samples):
        require(tuple(map(int,header.groups())) == (117489,0,117489,sample['generated_tokens']), 'native request binding differs')
    plans=admission.cap.matches(r'strata prefill: layer-major candidate, tokens=117488 chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=2406154240 residual_gpu_rows=117488 residual_gpu=2406154240 residual_host=0 residual_staging=0 layer_cache=1022361600; extra allocations, GPU headroom=536870912, RAM headroom=8589934592',log,len(samples))
    configurations=admission.cap.matches(r'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=81920 mixed_fp16_bytes=0',log,len(samples))
    require('effective_mode=2' not in log and all(headers[i].end() < c.start() < p.start() < headers[i+1].start()
        for i,(c,p) in enumerate(zip(configurations,plans))), 'actual all-GPU route/window differs')
    placement=cache_placement(log,arm,loan)
    if check:
        require(arm == 0 and len(samples) == 1, 'only old-owned check is admitted here')
        gen.common.extract_mixed_owners(log,admission.WINDOWS,admission.TAIL_OFFSET)
        entries=admission.cap.matches(r'strata MoE mixed owners: phase=entry [^\n]+',log,96)
        done=admission.cap.matches(r'strata MoE mixed owners: phase=retired [^\n]+',log,96)
        previous=headers[0].end()
        for a,b in zip(entries,done):
            require(previous<a.start()<b.start()<headers[1].start(), 'owned FFN retirement escaped measured request')
            previous=b.start()
    else:
        require(not any(x in log for x in ('strata trace:','strata prefill dense:','strata loan audit:',
            'strata MoE mixed owners:','SYCL_EVENT_QUERY','strata decode cpu:','strata decode timing:')), 'clean run emitted observer records')
    return dict(actual_route_checked=True,**placement,residual_gpu_rows=117488,
        mode=10,windows_per_layer=[list(w) for w in admission.WINDOWS],checked_direct_owner_retirements=96 if check else None)


def prepare(stage,cohort):
    check=stage == 'check'; old,current,proof,original=qualification(); review=reviewed()
    loan=admitted_loan()
    if check:
        require(not RESULT.exists(), 'fresh comparison only; never retry')
        source=gen.git('rev-parse','HEAD').decode().strip(); processes=[]; previous=None
    else:
        previous=gen.committed(RESULT)
        require(previous['complete'] and previous['passed'] and previous['ownership_closed'] and not previous['active'] and
            not previous['fresh_faults'] and previous['plan'] == comparison_plan() and
            ((cohort == 0 and previous['stage'] == 'check') or
             (cohort > 0 and previous['stage'] == 'clean' and previous['cohort'] == cohort-1)), 'close preceding frozen process first')
        source=previous['source_commit']; processes=previous['processes']
        require(len(processes) == cohort and proof == previous['retained_build_proof'] and
            gen.common.digest(admission.RESULT) == previous['capacity_result_sha256'], 'comparison admission changed')
    arm=0 if check else ORDER[cohort]
    root=B/('retained-long-comparison-v1-check' if check else f'retained-long-comparison-v1-{cohort}')
    require(not root.exists(), 'existing/closed root; no retry')
    pins=gen.benchmark_pins(source); settings=gen.committed(gen.SETTINGS)
    config=copy.deepcopy(original['config']); flags=config['args']
    require('--expert-cache-per-layer' in flags and all(flags.count(k) == 1 and
        flags[flags.index(k)+1] == '0' for k in ('--adapt-swaps','--pcie-frac')),
        'admitted static per-layer cache/profile route required')
    flags[flags.index('--expert-cache')+1]=str(1256 if arm else 128)
    flags[flags.index('--expert-profile')+1]=str(root/'expert-profile.bin')
    if not arm: flags.append('--no-prefill-borrow')
    config.update(log=str(root/'engine.stderr'),env={})
    env={k:v.replace(original['root'],str(root)) for k,v in original['environment'].items()
        if k != 'STRATA_TRACE' and k != 'UR_ENABLE_LAYERS' and not k.startswith('UR_LOG_')}
    env.update(STRATA_SYCL_CACHE_SCRATCH_LEASE=str(arm),STRATA_SYCL_CACHE_LOAN_AUDIT='0')
    for k in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE'):
        require(env[k] == settings['environment'][k], 'qualified safety setting changed')
    if check:
        env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',STRATA_TRACE='1')
        env.update({k:v for k,v in settings['environment'].items() if k.startswith('UR_LOG_') and k != 'UR_LOG_TRACING'})
    model=assets.model_pins(config,original['model_assets'],full=check)
    machine=assets.machine_identity()
    if not check: require(machine == previous['machine_identity'], 'cohort platform changed')
    root.mkdir(mode=0o700)
    for name in ('cache','tmp'): (root/name).mkdir()
    shutil.copyfile(P/'data/expert-profile.bin',root/'expert-profile.bin')
    require(gen.common.digest(root/'expert-profile.bin') == original['profile_sha256'], 'profile changed')
    (root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    count=1 if check else 4
    command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
        '--prefill-only','--prefill-tokens','98304','--think','--runs',str(count),'--workload-id','native_gdn_20261018',
        '--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]
    binary=old['binary'] if not arm else current['binary']
    record=dict(mode='engine-compare',stage=stage,cohort=None if check else cohort,arm=arm,source_commit=source,
        build_source_commit=old['build_source_commit'] if not arm else current['source_commit'],binary=binary,
        retained_build_proof=proof,capacity_result_sha256=gen.common.digest(admission.RESULT),source_review=review,
        admitted_long_loan=loan,
        plan=comparison_plan(),processes=processes,check_evidence=None if check else previous['check_evidence'],
        root=str(root),config=config,environment=env,command=command,
        runtime_asset_delta=original['runtime_asset_delta'],config_sha256=gen.common.digest(root/'config.json'),
        profile_sha256=original['profile_sha256'],settings_sha256=gen.common.digest(gen.SETTINGS),
        tokenizer_assets=original['tokenizer_assets'],model_assets=model,model_assets_fingerprint=assets.fingerprint(model),
        machine_identity=machine,source_count=len(pins),benchmark_fingerprint=assets.fingerprint(pins),
        complete=False,passed=False,active=False,adoption_qualified=False,
        budgets=dict(wall=2400,cpu=54000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20))
    gen.runtime_assets_unchanged(record)
    (root/'plan.json').write_text(json.dumps(record,indent=2)+'\n'); RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print('Prepared '+stage+'; commit exact plan before run.')


def run():
    record=gen.committed(RESULT); root=Path(record['root'])
    require(record == json.loads((root/'plan.json').read_text()) and not record['complete'] and
        not (root/'record.json').exists(), 'fresh frozen plan required; no retry')
    old,current,proof,original=qualification()
    require(proof == record['retained_build_proof'] and reviewed() == record['source_review'] and
        admitted_loan() == record['admitted_long_loan'] and
        record['binary'] == (old['binary'] if not record['arm'] else current['binary']) and
        gen.common.digest(admission.RESULT) == record['capacity_result_sha256'], 'build/admission changed')
    pins=gen.benchmark_pins(record['source_commit'])
    require(len(pins) == record['source_count'] and assets.fingerprint(pins) == record['benchmark_fingerprint'], 'benchmark source changed')
    for path,expected in [(gen.SETTINGS,record['settings_sha256']),(root/'config.json',record['config_sha256']),
            (root/'expert-profile.bin',record['profile_sha256']),*record['tokenizer_assets'].items()]:
        require(gen.common.digest(path) == expected, 'frozen input changed: '+str(path))
    gen.runtime_assets_unchanged(record)
    require(assets.model_pins(record['config'],record['model_assets']) == record['model_assets'] and
        assets.machine_identity() == record['machine_identity'], 'model/platform changed')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
    cursor=next(x.split(': ',1)[1] for x in before.splitlines() if x.startswith('-- cursor: '))
    owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        (root/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    owner=supervision.Owner(root/'owned',persist)
    try:
        entry,_,_=owner.run('bench',record['command'],record['environment'],P,**record['budgets'])
        record.update(complete=supervision.closed(entry),exit_code=entry.get('exit_code'))
        require(record['complete'] and entry['exit_code'] == 0, 'official benchmark process failed')
        path,=(root/'bench').glob('*/results.json')
        check=record['stage'] == 'check'
        samples,journal=official(root,json.loads(path.read_text()),1 if check else 4)
        record.update(requests=samples,request_journal=journal)
        native=json.loads((root/'engine-exit.json').read_text())
        require(native['exit_code'] == 0 and native['boot_id'] == boot and native['binary'] == record['binary'] and
            native['args'] == ['--serve',*record['config']['args']] and 'resource_collector' not in native, 'native launch/closure differs')
        log=(root/'engine.stderr').read_text(errors='replace'); assets.verify_pool(log,5)
        record['route_evidence']=native_evidence(log,samples,record['arm'],check,record['admitted_long_loan'])
        _,_,after_proof,_=qualification()
        require(after_proof == proof and gen.benchmark_pins(record['source_commit']) == pins and
            assets.machine_identity() == record['machine_identity'], 'source/platform changed during run')
        gen.runtime_assets_unchanged(record)
        assets.model_pins(record['config'],record['model_assets'],full=not check and record['cohort'] == 5)
        record.update(passed=True,native_identity=native['native'])
    except BaseException as error:
        record['error']=str(error)
        paths=list((root/'bench').glob('*/results.json'))
        if len(paths) == 1:
            record['requests']=json.loads(paths[0].read_text()).get('results',{}).get('prefill_only',{}).get('requests',[])
        for path in (root/'engine.stderr',root/'owned/bench.stderr'):
            if path.exists(): record['error_excerpt']=path.read_text(errors='replace')[-4096:]
    finally:
        finalize(record,owner,cursor,boot,persist,RESULT)
    print(json.dumps({k:record.get(k) for k in ('complete','passed','ownership_closed','error','comparison_summary')}))
    return 0 if record['passed'] else 1


def read_boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def finalize(record,owner,cursor,boot,persist,result_path):
    # Unknown fault evidence must never become a successful empty fault list.
    record.update(ownership_closed=False,fresh_faults=None,fault_probe_successful=False,boot_unchanged=False)
    try:
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        try:
            after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            record['fresh_faults']=[x for x in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
            record['fault_probe_successful']=True
        except Exception as error:
            record['kernel_journal_probe_error']=str(error)
        try: record['boot_unchanged']=boot == read_boot()
        except Exception as error: record['boot_probe_error']=str(error)
        try: record['driver_after']=guards.driver_guard()
        except Exception as error: record['driver_guard_error']=str(error)
        record['passed']=bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and
            record['fault_probe_successful'] and record['fresh_faults'] == [] and 'driver_guard_error' not in record)
        if record['passed']:
            if record['stage'] == 'check':
                record['check_evidence']=dict(passed=True,route_evidence=record['route_evidence'],native_identity=record['native_identity'])
            else:
                keys=('cohort','arm','build_source_commit','machine_identity','model_assets_fingerprint',
                    'benchmark_fingerprint','native_identity','requests','passed','ownership_closed','fresh_faults')
                row={k:record[k] for k in keys}
                row.update(binary_sha256=record['binary']['sha256'],observer_records_absent=True,actual_route_checked=True)
                record['processes'].append(row)
                if record['cohort'] == 5:
                    try: record['comparison_summary']=summarize(record['processes'])
                    except Exception as error: record.update(passed=False,error='Comparison eligibility: '+str(error))
    except BaseException as error:
        record.update(passed=False,finalization_error=str(error))
    finally:
        try: persist()
        except BaseException as error:
            record.update(passed=False,persistence_error=str(error))
            raise
        finally: result_path.write_text(json.dumps(record,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare-check','prepare','run'))
    parser.add_argument('--cohort',type=int,choices=range(6))
    args=parser.parse_args()
    require((args.action == 'prepare') == (args.cohort is not None), 'only prepare requires --cohort')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); guards.no_model_owners([]); guards.driver_guard()
        return run() if args.action == 'run' else prepare('check' if args.action == 'prepare-check' else 'clean',args.cohort)


if __name__ == '__main__': sys.exit(main())
