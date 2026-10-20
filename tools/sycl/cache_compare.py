#!/usr/bin/env python3
"""Main-owned matched owned128/borrowed1280 cache comparison; no inference rebuild."""
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

sys.dont_write_bytecode = True
import query_profile as q
import resource_engine as resources

gen, assets, guards, supervision = q.gen, q.assets, q.guards, q.supervision
P, B, A = gen.P, gen.B, gen.A
CAPACITY = A/'cache-capacity-current.json'
RESULT = A/'cache-comparison-current.json'
RESEARCH = A/'research-index.json'
SCOPE = 'cache_capacity_clean_comparison_scope'
REVIEW_INPUTS = ('tools/sycl/cache_compare.py','tools/test_sycl_cache_compare.py',
                'tools/sycl/generation_engine.py')
ORDER = (0,1,1,0,0,1)
PAIRS = ((0,1),(3,2),(4,5))


def comparison_plan():
    return dict(order=list(ORDER),pairs=[list(p) for p in PAIRS],
        question='Does reusing the prefill allocation as a larger decode cache improve the production32K+ strong-MTP route?',
        arms={'0':'Owned8192-row scratch,128 expert slots,no-prefill-borrow,cache_scratch_lease0.',
              '1':'Checked borrowed8192-row scratch,1280 expert slots,cache_scratch_lease1.'},
        constants='Same native executable, model/tokenizer/profile,131072 residentINT8 context,8192 chunk,384 staged expert bank, GPU BF16 residuals/mode10, GU/Down-large,5 CPU workers plus host,pcie-frac0,spec4,no-token-graph,no-spec-split,no adaptation. Same official workload ID and four warmups; one fresh regular-code request per independent process.',
        eligibility='All six clean processes close with the frozen native/benchmark/model/runtime/boot/config identities and actual selected long route. Each fresh input>=32768 and output>=1024 including thinking plus answer. Per pair require generation-count ratio within1/1.05..1.05 and MTP acceptance difference<=0.02, both acceptance>=0.60. Exact output/draft equality is not required; the cache placement may change rounding. A work-regime mismatch is inconclusive, not a discarded sample.',
        decisions='Use exact native ms clocks per actual generated/read token. Decode faster only if every candidate/base throughput ratio>1 and geomean>1.05; slower only if every ratio<1 and geomean<1/1.05. Prefill faster/slower screens use the same all-pair rule with1.02 and1/1.02. Otherwise inconclusive. Prefill compares identical fresh inputs independently of decode generation/MTP regime. Its nonregression requires geomean>=0.98 and each ratio>=0.95; no prefill/decode tradeoff adoption. Keep all samples and spreads. No automatic extra repeats or retry; a new cohort plan is needed after a failed/ineligible result.',
        limits='A strong-workload mechanism screen, not an isolated kernel/resource ceiling or adoption. Weak-MTP,agent-session,quality,physical262144/session/RESTORE and sustained70tok/s remain open. CPU/RAM/PCIe traffic is not measured by logical miss counts.')


def invariant_signature(config, environment, root):
    config = copy.deepcopy(config);flags = config['args']
    flags[flags.index('--expert-cache')+1] = '<cache-size>'
    if '--no-prefill-borrow' in flags: flags.remove('--no-prefill-borrow')
    environment = dict(environment);environment['STRATA_SYCL_CACHE_SCRATCH_LEASE'] = '<lease>'
    value = json.dumps(dict(config=config,environment=environment),sort_keys=True).replace(str(root),'<root>')
    return hashlib.sha256(value.encode()).hexdigest()


def summarize(rows):
    guards.require(len(rows)==6 and [r['arm'] for r in rows]==list(ORDER), 'six frozen-order samples required')
    guards.require(len({(r['native_identity']['pid'],r['native_identity']['start_ticks']) for r in rows})==6,
        'native process reused')
    for key in ('source_commit','binary_sha256','invariant_signature','machine_identity','model_assets_fingerprint'):
        guards.require(all(r[key]==rows[0][key] for r in rows), 'comparison identity changed: '+key)
    guards.require(all(r['passed'] and r['ownership_closed'] and not r['fresh_faults'] and
        r['observer_records_absent'] and r['successful_warmups']==4 and gen.sample_eligible(r['request']) and
        r['route_evidence']['actual_mode']==10 and
        r['route_evidence']['expert_cache_slots']==(1280 if r['arm'] else 128) and
        r['route_evidence']['gpu_residual_rows']==r['request']['read_tokens']-1 and
        r['request']['prompt_tps_source']=='engine' and isinstance(r['request']['prompt_ms'],(int,float)) and
        not isinstance(r['request']['prompt_ms'],bool) and math.isfinite(r['request']['prompt_ms']) and
        r['request']['prompt_ms']>0 for r in rows), 'ineligible clean sample')
    prompts={(r['request']['request_sha256'],r['request']['read_tokens'],r['request']['prompt_tokens']) for r in rows}
    guards.require(len(prompts)==1, 'request/input work changed')
    decode,prefill,regimes = [],[],[]
    for left,right in PAIRS:
        base,candidate = rows[left]['request'],rows[right]['request']
        length_ratio=candidate['generated_tokens']/base['generated_tokens']
        regimes.append(1/1.05 <= length_ratio <= 1.05 and
            min(base['acceptance_rate'],candidate['acceptance_rate'])>=0.60 and
            abs(base['acceptance_rate']-candidate['acceptance_rate'])<=0.02)
        decode.append((candidate['generated_tokens']/candidate['decode_ms']) /
                      (base['generated_tokens']/base['decode_ms']))
        prefill.append((candidate['read_tokens']/candidate['prompt_ms']) /
                       (base['read_tokens']/base['prompt_ms']))
    geo=lambda values:math.exp(statistics.mean(map(math.log,values)))
    dg,pg=geo(decode),geo(prefill)
    matched=all(regimes)
    decision=('inconclusive_generation_or_MTP_regime' if not matched else
        'candidate_faster_screen' if all(x>1 for x in decode) and dg>1.05 else
        'candidate_slower_screen' if all(x<1 for x in decode) and dg<1/1.05 else 'inconclusive_variability_or_small_effect')
    spreads={}
    for arm in (0,1):
        selected=[r['request'] for r in rows if r['arm']==arm]
        values=[1000*r['generated_tokens']/r['decode_ms'] for r in selected]
        spreads[str(arm)]=dict(decode_tps_median=statistics.median(values),decode_tps_min=min(values),
            decode_tps_max=max(values),decode_tps_sample_sd=statistics.stdev(values))
    return dict(work_regime_matched=matched,decode_ratio=decode,decode_ratio_geomean=dg,
        prefill_ratio=prefill,prefill_ratio_geomean=pg,prefill_nonregression=pg>=0.98 and all(x>=0.95 for x in prefill),
        prefill_decision=('candidate_faster_screen' if all(x>1 for x in prefill) and pg>1.02 else
            'candidate_slower_screen' if all(x<1 for x in prefill) and pg<1/1.02 else 'inconclusive_variability_or_small_effect'),
        decode_decision=decision,spread=spreads,adoption_qualified=False,
        candidate70tok_s_all_samples=all(1000*r['request']['generated_tokens']/r['request']['decode_ms']>=70 for r in rows if r['arm']==1),
        limits=comparison_plan()['limits'])


def native_evidence(log, sample, arm, check):
    headers=list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$',log,re.M))
    guards.require(len(headers)==5 and tuple(map(int,headers[-1].groups()))==
        (sample['prompt_tokens'],0,sample['read_tokens'],sample['generated_tokens']), 'native request binding differs')
    text=log[headers[-2].end():headers[-1].start()];rows=sample['read_tokens']-1
    guards.require(32767<=rows<=65536, 'primary32K window required')
    pattern=r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) residual_host=0 residual_staging=0 layer_cache=1022361600; extra allocations, GPU headroom=536870912, RAM headroom=8589934592$'
    plans=re.findall(pattern,text,re.M)
    guards.require(len(plans)==1 and tuple(map(int,plans[0]))==(rows,rows*20480,rows,rows*20480) and
        len(re.findall(r'^strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows='+str(rows)+r' mixed_fp16_bytes=0$',text,re.M))==1 and 'effective_mode=2' not in text,
        'actual all-GPU BF16/mode10 route differs')
    cache=re.findall(r'^strata generate: expert cache (\d+) slots, [^\n]+$',log,re.M)
    loan=re.findall(r'^strata serve: the prompt path borrows (\d+) CUDA0 cache slots [^\n]+$',log,re.M)
    owned='strata serve: the prompt path allocates its own buffers (too few cache slots to borrow)' in log
    guards.require(cache==[str(1280 if arm else 128)] and
        (len(loan)==1 and 0<int(loan[0])<=1152 and not owned if arm else not loan and owned), 'actual scratch/cache placement differs')
    hit=re.findall(r'^strata serve: decode expert cache hit rate: ([0-9.]+)% \((\d+) hits / (\d+) lookups\)$',log[headers[-1].start():],re.M)
    counts=re.findall(r'^strata serve: SYCL_EXPERT_COPY submitted_blobs=0 submitted_bytes=0 completed_blobs=0 completed_bytes=0 host_view_groups=0 gpu_miss_groups=0 cpu_unique_misses=(\d+) cpu_entries=(\d+)$',log[headers[-1].start():],re.M)
    guards.require(len(hit)==len(counts)==1 and int(hit[0][2])>0 and
        0<=int(hit[0][1])<=int(hit[0][2]) and 0<int(counts[0][0])<=int(counts[0][1]), 'logical miss evidence differs')
    if check:
        entries=list(re.finditer(r'^strata MoE mixed owners: phase=entry layer=(\d+) first=0 rows=(\d+)[^\n]+ direct_owners=1 previous_attention=checked_drain$',text,re.M))
        done=list(re.finditer(r'^strata MoE mixed owners: phase=retired layer=(\d+) first=0 rows=(\d+) route_status=0 shared_HC=checked_drain callbacks=after_return$',text,re.M))
        guards.require(not arm and [tuple(map(int,x.groups())) for x in entries]==[tuple(map(int,x.groups())) for x in done]==[(i,rows) for i in range(48)], 'owned model FFN evidence incomplete')
        previous=-1
        for entry,retired in zip(entries,done):
            guards.require(previous<entry.start()<retired.start(),'owned FFN retirement order differs')
            previous=retired.start()
    else:
        guards.require(not any(x in log for x in ('strata trace:','strata prefill dense:',
            'strata MoE mixed owners:','strata loan audit:','SYCL_EVENT_QUERY','strata decode cpu:','strata decode timing:')), 'clean process emitted observer records')
    return dict(actual_mode=10,gpu_residual_rows=rows,expert_cache_slots=1280 if arm else 128,
        scratch='checked-borrowed' if arm else 'owned',cache_hits=int(hit[0][1]),cache_lookups=int(hit[0][2]),
        rounded_hit_percent=float(hit[0][0]),cpu_unique_miss_jobs=int(counts[0][0]),cpu_entries=int(counts[0][1]),
        limits='Runtime mode10/residual/placement logs plus unchanged checked entry gates and complete generation. Clean runs omit per-FFN/readback audits; full audit was qualified separately. Logical counts are not physical traffic.')


def qualification():
    full,count=q.current_build();cap=gen.committed(CAPACITY)
    guards.require(cap['complete'] and cap['passed'] and cap['ownership_closed'] and not cap['active'] and
        not cap['fresh_faults'] and cap['binary']==full['binary'] and cap['long_capacity_evidence']['checked_retirements']==48 and
        cap['full_model_hashes_before_and_after_match'], 'matching completed long loan admission required')
    ref=cap['original_plan'];raw=gen.git('show',ref['commit']+':'+ref['path'])
    guards.require(hashlib.sha256(raw).hexdigest()==ref['sha256'], 'original admitted plan changed')
    original=json.loads(raw)
    guards.require(original['binary']==full['binary'] and gen.common.digest(q.BUILD_RECORD)==original['source_build_record']['sha256'] and
        gen.common.digest(q.COMPONENT)==original['component_result_sha256'], 'qualification build/component changed')
    for name,sha in original['source_review']['frozen_inputs'].items():
        guards.require(gen.common.digest(P/name)==sha,'reviewed source changed: '+name)
    return full,count,cap,original


def source_review():
    row=gen.committed(RESEARCH)[SCOPE]
    guards.require(row.get('clean_comparison_accepted') is True and row.get('worker_closed') is True and
        set(row['frozen_inputs'])==set(REVIEW_INPUTS), 'closed independent comparison review required')
    for name,sha in row['frozen_inputs'].items():
        guards.require(gen.common.digest(P/name)==sha and
            hashlib.sha256(gen.git('show',row['source_commit']+':'+name)).hexdigest()==sha,
            'reviewed comparison source changed: '+name)
    return {key:row[key] for key in ('source_commit','frozen_inputs','report_sha256','native_review_sha256')}


def prepare(stage,cohort):
    check=stage=='check';full,count,cap,original=qualification()
    review=None if check else source_review()
    if check:
        guards.require(not RESULT.exists(), 'fresh comparison only; never retry')
        source=gen.git('rev-parse','HEAD').decode().strip();processes=[];prior_check=None
    else:
        previous=gen.committed(RESULT)
        guards.require(previous['complete'] and previous['passed'] and previous['ownership_closed'] and
            not previous['active'] and not previous['fresh_faults'] and previous['plan']==comparison_plan() and
            ((cohort==0 and previous['stage']=='check') or
             (cohort>0 and previous['stage']=='clean' and previous['cohort']==cohort-1)), 'close frozen check/previous process first')
        source=previous['source_commit'];processes=previous['processes'];prior_check=previous['check_evidence']
        guards.require(prior_check['passed'] and len(processes)==cohort,'missing owned check or cohort samples')
        guards.require(gen.common.digest(CAPACITY)==previous['capacity_qualification_sha256'] and
            full['binary']==previous['binary'] and gen.common.digest(q.COMPONENT)==previous['component_result_sha256'], 'frozen qualification changed')
    arm=0 if check else ORDER[cohort]
    root=B/('cache-comparison-v1-check' if check else f'cache-comparison-v1-{cohort}')
    guards.require(not root.exists(), 'closed/live root exists; no retry')
    pins=gen.benchmark_pins(source);settings=gen.committed(gen.SETTINGS)
    config=copy.deepcopy(original['config']);flags=config['args']
    flags[flags.index('--expert-cache')+1]=str(1280 if arm else 128)
    flags[flags.index('--expert-profile')+1]=str(root/'expert-profile.bin')
    if not arm: flags.append('--no-prefill-borrow')
    config.update(log=str(root/'engine.stderr'),env={})
    env={k:v.replace(original['root'],str(root)) for k,v in original['environment'].items()
        if k!='STRATA_TRACE' and k!='UR_ENABLE_LAYERS' and not k.startswith('UR_LOG_')}
    env.update(STRATA_SYCL_CACHE_SCRATCH_LEASE=str(arm),STRATA_SYCL_CACHE_LOAN_AUDIT='0')
    for key in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[key]==settings['environment'][key],'qualified safety setting changed')
    signature=invariant_signature(config,env,root)
    if check:
        model=assets.model_pins(config,original['model_assets'],full=True)
        env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',STRATA_TRACE='1')
        env.update({k:v for k,v in settings['environment'].items() if k.startswith('UR_LOG_') and k!='UR_LOG_TRACING'})
    else:
        guards.require(signature==previous['invariant_signature'], 'unplanned arm difference')
        model=assets.model_pins(config,previous['model_assets'])
    machine=assets.machine_identity()
    if not check: guards.require(machine==previous['machine_identity'],'cohort platform changed')
    root.mkdir(mode=0o700)
    for name in ('cache','tmp'): (root/name).mkdir()
    shutil.copyfile(P/'data/expert-profile.bin',root/'expert-profile.bin')
    guards.require(gen.common.digest(root/'expert-profile.bin')==original['profile_sha256'],'profile changed')
    (root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
        '--long-generation-only','--long-generation-workload','regular-code','--think','--runs','1',
        '--workload-id',gen.WORKLOAD_ID,'--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]
    record=dict(mode='cache-owned-check' if check else 'cache-capacity-clean',stage=stage,cohort=cohort,arm=arm,source_commit=source,
        build_source_commit=full['source_commit'],binary=full['binary'],original_native_inputs_recovered=count,
        source_build_record=original['source_build_record'],component_result_sha256=gen.common.digest(q.COMPONENT),
        capacity_qualification_sha256=gen.common.digest(CAPACITY),capacity_original_plan=cap['original_plan'],
        root=str(root),config=config,environment=env,command=command,invariant_signature=signature,plan=comparison_plan(),
        config_sha256=gen.common.digest(root/'config.json'),profile_sha256=original['profile_sha256'],settings_sha256=gen.common.digest(gen.SETTINGS),
        model_assets=model,model_assets_fingerprint=assets.fingerprint(model),machine_identity=machine,
        runtime_asset_delta=original['runtime_asset_delta'],source_count=len(pins),source_aggregate_sha256=assets.fingerprint(pins),
        processes=processes,check_evidence=prior_check,clean_source_review=review,
        complete=False,passed=False,active=False,adoption_qualified=False,
        budgets=dict(wall=2400,cpu=54000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20))
    gen.runtime_assets_unchanged(record)
    (root/'plan.json').write_text(json.dumps(record,indent=2)+'\n');RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print('Prepared '+stage+'; commit the exact plan before run.')


def run():
    record=gen.committed(RESULT);root=Path(record['root']);check=record['stage']=='check'
    guards.require(record==json.loads((root/'plan.json').read_text()) and not record['complete'] and
        not (root/'record.json').exists(),'fresh frozen plan required; no retry')
    full,count,_,_=qualification()
    guards.require(full['binary']==record['binary'] and count==record['original_native_inputs_recovered'] and
        gen.common.digest(CAPACITY)==record['capacity_qualification_sha256'],'qualification changed')
    if not check: guards.require(source_review()==record['clean_source_review'],'comparison review changed')
    pins=gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins)==record['source_count'] and assets.fingerprint(pins)==record['source_aggregate_sha256'],'benchmark changed')
    for path,expected in [(gen.SETTINGS,record['settings_sha256']),(root/'config.json',record['config_sha256']),
                          (root/'expert-profile.bin',record['profile_sha256'])]:
        guards.require(gen.common.digest(path)==expected,'input changed: '+str(path))
    gen.runtime_assets_unchanged(record)
    guards.require(assets.model_pins(record['config'],record['model_assets'])==record['model_assets'] and
        assets.machine_identity()==record['machine_identity'],'model/platform changed')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
    cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '));owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        (root/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    owner=supervision.Owner(root/'owned',persist)
    try:
        entry,_,_=owner.run('bench',record['command'],record['environment'],P,**record['budgets'])
        record.update(complete=supervision.closed(entry),exit_code=entry.get('exit_code'))
        guards.require(record['complete'] and entry['exit_code']==0,'owned official benchmark failed')
        path,=(root/'bench').glob('*/results.json');suite=json.loads(path.read_text())['results']['long_generation']
        samples=gen.collect_samples(suite);record['requests']=samples;gen.require_complete_generation(suite,samples,('regular-code',))
        journal,rows=gen.journal_evidence(root);resources.require_journal_binding(samples,rows)
        record['warmups']=gen.warmup_evidence(root,journal)
        native=json.loads((root/'engine-exit.json').read_text())
        guards.require(native['exit_code']==0 and native['boot_id']==boot and native['binary']==record['binary'] and
            native['args']==['--serve',*record['config']['args']] and 'resource_collector' not in native,'native launch/closure differs')
        log=(root/'engine.stderr').read_text(errors='replace');assets.verify_pool(log,5)
        record['route_evidence']=native_evidence(log,samples[0],record['arm'],check)
        record['native_identity']=native['native'];record['observer_records_absent']=not check
        qualification();gen.runtime_assets_unchanged(record)
        guards.require(gen.benchmark_pins(record['source_commit'])==pins and assets.machine_identity()==record['machine_identity'] and
            assets.model_pins(record['config'],record['model_assets'],full=not check and record['cohort']==5)==record['model_assets'], 'inputs changed during run')
        record['passed']=True
    except BaseException as error:
        record['error']=str(error)
        try:
            journal,rows=gen.journal_evidence(root);record['request_journal']=journal
            if 'requests' not in record:record['requests']=rows
        except Exception as failure:record['journal_recovery_error']=str(failure)
        for path in (root/'engine.stderr',root/'owned/bench.stderr'):
            if path.exists():record['error_excerpt']=path.read_text(errors='replace')[-4096:]
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True)
        record['fresh_faults']=[x for x in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
        record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        try:record['driver_after']=guards.driver_guard()
        except Exception as error:record.update(driver_guard_error=str(error),passed=False)
        record['passed']=bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and not record['fresh_faults'])
        row={k:record.get(k) for k in ('arm','source_commit','native_identity','invariant_signature','machine_identity','model_assets_fingerprint','passed','ownership_closed','fresh_faults','observer_records_absent','route_evidence')}
        row.update(binary_sha256=record['binary']['sha256'],request=(record.get('requests') or [{}])[0],
            successful_warmups=record.get('warmups',{}).get('successful_warmups',0))
        if check:record['check_evidence']=dict(passed=record['passed'],source_commit=record['source_commit'],native_identity=record.get('native_identity'))
        else:
            record['processes'].append(row)
            if record['passed'] and record['cohort']==5:
                try:record['comparison_summary']=summarize(record['processes'])
                except Exception as error:record.update(passed=False,summary_error=str(error))
        persist();RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ('stage','cohort','arm','complete','passed','ownership_closed','error')}))
    return 0 if record['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('stage',nargs='?',choices=('check','clean'),default='check');parser.add_argument('cohort',nargs='?',type=int,choices=range(6),default=0);args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare(args.stage,args.cohort) if args.action=='prepare' else run()


if __name__=='__main__':sys.exit(main())
