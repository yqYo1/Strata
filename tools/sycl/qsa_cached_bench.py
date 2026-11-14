#!/usr/bin/env python3
"""Same-binary ordered-exp/cached-subgroup long-prefill comparison.

Main-authored; repository LICENSE applies. Recover closed model evidence from
Git/public results, without depending on its retired diagnostic directory.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import qsa_cached_model_entry as model
import qsa16_stream_bench as original

device = model.device
P, B, RESULT = model.P, model.B, original.RESULT
THIS = 'tools/sycl/qsa_cached_bench.py'
ROOT, KEY = B/'qsa-cached-subgroup-prefill-compare-v1', 'qsa_cached_subgroup_prefill_compare_v1'
ARMS = (0, 1, 1, 0, 0, 1)
CASES = tuple((read, arm) for read in (32769, 65537) for arm in ARMS)
SELECTORS = ('bf16-prefix-ordered-exp', 'bf16-prefix-cached-subgroup')
RUNS = 3
require, sha = model.require, model.sha


def basis(*, observer_files=frozenset()):
    observers = {THIS, *observer_files}
    native = device.native(observers={device.THIS, model.THIS, *observers})
    prior = device.original.validation.committed(RESULT, (model.KEY,))
    dev = device.original.validation.committed(RESULT, (device.KEY,))
    require(prior['stage'] == 'closed' and all(prior[k] for k in
        ('passed','complete','ownership_closed','model_executed')) and not prior['active']
        and prior['native_source_commit'] == native['native_source_commit']
        and prior['native_build']['binaries'] == native['binaries']
        and sha(prior['native_record']['path']) == prior['native_record']['sha256'],
        'closed exact eight-ELF actual-model qualification required')
    admission = prior['native_observer_admission']
    require(dev['stage'] == 'closed' and dev['passed'] and dev['complete']
        and dev['ownership_closed'] and not dev['active']
        and dev['binaries'] == native['binaries']
        and admission['fresh_nine_stage_controls']
        and admission['device_evidence_sha256'] == prior['device_evidence_sha256']
        == original.source.signature(dev)
        and admission['exact_seed_recipe_sha256'] == native['observer_admission']['exact_seed_recipe_sha256']
        and [v['mode'] for v in dev['results']] == list(device.STAGES)
        and all(v['passed'] and v['ownership_closed'] and v['native_exit_code'] == 0 for v in dev['results']),
        'unchanged original nine-stage same-native evidence required')
    require(set(admission['added_observers']) == {device.THIS, model.THIS}
        and all(native['observer_admission']['added_observers'][p] == h
            for p,h in admission['added_observers'].items()), 'both original qualified observers changed')
    result, = prior['results']
    routes = result['routes']
    require(result['passed'] and result['native_exit_code'] == 0 and result['ownership_closed']
        and result['fresh_faults'] == [] and result['boot_unchanged']
        and result['driver_after'] == prior['driver']
        and [(v['prompt_tokens'],v['generated'],v['complete']) for v in result['requests']]
            == [(32769,32,True),(82,32,True),(32769,32,True)]
        and routes['actual_attention_math'] == SELECTORS[1]
        and routes['actual_bulk_calls'] == 96 and routes['accepted_copy_commands'] == 384
        and routes['requested_D2H_bytes'] == 830472192
        and routes['MTP_expert_restore_bytes'] == [707788800]*2
        and routes['cache_binding_and_final_drains'], 'actual cached-QSA/bulk/decode handoff required')
    pin = prior['controller']
    prefixes = ['tools/sycl','tools/strata_tokenizer.py','data/expert-profile.bin']
    names = device.original.common.source.git(P,'ls-tree','-r','--name-only',pin['revision'],'--',*prefixes).decode().splitlines()
    before = {p:hashlib.sha256(raw).hexdigest() for p,raw in
        device.original.common.source.blobs(P,pin['revision'],names)}
    aggregate = hashlib.sha256(json.dumps(before,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    after = device.original.common.source.pin(P,prefixes)['source_sha256']
    require(len(before) == pin['count'] and aggregate == pin['aggregate_sha256']
        and set(after)-set(before) == observers and not set(before)-set(after)
        and all(after[p] == h for p,h in before.items()),
        'recover exact original model controls;only explicitly declared observers may extend them')
    reference = json.loads(model.original.REFERENCE.read_text())
    require(reference['models_sha256'] == prior['models_sha256']
        == model.original.signature(reference['models']), 'original15asset manifest changed')
    require(sha(prior['runtime_manifest']['path']) == prior['runtime_manifest']['sha256'],
        'original shared runtime manifest changed')
    providers = json.loads(Path(prior['runtime_manifest']['path']).read_text())
    require(all(sha(p) == h for p,h in providers.items())
        and original.source.model.guards.driver_guard() == prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot']
        and all(prior['environment'].get(k) == v for k,v in original.FAST.items()),
        'original runtime,boot,driver or qualified fast configuration changed')
    prior = copy.deepcopy(prior)
    prior.update(models=reference['models'],runtime_files=providers)
    model.original.previous.assets(prior)
    diagnostic = dict(key=model.KEY,evidence_sha256=original.source.signature(
        device.original.validation.committed(RESULT,(model.KEY,))),
        original_controller_source=prior['source_commit'],original_native_source=prior['native_source_commit'],
        original_nine_stage_and_actual_model_verified=True,full15final_hashes_reused_after_exact_identity_check=True,
        model_PP_TG_task_full262144_RESTORE_inherited=False)
    prior['observer_admission'] = dict(native=native['observer_admission'],
        original_model_evidence=diagnostic,original_model_controls_recovered=len(before),
        added_observers={p:after[p] for p in sorted(observers)},fast_configuration_diagnostic=diagnostic,
        native_recompiled=False,earlier_clean_result_inherited=False,
        retired_raw_model_directory_consumed=False)
    return native, prior, providers, None


def decision(processes):
    require(len(processes) == len(CASES) and all(p['passed'] and p['complete']
        and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
        and p['read'] == CASES[i][0] and len(p['samples']) == RUNS
        for i,p in enumerate(processes)), 'all12ordered clean processes required')
    points = []
    for offset,read in ((0,32769),(6,65537)):
        local = processes[offset:offset+6]
        def arm_summary(arm):
            selected = [p for i,p in enumerate(local) if ARMS[i] == arm]
            warm = [p['subsequent_median_ms'] for p in selected]
            pp = [read*1000/v for v in warm]
            first = [p['samples'][0]['prompt_ms'] for p in selected]
            return dict(processes=3,subsequent_median_prompt_ms=statistics.median(warm),
                subsequent_PP=read*1000/statistics.median(warm),
                subsequent_process_PP_min=min(pp),subsequent_process_PP_max=max(pp),
                subsequent_median_client_TTFT_s=statistics.median(
                    statistics.median(s['ttft_s'] for s in p['samples'][1:]) for p in selected),
                first_median_prompt_ms=statistics.median(first),first_PP=read*1000/statistics.median(first),
                first_median_client_TTFT_s=statistics.median(p['samples'][0]['ttft_s'] for p in selected))
        ratios = [local[b]['subsequent_median_ms']/local[a]['subsequent_median_ms']
            for a,b in ((0,1),(3,2),(4,5))]
        points.append(dict(read=read,baseline=arm_summary(0),cached=arm_summary(1),
            paired_subsequent_prompt_time_ratios=ratios,
            geometric_mean_time_ratio=statistics.geometric_mean(ratios),
            all_three_pairs_favor_cached=all(v<1 for v in ratios)))
    gain = all(p['all_three_pairs_favor_cached'] for p in points)
    return dict(points=points,clean_speed=True,matched_same_binary=True,
        independent_processes_per_arm_per_length=3,
        mechanism_evidence='repeated_improvement_at_both_lengths' if gain else 'not_consistently_faster',
        advance_to_further_model_gates=gain,
        prefill1000_on_64K_subsequent_median_observed=points[1]['cached']['subsequent_PP']>=1000,
        local_adjacent_ms_per_token={a:(points[1][a]['subsequent_median_prompt_ms']-
            points[0][a]['subsequent_median_prompt_ms'])/(65537-32769) for a in ('baseline','cached')},
        curve_scope='Two compatible lengths give local slopes,not a fitted law or physical fixed-cost estimate.',
        sustained_decode_qualified=False,agent_task_quality_qualified=False,
        physical262144_qualified=False,resource_saturation_qualified=False,inference_adopted=False,
        next='Keep all first/warmed samples and spread. Mixed evidence needs a new repetition plan;no noise relabeling. Preserve independent decode and both selectors;full-context/RESTORE,task/agent and sustained decode remain separate gates.')


def context():
    spec = importlib.util.spec_from_file_location('qsa_cached_clean_comparison',original.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT,ctx.KEY,ctx.THIS = ROOT,KEY,THIS
    ctx.READS,ctx.ORDER = tuple(r for r,_ in CASES),[(r,1) for r,_ in CASES]
    ctx.RUNS,ctx.CLEAN_FAST,ctx.basis,ctx.decision = RUNS,True,basis,decision
    base_environment,base_current,base_route = ctx.environment,ctx.current,ctx.request_route
    def environment(row,plan):
        i=plan['index']; require(type(i) is int and 0<=i<len(CASES)
            and plan['read']==CASES[i][0], 'unplanned QSA comparison arm')
        env=base_environment(row,plan)
        env.update(STRATA_SYCL_PREFILL_QSA_MATH=SELECTORS[CASES[i][1]],
            STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR='1',
            STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING='0')
        return env
    def current(row):
        require(row['arm_order']==[list(v) for v in CASES] and len(row['plans'])==len(CASES)
            and all((p['read'],p['qsa_math'])==(CASES[i][0],SELECTORS[CASES[i][1]])
                and p['index']==i and p['heads']==1 for i,p in enumerate(row['plans'])),
            'frozen12process arm order changed')
        return base_current(row)
    def request_route(text,plan):
        route=base_route(text,plan)
        values=re.findall(r'^strata prefill: QSA BF16 operands, FP32 online state/accumulation; '
            r'KV=(\S+) transient_bytes=(\d+); source=resident-Mode0 score_pitch=(\d+) '
            r'operand_layout=(\S+) softmax=(\S+)$',text,re.M)
        arm=CASES[plan['index']][1]
        require(len(values)==1 and values[0][0]=='active-logical-prefix' and int(values[0][1])>0
            and values[0][2:]==('33','row-major',('parallel-exp-ascending-sum','subgroup')[arm]),
            'actual timed attention arithmetic differs from the immutable arm')
        route.update(actual_attention_math=SELECTORS[arm],bulk_mirror=1,expert_upload_staging=0,
            selector_evidence='Actual request-entry softmax/prefix/pitch/layout plus live immutable environment at both request gates;exact native dispatch. No per-launch bulk count or physical traffic claim.')
        return route
    ctx.environment,ctx.current,ctx.request_route=environment,current,request_route
    return ctx


def prepare():
    ctx=context();ctx.prepare()
    row=json.loads((ROOT/'record.json').read_text())
    row['arm_order']=[list(v) for v in CASES]
    for i,p in enumerate(row['plans']):
        p['qsa_math']=SELECTORS[CASES[i][1]]
        require(p['environment_sha256']==ctx.comparison.fingerprint(ctx.environment(row,p)),
            'prepared immutable environment differs')
    row['whole_bytes']=1536<<20
    row['plan']=dict(question='Does cached-subgroup QSA improve long-input PP/clientTTFT over ordered-exp on the same current9af engine?',
        order='12fresh processes:32K ABBAAB then64K ABBAAB;three independent processes/arm/length. Three exact-count thinking-on bench.sh requests each,24outputs;first request separate from subsequent2. Matched per-length request body/nonce sequences;read=input,reuse0. Commit every closed process before advancing.',
        settings='Exact cf7/9af and eight same-build fixtures,logical262144/resident20480INT8,8192chunk,all residual rows onGPU,packed384bank/source-view1,owned264MiBstage,spec4,head2/dense1/backing0,MTPFP16batch1 and all9FASTsettings. Bulk mirror1 and expert-upload staging0 in BOTH arms. Only QSA math and private process paths differ;qualified runtime safety and decode selectors remain fixed.',
        acceptance='Fresh current-source CPU/frontend controls before native;exact source/recipe/SDK/8ELFs/providers/model identities,pre-main validation,owned single engine,exact thinking/read/reuse/output counts,valid prompt/client clocks,actual8192/typed/prefix/layout/softmax and packed-bank/head/MTP handoff,quiet validation,normal native+bench0,reaped/empty,unchanged boot/no fresh GPU fault.',
        comparison='Compare3paired subsequent-request process-median prompt-time ratios at each length;require all3favor cached at both lengths for repeated-gain advancement,with no arbitrary minimum gain. Keep first use,PP,TTFT and all individual samples/spread. TTFT and warmed throughput are separate strategy results;do not infer sustained TG from24outputs.',
        stopping='Stop first source,owner,provider,token/clock/route/validation/fault or finite-budget failure. No retry,pin refresh,criterion change,speed-based stop or dropped samples. Reuse original closed-model final15full hashes after exact identities;hash all15again after the cohort.',
        limits='Same-binary prefill mechanism comparison only. No full262144fill/RESTORE,task/agent quality,sustainedTG,ordinary-provider teardown,hardware saturation,default adoption or whole-goal completion. Two lengths provide local slopes only.',
        retention='Keep36compact samples and paired decisions;retire successful raw logs/journals/cache after verification. No new engine generation. At matched phase/decode decision select current9af or56cad9 and retire the superseded generation after exact-source recovery;resolve the separate two-QSA-fixture proof hold before deleting those bytes.')
    ctx.save(row);print('Prepared12process same-binary QSA comparison;commit before host.',flush=True)


if __name__=='__main__':
    require(sys.executable==model.PYTHON,'Use pinned benchmark interpreter:'+model.PYTHON)
    require(len(sys.argv)==2 and (sys.argv[1] in ('prepare','host')
        or sys.argv[1].isdigit() and int(sys.argv[1])<len(CASES)),'prepare|host|INDEX0..11')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model.original.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='host':original.comparison.host(context())
        else:original.comparison.run(int(sys.argv[1]),context())
