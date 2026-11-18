#!/usr/bin/env python3
"""Measure only the order of the saved-prefix MoE router on the admitted image.

Main-authored from the same-build request controller. Repository LICENSE.
"""
import copy
import fcntl
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_prefix_router as prefix

P, B, RESULT = prefix.P, prefix.B, prefix.RESULT
ROOT, KEY = B/'prefill-prefix-router-timing-v1', 'prefill_prefix_router_timing_v1'
THIS = 'tools/sycl/prefill_prefix_router_timing.py'
READ, RUNS = 65537, 3
ORDER = ('late', 'early', 'early', 'late', 'late', 'early')
basis, require, sha, signature = prefix.basis, prefix.require, prefix.sha, prefix.signature
EVIDENCE = '36460f9586df1081ee882db2ebdcc8954056f0ab'


def baseline():
    replay = basis.validation.committed(RESULT, ('prefill_prefix_router_replay_v1',))
    original = basis.validation.committed(RESULT, (prefix.KEY,))
    archive = json.loads(prefix.source.git(P,'show',EVIDENCE+':'+str(RESULT.relative_to(P))))
    retirement = replay.pop('artifact_retirement')
    require(replay == archive['prefill_prefix_router_replay_v1']
        and original == archive[prefix.KEY]
        and replay['stage']=='closed' and replay['ownership_closed'] and replay['passed']
        and replay['decision']['router_equivalence'] and replay['replayed_requests']==6
        and replay['model_assets_rehashed']==15
        and original['stage']=='failed' and original['ownership_closed'] and not original['passed'],
        'original aggregation failure and exact qualified replay must remain intact')
    require(retirement['complete'] and retirement['evidence_commit']==EVIDENCE
        and retirement['retained_build']['binary']==original['binary']==replay['binary']
        and not prefix.ROOT.exists(),'need reconciled original outputs and the same retained ELF')
    require(all(p['complete'] and p['ownership_closed'] and p['passed'] and not p['fresh_faults']
        and p['native_exit_code']==0 and p['binary']==original['binary'] for p in original['processes']),
        'earlier actual native request and fault closure required')
    native, prior, providers, unused = prefix.model_context().basis()
    require(native['binaries']['strata']==original['binary']
        and native['native_source_commit']==original['native_source_commit'],
        'the timing cohort must reuse the original tested image')
    prior['observer_admission'] = dict(prior['observer_admission'],
        numerical_evidence=dict(commit=EVIDENCE,replay_sha256=signature(replay),
            original_failed_aggregation=True,observed_equivalent_rows=32768,
            native_recompiled=False,retirement_sha256=signature(retirement)))
    return native,prior,providers,unused


def actual_route(text, plan):
    route = prefix.workspace.request_route(text,plan)
    layout=route['layout']
    mode=plan['variant']
    admissions=prefix.workspace.records(text,'strata MoE prefix router: ')
    completions=prefix.workspace.records(text,'strata MoE prefix router complete: ')
    require(admissions==[dict(requested=mode,effective=mode,
        workspace_bytes=str(prefix.PREFIX_BYTES),decode_changed='0')]
        and completions==[dict(effective=mode,layers='48',rows='8192',final_drain='checked')]
        and layout['fixed_including_headroom_bytes']+prefix.PREFIX_BYTES<=layout['free_gpu_bytes']
        and prefix.PREFIX_BYTES<=layout['allocation_limit_bytes'],
        'effective prefix producer, fixed geometry, coverage or capacity differs')
    require('strata MoE router census:' not in text and 'strata MoE first-route census:' not in text
        and 'strata MoE upload census:' not in text and route['permanent_hits']==0,
        'unexpected observer or changed demand upload path')
    route['prefix_router']=dict(mode=mode,layer_count=48,prefix_rows=8192,workspace_bytes=prefix.PREFIX_BYTES,
        final_drain='checked')
    route['MTP_expert_restore_bytes']=707788800  # Checked in the actual request log by request_route above.
    return route


def geometric_summary(ratios):
    return dict(paired=ratios,geometric=statistics.geometric_mean(ratios),range=[min(ratios),max(ratios)])


def decision(processes):
    pairs=[]
    for i in range(0,len(ORDER),2):
        matched={mode:processes[j] for j,mode in enumerate(ORDER[i:i+2],i)}
        late,early=matched['late'],matched['early']
        work=[]
        for process in (late,early):
            work.append([dict(body=s['request_sha256'],actual_read=s['actual_read_tokens'],
                actual_reuse=s['actual_reused_tokens'],actual_output=s['generated_tokens'],
                uploads=r['uploads'],bank_H2D=r['requested_bank_H2D_bytes'],
                MTP_restore=r['MTP_expert_restore_bytes'],job_counts=r['actual_GU_dispatch'])
                for s,r in zip(process['samples'],process['routes'],strict=True)])
        def timing(process):
            first,*warm=process['samples']
            return dict(first_prompt_ms=first['prompt_ms'],first_ttft_s=first['ttft_s'],
                warmed_prompt_ms=statistics.median(s['prompt_ms'] for s in warm),
                warmed_ttft_s=statistics.median(s['ttft_s'] for s in warm))
        late_time,early_time=timing(late),timing(early)
        pairs.append(dict(paired_indices=[i,i+1],work_matches=work[0]==work[1],
            same_binary=late['binary']==early['binary'],late=late_time,early=early_time,
            early_minus_late_warmed_prompt_ms=early_time['warmed_prompt_ms']-late_time['warmed_prompt_ms']))
    metrics=('first_prompt_ms','first_ttft_s','warmed_prompt_ms','warmed_ttft_s')
    ratios={k:geometric_summary([p['early'][k]/p['late'][k] for p in pairs]) for k in metrics}
    eligible=all(p['work_matches'] and p['same_binary'] for p in pairs)
    ratios_pp=ratios['warmed_prompt_ms']['paired']
    outcome='mixed'
    if all(r<1 for r in ratios_pp): outcome='early_faster_in_all_pairs'
    elif all(r>1 for r in ratios_pp): outcome='early_slower_in_all_pairs'
    return dict(pairs=pairs,ratios=ratios,matched_work=eligible,
        warmed_prompt_observation=outcome if eligible else 'changed_work',
        isolated_producer_order=True,model_adopted=False,transfer_overlap_measured=False)


def sample(sample,count,read,minimum_ms=10000):
    basis.comparison.check_sample(sample,count,read,minimum_ms)
    require(sample['generated_tokens']==24,'match the profiled prefill request extent')


def summary_control():
    # Use previously committed route and sample schemas to exercise finalization.
    original = basis.validation.committed(RESULT,(prefix.KEY,))
    late,early = original['processes'][1:]
    prototypes = {'early':early,'late':late}
    cases = [copy.deepcopy(prototypes[mode]) for mode in ORDER]
    for process in cases:
        for route in process['routes']:
            route['MTP_expert_restore_bytes']=707788800
    require(decision(cases)['matched_work'],'committed equivalent requests must pass')
    cases[1]['routes'][0]['requested_bank_H2D_bytes']+=1
    require(decision(cases)['warmed_prompt_observation']=='changed_work',
        'a changed payload must be ineligible for producer-order attribution')


def model_context():
    ctx=prefix.private_module(basis.core,'prefix_router_timing')
    ctx.ROOT,ctx.KEY,ctx.THIS=ROOT,KEY,THIS
    ctx.READS,ctx.ORDER,ctx.RUNS,ctx.CLEAN_FAST=(READ,)*len(ORDER),[(READ,1)]*len(ORDER),RUNS,True
    old_env,old_current=ctx.environment,ctx.current
    def environment(row,plan):
        env=old_env(row,plan)
        env.update({prefix.MODE:ORDER[plan['index']],
            'STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS':'512',
            'STRATA_SYCL_PREFILL_QSA_MATH':'bf16-prefix-ordered-exp',
            'STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR':'1',
            'STRATA_SYCL_PREFILL_GU_SWIGLU':'large',
            'STRATA_SYCL_PREFILL_MOE_CENSUS':'0',
            'STRATA_PREFILL_LAYER_MAJOR_PROFILE':'0',
            'STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING':'0',
            'STRATA_SYCL_PREFILL_FULLFIT_UPLOAD':'jit'})
        require('STRATA_TRACE' not in env and 'UR_LOG_TRACING' not in env,
                'clean order comparison requires the default quiet observer policy')
        return env
    def current(row):
        require([p['variant'] for p in row['plans']]==list(ORDER)
            and all(p['bank']==512 and p['window']==10 and p['upload']=='jit' and p['relocate']==0
                and not p['trace_progress'] and p['read']==READ for p in row['plans']),
            'frozen same-image timing plan differs')
        return old_current(row)
    ctx.basis,ctx.environment,ctx.current=baseline,environment,current
    ctx.request_route,ctx.decision,ctx.check_sample=actual_route,decision,sample
    return ctx


def prepare():
    summary_control()
    ctx=model_context();ctx.prepare()
    row=json.loads((ROOT/'record.json').read_text())
    environments=[]
    for plan,mode in zip(row['plans'],ORDER,strict=True):
        plan.update(variant=mode,bank=512,window=10,relocate=0,trace_progress=False,upload='jit')
        env=ctx.environment(row,plan)
        require(plan['environment_sha256']==basis.comparison.fingerprint(env),'prepared environment differs')
        environments.append({k:v.replace(plan['root'],'<case>') for k,v in env.items() if k!=prefix.MODE})
    require(all(e==environments[0] for e in environments),'another pre-exec setting changed across the timing arms')
    row['plan']=dict(unit='Keep the same saved prefix owner, split readback, tensors, JIT transfers and compute queue. Change only whether the prefix producer executes after all attention or after its first piece.',
        order='Late/early, early/late, late/early in six independent processes. Three matching 65537-input thinking-enabled 24-output requests per process. Separate first request and warmed pair median.',
        gate='Identical actual work per request and same executable. Compare all three paired warmed prompt ratios and TTFT, preserving first-use separately. Every eligible pair remains in the record. No minimum gain cutoff and no speed-based sample replacement.',
        limits='Same-build isolated producer order, no actual concurrent expert upload. Clean clocks with existing quiet validation and no timing/census observers. This does not cover the storage-only control, sustained decode, agent, quality or physical context.',
        retention='Keep individual samples, measured placement and matched comparison; retire captured output after closure. Review the additional build against the next named experiment.')
    ctx.save(row)


if __name__=='__main__':
    require(sys.executable==prefix.PYTHON and len(sys.argv)==2,'use pinned Python and one frozen action')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prefix.builder.guards.no_model_owners([])
        action=sys.argv[1]
        if action=='prepare': prepare()
        elif action=='host': basis.comparison.host(model_context())
        elif action in tuple(map(str,range(len(ORDER)))): basis.comparison.run(int(action),model_context())
        else: raise RuntimeError('unknown stage')
