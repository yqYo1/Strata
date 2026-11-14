#!/usr/bin/env python3
"""Compare eight versus32 adaptive replacements on the same owned engine.

Main-authored; repository LICENSE. Source preparation supplies no execution
qualification. Close both old cohorts,port/test the required frontend change,
then qualify real32-copy publication and the checked record refusal first.
"""
import copy
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import compare_adaptive_generation as clean
import check_adaptive_budget_failure as refusal

SELF = Path(__file__).resolve()
ARMS = (('reference',8),('candidate',32),('candidate',32),
        ('reference',8),('reference',8),('candidate',32))
FRONTEND_FILES = ('serve/frontend.py','tools/test_frontend_stream_scan.py',
                  'serve/test_streamed_tool_tags.py')
require = clean.require


def prerequisites(full=False):
    ctx = refusal.context()
    refusal.budget.prerequisite_rows(ctx)
    data = clean.base.committed(clean.RESULT)
    port = data['upstream_streamed_tool_tags_host_v2']
    require(port['stage']=='closed' and port['passed'] and port['complete'] and
            port['ownership_closed'] and not port['active'] and not port['GPU_executed'] and
            port['origin_commit']=='3bbf0f399e2a1b3d38642eca4342721fc7994201' and
            all(port['sources'][name]==clean.digest(clean.P/name) for name in FRONTEND_FILES) and
            len(port['tests'])>=3 and all(t['exit_code']==0 for t in port['tests']),
            'port and test the required streamed-tool behavior before another long cohort')
    for name in FRONTEND_FILES:
        raw = clean.base.basis.build.git(clean.P,'show',port['source_commit']+':'+name)
        require((clean.P/name).read_bytes()==raw,'recover exact tested frontend: '+name)
    positive = ctx.normal_entry()
    failed = json.loads((ctx.ROOT/'record.json').read_text())
    require(data[ctx.positive.KEY]==ctx.positive.compact(positive) and
            data[ctx.KEY]==ctx.compact(failed) and failed['stage']=='closed' and
            failed['passed'] and failed['complete'] and failed['ownership_closed'] and
            not failed['active'] and failed['outcome']['native_exit_code']==1 and
            failed['outcome']['adaptive']['first_swaps']==32 and
            failed['outcome']['adaptive']['queued_second_swaps']==32 and
            failed['outcome']['adaptive']['stale_event_not_published'],
            'closed full32-copy positive and checked second-record refusal required')
    ctx.positive.current(positive)
    ctx.current(failed)
    normal,prior,contract,manifest,providers,refs = clean.basis(full=full)
    require(positive['binary']==failed['binary']==normal['binary'] and
            positive['native_source_commit']==failed['native_source_commit']==normal['native_source_commit'],
            'same unchanged native engine required')
    refs.update({key:dict(path=str(root/'record.json'),sha256=clean.digest(root/'record.json'))
        for key,root in (('positive32',ctx.positive.ROOT),('record_failure32',ctx.ROOT))})
    refs['frontend']=dict(key='upstream_streamed_tool_tags_host_v2',sha256=clean.signature(port))
    return normal,prior,contract,manifest,providers,refs


def exact_decision(processes,printed):
    result = copy.deepcopy(printed)
    result['printed_TG_decision'] = printed
    ratios = []
    for i in (0,2,4):
        pair = {p['kernel']:p['samples'][0] for p in processes[i:i+2]}
        exact = lambda s:s['generated_tokens']*1000.0/s['decode_ms']
        require(all(s['decode_ms']>0 and s['generated_tokens']>0 for s in pair.values()),
                'positive measured generation extent')
        ratios.append(exact(pair['candidate'])/exact(pair['reference']))
    require(all(math.isfinite(r) and r>0 for r in ratios),'finite paired TG ratios')
    for arm in ('reference','candidate'):
        values = [p['samples'][0]['generated_tokens']*1000.0/p['samples'][0]['decode_ms']
                  for p in processes if p['kernel']==arm]
        result['arms'][arm].update(median_TG=statistics.median(values),minimum_TG=min(values),
                                  maximum_TG=max(values))
    result.update(candidate_over_reference_TG=ratios,
        geometric_mean_TG_ratio=math.exp(statistics.mean(map(math.log,ratios))),
        repeated_TG_gain=all(r>1 for r in ratios),
        TG_expression='generated_tokens*1000.0/decode_ms',
        limits='Same-ELF adaptive8/32 focused comparison only. Exact TG is primary;printed decision retained. Judge prompt/TTFT independently. No other length,quality,agent,physical262144/RESTORE,ordinary teardown,adoption or70TG qualification.')
    return result


def controls():
    rows = [dict(kernel=arm,adapt_swaps=swaps,passed=True,complete=True,ownership_closed=True,
        samples=[dict(decode_tps=10.0,prompt_ms=1000.0,ttft_s=1.1,acceptance_rate=.7,
            generated_tokens=2048,decode_ms=204780.0 if arm=='candidate' else 204800.0)])
        for arm,swaps in ARMS]
    ctx = context('regular-code')
    result = ctx.decision(rows)
    require(result['repeated_TG_gain'] and not result['printed_TG_decision']['repeated_TG_gain'],
            'prospective exact arithmetic must retain a small gain hidden by printing')
    changed = copy.deepcopy(rows)
    changed[1]['samples'][0]['decode_ms']=204801.0
    require(not ctx.decision(changed)['repeated_TG_gain'],'retain one slower exact pair')
    changed = copy.deepcopy(rows)
    for proc in changed:
        if proc['kernel']=='candidate':
            proc['samples'][0].update(prompt_ms=1010.0,ttft_s=1.2)
    phases = ctx.decision(changed)
    require(phases['repeated_TG_gain'] and not phases['prompt_nonregression'] and
            not phases['TTFT_nonregression'],'TG gain must not conceal slower prefill/TTFT')
    for workload in clean.CAPS:
        ctx = context(workload)
        row = json.loads((clean.B/('adaptive-generation-'+workload+'-v2')/'record.json').read_text())
        plans = [dict(index=i,kernel=arm,adapt_swaps=swaps,heads=1,
            root=str(ctx.ROOT/('control-'+str(i)))) for i,(arm,swaps) in enumerate(ARMS[:2])]
        require(ctx.invariant(row,plans[0])==ctx.invariant(row,plans[1]),'only budget/private paths differ')
        for plan in plans:
            args = ctx.configuration(row,plan)['args']
            require(args[args.index('--adapt-swaps')+1]==str(plan['adapt_swaps']) and
                    args[args.index('--adapt-every')+1]=='8' and
                    args[args.index('--mtp-max-t')+1]==str(clean.CAPS[workload]),'actual budget/width selectors')
    return dict(passed=True,small_exact_gain_retained=True,slower_pair_retained=True,phase_decisions_separate=True,
                immutable_configuration_controls=4,GPU_executed=False)


def context(workload):
    spec = importlib.util.spec_from_file_location('strata_adaptive_budget_generation',clean.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ARMS,module.SELF,module.basis,module.controls = ARMS,SELF,prerequisites,controls
    printed = module.decision
    module.decision = lambda rows:exact_decision(rows,printed(rows))
    ctx = module.context(workload)
    ctx.ROOT = clean.B/('adaptive-budget-generation-'+workload+'-v1')
    ctx.KEY = 'adaptive_budget_generation_'+workload.replace('-','_')+'_v1'
    ctx.WORKLOAD_ID = 'adaptive_budget_generation_20261107_'+workload
    ctx.configuration,ctx.invariant = module.configuration,module.invariant
    ctx.prepare_budget = lambda:prepare(module,ctx)
    return ctx


def prepare(module,ctx):
    module.prepare(ctx)
    row = json.loads((ctx.ROOT/'record.json').read_text())
    row['plan'] = ('Six fresh owned processes adaptive8/32,32/8,8/32 on unchanged429b;every8rounds in both. '
        'Four official warmups then one focused v5 thinking-on32K+/1K+ generated-token request with an answer. '
        'Code cap4/prose cap2 in separate cohorts;16K output allowance. All placement,precision,providers,runtime safety '
        'and clean observers match the earlier plan. Require the actual32 positive publication/dispatch/loan and '
        'second-record refusal first,plus the required upstream frontend functional port and fresh frontend probes. '
        'Keep original body/native-width/head/clock/journal/validation/ownership/no-fault gates and all15asset hashes. '
        'Three exact paired TG ratios>1 establish focused repeatability without a minimum gain floor;retain the '
        'printed decision separately. Prompt and client TTFT are independent decisions. First long request after '
        'four warmups is the measured history;do not call it engine first use or a subsequent-long-request result. '
        'Stop first safety/ownership/eligibility/budget failure;no retries,exclusions or hash refresh. '
        'Commit each closed process before the next;no new engine generation or inference adoption.')
    ctx.save(row)


if __name__=='__main__':
    args = sys.argv[1:]
    require(args==['controls'] or len(args)==2 and args[0] in clean.CAPS and
        (args[1] in ('prepare','host') or args[1].isdigit() and int(args[1])<6),
        'use controls | regular-code|varied-prose prepare|host|INDEX0..5')
    with (clean.B/'owned-v0141-measurement.lock').open('a+') as lock:
        clean.positive.fcntl.flock(lock,clean.positive.fcntl.LOCK_EX|clean.positive.fcntl.LOCK_NB)
        clean.base.guards.no_model_owners([])
        if args==['controls']:print(json.dumps(controls(),sort_keys=True))
        else:
            ctx = context(args[0])
            if args[1]=='prepare':ctx.prepare_budget()
            elif args[1]=='host':ctx.comparison.host(ctx)
            else:ctx.run(int(args[1]))
