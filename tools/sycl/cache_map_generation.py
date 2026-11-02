#!/usr/bin/env python3
"""Corrected-engine answer-room feasibility probe; repository LICENSE.

Use bench.sh's existing32K output option in a private policy context. Earlier
16K comparisons remain failed. Only total output room changes;thinking,sampling,
native mathematics and cache8placement stay fixed. This is not a speed comparison.
Main owns all serialized host/model execution.
"""
import fcntl
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import cache_map_model_entry as entry
import pool_spin as maintained

base, P, B, RESULT = entry.base, entry.P, entry.B, entry.RESULT
require, digest = base.require, base.digest
SOURCE = 'tools/sycl/cache_map_generation.py'
ROOT = entry.model.ROOT / 'generation-room-v1'
KEY = 'cache_map_generation_room_v1'
OLD_KEY = 'adaptive_cache_generation_speed_v2'
FAILED_KEY = 'cache_map_generation_speed_v1'
FAILED_ROOT = entry.model.ROOT / 'generation-speed-v1'
MAXIMUM_OUTPUT = 32768
WORKLOADS = ('regular-code', 'varied-prose')


def pinned_sources(revision):
    pins = entry.pinned_sources(revision)
    pins.update(base.controls(revision, controller=Path(__file__).resolve()))
    return pins


def basis(revision):
    prior_ctx = entry.context('positive')
    prior = json.loads((prior_ctx.ROOT / 'record.json').read_text())
    providers, manifest = prior_ctx.current(prior)
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete'] and
            prior['ownership_closed'] and not prior['active'] and prior['host_passed'] and
            prior['outcome']['native_exit_code'] == 0 and not prior['outcome']['fresh_faults'] and
            len(prior['outcome']['samples']) == len(prior['outcome']['routes']) == 2 and
            prior['outcome']['adaptive_work']['deferred_next_long_prefill_boundaries'] > 0 and
            prior['environment']['STRATA_SYCL_ADAPT_SYNC'] == '1',
            'closed current-engine positive-swap next-GEN model qualification required')
    zero_ctx = entry.context('zero')
    zero = json.loads((zero_ctx.ROOT / 'record.json').read_text())
    zero_ctx.current(zero)
    require(zero['stage'] == 'closed' and zero['passed'] and zero['complete'] and
            zero['ownership_closed'] and not zero['active'] and
            zero['binary'] == prior['binary'] and zero['outcome']['native_exit_code'] == 0 and
            not zero['outcome']['fresh_faults'], 'same-engine zero-swap model qualification required')
    original = {name: hashlib.sha256(raw).hexdigest()
                for name, raw in prior_ctx.shared.gen.blobs(P, prior['source_commit'], prior['sources'])}
    current = pinned_sources(revision)
    require(original == prior['sources'] and
            all(current.get(name) == sha for name, sha in original.items()) and
            set(current) - set(original) == {SOURCE},
            'only the new adapter may extend unchanged model/controller/workload inputs')
    failed = base.committed(RESULT)[OLD_KEY]
    require(failed['stage'] == 'failed' and not failed['passed'] and not failed['active'] and
            failed['ownership_closed'] and len(failed['processes']) == 2 and
            failed['binary'] != prior['binary'],
            'preserve the older engine failed cohort;never resume or relabel it')
    recent = json.loads((FAILED_ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[FAILED_KEY] == maintained.compact(recent) and
            recent['stage'] == 'failed' and not recent['passed'] and not recent['active'] and
            recent['ownership_closed'] and len(recent['processes']) == 2 and
            recent['binary'] == prior['binary'] and
            recent['failure_diagnosis']['observed_native_exit_code'] == 0 and
            recent['failure_diagnosis']['observed_benchmark_exit_code'] == 1 and
            recent['processes'][1]['samples'][1]['generated_tokens'] == 16384 and
            recent['processes'][1]['samples'][1]['output_chars'] == 0,
            'closed corrected-engine thinking-only failure must remain failed')
    recovered = {name: hashlib.sha256(raw).hexdigest()
                 for name, raw in maintained.shared.gen.blobs(P, recent['source_commit'], recent['sources'])}
    require(recovered == recent['sources'] and set(current) == set(recovered) and
            {name for name in current if current[name] != recovered[name]} == {SOURCE},
            'only this policy adapter may differ from the preserved16K comparison')
    native = entry.model.verify()
    names = ('src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/pool.hpp')
    pool = {name: native['native_sources'][name] for name in names}
    require(all(digest(entry.model.ROOT / 'source' / name) == sha for name, sha in pool.items()),
            'compiled pool inputs differ')
    require(prior['binary'] == native['binaries']['strata'] and
            prior['native_source_commit'] == native['source_commit'], 'current engine differs')
    return prior, providers, manifest, pool


def generation_contract(suite, rows):
    require(suite.get('workload_version') == 5 and suite.get('thinking') is True and
            suite.get('reasoning_budget_tokens') == 0 and suite.get('temperature') == 0 and
            suite.get('max_tokens') == MAXIMUM_OUTPUT and suite.get('diagnostic_subset') is False and
            suite.get('selected_workloads') == list(WORKLOADS),
            'unexpected32K output/thinking/sampling/workload policy')
    require(len(rows) == 2 and
            [(r.get('workload'), r.get('run'), r.get('order')) for r in rows] ==
            [(name, 0, i) for i, name in enumerate(WORKLOADS)] and
            all(r.get('controller_eligible') is True for r in rows),
            'incomplete/ineligible answer-room requests')


def decision(processes):
    require(len(processes) == 1 and processes[0]['index'] == 0 and
            processes[0]['kernel'] == 'candidate' and processes[0]['adapt_swaps'] == 8 and
            processes[0]['passed'] and processes[0]['complete'] and processes[0]['ownership_closed'] and
            processes[0]['native_exit_code'] == 0 and len(processes[0]['samples']) == 2 and
            all(s['controller_eligible'] for s in processes[0]['samples']),
            'one closed eligible cache8answer-room process required')
    code, prose = processes[0]['samples']
    return dict(answer_room_feasible=True, maximum_output=MAXIMUM_OUTPUT,
        observed_code_strong_MTP=code['acceptance_rate'] >= .60,
        observed_prose_weak_MTP=prose['acceptance_rate'] <= .40,
        repeated_MTP_class_qualified=False, repeated_gain_qualified=False,
        inference_adopted=False, physical_resource_saturation_qualified=False,
        limits='One process,two answer-bearing requests with thinking. Larger output allowance is not a matched speed gain or causal diagnosis. Input/history/temperature/uncapped thinking/cache8/native remain fixed;no budget-wrap GEN or sampler change.')


def selfcheck(ctx):
    legacy = maintained.selfcheck()
    row = dict(eligible=True, speed_eligible=True, ok=True, stream_complete=True,
        finish_reason='stop', reused_tokens=0, prompt_tokens=33792, read_tokens=33792,
        usage_generated_tokens=20000, generated_tokens=20000, generated_tokens_source='usage',
        output_chars=2000, reasoning_chars=1000, completion_includes_thinking=True,
        drafts_offered=1000, drafts_accepted=600, acceptance_rate=.6,
        decode_tps_source='engine', decode_ms=2000000, decode_tps=10., total_s=2100., run=0, order=0)
    require(ctx.shared is not maintained.shared and ctx.shared.gen is not maintained.shared.gen and
            not maintained.shared.gen.sample_eligible(row) and ctx.shared.gen.sample_eligible(row),
            'private32K collector must not change imported16K defaults')
    invalid = [dict(row, **patch) for patch in (
        dict(usage_generated_tokens=32769, generated_tokens=32769), dict(output_chars=0),
        dict(reasoning_chars=0), dict(stream_complete=False))]
    require(all(not ctx.shared.gen.sample_eligible(r) for r in invalid),
            'larger allowance must preserve answer/extent/completion gates')
    suite = dict(workload_version=5, thinking=True, reasoning_budget_tokens=0, temperature=0,
        max_tokens=MAXIMUM_OUTPUT, diagnostic_subset=False, selected_workloads=list(WORKLOADS),
        workloads={name: dict(requests=[dict(row, order=i)]) for i, name in enumerate(WORKLOADS)})
    rows = ctx.shared.gen.collect_samples(suite)
    generation_contract(suite, rows)
    wrong = [dict(suite, **patch) for patch in (
        dict(max_tokens=16384), dict(thinking=False), dict(reasoning_budget_tokens=12288),
        dict(temperature=1), dict(diagnostic_subset=True), dict(selected_workloads=[]))]
    for value in wrong:
        try:
            generation_contract(value, rows)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('changed room-probe contract accepted')
    events = []
    requests = [('warm-up/chat', None, 24, True), ('calibration', 200, 1, False),
        ('calibration', 800, 1, False), ('warm-up/prompt', None, 8, True)]
    requests += [('long-generation/' + name, 0, MAXIMUM_OUTPUT, True) for name in WORKLOADS]
    for i, (phase, run, maximum, thinking) in enumerate(requests):
        identity = dict(phase=phase, run=run, turn=None)
        sha = f'{i+1:064x}'
        events += [dict(identity, kind='start', index=i, max_tokens=maximum,
                        thinking=thinking, request_sha256=sha),
                   dict(identity, kind='result', request_index=i,
                        sample=dict(ok=True, request_sha256=sha))]
    require(ctx.shared.gen.warmup_reconciliation(events, WORKLOADS)['successful_long_requests'] == 2,
            '32K journal must bind both long requests and unchanged warmups')
    try:
        maintained.shared.gen.warmup_reconciliation(events, WORKLOADS)
    except RuntimeError:
        pass
    else:
        raise RuntimeError('imported16K journal gate was changed')
    return dict(passed=True, legacy_controls=legacy, private_default_isolation=True,
        eligible20000_token_control=True, invalid_extended_samples_rejected=len(invalid),
        invalid_room_contracts_rejected=len(wrong), extended_journal_binding=True, GPU_executed=False)


def context():
    spec = importlib.util.spec_from_file_location('strata_cache_map_generation_context', maintained.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.select_mode('cohort')
    ctx.ROOT, ctx.KEY = ROOT, KEY
    ctx.ARMS = (('candidate', 8),)
    ctx.model = entry.context('positive')
    ctx.model.pinned_sources = pinned_sources
    ctx.basis = basis
    # Pure helper facades change defaults only inside this private policy.
    # The maintained entry/device controllers retain their original16K gates.
    ctx.shared = SimpleNamespace(**vars(ctx.shared))
    ctx.shared.gen = SimpleNamespace(**vars(ctx.shared.gen))
    for name in ('sample_eligible', 'collect_samples', 'warmup_reconciliation'):
        setattr(ctx.shared.gen, name, partial(getattr(maintained.shared.gen, name), maximum_output=MAXIMUM_OUTPUT))
    ctx.generation_contract, ctx.decision = generation_contract, decision
    original_official = ctx.official
    def official(root, result):
        require(result['params']['long_generation_max_tokens'] == MAXIMUM_OUTPUT and
                result['comparison_identity']['long_generation_max_tokens'] == MAXIMUM_OUTPUT,
                'frontend cap metadata differs from the committed room probe')
        return original_official(root, result)
    ctx.official = official
    ctx.selfcheck = lambda: selfcheck(ctx)
    return ctx


def prepare(ctx):
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    require(len(row['plans']) == 1 and ctx.environment(row, row['plans'][0])['STRATA_SYCL_ADAPT_SYNC'] == '1',
            'one immutable same-engine cache8process required')
    row['plans'][0]['command'] += ['--long-generation-max-tokens', str(MAXIMUM_OUTPUT)]
    row['whole_bytes'] = 96 << 20
    row['budgets'].update(wall=3600, cpu=21600)
    row['qualification_reuse'] = dict(
        zero_model_result_key=entry.context('zero').KEY,
        positive_model_result_key=entry.context('positive').KEY,
        worker_device_result_key=entry.device.KEY,
        source_review_scope=entry.SCOPE,
        unchanged_math_fixture_count=8,
        added_control_paths=[SOURCE],
        preserved_failed_cohort_keys=[OLD_KEY, FAILED_KEY],
        scope='Exact corrected worker/map engine,unchanged eight mathematical fixtures,qualified real-copy/error and zero/positive next-GEN lifecycle. Recover original273inputs of failed16Kcohort;only this adapter differs. No failed-cohort qualification is inherited.')
    row['plan'].update(
        question='With cache8fixed,does the existing32K total-output option permit both v5code and after-code prose to return thought plus an actual answer?',
        motivation='The corrected-engine16Kcandidate exits normally but prose exhausts all16384tokens in thought. Use the already-maintained output-cap option rather than adding a reasoning-budget STOP/wrap/re-GEN or changing sampling. This is the separate larger-total-room option in the reviewed sampling contract.',
        order='One fresh cache8process,four unchanged official warmups,regular-code then varied-prose. Keep the previous fixed workload ID and instructions. Both requests explicitly allow32768output tokens instead of16384.',
        constants='Exact corrected worker/map engine479a6fdac and8unchanged fixtures;same providers/safety/private preload,throughput BF16 HC/GDN16/MTP16,heads1,131072INT8resident,1256slots,5workers,spec4,chunk8192,no token graph,caches0,20ms spin,adapt8/every1,immutable ADAPT_SYNC1. Thinking=true,preserved thought,xhigh,reasoning_budget0,temperature0 remain. Only total output allowance changes;live config penalties remain unobserved.',
        acceptance='Both actual fresh inputs>=32768,1024..32768generated including nonempty thought,>=512answer chars,>=256offered drafts,engine clocks,official journal/body/source/native/provider/route/owner gates,normal exits0,quiet validation,no freshfault. Preserve first-long code and subsequent prose separately. Full15model hashes before and after this one-process probe.',
        decision='One eligible process establishes answer-room feasibility only. Record actual MTP acceptance;one sample does not establish strong/weak repeated classes. Do not compare different-output-budget TG as a speedup or adopt cache policy. If prose still has no answer,keep failure and choose a separate budget-wrap or sampling plan rather than resuming the failed16Kcohort.',
        stopping='Stop first source/owner/provider/clock/route/validation/fault/budget/sample failure. No retries,exclusions,speed stop,cap escalation or refreshed hashes. One native process,wall3600s,cpu21600s,104GiBRSS,16MiBtext/32MiBfile/64MiBtotal,96MiBwhole root.',
        retention='Keep compact source/config,individual eligible and failed samples,actual route/MTP/cache counts,fault and decision limits. Retire raw output/JIT after verified closure. No new native generation;previous failed receipts remain unchanged.')
    ctx.save(row)


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        maintained.build.guards.no_model_owners([])
        ctx = context()
        args = sys.argv[1:]
        if args == ['prepare']:
            prepare(ctx)
        elif args == ['host']:
            ctx.comparison.host(ctx)
        elif args == ['selfcheck']:
            print(json.dumps(ctx.selfcheck()))
        elif len(args) == 1 and args[0].isdigit() and int(args[0]) < len(ctx.ARMS):
            ctx.run(int(args[0]))
        else:
            raise RuntimeError('cache_map_generation.py prepare|host|selfcheck|0')
