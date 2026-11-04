#!/usr/bin/env python3
"""Matched sustained Q6 decode comparison through bench.sh; repository LICENSE.

Reuse the owned long-generation supervisor in a private policy context. Keep
the same engine, model, cache, prefill and startup safety in all six processes.
Only the immutable decode-column selector changes. Main owns every execution.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import verify_window_model_entry as entry
import pool_spin as protocol

P, B, RESULT, base = entry.P, entry.B, entry.RESULT, entry.base
ROOT = entry.build.ROOT / 'generation-speed-v1'
KEY = 'verify_window_generation_speed_v1'
SELF = Path(__file__).resolve()
ARMS = (('reference', 0), ('candidate', 0), ('candidate', 0),
        ('reference', 0), ('reference', 0), ('candidate', 0))
require, digest = entry.require, entry.digest


def pinned_sources(revision):
    pins = entry.pinned_sources(revision)
    pins.update(base.controls(revision, controller=SELF))
    return pins


def basis(revision):
    ref = entry.context('reference')
    prior = json.loads((ref.ROOT / 'record.json').read_text())
    providers, manifest = ref.current(prior)
    col = entry.context('columns')
    other = json.loads((col.ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[col.KEY] == col.compact(other), 'committed column entry required')
    for row in (prior, other):
        require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
                row['ownership_closed'] and not row['active'] and row['host_passed'] and
                row['outcome']['native_exit_code'] == 0 and row['outcome']['fresh_faults'] == [] and
                len(row['outcome']['samples']) == len(row['outcome']['routes']) == 2,
                'both closed same-build32K model entries required')
    require(prior['binary'] == other['binary'] and
            prior['native_source_commit'] == other['native_source_commit'] and
            prior['sources'] == other['sources'] and prior['model_assets'] == other['model_assets'] and
            prior['configuration']['args'] == other['configuration']['args'] and
            digest(ref.ROOT / 'record.json') == other['matched_reference_plan']['record_sha256'],
            'matched model/source/recipe entry evidence differs')
    def environment(row):
        root = str(Path(row['configuration']['log']).parent)
        return {k: v.replace(root, '<entry>') for k, v in row['environment'].items()
                if k != 'STRATA_SYCL_DECODE_Q6_COLUMNS'}
    require(environment(prior) == environment(other) and
            [s['request_sha256'] for s in prior['outcome']['samples']] ==
            [s['request_sha256'] for s in other['outcome']['samples']],
            'entry startup/workload comparison changed')
    inv = other['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'] and
            json.loads(Path(inv['path']).read_text()) == providers, 'column-entry provider evidence differs')
    original = {name: hashlib.sha256(raw).hexdigest() for name, raw in
                protocol.shared.gen.blobs(P, prior['source_commit'], prior['sources'])}
    current = pinned_sources(revision)
    require(original == prior['sources'] and
            all(current.get(name) == sha for name, sha in original.items()) and
            set(current) - set(original) == {str(SELF.relative_to(P))},
            'only this new comparison policy may extend qualified model inputs')
    native = json.loads((entry.build.ROOT / 'record.json').read_text())
    names = ('src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/pool.hpp')
    pool = {name: native['native_sources'][name] for name in names}
    require(all(digest(entry.build.ROOT / 'source' / name) == sha for name, sha in pool.items()),
            'unchanged compiled pool required')
    return prior, providers, manifest, pool


def context(index=None):
    spec = importlib.util.spec_from_file_location('strata_q6_generation_policy', protocol.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.select_mode('cohort')
    ctx.ROOT, ctx.KEY, ctx.ARMS = ROOT, KEY, ARMS
    ctx.model = entry.context('reference')
    ctx.model.pinned_sources = pinned_sources
    ctx.basis = basis
    original_environment, original_route, original_decision = ctx.environment, ctx.route, ctx.decision
    original_checks = ctx.selfcheck

    def environment(row, plan):
        env = original_environment(row, plan)
        env['STRATA_SYCL_DECODE_Q6_COLUMNS'] = str(int(plan['kernel'] == 'candidate'))
        require(env['STRATA_SYCL_ADAPT_SYNC'] == '1' and
                env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and
                env['EnableDirectSubmission'] == '0' and
                env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0', 'runtime safety changed')
        return env

    def invariant(row, plan):
        config = json.loads((Path(plan['root']) / 'config.json').read_text())
        require(config['args'] == row['configuration']['args'] and plan['adapt_swaps'] == 0,
                'only the Q6 selector may vary;native arguments stay fixed')
        env = environment(row, plan)
        env.pop('STRATA_SYCL_DECODE_Q6_COLUMNS')
        value = json.dumps(dict(config=config, environment=env), sort_keys=True)
        return hashlib.sha256(value.replace(plan['root'], '<process>').encode()).hexdigest()

    def route(text, read):
        require(index is not None and 0 <= index < len(ARMS), 'actual process arm required')
        result = original_route(text, read)
        arm = 'columns' if ARMS[index][0] == 'candidate' else 'reference'
        result['Q6_captures'] = entry.captures(text, arm)
        return result

    def decision(processes):
        result = original_decision(processes)
        result['next'] = ('Judge Q6 selectors separately for both actual MTP classes;no minimum gain floor. '
            'Keep all pairs and unresolved variability. Dedicated64K/117K prefill and client TTFT,'
            'agent/task-quality,prompt-verify,physical262144/RESTORE and adoption remain separate.')
        return result

    def checks():
        result = original_checks()
        rows = [dict(kernel=arm, adapt_swaps=swaps, passed=True, complete=True,
            ownership_closed=True, native_identity=dict(pid=i+1, start_ticks=i+10),
            samples=[dict(generated_tokens=1024, decode_ms=999.9 if arm == 'candidate' else 1000.,
                prompt_ms=1000., ttft_s=1.1, acceptance_rate=.8 if name == 'regular-code' else .3)
                for name in ctx.WORKLOADS]) for i, (arm, swaps) in enumerate(ARMS)]
        good = decision(rows)
        require(all(v['reproducible_TG_gain'] for v in good['workloads'].values()),
                'small paired gains must remain eligible')
        slower = copy.deepcopy(rows)
        slower[1]['samples'][0]['decode_ms'] = 1000.1
        require(not decision(slower)['workloads']['regular-code']['reproducible_TG_gain'],
                'one slower pair must not disappear in the aggregate')
        duplicate = copy.deepcopy(rows)
        duplicate[1]['native_identity'] = duplicate[0]['native_identity']
        for invalid in (rows[:-1], duplicate):
            try:
                decision(invalid)
            except RuntimeError:
                continue
            raise RuntimeError('incomplete/reused-process cohort accepted')
        result.update(Q6_capture_parser=entry.parser_check(), paired_TG_controls=True,
                      small_gain_retained=True, slower_pair_retained=True, invalid_cohorts_rejected=2)
        return result

    ctx.environment, ctx.invariant, ctx.route, ctx.decision = environment, invariant, route, decision
    ctx.selfcheck = checks
    return ctx


def prepare(ctx):
    checked = ctx.selfcheck()
    ctx.selfcheck = lambda: checked
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['qualification_reuse'] = dict(build_result_key=entry.build.KEY,
        model_result_keys=['verify_window_model_entry_reference_v1', 'verify_window_model_entry_columns_v1'],
        native_source_commit=row['native_source_commit'], engine_sha256=row['binary']['sha256'],
        scope='Same fixed-Q6 engine and closed entry source/provider/phase checks. Only this Python comparison policy is new;no native rebuild or inherited sustained-speed claim.')
    row['plan'] = dict(mode='cohort', workload_version=5,
        question='Does decode-only Q6 column selection improve sustained generation in the same fixed engine?',
        order='Six fresh processes,R/C,C/R,R/C. Each four official warmups,then regular-code and varied-prose once. Same fixed v5 request bodies across all processes;code precedes prose.',
        constants='Same a99aa1 engine/source05a9,oneAPI2026.2/safety/preload,throughput BF16 HC,GDN16/MTP16,heads1,131072INT8resident context,1256static cache slots,5workers,spec4,8192chunk,short-read0,caches0,adapt0,20ms pool spin. Only immutable STRATA_SYCL_DECODE_Q6_COLUMNS0/1 changes;no process-environment mutation.',
        observers='Remove detailed trace,CPU census,decode/draft timing and other profile/audit observers. Keep parameter validation,private activation preload and existing sparse capture/native16/MTP headers. Request gates and identity/provider/cache checks occur outside reported clocks;no profiler.',
        acceptance='All6processes and12long requests must satisfy frozen source/build/model/provider/body/route/journal/clock/ownership checks,>=32768fresh input,>=1024generated including thinking,>=512answer characters,>=256drafts,normal native/benchmark0,quiet validation and no fresh fault. Each long request must expose decoded Q6 captures for its actual selector;any prompt capture must remain reference. Keep all15model hashes before/after cohort and quick identities throughout.',
        decision='Retain all3paired exact TG ratios/geometric mean and each arm median/min/max,PP/TTFT,lengths and acceptance. Reproducible TG gain requires all3ratios>1 per workload;no minimum gain floor. Code is MTP-strong only if every sample>=.60,prose weak only if every sample<=.40 per arm. Classification failure does not stop collection or erase samples. No adoption or target claim from a component or one process.',
        stopping='Stop first source/owner/provider/budget/route/clock/sample/validation/fault failure. No retry,replacement,exclusion,speed stop or hash refresh. Commit each closed process before the next.',
        retention='Keep compact individual samples,actual Q6/route/cache conservation,source/configuration and decisions. Retire successful logs/journals/JIT at each closed process after evidence/source verification. Retain global record/runtime and generated plans only while the comparison reader uses them;no new binary generation.')
    row.update(budgets=dict(wall=3600, cpu=43200, rss_cap=104 << 30,
                           text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20))
    ctx.save(row)
    print('Prepared six clean same-binary Q6 processes and12long-generation requests;commit before host probes.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and (sys.argv[1] in ('prepare', 'host') or
            sys.argv[1].isdigit() and int(sys.argv[1]) < len(ARMS)), 'use prepare|host|INDEX0..5')
    index = int(sys.argv[1]) if sys.argv[1].isdigit() else None
    ctx = context(index)
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare(ctx)
        elif sys.argv[1] == 'host':
            ctx.comparison.host(ctx)
        else:
            ctx.run(index)
