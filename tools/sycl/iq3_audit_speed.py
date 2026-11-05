#!/usr/bin/env python3
"""Same-binary enabled-IQ3-audit effect on timed generation; repository LICENSE.

Main-authored from the owned v5 comparison and supervisor. This changes no
inference implementation. Audit-off timed dispatch is unobserved, and an
audit-on replay cannot supply its missing telemetry. No adoption claim.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import iq3_generation_speed as gain
import pool_spin as protocol

P, B, RESULT, base = gain.P, gain.B, gain.RESULT, gain.base
ROOT = B / 'decode-iq3-audit-speed-v2'
KEY = 'decode_iq3_audit_speed_v2'
SELF = Path(__file__).resolve()
# These later tools were absent from the closed comparison's source snapshot.
# Revalidate that snapshot first; pin the exact additions in the new caller.
ADDITIONS = {'tools/sycl/decode_service_contract.py',
    'tools/sycl/iq3_audit_speed.py', 'tools/sycl/mapped_row_copy_component.py',
    'tools/sycl/profile_iq3_decode.py'}
# Reference is audit-on; candidate is audit-off. Both use IQ3_PAIR=1.
ARMS = gain.ARMS
require, digest = gain.require, gain.digest


def pinned_sources(revision):
    pins = gain.pinned_sources(revision)
    pins.update(base.controls(revision, controller=SELF))
    return pins


def basis(revision):
    cohort = json.loads((gain.ROOT / 'record.json').read_text())
    data = base.committed(RESULT)
    failed = data['decode_iq3_audit_speed_v1']
    require(failed['stage'] == 'failed' and not failed['passed']
        and not failed['model_executed'] and not failed['GPU_executed']
        and not failed['active'] and failed['ownership_closed']
        and not failed['root_created'], 'preserve original V1 prelaunch failure')
    require(data[gain.KEY] == gain.context().compact(cohort)
        and cohort['stage'] == 'closed' and cohort['passed'] and cohort['complete']
        and cohort['ownership_closed'] and not cohort['active']
        and len(cohort['processes']) == 6
        and all(p['passed'] and p['complete'] and p['ownership_closed']
            and p['native_exit_code'] == 0 and not p['fresh_faults']
            for p in cohort['processes']), 'closed route-verified IQ3 cohort required')
    # Never label the old qualification with the new caller revision. Its
    # original inputs still have to match Git and the current files exactly.
    _, providers, manifest, pool = gain.basis(cohort['source_commit'])
    current = pinned_sources(revision)
    previous = gain.pinned_sources(cohort['source_commit'])
    require(previous == cohort['sources'] and set(current) - set(previous) == ADDITIONS
        and set(previous) <= set(current)
        and all(current[n] == h for n, h in previous.items()),
        'only the four declared later tools may extend unchanged qualified inputs')
    selected = gain.short.context(1)
    prior = json.loads((selected.ROOT / 'record.json').read_text())
    gain.short.contract(prior, 1)
    return prior, providers, manifest, pool


def audit_enabled(index):
    require(type(index) is int and 0 <= index < len(ARMS), 'actual process index required')
    return ARMS[index][0] == 'reference'


def context(index=None):
    spec = importlib.util.spec_from_file_location('strata_iq3_audit_context', protocol.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.select_mode('cohort')
    ctx.ROOT, ctx.KEY, ctx.ARMS = ROOT, KEY, ARMS
    ctx.model = gain.short.context(1)
    ctx.model.pinned_sources = pinned_sources
    ctx.basis = basis
    old_environment, old_route, old_decision = ctx.environment, ctx.route, ctx.decision
    ctx.boundary_checks = []

    def environment(row, plan):
        # Preserve the established clean constructor's rejection of unknown
        # observers, then add only this prospectively declared treatment.
        baseline = dict(row, environment_template=dict(row['environment_template'],
            STRATA_IQ3_PAIR_AUDIT='0'))
        env = old_environment(baseline, plan)
        env.update(STRATA_IQ3_PAIR='1',
            STRATA_IQ3_PAIR_AUDIT=str(int(plan['kernel'] == 'reference')),
            STRATA_IQ_SINGLE_EXACT='0', STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0',
            STRATA_SYCL_DECODE_Q6_COLUMNS='1', STRATA_SYCL_ADAPT_SYNC='1')
        require(env['EnableDirectSubmission'] == '0'
            and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
            and env['STRATA_POOL_SPIN_US'] == '20000' and plan['adapt_swaps'] == 0
            and 'STRATA_DECODE_TIMING' not in env,
            'preserve native safety,static cache and declared timing envelope')
        return env

    def invariant(row, plan):
        config = json.loads((Path(plan['root']) / 'config.json').read_text())
        require(config['args'] == row['configuration']['args'], 'CLI arguments must match')
        env = environment(row, plan)
        env.pop('STRATA_IQ3_PAIR_AUDIT')
        return hashlib.sha256(json.dumps(dict(config=config, environment=env), sort_keys=True)
            .replace(plan['root'], '<process>').encode()).hexdigest()

    def route(text, read):
        result = old_route(text, read)
        if audit_enabled(index):
            rows = gain.short.dispatches(text, 1, positive=True)
            require(len(rows) == 1 and rows[0]['request'] in (5, 6),
                'one positive audit-on long-request footer required')
            result.update(IQ3_dispatch=rows[0], timed_IQ3_route_telemetry_available=True)
        else:
            require('IQ3_GU_DELTA' not in text, 'disabled audit emitted a footer')
            result['timed_IQ3_route_telemetry_available'] = False
        return result

    def boundary(root, expected):
        raw, history = protocol.cache_boundary(root, expected)
        if audit_enabled(index):
            rows = gain.short.dispatches(raw.decode(), 1, positive=False)
            require([r['request'] for r in rows] == list(range(1, expected + 1))
                and all(r['calls'] > 0 and r['nt1_rows'] > 0 for r in rows[4:]),
                'audit-on request history differs')
        else:
            require(b'IQ3_GU_DELTA' not in raw, 'audit-off history contains telemetry')
        ctx.boundary_checks.append(expected)
        return raw, history

    def decision(processes):
        result = old_decision(processes)
        for name, value in result['workloads'].items():
            value['all_three_audit_off_faster'] = value.pop('reproducible_TG_gain')
            wanted = 'strong' if name == 'regular-code' else 'weak'
            value['expected_MTP_class_observed'] = all(
                a['observed_MTP_class'] == wanted for a in value['arms'].values())
            value['contrast'] = 'audit-off candidate / audit-on reference;fixed IQ3_PAIR1'
        result.update(enabled_audit_timed_speed_comparison_complete=True,
            audit_off_timed_IQ3_route_qualified=False,
            total_observer_wall_cost_qualified=False, cross_build_gap_causality_qualified=False,
            inference_adopted=False,
            next='Use paired samples to measure the enabled-audit effect on timed speed in b310 only. '
                'Request snapshots/footer lie outside decode_ms. Keep the earlier route-verified IQ3 '
                'comparison and faster05a9/a99 reference. No audit-off route/adoption,physical resource '
                'attribution or70TG claim;consider compiled disabled scaffolding only for a concrete remaining decision.')
        return result

    ctx.environment, ctx.invariant, ctx.route = environment, invariant, route
    ctx.cache_boundary, ctx.decision = boundary, decision
    return ctx


def prepare_plan(ctx):
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['qualification_reuse'] = dict(build_result_key='decode_iq3_pair_model_build_v1',
        model_result_keys=['decode_iq3_model_entry_arm0_v1', 'decode_iq3_model_entry_arm1_v1'],
        closed_route_comparison_key=gain.KEY, native_source_commit=row['native_source_commit'],
        engine_sha256=row['binary']['sha256'],
        original_caller_source_commit=json.loads((gain.ROOT / 'record.json').read_text())['source_commit'],
        declared_caller_additions=sorted(ADDITIONS),
        scope='Exact current native/fixtures and closed audit-on short/long admissions at their original caller revision. The new caller pins four later tools;all original sources remain unchanged. Preserve failed V1 preparation. Audit-off timed IQ3 route is unobserved;replay is not a substitute.')
    row['plan'] = dict(mode='cohort', workload_version=5,
        question='How much does enabling IQ3 dispatch audit change timed generation on the same b310 binary?',
        order='Six fresh processes AUDIT1,0,0,1,1,0;three counterbalanced adjacent pairs. Each four official warmups then fixed v5 regular-code and varied-prose. Commit each closed process before next;no retries.',
        constants='Same9922/b310 source/ELF/providers/safety/preload,IQ3_PAIR1,Q6columns1,IQ_SINGLE_EXACT0,static1256cache,5workers,spec4,8192chunk,20ms pool spin,131072INT8resident,throughput BF16/GDN16/MTP16. Only immutable IQ3_PAIR_AUDIT1/0 differs;no native build.',
        observers='Parameter validation retained. TRACE,CPU_SERVICE,CPU_WORK_CENSUS,DECODE_TIMING,draft/CPU attribution,full API trace and profiler disabled. CPU_SERVICE0 leaves compiled guards/object layout/request scaffolding;its cost is unmeasured. No process environment mutation.',
        acceptance='Frozen source/build/ELF/model/provider/body/ordinary native16-GDN-MTP-Q6/cache/journal/clock/validation/fault/owner gates. Every long request>=32768fresh input,>=1024generated including thinking,>=512answer characters and>=256drafts. Audit-on must bind all6ordered IQ3 IDs with positive long selected NT1 rows;Audit-off must emit no IQ3 footer and has unavailable timed IQ3 route telemetry. All4paused common-cache gates apply to both. Full15asset hashes before/after cohort.',
        decision='Keep all3audit-off/on exact TG ratios,geomean,arm median/min/max and individual PP/TTFT,count/body/output/reasoning hashes,MTP acceptance/class. Do not require identical text or a minimum percentage. Speed/class differences do not stop collection. This is an observer-cost diagnostic,not an audit-off route-qualified inference comparison;no adoption gate is passed by it.',
        interpretation='TG captures in-decode audit checks/counters and interactions. Request-begin snapshots precede d0;end snapshots/validation/footer follow decode_ms. It does not measure whole observer wall cost or causally explain the cross-build absolute gap. Preserve all original comparison outcomes.',
        stopping='Stop first source,owner,provider,budget,ordinary route,body/token,clock,journal,counter,validation or driver-fault failure. No retry,replacement,hash refresh or changed criteria.',
        retention='Keep useful samples,available audit-on counters,explicit unavailable audit-off telemetry and limits. Retire successful raw logs/journals/JIT after result verification;retain record/runtime/config/plans only while current admission consumes them. No new engine generation.')
    row['budgets'] = dict(wall=3600, cpu=43200, rss_cap=104 << 30,
        text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20)
    ctx.save(row)
    print('Prepared six same-binary audit-cost processes;commit before frontend checks.', flush=True)


def prepare(ctx):
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'fresh V2 preparation only')
    try:
        prepare_plan(ctx)
    except BaseException as error:
        record = ROOT / 'record.json'
        row = json.loads(record.read_text()) if record.exists() else dict(
            source_commit=protocol.shared.gen.git('rev-parse', 'HEAD').decode().strip(),
            preparation_only=True)
        row.update(stage='failed', passed=False, complete=False, active=False,
            ownership_closed=True, model_executed=False, GPU_executed=False,
            preparation_error=repr(error))
        ROOT.mkdir(mode=0o700, exist_ok=True)
        if 'sources' in row:
            ctx.save(row)
        else:
            ctx.write(record, row)
            data = json.loads(RESULT.read_text())
            data[KEY] = row
            ctx.write(RESULT, data)
        raise


def run(ctx, index):
    ctx.run(index)
    row = json.loads((ROOT / 'record.json').read_text())
    try:
        require(ctx.boundary_checks == [4, 5, 5, 6], 'all four common-cache boundary gates required')
        proc = row['processes'][index]
        enabled = audit_enabled(index)
        proc['IQ3_audit'] = dict(enabled=enabled, boundary_checks=ctx.boundary_checks,
            fixed_math_selector=1, timed_route_telemetry_available=enabled,
            limits='Audit-on telemetry applies only to this process;no replay qualifies audit-off timed dispatch.')
        require(all(r['timed_IQ3_route_telemetry_available'] is enabled for r in proc['routes']),
            'audit availability differs from actual timed treatment')
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, IQ3_audit_error=repr(error))
    ctx.save(row)
    require(row['stage'] != 'failed', row.get('IQ3_audit_error', 'audit-cost comparison failed'))


if __name__ == '__main__':
    args = sys.argv[1:]
    require(len(args) == 1 and (args[0] in ('prepare', 'host') or
        args[0].isdigit() and int(args[0]) < len(ARMS)), 'use prepare|host|INDEX0..5')
    index = int(args[0]) if args[0].isdigit() else None
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        ctx = context(index)
        if args[0] == 'prepare':
            prepare(ctx)
        elif args[0] == 'host':
            ctx.comparison.host(ctx)
        else:
            run(ctx, index)
