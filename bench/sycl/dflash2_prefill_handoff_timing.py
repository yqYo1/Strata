"""Bracket target batching and subsequent bookkeeping on the same DFlash2 build."""

from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_prefill_kv_stage_speed as speed

gate, bank, basis, require = speed.gate, speed.bank, speed.basis, speed.require
READ = gate.SPLIT_READ
PREVIOUS = 'dflash2_prefill_handoff_timing_v1'
FIRST_CLEAN = '4e68b1ac7'
KEY = 'dflash2_prefill_handoff_timing_v2'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ROLES = ('traced', 'clean')
BATCH = re.compile(r'^strata trace: read (\d+) tokens \(batched\) in (\d+(?:\.\d+)?) ms$', re.M)


def first_control():
    row = bank.previous.historic(FIRST_CLEAN, PREVIOUS)
    item, = row['processes']
    require(row['host_passed'] and row['ownership_closed'] and not row['active']
            and item['index'] == 0 and item['clean_speed'] and item['complete']
            and item['passed'] and item['ownership_closed'] and item['native_exit_code'] == 0
            and item['fresh_faults'] == [], 'recover the original clean independent process')
    return row, item


def baseline():
    parent, control = first_control()
    failed = basis.validation.committed(bank.RESULT, (PREVIOUS,))
    observed = failed['processes'][1]['gate_refusal']
    require(failed['stage'] == 'failed' and not failed['active'] and not failed['passed']
            and not failed['ownership_closed'] and failed['processes'][0] == control
            and observed['reason'] == "ValueError('clean request contains the diagnostic trace')"
            and not observed['controller_observation_complete'] and observed['direct_child_reaped']
            and observed['session_empty'] and observed['native_pid_gone'] and observed['owner_gone'],
            'preserve the rejected traced process and reconcile its termination')
    row = speed.baseline(READ)
    require(all(parent[k] == row[k] for k in ('binary', 'native_source_commit', 'driver', 'boot')),
            'clean control changed platform or native build')
    comparison = basis.validation.committed(bank.RESULT, (speed.SPLIT_KEY,))
    require(comparison['stage'] == 'closed' and comparison['complete'] and comparison['passed']
            and comparison['ownership_closed'] and not comparison['active']
            and all(p['passed'] and p['ownership_closed'] and not p['fresh_faults']
                    for p in comparison['processes'])
            and comparison['binary'] == row['binary']
            and comparison['native_source_commit'] == row['native_source_commit']
            and comparison['boot'] == row['boot'] and comparison['driver'] == row['driver'],
            'require the closed six-process same-build split comparison')
    processes = comparison['processes']
    require(tuple(p['arm'] for p in processes) == speed.ORDER and all(p['read'] == READ for p in processes),
            'split predecessor order or extent differs')
    decision = speed.decision(processes, READ)
    require(comparison['decision'] == decision, 'original comparison decision differs')
    chosen = 'retain' if decision['repeated_warm_gain'] else 'reload'
    reference = next(p for p in processes if p['arm'] == chosen)
    return dict(row, handoff_reference=dict(arm=chosen,
        timing_sha256=bank.signature(comparison),
        requests=[s['request_sha256'] for s in reference['samples']],
        routes_sha256=bank.signature(reference['routes'])))


def request_route(text, plan):
    times = list(BATCH.finditer(text))
    route_plan = dict(plan, arm=plan['kv_arm'])
    if plan['trace_progress']:
        traced = speed.KV.check_request(text, 0, READ - 1, 8192,
            reuse=plan['kv_arm'] == 'retain', tap_layers=gate.TAPPED_QSA_LAYERS)
        counters = ''.join(line for line in text.splitlines(keepends=True)
                           if not line.startswith('strata trace:'))
        route = speed.request_route(counters, route_plan)
        require(all(route['kv'][key] == traced[key] for key in ('reloaded_pages', 'reused_pages')),
                'traced KV stage and invariant counters disagree')
        require(len(times) == 1 and int(times[0][1]) == READ - 1
                and float(times[0][2]) > 0, 'incomplete exact target batch timing')
        refill = text.find('strata trace: prompt done (slots refilled)')
        activation = text.find('strata DFlash2: eager,')
        require(times[0].end() <= refill < activation, 'unexpected prefill handoff order')
        route['target_batch_ms'] = float(times[0][2])
    else:
        require(not times, 'target batch clock present in clean control')
        route = speed.request_route(text, route_plan)
    return route


def after_execution(outcome, row, plan):
    routes = []
    for route, sample in zip(outcome['routes'], outcome['samples'], strict=True):
        work = dict(route)
        batch = work.pop('target_batch_ms', None)
        if plan['trace_progress']:
            require(batch is not None and 0 < batch < sample['prompt_ms'],
                    'target batch interval exceeds the request prompt interval')
            sample['outside_batch_ms'] = round(sample['prompt_ms'] - batch, 3)
        routes.append(work)
    require(bank.signature(routes) == row['handoff_reference']['routes_sha256']
            and outcome['clean_speed'] == (not plan['trace_progress'])
            and outcome['native_identity'] not in [p['native_identity'] for p in row['processes']],
            'request has changed ownership or work outside the observer')


def decision(processes):
    traced, after = processes
    _, before = first_control()
    require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                 for p in (before, traced, after)}) == 3, 'three independent native processes required')
    samples = []
    for left, observed, right, route in zip(before['samples'], traced['samples'],
                                           after['samples'], traced['routes'], strict=True):
        delta = [observed['prompt_ms'] - s['prompt_ms'] for s in (left, right)]
        samples.append(dict(request_sha256=observed['request_sha256'],
            prompt_ms=[s['prompt_ms'] for s in (left, observed, right)],
            ttft_s=[s['ttft_s'] for s in (left, observed, right)],
            observed_batch_ms=route['target_batch_ms'],
            observed_outside_batch_ms=observed['outside_batch_ms'],
            observer_delta_ms=delta,
            outside_batch_exceeds_observed_delta=observed['outside_batch_ms'] > max(map(abs, delta))))
    return dict(first_request=samples[0], subsequent_requests=samples[1:],
                scope='Outside batch includes lending, refill, request bookkeeping and DFlash2 activation. '
                      'Report the observed total-time differences against both clean controls.')


def experiment():
    reference = speed.experiment(READ)
    ctx = speed.generation.experiment(synchronous_commit=True)
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ROLES
    ctx.baseline, ctx.configuration = baseline, speed.configuration
    ctx.command = lambda plan: speed.command(plan, READ)
    ctx.protocol = dict(order=list(ROLES), read=READ, requests_per_process=speed.RUNS,
        first_clean_evidence=FIRST_CLEAN, rejected_protocol=PREVIOUS,
        unit='Reuse the committed clean baseline. Collect the observed process and a final fresh clean control.',
        acceptance='Replay the same official bodies, selected KV policy, split geometry and draft work. '
                   'Record first use and subsequent samples, each clean prompt time, target batch time '
                   'and its remaining native request time. Compare with both clean controls.',
        stop='Stop on changed input, clock, owner, transfer route, validation, fault or budget. '
             'Keep this diagnostic evidence for the next optimization decision.')

    def pins(revision):
        path = bank.P / THIS
        require(path.read_bytes() == basis.validation.git('show', revision + ':' + THIS),
                'changed handoff timing controller')
        return reference.pins(revision) | {THIS: bank.sha(path)}

    def environment(row, plan):
        env = speed.environment(row, dict(plan, arm=row['handoff_reference']['arm']), READ)
        if ROLES[plan['index']] == 'traced':
            env['STRATA_TRACE'] = '1'
        return env

    old_save, old_current = ctx.save, ctx.current

    def save(row):
        if row['stage'] == 'prepared':
            row['request_hashes'] = {str(READ): row['handoff_reference']['requests']}
            for plan in row['plans']:
                plan.update(read=READ, heads=speed.HEADS, kv_arm=row['handoff_reference']['arm'],
                            trace_progress=ROLES[plan['index']] == 'traced')
        old_save(row)

    def current(row):
        require(all((plan['read'], plan['heads'], plan['kv_arm'], plan['trace_progress']) ==
                    (READ, speed.HEADS, row['handoff_reference']['arm'], ROLES[i] == 'traced')
                    for i, plan in enumerate(row['plans'])), 'frozen handoff bracket differs')
        normalized = []
        for plan in row['plans']:
            env = environment(row, plan)
            env.pop('STRATA_TRACE', None)
            normalized.append(bank.signature({key: value.replace(plan['root'], '<case>')
                                              for key, value in env.items()}))
        require(len(set(normalized)) == 1, 'effective environments differ outside tracing')
        return old_current(row)

    ctx.pins, ctx.environment, ctx.save, ctx.current = pins, environment, save, current
    runner = SimpleNamespace(ROOT=ROOT, ORDER=[(READ, speed.HEADS)] * len(ROLES),
        RUNS=speed.RUNS, current=current, environment=environment,
        environment_fingerprint=speed.generation.suite.signature, save=save, request_route=request_route,
        after_execution=after_execution, decision=decision, validate_model_text=speed.validate_model_text,
        verify_model_assets=speed.verify_model_assets)
    ctx.run = lambda index: speed.comparison.run(index, runner)
    return ctx


if __name__ == '__main__':
    speed.generation.clock.main(experiment())
