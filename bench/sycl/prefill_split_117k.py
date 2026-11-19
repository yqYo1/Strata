"""Revalidate the owned 117489-token workload with the full configured context."""
import fcntl
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'tools/sycl'))
import prefill_resident_input_timing as bank
import prefill_workspace_admission as workspace

model, build, basis, require = bank.model, bank.build, bank.basis, bank.require
READ, ROWS = 117489, 117488
HISTORY = '792ec1ee5748647252c7d1397e9ef8afec17b1c6'
KEY = 'prefill_split_117k_v1'
ROOT = model.B / 'prefill-split-117k-v1'
THIS = str(Path(__file__).resolve().relative_to(PROJECT))


def admission():
    native, prior, providers, _ = bank.admission()
    control = basis.validation.committed(model.RESULT, ('prefill_resident_input_timing_v1',))
    require(control['stage'] == 'closed' and control['complete'] and control['passed']
            and control['ownership_closed'] and not control['active']
            and control['binary'] == native['binaries']['strata']
            and len(control['processes']) == 16 and all(p['passed'] and p['complete']
                and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
                for p in control['processes']), 'finish the same-engine 64K comparison first')
    historic = build.previous.historic(HISTORY, 'native_prefill_preservation_v2')
    hashes = historic['request_hashes'][str(READ)]
    require(historic['passed'] and historic['complete'] and len(hashes) == len(set(hashes)) == 4,
            'the four original 117489-token request hashes are required')
    prior['observer_admission'].update(historical_requests=hashes, historical_evidence=HISTORY,
        copy_control_sha256=build.signature(control['processes'][0]))
    return native, prior, providers, None


def check_split_budget(text):
    split = workspace.one(text, 'strata MoE split admission: ')
    accum = workspace.one(text, 'strata MoE accumulator admission: ')
    actual = {k: int(split[k]) for k in ('active', 'rows', 'extra_allocation_bytes', 'prefill_slots')}
    actual.update({k: int(accum[k]) for k in
                   ('gpu_prefix_rows', 'free_gpu_bytes', 'fixed_including_headroom_bytes', 'shared_arena_bytes')})
    require(actual['active'] == 1 and actual['rows'] == actual['gpu_prefix_rows'] == ROWS
            and actual['prefill_slots'] == 128 and actual['extra_allocation_bytes'] == 531988736
            and actual['free_gpu_bytes'] >= actual['fixed_including_headroom_bytes'] == 4057356672
            and int(split['max_single_allocation_bytes']) >= ROWS * 20480,
            f'117K split or memory admission differed: {actual}')
    mixed = workspace.one(text, 'strata MoE mixed banks: ')
    require(mixed['storage'] == 'scratch' and mixed['phase'] == 'FFN_only_after_attention_drain'
            and int(mixed['primary_bytes']) == 65536 * 5120
            and int(mixed['tail_offset']) == 897762560 and int(mixed['tail_bytes']) == (ROWS - 65536) * 5120
            and int(mixed['extra_allocation_bytes']) == 0
            and int(mixed['tail_offset']) + int(mixed['tail_bytes']) <= actual['shared_arena_bytes'],
            f'117K mixed storage differed: {mixed}')
    return actual


def check_work(outcome, row, plan):
    require([s['request_sha256'] for s in outcome['samples']] == row['observer_admission']['historical_requests'],
            'actual 117489-token request bodies differ from the historical four')
    bank.ordinary.check_work(outcome, row, plan)
    if plan['index'] == 0:
        history = build.previous.historic(HISTORY, 'native_prefill_preservation_v2')
        outcome['historical_work_delta'] = historical_work_delta(outcome, history)


def historical_work_delta(outcome, history):
    # These counters have the same producer on both builds. The draft-head
    # restore policy changed between builds and is deliberately outside this comparison.
    counter_names = {
        'jobs': 'jobs', 'fused_down_jobs': 'fused_down_jobs',
        'requested_bank_H2D_bytes': 'requested_bank_H2D_bytes',
        'requested_one_routing_map_H2D_bytes': 'requested_dequant_H2D_bytes',
    }
    hashes = [s['request_sha256'] for s in outcome['samples']]
    refs = [p for p in history['processes'] if p['read'] == READ and p['kernel'] == 'reference']
    draft_counts = ('drafts_offered', 'drafts_accepted')
    require(len(refs) == 3 and len(outcome['routes']) == len(hashes) == len(set(hashes)) == 4 and all(
        p['passed'] and p['complete'] and p['route_shape']['split_guard_checked']
        and p['route_shape']['prefill_slots'] == 128
        and [s['request_sha256'] for s in p['samples']] == hashes
        and all(s['actual_read_tokens'] == READ and s['actual_reused_tokens'] == 0
                and s['generated_tokens'] == 24 and all(s[k] == baseline[k] for k in draft_counts)
                for s, baseline in zip(p['samples'], refs[0]['samples'], strict=True))
        and p['request_route_checks'] == refs[0]['request_route_checks'] for p in refs),
        'historical 117K work must agree across the three reference processes')
    deltas = []
    for route, prior, sample, baseline in zip(outcome['routes'], refs[0]['request_route_checks'],
                                              outcome['samples'], refs[0]['samples'], strict=True):
        delta = {name: route['fast_window'][current] - prior[name] for name, current in counter_names.items()}
        delta.update({k: sample[k] - baseline[k] for k in draft_counts})
        deltas.append(delta)
    return deltas


def decision(processes):
    require(len(processes) == 4 and all(p['index'] == i and p['read'] == READ and p['passed']
            and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
            and p['clean_speed'] == (i > 0) and len(p['samples']) == 4 for i, p in enumerate(processes))
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == 4,
            'one traced and three independent clean processes are required')
    times = [bank.ordinary.store.stage.schedule.timing(p) for p in processes[1:]]
    pp = [1000 * READ / sample['warm_prompt_ms'] for sample in times]
    return dict(clean_times=times, warmed_process_PP=pp, median_PP=statistics.median(pp),
                repeated_1000=all(rate >= 1000 for rate in pp),
                historical_expert_and_MTP_counts_match=all(
                    value == 0 for request in processes[0]['historical_work_delta'] for value in request.values()),
                sustained_decode_qualified=False, physical_final_slot_qualified=False)


def experiment():
    ctx = basis.core.source.device.load_module('split_117k_benchmark', basis.core.__file__)
    reference = basis.reference.context()
    ctx.THIS, ctx.ROOT, ctx.KEY = THIS, ROOT, KEY
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (READ,) * 4, [(READ, 1)] * 4, 4, True
    ctx.basis, ctx.check_sample, ctx.decision = admission, reference.check_sample, decision
    ctx.after_execution = check_work
    base_save, base_current = ctx.save, ctx.current

    def environment(row, plan):
        env = model.ordinary_environment(row, plan['root'], row['observer_admission']['ordinary_control'])
        env[model.INPUT] = env[model.OUTPUT] = '0'
        if plan['index'] == 0:
            env['STRATA_TRACE'] = '1'
        return env

    def route(text, plan):
        budget = check_split_budget(text)
        value = reference.request_route(text, plan)
        store = model.restore.require_store(text, dict(direct=False), plan)
        require(store['maximum_rows'] == store['gpu_rows'] == ROWS and store['host_rows'] == 0,
                'long prompt did not run in one GPU window per layer')
        value['attention_store'] = store
        value['attention_input'] = model.input_receipt(text, ROWS, store, False, trace_layers=plan['index'] == 0)
        value['admission'] = budget
        return value

    def save(row):
        if row['stage'] == 'prepared':
            for i, plan in enumerate(row['plans']):
                plan['trace_progress'] = i == 0
            row['plan'] = dict(order='One traced process then three independent clean processes; four historical '
                               'thinking-on requests per process. Gate the actual split at each request.',
                               scope='At 117489 tokens with the current 262144 context and 20480 resident KV, '
                               'keep both BF16 bank controls off. Require the 128-slot guarded split, exact requests, '
                               'MoE and MTP work, normal exits, unchanged driver and final full model hashes.',
                               acceptance='First-use and subsequent-three medians separately. All three clean warmed '
                               'process rates must reach 1000 PP. No speed-based exclusion or retry.')
        base_save(row)

    def current(row):
        args = row['configuration']['args']
        previous = basis.validation.committed(model.RESULT, ('prefill_resident_input_timing_v1',))
        require(row['configuration'] == previous['configuration']
                and args[args.index('--max-context') + 1] == '262144'
                and args[args.index('--kv-resident') + 1] == '20480'
                and [(p['index'], p['read'], p['trace_progress']) for p in row['plans']]
                    == [(i, READ, i == 0) for i in range(4)]
                and THIS in row['sources'] and bank.THIS in row['sources'],
                'the current 64K configuration, controller and frozen run order are required')
        return base_current(row)

    ctx.environment, ctx.request_route, ctx.current, ctx.save = environment, route, current, save
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'host', '0', '1', '2', '3'),
            'use the pinned interpreter and one planned stage')
    with (model.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            basis.comparison.run(int(action), ctx)
