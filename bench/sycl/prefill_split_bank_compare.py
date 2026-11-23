"""Compare the actual split-prefill bank admission at 117489 input tokens."""

import fcntl
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_split_bank_build as build
import prefill_kv_gather_speed as speed
import prefill_workspace_admission as workspace

bank, model, basis, require = speed.bank, speed.model, speed.basis, speed.require
ROOT = bank.B / 'prefill-split-bank-compare-v1'
KEY = 'prefill_split_bank_compare_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
READ, RUNS = build.READ, 4
SLOTS = (128, 1, *build.SLOTS)
TRACED = 2
SLOT_OPTION = 'STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS'


def work(route, sample):
    return dict(request=sample['request_sha256'], reads=sample['actual_read_tokens'],
                offered=sample['drafts_offered'], accepted=sample['drafts_accepted'],
                generated=sample['generated_tokens'], GU=route['actual_GU_dispatch'],
                bank_bytes=route['requested_bank_H2D_bytes'], store=route['attention_store'],
                kv=route['kv'], gather=route['gather'])


def admission():
    native = bank.previous.read(build.ROOT / 'record.json')
    build.check(native)
    require(native['stage'] == 'built' and native['passed'] and native['complete']
            and native['ownership_closed'] and not native['active'] and native['host_control']['passed'],
            'complete the exact candidate build and its host fixture')
    parent = bank.previous.historic(build.HISTORY, 'prefill_kv_gather_speed_117489_v1')
    require(parent['passed'] and parent['complete'] and parent['ownership_closed'] and not parent['active']
            and parent['binary'] == native['projection']['parent_binary']
            and parent['decision']['contrast_passes']['reuse/base'], 'require the closed KV-reuse reference')
    samples = parent['processes'][speed.ORDER.index('reuse')]
    old_work = [work(dict(r, requested_bank_H2D_bytes=r['fast_window']['requested_bank_H2D_bytes']), s)
                for r, s in zip(samples['routes'], samples['samples'], strict=True)]
    require(len(old_work) == RUNS and [r['request'] for r in old_work] == parent['request_hashes'][str(READ)],
            'require the four measured 117K request bodies')
    prior = bank.previous.read(basis.reference.model.ROOT / 'record.json')
    require(prior['probe'] == parent['native_validation_probe'] and prior['boot'] == parent['boot']
            and prior['driver'] == parent['driver']
            and bank.signature(prior['models']) == parent['model_assets_sha256']
            and prior['runtime_manifest']['sha256'] == parent['runtime_manifest']['sha256']
            and bank.sha(prior['runtime_manifest']['path']) == prior['runtime_manifest']['sha256']
            and bank.previous.read(Path(prior['runtime_manifest']['path'])) == prior['runtime_files'],
            'keep the same model assets, providers and workstation state')
    prior['observer_admission'] = dict(fast_configuration_diagnostic=
        parent['observer_admission']['fast_configuration_diagnostic'],
        ordinary_control=parent['observer_admission']['ordinary_control'],
        reference_configuration=parent['configuration'], expected_work=old_work)
    return native, prior, prior['runtime_files'], None


def request_route(text, plan):
    value = workspace.request_route(text, dict(plan, window=10, relocate=0, upload='jit'))
    slots = plan['bank']
    require(value['layout']['split'] and value['layer_cache_bytes'] == slots * workspace.STRIDE
            and value['permanent_hits'] == 0 and value['uploads'] == value['actual_GU_dispatch']['dequants']
            and 'strata MoE prefix expert upload layer:' not in text, 'unexpected slot reuse or copy schedule')
    value['attention_store'] = model.restore.require_store(text, dict(direct=plan.get('direct_output', False)), plan)
    model.input_receipt(text, READ - 1, value['attention_store'], False, trace_layers=plan['trace_progress'])
    if plan['trace_progress']:
        kv = speed.state.kv.check_request(text, 0, READ - 1, 8192, reuse=True)
        value['kv'] = {k: kv[k] for k in ('reloaded_pages', 'reused_pages')}
    else:
        value['kv'] = speed.state.kv.check_clean_request(text, 0, READ - 1, 8192, reuse=True)
    value['gather'] = speed.state.gather_counts(text, False)
    receipt = {k: value[k] for k in ('layer_cache_bytes', 'uploads', 'actual_GU_dispatch', 'requested_bank_H2D_bytes',
                                     'attention_store', 'kv', 'gather')}
    if plan['trace_progress']:
        for phase in ('lend', 'refill'):
            table = workspace.one(text, 'strata loan audit: phase=' + phase + ' ')
            require(table['table_entries'] == '24576' and table['table_match'] == '1',
                    'decode expert residency table differs')
        payload = workspace.one(text, 'strata loan audit: phase=payload ')
        require(payload['rows'] == '1256' and payload['bytes'] == '2575206400'
                and int(payload['lent']) > 0 and payload['same_slots'] == payload['payload_match'] == '1',
                'decode expert cache was not restored')
        receipt['loaned_slots'] = int(payload['lent'])
    return receipt


def check_work(outcome, row, plan):
    routes = outcome['routes']
    require([work(r, s) for r, s in zip(routes, outcome['samples'], strict=True)]
            == row['observer_admission']['expected_work'], 'request, expert or handoff work differs')
    require(outcome['clean_speed'] == (not plan['trace_progress']), 'observer charge differs')
    if plan['trace_progress']:
        path, = (Path(plan['root']) / 'bench').glob('*/results.json')
        samples = bank.previous.read(path)['results']['prefill_only']['requests']
        require([s['request_sha256'] for s in samples] == [s['request_sha256'] for s in outcome['samples']],
                'traced response identity differs')
        outcome['response_sha256'] = [[s['reasoning_sha256'], s['output_sha256']] for s in samples]
    if plan['index'] == 1:
        first = row['processes'][0]
        require(outcome['response_sha256'] == first['response_sha256']
                and all(a['loaned_slots'] == b['loaned_slots'] and a['uploads'] == b['uploads']
                        for a, b in zip(routes, first['routes'], strict=True)),
                'same-ELF response or decode cache handoff differs')


def decision(processes):
    require(len(processes) == len(SLOTS) and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
            for p in processes}) == len(SLOTS), 'require all independent processes')
    timings, ratios = [], []
    for i in range(TRACED, len(processes), 2):
        pair = {}
        for j in (i, i + 1):
            p = processes[j]
            require(p['index'] == j and p['passed'] and p['complete'] and p['clean_speed']
                    and p['ownership_closed'] and not p['fresh_faults'], 'ineligible clean process')
            first, *warm = p['samples']
            pair[SLOTS[j]] = dict(first_prompt_ms=first['prompt_ms'], first_ttft_s=first['ttft_s'],
                                  warm_prompt_ms=statistics.median(s['prompt_ms'] for s in warm),
                                  warm_ttft_s=statistics.median(s['ttft_s'] for s in warm))
        timings.append(pair)
        ratios.append({k: pair[1][k] / pair[128][k] for k in pair[1]})
    return dict(time_ratios_1_over_128=ratios, pair_times=timings, bank_bytes_saved=127 * workspace.STRIDE,
                traced_response_and_decode_loan_matched=True, physical_full_context_qualified=False,
                sustained_decode_qualified=False, adopted=False)


def experiment():
    ctx = basis.core.source.device.load_module('split_prefill_bank', basis.core.__file__)
    ctx.THIS, ctx.ROOT, ctx.KEY = THIS, ROOT, KEY
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (READ,) * len(SLOTS), [(READ, 1)] * len(SLOTS), RUNS, True
    ctx.basis, ctx.check_sample = admission, basis.reference.context().check_sample
    ctx.request_route, ctx.after_execution, ctx.decision = request_route, check_work, decision
    old_pins, old_save, old_current = ctx.pins, ctx.save, ctx.current

    def pins(revision):
        values = old_pins(revision)
        paths = [THIS, build.THIS, speed.THIS, speed.capacity.THIS, speed.dispatch.THIS,
                 speed.dispatch.split.THIS, *speed.state.SCRIPTS]
        values.update(bank.stage.source.pin(bank.P, paths)['source_sha256'])
        return values

    def environment(row, plan):
        env = model.ordinary_environment(row, plan['root'], row['observer_admission']['ordinary_control'])
        env.update({SLOT_OPTION: str(SLOTS[plan['index']]), model.INPUT: '0', model.OUTPUT: '0',
                    speed.state.KV: '1', speed.state.GATHER: '0'})
        require('STRATA_TRACE' not in env and 'STRATA_STATE_HASH' not in env
                and env.get('STRATA_SYCL_CACHE_LOAN_AUDIT', '0') == '0', 'clean reference required')
        if plan['index'] < TRACED:
            env.update(STRATA_TRACE='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1')
        return env

    def save(row):
        if row['stage'] == 'prepared':
            for plan, slots in zip(row['plans'], SLOTS, strict=True):
                plan.update(bank=slots, trace_progress=plan['index'] < TRACED)
            row.update(whole_bytes=len(SLOTS) * row['budgets']['total_cap'] + (8 << 20),
                       plan=dict(order=list(SLOTS), traced_processes=TRACED, requests=RUNS,
                                 acceptance='Match actual split placement, work, traced responses and decode loan payload. '
                                            'Retain every clean pair and separate first and subsequent clocks.',
                                 stop='Stop at the first identity, work, response, runtime, fault or ownership failure. '
                                      'No speed-based exclusion or automatic selection.'))
        old_save(row)

    def current(row):
        require(row['configuration'] == row['observer_admission']['reference_configuration']
                and row['plan']['order'] == list(SLOTS) and row['plan']['requests'] == RUNS
                and row['plan']['traced_processes'] == TRACED
                and [(p['index'], p['read'], p['bank'], p['trace_progress']) for p in row['plans']]
                    == [(i, READ, slots, i < TRACED) for i, slots in enumerate(SLOTS)],
                'frozen 117K configuration, order or arm differs')
        return old_current(row)

    ctx.pins, ctx.environment, ctx.current, ctx.save = pins, environment, current, save
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and
            (action in ('prepare', 'host') or action in map(str, range(len(SLOTS)))), 'select one comparison stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            basis.comparison.run(int(action), ctx)
