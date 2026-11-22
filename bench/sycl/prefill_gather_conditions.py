"""Measure host and Xe conditions on the measured 65K and 117K copy routes."""

import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import client_resources
import hardware_conditions
import prefill_gather_store as store

bank, basis, require = store.bank, store.basis, store.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
READER = Path(hardware_conditions.__file__).resolve()
SOURCES = (THIS, str(READER.relative_to(bank.P)),
           str(Path(client_resources.__file__).resolve().relative_to(bank.P)))
DEFAULT_READ = 65537
ORDER = ('clean', 'observed', 'clean')
MAX_RATIO = 1.05
PLAN = dict(order=list(ORDER), observed_index=1, seconds=900, interval=1,
    question='Record GPU frequency, throttle, scheduled CPU and host pressure on the '
             'selected copy route that slowed between timing periods.',
    acceptance='Require the warmed observed prompt and client TTFT to be within 1.05x '
               'of both clean controls. Record first use separately. Require the '
               'owned reader, exact request work, coverage, and normal fault-free exit.',
    limits='Frequency and throttle are sampled global signals. The clean controls '
           'screen the observer cost in this bracket. Physical saturation and '
           'historical conditions need separate evidence.')


def summarize(root, outcome, boot):
    path = root / 'client-resources.jsonl'
    require(0 < path.stat().st_size <= 32 << 20, 'passive reader output budget')
    records = [json.loads(line) for line in path.read_text().splitlines()]
    result_path, = (root / 'bench').glob('*/results.json')
    requests = json.loads(result_path.read_text())['results']['prefill_only']['requests']
    summaries = []
    for sample, actual in zip(outcome['samples'], requests, strict=True):
        require(all(actual[key] == value for key, value in sample.items()),
                'condition window timing differs from the admitted request')
        interval = actual['client_interval']
        resources = client_resources.summarize_capture(
            records, outcome['native_identity'], boot, interval, before_only=True)['before_first_output']
        conditions = hardware_conditions.summarize(
            records, interval['request_begin_monotonic_ns'], interval['first_output_monotonic_ns'])
        summaries.append(dict(
            request_sha256=sample['request_sha256'],
            contained_samples=conditions['contained_samples'],
            gpu_gauges={k: v for k, v in conditions['gauges'].items()
                        if not k.startswith('cpu/')},
            cpu_reported_khz={k: v['median'] for k, v in conditions['gauges'].items()
                              if k.startswith('cpu/') and k.endswith('/scaling_cur_freq')},
            gpu_energy=conditions['energy'],
            pci_and_throttle={k: v for k, v in conditions['text'].items()
                              if k.startswith('pci/') or '/throttle/' in k},
            host_counters=conditions['host_counters'],
            scheduled_cpu_seconds=resources['process_cpu_seconds'],
            cpu_observed_wall_seconds=resources['contained_CPU_wall_seconds'],
            drm_engines=resources['engines'],
            condition_read_ms=conditions['reader_wall_sum_ms'],
            max_condition_read_ms=conditions['maximum_reader_ms']))
    return summaries


def decision(processes):
    require(len(processes) == len(ORDER) and
            len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                 for p in processes}) == len(ORDER) and
            all(p['passed'] and p['complete'] and p['ownership_closed']
                and p['native_exit_code'] == 0 and not p['fresh_faults']
                and p['clean_speed'] == (arm == 'clean')
                and ('conditions' in p) == (arm == 'observed')
                for p, arm in zip(processes, ORDER, strict=True)),
            'complete all independent observer bracket cases')
    first, observed, last = [store.timing(p) for p in processes]
    ratios = {metric: [observed[metric] / clean[metric] for clean in (first, last)]
              for metric in observed}
    return dict(observer_over_clean=ratios, maximum_warmed_ratio=MAX_RATIO,
                observer_cost_screen_passed=all(ratio <= MAX_RATIO for metric in
                    ('warm_prompt_ms', 'warm_ttft_s') for ratio in ratios[metric]),
                attribution_scope='Current timing period before client first output.')


def experiment(read):
    # Keep the copy arm fixed. Only the middle native wrapper spawns a reader.
    ctx = store.experiment(read, False, timing_order=(False,) * len(ORDER))
    key = f'prefill_gather_conditions_{read}_v2'
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.decision = bank.B / key.replace('_', '-'), key, THIS, decision
    plan = dict(PLAN, read=read, requests=ctx.RUNS)
    pin_sources, store_save = ctx.pins, ctx.save
    store_current, store_native, store_work = ctx.current, ctx.native_context, ctx.after_execution

    def pins(revision):
        sources = pin_sources(revision)
        for path in SOURCES:
            original = bank.stage.source.git(bank.P, 'show', revision + ':' + path)
            require((bank.P / path).read_bytes() == original, 'changed condition source: ' + path)
            sources[path] = bank.sha(bank.P / path)
        return sources

    def collection(sources):
        return dict(source=dict(path=str(READER), sha256=sources[SOURCES[1]]),
                    seconds=PLAN['seconds'], interval=PLAN['interval'])

    def save(row):
        if row['stage'] == 'prepared':
            row['condition_plan'] = plan
            case = row['plans'][PLAN['observed_index']]
            path = Path(case['root']) / 'plan.json'
            native = json.loads(path.read_text())
            require('resource_collection' not in native, 'unexpected inherited collector')
            basis.comparison.write(path, dict(native, resource_collection=collection(row['sources'])))
            case['generated']['plan.json'] = bank.sha(path)
        store_save(row)

    def current(row):
        require(row['condition_plan'] == plan, 'frozen conditions plan differs')
        for case in row['plans']:
            native = json.loads((Path(case['root']) / 'plan.json').read_text())
            if case['index'] == PLAN['observed_index']:
                require(native['resource_collection'] == collection(row['sources']), 'collector plan differs')
            else:
                require('resource_collection' not in native, 'clean case has a collector')
        return store_current(row)

    def native_context(row, case):
        native = store_native(row, case)
        if case['index'] == PLAN['observed_index']:
            native['resource_collection'] = collection(row['sources'])
        return native

    def after_execution(outcome, row, case):
        store_work(outcome, row, case)
        if case['index'] == PLAN['observed_index']:
            outcome['clean_speed'] = False
            outcome['conditions'] = summarize(Path(case['root']), outcome, row['boot'])

    ctx.pins, ctx.save, ctx.current = pins, save, current
    ctx.native_context, ctx.after_execution = native_context, after_execution
    return ctx


if __name__ == '__main__':
    require(len(sys.argv) in (2, 3), 'use prepare|host|INDEX [65537|117489]')
    action, *length = sys.argv[1:]
    read = int(length[0]) if length else DEFAULT_READ
    require(sys.executable == basis.reference.model.PYTHON and
            read in store.SELECTION and
            action in ('prepare', 'host', *map(str, range(len(ORDER)))), 'choose a planned conditions stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment(read)
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            basis.comparison.run(int(action), ctx)
