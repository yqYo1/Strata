"""Attribute first-use prefill on the retained mode16 engine."""

from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_primary_prefill_curve as curve
import dflash2_primary_retained as retained
import bench_engine as observer

speed, bank, basis, require = curve.speed, curve.bank, curve.basis, curve.require
KEY = 'dflash2_primary_prefill_cpu_wait_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
OBSERVER = str(Path(observer.__file__).resolve().relative_to(bank.P))
REFERENCE = '71ddd9046'
READ = 118785
ORDER = ('clean', 'cpu-wait', 'cpu-wait', 'clean')
SELECTOR = 'STRATA_PREFILL_CPU_WAIT_PROFILE'
WAIT_KINDS = ('compute', 'transfer', 'event', 'compat', 'direct')
HOST_KINDS = observer.CPU_HOST_BASE + observer.CPU_HOST_MOE
PLAN = dict(order=list(ORDER), read=READ, requests_per_process=speed.RUNS,
    scope='Same selected mode16 engine, inputs, private cache policy and three requests per process. '
          'Enable only the native layer-major CPU wait observer for the two observed processes.',
    decision='Keep first and subsequent clocks separate. Require both paired subsequent medians '
             'within five percent before using the observer for attribution. Report first-use ratios '
             'separately because first-use variance can exceed observer overhead.',
    stop='Stop on any changed work, unavailable accounting, exception, driver change or ownership failure.')


def comparable_work(frame):
    return {key: value for key, value in frame.items() if key not in ('free_bytes', 'cpu_wait')}


def baseline():
    row = retained.baseline()
    closed = bank.previous.historic(REFERENCE, curve.KEY)
    require(retained.closed(closed, 'closed')
            and all(closed[k] == row[k] for k in ('binary', 'native_source_commit', 'native_build_sha256')),
            'require the closed exact-length comparison on the retained engine')
    matches = [p for p in closed['processes'] if p['read'] == READ]
    require(len(matches) == 2 and all(p['passed'] and p['complete'] and p['ownership_closed']
            and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in matches),
            'require the two closed same-length controls')
    original, repeat = matches
    hashes = [sample['request_sha256'] for sample in original['samples']]
    work = [comparable_work(frame) for frame in original['routes']]
    require(len(hashes) == len(work) == speed.RUNS
            and hashes == [sample['request_sha256'] for sample in repeat['samples']]
            and work == [comparable_work(frame) for frame in repeat['routes']],
            'the original repeated bodies or native work differ')
    return dict(row, request_hashes={str(READ): hashes}, frozen_prefill_work=work)


def command(plan):
    return curve.command(dict(plan, arm=READ))


def environment(row, plan):
    env = curve.environment(row, plan)
    env[SELECTOR] = '1' if plan['arm'] == 'cpu-wait' else '0'
    return env


def route(text, plan):
    frame = curve.route(text, plan)
    if plan['arm'] == 'clean':
        require(not any(line.startswith(('strata prefill cpu wait:', 'strata prefill cpu host:'))
                        for line in text.splitlines()), 'unexpected CPU observer on clean request')
        return frame
    data = observer.extract_cpu_observers(text, expected_version=4)
    wait, host = data['wait'], data['host']
    counts = [int(wait[name + '_count']) for name in WAIT_KINDS]
    require(sum(counts) == int(wait['wait_count']), 'CPU wait counts do not conserve')
    for unit in ('cpu_s', 'wall_s'):
        require(abs(sum(float(wait[f'{name}_{unit}']) for name in WAIT_KINDS)
                    - float(wait['wait_' + unit])) < 1e-7, 'CPU wait intervals do not conserve')
    return dict(frame, cpu_wait=dict(
        wall_s=float(wait['wall_s']), cpu_s=float(wait['main_cpu_s']),
        wait_cpu_s=float(wait['wait_cpu_s']), wait_wall_s=float(wait['wait_wall_s']),
        waits={name: [int(wait[name + '_count']), float(wait[name + '_cpu_s']),
                      float(wait[name + '_wall_s'])] for name in WAIT_KINDS},
        outside_wait={name: [float(host[name + '_outside_wait_cpu_s']),
                             float(host[name + '_outside_wait_wall_s'])] for name in HOST_KINDS},
        unclassified=[float(host['unclassified_cpu_s']), float(host['unclassified_wall_s'])]))


def after_execution(outcome, row, plan):
    observed = plan['arm'] == 'cpu-wait'
    require(outcome['request_routes_checked_before_continue'] == speed.RUNS
            and [comparable_work(f) for f in outcome['routes']] == row['frozen_prefill_work']
            and all(('cpu_wait' in f) is observed for f in outcome['routes']),
            'observer changed the actual prefill or decode work')
    if observed:
        for frame, sample in zip(outcome['routes'], outcome['samples'], strict=True):
            require(0 < frame['cpu_wait']['wall_s'] <= sample['prompt_ms'] / 1000 + 0.001,
                    'the layer-major observer exceeded the enclosing prompt interval')
    outcome.update(arm=plan['arm'], clean_speed=not observed)


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and all(p['passed'] and p['complete'] and p['ownership_closed']
                    and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(ORDER), 'require four independently closed processes')
    pairs = []
    for x, y in zip(processes[::2], processes[1::2], strict=True):
        clean, observed = (x, y) if x['arm'] == 'clean' else (y, x)
        ratios = {'first': observed['samples'][0]['prompt_ms'] / clean['samples'][0]['prompt_ms'],
                  'subsequent': observed['subsequent_median_ms'] / clean['subsequent_median_ms']}
        pairs.append(ratios)
    return dict(observed_over_clean=pairs,
                warm_observer_gate=all(.95 <= p['subsequent'] <= 1.05 for p in pairs),
                first_overhead_qualified=False, adopted=False)


def experiment():
    ctx = speed.clock.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol = baseline, PLAN
    ctx.configuration, ctx.command, ctx.environment = speed.generation.configuration, command, environment
    original_save, original_current = ctx.save, ctx.current
    speed_pins = speed.experiment().pins

    def pins(revision):
        values = speed_pins(revision)
        for name in (THIS, curve.THIS, retained.THIS, OBSERVER):
            source = bank.P / name
            require(source.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed cold-prefill comparison input: ' + name)
            values[name] = bank.sha(source)
        return values

    def save(row):
        if row['stage'] == 'prepared':
            for plan in row['plans']:
                plan.update(read=READ, heads=speed.HEADS, kv_arm='retain')
        original_save(row)

    def current(row):
        require(all((p['read'], p['heads'], p['kv_arm']) == (READ, speed.HEADS, 'retain')
                    for p in row['plans']), 'frozen read extent or KV policy differs')
        comparable = [{k: v.replace(p['root'], '<process>') for k, v in environment(row, p).items()
                       if k != SELECTOR} for p in row['plans']]
        require(all(env == comparable[0] for env in comparable), 'another setting differs between arms')
        return original_current(row)

    def run(index):
        row = bank.previous.read(ROOT / 'record.json')
        runner = basis.core.source.device.load_module('mode16_cold_wait', basis.comparison.__file__)
        original_models = runner.shared.models
        runner.shared = SimpleNamespace(**vars(runner.shared))

        def model_pins(config, assets, full=False):
            original_models.model_pins(row['configuration'], assets, full=full)
            for flag in ('--draft-model', '--mtp-draft-vocab'):
                path = config['args'][config['args'].index(flag) + 1]
                require(bank.sha(path) == row['dflash2_assets'][path], 'actual DFlash2 input changed')

        def validate_model(text, _):
            require(text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
                    and not re.search(r'(?m)^strata mtp: .*loaded|strata loan audit:|strata DFlash2 tap payload:', text),
                    'unexpected head, MTP layer or observer')

        runner.shared.models = SimpleNamespace(model_pins=model_pins)
        runner.fingerprint = bank.signature
        policy = SimpleNamespace(ROOT=ROOT, ORDER=[(READ, speed.HEADS)] * len(ORDER), RUNS=speed.RUNS,
            current=current, native_context=lambda data, _: data, environment=environment, save=save,
            request_route=route, after_execution=after_execution, decision=decision, validate_model_text=validate_model)
        runner.run(index, policy)

    ctx.pins, ctx.save, ctx.current, ctx.run = pins, save, current, run
    return ctx


if __name__ == '__main__':
    speed.clock.main(experiment())
