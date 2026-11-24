"""Compare the actual copy and direct writers on the same DFlash2 window."""

from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_prefill_direct_store_speed as previous_writer
import dflash2_prefill_window_direct as direct
import dflash2_prefill_window_speed as window_speed

bank, basis, require = direct.bank, direct.basis, direct.require
speed, window = window_speed.speed, window_speed.window
READ, HEADS, RUNS = window_speed.READ, window_speed.HEADS, window_speed.RUNS
ORDER = previous_writer.ORDER
KEY = 'dflash2_prefill_window_writer_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
WINDOW_TIMING = '6fb701c7fdf5d55c6b9503e8d3b43e719f2c4cc5'
DIRECT_TRACE = '3b8cdaf515ac7d8ad3ffafefabae754847a917dd'
WRITER = window.store.SELECTOR
WORK = ('requested_bank_H2D_bytes', 'uploads', 'jobs', 'dequants', 'windows',
        'maximum_rows', 'kv', 'tap_vram_bytes', 'request_work')


def unchanged_work(route):
    return {key: route[key] for key in WORK}


def baseline():
    row = window_speed.baseline()
    admission = basis.validation.committed(bank.RESULT, (direct.KEY,))
    require(admission == bank.previous.historic(DIRECT_TRACE, direct.KEY)
            and admission['stage'] == 'closed' and admission['complete'] and admission['passed']
            and admission['ownership_closed'] and not admission['active'],
            'recover the original closed direct-writer trace')
    tested, plan = admission['outcome'], admission['plan']
    require(tested['complete'] and tested['passed'] and tested['ownership_closed']
            and tested['native_exit_code'] == 0 and not tested['fresh_faults']
            and admission['comparison'] == dict(matched_initial_short_state=True,
                matched_state_and_generated_work=True, matched_tap_coverage=True, matched_expert_work=True,
                independent_processes=True, requested_D2D_bytes_removed=direct.COPY_BYTES)
            and all(plan[k] == row[k] for k in ('binary', 'native_source_commit', 'native_build_sha256',
                    'boot', 'driver', 'runtime_manifest'))
            and plan['validation'] == row['native_validation_probe']
            and plan['drafter_assets'] == row['dflash2_assets'],
            'require the matched mode16 traced writer admission on the current build')

    previous = basis.validation.committed(bank.RESULT, (window_speed.KEY,))
    require(previous == bank.previous.historic(WINDOW_TIMING, window_speed.KEY)
            and bank.signature(previous) == plan['quiet_window_sha256']
            and previous['decision']['repeated_warm_gain'], 'recover the original one-window timing')
    selected = [p for p in previous['processes'] if p['arm'] == 'window16']
    require(len(selected) == 3 and all(p['passed'] and p['clean_speed'] and p['ownership_closed']
                and p['native_exit_code'] == 0 and not p['fresh_faults']
                and len(p['routes']) == len(p['samples']) == RUNS
                and all(r['windows'] == 48 and r['logical_copy_bytes'] == direct.COPY_BYTES
                        for r in p['routes']) for p in selected), 'require the original three copy processes')
    hashes = previous['request_hashes'][str(READ)]
    work = [unchanged_work(r) for r in selected[0]['routes']]
    require(all([unchanged_work(r) for r in p['routes']] == work
                and [s['request_sha256'] for s in p['samples']] == hashes for p in selected),
            'original copy work or request bodies differ')
    return dict(row, writer_reference=dict(admission_sha256=bank.signature(admission),
        trace_commit=DIRECT_TRACE, timing_commit=WINDOW_TIMING, hashes=hashes, work_sha256=bank.signature(work)))


def environment(row, plan):
    env = window_speed.environment(row, dict(plan, arm='window16'))
    require(env[WRITER] == '0', 'expected the fixed copy reference')
    env[WRITER] = str(int(plan['arm'] == 'direct'))
    return env


def request_route(text, plan):
    return window_speed.request_route(text, dict(plan, arm='window16'), direct=plan['arm'] == 'direct')


def after_execution(outcome, row, plan):
    window_speed.after_execution(outcome, row, plan)
    require(bank.signature([unchanged_work(r) for r in outcome['routes']])
            == row['writer_reference']['work_sha256'], 'work outside the attention writer changed')


def experiment():
    previous = window_speed.experiment()
    ctx = speed.generation.experiment(synchronous_commit=True)
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment = baseline, speed.configuration, environment
    ctx.command = lambda plan: speed.command(plan, READ)
    ctx.protocol = dict(order=list(ORDER), read=READ, requests_per_process=RUNS,
        unit='Change only the attention writer before exec. Keep mode16, the selected reader, '
             'checked drains, demand uploads, bank128, BF16 residuals, retained KV and DFlash2 fixed.',
        acceptance='Three counterbalanced fresh-process pairs through the pinned exact-prefill benchmark. '
                   'Verify actual stores, retained KV and all expert work against the original mode16 copy prompts. '
                   'Keep first-use and warmed prompt time and client TTFT separate. '
                   'Advance a repeated warm gain when all three paired warm prompt-time ratios are below one. '
                   'Record the 1000 tok/s check separately.',
        stop='Stop on changed inputs, route, state, runtime, ownership, budget or GPU faults. '
             'Keep all eligible samples. This protocol uses short generation to check handoff.')
    old_current, old_save = ctx.current, ctx.save

    def pins(revision):
        files = previous.pins(revision)
        for name in (THIS, direct.THIS, previous_writer.THIS):
            require((bank.P / name).read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed window writer timing source: ' + name)
            files[name] = bank.sha(bank.P / name)
        return files

    def save(row):
        if row['stage'] == 'prepared':
            row['request_hashes'] = {str(READ): row['writer_reference']['hashes']}
            for plan in row['plans']:
                plan.update(read=READ, heads=HEADS)
        old_save(row)

    def current(row):
        normalized = []
        for plan in row['plans']:
            require((plan['read'], plan['heads']) == (READ, HEADS), 'planned extent changed')
            env = environment(row, plan)
            require(env.pop(WRITER) == str(int(plan['arm'] == 'direct')), 'writer policy changed')
            normalized.append(bank.signature({k: v.replace(plan['root'], '<process>') for k, v in env.items()}))
        require(len(set(normalized)) == 1, 'process settings differ outside the writer')
        return old_current(row)

    ctx.pins, ctx.current, ctx.save = pins, current, save
    runner = SimpleNamespace(ROOT=ROOT, ORDER=[(READ, HEADS)] * len(ORDER), RUNS=RUNS,
        current=current, environment=environment, environment_fingerprint=speed.generation.suite.signature,
        save=save, request_route=request_route, after_execution=after_execution,
        decision=lambda rows: previous_writer.decision(rows, read=READ, copy_bytes=direct.COPY_BYTES),
        validate_model_text=speed.validate_model_text, verify_model_assets=speed.verify_model_assets)
    ctx.run = lambda index: basis.comparison.run(index, runner)
    return ctx


if __name__ == '__main__':
    speed.generation.clock.main(experiment())
