"""Time independent attention-input borrowing on the 122,881-input path."""

from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_prefill_window_long_direct as admission
import dflash2_prefill_window_writer_speed as writer
from prefill_resident_input_model import input_receipt

bank, basis, require = writer.bank, writer.basis, writer.require
window, speed = writer.window, writer.speed
KEY = 'dflash2_prefill_window_122881_reader_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
INPUT_RECEIPT = 'tools/sycl/prefill_resident_input_model.py'
READ, SLOTS = writer.bank_speed.ARMS['longer']
ROWS = READ - 1
LAYERS, ELIGIBLE = 48, 690
INPUT_BYTES = (LAYERS - 2) * ROWS * window.workspace.ROW
OUTPUT_BYTES = LAYERS * ROWS * window.workspace.ROW
READER = window.store.ATTENTION_READER
ORDER = writer.ORDER


def baseline():
    row = writer.long_baseline()
    cohort = basis.validation.committed(bank.RESULT, (writer.LONG_KEY,))
    same_build = writer.bank_speed.BUILD_KEYS
    require(cohort['stage'] == 'closed' and cohort['complete'] and cohort['passed']
            and cohort['ownership_closed'] and not cohort['active']
            and all(cohort[key] == row[key] for key in (*same_build, 'writer_reference'))
            and tuple(p['arm'] for p in cohort['processes']) == ORDER,
            'require the closed same-build writer comparison')
    reference = basis.validation.committed(bank.RESULT, (admission.INPUT_KEY,))
    plan, observed = reference['plan'], reference['outcome']
    require(reference['stage'] == 'closed' and reference['complete'] and reference['passed']
            and reference['ownership_closed'] and not reference['active']
            and observed['complete'] and observed['passed'] and observed['ownership_closed']
            and observed['native_exit_code'] == 0 and not observed['fresh_faults']
            and all(plan[key] == row[key] for key in same_build)
            and plan['validation'] == row['native_validation_probe']
            and plan['drafter_assets'] == row['dflash2_assets']
            and plan['copy_trace_sha256'] == row['longer_admission_sha256']
            and plan['cases'][admission.CASE]['budget'] == window.budget(16, slots=SLOTS, rows=ROWS)
            and observed['first_verifier_window'] == [ROWS, 1]
            and observed['route']['logical_copy_bytes'] == OUTPUT_BYTES
            and observed['attention_input'] == dict(eligible=ELIGIBLE, borrowed=ELIGIBLE,
                borrowed_bytes=INPUT_BYTES, copy_bytes=0, checked_layers=LAYERS)
            and reference['comparison'] == dict(matched_state=True, matched_work=True, distinct_process=True,
                requested_D2D_bytes_removed=INPUT_BYTES),
            'require the closed reader admission with copied output')
    return dict(row, reader_reference=dict(writer_cohort_sha256=bank.signature(cohort),
        reader_admission_sha256=bank.signature(reference)))


def environment(row, plan):
    env = writer.environment(row, dict(plan, arm='copy'), read=READ)
    require(env[READER] == env[writer.WRITER] == '0', 'require original copied attention')
    env[READER] = str(int(plan['arm'] == 'direct'))
    return env


def request_route(text, plan):
    route = writer.window_speed.request_route(text, dict(plan, arm='window16'), read=READ, slots=SLOTS)
    require(route['logical_copy_bytes'] == OUTPUT_BYTES, 'output copy changed')
    chunks = (ROWS + window.store.CHUNK - 1) // window.store.CHUNK
    layout = dict(gpu_rows=ROWS, maximum_rows=route['maximum_rows'], pieces=LAYERS * chunks)
    inputs = input_receipt(text, ROWS, layout, plan['arm'] == 'direct', trace_layers=False)
    require(inputs['eligible'] == ELIGIBLE, 'reader extent changed')
    return dict(route, attention_input=inputs)


def after_execution(outcome, row, plan):
    writer.window_speed.after_execution(outcome, row, plan, read=READ)
    require(bank.signature([writer.unchanged_work(r) for r in outcome['routes']])
            == row['writer_reference']['work_sha256'], 'work outside the attention reader changed')


def decision(processes):
    summary = writer.previous_writer.decision(processes, read=READ, copy_bytes=INPUT_BYTES)
    summary['all_warm_requests_1000'] = {
        arm: all(s['prompt_ms'] <= READ for p in processes if p['arm'] == arm for s in p['samples'][1:])
        for arm in ('copy', 'direct')
    }
    return summary


def experiment():
    previous = writer.experiment(read=READ)
    ctx = speed.generation.experiment(synchronous_commit=True)
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment = baseline, speed.configuration, environment
    ctx.command = lambda plan: speed.command(plan, READ)
    ctx.protocol = dict(order=list(ORDER), read=READ, requests_per_process=writer.RUNS,
        unit='Direct selects input borrowing before exec. Both arms keep copied output, the 32-slot bank, '
             'DFlash2, demand expert uploads and 48 MoE windows.',
        acceptance='Use three counterbalanced independent-process pairs, each with one first-use and two '
                   'subsequent timings. Match the original request bodies and expert, KV, tap and verifier work. '
                   'Check all 690 eligible input spans and unchanged output copies. A repeated gain requires '
                   'all paired warmed prompt ratios below one. Record the 1000 tok/s result separately.',
        stop='Stop on changed inputs, route, state, runtime, ownership, budget or GPU faults. Preserve all eligible samples.')
    old_current, old_save = ctx.current, ctx.save

    def pins(revision):
        files = previous.pins(revision)
        for source in (THIS, INPUT_RECEIPT):
            path = bank.P / source
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + source),
                    'changed reader comparison input: ' + source)
            files[source] = bank.sha(path)
        return files

    def save(row):
        if row['stage'] == 'prepared':
            row['request_hashes'] = {str(READ): row['writer_reference']['hashes']}
            for plan in row['plans']:
                plan.update(read=READ, heads=writer.HEADS)
        old_save(row)

    def current(row):
        fingerprints = []
        for plan in row['plans']:
            require((plan['read'], plan['heads']) == (READ, writer.HEADS), 'prepared reader extent changed')
            env = environment(row, plan)
            require(env.pop(READER) == str(int(plan['arm'] == 'direct')), 'reader selector changed')
            fingerprints.append(bank.signature({key: value.replace(plan['root'], '<process>')
                                               for key, value in env.items()}))
        require(len(set(fingerprints)) == 1, 'process settings differ outside the attention reader')
        return old_current(row)

    ctx.pins, ctx.current, ctx.save = pins, current, save
    runner = SimpleNamespace(ROOT=ROOT, ORDER=[(READ, writer.HEADS)] * len(ORDER), RUNS=writer.RUNS,
        current=current, environment=environment, environment_fingerprint=speed.generation.suite.signature,
        save=save, request_route=request_route, after_execution=after_execution, decision=decision,
        validate_model_text=speed.validate_model_text, verify_model_assets=speed.verify_model_assets)
    ctx.run = lambda index: basis.comparison.run(index, runner)
    return ctx


if __name__ == '__main__':
    speed.generation.clock.main(experiment())
