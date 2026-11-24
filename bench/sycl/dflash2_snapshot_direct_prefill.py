"""Compare retained DFlash2 direct prefill before and after target-only snapshots."""

from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_prefill_window_writer_speed as writer
import dflash2_target_snapshot_build as snapshot
import dflash2_target_snapshot_restore as restore

bank, basis, require = writer.bank, writer.basis, writer.require
speed, generation = writer.speed, writer.speed.generation
clock = generation.clock
READ, HEADS, RUNS = writer.bank_speed.LONG_READ, writer.HEADS, writer.RUNS
SLOTS = writer.bank_speed.admission.SLOTS
ORDER = ('incumbent', 'snapshot', 'snapshot', 'incumbent', 'incumbent', 'snapshot')
KEY = 'dflash2_snapshot_direct_prefill_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
RESTORE = '962acee66e7e3ff4969370f8fadb28e673fba006'
INITIAL_SPEED = 'bd179280ae65b3b84771cf6facef2ec718afdb49'
PROTOCOL = dict(
    order=list(ORDER), read=READ, requests_per_process=RUNS,
    unit='Compare the coherent target-only snapshot engine with its measured parent. '
         'Use the same mode16 direct attention writer at 122881 fresh inputs.',
    acceptance='Require both exact builds, byte-identical request bodies, actual direct writer, '
               'matched expert, KV, tap coverage and verifier work, normal owner closure and zero fresh Xe faults. '
               'Report first use, all individual subsequent times, matched prompt/TTFT ratios and their directions. '
               'Record the repeated 1000 tok/s target independently of paired direction.',
    stop='Stop on a source, input, route, runtime, correctness, owner, budget or fault mismatch. '
         'Complete every eligible pair without speed-based exclusions.',
    limits='The benchmark uses a short fixed generation to control prefill handoff. Sustained decode '
           'and direct tap content have their own controls. The prior live restore comparison uses the copy route.')


def baseline():
    row = writer.long_baseline()
    measured = bank.previous.historic(INITIAL_SPEED, writer.LONG_KEY)
    require(measured['stage'] == 'closed' and measured['passed'] and measured['ownership_closed']
            and measured['binary'] == row['binary']
            and measured['decision']['all_warm_requests_1000']['direct'],
            'require the repeated measured direct reference')
    native = bank.previous.read(snapshot.ROOT / 'record.json')
    snapshot.context().check(native)
    require(native['stage'] == 'built' and native['passed'] and native['complete']
            and native['ownership_closed'] and not native['active'] and native['host_control']['passed']
            and native['projection']['target_snapshot_parent_hash'] == row['binary']['sha256'],
            'require the exact same-source target-only engine delta')
    tested = bank.previous.historic(RESTORE, restore.KEY)
    binary = native['binaries']['strata']
    record_sha = bank.signature(bank.builder.compact(native))
    require(tested['stage'] == 'closed' and tested['passed'] and tested['complete']
            and tested['ownership_closed'] and not tested['active']
            and tested['binary'] == binary and tested['native_build_sha256'] == record_sha
            and all(tested[k] == row[k] for k in
                    ('boot', 'driver', 'runtime_manifest', 'native_validation_probe'))
            and tested['outcomes']['short-copy']['restored_tokens'] == 32769
            and tested['outcomes']['near-copy']['restored_tokens'] == 253919
            and tested['outcomes']['near-copy']['requests'][-1]['prompt'] == 262128,
            'require both original live target-only restore cases on this executable')
    return dict(row, snapshot_reference=dict(binary=binary, source=native['native_source_commit'],
        native_build_sha256=record_sha, restore_commit=RESTORE), direct_reference_commit=INITIAL_SPEED)


def native_context(row, plan):
    arm = ORDER[plan['index']]
    require(plan['arm'] == arm, 'planned native arm changed')
    if arm == 'snapshot':
        selected = row['snapshot_reference']
        binary, source = selected['binary'], selected['source']
    else:
        binary, source = row['binary'], row['native_source_commit']
    return dict(binary=binary, native_source_commit=source,
                native_validation_probe=row['native_validation_probe'], boot=row['boot'])


def environment(row, plan):
    return writer.environment(row, dict(plan, arm='direct'), read=READ)


def decision(processes):
    require([p['arm'] for p in processes] == list(ORDER), 'all matched pairs required')
    ratios = []
    for a, b in zip(processes[::2], processes[1::2], strict=True):
        incumbent, candidate = (a, b) if a['arm'] == 'incumbent' else (b, a)
        ratios.append(dict(
            first_ms=candidate['samples'][0]['prompt_ms'] / incumbent['samples'][0]['prompt_ms'],
            warm_ms=candidate['subsequent_median_ms'] / incumbent['subsequent_median_ms'],
            warm_ttft=statistics.median(s['ttft_s'] for s in candidate['samples'][1:]) /
                      statistics.median(s['ttft_s'] for s in incumbent['samples'][1:])))
    warm = {arm: [READ * 1000 / p['subsequent_median_ms'] for p in processes if p['arm'] == arm]
            for arm in ('incumbent', 'snapshot')}
    return dict(snapshot_over_incumbent=ratios, warmed_process_PP=warm,
        all_pairs_faster=all(pair['warm_ms'] < 1 for pair in ratios),
        all_pairs_slower=all(pair['warm_ms'] > 1 for pair in ratios),
        all_warm_requests_1000={arm: all(s['prompt_ms'] <= READ
            for p in processes if p['arm'] == arm for s in p['samples'][1:]) for arm in warm},
        sustained_decode_qualified=False, direct_tap_payloads_qualified=False, adopted=False)


def experiment():
    prior = writer.experiment(read=READ)
    ctx = clock.experiment()
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    ctx.baseline, ctx.protocol = baseline, PROTOCOL
    ctx.native_context, ctx.environment = native_context, environment
    ctx.configuration = speed.configuration
    ctx.command = lambda plan: speed.command(plan, READ)
    old_current, old_save = ctx.current, ctx.save

    def pins(revision):
        selected = prior.pins(revision)
        for name in (THIS, clock.THIS, snapshot.THIS, restore.THIS):
            require((bank.P / name).read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed snapshot prefill comparison input: ' + name)
            selected[name] = bank.sha(bank.P / name)
        return selected

    def save(row):
        if row['stage'] == 'prepared':
            row['request_hashes'] = {str(READ): row['writer_reference']['hashes']}
            for plan in row['plans']:
                plan.update(read=READ, heads=HEADS)
        old_save(row)

    def current(row):
        require(all((p['read'], p['heads']) == (READ, HEADS) for p in row['plans']),
                'planned direct prefill shape changed')
        require(len({bank.signature({k: v.replace(p['root'], '<process>')
                    for k, v in environment(row, p).items()}) for p in row['plans']}) == 1,
                'normalized runtime environment changed between executables')
        return old_current(row)

    def route(text, plan):
        return writer.request_route(text, dict(plan, arm='direct'), read=READ, slots=SLOTS)

    ctx.pins, ctx.current, ctx.save = pins, current, save
    runner = SimpleNamespace(ROOT=ROOT, ORDER=[(READ, HEADS)] * len(ORDER), RUNS=RUNS,
        current=current, native_context=native_context, environment=environment,
        environment_fingerprint=generation.suite.signature, save=save, request_route=route,
        after_execution=lambda outcome, row, plan: writer.after_execution(outcome, row, plan, read=READ),
        decision=decision, validate_model_text=speed.validate_model_text,
        verify_model_assets=speed.verify_model_assets)
    ctx.run = lambda index: basis.comparison.run(index, runner)
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
