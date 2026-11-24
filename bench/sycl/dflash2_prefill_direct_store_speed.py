"""Measure the DFlash2 attention writer with the official exact-prefill workload."""

from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_prefill_direct_store as admission

speed, bank, basis, require = admission.speed, admission.bank, admission.basis, admission.require
GATE = speed.gate
READ, HEADS, RUNS = GATE.SPLIT_READ, speed.HEADS, speed.RUNS
ORDER = ('copy', 'direct', 'direct', 'copy', 'copy', 'direct')
KEY = 'dflash2_prefill_direct_store_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PIECES, COPY_BYTES = admission.model.restore.capacity.expected_store(READ - 1, READ - 1, READ - 1)


def unchanged_work(route):
    placement = {key: value for key, value in route['store'].items() if key not in ('direct', 'copy_bytes')}
    return dict(route, store=placement)


def baseline():
    row = speed.baseline(READ)
    checked = basis.validation.committed(bank.RESULT, (admission.KEY,))
    plan = checked['plan']
    require(checked['stage'] == 'closed' and checked['complete'] and checked['passed']
            and checked['ownership_closed'] and not checked['active']
            and tuple(checked['outcomes']) == tuple(admission.ARMS)
            and all(case['passed'] and case['complete'] and case['ownership_closed']
                    and case['native_exit_code'] == 0 and case['fresh_faults'] == []
                    for case in checked['outcomes'].values())
            and checked['comparison'] == dict(matched=True, requested_D2D_bytes_removed=admission.COPY_BYTES)
            and all(plan[key] == row[key] for key in ('binary', 'native_source_commit', 'native_build_sha256',
                        'boot', 'driver', 'runtime_manifest'))
            and plan['validation'] == row['native_validation_probe']
            and plan['drafter_assets'] == row['dflash2_assets'],
            'complete the matched traced split admission on this exact DFlash2 build first')

    earlier = bank.previous.historic(admission.SPLIT_TIMING, speed.SPLIT_KEY)
    require(earlier['stage'] == 'closed' and earlier['complete'] and earlier['passed']
            and earlier['ownership_closed'] and not earlier['active'] and earlier['decision']['repeated_warm_gain']
            and all(earlier[key] == row[key] for key in ('binary', 'native_source_commit', 'native_build_sha256',
                        'boot', 'driver', 'runtime_manifest')),
            'recover the measured retained-KV copy control')
    selected = [p for p in earlier['processes'] if p['arm'] == 'retain']
    require(len(selected) == 3 and all(p['clean_speed'] and p['read'] == READ and p['complete'] and p['passed']
                and p['ownership_closed'] and p['native_exit_code'] == 0 and p['fresh_faults'] == []
                and len(p['routes']) == len(p['samples']) == RUNS
                and all(r['store']['pieces'] == PIECES and r['store']['direct'] == 0
                        and r['store']['copy_bytes'] == COPY_BYTES for r in p['routes']) for p in selected),
            'the three controls must actually run the all-GPU copy route')
    work = [unchanged_work(route) for route in selected[0]['routes']]
    hashes = earlier['request_hashes'][str(READ)]
    require(all([unchanged_work(r) for r in p['routes']] == work
                and [s['request_sha256'] for s in p['samples']] == hashes for p in selected),
            'official request bodies or native work differ between prior processes')
    return dict(row, attention_store_reference=dict(admission_sha256=bank.signature(checked),
        timing=admission.SPLIT_TIMING, request_hashes=hashes, unchanged_work_sha256=bank.signature(work)))


def environment(row, plan):
    return admission.environment(row, Path(plan['root']), admission.ARMS[plan['arm']])


def request_route(text, plan):
    direct = admission.ARMS[plan['arm']]
    route = speed.request_route(text, dict(plan, arm='retain'), direct=direct)
    store = route['store']
    require(store['pieces'] == PIECES and store['direct'] == (PIECES if direct else 0)
            and store['copy_bytes'] == (0 if direct else COPY_BYTES),
            'actual attention write count or requested copy bytes differ')
    return route


def after_execution(outcome, row, plan):
    actual = [unchanged_work(r) for r in outcome['routes']]
    require(outcome['clean_speed'] and outcome['native_identity'] not in
            [p['native_identity'] for p in row['processes']]
            and bank.signature(actual) == row['attention_store_reference']['unchanged_work_sha256'],
            'clean process ownership, or work outside attention output, changed')
    for route, sample in zip(actual, outcome['samples'], strict=True):
        require(route['request_work'] == [READ, 0, READ, sample['generated_tokens'],
                                          sample['drafts_accepted'], sample['drafts_offered']],
                'actual native and client request work differed')
    outcome['arm'] = plan['arm']


def decision(processes, *, read=READ, copy_bytes=COPY_BYTES):
    require(tuple(p['arm'] for p in processes) == ORDER, 'expected all three counterbalanced pairs')
    records, ratios = [], []
    for process in processes:
        first, *warm = process['samples']
        warm_ms = statistics.median(s['prompt_ms'] for s in warm)
        records.append(dict(arm=process['arm'], first_ms=first['prompt_ms'], first_ttft_s=first['ttft_s'],
                            warm_ms=warm_ms, warm_pp=read * 1000 / warm_ms,
                            warm_ttft_s=statistics.median(s['ttft_s'] for s in warm)))
    for left, right in zip(records[::2], records[1::2], strict=True):
        copy, direct = (left, right) if left['arm'] == 'copy' else (right, left)
        fields = ('first_ms', 'first_ttft_s', 'warm_ms', 'warm_ttft_s')
        ratios.append({key: direct[key] / copy[key] for key in fields})
    rates = {arm: [r['warm_pp'] for r in records if r['arm'] == arm] for arm in admission.ARMS}
    return dict(process_times=records, direct_over_copy=ratios, warmed_PP=rates,
                median_warmed_PP={arm: statistics.median(values) for arm, values in rates.items()},
                repeated_warm_gain=all(r['warm_ms'] < 1 for r in ratios),
                repeated_1000={arm: all(rate >= 1000 for rate in values) for arm, values in rates.items()},
                requested_D2D_bytes_removed=copy_bytes, full_context_qualified=False,
                sustained_decode_qualified=False, adopted=False)


def experiment():
    reference = speed.experiment(READ)
    ctx = speed.generation.experiment(synchronous_commit=True)
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment = baseline, speed.configuration, environment
    ctx.command = lambda plan: speed.command(plan, READ)
    ctx.protocol = dict(order=list(ORDER), read=READ, requests_per_process=RUNS,
        unit='Change the attention writer at process start. Keep the scratch reader, checked drains, retained KV, '
             'batched uploads, head placement and DFlash2 spec4 fixed.',
        acceptance='Run three counterbalanced fresh-process pairs at the exact official 117489-token inputs. '
                   'Require identical request bodies and actual work outside the copied output. Keep first-use '
                   'and warmed prompt time and TTFT separately. Advance a repeated warm gain only when all three '
                   'paired warm prompt-time ratios are below one. Record mixed and smaller effects without declaring '
                   'a winner. Report the 1000 tok/s check separately.',
        stop='Stop on the first input, runtime, correctness, route, ownership, budget or fault failure. '
             'Keep all eligible samples. No speed-based exclusions. This is prefill timing with short output.')
    old_save, old_current = ctx.save, ctx.current

    def pins(revision):
        names = (THIS, admission.THIS)
        for name in names:
            require((bank.P / name).read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed attention-store timing input: ' + name)
        return reference.pins(revision) | {name: bank.sha(bank.P / name) for name in names}

    def save(row):
        if row['stage'] == 'prepared':
            row['request_hashes'] = {str(READ): row['attention_store_reference']['request_hashes']}
            for plan in row['plans']:
                plan.update(read=READ, heads=HEADS)
        old_save(row)

    def current(row):
        normalized = []
        for plan in row['plans']:
            require((plan['read'], plan['heads']) == (READ, HEADS), 'planned prefill extent changed')
            env = environment(row, plan)
            require(env.pop(admission.SELECTOR) == str(int(admission.ARMS[plan['arm']])),
                    'unexpected attention writer selector')
            normalized.append(bank.signature({key: value.replace(plan['root'], '<process>')
                                              for key, value in env.items()}))
        require(len(set(normalized)) == 1, 'environment changed outside the attention writer')
        return old_current(row)

    ctx.pins, ctx.current, ctx.save = pins, current, save
    runner = SimpleNamespace(ROOT=ROOT, ORDER=[(READ, HEADS)] * len(ORDER), RUNS=RUNS,
        current=current, environment=environment, environment_fingerprint=speed.generation.suite.signature,
        save=save, request_route=request_route, after_execution=after_execution, decision=decision,
        validate_model_text=speed.validate_model_text, verify_model_assets=speed.verify_model_assets)
    ctx.run = lambda index: basis.comparison.run(index, runner)
    return ctx


if __name__ == '__main__':
    speed.generation.clock.main(experiment())
