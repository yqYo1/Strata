"""Measure the selective DFlash2 width-four schedules on one long code workload."""

import math
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_decode_t4_model as model
import dflash2_primary_continuation as primary

build, bank, require, validation = model.build, model.bank, model.require, model.validation
PRODUCTION = 'dflash2_decode_t4_production_dpas_v1'
REFERENCE = '9261c4b44'
KEY = 'dflash2_decode_t4_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
WORKLOAD = 'regular-code'
WORKLOADS = (WORKLOAD,)
ORDER = ('stock', 'after-cpu', 'early', 'after-cpu', 'early', 'stock', 'early', 'stock', 'after-cpu')
COMPARISONS = (('early', 'after-cpu'), ('early', 'stock'), ('after-cpu', 'stock'))
SCHEDULE = 'STRATA_SYCL_DECODE_T4_SCHEDULE'
AUDIT = 'STRATA_SYCL_DECODE_T4_AUDIT'
PLAN = dict(order=list(ORDER), workload=WORKLOAD, input_case=primary.continuation.INPUT_CASE,
    comparison='Three counterbalanced blocks on one executable. Compare stock with deferred to price the '
               'split and compare early with deferred to isolate its submission order. Keep production DPAS on.',
    scope='One complete thinking-on code request per fresh server, starting at 96K input. The native T4 '
          'per-window audit is enabled; kernel, state and runtime tracing are disabled. This is a route-checked '
          'speed screen. Clean repetitions and task, agent, weak-draft and physical-context checks are separate.',
    eligibility='Match the committed body and read extent, actual throughput prefill and KV placement, '
                'client/native token and draft counts, and valid output of at least 1024 tokens. '
                'Check the actual split modes on every recorded width-four window.',
    decision='Recommend a separate clean comparison when early improves TG against deferred and stock '
             'in at least two blocks each with geometric mean ratios above one, with equal logical prefill '
             'work. Report prompt time and TTFT ratios independently. Keep work differences and all samples.',
    stop='Stop on the first identity, workload, route, validation, ownership, fault or budget failure.')


def reference():
    row = bank.previous.historic(REFERENCE, primary.KEY)
    require(build.retained.closed(row, 'closed') and len(row['processes']) == 3,
            'require the retained actual long-code workload')
    samples = [next(s for s in p['evidence']['samples'] if s.get('workload') == WORKLOAD)
               for p in row['processes']]
    bodies = {(s['request_sha256'], s['read_tokens']) for s in samples}
    require(len(bodies) == 1 and samples[0]['read_tokens'] == 96204
            and all(s['generated_tokens'] >= 1024 and s['reused_tokens'] == 0 for s in samples),
            'the verified long-code bodies must remain fixed')
    body, extent = next(iter(bodies))
    return dict(source_commit=REFERENCE, body_sha256=body, read_tokens=extent)


def baseline():
    native = bank.previous.read(build.ROOT / 'record.json')
    build.check(native)
    proof = validation.committed(bank.RESULT, (PRODUCTION,))
    identity = bank.signature(build.builder.compact(native))
    binary = native['binaries']['strata']
    require(build.retained.closed(proof, 'closed') and proof['production_dpas_preserved']
            and proof['plan']['native_build_sha256'] == identity and proof['plan']['binary'] == binary
            and proof['plan']['dense_dpas'] == '1', 'close the production DPAS model comparison first')
    row = build.retained.baseline()
    row['base_physical_context_sha256'] = row.pop('physical_context_sha256')
    return dict(row, binary=binary, native_source_commit=native['native_source_commit'],
                native_build_sha256=identity, native_context_qualified=False,
                production_dpas_proof=bank.signature(proof), split_layers=proof['plan']['split_layers'],
                request_reference=reference())


def environment(row, plan):
    env = primary.curve.environment(row, plan)
    require(SCHEDULE not in env and AUDIT not in env
            and env.get('STRATA_SYCL_DECODE_Q6_DPAS_DENSE') == '1'
            and env.get('STRATA_SYCL_Q6_ROUTE_AUDIT', '0') == '0', 'unexpected inherited T4 or DPAS selector')
    return dict(env, **{SCHEDULE: plan['arm'], AUDIT: '1'})


def command(plan):
    return primary.continuation.command(plan) + ['--long-generation-workload', WORKLOAD]


def read_subset(root, workload_id):
    out = primary.continuation.read_complete_suite(root, workload_id, workloads=WORKLOADS)
    sample = out['samples'][-1]
    fixed = reference()
    require(sample['workload'] == WORKLOAD and sample['request_sha256'] == fixed['body_sha256']
            and sample['read_tokens'] == fixed['read_tokens'], 'long-code request differs from the pinned control')
    return out


def split_windows(frame, sample, arm, layers):
    records = [(mode, *map(int, values)) for mode, *values in model.T4.findall(frame)]
    if arm == 'stock':
        require('DECODE_T4 ' not in frame, 'the stock scheduler published a split')
        return dict(width4_windows=None, early_submissions=0)
    positions = [p for _, p, *_ in records]
    generated, accepted = (sample[k] for k in ('generated_tokens', 'drafts_accepted'))
    prompt = sample['prompt_tokens']
    early_layers = layers if arm == 'early' else 0
    require(0 < len(records) == frame.count('DECODE_T4 ') <= generated - accepted + 3
            and positions == sorted(set(positions)) and prompt - 1 <= positions[0] <= positions[-1] <= prompt + generated
            and all(kind == arm and total == 48 and split == ready == layers and early == early_layers
                    for kind, _, total, split, ready, early in records), 'actual long-code T4 schedule changed')
    return dict(width4_windows=len(records), early_submissions=len(records) * early_layers)


def routes(text, samples, arm):
    out = primary.routes(text, samples, 'dflash2', workloads=WORKLOADS)
    headers = list(re.finditer(primary.footer.HEADER, text, re.M))
    frame = text[headers[-2].end() if len(headers) > 1 else 0:headers[-1].start()]
    expected = bank.previous.read(ROOT / 'record.json')['split_layers']
    out[WORKLOAD]['t4'] = split_windows(frame, samples[-1], arm, expected)
    return out


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER and all(
            p['complete'] and p['passed'] and p['ownership_closed'] and p['native_exit_code'] == 0
            and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER),
            'the speed screen requires nine independent accepted processes')
    prefill_fields = ('residual_gpu_rows', 'direct_pieces', 'kv', 'bank_h2d_requested_bytes', 'placement_preserved')
    decode_fields = ('generated_tokens', 'drafts_accepted', 'drafts_offered', 'output_sha256')
    records = []
    for offset in range(0, len(ORDER), 3):
        block = {p['arm']: p for p in processes[offset:offset + 3]}
        samples = {arm: p['evidence']['samples'][-1] for arm, p in block.items()}
        calls = {arm: p['routes'][WORKLOAD] for arm, p in block.items()}
        fixed = [[route[k] for k in prefill_fields] for route in calls.values()]
        rates = {arm: p['evidence']['decode_rates'][WORKLOAD] for arm, p in block.items()}
        records.append(dict(tg=rates, matched_prefill=all(entry == fixed[0] for entry in fixed),
            ratios={f'{candidate}/{control}': dict(tg=rates[candidate] / rates[control],
                prefill_time=samples[candidate]['prompt_ms'] / samples[control]['prompt_ms'],
                ttft=samples[candidate]['ttft_s'] / samples[control]['ttft_s'])
                for candidate, control in COMPARISONS},
            same_decode_work={f'{candidate}/{control}': all(samples[candidate][k] == samples[control][k]
                                                           for k in decode_fields)
                              and calls[candidate]['expert_work'] == calls[control]['expert_work']
                              for candidate, control in COMPARISONS}))
    times = {name: dict(tg_geomean=math.exp(statistics.mean(math.log(v['ratios'][name]['tg']) for v in records)),
                        faster=sum(v['ratios'][name]['tg'] > 1 for v in records))
             for name in records[0]['ratios']}
    return dict(blocks=records, summary=times, speed_screen_passed=all(r['matched_prefill'] for r in records)
                and all(times[name]['tg_geomean'] > 1 and times[name]['faster'] >= 2
                        for name in ('early/after-cpu', 'early/stock')),
                timing_only_work_matched=all(r['same_decode_work']['early/after-cpu'] for r in records),
                clean_speed_qualified=False, adopted=False)


def experiment():
    ctx = primary.experiment()
    base_pins, current = ctx.pins, ctx.current
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.environment, ctx.command, ctx.routes = baseline, environment, command, routes
    ctx.evidence_reader.read_complete_suite, ctx.evidence_reader.decide = read_subset, decision
    ctx.protocol = PLAN

    def pins(revision):
        sources = base_pins(revision)
        for name in (THIS, model.THIS, model.arc.THIS, build.THIS, build.fixture.THIS):
            path = bank.P / name
            require(path.read_bytes() == validation.git('show', revision + ':' + name),
                    'commit the exact selective T4 comparison input: ' + name)
            sources[name] = bank.sha(path)
        return sources

    def check_environment(row):
        environments = [{k: v.replace(plan['root'], '<process>') for k, v in environment(row, plan).items()
                         if k != SCHEDULE} for plan in row['plans']]
        require(all(env == environments[0] for env in environments), 'another launch setting varies by schedule')
        return current(row)

    ctx.pins, ctx.current = pins, check_environment
    return ctx


if __name__ == '__main__':
    primary.clock.main(experiment())
