"""Compare decode placement on a fixed measured per-layer, whole-horizon PLE route."""

import math
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_decode_layer_placement_check as checked
import dflash2_prefill_ple_horizon_continuation as horizon

primary = horizon.primary
build, bank, require, validation = checked.build, checked.bank, checked.require, checked.validation
KEY = 'dflash2_decode_layer_placement_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = 'bench/dflash2_decode_layer_placement_speed.py'
WORKLOAD_REFERENCE = 'b7b333ebdc7fe1842cdf2d9baf8451a7e45a7eca'
ORDER = ('stock', 'blend', 'blend', 'stock', 'stock', 'blend')
EVENT = re.compile(checked.EVENT.pattern.removesuffix('audit=1$') + r'audit=0$', re.M)
PREFILL = ('residual_gpu_rows', 'direct_pieces', 'kv', 'bank_h2d_requested_bytes', 'placement_preserved',
           'ple_record_window_chunks')
PLAN = dict(primary.PLAN, order=list(ORDER),
    scope='One coherent executable and measured initial layer-local placement. Fix the dedicated PLE cache '
          'and whole-request record window for both arms. Alternate stock and delayed placement in three '
          'counterbalanced process pairs with the same two 96K thinking bodies.',
    timing='First nonempty client output precedes the one-time preparation and installation. Charge both transitions '
           'and preparation to candidate decode. Charge final commit retirement to both arms. Record native '
           'prefill and client TTFT separately. Reconcile native verification width counts with target rows and drafts '
           'to report time and output per window and the 70 tok/s window budget at observed yield. Record requested '
           'H2D bytes and the wall spans of preparation, installation and restoration. Estimate a break-even output '
           'length after the observed stock prefix, assuming its per-token cost matches the paired stock arm. The estimate '
           'assumes those unit costs and output paths persist. Bound logical CPU packed-weight visits using the '
           'pinned expert layout and the actual verified rows. Physical bus and RAM traffic require separate evidence.',
    decision='Require equal fresh prompt and logical prefill work, at least two decode wins and a geometric mean '
             'rate ratio above one per workload, and paired median prefill and TTFT ratios at most one. Placement may change '
             'the output and the accepted drafts. Require separate task, weak-draft, agent and full-context '
             'qualification before adoption.',
    stop='Stop on the first source, runtime, ownership, workload, actual route or validation failure.')


def baseline():
    native = bank.previous.read(build.ROOT / 'record.json')
    build.context().check(native)
    proof = validation.committed(bank.RESULT, (checked.KEY,))
    binary = native['binaries']['strata']
    identity = bank.signature(build.builder.compact(native))
    require(build.grouped.wait.retained.closed(proof, 'closed') and proof['build_sha256'] == identity
            and proof['native_source_commit'] == native['native_source_commit']
            and proof['prefill_build_sha256'] == native['prefill_build_sha256']
            and tuple(proof['outcomes']) == checked.ORDER
            and all(p['passed'] and p['ownership_closed'] and p['final_assets_verified'] and not p['fresh_faults']
                    and p['native_exit_code'] == (1 if arm == 'copy-failure' else 0)
                    for arm, p in proof['outcomes'].items())
            and [proof['outcomes']['copy-failure']['copy_failure'][key]
                 for key in ('first_slot', 'copied_bytes')] == proof['first_copy']
            and proof['outcomes']['copy-failure']['copy_failure']['masked']
            and proof['outcomes']['copy-failure']['copy_failure']['drained']
            and all(proof['cases'][arm]['binary'] == binary for arm in ('stock', 'blend'))
            and proof['outcomes']['blend']['requests'][1]['placement_preparation']['slots'] == checked.GEOMETRY[0]
            and all(len(proof['outcomes'][arm]['requests']) == len(proof['requests'])
                    and all(r['verify_windows'] > 0 and r['verified_rows'] >= r['verify_windows']
                            for r in proof['outcomes'][arm]['requests']) for arm in ('stock', 'blend'))
            and proof['profiles'] == native['profiles']
            and all(bank.sha(p['path']) == p['sha256'] for p in proof['profiles'].values()),
            'require closed same-build placement, cancellation, restore and short output checks')
    row = horizon.baseline()
    require(proof['model_assets_sha256'] == bank.signature(row['model_assets'])
            and proof['drafter_assets_sha256'] == bank.signature(row['dflash2_assets']),
            'the placement check must qualify the same target and drafter assets as the speed run')
    reference = bank.previous.historic(WORKLOAD_REFERENCE, horizon.KEY)
    control = reference['processes'][2]
    require(control['arm'] == 'dedicated' and control['complete'] and control['passed']
            and control['ownership_closed'] and control['native_exit_code'] == 0 and not control['fresh_faults']
            and reference['binary'] == row['binary']
            and reference['native_source_commit'] == build.PREIMAGE
            and reference['native_build_sha256'] == row['native_build_sha256'] == native['prefill_build_sha256'],
            'require the closed clean same-PLE workload from the reference commit')
    requests = {}
    for name in primary.WORKLOADS:
        sample, = [s for s in control['evidence']['samples'] if s.get('workload') == name]
        require(sample['reused_tokens'] == 0 and sample['generated_tokens'] >= 1024,
                'the reference needs a fresh long request and sustained output')
        requests[name] = {key: sample[key] for key in ('read_tokens', 'request_sha256')}
    row['base_physical_context_sha256'] = row.pop('physical_context_sha256')
    row['base_live_restore_sha256'] = row.pop('live_restore_sha256')
    return dict(row, binary=binary, native_source_commit=native['native_source_commit'],
                native_build_sha256=identity, native_context_qualified=False,
                placement_validation_sha256=bank.signature(proof), qualified_profiles=proof['profiles'],
                archived_requests=requests)


def environment(row, plan):
    require(plan['arm'] in ORDER, 'unexpected placement arm')
    env = horizon.environment(row, dict(plan, arm='dedicated'))
    require(not any(name in env for name in (checked.SELECTOR, checked.CLOCK, checked.AUDIT, checked.FAULT)),
            'unexpected placement setting in the measured template')
    env[checked.CLOCK] = '1'
    if plan['arm'] == 'blend':
        env[checked.SELECTOR] = row['qualified_profiles']['blend']['path']
    return env


def routes(text, samples, arm):
    row = bank.previous.read(ROOT / 'record.json')
    clocks = horizon.clocks
    manifest = bank.P / clocks.EXPERT_LAYOUT
    require(bank.sha(manifest) == row['expert_manifest_sha256'], 'the qualified expert layout changed')
    blob_sizes = [size for _, _, size in clocks.census.manifest_formats(manifest)]
    require('decode phase profile inactive' not in text, 'placement admission failed')
    require(tuple(s['workload'] for s in samples if s.get('workload') in primary.WORKLOADS) == primary.WORKLOADS,
            'require the original code and prose order')
    report = horizon.routes(text, samples, 'dedicated')
    start = 0
    for header, sample in zip(re.finditer(primary.footer.HEADER, text, re.M), samples, strict=True):
        frame = text[start:header.start()]
        start = header.end()
        events = EVENT.findall(frame)
        preparation = checked.PREPARE.findall(frame)
        name = sample.get('workload')
        switched = arm == 'blend' and name in primary.WORKLOADS
        expected_prepare = switched and name == primary.WORKLOADS[0]
        require([e[0] for e in events] == (['blend', 'stock'] if switched else [])
                and len(events) == frame.count('strata decode placement:')
                and len(preparation) == frame.count('strata decode placement prepare:') == int(expected_prepare),
                'actual preparation or placement differs from plan')
        if name not in primary.WORKLOADS:
            continue
        require(all(sample[k] == v for k, v in row['archived_requests'][name].items()), 'sustained request body changed')
        require(sample['decode_ms'] > 0, 'require a positive native decode interval')
        report[name].update(checked.verification_work(
            frame, sample['generated_tokens'], sample['drafts_accepted'], sample['drafts_offered']))
        route = report[name]
        weights = clocks.weight_visit_bounds(route['expert_work']['cpu_unique_misses'], route['verified_rows'], blob_sizes)
        route['logical_cpu_weight_gbps_at_70'] = [amount * 70 / sample['generated_tokens'] / 1e9 for amount in weights]
        report[name]['transitions'] = []
        if switched:
            install, restored = [int(e[1]) for e in events]
            when = [int(e[4]) for e in events]
            ms = [float(e[5]) for e in events]
            interval = sample['client_interval']
            first, end = (interval[k] for k in ('first_output_monotonic_ns', 'request_end_monotonic_ns'))
            prepare_ms = 0.0
            if preparation:
                (emitted, slots, prepare_ns, elapsed), = preparation
                prepare_ms = float(elapsed)
                prepare_start = int(prepare_ns)
                prepare_done = prepare_start + round(prepare_ms * 1_000_000)
                require(int(emitted) == install and int(slots) == checked.GEOMETRY[0]
                        and first < prepare_start < when[0] and prepare_ms > 0 and prepare_done <= when[0] + 500,
                        'profile preparation must be timed after first client output and before installation')
                report[name]['placement_preparation'] = dict(wall_ms=prepare_ms,
                    after_first_ms=(prepare_start - first) / 1e6)
            # Native wall_ms is rounded to the nearest microsecond.
            install_done = when[0] + round(ms[0] * 1_000_000)
            restore_done = when[1] + round(ms[1] * 1_000_000)
            require(32 <= install <= 35 and install < restored == sample['generated_tokens']
                    and first < when[0] < when[1] < end and restore_done <= end + 500
                    and all(value > 0 for value in ms) and install_done <= when[1] + 500
                    and 0 < prepare_ms + sum(ms) < sample['decode_ms']
                    and all(tuple(map(int, e[2:4])) == checked.GEOMETRY for e in events),
                    'transition extent, client ordering or charged decode time changed')
            report[name]['transitions'] = [dict(profile=e[0], emitted=int(e[1]), wall_ms=float(e[5]),
                                                requested_h2d_bytes=int(e[3]),
                                                after_first_ms=(int(e[4]) - first) / 1e6) for e in events]
    require(not any(prefix in text[start:] for prefix in ('strata decode placement:',
                                                          'strata decode placement prepare:',
                                                          checked.WORK_PREFIX)),
            'unexpected placement data outside the request')
    return report


def decode_work(sample, route):
    emitted, windows = sample['generated_tokens'], route['verify_windows']
    decode_ms = sample['decode_ms']
    verified_rows = route['verified_rows']
    cpu_jobs = route['expert_work']['cpu_unique_misses']
    return dict(emitted=emitted, accepted=sample['drafts_accepted'], offered=sample['drafts_offered'],
                verify_windows=windows, verified_rows=verified_rows, windows_by_width=route['windows_by_width'],
                decode_ms_per_window=decode_ms / windows,
                tokens_per_window=emitted / windows,
                window_budget_70_ms=1000 * emitted / (70 * windows),
                cpu_jobs_per_output=cpu_jobs / emitted, cpu_jobs_per_verified_row=cpu_jobs / verified_rows)


def placement_cost(stock, blend, route):
    """Compute a fixed-unit-cost estimate from the three disjoint charged spans."""
    preparation_ms = route.get('placement_preparation', {}).get('wall_ms', 0.0)
    install, restore = route['transitions']
    fixed_ms = preparation_ms + install['wall_ms'] + restore['wall_ms']
    stock_prefix = install['emitted']
    blend_suffix = blend['generated_tokens'] - stock_prefix
    require(0 < fixed_ms < blend['decode_ms'] and blend_suffix > 0,
            'placement costs require a charged decode interval and generated suffix')
    stock_ms_per_output = stock['decode_ms'] / stock['generated_tokens']
    estimated_ms_per_output = (blend['decode_ms'] - fixed_ms - stock_prefix * stock_ms_per_output) / blend_suffix
    break_even = None
    if 0 < estimated_ms_per_output < stock_ms_per_output:
        break_even = stock_prefix + math.ceil(fixed_ms / (stock_ms_per_output - estimated_ms_per_output))
    return dict(measured_fixed_ms=fixed_ms,
                requested_h2d_bytes=install['requested_h2d_bytes'] + restore['requested_h2d_bytes'],
                fixed_ms_per_output=fixed_ms / blend['generated_tokens'],
                stock_ms_per_output=stock_ms_per_output,
                estimated_blend_suffix_ms_per_output=estimated_ms_per_output,
                fixed_unit_cost_break_even_tokens=break_even)


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER),
            'six independent clean placement processes are required')
    pairs = {name: [] for name in primary.WORKLOADS}
    for left, right in zip(processes[::2], processes[1::2], strict=True):
        stock, blend = (left, right) if left['arm'] == 'stock' else (right, left)
        for name in pairs:
            a, b = [next(s for s in p['evidence']['samples'] if s.get('workload') == name) for p in (stock, blend)]
            control, candidate = [p['routes'][name] for p in (stock, blend)]
            rates = [p['evidence']['decode_rates'][name] for p in (stock, blend)]
            pairs[name].append(dict(tg=rates, tg_ratio=rates[1] / rates[0],
                prefill_time_ratio=b['prompt_ms'] / a['prompt_ms'], ttft_ratio=b['ttft_s'] / a['ttft_s'],
                matched_prefill=all(a[k] == b[k] for k in ('request_sha256', 'read_tokens'))
                    and all(control[k] == candidate[k] for k in PREFILL),
                placement_cost=placement_cost(a, b, candidate),
                actual_work=[decode_work(a, control), decode_work(b, candidate)]))
    summary = {}
    for name, samples in pairs.items():
        summary[name] = dict(tg_geomean=statistics.geometric_mean(p['tg_ratio'] for p in samples),
            tg_wins=sum(p['tg_ratio'] > 1 for p in samples), matched_prefill=all(p['matched_prefill'] for p in samples),
            median_prefill_time_ratio=statistics.median(p['prefill_time_ratio'] for p in samples),
            median_ttft_ratio=statistics.median(p['ttft_ratio'] for p in samples))
    eligible = all(r['matched_prefill'] and r['tg_wins'] >= 2 and r['tg_geomean'] > 1
                   and r['median_prefill_time_ratio'] <= 1 and r['median_ttft_ratio'] <= 1 for r in summary.values())
    classify = primary.continuation.suite.official.d.client.long_generation.classify_pair
    classes = {}
    for arm in ('stock', 'blend'):
        samples = [s for p in processes if p['arm'] == arm for s in p['evidence']['samples']]
        cases = {name: dict(requests=[s for s in samples if s.get('workload') == name])
                 for name in primary.WORKLOADS}
        classes[arm] = classify(cases, len(ORDER) // 2)
    return dict(pairs=pairs, summary=summary, eligible_for_qualification=eligible,
                observed_draft_classes=classes, adopted=False)


def experiment():
    ctx = primary.experiment()
    base_pins, current = horizon.experiment().pins, ctx.current
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.environment = baseline, PLAN, environment
    ctx.routes, ctx.evidence_reader.decide = routes, decision

    def pins(revision):
        values = base_pins(revision)
        for path in (THIS, checked.THIS, checked.stock.THIS, build.THIS):
            file = bank.P / path
            require(file.read_bytes() == validation.git('show', revision + ':' + path),
                    'commit the exact placement timing input: ' + path)
            values[path] = bank.sha(file)
        return values

    def checked_environment(row):
        envs = []
        for plan in row['plans']:
            envs.append({k: v.replace(plan['root'], '<process>')
                         for k, v in ctx.environment(row, plan).items() if k != checked.SELECTOR})
        require(all(env == envs[0] for env in envs),
                'only the decode phase placement profile may differ between arms')
        return current(row)

    ctx.pins, ctx.current = pins, checked_environment
    return ctx


if __name__ == '__main__':
    primary.clock.main(experiment())
