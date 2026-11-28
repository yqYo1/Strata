"""Compare sustained DFlash2 decode with the dedicated long-prefill cache."""

from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_early_continuation as earlier
import dflash2_prefill_ple_horizon_restore as restore
import dflash2_primary_decode_clocks as clocks

primary, bank, basis, require = earlier.primary, earlier.bank, earlier.basis, earlier.require
context, phase, cache, grouped = restore.context, restore.phase, restore.cache, restore.grouped
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_horizon_continuation_v2'
ROOT = bank.B / KEY.replace('_', '-')
CLEAN_ORDER = ('stock', 'dedicated', 'dedicated', 'stock', 'stock', 'dedicated')
OBSERVED = 'dedicated-clocks'
ORDER = (*CLEAN_ORDER[:3], OBSERVED, *CLEAN_ORDER[3:])
WORKLOADS = primary.WORKLOADS
DECODE_DEFAULTS = {
    'STRATA_SYCL_DECODE_T4_SCHEDULE': 'stock',
    'STRATA_SYCL_DECODE_T4_AUDIT': '0',
    'STRATA_SYCL_DECODE_T4_PREFIX_HELP': '0',
    'STRATA_SYCL_DECODE_HANDOFF_MEMORY': '0',
    'STRATA_SYCL_DFLASH2_KV_12288': '0',
}
DECODE_ARGS = {'--kv': 'int8', '--kv-resident': '20480', '--pcie-frac': '0',
               '--spec': '4', '--draft-conf': '0.6', '--max-context': '262144'}
PLAN = dict(order=list(ORDER), input_case=primary.continuation.INPUT_CASE,
    scope='Three opposite-order process pairs on the exact same timed and final-slot checked binary. '
          'Keep the whole-request reader, ordinary cache, decode placement and DFlash2 controls fixed. '
          'Vary the dedicated long-prefill cache. Use the thinking-enabled split-v1 code and prose bodies.',
    observation='Run one independent dedicated-cache decode-clock arm between clean dedicated requests. '
                'Compare exact output, reasoning and native work against two clean dedicated processes. '
                'Both decode latency ratios must be within the predeclared factor of 1.10. Keep the '
                'observed request out of the six clean throughput samples.',
    route='Require actual dedicated or stock PLE selection, whole-request dispatch on both fresh long '
          'requests, unchanged warmups, mode16 and matching prefill and decode work.',
    decision='Retain all paired prompt, TTFT and TG ratios. Initial decode preservation requires matching '
             'prefill and decode work, repeated clean process completions and TG geomean at least one '
             'for each workload. Preserve individual output and acceptance samples.',
    limits='Classify drafts by actual acceptance. A varied-prose case may still leave weak-draft '
           'coverage open. Agent-session and 70 tok/s qualification have separate checks.',
    stop=earlier.PLAN['stop'])


def baseline():
    native, row, measured_sha = restore.prerequisites()
    physical = basis.validation.committed(bank.RESULT, (context.KEY,))
    live = basis.validation.committed(bank.RESULT, (restore.KEY,))
    require(context.wait.retained.closed(live, 'closed')
            and all(live[key] == row[key] for key in ('binary', 'native_source_commit', 'native_build_sha256',
                                                     'boot', 'driver', 'runtime_manifest', 'dflash2_assets'))
            and live['physical_sha256'] == bank.signature(physical)
            and live['timing_sha256'] == measured_sha
            and tuple(live['outcomes']) == restore.ORDER
            and physical['binary'] == native['binaries']['strata_full_context_diagnostic']
            and all(case['passed'] and case['complete'] and case['ownership_closed']
                    and case['native_exit_code'] == 0 and not case['fresh_faults']
                    for arm, case in live['outcomes'].items() if arm != 'host'),
            'require the checked whole-request final slot and paired live RESTORE evidence')
    args = row['validated_dflash2_args']
    require(all(key in args and list(args[args.index(key) + 1:args.index(key) + 2]) == [value]
                for key, value in DECODE_ARGS.items()) and '--no-token-graph' in args,
            'the frozen stock decode precision, placement or DFlash2 policy changed')
    actual_layout = Path(args[args.index('--pack') + 1]) / 'native_experts.txt'
    expected_layout = bank.P / clocks.EXPERT_LAYOUT
    require(actual_layout.read_bytes() == expected_layout.read_bytes(), 'the observed expert manifest changed')
    return dict(row, native_context_qualified=True, physical_context_sha256=bank.signature(physical),
                live_restore_sha256=bank.signature(live), horizon_cache_speed_sha256=measured_sha,
                expert_manifest_sha256=bank.sha(expected_layout))


def environment(row, plan):
    env = earlier.timing.environment(row, dict(plan, arm='early'))
    require(not any(name in env for name in (phase.SELECTOR, grouped.SELECTOR))
            and env.get(grouped.geometry.SELECTOR, '0') == '0',
            'unexpected PLE cache or record observer')
    require(plan['arm'] in ORDER and not any(name in env for name in clocks.adaptive.OBSERVERS)
            and all(env.get(key, value) == value for key, value in DECODE_DEFAULTS.items()),
            'unexpected decode placement, schedule, observer or comparison arm')
    rows = 0 if plan['arm'] == 'stock' else cache.capacity.LARGE_ROWS
    env.update({phase.SELECTOR: str(rows), grouped.SELECTOR: 'all'})
    if plan['arm'] == OBSERVED:
        env.update(dict.fromkeys(clocks.adaptive.OBSERVERS, '1'))
    return env


def routes(text, samples, arm):
    checked = clocks.routes(text, samples, 'clocks' if arm == OBSERVED else 'clean-before')
    dedicated = arm != 'stock'
    expected = (f'{phase.START}long_prefill={cache.capacity.LARGE_ROWS} '
                f'ordinary={cache.capacity.DEFAULT_ROWS}')
    require([line for line in text.splitlines() if line.startswith(phase.START)] == ([expected] if dedicated else [])
            and not any(marker in text for marker in (grouped.PRODUCER, grouped.CONSUMER, phase.TICKET,
                                                      earlier.timing.MARKER, *earlier.timing.OBSERVED_LINES)),
            'dedicated cache startup, grouped dispatch or observer differs')
    end = 0
    for header, sample in zip(re.finditer(primary.footer.HEADER, text, re.M), samples, strict=True):
        frame = text[end:header.start()]
        end = header.end()
        name = sample.get('workload')
        phase.check_first_cache(frame, 'large' if dedicated else 'stock', int(name in WORKLOADS))
        windows = [line for line in frame.splitlines() if line.startswith(grouped.MARKER)]
        require(frame.count(grouped.MARKER) == len(windows), 'malformed grouped dispatch marker')
        if name in WORKLOADS:
            chunk = primary.speed.CHUNK
            depth = (sample['read_tokens'] - 2) // chunk + 1
            require(windows == [f'{grouped.MARKER}tokens_per_chunk={chunk} window_chunks={depth}'],
                    'the full record window must cover every fresh long-prefill chunk')
            checked[name]['ple_record_window_chunks'] = depth
        else:
            require(not windows, 'short warmup unexpectedly used the grouped window')
    require(all(marker not in text[end:] for marker in (phase.FIRST, grouped.MARKER)),
            'unchecked PLE dispatch after the last request')
    return checked


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER, 'require the exact counterbalanced order')
    clean = [p for p in processes if p['arm'] != OBSERVED]
    require(tuple(p['arm'] for p in clean) == CLEAN_ORDER, 'require all six clean throughput processes')
    mapped = [dict(p, arm='early' if p['arm'] == 'dedicated' else 'stock') for p in clean]
    result = earlier.decision(mapped)
    classes = result['observed_draft_classes']
    result['observed_draft_classes'] = dict(stock=classes['stock'], dedicated=classes['early'])
    result['decode_preservation_screen'] = bool(result['decode_preservation_screen']
        and all(row['all_decode_work_matched'] for row in result['summary'].values()))
    comparison = [dict(processes[index], arm=arm) for index, arm in
                  zip((2, 3, 6), clocks.ORDER, strict=True)]
    result['decode_clock_attribution'] = clocks.decision(comparison)
    return result


def experiment():
    ctx = primary.experiment()
    base_pins, check = ctx.pins, ctx.current
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.environment, ctx.routes = baseline, environment, routes
    ctx.evidence_reader.decide, ctx.protocol = decision, PLAN

    def pins(revision):
        values = base_pins(revision) | earlier.timing.experiment().pins(revision)
        for name in (THIS, earlier.THIS, restore.THIS, restore.restore.THIS,
                     context.THIS, context.earlier.THIS, phase.THIS, cache.THIS,
                     cache.capacity.THIS, cache.horizon.THIS, grouped.THIS,
                     clocks.THIS, clocks.adaptive.THIS, clocks.EXPERT_LAYOUT,
                     *(Path(module.__file__).resolve().relative_to(bank.P).as_posix()
                       for module in (clocks.caller, clocks.footer, clocks.census))):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact grouped continuation input: ' + name)
            values[name] = bank.sha(path)
        return values

    def current(row):
        layouts = [{key: value.replace(plan['root'], '<process>')
                    for key, value in ctx.environment(row, plan).items()
                    if key not in (phase.SELECTOR, *clocks.adaptive.OBSERVERS)}
                   for plan in row['plans']]
        require(all(env == layouts[0] for env in layouts), 'another setting changed between the two cache arms')
        return check(row)

    ctx.pins, ctx.current = pins, current
    return ctx


if __name__ == '__main__':
    primary.clock.main(experiment())
