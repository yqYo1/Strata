"""Compare sustained DFlash2 generation with and without early PLE prefill."""

from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_early_restore as restore
import dflash2_primary_continuation as primary

bank, basis, require = primary.bank, primary.basis, primary.require
capacity, timing = restore.capacity, restore.timing
SELECTOR = capacity.SELECTOR
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_early_continuation_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = timing.ORDER
WORKLOADS = primary.WORKLOADS
PREFILL_WORK = ('residual_gpu_rows', 'direct_pieces', 'kv', 'bank_h2d_requested_bytes', 'placement_preserved')
DECODE_WORK = ('generated_tokens', 'drafts_accepted', 'drafts_offered')
PLAN = dict(order=list(ORDER), input_case=primary.continuation.INPUT_CASE,
    scope='Compare the same final-slot and live-RESTORE checked engine in three counterbalanced pairs. '
          'Change only early PLE prefill scheduling. Use the maintained thinking-enabled split-v1 long '
          'code and prose workloads, with the same decode placement and DFlash2 settings.',
    route='Check the native first PLE consumption for both long requests and identical prefill work '
          'for each matched body. Keep the actual decode work and acceptance for every sample.',
    decision='Report first and subsequent prompt time, client TTFT and paired decode ratios for each '
             'workload. Require matched prefill and geometric mean TG ratios at least one for both '
             'workloads to pass the decode preservation screen. Retain decode work differences for '
             'attribution. Isolate a repeatable decrease before adoption.',
    limits='Draft class follows observed acceptance. Weak-draft and agent-session qualification remain separate.',
    stop='Stop on input, driver, route, ownership or validation failure. Preserve closed samples.')


def baseline(native_context):
    native, row, measured_sha = capacity.prerequisites(native_context)
    physical = basis.validation.committed(bank.RESULT, (capacity.KEY,))
    live = basis.validation.committed(bank.RESULT, (restore.KEY,))
    retained = timing.wait.retained
    require(retained.closed(physical, 'closed') and retained.closed(live, 'closed')
            and physical['outcomes']['full-copy']['final_slot_consumed']
            and physical['binary'] == native['binaries']['strata_full_context_diagnostic']
            and physical['native_build_sha256'] == row['native_build_sha256']
            and all(live[k] == row[k] for k in ('binary', 'native_source_commit', 'native_build_sha256',
                                                'boot', 'driver', 'runtime_manifest', 'dflash2_assets'))
            and live['physical_sha256'] == bank.signature(physical)
            and live['timing_sha256'] == measured_sha
            and tuple(live['outcomes']) == restore.ORDER
            and all(case['passed'] and case['complete'] and case['ownership_closed']
                    and case['native_exit_code'] == 0 and not case['fresh_faults']
                    for arm, case in live['outcomes'].items() if arm != 'host'),
            'require all early PLE safety and timing results on the exact launch build')
    return dict(row, native_context_qualified=True, physical_context_sha256=bank.signature(physical),
                live_restore_sha256=bank.signature(live), early_speed_sha256=measured_sha)


def routes(text, samples, arm):
    checked = primary.routes(text, samples, 'dflash2')
    end = 0
    for header, sample in zip(re.finditer(primary.footer.HEADER, text, re.M), samples, strict=True):
        frame = text[end:header.start()]
        end = header.end()
        name = sample.get('workload')
        expected = [timing.check.CONSUMED] if arm == 'early' and name in WORKLOADS else []
        markers = [line for line in frame.splitlines() if line.startswith(timing.MARKER)]
        require(markers == expected and frame.count(timing.MARKER) == len(markers)
                and not any(prefix in frame for prefix in timing.OBSERVED_LINES),
                'unexpected PLE first-chunk schedule or timing observer')
        if name in WORKLOADS:
            checked[name]['ple_early_reads'] = len(markers)
    require(timing.MARKER not in text[end:], 'early read outside a checked request')
    return checked


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(ORDER), 'six independent complete early PLE processes required')
    arms = ('stock', 'early')
    records = {name: [] for name in WORKLOADS}
    for left, right in zip(processes[::2], processes[1::2], strict=True):
        control, candidate = (left, right) if left['arm'] == 'stock' else (right, left)
        for name in WORKLOADS:
            samples = [next(s for s in p['evidence']['samples'] if s.get('workload') == name)
                       for p in (control, candidate)]
            a, b = samples
            before, after = (p['routes'][name] for p in (control, candidate))
            rates = [p['evidence']['decode_rates'][name] for p in (control, candidate)]
            records[name].append(dict(
                pp_time_ratio=b['prompt_ms'] / a['prompt_ms'], ttft_ratio=b['ttft_s'] / a['ttft_s'],
                tg=rates, tg_ratio=rates[1] / rates[0],
                matched_prefill=all(a[k] == b[k] for k in ('request_sha256', 'read_tokens'))
                    and all(before[k] == after[k] for k in PREFILL_WORK),
                matched_decode_work=all(a[k] == b[k] for k in DECODE_WORK)
                    and before['expert_work'] == after['expert_work']))
    summary = {}
    classify = primary.continuation.suite.official.d.client.long_generation.classify_pair
    for name, pairs in records.items():
        summary[name] = dict(tg_geomean=statistics.geometric_mean(pair['tg_ratio'] for pair in pairs),
                             all_prefill_matched=all(pair['matched_prefill'] for pair in pairs),
                             all_decode_work_matched=all(pair['matched_decode_work'] for pair in pairs))
    classes = {}
    for arm in arms:
        samples = [s for p in processes if p['arm'] == arm for s in p['evidence']['samples']]
        cases = {name: dict(requests=[s for s in samples if s.get('workload') == name]) for name in WORKLOADS}
        classes[arm] = classify(cases, len(ORDER) // 2)
    return dict(pairs=records, summary=summary, observed_draft_classes=classes,
                decode_preservation_screen=all(s['all_prefill_matched'] and s['tg_geomean'] >= 1
                                               for s in summary.values()),
                adopted=False)


def experiment():
    native = timing.build.context()
    ctx = primary.experiment()
    primary_pins, timing_pins, current = ctx.pins, timing.experiment().pins, ctx.current
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.environment, ctx.routes = lambda: baseline(native), timing.environment, routes
    ctx.evidence_reader.decide, ctx.protocol = decision, PLAN

    def pins(revision):
        sources = primary_pins(revision) | timing_pins(revision)
        for name in (THIS, capacity.THIS, restore.THIS):
            file = bank.P / name
            require(file.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact early PLE continuation input: ' + name)
            sources[name] = bank.sha(file)
        return sources

    def check_environment(row):
        envs = [{k: v.replace(plan['root'], '<process>') for k, v in ctx.environment(row, plan).items()
                 if k != SELECTOR} for plan in row['plans']]
        require(all(env == envs[0] for env in envs), 'another setting changes between PLE arms')
        return current(row)

    ctx.pins, ctx.current = pins, check_environment
    return ctx


if __name__ == '__main__':
    primary.clock.main(experiment())
