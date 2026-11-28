"""Attribute sustained DFlash2 decode after the selected long-prefill route."""

from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_primary_continuation as primary
import dflash2_v2_decode_clocks as adaptive
import cpu_work_census as census

bank, basis, require, footer = primary.bank, primary.basis, primary.require, primary.footer
caller, clock = adaptive.caller, primary.clock
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_primary_decode_clocks_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('clean-before', 'clocks', 'clean-after')
TIMING = re.compile(footer.TIMING, re.M)
LIMIT, TARGET = adaptive.LIMIT, 70
EXPERT_LAYOUT = 'tests/data/native_experts/iq3_s.txt'
WORK = (*primary.continuation.WORK_FIELDS, 'request_sha256', 'output_sha256', 'reasoning_sha256')
REGION_CANDIDATES = ('draft', 'verify', 'commit_emit', 'GPU_reach_wait', 'CPU_pool_run')


def weight_visit_bounds(jobs, verified_rows, blob_sizes):
    """Bound one full packed-weight visit per job using the layer top-10 cap."""
    cap = verified_rows * 10
    require(len(blob_sizes) == 48 and min(blob_sizes) > 0 and 0 < jobs <= cap * len(blob_sizes),
            'CPU jobs exceed the manifest or verifier-row bounds')

    def allocate(sizes):
        return sum(min(cap, max(0, jobs - layer * cap)) * amount for layer, amount in enumerate(sizes))

    sizes = sorted(blob_sizes)
    return [allocate(sizes), allocate(reversed(sizes))]


def baseline():
    selected = primary.retained.baseline()
    args = selected['validated_dflash2_args']
    manifest = Path(args[args.index('--pack') + 1]) / 'native_experts.txt'
    expected = bank.P / EXPERT_LAYOUT
    require(manifest.read_bytes() == expected.read_bytes(), 'the actual expert pack differs from the pinned layout')
    return dict(selected, expert_manifest_sha256=bank.sha(expected))


def routes(text, samples, arm):
    observing = arm == 'clocks'
    headers = list(re.finditer(footer.HEADER, text, re.M))
    timings = list(TIMING.finditer(text))
    require(arm in ORDER and len(headers) == len(samples)
            and len(timings) == text.count('strata decode timing:') == (len(samples) if observing else 0),
            'unexpected native request or decode timing coverage')
    captured = caller.extract(text, list(range(1, len(samples) + 1)) if observing else ())
    checked = primary.routes(TIMING.sub('', text), samples, 'dflash2')
    if not observing:
        return checked

    blob_sizes = [size for _, _, size in census.manifest_formats(bank.P / EXPERT_LAYOUT)]
    for i, (header, timing, sample, service) in enumerate(zip(headers, timings, samples, captured, strict=True)):
        require((headers[i - 1].end() if i else 0) <= timing.start() < header.start(),
                'decode timing belongs to a different request')
        name = sample.get('workload')
        if name is None:
            continue
        row = checked[name]
        phases = adaptive.decode_clocks(timing.groups(), sample, row['expert_work'])
        ms = phases['wall_ms_per_window']
        require(ms['verify'] + .021 >= sum(ms[k] for k in ('GPU_reach_wait', 'per_layer_host', 'stage'))
                and ms['per_layer_host'] + .026 >= sum(ms[k] for k in (
                    'plan', 'activation_quant', 'jobs', 'CPU_pool_run')),
                'nested verifier timings exceed the enclosing wall interval')
        emitted = sample['generated_tokens']
        budget = 1000 * emitted / (TARGET * phases['windows'])
        weights = weight_visit_bounds(row['expert_work']['cpu_unique_misses'], phases['target_rows'], blob_sizes)
        row['decode_clocks'] = dict(phases, target_ms_per_window=budget,
                                   remaining_after_draft_ms_per_window=budget - ms['draft'],
                                   packed_weight_bounds_bytes=weights,
                                   logical_weight_GBps_at_70=[w * TARGET / emitted / 1e9 for w in weights])
        row['caller_decode_service'] = caller.compact(service)
    return checked


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(ORDER)
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in processes),
            'require three independently closed decode-clock processes')
    result = {}
    for name in primary.WORKLOADS:
        samples = [next(s for s in p['evidence']['samples'] if s.get('workload') == name) for p in processes]
        paths = [p['routes'][name] for p in processes]
        work = [[s[k] for k in WORK] for s in samples]
        copy = [r['expert_work'] for r in paths]
        matched = work[0] == work[1] == work[2] and copy[0] == copy[1] == copy[2]
        clocks = [s['decode_ms'] for s in samples]
        ratios = [clocks[1] / clocks[i] for i in (0, 2)]
        observed = paths[1]['decode_clocks']
        eligible = matched and all(1 / LIMIT <= v <= LIMIT for v in ratios)
        result[name] = dict(matched_work=matched, observed_over_clean=ratios,
                            eligible=eligible,
                            clean_tg=[processes[i]['evidence']['decode_rates'][name] for i in (0, 2)],
                            target_ms_per_window=observed['target_ms_per_window'],
                            remaining_after_draft_ms_per_window=observed['remaining_after_draft_ms_per_window'],
                            logical_weight_GBps_at_70=observed['logical_weight_GBps_at_70'])
        if eligible:
            # Give each region its full printed duration plus rounding, independently.
            # A nested or overlapped region cannot promise that much wall-time savings.
            gap = (clocks[1] - .05) / observed['windows'] - observed['target_ms_per_window']
            wall = observed['wall_ms_per_window']
            result[name]['remaining_gap_ms_per_window_after_removing_one_region'] = {
                region: max(0.0, gap - wall[region] - .005) for region in REGION_CANDIDATES}
    return dict(workloads=result, adopted=False)


def experiment():
    ctx = primary.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    old_environment, old_pins, old_current = ctx.environment, ctx.pins, ctx.current

    def environment(row, plan):
        env = old_environment(row, plan)
        require(not any(key in env for key in (*adaptive.OBSERVERS, 'STRATA_TRACE', 'STRATA_DRAFT_TIMING'))
                and all(env.get(key, '0') == '0' for key in ('STRATA_CPU_SERVICE', 'STRATA_CPU_WORK_CENSUS')),
                'unexpected inherited decode observer')
        if plan['arm'] == 'clocks':
            env.update(dict.fromkeys(adaptive.OBSERVERS, '1'))
        return env

    def pins(revision):
        sources = old_pins(revision)
        modules = (sys.modules[__name__], adaptive, caller, footer, census)
        paths = [Path(module.__file__).resolve() for module in modules]
        for path in (*paths, bank.P / EXPERT_LAYOUT):
            name = path.relative_to(bank.P).as_posix()
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact decode clock input: ' + name)
            sources[name] = bank.sha(path)
        return sources

    def current(row):
        environments = [{k: v.replace(plan['root'], '<process>')
                         for k, v in environment(row, plan).items() if k not in adaptive.OBSERVERS}
                        for plan in row['plans']]
        require(len(environments) == len(ORDER) and all(v == environments[0] for v in environments),
                'a setting outside the decode observers changed')
        return old_current(row)

    ctx.baseline, ctx.environment, ctx.pins, ctx.current, ctx.routes = baseline, environment, pins, current, routes
    ctx.evidence_reader.decide = decision
    ctx.protocol = dict(primary.PLAN, order=list(ORDER),
        scope='Use the same thinking-on code and prose workloads after the selected mode16 long prefill. '
              'Bracket the two decode observers with independent clean processes on the same build.',
        decision='Check exact native work and both paired latency ratios within a factor of 1.10 per workload. '
                 'Report observed draft, verification, commit and caller clocks against the 70 tok/s budget. '
                 'For an eligible workload, report the remaining gap with each measured region removed '
                 'independently, holding verifier yield and all other measured work fixed.',
        limits='Use rounded request clocks and conserved caller regions. GPU waits and CPU service are nested '
               'inside verification. The individual region removals are optimistic and nonadditive. '
               'Packed-weight visits bound model work. Physical DRAM traffic and '
               'arithmetic ceilings need separate evidence.')
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
