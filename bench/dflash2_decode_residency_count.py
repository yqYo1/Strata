"""Measure legacy DFlash2 routed-row counting with adaptive swaps held idle."""

import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
import dflash2_width4_wait_attribution as wait

reference, leaves = wait.reference, wait.leaves
bank, basis, require = wait.bank, wait.basis, wait.require
KEY = 'dflash2_decode_residency_count_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
TRACE = 'STRATA_TRACE_ADAPT'
INTERVAL, REQUESTS, OUTPUT_CAP = 1_000_000, 5, 16_384
ORDER = ('off', 'on', 'on', 'off')
OVERHEAD_FACTOR = 1.05
SWAPS, EVERY = '--adapt-swaps', '--adapt-every'
PRODUCERS = ('src/program/generate.cpp', 'src/core/expert_source.cpp')


def baseline():
    row = wait.baseline()
    root = leaves.speed.build.DFLASH2_KV_ROOT
    record = bank.previous.read(root / 'record.json')
    require(record['binaries']['strata'] == row['binary']
            and bank.signature(bank.builder.compact(record)) == row['native_build_sha256']
            and REQUESTS * OUTPUT_CAP < INTERVAL,
            'bind the measured engine and the process-wide no-refill bound')
    # wait.baseline() verifies the original projection, including overlays, by
    # reconstructing the retained engine's source and matching its build inputs.
    compiled = {}
    for name in PRODUCERS:
        producer = root / 'source' / name
        value = bank.sha(producer)
        require(value == record['source_files'][name]
                == record['actual_inputs']['compiled'][str(producer)],
                'retained count producer was not compiled from the checked projection')
        compiled[name] = value
    return dict(row, residency_count_producers=compiled,
                residency_count_projection=record['projection']['dflash2_kv_source'])


def configuration(row, plan):
    cfg = reference.configuration(row, plan)
    args, overrides = list(cfg['args']), list(cfg['user_args'])
    require(plan['arm'] in ORDER and args.count(SWAPS) == 1 and args[args.index(SWAPS) + 1] == '0'
            and EVERY not in args and SWAPS not in overrides and EVERY not in overrides
            and not any(a.startswith(('--resident', '--expert-profile-save', '--peer-', '--layer-split')) for a in args),
            'freeze the retained file-backed non-adaptive route')
    requested = str(int(plan['arm'] == 'on'))
    args[args.index(SWAPS) + 1] = requested
    args.extend((EVERY, str(INTERVAL)))
    out = dict(cfg, args=args)
    server = reference.confidence.model.server
    resolved = server.engine_args(out, exe=out['exe'], announce=False)
    require(resolved.count(SWAPS) == resolved.count(EVERY) == 1
            and resolved[resolved.index(SWAPS) + 1] == requested
            and resolved[resolved.index(EVERY) + 1] == str(INTERVAL),
            'the two requested native arguments were not resolved once')
    other = list(resolved)
    interval_index = other.index(EVERY)
    del other[interval_index:interval_index + 2]
    other[other.index(SWAPS) + 1] = '0'
    require(other == cfg['args'], 'another effective native argument differs')
    out['args'] = resolved
    require(server.engine_args(out, exe=out['exe'], announce=False) == resolved,
            'the resulting launch is not idempotent')
    return out


def environment(row, plan):
    env = wait.environment(row, dict(plan, arm='clean-before'))
    require(TRACE not in env, 'unexpected adaptive logging in the original environment')
    return dict(env, **{TRACE: '1'})


def routes(text, samples, arm):
    require(arm in ORDER and len(samples) == REQUESTS
            and all(0 <= s['generated_tokens'] <= OUTPUT_CAP for s in samples)
            and 'strata serve: ADAPT_' not in text
            and 'resident RAM mode:' not in text,
            'a refill occurred or the file-backed request bound changed')
    return wait.routes(text, samples, 'clean-before')


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER and all(p['complete'] and p['passed']
            and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(ORDER), 'finish four independent native processes')
    _, expected = wait.measured_reference()
    matched = all(wait.work_signature(p) == expected for p in processes)
    samples = [p['evidence']['samples'][-1] for p in processes]
    ms = [s['decode_ms'] for s in samples]
    require(all(math.isfinite(x) and x > 0 for x in ms), 'native decode clocks required')
    ratios = [ms[1] / ms[0], ms[2] / ms[3]]
    rates = [1000 * s['generated_tokens'] / v for s, v in zip(samples, ms, strict=True)]
    return dict(matched_original_work=matched, on_over_off_latency=ratios,
                geometric_mean_latency_ratio=statistics.geometric_mean(ratios),
                bounded_count_overhead=matched and all(1 / OVERHEAD_FACTOR <= r <= OVERHEAD_FACTOR for r in ratios),
                decode_tok_s=rates, observed_refills=0, adopted=False)


def experiment():
    ctx = leaves.experiment()
    prior_pins = ctx.pins
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment, ctx.routes = baseline, configuration, environment, routes
    ctx.evidence_reader.decide = decision
    ctx.protocol = dict(order=list(ORDER), workload=leaves.WORKLOAD, adaptive_interval=INTERVAL,
        proposed_output_bound=REQUESTS * OUTPUT_CAP, paired_overhead_factor=OVERHEAD_FACTOR,
        prior_prelaunch_refusal=dict(source='395aee434cea05c3ffe7bc50bf4aac1ec69b5b10',
            failed_producer='src/program/generate.cpp', native_execution_attempted=False,
            reason='The original reader compared the projected producer against the nominal source commit.'),
        unit='Hold the effective adaptive interval and trace fixed. Toggle only routed-row counting before exec.',
        evidence='Use the retained regular-code body and its four warmups. Keep exact native work, output and route. '
                 'Require zero adaptive submits or commits and normal GPU/owner closure in every process.',
        interpretation='Two reversed pairs bound counting overhead in the existing route. '
                       'This covers the whole process. A later bounded decode-only counter and any cache swaps '
                       'need their own comparisons. No new engine source or model adoption.',
        stop='Stop on a changed input, route, ownership, budget, provider or driver fault. Preserve all collected rates.')

    def pins(revision):
        values = prior_pins(revision)
        for name in (THIS, wait.THIS, reference.THIS, reference.confidence.THIS):
            p = bank.P / name
            require(p.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit exact residency count inputs: ' + name)
            values[name] = bank.sha(p)
        return values

    ctx.pins = pins
    return ctx


if __name__ == '__main__':
    reference.clock.main(experiment())
