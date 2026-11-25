"""Measure the isolated width-five Q6 head on the official sustained workload."""

from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_c5_head_probe as probe
import dflash2_width_generation as width

bank, basis, require = width.bank, width.basis, width.require
snapshot = width.snapshot
KEY = 'dflash2_c5_head_generation_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('reference', 'c5', 'c5', 'reference', 'reference', 'c5')
PROTOCOL = dict(order=list(ORDER), workload='regular-code',
    only_factor='Use one executable and width five throughout. Change only the decode Q6 C5 head selector. '
                'Keep the native placement, cache, drafter confidence, prefill strategy and task fixed.',
    evidence='Independent processes run the same four warmups and one fresh sustained thinking-enabled request. '
             'Require 32768 or more fresh inputs, at least 1024 native outputs, a visible answer, actual native '
             'width-five capture counts, and exact conservation of verifier and expert work. Keep PP and TTFT separate.',
    decision='The head advances only when all three ordered pairs have matched work and lower candidate decode time. '
             'Preserve each original time and output difference. Draft acceptance, wider-vs-narrower width, '
             'weak acceptance, agent tasks, full context, RESTORE and default adoption are separate.',
    stop='Stop at the first source, workload, provider, route, ownership, or device fault mismatch.')


def admission():
    value = basis.validation.committed(bank.RESULT, (probe.KEY,))
    require(value['stage'] == 'closed' and value['complete'] and value['passed'] and value['ownership_closed']
            and tuple(value['outcomes']) == probe.ORDER and all(
                r['passed'] and r['ownership_closed'] and r['native_exit_code'] == 0 and not r['fresh_faults']
                for r in value['outcomes'].values()), 'both native head probes must complete')
    return value


def baseline():
    row = width.baseline()
    prior = basis.validation.committed(bank.RESULT, (width.KEY,))
    require(prior['stage'] == 'closed' and prior['complete'] and prior['ownership_closed'] and not prior['active']
            and [p['arm'] for p in prior['processes']] == list(width.ORDER),
            'close the unchanged fixed-width control comparison')
    gate = admission()
    compiled = bank.previous.read(probe.build.ROOT / 'record.json')
    probe.build.context().check(compiled)
    plan = gate['plan']
    require(plan['binary'] == compiled['binaries']['strata'] and plan['native_source_commit'] == compiled['native_source_commit']
            and plan['native_build_sha256'] == bank.signature(probe.build.builder.compact(compiled))
            and plan['driver'] == row['driver'] and plan['boot'] == row['boot'] and plan['runtime_manifest'] == row['runtime_manifest']
            and plan['model_assets_sha256'] == bank.signature(row['model_assets'])
            and plan['drafter_assets'] == row['dflash2_assets'], 'the traced model probe must bind this build and runtime')
    head_arm = {k: plan[k] for k in ('binary', 'native_source_commit', 'native_build_sha256')}
    return dict(row, c5_head_arm=head_arm, c5_probe_sha256=bank.signature(gate), fixed_width_sha256=bank.signature(prior))


def configuration(row, plan):
    return width.configuration(row, dict(plan, arm='5'))


def environment(row, plan):
    env = width.environment(row, dict(plan, arm='5'))
    require(probe.FLAG not in env and plan['arm'] in probe.ORDER, 'the C5 head must be selected only by the arm')
    return dict(env, **{probe.FLAG: str(int(plan['arm'] == 'c5'))})


def routes(text, samples, arm):
    summary = width.routes(text, samples, '5')
    captures = [dict(reference=int(ref), columns=int(cols)) for ref, cols in probe.DECODE_C5.findall(text)]
    predicted = admission()['outcomes'][arm]['capture_counts']
    require(captures and all(v == predicted for v in captures), 'the real width-five graph differs from the audited head route')
    return dict(summary, head_capture_counts=predicted, verified_head_captures=len(captures))


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER and len({
        (p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER)
        and all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                and not p['fresh_faults'] for p in processes), 'six independently completed model processes required')

    def rate(p):
        return p['routes']['generation']['exact_native_decode_tps']

    pairs = []
    for a, b in zip(processes[::2], processes[1::2], strict=True):
        ref, cand = (a, b) if a['arm'] == 'reference' else (b, a)
        same_work = (width.comparable_work(ref) == width.comparable_work(cand) and all(
            ref['routes'][key] == cand['routes'][key] for key in ('verifier_work', 'expert_work')))
        before, after = (p['evidence']['samples'][-1] for p in (ref, cand))
        pairs.append(dict(matched_work=same_work, tg_ratio=rate(cand) / rate(ref),
                          prompt_time_ratio=after['prompt_ms'] / before['prompt_ms'],
                          ttft_ratio=after['ttft_s'] / before['ttft_s'],
                          same_text=before['output_sha256'] == after['output_sha256']))
    rates = {arm: [rate(p) for p in processes if p['arm'] == arm] for arm in probe.ORDER}
    return dict(pairs=pairs, TG=rates, TG_median={arm: statistics.median(values) for arm, values in rates.items()},
                head_gain_qualified=all(p['matched_work'] and p['tg_ratio'] > 1 for p in pairs), adopted=False)


def experiment():
    ctx = width.leaves.experiment()
    run = ctx.run
    inherited_pins = width.experiment().pins
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment = baseline, configuration, environment
    ctx.routes, ctx.protocol = routes, PROTOCOL
    ctx.evidence_reader.read_complete_suite, ctx.evidence_reader.decide = width.read_suite, decision

    def native_context(row, plan):
        require(plan['arm'] == ORDER[plan['index']], 'the fixed head comparison order changed')
        return dict(row, **row['c5_head_arm'])

    def pins(revision):
        result = inherited_pins(revision)
        for name in (THIS, probe.THIS, probe.build.THIS):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed head measurement input: ' + name)
            result[name] = bank.sha(path)
        return result

    ctx.pins, ctx.native_context = pins, native_context
    ctx.run = lambda index: run(index, native_context=native_context)
    return ctx


if __name__ == '__main__':
    snapshot.clock.main(experiment())
