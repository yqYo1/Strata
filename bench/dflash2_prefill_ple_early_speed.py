"""Time the first PLE gather schedule on the qualified DFlash2 executable."""

from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_early_check as check

build, bank, require = check.build, check.bank, check.require
wait = build.wait
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_early_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('stock', 'early', 'early', 'stock', 'stock', 'early')
WARM_LIMIT = 1.01
MARKER = 'strata prefill PLE early read:'
OBSERVERS = (build.ple.SELECTOR, build.window.SELECTOR, build.window.layer.SELECTOR,
             build.window.layer.coarse.SELECTOR, wait.SELECTOR)
OBSERVED_LINES = (build.ple.PREFIX, build.window.PREFIX, build.window.layer.PREFIX,
                  build.window.layer.coarse.PREFIX)
PLAN = dict(
    order=list(ORDER), read=wait.READ, requests_per_process=wait.speed.RUNS,
    scope='Three counterbalanced pairs on the exact native executable that passed STOP and direct handoff. '
          'Change only the first PLE gather schedule. Match each bench.sh body and executed route at 118785 input '
          'tokens. Use independent processes and private runtime caches with the original warmups.',
    eligibility='Keep CPU, layer, PLE and native tracing clocks off. Require exactly one layer-1 consumption marker '
                'per enabled long request, none in stock, and the original expert, KV and generation work.',
    decision='Report the first long request separately from the next two fresh requests. Require first PP time '
             'and client TTFT each to improve in at least two pairs, with median paired ratios below one. '
             'Require median subsequent PP time and TTFT ratios at most 1.01 before further qualification. '
             'Record the 1000 tok/s milestone independently of the scheduling comparison.',
    stop='Stop on changed inputs, routing or work, unclosed ownership, runtime validation or a fresh Xe fault.',
    limits='Standard repeated-text workload. No causal device-capacity claim, decode adoption, physical fill, '
           'RESTORE, agent or varied-text qualification.')


def baseline(ctx):
    row = wait.baseline()
    native = bank.previous.read(build.ROOT / 'record.json')
    proof = wait.basis.validation.committed(bank.RESULT, (check.KEY,))
    require(wait.retained.closed(native, 'built') and wait.retained.closed(proof, 'closed')
            and proof == bank.previous.read(check.ROOT / 'record.json')
            and tuple(proof['outcomes']) == check.ORDER
            and all(p['complete'] and p['passed'] and p['ownership_closed']
                    and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in proof['outcomes'].values())
            and proof['outcomes']['cancel']['cancelled']['read'] == 0,
            'require terminal STOP and direct-handoff results on the same native build')
    ctx.check(native)
    identity = bank.signature(build.builder.compact(native))
    plan = proof['plan']
    require(native['base_binary'] == row['binary'] and native['base_build_sha256'] == row['native_build_sha256']
            and plan['binary'] == native['binaries']['strata']
            and plan['native_build_sha256'] == identity
            and plan['native_source_commit'] == native['overlay_commit'] == build.PATCH
            and all(plan[k] == row[k] for k in (
                'boot', 'driver', 'runtime_manifest', 'native_validation_probe', 'dflash2_assets')),
            'the speed input must use the exact checked engine and live model configuration')
    row['base_physical_context_sha256'] = row.pop('physical_context_sha256')
    return dict(row, binary=plan['binary'], native_source_commit=build.PATCH, native_build_sha256=identity,
                native_context_qualified=False, ple_check_sha256=bank.signature(proof))


def environment(row, plan):
    env = wait.environment(row, dict(plan, arm='clean'))
    require(all(env.get(key, '0') == '0' for key in OBSERVERS)
            and not any(key in env for key in ('STRATA_TRACE', 'STRATA_STATE_HASH', 'STRATA_DFLASH2_TAP_PAYLOAD_AUDIT')),
            'unexpected timing observer')
    return dict(env, **{check.SELECTOR: str(int(plan['arm'] == 'early'))})


def route(text, plan):
    frame = wait.route(text, dict(plan, arm='clean'))
    lines = text.splitlines()
    actual = [line for line in lines if line.startswith(MARKER)]
    expected = [check.CONSUMED] if plan['arm'] == 'early' else []
    require(actual == expected and text.count(MARKER) == len(actual)
            and not any(prefix in text for prefix in OBSERVED_LINES),
            'actual PLE scheduling or timing observers differ from the plan')
    return dict(frame, ple_early_reads=len(actual))


def after_execution(outcome, row, plan):
    routes = outcome['routes']
    count = int(plan['arm'] == 'early')
    require(all(frame['ple_early_reads'] == count for frame in routes),
            'unexpected first-gather scheduling')
    matched = dict(outcome, routes=[{key: val for key, val in frame.items() if key != 'ple_early_reads'}
                                   for frame in routes])
    wait.after_execution(matched, row, dict(plan, arm='clean'))
    outcome.update(arm=plan['arm'], clean_speed=True)


def decision(processes):
    require(all(p['clean_speed'] and len(p['samples']) == wait.speed.RUNS for p in processes),
            'require all clean matched processes')
    prompt_ratios = wait.decision(processes, ORDER)['observed_over_clean']
    pairs = []
    for x, y, prompt in zip(processes[::2], processes[1::2], prompt_ratios, strict=True):
        stock, early = (x, y) if x['arm'] == 'stock' else (y, x)
        a, *later_a = stock['samples']
        b, *later_b = early['samples']
        pairs.append(dict(first_prefill=prompt['first'],
                          first_ttft=b['ttft_s'] / a['ttft_s'],
                          subsequent_prefill=prompt['subsequent'],
                          subsequent_ttft=statistics.median(s['ttft_s'] for s in later_b)
                          / statistics.median(s['ttft_s'] for s in later_a)))
    medians = {key: statistics.median(pair[key] for pair in pairs) for key in pairs[0]}
    first_gain = all(medians[key] < 1 and sum(pair[key] < 1 for pair in pairs) >= 2
                     for key in ('first_prefill', 'first_ttft'))
    warm_preserved = all(medians[key] <= WARM_LIMIT for key in ('subsequent_prefill', 'subsequent_ttft'))
    return dict(paired_early_over_stock=pairs, median_ratios=medians, repeat_first_gain=first_gain,
                subsequent_preservation_screen=warm_preserved, advance=first_gain and warm_preserved,
                first_1000={arm: [p['samples'][0]['prompt_ms'] <= wait.READ for p in processes if p['arm'] == arm]
                            for arm in ('stock', 'early')}, adopted=False)


def experiment():
    native = build.context()
    ctx = wait.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.environment = lambda: baseline(native), PLAN, environment
    ctx.selector = check.SELECTOR
    ctx.extra_sources = (THIS, check.THIS, check.tap.THIS, build.THIS, build.ple.THIS, build.window.THIS,
                         build.window.layer.THIS, build.window.layer.coarse.THIS, build.base_build.THIS)
    ctx.request_route, ctx.after_execution, ctx.decision = route, after_execution, decision
    return ctx


if __name__ == '__main__':
    wait.speed.clock.main(experiment())
