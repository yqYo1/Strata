"""Screen existing live-prefix reuse on the pinned DFlash2 answer-room request."""

from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_reconciliation_model as model
from strata_control.user_args import groups


THIS = str(Path(__file__).resolve().relative_to(model.bank.P))
KEY = 'dflash2_answer_room_reuse_v1'
REFERENCE = 'c5ffe8f77'
REFERENCE_KEY = 'dflash2_reconciliation_model_v1'
CASE = model.reconciliation.RECONCILIATION
MAX_FRESH_CONTINUATION = 64
PLAN = dict(model.PLAN,
    scope='One private server with the retained DFlash2 executable and original reconciliation body. '
          'Enable the existing live prompt cache, with periodic, turn, root and parked snapshots disabled.',
    reference='Use the closed cache-off observation for original work and scale. A later counterbalanced '
              'comparison is required to claim a repeatable end-to-end gain.',
    acceptance='Require two owned native legs, a fresh first prompt, at most 64 freshly read tokens on '
               'the second leg, a live DFlash2 prefix, exact task oracle and sustained thinking-on output.',
    retention='Retain input and source identities, native leg and reference clocks, reuse and drafter counts, '
              'oracle and ownership outcomes. Retire routine output after the comparison decision.')


def set_live_cache(config):
    args = list(config['args'])
    user_args = list(config['user_args'])
    fixed = {'--prompt-cache': '0', '--turn-token': '-1', '--conversation-cache-mib': '0'}
    model.require(all(args.count(flag) == 1 and args[args.index(flag) + 1] == value
                      for flag, value in fixed.items()) and
                  user_args.count('--prompt-cache') == 1 and user_args[user_args.index('--prompt-cache') + 1] == '0',
                  'unexpected baseline cache settings')
    args[args.index('--prompt-cache') + 1] = '1'
    user_args[user_args.index('--prompt-cache') + 1] = '1'
    for flag in ('--prompt-cache-every', '--prompt-cache-root'):
        if flag not in args:
            args.extend((flag, '0'))
        else:
            model.require(args.count(flag) == 1 and args[args.index(flag) + 1] == '0',
                          'the interval and root checkpoints must be disabled')
    return dict(config, args=args, user_args=user_args)


def reference(row):
    closed = model.bank.previous.historic(REFERENCE, REFERENCE_KEY)
    model.require(closed['stage'] == 'closed' and closed['passed'] and closed['complete']
                  and closed['ownership_closed'] and not closed['active']
                  and all(closed[k] == row[k] for k in ('binary', 'native_source_commit', 'native_build_sha256')),
                  'require the original closed native reference')
    process, = closed['processes']
    evidence = process['evidence']
    sample, = evidence['samples']
    first, second = sample['native_segments']
    model.require(evidence['observation']['answer_correct'] and sample['request_sha256'] == CASE.wire_sha256
                  and (first['prompt'], first['reused'], second['read'], second['reused'])
                  == (CASE.input_tokens, 0, 48221, 0), 'cached prompt control does not match the original task')
    return dict(commit=REFERENCE, input=CASE.input_tokens, second_read=second['read'],
                second_prompt_ms=second['prompt_ms_rounded'])


def routes(log, samples, arm):
    result = model.routes(log, samples, arm)
    markers = re.findall(r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=([01]) '
                         r'vram_bytes=([1-9]\d*)$', log, re.M)
    model.require(len(markers) == len(samples[0]['native_segments']), 'missing DFlash2 leg marker')
    rows = re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) ', log, re.M)
    return dict(result, drafter_cold_prefix=[m[1] == '1' for m in markers], layer_major_rows=list(map(int, rows)))


def decision(processes):
    process, = processes
    evidence = process['evidence']
    sample, = evidence['samples']
    observation = evidence['observation']
    segments = sample['native_segments']
    continued = len(segments) == 2
    reuse = (continued and 0 < segments[1]['read'] <= MAX_FRESH_CONTINUATION
             and segments[1]['reused'] >= CASE.input_tokens + segments[0]['generated'] - 4)
    live_drafter = continued and process['routes']['drafter_cold_prefix'] == [False, False]
    expected_route = process['routes']['layer_major_rows'] == [CASE.input_tokens - 1]
    quality = observation['answer_correct'] and observation['native_generation_long_enough']
    return dict(two_native_legs=continued, target_prefix_reused=reuse, drafter_prefix_live=live_drafter,
                first_leg_batched_only=expected_route, correct_answer=observation['answer_correct'],
                native_tg=observation['native_tg'],
                continuation_read_tokens=segments[1]['read'] if continued else None,
                continuation_prompt_ms=segments[1]['prompt_ms_rounded'] if continued else None,
                eligible_for_repeated_comparison=bool(reuse and live_drafter and expected_route and quality
                    and observation['eligibility']['speed_eligible']))


def experiment():
    ctx = model.experiment(CASE, source=THIS, protocol=PLAN, decide=decision)
    configuration, baseline = ctx.configuration, ctx.baseline

    def configured(row, plan):
        cfg = set_live_cache(configuration(row, plan))
        args = model.primary.speed.generation.model.server.engine_args(cfg, exe=cfg['exe'], announce=False)
        model.require(sorted(groups(args)) == sorted(groups(cfg['args'])),
                      'server changed the requested live-cache settings')
        return dict(cfg, args=args)

    def baseline_with_reference():
        row = baseline()
        return dict(row, cache_off_reference=reference(row))

    ctx.KEY, ctx.ROOT = KEY, model.bank.B / KEY.replace('_', '-')
    ctx.configuration, ctx.baseline, ctx.routes = configured, baseline_with_reference, routes
    return ctx


if __name__ == '__main__':
    if sys.argv[1:2] == ['client']:
        model.collect(Path(sys.argv[2]), int(sys.argv[3]), case=CASE)
    else:
        model.clock.main(experiment())
