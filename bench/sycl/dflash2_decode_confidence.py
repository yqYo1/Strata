"""Screen DFlash2 confidence on the existing spec4 production path."""

import fcntl
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_cpu_leaves as leaves
import dflash2_v2_commit_order as width_check

bank, basis, require = leaves.bank, leaves.basis, leaves.require
model, clock = width_check.model, leaves.clock
KEY = 'dflash2_decode_confidence_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
CONFIDENCE = {'threshold': '0.6', 'full': '0'}
ORDER = ('threshold', 'full', 'full', 'threshold')
WIDTH, CHECK_PROMPT, CHECK_OUTPUT = 4, 11793, 32
CONTEXT, MAX_OUTPUT = 262144, 16384
CASE = 'width-check'
FLAG = '--draft-conf'


def baseline():
    row = leaves.baseline()
    prior = basis.validation.committed(bank.RESULT, (leaves.KEY,))
    require(prior['stage'] == 'closed' and prior['complete'] and prior['passed']
            and prior['ownership_closed'] and not prior['active']
            and tuple(p['arm'] for p in prior['processes']) == leaves.ORDER
            and all(p['passed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
                    for p in prior['processes'])
            and all(row[key] == prior[key] for key in ('binary', 'native_build_sha256', 'runtime_manifest',
                    'boot', 'driver', 'request_reference')),
            'require the closed worker-leaf comparison on the retained DFlash2 route')
    return dict(row, confidence_reference_sha256=bank.signature(prior))


def configuration(row, plan):
    cfg = leaves.configuration(row, plan)
    args, overrides = list(cfg['args']), list(cfg['user_args'])
    require(args.count(FLAG) == 1 and args[args.index(FLAG) + 1] == CONFIDENCE['threshold']
            and FLAG not in overrides, 'expected the measured confidence default')
    confidence = CONFIDENCE[plan['arm']]
    args[args.index(FLAG) + 1] = confidence
    overrides.extend((FLAG, confidence))
    out = dict(cfg, args=args, user_args=overrides)
    require(model.server.engine_args(out, exe=out['exe'], announce=False) == args,
            'the production resolver changed another native argument')
    return out


def environment(row, plan):
    return leaves.environment(row, dict(plan, arm='clean'))


def probe_inputs(row):
    root = ROOT / CASE
    choice = dict(root=str(root), arm='full')
    cfg = configuration(row, choice)
    env = environment(row, choice)
    require(env['STRATA_COMMIT_SYNC'] == '1' and env['STRATA_WATCHDOG_S'] == '240'
            and env.get('STRATA_SYCL_VERIFY_OVERLAP', '1') != '0', 'keep the qualified verifier order')
    env = model.native_environment(row, dict(env, STRATA_TRACE='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1'))
    short = model.request_ids(Path(cfg['tokenizer']))['ids']
    require(len(short) == 82, 'checked short prompt changed')
    middle = [i % 4096 for i in range(CHECK_PROMPT - len(short))] + short
    requests = [(ids, CHECK_OUTPUT) for ids in (short, middle, short)]
    case = dict(command=[row['binary']['path'], '--serve', *cfg['args']], environment_sha256=bank.signature(env),
                wire=[dict(prompt=len(ids), output=count,
                           sha256=model.restore.signature_bytes(model.wire(ids, count))) for ids, count in requests])
    plan = dict(cases={CASE: case}, boot=row['boot'], driver=row['driver'],
                validation=row['native_validation_probe'], drafter_assets=row['dflash2_assets'])
    return plan, env, requests


def check_width(ctx):
    require(sys.executable == basis.reference.model.PYTHON, 'use the pinned Python')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        row = bank.previous.read(ROOT / 'record.json')
        ctx.current(row)
        plan, env, requests = probe_inputs(row)
        root = ROOT / CASE
        require(row['stage'] == 'ready' and row['host_passed'] and row['ownership_closed'] and not row['active']
                and not row['processes'] and row['width_check'] is None
                and not (root / 'execution.json').exists() and not (root / 'owned').exists(),
                'require the committed unused width check')

        def inspect(log, pid, answers):
            result = width_check.inspect(log, plan, pid, CHECK_PROMPT, answers)
            first, *rest = result['middle_windows']
            require(first == [CHECK_PROMPT - 1, 1] and rest and all(w == WIDTH for _, w in rest),
                    'the no-threshold middle request must exercise full-width verification')
            return result

        result = width_check.run_case(plan, root, CASE, env, requests, inspect)
        row.update(width_check=result, ownership_closed=result['ownership_closed'], active=not result['ownership_closed'])
        if not result['passed']:
            row['stage'] = 'failed'
        ctx.save(row)
        print({k: result.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
        require(result['passed'], 'close the failed width check before further work')


def window_counts(sample, arm):
    offered, accepted, emitted = (sample[k] for k in ('drafts_offered', 'drafts_accepted', 'generated_tokens'))
    require(sample['read_tokens'] + MAX_OUTPUT + WIDTH < CONTEXT and 0 <= accepted <= offered,
            'the exact served request needs context and valid draft work')
    # Each window offers at most three draft rows. A terminal window can leave
    # up to four of its accepted rows or bonus un-emitted. Its first window is width one.
    minimum = max(1 + (offered + WIDTH - 2) // (WIDTH - 1), emitted - accepted)
    maximum = emitted - accepted + WIDTH
    require(minimum <= maximum, 'inconsistent emitted/accepted window bounds')
    if arm == 'full':
        windows, remainder = divmod(offered, WIDTH - 1)
        windows += 1
        require(not remainder and minimum <= windows <= maximum, 'non-full window or invalid terminal work')
        minimum = maximum = windows
    return dict(windows_min=minimum, windows_max=maximum, offered=offered, accepted=accepted,
                emitted=emitted, verifier_rows_min=offered + minimum, verifier_rows_max=offered + maximum)


def routes(text, samples, arm):
    sample = samples[-1]
    require(all(math.isfinite(sample[k]) and sample[k] > 0
                for k in ('decode_ms', 'prompt_ms', 'ttft_s')), 'positive clock values required')
    counts = window_counts(sample, arm)
    return dict(leaves.routes(text, samples, 'clean'), window_work=counts)


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER and len({
        (p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER)
        and all(p['complete'] and p['passed'] and p['ownership_closed'] and p['native_exit_code'] == 0
                and not p['fresh_faults'] for p in processes), 'four fresh clean processes required')
    rows = [process['evidence']['samples'][-1] for process in processes]
    pairs = []
    for i, j in ((0, 1), (3, 2)):
        control, full = rows[i], rows[j]
        work = full['generated_tokens'] == control['generated_tokens']
        tg = (full['generated_tokens'] / full['decode_ms']) / (control['generated_tokens'] / control['decode_ms'])
        pairs.append(dict(tg_ratio=tg, same_generation_length=work,
                          matching_output_hash=full['output_sha256'] == control['output_sha256'],
                          prefill_time_ratio=full['prompt_ms']/control['prompt_ms'],
                          ttft_ratio=full['ttft_s']/control['ttft_s']))
    return dict(pairs=pairs, windows=[p['routes']['window_work'] for p in processes],
                repeated_matched_decode_gain=all(p['same_generation_length'] and p['tg_ratio'] > 1 for p in pairs),
                sustained_weak_qualified=False, live_restore_qualified=False, adopted=False)


def experiment():
    ctx = leaves.experiment()
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment = baseline, configuration, environment
    ctx.evidence_reader.decide = decision
    ctx.routes = routes
    ctx.protocol = dict(order=list(ORDER), threshold=CONFIDENCE,
        only_factor='Set the native --draft-conf threshold to 0.6 or 0 before exec. Keep width four and all placements.',
        gate='Before long runs, require the bounded traced 82 / 11793 / 82 no-threshold check at 32 outputs.',
        comparison='Two reversed fresh-process pairs of the same official thinking-on code request. Preserve prompt work, '
                   'four warmups, actual route, per-output latency, PP, client TTFT and exact accepted/offered totals. '
                   'A matched code gain needs the same generation length and better TG in both pairs. '
                   'Record output/reasoning hashes for diagnosis and keep broader quality and RESTORE gates separate.',
        stop='Stop on the first input, provider, route, ownership or fault failure. Do not discard measured speed samples.')
    original_pins, original_save, original_current, original_run = ctx.pins, ctx.save, ctx.current, ctx.run

    def pins(revision):
        files = original_pins(revision)
        for name in (THIS, width_check.THIS):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'confidence comparison source changed: ' + name)
            files[name] = bank.sha(path)
        return files

    def save(row):
        if row['stage'] == 'prepared':
            row['width_plan'], _, _ = probe_inputs(row)
            row['width_check'] = None
            for folder in ('cache', 'tmp'):
                (ROOT / CASE / folder).mkdir(mode=0o700, parents=True)
        original_save(row)

    def current(row):
        require(row['width_plan'] == probe_inputs(row)[0], 'bounded device check inputs changed')
        return original_current(row)

    def run(index):
        row = bank.previous.read(ROOT / 'record.json')
        short = row['width_check']
        require(short and short['passed'] and short['complete'] and short['ownership_closed']
                and short['native_exit_code'] == 0 and not short['fresh_faults'], 'finish the committed width check')
        original_run(index)

    ctx.pins, ctx.save, ctx.current, ctx.run = pins, save, current, run
    return ctx


if __name__ == '__main__':
    ctx = experiment()
    if sys.argv[1:] == ['check-width']:
        check_width(ctx)
    else:
        clock.main(ctx)
