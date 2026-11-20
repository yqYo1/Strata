"""Measure the cost of recording and replaying the empty PCIe expert projection."""

import fcntl
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_kv_gather_speed as speed
import decode_footer as footer

bank, basis, require = speed.bank, speed.basis, speed.require
KEY = 'decode_empty_pcie_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
READ, RUNS = 65537, 3
ORDER = ('inline', 'separate', 'separate', 'inline')
SELECTOR = 'STRATA_SYCL_VERIFY_PCIE_EARLY'
EARLY = 'strata serve: SYCL_PCIE_EARLY '
ZERO_EARLY = EARLY + 'launches=0 groups=0 ready_at_pool_return=0 pending_at_pool_return=0'
START = 'strata verify: early PCIe expert graph enabled mode=0 audit=0'
CAPTURE = r'^strata verify: capture ([14]) tokens (\d+\.\d+) ms, free VRAM \d+\.\d+ MiB$'
PLAN = dict(order=list(ORDER), input_tokens=READ, requests_per_process=RUNS,
    mechanism='At pcie-frac 0 the inline verifier records and replays a grouped projection with count zero. '
              'Separating it adds a capture but the empty graph never launches.',
    fixed='Same retained native ELF, ordinary-Arena copy and shared/hit overlap in both arms, host expert pool, '
          'placement, original 65K bodies, scalar gather, KV reload and spec4. Only the early selector differs.',
    scope='Require zero expert copies and no early replay before the first measured request and throughout. '
          'Keep native 24-output decode time with thinking, graph capture time, prompt time and client TTFT. '
          'Charge capture to the actual request clock and retain its own logged time for attribution.',
    gate='Candidate must reduce median subsequent native decode ms in both ordered pairs with matched work. '
         'First use and PP/TTFT are separate. This screens the graph arrangement for a decode-only implementation; '
         'the selector also covers prompt verification and cannot qualify independent prefill or sustained TG.',
    stop='First source, ownership, validation, fresh fault, output-budget, clock or work mismatch. '
         'Keep the independent processes and individual samples. No speed-based exclusion.',
    retention='Keep compact matching counters, individual clocks and the exact original input/engine identities. '
              'Review temporary logs and the controller at this comparison closure.')


def native_requests(text, separate):
    """Bind zero-copy and zero-replay footers directly to their native request."""
    require('strata verify: ordinary-Arena copy enabled mode=0 ' in text
            and ('strata verify: early PCIe expert graph enabled ' in text) == separate
            and (START in text) == separate, 'unexpected verifier setup')
    heads = list(re.finditer(footer.HEADER, text, re.M))
    require(heads and text.count('strata serve: SYCL_EXPERT_COPY ') == len(heads)
            and text.count(EARLY) == len(heads) * int(separate), 'missing or extra expert summary')
    requests, previous = [], 0
    for h in heads:
        following = text[h.end() + 1:].split('\n', 2)
        copy = re.fullmatch(footer.COPY, following[0])
        require(copy and not any(map(int, copy.groups()[:6])) and
                (not separate or len(following) > 1 and following[1] == ZERO_EARLY),
                'PCIe work differs from the zero-group plan')
        jobs, entries = map(int, copy.groups()[6:])
        require(0 < jobs <= entries, 'invalid native CPU expert counts')
        position = tuple(map(int, h.groups()))
        rounded_ms = re.search(r' generated in (\d+) ms ', h.group())
        require(rounded_ms, 'missing native decode duration')
        frame = text[previous:h.start()]
        captures = [(int(width), float(ms)) for width, ms in re.findall(CAPTURE, frame, re.M)]
        phases = tuple(map(int, re.findall(r'^strata verify: phase=decode tokens=([14]) ', frame, re.M)))
        requests.append(dict(request=position, cpu_jobs=jobs, cpu_entries=entries,
            reported_decode_ms=int(rounded_ms[1]), captures=captures, decode_capture_widths=phases))
        previous = h.end()
    return requests


def compare(processes):
    # Each process is admitted independently against the first process before
    # it is saved. Keep the last outcome even if a comparison is ineligible.
    if len(processes) != len(ORDER) or any(p.get('variant') != arm or not p.get('passed')
            for p, arm in zip(processes, ORDER)):
        return dict(eligible=False, reason='need two complete matched pairs', adopted=False)

    def times(process, warmed):
        samples = process['samples'][1:] if warmed else process['samples'][:1]
        return {key: statistics.median(s[key] for s in samples)
                for key in ('decode_ms', 'capture_total_ms', 'prompt_ms', 'ttft_s')}

    ratios = {scope: [] for scope in ('first', 'warm')}
    for start in (0, 2):
        pair = {p['variant']: p for p in processes[start:start + 2]}
        for scope, values in ratios.items():
            a, b = (times(pair[k], scope == 'warm') for k in ('inline', 'separate'))
            values.append({key: b[key] / a[key] for key in a})
    return dict(eligible=True, separate_over_inline=ratios,
        repeated_short_decode_gain=all(v['decode_ms'] < 1 for v in ratios['warm']),
        sustained_decode_qualified=False, adopted=False)


def experiment():
    baseline = speed.experiment(READ)
    ctx = basis.core.source.device.load_module('decode_empty_pcie', basis.core.__file__)
    ctx.THIS, ctx.KEY, ctx.ROOT = THIS, KEY, bank.B / 'decode-empty-pcie-v1'
    ctx.READS, ctx.ORDER = (READ,) * len(ORDER), [(READ, 1)] * len(ORDER)
    ctx.RUNS, ctx.CLEAN_FAST = RUNS, True
    ctx.basis, ctx.check_sample = baseline.basis, baseline.check_sample
    ctx.decision = compare
    old_pins, old_save, old_current = ctx.pins, ctx.save, ctx.current

    def pins(revision):
        values = old_pins(revision)
        require(({THIS, speed.THIS, str(Path(footer.__file__).relative_to(bank.P))} |
                 set(baseline.pins(revision))) <= values.keys(), 'uncommitted experiment or inherited input')
        return values

    def environment(row, plan):
        args = row['configuration']['args']
        require(args.count('--pcie-frac') == 1 and args[args.index('--pcie-frac') + 1] == '0'
                and args.count('--spec') == 1 and args[args.index('--spec') + 1] == '4'
                and '--no-spec-split' in args, 'need the fixed unsplit width-four pcie0 configuration')
        env = baseline.environment(row, dict(plan, index=0))
        require(env.get('STRATA_SYCL_VERIFY_NATIVE_CAPTURE') == '0', 'need the qualified capture path')
        env.update(STRATA_SYCL_VERIFY_ARENA_COPY='1', STRATA_SYCL_VERIFY_OVERLAP='1',
                   STRATA_SYCL_VERIFY_PCIE_EARLY_AUDIT='0')
        env[SELECTOR] = str(int(ORDER[plan['index']] == 'separate'))
        return env

    def route(text, plan):
        return baseline.request_route(text, dict(plan, index=0))

    def begin_gate(outcome, request, stage, _now):
        if stage == 'begin' and request == 0:
            index = outcome['index']
            log = (ctx.ROOT / str(index) / 'engine.stderr').read_text()
            warmup, = native_requests(log, ORDER[index] == 'separate')
            require(0 < warmup['request'][2] < READ, 'short warmup did not qualify the no-copy path')

    def after_execution(outcome, row, plan):
        baseline.after_execution(outcome, row, plan)
        root = Path(plan['root'])
        result, = (root / 'bench').glob('*/results.json')
        requests = json.loads(result.read_text())['results']['prefill_only']['requests']
        warmup, *native = native_requests((root / 'engine.stderr').read_text(), plan['variant'] == 'separate')
        require(0 < warmup['request'][2] < READ and len(native) == RUNS, 'unexpected warmup or sample count')
        work = []
        for sample, measured, observed in zip(outcome['samples'], requests, native, strict=True):
            clock = measured['decode_ms']
            fields = ('actual_prompt_tokens', 'actual_reused_tokens', 'actual_read_tokens', 'generated_tokens',
                      'drafts_accepted', 'drafts_offered')
            require(tuple(measured[k] for k in fields) == observed['request'] and
                    measured['request_sha256'] == sample['request_sha256'] and measured['decode_tps_source'] == 'engine'
                    and type(clock) in (int, float) and math.isfinite(clock) and clock >= 100
                    and abs(clock - observed['reported_decode_ms']) <= .55,
                    'native/API work or decode clock differs')
            captures = observed['captures']
            require(observed['decode_capture_widths'] == (1, 4) and [w for w, _ in captures] == [1, 4]
                    and all(math.isfinite(ms) and ms > 0 for _, ms in captures), 'changed native capture path')
            sample.update(decode_ms=clock, capture_ms_by_width=dict(captures),
                          capture_total_ms=sum(ms for _, ms in captures))
            work.append(dict(cpu_jobs=observed['cpu_jobs'], cpu_entries=observed['cpu_entries']))
        require(not row['processes'] or work == row['processes'][0]['cpu_expert_work'], 'CPU expert work changed')
        identity = (outcome['native_identity']['pid'], outcome['native_identity']['start_ticks'])
        require(all(identity != (p['native_identity']['pid'], p['native_identity']['start_ticks'])
                    for p in row['processes']), 'need an independent native process')
        outcome.update(variant=plan['variant'], cpu_expert_work=work,
                       zero_expert_copy=True, empty_pcie_launches=0 if plan['variant'] == 'separate' else None)

    def save(row):
        if row['stage'] == 'prepared':
            for plan, variant in zip(row['plans'], ORDER, strict=True):
                plan['variant'] = variant
            row['budgets']['total_cap'] = 64 << 20
            row.update(plan=PLAN, whole_bytes=len(ORDER) * (64 << 20) + (8 << 20))
        old_save(row)

    def current(row):
        require(row['plan'] == PLAN and [(p['index'], p['variant']) for p in row['plans']]
                == list(enumerate(ORDER)), 'frozen comparison changed')
        return old_current(row)

    ctx.pins, ctx.environment, ctx.save, ctx.current = pins, environment, save, current
    ctx.request_route, ctx.observe_gate, ctx.after_execution = route, begin_gate, after_execution
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and
            (action in ('prepare', 'host') or action in map(str, range(len(ORDER)))), 'invalid comparison stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            basis.comparison.run(int(action), ctx)
