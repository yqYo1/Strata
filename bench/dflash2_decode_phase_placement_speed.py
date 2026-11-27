"""Measure DFlash2 placement with a common prefill and decode clock."""

from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_decode_phase_placement_check as checked
import dflash2_decode_phase_placement_model_check as model_check
import dflash2_primary_continuation as primary

build, bank, require, validation = checked.build, checked.bank, checked.require, checked.validation
KEY = 'dflash2_decode_phase_placement_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = Path(__file__).resolve().relative_to(bank.P).as_posix()
ORDER = ('stock', 'blend', 'blend', 'stock', 'stock', 'blend')
EVENT = re.compile(checked.EVENT.pattern.removesuffix('audit=1$') + r'audit=0$', re.M)
PREFILL = ('residual_gpu_rows', 'direct_pieces', 'kv', 'bank_h2d_requested_bytes', 'placement_preserved')
PLAN = dict(primary.PLAN, order=list(ORDER),
    scope='One coherent executable and initial placement. Alternate stock and decode-only placement in three '
          'counterbalanced process pairs with the same two 96K thinking bodies. Keep T4 stock and caller help off.',
    timing='First nonempty client output precedes installation. Charge the final commit in both arms and both '
           'transitions in the candidate. Record prefill and TTFT independently and preserve actual draft work.',
    decision='Require equal fresh prompt and logical prefill work, repeated decode gains with positive geometric '
             'mean on both workloads, and paired median prefill and TTFT ratios at most one. Placement may change '
             'the output and the accepted drafts. Require separate task, weak-draft, agent and full-context '
             'qualification before adoption.',
    stop='Stop on the first source, runtime, ownership, workload, actual route or validation failure.')


def baseline():
    native = bank.previous.read(build.ROOT / 'record.json')
    build.check(native)
    proof = validation.committed(bank.RESULT, (model_check.KEY,))
    binary = native['binaries']['strata']
    identity = bank.signature(build.builder.compact(native))
    require(build.retained.closed(proof, 'closed') and proof['build_sha256'] == identity
            and proof['native_source_commit'] == native['native_source_commit']
            and tuple(proof['outcomes']) == checked.ORDER
            and all(p['passed'] and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
                    for p in proof['outcomes'].values())
            and all(proof['cases'][arm]['binary'] == binary for arm in ('stock', 'blend'))
            and all(bank.sha(p['path']) == p['sha256'] for p in proof['profiles'].values()),
            'require closed same-build placement, cancellation, restore and short output checks')
    row = build.retained.baseline()
    row['base_physical_context_sha256'] = row.pop('physical_context_sha256')
    return dict(row, binary=binary, native_source_commit=native['native_source_commit'],
                native_build_sha256=identity, native_context_qualified=False,
                placement_validation_sha256=bank.signature(proof), qualified_profiles=proof['profiles'])


def configuration(row, plan):
    cfg = primary.speed.generation.configuration(row, plan)
    args = list(cfg['args'])
    require(args.count('--expert-cache') == args.count('--expert-profile') == args.count('--expert-cache-per-layer') == 1
            and args[args.index('--expert-cache') + 1] == '1256'
            and args[args.index('--adapt-swaps') + 1] == '0' and '--mmap-experts' not in args,
            'the validated fixed global cache options changed')
    args[args.index('--expert-cache') + 1] = '968'
    args[args.index('--expert-profile') + 1] = row['qualified_profiles']['stock']['path']
    args.remove('--expert-cache-per-layer')
    return dict(cfg, args=args)


def environment(row, plan):
    env = primary.curve.environment(row, plan)
    require(not any(k in env for k in (checked.SELECTOR, checked.AUDIT, checked.CLOCK,
                                     'STRATA_SYCL_DECODE_T4_SCHEDULE', 'STRATA_SYCL_DECODE_T4_PREFIX_HELP')),
            'the timing template contains an unexpected placement or T4 selector')
    env.update(STRATA_SYCL_DECODE_T4_SCHEDULE='stock', STRATA_SYCL_DECODE_T4_PREFIX_HELP='0', **{checked.CLOCK: '1'})
    if plan['arm'] == 'blend':
        env[checked.SELECTOR] = row['qualified_profiles']['blend']['path']
    return env


def routes(text, samples, arm):
    row = bank.previous.read(ROOT / 'record.json')
    profile = row['qualified_profiles']['stock']['path']
    require('decode phase profile inactive' not in text, 'placement admission failed')
    report = primary.routes(text, samples, 'dflash2',
                            profile_check=lambda log: checked.stock.check_loader(log, profile, per_layer=False))
    start = 0
    for header, sample in zip(re.finditer(primary.footer.HEADER, text, re.M), samples, strict=True):
        frame = text[start:header.start()]
        start = header.end()
        events = EVENT.findall(frame)
        name = sample.get('workload')
        switched = arm == 'blend' and name in primary.WORKLOADS
        require([e[0] for e in events] == (['blend', 'stock'] if switched else [])
                and len(events) == frame.count('strata decode placement:'), 'actual placement differs from plan')
        if name not in primary.WORKLOADS:
            continue
        report[name]['transitions'] = []
        if switched:
            install, restored = [int(e[1]) for e in events]
            when = [int(e[4]) for e in events]
            ms = [float(e[5]) for e in events]
            interval = sample['client_interval']
            first, end = (interval[k] for k in ('first_output_monotonic_ns', 'request_end_monotonic_ns'))
            require(32 <= install <= 35 and install < restored == sample['generated_tokens']
                    and first < when[0] < when[1] < end
                    and 0 < sum(ms) < sample['decode_ms']
                    and all(tuple(map(int, e[2:4])) == checked.GEOMETRY for e in events),
                    'transition extent, client ordering or charged decode time changed')
            report[name]['transitions'] = [dict(profile=e[0], emitted=int(e[1]), wall_ms=float(e[5]),
                                                after_first_ms=(int(e[4]) - first) / 1e6) for e in events]
    require('strata decode placement:' not in text[start:], 'unexpected switch outside the request')
    return report


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
                actual_work=[dict(emitted=s['generated_tokens'], accepted=s['drafts_accepted'],
                                  offered=s['drafts_offered'],
                                  cpu_jobs_per_output=r['expert_work']['cpu_unique_misses'] / s['generated_tokens'])
                             for s, r in ((a, control), (b, candidate))]))
    summary = {}
    for name, samples in pairs.items():
        summary[name] = dict(tg_geomean=statistics.geometric_mean(p['tg_ratio'] for p in samples),
            tg_wins=sum(p['tg_ratio'] > 1 for p in samples), matched_prefill=all(p['matched_prefill'] for p in samples),
            median_prefill_time_ratio=statistics.median(p['prefill_time_ratio'] for p in samples),
            median_ttft_ratio=statistics.median(p['ttft_ratio'] for p in samples))
    eligible = all(r['matched_prefill'] and r['tg_wins'] >= 2 and r['tg_geomean'] > 1
                   and r['median_prefill_time_ratio'] <= 1 and r['median_ttft_ratio'] <= 1 for r in summary.values())
    return dict(pairs=pairs, summary=summary, eligible_for_qualification=eligible, adopted=False)


def experiment():
    ctx = primary.experiment()
    base_pins = ctx.pins
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.configuration, ctx.environment = baseline, PLAN, configuration, environment
    ctx.routes, ctx.evidence_reader.decide = routes, decision

    def pins(revision):
        results = base_pins(revision)
        for path in (THIS, checked.THIS, checked.stock.THIS, build.THIS, model_check.THIS):
            file = bank.P / path
            require(file.read_bytes() == validation.git('show', revision + ':' + path), 'commit timing input: ' + path)
            results[path] = bank.sha(file)
        return results

    ctx.pins = pins
    return ctx


if __name__ == '__main__':
    primary.clock.main(experiment())
