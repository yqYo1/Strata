"""Measure decode-only scalar clock wrap on two fixed 96K DFlash2 workloads."""

from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_decode_kv_clock_wrap_admission as admission
import dflash2_primary_continuation as primary

bank, basis, require = admission.bank, admission.basis, admission.require
validation = basis.validation
KEY = 'dflash2_decode_kv_clock_wrap_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
REFERENCE = '9261c4b449ce811b13c3b2238ae4e0b35ae50625'
ORDER = ('scalar', 'clock-wrap', 'clock-wrap', 'scalar', 'scalar', 'clock-wrap')
PREFILL = ('residual_gpu_rows', 'direct_pieces', 'kv', 'bank_h2d_requested_bytes', 'placement_preserved')
KV_TOTALS = re.compile(r'^strata serve: KV streaming: (\d+\.\d+)% of (\d+) block reads hit VRAM, '
                       r'(\d+\.\d+) MiB read from RAM$', re.M)
PLAN = dict(primary.PLAN, order=list(ORDER),
    scope='Same coherent executable and per-layer stock placement. Switch only scalar clock wrap at exec '
          'in three counterbalanced process pairs with the two original 96K thinking work items.',
    route='The production prefill route stays stock. Check phase-specific KV graph captures, '
          'the actual streamed KV totals and the existing long-prefill placement and handoff.',
    decision='In each workload require identical logical prefill work, at least two decode wins '
             'and geometric mean decode ratio above one. Require paired median prefill and TTFT ratios '
             'at most one. Retain work and draft acceptance even when they differ. A speed pass '
             'permits physical context, live RESTORE and agent and task qualification.')


def baseline():
    base = admission.baseline()
    proof = validation.committed(bank.RESULT, (admission.KEY,))
    require(admission.build.retained.closed(proof, 'closed')
            and proof['binary'] == base['binary'] and proof['build_sha256'] == base['native_build_sha256']
            and proof['native_source_commit'] == base['native_source_commit']
            and tuple(proof['outcomes']) == admission.ORDER
            and all(p['passed'] and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
                    for p in proof['outcomes'].values())
            and proof['outcomes']['scalar']['request'] == proof['outcomes']['clock-wrap']['request']
            and bank.sha(proof['stock']['path']) == proof['stock']['sha256'],
            'require both closed same-build route and output checks')
    prior = bank.previous.historic(REFERENCE, primary.KEY)
    require(prior['stage'] == 'closed' and prior['ownership_closed']
            and prior['native_source_commit'] == admission.build.foundation.BASE,
            'require the earlier 96K primary workload identity')
    bodies = {}
    for name in primary.WORKLOADS:
        samples = [s for process in prior['processes'] for s in process['evidence']['samples']
                   if s.get('workload') == name]
        (request_sha, read), = {(s['request_sha256'], s['read_tokens']) for s in samples}
        require(len(samples) == 3 and all(s['reused_tokens'] == 0 for s in samples), 'unexpected archived request')
        bodies[name] = dict(request_sha256=request_sha, read_tokens=read)
    return dict(base, short_admission_sha256=bank.signature(proof), stock=proof['stock'],
                archived_requests=bodies)


def environment(row, plan):
    env = primary.curve.environment(row, plan)
    require(admission.FLAG not in env and env.get(admission.native.FLAG, '0') == '0'
            and env.get('STRATA_SYCL_KV_COPY_GROUPS') == '0',
            'keep scalar copy configuration in both timing arms')
    return dict(env, **{admission.FLAG: str(int(plan['arm'] == 'clock-wrap'))})


def routes(text, samples, arm):
    row = bank.previous.read(ROOT / 'record.json')
    report = primary.routes(text, samples, 'dflash2', profile_check=lambda log:
                            admission.check_profile(log, row['stock']['path']))
    selected = arm == 'clock-wrap'
    all_captures = admission.routes(text, selected)
    require(any(x['phase'] == 'prompt' for x in all_captures), 'require an actual verifier prompt capture')
    headers = list(re.finditer(primary.footer.HEADER, text, re.M))
    totals = list(KV_TOTALS.finditer(text))
    require(len(totals) == len(headers) == len(samples)
            and text.count('strata serve: KV streaming: ') == len(totals),
            'require one non-overflowing KV totals footer per request')
    for i, (header, footer, sample) in enumerate(zip(headers, totals, samples, strict=True)):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        require(header.end() < footer.start() < end, 'wrong order of request and KV footers')
        if not sample.get('workload'):
            continue
        name = sample['workload']
        require({k: sample[k] for k in ('request_sha256', 'read_tokens')} == row['archived_requests'][name],
                'the sustained request differs from the committed 96K input')
        start = headers[i - 1].end() if i else 0
        captures = admission.routes(text[start:header.start()], selected)
        require(any(x['phase'] == 'decode' and 1 <= x['width'] <= 4 for x in captures),
                'each 96K request must show a supported decode capture')
        report[name].update(kv_captures=captures, kv_stream_totals=dict(
            hit_percent=float(footer[1]), block_reads=int(footer[2]),
            requested_host_payload_mib_rounded=float(footer[3])))
    return report


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER),
            'six independent complete timing processes required')
    pairs = {name: [] for name in primary.WORKLOADS}
    for first, second in zip(processes[::2], processes[1::2], strict=True):
        stock, candidate = (first, second) if first['arm'] == 'scalar' else (second, first)
        for name in pairs:
            a, b = [next(s for s in p['evidence']['samples'] if s.get('workload') == name)
                    for p in (stock, candidate)]
            control, trial = stock['routes'][name], candidate['routes'][name]
            rates = [p['evidence']['decode_rates'][name] for p in (stock, candidate)]
            pairs[name].append(dict(tg=rates, tg_ratio=rates[1] / rates[0],
                prefill_time_ratio=b['prompt_ms'] / a['prompt_ms'], ttft_ratio=b['ttft_s'] / a['ttft_s'],
                matched_prefill=all(a[k] == b[k] for k in ('request_sha256', 'read_tokens'))
                    and all(control[k] == trial[k] for k in PREFILL),
                matched_decode_counts=all(a[k] == b[k] for k in ('generated_tokens', 'drafts_accepted', 'drafts_offered'))
                    and control['expert_work'] == trial['expert_work'],
                reported_kv_totals=[control['kv_stream_totals'], trial['kv_stream_totals']]))
    summary = {name: dict(tg_geomean=statistics.geometric_mean(p['tg_ratio'] for p in samples),
        tg_wins=sum(p['tg_ratio'] > 1 for p in samples), matched_prefill=all(p['matched_prefill'] for p in samples),
        median_prefill_time_ratio=statistics.median(p['prefill_time_ratio'] for p in samples),
        median_ttft_ratio=statistics.median(p['ttft_ratio'] for p in samples)) for name, samples in pairs.items()}
    eligible = all(p['matched_prefill'] and p['tg_wins'] >= 2 and p['tg_geomean'] > 1
                   and p['median_prefill_time_ratio'] <= 1 and p['median_ttft_ratio'] <= 1 for p in summary.values())
    return dict(pairs=pairs, summary=summary, eligible_for_qualification=eligible, adopted=False)


def experiment():
    ctx = primary.experiment()
    base_pins = ctx.pins
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.environment, ctx.routes = baseline, PLAN, environment, routes
    ctx.evidence_reader.decide = decision

    def pins(revision):
        values = base_pins(revision)
        for name in (THIS, admission.THIS, admission.build.THIS, admission.native.THIS):
            file = bank.P / name
            require(file.read_bytes() == validation.git('show', revision + ':' + name),
                    'commit the exact clock-wrap timing input: ' + name)
            values[name] = bank.sha(file)
        return values

    ctx.pins = pins
    return ctx


if __name__ == '__main__':
    primary.clock.main(experiment())
