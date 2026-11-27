"""Measure sustained generation after the selected primary DFlash2 long prefill."""

from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_primary_prefill_curve as curve
import dflash2_primary_retained as retained
import prefill_split_continuation as continuation

speed, bank, basis, require = curve.speed, curve.bank, curve.basis, curve.require
clock, footer = speed.clock, speed.clock.footer
KEY = 'dflash2_primary_continuation_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('dflash2',) * 3
WORKLOADS = continuation.WORKLOADS
PLAN = dict(order=list(ORDER), input_case=continuation.INPUT_CASE,
    scope='Three independent processes on the measured primary binary. Use the two maintained split-v1 '
          'thinking workloads to measure sustained decode after actual mode16 split prefill.',
    route='Keep the selected KV stage retention, direct attention store, BF16 all-GPU residuals, '
          '8192-row chunks, demand uploads and private 32-slot bank. Preserve decode placement and '
          'DFlash2 spec4/confidence0.6. Check actual route and request extent, including a ragged tail.',
    evidence='Require both exact count-preflight bodies and their fresh native read extents. '
             'Require a visible answer, >=1024 actual native outputs, quiet validation and normal owner closure. '
             'Separate first and subsequent long-request scope, report observed draft classes and all TG samples. '
             'This checks the long-route handoff. The standard agent and independent performance comparisons '
             'have separate plans.',
    stop='Stop on source, runtime, workload, route, ownership or fault failure. Preserve the outcome. '
         'No speed-based exclusion, alternate extent or automatic adoption.')


def long_route(frame, read):
    n, chunk = read - 1, speed.CHUNK
    one = speed.workspace.one
    residual = one(frame, 'strata prefill: layer-major candidate, ')
    mode = one(frame, 'strata MoE window configuration: ')
    admission = one(frame, 'strata MoE accumulator admission: ')
    cache = one(frame, 'strata MoE expert bank: ')
    work = one(frame, 'strata MoE accumulator window: ')
    uploads = one(frame, 'strata MoE expert bank complete: ')
    split = one(frame, 'strata MoE split complete: ')
    shared = one(frame, 'strata MoE long workspace complete: ')
    require(residual['tokens'] == residual['residual_gpu_rows'] == admission['gpu_prefix_rows']
            == mode['maximum_rows'] == work['maximum_rows'] == str(n)
            and residual['residual_dtype'] == 'bf16' and residual['residual_element_bytes'] == '2'
            and residual['residual_host'] == residual['residual_staging'] == '0'
            and int(residual['residual_gpu']) == n * 20480
            and int(residual['chunk']) == int(work['operator_max']) == chunk
            and mode['requested_mode'] == mode['effective_mode'] == '16'
            and work['windows'] == '48' and work['GPU_resident_windows_only'] == '1',
            'mode16 and all-GPU residual placement required after count-only admission')
    require(cache['requested_slots'] == cache['effective_slots'] == str(speed.SLOTS)
            and cache['staged'] == '1' and cache['general_map'] == 'private_scratch_never_published'
            and cache['residual_gpu_rows'] == str(n)
            and int(admission['fixed_including_headroom_bytes']) <= int(admission['free_gpu_bytes'])
            and int(cache['device_bytes']) == speed.SLOTS * speed.workspace.STRIDE
            and 0 < int(uploads['uploads']) + int(uploads['permanent_hits']) == int(work['dequants'])
            and uploads['final_drain'] == split['final_drain'] == shared['final_drain'] == 'checked'
            and shared['guards'] == '3_checked' and split['guard'] == 'checked'
            and shared['decode_changed'] == split['decode_changed'] == '0'
            and shared['rows'] == split['rows'] == str(n)
            and 'strata MoE split expert upload: ' not in frame, 'bank, demand uploads or handoff changed')
    pieces = (n + chunk - 1) // chunk
    curve.check_attention(frame, n, pieces)
    require(n > 81920, 'expected the long split accumulator')
    return dict(residual_gpu_rows=n, direct_pieces=48 * pieces, kv=curve.kv_pages(frame, n, 'retain'),
                gpu_headroom_bytes=int(admission['free_gpu_bytes']) - int(admission['fixed_including_headroom_bytes']),
                bank_h2d_requested_bytes=int(uploads['requested_H2D_bytes']), placement_preserved=True)


def routes(text, samples, arm, *, profile_check=None, workloads=WORKLOADS):
    require(arm == 'dflash2' and text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
            and not re.search(r'(?m)^strata mtp: .*loaded|strata loan audit:|strata DFlash2 tap payload:|'
                              r'^strata (?:trace:|decode timing:)', text), 'unexpected DFlash2 route or observer')
    if profile_check is None:
        continuation.standard.profiles.check_profile_startup(text, 'original')
    else:
        profile_check(text)
    headers = list(re.finditer(footer.HEADER, text, re.M))
    expert = list(re.finditer(footer.COPY, text, re.M))
    require(len(headers) == len(expert) == len(samples), 'one native request and expert footer per wire request')
    checked = {}
    for i, (header, copy, sample) in enumerate(zip(headers, expert, samples, strict=True)):
        start = headers[i - 1].end() if i else 0
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        fields = ('prompt_tokens', 'reused_tokens', 'read_tokens', 'generated_tokens', 'drafts_accepted', 'drafts_offered')
        require(header.end() <= copy.start() < end
                and tuple(map(int, header.groups())) == tuple(sample[k] for k in fields),
                'native and client request work differs')
        if 'workload' not in sample:
            continue
        read = sample['read_tokens']
        require(90114 <= read <= 117489 and sample['reused_tokens'] == 0 and sample['generated_tokens'] >= 1024,
                'insufficient fresh long-prefill or native generation work')
        counts = dict(zip(footer.COPY_KEYS, map(int, copy.groups()), strict=True))
        require(all(counts[k] == 0 for k in footer.COPY_KEYS[:6])
                and 0 < counts['cpu_unique_misses'] <= counts['cpu_entries'], 'post-prefill expert route changed')
        checked[sample['workload']] = dict(long_route(text[start:header.start()], read), expert_work=counts)
    require(tuple(checked) == workloads, 'the selected split-prefill decode handoffs are required')
    return checked


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER and all(p['passed'] and p['complete']
            and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER),
            'three independent complete primary-binary processes required')
    cases = {name: dict(requests=[s for p in processes for s in p['evidence']['samples']
                                 if s.get('workload') == name]) for name in WORKLOADS}
    rates = {name: [p['evidence']['decode_rates'][name] for p in processes] for name in WORKLOADS}
    return dict(decode_tps=rates, decode_median={name: statistics.median(v) for name, v in rates.items()},
                observed_draft_classes=continuation.suite.official.d.client.long_generation.classify_pair(cases, len(ORDER)),
                sustained_70_all_samples=all(min(samples) >= 70 for samples in rates.values()), adopted=False)


def experiment():
    ctx = clock.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.WORKLOAD_ID = continuation.WORKLOAD_ID
    ctx.configuration, ctx.environment, ctx.command = speed.generation.configuration, curve.environment, continuation.command
    ctx.routes, ctx.protocol, ctx.baseline = routes, PLAN, retained.baseline
    ctx.evidence_reader.read_complete_suite, ctx.evidence_reader.decide = continuation.read_complete_suite, decision
    base_pins = speed.experiment().pins

    def pins(revision):
        values = base_pins(revision)
        profile = continuation.standard.profiles
        for name in (THIS, retained.THIS, curve.THIS, clock.THIS, continuation.THIS, continuation.standard.THIS,
                     profile.THIS, *profile.profile.SOURCES):
            file = bank.P / name
            require(file.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact continuation input: ' + name)
            values[name] = bank.sha(file)
        return values

    ctx.pins = pins
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
