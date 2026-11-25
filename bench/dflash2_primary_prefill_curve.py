"""Measure and fit the primary DFlash2 mode-16 prefill route."""

import math
from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_primary_prefill_speed as speed

bank, basis, require = speed.bank, speed.basis, speed.require
KEY = 'dflash2_primary_prefill_curve_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
READS = (94209, 102401, 110593, 118785, 126977)
HOLDOUT = 110593
FORWARD = (110593, 94209, 126977, 102401, 118785)
ORDER = FORWARD + FORWARD[::-1]
PLAN = dict(order=list(ORDER), actual_input_tokens=list(READS), held_out=HOLDOUT,
    requests_per_process=speed.RUNS, final_chunk=4096,
    scope='Two opposite-order blocks on the selected primary KV arm. Keep mode16, bank32, '
          'direct attention, copied input, demand upload and decode settings fixed.',
    fit='Fit affine, quadratic and n log n separately on first-use and subsequent medians '
        'for engine PP time and client TTFT. Exclude the reserved length from all fits. '
        'Report every prediction, residual and independent-process spread.',
    stop='Stop on the first changed route, allocation, work, input or safety outcome. '
         'Preserve each observed timing. No replacement length or automatic adoption.')


def baseline():
    base = speed.baseline()
    result = basis.validation.committed(bank.RESULT, (speed.KEY,))
    require(result['stage'] == 'closed' and result['passed'] and result['complete']
            and result['ownership_closed'] and not result['active']
            and all(result[k] == base[k] for k in ('binary', 'native_source_commit', 'native_build_sha256'))
            and result['decision'] == speed.decision(result['processes']),
            'finish the matched primary KV comparison on this build')
    verdict = result['decision']
    base.pop('request_hashes')
    return dict(base, selected_arm='retain' if verdict['all_pairs_faster'] else 'reload',
                selection_sha256=bank.signature(verdict))


def environment(row, plan):
    return speed.environment(row, dict(plan, arm=row['selected_arm']))


def command(plan):
    cmd = speed.command(plan)
    cmd[cmd.index('--prefill-exact-tokens') + 1] = str(plan['arm'])
    return cmd


def route(text, plan, *, batched_upload=False):
    n = plan['read'] - 1
    require(plan['read'] in READS and n % speed.CHUNK == 4096, 'the curve requires the fixed partial chunk')
    pieces = (n + speed.CHUNK - 1) // speed.CHUNK
    layout = speed.workspace.workspace(n, split=True, relocated=True)
    one = speed.workspace.one
    main = one(text, 'strata prefill: layer-major candidate, ')
    mode = one(text, 'strata MoE window configuration: ')
    bank_owner = one(text, 'strata MoE expert bank: ')
    admit = one(text, 'strata MoE accumulator admission: ')
    split = one(text, 'strata MoE split admission: ')
    mixed = one(text, 'strata MoE mixed banks: ')
    finish = one(text, 'strata MoE long workspace complete: ')
    tail = one(text, 'strata MoE split complete: ')
    work = one(text, 'strata MoE accumulator window: ')
    uploads = one(text, 'strata MoE expert bank complete: ')
    require(main['tokens'] == main['residual_gpu_rows'] == mode['maximum_rows'] == str(n)
            and main['chunk'] == '8192' and main['residual_dtype'] == 'bf16'
            and main['residual_element_bytes'] == '2' and int(main['residual_gpu']) == n * 20480
            and main['residual_host'] == main['residual_staging'] == '0'
            and mode['requested_mode'] == mode['effective_mode'] == '16'
            and mode['mixed_fp16_bytes'] == '0' and work['maximum_rows'] == str(n)
            and work['windows'] == '48' and work['operator_max'] == '8192'
            and work['GPU_resident_windows_only'] == '1', 'curve changed the all-GPU mode16 route')
    cache = speed.SLOTS * speed.workspace.STRIDE
    prefix = n * 2056 + 256
    fixed = cache + prefix + n * 20480 + layout['extra'] + (512 << 20)
    require(bank_owner['staged'] == '1' and bank_owner['requested_slots'] == bank_owner['effective_slots'] == '32'
            and int(bank_owner['device_bytes']) == int(main['layer_cache']) == cache
            and bank_owner['residual_gpu_rows'] == str(n)
            and bank_owner['general_map'] == 'private_scratch_never_published'
            and admit['rows'] == admit['gpu_prefix_rows'] == split['rows'] == str(n)
            and int(admit['fixed_including_headroom_bytes']) == fixed <= int(admit['free_gpu_bytes'])
            and max(cache, prefix, n * 20480, layout['extra']) <= int(split['max_single_allocation_bytes']),
            'curve changed total/single-allocation admission or the staged bank')
    require(split['active'] == '1' and split['decode_changed'] == '0' and split['primary_rows'] == '65536'
            and int(split['tail_payload_bytes']) == layout['tail_payload']
            and int(split['extra_allocation_bytes']) == layout['extra']
            and mixed['storage'] == 'scratch' and mixed['extra_allocation_bytes'] == '0'
            and int(mixed['tail_offset']) == layout['region']
            and int(mixed['primary_bytes']) == layout['mixed_primary']
            and int(mixed['tail_bytes']) == layout['mixed_tail']
            and layout['region'] + layout['mixed_tail'] <= int(admit['shared_arena_bytes'])
            and finish == dict(rows=str(n), extra_bytes=str(layout['extra']), guards='3_checked',
                final_drain='checked', decode_changed='0')
            and tail == dict(rows=str(n), primary_rows='65536', tail_bytes=str(layout['tail_payload']),
                guard='checked', final_drain='checked', decode_changed='0'), 'relocated planes/guards differ')
    require(uploads['final_drain'] == 'checked' and int(uploads['uploads']) + int(uploads['permanent_hits'])
            == int(work['dequants']) > 0, 'actual upload count or bank retirement changed')
    batch_prefix = 'strata MoE split expert upload: '
    if batched_upload:
        require(uploads['permanent_slots'] == uploads['permanent_hits'] == '0'
                and one(text, batch_prefix) == dict(windows='48', experts=uploads['uploads'],
                    copy_commands=uploads['uploads'], final_drain='checked'),
                'actual batched upload count, copy commands or bank retirement changed')
    else:
        require(batch_prefix not in text, 'demand upload unexpectedly used a batch')
    check_attention(text, n, pieces)
    request, = re.findall(speed.clock.footer.HEADER, text, re.M)
    return dict(work=work, uploads=uploads, kv=kv_pages(text, n, plan['kv_arm']),
                fixed_bytes=fixed, free_bytes=int(admit['free_gpu_bytes']), direct_pieces=48 * pieces,
                native_request=list(map(int, request)))


def check_attention(text, n, pieces):
    one = speed.workspace.one
    drafter = one(text, 'strata DFlash2: eager, ')
    require(one(text, 'strata prefill attention input: ') == dict(requested='0', eligible=str(46 * pieces),
                borrowed='0', borrowed_bytes='0', requested_D2D_bytes=str(46 * n * 20480))
            and one(text, 'strata prefill attention bank: ') == dict(requested='1',
                pieces=str(48 * pieces), direct=str(48 * pieces), requested_D2D_bytes='0')
            and 'strata prefill heads: draft_reader=absent-dflash2' in text
            and 'strata prefill heads: target detached ' in text
            and drafter['prompt_tail_rows'] == '2046' and drafter['cold_prefix'] == '0'
            and int(drafter['vram_bytes']) > 0,
            'attention copy, final partial direct store, or draft handoff changed')


def kv_pages(text, n, arm):
    kv = speed.physical.kv
    expected = [(layer, *kv.expected_prefix(range(0, n, speed.CHUNK), layer,
                 reuse=arm == 'retain' and layer not in speed.physical.TAPPED_QSA_LAYERS)[:2])
                for layer in range(3, 48, 4)]
    actual = [tuple(map(int, m.groups())) for line in text.splitlines() if (m := kv.STAGE.fullmatch(line))]
    require(actual == expected and text.count('strata prefill KV stage:') == len(expected),
            'curve KV page ownership or partial-chunk work changed')
    return dict(reloaded=sum(row[1] for row in actual), reused=sum(row[2] for row in actual))


def fit(points, holdout, feature=None):
    # Orthogonalize the third predictor against [1, x] to avoid subtracting
    # nearly equal large normal-equation terms over the narrow length range.
    train = [n for n in READS if n != holdout]
    x = [n / 65536 for n in train]
    y = [points[n] for n in train]
    line = statistics.linear_regression(x, y)
    c = None
    if feature:
        z = [feature(t) for t in x]
        p = statistics.linear_regression(x, z)
        z_residual = [v - p.intercept - p.slope * t for t, v in zip(x, z, strict=True)]
        y_residual = [v - line.intercept - line.slope * t for t, v in zip(x, y, strict=True)]
        c = math.fsum(a * b for a, b in zip(z_residual, y_residual, strict=True)) / math.fsum(a * a for a in z_residual)
        line = statistics.linear_regression(x, [v - c * t for v, t in zip(y, z, strict=True)])

    def evaluate(n):
        t = n / 65536
        return line.intercept + line.slope * t + (c * feature(t) if feature else 0.0)

    forecast = evaluate(holdout)
    return dict(coefficients_ms=[line.intercept, line.slope, c] if feature else [line.intercept, line.slope],
                training_residual_ms=[points[n] - evaluate(n) for n in train],
                held_out_forecast_ms=forecast, held_out_error_ms=forecast - points[holdout],
                held_out_relative_error=abs(forecast - points[holdout]) / points[holdout])


def characterize(processes):
    require(tuple(p['read'] for p in processes) == ORDER
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(ORDER), 'need both independent length blocks')
    results = {}
    for phase in ('first', 'subsequent'):
        for field, scale in (('prompt_ms', 1), ('ttft_s', 1000)):
            values = {n: [] for n in READS}
            for p in processes:
                samples = p['samples'][:1] if phase == 'first' else p['samples'][1:]
                values[p['read']].append(statistics.median(s[field] for s in samples) * scale)
            points = {n: statistics.median(v) for n, v in values.items()}
            require(all(math.isfinite(v) and v > 0 for v in points.values()), 'invalid independent length clocks')
            results[phase + '_' + field] = dict(process_ms=values, median_ms=points,
                adjacent_ms_per_input_token=[(points[b] - points[a]) / (b - a)
                    for a, b in zip(READS, READS[1:])],
                affine=fit(points, HOLDOUT), quadratic=fit(points, HOLDOUT, lambda x: x * x),
                n_log_n=fit(points, HOLDOUT, lambda x: x * math.log2(x)))
    return dict(series=results, input_scale='n/65536', held_out=HOLDOUT,
                used_for_fitting=[n for n in READS if n != HOLDOUT], adopted=False)


def experiment():
    ctx = speed.clock.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.protocol, ctx.baseline = PLAN, baseline
    ctx.configuration, ctx.command, ctx.environment = speed.generation.configuration, command, environment
    speed_pins = speed.experiment().pins
    prior_save, prior_current = ctx.save, ctx.current

    def pins(revision):
        paths = speed_pins(revision)
        actual = bank.P / THIS
        require(actual.read_bytes() == basis.validation.git('show', revision + ':' + THIS),
                'commit the length controller before preparation')
        return dict(paths, **{THIS: bank.sha(actual)})

    def save(row):
        if row['stage'] == 'prepared':
            row['request_hashes'] = {}
            for plan in row['plans']:
                plan.update(read=plan['arm'], heads=speed.HEADS, kv_arm=row['selected_arm'])
        prior_save(row)

    def current(row):
        require(all((p['read'], p['heads'], p['kv_arm']) == (p['arm'], speed.HEADS, row['selected_arm'])
                    for p in row['plans']), 'frozen extent or KV policy differs')
        hashes = {str(p['read']): [s['request_sha256'] for s in p['samples']] for p in row['processes']}
        require(row['request_hashes'] == hashes, 'request body history differs')
        return prior_current(row)

    def after_execution(outcome, row, plan):
        def work(p):
            return [{k: v for k, v in r.items() if k != 'free_bytes'} for r in p['routes']]
        require(outcome['clean_speed'] and outcome['request_routes_checked_before_continue'] == speed.RUNS
                and all(work(p) == work(outcome) for p in row['processes'] if p['read'] == plan['read']),
                'repeated length changed route or expert work')
        read = plan['read']
        for frame, sample in zip(outcome['routes'], outcome['samples'], strict=True):
            require(frame['native_request'] == [read, 0, read, 24, sample['drafts_accepted'], sample['drafts_offered']],
                    'native and client request work differs')

    def run(index):
        runner = basis.core.source.device.load_module('primary_prefill_curve_timing', basis.comparison.__file__)
        original_models = runner.shared.models
        runner.shared = SimpleNamespace(**vars(runner.shared))
        row = bank.previous.read(ROOT / 'record.json')

        def model_pins(config, assets, full=False):
            original_models.model_pins(row['configuration'], assets, full=full)
            for flag in ('--draft-model', '--mtp-draft-vocab'):
                path = config['args'][config['args'].index(flag) + 1]
                require(bank.sha(path) == row['dflash2_assets'][path], 'selected DFlash2 input changed')

        def validate_model(text, _):
            require(text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
                    and not re.search(r'(?m)^strata mtp: .*loaded|strata loan audit:|strata DFlash2 tap payload:', text),
                    'unexpected decode head, MTP layer or observer')

        runner.shared.models = SimpleNamespace(model_pins=model_pins)
        runner.fingerprint = bank.signature
        policy = SimpleNamespace(ROOT=ROOT, ORDER=[(n, speed.HEADS) for n in ORDER], RUNS=speed.RUNS,
            current=current, native_context=lambda r, _: r, environment=environment, save=save,
            request_route=route, after_execution=after_execution, decision=characterize, validate_model_text=validate_model)
        runner.run(index, policy)

    ctx.pins, ctx.save, ctx.current, ctx.run = pins, save, current, run
    return ctx


if __name__ == '__main__':
    speed.clock.main(experiment())
