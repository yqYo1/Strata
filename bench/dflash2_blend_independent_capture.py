"""Test the frozen blended expert map on the separate 96K decode routes."""

from collections import Counter
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_decode_blended_rank_replay as blend
import dflash2_primary_continuation as primary


bank, basis, require = primary.bank, primary.basis, primary.require
replay, quota = blend.replay, blend.quota
KEY = 'dflash2_blend_independent_capture_v2'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
TRAINED = '03d0b3cac'
LONG_REFERENCE = '9261c4b44'
BLEND = 'dflash2_decode_blended_rank_replay_v1'
TRACE = 'expert-routing.dat'
ORDER = primary.WORKLOADS
PLAN = dict(primary.PLAN, order=list(ORDER),
    scope='One stock DFlash2 process per existing 96K thinking request. Dump routing only to score the '
          'already selected 75%-code map at the same per-layer placements and packed-byte budget.',
    route='Require the retained binary and ordinary 96K prefill/draft handoff. Use fixed resident masks. '
          'Require the same independent request bodies as the closed primary continuation and bind their '
          'last trace frames to actual window, row, CPU-job and CPU-entry counts. Match warm-ups across '
          'processes while requiring distinct long bodies.',
    evidence='Reproduce the selected rank digest before both captures. Advance only if logical packed bytes '
             'and CPU jobs both fall on both preserved request routes. The trace is diagnostic; resume '
             'normal measurement without it for native placement and speed. Keep stock selected.')


def placement():
    old = json.loads(replay.git('show', TRAINED + ':' + replay.EVIDENCE))[BLEND]
    require(basis.validation.committed(bank.RESULT, (BLEND,)) == old
            and old['advance_to_independent_capture'] and old['selected_code_weight_quarters'] == 3,
            'require the original frozen blended ranking')
    _, data, windows, masks, formats, observed = replay.inputs()
    archive = json.loads(replay.git('show', replay.CAPTURE + ':' + replay.EVIDENCE))
    code, code_hash, code_windows = blend.code_counts(archive, masks, formats)
    prose, suffix = replay.count_routes(data, [width for _, width in windows])
    blend.verify_stock([a + b for a, b in zip(prose, suffix, strict=True)], masks, formats, observed)
    workloads = dict(code_train=code, prose_train=prose, prose_suffix=suffix)
    stock = {name: {k: v for k, v in replay.score(counts, masks, formats).items() if k in blend.METRICS}
             for name, counts in workloads.items()}
    require(stock == old['stock'] and code_hash == old['code_usage_sha256']
            and code_windows == old['code_train_windows'] and observed['routing_sha256'] == old['prose_trace_sha256'],
            'training inputs changed')
    order, slots, sizes, widths = quota.stock_geometry(replay.ROOT / quota.PROFILE, masks, formats)
    candidate = quota.ranked_slots(slots, order, blend.blend(code, prose, 3, stock), lambda pair: pair[0])
    ranking, requested = quota.global_profile(candidate, order, sizes)
    target = old['candidates'][3]
    require(quota.rank_sha256(ranking) == target['ranking_sha256'] and requested == target['requested_expert_cache']
            and candidate[:quota.PROTECTED] == slots[:quota.PROTECTED]
            and [sizes[layer] for layer, _ in candidate] == widths and sum(widths) == old['cache_bytes'],
            'the preselected rank, protected prefix or packed slot layout changed')
    training = {archive[k]['processes'][0]['evidence']['samples'][-1]['request_sha256']
                for k in (blend.CODE_RUN, replay.KEY)}
    return masks, quota.masks(candidate), formats, target['ranking_sha256'], sorted(training)


def baseline():
    row = primary.retained.baseline()
    _, _, _, digest, training = placement()
    prior = bank.previous.historic(LONG_REFERENCE, primary.KEY)
    require(primary.retained.closed(prior, 'closed') and all(prior[k] == row[k] for k in (
            'binary', 'native_source_commit', 'native_build_sha256')), 'require the measured independent 96K bodies')
    bodies = {}
    for arm in ORDER:
        checks = [s for p in prior['processes'] for s in p['evidence']['samples'] if s.get('workload') == arm]
        pairs = {(s['request_sha256'], s['read_tokens']) for s in checks}
        ((sha, tokens),) = pairs
        require(len(checks) == 3 and sha not in training and all(s['reused_tokens'] == 0 for s in checks),
                'holdout must use a known fresh body excluded from training')
        bodies[arm] = dict(request_sha256=sha, read_tokens=tokens)
    return dict(row, frozen_rank_sha256=digest, rank_training_request_sha256=training, independent_bodies=bodies)


def config(original, row, plan):
    cfg = original(row, dict(plan, arm='dflash2'))
    args = cfg['args']
    require('--dump-routing' not in args and '--no-spec-split' in args and all(
            args.count(flag) == 1 and args[args.index(flag) + 1] == value for flag, value in (
                ('--pcie-frac', '0'), ('--adapt-swaps', '0'), ('--spec', '4'))), 'unexpected stock trace route')
    return dict(cfg, args=[*args, '--dump-routing', str(Path(plan['root']) / TRACE)])


def routes(text, samples, arm):
    clean = re.sub(r'(?m)^strata trace:[^\n]*\n', '', text)
    validated = primary.routes(clean, samples, 'dflash2', workloads=(arm,))[arm]
    sample = samples[-1]
    headers = list(re.finditer(primary.footer.HEADER, text, re.M))
    frame = text[headers[-2].end():headers[-1].start()]
    windows = [tuple(map(int, x)) for x in re.findall(r'^strata trace: window (\d+) (\d+)$', frame, re.M)]
    widths = [n for _, n in windows]
    require(windows and windows[0] == (sample['prompt_tokens'] - 1, 1)
            and sum(widths) - len(widths) == sample['drafts_offered']
            and all(1 <= n <= 4 and 0 <= pos <= 262144 - n for pos, n in windows)
            and all(1 <= following - pos <= n for (pos, n), (following, _) in zip(windows, windows[1:])),
            'held-out verifier frames or target rows differ')
    row = bank.previous.read(ROOT / 'record.json')
    require(sample['workload'] == arm and {k: sample[k] for k in ('request_sha256', 'read_tokens')}
            == row['independent_bodies'][arm], 'holdout request body differs')
    trace = ROOT / str(ORDER.index(arm)) / TRACE
    require(trace.is_file() and trace.stat().st_size <= row['budgets']['file_cap'], 'missing or oversized route trace')
    data = trace.read_bytes()
    masks, candidate, formats, digest, _ = placement()
    require(digest == row['frozen_rank_sha256'], 'ranking changed before scoring')
    counts = [Counter() for _ in masks]
    for window in replay.window_counts(data, widths):
        for layer, jobs in zip(counts, window, strict=True):
            layer.update(jobs.items())
    stock, shadow = (replay.score(counts, layout, formats) for layout in (masks, candidate))
    native = validated['expert_work']
    require(stock['cpu_jobs'] == native['cpu_unique_misses'] and stock['cpu_entries'] == native['cpu_entries'],
            'stock replay disagrees with the native request')
    return dict(workload=arm, validated_route=validated, request_sha256=sample['request_sha256'],
                trace_bytes=len(data), routing_sha256=replay.sha(data), windows=len(windows), target_rows=sum(widths),
                stock=stock, blended=shadow)


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER and all(p['passed'] and p['complete']
            and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER),
            'both independent trace processes must close normally')
    results = {p['arm']: {k: p['routes']['blended'][k] / p['routes']['stock'][k] for k in blend.METRICS}
               for p in processes}
    return dict(candidate_over_stock=results, advance_to_native_placement=all(
                max(m.values()) < 1 for m in results.values()), model_speed_qualified=False, adopted=False)


def match_holdouts(current, previous):
    require(len(current) == len(previous) == 5 and (previous[-1]['workload'], current[-1]['workload']) == ORDER
            and current[-1]['request_sha256'] != previous[-1]['request_sha256'],
            'require distinct code and prose holdout bodies after the standard warm-ups')
    primary.clock.suite.official.matched(current[:-1], previous[:-1])


def experiment():
    ctx = primary.experiment()
    original_config, original_env, original_command, original_pins = ctx.configuration, ctx.environment, ctx.command, ctx.pins
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.protocol, ctx.baseline, ctx.routes = PLAN, baseline, routes
    ctx.configuration = lambda row, plan: config(original_config, row, plan)
    ctx.environment = lambda row, plan: dict(original_env(row, dict(plan, arm='dflash2')), STRATA_TRACE='1')
    ctx.command = lambda plan: [*original_command(plan), '--long-generation-workload', plan['arm']]
    ctx.evidence_reader.read_complete_suite = lambda root, name: primary.continuation.read_complete_suite(
        root, name, workloads=(ORDER[int(root.name)],))
    ctx.evidence_reader.decide = decision
    ctx.match_samples = match_holdouts

    def pins(revision):
        values = original_pins(revision)
        names = (THIS, 'tools/make_profile.py', primary.clock.suite.THIS,
                 *(str(Path(mod.__file__).resolve().relative_to(bank.P)) for mod in (blend, replay, quota)))
        for name in names:
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact rank and route reader: ' + name)
            values[name] = bank.sha(path)
        return values

    ctx.pins = pins
    return ctx


if __name__ == '__main__':
    primary.clock.main(experiment())
