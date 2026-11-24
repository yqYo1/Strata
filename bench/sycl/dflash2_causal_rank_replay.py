"""Score early-routing-only expert placements on the unchanged suffix of a closed trace."""

from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_expert_rank_forecast as forecast

bank, capture, route, census = forecast.bank, forecast.capture, forecast.routing, forecast.census
require = forecast.require
HOLDOUT_COMMIT = 'b28a9372bc1943812a87d170402435cc1c7425f0'
SCORE_COMMIT = '20a14c34c388b4ba9612e73c21f36407cc29db08'
TRANSFER_REFERENCE = '68f8c61b6bee758d4ce74963d2aafbfead495dea'
HORIZONS = (64, 256, 1024)
SWAP_LIMITS = (32, 96, 256)
COPY_GBS = (6.292, 6.485)


def digest(data):
    return sha256(data).hexdigest()


def scan_intervals(path, widths, boundaries):
    """Split validated verifier windows into adjacent intervals."""
    histograms = [Counter() for _ in range(48)]
    intervals = []
    with path.open('rb') as stream:
        size = stream.seek(0, 2)
        start = size - sum(widths) * len(histograms) * route.ROUTE_ROW.size
        require(start >= 0 and start % route.ROUTE_ROW.size == 0, 'trace does not contain the verified suffix')
        stream.seek(start)
        for i, width in enumerate(widths, 1):
            for layer, hist in enumerate(histograms):
                counts = Counter()
                for _ in range(width):
                    row = stream.read(route.ROUTE_ROW.size)
                    actual, k, *ids, weights = route.ROUTE_ROW.unpack(row)
                    require(actual == layer and k == 10 and weights == route.ZERO_WEIGHTS
                            and len(set(ids)) == k and all(0 <= e < 512 for e in ids),
                            'unexpected verifier routing row')
                    counts.update(ids)
                hist.update(counts.items())
            if i in boundaries:
                intervals.append(histograms)
                histograms = [Counter() for _ in range(48)]
        require(stream.tell() == size and not stream.read(1), 'unexpected routing suffix')
    require(len(intervals) == len(boundaries), 'missing interval boundary')
    return intervals


def as_usage(slices):
    usage = []
    for layer in range(48):
        visits = Counter()
        for segment in slices:
            visits.update(segment[layer])
        rows = {}
        for (expert, nt), count in visits.items():
            rows.setdefault(expert, [0] * 4)[nt - 1] = count
        usage.extend((layer, e, hits) for e, hits in sorted(rows.items()))
    return usage


def physical_slots(rank):
    return [[(layer, expert) for expert in experts[:cap]]
            for layer, (experts, cap) in enumerate(zip(forecast.by_layer(rank), forecast.CAPACITY, strict=True))]


def transition(before, after, formats):
    newly_admitted = changed_slots = new_payload = moved_payload = 0
    for stock, trained, (_, _, bytes_per_slot) in zip(physical_slots(before), physical_slots(after), formats, strict=True):
        new = len(set(trained) - set(stock))
        changed = sum(a != b for a, b in zip(stock, trained, strict=True))
        newly_admitted += new
        changed_slots += changed
        new_payload += new * bytes_per_slot
        moved_payload += changed * bytes_per_slot
    require(newly_admitted <= changed_slots <= forecast.CACHE_SLOTS - forecast.PREFIX,
            'swap changes the protected cache prefix')
    return dict(new_members=newly_admitted, changed_slots=changed_slots, new_payload_bytes=new_payload,
                ordered_slot_payload_bytes=moved_payload,
                new_payload_service_s_at_reference=[new_payload / (g*1e9) for g in reversed(COPY_GBS)])


def fit_transfer_budget(exchanges, formats, limit, payload_bytes, *, by_efficiency):
    """Pick distinct exchanges within the same outbound and restore byte limit."""
    require(limit > 0 and payload_bytes > 0 and len(formats) == 48
            and all(n > 0 for _, _, n in formats), 'invalid transfer budget')
    if by_efficiency:
        exchanges = sorted(exchanges, key=lambda row: (Fraction(row[0], formats[row[1]][2]), row))
    selected = []
    for row in exchanges:
        size = formats[row[1]][2]
        if size > payload_bytes:
            continue
        selected.append(row)
        payload_bytes -= size
        if len(selected) == limit:
            break
    return selected


def limited_rank(stock_rank, early_usage, limit, *, rows=False, minimum_gain=1, transfer=None):
    """Rank exchanges using early requests, with an optional limit on copied bytes."""
    visits = {(l, e): sum((i+1 if rows else 1)*n for i, n in enumerate(counts))
              for l, e, counts in early_usage}
    layer_ranks = forecast.by_layer(stock_rank)
    exchanges, offset = [], 0
    for layer, (rank, cap) in enumerate(zip(layer_ranks, forecast.CAPACITY, strict=True)):
        locked = min(cap, max(0, forecast.PREFIX - offset))
        offset += cap
        score = lambda expert: visits.get((layer, expert), 0)
        victims = sorted(range(locked, cap), key=lambda slot: score(rank[slot]))
        proposals = sorted(rank[cap:], key=score, reverse=True)
        for slot, expert in zip(victims, proposals):
            gain = score(expert) - score(rank[slot])
            if gain < minimum_gain:
                break
            exchanges.append((-gain, layer, slot, expert))
    ordered = sorted(exchanges)
    if transfer is None:
        selected = ordered[:limit]
    else:
        formats, size_limit, by_efficiency = transfer
        selected = fit_transfer_budget(ordered, formats, limit, size_limit, by_efficiency=by_efficiency)
    changed = {}
    for _, layer, slot, expert in selected:
        admitted = changed.setdefault(str(layer), layer_ranks[layer][:forecast.CAPACITY[layer]])
        admitted[slot] = expert
    return forecast.apply_delta(stock_rank, changed)


def compare_transfer_budget(past, future, stock, base, formats, validated):
    prior = bank.previous.historic(TRANSFER_REFERENCE, 'dflash2_causal_rank_replay_v1')
    require(prior['only_existing_trace'] and prior['record_commit'] == HOLDOUT_COMMIT
            and prior['trace_sha256'] == validated['routing_sha256'], 'recover the same qualified held-out trace')
    early, = (cell for cell in prior['evaluated'] if cell['early_windows'] == 256)
    anchor, = (row for row in early['matched_rank_metric'] if row['metric'] == 'routed-rows' and row['cap'] == 96)
    budget = anchor['requested_new_weight_bytes']
    require(budget == 197_913_600 and anchor['minimum_gain'] == 2, 'original transfer reference differs')
    stock_work = route.projected_cpu_expert_work(future, base, formats)
    require(stock_work['cpu_jobs'] == early['stock']['cpu_jobs'], 'stock suffix changed')
    comparisons = []
    for metric in ('unique-jobs', 'routed-rows'):
        for efficient in (False, True):
            rank = limited_rank(stock, past, 96, rows=(metric == 'routed-rows'), minimum_gain=2,
                                transfer=(formats, budget, efficient))
            moved = transition(stock, rank, formats)
            require(moved['changed_slots'] == moved['new_members'] <= 96
                    and moved['ordered_slot_payload_bytes'] <= budget, 'exchange exceeds the frozen cap')
            work = route.projected_cpu_expert_work(future, forecast.membership(rank), formats)
            comparisons.append(dict(metric=metric, priority='gain-per-byte' if efficient else 'gain',
                work=work, transition=moved, jobs_removed=stock_work['cpu_jobs'] - work['cpu_jobs']))
    reference, = (row for row in comparisons if row['metric'] == 'routed-rows' and row['priority'] == 'gain')
    require(reference['work']['cpu_jobs'] == anchor['cpu_jobs']
            and reference['transition']['new_payload_bytes'] == budget,
            'gain-first under its own byte ceiling failed to reproduce')
    return dict(early_windows=256, max_swaps=96, max_bytes_each_direction=budget,
                stock=stock_work, rows=comparisons, prior_commit=TRANSFER_REFERENCE)


def main(*, transfer_budget=False):
    key = capture.experiment('holdout').KEY
    original = bank.previous.historic(HOLDOUT_COMMIT, key)
    score = bank.previous.historic(SCORE_COMMIT, forecast.KEY)
    require(original['stage'] == 'closed' and original['passed'] and original['complete']
            and original['ownership_closed'] and not original['active'], 'require a closed qualified routing capture')
    process, = original['processes']
    require(process['native_exit_code'] == 0 and process['ownership_closed'] and process['fresh_faults'] == [],
            'held-out owner or device outcome differs')
    require(score['holdout']['candidate_freeze'] == original['rank_candidate_freeze'],
            'held-out placement plan differs')
    role = capture.experiment('holdout')
    raw = bank.previous.read(role.ROOT / 'record.json')
    require(role.compact(raw) == original, 'original capture receipt differs')
    root = Path(original['plans'][0]['root'])
    exit_record = bank.previous.read(root / 'engine-exit.json')
    require(exit_record['exit_code'] == 0, 'original native exit differs')
    execution = bank.previous.read(root / 'execution.json')
    require(not execution['active'], 'capture owner is still active')
    sid = str(process['native_identity']['sid'])
    members = subprocess.run(['ps', '-o', 'pid=', '--sid', sid], capture_output=True, timeout=15)
    require(members.returncode == 1 and not members.stdout.strip(), 'original native session is still present')
    log = (root / 'engine.stderr').read_bytes()
    samples = process['evidence']['samples']
    require(samples[-1]['request_sha256'] == score['holdout']['request_sha256'], 'held-out body differs')
    text = log.decode()
    native, _ = capture.leaves.leaf_reader.evidence(text, samples, 4, False)
    bindings, widths = capture.census_inputs(text, samples, native)
    metadata = original['routing_metadata']
    for key in ('profile','manifest'):
        require(bank.sha(metadata[key]) == metadata[key+'_sha256'], 'profile or expert manifest differs')
    stock_profile = metadata['profile']
    stock_rank = forecast.make_profile.read_profile(stock_profile)
    base = census.expected_membership(stock_profile)
    formats = census.manifest_formats(metadata['manifest'])
    counted = census.extract(text, bindings, formats, base)[-1]
    trace = root / capture.ROUTING_FILE
    validated = process['routes']
    require(trace.stat().st_size == validated['routing_bytes'] and bank.sha(trace) == validated['routing_sha256'],
            'original raw trace identity differs')
    require(len(widths) == validated['windows'] == counted['windows'] and sum(widths) == counted['rows'],
            'number of recorded verifier rows or windows differs')
    whole = route.verifier_routing_usage(trace, widths, counted, formats, base,
                                      max_bytes=original['budgets']['file_cap'])
    recorded = json.loads(Path(validated['usage_file']).read_bytes())['usage']
    require([[l, e, c] for l, e, c in whole] == recorded
            and bank.sha(validated['usage_file']) == validated['usage_sha256'], 'verified expert counts differ')
    require(all(n < len(widths) for n in HORIZONS), 'no held-out suffix after an early interval')
    intervals = scan_intervals(trace, widths, (*HORIZONS, len(widths)))
    require(as_usage(intervals) == whole, 'the intervals do not conserve full-trace counts')
    trained = forecast.apply_delta(stock_rank, score['profile']['profile_delta'])
    headers = list(re.finditer(capture.footer.HEADER, text, re.M))
    frame = text[headers[-2].end():headers[-1].start()]
    positions = [(int(p), int(w)) for p, w in re.findall(r'^strata trace: window (\d+) (\d+)$', frame, re.M)]
    require([w for _, w in positions] == widths and positions[0][0] == samples[-1]['read_tokens']-1,
            'late score must use the final fresh request windows')
    if transfer_budget:
        index = HORIZONS.index(256) + 1
        result = compare_transfer_budget(as_usage(intervals[:index]), as_usage(intervals[index:]),
                                         stock_rank, base, formats, validated)
        print(json.dumps(dict(only_existing_trace=True, record_commit=HOLDOUT_COMMIT,
            native_source=original['native_source_commit'], trace_sha256=validated['routing_sha256'],
            input=samples[-1]['read_tokens'], output=samples[-1]['generated_tokens'],
            transfer_budget=result), indent=2))
        return

    summary = []
    for i, h in enumerate(HORIZONS, 1):
        past, future = as_usage(intervals[:i]), as_usage(intervals[i:])
        policy = route.rank_by_verifier_jobs(stock_rank, [past], cache_slots=forecast.CACHE_SLOTS,
                                             protected_prefix=forecast.PREFIX)
        control, candidate, code_candidate = [route.projected_cpu_expert_work(future, m, formats)
                  for m in (base, forecast.membership(policy), forecast.membership(trained))]
        change = transition(stock_rank, policy, formats)
        capped_candidates = []
        for limit in SWAP_LIMITS:
            rank = limited_rank(stock_rank, past, limit)
            moved = transition(stock_rank, rank, formats)
            require(moved['changed_slots'] == moved['new_members'] <= limit,
                    'bounded swap moves an already resident slot')
            work = route.projected_cpu_expert_work(future, forecast.membership(rank), formats)
            capped_candidates.append(dict(max_swaps=limit, work=work, transition=moved,
                                          jobs_removed=control['cpu_jobs']-work['cpu_jobs']))
        metric_pair = []
        if h == 256:
            for metric in ('unique-jobs', 'routed-rows'):
                for limit in (96, 256):
                    rank = limited_rank(stock_rank, past, limit, rows=(metric == 'routed-rows'), minimum_gain=2)
                    work = route.projected_cpu_expert_work(future, forecast.membership(rank), formats)
                    metric_pair.append(dict(metric=metric, limit=limit, minimum_gain=2, work=work,
                                            transition=transition(stock_rank, rank, formats)))
        summary.append(dict(early_windows=h, remaining_windows=len(widths)-h, early_rows=sum(widths[:h]),
            early_output_tokens=positions[h][0] - positions[0][0],
            stock=control, early_policy=candidate, code_trained=code_candidate, transition=change,
            capped_candidates=capped_candidates, matched_rank_metric=metric_pair))
    print(json.dumps(dict(only_existing_trace=True, record_commit=HOLDOUT_COMMIT, score_commit=SCORE_COMMIT,
        native_source=original['native_source_commit'], native_sha256=original['binary']['sha256'],
        trace_sha256=validated['routing_sha256'],
        stderr_sha256=digest(log), horizons=HORIZONS, input=samples[-1]['read_tokens'], output=samples[-1]['generated_tokens'],
        evaluated=summary),indent=2))


if __name__ == '__main__':
    require(sys.argv[1:] in ([], ['--transfer-budget']), 'select the original or the fixed-transfer replay')
    main(transfer_budget=bool(sys.argv[1:]))
