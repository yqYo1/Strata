"""Compare full and partial expert residency on captured DFlash2 decode routes."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/sycl'))
from cpu_work_census import expected_membership, manifest_formats

CAPTURE = 'b28a9372bc1943812a87d170402435cc1c7425f0'
EVIDENCE = 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'dflash2_long_routing_holdout_v1'
ROUTE = struct.Struct('<ii10i40s')
LAYERS, EXPERTS = 48, 512
EMBEDDING, EXPERT_HIDDEN = 2560, 640
# DFlash2 IQ3_S pack formats from ggml-common.h. Validate the derived
# sum against every actual full expert span in the packed manifest.
BLOCKS = {18: (256, 98), 20: (32, 18), 21: (256, 110), 22: (256, 82),
          23: (256, 136), 42: (64, 18)}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inputs():
    revision = git('rev-parse', 'HEAD').decode().strip()
    for name in ('bench/dflash2_decode_residency_replay.py', 'tools/sycl/cpu_work_census.py',
                 'tools/sycl/measurement_guards.py', 'tools/gguf_reader.py', 'third_party/ggml/ggml-common.h'):
        require((ROOT / name).read_bytes() == git('show', revision + ':' + name), 'commit the replay reader')
    capture = json.loads(git('show', CAPTURE + ':' + EVIDENCE))[KEY]
    process, = capture['processes']
    require(capture['stage'] == 'closed' and capture['passed'] and capture['complete']
            and capture['ownership_closed'] and not capture['active'] and process['native_exit_code'] == 0
            and process['ownership_closed'] and process['fresh_faults'] == [], 'capture did not close')
    observed = process['routes']
    assets = capture['routing_metadata']
    for name in ('profile', 'manifest'):
        require(sha(Path(assets[name]).read_bytes()) == assets[name + '_sha256'], 'rank or weights changed')
    baseline = expected_membership(assets['profile'])
    require(sha(b''.join(mask.to_bytes(64, 'little') for mask in baseline)) == observed['membership_sha256'],
            'initial resident membership disagrees with the native capture')
    folder = Path(capture['plans'][0]['root'])
    trace = folder / 'expert-routing.dat'
    require(trace.stat().st_size == observed['routing_bytes'] <= 64 << 20, 'routing extent changed')
    data = trace.read_bytes()
    require(sha(data) == observed['routing_sha256'], 'routing trace changed')
    log = folder / 'engine.stderr'
    require(log.stat().st_size <= 1 << 20, 'native routing log exceeds its recorded envelope')
    text = log.read_text()
    headers = list(re.finditer(r'^strata serve: prompt (\d+) tokens[^\n]*$', text, re.M))
    samples = process['evidence']['samples']
    require(len(headers) == len(samples) == 5
            and [int(h[1]) for h in headers] == [s['prompt_tokens'] for s in samples]
            and samples[-1]['phase'] == 'long-generation/varied-prose', 'capture requests changed')
    windows = [list(map(int, row)) for row in re.findall(r'^strata trace: window (\d+) (\d+)$',
               text[headers[-2].end():headers[-1].start()], re.M)]
    widths = [width for _, width in windows]
    require(len(windows) == observed['windows'] and sum(widths) == observed['target_rows']
            and sum(widths) - len(windows) == samples[-1]['drafts_offered']
            and windows[0] == [samples[-1]['prompt_tokens'] - 1, 1]
            and all(1 <= w <= 4 and 0 <= p <= 262144 - w for p, w in windows)
            and all(1 <= b - a <= w for (a, w), (b, _) in zip(windows, windows[1:])), 'verifier windows changed')
    return revision, data, windows, baseline, manifest_formats(assets['manifest']), observed


def window_counts(data, widths):
    offset = len(data) - sum(widths) * LAYERS * ROUTE.size
    require(offset >= 0 and offset % ROUTE.size == 0, 'unbound routing extent')
    empty_weights = bytes(40)
    for width in widths:
        frame = []
        for layer in range(LAYERS):
            jobs = Counter()
            for _ in range(width):
                actual, fanout, *experts, weights = ROUTE.unpack_from(data, offset)
                offset += ROUTE.size
                require((actual, fanout, weights) == (layer, 10, empty_weights)
                        and len(set(experts)) == 10 and all(0 <= e < EXPERTS for e in experts), 'bad route')
                jobs.update(experts)
            frame.append(jobs)
        yield frame
    require(offset == len(data), 'trailing capture bytes')


def count_routes(data, widths):
    counts = [[Counter() for _ in range(LAYERS)] for _ in range(2)]
    for window, frame in enumerate(window_counts(data, widths)):
        half = counts[int(window >= len(widths) // 2)]
        for histogram, jobs in zip(half, frame, strict=True):
            histogram.update(jobs.items())
    return counts


def score(layers, masks, formats):
    nt = [0] * 4
    logical_bytes = 0
    for layer, mask, fmt in zip(layers, masks, formats, strict=True):
        for (expert, width), count in layer.items():
            if not mask >> expert & 1:
                nt[width - 1] += count
                logical_bytes += count * fmt[2]
    return dict(cpu_jobs=sum(nt), cpu_entries=sum((i + 1) * n for i, n in enumerate(nt)),
                logical_job_blob_bytes=logical_bytes, jobs_NT=nt)


def select(train, baseline, *, for_entries):
    masks = []
    for visits, current in zip(train, baseline, strict=True):
        counts = Counter()
        for (expert, width), count in visits.items():
            counts[expert] += count * (width if for_entries else 1)
        # Break equal scores in favor of existing residents, then expert ID.
        ranked = sorted(range(EXPERTS), key=lambda e: (-counts[e], -((current >> e) & 1), e))
        masks.append(sum(1 << e for e in ranked[:current.bit_count()]))
    return masks


def projection_bytes(layout):
    def matrix(rows, columns, quant):
        block, byte_count = BLOCKS[quant]
        require(columns % block == 0, 'partial quant block')
        return rows * (columns // block) * byte_count

    gu, down, total = layout
    first = 2 * matrix(EXPERT_HIDDEN, EMBEDDING, gu)
    second = matrix(EMBEDDING, EXPERT_HIDDEN, down)
    require(first + second == total, 'projection spans differ from the expert manifest')
    return first, second


def aligned(byte_count):
    return (byte_count + 255) // 256 * 256


def choose_projections(train, baseline, sizes, fraction):
    """Select only with training windows and the original per-layer byte caps."""
    choices = []
    for visits, current, (gu, down) in zip(train, baseline, sizes, strict=True):
        jobs = Counter()
        for (expert, _), count in visits.items():
            jobs[expert] += count
        ordered = sorted(range(EXPERTS), key=lambda e: (-jobs[e], -((current >> e) & 1), e))
        quota = current.bit_count()
        budget = quota * aligned(gu + down)
        prefix = [0]
        for expert in ordered:
            prefix.append(prefix[-1] + jobs[expert])

        def geometry(full):
            remaining = budget - full * aligned(gu + down)
            gu_only = min(EXPERTS - full, remaining // aligned(gu))
            return full, full + gu_only

        counts = [geometry(n) for n in range(quota + 1)] if fraction is None else [geometry(quota * fraction // 4)]
        # Full-resident experts avoid both CPU projections. A partial expert
        # avoids only Gate+Up, so at fixed counts it follows the full experts.
        full, last = max(counts, key=lambda n: (prefix[n[1]] * gu + prefix[n[0]] * down, n[0]))
        require(full * aligned(gu + down) + (last - full) * aligned(gu) <= budget,
                'projection residency exceeds the retained layer budget')
        choices.append((sum(1 << e for e in ordered[:full]), sum(1 << e for e in ordered[full:last])))
    return choices


def projection_score(holdout, choices, sizes):
    gu_jobs = down_jobs = gu_bytes = down_bytes = intermediate_rows = 0
    for visits, (full, gu_only), (gu, down) in zip(holdout, choices, sizes, strict=True):
        for (expert, nt), count in visits.items():
            if full >> expert & 1:
                continue
            down_jobs += count
            down_bytes += count * down
            if gu_only >> expert & 1:
                intermediate_rows += count * nt
            else:
                gu_jobs += count
                gu_bytes += count * gu
    return dict(cpu_gu_jobs=gu_jobs, cpu_down_jobs=down_jobs, cpu_gu_bytes=gu_bytes, cpu_down_bytes=down_bytes,
                total_cpu_weight_bytes=gu_bytes + down_bytes, fp32_intermediate_rows=intermediate_rows,
                requested_fp32_intermediate_D2H_bytes=intermediate_rows * EXPERT_HIDDEN * 4,
                allocated_expert_bytes=sum(full.bit_count() * aligned(gu+down) + gu_only.bit_count() * aligned(gu)
                    for (full, gu_only), (gu, down) in zip(choices, sizes, strict=True)),
                full_slots=sum(full.bit_count() for full, _ in choices),
                gu_only_slots=sum(gu_only.bit_count() for _, gu_only in choices))


def partial_replay(train, holdout, baseline, formats):
    sizes = [projection_bytes(layout) for layout in formats]
    results = {'stock_full': projection_score(holdout, [(s, 0) for s in baseline], sizes)}
    cases = (('ranked_full', 4), ('three_quarters_full', 3), ('half_full', 2),
             ('quarter_full', 1), ('all_gu_only', 0), ('maximize_training_logical_bytes', None))
    for name, fraction in cases:
        plans = choose_projections(train, baseline, sizes, fraction)
        results[name] = projection_score(holdout, plans, sizes)
    return results


def replay():
    revision, data, windows, baseline, formats, observed = inputs()
    train, holdout = count_routes(data, [w for _, w in windows])
    full = score([a + b for a, b in zip(train, holdout, strict=True)], baseline, formats)
    require(all(value == (observed[key][:4] if key == 'jobs_NT' else observed[key])
                for key, value in full.items()), 'replayed baseline does not conserve native work')
    quota = [mask.bit_count() for mask in baseline]
    center = len(windows) // 2
    results = {}
    for name, masks in (('incumbent', baseline), ('first_half_window_jobs', select(train, baseline, for_entries=False)),
                        ('first_half_routed_rows', select(train, baseline, for_entries=True))):
        require([mask.bit_count() for mask in masks] == quota, 'expert cache geometry changed')
        results[name] = dict(replaced_slots=sum((old & ~new).bit_count()
                            for old, new in zip(baseline, masks, strict=True)),
                            train=score(train, masks, formats), holdout=score(holdout, masks, formats))
    projections = partial_replay(train, holdout, baseline, formats)
    stock = projections['stock_full']
    original = results['incumbent']['holdout']
    require(stock['cpu_gu_jobs'] == stock['cpu_down_jobs'] == original['cpu_jobs']
            and stock['total_cpu_weight_bytes'] == original['logical_job_blob_bytes'],
            'the split model does not conserve incumbent full-resident work')
    return dict(source_commit=revision, capture_commit=CAPTURE, route_sha256=observed['routing_sha256'],
                train_windows=center, holdout_windows=len(windows) - center,
                holdout_starts_at_prompt_position=windows[center][0], cache_slots=sum(quota),
                aligned_cache_bytes=sum(n * ((f[2] + 255) // 256 * 256) for n, f in zip(quota, formats, strict=True)),
                conserved_incumbent=full, results=results, projection_residency=projections)


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2))
