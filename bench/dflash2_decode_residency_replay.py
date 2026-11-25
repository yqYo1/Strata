"""Compare fixed-quota expert rankings on captured DFlash2 decode routes."""

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
                 'tools/sycl/measurement_guards.py'):
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


def count_routes(data, widths):
    offset = len(data) - sum(widths) * LAYERS * ROUTE.size
    require(offset >= 0 and offset % ROUTE.size == 0, 'unbound routing extent')
    counts = [[Counter() for _ in range(LAYERS)] for _ in range(2)]
    empty_weights = bytes(40)
    for window, width in enumerate(widths):
        half = counts[int(window >= len(widths) // 2)]
        for layer in range(LAYERS):
            jobs = Counter()
            for _ in range(width):
                actual, fanout, *experts, weights = ROUTE.unpack_from(data, offset)
                offset += ROUTE.size
                require((actual, fanout, weights) == (layer, 10, empty_weights)
                        and len(set(experts)) == 10 and all(0 <= e < EXPERTS for e in experts), 'bad route')
                jobs.update(experts)
            half[layer].update(jobs.items())
    require(offset == len(data), 'trailing capture bytes')
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
    return dict(source_commit=revision, capture_commit=CAPTURE, route_sha256=observed['routing_sha256'],
                train_windows=center, holdout_windows=len(windows) - center,
                holdout_starts_at_prompt_position=windows[center][0], cache_slots=sum(quota),
                aligned_cache_bytes=sum(n * ((f[2] + 255) // 256 * 256) for n, f in zip(quota, formats, strict=True)),
                conserved_incumbent=full, results=results)


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2))
