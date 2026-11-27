"""Count resident rows and CPU publication dependencies in the retained DFlash2 trace."""

from collections import Counter
import fcntl
from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
import dflash2_decode_residency_replay as replay
from measurement_guards import no_model_owners


COPY_RESULT = '4a3526351f0230fe7120164e0a81782c796d1e75'
COPY_KEY = 'dflash2_decode_result_copy_v1'
HERE = str(Path(__file__).resolve().relative_to(replay.ROOT))
WIDTH = 4
EXPERTS_PER_TOKEN = 10


def census(revision, data, windows, masks, formats, observed):
    capture = json.loads(replay.git('show', replay.CAPTURE + ':' + replay.EVIDENCE))[replay.KEY]
    sample = capture['processes'][0]['evidence']['samples'][-1]
    plan, = capture['plans']
    config_bytes = (Path(plan['root']) / 'config.json').read_bytes()
    replay.require(replay.sha(config_bytes) == plan['generated']['config.json'], 'captured configuration changed')
    config = json.loads(config_bytes)
    args = config['args']
    replay.require('--no-spec-split' in args and all(args.count(k) == 1 and args[args.index(k) + 1] == value
                       for k, value in (('--pcie-frac', '0'), ('--adapt-swaps', '0'), ('--spec', str(WIDTH))))
                   and config['env'].get('STRATA_DEC_BATCH', '1') == '1',
                   'require unsplit, batched stock decode with fixed residency and no PCIe expert route')
    binary = capture['binary']
    executable = Path(binary['path'])
    replay.require(binary == plan['binary'] and executable.stat().st_size == binary['bytes']
                   and replay.sha(executable.read_bytes()) == binary['sha256'], 'captured decode executable changed')
    component = json.loads(replay.git('show', COPY_RESULT + ':' + replay.EVIDENCE))[COPY_KEY]
    replay.require(component['stage'] == 'closed' and component['complete'] and component['passed']
                   and component['ownership_closed'] and not component['model_speed_qualified'],
                   'require the closed copy-only component comparison')

    widths = [width for _, width in windows]
    hits = Counter()
    publication_layers = Counter()
    publication_jobs, publication_rows, publication_bytes = Counter(), Counter(), Counter()
    cpu_jobs = cpu_entries = 0
    for width, frame in zip(widths, replay.window_routes(data, widths), strict=True):
        for rows, mask, fmt in zip(frame, masks, formats, strict=True):
            jobs = Counter(expert for row in rows for expert in row)
            resident = 0
            for expert, count in jobs.items():
                if mask >> expert & 1:
                    resident += count
                else:
                    cpu_jobs += 1
                    cpu_entries += count
            replay.require(sum(jobs.values()) == width * EXPERTS_PER_TOKEN, 'routed row count changed')
            if width == WIDTH:
                hits[resident] += 1
                first = {expert for row in rows[:2] for expert in row if not (mask >> expert & 1)}
                second = {expert for row in rows[2:] for expert in row if not (mask >> expert & 1)}
                groups = dict(first_only=first - second, shared=first & second, second_only=second - first)
                late_only = len(groups['second_only'])
                # The dispatcher's first-seen expert order already places every
                # first-pair dependency before the second-pair-only work.
                order = [expert for expert in jobs if not (mask >> expert & 1)]
                replay.require(set(order[:len(first)]) == first and len(order) == len(first) + late_only,
                               'first-pair dependency prefix disagrees with route order')
                publication_layers.update({
                    'no_first_cpu_dependencies': int(not first),
                    'no_shared_experts': int(not groups['shared']),
                    'has_second_only_work': int(late_only > 0),
                    'first_dependencies_at_most_half_with_second_only_jobs': int(0 < len(order) and len(first) <= late_only),
                })
                for group, experts in groups.items():
                    count = len(experts)
                    publication_jobs[group] += count
                    publication_rows[group] += sum(jobs[expert] for expert in experts)
                    publication_bytes[group] += count * fmt[2]
    width_four_layers = widths.count(WIDTH) * replay.LAYERS
    replay.require(cpu_jobs == observed['cpu_jobs'] and cpu_entries == observed['cpu_entries']
                   and sum(hits.values()) == width_four_layers > 0, 'resident mask replay differs from native work')
    replay.require(sum(publication_rows.values()) == sum((WIDTH * EXPERTS_PER_TOKEN - resident) * cells
                                                        for resident, cells in hits.items()),
                   'publication dependencies must cover every width-four CPU row')

    compared = component['plan']['hits']
    return dict(source_commit=revision, captured_commit=replay.CAPTURE, route_sha256=observed['routing_sha256'],
                captured_binary_sha256=binary['sha256'], membership_sha256=observed['membership_sha256'],
                copy_evidence_commit=COPY_RESULT, input_tokens=sample['read_tokens'],
                output_tokens=sample['generated_tokens'], width_windows=dict(sorted(Counter(widths).items())),
                width_four_layers=width_four_layers, exact_hit_counts={n: hits[n] for n in compared},
                eligible_exact_layers=sum(hits[n] for n in component['eligible_hit_counts']),
                bands=[sum(hits[n] for n in range(lo, hi + 1)) for lo, hi in ((0, 3), (4, 10), (11, 19), (20, 40))],
                most_common=sorted(hits.items(), key=lambda row: (-row[1], row[0]))[:6],
                cpu_jobs=cpu_jobs, cpu_entries=cpu_entries,
                width_four_publication=dict(layers=dict(publication_layers), jobs=dict(publication_jobs),
                    routed_cpu_rows=dict(publication_rows), logical_blob_bytes=dict(publication_bytes),
                    first_pair_cpu_dependencies_are_ordered_prefix=True))


if __name__ == '__main__':
    base = Path.home() / '.local/state/strata-sycl/post-reboot-tuning-20261007'
    with (base / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        no_model_owners([])
        revision, data, windows, masks, formats, observed = replay.inputs()
        replay.require(Path(__file__).read_bytes() == replay.git('show', revision + ':' + HERE),
                       'commit the census before replay')
        print(json.dumps(census(revision, data, windows, masks, formats, observed), sort_keys=True))
