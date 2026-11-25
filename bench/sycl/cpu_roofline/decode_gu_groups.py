"""Compare native Gate/Up scheduling with identical captured CPU work."""

import fcntl
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import fixed_replay as fixed

P, B, recipe = fixed.P, fixed.B, fixed.recipe
sys.path.insert(0, str(P / 'bench'))
import dflash2_decode_residency_replay as route

need, sha, git = fixed.need, fixed.sha, fixed.git
KEY = 'cpu_decode_gu_task_groups_v1'
ROOT = B / KEY.replace('_', '-')
RESULT = recipe.RESULT
PROJECT = (*fixed.PROJECT, 'bench/dflash2_decode_residency_replay.py', route.PROFILE)
FORMATS = (18, 21, 22)
GROUPS = (1, 2, 4)
CASES = tuple(f'{gu}-{pattern}' for gu in FORMATS for pattern in ('hot', 'rotating'))
CHECKS = tuple((case, group) for case in CASES for group in GROUPS)
ORDER = tuple((round_id, case, group)
              for round_id, cycle in enumerate((GROUPS, (4, 1, 2), (2, 4, 1)))
              for case in CASES for group in cycle)
PLAN = dict(order=[list(x) for x in ORDER], check_order=[list(x) for x in CHECKS],
            scope='Retain 15 logical GU partitions and identical native calls. Group 1, 2 or 4 existing '
            'partitions per epoch claim in the CPU fixture. Keep five pinned workers and a reserved caller. '
            'The eight submissions per pass are the same across arms. Hidden quantization and Down are fixed.',
            cases='Three real IQ3_XXS, IQ3_S or IQ2_S / IQ4_NL layer formats. Select eight NT1 experts per '
            'batch from eight width-four windows in a held-out suffix. The hot case repeats the first batch. '
            'Rotate at least six distinct previously unselected blobs per later batch.',
            timing='Three cyclic independent process rounds per case. Use the existing warm-up and five '
            'scored batches per process, each at least one second. Report wall and GU time with all samples.',
            decision='A format and group advances only if all three rotating pairs are faster end to end '
            'and all three improve GU time. Keep hot and rotating outcomes separate. Stop at the first '
            'source, actual claim, execution, clock or bitwise output failure. No model speed claim.')


def fixtures():
    _, raw, windows, stock, formats, observed = route.inputs()
    offsets = list(itertools.accumulate((width for _, width in windows), initial=0))
    start = len(raw) - offsets[-1] * route.LAYERS * route.ROUTE.size
    by_format = {f['gu']: f for f in fixed.calibration.formats() if f['down'] == 20 and f['gu'] in FORMATS}
    need(set(by_format) == set(FORMATS) and start >= 0 and start % route.ROUTE.size == 0, 'capture or layer formats differ')
    selection = {}
    for gu in FORMATS:
        layer_info = by_format[gu]
        layer = layer_info['layer']
        taken, union = [], set()
        for w in range(len(windows) // 2, len(windows)):
            _, width = windows[w]
            if width != 4:
                continue
            position = start + (offsets[w] * route.LAYERS + layer * width) * route.ROUTE.size
            jobs = {}
            for row in range(width):
                actual, fanout, *experts, weights = route.ROUTE.unpack_from(raw, position + row * route.ROUTE.size)
                need((actual, fanout, weights) == (layer, 10, bytes(40)) and len(set(experts)) == 10
                     and all(0 <= e < route.EXPERTS for e in experts), 'malformed captured expert route')
                for expert in experts:
                    if not stock[layer] >> expert & 1:
                        jobs.setdefault(expert, []).append(row)
            eligible = [e for e, rows in jobs.items() if len(rows) == 1]
            new = [e for e in eligible if e not in union]
            if len(eligible) < 8 or len(new) < 6:
                continue
            chosen = set((new + [e for e in eligible if e in union])[:8])
            ordered = [(e, jobs[e]) for e in eligible if e in chosen]
            taken.append(dict(window=w, width=width, layer=layer, jobs=ordered))
            union.update(chosen)
            if len(taken) == 8:
                break
        width_bytes = layer_info['blob_bytes']
        need(len(taken) == 8 and formats[layer] == (gu, 20, width_bytes)
             and len(union) * width_bytes > 64 << 20 and 8 * width_bytes <= 32 << 20, 'working sets differ from the plan')
        for name, blocks, n in (('hot', [taken[0]] * 8, 8), ('rotating', taken, len(union))):
            selection[f'{gu}-{name}'] = dict(layers=[layer_info], blocks=json.dumps(blocks, separators=(',', ':')),
                                             batches=8, jobs=64, entries=64, logical_bytes=64 * width_bytes,
                                             unique_blobs=n, unique_bytes=n * width_bytes)
    return selection, dict(commit=route.CAPTURE, route_sha256=observed['routing_sha256'])


def descriptor(fixture):
    return fixed.native_descriptor(fixture).replace('strata-expert-replay-v1', 'strata-gu-group-replay-v1', 1)


def save(row):
    fixed.atomic_json(ROOT / 'record.json', row)
    result = json.loads(RESULT.read_text())
    result[KEY] = recipe.compact(row)
    fixed.atomic_json(RESULT, result)


def prepare():
    need(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'one fresh grouped replay')
    revision = git(P, 'rev-parse', 'HEAD').decode().strip()
    cases, capture = fixtures()
    weights = {name: fixed.weights(cases[name]) for name in CASES}
    row = fixed.build_plan(ROOT, revision, project_scope=PROJECT,
                           definitions=('-DSTRATA_BENCH_DECODE_GU_TASK_GROUPS:BOOL=ON',))
    workload_hashes = {}
    for name, fixture in cases.items():
        path = ROOT / (name + '.txt')
        path.write_text(descriptor(fixture))
        workload_hashes[name] = sha(path)
    row.update(protocol=PLAN, capture=capture, fixtures=cases, weights=weights, workload_sha256=workload_hashes)
    save(row)


def load():
    row = json.loads((ROOT / 'record.json').read_text())
    evidence = git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P)))
    need(RESULT.read_bytes() == evidence and recipe.compact(row) == json.loads(evidence)[KEY], 'commit the preceding stage')
    need(row['protocol'] == PLAN and row['machine'] == fixed.calibration.machine()
         and row['project_inputs'] == recipe.pins(P, row['source_commit'], PROJECT)
         and row['ggml_inputs'] == recipe.pins(recipe.GGML, recipe.GGML_REV, ('ggml',))
         and all(sha(p) == h for p, h in row['tools'].items())
         and sha(fixed.calibration.METADATA) == row['metadata_sha256'], 'source, machine or format changed')
    for name in CASES:
        path = ROOT / (name + '.txt')
        need(sha(path) == row['workload_sha256'][name] and path.read_text() == descriptor(row['fixtures'][name]),
             'grouped replay descriptor changed')
    return row


def binary_check(row):
    binary = row['binary']
    compiler = row['build_evidence']['compiler_inputs']
    need(sha(compiler['path']) == compiler['sha256'] and Path(binary['path']).stat().st_size == binary['bytes']
         and sha(binary['path']) == binary['sha256']
         and all(sha(p) == h for p, h in row['runtime_inputs'].items()), 'replay build or provider changed')


def decision(processes):
    cells = {}
    for case in CASES:
        entries = {(p['round'], p['claim_group']): p for p in processes if p['case'] == case}
        for group in GROUPS[1:]:
            wall, gu = [], []
            for i in range(3):
                a, b = entries[i, 1], entries[i, group]
                wall.append(b['median_s_per_pass'] / a['median_s_per_pass'])
                gu.append(b['median_gu_s_per_pass'] / a['median_gu_s_per_pass'])
            cells[f'{case}-{group}'] = dict(paired_wall_ratio=wall, paired_gu_ratio=gu,
                                            wall_geomean=math.prod(wall) ** (1 / 3),
                                            advances=case.endswith('-rotating') and max(wall + gu) < 1)
    return cells


def execute(stage):
    row = load()
    need(row['stage'] == {'build': 'prepared', 'check': 'built', 'measure': 'checked'}[stage]
         and not row['active'] and row['ownership_closed'], 'close and commit the preceding stage')
    scope = ROOT / ('owned-' + stage)
    need(not scope.exists(), 'inspect the preceding attempt before continuing')
    owner = fixed.owned.Owner(scope, lambda: fixed.atomic_json(
        ROOT / 'execution.json', dict(stage=stage, active=bool(owner.active), commands=owner.commands)))

    def run(label, argv, *, build=False):
        budget = row['budgets']
        entry, out, err = owner.run(label, argv, row['environment'], ROOT,
                                    wall=budget['build_wall' if build else 'run_wall'],
                                    rss_cap=budget['build_rss' if build else 'run_rss'], cpu=1200 if build else 180,
                                    text_cap=budget['text'], file_cap=budget['file'], total_cap=budget['total'])
        need(fixed.owned.closed(entry) and entry['exit_code'] == 0, 'owned CPU replay failed: ' + label)
        return entry, out, err

    try:
        if stage == 'build':
            fixed.build_fixture(row, scope, run, ROOT / 'build')
            row['stage'] = 'built'
        else:
            need(all(fixed.weights(row['fixtures'][name]) == row['weights'][name] for name in CASES), 'weight bytes changed')
            timed = stage == 'measure'
            order = ORDER if timed else [(0, case, group) for case, group in CHECKS]
            start = len(row['processes'])
            for i, (round_id, case, group) in enumerate(order):
                binary_check(row)
                path = ROOT / (case + '.txt')
                need(sha(path) == row['workload_sha256'][case] and
                     all(recipe.identity(p) == v for p, v in row['weights'][case]['shards'].items()), 'arm inputs changed')
                entry, out, err = run(f'case-{i}', [row['binary']['path'], 'replay-gu-groups', str(path), str(group), stage])
                need(err.stat().st_size == 0, 'CPU replay emitted stderr')
                stats = fixed.summaries(dict(row, fixture=row['fixtures'][case]), out.read_text(),
                                        'pool', 5, timed, claim_group=group)
                if timed:
                    value = statistics.median(s['gu_s'] / s['iterations'] for s in stats['samples'])
                    need(value > 0, 'expected a positive GU phase time')
                    stats['median_gu_s_per_pass'] = value
                row['processes'].append(dict(round=round_id, case=case, claim_group=group, owner=recipe.receipt(entry), **stats))
            need(all(fixed.weights(row['fixtures'][name]) == row['weights'][name] for name in CASES), 'weight bytes changed')
            observed = row['processes'][start:]
            need(len({(p['owner']['pid'], p['owner']['direct_identity']['start_ticks']) for p in observed}) == len(order),
                 'expected independent processes')
            if timed:
                row['decision'] = decision(observed)
            row['stage'] = 'closed' if timed else 'checked'
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        raise
    finally:
        row.update(active=bool(owner.active), ownership_closed=not owner.active and all(fixed.owned.closed(x) for x in owner.commands))
        save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    need(sys.executable == '/home/yayoi/.local/share/strata-sycl/venv/bin/python3'
         and action in ('prepare', 'build', 'check', 'measure'), 'use the pinned Python and one planned stage')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixed.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        else:
            execute(action)
