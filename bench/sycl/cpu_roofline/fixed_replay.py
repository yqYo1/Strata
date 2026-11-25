#!/usr/bin/env python3
"""Select and measure fixed native CPU expert work from a closed DFlash2 capture."""

import fcntl
import hashlib
import itertools
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path
import re
import statistics
import struct
import sys

sys.dont_write_bytecode = True
import iq3_pair as recipe
import cpu_roofline as calibration
import cpu_work_census as census
import measurement_guards as guards
import measurement_owner as owned

P, B = recipe.P, recipe.B
ROOT = B / 'cpu-fixed-expert-replay-v1'
BUILD = ROOT / 'build'
RESULT = recipe.RESULT.with_name('cpu-fixed-expert-replay-current.json')
CAPTURE_COMMIT = 'b28a9372bc1943812a87d170402435cc1c7425f0'
CAPTURE_KEY = 'dflash2_long_routing_holdout_v1'
ROUTE = struct.Struct('<ii10i40s')
SELECTED = '931fa301815eb3824492e10c571fcd0e63d1f3124ff9d09a3b93df305c080933'
PROTOCOL = (
    'Six physical-core counts, with the caller participating only in the six-core pool arm. '
    'Use the same real expert weights, seeded native activations and ordered jobs throughout. '
    'The scan reads an independent contiguous buffer equal to the unique weight footprint. '
    'Match worker cores, caller placement, first touch and caller participation between workloads. '
    'Three independent process rounds per cell, five scored batches per process. '
    'Hashing, allocation, quantizing initial activations, output validation and warmup precede timing. '
    'Production pool scheduling, GU, hidden quantization and Down are timed. '
    'Logical weight bytes and the sequential-read reference are component evidence.'
)
REQUIRED = ('bench/sycl/cpu_roofline/main.cpp', 'src/kernels/cpu/pool.cpp',
            'src/kernels/cpu/native_expert.cpp', 'src/kernels/cpu/iq_avx2.cpp')
PROJECT = (*recipe.PROJECT, 'tools/sycl/cpu_roofline.py', 'tools/sycl/cpu_work_census.py')
CHECK_ORDER = [(0, cores, mode) for cores in (1, 6) for mode in ('pool', 'scan')]
ORDER = [(round_id, cores, mode)
         for round_id, counts in enumerate((range(1, 7), range(6, 0, -1), (3, 6, 1, 5, 2, 4)))
         for cores in counts for mode in (('scan', 'pool') if round_id == 1 else ('pool', 'scan'))]
need, sha, git = recipe.need, owned.sha, recipe.git


def atomic_json(path, data):
    pending = path.with_suffix('.tmp')
    recipe.write(pending, data)
    pending.replace(path)


def fixture_from_capture():
    path = recipe.RESULT.relative_to(P).as_posix()
    capture = json.loads(git(P, 'show', CAPTURE_COMMIT + ':' + path))[CAPTURE_KEY]
    process, = capture['processes']
    need(capture['stage'] == 'closed' and capture['passed'] and not capture['active']
         and capture['ownership_closed'] and process['passed'] and process['complete']
         and process['native_exit_code'] == 0 and process['ownership_closed'] and not process['fresh_faults'],
         'closed qualified held-out capture required')
    case = Path(capture['plans'][0]['root'])
    execution = json.loads((case / 'execution.json').read_text())
    need(not execution['active'] and execution['commands'] and
         all(owned.closed(c) and c['exit_code'] == 0 for c in execution['commands']),
         'capture execution ownership')
    trace = case / 'expert-routing.dat'
    observed = process['routes']
    need(trace.stat().st_size == observed['routing_bytes'] <= 64 << 20
         and sha(trace) == observed['routing_sha256'], 'original routing capture changed')
    assets = capture['routing_metadata']
    need(all(sha(assets[k]) == assets[k + '_sha256'] for k in ('manifest', 'profile'))
         and Path(assets['manifest']) == calibration.METADATA, 'original native format or cache profile changed')
    membership = census.expected_membership(assets['profile'])
    need(hashlib.sha256(b''.join(x.to_bytes(64, 'little') for x in membership)).hexdigest()
         == observed['membership_sha256'], 'captured resident-expert membership')
    format_rows = census.manifest_formats(assets['manifest'])
    layers = [r for r in calibration.formats() if (r['gu'], r['down']) in ((18, 20), (21, 20))]
    layer_ids = [r['layer'] for r in layers]
    need(layer_ids == [0, 17], 'first matching native layer for each format')
    log_path = case / 'engine.stderr'
    need(log_path.stat().st_size <= 1 << 20, 'unbounded original routing log')
    log = log_path.read_text()
    headers = list(re.finditer(r'^strata serve: prompt (\d+) tokens[^\n]*$', log, re.M))
    samples = process['evidence']['samples']
    need(len(headers) == len(samples) == 5 and
         all(int(h[1]) == s['prompt_tokens'] for h, s in zip(headers, samples)), 'capture request bounds')
    windows = re.findall(r'^strata trace: window (\d+) (\d+)$',
                         log[headers[-2].end():headers[-1].start()], re.M)
    positions, widths = map(list, zip(*((int(pos), int(w)) for pos, w in windows), strict=True))
    need(len(widths) == observed['windows'] and sum(widths) == observed['target_rows']
         and sum(widths) - len(widths) == samples[-1]['drafts_offered']
         and (positions[0], widths[0]) == (samples[-1]['prompt_tokens'] - 1, 1)
         and all(1 <= w <= 4 and 0 <= p <= 262144 - w for p, w in zip(positions, widths))
         and all(1 <= b - a <= w for a, b, w in zip(positions, positions[1:], widths)),
         'capture verifier window conservation')
    selected = []
    for width in (2, 4):
        matching = [i for i, value in enumerate(widths) if value == width]
        need(len(matching) >= 4, 'insufficient windows at selected width')
        selected.extend(matching[len(matching) * q // 4] for q in (1, 3))
    suffix = trace.stat().st_size - sum(widths) * 48 * ROUTE.size
    need(suffix >= 0 and suffix % ROUTE.size == 0, 'capture trace extent')
    offsets = list(itertools.accumulate(widths, initial=0))
    blocks = []
    with trace.open('rb') as stream:
        for window in sorted(selected):
            width = widths[window]
            for layer in layer_ids:
                stream.seek(suffix + (offsets[window] * 48 + layer * width) * ROUTE.size)
                jobs = {}
                for t in range(width):
                    actual_layer, fanout, *ids, weights = ROUTE.unpack(stream.read(ROUTE.size))
                    need((actual_layer, fanout) == (layer, 10) and weights == bytes(40)
                         and len(set(ids)) == 10 and all(0 <= e < 512 for e in ids),
                         'invalid original batch routing row')
                    for expert in ids:
                        if not (membership[layer] >> expert) & 1:
                            jobs.setdefault(expert, []).append(t)
                blocks.append(dict(window=window, width=width, layer=layer, fmt=format_rows[layer],
                                   jobs=list(jobs.items())))
    encoded = json.dumps(blocks, separators=(',', ':'))
    need(hashlib.sha256(encoded.encode()).hexdigest() == SELECTED, 'original ordered replay selection changed')
    unique = {(b['layer'], e) for b in blocks for e, _ in b['jobs']}
    return dict(capture=dict(commit=CAPTURE_COMMIT, key=CAPTURE_KEY, trace_sha256=observed['routing_sha256']),
                descriptor_sha256=SELECTED, blocks=encoded, layers=layers,
                unique_blobs=len(unique), unique_bytes=sum(format_rows[l][2] for l, _ in unique),
                jobs=sum(len(b['jobs']) for b in blocks),
                entries=sum(len(rows) for b in blocks for _, rows in b['jobs']),
                logical_bytes=sum(b['fmt'][2] * len(b['jobs']) for b in blocks))


def native_descriptor(fixture):
    text = ['strata-expert-replay-v1']
    layers = fixture['layers']
    for layer in layers:
        values = [layer[k] for k in ('gu', 'down', 'blob_bytes', 'gate', 'up', 'down_offset')]
        text.append(' '.join(map(str, values)) + ' ' + json.dumps(layer['shard'], ensure_ascii=False))
    ordinal = {layer['layer']: i for i, layer in enumerate(layers)}
    for block in json.loads(fixture['blocks']):
        jobs = block['jobs']
        text.append(f"{block['window']} {block['width']} {ordinal[block['layer']]} {len(jobs)}")
        text.extend(' '.join(map(str, (expert, len(rows), *rows))) for expert, rows in jobs)
    return '\n'.join(text) + '\n'


def weights(fixture):
    layers = {r['layer']: r for r in fixture['layers']}
    unique = sorted({(b['layer'], e) for b in json.loads(fixture['blocks']) for e, _ in b['jobs']})
    identities = {r['shard']: recipe.identity(r['shard']) for r in layers.values()}
    digest = hashlib.sha256()
    total = 0
    with ExitStack() as stack:
        streams = {p: stack.enter_context(Path(p).open('rb', buffering=0)) for p in identities}
        for layer, expert in unique:
            row = layers[layer]
            for offset, size in ((row['gate'], row['gu_bytes']), (row['up'], row['gu_bytes']),
                                 (row['down_offset'], row['down_bytes'])):
                start = offset + expert * size
                need(0 <= start <= identities[row['shard']]['bytes'] - size, 'expert shard slice bounds')
                data = os.pread(streams[row['shard']].fileno(), size, start)
                need(len(data) == size, 'short expert shard slice')
                digest.update(data)
                total += size
    need(len(unique) == fixture['unique_blobs'] and total == fixture['unique_bytes'] and
         all(recipe.identity(path) == identity for path, identity in identities.items()), 'fixed weight extent or identity')
    return dict(shards=identities, sha256=digest.hexdigest(), bytes=total)


def build_plan(root, revision, *, project_scope=PROJECT, definitions=()):
    build_root = root / 'build'
    need(git(recipe.GGML, 'rev-parse', 'HEAD').decode().strip() == recipe.GGML_REV, 'pinned GGML source')
    project = recipe.pins(P, revision, project_scope)
    ggml = recipe.pins(recipe.GGML, recipe.GGML_REV, ('ggml',))
    need(sha(recipe.SDK) == recipe.SDK_SHA, 'pinned SDK manifest changed')
    allowed = json.loads(recipe.SDK.read_text())
    need(len(allowed) == 10945 and all(sha(p) == h for p, h in allowed.items()), 'SDK input drift')
    env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
               LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib', OMP_NUM_THREADS='1',
               STRATA_IQ_SINGLE_EXACT='0', STRATA_IQ3_PAIR='0', STRATA_CPU_SERVICE='0',
               TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
    allowed.update(recipe.host_inventory(env)['files'])
    tools = {str(Path(p).resolve()): sha(p) for p in
             (recipe.CC, recipe.CXX, '/usr/bin/cmake', '/usr/bin/ninja', '/usr/bin/ldd', sys.executable)}
    allowed.update(tools)
    for directory in Path('/usr/share').glob('cmake-*/Modules'):
        allowed.update({str(p.resolve()): sha(p) for p in directory.rglob('*') if p.is_file()})
    root.mkdir(mode=0o700)
    for name in ('cache', 'tmp'):
        (root / name).mkdir()
    external = root / 'external-inputs.json'
    atomic_json(external, allowed)
    header = '#pragma once\n\n#define GGML_VERSION "0.24.0"\n#define GGML_COMMIT  "3cf03257f"\n'
    return dict(stage='prepared', source_commit=revision, project_inputs=project, ggml_inputs=ggml,
               machine=calibration.machine(), tools=tools, environment=env,
               metadata_sha256=sha(calibration.METADATA), ggml_revision=recipe.GGML_REV,
               external_inputs=dict(path=str(external), sha256=sha(external), files=len(allowed)),
               generated_expected={str(build_root / 'ggml/src/ggml-version.h'): hashlib.sha256(header.encode()).hexdigest()},
               configure=['/usr/bin/cmake', '-S', str(P / 'bench/sycl/cpu_roofline'), '-B', str(build_root),
                          '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                          '-DCMAKE_C_COMPILER=' + recipe.CC, '-DCMAKE_CXX_COMPILER=' + recipe.CXX,
                          '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t', '-DGGML_CCACHE=OFF',
                          '-DSTRATA_GGML_DIR=' + str(recipe.GGML), *definitions],
               build=['/usr/bin/cmake', '--build', str(build_root), '--target', 'cpu_roofline', '--parallel', '4'],
               budgets=dict(build_wall=300, run_wall=45, build_rss=8 << 30, run_rss=2 << 30,
                            text=4 << 20, file=64 << 20, total=384 << 20),
               processes=[], active=False, ownership_closed=True, gpu_executed=False)


def prepare():
    need(not ROOT.exists() and not RESULT.exists(), 'one fresh replay plan')
    revision = git(P, 'rev-parse', 'HEAD').decode().strip()
    fixture = fixture_from_capture()
    weight_pin = weights(fixture)
    row = build_plan(ROOT, revision)
    target = ROOT / 'replay.txt'
    target.write_text(native_descriptor(fixture))
    row.update(fixture=fixture, weights=weight_pin, protocol=PROTOCOL, replay_input_sha256=sha(target),
               order=[list(c) for c in ORDER], check_order=[list(c) for c in CHECK_ORDER])
    save(row)
    print('Prepared fixed CPU replay; commit the plan before compiling.', flush=True)


def save(row):
    atomic_json(ROOT / 'record.json', row)
    atomic_json(RESULT, recipe.compact(row))


def load():
    row = json.loads((ROOT / 'record.json').read_text())
    relative = RESULT.relative_to(P).as_posix()
    need(RESULT.read_bytes() == git(P, 'show', 'HEAD:' + relative)
         and json.loads(RESULT.read_text()) == recipe.compact(row), 'commit exact replay state first')
    need(recipe.pins(P, row['source_commit'], PROJECT) == row['project_inputs']
         and recipe.pins(recipe.GGML, recipe.GGML_REV, ('ggml',)) == row['ggml_inputs'], 'replay source changed')
    need(row['machine'] == calibration.machine() and all(sha(p) == h for p, h in row['tools'].items())
         and row['metadata_sha256'] == sha(calibration.METADATA), 'machine, tool or model manifest changed')
    target = ROOT / 'replay.txt'
    need(sha(target) == row['replay_input_sha256'] and target.read_text() == native_descriptor(row['fixture'])
         and row['check_order'] == [list(c) for c in CHECK_ORDER]
         and row['order'] == [list(c) for c in ORDER] and row['protocol'] == PROTOCOL, 'replay input or plan drift')
    return row


def verify_binary(row):
    proof = row['build_evidence']['compiler_inputs']
    binary = row['binary']
    need(sha(proof['path']) == proof['sha256'] and Path(binary['path']).stat().st_size == binary['bytes']
         and sha(binary['path']) == binary['sha256']
         and all(sha(p) == h for p, h in row['runtime_inputs'].items()), 'replay build or runtime inputs changed')
    need(sha(ROOT / 'replay.txt') == row['replay_input_sha256'] and
         all(recipe.identity(path) == identity for path, identity in row['weights']['shards'].items()),
         'route descriptor or model shard identity changed')


def summaries(row, text, mode, cores, timed, *, claim_group=None):
    rows = [json.loads(line) for line in text.splitlines()]
    by_kind = {kind: [v for v in rows if v['kind'] == kind]
               for kind in ('replay_placement', 'replay_input', 'timer_check', 'sample', 'closed')}
    need(sum(map(len, by_kind.values())) == len(rows) and
         by_kind['closed'] == [dict(kind='closed', passed=True)] and rows[-1] == by_kind['closed'][0],
         'complete CPU replay output required')
    fixture = row['fixture']
    host, *available = row['machine']['physical_cpus']
    workers = available[:min(cores, 5)]
    placement, = by_kind['replay_placement']
    need(placement['host_cpu'] == host and placement['workers'] == workers
         and placement['caller_computes'] == (cores == 6), 'replay caller and worker placement')
    if mode == 'pool':
        actual, = by_kind['replay_input']
        need(all(actual[k] == fixture[k] for k in ('jobs', 'entries', 'logical_bytes', 'unique_blobs', 'unique_bytes'))
             and actual['batches'] == 8 and actual['threads'] == cores and actual['host_works'] == (cores == 6)
             and actual['serial_output_equal'] and actual['verified_affinity'], 'pool replay conservation or numerics')
        if claim_group is not None:
            need(actual['gu'] == fixture['layers'][0]['gu'] and actual['claim_group'] == claim_group
                 and actual['gu_claims_per_pass'] == fixture['batches'] * ((3 * cores + claim_group - 1) // claim_group)
                 and actual['bitwise_output_equal'], 'actual grouped Gate/Up work or bitwise parity differs')
    else:
        need(not by_kind['replay_input'], 'unexpected expert work in the read control')
    result = dict(mode=mode, cores=cores, placement=placement)
    samples = by_kind['sample']
    need(len(samples) == (5 if timed else 0), 'scored replay sample count')
    if not timed:
        need(not by_kind['timer_check'], 'unexpected timed check')
        return result
    timer, = by_kind['timer_check']
    need(0 < timer['query_s'] < timer['fastest_pass_s'] * .001, 'timer overhead admission')
    byte_count = fixture['logical_bytes' if mode == 'pool' else 'unique_bytes']
    visits = fixture['jobs'] if mode == 'pool' else 1
    expected_mode = 'replay-gu-groups' if claim_group is not None else ('replay' if mode == 'pool' else 'scan')
    for rep, sample in enumerate(samples):
        repeats = sample['iterations']
        need(sample['mode'] == expected_mode and sample['threads'] == cores
             and (claim_group is None or sample['claim_group'] == claim_group)
             and sample['rep'] == rep and sample['attempt'] == 0 and 0 < repeats < 1000000
             and sample['requested_bytes'] == repeats * byte_count and sample['visits'] == repeats * visits
             and math.isfinite(sample['wall_s']) and sample['wall_s'] >= 1
             and math.isfinite(sample['cpu_s'])
             and 0 <= sample['host_cpu_s'] <= sample['cpu_s'] + .005
             and sample['host_cpu_s'] <= sample['wall_s'] + .005, 'replay timing or work admission')
        if claim_group is not None:
            need(sample['gu_claims'] == repeats * actual['gu_claims_per_pass'],
                 'timed Gate/Up claim count differs from its validated replay')
        if mode == 'pool':
            phases = [sample[k] for k in ('gu_s', 'q_s', 'down_s')]
            need(all(math.isfinite(v) and v >= 0 for v in phases) and sum(phases) <= sample['wall_s'] + .001,
                 'bounded disjoint pool stage times')
    return dict(result, samples=samples, timer=timer,
                median_s_per_pass=statistics.median(v['wall_s'] / v['iterations'] for v in samples))


def build_fixture(row, scope, run, build_root):
    allowed = recipe.input_map(row)
    run('configure', row['configure'], build=True)
    run('compile', row['build'], build=True)
    run('deps', ['/usr/bin/ninja', '-C', str(build_root), '-t', 'deps'])
    row['build_evidence'] = recipe.build_check(row, scope, build_root=build_root, required_units=REQUIRED)
    binary = build_root / 'cpu_roofline'
    row['binary'] = dict(path=str(binary), bytes=binary.stat().st_size, sha256=sha(binary))
    _, out, _ = run('runtime', ['/usr/bin/ldd', str(binary)])
    info = out.read_text()
    row['runtime_inputs'] = {str(Path(p).resolve()): sha(Path(p).resolve()) for p in re.findall(r'(/[^\s]+)', info)}
    need('not found' not in info and row['runtime_inputs']
         and not re.search(r'sycl|libze_|libur_|cuda|amdhip', info, re.I)
         and all(allowed.get(p) == h for p, h in row['runtime_inputs'].items()), 'unpinned CPU runtime')


def execute(stage):
    row = load()
    expected = dict(build='prepared', check='built', measure='checked')
    need(row['stage'] == expected[stage] and not row['active'] and row['ownership_closed'],
         'close and commit the preceding stage before advancing')
    need(len(row['processes']) == (len(CHECK_ORDER) if stage == 'measure' else 0), 'unexpected prior replay cases')
    scope = ROOT / ('owned-' + stage)
    need(not scope.exists(), 'inspect the prior owned stage; no retry')

    def persist():
        atomic_json(ROOT / 'execution.json', dict(stage=stage, active=bool(owner.active), commands=owner.commands))

    owner = owned.Owner(scope, persist=persist)

    def run(label, argv, *, build=False):
        entry, out, err = owner.run(label, argv, row['environment'], ROOT,
            wall=row['budgets']['build_wall' if build else 'run_wall'],
            rss_cap=row['budgets']['build_rss' if build else 'run_rss'],
            cpu=1200 if build else 180, text_cap=row['budgets']['text'],
            file_cap=row['budgets']['file'], total_cap=row['budgets']['total'])
        need(owned.closed(entry) and entry['exit_code'] == 0,
             label + ' failed: ' + err.read_text(errors='replace')[-1500:])
        return entry, out, err

    try:
        if stage == 'build':
            build_fixture(row, scope, run, BUILD)
            row['stage'] = 'built'
        else:
            verify_binary(row)
            need(weights(row['fixture']) == row['weights'], 'original model weight slices changed')
            cases = CHECK_ORDER if stage == 'check' else ORDER
            for i, (round_id, cores, mode) in enumerate(cases):
                verify_binary(row)
                args = (['replay', str(ROOT / 'replay.txt'), str(cores), stage] if mode == 'pool'
                        else ['replay-scan', str(cores), str(row['fixture']['unique_bytes']), stage])
                entry, out, err = run(f'case-{i}', [row['binary']['path'], *args])
                need(err.stat().st_size == 0, 'replay emitted stderr')
                data = summaries(row, out.read_text(), mode, cores, stage == 'measure')
                row['processes'].append(dict(round=round_id, **data, owner=recipe.receipt(entry)))
            need(weights(row['fixture']) == row['weights'], 'expert weight slices changed during replay')
            if stage == 'check':
                row['stage'] = 'checked'
            else:
                measured = row['processes'][len(CHECK_ORDER):]
                need(len({(x['owner']['pid'], x['owner']['direct_identity']['start_ticks'])
                          for x in measured}) == len(ORDER), 'independent replay process count')
                row['medians_s_per_pass'] = {mode: {
                    str(cores): statistics.median(x['median_s_per_pass'] for x in measured
                                                   if x['mode'] == mode and x['cores'] == cores)
                    for cores in range(1, 7)} for mode in ('pool', 'scan')}
                row['stage'] = 'closed'
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        raise
    finally:
        row.update(active=bool(owner.active), ownership_closed=not owner.active and all(owned.closed(e) for e in owner.commands))
        save(row)
    print(row['stage'], row.get('medians_s_per_pass', {}), flush=True)


if __name__ == '__main__':
    need(sys.executable == '/home/yayoi/.local/share/strata-sycl/venv/bin/python3' and len(sys.argv) == 2
         and sys.argv[1] in ('prepare', 'build', 'check', 'measure'), 'pinned Python and prepare|build|check|measure required')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            execute(sys.argv[1])
