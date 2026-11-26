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
PROFILE = 'data/expert-profile.bin'
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
PROJECT = (*recipe.PROJECT, 'tools/sycl/cpu_roofline.py', 'tools/sycl/cpu_work_census.py', PROFILE)
CHECK_ORDER = [(0, cores, mode) for cores in (1, 6) for mode in ('pool', 'scan')]
ORDER = [(round_id, cores, mode)
         for round_id, counts in enumerate((range(1, 7), range(6, 0, -1), (3, 6, 1, 5, 2, 4)))
         for cores in counts for mode in (('scan', 'pool') if round_id == 1 else ('pool', 'scan'))]
LOCALITY_ARMS = {f'{layout}-{batch}': (layout, batch) for batch in (4, 5) for layout in ('hot', 'rotate')}
LOCALITY_PROTOCOL = (
    'Keep the first eight ordered CPU jobs from each of the two held-out width-two layer batches, '
    'their production packed weights, seeded activations and five pinned workers with a reserved caller. '
    'Allocate four identical copies in each process. Compare reusing one copy with reading each copy in '
    'turn across four otherwise identical submissions. Validate actual work, exact serial outputs, guards, '
    'and observed worker rows and calls outside the timed processes. Three counterbalanced independent '
    'pairs per format, with the existing five scored batches per process and no CPU service observer. '
    'Report whole-pass and gate/up times separately and retain every pair, including mixed or flat results. '
    'This tests address reuse sensitivity of the fixed work. Dynamic claim timing and physical DRAM bytes '
    'are separate questions, and the component does not qualify a model speedup.'
)
LOCALITY_MODE = False
SCHEDULE_ARMS = ('split', 'by-expert')
SCHEDULE_SHAPES = ((0, 0), (18, 2), (18, 4), (21, 2), (21, 4))
SCHEDULE_MODE = False
PUBLICATION_MODE = False
IQ3S_SINGLE_MODE = False
IQ3S_SINGLE_ARMS = ('reference', 'iq3s')
SCHEDULE_PROTOCOL = (
    'Replay the same eight held-out width-two and width-four batches with six physical cores and the '
    'DFlash2 decode IQ3 pair selection. Use the same compiled expert arithmetic, inputs, weights and '
    'pool in both arms. Compare the current row-split Gate/Up and Down phases against one complete '
    'expert per claim. The latter keeps the intermediate on the worker and retires all claims before '
    'the caller proceeds. Prove identical row outputs, guards, total work and affinity. Measure whole '
    'passes, with five scored samples and three counterbalanced independent pairs. In each process, '
    'time the full replay first, then each captured Gate/Up format and verifier width separately. '
    'Require by-expert to be faster in all three paired medians for the proposed scope. A shape '
    'passing alone permits a scoped follow-up. A broad follow-up requires the full replay and every '
    'shape to pass. These are CPU scheduling screens.'
)
need, sha, git = recipe.need, owned.sha, recipe.git


def select_locality():
    global ROOT, BUILD, RESULT, PROTOCOL, CHECK_ORDER, ORDER, LOCALITY_MODE
    LOCALITY_MODE = True
    ROOT = B / 'cpu-fixed-expert-locality-v1'
    BUILD = ROOT / 'build'
    RESULT = recipe.RESULT.with_name('cpu-fixed-expert-locality-current.json')
    PROTOCOL = LOCALITY_PROTOCOL
    CHECK_ORDER = [(0, 5, mode) for mode in LOCALITY_ARMS]
    ORDER = [(round_id, 5, f'{layout}-{batch}')
             for round_id, batches in enumerate(((4, 5), (5, 4), (4, 5)))
             for batch in batches
             for layout in (('rotate', 'hot') if round_id == 1 else ('hot', 'rotate'))]


def select_schedule():
    global ROOT, BUILD, RESULT, PROTOCOL, CHECK_ORDER, ORDER, SCHEDULE_MODE
    SCHEDULE_MODE = True
    ROOT = B / 'cpu-fixed-expert-schedule-v1'
    BUILD = ROOT / 'build'
    RESULT = recipe.RESULT.with_name('cpu-fixed-expert-schedule-current.json')
    PROTOCOL = SCHEDULE_PROTOCOL
    CHECK_ORDER = [(0, 6, mode) for mode in SCHEDULE_ARMS]
    ORDER = [(round_id, 6, mode) for round_id, arms in
             enumerate((SCHEDULE_ARMS, SCHEDULE_ARMS[::-1], SCHEDULE_ARMS)) for mode in arms]


def select_publication():
    global PUBLICATION_MODE, ROOT, BUILD, RESULT, PROTOCOL, CHECK_ORDER, ORDER
    select_schedule()
    PUBLICATION_MODE = True
    ROOT = B / 'cpu-fixed-expert-publication-v1'
    BUILD = ROOT / 'build'
    RESULT = recipe.RESULT.with_name('cpu-fixed-expert-publication-current.json')
    PROTOCOL = (
        'On the four captured width-four layers, reserve the caller and run five pinned workers. '
        'Check a one-epoch by-expert schedule with the actual first-pair dependency prefix, '
        'zero dependencies, and the full batch. Compare callback-after-prefix and callback-after-all. '
        'Prove the caller sees the completed prefix exactly, every job returns exactly the serial bits '
        'at retirement, the callback fires once per epoch, and the original guards and placement hold. '
        'No GPU execution or timing. Leave the complete original replay descriptor unchanged.'
    )
    CHECK_ORDER = [(0, 5, 'publication')]
    ORDER = []


def select_iq3s_single():
    global IQ3S_SINGLE_MODE, ROOT, BUILD, RESULT, PROTOCOL, CHECK_ORDER, ORDER
    IQ3S_SINGLE_MODE = True
    ROOT = B / 'cpu-fixed-expert-iq3s-single-v1'
    BUILD = ROOT / 'build'
    RESULT = recipe.RESULT.with_name('cpu-fixed-expert-iq3s-single-current.json')
    PROTOCOL = (
        'Replay all eight original captured batches with five pinned workers. Use the production row-split '
        'schedule in both arms and change only the decode IQ3_S single-token Gate/Up entry. Keep stock IQ3_XXS '
        'pairing, all multi-token routes, and the original packed weights. Check the candidate predicate with '
        'a separate dispatch audit in each arm. Require identical guarded output digests across all runs. '
        'Measure each arm without counters in three counterbalanced process pairs and five eligible batches '
        'per process. Advance only if all pairs improve both Gate/Up and whole-pool time.'
    )
    CHECK_ORDER = [(0, 5, arm) for arm in IQ3S_SINGLE_ARMS]
    ORDER = [(index, 5, arm) for index, pair in enumerate(
        (IQ3S_SINGLE_ARMS, IQ3S_SINGLE_ARMS[::-1], IQ3S_SINGLE_ARMS)) for arm in pair]


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
    profile = P / PROFILE
    original_profile = git(P, 'show', capture['source_commit'] + ':' + PROFILE)
    need(sha(profile) == hashlib.sha256(original_profile).hexdigest() == assets['profile_sha256'],
         'tracked profile differs from the original capture')
    need(Path(assets['manifest']) == calibration.METADATA and sha(calibration.METADATA) == assets['manifest_sha256'],
         'original native format manifest changed')
    membership = census.expected_membership(profile)
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


def prepare():
    need(not ROOT.exists() and not RESULT.exists(), 'one fresh replay plan')
    revision = git(P, 'rev-parse', 'HEAD').decode().strip()
    need(git(recipe.GGML, 'rev-parse', 'HEAD').decode().strip() == recipe.GGML_REV, 'pinned GGML source')
    project = recipe.pins(P, revision, PROJECT)
    ggml = recipe.pins(recipe.GGML, recipe.GGML_REV, ('ggml',))
    fixture = fixture_from_capture()
    weight_pin = weights(fixture)
    need(sha(recipe.SDK) == recipe.SDK_SHA, 'pinned SDK manifest changed')
    allowed = json.loads(recipe.SDK.read_text())
    need(len(allowed) == 10945 and all(sha(p) == h for p, h in allowed.items()), 'SDK input drift')
    env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
               LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib', OMP_NUM_THREADS='1',
               STRATA_IQ_SINGLE_EXACT='0', STRATA_IQ3_PAIR='0', STRATA_CPU_SERVICE='0',
               TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    if SCHEDULE_MODE or IQ3S_SINGLE_MODE:
        env['STRATA_IQ3_PAIR'] = '1'
    allowed.update(recipe.host_inventory(env)['files'])
    tools = {str(Path(p).resolve()): sha(p) for p in
             (recipe.CC, recipe.CXX, '/usr/bin/cmake', '/usr/bin/ninja', '/usr/bin/ldd', sys.executable)}
    allowed.update(tools)
    for directory in Path('/usr/share').glob('cmake-*/Modules'):
        allowed.update({str(p.resolve()): sha(p) for p in directory.rglob('*') if p.is_file()})
    ROOT.mkdir(mode=0o700)
    for name in ('cache', 'tmp'):
        (ROOT / name).mkdir()
    external = ROOT / 'external-inputs.json'
    atomic_json(external, allowed)
    target = ROOT / 'replay.txt'
    target.write_text(native_descriptor(fixture))
    header = '#pragma once\n\n#define GGML_VERSION "0.24.0"\n#define GGML_COMMIT  "3cf03257f"\n'
    row = dict(stage='prepared', source_commit=revision, project_inputs=project, ggml_inputs=ggml,
               machine=calibration.machine(), tools=tools, environment=env, fixture=fixture, weights=weight_pin,
               metadata_sha256=sha(calibration.METADATA), ggml_revision=recipe.GGML_REV, protocol=PROTOCOL,
               replay_input_sha256=sha(target), order=[list(c) for c in ORDER],
               check_order=[list(c) for c in CHECK_ORDER],
               external_inputs=dict(path=str(external), sha256=sha(external), files=len(allowed)),
               generated_expected={str(BUILD / 'ggml/src/ggml-version.h'): hashlib.sha256(header.encode()).hexdigest()},
               configure=['/usr/bin/cmake', '-S', str(P / 'bench/sycl/cpu_roofline'), '-B', str(BUILD),
                          '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                          '-DCMAKE_C_COMPILER=' + recipe.CC, '-DCMAKE_CXX_COMPILER=' + recipe.CXX,
                          '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t', '-DGGML_CCACHE=OFF',
                          '-DSTRATA_GGML_DIR=' + str(recipe.GGML)],
               build=['/usr/bin/cmake', '--build', str(BUILD), '--target', 'cpu_roofline', '--parallel', '4'],
               budgets=dict(build_wall=300, run_wall=60 if SCHEDULE_MODE else 45, build_rss=8 << 30, run_rss=2 << 30,
                            text=4 << 20, file=64 << 20, total=384 << 20),
               processes=[], active=False, ownership_closed=True, gpu_executed=False)
    if LOCALITY_MODE:
        row['observer_environment'] = dict(env, STRATA_CPU_SERVICE='1')
    if SCHEDULE_MODE or IQ3S_SINGLE_MODE:
        row['configure'].append('-DSTRATA_CPU_EXPERT_SCHEDULE_TEST=ON')
    if SCHEDULE_MODE:
        row['schedule_shapes'] = [list(shape) for shape in scheduling_shapes(fixture)]
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
    if LOCALITY_MODE:
        need(row['environment']['STRATA_CPU_SERVICE'] == '0'
             and row['observer_environment'] == dict(row['environment'], STRATA_CPU_SERVICE='1'),
             'separate clean and observed worker envelopes required')
    if SCHEDULE_MODE:
        need(row['environment']['STRATA_IQ3_PAIR'] == '1'
             and row['environment']['STRATA_IQ_SINGLE_EXACT'] == row['environment']['STRATA_CPU_SERVICE'] == '0'
             and row['configure'][-1] == '-DSTRATA_CPU_EXPERT_SCHEDULE_TEST=ON'
             and row['schedule_shapes'] == [list(shape) for shape in scheduling_shapes(row['fixture'])]
             and row['budgets']['run_wall'] == 60, 'fixed decode scheduling inputs')
    if IQ3S_SINGLE_MODE:
        need(row['environment']['STRATA_IQ3_PAIR'] == '1'
             and row['environment']['STRATA_IQ_SINGLE_EXACT'] == row['environment']['STRATA_CPU_SERVICE'] == '0'
             and row['configure'][-1] == '-DSTRATA_CPU_EXPERT_SCHEDULE_TEST=ON', 'fixed IQ3_S single replay inputs')
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


def locality_summary(fixture, mode, by_kind, timed):
    layout, batch = LOCALITY_ARMS[mode]
    source = json.loads(fixture['blocks'])[batch]
    gu, down, blob_bytes = source['fmt']
    jobs = source['jobs'][:8]
    entries = sum(len(rows) for _, rows in jobs)
    expected = dict(kind='locality_input', layout=layout, batch=batch, gu=gu, down=down, jobs=8,
                    entries=entries, logical_bytes=32 * blob_bytes, allocated_bytes=32 * blob_bytes,
                    address_bytes=(32 if layout == 'rotate' else 8) * blob_bytes, serial_output_equal=True)
    need(source['window'] == 4801 and source['width'] == 2 and len(jobs) == 8
         and by_kind['locality_input'] == [expected] and not by_kind['replay_input'],
         'locality decoded work or address footprint changed')
    result = dict(locality=expected)
    if timed:
        need(not by_kind['locality_workers'], 'profile observer active in a clean timing arm')
    else:
        profile, = by_kind['locality_workers']
        counts = profile['slots']
        need(profile['passes'] == 32 and {v['slot'] for v in counts} == {0, 1, 2, 3, 4, 64}
             and len(counts) == 6 and all(v['calls'] > 0 and 0 < v['rows'] <= v['token_rows'] <= 2 * v['rows']
                                         and 0 <= v['cpu_ns'] <= v['wall_ns'] + 1_000_000 for v in counts),
             'missing or inconsistent observed CPU workers')
        host, = (v for v in counts if v['slot'] == 64)
        need(host['calls'] == 32 * 4 * 8, 'the caller may only quantize the hidden rows')
        result['worker_census'] = profile
    return result


def scheduling_shapes(fixture):
    blocks = json.loads(fixture['blocks'])
    shapes = []
    for gu, width in SCHEDULE_SHAPES[1:]:
        matching = [b for b in blocks if (b['fmt'][0], b['width']) == (gu, width)]
        need(len(matching) == 2, 'two captured batches required per scheduling shape')
        jobs = sum(len(b['jobs']) for b in matching)
        byte_count = sum(b['fmt'][2] * len(b['jobs']) for b in matching)
        shapes.append((gu, width, byte_count, jobs))
    need(sum(s[2] for s in shapes) == fixture['logical_bytes']
         and sum(s[3] for s in shapes) == fixture['jobs'], 'shape work must cover the full schedule')
    return [(0, 0, fixture['logical_bytes'], fixture['jobs']), *shapes]


def summaries(row, text, mode, cores, timed):
    rows = [json.loads(line) for line in text.splitlines()]
    by_kind = {kind: [v for v in rows if v['kind'] == kind]
               for kind in ('replay_placement', 'replay_input', 'timer_check', 'sample', 'closed',
                            'locality_input', 'locality_workers', 'iq3s_single')}
    need(sum(map(len, by_kind.values())) == len(rows) and
         by_kind['closed'] == [dict(kind='closed', passed=True)] and rows[-1] == by_kind['closed'][0],
         'complete CPU replay output required')
    fixture = row['fixture']
    host, *available = row['machine']['physical_cpus']
    workers = available[:min(cores, 5)]
    placement, = by_kind['replay_placement']
    need(placement['host_cpu'] == host and placement['workers'] == workers
         and placement['caller_computes'] == (cores == 6), 'replay caller and worker placement')
    result = dict(mode=mode, cores=cores, placement=placement)
    if IQ3S_SINGLE_MODE:
        proof, = by_kind['iq3s_single']
        nt1 = sum(len(tokens) == 1 for batch in json.loads(fixture['blocks']) if batch['fmt'][0] == 21
                  for _, tokens in batch['jobs'])
        # The check runs the pool and the serial comparison once. Timed
        # processes keep dispatch counters disabled for every replay.
        need(mode in IQ3S_SINGLE_ARMS and cores == 5 and nt1 == 81 and proof['audit'] is (not timed)
             and type(proof['output_fnv64']) is int and 0 <= proof['output_fnv64'] < 1 << 64
             and proof['single_rows'] == (2 * nt1 * 640 if not timed and mode == 'iq3s' else 0),
             'actual IQ3_S single-token dispatch or output proof differs')
        result.update(arm=mode, output_fnv64=proof['output_fnv64'], audit_rows=proof['single_rows'])
        mode = 'pool'
    else:
        need(not by_kind['iq3s_single'], 'unexpected IQ3_S single report')
    schedule = mode in SCHEDULE_ARMS
    if schedule:
        need(cores == 6 and all(v.get('schedule') == mode for v in
             by_kind['replay_input'] + by_kind['timer_check'] + by_kind['sample']),
             'wrong decode expert schedule')
    if mode in LOCALITY_ARMS:
        need(cores == 5, 'locality replay requires five pinned workers')
        result.update(locality_summary(fixture, mode, by_kind, timed))
    else:
        need(not by_kind['locality_input'] and not by_kind['locality_workers'], 'unexpected locality records')
    if mode == 'pool' or schedule:
        actual, = by_kind['replay_input']
        need(all(actual[k] == fixture[k] for k in ('jobs', 'entries', 'logical_bytes', 'unique_blobs', 'unique_bytes'))
             and actual['batches'] == 8 and actual['threads'] == cores and actual['host_works'] == (cores == 6)
             and actual['serial_output_equal'] and actual['verified_affinity'], 'pool replay conservation or numerics')
    else:
        need(not by_kind['replay_input'], 'unexpected expert work in the read control')
    samples = by_kind['sample']
    timings = by_kind['timer_check']
    if not timed:
        need(not timings and not samples, 'unexpected timed check')
        return result
    if mode in LOCALITY_ARMS:
        byte_count, visits, label = result['locality']['logical_bytes'], 32, 'locality'
    elif mode == 'pool' or schedule:
        byte_count, visits, label = fixture['logical_bytes'], fixture['jobs'], 'replay'
    else:
        byte_count, visits, label = fixture['unique_bytes'], 1, 'scan'
    cases = row['schedule_shapes'] if schedule else [(0, 0, byte_count, visits)]
    need(len(timings) == len(cases) and len(samples) == 5 * len(cases), 'scored replay sample count')
    cells = []
    for i, (gu, width, byte_count, visits) in enumerate(cases):
        timer = timings[i]
        cell_samples = samples[5*i:5*(i+1)]
        need(0 < timer['query_s'] < timer['fastest_pass_s'] * .001, 'timer overhead admission')
        if schedule:
            need(all((v.get('gu'), v.get('width'), v.get('mode'), v.get('threads')) == (gu, width, label, cores)
                     for v in (timer, *cell_samples)), 'wrong measured scheduling shape')
        if label == 'locality':
            need(timer['mode'] == label and all(v['layout'] == result['locality']['layout']
                 for v in (timer, *cell_samples)), 'wrong locality timing arm')
        for rep, sample in enumerate(cell_samples):
            repeats = sample['iterations']
            need(sample['mode'] == label and sample['threads'] == cores
                 and sample['rep'] == rep and sample['attempt'] == 0 and 0 < repeats < 1000000
                 and sample['requested_bytes'] == repeats * byte_count and sample['visits'] == repeats * visits
                 and math.isfinite(sample['wall_s']) and sample['wall_s'] >= 1
                 and math.isfinite(sample['cpu_s'])
                 and 0 <= sample['host_cpu_s'] <= sample['cpu_s'] + .005
                 and sample['host_cpu_s'] <= sample['wall_s'] + .005, 'replay timing or work admission')
            if label != 'scan':
                phases = [sample[k] for k in ('gu_s', 'q_s', 'down_s')]
                need(all(math.isfinite(v) and v >= 0 for v in phases) and sum(phases) <= sample['wall_s'] + .001,
                     'bounded disjoint pool stage times')
                if mode == 'by-expert':
                    need(phases == [0, 0, 0], 'by-expert work has no disjoint stage clocks')
        cells.append(dict(gu=gu, width=width, median_s_per_pass=statistics.median(
            v['wall_s'] / v['iterations'] for v in cell_samples)))
    if schedule:
        result.update(schedule_cells=cells, shape_timers=timings[1:])
    return dict(result, samples=samples, timer=timings[0], median_s_per_pass=cells[0]['median_s_per_pass'])


def publication_summary(row, text):
    records = [json.loads(line) for line in text.splitlines()]
    host, *available = row['machine']['physical_cpus']
    placement = dict(kind='replay_placement', host_cpu=host, workers=available[:5], caller_computes=False)
    expected = []
    for batch in json.loads(row['fixture']['blocks']):
        if batch['width'] != 4:
            continue
        jobs = batch['jobs']
        expected.append(dict(kind='publication_input', jobs=len(jobs), gu=batch['fmt'][0],
            prefix_jobs=sum(rows[0] < 2 for _, rows in jobs), logical_bytes=batch['fmt'][2] * len(jobs),
            cases=8, bitwise=True))
    need(len(expected) == 4 and sum(v['jobs'] for v in expected) == 137
         and sum(v['logical_bytes'] for v in expected) == 308249600
         and records == [placement, *expected, dict(kind='closed', passed=True)],
         'CPU publication placement, captured work or readiness proof changed')
    return dict(placement=placement, slices=expected, gpu_executed=False)


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

    def run(label, argv, *, build=False, observe=False, environment=None):
        env = environment if environment is not None else (row['observer_environment'] if observe else row['environment'])
        entry, out, err = owner.run(label, argv, env, ROOT,
            wall=row['budgets']['build_wall' if build else 'run_wall'],
            rss_cap=row['budgets']['build_rss' if build else 'run_rss'],
            cpu=1200 if build else 180, text_cap=row['budgets']['text'],
            file_cap=row['budgets']['file'], total_cap=row['budgets']['total'])
        need(owned.closed(entry) and entry['exit_code'] == 0,
             label + ' failed: ' + err.read_text(errors='replace')[-1500:])
        return entry, out, err

    try:
        if stage == 'build':
            allowed = recipe.input_map(row)
            run('configure', row['configure'], build=True)
            run('compile', row['build'], build=True)
            run('deps', ['/usr/bin/ninja', '-C', str(BUILD), '-t', 'deps'])
            row['build_evidence'] = recipe.build_check(row, scope, build_root=BUILD, required_units=REQUIRED)
            binary = BUILD / 'cpu_roofline'
            row['binary'] = dict(path=str(binary), bytes=binary.stat().st_size, sha256=sha(binary))
            _, out, _ = run('runtime', ['/usr/bin/ldd', str(binary)])
            info = out.read_text()
            row['runtime_inputs'] = {str(Path(p).resolve()): sha(Path(p).resolve())
                                     for p in re.findall(r'(/[^\s]+)', info)}
            need('not found' not in info and row['runtime_inputs'] and
                 not re.search(r'sycl|libze_|libur_|cuda|amdhip', info, re.I)
                 and all(allowed.get(p) == h for p, h in row['runtime_inputs'].items()), 'unpinned CPU runtime')
            row['stage'] = 'built'
        else:
            verify_binary(row)
            need(weights(row['fixture']) == row['weights'], 'original model weight slices changed')
            cases = CHECK_ORDER if stage == 'check' else ORDER
            for i, (round_id, cores, mode) in enumerate(cases):
                verify_binary(row)
                environment = None
                if PUBLICATION_MODE:
                    args = ['replay-publication-check', str(ROOT / 'replay.txt')]
                elif IQ3S_SINGLE_MODE:
                    args = ['replay-iq3s-single', str(ROOT / 'replay.txt'), stage]
                    environment = dict(row['environment'], STRATA_IQ_SINGLE_EXACT='iq3s' if mode == 'iq3s' else '0',
                                       STRATA_IQ3_PAIR_AUDIT='1' if stage == 'check' else '0')
                elif mode in LOCALITY_ARMS:
                    layout, batch = LOCALITY_ARMS[mode]
                    args = ['replay-locality', str(ROOT / 'replay.txt'), str(batch), layout, stage]
                elif mode in SCHEDULE_ARMS:
                    args = ['replay-schedule', str(ROOT / 'replay.txt'), mode, stage]
                elif mode == 'pool':
                    args = ['replay', str(ROOT / 'replay.txt'), str(cores), stage]
                else:
                    args = ['replay-scan', str(cores), str(row['fixture']['unique_bytes']), stage]
                entry, out, err = run(f'case-{i}', [row['binary']['path'], *args],
                                      observe=LOCALITY_MODE and stage == 'check', environment=environment)
                need(err.stat().st_size == 0, 'replay emitted stderr')
                data = (publication_summary(row, out.read_text()) if PUBLICATION_MODE
                        else summaries(row, out.read_text(), mode, cores, stage == 'measure'))
                if environment is not None:
                    data['environment_sha256'] = recipe.fingerprint(environment)
                row['processes'].append(dict(round=round_id, **data, owner=recipe.receipt(entry)))
            if IQ3S_SINGLE_MODE:
                need(len({p['output_fnv64'] for p in row['processes']}) == 1, 'IQ3_S arms differ in replay output')
            need(weights(row['fixture']) == row['weights'], 'expert weight slices changed during replay')
            if stage == 'check':
                if PUBLICATION_MODE:
                    row.update(stage='closed', complete=True, passed=True, publication_cpu_correctness_qualified=True)
                else:
                    row['stage'] = 'checked'
            else:
                measured = row['processes'][len(CHECK_ORDER):]
                need(len({(x['owner']['pid'], x['owner']['direct_identity']['start_ticks'])
                          for x in measured}) == len(ORDER), 'independent replay process count')
                if LOCALITY_MODE:
                    row['locality_pairs'] = locality_pairs(measured)
                elif SCHEDULE_MODE:
                    row['schedule_result'] = schedule_result(measured)
                elif IQ3S_SINGLE_MODE:
                    row['iq3s_single_result'] = iq3s_single_result(measured)
                else:
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
    summary = next((row[k] for k in ('locality_pairs', 'schedule_result', 'iq3s_single_result', 'medians_s_per_pass')
                    if k in row), {})
    print(row['stage'], summary, flush=True)


def locality_pairs(measured):
    def costs(case):
        return (case['median_s_per_pass'], statistics.median(s['gu_s'] / s['iterations'] for s in case['samples']))

    result = []
    for a, b in zip(measured[::2], measured[1::2], strict=True):
        hot, rotate = (a, b) if a['locality']['layout'] == 'hot' else (b, a)
        need(hot['round'] == rotate['round'] and hot['locality']['batch'] == rotate['locality']['batch']
             and hot['locality']['layout'] == 'hot' and rotate['locality']['layout'] == 'rotate',
             'unpaired locality timing arms')
        full_hot, gu_hot = costs(hot)
        full_rotate, gu_rotate = costs(rotate)
        need(min(full_hot, gu_hot, full_rotate, gu_rotate) > 0, 'missing positive locality clocks')
        result.append(dict(round=hot['round'], batch=hot['locality']['batch'],
                           full_rotate_over_hot=full_rotate/full_hot, gu_rotate_over_hot=gu_rotate/gu_hot))
    return result


def schedule_result(measured):
    ratios = {shape: [] for shape in SCHEDULE_SHAPES}
    for a, b in zip(measured[::2], measured[1::2], strict=True):
        arms = {p['mode']: p['schedule_cells'] for p in (a, b)}
        need(a['round'] == b['round'] and set(arms) == set(SCHEDULE_ARMS), 'unpaired decode expert schedule')
        for shape, control, candidate in zip(SCHEDULE_SHAPES, arms['split'], arms['by-expert'], strict=True):
            need(all((cell['gu'], cell['width']) == shape and cell['median_s_per_pass'] > 0
                     for cell in (control, candidate)), 'unpaired decode scheduling shape')
            ratios[shape].append(candidate['median_s_per_pass'] / control['median_s_per_pass'])
    need(all(len(values) == 3 for values in ratios.values()), 'expected three decode scheduling pairs')
    cells = [dict(gu=gu, width=width, by_expert_over_split=values, eligible=all(v < 1.0 for v in values))
             for (gu, width), values in ratios.items()]
    return dict(by_expert_over_split=cells[0]['by_expert_over_split'],
                eligible=all(cell['eligible'] for cell in cells), shapes=cells[1:])


def iq3s_single_result(measured):
    pairs = []
    for a, b in zip(measured[::2], measured[1::2], strict=True):
        arms = {p['arm']: p for p in (a, b)}
        need(a['round'] == b['round'] and set(arms) == set(IQ3S_SINGLE_ARMS), 'unpaired IQ3_S single timing')
        control, candidate = (arms[k] for k in IQ3S_SINGLE_ARMS)
        gu = [statistics.median(s['gu_s'] / s['iterations'] for s in p['samples']) for p in (control, candidate)]
        need(min(gu) > 0, 'missing IQ3_S Gate/Up timing')
        pairs.append(dict(gu_ratio=gu[1] / gu[0],
                          pool_ratio=candidate['median_s_per_pass'] / control['median_s_per_pass']))
    need(len(pairs) == 3, 'three independent IQ3_S single pairs required')
    return dict(pairs=pairs, eligible=all(max(p.values()) < 1 for p in pairs), model_qualified=False)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--locality']:
        sys.argv.pop(1)
        select_locality()
    elif sys.argv[1:2] == ['--schedule']:
        sys.argv.pop(1)
        select_schedule()
    elif sys.argv[1:2] == ['--publication']:
        sys.argv.pop(1)
        select_publication()
    elif sys.argv[1:2] == ['--iq3s-single']:
        sys.argv.pop(1)
        select_iq3s_single()
    stages = ('prepare', 'build', 'check') if PUBLICATION_MODE else ('prepare', 'build', 'check', 'measure')
    need(sys.executable == '/home/yayoi/.local/share/strata-sycl/venv/bin/python3' and len(sys.argv) == 2
         and sys.argv[1] in stages, 'pinned Python and one declared CPU replay stage required')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            execute(sys.argv[1])
