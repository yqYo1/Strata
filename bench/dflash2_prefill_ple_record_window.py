"""Replay the exact prose workload on legacy, one-chunk and two-chunk PLE."""

import fcntl
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_cache_capacity as capacity
import dflash2_prefill_ple_phase_cache as phase

corpus, bank, wait, geometry = capacity.corpus, capacity.bank, capacity.wait, capacity.geometry
early, attribution = geometry.early, geometry.attribution
validate, require = capacity.validate, capacity.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_record_window_v1'
BUILD_KEY = 'dflash2_prefill_ple_record_window_native_v1'
BUILD_ROOT = bank.B / BUILD_KEY.replace('_', '-')
TREE = BUILD_ROOT / 'source'
HOST_KEY = 'dflash2_prefill_ple_record_window_host_v1'
HOST_ROOT = BUILD_ROOT / 'host-fixture'
SMOKE_KEY = 'dflash2_prefill_ple_record_window_check_v1'
FULL_SMOKE_KEY = 'dflash2_prefill_ple_record_horizon_check_v1'
SMOKE_CONTROLLER = 'bench/dflash2_prefill_ple_record_window_check.py'
PREIMAGE = '9cba62818dcc4f0455891f67675742f3e618bc55'
ORDER = ('legacy', 'one', 'two', 'two', 'one', 'legacy')
WINDOW = dict(legacy=0, one=1, two=2, full='all')
FULL_KEY = 'dflash2_prefill_ple_record_horizon_v1'
FULL_ORDER = ('two', 'full', 'full', 'two')
SELECTOR = 'STRATA_SYCL_PREFILL_PLE_RECORD_WINDOW'
MARKER = 'strata prefill PLE record window: '
PRODUCER = 'strata prefill PLE record producer: '
CONSUMER = 'strata prefill PLE record consumer: '
CPP = ('src/platform/direct_file.cpp', 'src/ngram/ple_reader.cpp',
       'src/kernels/ngram.cpp', 'src/prefill/prefill.cpp')
EXISTING = (*CPP, 'include/strata/platform/direct_file.hpp', 'include/strata/ngram/ple_reader.hpp',
            'include/strata/kernels/ngram.hpp', 'src/prefill/layer_major.inc')
NEW = ('include/strata/ngram/prefill_record_plan.hpp', 'include/strata/ngram/prefill_record_reader.hpp',
       'tests/core/prefill_record_plan_test.cpp', 'tests/core/prefill_record_reader_test.cpp')
PLANNER, READER = NEW[2:]
HOST_CPP = (PLANNER, READER, 'src/ngram/ple_reader.cpp', 'src/platform/direct_file.cpp')
HOST_INPUTS = (*HOST_CPP, *NEW[:2], 'include/strata/ngram/ple_reader.hpp', 'include/strata/platform/direct_file.hpp')
HOST_SUCCESS = dict(planner='prefill PLE record planner checks passed',
                    reader='prefill PLE DirectFile read window checks passed')


def host_recipe(cfg, toolchain, source_hashes):
    compiler, = (arg.split('=', 1)[1] for arg in cfg if arg.startswith('-DCMAKE_CXX_COMPILER='))
    compiler_sha = toolchain[compiler]
    require(bank.sha(compiler) == compiler_sha, 'the host fixture compiler changed')
    common = [compiler, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-pthread',
              '-I', str(TREE / 'include')]
    cases = {}
    for role, inputs in (('planner', (PLANNER,)), ('reader', (READER, *HOST_CPP[2:]))):
        exe = str(HOST_ROOT / role)
        argv = [exe, str(HOST_ROOT / 'cases')] if role == 'reader' else [exe]
        cases[role] = dict(compile=[*common, *(str(TREE / path) for path in inputs), '-o', exe], command=argv)
    return dict(compiler=compiler, compiler_sha256=compiler_sha, cases=cases,
                source_sha256={path: source_hashes[path] for path in HOST_INPUTS})


def build_inputs(revision):
    fields, payload, original, archive = phase.build_inputs(PREIMAGE)
    archived = bank.previous.read(phase.BUILD_ROOT / 'record.json')
    identity = early.builder.compact(archived)
    require(wait.retained.closed(archived, 'built')
            and identity == validate.committed(bank.RESULT, (phase.BUILD_KEY,))
            and all(archived[k] == fields[k] for k in ('source_files', 'recipe', 'native_source_commit')),
            'recover the exact dedicated-cache source and recipe')
    fields.pop('host_fixture_recipe')
    make_blobs = early.source.blobs
    before = dict(make_blobs(bank.P, PREIMAGE, EXISTING))
    overlay = dict(make_blobs(bank.P, revision, (*EXISTING, *NEW)))
    require(before == {name: payload[name] for name in EXISTING}
            and all(overlay[name] != raw for name, raw in before.items())
            and not set(NEW) & payload.keys(), 'record window source preimages differ')
    payload.update(overlay)
    control = early.source.pin(bank.P, (THIS, phase.THIS, corpus.THIS, capacity.THIS, SMOKE_CONTROLLER))['source_sha256']
    require(bank.stage.hashed(dict(make_blobs(bank.P, revision, control))) == control,
            'commit the record window controllers first')
    old, new = str(phase.BUILD_ROOT), str(BUILD_ROOT)
    recipe = fields['recipe']
    cfg = [arg.replace(old, new) for arg in recipe['configure']]
    require(cfg[2:5] == [str(TREE), '-B', str(BUILD_ROOT / 'build')], 'relocate only build paths')
    source_hashes = bank.stage.hashed(payload)
    fields.update(source_commit=revision, native_source_commit=revision, overlay_commit=revision,
                  overlays=bank.stage.hashed(overlay), source_files=source_hashes,
                  host_fixture_recipe=host_recipe(cfg, original['toolchain'], source_hashes),
                  controller_sources=fields['controller_sources'] | control,
                  phase_build_sha256=bank.signature(identity),
                  recipe=dict(configure=cfg, targets=recipe['targets'],
                              environment={k: v.replace(old, new) for k, v in recipe['environment'].items()}))
    return fields, payload, original, archive


def build_context():
    ctx = bank.basis.core.source.device.load_module('ple_record_window_build', early.base_build.__file__)
    ctx.KEY, ctx.ROOT, ctx.TREE, ctx.OVERLAYS = BUILD_KEY, BUILD_ROOT, TREE, CPP
    ctx.inputs = build_inputs

    def prepare():
        require(not BUILD_ROOT.exists() and BUILD_KEY not in validate.committed(bank.RESULT),
                'inspect the existing window build')
        revision = early.source.git(bank.P, 'rev-parse', 'HEAD').decode().strip()
        fields, payload, _, _ = build_inputs(revision)
        BUILD_ROOT.mkdir(mode=0o700)
        for subdir in ('owned', 'tmp', 'cache'):
            (BUILD_ROOT / subdir).mkdir(mode=0o700)
        for name, raw in payload.items():
            path = TREE / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        row = dict(fields, stage='prepared', commands=[], active=False, passed=False, complete=False,
                   ownership_closed=True, scope='Rebuild the measured reader with a selectable prefill record window.',
                   retention='Review at the first failed check or the close of the window attribution comparison.')
        ctx.check(row, committed=False)
        ctx.save(row)

    ctx.prepare = prepare
    return ctx


def host_fixture():
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder = early.builder
        builder.guards.no_model_owners([])
        built = bank.previous.read(BUILD_ROOT / 'record.json')
        require(wait.retained.closed(built, 'built') and not HOST_ROOT.exists()
                and HOST_KEY not in validate.committed(bank.RESULT), 'require a fresh same-source host check')
        build_context().check(built)
        recipe = built['host_fixture_recipe']
        source = recipe['source_sha256']
        require(all(bank.sha(TREE / path) == digest for path, digest in source.items())
                and bank.sha(recipe['compiler']) == recipe['compiler_sha256'], 'host fixture inputs changed')
        HOST_ROOT.mkdir(mode=0o700)
        (HOST_ROOT / 'cases').mkdir(mode=0o700)
        env = dict(built['recipe']['environment'], STRATA_IO_THREADS='2')
        receipt = dict(stage='running', active=True, complete=False, passed=False, ownership_closed=False,
                       native_build_sha256=bank.signature(builder.compact(built)), recipe=recipe,
                       environment_sha256=bank.signature(env), commands=[], executables={})
        bank.stage.write(HOST_ROOT / 'record.json', receipt)
        owner = builder.ownership.Owner(HOST_ROOT / 'owned')
        limits = dict(wall=90, cpu=60, text_cap=128 << 10, file_cap=8 << 20,
                      total_cap=24 << 20, rss_cap=1 << 30)

        def execute(label, argv):
            require(all(bank.sha(TREE / path) == digest for path, digest in source.items())
                    and bank.sha(recipe['compiler']) == recipe['compiler_sha256'], 'host fixture inputs changed')
            entry, stdout, _ = owner.run(label, argv, env, TREE, **limits)
            closed = bool(builder.ownership.closed(entry))
            receipt['commands'].append(dict(label=label, exit_code=entry.get('exit_code'), ownership_closed=closed))
            require(closed and entry['exit_code'] == 0, 'host command failed: ' + label)
            return stdout

        try:
            for role, case in recipe['cases'].items():
                execute(role + '-compile', case['compile'])
                binary = Path(case['command'][0])
                image = dict(path=str(binary), bytes=binary.stat().st_size, sha256=bank.sha(binary))
                checked = execute(role + '-test', case['command'])
                require(checked.read_text().splitlines() == [HOST_SUCCESS[role]]
                        and bank.sha(binary) == image['sha256'], 'host verification failed: ' + role)
                receipt['executables'][role] = image
            receipt.update(stage='closed', complete=True, passed=True)
        except Exception as error:
            receipt.update(stage='failed', error=str(error))
        finally:
            receipt.update(active=owner.active is not None,
                           ownership_closed=owner.active is None and all(
                               builder.ownership.closed(command) for command in owner.commands))
            bank.stage.write(HOST_ROOT / 'record.json', receipt)
            results = bank.previous.read(bank.RESULT)
            results[HOST_KEY] = receipt
            bank.stage.write(bank.RESULT, results)
        require(receipt['passed'] and receipt['ownership_closed'], 'host record-window checks stopped')
        print('Host planner and DirectFile checks passed. Commit the receipt before timing.', flush=True)


def checked_host(built):
    receipt = bank.previous.read(HOST_ROOT / 'record.json')
    require(wait.retained.closed(receipt, 'closed')
            and receipt == validate.committed(bank.RESULT, (HOST_KEY,))
            and receipt['native_build_sha256'] == bank.signature(early.builder.compact(built))
            and receipt['recipe'] == built['host_fixture_recipe']
            and receipt['executables'].keys() == HOST_SUCCESS.keys()
            and len(receipt['commands']) == 2 * len(HOST_SUCCESS)
            and all(entry['ownership_closed'] and entry['exit_code'] == 0 for entry in receipt['commands'])
            and all(Path(image['path']).stat().st_size == image['bytes'] and bank.sha(image['path']) == image['sha256']
                    for image in receipt['executables'].values()), 'require the committed same-build host checks')
    return bank.signature(receipt)


def timed_phase_reference(native):
    measured = bank.previous.read(phase.ROOT / 'record.json')
    sealed = validate.committed(bank.RESULT, (phase.KEY,))
    archived = validate.committed(bank.RESULT, (phase.BUILD_KEY,))
    require(wait.retained.closed(measured, 'closed') and wait.basis.compact(measured) == sealed
            and measured['decision'] == capacity.decision(measured['processes'])
            and measured['native_build_sha256'] == native['phase_build_sha256'] == bank.signature(archived)
            and measured['native_source_commit'] == archived['native_source_commit'] == PREIMAGE
            and measured['capacity_proof_sha256'] == native['capacity_proof_sha256']
            and measured['binary'] == archived['binaries']['strata'],
            'recover the closed timing on the exact phase-cache source and build')
    selected = wait.baseline()
    # Validate current model and runtime through the primary fallback. The
    # phase and early candidates supply evidence through their sealed receipts.
    physical = selected.pop('physical_context_sha256')
    for name in ('binary', 'native_source_commit', 'native_build_sha256', 'request_hashes', 'frozen_prefill_work'):
        selected.pop(name)
    require(all(measured[k] == v for k, v in selected.items())
            and measured['base_physical_context_sha256'] == physical
            and measured['storage_geometry'] == attribution.storage_geometry(measured)
            and capacity.FLAG not in selected['configuration']['args'],
            'the selected model, runtime, ordinary cache or storage geometry changed')
    controls = ('ple_check_sha256', 'clean_timing_sha256', 'clean_controls', 'storage_geometry',
                'attribution_sha256', 'early_extra_wait_ms')
    return dict(selected, **{key: measured[key] for key in controls}, native_context_qualified=False,
                base_physical_context_sha256=physical, phase_timing_sha256=bank.signature(sealed))


def baseline():
    native = bank.previous.read(BUILD_ROOT / 'record.json')
    require(wait.retained.closed(native, 'built'), 'finish the record window build')
    build_context().check(native)
    host_proof = checked_host(native)
    smoke = validate.committed(bank.RESULT, (SMOKE_KEY,))
    require(wait.retained.closed(smoke, 'closed') and smoke['plan']['binary'] == native['binaries']['strata']
            and smoke['plan']['native_build_sha256'] == bank.signature(early.builder.compact(native))
            and smoke['plan']['host_fixture_sha256'] == host_proof
            and smoke['outcomes']['cancel']['cancelled']['read'] == 0
            and smoke['outcomes']['cancel']['first_direct_read_submitted']
            and tuple(smoke['outcomes']) == ('cancel', 'one', 'direct')
            and all(smoke['outcomes'][name]['record_window_chunks'] == depth
                    and smoke['outcomes'][name]['consumed_chunks'] == 15
                    for name, depth in (('one', 1), ('direct', 2)))
            and all(p['passed'] and p['ownership_closed'] and not p['fresh_faults'] for p in smoke['outcomes'].values()),
            'finish native STOP, first-chunk use and direct-writer parity on this exact binary')
    capacity_proof = validate.committed(bank.RESULT, (capacity.KEY,))
    require(wait.retained.closed(capacity_proof, 'closed') and len(capacity_proof['processes']) == 4
            and bank.signature(capacity_proof) == native['capacity_proof_sha256'],
            'finish the same-binary capacity comparison first')
    source = validate.committed(bank.RESULT, (corpus.campaign_key(capacity.WORKLOAD),))
    require(wait.retained.closed(source, 'closed') and len(source['processes']) == 2
            and source['binary'] == capacity_proof['binary']
            and source['plan']['corpus_sha256'] == corpus.CORPORA[capacity.WORKLOAD][0],
            'recover the independent matched prose requests')
    hashes = [[s['request_sha256'] for s in run['samples']] for run in source['processes']]
    work = [[(r['request_work'], r['bank_upload_bytes']) for r in run['routes']]
            for run in source['processes']]
    require(hashes[0] == hashes[1] and work[0] == work[1] and len(hashes[0]) == wait.speed.RUNS,
            'the independent reference requests differ')
    selected = timed_phase_reference(native)
    return dict(selected, binary=native['binaries']['strata'], native_source_commit=native['native_source_commit'],
                native_build_sha256=bank.signature(early.builder.compact(native)),
                capacity_proof_sha256=bank.signature(capacity_proof), host_fixture_sha256=host_proof,
                native_smoke_sha256=bank.signature(smoke),
                request_hashes={str(wait.READ): hashes[0]}, constant_request_work=[list(r) for r in work[0]])


def fields(text, prefix, pattern):
    hits = [pattern.fullmatch(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]
    require(len(hits) == text.count(prefix) and all(hits), 'malformed record window observer: ' + prefix)
    return [m.groups() for m in hits]


PRODUCER_FIELDS = re.compile(
    r'rows=(\d+) chunks=(\d+) reads=(\d+) requested_bytes=(\d+) returned_bytes=(\d+) '
    r'first_ready_ms=(\d+\.\d{3}) complete_ms=(\d+\.\d{3})')
CONSUMER_FIELDS = re.compile(
    r'offset=(\d+) tokens=(\d+) host_slot=([01]) wait_ms=(\d+\.\d{3}) worker_expand_ms=(\d+\.\d{3})')


def route(text, plan):
    depth = WINDOW[plan['arm']]
    if depth == 0:
        require(not any(mark in text for mark in (MARKER, PRODUCER, CONSUMER)),
                'record window observed in the legacy arm')
        frame = geometry.route(text, dict(plan, arm='geometry'))
        return dict(corpus.compact_request(frame),
                    first_chunk_exposed_ms=frame['ple_host_timeline'][0]['join_ns'] / 1e6)

    removed = (MARKER, PRODUCER, CONSUMER, attribution.ple.PREFIX)
    clean = ''.join(line for line in text.splitlines(keepends=True) if not line.startswith(removed))
    require(not any(mark in clean for mark in (*geometry.OBSERVED_PREFIXES, geometry.timing.MARKER)),
            'legacy gathers observed in the record window arm')
    frame = wait.route(clean, dict(plan, arm='clean'))
    rows, chunk = plan['read'] - 1, int(frame['work']['operator_max'])
    chunks = [(i, min(chunk, rows - i)) for i in range(0, rows, chunk)]
    if depth == 'all':
        depth = len(chunks)
    require([line for line in text.splitlines() if line.startswith(MARKER)]
            == [f'{MARKER}tokens_per_chunk={chunk} window_chunks={depth}'],
            'actual record window dispatch differs')
    wait_data = attribution.ple.gather_accounting(text, rows, chunk)
    consumed = []
    for index, (expected, values) in enumerate(zip(chunks, fields(text, CONSUMER, CONSUMER_FIELDS), strict=True)):
        i, n, slot, *clocks = values
        require((int(i), int(n)) == expected and int(slot) == (index & 1),
                'record consumer extent or host slot differs')
        consumed.append(list(map(float, clocks)))
    require(sum(row[0] for row in consumed) <= wait_data['future_get_ms'] + 0.001 * len(chunks) + 0.001,
            'consumed timings exceed the enclosing PLE interval')
    produced = []
    reads = requested = returned = 0
    windows = [chunks[i:i+depth] for i in range(0, len(chunks), depth)]
    for expected, values in zip(windows, fields(text, PRODUCER, PRODUCER_FIELDS), strict=True):
        count, n, submitted, requested_bytes, returned_bytes = map(int, values[:5])
        first_ms, complete_ms = map(float, values[5:])
        require(n == len(expected) and count == geometry.ROWS_PER_TOKEN * sum(tokens for _, tokens in expected)
                and 0 <= submitted <= 2 * count and submitted * geometry.PAGE_BYTES <= requested_bytes <= submitted * geometry.RECORD_BYTES
                and requested_bytes % geometry.PAGE_BYTES == 0 and 0 <= returned_bytes <= requested_bytes
                and first_ms <= complete_ms, 'record producer measurements differ')
        reads += submitted
        requested += requested_bytes
        returned += returned_bytes
        produced.append([first_ms, complete_ms])
    return dict(request_work=frame['native_request'], bank_upload_bytes=int(frame['uploads']['requested_H2D_bytes']),
                read_jobs=reads, requested_read_bytes=requested, returned_read_bytes=returned,
                first_producer_ms=produced[0], first_chunk_exposed_ms=consumed[0][0],
                producer_ms=produced, consumer_ms=consumed, ple_wait=wait_data)


def after_execution(outcome, row, plan):
    require('storage_begin' not in outcome and outcome['request_routes_checked_before_continue'] == wait.speed.RUNS
            and len(outcome['storage_counters']) == wait.speed.RUNS
            and [[r['request_work'], r['bank_upload_bytes']] for r in outcome['routes']]
                == row['constant_request_work'], 'storage/request coverage or native work differs')
    for frame, sample, storage in zip(outcome['routes'], outcome['samples'], outcome['storage_counters'], strict=True):
        require(frame['ple_wait']['host_ple_ms'] <= sample['prompt_ms'] + 0.001
                and storage['request_window_ns'] >= sample['prompt_ms'] * 1e6,
                'the enclosing prompt/storage interval differs')
    outcome.update(arm=plan['arm'], clean_speed=False)


def request_time_ratios(a, b):
    x, y = a['samples'][0], b['samples'][0]
    ttft = [statistics.median(s['ttft_s'] for s in p['samples'][1:]) for p in (a, b)]
    return dict(first_pp_time=y['prompt_ms'] / x['prompt_ms'],
                first_ttft=y['ttft_s'] / x['ttft_s'],
                subsequent_pp_time=b['subsequent_median_ms'] / a['subsequent_median_ms'],
                subsequent_ttft=ttft[1] / ttft[0])


def decision(processes, order=ORDER, contrasts=(('legacy', 'one'), ('legacy', 'two'), ('one', 'two'))):
    require(tuple(p['arm'] for p in processes) == order
            and all(p['passed'] and p['complete'] and p['ownership_closed']
                    and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(order), 'require independently closed model processes')
    results = {}
    size = len(order) // 2
    blocks = ({p['arm']: p for p in processes[:size]}, {p['arm']: p for p in processes[size:]})
    for base, candidate in contrasts:
        pairs = []
        for block in blocks:
            a, b = block[base], block[candidate]
            first = [p['routes'][0] for p in (a, b)]
            storage = [p['storage_counters'][0] for p in (a, b)]
            pairs.append(dict(
                first_reads=[r['read_jobs'] for r in first],
                first_requested_read_bytes=[r['requested_read_bytes'] for r in first],
                first_chunk_exposed_ms=[r['first_chunk_exposed_ms'] for r in first],
                first_ple_wait_ms=[r['ple_wait']['future_get_ms'] for r in first],
                first_zfs_direct_bytes=[s['zfs_direct_bytes'] for s in storage],
                first_partition_read_bytes=[s['partition_read_bytes'] for s in storage],
                first_storage_interval_ns=[s['request_window_ns'] for s in storage],
                **request_time_ratios(a, b)))
        results[candidate + '_over_' + base] = dict(
            pairs=pairs, advance=all(p['first_reads'][1] < p['first_reads'][0]
                                    and p['first_pp_time'] < 1 and p['first_ttft'] < 1 for p in pairs),
            exposed_wait_falls=all(p['first_ple_wait_ms'][1] < p['first_ple_wait_ms'][0] for p in pairs),
            shared_read_counters_fall=all(p['first_zfs_direct_bytes'][1] < p['first_zfs_direct_bytes'][0]
                                         and p['first_partition_read_bytes'][1] < p['first_partition_read_bytes'][0]
                                         for p in pairs))
    return dict(contrasts=results, adopted=False)


def full_baseline():
    selected = baseline()
    prior = validate.committed(bank.RESULT, (KEY,))
    require(wait.retained.closed(prior, 'closed') and prior['decision'] == decision(prior['processes'])
            and prior['decision']['contrasts']['two_over_legacy']['advance']
            and all(prior[k] == selected[k] for k in ('binary', 'native_source_commit', 'native_build_sha256')),
            'require the repeated two-chunk gain on this executable before the full horizon')
    smoke = validate.committed(bank.RESULT, (FULL_SMOKE_KEY,))
    require(wait.retained.closed(smoke, 'closed') and tuple(smoke['outcomes']) == ('direct',)
            and smoke['plan']['record_window_chunks'] == {'direct': 'all'}
            and all(smoke['plan'][k] == selected[k] for k in (
                'binary', 'native_source_commit', 'native_build_sha256', 'host_fixture_sha256')),
            'verify the whole-request direct writer on this exact build')
    outcome = smoke['outcomes']['direct']
    require(outcome['passed'] and outcome['ownership_closed'] and not outcome['fresh_faults']
            and outcome['record_window_chunks'] == outcome['consumed_chunks'] == 15,
            'the whole-request reader must expose all 15 validated chunks')
    return dict(selected, two_chunk_proof_sha256=bank.signature(prior),
                full_native_smoke_sha256=bank.signature(smoke))


def run_clean(ctx, index):
    loader = wait.basis.core.source.device.load_module
    runner = loader('record_window_clean_benchmark', attribution.__file__)
    # The shared runner gates every request. In the private clean runner
    # those gates still verify the route without sampling storage counters.
    runner.observe_storage = lambda *_: None
    runner.run_observed(ctx, index)


def clean_baseline(candidate):
    full = candidate == 'full'
    selected = full_baseline() if full else baseline()
    prior = validate.committed(bank.RESULT, ((FULL_KEY if full else KEY),))
    require(wait.retained.closed(prior, 'closed'), 'complete the observed reader comparison first')
    processes = prior['processes']
    observed = decision(processes, FULL_ORDER, (('two', 'full'),)) if full else decision(processes)
    contrast = 'full_over_two' if full else candidate + '_over_legacy'
    identities = ('binary', 'native_source_commit', 'native_build_sha256')
    if full:
        identities += ('full_native_smoke_sha256',)
    require(prior['decision'] == observed and observed['contrasts'][contrast]['advance']
            and all(prior[k] == selected[k] for k in identities),
            'require the same-executable observed PLE comparison before clean timing')
    return dict(selected, observed_window_proof_sha256=bank.signature(prior))


def clean_route(text, plan):
    require(not any(marker in text for marker in (
        PRODUCER, CONSUMER, attribution.ple.PREFIX, *geometry.OBSERVED_PREFIXES)),
        'PLE timing observers must stay off for clean comparison')
    depth = WINDOW[plan['arm']]
    if depth == 0:
        require(MARKER not in text, 'grouped reader executed in the legacy arm')
        frame = geometry.timing.route(text, dict(plan, arm='early'))
    else:
        require(geometry.timing.MARKER not in text, 'legacy gather ran in the grouped arm')
        bare = ''.join(line for line in text.splitlines(keepends=True) if not line.startswith(MARKER))
        frame = wait.route(bare, dict(plan, arm='clean'))
        chunk = int(frame['work']['operator_max'])
        if depth == 'all':
            depth = 1 + (plan['read'] - 2) // chunk
        require([line for line in text.splitlines() if line.startswith(MARKER)]
                == [f'{MARKER}tokens_per_chunk={chunk} window_chunks={depth}']
                and text.count(MARKER) == 1, 'the clean grouped reader dispatch differs')
    return dict(request_work=frame['native_request'], bank_upload_bytes=int(frame['uploads']['requested_H2D_bytes']))


def clean_after_execution(outcome, row, plan):
    require('storage_begin' not in outcome and 'storage_counters' not in outcome
            and outcome['request_routes_checked_before_continue'] == wait.speed.RUNS
            and [[r['request_work'], r['bank_upload_bytes']] for r in outcome['routes']]
                == row['constant_request_work'], 'clean observer state or model work differs')
    outcome.update(arm=plan['arm'], clean_speed=True)


def clean_decision(processes, candidate):
    order = ('legacy', candidate, candidate, 'legacy')
    require(tuple(p['arm'] for p in processes) == order and all(
        p['passed'] and p['complete'] and p['ownership_closed'] and p['clean_speed']
        and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in processes)
        and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                 for p in processes}) == len(order), 'require four independently closed clean processes')
    pairs = [request_time_ratios(a, b) for a, b in ((processes[0], processes[1]), (processes[3], processes[2]))]
    return dict(candidate=candidate, pairs=pairs,
                repeated_first_gain=all(p['first_pp_time'] < 1 and p['first_ttft'] < 1 for p in pairs),
                subsequent_ratios={name: statistics.median(p[name] for p in pairs)
                                   for name in ('subsequent_pp_time', 'subsequent_ttft')},
                first_1000=[p['samples'][0]['prompt_tps'] >= 1000 for p in processes if p['arm'] == candidate],
                adopted=False)


def experiment(full=False, clean=None):
    ctx = geometry.experiment()
    key, order = (FULL_KEY, FULL_ORDER) if full else (KEY, ORDER)
    if clean:
        key = KEY.removesuffix('_v1') + '_clean_' + clean + '_v1'
        order = ('legacy', clean, clean, 'legacy')
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = key, bank.B / key.replace('_', '-'), THIS, order
    payload = ctx.ROOT / 'corpus.txt'
    base_command, base_save, base_current = ctx.command, ctx.save, ctx.current
    digest, paths = corpus.CORPORA[capacity.WORKLOAD]

    def save(row):
        if row['stage'] == 'prepared':
            with payload.open('xb') as target:
                target.write(corpus.corpus_payload(capacity.WORKLOAD))
            payload.chmod(0o400)
        base_save(row)

    def current(row):
        require(bank.sha(payload) == digest, 'corpus bytes differ')
        return base_current(row)

    ctx.baseline, ctx.save, ctx.current = full_baseline if full else baseline, save, current
    ctx.command = lambda plan: [*base_command(plan), '--prefill-corpus', str(payload)]
    def environment(row, plan):
        arm = plan['arm']
        if clean:
            env = geometry.timing.environment(row, dict(plan, arm='early'))
            env[attribution.IO_WORKERS] = '16'
        else:
            # Freeze the same observer settings across all arms. Organic
            # legacy tickets alone emit their page and record census.
            env = geometry.environment(row, dict(plan, arm='geometry'))
        require(phase.SELECTOR not in env, 'hold the dedicated row cache off in the grouping comparison')
        return dict(env, **{SELECTOR: str(WINDOW[arm])})

    ctx.environment = environment
    ctx.request_route, ctx.after_execution, ctx.decision = route, after_execution, decision
    ctx.extra_sources = (*ctx.extra_sources, THIS, corpus.THIS, capacity.THIS)
    ctx.selector = SELECTOR
    ctx.protocol = dict(order=list(order), read=wait.READ, requests_per_process=wait.speed.RUNS,
        corpus=capacity.WORKLOAD, corpus_revision=corpus.CORPUS_REVISION, corpus_sha256=digest,
        corpus_paths=list(paths), capacity_rows=capacity.DEFAULT_ROWS,
        scope='Use one rebuilt executable and the original cache capacity across the three prefill paths. '
              'Change only the record window selector. Use two reverse-order process blocks.',
        decision='Advance a scoped contrast if both blocks reduce first-request reads, prefill and TTFT. '
                 'Retain subsequent effects individually and compare clean timing before model qualification.',
        limits='One exact technical-prose workload. ZFS and partition counts are shared request-level deltas. '
               'The one- to two-chunk contrast includes the change in number of worker starts. '
               'Capacity interaction and sustained decode remain separate comparisons.',
        stop=geometry.PLAN['stop'])
    if full:
        ctx.decision = lambda processes: decision(processes, FULL_ORDER, (('two', 'full'),))
        ctx.protocol.update(
            scope='Hold the coherent reader, stock capacity and first-chunk priority fixed. '
                  'Compare two chunks with a whole-request horizon in both process orders.',
            limits='The whole-request planner is bounded to 262144 tokens. '
                   'Track the first exposed chunk, actual physical I/O counters and request TTFT.')
    if clean:
        ctx.baseline = lambda: clean_baseline(clean)
        ctx.run = lambda index: run_clean(ctx, index)
        ctx.request_route, ctx.after_execution = clean_route, clean_after_execution
        ctx.decision = lambda processes: clean_decision(processes, clean)
        ctx.protocol.update(
            scope='Compare the selected record horizon with the legacy reader in both process orders. '
                  'Same binary, input, ordinary row cache and native work. Disable PLE and storage observers.',
            decision='Require faster first prefill and client TTFT in both pairs for the clean gain. '
                     'Report subsequent results and the 1000 tok/s target separately.',
            limits='One technical-prose workload. The observed storage comparison and model qualification are separate.')
    return ctx


if __name__ == '__main__':
    require(sys.executable == bank.basis.reference.model.PYTHON, 'use the pinned controller interpreter')
    if sys.argv[1:2] == ['--build']:
        action, = sys.argv[2:]
        build_context().main(action)
    elif sys.argv[1:] == ['--host-fixture']:
        host_fixture()
    elif sys.argv[1:2] == ['--clean']:
        require(len(sys.argv) == 4 and sys.argv[2] in ('one', 'two', 'full'), 'select a measured record horizon')
        name = sys.argv.pop(2)
        sys.argv.pop(1)
        wait.speed.clock.main(experiment(clean=name))
    else:
        full = sys.argv[1:2] == ['--full-horizon']
        if full:
            sys.argv.pop(1)
        wait.speed.clock.main(experiment(full))
