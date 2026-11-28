"""Build and screen the dedicated long-prefill PLE row cache."""

import fcntl
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_cache_capacity as capacity

corpus, bank, wait, geometry = capacity.corpus, capacity.bank, capacity.wait, capacity.geometry
early, attribution = geometry.early, geometry.attribution
validate, require = capacity.validate, capacity.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_phase_cache_v1'
ROOT = bank.B / KEY.replace('_', '-')
BUILD_KEY = 'dflash2_prefill_ple_phase_cache_native_v1'
BUILD_ROOT = bank.B / BUILD_KEY.replace('_', '-')
TREE = BUILD_ROOT / 'source'
HOST_KEY = 'dflash2_prefill_ple_phase_cache_host_v1'
HOST_ROOT = BUILD_ROOT / 'host-fixture'
FILE = ROOT / 'corpus.txt'
SELECTOR = 'STRATA_SYCL_PREFILL_PLE_CACHE_ROWS'
CPP = ('src/ngram/ple_reader.cpp', 'src/kernels/ngram.cpp', 'src/prefill/prefill.cpp')
MODIFIED = (*CPP, 'include/strata/ngram/ple_reader.hpp', 'include/strata/kernels/ngram.hpp')
FIXTURE = 'tests/core/ple_phase_cache_test.cpp'
HOST_CPP = (FIXTURE, 'src/ngram/ple_reader.cpp', 'src/platform/direct_file.cpp')
HOST_INPUTS = (*HOST_CPP, 'include/strata/ngram/ple_reader.hpp', 'include/strata/platform/direct_file.hpp')
ARCHIVE = '1dba95fb1702e40139553fbdc8f842e73abda08a'
PREIMAGE = 'ebb8b91c95a424c6f7c013e97ef993c680e13179'
START = 'strata PLE row cache: '
TICKET = 'strata prefill PLE cache ticket: '
FIRST = 'strata prefill PLE cache route: '
HANDOFF_KEY = 'dflash2_prefill_ple_phase_cache_handoff_v1'
HANDOFF_ORDER = ('stock', 'large', 'large', 'stock', 'stock', 'large')
TICKET_FIELDS = re.compile(r'rows=(\d+) capacity=(\d+)')


def capacity_proof():
    row = validate.committed(bank.RESULT, (capacity.KEY,))
    require(wait.retained.closed(row, 'closed') and row['decision'] == capacity.decision(row['processes'])
            and row['decision']['warm_cache_mechanism_advances'], 'require repeated measured shared capacity gain')
    return row


def build_inputs(revision):
    fields, payload, original, early_archive = geometry.build_inputs(revision)
    archived = bank.previous.read(geometry.BUILD_ROOT / 'record.json')
    compact = early.builder.compact(archived)
    proof = capacity_proof()
    require(wait.retained.closed(archived, 'built')
            and compact == bank.previous.historic(ARCHIVE, geometry.BUILD_KEY)
            and all(archived[k] == fields[k] for k in ('source_files', 'recipe', 'native_source_commit'))
            and proof['binary'] == archived['binaries']['strata']
            and proof['native_build_sha256'] == bank.signature(compact),
            'recover the exact measured observer build from its sealed receipt')
    blobs = early.source.blobs
    preimage = dict(blobs(bank.P, PREIMAGE, MODIFIED))
    overlay = dict(blobs(bank.P, revision, (*MODIFIED, FIXTURE)))
    require(preimage == {p: payload[p] for p in MODIFIED}
            and all(overlay[p] != data for p, data in preimage.items())
            and FIXTURE not in payload, 'require exact original inputs and a new fixture')
    payload.update(overlay)
    controls = early.source.pin(bank.P, (THIS, corpus.THIS, capacity.THIS))['source_sha256']
    require(bank.stage.hashed(dict(blobs(bank.P, revision, controls))) == controls,
            'commit the exact phase controller inputs')
    recipe = fields['recipe']
    old, new = str(geometry.BUILD_ROOT), str(BUILD_ROOT)
    configure = [arg.replace(old, new) for arg in recipe['configure']]
    require(configure[2:5] == [str(TREE), '-B', str(BUILD_ROOT / 'build')], 'relocate only build paths')
    compiler, = (arg.split('=', 1)[1] for arg in configure if arg.startswith('-DCMAKE_CXX_COMPILER='))
    digest = original['toolchain'][compiler]
    require(bank.sha(compiler) == digest, 'the host fixture compiler differs from the pinned native toolchain')
    host_exe = str(HOST_ROOT / 'ple_phase_cache_test')
    source_hashes = bank.stage.hashed(payload)
    host_recipe = dict(compiler=compiler, compiler_sha256=digest,
        compile=[compiler, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-pthread',
                 '-I', str(TREE / 'include'), *(str(TREE / p) for p in HOST_CPP), '-o', host_exe],
        command=[host_exe, str(HOST_ROOT / 'cases')], source_sha256={p: source_hashes[p] for p in HOST_INPUTS})
    fields.update(native_source_commit=revision, overlay_commit=revision, host_fixture_recipe=host_recipe,
                  source_files=source_hashes, overlays=bank.stage.hashed(overlay),
                  controller_sources=fields['controller_sources'] | controls,
                  geometry_build_sha256=bank.signature(compact), capacity_proof_sha256=bank.signature(proof),
                  recipe=dict(configure=configure, targets=recipe['targets'],
                              environment={k: v.replace(old, new) for k, v in recipe['environment'].items()}))
    return fields, payload, original, early_archive


def build_context():
    ctx = bank.basis.core.source.device.load_module('ple_phase_cache_build', early.base_build.__file__)
    ctx.KEY, ctx.ROOT, ctx.TREE, ctx.OVERLAYS = BUILD_KEY, BUILD_ROOT, TREE, CPP
    ctx.inputs = build_inputs

    def prepare():
        require(not BUILD_ROOT.exists() and BUILD_KEY not in validate.committed(bank.RESULT),
                'inspect the existing phase cache build')
        revision = early.source.git(bank.P, 'rev-parse', 'HEAD').decode().strip()
        fields, payload, _, _ = build_inputs(revision)
        BUILD_ROOT.mkdir(mode=0o700)
        for subdir in ('owned', 'tmp', 'cache'):
            (BUILD_ROOT / subdir).mkdir(mode=0o700)
        for name, data in payload.items():
            path = TREE / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        row = dict(fields, stage='prepared', commands=[], active=False, passed=False, complete=False,
                   ownership_closed=True, scope='Rebuild the original geometry executable with phase-local cache storage.',
                   retention='Review at the first failed gate or the close of the prefill and decode comparison.')
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
        require(wait.retained.closed(built, 'built')
                and HOST_KEY not in validate.committed(bank.RESULT) and not HOST_ROOT.exists(),
                'require a fresh host fixture on the coherent native build')
        build_context().check(built)
        plan = built['host_fixture_recipe']
        paths = plan['source_sha256']
        require(all(bank.sha(TREE / path) == digest for path, digest in paths.items())
                and bank.sha(plan['compiler']) == plan['compiler_sha256'], 'host source or toolchain changed')
        HOST_ROOT.mkdir(mode=0o700)
        (HOST_ROOT / 'cases').mkdir(mode=0o700)
        env = dict(built['recipe']['environment'], STRATA_IO_THREADS='2')
        record = dict(stage='running', passed=False, complete=False, active=True, ownership_closed=False,
                      native_build_sha256=bank.signature(builder.compact(built)), recipe=plan,
                      environment_sha256=bank.signature(env))
        bank.stage.write(HOST_ROOT / 'record.json', record)
        owner = builder.ownership.Owner(HOST_ROOT / 'owned')
        commands = []
        limits = dict(wall=90, cpu=60, text_cap=128 << 10, file_cap=2 << 20,
                      total_cap=8 << 20, rss_cap=1 << 30)
        try:
            compiled, _, _ = owner.run('compile', plan['compile'], env, TREE, **limits)
            commands.append(dict(label='compile', exit_code=compiled.get('exit_code'),
                                 ownership_closed=bool(builder.ownership.closed(compiled))))
            require(commands[-1]['ownership_closed'] and commands[-1]['exit_code'] == 0,
                    'host fixture compilation failed')
            executable = Path(plan['command'][0])
            identity = dict(path=str(executable), sha256=bank.sha(executable), bytes=executable.stat().st_size)
            require(all(bank.sha(TREE / p) == h for p, h in paths.items())
                    and bank.sha(executable) == identity['sha256'], 'host fixture inputs or image changed')
            tested, stdout, _ = owner.run('test', plan['command'], env, TREE, **limits)
            commands.append(dict(label='test', exit_code=tested.get('exit_code'),
                                 ownership_closed=bool(builder.ownership.closed(tested))))
            require(commands[-1]['ownership_closed'] and commands[-1]['exit_code'] == 0
                    and stdout.read_text().splitlines() == ['PLE phase cache isolation checks passed']
                    and bank.sha(executable) == identity['sha256'], 'host cache fixture failed')
            record.update(stage='closed', executable=identity, passed=True, complete=True)
        except Exception as exc:
            record.update(stage='failed', error=str(exc))
        finally:
            record.update(commands=commands, active=owner.active is not None,
                          ownership_closed=owner.active is None and all(
                              builder.ownership.closed(c) for c in owner.commands))
            bank.stage.write(HOST_ROOT / 'record.json', record)
            results = bank.previous.read(bank.RESULT)
            results[HOST_KEY] = record
            bank.stage.write(bank.RESULT, results)
        require(record['passed'] and record['ownership_closed'], 'host phase cache fixture did not close')
        print('Host phase cache checks passed. Commit the fixture receipt before model timing.', flush=True)


def baseline():
    built = bank.previous.read(BUILD_ROOT / 'record.json')
    require(wait.retained.closed(built, 'built'), 'finish the coherent phase cache build')
    build_context().check(built)
    host = bank.previous.read(HOST_ROOT / 'record.json')
    native_identity = bank.signature(early.builder.compact(built))
    require(wait.retained.closed(host, 'closed') and host == validate.committed(bank.RESULT, (HOST_KEY,))
            and host['native_build_sha256'] == native_identity
            and host['recipe'] == built['host_fixture_recipe']
            and bank.sha(host['executable']['path']) == host['executable']['sha256'],
            'require exact same-source closed host cache isolation')
    proof = capacity_proof()
    old = attribution.worker_baseline(early.context())
    require(capacity.FLAG not in old['configuration']['args']
            and built['capacity_proof_sha256'] == bank.signature(proof), 'preserve the ordinary cache capacity')
    old.pop('frozen_prefill_work')
    matching = [p for p in proof['processes'] if p['arm'] == 'stock']
    expected = [[[r['request_work'], r['bank_upload_bytes']] for r in p['routes']] for p in matching]
    require(len(expected) == 2 and expected[0] == expected[1], 'require matched stock request work')
    return dict(old, binary=built['binaries']['strata'], native_source_commit=built['native_source_commit'],
                native_build_sha256=native_identity, host_fixture_sha256=bank.signature(host),
                request_hashes=proof['request_hashes'], frozen_work=expected[0],
                capacity_proof_sha256=built['capacity_proof_sha256'])


def check_first_cache(text, arm, requests):
    isolated = arm == 'large'
    row_count = capacity.LARGE_ROWS if isolated else capacity.DEFAULT_ROWS
    expected = f'{FIRST}dedicated={int(isolated)} active_rows={row_count} ordinary_rows={capacity.DEFAULT_ROWS}'
    require([line for line in text.splitlines() if line.startswith(FIRST)] == [expected] * requests
            and text.count(FIRST) == requests, 'actual long-prefill PLE phase or capacity differs')


def route(text, plan):
    check_first_cache(text, plan['arm'], 1)
    kept = ''.join(line for line in text.splitlines(keepends=True) if not line.startswith(TICKET))
    frame = geometry.route(kept, dict(plan, arm='geometry'))
    matches = [TICKET_FIELDS.fullmatch(line.removeprefix(TICKET)) for line in text.splitlines()
               if line.startswith(TICKET)]
    require(len(matches) == text.count(TICKET) and all(matches), 'malformed cache route marker')
    sizes = [(ticket['rows'], capacity.LARGE_ROWS) for ticket in frame['ple_read_geometry']]
    require([tuple(map(int, m.groups())) for m in matches] == (sizes if plan['arm'] == 'large' else []),
            'the active ticket did not use the planned prefill-only cache')
    return frame


def after_execution(outcome, row, plan):
    text = (Path(plan['root']) / 'engine.stderr').read_text()
    starts = [line for line in text.splitlines() if line.startswith(START)]
    expected = [f'{START}long_prefill={capacity.LARGE_ROWS} ordinary={capacity.DEFAULT_ROWS}']
    require(starts == (expected if plan['arm'] == 'large' else [])
            and [[r['native_request'], int(r['uploads']['requested_H2D_bytes'])] for r in outcome['routes']]
                == row['frozen_work'], 'actual capacity or native inference work differs')
    corpus.after_execution(outcome, row, plan)


def experiment():
    ctx = geometry.experiment()
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, capacity.ORDER
    base_command, base_save, base_current = ctx.command, ctx.save, ctx.current
    digest, paths = corpus.CORPORA[capacity.WORKLOAD]

    def save(row):
        if row['stage'] == 'prepared':
            with FILE.open('xb') as target:
                target.write(corpus.corpus_payload(capacity.WORKLOAD))
            FILE.chmod(0o400)
        base_save(row)

    def current(row):
        require(bank.sha(FILE) == digest, 'corpus bytes differ')
        return base_current(row)

    ctx.baseline, ctx.current, ctx.save = baseline, current, save
    ctx.command = lambda plan: [*base_command(plan), '--prefill-corpus', str(FILE)]
    ctx.environment = lambda row, plan: dict(geometry.environment(row, dict(plan, arm='geometry')),
        **{SELECTOR: str(capacity.LARGE_ROWS if plan['arm'] == 'large' else 0)})
    ctx.request_route, ctx.after_execution, ctx.decision = route, after_execution, capacity.decision
    ctx.extra_sources = (*ctx.extra_sources, THIS, corpus.THIS, capacity.THIS)
    ctx.selector = SELECTOR
    ctx.protocol = dict(order=list(capacity.ORDER), read=wait.READ, requests_per_process=wait.speed.RUNS,
        corpus=capacity.WORKLOAD, corpus_revision=corpus.CORPUS_REVISION, corpus_sha256=digest, corpus_paths=list(paths),
        ordinary_cache_rows=capacity.DEFAULT_ROWS, dedicated_cache_rows=capacity.LARGE_ROWS,
        memory_addition_bytes=capacity.LARGE_ROWS * 94 + capacity.LARGE_ROWS // 8,
        scope='Hold the ordinary PLE cache at 1,048,576 rows. On one coherent binary, compare the disabled and '
              '4,194,304-row long-prefill cache in two reverse-order pairs. Check per-ticket selection.',
        decision='Advance only after fewer subsequent reads and faster subsequent prefill and TTFT repeat. '
                 'Run the long-prefill-to-DFlash2 thinking comparison, ordinary decode, physical 262144 and '
                 'live RESTORE before selecting a production cache handoff.',
        stop=geometry.PLAN['stop'])
    return ctx


def handoff_experiment():
    import dflash2_primary_continuation as primary

    ctx = primary.experiment()
    ctx.KEY, ctx.THIS, ctx.ORDER = HANDOFF_KEY, THIS, HANDOFF_ORDER
    ctx.ROOT = bank.B / HANDOFF_KEY.replace('_', '-')
    base_pins, base_environment = ctx.pins, ctx.environment

    def model_baseline():
        phase = baseline()
        closed = validate.committed(bank.RESULT, (KEY,))
        require(wait.retained.closed(closed, 'closed')
                and closed['decision'] == capacity.decision(closed['processes'])
                and closed['decision']['warm_cache_mechanism_advances']
                and all(phase[name] == closed[name] for name in (
                    'binary', 'native_source_commit', 'native_build_sha256', 'host_fixture_sha256')),
                'require the measured same-build prefill-only gain first')
        reference = primary.retained.baseline()
        require(all(reference[k] == phase[k] for k in (
            'driver', 'boot', 'runtime_manifest', 'dflash2_assets', 'native_validation_probe',
            'python_contract', 'python_selected')),
            'the sustained workload must use the same runtime and DFlash2 model assets')
        return dict(reference, binary=phase['binary'], native_source_commit=phase['native_source_commit'],
                    native_build_sha256=phase['native_build_sha256'],
                    phase_prefill_proof_sha256=bank.signature(closed))

    def environment(row, plan):
        env = base_environment(row, plan)
        require(env.get('STRATA_PREFILL_PLE_RECORD_AUDIT', '0') == '0' and SELECTOR not in env,
                'sustained handoff requires the uninstrumented prefill runtime')
        return dict(env, **{SELECTOR: str(capacity.LARGE_ROWS if plan['arm'] == 'large' else 0)})

    def pins(revision):
        files = base_pins(revision)
        for name in (THIS, capacity.THIS, corpus.THIS, geometry.THIS):
            path = bank.P / name
            require(path.read_bytes() == validate.git('show', revision + ':' + name),
                    'commit exact phase handoff control: ' + name)
            files[name] = bank.sha(path)
        return files

    def routes(text, samples, arm):
        check_first_cache(text, arm, len(primary.WORKLOADS))
        expected = ([f'{START}long_prefill={capacity.LARGE_ROWS} ordinary={capacity.DEFAULT_ROWS}']
                    if arm == 'large' else [])
        require([line for line in text.splitlines() if line.startswith(START)] == expected
                and TICKET not in text, 'unexpected cache size or ticket observer in sustained generation')
        return primary.routes(text, samples, 'dflash2')

    def decision(processes):
        require(tuple(p['arm'] for p in processes) == HANDOFF_ORDER
                and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                         for p in processes}) == len(HANDOFF_ORDER)
                and all(p['passed'] and p['complete'] and p['ownership_closed']
                        and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in processes),
                'require the three independent completed handoff pairs')
        comparisons = {name: [] for name in primary.WORKLOADS}
        for first, second in zip(processes[::2], processes[1::2], strict=True):
            stock, large = (first, second) if first['arm'] == 'stock' else (second, first)
            for name, pairs in comparisons.items():
                x, y = [next(s for s in p['evidence']['samples'] if s.get('workload') == name)
                        for p in (stock, large)]
                route_x, route_y = stock['routes'][name], large['routes'][name]
                fixed = ('residual_gpu_rows', 'direct_pieces', 'kv', 'bank_h2d_requested_bytes', 'placement_preserved')
                require(x['request_sha256'] == y['request_sha256'] and x['read_tokens'] == y['read_tokens']
                        and x['reused_tokens'] == y['reused_tokens'] == 0
                        and min(x['generated_tokens'], y['generated_tokens']) >= 1024
                        and all(route_x[k] == route_y[k] for k in fixed), 'paired inference work differs')
                matched_decode = (all(x[k] == y[k] for k in primary.continuation.WORK_FIELDS)
                                  and route_x['expert_work'] == route_y['expert_work'])
                pairs.append(dict(matched_decode=matched_decode, generated=[x['generated_tokens'], y['generated_tokens']],
                                  first_client_ttft_s=[x['ttft_s'], y['ttft_s']],
                                  prefill_ms=[x['prompt_ms'], y['prompt_ms']],
                                  tg_ratio=large['evidence']['decode_rates'][name] / stock['evidence']['decode_rates'][name],
                                  drafts=[[s['drafts_accepted'], s['drafts_offered']] for s in (x, y)]))
        screens = {name: dict(all_work_matched=all(p['matched_decode'] for p in pairs),
                              geometric_mean_tg_ratio=statistics.geometric_mean(p['tg_ratio'] for p in pairs),
                              paired_tg_ratios=[p['tg_ratio'] for p in pairs])
                   for name, pairs in comparisons.items()}
        return dict(samples=comparisons, screens=screens,
                    sustained_handoff_preserved=all(s['all_work_matched'] and s['geometric_mean_tg_ratio'] >= 1
                                                    for s in screens.values()), adopted=False)

    ctx.baseline, ctx.environment, ctx.pins, ctx.routes = model_baseline, environment, pins, routes
    ctx.evidence_reader.decide = decision
    ctx.protocol = dict(primary.PLAN, order=list(HANDOFF_ORDER),
        scope='Three reverse-order pairs on the same phase-cache binary. Keep the ordinary capacity fixed. '
              'Measure client TTFT and at least 1024 generated tokens with thinking after actual split long prefill.',
        decision='Require the same native prefill work, same sustained decode work and at least stock geometric '
                 'mean TG on both workloads. Retain individual PP, TG, TTFT and draft counts. Use a new plan '
                 'for ambiguous repetitions, weak acceptance, agent session and physical-context qualification.')
    return primary.clock, ctx


if __name__ == '__main__':
    require(sys.executable == bank.basis.reference.model.PYTHON, 'use the pinned controller interpreter')
    if sys.argv[1:2] == ['--handoff']:
        sys.argv.pop(1)
        clock, campaign = handoff_experiment()
        clock.main(campaign)
    elif sys.argv[1:] == ['--host-fixture']:
        host_fixture()
    elif sys.argv[1:2] == ['--build']:
        action, = sys.argv[2:]
        build_context().main(action)
    else:
        wait.speed.clock.main(experiment())
