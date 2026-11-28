"""Count per-gather PLE read overlap on the measured early-prefill route."""

from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_early_wait as attribution

timing, bank, wait, require = attribution.timing, attribution.bank, attribution.wait, attribution.require
early = timing.build
THIS = str(Path(__file__).resolve().relative_to(bank.P))
READER = 'src/ngram/ple_reader.cpp'
READER_BEFORE = '17de86439c2b72e04eb62769b9853c28a18e05d4'
READER_AFTER = 'ef9dcaba575945ba4acd0e32e4be440e635d6218'
TIMELINE_AFTER = '5b82c9f60b26d1c4795c023dad5bcd2e300ca577'
TIMELINE_SOURCES = ('src/kernels/ngram.cpp', 'src/prefill/prefill.cpp')
SELECTOR = 'STRATA_PREFILL_PLE_RECORD_AUDIT'
PREFIX = 'strata prefill PLE read geometry: '
HOST_PREFIX = 'strata prefill PLE host timeline: '
JOIN_PREFIX = 'strata prefill PLE join timeline: '
OBSERVED_PREFIXES = (PREFIX, HOST_PREFIX, JOIN_PREFIX)
PAGE_BYTES = 4096
RECORD_BYTES = 131072
ROWS_PER_TOKEN = 16
BUILD_KEY = 'dflash2_prefill_ple_record_native_v1'
BUILD_ROOT = bank.B / BUILD_KEY.replace('_', '-')
TREE = BUILD_ROOT / 'source'
KEY = 'dflash2_prefill_ple_record_geometry_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('control', 'geometry', 'geometry', 'control')
FIELDS = ('rows', 'jobs', 'requested_bytes', 'in_file_bytes', 'page_touches', 'unique_pages',
          'record_bytes', 'record_touches', 'unique_records')
HOST_STAMPS = ('allocate_ns', 'issue_ns', 'collect_ns', 'expand_ns', 'end_ns')
HOST_PHASES = ('allocation', 'issue', 'collect', 'expand')
PLAN = dict(order=list(ORDER), read=wait.READ, requests_per_process=wait.speed.RUNS,
    scope='Hold early PLE and 16 DirectFile workers fixed on one observer build. Compare two counterbalanced '
          'pairs with PLE wait and request-gated storage counters in both arms. Collect per-ticket record '
          'counts and host/join clocks in the geometry arm. Keep the verified 118785-token request bodies.',
    accounting='Match one ticket per 8192-token chunk and the 4096-token tail. Intersect the four disjoint '
               'host phases with the matching future join. Reconcile join sum with the full request wait. '
               'Keep intended job and record counts separate from shared-device counter deltas.',
    decision='Require each paired subsequent prompt-time ratio within [0.95, 1.05] for attribution. '
             'Keep per-ticket counts and first-use intervals for both independent observed processes. '
             'Use removable logical record touches to decide whether to study grouping on varied prompts.',
    limits='Ticket unions describe within-ticket reuse. The stage clocks are elapsed times and may include '
           'preemption. Shared-device counters and observer clocks provide attribution. Estimate clean '
           'speed with a separate comparison.',
    stop=timing.PLAN['stop'])


def build_inputs(revision):
    fields, payload, original, archive = early.inputs(revision)
    retained = bank.previous.read(early.ROOT / 'record.json')
    compact = early.builder.compact(retained)
    require(wait.retained.closed(retained, 'built')
            and compact == wait.basis.validation.committed(bank.RESULT, (early.KEY,))
            and all(retained[k] == fields[k] for k in ('source_files', 'recipe', 'native_source_commit')),
            'recover the exact measured early PLE source and recipe')

    hashes = early.source.pin(bank.P, (THIS, attribution.THIS, early.THIS))['source_sha256']
    require(bank.stage.hashed(dict(early.source.blobs(bank.P, revision, hashes))) == hashes,
            'commit the reader geometry controllers first')
    before = wait.basis.validation.git('show', READER_BEFORE + ':' + READER)
    after = wait.basis.validation.git('show', READER_AFTER + ':' + READER)
    require(payload[READER] == before and after != before,
            'the reader-only observer requires the exact reviewed preimage')
    original_timeline = dict(early.source.blobs(bank.P, TIMELINE_AFTER + '^', TIMELINE_SOURCES))
    timeline = dict(early.source.blobs(bank.P, TIMELINE_AFTER, TIMELINE_SOURCES))
    require(all(payload[p] == original_timeline[p] and timeline[p] != payload[p] for p in TIMELINE_SOURCES),
            'the host timeline requires the exact measured gather and prefill join sources')
    payload[READER] = after
    payload.update(timeline)
    files = bank.stage.hashed(payload)
    old, new = str(early.ROOT), str(BUILD_ROOT)
    recipe = fields['recipe']
    arguments = [arg.replace(old, new) for arg in recipe['configure']]
    require(arguments[2:5] == [str(TREE), '-B', str(BUILD_ROOT / 'build')],
            'relocate only the original build paths')
    fields.update(source_files=files, overlay_commit=TIMELINE_AFTER,
                  overlays={p: files[p] for p in (READER, *TIMELINE_SOURCES)},
                  reader_overlay_commit=READER_AFTER,
                  controller_sources=fields['controller_sources'] | hashes,
                  base_early_build_sha256=bank.signature(compact),
                  recipe=dict(configure=arguments, targets=recipe['targets'],
                              environment={k: v.replace(old, new) for k, v in recipe['environment'].items()}))
    return fields, payload, original, archive


def build_context():
    ctx = bank.basis.core.source.device.load_module('ple_record_build', early.base_build.__file__)
    ctx.KEY, ctx.ROOT, ctx.TREE, ctx.OVERLAYS = BUILD_KEY, BUILD_ROOT, TREE, (READER, *TIMELINE_SOURCES)
    ctx.inputs = build_inputs

    def prepare():
        require(not BUILD_ROOT.exists() and BUILD_KEY not in wait.basis.validation.committed(bank.RESULT),
                'inspect the existing geometry build before preparing')
        revision = early.source.git(bank.P, 'rev-parse', 'HEAD').decode().strip()
        fields, payload, _, _ = build_inputs(revision)
        BUILD_ROOT.mkdir(mode=0o700)
        for name in ('owned', 'tmp', 'cache'):
            (BUILD_ROOT / name).mkdir(mode=0o700)
        for name, raw in payload.items():
            path = TREE / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        row = dict(fields, stage='prepared', commands=[], active=False, passed=False, complete=False,
                   ownership_closed=True, scope='Measured early PLE source with host and reader observers.',
                   retention='Review at the end or first failed gate of the record-geometry comparison.')
        ctx.check(row, committed=False)
        ctx.save(row)

    ctx.prepare = prepare
    return ctx


def baseline():
    row = attribution.worker_baseline(early.context())
    native = bank.previous.read(BUILD_ROOT / 'record.json')
    require(wait.retained.closed(native, 'built') and native['base_early_build_sha256'] == row['native_build_sha256'],
            'require the built host and reader observers on the measured early PLE source')
    build_context().check(native)
    return dict(row, binary=native['binaries']['strata'],
                native_build_sha256=bank.signature(early.builder.compact(native)),
                reader_overlay_commit=READER_AFTER, host_timeline_commit=TIMELINE_AFTER)


def read_records(text, prefix, fields, count):
    lines = [line.removeprefix(prefix) for line in text.splitlines() if line.startswith(prefix)]
    require(len(lines) == text.count(prefix) == count, 'unexpected PLE record coverage: ' + prefix)
    records = []
    for line in lines:
        pairs = [word.split('=', 1) for word in line.split()]
        require(all(len(pair) == 2 and pair[1].isascii() and pair[1].isdecimal() for pair in pairs)
                and tuple(pair[0] for pair in pairs) == fields, 'malformed PLE record: ' + prefix)
        records.append({key: int(value) for key, value in pairs})
    return records


def read_geometry(text, pieces):
    tickets = read_records(text, PREFIX, FIELDS, len(pieces))
    for data, (_, tokens) in zip(tickets, pieces, strict=True):
        rows = tokens * ROWS_PER_TOKEN
        jobs, pages = data['jobs'], data['page_touches']
        require(data['rows'] == rows and data['record_bytes'] == RECORD_BYTES and jobs <= rows
                and jobs <= data['unique_pages'] <= pages <= 2 * jobs
                and data['in_file_bytes'] <= data['requested_bytes'] == pages * PAGE_BYTES
                and data['unique_records'] <= data['unique_pages']
                and jobs <= data['record_touches']
                and data['unique_records'] <= data['record_touches'] <= pages
                and bool(data['in_file_bytes']) == bool(jobs)
                and bool(data['unique_records']) == bool(jobs), 'inconsistent PLE read geometry counts')
    return tickets


def host_timeline(text, pieces, expected_wait_ms):
    host = read_records(text, HOST_PREFIX, ('tokens', *HOST_STAMPS), len(pieces))
    joins = read_records(text, JOIN_PREFIX, ('offset', 'tokens', 'begin_ns', 'end_ns'), len(pieces))
    last_join_end = 0
    tickets = []
    for (offset, tokens), worker, join in zip(pieces, host, joins, strict=True):
        bounds = [worker[key] for key in HOST_STAMPS]
        begin, end = join['begin_ns'], join['end_ns']
        require(join['offset'] == offset and join['tokens'] == worker['tokens'] == tokens
                and last_join_end <= min(begin, bounds[0]) and bounds[0] > 0
                and bounds == sorted(bounds) and max(begin, bounds[-1]) <= end,
                'PLE host or join timeline violates gather order or extent')
        phases = {}
        for name, start, stop in zip(HOST_PHASES, bounds[:-1], bounds[1:], strict=True):
            phases[name] = dict(wall_ns=stop - start, exposed_ns=max(0, min(stop, end) - max(start, begin)))
        join_ns = end - begin
        unaccounted = join_ns - sum(v['exposed_ns'] for v in phases.values())
        require(unaccounted >= 0, 'PLE host phases exceed the join window')
        tickets.append(dict(join_ns=join_ns, unattributed_join_ns=unaccounted, phases=phases))
        last_join_end = end
    if pieces:
        require(abs(sum(t['join_ns'] for t in tickets) / 1e6 - expected_wait_ms) <= 0.001,
                'PLE join windows do not reconcile with the enclosing wait')
    return tickets


def route(text, plan):
    clean = ''.join(line for line in text.splitlines(keepends=True) if not line.startswith(OBSERVED_PREFIXES))
    frame = attribution.route(clean, dict(plan, arm='early'))
    rows, chunk = plan['read'] - 1, int(frame['work']['operator_max'])
    pieces = ([(start, min(chunk, rows - start)) for start in range(0, rows, chunk)]
              if plan['arm'] == 'geometry' else [])
    return dict(frame, ple_read_geometry=read_geometry(text, pieces),
                ple_host_timeline=host_timeline(text, pieces, frame['ple_wait']['future_get_ms']))


def environment(row, plan):
    env = attribution.environment(row, dict(plan, arm='early'))
    return dict(env, **{attribution.IO_WORKERS: '16', SELECTOR: str(int(plan['arm'] == 'geometry'))})


def after_execution(outcome, row, plan):
    bare = [{k: v for k, v in frame.items() if k not in ('ple_read_geometry', 'ple_host_timeline')}
            for frame in outcome['routes']]
    attribution.after_execution(dict(outcome, routes=bare), row, dict(plan, arm='early'))
    outcome.update(arm=plan['arm'], clean_speed=False)


def decision(processes):
    pairs = wait.decision(processes, ORDER)['observed_over_clean']
    lo, hi = attribution.WARM_OBSERVER_RANGE
    observed = [p for p in processes if p['arm'] == 'geometry']
    totals = [[sum(t['record_touches'] - t['unique_records'] for t in frame['ple_read_geometry'])
               for frame in process['routes']] for process in observed]
    expand = [[sum(t['phases']['expand']['exposed_ns'] for t in frame['ple_host_timeline']) / 1e6
               for frame in process['routes']] for process in observed]
    eligible = all(lo <= p['subsequent'] <= hi for p in pairs)
    return dict(observer_over_control=pairs, subsequent_attribution_eligible=eligible,
                duplicate_record_touches_per_request=totals,
                exposed_cpu_expand_ms_per_request=expand,
                first_wait_ms=[p['routes'][0]['ple_wait']['future_get_ms'] for p in observed],
                varied_prompt_followup=eligible and all(first > 0 for first, *_ in totals),
                adopted=False)


def experiment():
    ctx = attribution.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.environment = baseline, PLAN, environment
    ctx.selector, ctx.extra_sources = SELECTOR, (*ctx.extra_sources, THIS)
    ctx.request_route, ctx.after_execution, ctx.decision = route, after_execution, decision
    return ctx


if __name__ == '__main__':
    require(sys.executable == bank.basis.reference.model.PYTHON, 'use the pinned controller interpreter')
    if sys.argv[1:2] == ['--build']:
        action, = sys.argv[2:]
        build_context().main(action)
    else:
        wait.speed.clock.main(experiment())
