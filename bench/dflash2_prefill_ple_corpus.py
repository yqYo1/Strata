"""Check first-use PLE record and wait geometry across exact-length corpora."""

from hashlib import sha256
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_record_geometry as geometry

bank, wait, require = geometry.bank, geometry.wait, geometry.require
validate = wait.basis.validation
THIS = str(Path(__file__).resolve().relative_to(bank.P))
GEOMETRY_PROOF = '5476afa882c536c5fd61cc9ef0c3c93b102c262b'
CORPUS_REVISION = 'a85356c6a54634e8be731ef16d03b2df13481731'
CORPORA = {
    'cxx': (
        '0f722fc22123dc2d17007681de9ff05886f2c467a52d9cd70cf60314849eb618',
        ('src/core/expert_source.cpp', 'src/core/verify.cpp', 'src/prefill/prefill.cpp',
         'src/prefill/layer_major.inc', 'src/prefill/moe_window.inc'),
    ),
    'technical-prose': (
        'f58e5b223152a6d566b35cbfdb8cf25568823ab85eb877517de245b756d995d0',
        ('README.md', 'docs/HOW_IT_WORKS.md', 'docs/DETAILS.md', 'docs/INSTALL.md',
         'docs/MODELS.md', 'docs/COMMUNITY_BENCHMARKS.md', 'docs/MCP_SERVER.md',
         'docs/SYCL_BATTLEMAGE_OPERATIONS.md', 'docs/SYCL_IMPLEMENTATION.md',
         'docs/RESOURCE_PROFILING.md'),
    ),
}
ORDER = ('geometry', 'geometry')
WORKLOADS = ('cxx', 'technical-prose', 'technical-prose', 'cxx')


def baseline():
    prior = bank.previous.historic(GEOMETRY_PROOF, geometry.KEY)
    compact = validate.committed(bank.RESULT, (geometry.KEY,))
    require(wait.retained.closed(prior, 'closed') and wait.retained.closed(compact, 'closed')
            and compact['archive_commit'] == GEOMETRY_PROOF
            and prior['decision'] == compact['decision'] == geometry.decision(prior['processes'])
            and prior['decision']['varied_prompt_followup'],
            'require the closed counterbalanced PLE attribution and its varied-prompt gate')
    selected = geometry.baseline()
    require(all(prior[k] == selected[k] for k in ('binary', 'native_source_commit', 'native_build_sha256')),
            'the corpus comparison uses the same instrumented build')
    selected.pop('frozen_prefill_work')
    selected.pop('request_hashes')
    return dict(selected, prior_ple_decision_sha256=bank.signature(prior['decision']))


def compact_request(frame):
    read = frame['ple_read_geometry']
    host = frame['ple_host_timeline']
    record_touches = sum(g['record_touches'] for g in read)
    duplicates = record_touches - sum(g['unique_records'] for g in read)
    exposed = {name: sum(t['phases'][name]['exposed_ns'] for t in host) / 1e6
               for name in geometry.HOST_PHASES}
    return dict(ple_wait=frame['ple_wait'], request_work=frame['native_request'],
                bank_upload_bytes=int(frame['uploads']['requested_H2D_bytes']),
                read_jobs=sum(g['jobs'] for g in read), record_touches=record_touches,
                requested_read_bytes=sum(g['requested_bytes'] for g in read),
                within_ticket_duplicates=duplicates,
                first_two_duplicates=sum(g['record_touches'] - g['unique_records'] for g in read[:2]),
                exposed_ms=exposed)


def after_execution(outcome, row, plan):
    runs = row['plan']['requests_per_process']
    require('storage_begin' not in outcome and outcome['request_routes_checked_before_continue'] == runs
            and len(outcome['storage_counters']) == runs, 'incomplete request gates')
    for frame, sample, storage in zip(outcome['routes'], outcome['samples'], outcome['storage_counters'], strict=True):
        require(frame['ple_early_reads'] == 1 and frame['ple_read_geometry'] and frame['ple_host_timeline']
                and frame['ple_wait']['host_ple_ms'] <= sample['prompt_ms'] + 0.001
                and storage['request_window_ns'] >= sample['prompt_ms'] * 1e6,
                'observe all early-read work and intervals')
    outcome.update(arm=plan['arm'], clean_speed=False, routes=[compact_request(f) for f in outcome['routes']])


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and all(p['complete'] and p['passed'] and p['ownership_closed']
                    and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER),
            'two independently closed corpus processes required')
    first = [p['routes'][0] for p in processes]
    return dict(first_wait_ms=[r['ple_wait']['future_get_ms'] for r in first],
                first_collect_ms=[r['exposed_ms']['collect'] for r in first],
                first_expand_ms=[r['exposed_ms']['expand'] for r in first],
                duplicate_record_touches=[r['within_ticket_duplicates'] for r in first],
                duplicate_fraction=[r['within_ticket_duplicates'] / max(1, r['record_touches']) for r in first],
                repeat_duplicate_reads=all(r['within_ticket_duplicates'] > 0 for r in first), adopted=False)


def campaign_key(name):
    return 'dflash2_prefill_ple_corpus_' + name.replace('-', '_') + '_v1'


def corpus_payload(name):
    digest, paths = CORPORA[name]
    data = b'\n\n'.join(validate.git('show', CORPUS_REVISION + ':' + path) for path in paths)
    require(sha256(data).hexdigest() == digest, 'committed corpus inputs differ')
    return data


def experiment(name):
    digest, paths = CORPORA[name]
    ctx = geometry.experiment()
    ctx.KEY = campaign_key(name)
    ctx.ROOT, ctx.THIS, ctx.ORDER = bank.B / ctx.KEY.replace('_', '-'), THIS, ORDER
    corpus = ctx.ROOT / 'corpus.txt'
    base_command, base_save, base_current, base_run = ctx.command, ctx.save, ctx.current, ctx.run

    def save(row):
        if row['stage'] == 'prepared':
            row['request_hashes'] = {}
            with corpus.open('xb') as output:
                output.write(corpus_payload(name))
            corpus.chmod(0o400)
        base_save(row)

    def current(row):
        require(bank.sha(corpus) == digest, 'generated corpus bytes changed')
        return base_current(row)

    def run(index):
        counts = {}
        for workload in CORPORA:
            key = campaign_key(workload)
            record = bank.previous.read(bank.B / key.replace('_', '-') / 'record.json')
            committed = validate.committed(bank.RESULT, (key,))
            require(record['host_passed'] and record['ownership_closed'] and not record['active']
                    and record['processes'] == committed['processes']
                    and all(p['passed'] for p in record['processes']), 'prepare both corpus cohorts')
            counts[workload] = len(record['processes'])
        done = sum(counts.values())
        require(done < len(WORKLOADS) and WORKLOADS[done] == name and counts[name] == index
                and counts == {w: WORKLOADS[:done].count(w) for w in CORPORA},
                'follow the committed interleaved corpus order')
        base_run(index)

    ctx.extra_sources = (*ctx.extra_sources, THIS)
    ctx.baseline, ctx.save, ctx.current, ctx.run = baseline, save, current, run
    ctx.command = lambda plan: [*base_command(plan), '--prefill-corpus', str(corpus)]
    ctx.after_execution, ctx.decision = after_execution, decision
    ctx.protocol = dict(order=list(ORDER), request_order=list(WORKLOADS), read=wait.READ,
        requests_per_process=wait.speed.RUNS, corpus=name, corpus_revision=CORPUS_REVISION,
        corpus_paths=list(paths), corpus_sha256=digest,
        scope='Two first-use and subsequent measurements for each corpus. Alternate independent processes '
              'across the two corpora, keeping early PLE, the route and all observer settings fixed.',
        decision='Use within-ticket duplicate count and exposed Direct File collection time per corpus. '
                 'Keep fresh first-use and subsequent samples. This is a diagnostic input-variation screen.',
        stop=geometry.PLAN['stop'])
    return ctx


if __name__ == '__main__':
    name = sys.argv.pop(1)
    require(name in CORPORA, 'select the cxx or technical-prose corpus')
    wait.speed.clock.main(experiment(name))
