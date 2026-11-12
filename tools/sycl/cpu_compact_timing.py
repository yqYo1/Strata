#!/usr/bin/env python3
"""Prospective CPU-only lookup/GU timing using V4-qualified source-built archives.

Main-authored; repository LICENSE. Native source, archives and every timing arm
are immutable. No engine/GPU work or production-pool speed claim.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import sys

sys.dont_write_bytecode = True
import admit_cpu_compact_numeric as prior

hot, producer = prior.hot, prior.producer
P, B = hot.P, hot.B
ROOT = B / 'cpu-compact-hot-timing-v1'
PLAN = P / 'bench/sycl/cpu_compact_hot/timing-plan-v1.json'
THIS = 'tools/sycl/cpu_compact_timing.py'
SOURCE = 'bench/sycl/cpu_compact_hot/timing.cpp'
need, sha, write = hot.need, hot.sha, hot.write
ORDER = ['reference', 'compact', 'compact', 'reference', 'reference', 'compact']


def std_headers():
    pins = {str(p.resolve()): sha(p) for root in
        ('/usr/include/c++/13', '/usr/include/x86_64-linux-gnu/c++/13')
        for p in Path(root).rglob('*') if p.is_file()}
    need(len(pins) > 500, 'installed standard headers incomplete')
    return pins


def signature(pins):
    return hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def numeric_gate():
    plan = json.loads(prior.PLAN.read_text()); prior.verify(plan)
    check = json.loads((prior.ROOT / 'check.json').read_text())
    need(check['passed'] and check['complete'] and check['ownership_closed'] and not check['active']
         and check['numerical_qualified'] and check['binary'] == plan['binary']
         and check['plan_sha256'] == sha(prior.PLAN)
         and all(hot.owner.closed(c) for c in check['commands']), 'closed exact numeric pass required')
    return plan


def prepare():
    need(not ROOT.exists() and not PLAN.exists(), 'fresh timing root/plan required')
    old = numeric_gate(); original = json.loads(producer.PLAN.read_text())
    revision = hot.git('rev-parse', 'HEAD').decode().strip()
    sources = {}
    for name in (THIS, SOURCE, 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md'):
        raw = hot.git('show', revision + ':' + name)
        need(raw == (P / name).read_bytes(), 'commit source first: ' + name)
        sources[name] = hashlib.sha256(raw).hexdigest()
    build = producer.ROOT / 'build'
    archives = {str(build / name): sha(build / name) for name in
        ('libcompact_reference_cpu.a', 'ggml/src/libggml-cpu.a', 'ggml/src/libggml-base.a')}
    need(all(old['artifacts'].get(p) == h for p, h in archives.items()), 'unqualified archive')
    flags = ['-Wno-error=deprecated-declarations',
        '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13', '-O3', '-DNDEBUG',
        '-std=gnu++20', '-march=znver3', '-mtune=znver3', '-DSTRATA_CPU_SERVICE_PROFILE=1',
        '-DSTRATA_NATIVE_EXPERTS=1']
    ggml = Path(original['GGML']['root'])
    includes = ['-I' + str(p) for p in (P / 'include', ggml / 'ggml/src', ggml / 'ggml/include')]
    environment = dict(old['environment'], TMPDIR=str(ROOT / 'tmp'))
    native = dict(old['native_environment'], TMPDIR=str(ROOT / 'tmp'), STRATA_IQ3_PAIR_AUDIT='0')
    plan = dict(schema='compact-cache-hot-cold-v1', source_commit=revision, sources=sources,
        qualified_archive_source=original['source_commit'], qualified_admission_source=old['admission_source_commit'],
        numeric_plan_sha256=sha(prior.PLAN), numeric_check_sha256=sha(prior.ROOT / 'check.json'),
        archives=archives, standard_headers=dict(count=len(std_headers()), sha256=signature(std_headers())),
        compiler=hot.CXX, compiler_sha256=sha(hot.CXX), formats=old['formats'],
        compile=[hot.CXX, *flags, *includes, '-MD', '-MF', str(ROOT / 'timing.d'),
                 '-c', str(P / SOURCE), '-o', str(ROOT / 'timing.o')],
        link=[hot.CXX, '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
              '-O3', '-DNDEBUG', '-Wl,-t', str(ROOT / 'timing.o'), *archives, '-lm',
              '-o', str(ROOT / 'timing')],
        environment=environment, native_environment=native,
        check_environment=dict(native, STRATA_IQ3_PAIR_AUDIT='1'),
        order=ORDER, physical_cpus=[0, 1, 2, 3, 4, 5],
        scope='New CPU-only fixture omits the unused SYCL-id-query flag;all qualified production archives unchanged.'
              'Isolate actual native_gu_rows versus immutable Cache.lookup+Lease.gu_rows NT1fullGU.'
              '1or6physical-core workers,one full640-row expert job per worker iteration.'
              'Not the production ExpertPool launch,partial-row scheduler,Down or whole model.',
        workloads='Each Down20/42 uses32actual experts and4native-Q8_K activations. Reuse keeps'
              'one expert per worker;stream32 traverses32experts. Both are synthetic reuse policies,'
              'not observed model hotness or controlled DRAM-cold access. Raw Down bytes remain resident.',
        timing='Fresh process per arm and Down format. Before each cell: deterministic power-of-two'
              'calibration32..65536rounds per worker until>=200ms,excluded from speed. Three batches'
              'each>=100ms;timers<0.1%wall. Threads/affinity/reference/reset/checks outside timed region.'
              'Start/drain waits remain inside. Arms may use different calibrated counts;compare seconds/job.',
        setup='Record first raw allocation/read separately(page-cache state unknown),then candidate-only'
              'Cache construction,allocation,encoding and freeze. Keep raw+view bytes and observed peak RSS.'
              'For each positive cell gain compute encoding/(reference_seconds_per_job-candidate_seconds_per_job);'
              'this is a synthetic amortization estimate,not an observed model reuse count.',
        acceptance='Both format checks:128bitwise full-GU cases and81920actual paired rows;quiet normal0'
              'closed owners. Timing requires all4cells/3eligible batches per process,finite outputs and'
              'exact final reference bits. Judge each(thread,pattern,format)cell independently over the'
              '3adjacent R/Cpairs:positive gain in all3pairs permits that synthetic cell only.'
              'Stream regressions forbid blanket caching. No model trial until actual reuse/selection'
              'and bounded RSS/prefill-independent production bridge are established. No minimum gain threshold.',
        stopping='Single-use build/check/measure-N;stop on missing source,archive,ELF,library,dispatch,'
              'numeric,owner,budget or batch eligibility. Preserve failure;no hash refresh or retry.',
        budgets=dict(build_wall=120, check_wall=45, measure_wall=120, rss=3 << 30,
                     output=256 << 10, total=128 << 20, cpu=720),
        retention='Reuse current V3CPUarchives;one new timing ELF alongside numerical fallback for'
              'this immediate comparison. Preserve compact individual speed/setup samples,source/recipe.'
              'Retire routine output and ended ownership ELF after exact recovery;legacy engine holds remain.')
    ROOT.mkdir(mode=0o700); (ROOT / 'tmp').mkdir(); write(PLAN, plan)
    print('Prepared source-pinned hot/cold plan;commit before build/check/measure.', flush=True)


def verify(plan):
    need(hot.git('show', 'HEAD:' + str(PLAN.relative_to(P))) == PLAN.read_bytes(), 'commit exact timing plan')
    for name, digest in plan['sources'].items():
        raw = hot.git('show', plan['source_commit'] + ':' + name)
        need(hashlib.sha256(raw).hexdigest() == digest == sha(P / name), 'timing input drift: ' + name)
    need(sha(hot.CXX) == plan['compiler_sha256'], 'compiler drift')
    need(sha(prior.PLAN) == plan['numeric_plan_sha256'] and
         sha(prior.ROOT / 'check.json') == plan['numeric_check_sha256'], 'numeric evidence drift')
    numeric_gate()
    need(all(sha(p) == h for p, h in plan['archives'].items()), 'archive drift')
    headers = std_headers()
    need(len(headers) == plan['standard_headers']['count'] and
         signature(headers) == plan['standard_headers']['sha256'], 'standard headers drift')


def dependencies(plan):
    original = json.loads(producer.PLAN.read_text())
    census = json.loads(Path(original['compiler_inputs']['path']).read_text())
    known = dict(census['compiled']) | dict(census['links']) | std_headers()
    known.update({str(P / p): h for p, h in original['sources'].items()})
    known.update({str(P / p): h for p, h in plan['sources'].items()})
    known.update({str(Path(original['GGML']['root']) / p): h for p, h in original['GGML']['source_sha256'].items()})
    known = {str(Path(p).resolve()): h for p, h in known.items()}
    text = (ROOT / 'timing.d').read_text().replace('\\\n', '')
    paths = [Path(p).resolve() for p in shlex.split(text.split(':', 1)[1])]
    need(0 < len(paths) < 4096, 'dependency count')
    actual = {}
    for path in paths:
        name = str(path)
        need(name in known and sha(path) == known[name], 'uncovered compiler input: ' + name)
        actual[name] = known[name]
    return actual


def built(plan):
    value = json.loads((ROOT / 'build.json').read_text())
    need(value['passed'] and value['ownership_closed'] and not value['active']
         and value['plan_sha256'] == sha(PLAN) and value['dependencies'] == dependencies(plan)
         and sha(value['binary']['path']) == value['binary']['sha256'], 'qualified timing executable required')
    return value


def execute(action):
    with (ROOT / (action + '-attempt.json')).open('x') as stream:
        json.dump(dict(action=action, no_retry=True), stream)
    state = dict(stage=action, active=True, passed=False, complete=False, GPU_executed=False, commands=[])
    def persist(): write(ROOT / (action + '.json'), state)
    persist()
    try:
        plan = json.loads(PLAN.read_text()); verify(plan)
        scope = hot.owner.Owner(ROOT / 'owned', persist); state['commands'] = scope.commands
        state.update(source_commit=plan['source_commit'], plan_sha256=sha(PLAN))
        def run(label, command, wall, env):
            entry, out, err = scope.run(action + '-' + label,
                [hot.PRLIMIT, '--as=8589934592', '--', *command], env, P,
                wall=wall, text_cap=plan['budgets']['output'], rss_cap=plan['budgets']['rss'],
                total_cap=plan['budgets']['total'], cpu=plan['budgets']['cpu'], file_cap=plan['budgets']['total'])
            need(hot.owner.closed(entry) and entry['exit_code'] == 0, 'failed/unclosed ' + label)
            return out.read_text(), err.read_text()
        if action == 'build':
            _, error = run('compile', plan['compile'], plan['budgets']['build_wall'], plan['environment'])
            need(not error, 'compile stderr')
            deps = dependencies(plan)
            text, error = run('link', plan['link'], plan['budgets']['build_wall'], plan['environment'])
            need(not error, 'link stderr')
            original = json.loads(producer.PLAN.read_text())
            external = json.loads(Path(original['external_inputs']['path']).read_text())
            allowed = {str(Path(p).resolve()): h for p, h in external.items()}
            allowed.update({str(Path(p).resolve()): h for p, h in plan['archives'].items()})
            allowed[str(ROOT / 'timing.o')] = sha(ROOT / 'timing.o')
            links = {}
            for line in text.splitlines():
                path = Path(line.strip())
                if not path.is_absolute(): continue
                name = str(path.resolve())
                need(name in allowed and sha(path) == allowed[name], 'uncovered link input: ' + name)
                links[name] = allowed[name]
            need(set(plan['archives']) <= set(links) and len(links) >= 20, 'incomplete archive/link census')
            text, error = run('libraries', [hot.READELF, '--dynamic', str(ROOT / 'timing')], 10, plan['environment'])
            libraries = re.findall(r'Shared library: \[([^\]]+)\]', text)
            need(not error and set(libraries) == {'libm.so.6', 'libstdc++.so.6', 'libgcc_s.so.1', 'libc.so.6'}, 'library census')
            state.update(binary=dict(path=str(ROOT / 'timing'), sha256=sha(ROOT / 'timing')),
                         dependencies=deps, links=links, libraries=libraries)
        else:
            build = built(plan); results = []
            arm = 'check' if action == 'check' else plan['order'][int(action.split('-')[1])]
            if action != 'check':
                checked = json.loads((ROOT / 'check.json').read_text())
                need(checked['passed'] and checked['ownership_closed'] and not checked['active']
                     and checked['binary'] == build['binary'] and checked['plan_sha256'] == sha(PLAN), 'check exact binary first')
            for row in plan['formats']:
                command = [build['binary']['path'], row['shard'], str(row['down']),
                           *map(str, row['offsets']), str(row['blob_bytes']), arm]
                text, error = run('down' + str(row['down']), command,
                    plan['budgets']['check_wall' if action == 'check' else 'measure_wall'],
                    plan['check_environment'] if action == 'check' else plan['native_environment'])
                need(not error, 'native stderr'); result = json.loads(text); need(result['passed'], 'native check')
                if action == 'check':
                    need(result == dict(passed=True, cases=128, raw_pair_rows=81920), 'numeric/actual pair differs')
                else:
                    need(result['arm'] == arm and result['down'] == row['down'] and len(result['cells']) == 4, 'arm/shape')
                    for cell in result['cells']:
                        need(len(cell['samples']) == 3 and all(s['wall_s'] >= .1 and s['jobs'] ==
                            cell['rounds_per_thread'] * cell['threads'] for s in cell['samples']), 'batch eligibility')
                        cell['median_seconds_per_job'] = statistics.median(s['wall_s'] / s['jobs'] for s in cell['samples'])
                results.append(result)
            state.update(binary=build['binary'], results=results)
        verify(plan); state.update(passed=True, complete=True, ownership_closed=True)
    except BaseException as error:
        state.update(error=repr(error), ownership_closed=all(hot.owner.closed(c) for c in state['commands'])); raise
    finally:
        state['active'] = False; persist()


if __name__ == '__main__':
    need(sys.executable == hot.PYTHON, 'qualified Python required')
    need(len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'build', 'check'] +
         ['measure-' + str(i) for i in range(6)], 'prepare|build|check|measure-0..5')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); hot.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
