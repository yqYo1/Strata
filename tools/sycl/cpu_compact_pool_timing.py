#!/usr/bin/env python3
"""Pinned whole-pool CPU comparison using the coherently built pool archives.

Main-authored; repository LICENSE. No old CPU generation, GPU work or model
adoption. Each action is single-use under Main's shared measurement lock.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import cpu_compact_pool as pool

hot, inputs = pool.hot, pool.inputs
P, B = pool.P, pool.B
ROOT = B / 'cpu-compact-production-pool-timing-v1'
PLAN = P / 'bench/sycl/cpu_compact_hot/pool-timing-plan-v1.json'
THIS = 'tools/sycl/cpu_compact_pool_timing.py'
SOURCE = 'bench/sycl/cpu_compact_hot/pool_timing.cpp'
RESULT = 'bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-current.json'
KEY = 'compact_production_pool_v1'
ORDER = ['reference', 'compact', 'compact', 'reference', 'reference', 'compact']
AR = '/usr/bin/ar'
LIBC_HEADERS = ('/usr/include/x86_64-linux-gnu/sys/resource.h',
    '/usr/include/x86_64-linux-gnu/bits/resource.h',
    '/usr/include/x86_64-linux-gnu/bits/types/struct_rusage.h')
need, sha, write = hot.need, hot.sha, hot.write


def archive_proof(build):
    """Match every archive member byte-for-byte to the original compiled object.

    Archive names/hashes alone do not prove qualified library inputs. Duplicate
    basenames in ggml are disambiguated by original object size and digest.
    """
    root = pool.ROOT / 'build'
    objects = build['actual_inputs']['objects']
    remaining = {p: h for p, h in objects.items() if not p.startswith('CMakeFiles/compact_cache_pool.dir/')}
    matched, digests, counts = [], {}, {}
    for name in ('libcompact_reference_cpu.a', 'ggml/src/libggml-cpu.a', 'ggml/src/libggml-base.a'):
        path = root / name
        need(0 < path.stat().st_size < 128 << 20, 'bounded qualified archive')
        members = subprocess.check_output([AR, 't', str(path)], text=True, timeout=20).splitlines()
        payload = subprocess.check_output([AR, 'p', str(path)], timeout=20)
        need(0 < len(members) <= 35 and len(payload) < 128 << 20, 'archive member budget')
        offset = 0
        for index, member in enumerate(members):
            found = []
            for p, digest in remaining.items():
                if Path(p).name != member: continue
                size = (root / p).stat().st_size
                if hashlib.sha256(payload[offset:offset + size]).hexdigest() == digest: found.append((p, size))
            need(len(found) == 1, 'archive member does not match one qualified object: ' + member)
            p, size = found[0]; offset += size
            matched.append((name, index, p, remaining.pop(p)))
        need(offset == len(payload), 'archive payload coverage')
        digests[str(path)] = sha(path); counts[name] = len(members)
    need(not remaining and len(matched) == 35, 'all35CPU/GGML objects must match reused archives')
    return dict(archives=digests, counts=counts, member_object_map_sha256=pool.signature(matched),
                tool=dict(path=AR, sha256=sha(AR)))


def numerical():
    original = json.loads(pool.PLAN.read_text()); pool.verify(original)
    build = json.loads((pool.ROOT / 'build.json').read_text())
    check = json.loads((pool.ROOT / 'check.json').read_text())
    for value in (build, check):
        need(value['passed'] and value['complete'] and value['ownership_closed'] and not value['active']
            and value['source_commit'] == original['source_commit']
            and value['plan_sha256'] == sha(pool.PLAN)
            and all(hot.owner.closed(c) for c in value['commands']), 'closed whole-pool evidence required')
    need(build['actual_inputs'] == pool.dependencies(original) and check['binary'] == build['binary']
         and sha(build['binary']['path']) == build['binary']['sha256']
         and check['pool_numerical_qualified'] and not check['performance_qualified'], 'exact numeric build')
    return original, build, check


def expected_numeric():
    cases = rows = selected = 0
    for workers in (1, 5):
        for n in (1, 6, 32, 97, 193):
            for policy in (0, 1, 2):
                for decode in (True, False):
                    for misses in (False, True):
                        r = cases + 11
                        for j in range(n):
                            nt = 1 if policy == 0 else 2 if policy == 1 else 1 + (j & 1)
                            if nt == 1:
                                rows += 640
                                if decode and (not misses or (j + j // 17 + r) % 5 != 0): selected += 640
                        cases += 1
    return dict(passed=True, cases=cases, nt1_rows=rows, compact_rows=selected, offset_control_detected=True)


def evidence(revision):
    raw = hot.git('show', revision + ':' + RESULT)
    node = json.loads(raw)[KEY]
    need(node == json.loads((P / RESULT).read_text())[KEY] and node['pool_numerical_qualified']
         and not node['model_adopted'], 'scoped committed numerical evidence differs')
    return pool.signature(node)


def prepare():
    need(not ROOT.exists() and not PLAN.exists(), 'fresh timing plan/root required')
    old, build, check = numerical()
    revision = hot.git('rev-parse', 'HEAD').decode().strip()
    sources = {}
    for name in (THIS, SOURCE, 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md'):
        raw = hot.git('show', revision + ':' + name)
        need(raw == (P / name).read_bytes(), 'commit source first: ' + name)
        sources[name] = hashlib.sha256(raw).hexdigest()
    proof = archive_proof(build); archives = proof['archives']
    flags = ['-Wno-error=deprecated-declarations', '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
        '-O3', '-DNDEBUG', '-std=gnu++20', '-march=znver3', '-mtune=znver3',
        '-DSTRATA_NATIVE_EXPERTS=1', '-DSTRATA_CPU_SERVICE_PROFILE=1']
    ggml = Path(old['GGML']['root'])
    includes = ['-I' + str(p) for p in (P / 'include', ggml / 'ggml/src', ggml / 'ggml/include')]
    environment = dict(old['environment'], TMPDIR=str(ROOT / 'tmp'))
    native = dict(old['native_environment'], TMPDIR=str(ROOT / 'tmp'), STRATA_IQ3_PAIR_AUDIT='0')
    plan = dict(schema='compact-production-pool-timing-v1', source_commit=revision, sources=sources,
        numeric_plan=dict(path=str(pool.PLAN), sha256=sha(pool.PLAN)),
        numeric_records={str(pool.ROOT / (name + '.json')): sha(pool.ROOT / (name + '.json')) for name in ('build', 'check')},
        numeric_evidence=dict(commit=revision, path=RESULT, key=KEY, canonical_sha256=evidence(revision)),
        archives=archives, archive_source_proof=proof,
        operating_system_headers={p: sha(p) for p in LIBC_HEADERS},
        compile=[hot.CXX, *flags, *includes, '-MD', '-MF', str(ROOT / 'timing.d'), '-c', str(P / SOURCE),
                 '-o', str(ROOT / 'timing.o')],
        link=[hot.CXX, '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13', '-O3', '-DNDEBUG', '-Wl,-t',
              str(ROOT / 'timing.o'), *archives, '-lm', '-o', str(ROOT / 'timing')],
        environment=environment, native_environment=native,
        check_environment=dict(native, STRATA_IQ3_PAIR_AUDIT='1'), formats=old['formats'], order=ORDER,
        expected_numeric=expected_numeric(), topology=old['topology'],
        scope='Actual unchanged ExpertPool::run_split_multi_native DecodeNT1entry;both arms prepare job'
          'coordinates/activations,compact additionally performs Cache.lookup/Lease assignment every job/call.'
          'All GU,row scheduling,host hidden quantization,Down,publication/waits and normal stage clocks are timed.'
          '1worker onCPU1 with host CPU0not draining;or5workers onCPU1..5 plus host CPU0draining.'
          'Controller admits six unique physical cores;child checks actual worker TIDs/singleton affinities'
          'before/after each batch. Full allowed affinity is restored between pool constructions.',
        workloads='Both actual Down20/42 slices;32encoded experts plus retained raw weights,4native Q8_K activations.'
          'Timed jobs/call1,6,32;reuse fixes keys per call shape,stream32 rotates a permutation of the32experts.'
          'Both synthetic patterns can reuse CPUcache;neither is controlled cold DRAM or observed model reuse.'
          'NT1all-hit timing only. Untimed numeric checks include NT1/2/mixed,Prefill,misses and97/193jobs.'
          'key(j)=13*j+7*floor(j/32)+13*round+3 mod32;misses use(j+floor(j/17)+round)mod5.'
          'Deliberately wrong96job-offset leases must produce a guarded output difference in both pool geometries.',
        timing='Separate native process per format/arm,in R,C,C,R,R,C order. For each of12cells,calibrate'
          '1..65536calls by powers of2until>=200ms;then3batches each>=100ms and timer overhead<0.1%wall.'
          'Compare wall seconds/expert job despite independent calibrated call counts. Preparation,oracle,'
          'pool construction,affinity checks,output/guard checks and resets are outside timed region.'
          'Coordinates/lookup,normal pool clocks and synchronous waits remain inside.',
        acceptance='New nonperiodic120case checks per format must exactly match expected counters/bits and'
          'detect the wrong-offset control;all owners normal0,quiet and closed. All12timing cells need3'
          'eligible finite samples per process. Judge each format/worker/jobs/pattern over3adjacent R/Cpairs:'
          'all3positive gains qualify only that synthetic cell. An actual-model trial requires all6tested'
          'five-workers+host cells to win in both formats and all3pairs,plus separate source/safety admission.'
          'One-worker cells remain independent. No minimum percentage. No model/adoption/70TGclaim.',
        accounting='Existing stage clocks partition GU,quantization and Down critical-path wall within each'
          'pool call;their sum must not exceed outer batch wall. They are not pure arithmetic or resource'
          'saturation. Process CPU time includes work,polling and submission. Setup RSS precedes oracle/workers;'
          'final ru_maxrss is separately reported. Requested blob/view sizes are not physical memory traffic.',
        stopping='Single-use build/check/measure-N. Stop on missing/changed input,evidence,archive,ELF,library,'
          'owner,numeric,counter,topology,budget or timing eligibility. Preserve original failure;no hash refresh/retry.',
        budgets=dict(build_wall=120, check_wall=120, measure_wall=180, rss=3 << 30,
                     output=256 << 10, total=192 << 20, cpu=900),
        retention='Reuse fresh qualified CPU/GGMLarchives and preserve coherent GPUfallback unchanged.'
          'Keep this new timing ELF only for imminent pool comparison/model-bridge admission;compact samples'
          'and source/recipe go to Git. Retire ended old CPUgeneration/child after exact source recovery and'
          'removing path-bound readers;old GPUfixture/deployment provenance holds stay independent.')
    ROOT.mkdir(mode=0o700); (ROOT / 'tmp').mkdir(); write(PLAN, plan)
    print('Prepared whole-pool comparison;commit plan before build/check/measure.', flush=True)


def verify(plan):
    need(hot.git('show', 'HEAD:' + str(PLAN.relative_to(P))) == PLAN.read_bytes(), 'commit exact timing plan')
    for name, digest in plan['sources'].items():
        raw = hot.git('show', plan['source_commit'] + ':' + name)
        need(hashlib.sha256(raw).hexdigest() == digest == sha(P / name), 'source drift: ' + name)
    need(sha(pool.PLAN) == plan['numeric_plan']['sha256'] and
         all(sha(p) == h for p, h in plan['numeric_records'].items()), 'numeric proof drift')
    _, build, _ = numerical()
    need(archive_proof(build) == plan['archive_source_proof'], 'qualified archive object/source proof differs')
    need(evidence(plan['numeric_evidence']['commit']) == plan['numeric_evidence']['canonical_sha256'], 'evidence drift')
    need(all(sha(p) == h for p, h in plan['archives'].items()) and
         all(sha(p) == h for p, h in plan['operating_system_headers'].items()), 'archive/header drift')
    need(pool.topology() == plan['topology'], 'physical topology drift')


def dependencies(plan):
    old = json.loads(pool.PLAN.read_text())
    census = json.loads(Path(old['compiler_inputs']['path']).read_text())
    known = dict(census['compiled']) | dict(census['links']) | pool.std_headers() | plan['operating_system_headers']
    known.update({str(P / p): h for p, h in old['sources'].items()})
    known.update({str(P / p): h for p, h in plan['sources'].items()})
    known.update({str(Path(old['GGML']['root']) / p): h for p, h in old['GGML']['source_sha256'].items()})
    known = {str(Path(p).resolve()): h for p, h in known.items()}
    text = (ROOT / 'timing.d').read_text().replace('\\\n', '')
    paths = [Path(p).resolve() for p in shlex.split(text.split(':', 1)[1])]
    need(0 < len(paths) < 4096, 'dependency count')
    actual = {}
    for path in paths:
        name = str(path); need(name in known and sha(path) == known[name], 'uncovered compiler input: ' + name)
        actual[name] = known[name]
    return actual


def built(plan):
    value = json.loads((ROOT / 'build.json').read_text())
    need(value['passed'] and value['complete'] and value['ownership_closed'] and not value['active']
         and value['plan_sha256'] == sha(PLAN) and value['dependencies'] == dependencies(plan)
         and all(sha(p) == h for p, h in value['links'].items())
         and sha(value['binary']['path']) == value['binary']['sha256'], 'exact built timing ELF required')
    return value


def execute(action):
    with (ROOT / (action + '-attempt.json')).open('x') as stream: json.dump(dict(action=action, no_retry=True), stream)
    state = dict(stage=action, active=True, passed=False, complete=False, GPU_executed=False, commands=[])
    def persist(): write(ROOT / (action + '.json'), state)
    persist()
    try:
        plan = json.loads(PLAN.read_text()); verify(plan)
        scope = hot.owner.Owner(ROOT / 'owned', persist); state['commands'] = scope.commands
        state.update(source_commit=plan['source_commit'], plan_sha256=sha(PLAN))
        def run(label, command, wall, env):
            entry, out, err = scope.run(action + '-' + label, [hot.PRLIMIT, '--as=8589934592', '--', *command],
                env, P, wall=wall, text_cap=plan['budgets']['output'], rss_cap=plan['budgets']['rss'],
                total_cap=plan['budgets']['total'], cpu=plan['budgets']['cpu'], file_cap=plan['budgets']['total'])
            need(hot.owner.closed(entry) and entry['exit_code'] == 0, 'failed/unclosed ' + label)
            return out.read_text(), err.read_text()
        if action == 'build':
            _, error = run('compile', plan['compile'], plan['budgets']['build_wall'], plan['environment'])
            need(not error, 'compile stderr'); deps = dependencies(plan)
            text, error = run('link', plan['link'], plan['budgets']['build_wall'], plan['environment'])
            need(not error, 'link stderr')
            old = json.loads(pool.PLAN.read_text())
            allowed = {str(Path(p).resolve()): h for p, h in json.loads(Path(old['external_inputs']['path']).read_text()).items()}
            allowed.update({str(Path(p).resolve()): h for p, h in plan['archives'].items()})
            allowed[str(ROOT / 'timing.o')] = sha(ROOT / 'timing.o'); links = {}
            for line in text.splitlines():
                path = Path(line.strip())
                if not path.is_absolute(): continue
                name = str(path.resolve()); need(name in allowed and sha(path) == allowed[name], 'uncovered link input: ' + name)
                links[name] = allowed[name]
            need(set(plan['archives']) <= set(links) and len(links) >= 20, 'incomplete archive/link census')
            text, error = run('libraries', [hot.READELF, '--dynamic', str(ROOT / 'timing')], 10, plan['environment'])
            libraries = re.findall(r'Shared library: \[([^\]]+)\]', text)
            need(not error and set(libraries) == {'libm.so.6', 'libstdc++.so.6', 'libgcc_s.so.1', 'libc.so.6'}, 'CPU-only library census')
            state.update(binary=dict(path=str(ROOT / 'timing'), sha256=sha(ROOT / 'timing')),
                         dependencies=deps, links=links, libraries=libraries)
        else:
            build = built(plan); results = []
            arm = 'check' if action == 'check' else plan['order'][int(action.split('-')[1])]
            if action != 'check':
                checked = json.loads((ROOT / 'check.json').read_text())
                need(checked['passed'] and checked['complete'] and checked['ownership_closed'] and not checked['active']
                    and checked['binary'] == build['binary'] and checked['plan_sha256'] == sha(PLAN)
                    and all(hot.owner.closed(c) for c in checked['commands']), 'exact numerical pass first')
            for row in plan['formats']:
                text, error = run('down' + str(row['down']), [build['binary']['path'], row['shard'], str(row['down']),
                    *map(str, row['offsets']), str(row['blob_bytes']), arm],
                    plan['budgets']['check_wall' if action == 'check' else 'measure_wall'],
                    plan['check_environment'] if action == 'check' else plan['native_environment'])
                need(not error, 'native stderr'); result = json.loads(text); need(result['passed'], 'native check')
                if action == 'check': need(result == plan['expected_numeric'], 'numerical output/dispatch/control differs')
                else:
                    need(result['arm'] == arm and result['down'] == row['down'] and len(result['cells']) == 12, 'arm/shape')
                    seen = set()
                    for cell in result['cells']:
                        k = (cell['workers'], cell['jobs_per_call'], cell['pattern']); need(k not in seen, 'duplicate cell'); seen.add(k)
                        need(cell['host_cpu'] == 0 and cell['host_works'] == (cell['workers'] == 5) and
                             cell['worker_cpus'] == list(range(1, cell['workers'] + 1)), 'actual thread affinity')
                        need(len(cell['samples']) == 3 and all(s['wall_s'] >= .1 and s['calls'] == cell['rounds']
                            and s['jobs'] == cell['rounds'] * cell['jobs_per_call'] and
                            0 <= s['pool_gu_s'] + s['pool_quant_s'] + s['pool_down_s'] <= s['wall_s'] + 1e-6
                            and result['timer_pair_s'] / s['wall_s'] < .001 for s in cell['samples']), 'batch eligibility')
                        cell['median_seconds_per_job'] = statistics.median(s['wall_s'] / s['jobs'] for s in cell['samples'])
                    need(seen == {(w, n, p) for w in (1, 5) for n in (1, 6, 32) for p in ('reuse', 'stream32')}, 'cell coverage')
                results.append(result)
            state.update(binary=build['binary'], results=results)
        verify(plan); state.update(passed=True, complete=True, ownership_closed=True)
    except BaseException as error:
        state.update(error=repr(error), ownership_closed=all(hot.owner.closed(c) for c in state['commands'])); raise
    finally:
        state['active'] = False; persist()


if __name__ == '__main__':
    need(sys.executable == hot.PYTHON and len(sys.argv) == 2 and
         sys.argv[1] in ('prepare', 'build', 'check', *(f'measure-{i}' for i in range(6))), 'qualified Python;prepare|build|check|measure-0..5')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); hot.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
