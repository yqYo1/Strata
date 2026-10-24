#!/usr/bin/env python3
"""Owned host collector and four-clock reference; no model/GPU qualification.

Extends the closed original host result without relabeling its source or pass.
Temporary runner: retire after this finite gate and original-input recovery.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import build_engine as build
import measurement_guards as guards
import measurement_owner as supervision

P, B = build.P, build.B
ROOT = B / 'cpu-service-host-clock-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/cpu-service-current.json'
KEY = 'four_clock_reference'
NATIVE = ('tests/core/cpu_service_profile_test.cpp',
    'include/strata/kernels/cpu/service_profile.hpp', 'include/strata/kernels/cpu/native_expert.hpp')
INPUTS = (*NATIVE, 'tools/sycl/cpu_service_profile.py', 'tools/sycl/build_engine.py',
    'tools/sycl/measurement_guards.py', 'tools/sycl/measurement_owner.py',
    'tools/sycl/cpu_service_evidence.py', 'tests/core/test_cpu_service_evidence.py',
    'tools/sycl/cpu_work_census.py', 'tests/core/test_cpu_work_census.py')
PACKAGES = ('g++-13', 'gcc-13', 'libstdc++-13-dev', 'libgcc-13-dev',
    'libc6-dev', 'linux-libc-dev', 'libstdc++6', 'libgcc-s1', 'libc6', 'binutils',
    'python3.12-minimal', 'libpython3.12-stdlib')
EXPECTED = 'cpu_service_profile counters,phase isolation,thread ownership,clocks,refusals=PASS\n'
require = guards.require


def write(path, row):
    path.write_text(json.dumps(row, indent=2) + '\n')


def fingerprint(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def pins(revision):
    result = {}
    for name, raw in build.blobs(P, revision, INPUTS):
        require(raw == (P / name).read_bytes(), 'commit exact host input: ' + name)
        result[name] = build.digest(P / name)
    return result


def packages():
    # Provider versions supplement the exact compiler/stdlib hashes. Actual
    # non-system project include closure is checked after compilation.
    return subprocess.check_output(['/usr/bin/dpkg-query', '-W',
        '-f=${binary:Package}=${Version}\n', *PACKAGES], text=True, timeout=10).splitlines()


def historical(row):
    return {k: v for k, v in row.items() if k != KEY}


def current(plan):
    raw = RESULT.read_bytes()
    require(raw == build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))), 'commit exact host plan')
    row = json.loads(raw)
    require(row[KEY] == plan == json.loads((ROOT / 'plan.json').read_text()), 'host plan changed')
    require(fingerprint(historical(row)) == plan['historical_result_sha256'], 'original host outcome changed')
    require(pins(plan['source_commit']) == plan['input_sha256'], 'host source changed')
    require(packages() == plan['provider_packages'], 'host provider package versions changed')
    for item in (plan['compiler'], plan['stdlib'], plan['python']):
        require(build.digest(item['path']) == item['sha256'], 'host provider changed')


def prepare():
    require(not ROOT.exists(), 'fresh gate only; no duplicate or retry')
    raw = RESULT.read_bytes()
    require(raw == build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))), 'original result must be committed')
    row = json.loads(raw)
    require(KEY not in row and row['passed'] and row['complete'] and not row['active'] and
        row['ownership_closed'] and row['artifact_retirement']['complete'], 'preserve the closed original host gate')
    revision = build.git(P, 'rev-parse', 'HEAD').decode().strip()
    source = pins(revision)
    compiler = '/usr/bin/g++'
    stdlib = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33'
    ROOT.mkdir(mode=0o700)
    (ROOT / 'tmp').mkdir()
    plan = dict(stage='prepared', source_commit=revision, input_sha256=source,
        historical_result_sha256=fingerprint(row), provider_packages=packages(),
        compiler=dict(path=compiler, sha256=build.digest(compiler),
            version=subprocess.check_output([compiler, '--version'], text=True, timeout=10).splitlines()[0]),
        stdlib=dict(path=stdlib, sha256=build.digest(stdlib)),
        python=dict(path='/usr/bin/python3', sha256=build.digest('/usr/bin/python3'),
            version=subprocess.check_output(['/usr/bin/python3', '--version'], text=True, timeout=10).strip()),
        command=[compiler, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic', '-pthread',
            '-DSTRATA_CPU_SERVICE_PROFILE=1', '-I', str(P / 'include'), '-MMD', '-MF', str(ROOT / 'fixture.d'),
            str(P / NATIVE[0]), '-o', str(ROOT / 'fixture')],
        environment=dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8', TMPDIR=str(ROOT / 'tmp')),
        budgets=dict(wall=90, cpu=120, rss_cap=2 << 30, text_cap=2 << 20, file_cap=8 << 20, total_cap=16 << 20),
        criteria='Owned same-request leaf/census reader controls,single compile,default collector controls,then --clock-screen;all exit0 and ownership closed. Actual project includes exactly the three pinned native files. Default counters/phase/thread/refusal controls still pass. Clock output exactly six distinct positive calling TIDs/lanes,65536 observations+512 warmups each,zero errors,ordered finite integer min/p50/p99/max wall/CPU values. No model or GPU;failure stops this gate without retry.',
        interpretation='Four ordered clock reads only,without a kernel/counter update. Affinity uncontrolled. Observed per-lane distributions are not a universal upper bound on full Scope/model cost and must not be subtracted from leaf timings. Separate production-pool parity,same-request census and model observer-off/on overhead gates remain open.',
        retention='Keep source/recipe,provider identities,per-lane compact distributions and original collector status. After committed source/recipe recovery,delete the standalone executable,successful raw output and this one-shot runner. No extra native engine generation.',
        passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, production_pool_checked=False, model_adoption_qualified=False)
    write(ROOT / 'plan.json', plan)
    row[KEY] = plan
    write(RESULT, row)
    print('Prepared host controls and clock reference; commit plan before compile/test.', flush=True)


def read_clocks(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(rows) == 6 and all(type(r.get('lane')) is int for r in rows) and
        [r['lane'] for r in rows] == list(range(6)), 'clock lane extent differs')
    tids = set()
    for row in rows:
        require(row.get('kind') == 'cpu_leaf_four_clock_reference_v1' and row.get('valid') is True and
            type(row.get('tid')) is int and row['tid'] > 0 and row['tid'] not in tids and
            type(row.get('samples')) is int and row['samples'] == 65536 and
            type(row.get('warmup')) is int and row['warmup'] == 512 and
            type(row.get('errors')) is int and row['errors'] == 0 and
            row.get('affinity') == 'uncontrolled' and
            row.get('sequence') == 'MONOTONIC,THREAD_CPU,THREAD_CPU,MONOTONIC' and
            all(row.get(k) is False for k in ('kernel', 'overhead_bound', 'subtracted')), 'invalid clock reference')
        tids.add(row['tid'])
        for kind in ('wall_ns', 'cpu_ns'):
            values = row.get(kind, {})
            require(set(values) == {'min', 'p50', 'p99', 'max'}, 'clock quantile fields differ')
            ordered = [values[k] for k in ('min', 'p50', 'p99', 'max')]
            require(all(type(v) is int and v >= 0 for v in ordered) and ordered == sorted(ordered),
                'clock quantiles invalid')
        require(all(row['cpu_ns'][k] <= row['wall_ns'][k] for k in ('min', 'p50', 'p99', 'max')),
            'clock CPU interval exceeds enclosing wall interval')
    return [dict(lane=r['lane'], samples=r['samples'], errors=r['errors'],
        wall_ns=r['wall_ns'], cpu_ns=r['cpu_ns']) for r in rows]


def host(plan):
    current(plan)
    require(plan['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use host gate')
    outcome = copy.deepcopy(plan)
    owner = None

    def persist():
        write(ROOT / 'record.json', dict(outcome, commands=owner.commands if owner else [],
            active=bool(owner and owner.active is not None)))

    try:
        owner = supervision.Owner(ROOT / 'owned', persist)
        entry, _, _ = owner.run('reader', [plan['python']['path'], '-B', 'tests/core/test_cpu_service_evidence.py'],
            plan['environment'], P, **plan['budgets'])
        require(supervision.closed(entry) and entry['exit_code'] == 0, 'CPU service reader controls failed')
        entry, _, _ = owner.run('compile', plan['command'], plan['environment'], P, **plan['budgets'])
        require(supervision.closed(entry) and entry['exit_code'] == 0, 'host compile failed')
        deps = shlex.split((ROOT / 'fixture.d').read_text().replace('\\\n', ' ').partition(':')[2])
        found = {str(Path(p).resolve().relative_to(P)) for p in deps}
        require(found == set(NATIVE), 'actual project include closure differs')
        outcome['compiled_project_inputs'] = len(found)
        fixture = ROOT / 'fixture'
        outcome['fixture'] = dict(path=str(fixture), sha256=build.digest(fixture), bytes=fixture.stat().st_size)
        entry, out, err = owner.run('check', [str(fixture)], plan['environment'], ROOT, **plan['budgets'])
        require(supervision.closed(entry) and entry['exit_code'] == 0 and out.read_text() == EXPECTED,
            'collector controls failed')
        require(err.read_text().startswith('strata CPU_SERVICE request=9 valid=0 ') and
            len(err.read_text().splitlines()) == 1, 'expected negative-delta diagnostic differs')
        entry, out, err = owner.run('clock', [str(fixture), '--clock-screen'], plan['environment'], ROOT, **plan['budgets'])
        require(supervision.closed(entry) and entry['exit_code'] == 0 and not err.read_bytes(), 'clock reference failed')
        outcome['lanes'] = read_clocks(out)
        require(build.digest(fixture) == outcome['fixture']['sha256'], 'host fixture changed')
        current(plan)
        outcome.update(stage='host-checked', passed=True, complete=True)
    except BaseException as error:
        outcome.update(stage='failed', error=repr(error))
    finally:
        outcome.update(active=False, ownership_closed=bool(owner and owner.active is None and
            all(supervision.closed(c) for c in owner.commands)),
            command_outcomes=[dict(label=c['label'], exit_code=c.get('exit_code')) for c in owner.commands] if owner else [])
        persist()
        row = json.loads(RESULT.read_text())
        row[KEY] = outcome
        write(RESULT, row)
    print(json.dumps({k: outcome[k] for k in ('stage', 'passed', 'complete', 'ownership_closed')}), flush=True)
    return int(not outcome['passed'])


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host'), 'prepare|host')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            sys.exit(host(json.loads(RESULT.read_text())[KEY]))
