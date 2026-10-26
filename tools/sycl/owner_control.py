#!/usr/bin/env python3
"""Main-owned, CPU-only supervisor comparison; repository LICENSE.

Default Owner and the GPU runtime are unchanged. No inference speed claim.
"""
import ast
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import select
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
import unittest.mock

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'tools/sycl'))
import measurement_owner as baseline
import owner_candidate as candidate
import measurement_guards as guards

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'owner-scan-control-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/prefill-runtime-comparison-current.json'
KEY = 'owner_scan_control_v1'
SOURCES = ('tools/sycl/measurement_owner.py', 'tools/sycl/owner_candidate.py',
    'tools/sycl/owner_control.py', 'tools/test_sycl_owner_candidate.py',
    'tools/sycl/measurement_guards.py', 'bench/sycl/owner_fixture.py')
ORDER = ('baseline', 'candidate', 'candidate', 'baseline', 'baseline', 'candidate')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
require = guards.require


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def git(*args): return subprocess.check_output(['/usr/bin/git', *args], cwd=P)
def write(path, value): path.write_text(json.dumps(value, indent=2) + '\n')


def sources(revision):
    out = {}
    for name in SOURCES:
        data = git('show', revision + ':' + name)
        require((P / name).read_bytes() == data, 'commit exact supervisor input: ' + name)
        out[name] = hashlib.sha256(data).hexdigest()
    return out


def providers():
    paths = {str(Path(sys.executable).resolve())}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).is_file() and not Path(filename).resolve().is_relative_to(P):
            paths.add(str(Path(filename).resolve()))
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and 'x' in parts[1] and parts[5].startswith('/'):
            require(not parts[5].endswith(' (deleted)'), 'deleted executable provider')
            paths.add(str(Path(parts[5]).resolve()))
    return {name: sha(name) for name in sorted(paths)}


def save(row):
    raw = RESULT.read_text(); original = json.loads(raw)
    literal = json.dumps(KEY) + ':'; value = json.dumps(row, indent=2).replace('\n', '\n    ')
    if KEY in original['historical_regression_audit']:
        require(raw.count(literal) == 1, 'ambiguous supervisor scope')
        start = raw.index(literal) + len(literal)
        while raw[start].isspace(): start += 1
        _, count = json.JSONDecoder().raw_decode(raw[start:]); raw = raw[:start] + value + raw[start + count:]
    else:
        anchor = '"historical_regression_audit": {'
        require(raw.count(anchor) == 1, 'ambiguous audit scope')
        raw = raw.replace(anchor, anchor + '\n    ' + literal + ' ' + value + ',', 1)
    original['historical_regression_audit'][KEY] = row
    require(json.loads(raw) == original, 'supervisor update changed unrelated evidence')
    write(ROOT / 'record.json', row); RESULT.write_text(raw)


def current(row):
    committed = json.loads(git('show', 'HEAD:' + str(RESULT.relative_to(P))))['historical_regression_audit'][KEY]
    # Plan-only scoped admission deliberately excludes completed output samples.
    require(committed['plan'] == row['plan'], 'commit immutable supervisor plan')
    plan = row['plan']
    require(sources(plan['source_commit']) == plan['sources'] and
        providers() == json.loads((ROOT / 'providers.json').read_text()) and
        sha(ROOT / 'providers.json') == plan['providers_sha256'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == plan['boot'], 'host input/provider/boot changed')


def prepare():
    require(not ROOT.exists(), 'single-use supervisor comparison')
    require(KEY not in json.loads(RESULT.read_text())['historical_regression_audit'], 'preserve previous comparison')
    revision = git('rev-parse', 'HEAD').decode().strip(); pins = sources(revision)
    ROOT.mkdir(mode=0o700); write(ROOT / 'providers.json', providers())
    row = dict(stage='prepared', plan=dict(source_commit=revision, sources=pins,
        providers_sha256=sha(ROOT / 'providers.json'), python=str(Path(sys.executable).resolve()), environment=ENV,
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), order=list(ORDER),
        question='Does direct text/proc-name I/O reduce observed supervisor CPU while preserving250ms reconciliation and ownership?',
        mechanism='Global identity/path allocation dominates measured scan CPU. Sparse owned-identity reuse alone cannot remove most unrelated-PID parsing. Candidate removes Path allocation;new registration accounts only fresh accepted identity and rejects newly dead nonleaders. No cache across scans or cadence reduction.',
        correctness='9source-pinned controls before timing:parser/missing/reuse/disappearance/zombie/freshRSS/knownescape/caps plus normal,escaped32MiBRSS and owned-only cleanup native processes. Stop first failure. No GPU or native rebuild.',
        timing='Six fresh controller processes in baseline/candidate/candidate/baseline/baseline/candidate order. Each times full Owner.run of same10tick/10s CPU-only fixture,including native startup,250ms scans,EOF,closure/reap and final log hashes. Record controller CPU and BOOTTIME;reject suspend divergence>50ms. No profiler in speed processes.',
        gate='Every normal owner/provider/protocol/closure check passes;all3 candidate/baseline CPU-fraction ratios<=0.8 and geometric mean<=0.75;every candidate controller fraction<=0.05. No speed-based stop/retry. A failed gate forbids model advancement. Passing permits a separate matched model plan only;no historical-cause/inference claim.',
        budgets=dict(wall=18, cpu=3, rss_cap=64 << 20, text_cap=1 << 20, file_cap=1 << 20, total_cap=8 << 20),
        retention='Keep9control outcome and6individual CPU/wall/fraction samples,source/provider/environment and decision. Retire normal fixture/process logs and candidate duplication after decision;no model or engine generation.'),
        tests=None, processes=[], complete=False, passed=False, active=False, ownership_closed=True, GPU_executed=False)
    save(row); print('Prepared supervisor host tests and6process CPU comparison;commit before execution.', flush=True)


def tests(row):
    current(row); require(row['stage'] == 'prepared' and row['tests'] is None, 'single-use correctness gate')
    root = ROOT / 'tests'; root.mkdir(); env = dict(ENV, STRATA_OWNER_TEST_ROOT=str(root))
    command = [row['plan']['python'], '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(ROOT / 'no-cache'), str(P / 'tools/test_sycl_owner_candidate.py')]
    error = None; code = None
    try:
        with (root / 'stdout').open('xb') as out, (root / 'stderr').open('xb') as err:
            result = subprocess.run(command, cwd=P, env=env, stdout=out, stderr=err, timeout=45)
            code = result.returncode
    except BaseException as exception: error = repr(exception)
    text = (root / 'stderr').read_text()
    receipts = [json.loads(p.read_text()) for p in root.glob('*/execution.json')]
    closed = error is None and all(not r['active'] and all(c['direct_child_reaped'] and c['session_empty'] and
        not c.get('survivors') for c in r['commands']) for r in receipts)
    accepted = code == 0 and closed and 'Ran 9 tests' in text and text.rstrip().endswith('OK')
    row['tests'] = dict(passed=accepted, exit_code=code, cases=9 if accepted else None, error=error,
        ownership_closed=closed,
        command=command, output_bytes=(root / 'stdout').stat().st_size + (root / 'stderr').stat().st_size)
    row['stage'] = 'tests-passed' if accepted else 'failed'; row['ownership_closed'] = closed; save(row)
    print(json.dumps(row['tests']), flush=True); return int(not accepted)


def run_one(row):
    current(row); index = len(row['processes'])
    require(row['tests']['passed'] and row['stage'] in ('tests-passed', 'running') and index < 6 and
        all(p['passed'] for p in row['processes']), 'ordered run after host correctness only')
    arm = ORDER[index]; module = baseline if arm == 'baseline' else candidate
    root = ROOT / str(index); root.mkdir(); owner = None; result = dict(index=index, arm=arm, passed=False)
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def awake():
        now = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
        delta = [a-b for a,b in zip(now, initial)]
        require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
        return delta[0]
    def persist(): write(root / 'execution.json', dict(active=bool(owner and owner.active), commands=owner.commands if owner else []))
    def interact(io, entry):
        poll = io.poll
        def checked_poll(): require(awake() < 18000000000, 'absolute host deadline'); poll()
        io.poll = checked_poll
        require(io.line(3) == 'ready', 'fixture ready protocol')
        for tick in range(1, 11): require(io.line(3) == 'tick=' + str(tick), 'fixture tick protocol')
        io.proc.stdin.close(); require(io.line(3) == 'done', 'fixture EOF protocol'); io.exit(3)
    try:
        owner = module.Owner(root / 'owned', persist)
        population = len(os.listdir('/proc')); cpu = time.process_time_ns(); start = awake()
        entry, _, err = owner.run('wait', [row['plan']['python'], '-I', '-S', '-B', str(P / 'bench/sycl/owner_fixture.py'), 'wait'],
            ENV, P, interaction=interact, **row['plan']['budgets'])
        elapsed = awake() - start; consumed = time.process_time_ns() - cpu
        require(module.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size and elapsed >= 9000000000, 'owner fixture incomplete')
        current(row)
        result.update(passed=True, elapsed_ns=elapsed, controller_cpu_ns=consumed, cpu_fraction=consumed / elapsed,
            proc_entries_before=population, native_pid=entry['pid'], module_sha256=sha(module.__file__))
    except BaseException as error: result['error'] = repr(error)
    finally:
        result['ownership_closed'] = bool(owner and owner.active is None and all(module.closed(c) for c in owner.commands))
        result['passed'] = bool(result['passed'] and result['ownership_closed']); persist()
        row['processes'].append(result); row['stage'] = 'running' if result['passed'] else 'failed'
        row['ownership_closed'] = result['ownership_closed']
        if len(row['processes']) == 6 and all(p['passed'] for p in row['processes']):
            pairs = []
            for offset in (0, 2, 4):
                group = {p['arm']: p['cpu_fraction'] for p in row['processes'][offset:offset+2]}
                pairs.append(group['candidate'] / group['baseline'])
            ratio = statistics.geometric_mean(pairs)
            row.update(stage='closed', complete=True, passed=True,
                analysis=dict(candidate_over_baseline=pairs, geometric_mean=ratio,
                    host_gate_passed=all(v <= .8 for v in pairs) and ratio <= .75 and
                        all(p['cpu_fraction'] <= .05 for p in row['processes'] if p['arm'] == 'candidate'),
                    inference_adoption=False, historical_cause_established=False))
        save(row)
    print(json.dumps(result), flush=True); return int(not result['passed'])


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare(); return 0
        row = json.loads((ROOT / 'record.json').read_text())
        if sys.argv[1:] == ['tests']: return tests(row)
        if sys.argv[1:] == ['run-one']: return run_one(row)
        raise RuntimeError('use prepare,tests,run-one')


if __name__ == '__main__': sys.exit(main())
