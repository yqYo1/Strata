#!/usr/bin/env python3
"""Main-owned bounded CPU-only qualification of the decode CPU observer.

Use the prospectively committed child source and recipe. This builds no engine,
performs no GPU operation and never retries an existing output directory.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
INDEX = P / 'bench/results/2026-10-15-pr6-recovery-audit/research-index.json'
BASELINE = INDEX.with_name('decode-long-generation-current.json')


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def committed(path):
    raw = path.read_bytes()
    guards.require(raw == git(P, 'show', 'HEAD:' + str(path.relative_to(P))),
                   'commit exact qualification inputs first: ' + str(path))
    return json.loads(raw)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run():
    committed_source = git(P, 'show', 'HEAD:' + str(Path(__file__).relative_to(P)))
    guards.require(Path(__file__).read_bytes() == committed_source, 'commit controller first')
    index = committed(INDEX)
    baseline = committed(BASELINE)
    summary = baseline.get('independent_process_summary', {})
    guards.require(baseline['complete'] and baseline['passed'] and baseline['ownership_closed'] and
                   not baseline['active'] and summary.get('original_benchmark_inputs_recovered') == 114,
                   'close the frozen generation cohort and reducer first')
    scope = index['current_decode_mtp_verifier_reuse_source_audit']['cpu_observer_source_audit']
    plan = scope['cpu_only_qualification_plan']
    child = Path(scope['revised_candidate']['source_root'])
    guards.require(git(child, 'rev-parse', 'HEAD').decode().strip() == plan['source_commit'] and
                   not git(child, 'status', '--porcelain'), 'qualification child differs from pinned source')
    for arg in plan['configure']:
        if arg.startswith(('-DCMAKE_C_COMPILER=', '-DCMAKE_CXX_COMPILER=')):
            guards.require(digest(arg.split('=', 1)[1]) == plan['compiler_sha256'], 'compiler changed')
    ggml = Path(next(a.split('=', 1)[1] for a in plan['configure'] if a.startswith('-DSTRATA_GGML_DIR=')))
    guards.require(git(ggml, 'rev-parse', 'HEAD').decode().strip() == plan['ggml_commit'] and
                   not git(ggml, 'status', '--porcelain', '--untracked-files=no'), 'ggml inputs changed')
    for name in ('cmake', 'ninja'):
        guards.require(subprocess.check_output([plan[name + '_path'], '--version'], text=True).splitlines()[0]
                       == plan[name + '_version'], name + ' version changed')
    root = Path(plan['configure'][plan['configure'].index('-B') + 1])
    guards.require(root.parent == B and not root.exists(), 'never overwrite or retry a qualification root')
    guards.no_model_owners([])
    root.mkdir(mode=0o700)
    (root / 'tmp').mkdir(mode=0o700)
    compiler = Path(next(a.split('=', 1)[1] for a in plan['configure'] if a.startswith('-DCMAKE_CXX_COMPILER=')))
    environment = dict(PATH=str(compiler.parent) + ':/usr/bin:/bin', HOME=os.environ['HOME'],
                       TMPDIR=str(root / 'tmp'), LC_ALL='C.UTF-8', LANG='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
    for key in plan['environment_unset']:
        guards.require(key not in environment, 'observer arm must be selected by CTest before exec')
    record = dict(source_commit=plan['source_commit'], controller_source_commit=git(P, 'rev-parse', 'HEAD').decode().strip(),
                  plan_commit=git(P, 'log', '-1', '--format=%H', '--', str(INDEX.relative_to(P))).decode().strip(),
                  environment=environment, plan=plan, complete=False, passed=False, active=False)
    owner = None

    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (root / 'record.json').write_text(json.dumps(dict(record, commands=owner.commands if owner else []), indent=2) + '\n')

    owner = supervision.Owner(root / 'owned', persist)
    try:
        budget = plan['budgets']
        for phase in ('configure', 'build', 'test'):
            guards.no_model_owners([])
            entry, _, _ = owner.run(phase, plan[phase], environment, P,
                wall=budget[phase + '_wall_seconds'], cpu=budget['cpu_seconds'],
                text_cap=budget['raw_log_bytes'], rss_cap=budget['rss_bytes'],
                file_cap=budget['file_bytes'], total_cap=budget['total_build_bytes'])
            guards.require(supervision.closed(entry) and entry.get('exit_code') == 0, phase + ' failed; inspect original owned logs')
        log = root / 'Testing/Temporary/LastTest.log'
        text = log.read_text()
        names = re.findall(r'^\d+/\d+ Testing: (\S+)\s*$', text, re.M)
        expected = {'cpu_clock_off', 'cpu_clock_on', 'pool_cpu_clock_off', 'pool_cpu_clock_on'}
        guards.require(len(names) == 4 and set(names) == expected and text.count('Test Passed.') == 4,
                       'four exact successful CPU tests required')
        record['tests'] = names
        record['test_evidence'] = [s for s in text.splitlines() if s.startswith(('cpu_clock:', 'pool_cpu_clock:'))]
        record['last_test_log_sha256'] = digest(log)
        record['binaries'] = {n: dict(sha256=digest(root / n), bytes=(root / n).stat().st_size)
                              for n in ('cpu_clock_test', 'pool_cpu_clock_test')}
        record['passed'] = True
    except BaseException as error:
        record['error'] = str(error)
    finally:
        record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        record['complete'] = record['ownership_closed']
        record['passed'] = bool(record['passed'] and record['ownership_closed'])
        persist()
        compact = {k: record[k] for k in ('source_commit', 'controller_source_commit', 'plan_commit',
                   'complete', 'passed', 'ownership_closed')}
        compact['phases'] = [{k: entry.get(k) for k in ('label', 'exit_code', 'elapsed_seconds', 'error')}
                             for entry in owner.commands]
        for key in ('tests', 'test_evidence', 'last_test_log_sha256', 'binaries', 'error'):
            if key in record:
                compact[key] = record[key]
        compact['GPU_executed'] = False
        compact['limits'] = 'CPU observer accounting/output checks only; no IQ-kernel, GPU/runtime, overhead, model throughput, quality or physical-context qualification.'
        scope['cpu_only_qualification'] = compact
        scope['revised_candidate']['built'] = any(c['label'] == 'build' and c.get('exit_code') == 0 for c in owner.commands)
        scope['revised_candidate']['tested'] = any(c['label'] == 'test' for c in owner.commands)
        INDEX.write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(compact))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.exit(run())
