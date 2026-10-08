"""Finite same-day update comparison; only launches after the owned first gate exits."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

base = Path(__file__).parent
sys.path.insert(0, '/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/sycl/tools')
from owned_gdb import process_identity
controller = base / 'run_owned_upstream_v0141_code32k_v2.py'
controller_sha = '3cf4f54fdeaf06893f40a88b31f2b72b91b94f5d93d2894e03cff3d703de188c'
assert hashlib.sha256(controller.read_bytes()).hexdigest() == controller_sha
out = base / 'upstream-v0141-matched32k-comparison-sequence-v1'
assert not out.exists()
out.mkdir(mode=0o700)
record = {'active': True, 'passed': False,
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'Logged pure/integrated/control first gates, then quiet control/pure/integrated/integrated/pure/control. Four fresh32K A/B/A/B per process. First reads and later complete reads reported separately. Diagnostic times excluded. No adoption or physical256K claim.',
          'controller_sha256': controller_sha, 'steps': [],
          'active_stage': 'wait-owned-pure-diagnostic-r2', 'deadline_seconds': 12600}
started = time.monotonic()
child = None


def save():
    record['elapsed_seconds'] = time.monotonic() - started
    temp = out / 'record.json.tmp'
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(out / 'record.json')


def finished(path):
    data = json.loads(path.read_text())
    assert not data['active'] and data['completed'] and data['healthy'] and data['math_gate_passed']
    assert data['exit_code'] == 0 and not data['exit_signal'] and not data['new_fault_messages']
    assert not any(data['cleanup'].values()) and len(data['requests']) == 4
    assert data['source_sha256'][str(controller)] == controller_sha
    assert data['actual_executable_identity']['sha256'] == data['binary_sha256']
    for role in ['inferior', 'debugger']:
        old = data[role]
        now = process_identity(old['pid'])
        assert not now or now['start_ticks'] != old['start_ticks']
    return data


save()
try:
    first = base / 'owned-v0141-code32k-pure-diagnostic-r2/record.json'
    wait_end = time.monotonic() + 3900
    while json.loads(first.read_text())['active']:
        assert time.monotonic() < wait_end, 'initial owned gate remains active; no overlapping model'
        save()
        time.sleep(5)
    finished(first)
    record['initial_pure_gate_sha256'] = hashlib.sha256(first.read_bytes()).hexdigest()
    jobs = [('integrated', 'diagnostic', 1), ('control', 'diagnostic', 1)]
    jobs += [(mode, 'clean', rep) for mode, rep in [('control', 1), ('pure', 1), ('integrated', 1), ('integrated', 2), ('pure', 2), ('control', 2)]]
    for mode, phase, rep in jobs:
        assert time.monotonic() - started < record['deadline_seconds']
        assert hashlib.sha256(controller.read_bytes()).hexdigest() == controller_sha
        record['active_stage'] = f'{mode}-{phase}-r{rep}'
        receipt = base / f'owned-v0141-code32k-{mode}-{phase}-r{rep}/record.json'
        argv = ['/usr/bin/python3', str(controller), mode, phase, str(rep)]
        step = {'argv': argv, 'mode': mode, 'phase': phase, 'repetition': rep,
                'receipt': str(receipt), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        record['steps'].append(step)
        save()
        with (out / (record['active_stage'] + '.stdout')).open('w') as stdout, (out / (record['active_stage'] + '.stderr')).open('w') as stderr:
            child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
            step['controller_identity'] = process_identity(child.pid)
            save()
            while child.poll() is None:
                assert time.monotonic() - started < record['deadline_seconds']
                save()
                time.sleep(5)
            step['exit_code'] = child.returncode
            child = None
        assert step['exit_code'] == 0, record['active_stage']
        data = finished(receipt)
        step['receipt_sha256'] = hashlib.sha256(receipt.read_bytes()).hexdigest()
        step['finished_utc'] = data['finished_utc']
        save()
    quiet = {key: [] for key in ['control', 'pure', 'integrated']}
    for step in record['steps']:
        if step['phase'] != 'clean':
            continue
        data = finished(Path(step['receipt']))
        assert data['phase'] == 'clean' and data['performance_eligible']
        quiet[step['mode']].append(data)
    for key in quiet:
        assert len(quiet[key]) == 2
    argv_sets = {tuple(d['argv'][1:]) for jobs in quiet.values() for d in jobs}
    env_sets = {json.dumps(d['actual_target_environment'], sort_keys=True) for jobs in quiet.values() for d in jobs}
    assert len(argv_sets) == len(env_sets) == 1
    summaries = {}
    for mode, jobs in quiet.items():
        summaries[mode] = {}
        for name, first_read in [('first_process_read', True), ('later_fresh_reads', False)]:
            reads = [r for d in jobs for r in d['requests'] if r['first_process_read'] == first_read]
            count = len(reads)
            summaries[mode][name] = {
                'count': count,
                'prefill_tok_s': 1000 * sum(r['measurement']['prompt_tokens'] for r in reads) / sum(r['measurement']['prompt_ms'] for r in reads),
                'decode_tok_s': 1000 * sum(r['measurement']['generated_tokens'] for r in reads) / sum(r['measurement']['decode_ms'] for r in reads),
                'reads': [{'name': r['name'], 'measurement': r['measurement']} for r in reads],
            }
    cross = []
    for index in range(4):
        reference = quiet['control'][0]['requests'][index]
        for mode in ['pure', 'integrated']:
            actual = quiet[mode][0]['requests'][index]
            cross.append({'mode': mode, 'read_index': index,
                          'ids_equal_control': actual['ids'] == reference['ids'],
                          'logprobs_equal_control': actual['logprobs'] == reference['logprobs'],
                          'mtp_counts_equal_control': actual['mtp_counts'] == reference['mtp_counts'],
                          'note': 'Cross-version differences are reported, not silently accepted as numerical equivalence. Raw hidden-tail accepted counts and actual chunk sizing can differ. Own logged/repeated-output checks are mandatory.'})
    record['summary'] = summaries
    record['cross_version_output_comparison'] = cross
    record['paired_requested_configuration'] = True
    record['actual_geometry_note'] = 'Actual chunk sizes/cache admission are preserved separately in each project-messages.txt. Same requested geometry does not imply identical allocation decisions.'
    record['passed'] = True
except BaseException as error:
    record['error'] = repr(error)
    if child and child.poll() is None:
        # Ask only our controller to run its own bounded snapshot/cleanup.
        os.kill(child.pid, signal.SIGINT)
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            record['controller_cleanup_pending'] = process_identity(child.pid)
finally:
    record['active'] = False
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
print(json.dumps({k: record.get(k) for k in ['active', 'passed', 'elapsed_seconds', 'active_stage', 'error', 'summary', 'cross_version_output_comparison']}))
if not record['passed']:
    raise SystemExit(1)
