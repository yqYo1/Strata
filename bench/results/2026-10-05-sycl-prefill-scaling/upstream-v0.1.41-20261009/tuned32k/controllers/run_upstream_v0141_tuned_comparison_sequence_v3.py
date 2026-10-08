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
controller = base / 'run_owned_upstream_v0141_tuned32k_v3.py'
controller_sha = '80fe9ba0ee4127ded95c0fd5a09e25f4ad754badde381253746ea022b04c11ac'
assert hashlib.sha256(controller.read_bytes()).hexdigest() == controller_sha
default_path=base/'upstream-v0141-matched32k-comparison-sequence-v1/record.json'
default=json.loads(default_path.read_text())
assert not default['active'], 'default32K sequence active; no tuned sequence'
assert default['passed']
physical_path=base/'owned-upstream-v0141-integrated-full256k-diagnostic-r2/record.json'
physical=json.loads(physical_path.read_text())
assert not physical['active'] and physical['completed'] and physical['healthy'] and physical['math_gate_passed']
assert physical['physical256k_sequence_completed'] and physical['capacity_sequence_completed']
assert physical['exit_code']==0 and not physical['exit_signal'] and not any(physical['cleanup'].values())
for role in ['inferior','debugger']:
    old=physical[role];now=process_identity(old['pid'])
    assert not now or now['start_ticks']!=old['start_ticks']
out = base / 'upstream-v0141-tuned-matched32k-comparison-sequence-v1'
assert not out.exists()
out.mkdir(mode=0o700)
record = {'active': True, 'passed': False,
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'Separate previous tuning configuration control/integrated/integrated/control,4 fresh32K A/B/A/B per process, context262144, actualchunk8192, int8/resident32768, compact2/attention-layout1/own-KV-stage1/prefetch0, PC1/ckpt1/every262139/root0/turn-token-1, MTP4,128 actual cache slots. Initial physical/head/state/disk gate required before launch. First vs later fresh reads separate; no adoption claim.',
          'controller_sha256': controller_sha, 'steps': [],
          'active_stage': 'not_started', 'deadline_seconds': 3600}
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
    info=next(value for value in data['startup'] if value.startswith('INFO '))
    assert 'expert_slots=128 ' in info and 'kv_resident=32768 ' in info
    for req in data['requests']:
        pp=[int(value.split()[1]) for value in req['protocol'] if value.startswith('PP ')]
        assert pp==[8192,16384,24576,32767], 'actual chunk differs from qualified tuning geometry'
    for role in ['inferior', 'debugger']:
        old = data[role]
        now = process_identity(old['pid'])
        assert not now or now['start_ticks'] != old['start_ticks']
    return data


save()
try:
    record['new_full_receipt_sha256']=hashlib.sha256(physical_path.read_bytes()).hexdigest()
    record['default_comparison_receipt_sha256']=hashlib.sha256(default_path.read_bytes()).hexdigest()
    jobs=[('control','clean',1),('integrated','clean',1),('integrated','clean',2),('control','clean',2)]
    for mode, phase, rep in jobs:
        assert time.monotonic() - started < record['deadline_seconds']
        assert hashlib.sha256(controller.read_bytes()).hexdigest() == controller_sha
        record['active_stage'] = f'{mode}-{phase}-r{rep}'
        receipt = base / f'owned-v0141-tuned32k-{mode}-{phase}-r{rep}/record.json'
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
    quiet = {key: [] for key in ['control', 'integrated']}
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
        for mode in ['integrated']:
            actual = quiet[mode][0]['requests'][index]
            cross.append({'mode': mode, 'read_index': index,
                          'ids_equal_control': actual['ids'] == reference['ids'],
                          'logprobs_equal_control': actual['logprobs'] == reference['logprobs'],
                          'mtp_counts_equal_control': actual['mtp_counts'] == reference['mtp_counts'],
                          'note': 'Cross-version differences are reported, not silently accepted as numerical equivalence. Matched actual8192 chunks/128 slots and qualified per-mode numerical references are required. Own logged/repeated-output checks are mandatory.'})
    record['summary'] = summaries
    record['cross_version_output_comparison'] = cross
    record['paired_requested_configuration'] = True
    record['actual_geometry_note'] = 'Actual8192 chunks and128 slots enforced for both modes; same requested argv and runtime settings.'
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
