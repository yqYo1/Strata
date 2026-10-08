"""Finite quiet baseline/off/on/on/off/baseline comparison after the first diagnostic gate."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

b = Path(__file__).parent
observer = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0, str(observer / 'sycl/tools'))
from owned_gdb import process_identity

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
controller = b / 'run_owned_host_accounting_v0141_quiet32k_v1.py'
controller_sha = 'a23c23b8a55488dce3be96ec5a3daf54772739ac5d72fd69970374b32f47deb0'
assert sha(controller) == controller_sha
first_path = b / 'owned-host-accounting-v0141-code32k-diagnostic-r1/record.json'
assert sha(first_path) == '6605f3c24444b6069ff10ab594e429ee66c6498a8a9f0cc21d6fcd33e388b2ed'
first = json.loads(first_path.read_text())
assert not first['active'] and first['completed'] and first['healthy'] and first['math_gate_passed']
assert first['host_accounting_gate_passed'] and first['exit_code'] == 0
assert not any(first['cleanup'].values()) and not first['new_fault_messages']
preflight_path = b / 'host-prefill-accounting-v0141-quiet-controller-preparation-v1.json'
assert sha(preflight_path) == '442f209105f26402acd510fde195adf3cd2f7f8e0c8c212c95996623325c8f54'
preflight = json.loads(preflight_path.read_text())
assert preflight['passed'] and preflight['controller_sha256'] == controller_sha
out = b / 'host-prefill-accounting-v0141-quiet32k-comparison-sequence-v1'
assert not out.exists()
out.mkdir(mode=0o700)
record = {
    'active': True, 'passed': False,
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Same-day quiet baseline/host-counter-off/host-counter-on/on/off/baseline. Four fresh32768 A/B/A/B and64 outputs per process, context262144, actual8192 chunks and128 expert slots. First/later separated. No native event queries, profiler, extra waits, API logging or dumps. Instrumented code is private and not adopted.',
    'controller_sha256': controller_sha, 'first_diagnostic_receipt_sha256': sha(first_path),
    'cpu_preparation_receipt_sha256': sha(preflight_path),
    'steps': [], 'deadline_seconds': 4800, 'active_stage': None,
}
started = time.monotonic()
child = None


def save():
    record['elapsed_seconds'] = time.monotonic() - started
    tmp = out / 'record.json.tmp'
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(out / 'record.json')


def terminal(path):
    data = json.loads(path.read_text())
    assert not data['active'] and data['completed'] and data['healthy'] and data['math_gate_passed']
    assert data['performance_eligible'] and data['host_accounting_gate_passed']
    assert data['exit_code'] == 0 and not data['exit_signal'] and not any(data['cleanup'].values()) and not data['new_fault_messages']
    assert len(data['requests']) == 4 and data['phase'] == 'clean'
    assert data['source_sha256'][str(controller)] == controller_sha
    assert data['actual_executable_identity']['sha256'] == data['binary_sha256']
    info = next(x for x in data['startup'] if x.startswith('INFO '))
    assert 'expert_slots=128 ' in info and 'kv_resident=32768 ' in info
    for request in data['requests']:
        assert request['math_gate_passed']
        assert [int(x.split()[1]) for x in request['protocol'] if x.startswith('PP ')] == [8192, 16384, 24576, 32767]
    for role in ['inferior', 'debugger']:
        old = data[role]
        now = process_identity(old['pid'])
        assert not now or now['start_ticks'] != old['start_ticks']
    return data


save()
try:
    for mode, rep in [('baseline', 1), ('hostoff', 1), ('hoston', 1), ('hoston', 2), ('hostoff', 2), ('baseline', 2)]:
        assert time.monotonic() - started < record['deadline_seconds']
        assert sha(controller) == controller_sha
        record['active_stage'] = f'{mode}-r{rep}'
        receipt = b / f'owned-host-accounting-v0141-code32k-{mode}-clean-r{rep}/record.json'
        argv = ['/usr/bin/python3', str(controller), mode, 'clean', str(rep)]
        step = {'mode': mode, 'repetition': rep, 'argv': argv, 'receipt': str(receipt), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
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
        assert step['exit_code'] == 0
        terminal(receipt)
        step['receipt_sha256'] = sha(receipt)
        save()
    groups = {mode: [terminal(Path(s['receipt'])) for s in record['steps'] if s['mode'] == mode] for mode in ['baseline', 'hostoff', 'hoston']}
    argvs = {tuple(d['argv'][1:]) for jobs in groups.values() for d in jobs}
    envs = set()
    for mode, jobs in groups.items():
        for d in jobs:
            env = dict(d['actual_target_environment'])
            flag = env.pop('STRATA_PREFILL_HOST_TIMING', None)
            assert flag == {'baseline': None, 'hostoff': '0', 'hoston': '1'}[mode]
            assert not any(k.startswith(('UR_LOG_', 'ZE_ENABLE_', 'ZEL_')) or k in ['UR_ENABLE_LAYERS', 'STRATA_TRACE', 'STRATA_DUMP_FIRST_LOGITS', 'STRATA_PREFILL_DUMP_STATE', 'STRATA_PREFILL_TRANSFER_TIMING'] for k in env)
            envs.add(json.dumps(env, sort_keys=True))
    assert len(argvs) == len(envs) == 1
    record['summary'] = {}
    for mode, jobs in groups.items():
        assert len(jobs) == 2
        record['summary'][mode] = {}
        for label, first_read in [('first_process_read', True), ('later_fresh_reads', False)]:
            reads = [r for d in jobs for r in d['requests'] if r['first_process_read'] == first_read]
            assert len(reads) == (2 if first_read else 6)
            record['summary'][mode][label] = {
                'count': len(reads),
                'prefill_tok_s': 1000 * sum(r['measurement']['prompt_tokens'] for r in reads) / sum(r['measurement']['prompt_ms'] for r in reads),
                'decode_tok_s': 1000 * sum(r['measurement']['generated_tokens'] for r in reads) / sum(r['measurement']['decode_ms'] for r in reads),
                'reads': [{'name': r['name'], 'measurement': r['measurement']} for r in reads],
            }
    record['host_reports'] = [r for d in groups['hoston'] for r in d['host_accounting_reports']]
    assert len(record['host_reports']) == 8
    record['configuration_equal_except_explicit_host_accounting_flag'] = True
    record['passed'] = True
except BaseException as error:
    record['error'] = repr(error)
    if child and child.poll() is None:
        os.kill(child.pid, signal.SIGINT)
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            record['controller_cleanup_pending'] = process_identity(child.pid)
finally:
    record['active'] = False
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
print(json.dumps({key: record.get(key) for key in ['active', 'passed', 'elapsed_seconds', 'active_stage', 'error', 'summary']}))
if not record['passed']:
    raise SystemExit(1)
