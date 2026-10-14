#!/usr/bin/env python3
"""CPU-only mutation checks against copies of the closed suite receipts."""
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile

D = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ORIGINAL = B/'main-primary-long-generation-suite-root-v4-upload-fix-v1'


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('finalizer_fixture', D/'finalize_upload_fix_suite.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index = load(D/'receipts'/ORIGINAL.name/'index.json')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = {Path(e['original']).relative_to(ORIGINAL): Path(e['original']).read_bytes()
                    for e in index['files']}
        assert all(digest(original[Path(e['original']).relative_to(ORIGINAL)]) == e['sha256']
                   for e in index['files'])
        assert load(ORIGINAL/'record.json')['active'] is False
        outcomes = []
        for case in ('normal-idempotent', 'active', 'ownership', 'suite-hash', 'journal-hash',
                     'server-result-anchor', 'short-long-input', 'capped-agent'):
            with tempfile.TemporaryDirectory(prefix='strata-finalizer-fixture-') as name:
                root = Path(name)
                module.O, module.D, module.B = root/'run', root/'out', root/'locks'
                module.D.mkdir(); module.B.mkdir()
                for path, raw in original.items():
                    target = module.O/path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                record_path = module.O/'record.json'
                record = load(record_path)
                suite_path = next((module.O/'stock-results').glob('*/results.json'))
                journal_path = suite_path.parent/'requests.jsonl'
                refusal = case in ('active', 'ownership', 'suite-hash', 'journal-hash', 'server-result-anchor')
                if case == 'active':
                    record['active'] = True
                elif case == 'ownership':
                    record['commands'][1]['session_empty'] = False
                elif case == 'suite-hash':
                    suite_path.write_bytes(suite_path.read_bytes() + b' ')
                elif case == 'journal-hash':
                    journal_path.write_bytes(journal_path.read_bytes().replace(b'\n', b' \n', 1))
                    from strata_bench import verify_request_journal
                    assert verify_request_journal(suite_path.parent)['valid']
                elif case == 'server-result-anchor':
                    result = load(module.O/'result.json')
                    result['server_exit_code'] = 1
                    save(module.O/'result.json', result)
                elif case in ('short-long-input', 'capped-agent'):
                    suite = load(suite_path)
                    rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
                    if case == 'short-long-input':
                        target = next(r for r in rows if r['kind'] == 'result' and
                                      r['phase'] == 'long-generation/regular-code' and r['run'] == 0)
                        target['sample']['prompt_tokens'] = 32767
                        suite['results']['long_generation']['workloads']['regular-code']['requests'][0]['prompt_tokens'] = 32767
                    else:
                        target = next(r for r in rows if r['kind'] == 'result' and
                                      r['phase'] == 'agent' and r['run'] == r['turn'] == 0)
                        target['sample']['finish_reason'] = 'length'
                        suite['results']['agent']['sessions'][0][0]['finish_reason'] = 'length'
                    previous = None
                    for row in rows:
                        row.pop('sha256')
                        row['previous_sha256'] = previous
                        previous = digest(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                                     separators=(',', ':')).encode())
                        row['sha256'] = previous
                    journal_path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                    state_path = suite_path.parent/'journal-state.json'
                    state = load(state_path); state['last_sha256'] = previous; save(state_path, state)
                    save(suite_path, suite)
                    record.update(benchmark_result=suite, benchmark_result_sha256=module.sha(suite_path))
                    entry = record['request_journals'][0]
                    entry.update(sha256=module.sha(journal_path), bytes=journal_path.stat().st_size, state=state)
                    from strata_bench import verify_request_journal
                    entry['verification'] = verify_request_journal(suite_path.parent)
                save(record_path, record)
                before = record_path.read_bytes()
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        module.main()
                        if case == 'normal-idempotent':
                            module.main()
                    assert not refusal, 'inconsistent fixture was accepted'
                    summary = load(module.D/'suite-v4-upload-fix-final-summary.json')
                    assert summary['benchmark_protocol_qualified'] == (case == 'normal-idempotent')
                    if case == 'short-long-input':
                        assert not summary['gates']['strong_weak_bands_confirmed']
                    if case == 'capped-agent':
                        assert not summary['gates']['agent_complete']
                    assert summary['original_passed'] is True
                    assert not any(summary[k] for k in ('quality_qualified', 'physical_262144_qualified',
                               'restore_qualified', 'speed_improvement_qualified', 'adopted'))
                except RuntimeError as error:
                    assert refusal, repr(error)
                    assert not (module.D/'suite-v4-upload-fix-final-summary.json').exists()
                assert record_path.read_bytes() == before
                outcomes.append({'case': case, 'passed': True, 'expected_refusal': refusal})
        assert all((ORIGINAL/path).read_bytes() == raw for path, raw in original.items())
    save(D/'finalizer-cpu-validation.json', {'passed': True, 'cases': outcomes,
         'originals_unchanged': True, 'GPU_used': False, 'active_run_touched': False,
         'collector_sha256': module.sha(D/'finalize_upload_fix_suite.py'),
         'fixture_sha256': module.sha(Path(__file__)),
         'scope': 'Copies of closed receipts under the exclusive measurement lock; integrity refusals, independent input/agent gates, original-status preservation and idempotence only.'})
    print(json.dumps({'passed': True, 'cases': len(outcomes), 'GPU_used': False}))


if __name__ == '__main__':
    main()
