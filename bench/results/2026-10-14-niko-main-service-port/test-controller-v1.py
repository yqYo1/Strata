#!/usr/bin/env python3
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
O = B / 'main-niko-service-port-tests-root-v1'
D = P / 'bench/results/2026-10-14-niko-main-service-port'
OWNER = B / 'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(OWNER.read_bytes()).hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec = importlib.util.spec_from_file_location('direct_owner', OWNER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
env = json.loads((B / 'main-layer-major-bulk-full-lints-root-v3/record.json').read_text())['environment']
env = {k: v for k, v in env.items() if k != 'STRATA_LINT_BUILD'}
env['PYTHONPATH'] = str(P / 'tools') + ':' + str(P)

def main():
    assert not O.exists()
    O.mkdir()
    record = {'active': True, 'passed': False, 'complete': False, 'GPU_model_executed': False,
              'environment': env, 'commands': [], 'source_import_sha256': hashlib.sha256((D / 'source-import.json').read_bytes()).hexdigest()}
    def persist():
        (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    owner = module.Owner(O, persist)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            python = '/home/yayoi/.local/share/strata-sycl/venv/bin/python'
            entry, stdout, stderr = owner.run('service-unit', [python, str(D / 'run_service_tests.py'), str(O / 'test-results.json')], env, P, wall=480, text_cap=16 << 20, file_cap=16 << 20)
            record['commands'].append(entry)
            assert module.closed(entry), 'test process ownership did not close'
            samples = json.loads((O / 'test-results.json').read_text())
            record['tests_run'] = samples['tests_run']
            record['passed'] = entry['exit_code'] == 0 and samples['passed']
            record['complete'] = True
            record['result_sha256'] = hashlib.sha256((O / 'test-results.json').read_bytes()).hexdigest()
            if not record['passed']:
                record['failures'] = [x for x in samples['samples'] if x['status'] in ['failed', 'error', 'unexpected_success']]
            print(json.dumps({'complete': True, 'passed': record['passed'], 'tests': samples['tests_run'], 'errors': samples['errors'], 'failures': samples['failures']}), flush=True)
        finally:
            record['active'] = False
            persist()
    (D / 'test-controller.py').write_bytes(Path(__file__).read_bytes())

if __name__ == '__main__':
    main()
