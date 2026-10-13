#!/usr/bin/env python3
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
O = B / 'main-niko-native-setup-tests-root-v3'
D = P / 'bench/results/2026-10-14-niko-main-service-port'
spec = importlib.util.spec_from_file_location('owner', B / 'xe-llvm-v711-source-v5/direct_owner.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def main():
    assert not O.exists()
    O.mkdir()
    env = json.loads((B / 'main-niko-primary-host-qualification-root-v2/record.json').read_text())['environment']
    record = {'active': True, 'passed': False, 'complete': False, 'GPU_model_installer_executed': False,
              'commands': [], 'environment': env, 'source_pins': {}}
    for f in [P / 'setup.py', P / 'tools/setup_intel.py', D / 'run_setup_tests.py', *sorted((P / 'tools').glob('test_setup_*.py')), *[P / f for f in ['tools/strata_mcp.py', 'tools/test_strata_mcp.py', 'tools/draft_vocab.py', 'tools/test_draft_vocab.py', 'tools/make_profile.py', 'tools/test_make_profile.py']]]:
        record['source_pins'][str(f.relative_to(P))] = hashlib.sha256(f.read_bytes()).hexdigest()
    def persist(): (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    owner = module.Owner(O / 'owned', persist)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            argv = ['/home/yayoi/.local/share/strata-sycl/venv/bin/python', str(D / 'run_setup_tests.py'), str(O / 'test-results.json')]
            entry, stdout, stderr = owner.run('setup-unit', argv, env, P, wall=300, total_cap=128 << 20)
            record['commands'].append(entry)
            assert module.closed(entry), 'test ownership did not close'
            result = json.loads((O / 'test-results.json').read_text())
            record.update({'passed': result['passed'] and entry['exit_code'] == 0, 'complete': True, 'tests': result['tests_run'],
                           'result_sha256': hashlib.sha256((O / 'test-results.json').read_bytes()).hexdigest()})
            print(json.dumps({'complete': True, 'passed': record['passed'], 'tests': result['tests_run'], 'errors': result['errors'], 'failures': result['failures']}), flush=True)
        finally:
            record['active'] = False
            persist()
            (D / 'setup-test-controller.py').write_bytes(Path(__file__).read_bytes())

if __name__ == '__main__': main()
