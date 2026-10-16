#!/usr/bin/env python3
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
O = B / 'main-niko-setup-isolation-root-v1'
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
    for f in [P / 'setup.py', P / 'tools/setup_intel.py', D / 'run_setup_tests.py', *sorted((P / 'tools').glob('test_setup_*.py'))]:
        record['source_pins'][str(f.relative_to(P))] = hashlib.sha256(f.read_bytes()).hexdigest()
    def persist(): (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    owner = module.Owner(O / 'owned', persist)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for label, root in [('primary', P), ('main', Path('/home/yayoi/ghq/github.com/yqYo1/Strata'))]:
                out = O / (label + '.json')
                argv = ['/home/yayoi/.local/share/strata-sycl/venv/bin/python', str(D / 'setup_isolation.py'), str(root), str(out)]
                entry, stdout, stderr = owner.run(label, argv, env, root, wall=90, total_cap=128 << 20)
                record['commands'].append(entry)
                assert module.closed(entry), 'test ownership did not close'
                assert entry['exit_code'] == 0
                print(stdout.decode(), flush=True)
            record.update(complete=True, passed=True)
        finally:
            record['active'] = False
            persist()
            (D / 'setup-isolation-controller.py').write_bytes(Path(__file__).read_bytes())

if __name__ == '__main__': main()
