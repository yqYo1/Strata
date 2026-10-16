#!/usr/bin/env python3
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
O = B / 'main-niko-primary-syntax-finish-root-v1'
D = P / 'bench/results/2026-10-14-niko-main-service-port'
T = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
GG = Path('/home/yayoi/ghq/github.com/ggml-org/llama.cpp/.worktree/dep-ggml-3cf-xe-e32-20261011')
spec = importlib.util.spec_from_file_location('direct_owner', B / 'xe-llvm-v711-source-v5/direct_owner.py')
owner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner_module)

def main():
    prior = json.loads((B / 'main-niko-primary-host-qualification-root-v2/record.json').read_text())
    assert not prior['active']
    assert prior['service_result']['tests_run'] == 268
    assert all(c['exit_code'] == 0 and owner_module.closed(c) for c in prior['commands'][:-1])
    for path, digest in prior['source_pins'].items():
        assert hashlib.sha256((P / path).read_bytes()).hexdigest() == digest
    assert not O.exists()
    O.mkdir()
    record = {'active': True, 'passed': False, 'complete': False, 'GPU_model_executed': False,
              'commands': [], 'environment': prior['environment'],
              'reused_exact_unchanged_qualified_checks': [c['label'] for c in prior['commands'][:-1]],
              'original_host_qualification_failed_status_preserved': True,
              'source_pins': {**prior['source_pins'], 'include/strata/core/expert_source.hpp': hashlib.sha256((P / 'include/strata/core/expert_source.hpp').read_bytes()).hexdigest()}}
    def persist(): (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    owner = owner_module.Owner(O / 'owned', persist)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            argv = [T / 'bin/clang++', '-std=c++20', '-Wall', '-Wextra', '-Werror', '-UNDEBUG', '-fsycl', '-fsyntax-only',
                    '-DSTRATA_ENABLE_SYCL=1', '-DSTRATA_NATIVE_EXPERTS=1', '-DSTRATA_VERSION="0.1.38"',
                    '-I' + str(P / 'include'), '-I' + str(P / 'include/strata/sycl/compat'),
                    '-I' + str(GG / 'ggml/include'), '-I' + str(GG / 'ggml/src'), P / 'src/program/generate.cpp']
            entry, stdout, stderr = owner.run('SYCL-generator-syntax', [str(x) for x in argv], record['environment'], P, wall=180, total_cap=128 << 20)
            record['commands'].append(entry)
            assert owner_module.closed(entry) and entry['exit_code'] == 0, stderr.read_text()[-2000:]
            record.update(passed=True, complete=True, source_generator_qualified='syntax only, no GPU/link/model proof')
            print(json.dumps({'passed': True, 'scope': 'host fixtures, service, SYCL source syntax; no GPU or inference speed'}), flush=True)
        finally:
            record['active'] = False
            persist()
            (D / 'syntax-finish-controller.py').write_bytes(Path(__file__).read_bytes())

if __name__ == '__main__': main()
