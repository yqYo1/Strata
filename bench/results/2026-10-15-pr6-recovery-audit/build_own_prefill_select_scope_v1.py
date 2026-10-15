#!/usr/bin/env python3
"""Main reviewed host-only dispatch isolation; actual SYCL build, no GPU."""
import fcntl
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
S = A / 'own-prefill-selector-scope-v1'
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bp = A / 'own-qsa-select-build-v1.json'
        old = json.loads(bp.read_text())
        cp = A / 'own-qsa-select-model-a2-comparison-v1.json'
        comparison = json.loads(cp.read_text())
        require(old['passed'] and old['complete'] and not old['active'] and comparison['passed']
                and comparison['complete'] and comparison['sampled_FP32_values_equal'], 'preceding build/model comparison closed')
        require(all(sha(P / n) == h for n, h in old['source_pins'].items()), 'compiled working base stable')
        manifest = json.loads((S / 'manifest.json').read_text())
        require(sha(S / 'source.diff') == manifest['source_diff_sha256'] ==
                'db2f12de832e974b7ae1ea83e5ffd5459542c49bb2ea6c12878be9ff971fd74a', 'reviewed exact patch')
        base = {v['path']: v for v in manifest['base_files']}
        proposed = {v['path']: v for v in manifest['proposed_files']}
        require(set(base) == set(proposed) == {'src/prefill/prefill.cpp', 'src/kernels/sycl/qsa_select.cpp',
                'include/strata/kernels/qsa_select.hpp', 'tests/sycl/qsa_select_radix.cpp'}, 'reviewed four-file scope')
        for n in base:
            require(sha(P / n) == base[n]['sha256'] and sha(S / 'files' / n) == proposed[n]['sha256'], 'exact base/proposal bytes')
        before = (P / 'src/kernels/sycl/qsa_select.cpp').read_text()
        after = (S / 'files/src/kernels/sycl/qsa_select.cpp').read_text()
        kernel0 = before[before.index('template <int Mapping>'):before.index('\nint selection_mapping')].rstrip()
        kernel1 = after[after.index('template <int Mapping>'):after.index('\nint64_t validate_topk')].rstrip()
        require(kernel0 == kernel1, 'entire device kernel template/lambda byte-identical')
        build = Path(old['build_path'])
        require(sha(build / 'strata') == old['binary_sha256'] and sha(old['compiler']) == old['compiler_sha256'], 'native/compiler pins')
        root = B / 'main-own-prefill-select-scope-build-root-v1'
        target = A / 'own-prefill-select-scope-build-v1.json'
        require(not root.exists() and not target.exists(), 'fresh build')
        root.mkdir(mode=0o700)
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[],
            references={str(p.relative_to(A)): sha(p) for p in (bp, cp, S / 'manifest.json')},
            recipe_sha256=sha(Path(__file__)), source_diff_sha256=sha(S / 'source.diff'),
            transformations={n: dict(base_sha256=base[n]['sha256'], proposed_sha256=proposed[n]['sha256']) for n in base},
            device_kernel_template_byte_identical=True, source_pins=dict(old['source_pins']),
            build_path=str(build), environment=old['environment'], compiler=old['compiler'],
            compiler_sha256=old['compiler_sha256'], canonical_policy_passed=old['canonical_policy_passed'],
            retained_reference_binary=old['retained_reference_binary'], retained_prompt_component=old['retained_prompt_component'],
            replaced_native_sha256=old['binary_sha256'], component_qualified=False, performance_qualified=False,
            scope='Main host-dispatch isolation to dedicated SYCL layer-major prefill. Public decode/verify/short/ref API alwaysMapping0. GPU template unchanged, no Xe source. Build only.')
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        hostenv = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        try:
            for label, argv in [('patch-check', ['git', 'apply', '--check', str(S / 'source.diff')]),
                                ('patch-apply', ['git', 'apply', str(S / 'source.diff')])]:
                c, out, err = owner.run(label, argv, hostenv, P, wall=60, cpu=60,
                    rss_cap=2 << 30, text_cap=1 << 20, total_cap=1 << 30, file_cap=1 << 20)
                require(module.closed(c) and c['exit_code'] == 0, label + ': ' + err.read_text()[-2000:])
            record['source_pins'].update({n: v['sha256'] for n, v in proposed.items()})
            require(all(sha(P / n) == h for n, h in record['source_pins'].items()), 'exact applied source')
            record['source_applied'] = True
            persist()
            c, out, err = owner.run('incremental-sycl-build',
                ['cmake', '--build', str(build), '--target', 'strata', 'sycl_qsa_select_radix_test', '--parallel', '2'],
                record['environment'], P, wall=1200, cpu=14400, rss_cap=32 << 30,
                text_cap=16 << 20, total_cap=1 << 30, file_cap=512 << 20)
            require(module.closed(c) and c['exit_code'] == 0, 'SYCL build: ' + err.read_text()[-3000:] + '\n' + out.read_text()[-3000:])
            require(all(sha(P / n) == h for n, h in record['source_pins'].items()), 'compiled source stable')
            exe = build / 'sycl_qsa_select_radix_test'
            record.update(passed=True, complete=True, binary=str(build / 'strata'), binary_sha256=sha(build / 'strata'),
                component_binary=str(exe), component_binary_sha256=sha(exe))
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):
                record.update(passed=False, complete=False)
            persist()
            with target.open('x') as f:
                json.dump(record, f, indent=2)
                f.write('\n')
        print(json.dumps({k: record[k] for k in ('passed', 'binary_sha256', 'component_binary_sha256')}), flush=True)


if __name__ == '__main__':
    main()
