#!/usr/bin/env python3
"""Main-reviewed own pitch40 integration/build; no GPU execution."""
import fcntl
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
S = A / 'own-prompt-k-padded-v1'
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old_path = A / 'own-prompt-k-cellmajor-build-v3.json'
        comparison_path = A / 'own-prompt-k-cellmajor-comparison-v2.json'
        old = json.loads(old_path.read_text())
        comparison = json.loads(comparison_path.read_text())
        require(old['passed'] and old['complete'] and not old['active'], 'previous build closed')
        require(comparison['passed'] and comparison['complete'], 'previous component comparison complete')
        for c in old['commands']:
            require(module.closed(c), 'previous build command closed')
            for identity in c['owners']:
                now = module.identity(identity['pid'])
                require(now is None or now['start_ticks'] != identity['start_ticks'], 'previous owner absent')
        require(all(sha(P / n) == h for n, h in old['source_pins'].items()), 'compiled source base stable')
        manifest = json.loads((S / 'manifest.json').read_text())
        require(sha(S / 'source.diff') == manifest['diff_sha256'] ==
                'ca4233235911c3593db6c8fda715681875b122a8d08c3f2f7c7b24256807668f', 'reviewed exact patch')
        require(set(manifest['files']) == {'src/kernels/sycl/prompt_attention.cpp',
                'tests/sycl/prompt_k_cellmajor.cpp', 'cmake/sycl_backend.cmake'}, 'reviewed file scope')
        for n, value in manifest['files'].items():
            require(sha(P / n) == value['base_sha256'], 'exact working base')
            require(sha(S / 'files' / n) == value['proposal_sha256'], 'exact reviewed proposal')
        require(manifest['files']['cmake/sycl_backend.cmake']['base_sha256'] ==
                manifest['files']['cmake/sycl_backend.cmake']['proposal_sha256'], 'CMake unchanged')
        require(sha(old['compiler']) == old['compiler_sha256'], 'compiler unchanged')
        build = Path(old['build_path'])
        products = {name: {'path': str(build / name), 'sha256': sha(build / name),
                          'bytes': (build / name).stat().st_size}
                    for name in ('strata', 'sycl_prompt_k_cellmajor_test')}
        require(products['strata']['sha256'] == old['binary_sha256'] and
                products['sycl_prompt_k_cellmajor_test']['sha256'] == old['component_binary_sha256'],
                'old products exact')
        root = B / 'main-own-prompt-k-padded-build-root-v1'
        target = A / 'own-prompt-k-padded-build-v1.json'
        require(not root.exists() and not target.exists(), 'fresh build')
        root.mkdir(mode=0o700)
        # Ruby lint paths are unrelated to this C++ build; retain only its required environment.
        env = {k: v for k, v in old['environment'].items() if k not in ('RUBYLIB', 'GEM_PATH')}
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[],
            references={p.name: sha(p) for p in (old_path, comparison_path)},
            recipe_sha256=sha(Path(__file__)), source_diff_sha256=sha(S / 'source.diff'),
            proposal_manifest_sha256=sha(S / 'manifest.json'), transformations=manifest['files'],
            environment=env, source_pins=dict(old['source_pins']), build_path=str(build),
            compiler=old['compiler'], compiler_sha256=old['compiler_sha256'],
            canonical_policy_passed=old['canonical_policy_passed'],
            retained_reference_binary=old['retained_reference_binary'], replaced_products=products,
            component_qualified=False, performance_qualified=False, quality_qualified=False,
            restore_qualified=False, physical_262144_qualified=False,
            scope='Own pitch40 K tile only, defaultOFF, reviewed two-file patch; actual SYCL build, no GPU. '
                  'Prior plain cell-major remains rejected. Product replacement recorded; no extra binary copy.')
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
            record['source_pins'].update({n: v['proposal_sha256'] for n, v in manifest['files'].items()})
            require(all(sha(P / n) == h for n, h in record['source_pins'].items()), 'exact applied source')
            record['source_applied'] = True
            persist()
            c, out, err = owner.run('incremental-sycl-build',
                ['cmake', '--build', str(build), '--target', 'strata', 'sycl_prompt_k_cellmajor_test', '--parallel', '2'],
                env, P, wall=1200, cpu=14400, rss_cap=32 << 30,
                text_cap=16 << 20, total_cap=1 << 30, file_cap=512 << 20)
            require(module.closed(c) and c['exit_code'] == 0,
                    'SYCL build: ' + err.read_text()[-3000:] + '\n' + out.read_text()[-3000:])
            require(all(sha(P / n) == h for n, h in record['source_pins'].items()), 'compiled source stable')
            record.update(passed=True, complete=True, binary=str(build / 'strata'),
                binary_sha256=sha(build / 'strata'), component_binary=str(build / 'sycl_prompt_k_cellmajor_test'),
                component_binary_sha256=sha(build / 'sycl_prompt_k_cellmajor_test'))
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
