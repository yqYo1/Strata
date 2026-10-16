#!/usr/bin/env python3
"""Build the reviewed private GDN fixture after the current profile closes.

No product edit or model execution. The compiler path is a dependency only;
no source from the excluded project is read.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

if not __debug__:
    raise RuntimeError('Run without -O: the qualification guards must remain enabled')
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
PACKAGE = A/'gdn-output16-only-v2'
HOST = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8',
            LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), realpath=str(path.resolve()),
                bytes=path.stat().st_size, sha256=digest)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    root = B/'main-gdn-output16-only-build-v2'
    dst = A/'own-gdn-output16-only-build-v2.json'
    assert not root.exists() and not dst.exists(), 'never overwrite a result'
    manifest_path = PACKAGE/'manifest.json'
    assert pin(manifest_path)['sha256'] == '4815a33772cd154b48314c184f5b5bf5ad96384eb921552951e7e7d29d2162ff'
    manifest = json.loads(manifest_path.read_text())
    patch, fixture = PACKAGE/'proposal.patch', PACKAGE/'fixture.cpp'
    assert pin(patch)['sha256'] == manifest['patch_sha256']
    assert pin(fixture)['sha256'] == manifest['fixture']['sha256']
    review = A/'research-review-round964-v1.json'
    assert pin(review)['sha256'] == '13eae04202e264e96723bf5b654e110426906b8d529d50cf92c5146ad28d84e7'
    owner_path = A/'direct_owner_transient_artifacts_v1.py'
    assert pin(owner_path)['sha256'] == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owner_module = module('gdn_compile_owner', owner_path)
    guard_package = A/'own-vllm-xe2-qualification-v1'
    guard_meta = guard_package/'manifest.json'
    assert pin(guard_meta)['sha256'] == '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    meta = json.loads(guard_meta.read_text())
    guard_source = guard_package/'main_controller.py'
    assert pin(guard_source)['sha256'] == meta['package_files']['main_controller.py']['sha256']
    guards = module('gdn_compile_guards', guard_source)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resource_guard = guards.no_model_owners(meta['known_model_files'])
        # Preserve the already queued paired profile's precedence.
        profile_path = A/'qsa-bf16-prefix-tilepack-profile-pair-v2.json'
        profile = json.loads(profile_path.read_text())
        assert profile['passed'] and profile['complete'] and profile['active_controller'] is None
        analysis_path = A/'qsa-bf16-prefix-tilepack-profile-summary-v2.json'
        analysis = json.loads(analysis_path.read_text())
        assert analysis['passed'] and analysis['complete']
        assert analysis['pair_receipt']['sha256'] == pin(profile_path)['sha256']
        engine_path = A/'own-qsa-bf16-prefix-tilepack-engine-build-v2.json'
        engine = json.loads(engine_path.read_text())
        assert engine['passed'] and engine['complete'] and engine['ownership_closed'] and not engine['active']
        assert pin(engine['binary']['path']) == engine['binary']
        for item in profile['completed']:
            result_path = Path(item['path'])
            assert pin(result_path)['sha256'] == item['sha256']
            result = json.loads(result_path.read_text())
            assert result['passed'] and result['complete'] and result['ownership_closed'] and not result['active']
            for command in result['commands']:
                assert owner_module.closed(command) and command['exit_code'] == 0
                for old in command['owners'] + [command['direct_identity']]:
                    now = owner_module.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
        compiler = pin(L/'bin/clang++')
        assert compiler['sha256'] == '1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
        libraries = [Path(engine['binary']['path']).parent/name for name in
                     ('libstrata_kernels.a', 'libstrata_core.a', 'libstrata_sycl_runtime.a')]
        frozen = dict(engine['frozen_files'])
        for name, digest in {**manifest['base_files'], **manifest['safety_source_pins']}.items():
            assert pin(P/name)['sha256'] == digest, name
        for path in [Path(__file__), manifest_path, patch, fixture, review, owner_path,
                     guard_meta, guard_source, profile_path, analysis_path, engine_path,
                     L/'bin/clang++', *libraries]:
            frozen[str(path)] = pin(path)
        host_check = A/'transient-artifact-owner-check-v1.json'
        check = json.loads(host_check.read_text())
        assert check['passed'] and check['complete'] and check['helper_sha256'] == pin(owner_path)['sha256']
        frozen[str(host_check)] = pin(host_check)
        root.mkdir(mode=0o700)
        (root/'tmp').mkdir()
        (root/'cache').mkdir()
        rec = dict(schema='strata-gdn-output16-only-build-v2', passed=False, complete=False,
                   active=False, ownership_closed=False, GPU_executed=False, commands=[],
                   frozen_files=frozen, base_commit=manifest['base_commit'], compiler=compiler,
                   engine_source_commit=engine['source_commit'], engine_binary=engine['binary'],
                   source_manifest=pin(manifest_path), fixture=pin(fixture), resource_guard=resource_guard,
                   library_inputs={str(path): pin(path) for path in libraries},
                   proposed_source_hashes={}, product_source_changed=False,
                   performance_adopted=False, model_execution=False,
                   scope='Private same-source GDN float-output-store omission with pre-offset registered span/domain checks. Exact private index, original public header plus fixture forward declaration, candidate translation unit before existing archives. No product caller compile or decode/short/default runtime qualification.',
                   provenance='Main-authored Strata proposal/fixture and controller; no excluded-project source. Libraries are outputs of the qualified current Strata build, first observed as inputs here.')
        owner = None

        def persist():
            rec['active'] = bool(owner and owner.active is not None)
            (root/'record.json').write_text(json.dumps(rec, indent=2)+'\n')

        owner = owner_module.Owner(root/'owned', persist)
        rec['commands'] = owner.commands

        def run(label, argv, env=HOST, wall=30):
            command, out, err = owner.run(label, argv, env, P, wall=wall, cpu=900,
                                         rss_cap=8 << 30, text_cap=64 << 20,
                                         file_cap=128 << 20, total_cap=1 << 30)
            assert owner_module.closed(command) and command['exit_code'] == 0, label
            return out, err

        def verify():
            for path, value in frozen.items():
                assert pin(path) == value, path

        try:
            verify()
            git_env = {**HOST, 'GIT_INDEX_FILE': str(root/'private-index')}
            run('index-base', ['/usr/bin/git', 'read-tree', manifest['base_commit']], git_env)
            run('index-check', ['/usr/bin/git', 'apply', '--cached', '--check', str(patch)], git_env)
            run('index-apply', ['/usr/bin/git', 'apply', '--cached', str(patch)], git_env)
            candidate = root/'fused_gdn.cpp'
            for index, (name, digest) in enumerate(manifest['proposed_files'].items()):
                out, _ = run('extract-'+str(index), ['/usr/bin/git', 'show', ':'+name], git_env)
                assert pin(out)['sha256'] == digest, name
                rec['proposed_source_hashes'][name] = digest
                if name == 'src/kernels/sycl/fused_gdn.cpp':
                    candidate.write_bytes(out.read_bytes())
            rec['candidate_source'] = pin(candidate)
            frozen[str(candidate)] = pin(candidate)
            binary = root/'sycl_gdn_output16_only'
            env = {**HOST, 'PATH': str(L/'bin')+':/usr/bin:/bin',
                   'LD_LIBRARY_PATH': str(L/'lib'), 'TMPDIR': str(root/'tmp'),
                   'XDG_CACHE_HOME': str(root/'cache')}
            argv = [compiler['path'], '-std=c++20', '-O3', '-DNDEBUG',
                    '-DSTRATA_ENABLE_SYCL=1', '-I'+str(P/'include'),
                    '-I'+str(P/'include/strata/sycl/compat'), '-fno-fast-math',
                    '-ffp-contract=off', '-fsycl', '-fsycl-id-queries-range=size_t',
                    '-Wall', '-Wextra', '-Werror', str(candidate), str(fixture),
                    *map(str, libraries), '-o', str(binary)]
            out, err = run('compile-link', argv, env, wall=300)
            rec.update(compile_stdout=out.read_text(), compile_stderr=err.read_text())
            verify()
            rec.update(binary=pin(binary), passed=True, complete=True)
        except BaseException as error:
            import traceback
            rec.update(error=repr(error), traceback=traceback.format_exc())
        finally:
            rec['ownership_closed'] = owner.active is None and all(owner_module.closed(c) for c in owner.commands)
            if not rec['ownership_closed']:
                rec.update(passed=False, complete=False)
            persist()
            dst.write_bytes((root/'record.json').read_bytes())
        print(json.dumps(dict(receipt=str(dst), passed=rec['passed'], error=rec.get('error'), GPU_executed=False)))
        return 0 if rec['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
