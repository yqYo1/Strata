#!/usr/bin/env python3
"""Build a private 16-cell attention proposal and independent fixtures, without GPU work."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), realpath=str(path.resolve()), bytes=path.stat().st_size, sha256=digest)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    root = B / 'main-qsa-prefix16-build-v1'
    out = A / 'qsa-prefix16-build-v1.json'
    assert not root.exists() and not out.exists()
    # Serialize after both complete model cohorts; do not contend for CPU/RAM.
    gates = []
    for arm in (8, 10):
        path = A / f'moe-accum-model-profile-window{arm}-tokens65536-warm4-v5.json'
        value = json.loads(path.read_text())
        assert value['passed'] and value['complete'] and value['ownership_closed'] and not value['active']
        assert value['fresh_faults'] == [] and not value['driver_dump_exists'] and value['boot_unchanged']
        gates.append((path, value))
    build_path = A / 'moe-accum-engine-build-v2.json'
    build = json.loads(build_path.read_text())
    assert build['passed'] and build['complete'] and build['ownership_closed'] and not build['active']
    proposal = A / 'qsa-prefix16-candidate-v1'
    proposal_path = proposal / 'proposal.json'
    meta = json.loads(proposal_path.read_text())
    for name, expected in meta['base_source_hashes'].items():
        assert pin(P / name)['sha256'] == expected
    for name, expected in meta['candidate_hashes'].items():
        assert pin(proposal / 'files' / name)['sha256'] == expected
    assert pin(proposal / 'candidate.patch')['sha256'] == meta['patch_sha256']
    compiler = pin(L / 'bin/clang++')
    assert compiler['sha256'] == '1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
    owner_path = A / 'direct_owner_transient_artifacts_v1.py'
    assert pin(owner_path)['sha256'] == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owned = module('qsa_prefix16_build_owner', owner_path)
    guard_package = A / 'own-vllm-xe2-qualification-v1'
    guard_meta = json.loads((guard_package / 'manifest.json').read_text())
    guard_path = guard_package / 'main_controller.py'
    assert pin(guard_path)['sha256'] == guard_meta['package_files']['main_controller.py']['sha256']
    guards = module('qsa_prefix16_build_guard', guard_path)
    library_root = Path(build['binary']['path']).parent
    libraries = [library_root / name for name in ('libstrata_kernels.a', 'libstrata_core.a', 'libstrata_sycl_runtime.a')]
    inputs = [build_path, proposal_path, proposal / 'candidate.patch', Path(__file__), owner_path, guard_path,
              L / 'bin/clang++', *libraries, *(path for path, _ in gates)]
    inputs += [proposal / 'files' / name for name in meta['candidate_hashes']]
    inputs += list((P / 'include').rglob('*.hpp'))
    inputs += [P / 'tests/sycl/qsa_bf16_prefix.cpp']
    frozen = {str(path): pin(path) for path in inputs}
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners(guard_meta['known_model_files'])
        for _, gate in gates:
            for command in gate['commands']:
                assert owned.closed(command) and command['exit_code'] == 0
                for old in command['owners'] + [command['direct_identity']]:
                    now = owned.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
        root.mkdir(mode=0o700)
        (root / 'tmp').mkdir()
        env = dict(PATH=str(L / 'bin') + ':/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8',
                   LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=str(L / 'lib'),
                   TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
        record = dict(schema='strata-qsa-prefix16-build-v1', passed=False, complete=False, active=False,
                      GPU_executed=False, production_selected=False, performance_adopted=False,
                      commands=[], frozen_files=frozen, compiler=compiler, environment=env,
                      proposal=pin(proposal_path), products={},
                      scope='Private 16-cell/score17 candidate plus unchanged32-cell smoke, independent16-cell oracle and matched32/64 timing fixture. Compile/link only; no numerical or performance qualification.')
        owner = None

        def save():
            record['active'] = bool(owner and owner.active is not None)
            out.write_text(json.dumps(record, indent=2) + '\n')

        owner = owned.Owner(root / 'owned', save)
        record['commands'] = owner.commands

        def verify():
            for path, expected in frozen.items():
                assert pin(path) == expected, path

        def run(label, argv):
            command, stdout, stderr = owner.run(label, argv, env, P, wall=300, cpu=900,
                rss_cap=12 << 30, text_cap=64 << 20, file_cap=256 << 20, total_cap=2 << 30)
            assert owned.closed(command) and command['exit_code'] == 0, (label, stderr.read_text()[-4000:])
            return stdout.read_text(), stderr.read_text()

        flags = [compiler['path'], '-std=c++20', '-O3', '-DNDEBUG', '-DSTRATA_ENABLE_SYCL=1',
                 '-I' + str(proposal / 'files/include'), '-I' + str(P / 'include'),
                 '-I' + str(P / 'include/strata/sycl/compat'), '-fno-fast-math', '-ffp-contract=off',
                 '-fsycl', '-fsycl-id-queries-range=size_t', '-Wall', '-Wextra', '-Werror']
        save()
        try:
            verify()
            obj = root / 'qsa-prefix16.o'
            run('compile-proposal', flags + ['-c', str(proposal / 'files/src/kernels/sycl/qsa_bf16_prefix.cpp'), '-o', str(obj)])
            record['object'] = pin(obj)
            for name, fixture in [('baseline32', P / 'tests/sycl/qsa_bf16_prefix.cpp'),
                                  ('candidate16', proposal / 'files/tests/sycl/qsa_bf16_prefix16.cpp'),
                                  ('timing', proposal / 'files/tests/sycl/qsa_bf16_prefix16_bench.cpp')]:
                exe = root / ('sycl_qsa_prefix16_' + name)
                run('link-' + name, flags + [str(fixture), str(obj), *map(str, libraries), '-o', str(exe)])
                dynamic, _ = run('readelf-' + name, ['/usr/bin/readelf', '-d', str(exe)])
                record['products'][name] = dict(binary=pin(exe), dynamic=dynamic, fixture=pin(fixture))
            record['binary'] = record['products']['candidate16']['binary']
            verify()
            record.update(passed=True, complete=True)
        except BaseException as error:
            import traceback
            record.update(error=repr(error), traceback=traceback.format_exc(), passed=False, complete=False)
        finally:
            record['ownership_closed'] = owner.active is None and all(owned.closed(c) for c in owner.commands)
            record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save()
        print(json.dumps(dict(receipt=str(out), passed=record['passed'], error=record.get('error'))), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
