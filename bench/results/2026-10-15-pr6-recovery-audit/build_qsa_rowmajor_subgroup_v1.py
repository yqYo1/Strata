#!/usr/bin/env python3
"""Build the reviewed subgroup softmax on the CURRENT row-major score33 path."""
import difflib
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-moe-window')
C = W / 'bench/results/2026-10-15-pr6-recovery-audit'
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), realpath=str(path.resolve()), bytes=path.stat().st_size, sha256=value)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def candidate(original):
    # Same checked numerical design as the older packed proposal; current
    # optimized row-major Q/K/V/P layout, not the separately stalled experiment.
    start = original.index('        if (tid < 16) {\n          float m = maximum[tid];')
    end = original.index('        for (int i = tid; i < Cells * Dim;', start)
    norm = """        for (int h = subgroup; h < 16; h += 8) {
          const int c0 = lane, c1 = lane + 16;
          const float s0 = scores[h * ScoreStride + c0];
          const float s1 = scores[h * ScoreStride + c1];
          const bool row0 = rows[c0] >= 0, row1 = rows[c1] >= 0;
          const bool valid0 = row0 && sycl::isfinite(s0);
          const bool valid1 = row1 && sycl::isfinite(s1);
          const int bad = (row0 && !valid0) || (row1 && !valid1);
          const int score_bad = sycl::reduce_over_group(sg, bad, sycl::maximum<int>());
          const float lane_max = sycl::fmax(valid0 ? s0 : -INFINITY,
                                           valid1 ? s1 : -INFINITY);
          const float tile_max = sycl::reduce_over_group(sg, lane_max, sycl::maximum<float>());
          const float prior = !lane ? maximum[h] : -INFINITY;
          const float old_m = sycl::group_broadcast(sg, prior, 0);
          const float m = sycl::fmax(old_m, tile_max);
          const float p0 = valid0 ? sycl::exp2(s0 - m) : 0.f;
          const float p1 = valid1 ? sycl::exp2(s1 - m) : 0.f;
          const float sum = sycl::reduce_over_group(sg, p0 + p1, sycl::plus<float>());
          probability[h * Cells + c0] = BFloat(p0);
          probability[h * Cells + c1] = BFloat(p1);
          if (!lane) {
            const float old_norm = normalizer[h];
            const float a = old_norm > 0.f ? sycl::exp2(old_m - m) : 0.f;
            bad_head[h] |= score_bad;
            maximum[h] = m; alpha[h] = a;
            normalizer[h] = sycl::fma(old_norm, a, sum);
          }
        }
"""
    changed = original[:start] + norm + original[end:]
    # A unique implementation name also separates compiler-generated kernel IDs
    # from the baseline template linked from the current engine archive.
    changed = changed.replace('prefix_batch_impl', 'rowmajor_subgroup_impl')
    entry = changed.index('bool qsa_bf16_prefix_optimized_batch(')
    ending = changed.index('// Dense16 A tiles', entry)
    exported = changed[entry:ending].replace('qsa_bf16_prefix_optimized_batch',
                                           'qsa_bf16_prefix_rowmajor_subgroup_batch')
    changed = changed[:changed.index('bool qsa_bf16_prefix_batch(')] + exported + '} // namespace strata::kernels\n'
    assert 'rowmajor_subgroup_impl<Cells + 1>' in changed
    assert 'rowmajor_subgroup_impl<Cells + 1, true>' not in changed
    return changed


def main():
    root = B / 'main-qsa-rowmajor-subgroup-build-v1'
    out = A / 'qsa-rowmajor-subgroup-build-v1.json'
    assert not root.exists() and not out.exists()
    cohort_path = A / 'moe-window-clean-cohort-v3.json'
    cohort = json.loads(cohort_path.read_text())
    assert cohort['passed'] and cohort['complete'] and cohort['active_sample'] is None and len(cohort['completed']) == 6
    build_path = C / 'moe-window-engine-build-v1.json'
    build = json.loads(build_path.read_text())
    assert build['passed'] and build['complete'] and build['ownership_closed'] and not build['active']
    owner_path = A / 'direct_owner_transient_artifacts_v1.py'
    assert pin(owner_path)['sha256'] == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owner_module = module('rowmajor_build_owner', owner_path)
    package = A / 'own-vllm-xe2-qualification-v1'
    manifest_path = package / 'manifest.json'
    assert pin(manifest_path)['sha256'] == '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    manifest = json.loads(manifest_path.read_text())
    helper = package / 'main_controller.py'
    assert pin(helper)['sha256'] == manifest['package_files']['main_controller.py']['sha256']
    guard = module('rowmajor_build_guard', helper)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.no_model_owners(manifest['known_model_files'])
        guard.driver_guard()
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == build['boot_id']
        for item in cohort['completed']:
            assert pin(item['path'])['sha256'] == item['sha256']
            receipt = json.loads(Path(item['path']).read_text())
            assert receipt['passed'] and receipt['complete'] and receipt['ownership_closed'] and not receipt['active']
            assert receipt['fresh_faults'] == [] and not receipt['driver_dump_exists'] and receipt['boot_unchanged']
            for command in receipt['commands']:
                assert owner_module.closed(command) and command['exit_code'] == 0
                for old in command['owners'] + [command['direct_identity']]:
                    now = owner_module.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
        frozen = {path: pin(path) for path in build['source_pins']}
        assert all(frozen[path]['sha256'] == value for path, value in build['source_pins'].items())
        source = W / 'src/kernels/sycl/qsa_bf16_prefix.cpp'
        assert pin(source)['sha256'] == '0733dbdc8687c032c2510bc3f350aed0279233a7c0ebb867171c2b2f93ac452c'
        compiler = pin(L / 'bin/clang++')
        assert compiler['sha256'] == '1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
        libdir = Path(build['products']['strata']['path']).parent
        libraries = [libdir / name for name in ('libstrata_kernels.a', 'libstrata_core.a', 'libstrata_sycl_runtime.a')]
        for path in [*libraries, cohort_path, build_path, owner_path, helper, manifest_path,
                     Path(__file__), L / 'bin/clang++', A / 'qsa-rowmajor-subgroup-contract-v1.ja.md']:
            frozen[str(path)] = pin(path)
        original = source.read_text()
        changed = candidate(original)
        patch = ''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                       fromfile='a/src/kernels/sycl/qsa_bf16_prefix.cpp', tofile='b/private/qsa_rowmajor_subgroup.cpp'))
        root.mkdir(mode=0o700)
        (root / 'tmp').mkdir()
        private = root / 'qsa_rowmajor_subgroup.cpp'
        private.write_text(changed)
        frozen[str(private)] = pin(private)
        rec = dict(schema='strata-qsa-rowmajor-subgroup-build-v1', passed=False, complete=False, active=False,
                   GPU_executed=False, commands=[], frozen_files=frozen, binary=None, timing_binary=None,
                   base_engine_source_commit=build['source_commit'], base_source=pin(source),
                   candidate_source=pin(private), source_diff=patch,
                   library_pin_scope='Freshly hashed completed build outputs, first observed as inputs here.',
                   provenance='Main-authored subgroup collective softmax using the already reviewed SGLang numerical design contract; current independently authored Strata row-major kernel and test fixtures only. No XeStrata project implementation.',
                   production_selected=False, performance_adopted=False)
        owner = None

        def save():
            rec['active'] = bool(owner and owner.active is not None)
            rec['ownership_closed'] = bool(owner and owner.active is None and all(owner_module.closed(c) for c in owner.commands))
            out.write_text(json.dumps(rec, indent=2) + '\n')

        owner = owner_module.Owner(root / 'owned', save)
        rec['commands'] = owner.commands
        env = dict(PATH=str(L / 'bin') + ':/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                   PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=str(L / 'lib'), TMPDIR=str(root / 'tmp'),
                   XDG_CACHE_HOME=str(root / 'cache'))
        common = [compiler['path'], '-std=c++20', '-O3', '-DNDEBUG', '-DSTRATA_ENABLE_SYCL=1',
                  '-I' + str(W / 'include'), '-I' + str(W / 'include/strata/sycl/compat'),
                  '-fno-fast-math', '-ffp-contract=off', '-fsycl', '-fsycl-id-queries-range=size_t',
                  '-Wall', '-Wextra', '-Werror']
        try:
            for name, fixture, flags in [
                    ('binary', 'qsa_bf16_prefix.cpp', ['-Dqsa_bf16_prefix_batch=qsa_bf16_prefix_rowmajor_subgroup_batch']),
                    ('timing_binary', 'qsa_bf16_prefix_layout_bench.cpp', ['-Dqsa_bf16_prefix_batch=qsa_bf16_prefix_optimized_batch',
                         '-DSTRATA_QSA_PREFIX_CANDIDATE=qsa_bf16_prefix_rowmajor_subgroup_batch',
                         '-DSTRATA_QSA_PREFIX_CANDIDATE_NAME="rowmajor-subgroup"'])]:
                exe = root / ('sycl_qsa_rowmajor_subgroup_' + name)
                argv = [*common, *flags, str(private), str(W / 'tests/sycl' / fixture), *map(str, libraries), '-o', str(exe)]
                command, _, stderr = owner.run('compile-' + name, argv, env, P, wall=300, cpu=900,
                                              rss_cap=8 << 30, text_cap=64 << 20, file_cap=128 << 20, total_cap=1 << 30)
                assert owner_module.closed(command) and command['exit_code'] == 0, str(stderr)
                assert all(pin(path) == value for path, value in frozen.items())
                rec[name] = pin(exe)
                save()
            rec.update(passed=True, complete=True)
        except BaseException as error:
            rec['error'] = repr(error)
        finally:
            save()
        print(json.dumps(dict(passed=rec['passed'], complete=rec['complete'], receipt=str(out), error=rec.get('error'))))
        return 0 if rec['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
