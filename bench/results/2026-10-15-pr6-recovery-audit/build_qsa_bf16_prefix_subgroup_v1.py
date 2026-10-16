#!/usr/bin/env python3
"""Compile a single-layout QSA experiment; preserve the measured production tree."""
import difflib
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')


def pin(p):
    p = Path(p)
    with p.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(p), realpath=str(p.resolve()), bytes=p.stat().st_size, sha256=digest)


def main():
    root = B/'main-qsa-bf16-prefix-subgroup-build-v1'
    dst = A/'own-qsa-bf16-prefix-subgroup-build-v1.json'
    assert not root.exists() and not dst.exists()
    build_path = A/'own-qsa-bf16-prefix-tilepack-engine-build-v2.json'
    build = json.loads(build_path.read_text())
    assert build['passed'] and build['complete'] and build['ownership_closed'] and not build['active']
    owner_path = A/'direct_owner_transient_artifacts_v1.py'
    assert pin(owner_path)['sha256'] == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec = importlib.util.spec_from_file_location('subgroup_build_owner', owner_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    compiler = pin(L/'bin/clang++')
    assert compiler['sha256'] == '1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
    origin_path=A/'own-qsa-bf16-prefix-tilepack-build-v1.json'
    origin=json.loads(origin_path.read_text())
    assert origin['passed'] and origin['complete'] and origin['ownership_closed']
    source=Path(origin['candidate_source']['path'])
    assert pin(source)==origin['candidate_source']
    assert pin(source)['sha256']=='058ab2235f318dde4d39537fff93315a38c60a8c331587ab0db63755615add68'
    fixture = P/'tests/sycl/qsa_bf16_prefix.cpp'
    frozen = dict(build['frozen_files'])
    for p in [build_path, owner_path, origin_path, Path(__file__), source, fixture, L/'bin/clang++']:
        frozen[str(p)] = pin(p)
    original = source.read_text()
    cohort_path=A/'qsa-bf16-prefix-tilepack-clean-cohort-v2.json'
    cohort=json.loads(cohort_path.read_text())
    assert cohort['passed'] and cohort['complete'] and cohort['active_controller'] is None and len(cohort['completed'])==6
    frozen[str(cohort_path)]=pin(cohort_path)
    # Fresh experiment, accepted framework reduction contract defined BEFORE run.
    # All sixteen lanes participate for two fixed row waves, including padding.
    # Sum FP32 p BEFORE BF16 operand rounding; all persistent local state stays FP32.
    before_start=original.index('        if (tid < 16) {\n          float m = maximum[tid];')
    before_end=original.index('        for (int i = tid; i < Cells * Dim;',before_start)
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
          // Only lane0 accesses writable row state; broadcasts move values,
          // rather than relying on subgroup collectives as memory fences.
          const float prior = !lane ? maximum[h] : -INFINITY;
          const float old_m = sycl::group_broadcast(sg, prior, 0);
          const float m = sycl::fmax(old_m, tile_max);
          const float p0 = valid0 ? sycl::exp2(s0 - m) : 0.f;
          const float p1 = valid1 ? sycl::exp2(s1 - m) : 0.f;
          const float sum = sycl::reduce_over_group(sg, p0 + p1, sycl::plus<float>());
          probability[h * 16 + c0] = BFloat(p0);
          probability[256 + h * 16 + lane] = BFloat(p1);
          if (!lane) {
            const float old_norm = normalizer[h];
            const float a = old_norm > 0.f ? sycl::exp2(old_m - m) : 0.f;
            bad_head[h] |= score_bad;
            maximum[h] = m; alpha[h] = a;
            normalizer[h] = sycl::fma(old_norm, a, sum);
          }
        }
"""
    changed=original[:before_start]+norm+original[before_end:]
    changed=changed.replace('tilepack_batch_impl','subgroup_batch_impl')
    changed=changed.replace('qsa_bf16_prefix_tilepack_batch','qsa_bf16_prefix_subgroup_batch')
    for p in [A/'qsa-bf16-prefix-subgroup-contract-v1.ja.md',A/'research-review-round955-v1.json']:
      frozen[str(p)]=pin(p)
    host_check=A/'transient-artifact-owner-check-v1.json'
    host=json.loads(host_check.read_text());assert host['passed'] and host['complete'] and host['helper_sha256']==pin(owner_path)['sha256']
    frozen[str(host_check)]=pin(host_check)
    # The lock serializes Main's jobs. A user-owned CPU or GPU benchmark need
    # not share that lock, so also apply the qualified read-only owner guard.
    package=A/'own-vllm-xe2-qualification-v1'
    metadata=package/'manifest.json'
    assert pin(metadata)['sha256']=='7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    meta=json.loads(metadata.read_text());helper=package/'main_controller.py'
    assert pin(helper)['sha256']==meta['package_files']['main_controller.py']['sha256']
    spec=importlib.util.spec_from_file_location('subgroup_resource_guards',helper)
    guards=importlib.util.module_from_spec(spec);spec.loader.exec_module(guards)
    frozen[str(metadata)]=pin(metadata);frozen[str(helper)]=pin(helper)

    patch = ''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                    fromfile='a/private/qsa_bf16_prefix_tilepack.cpp', tofile='b/private/qsa_bf16_prefix_subgroup.cpp'))
    libraries = [Path(build['binary']['path']).parent/n for n in
                 ['libstrata_kernels.a', 'libstrata_core.a', 'libstrata_sycl_runtime.a']]
    # These archives are build OUTPUTS, not entries in its frozen INPUT table.
    # Establish their first observed input pins for this independent fixture.
    library_pins = {str(p): pin(p) for p in libraries}
    frozen.update(library_pins)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for c in build['commands']:
            assert m.closed(c) and c['exit_code'] == 0
            for old_id in c['owners'] + [c['direct_identity']]:
                now = m.identity(old_id['pid'])
                assert now is None or now['start_ticks'] != old_id['start_ticks']
        resource_guard=guards.no_model_owners(meta['known_model_files'])
        root.mkdir(mode=0o700)
        (root/'tmp').mkdir()
        (root/'cache').mkdir()
        candidate = root/'qsa_bf16_prefix_subgroup.cpp'
        candidate.write_text(changed)
        frozen[str(candidate)] = pin(candidate)
        rec = dict(schema='strata-bf16-prefix-subgroup-build-v1', passed=False, complete=False,
                   active=False, ownership_closed=False, GPU_executed=False, commands=[],
                   frozen_files=frozen, compiler=compiler, base_source=pin(source), private_origin_receipt=pin(origin_path), private_origin_source_commit=origin['source_commit'],
                   base_engine_source_commit=build['source_commit'], source_diff=patch,
                   candidate_source=pin(candidate), source_commit=subprocess.check_output(
                       ['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
                   library_inputs=library_pins,
                   resource_guard=resource_guard,
                   library_pin_scope='Freshly observed outputs under the completed current engine build; first hashed here as inputs, not claimed to be prior build-receipt output hashes.',
                   provenance='Main-authored row-owned subgroup FP32 max/sum on independently authored packed BF16 Strata kernel. SGLang source supplies numerical design contract and SYCL spec supplies collective API requirements, no copied implementation/sample code. R954 targeted15–34 used as reviewed decision support. Excluded-project path compiler dependency only.',
                   production_selected=False, performance_adopted=False,
                   scope='Private packed score33 with two fixed head waves/subgroup, FP32 max/sum tree, P rounded only after FP32 sum inputs, FP32 normalizer/alpha/accumulator. Local payload28224, same WG barriers/source loads/selected IDs/matrix math. Reduction tree rounding differs prospectively under pinned SGLang contract; no FP32/llama bit equality gate or speed claim. Compile only, after Main closes current packed model cohort.')
        owner = None
        def persist():
            rec['active'] = bool(owner and owner.active is not None)
            (root/'record.json').write_text(json.dumps(rec, indent=2)+'\n')
        owner = m.Owner(root/'owned', persist)
        rec['commands'] = owner.commands
        def verify():
            for p, value in frozen.items():
                assert pin(p) == value, p
        try:
            verify()
            exe = root/'sycl_qsa_bf16_prefix_subgroup'
            env = dict(PATH=str(L/'bin')+':/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                       PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=str(L/'lib'), TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'))
            argv = [compiler['path'], '-std=c++20', '-O3', '-DNDEBUG', '-DSTRATA_ENABLE_SYCL=1',
                    '-Dqsa_bf16_prefix_batch=qsa_bf16_prefix_subgroup_batch',
                    '-I'+str(P/'include'), '-I'+str(P/'include/strata/sycl/compat'),
                    '-fno-fast-math', '-ffp-contract=off', '-fsycl', '-fsycl-id-queries-range=size_t',
                    '-Wall', '-Wextra', '-Werror', str(candidate), str(fixture), *map(str, libraries), '-o', str(exe)]
            c, out, err = owner.run('compile-link', argv, env, P, wall=300, cpu=900,
                                    rss_cap=8<<30, text_cap=64<<20, file_cap=128<<20, total_cap=1<<30)
            rec.update(exit_code=c['exit_code'], stdout=out.read_text(), stderr=err.read_text())
            assert m.closed(c) and c['exit_code'] == 0
            verify()
            rec.update(binary=pin(exe), passed=True, complete=True)
        except BaseException as error:
            import traceback
            rec.update(error=repr(error), traceback=traceback.format_exc())
        finally:
            rec['ownership_closed'] = owner.active is None and all(m.closed(c) for c in owner.commands)
            persist()
            dst.write_bytes((root/'record.json').read_bytes())
        print(json.dumps(dict(receipt=str(dst), passed=rec['passed'], error=rec.get('error'), GPU_executed=False)))
        return 0 if rec['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
