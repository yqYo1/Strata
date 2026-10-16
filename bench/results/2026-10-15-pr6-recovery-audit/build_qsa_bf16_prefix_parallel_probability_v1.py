#!/usr/bin/env python3
"""Compile same-order parallel probability experiment; preserve production tree."""
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
    root = B/'main-qsa-bf16-prefix-parallel-probability-build-v1'
    dst = A/'own-qsa-bf16-prefix-parallel-probability-build-v1.json'
    assert not root.exists() and not dst.exists()
    build_path = A/'own-qsa-bf16-prefix-engine-build-v1.json'
    build = json.loads(build_path.read_text())
    assert build['passed'] and build['complete'] and build['ownership_closed'] and not build['active']
    owner_path = P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert pin(owner_path)['sha256'] == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('parallel_probability_build_owner', owner_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    compiler = pin(L/'bin/clang++')
    assert compiler['sha256'] == '1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
    source = P/'src/kernels/sycl/qsa_bf16_prefix.cpp'
    fixture = P/'tests/sycl/qsa_bf16_prefix.cpp'
    frozen = dict(build['frozen_files'])
    for p in [build_path, owner_path, Path(__file__), source, fixture, L/'bin/clang++']:
        frozen[str(p)] = pin(p)
    original = source.read_text()
    old = '          const float a = normalizer[tid] > 0.f ? sycl::exp2(maximum[tid] - m) : 0.f;\n          float sum = 0.f;\n          for (int c = 0; c < Cells; ++c) {\n            const float score = scores[tid * Cells + c];\n            const float p = rows[c] >= 0 && sycl::isfinite(score)\n                ? sycl::exp2(score - m) : 0.f;\n            sum += p;\n            probability[tid * Cells + c] = BFloat(p);\n          }\n          maximum[tid] = m; alpha[tid] = a;\n          normalizer[tid] = sycl::fma(normalizer[tid], a, sum);\n        }'
    new = '          const float a = normalizer[tid] > 0.f ? sycl::exp2(maximum[tid] - m) : 0.f;\n          maximum[tid] = m; alpha[tid] = a;\n        }\n        // All row maxima must be consumed before reusing scores as FP32 p.\n        item.barrier(sycl::access::fence_space::local_space);\n        for (int i = tid; i < 16 * Cells; i += Workgroup) {\n          const int head = i / Cells, cell = i % Cells;\n          const float score = scores[i];\n          const float p = rows[cell] >= 0 && sycl::isfinite(score)\n              ? sycl::exp2(score - maximum[head]) : 0.f;\n          scores[i] = p;\n          probability[i] = BFloat(p);\n        }\n        item.barrier(sycl::access::fence_space::local_space);\n        if (tid < 16) {\n          // Preserve the original FP32 sum order and one explicit fma/update.\n          float sum = 0.f;\n          for (int c = 0; c < Cells; ++c) sum += scores[tid * Cells + c];\n          normalizer[tid] = sycl::fma(normalizer[tid], alpha[tid], sum);\n        }'
    assert original.count(old) == 1
    changed = original.replace(old, new)
    patch = ''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                    fromfile='a/src/kernels/sycl/qsa_bf16_prefix.cpp', tofile='b/src/kernels/sycl/qsa_bf16_prefix.cpp'))
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
        root.mkdir(mode=0o700)
        (root/'tmp').mkdir()
        (root/'cache').mkdir()
        candidate = root/'qsa_bf16_prefix_parallel_probability.cpp'
        candidate.write_text(changed)
        frozen[str(candidate)] = pin(candidate)
        rec = dict(schema='strata-bf16-prefix-parallel-probability-build-v1', passed=False, complete=False,
                   active=False, ownership_closed=False, GPU_executed=False, commands=[],
                   frozen_files=frozen, compiler=compiler, base_source=pin(source),
                   base_engine_source_commit=build['source_commit'], source_diff=patch,
                   candidate_source=pin(candidate), source_commit=subprocess.check_output(
                       ['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
                   library_inputs=library_pins,
                   library_pin_scope='Freshly observed outputs under the completed current engine build; first hashed here as inputs, not claimed to be prior build-receipt output hashes.',
                   provenance='Main-authored same-order probability work distribution in own MIT Strata kernel; no external project source. Installed toolchain dependency only at excluded-project path.',
                   production_selected=False, performance_adopted=False,
                   scope='Keep current K mapping, BF16 Q/K/V/P, FP32 matrix/online-state math and sequential32value sum/fma order. Distribute p exp2 from16 to128 threads, reuse FP32 score SLM as unrounded p after all maxima consumed. Two additional uniform local barriers per tile, no extra allocation/SLM, no speedup assumed. Compile only; no GPU/model execution.')
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
            exe = root/'sycl_qsa_bf16_prefix_parallel_probability'
            env = dict(PATH=str(L/'bin')+':/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                       PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=str(L/'lib'), TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'))
            argv = [compiler['path'], '-std=c++20', '-O3', '-DNDEBUG', '-DSTRATA_ENABLE_SYCL=1',
                    '-Dqsa_bf16_prefix_batch=qsa_bf16_prefix_parallel_probability_batch',
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
