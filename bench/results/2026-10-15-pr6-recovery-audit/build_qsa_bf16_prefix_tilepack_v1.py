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
    root = B/'main-qsa-bf16-prefix-tilepack-build-v1'
    dst = A/'own-qsa-bf16-prefix-tilepack-build-v1.json'
    assert not root.exists() and not dst.exists()
    build_path = A/'own-qsa-bf16-prefix-optimized-engine-build-v1.json'
    build = json.loads(build_path.read_text())
    assert build['passed'] and build['complete'] and build['ownership_closed'] and not build['active']
    owner_path = P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert pin(owner_path)['sha256'] == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('tilepack_build_owner', owner_path)
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
    cohort_path=A/'qsa-bf16-prefix-optimized-clean-cohort-v1.json'
    cohort=json.loads(cohort_path.read_text())
    assert cohort['passed'] and cohort['complete'] and cohort['active_controller'] is None and len(cohort['completed'])==6
    frozen[str(cohort_path)]=pin(cohort_path)
    # API-defined BF16 VNNI pair packing. The independent mapping checks below
    # verify every destination once, and every operand lane against its semantic
    # source coordinate before compilation or GPU execution.
    q_index=lambda h,d:(d//16)*256+h*16+d%16
    k_index=lambda d,c:((d//16)*2+c//16)*256+(d%16//2)*32+(c%16)*2+d%2
    p_index=lambda h,c:(c//16)*256+h*16+c%16
    v_index=lambda c,d:((c//16)*16+d//16)*256+(c%16//2)*32+(d%16)*2+c%2
    assert sorted(q_index(h,d) for h in range(16) for d in range(256))==list(range(4096))
    assert sorted(k_index(d,c) for d in range(256) for c in range(32))==list(range(8192))
    assert sorted(p_index(h,c) for h in range(16) for c in range(32))==list(range(512))
    assert sorted(v_index(c,d) for c in range(32) for d in range(256))==list(range(8192))
    # Actual loads use an 8x16 row-major A and a 16x16 pair-packed B.
    for hb in [0,8]:
      for kb in range(0,256,16):
        for r in range(8):
          for c in range(16):
            assert q_index(hb+r,kb+c)==(kb//16)*256+hb*16+r*16+c
    for kb in range(0,256,16):
      for cb in [0,16]:
        for r in range(16):
          for c in range(16):
            assert k_index(kb+r,cb+c)==((kb//16)*2+cb//16)*256+(r//2)*32+c*2+r%2
    for hb in [0,8]:
      for kb in [0,16]:
        for r in range(8):
          for c in range(16):
            assert p_index(hb+r,kb+c)==(kb//16)*256+hb*16+r*16+c
    for kb in [0,16]:
      for db in range(0,256,16):
        for r in range(16):
          for c in range(16):
            assert v_index(kb+r,db+c)==((kb//16)*16+db//16)*256+(r//2)*32+c*2+r%2
    changed=original
    replacements=[
      ('if (!device.is_gpu() ||', 'if (!device.has(sycl::aspect::ext_intel_matrix) || !device.is_gpu() ||',1),
      ('qtile[head * Dim + d] = BFloat(value);','qtile[(d / 16) * 256 + head * 16 + d % 16] = BFloat(value);',1),
      ('mx::use::b, 16, 16,\n                                     mx::layout::row_major>',
       'mx::use::b, 16, 16,\n                                     mx::layout::ext_intel_packed>',1),
      ('kv[i] = sycl::bit_cast<BFloat>(bits); // Same local K transpose.',
       'kv[((d / 16) * 2 + cell / 16) * 256 + (d % 16 / 2) * 32 +\n             (cell % 16) * 2 + d % 2] = sycl::bit_cast<BFloat>(bits);',1),
      ('qtile + head * Dim + d, Dim','qtile + (d / 16) * 256 + head * 16, 16',1),
      ('kv + d * Cells + cell, Cells','kv + ((d / 16) * 2 + cell / 16) * 256, 32',1),
      ('probability[tid * Cells + c] = BFloat(p);',
       'probability[(c / 16) * 256 + tid * 16 + c % 16] = BFloat(p);',1),
      ('kv[i] = sycl::bit_cast<BFloat>(bits);',
       'kv[((cell / 16) * 16 + d / 16) * 256 + (cell % 16 / 2) * 32 +\n             (d % 16) * 2 + cell % 2] = sycl::bit_cast<BFloat>(bits);',1),
      ('probability + head * Cells + c, Cells','probability + (c / 16) * 256 + head * 16, 16',1),
      ('kv + c * Dim + dim_base + j * 16, Dim',
       'kv + ((c / 16) * 16 + dim_base / 16 + j) * 256, 32',1)]
    for before,after,count in replacements:
      assert changed.count(before)==count,before
      changed=changed.replace(before,after)
    start=changed.index('bool qsa_bf16_prefix_batch(')
    end=changed.index('bool qsa_bf16_prefix_optimized_batch(',start)
    changed=changed[:start]+changed[end:]
    changed=changed.replace('bool qsa_bf16_prefix_optimized_batch(', 'bool qsa_bf16_prefix_tilepack_batch(')
    changed=changed.replace('prefix_batch_impl','tilepack_batch_impl')
    changed=changed.replace('#include <type_traits>','#include <type_traits>\n#include <cstddef>')
    guard = """static_assert(sizeof(BFloat) == 2);
template<int Count> constexpr bool tilepack_storage_contract() {
  using Tile = MatrixTile<Count>;
  static_assert(std::is_standard_layout_v<Tile> && std::is_trivially_copyable_v<Tile>);
  static_assert(alignof(Tile) >= 4 && offsetof(Tile, values) == 0);
  static_assert(sizeof(Tile) == Count * sizeof(BFloat));
  return true;
}
static_assert(tilepack_storage_contract<16 * Dim>());
static_assert(tilepack_storage_contract<Cells * Dim>());
static_assert(tilepack_storage_contract<16 * Cells>());
static_assert(16 * sizeof(BFloat) % 8 == 0 && 32 * sizeof(BFloat) % 8 == 0);
"""
    seam='static_assert(sizeof(MatrixTile<16 * Dim>) == 16 * Dim * sizeof(BFloat));'
    assert changed.count(seam)==1
    changed=changed.replace(seam,seam+'\n'+guard)
    for name in ['sycl_ext_intel_matrix.asciidoc','sycl_ext_oneapi_matrix.asciidoc']:
      path=A/'own-native16-attention-v1'/('provenance-'+name+'.json')
      frozen[str(path)]=pin(path)
      api=B/('tilepack-api-'+name)
      assert pin(api)['sha256']==json.loads(path.read_text())['sha256']
      frozen[str(api)]=pin(api)

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
        candidate = root/'qsa_bf16_prefix_tilepack.cpp'
        candidate.write_text(changed)
        frozen[str(candidate)] = pin(candidate)
        rec = dict(schema='strata-bf16-prefix-tilepack-build-v1', passed=False, complete=False,
                   active=False, ownership_closed=False, GPU_executed=False, commands=[],
                   frozen_files=frozen, compiler=compiler, base_source=pin(source),
                   base_engine_source_commit=build['source_commit'], source_diff=patch,
                   candidate_source=pin(candidate), source_commit=subprocess.check_output(
                       ['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
                   library_inputs=library_pins,
                   library_pin_scope='Freshly observed outputs under the completed current engine build; first hashed here as inputs, not claimed to be prior build-receipt output hashes.',
                   provenance='Main-authored dense16 A tiles and pair-packed BF16 B tiles on own aligned score33 MIT Strata kernel. Pinned Intel matrix extension defines VNNI layout/API constraints only; no implementation/sample source copied. Excluded-project path is compiler dependency only.',
                   production_selected=False, performance_adopted=False,
                   layout_mapping_bijections_passed=True, matrix_load_semantic_mapping_passed=True, local_payload_bytes=28224, scope='Private score33 candidate: dense16 row-major A tiles Q/P and BF16 pair-packed16x16 B tiles K/V. Original global K and V producer source coordinates, operand values, numerical32cell updates/barriers unchanged. Actual matrix-load mappings checked on host; explicit Intel aspect and full object contracts. No assumed ISA/bank/occupancy/speedup. Compile only.')
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
            exe = root/'sycl_qsa_bf16_prefix_tilepack'
            env = dict(PATH=str(L/'bin')+':/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                       PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=str(L/'lib'), TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'))
            argv = [compiler['path'], '-std=c++20', '-O3', '-DNDEBUG', '-DSTRATA_ENABLE_SYCL=1',
                    '-Dqsa_bf16_prefix_batch=qsa_bf16_prefix_tilepack_batch',
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
