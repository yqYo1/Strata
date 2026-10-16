#!/usr/bin/env python3
"""Source-only package preparation. Never builds, tests, invokes SYCL, or edits live files."""
from pathlib import Path
import hashlib, json, difflib
P=Path(__file__).resolve().parent
ROOT=P.parents[3]
paths=['src/kernels/sycl/qsa_select.cpp','cmake/sycl_backend.cmake']
def sha(b): return hashlib.sha256(b).hexdigest()
# Capture actual source bytes once. Refuse stale package regeneration against live files.
for rel in paths:
    dst=P/'base'/rel
    if not dst.exists():
        b=(ROOT/rel).read_bytes()
        if rel.endswith('qsa_select.cpp') and sha(b)!='45084756ca5e8e36085dae46581049c81366a45cdfde23e8c6f38defbba0c809':
            raise SystemExit('selector base changed; explicit review required')
        dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(b)
src=(P/'base'/paths[0]).read_text()
start=src.index('  auto event = queue_for(stream).parallel_for(',src.index('void qsa_block_topk('))
end=src.index('  finish(stream, event);',start)
launch=src[start:end].replace('  auto event = queue_for(stream).parallel_for(', '  return queue.parallel_for(',1)
old='''        for (int bit = 31; bit >= 0; --bit) {
          const uint32_t candidate = threshold | (1u << bit);
          int count = 0;
          for (int64_t b = lo; b < hi; ++b)
            if (order(sc[b]) >= candidate)
              count += weight(b);
          if (sycl::reduce_over_group(it.get_group(), count,
                                      sycl::plus<int>()) >= width)
            threshold = candidate;
        }
'''
assert old in launch
radix='''        if constexpr (Radix4) {
          // Rank is in cells, including the partial final block. All counts
          // are subsets of nonnegative int32 n, hence fit int without adding
          // a physical-context restriction to the existing API.
          int rank = int(width);
          uint32_t mask = 0;
          for (int shift = 28; shift >= 0; shift -= 4) {
            int histogram[16] = {};
            // Every round loads every key before checking the prefix.
            for (int64_t b = lo; b < hi; ++b) {
              const uint32_t key = order(sc[b]);
              if ((key & mask) == threshold)
                histogram[(key >> shift) & 15u] += weight(b);
            }
            for (int digit = 0; digit < 16; ++digit)
              histogram[digit] = sycl::reduce_over_group(
                  it.get_group(), histogram[digit], sycl::plus<int>());
            int digit = 15;
            while (digit > 0 && rank > histogram[digit]) {
              rank -= histogram[digit];
              --digit;
            }
            threshold |= uint32_t(digit) << shift;
            mask |= 15u << shift;
          }
        } else {
'''+''.join('  '+line+'\n' for line in old.splitlines())+'''        }
'''
launch=launch.replace(old,radix)
helper='''template <bool Radix4>
sycl::event submit_topk(sycl::queue &queue, const float *scores,
                        const int32_t *steps, int64_t nq, int64_t blocks,
                        int64_t cap, int32_t *ids, int64_t launched) {
'''+launch+'''}

bool select_radix4() {
  // Experimental, default off. Set before exec; process environment must
  // remain immutable once runtime/provider initialization starts.
  static const bool enabled = [] {
    const char *value = std::getenv("STRATA_SYCL_SELECT_RADIX4");
    return value && value[0] == '1' && value[1] == '\\0';
  }();
  return enabled;
}
'''
# Replace just the top-k submission, then put helper inside the anonymous namespace.
src=src[:start]+'''  auto &queue = queue_for(stream);
  const auto event = select_radix4()
      ? submit_topk<true>(queue, scores, steps, nq, blocks, cap, ids, launched)
      : submit_topk<false>(queue, scores, steps, nq, blocks, cap, ids, launched);
'''+src[end:]
src=src.replace('#include <climits>','#include <climits>\n#include <cstdlib>')
src=src.replace('} // namespace\nvoid qsa_block_scores',helper+'} // namespace\nvoid qsa_block_scores',1)
dst=P/'files'/paths[0];dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(src)
cm=(P/'base'/paths[1]).read_text()
needle='  # Manual source-only K mapping candidate; Main serializes all GPU runs.\n  add_executable(sycl_prompt_k_cellmajor_test EXCLUDE_FROM_ALL tests/sycl/prompt_k_cellmajor.cpp)'
assert cm.count(needle)==1
cm=cm.replace(needle,'''  # Manual exact selector experiment; use separate immutable-environment processes.
  add_executable(sycl_qsa_select_radix_test EXCLUDE_FROM_ALL tests/sycl/qsa_select_radix.cpp)
  target_link_libraries(sycl_qsa_select_radix_test PRIVATE strata_kernels)
  target_compile_options(sycl_qsa_select_radix_test PRIVATE -fno-fast-math -ffp-contract=off)
'''+needle)
dst=P/'files'/paths[1];dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(cm)
fixture='tests/sycl/qsa_select_radix.cpp'
paths.append(fixture)
patch=''
for rel in paths:
    a=(P/'base'/rel)
    b=P/'files'/rel
    if not b.exists(): raise SystemExit('write fixture before preparing package')
    patch+=''.join(difflib.unified_diff(a.read_text().splitlines(True) if a.exists() else [],b.read_text().splitlines(True),fromfile='a/'+rel if a.exists() else '/dev/null',tofile='b/'+rel))
(P/'proposal.patch').write_text(patch)
files={}
for folder in ('base','files'):
    for f in sorted((P/folder).rglob('*')):
        if f.is_file(): files[str(f.relative_to(P))]={'sha256':sha(f.read_bytes()),'bytes':f.stat().st_size}
files['proposal.patch']={'sha256':sha(patch.encode()),'bytes':len(patch.encode())}
for name in ('README.md','prepare.py'):
    data=(P/name).read_bytes()
    files[name]={'sha256':sha(data),'bytes':len(data)}
references={}
for rel in ('include/strata/kernels/qsa_select.hpp','include/strata/kernels/qsa.hpp',
            'include/strata/sycl/launch.hpp','include/strata/sycl/runtime.hpp',
            'tests/sycl/prompt_k_cellmajor.cpp',
            'bench/results/2026-10-15-pr6-recovery-audit/research-index.json',
            'bench/results/2026-10-15-pr6-recovery-audit/research-exact-selection-radix-round738-v1.txt'):
    data=(ROOT/rel).read_bytes()
    references[rel]={'sha256':sha(data),'bytes':len(data),'role':'read-only reference; not a patch base'}
snapshot=''.join(v['sha256']+'  '+k+'\n' for k,v in sorted(files.items()))
(P/'SHA256SUMS').write_text(snapshot)
manifest={'qualification':{'build':'UNRUN','correctness':'UNRUN','performance':'UNRUN','model':'UNRUN','physical_context':'UNRUN','restore':'UNRUN','fault_teardown':'UNRUN'},'base_kind':'actual-byte source snapshot; never inferred from HEAD','base_files':paths[:2],'new_files':[fixture],'files':files,'snapshot_sha256':sha(snapshot.encode()),'origin':'Independent implementation from own current selector and weighted MSD algorithm reasoning; no XeStrata source inspected or used.','scope':'source-only proposal; active files untouched; no build/test/GPU/runtime/environment/service/cleanup/Git mutation','owner':'Main reviews patch before apply and owns serialized qualification','next_review':'Main source review then build and exact fixture in separate immutable-environment processes','load_count':'8*B histogram key visits + 2*B final key visits; no prefix-distribution reduction without compaction','count_bound':'0 <= weighted subset <= n <= INT_MAX; no new arbitrary context validation','references':references}
(P/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'package':str(P),'snapshot_sha256':manifest['snapshot_sha256'],'patch_sha256':files['proposal.patch']['sha256']},indent=2))
