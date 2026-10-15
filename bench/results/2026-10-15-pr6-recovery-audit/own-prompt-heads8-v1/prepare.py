#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate proposal bytes/diff only. Never applies, builds, tests or runs GPU work."""
from pathlib import Path
import hashlib, json, difflib
ROOT=Path(__file__).resolve().parents[4]
PKG=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
paths=['src/kernels/sycl/prompt_attention.cpp','tests/sycl/prompt_k_cellmajor.cpp','cmake/sycl_backend.cmake','LICENSE']
# A rerun consumes the immutable package base, never changed live files.
base={p:(PKG/'base'/p).read_bytes() for p in paths}
s=base[paths[0]].decode()
s=s.replace('int KMapping = 0>','int KMapping = 0, int HRows = 16>')
s=s.replace('  static_assert(KMapping >=', '  static_assert(HRows == 16 || (HRows == 8 && Mode == 1 && Direct && !Native && KMapping == 0));\n  constexpr int WG = HRows * 8, HTiles = HRows == 8 ? 2 : 1;\n  static_assert(KMapping >=',1)
start=s.index('  auto event =');end=s.index('// Select distinct kernels')
a=s[start:end]
for x,y in [('16 * HD','HRows * HD'),('16 * CH','HRows * CH'),('maxima(16','maxima(HRows'),('sums(16','sums(HRows'),('alpha(16','alpha(HRows'),('tid < 16','tid < HRows'),('% 16, c = i % CH','% HRows, c = i % CH'),('(vg * 16 +','(vg * HRows +')]: a=a.replace(x,y)
a=a.replace('shapes.n_head_kv * WG','shapes.n_head_kv * HTiles * WG')
a=a.replace('batch = index / shapes.n_head_kv,\n                       kvhead = index % shapes.n_head_kv;', 'batch = index / (shapes.n_head_kv * HTiles),\n                       kvhead = (index / HTiles) % shapes.n_head_kv;\n          const int headbase = (index % HTiles) * HRows;')
a=a.replace('const float x = i < HEADS * HD ? q[i] * qup : 0.f;', 'const int qi = headbase * HD + i;\n            const float x = qi < HEADS * HD ? q[qi] * qup : 0.f;')
a=a.replace('subgroup < 4','subgroup < HRows / 4')
a=a.replace('if (hb * 8 + r < HEADS)','if (headbase + hb * 8 + r < HEADS)')
a=a.replace('kvhead * HEADS + hb * 8 + r','kvhead * HEADS + headbase + hb * 8 + r')
s=s[:start]+a+s[end:]
needle='  const char *direct = std::getenv("STRATA_SYCL_PROMPT_DIRECT");'
s=s.replace(needle,needle+'''
  if constexpr (Mode == 1) {
    const char *heads8 = std::getenv("STRATA_SYCL_PROMPT_HEADS8");
    if (heads8 && heads8[0] == '1' && heads8[1] == '\\0') {
      const char *mapping = std::getenv("STRATA_SYCL_PROMPT_K_CELL_MAJOR");
      if (!(direct && direct[0] == '1') ||
          (mapping && (mapping[0] == '1' || mapping[0] == '2') && mapping[1] == '\\0'))
        throw std::invalid_argument("prompt heads8 requires Mode1 Direct1 KMapping0");
      launch<1, true, false, 0, 8>(q, p, ids, steps, cap, s, out, nq, stream);
      return;
    }
  }''')
# Padded rows have zero probabilities, and no global stores in the candidate.
s=s.replace("rows[c] >= 0 ? sycl::exp(scores[tid * CH + c] - m) : 0.f;", "rows[c] >= 0 && (HRows == 16 || headbase + tid < HEADS)\n                        ? sycl::exp(scores[tid * CH + c] - m) : 0.f;")
# fixture extension is based solely on current own fixture, separate target.
f=base[paths[1]].decode()
f='// SPDX-License-Identifier: MIT\n// Own fixture extension; source-only, Main must review and qualify.\n'+f
f=f.replace('int benchmark_nq = 0;', 'int benchmark_nq = 0;\nint case_nq = 3, case_width = 2051;')
f=f.replace('benchmark_nq ? benchmark_nq : 3','benchmark_nq ? benchmark_nq : case_nq')
f=f.replace('std::vector<int> counts(nq, cap);\n  counts[0] = 13;\n  counts[1] = 67;', 'std::vector<int> counts(nq, benchmark_nq ? cap : case_width);')
f=f.replace('if (scaled)\n      ids[b * cap + 5] = -1;', 'ids[b * cap + 0] = -1;\n    ids[b * cap + 5] = -1;')
# Keep width1 live, while testing negative IDs at width>=31.
f=f.replace('    ids[b * cap + 0] = -1;\n','')
f=f.replace('const int count0', 'const int count0')
f=f.replace('std::vector<double> weights(counts[b]);','const int count = counts[b] >= 0 && counts[b] <= cap ? counts[b] : 0;\n      std::vector<double> weights(count);')
f=f.replace('j < counts[b]','j < count')
f=f.replace('out(query.size() + 16)', 'out(query.size() + 32)')
f=f.replace('out.put(std::vector<float>(query.size() + 16, 1234.f));','out.put(std::vector<float>(query.size() + 32, 1234.f));')
f=f.replace('out.data(), nq','out.data() + 16, nq')
f=f.replace('  auto y = out.get();','''  auto read_output = [&]() {
    auto all = out.get();
    for (int i=0;i<16;++i) check(all[i] == 1234.f, "output prefix guard");
    return std::vector<float>(all.begin()+16, all.end());
  };
  auto y = read_output();''')
f=f.replace('out.get() == y','read_output() == y').replace('auto zero = out.get();','auto zero = read_output();').replace('out.get() == zero','read_output() == zero')
f=f.replace('limited_oracle_queries=0,1,2,last','limited_oracle_queries=0,1,2,last')
f=f.replace('cap=2051 counts=13,67,then_all2051','cap=2051 counts=all2051')
# Every async borrowed-host transfer either drains successfully or fail-stops.
f=f.replace('    runtime->wait(\n        runtime->compute().memcpy(data(), v.data(), v.size() * sizeof(T)));','    try { runtime->wait(\n        runtime->compute().memcpy(data(), v.data(), v.size() * sizeof(T))); } catch (...) { std::abort(); }')
f=f.replace('    runtime->wait(runtime->compute().memcpy(v.data(), data(), mem.size()));','    try { runtime->wait(runtime->compute().memcpy(v.data(), data(), mem.size())); } catch (...) { std::abort(); }')
f=f.replace('catch (...) {}','catch (...) { std::abort(); }')
# Inputs must be unchanged after all repeated launches, before any deliberate fixture mutation.
needle='  if (benchmark_nq) {\n    std::cout'
imm='''  check(q.get() == query && di.get() == ids && ds.get() == steps &&
        dt.get() == table && dk8.get() == k8 && dv8.get() == v8 &&
        dks.get() == ks && dvs.get() == vs, "input immutability");
'''
f=f.replace(needle,imm+needle)
f=f.replace('(scaled ? (i / (heads * dim) == 0 ? 64.f : 0x1p-20f) : 1.f)', '(scaled ? (((i/dim)%12 < 8) ? 64.f : 0x1p-20f) : 1.f)')
# Unrelated legacy decode rejection allocations are excluded from this dedicated fixture.
begin=f.index('    s = qsa_real_shapes();\n    Buffer<float> q(')
end=f.index('    runtime->wait();',begin)
f=f[:begin]+f[end:]
# Initial result dump includes guards, and all small case initial results.
f=f.replace('''      for (int fmt = 0; fmt < 4; ++fmt)
        for (int page : {1, 4, 16})
          run(fmt, page, page != 1);
      for (int fmt = 0; fmt < 4; ++fmt)
        run(fmt, 16, true, true);''','''      for (case_nq : {1,3}) {} // replaced below
''')
f=f.replace('      for (case_nq : {1,3}) {} // replaced below', '''      for (int n : {1,3}) for (int width : {0,1,31,32,33,2051,-1,2052}) {
        case_nq=n; case_width=width;
        for (int page : {1,4,16}) for (bool mask : {false,true}) run(1,page,mask,true);
      }''')
f=f.replace('    return std::vector<float>(all.begin()+16, all.end());', '    return std::vector<float>(all.begin()+16, all.end());')
# Dump every checked output, including prefix/tail guards; never inside timing.
old_dump='  if (dump.is_open()) {\n    dump.write(reinterpret_cast<const char *>(y.data()), y.size()*sizeof(float));\n    check(bool(dump), "output dump failed");\n  }\n'
f=f.replace(old_dump,'')
f=f.replace('    return std::vector<float>(all.begin()+16, all.end());', '    if (dump.is_open()) {\n      dump.write(reinterpret_cast<const char *>(all.data()), all.size()*sizeof(float));\n      check(bool(dump), "full output dump failed");\n    }\n    return std::vector<float>(all.begin()+16, all.end());')
# Explicit logical coverage proof for both work-group geometries.
f=f.replace('    logical_tile();','''    for (int hr : {8,16}) {
      std::vector<int> coverage(12*256);
      for(int tile=0;tile<(hr==8?2:1);++tile)
        for(int sg=0;sg<hr/2;++sg) for(int j=0;j<4;++j)
          for(int r=0;r<8;++r) for(int c=0;c<16;++c) {
            int head=tile*hr+(sg/4)*8+r, d=((sg%4)*4+j)*16+c;
            if(head<12) ++coverage[head*256+d];
          }
      for(int writes:coverage) check(writes==1,"head8+4 unique complete output coverage");
    }
    logical_tile();''')
c=base[paths[2]].decode();needle='  add_executable(sycl_prompt_attention_test'
pos=c.index(needle)
c=c[:pos]+'''  # Separate opt-in heads8 fixture; baseline target unchanged.
  add_executable(sycl_prompt_heads8_test EXCLUDE_FROM_ALL tests/sycl/prompt_heads8.cpp)
  target_link_libraries(sycl_prompt_heads8_test PRIVATE strata_kernels)
  target_compile_definitions(sycl_prompt_heads8_test PRIVATE STRATA_TEST_PROMPT_MATRIX=1)
  target_compile_options(sycl_prompt_heads8_test PRIVATE -fno-fast-math -ffp-contract=off)
'''+c[pos:]
new={paths[0]:s,'tests/sycl/prompt_heads8.cpp':f,paths[2]:c}
patch=''
for p,t in new.items():
 old=base[p].decode() if p in base else ''
 patch+=''.join(difflib.unified_diff(old.splitlines(True),t.splitlines(True),fromfile='a/'+p if old else '/dev/null',tofile='b/'+p))
 dst=PKG/'files'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(t)
(PKG/'source.diff').write_text(patch)
manifest={'base_commit':'617647659f21487ff3bf61fa1fd4ba87ba2520bf','license':'MIT (root LICENSE; no skill metadata copied)','status':'SOURCE ONLY; all builds/tests/GPU UNRUN','base_sha256':{p:sha(b) for p,b in base.items()},'candidate_sha256':{p:sha(t.encode()) for p,t in new.items()},'patch_sha256':sha(patch.encode()),'origin':'Own current prompt_attention.cpp and prompt_k_cellmajor.cpp only; independent heads8 edits; no XeStrata implementation/source consulted'}
(PKG/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
