from pathlib import Path
import subprocess, hashlib,json,difflib
root=Path.cwd(); dest=root/'bench/results/2026-10-15-pr6-recovery-audit/own-prompt-k-cellmajor-v2'; base='3d873f9ca050db105122b8b18c8a8e1275aecdde'
def original(p):return subprocess.check_output(['git','show',base+':'+p]).decode()
p='src/kernels/sycl/prompt_attention.cpp'; old=original(p); new=old.replace('template <int Mode, bool Direct, bool Native = false>','template <int Mode, bool Direct, bool Native = false, bool CellMajor = false>')
new=new.replace('  constexpr int KG = scale_groups<Mode, false>(),','  static_assert(!CellMajor || (Mode == 1 && Direct && !Native));\n  constexpr int KG = scale_groups<Mode, false>(),',1)
a='''              const int dim = i / CH, c = i % CH;
              kv[i] = rows[c] >= 0
                          ? half_code<Mode, false, Native>(pools, rows[c], dim)
                          : sycl::half(0.f);'''
b='''              if constexpr (CellMajor) {
                // Read consecutive INT8 dimensions; retain the exact QK tile.
                const int c = i / HD, dim = i % HD;
                kv[dim * CH + c] = rows[c] >= 0
                    ? half_code<Mode, false, Native>(pools, rows[c], dim)
                    : sycl::half(0.f);
              } else {
                const int dim = i / CH, c = i % CH;
                kv[i] = rows[c] >= 0
                            ? half_code<Mode, false, Native>(pools, rows[c], dim)
                            : sycl::half(0.f);
              }'''
assert a in new; new=new.replace(a,b,1)
a='''  if (direct && direct[0] == '1') {
    if constexpr (Mode == 0) {'''
b='''  if (direct && direct[0] == '1') {
    if constexpr (Mode == 1) {
      // Fixed by the owning process before exec; do not mutate runtime env.
      const char *cell_major = std::getenv("STRATA_SYCL_PROMPT_K_CELL_MAJOR");
      if (cell_major && cell_major[0] == '1' && cell_major[1] == '\\0') {
        launch<1, true, false, true>(q, p, ids, steps, cap, s, out, nq, stream);
        return;
      }
    }
    if constexpr (Mode == 0) {'''
assert a in new;new=new.replace(a,b,1)
files={p:(old,new)}
p='tests/sycl/prompt_k_cellmajor.cpp'; old=original('tests/sycl/decode_attention.cpp');new=old
new=new.replace('#include <iostream>','#include <iostream>\n#include <fstream>\n#include <chrono>')
new=new.replace('std::shared_ptr<sycl_backend::Runtime> runtime;','std::shared_ptr<sycl_backend::Runtime> runtime;\nstd::ofstream dump;\nint repetitions = 0;')
start=new.index('#ifdef STRATA_TEST_PROMPT_MATRIX\n  setenv(');end=new.index('#endif',start)+len('#endif');new=new[:start]+new[end:]
start=new.index('#ifdef STRATA_TEST_PROMPT_MATRIX\n    setenv(');end=new.index('#endif',start)+len('#endif');new=new[:start]+new[end:]
new=new.replace('constexpr int cells = 141, cap = 133','constexpr int cells = 2113, cap = 2051').replace('{13, 67, 129}','{13, 67, 2051}')
needle='  auto y = out.get();'
new=new.replace(needle,needle+'''
  if (dump.is_open()) {
    dump.write(reinterpret_cast<const char *>(y.data()), y.size()*sizeof(float));
    check(bool(dump), "output dump failed");
  }
  if (repetitions) {
    runtime->wait();
    const auto begin = std::chrono::steady_clock::now();
    for (int r = 0; r < repetitions; ++r)
      attention(q.data(), p, di.data(), ds.data(), cap, s, scratch.data(),
                out.data(), nq);
    runtime->wait();
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    check(out.get() == y, "repeated output exact");
    std::cout << "bounded_component nq=3 cap=2051 counts=13,67,2051 repeats="
              << repetitions << " total_ms=" << ms << "\\n";
  }
''',1)
new=new.replace('int main() {','''int main(int argc, char **argv) {''')
new=new.replace('    runtime = sycl_backend::runtime_for();','''    // No process-environment writes, including before/after runtime creation.
    // Optional output dump is compared byte-for-byte across separate arms.
    check(argc == 1 || argc == 3 || argc == 4, "usage: fixture [--dump PATH [repeats1..20]]");
    if (argc >= 3) {
      check(std::string(argv[1]) == "--dump", "expected --dump");
      dump.open(argv[2], std::ios::binary | std::ios::trunc);
      check(bool(dump), "cannot open dump");
    }
    if (argc == 4) {
      char *end = nullptr;
      const long n = std::strtol(argv[3], &end, 10);
      check(end != argv[3] && *end == '\\0' && n >= 1 && n <= 20, "invalid repeats");
      repetitions = int(n);
    }
    runtime = sycl_backend::runtime_for();''')
new=new.replace('''    for (int fmt = 0; fmt < 4; ++fmt)
      for (int page : {1, 4, 16})
        run(fmt, page, page != 1);
    for (int fmt = 0; fmt < 4; ++fmt)
      run(fmt, 16, true, true);''','''    if (repetitions) {
      run(1, 4, false);
    } else {
      for (int fmt = 0; fmt < 4; ++fmt)
        for (int page : {1, 4, 16})
          run(fmt, page, page != 1);
      for (int fmt = 0; fmt < 4; ++fmt)
        run(fmt, 16, true, true);
    }''')
# Ensure exceptions drain before owners inside run leave scope: run allocations' queue work is waited
# by get() on normal path; catch in main drains pending work, allocation destructor also owns event behavior.
new=new.replace('''    std::cerr << e.what() << "\\n";''','''    if (runtime) { try { runtime->wait(); } catch (...) {} }
    std::cerr << e.what() << "\\n";''')
# Scope guard constructed AFTER all allocations and before first attention, drains before owners unwind.
new=new.replace('  attention(q.data(), p, di.data(), ds.data(), cap, s, scratch.data(),\n            out.data(), nq);','''  struct Drain {
    ~Drain() { try { runtime->wait(); } catch (...) {} }
  } drain; // Destroyed before any allocation owner, including exceptional paths.
  attention(q.data(), p, di.data(), ds.data(), cap, s, scratch.data(),
            out.data(), nq);''',1)
assert 'setenv(' not in new and 'unsetenv(' not in new
new=new.replace('int main(int argc, char **argv) {', r'''// Independent logical-tile proof: each map must write every [dim,slot] once.
void logical_tile() {
  constexpr int HD=256, CH=32, WG=128, page_size=4, kvheads=2;
  const int table[] = {3, -1, 0, 2};
  std::vector<int8_t> pool(4*page_size*kvheads*HD);
  for (size_t i=0;i<pool.size();++i) pool[i]=int8_t(int(i*7%255)-127);
  for (int count : {0,1,13,31,32}) for (int head=0;head<kvheads;++head) {
    int64_t rows[CH];
    for (int c=0;c<CH;++c) {
      const int cell = c==5 ? -1 : (c*7)%16;
      const int page = cell<0 ? -1 : table[cell/page_size];
      rows[c] = c<count && page>=0
          ? (page*kvheads+head)*page_size+cell%page_size : -1;
    }
    std::vector<uint16_t> old(HD*CH), candidate(HD*CH);
    std::vector<int> writes(HD*CH);
    for (int tid=0;tid<WG;++tid) for(int i=tid;i<HD*CH;i+=WG) {
      const int d=i/CH,c=i%CH;
      old[i]=half(rows[c]>=0 ? float(pool[rows[c]*HD+d]) : 0.f);
      const int nc=i/HD,nd=i%HD;
      candidate[nd*CH+nc]=half(rows[nc]>=0 ? float(pool[rows[nc]*HD+nd]) : 0.f);
      ++writes[nd*CH+nc];
    }
    check(old==candidate,"logical K tile exact");
    for(int w:writes) check(w==1,"logical tile unique complete coverage");
  }
}
int main(int argc, char **argv) {''')
new=new.replace('    runtime = sycl_backend::runtime_for();','    logical_tile();\n    runtime = sycl_backend::runtime_for();')
# V2 changes only the fixture performance path; kernel and CMake remain v1 bytes.
new=new.replace('#include <chrono>', '#include <chrono>\n#include <string>')
new=new.replace('int repetitions = 0;', 'int repetitions = 0;\nint benchmark_nq = 0;')
new=new.replace('constexpr int cells = 2113, cap = 2051, nq = 3, heads = 24, kvheads = 2,\n                dim = 256;',
'''constexpr int cells = 2113, cap = 2051, heads = 24, kvheads = 2,
                dim = 256;
  const int nq = benchmark_nq ? benchmark_nq : 3;''')
new=new.replace('const int counts[nq] = {13, 67, 2051};',
'''std::vector<int> counts(nq, cap);
  counts[0] = 13;
  counts[1] = 67;''')
new=new.replace('~Drain() { try { runtime->wait(); } catch (...) {} }',
'~Drain() { try { runtime->wait(); } catch (...) { std::abort(); } }')
new=new.replace('    std::cout << "bounded_component nq=3 cap=2051 counts=13,67,2051 repeats="',
'''    std::cout << "bounded_component nq=" << nq
              << " cap=2051 counts=13,67,then_all2051 limited_oracle_queries=0,1,2,last repeats="''')
start=new.index('  for (int b = 0; b < nq; ++b)\n    attention(q.data() + b * heads * dim')
end=new.index('  double worst = 0;', start)
oldblock=new[start:end]
new=new[:start]+'  if (!benchmark_nq) {\n'+oldblock+'  }\n'+new[end:]
new=new.replace('''  for (int b = 0; b < nq; ++b)
    for (int h = 0; h < heads; ++h) {''','''  for (size_t i = 0; i < query.size(); ++i)
    check(std::isfinite(y[i]), "full output finite");
  for (int b = 0; b < nq; ++b) {
    if (benchmark_nq && b != 0 && b != 1 && b != 2 && b != nq - 1)
      continue; // Benchmark-only limited scalar oracle; all outputs dumped.
    for (int h = 0; h < heads; ++h) {''')
new=new.replace('  for (size_t i = query.size(); i < y.size(); ++i)',
'  }\n  for (size_t i = query.size(); i < y.size(); ++i)',1)
new=new.replace('  // Reuse the same buffers with all pages missing, then with zero widths.',
'''  if (benchmark_nq) {
    std::cout << "benchmark oracle=FP64_queries_0_1_2_last max_abs=" << worst
              << " full_finite=PASS tail_canaries=PASS cross_arm_exact=EXTERNAL_REQUIRED\\n";
    return;
  }
  // Reuse the same buffers with all pages missing, then with zero widths.''')
start=new.index('    check(argc == 1 || argc == 3 || argc == 4,')
end=new.index('    logical_tile();',start)
new=new[:start]+'''    const bool bench = argc == 6 && std::string(argv[1]) == "--bench";
    check(argc == 1 || argc == 3 || bench,
          "usage: fixture [--dump PATH] | --bench 128|256 repeats1..20 --dump PATH");
    auto integer = [](const char *text) {
      char *end = nullptr;
      const long n = std::strtol(text, &end, 10);
      check(end != text && *end == '\\0', "invalid integer");
      return n;
    };
    if (bench) {
      const long n = integer(argv[2]);
      check(n == 128 || n == 256, "benchmark nq must be 128 or 256");
      benchmark_nq = int(n);
      const long r = integer(argv[3]);
      check(r >= 1 && r <= 20, "benchmark repeats must be 1..20");
      repetitions = int(r);
      check(std::string(argv[4]) == "--dump", "benchmark requires --dump");
    }
    if (argc != 1) {
      if (!bench) check(std::string(argv[1]) == "--dump", "expected --dump");
      dump.open(argv[bench ? 5 : 2], std::ios::binary | std::ios::trunc);
      check(bool(dump), "cannot open dump");
    }
'''+new[end:]
new=new.replace('    runtime->wait();\n  } catch (const std::exception &e) {',
'''    runtime->wait();
    if (dump.is_open()) {
      dump.flush();
      check(bool(dump), "output dump flush failed");
      dump.close();
      check(!dump.fail(), "output dump close failed");
    }
  } catch (const std::exception &e) {''')
assert 'setenv(' not in new and 'unsetenv(' not in new
files[p]=('',new)
p='cmake/sycl_backend.cmake';old=original(p);needle='  add_executable(sycl_prompt_attention_test tests/sycl/decode_attention.cpp)';new=old.replace(needle,'''  # Manual source-only K mapping candidate; Main serializes all GPU runs.
  add_executable(sycl_prompt_k_cellmajor_test EXCLUDE_FROM_ALL tests/sycl/prompt_k_cellmajor.cpp)
  target_link_libraries(sycl_prompt_k_cellmajor_test PRIVATE strata_kernels)
  target_compile_definitions(sycl_prompt_k_cellmajor_test PRIVATE STRATA_TEST_PROMPT_MATRIX=1)
  target_compile_options(sycl_prompt_k_cellmajor_test PRIVATE -fno-fast-math -ffp-contract=off)
'''+needle);files[p]=(old,new)
patch='';manifest={'base_commit':base,'status':'SOURCE_ONLY_UNBUILT_UNTESTED','files':{}}
for p,(old,new) in files.items():
 target=dest/'files'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(new)
 patch+=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+p if old else '/dev/null',tofile='b/'+p))
 manifest['files'][p]={'base_sha256':hashlib.sha256(old.encode()).hexdigest() if old else None,'proposal_sha256':hashlib.sha256(new.encode()).hexdigest()}
manifest['v1_kernel_identical'] = manifest['files']['src/kernels/sycl/prompt_attention.cpp']['proposal_sha256'] == 'a22ae0834ead98e0b129b3f31a15cb06db99fe855e96b03c10f216df713f07eb'
manifest['v1_cmake_identical'] = manifest['files']['cmake/sycl_backend.cmake']['proposal_sha256'] == '2544f5a426f3565ade25f3860c0f6317d1770fdc9e225cb0439ff5218ede14d2'
(dest/'source.diff').write_text(patch);manifest['diff_sha256']=hashlib.sha256(patch.encode()).hexdigest();(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
