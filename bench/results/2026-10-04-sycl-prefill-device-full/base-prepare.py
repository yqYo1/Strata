# Prepare only; do not modify the worktree while a performance run is active.
from pathlib import Path
import subprocess,sys
source_ref=sys.argv[1] if len(sys.argv)>1 else "HEAD"
header='#include "/tmp/strata-sycl-goal-prefill-device-trace.hpp"\n'
p=Path('src/prefill/prefill.cpp');s=subprocess.check_output(["git","show",f"{source_ref}:{p}"]).decode();s='#if defined(STRATA_ENABLE_SYCL)\n'+header+'#endif\n'+s
anchor='bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {\n';assert s.count(anchor)==1
s=s.replace(anchor,anchor+'#if defined(STRATA_ENABLE_SYCL)\n    strata_prefill_trace::Scope device_trace;\n#endif\n')
Path('/tmp/strata-sycl-goal-prefill-device-trace-prefill.cpp').write_text(s)
p=Path('src/sycl/compat.cpp');s=subprocess.check_output(["git","show",f"{source_ref}:{p}"]).decode();s=header+s
anchor='''    if (bytes)
      queue(st).memcpy(dst, src, bytes);
''';assert s.count(anchor)==1
s=s.replace('cudaMemcpyKind, cudaStream_t st) {','cudaMemcpyKind kind, cudaStream_t st) {',1)
s=s.replace(anchor,'''    if (bytes) {
      const auto start=strata_prefill_trace::Clock::now();
      const auto event=queue(st).memcpy(dst, src, bytes);
      strata_prefill_trace::put("copy", event, start, bytes, int(kind));
    }
''')
Path('/tmp/strata-sycl-goal-prefill-device-trace-compat.cpp').write_text(s)
p=Path('src/kernels/sycl/native_mmq_xmx.cpp');s=subprocess.check_output(["git","show",f"{source_ref}:{p}"]).decode();s=header+s
anchor='  queue_for(stream).parallel_for(\n';assert s.count(anchor)==1
s=s.replace(anchor,'  const auto start=strata_prefill_trace::Clock::now();\n  const auto event=queue_for(stream).parallel_for(\n')
anchor='      });\n}\ntemplate <int Type, int Tile, bool Exact, bool Packed>';assert s.count(anchor)==1
s=s.replace(anchor,'      });\n  strata_prefill_trace::put("xmx",event,start,0,Type,p.rows,p.cols,p.max_rows);\n}\ntemplate <int Type, int Tile, bool Exact, bool Packed>')
Path('/tmp/strata-sycl-goal-prefill-device-trace-xmx.cpp').write_text(s)

p=Path('/tmp/strata-sycl-goal-prefill-device-trace-prefill.cpp');s=p.read_text()
s=s.replace('    strata_prefill_trace::Scope device_trace;',"""    strata_prefill_trace::Scope device_trace;
    std::unique_ptr<std::FILE, decltype(&std::fclose)> shape_trace(
        device_trace.path ? std::fopen((std::string(device_trace.path) + ".shapes.tsv").c_str(), "a") : nullptr, &std::fclose);""")
anchor='                                for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);'
assert s.count(anchor)==1
s=s.replace(anchor,anchor+"""
#if defined(STRATA_ENABLE_SYCL)
                                if (shape_trace) {
                                    std::fprintf(shape_trace.get(), "%lld\\t%lld\\t%zu\\t%d\\t%lld\\t%lld\\t%d\\t%d", (long long)T, (long long)l, j0, ngx, (long long)nr, (long long)maxr, mmq_gt, mmq_dt);
                                    for(size_t i=j0;i<=j;++i) std::fprintf(shape_trace.get(), "\\t%lld", (long long)m.cnt[(size_t)order[i]]);
                                    std::fprintf(shape_trace.get(), "\\n");
                                }
#endif""")
p.write_text(s)
