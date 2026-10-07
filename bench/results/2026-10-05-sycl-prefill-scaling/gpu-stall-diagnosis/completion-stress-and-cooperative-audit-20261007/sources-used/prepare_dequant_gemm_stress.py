"""Derive a private diagnostic fixture adding unchanged production GEMM/SwiGLU consumers."""
from pathlib import Path
base=Path(__file__).parent
original=(base/'dequant_completion_stress.cpp').read_text()
extra=r'''
class Pipeline {
public:
    static constexpr int64_t rows=8;
    sycl::queue& q;
    sycl::half *x=nullptr,*h=nullptr,*gu,*down;
    float *g=nullptr,*d=nullptr;
    strata::prefill::Gemm gemm;
    std::vector<std::vector<uint8_t>> reference;
    Pipeline(sycl::queue& queue,sycl::half* weights_gu,sycl::half* weights_down)
        :q(queue),gu(weights_gu),down(weights_down) {
        bool found=false;
        for (const auto& id:sycl::get_kernel_ids()) {
            if (std::string(id.get_name()).find("swiglu_il_kernel_98d495")==std::string::npos) continue;
            auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(),{q.get_device()},{id});
            const auto kernel=bundle.get_kernel(id);
            const auto maximum=kernel.ext_oneapi_get_info<sycl::ext::oneapi::experimental::info::kernel_queue_specific::max_num_work_groups>(q,sycl::range<3>(1,1,256),0);
            const size_t needed=(rows*ff+255)/256;
            if (maximum<needed) throw std::runtime_error("SwiGLU cooperative limit too small");
            std::printf("{\"stage\":\"SwiGLU limit\",\"maximum\":%zu,\"groups\":%zu}\n",maximum,needed);
            std::fflush(stdout);found=true;
        }
        if (!found) throw std::runtime_error("actual SwiGLU kernel absent");
        x=sycl::malloc_device<sycl::half>(rows*embd,q);
        h=sycl::malloc_device<sycl::half>(rows*ff+2*guard,q);
        g=sycl::malloc_device<float>(rows*2*ff+2*guard,q);
        d=sycl::malloc_device<float>(rows*embd+2*guard,q);
        if (!x || !h || !g || !d) throw std::runtime_error("pipeline allocation failed");
        std::vector<sycl::half> host(rows*embd);
        for (size_t i=0;i<host.size();++i) host[i]=sycl::half(float(int(i%33)-16)/64.0f);
        q.memcpy(x,host.data(),host.size()*2);
        q.memset(h,0x5a,(rows*ff+2*guard)*2);
        q.memset(g,0x5a,(rows*2*ff+2*guard)*4);
        q.memset(d,0x5a,(rows*embd+2*guard)*4);q.wait_and_throw();
        std::string err;
        if (!gemm.init_external(&q,nullptr,0,nullptr,0,err)) throw std::runtime_error(err);
    }
    void run() {
        gemm.f16(reinterpret_cast<uint16_t*>(x),reinterpret_cast<uint16_t*>(gu+guard),g+guard,rows,2*ff,embd);
        strata::prefill::swiglu_interleaved(g+guard,reinterpret_cast<uint16_t*>(h+guard),rows,&q);
        q.wait_and_throw();
        gemm.f16(reinterpret_cast<uint16_t*>(h+guard),reinterpret_cast<uint16_t*>(down+guard),d+guard,rows,embd,ff);
        q.wait_and_throw();
    }
    std::vector<uint8_t> floats(float* input,size_t elements) {
        std::vector<uint32_t> words(elements+2*guard);
        q.memcpy(words.data(),input,words.size()*4).wait_and_throw();
        for (size_t i=0;i<guard;++i)
            if (words[i]!=0x5a5a5a5a || words[guard+elements+i]!=0x5a5a5a5a)
                throw std::runtime_error("GEMM output guard changed");
        for (size_t i=guard;i<guard+elements;++i)
            if (!std::isfinite(sycl::bit_cast<float>(words[i]))) throw std::runtime_error("nonfinite GEMM output");
        std::vector<uint8_t> bytes(words.size()*4);std::memcpy(bytes.data(),words.data(),bytes.size());return bytes;
    }
    std::vector<std::vector<uint8_t>> snapshot() {
        std::vector<uint16_t> words(rows*ff+2*guard);
        q.memcpy(words.data(),h,words.size()*2).wait_and_throw();guards(words,rows*ff);
        std::vector<uint8_t> hbytes(words.size()*2);std::memcpy(hbytes.data(),words.data(),hbytes.size());
        return {floats(g,rows*2*ff),hbytes,floats(d,rows*embd)};
    }
    void capture(const std::string& path) {
        reference=snapshot();
        for (size_t i=0;i<reference.size();++i) {
            std::ofstream out(path+"-"+std::to_string(i)+".bin",std::ios::binary);
            out.write(reinterpret_cast<const char*>(reference[i].data()),std::streamsize(reference[i].size()));
            out.close();if (!out) throw std::runtime_error("pipeline reference write failed");
        }
    }
    void check() {if (snapshot()!=reference) throw std::runtime_error("whole GEMM/SwiGLU output changed");}
    void finish() {q.wait_and_throw();sycl::free(d,q);sycl::free(g,q);sycl::free(h,q);sycl::free(x,q);}
};
'''
s=original.replace('#include <algorithm>','#include "strata/prefill/gemm.hpp"\n#include "strata/prefill/kernels.hpp"\n#include <memory>\n#include <cstring>\n#include <algorithm>')
s=s.replace('int main(int argc, char** argv) try {',extra+'\nint main(int argc, char** argv) try {')
s=s.replace('    auto launch=[&]', '    auto pipeline=std::make_unique<Pipeline>(q,gu,down);\n    auto launch=[&]')
s=s.replace('        launch(mapped); download', '        launch(mapped); pipeline->run(); download')
s=s.replace('    launch(arena);\n', '    launch(arena);pipeline->run();pipeline->capture(std::string(argv[5])+"-pipeline");\n')
s=s.replace('        launch(arena+((size_t(i)*37+11)%slots)*stride);', '        launch(arena+((size_t(i)*37+11)%slots)*stride);pipeline->run();')
s=s.replace('            ++full_checks;', '            pipeline->check();\n            ++full_checks;')
s=s.replace('    sycl::free(arena,q);', '    pipeline->finish();pipeline.reset();\n    sycl::free(arena,q);')
p=base/'dequant_gemm_completion_stress.cpp';assert not p.exists();p.write_text(s)

b=(base/'build_dequant_completion_stress.py').read_text()
b=b.replace('dequant-completion-stress-build','dequant-gemm-completion-stress-build').replace('dequant_completion_stress.cpp','dequant_gemm_completion_stress.cpp')
b=b.replace("    for label,args in [('compile',argv),('link',link)]:", "    link.insert(-2,'-qmkl=sequential')\n    for path in ['libstrata_prefill.a','libstrata_kernels.a']:\n        link.insert(-2,str(root/'build-sycl-refresh-20261007'/path))\n    for label,args in [('compile',argv),('link',link)]:")
# The linker -o flag must stay immediately before its output value.
b=b.replace("link.insert(-2,", "link.insert(-3,")
b=b.replace("LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu'", "LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu',MKLROOT='/opt/intel/oneapi/mkl/2026.1'")
b=b.replace("'scope':'Offline regular actual IQ object stress fixture build; no GPU execution'", "'scope':'Offline actual regular IQ wrappers plus unchanged production GEMM/SwiGLU fixture build; no GPU execution'")
p=base/'build_dequant_gemm_completion_stress.py';assert not p.exists();p.write_text(b)

c=(base/'run_dequant_completion_stress.py').read_text()
c=c.replace('dequant-completion-stress-', 'dequant-gemm-completion-stress-').replace('dequant_completion_stress.cpp','dequant_gemm_completion_stress.cpp')
c=c.replace('optional64MiB VMM warmup/retirement on samequeue', 'actual GEMM/SwiGLU/down consumers with synthetic eight-row FP16 input; optional64MiB VMM warmup/retirement on samequeue')
c=c.replace("    r['reference_heads']=", "    r['pipeline_heads']=[dict(file=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.glob('reference-pipeline-*.bin'))]\n    assert len(r['pipeline_heads'])==3\n    if mode=='retire':assert r['pipeline_heads']==before['pipeline_heads']\n    r['reference_heads']=")
c=c.replace("for p in sorted(out.glob('reference-*.bin'))", "for p in sorted(out.glob('reference-*.bin')) if '-pipeline-' not in p.name")
c=c.replace("env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')", "env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0');env['LD_LIBRARY_PATH']+=':/opt/intel/oneapi/mkl/2026.1/lib'")
p=base/'run_dequant_gemm_completion_stress.py';assert not p.exists();p.write_text(c)
print('Prepared actual IQ/GEMM/SwiGLU completion fixture')
