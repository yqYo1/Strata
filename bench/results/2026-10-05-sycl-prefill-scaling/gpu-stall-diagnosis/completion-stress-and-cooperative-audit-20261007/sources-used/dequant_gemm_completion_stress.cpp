// Diagnostic only: actual updated IQ wrappers with regular launches.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include <memory>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vm = sycl::ext::oneapi::experimental;
constexpr size_t guard = 64, slots = 512;
constexpr int64_t embd = 2560, ff = 640;
constexpr size_t gu_elements = 2 * ff * embd, down_elements = ff * embd;
struct Metadata { int gu = 0, down = 0; size_t bytes = 0, largest = 0;
                  uint64_t gate_at = 0, up_at = 0, down_at = 0; };

Metadata metadata(const char* path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read native metadata");
    Metadata m;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        int layer, gu, down; uint64_t offset, bytes, gate_at, up_at, down_at;
        if (!(row >> layer >> gu >> down >> offset >> bytes >> gate_at >> up_at >> down_at))
            throw std::runtime_error("invalid native metadata row");
        m.largest = std::max(m.largest, size_t(bytes));
        if (layer == 17) m.gu = gu, m.down = down, m.bytes = bytes,
            m.gate_at = gate_at, m.up_at = up_at, m.down_at = down_at;
    }
    if (!m.bytes || !m.largest) throw std::runtime_error("missing layer17");
    return m;
}
void read_at(std::ifstream& in, uint64_t offset, uint8_t* out, size_t bytes) {
    in.seekg(std::streamoff(offset));
    in.read(reinterpret_cast<char*>(out), std::streamsize(bytes));
    if (!in || size_t(in.gcount()) != bytes) throw std::runtime_error("short GGUF tensor read");
}
void write(const std::string& path, const std::vector<uint16_t>& bits) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bits.data()), std::streamsize(bits.size()*2));
    out.close(); if (!out) throw std::runtime_error("head write failed");
}
void guards(const std::vector<uint16_t>& bits, size_t elements) {
    for (size_t i=0; i<guard; ++i)
        if (bits[i]!=0x5a5a || bits[guard+elements+i]!=0x5a5a)
            throw std::runtime_error("output guard changed");
    for (size_t i=guard; i<guard+elements; ++i)
        if (!std::isfinite(float(sycl::bit_cast<sycl::half>(bits[i]))))
            throw std::runtime_error("nonfinite output");
}


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

int main(int argc, char** argv) try {
    if (argc!=6) throw std::runtime_error("shard metadata usm|retire iterations output-prefix required");
    const std::string mode=argv[3];
    const int iterations=std::stoi(argv[4]);
    if ((mode!="usm" && mode!="retire") || iterations<1 || iterations>20000)
        throw std::runtime_error("invalid bounded case");
    const auto meta=metadata(argv[2]);
    const auto layout=strata::kernels::native_expert_layout(meta.gu,meta.down,embd,ff);
    if (layout.bytes!=meta.bytes || !strata::kernels::native_expert_supported(meta.gu,meta.down,embd,ff))
        throw std::runtime_error("unexpected native layout");
    std::vector<uint8_t> blob(layout.bytes);
    std::ifstream shard(argv[1],std::ios::binary);
    read_at(shard,meta.gate_at,blob.data(),layout.up_off);
    read_at(shard,meta.up_at,blob.data()+layout.up_off,layout.up_off);
    read_at(shard,meta.down_at,blob.data()+layout.down_off,layout.bytes-layout.down_off);
    sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
    std::fprintf(stderr,"device=%s driver=%s mode=%s\n",
        q.get_device().get_info<sycl::info::device::name>().c_str(),
        q.get_device().get_info<sycl::info::device::driver_version>().c_str(),mode.c_str());
    auto* gu=sycl::malloc_device<sycl::half>(gu_elements+2*guard,q);
    auto* down=sycl::malloc_device<sycl::half>(down_elements+2*guard,q);
    if (!gu || !down) throw std::runtime_error("output allocation failed");
    q.memset(gu,0x5a,(gu_elements+2*guard)*2);
    q.memset(down,0x5a,(down_elements+2*guard)*2);
    q.wait_and_throw();
    auto pipeline=std::make_unique<Pipeline>(q,gu,down);
    auto launch=[&](const uint8_t* source) {
        strata::kernels::iq_dequant_gu_f16(meta.gu,source,source+layout.up_off,ff,embd,
            reinterpret_cast<uint16_t*>(gu+guard),&q);
        strata::kernels::iq_dequant_f16(meta.down,source+layout.down_off,down_elements,
            reinterpret_cast<uint16_t*>(down+guard),&q);
        q.wait_and_throw();
    };
    auto download=[&](sycl::half* source, size_t elements) {
        std::vector<uint16_t> bits(elements+2*guard);
        q.memcpy(bits.data(),source,bits.size()*2).wait_and_throw();
        guards(bits,elements); return bits;
    };
    if (mode=="retire") {
        if (!q.get_device().has(sycl::aspect::ext_oneapi_virtual_mem))
            throw std::runtime_error("VMM aspect absent");
        constexpr size_t segment=64*1024*1024;
        const size_t gran=vm::get_mem_granularity(q.get_context(),vm::granularity_mode::minimum);
        if (!gran || segment%gran) throw std::runtime_error("unexpected VMM granularity");
        const uintptr_t va=vm::reserve_virtual_mem(segment,q.get_context());
        // Keep raw ownership on failure: never unmap/free an in-flight allocation.
        auto* physical=new vm::physical_mem(q,segment);
        auto* mapped=static_cast<uint8_t*>(physical->map(va,segment,vm::address_access_mode::read_write));
        if (reinterpret_cast<uintptr_t>(mapped)!=va) throw std::runtime_error("VMM address mismatch");
        q.memcpy(mapped,blob.data(),blob.size()).wait_and_throw();
        launch(mapped); pipeline->run(); download(gu,gu_elements); download(down,down_elements);
        q.wait_and_throw();
        vm::unmap(mapped,segment,q.get_context()); delete physical;
        vm::free_virtual_mem(va,segment,q.get_context());
        std::printf("{\"stage\":\"VMM retired\",\"segment_bytes\":%zu}\n",segment);
        std::fflush(stdout);
    }
    const size_t stride=(meta.largest+255)/256*256, arena_bytes=slots*stride;
    auto* arena=sycl::malloc_device<uint8_t>(arena_bytes,q);
    if (!arena) throw std::runtime_error("source arena allocation failed");
    q.memset(arena,0,arena_bytes);
    for (size_t i=0; i<slots; ++i) q.memcpy(arena+i*stride,blob.data(),blob.size());
    q.wait_and_throw();
    launch(arena);pipeline->run();pipeline->capture(std::string(argv[5])+"-pipeline");
    const auto expected_gu=download(gu,gu_elements), expected_down=download(down,down_elements);
    write(std::string(argv[5])+"-gu.bin",expected_gu);
    write(std::string(argv[5])+"-down.bin",expected_down);
    std::printf("{\"stage\":\"reference\",\"layer\":17,\"expert\":0,\"gu_type\":%d,\"down_type\":%d,"
                "\"slots\":%zu,\"stride\":%zu,\"arena_bytes\":%zu,\"blob_bytes\":%zu}\n",
                meta.gu,meta.down,slots,stride,arena_bytes,blob.size());
    std::fflush(stdout);
    const auto start=std::chrono::steady_clock::now();
    size_t full_checks=0;
    for (int i=1; i<=iterations; ++i) {
        launch(arena+((size_t(i)*37+11)%slots)*stride);pipeline->run();
        if (i%512==0 || i==iterations) {
            if (download(gu,gu_elements)!=expected_gu || download(down,down_elements)!=expected_down)
                throw std::runtime_error("whole FP16 output changed");
            pipeline->check();
            ++full_checks;
        }
        if (i%128==0 || i==iterations) {
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            std::printf("{\"stage\":\"progress\",\"iterations\":%d,\"full_checks\":%zu,\"seconds\":%.6f}\n",i,full_checks,seconds);
            std::fflush(stdout);
        }
    }
    q.wait_and_throw();
    pipeline->finish();pipeline.reset();
    sycl::free(arena,q); sycl::free(down,q); sycl::free(gu,q);
    std::printf("{\"stage\":\"PASS\",\"iterations\":%d,\"dequant_launches\":%d,\"full_checks\":%zu}\n",
                iterations,2*(iterations+1+(mode=="retire")),full_checks);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr,"completion stress failed: %s\n",e.what()); return 1;
}
