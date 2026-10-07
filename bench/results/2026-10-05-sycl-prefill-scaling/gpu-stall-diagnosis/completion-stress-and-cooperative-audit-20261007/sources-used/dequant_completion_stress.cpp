// Diagnostic only: actual updated IQ wrappers with regular launches.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#include "strata/kernels/iq_kernels.hpp"
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
        launch(mapped); download(gu,gu_elements); download(down,down_elements);
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
    launch(arena);
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
        launch(arena+((size_t(i)*37+11)%slots)*stride);
        if (i%512==0 || i==iterations) {
            if (download(gu,gu_elements)!=expected_gu || download(down,down_elements)!=expected_down)
                throw std::runtime_error("whole FP16 output changed");
            ++full_checks;
        }
        if (i%128==0 || i==iterations) {
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            std::printf("{\"stage\":\"progress\",\"iterations\":%d,\"full_checks\":%zu,\"seconds\":%.6f}\n",i,full_checks,seconds);
            std::fflush(stdout);
        }
    }
    q.wait_and_throw();
    sycl::free(arena,q); sycl::free(down,q); sycl::free(gu,q);
    std::printf("{\"stage\":\"PASS\",\"iterations\":%d,\"dequant_launches\":%d,\"full_checks\":%zu}\n",
                iterations,2*(iterations+1+(mode=="retire")),full_checks);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr,"completion stress failed: %s\n",e.what()); return 1;
}
