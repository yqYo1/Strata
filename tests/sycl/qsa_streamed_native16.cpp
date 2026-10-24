// Main-authored memory/publication fixture; repository LICENSE applies.
// The qualified resident append is the byte oracle; its separate fixture owns
// the numerical oracle. No model, task quality or performance claim.
#include "strata/prefill/qsa_native16.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/compat/cuda_runtime.h"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sb = strata::sycl_backend;
namespace qa = strata::prefill::native16::qsa;
namespace sk = strata::kernels;
namespace {
constexpr size_t G = 64;
void need(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
struct Drain {
    std::shared_ptr<sb::Runtime> r;
    ~Drain() noexcept { sb::checked_cleanup([&] { r->wait(); }, [] {}, "QSA16 streamed fixture drain"); }
};
template<class T> struct Buffer {
    std::shared_ptr<sb::Runtime> r;
    sb::Allocation device;
    T* host = nullptr;
    size_t count;
    Buffer(std::shared_ptr<sb::Runtime> runtime, size_t n, bool mapped=false) :
        r(runtime), count(n+2*G) {
        if (mapped) {
            void* p=nullptr;
            need(cudaHostAlloc(&p,count*sizeof(T),cudaHostAllocMapped)==cudaSuccess,"mapped host owner");
            host=static_cast<T*>(p);
        } else device=sb::Allocation(r,count*sizeof(T),sb::MemoryKind::Device);
    }
    ~Buffer() noexcept {
        if (host && cudaFreeHost(host)!=cudaSuccess) sb::cleanup_fail_stop("QSA16 fixture host free");
    }
    T* base() { return host ? host : device.template as<T>(); }
    T* data() { return base()+G; }
    void fill() {
        if (host) std::memset(host,0xa5,count*sizeof(T));
        else r->compute().memset(base(),0xa5,count*sizeof(T));
    }
    void put(const std::vector<T>& values) {
        need(values.size()==count,"upload extent");
        r->wait(r->compute().memcpy(base(),values.data(),count*sizeof(T)));
    }
    std::vector<T> get() {
        std::vector<T> values(count);
        if (host) std::memcpy(values.data(),host,count*sizeof(T));
        else r->wait(r->compute().memcpy(values.data(),base(),count*sizeof(T)));
        return values;
    }
};
struct Pools {
    Buffer<int8_t> k,v;
    Buffer<uint16_t> ks,vs;
    Pools(std::shared_ptr<sb::Runtime> r, size_t cells, bool host=false) :
        k(r,cells*2*256,host),v(r,cells*2*256,host),ks(r,cells*2*4,host),vs(r,cells*2*4,host) {}
    void fill() { k.fill();v.fill();ks.fill();vs.fill(); }
    qa::Q8LinearPools linear() { return {k.data(),v.data(),ks.data(),vs.data()}; }
    qa::Q8WritePools mapped(const int32_t* table) { return {k.data(),v.data(),ks.data(),vs.data(),table}; }
};
struct Stop {};

void run(std::shared_ptr<sb::Runtime> r, int pages, int physical, int pos0, int tokens,
         bool unwind, bool rejects) {
    const int page=4,ld=530,cells=pages*page;
    Pools stage(r,cells),reference(r,cells),resident(r,physical*page),host(r,cells,true);
    Buffer<int32_t> table(r,pages),identity(r,pages);
    Buffer<uint16_t> k(r,size_t(tokens)*ld),v(r,size_t(tokens)*ld);
    Drain lifetime{r}; // every owner above survives all copies and checked errors
    stage.fill();reference.fill();resident.fill();host.fill();
    std::vector<int32_t> map(table.count,-99),ident(identity.count,-99);
    for (int b=0;b<pages;++b) { map[G+b]=-1;ident[G+b]=b; }
    map[G]=physical-1;
    if (pages>2) map[G+2]=0;
    if (pages>3) map[G+3]=physical; // absent/out-of-range pages both skip only the resident write
    if (pages>4) map[G+4]=1;
    if (pages>5) map[G+5]=INT32_MAX;
    if (pos0/4>5) map[G+pos0/4]=physical-1;
    table.put(map);identity.put(ident);
    std::vector<uint16_t> x(k.count,0xa5a5),y(v.count,0xa5a5);
    for (int t=0;t<tokens;++t) for (int h=0;h<2;++h) for (int d=0;d<256;++d) {
        x[G+t*ld+h*256+d]=sk::bf16_from_f32(float((t*7+h*13+d*17)%97-48)/8.f);
        y[G+t*ld+h*256+d]=sk::bf16_from_f32(float((t*11+h*19+d*23)%83-41)/16.f);
    }
    x[G]=0x7fc1;y[G]=0x7f80; // retain the resident quantizer's nonfinite diagnostics
    k.put(x);v.put(y);
    auto shapes=sk::qsa_real_shapes();shapes.page_size=page;
    sk::QsaPromptBf16Bounds bounds{pages,physical};
    auto execute=[&](qa::Q8WritePools slots,qa::Q8LinearPools staging,qa::Q8LinearPools mirror,
                     int n,int64_t start,void* queue) {
        qa::append_q8_streamed(k.data(),v.data(),n,ld,start,slots,staging,mirror,shapes,bounds,queue);
    };
    qa::append_q8(k.data(),v.data(),tokens,ld,pos0,reference.mapped(identity.data()),
        shapes,{pages,pages},&r->compute());
    bool caught=false;
    try {
        Drain pending{r};
        execute(resident.mapped(table.data()),stage.linear(),host.linear(),tokens,pos0,&r->compute());
        if (unwind) throw Stop{};
    } catch (const Stop&) { caught=true; }
    need(caught==unwind,"checked host unwind");
    std::vector<int> starts{pos0};
    auto check_field=[&](auto& full,auto& staged,auto& mirror,auto& slots,int width) {
        auto expected=full.get(),actual=staged.get(),mapped=mirror.get(),cached=slots.get();
        need(expected==actual && expected==mapped,"full stage/host quantized bytes or guards differ");
        using T=typename decltype(cached)::value_type;
        std::vector<T> wanted(cached.size());std::memset(wanted.data(),0xa5,wanted.size()*sizeof(T));
        for (int start:starts) for (int t=0;t<tokens;++t) {
            const int pos=start+t,block=pos/page,slot=map[G+block];
            if (slot<0 || slot>=physical) continue;
            for (int h=0;h<2;++h) for (int d=0;d<width;++d) {
                const size_t from=G+((size_t(block)*2+h)*page+pos%page)*width+d;
                const size_t to=G+((size_t(slot)*2+h)*page+pos%page)*width+d;
                wanted[to]=expected[from];
            }
        }
        need(cached==wanted,"resident map/partial page/tail guards differ");
    };
    auto check=[&] {
        r->wait();
        check_field(reference.k,stage.k,host.k,resident.k,256);
        check_field(reference.v,stage.v,host.v,resident.v,256);
        check_field(reference.ks,stage.ks,host.ks,resident.ks,4);
        check_field(reference.vs,stage.vs,host.vs,resident.vs,4);
        need(k.get()==x && v.get()==y && table.get()==map && identity.get()==ident,"input/map mutation");
    };
    check();
    unsigned rejected=0;
    auto invalid=[&](auto call) {
        bool failed=false;try { call(); } catch (const std::invalid_argument&) { failed=true; }
        need(failed,"invalid streamed append accepted");++rejected;
    };
    if (rejects) {
        const auto slots=resident.mapped(table.data());const auto st=stage.linear(),h=host.linear();
        auto go=[&](auto a,auto b,auto c) { execute(a,b,c,tokens,pos0,&r->compute()); };
        invalid([&] { auto bad=h;bad.k_q=nullptr;go(slots,st,bad); });
        invalid([&] { auto bad=h;bad.k_q=stage.k.data();go(slots,st,bad); });
        invalid([&] { auto bad=h;bad.k_q=host.k.base()+2*G+1;go(slots,st,bad); });
        std::vector<int8_t> ordinary(host.k.count);
        invalid([&] { auto bad=h;bad.k_q=ordinary.data();go(slots,st,bad); });
        invalid([&] { auto bad=h;bad.v_q=bad.k_q;go(slots,st,bad); });
        invalid([&] { auto bad=st;bad.k_q=stage.k.base()+2*G+1;go(slots,bad,h); });
        invalid([&] { auto bad=st;bad.v_q=resident.k.data();go(slots,bad,h); });
        invalid([&] { auto bad=slots;bad.page_table=nullptr;go(bad,st,h); });
        invalid([&] { auto bad=bounds;bad.physical_pages=pages+1;
            qa::append_q8_streamed(k.data(),v.data(),tokens,ld,pos0,slots,st,h,shapes,bad,&r->compute()); });
        invalid([&] { execute(slots,st,h,tokens,cells,&r->compute()); });
        invalid([&] { execute(slots,st,h,0,pos0,&r->compute()); });
        invalid([&] { execute(slots,st,h,8193,pos0,&r->compute()); });
        invalid([&] { qa::append_q8_streamed(reinterpret_cast<uint16_t*>(st.k_q),v.data(),tokens,
            ld,pos0,slots,st,h,shapes,bounds,&r->compute()); });
        sycl::queue foreign{sycl::context{r->device()},r->device(),sycl::property::queue::in_order{}};
        invalid([&] { execute(slots,st,h,tokens,pos0,&foreign); });
        check(); // refusals must precede *all* quantization/publication writes
        // Consecutive chunks retain their prefix; a later stage refill and
        // rewound append overwrite only their own cells, using the unchanged
        // streamed-cache host->stage API also used at RESTORE entry.
        for (int start:{pos0+tokens,pos0+2}) {
            if (start==pos0+2) {
                stage.fill();
                sk::QsaAttnPools destination{};destination.k_q=st.k_q;destination.v_q=st.v_q;
                destination.k_scale=st.k_scale;destination.v_scale=st.v_scale;destination.page_table=identity.data();
                sk::KvHostPools source{};source.k_q=h.k_q;source.v_q=h.v_q;
                source.k_scale=h.k_scale;source.v_scale=h.v_scale;
                sk::kv_stage_from_host(destination,source,sk::kKvInt8,pages,shapes,&r->compute());
                x[G+64]=sk::bf16_from_f32(127.f);k.put(x);
            }
            qa::append_q8(k.data(),v.data(),tokens,ld,start,reference.mapped(identity.data()),
                shapes,{pages,pages},&r->compute());
            execute(slots,st,h,tokens,start,&r->compute());
            starts.push_back(start);check();
        }
    }
    std::cout << "QSA16_streamed_case cells="<<cells<<" physical_pages="<<physical<<" pos0="<<pos0
        <<" tokens="<<tokens<<" rejects="<<rejected<<" unwind="<<caught
        <<" chunks="<<starts.size()
        <<" stage_host_reference=PASS resident_map=PASS guards=PASS inputs=PASS checked_drain=PASS\n";
}
} // namespace
int main(int argc,char** argv) try {
    need(argc==2 && (std::string(argv[1])=="--smoke" || std::string(argv[1])=="--component"),
        "usage: --smoke|--component");
    auto r=sb::runtime_for();
    run(r,3,2,1,5,true,true);
    if (std::string(argv[1])=="--component") {
        run(r,8,3,3,19,false,false);
        run(r,2049,3,2,8192,false,false);
        run(r,65536,3,262141,3,true,false);
    }
    r->wait();
    std::cout<<"QSA16 streamed publication PASS model=UNRUN physical262144_model_RESTORE_decode_performance=UNQUALIFIED\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr<<"QSA16 streamed fixture failure: "<<error.what()<<'\n';return 1;
}
