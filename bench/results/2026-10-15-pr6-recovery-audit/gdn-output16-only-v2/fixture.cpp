// Main-authored Strata fixture; repository LICENSE (MIT) applies.
// v2: real registered span/domain validation before packed-input offsets.
// Prospective qualification, not run. Compare the output-store specialization
// against the same-source FP32+FP16 kernel; no new precision standard is imposed.
#include "strata/prefill/kernels.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/runtime.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// The private kernel object exports this symbol before the product patch is
// applied. Never link both the original and private fused_gdn object explicitly.
namespace strata::prefill {
void gdn_recurrence_output16(float*, const float*, const float*, const float*,
                             const float*, const float*, float, uint16_t*,
                             int64_t, void*);
}
namespace p = strata::prefill;
namespace s = strata::sycl_backend;
std::shared_ptr<s::Runtime> runtime;
void need(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void drain() noexcept {
    s::checked_cleanup([] { if (runtime) runtime->wait(); }, [] {},
                       "GDN output16 fixture drain");
}
struct Drain { ~Drain() noexcept { drain(); } };
template <class T> struct Buffer {
    s::Allocation allocation;
    size_t count;
    explicit Buffer(size_t n) : allocation(runtime, n*sizeof(T), s::MemoryKind::Device), count(n) {}
    T* data() { return allocation.as<T>(); }
    void put(const std::vector<T>& values) {
        need(values.size()==count, "upload extent");
        runtime->wait(runtime->compute().memcpy(data(), values.data(), count*sizeof(T)));
    }
    std::vector<T> get() {
        std::vector<T> values(count);
        runtime->wait(runtime->compute().memcpy(values.data(), data(), count*sizeof(T)));
        return values;
    }
};
template <class T> bool bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size()==b.size() && !std::memcmp(a.data(), b.data(), a.size()*sizeof(T));
}
void gate(const char* phase) {
    std::cout << "GDN_OUTPUT16_READY " << phase << std::endl;
    need(std::cin.get()=='X', "Main gate token");
}
void run(int T, bool nonfinite) {
    constexpr int C=10240, HV=48, V=HV*128, ST=128*V, Pad=16;
    constexpr float Sentinel=1234.f;
    constexpr uint16_t HalfSentinel=0x5500;
    std::vector<float> h(size_t(T)*C), gate_values(size_t(T)*HV), beta(gate_values.size()),
        z(size_t(T)*V), gamma(128), initial(ST+2*Pad, Sentinel);
    for (size_t j=0; j<h.size(); ++j) h[j]=float(int(j%127)-63)*.001f;
    for (size_t j=0; j<gate_values.size(); ++j) {
        gate_values[j]=-.05f*float(1+j%7); beta[j]=.1f+.03f*float(j%5);
    }
    for (size_t j=0; j<z.size(); ++j) z[j]=float(int(j%43)-21)*.03f;
    for (size_t j=0; j<gamma.size(); ++j) gamma[j]=.9f+.01f*float(j%11);
    for (int j=0; j<ST; ++j) initial[Pad+j]=float(j%31-15)*.0001f;
    if (nonfinite) {
        z[0]=std::numeric_limits<float>::quiet_NaN();
        z[1]=std::numeric_limits<float>::infinity();
    }
    const std::vector<uint16_t> blank(size_t(T)*V+2*Pad, HalfSentinel);
    Buffer<float> H(h.size()), G(gate_values.size()), B(beta.size()), Z(z.size()),
        N(gamma.size()), old_state(initial.size()), new_state(initial.size()),
        old_output(size_t(T)*V+2*Pad);
    Buffer<uint16_t> old_half(blank.size()), new_half(blank.size());
    Drain pending; // Queue retirement precedes every allocation/input destructor.
    H.put(h); G.put(gate_values); B.put(beta); Z.put(z); N.put(gamma);
    old_state.put(initial); new_state.put(initial);
    old_output.put(std::vector<float>(size_t(T)*V+2*Pad, Sentinel));
    old_half.put(blank); new_half.put(blank);
    auto stream=&runtime->compute();
    auto candidate=[&](int first, int count) {
        p::gdn_recurrence_output16(new_state.data()+Pad, H.data()+size_t(first)*C,
            G.data()+size_t(first)*HV, B.data()+size_t(first)*HV,
            Z.data()+size_t(first)*V, N.data(), 1e-6f,
            new_half.data()+Pad+size_t(first)*V, count, stream);
    };
    p::gdn_recurrence(old_state.data()+Pad, H.data(), G.data(), B.data(), Z.data(),
        N.data(), 1e-6f, old_output.data()+Pad, old_half.data()+Pad, T, stream);
    candidate(0,T); runtime->wait();
    const auto state=old_state.get();
    const auto expected=old_half.get();
    need(bits(state,new_state.get()), "FP32 final state differs");
    need(bits(expected,new_half.get()), "FP16 output differs");
    const auto full=old_output.get();
    for (int j=0; j<Pad; ++j) {
        need(state[j]==Sentinel && state[state.size()-Pad+j]==Sentinel, "state guards");
        need(full[j]==Sentinel && full[full.size()-Pad+j]==Sentinel, "old output guards");
        need(expected[j]==HalfSentinel && expected[expected.size()-Pad+j]==HalfSentinel,
             "FP16 output guards");
    }
    if (nonfinite) need((expected[Pad]&0x7c00)==0x7c00 && (expected[Pad]&0x03ff), "NaN output diagnostic");
    else for (size_t j=Pad; j<expected.size()-Pad; ++j)
        need((expected[j]&0x7c00)!=0x7c00, "finite case output");
    new_state.put(initial); new_half.put(blank); candidate(0,T); runtime->wait();
    need(bits(expected,new_half.get()) && bits(state,new_state.get()), "repeat bits");
    if (T>1) {
        const int first=T==17 ? 5 : T/2;
        new_state.put(initial); new_half.put(blank);
        candidate(0,first); candidate(first,T-first); runtime->wait();
        need(bits(expected,new_half.get()) && bits(state,new_state.get()), "split recurrence/state");
    }
    bool refused=false;
    try { candidate(0,0); } catch (const std::invalid_argument&) { refused=true; }
    need(refused, "zero count admitted");
    refused=false;
    try {
        p::gdn_recurrence_output16(new_state.data()+Pad,H.data(),G.data(),B.data(),
            Z.data(),N.data(),1e-6f,nullptr,T,stream);
    } catch (const std::invalid_argument&) { refused=true; }
    need(refused, "null FP16 output admitted");
    refused=false;
    try {
        p::gdn_recurrence_output16(new_state.data()+Pad,nullptr,G.data(),B.data(),
            Z.data(),N.data(),1e-6f,new_half.data()+Pad,T,stream);
    } catch (const std::invalid_argument&) { refused=true; }
    need(refused, "null input reached pointer arithmetic");
    refused=false;
    try {
        p::gdn_recurrence_output16(new_state.data()+Pad,H.data(),G.data(),B.data(),
            Z.data(),N.data(),1e-6f,new_half.data()+Pad,T,nullptr);
    } catch (const std::invalid_argument&) { refused=true; }
    need(refused, "implicit queue admitted");
    refused=false;
    try {
        p::gdn_recurrence_output16(new_state.data()+Pad,H.data(),G.data(),B.data(),
            Z.data(),N.data(),1e-6f,reinterpret_cast<uint16_t*>(H.data()),T,stream);
    } catch (const std::invalid_argument&) { refused=true; }
    need(refused, "input/output overlap admitted");
    { Buffer<uint16_t> short_output(size_t(T)*V-1); Drain short_pending;
      refused=false;
      try { p::gdn_recurrence_output16(new_state.data()+Pad,H.data(),G.data(),B.data(),
                Z.data(),N.data(),1e-6f,short_output.data(),T,stream); }
      catch (const std::invalid_argument&) { refused=true; }
      need(refused, "short output extent admitted");
    }
    if (T==1) {
        // Every registered allocation extent, plus host and queue domains,
        // must be refused synchronously before any kernel submission.
        Buffer<float> tiny(1); Drain tiny_pending;
        auto reject=[&](auto action, const char* message) {
            bool caught=false;
            try { action(); } catch (const std::invalid_argument&) { caught=true; }
            need(caught,message);
        };
        for (int argument=0; argument<7; ++argument) {
            float* sp=new_state.data()+Pad;
            const float *hp=H.data(), *gp=G.data(), *bp=B.data(), *zp=Z.data(), *np=N.data();
            uint16_t* yp=new_half.data()+Pad;
            switch (argument) {
                case 0: sp=tiny.data(); break;
                case 1: hp=tiny.data(); break;
                case 2: gp=tiny.data(); break;
                case 3: bp=tiny.data(); break;
                case 4: zp=tiny.data(); break;
                case 5: np=tiny.data(); break;
                case 6: yp=reinterpret_cast<uint16_t*>(tiny.data()); break;
            }
            reject([&] { p::gdn_recurrence_output16(sp,hp,gp,bp,zp,np,
                       1e-6f,yp,T,stream); }, "short registered span admitted");
        }
        reject([&] { p::gdn_recurrence_output16(new_state.data()+Pad,h.data(),
                   G.data(),B.data(),Z.data(),N.data(),1e-6f,
                   new_half.data()+Pad,T,stream); }, "ordinary host input admitted");
        reject([&] { p::gdn_recurrence_output16(new_state.data()+Pad,H.data(),
                   G.data(),B.data(),Z.data(),N.data(),1e-6f,
                   new_half.data()+Pad+1,T,stream); }, "misaligned output admitted");
        sycl::queue unordered(runtime->context(),runtime->device());
        reject([&] { p::gdn_recurrence_output16(new_state.data()+Pad,H.data(),
                   G.data(),B.data(),Z.data(),N.data(),1e-6f,
                   new_half.data()+Pad,T,&unordered); }, "unordered queue admitted");
        sycl::context other_context(runtime->device());
        sycl::queue other(other_context,runtime->device(),sycl::property::queue::in_order{});
        reject([&] { p::gdn_recurrence_output16(new_state.data()+Pad,H.data(),
                   G.data(),B.data(),Z.data(),N.data(),1e-6f,
                   new_half.data()+Pad,T,&other); }, "foreign context admitted");
    }
    need(bits(expected,new_half.get()) && bits(state,new_state.get()), "rejection changed outputs");
    need(bits(H.get(),h) && bits(G.get(),gate_values) && bits(B.get(),beta) &&
         bits(Z.get(),z) && bits(N.get(),gamma), "input bytes changed");
    std::cout << "case T=" << T << " nonfinite=" << nonfinite
        << " state_bits=PASS FP16_bits=PASS split_repeat=PASS guards=PASS inputs=PASS rejection=PASS registered_extents=PASS domains=PASS\n";
}
int main(int argc, char** argv) {
    try {
        std::cout.setf(std::ios::unitbuf);
        need(argc==2 && (std::string(argv[1])=="--smoke" || std::string(argv[1])=="--component"), "mode");
        gate("before-runtime"); runtime=s::runtime_for();
        std::cout << "device=" << runtime->device().get_info<sycl::info::device::name>()
                  << " driver=" << runtime->device().get_info<sycl::info::device::driver_version>() << std::endl;
        gate("runtime-ready");
        run(1,false); run(17,true);
        if (std::string(argv[1])=="--component") { run(256,false); run(8192,false); }
        drain(); runtime.reset(); gate("after-cleanup");
        std::cout << "fixture_status=PASS model_quality=UNRUN model_PP=UNRUN physical262144_model=UNRUN faults=EXTERNAL_REQUIRED\n";
    } catch (const std::exception& error) {
        drain(); std::cerr << "fixture_status=FAIL " << error.what() << '\n'; return 1;
    }
}
