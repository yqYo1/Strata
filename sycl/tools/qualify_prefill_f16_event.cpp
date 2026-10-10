// ROOT-RUN qualification only: link the real SYCL Gemm implementation/library.
// Full small-output equality tests layout/type/completion, not internal-kernel interval coverage.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/prefill/gemm.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

int main() try {
    auto* q = dpct::get_current_device().create_in_order_queue(true);
    if (!q->has_property<sycl::property::queue::enable_profiling>() ||
        !q->get_device().has(sycl::aspect::queue_profiling)) throw std::runtime_error("profiling unavailable");
    if(q->get_device().get_info<sycl::info::device::name>() != "Intel(R) Arc(TM) B570 Graphics")
        throw std::runtime_error("B570 qualification device required");
    constexpr int64_t T = 7, N = 11, K = 13, ldy = 17;
    std::vector<uint16_t> x(T*K), w(N*K);
    // 0/1 half inputs make every expected product exactly representable as FP32.
    for (int64_t i=0;i<T*K;++i) x[i] = (i%3) ? 0x3c00 : 0;
    for (int64_t i=0;i<N*K;++i) w[i] = (i%5) ? 0x3c00 : 0;
    std::vector<float> expected(T*ldy,-123.0f), initial(T*ldy,-123.0f), observed(T*ldy), control(T*ldy);
    for(int64_t t=0;t<T;++t) for(int64_t n=0;n<N;++n) {
        float sum=0;
        for(int64_t k=0;k<K;++k) sum += (x[t*K+k] ? 1.0f : 0.0f)*(w[n*K+k] ? 1.0f : 0.0f);
        expected[t*ldy+n]=sum;
    }
    auto* dx = sycl::malloc_device<uint16_t>(x.size(),*q);
    auto* dw = sycl::malloc_device<uint16_t>(w.size(),*q);
    auto* dy = sycl::malloc_device<float>(expected.size(),*q);
    if(!dx || !dw || !dy) throw std::runtime_error("allocation failed");
    // Raw owner deliberately survives exception paths until _Exit; release only after successful completion.
    auto* gm = new strata::prefill::Gemm;
    std::string error;
    if(!gm->init(q,1,error)) throw std::runtime_error(error);
    q->memcpy(dx,x.data(),x.size()*sizeof(uint16_t));
    q->memcpy(dw,w.data(),w.size()*sizeof(uint16_t));
    q->memcpy(dy,initial.data(),initial.size()*sizeof(float));
    sycl::event returned;
    if(!gm->f16_event(dx,dw,dy,T,N,K,ldy,returned)) throw std::runtime_error("unsupported event path");
    // This qualification explicitly waits; engine capture never adds this wait.
    returned.wait_and_throw(); q->throw_asynchronous();
    if(returned.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete)
        throw std::runtime_error("returned event incomplete");
    const auto submit=returned.get_profiling_info<sycl::info::event_profiling::command_submit>();
    const auto start=returned.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end=returned.get_profiling_info<sycl::info::event_profiling::command_end>();
    if(submit>start || start>end) throw std::runtime_error("backward profiling fields");
    q->memcpy(observed.data(),dy,observed.size()*sizeof(float)).wait_and_throw();
    // Control exercises the unchanged wrapper once in this qualification.
    q->memcpy(dy,initial.data(),initial.size()*sizeof(float));
    gm->f16(dx,dw,dy,T,N,K,ldy);
    q->memcpy(control.data(),dy,control.size()*sizeof(float)).wait_and_throw();
    for(size_t i=0;i<expected.size();++i)
        if(observed[i]!=expected[i] || control[i]!=expected[i]) throw std::runtime_error("full output/padding mismatch");
    std::printf("f16 returned-event qualification PASS: T=%lld N=%lld K=%lld ldy=%lld half/half/float trans/nontrans; full %zu float outputs and padding; submit=%llu start=%llu end=%llu; not internal kernel coverage\n",
        (long long)T,(long long)N,(long long)K,(long long)ldy,expected.size(),
        (unsigned long long)submit,(unsigned long long)start,(unsigned long long)end);
    q->wait_and_throw();
    delete gm;
    sycl::free(dx,*q); sycl::free(dw,*q); sycl::free(dy,*q);
    return 0;
} catch(const std::exception& e) {
    std::fprintf(stderr,"f16 returned-event qualification FAIL: %s\n",e.what());
    std::fflush(stderr); std::_Exit(1); // Avoid cleanup with unknown completion.
}
