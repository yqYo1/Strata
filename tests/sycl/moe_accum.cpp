// Original Strata fixture; repository LICENSE applies. Synthetic expert outputs
// test routing/reduction, not GEMM arithmetic or model quality.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/prefill/moe_window_routes.hpp"
#include "strata/prefill/native16.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
namespace sb = strata::sycl_backend;
namespace pd = strata::prefill::detail;
namespace pn = strata::prefill::native16;
namespace kb = strata::kernels;
constexpr size_t N=2560, K=10, J=8192, G=256;
std::shared_ptr<sb::Runtime> runtime;
void need(bool ok,const char* text) { if (!ok) throw std::runtime_error(text); }
void gate(const char* phase) {
    const char* value=std::getenv("STRATA_TEST_MAIN_GATES");
    if (value && std::string(value)=="1") {
        std::cout<<"ACCUM_READY "<<phase<<'\n';need(std::cin.get()=='X',"Main gate");
    }
}
struct Pending {
    ~Pending() noexcept {
        std::cout<<"fixture drain begin\n";
        sb::checked_cleanup([] { if (runtime) runtime->wait(); },[] {},"accumulator fixture drain");
        std::cout<<"fixture drain end\n";
    }
};
template<class T> struct Buffer {
    sb::Allocation allocation;
    size_t count;
    Buffer(size_t n):allocation(runtime,n*sizeof(T)+2*G,sb::MemoryKind::Device),count(n) {}
    T* data() const { return reinterpret_cast<T*>(allocation.as<unsigned char>()+G); }
    void poison(sycl::queue& q) { q.memset(allocation.data(),0xa5,allocation.size()); }
    void guards() const {
        std::array<unsigned char,2*G> observed{};
        try {
            auto& q=runtime->compute();q.memcpy(observed.data(),allocation.data(),G);
            q.memcpy(observed.data()+G,data()+count,G);runtime->wait();
        } catch (...) { sb::cleanup_fail_stop("accumulator guard destination did not retire"); }
        for (auto x:observed) need(x==0xa5,"guard overwritten");
    }
};
float down_value(int token,int expert,int col,bool numeric) {
    if (numeric) {
        const int v=(token*13+expert*7+col*3)%97-48;
        return sycl::ldexp(float(v),(expert+col)%13-6);
    }
    return float(2*(token%13-6)+(expert%17-8)+4*(col%7-3))/16.f;
}
uint16_t shared_value(int token,int col) {
    return kb::bf16_from_f32(float(2*(token%5-2)+(col%11-5))/16.f);
}
struct Stop {};
void run(int rows,bool hot,bool numeric,int fault,bool unwind) {
    const size_t routes=size_t(rows)*K;
    std::vector<int32_t> ids(routes),order(routes);
    std::vector<float> weights(routes);
    std::vector<uint16_t> observed(size_t(rows)*N);
    uint32_t status_observed=~uint32_t(0);
    for (int t=0;t<rows;++t) for (size_t k=0;k<K;++k) {
        ids[size_t(t)*K+k]=int32_t(((hot?17:t*7)+int(k)*47)%512);
        weights[size_t(t)*K+k]=float(k+1)/(numeric?55.f:64.f);
    }
    std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int32_t a,int32_t b) { return ids[size_t(a)]<ids[size_t(b)]; });
    const auto plan=pd::MoeWindowRoutes::build_accum(rows,ids.data(),routes);
    for (size_t i=0;i<routes;++i) {
        need(plan->original_slots()[i]==order[i],"inverse differs from independent stable sort");
        need(plan->source_rows()[i]==order[i]/int(K),"source differs from independent stable sort");
    }
    Buffer<float> original_weights(routes),job_weights(routes),accumulator(size_t(rows)*N),sg{size_t(rows)};
    Buffer<int32_t> source(routes),inverse(routes);
    Buffer<uint32_t> status(1);
    Buffer<uint16_t> down(std::min<size_t>(size_t(rows),J)*N),shared(std::min<size_t>(size_t(rows),J)*N),output(size_t(rows)*N);
    Pending pending;
    auto& q=runtime->compute();
    original_weights.poison(q);job_weights.poison(q);accumulator.poison(q);sg.poison(q);
    source.poison(q);inverse.poison(q);status.poison(q);down.poison(q);shared.poison(q);output.poison(q);
    q.memcpy(original_weights.data(),weights.data(),routes*4);
    q.memcpy(source.data(),plan->source_rows().data(),routes*4);
    q.memcpy(inverse.data(),plan->original_slots().data(),routes*4);
    q.memset(accumulator.data(),0,size_t(rows)*N*4);q.memset(status.data(),0,4);q.memset(sg.data(),0,size_t(rows)*4);
    size_t launches=0;
    if (fault) {
        auto* s=source.data();auto* v=inverse.data();
        q.single_task([=] { if (fault==1) s[0]=(s[0]+1)%rows;else v[0]=int32_t(routes); });++launches;
    }
    pn::moe_job_weights(*plan,original_weights.data(),inverse.data(),job_weights.data(),status.data(),&q);++launches;
    for (size_t index=0;index<plan->jobs().size();++index) {
        const auto job=plan->jobs()[index];const auto* src=source.data();auto* d=down.data();
        q.parallel_for(sycl::range<1>(size_t(job.rows)*N),[=](sycl::id<1> item) {
            const size_t i=item[0];
            // Fault cases may deliberately change this private source map.
            // Bound the synthetic input independently; they must not commit output.
            const int t=src[size_t(job.global_row)+i/N];
            const int bounded=(t>=0 && t<rows)?t:0;
            d[i]=kb::bf16_from_f32(down_value(bounded,job.expert,int(i%N),numeric));
        });++launches;
        pn::moe_accumulate(*plan,index,down.data(),source.data(),inverse.data(),job_weights.data(),accumulator.data(),status.data(),&q);++launches;
        if (unwind) throw Stop{};
    }
    q.memcpy(&status_observed,status.data(),4);runtime->wait();
    need(status_observed==uint32_t(fault==1?2:fault==2?1:0),"status mismatch");
    if (!fault) for (size_t first=0;first<size_t(rows);first+=J) {
        const size_t count=std::min(J,size_t(rows)-first);auto* sh=shared.data();
        q.parallel_for(sycl::range<1>(count*N),[=](sycl::id<1> item) {
            const size_t i=item[0];sh[i]=shared_value(int(first+i/N),int(i%N));
        });++launches;
        pn::moe_finish(accumulator.data()+first*N,shared.data(),sg.data()+first,output.data()+first*N,int64_t(count),&q);++launches;
    }
    q.memcpy(observed.data(),output.data(),observed.size()*2);runtime->wait();
    size_t old_order_differences=0;
    for (int t=0;t<rows;++t) {
        std::array<size_t,K> sorted{};std::iota(sorted.begin(),sorted.end(),size_t(0));
        if (numeric) std::sort(sorted.begin(),sorted.end(),[&](size_t a,size_t b){return ids[size_t(t)*K+a]<ids[size_t(t)*K+b];});
        int integer_base=0;
        for (size_t k=0;k<K;++k)
            integer_base+=int(k+1)*(2*(t%13-6)+(ids[size_t(t)*K+k]%17-8));
        for (size_t col=0;col<N;++col) {
            if (fault) { need(observed[size_t(t)*N+col]==0xa5a5,"error published output");continue; }
            float sum=0,old=0;
            if (numeric) {
                for (auto k:sorted) sum=std::fma(weights[size_t(t)*K+k],kb::f32_from_bf16(kb::bf16_from_f32(down_value(t,ids[size_t(t)*K+k],int(col),true))),sum);
                for (size_t k=0;k<K;++k) old=std::fma(weights[size_t(t)*K+k],kb::f32_from_bf16(kb::bf16_from_f32(down_value(t,ids[size_t(t)*K+k],int(col),true))),old);
            } else {
                const int integer=integer_base+220*(int(col)%7-3);
                sum=float(integer)/1024.f;old=sum;
            }
            const float shared_term=kb::f32_from_bf16(shared_value(t,int(col)))*0.5f;
            const auto expected=kb::bf16_from_f32(sum+shared_term);
            old_order_differences+=size_t(expected!=kb::bf16_from_f32(old+shared_term));
            need(observed[size_t(t)*N+col]==expected,"complete FP32/BF16 numerical oracle mismatch");
        }
    }
    original_weights.guards();job_weights.guards();accumulator.guards();sg.guards();source.guards();inverse.guards();status.guards();down.guards();shared.guards();output.guards();
    std::cout<<"case rows="<<rows<<" hot="<<hot<<" numeric="<<numeric<<" fault="<<fault
             <<" jobs="<<plan->jobs().size()<<" expected_kernels="<<launches<<" status="<<status_observed
             <<" old_order_bf16_differences="<<old_order_differences<<" guards=PASS full_output=PASS host_plan=VALIDATED PASS\n";
}
void refused(const std::string& mode) {
    std::array<int32_t,10> ids{};std::iota(ids.begin(),ids.end(),0);
    const auto plan=pd::MoeWindowRoutes::build_accum(1,ids.data(),ids.size());
    const auto old=pd::MoeWindowRoutes::build(1,ids.data(),ids.size());
    auto& q=runtime->compute();
    Buffer<float> weight(10),job_weight(10),acc(N);
    Buffer<int32_t> inverse(10),source(10);Buffer<uint32_t> status(1);
    const size_t bytes=mode=="short-down"?N*2-2:10*4-4;
    sb::Allocation short_buffer(runtime,bytes,sb::MemoryKind::Device);
    Pending pending;
    bool rejected=false;
    try {
        if (mode=="short-weights") pn::moe_job_weights(*plan,short_buffer.as<float>(),inverse.data(),job_weight.data(),status.data(),&q);
        else if (mode=="short-down") pn::moe_accumulate(*plan,0,short_buffer.as<uint16_t>(),source.data(),inverse.data(),job_weight.data(),acc.data(),status.data(),&q);
        else pn::moe_job_weights(*old,nullptr,nullptr,nullptr,nullptr,&q);
    } catch (const std::invalid_argument&) { rejected=true; }
    need(rejected,"invalid span/plan was not refused before enqueue");
    std::cout<<"refusal="<<mode<<" expected_kernels=0 PASS\n";
}
// Isolated copy contract: arbitrary logical rows intentionally test both banks
// within one expert job. This is not an accumulator route/slot validity oracle.
void mixed_gather(int rows, int primary_rows) {
    const size_t cells=size_t(rows)*N, routes=size_t(rows)*K;
    std::vector<uint16_t> input(cells),observed(J*N),reference(J*N),capture;
    std::vector<int32_t> ids(routes),map(routes),observed_map(routes);
    for (size_t r=0;r<size_t(rows);++r) {
        for(size_t col=0;col<N;++col)input[r*N+col]=uint16_t((r*31337+col*97+(r>>3))&65535);
        for(size_t slot=0;slot<K;++slot)ids[r*K+slot]=int32_t(slot);
    }
    const auto plan=pd::MoeWindowRoutes::build_accum(rows,ids.data(),ids.size());
    const auto& job=plan->jobs().front();need(job.rows==int(J),"full gather job required");
    map=plan->source_rows();
    for(size_t i=0;i<J;++i) {
        const int edge[4]={0,primary_rows-1,std::min(primary_rows,rows-1),rows-1};
        map[size_t(job.global_row)+i]=i%5?edge[i%4]:int((i*193+17)%size_t(rows));
    }
    auto& q=runtime->compute();
    Buffer<uint16_t> contiguous(cells),primary(size_t(primary_rows)*N),tail(size_t(rows-primary_rows)*N),out(J*N),old(J*N);
    Buffer<int32_t> source(routes);Buffer<uint32_t> status(1);Pending pending;
    contiguous.poison(q);primary.poison(q);tail.poison(q);out.poison(q);old.poison(q);source.poison(q);status.poison(q);
    q.memcpy(contiguous.data(),input.data(),cells*2);
    q.memcpy(primary.data(),input.data(),size_t(primary_rows)*N*2);
    if(rows>primary_rows)q.memcpy(tail.data(),input.data()+size_t(primary_rows)*N,size_t(rows-primary_rows)*N*2);
    q.memcpy(source.data(),map.data(),routes*4);q.memset(status.data(),0,4);runtime->wait();
    auto launch=[&](const uint16_t* p,const uint16_t* t,int split,const int32_t* s,uint16_t* dst) {
        pn::moe_gather_mixed(*plan,0,p,t,split,s,dst,status.data(),&q);
    };
    const auto* second=rows>primary_rows?tail.data():nullptr;
    launch(primary.data(),second,primary_rows,source.data(),out.data());
    strata::prefill::gather_rows16(contiguous.data(),source.data()+job.global_row,old.data(),job.rows,N,&q);
    uint32_t observed_status=~uint32_t(0);
    q.memcpy(observed.data(),out.data(),observed.size()*2);q.memcpy(reference.data(),old.data(),reference.size()*2);
    q.memcpy(&observed_status,status.data(),4);runtime->wait();
    need(!observed_status && observed==reference,"two-bank/contiguous full-output mismatch");
    for(size_t i=0;i<observed.size();++i)
        need(observed[i]==input[size_t(map[size_t(job.global_row)+i/N])*N+i%N],"independent gather copy oracle");
    size_t rejects=0;
    auto reject=[&](auto call) {
        bool refused=false;try{call();}catch(const std::invalid_argument&){refused=true;}
        need(refused,"mixed gather bad admission submitted");++rejects;
    };
    reject([&]{launch(primary.data(),second,0,source.data(),out.data());});
    reject([&]{launch(primary.data(),second,rows+1,source.data(),out.data());});
    reject([&]{launch(primary.data()+size_t(primary_rows)*N-2,second,primary_rows,source.data(),out.data());});
    reject([&]{launch(primary.data(),second,primary_rows,source.data()+routes-2,out.data());});
    reject([&]{launch(primary.data(),second,primary_rows,source.data(),primary.data());});
    if(second)reject([&]{launch(primary.data(),tail.data()+size_t(rows-primary_rows)*N-2,primary_rows,source.data(),out.data());});
    else reject([&]{launch(primary.data(),tail.data(),primary_rows,source.data(),out.data());});
    const int bad[4]={-1,rows,INT32_MAX,INT32_MIN};
    for(size_t i=0;i<4;++i)map[size_t(job.global_row)+i]=bad[i];
    q.memcpy(source.data(),map.data(),routes*4);q.memset(status.data(),0,4);
    launch(primary.data(),second,primary_rows,source.data(),out.data());
    q.memcpy(observed.data(),out.data(),observed.size()*2);q.memcpy(&observed_status,status.data(),4);runtime->wait();
    need(observed_status==4,"invalid row sticky gather status");
    for(size_t i=0;i<observed.size();++i) {
        const int row=map[size_t(job.global_row)+i/N];
        need(observed[i]==(row<0 || row>=rows?0:input[size_t(row)*N+i%N]),"invalid row accessed source memory");
    }
    // Whole logical input banks and maps, not only guards, must be unchanged.
    capture.resize(size_t(primary_rows)*N);q.memcpy(capture.data(),primary.data(),capture.size()*2);runtime->wait();
    need(std::equal(capture.begin(),capture.end(),input.begin()),"primary bank mutated");
    if(second){capture.resize(size_t(rows-primary_rows)*N);q.memcpy(capture.data(),tail.data(),capture.size()*2);runtime->wait();need(std::equal(capture.begin(),capture.end(),input.begin()+size_t(primary_rows)*N),"tail bank mutated");}
    q.memcpy(observed_map.data(),source.data(),routes*4);runtime->wait();need(observed_map==map,"source map mutated");
    contiguous.guards();primary.guards();tail.guards();out.guards();old.guards();source.guards();status.guards();
    need(rejects==6,"mixed gather admission coverage");
    std::cout<<"mixed_gather rows="<<rows<<" primary_rows="<<primary_rows<<" compared_values="<<observed.size()
             <<" admission_rejects="<<rejects<<" invalid_rows=4 status=4 inputs_immutable=1 guards=PASS PASS\n";
}
} // namespace
int main(int argc,char** argv) {
    try {
        std::cout.setf(std::ios::unitbuf);need(argc==3 && std::string(argv[1])=="--case","case required");
        const std::string mode=argv[2];need(std::fegetround()==FE_TONEAREST,"host RNE required");
        gate("before-runtime");runtime=sb::runtime_for();
        std::cout<<"device="<<runtime->device().get_info<sycl::info::device::name>()<<" driver="<<runtime->device().get_info<sycl::info::device::driver_version>()<<'\n';
        gate("runtime-ready");
        if(mode=="mixed-gather-smoke")mixed_gather(8193,8192);
        else if(mode=="mixed-gather-contiguous")mixed_gather(16384,16384);
        else if(mode=="mixed-gather-65537")mixed_gather(65537,65536);
        else if(mode=="mixed-gather-partial")mixed_gather(78452,65536);
        else if(mode=="mixed-gather-full")mixed_gather(81920,65536);
        else if (mode=="short-weights" || mode=="short-down" || mode=="legacy-plan") refused(mode);
        else {
            need(mode=="smoke" || mode=="numeric" || mode=="hot65536" || mode=="balanced65536" || mode=="hot81920" || mode=="balanced81920" || mode=="partial78452" || mode=="negative-source" || mode=="negative-slot" || mode=="unwind","unsupported case");
            const int rows=mode=="smoke"?3:mode=="numeric"?37:(mode=="hot65536" || mode=="balanced65536")?65536:(mode=="hot81920" || mode=="balanced81920")?81920:mode=="partial78452"?78452:17;
            try { run(rows,mode=="hot65536" || mode=="hot81920",mode=="numeric",mode=="negative-source"?1:mode=="negative-slot"?2:0,mode=="unwind"); }
            catch (const Stop&) { need(mode=="unwind","unexpected stop");std::cout<<"unwind=PASS numeric=UNRUN expected_kernels=3\n"; }
        }
        runtime->wait();runtime.reset();gate("after-cleanup");std::cout<<"fixture_status=PASS model_PP=UNRUN\n";return 0;
    } catch (const std::exception& error) {
        sb::checked_cleanup([] { if (runtime) runtime->wait(); },[] {},"accumulator fixture error");
        std::cerr<<"accumulator fixture failed: "<<error.what()<<'\n';return 1;
    }
}
