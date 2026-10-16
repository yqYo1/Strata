// Manual component diagnostic. Executes the exact proposed production helper.
// No model, decode, service, profiler, extra queue, or CTest registration.
#include "strata/prefill/sycl_qsa_score_gemm.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata;
using namespace sycl_backend;
constexpr int64_t Blocks = 65538, Cap = 2051;
constexpr float Sentinel = -12345.5f;
void require(bool x, const char* message) { if (!x) throw std::runtime_error(message); }
int64_t integer(const char* p) {
    int64_t v; auto e=p+std::strlen(p); auto r=std::from_chars(p,e,v);
    if (r.ec!=std::errc{} || r.ptr!=e) throw std::invalid_argument("invalid exact integer");
    return v;
}
struct Buffer {
    Allocation storage;
    size_t bytes;
    Buffer(std::shared_ptr<Runtime> rt, size_t n) : storage(rt,n+512,MemoryKind::Device),bytes(n) {
        rt->compute().memset(storage.data(),0xa5,n+512);
    }
    template<class T> T* as() { return static_cast<T*>(storage.at(256,bytes)); }
    void guards(sycl::queue& q) {
        unsigned char a[256], b[256];
        q.memcpy(a,storage.data(),256);
        q.memcpy(b,storage.at(256+bytes,256),256).wait_and_throw();
        for (int j=0;j<256;++j) require(a[j]==0xa5 && b[j]==0xa5,"buffer redzone changed");
    }
};
float key_value(int64_t b, const std::string& kind) {
    if (kind=="near-tie") return 1.f+float(b%31)/4096;
    if (kind=="overflow") return 65536.f;
    if (kind=="nonfinite") return b%3==0 ? std::numeric_limits<float>::quiet_NaN() :
        b%3==1 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
    if (kind=="tie") return 0;
    return float(b%17-8)/16;
}
float query_value(int64_t qi,int h,const std::string& kind) {
    if (kind=="subnormal") return 0x1p-24f;
    if (kind=="overflow" || kind=="nonfinite" || kind=="tie") return 1;
    if (kind=="near-tie") return 1.f+float(qi%7)/4096;
    return float(h-1)*float(qi%3+1)/16;
}
std::vector<int32_t> independent_ids(int64_t n, int64_t width,
                                    const float* score) {
    // Truth over individual cells, independent of the production block radix
    // threshold algorithm. NaNs rank below all numbers, +/-zero tie by cell ID.
    std::vector<int32_t> cells(size_t(n),0);
    for (int64_t i=0;i<n;++i) cells[size_t(i)]=int32_t(i);
    auto rank = [&](int32_t a,int32_t b) {
        float x=score[a/4],y=score[b/4];
        if (std::isnan(x) || std::isnan(y)) {
            if (std::isnan(x) && std::isnan(y)) return a<b;
            return !std::isnan(x);
        }
        return x==y ? a<b : x>y;
    };
    if (width<n) std::nth_element(cells.begin(),cells.begin()+width,cells.end(),rank);
    cells.resize(size_t(width)); std::sort(cells.begin(),cells.end()); return cells;
}
}
int main(int argc,char** argv) try {
    if (argc!=6) throw std::invalid_argument(
        "usage: sycl_qsa_score_gemm_component --validate|--benchmark nq last_n samples exact|near-tie|overflow|subnormal|nonfinite|tie");
    const std::string action=argv[1],kind=argv[5];
    const auto nq=integer(argv[2]),last=integer(argv[3]),samples=integer(argv[4]);
    require(action=="--validate" || action=="--benchmark","unknown action");
    require(nq>=1 && nq<=256 && last>=nq && last<=262144 && samples>=1 && samples<=9,"bounded shape/sample contract");
    require(kind=="exact" || kind=="near-tie" || kind=="overflow" || kind=="subnormal" || kind=="nonfinite" || kind=="tie","unknown kind");
    require(prefill::qsa_score_mode()!=prefill::QsaScoreMode::off,"set the precision arm before startup");
    const size_t scratch=prefill::qsa_score_scratch_bytes(Blocks,256);
    const size_t poolbytes=size_t(Blocks)*128*4, querybytes=size_t(nq)*512*4,
        stepbytes=size_t(nq)*kernels::kStepCount*4, scorebytes=size_t(nq)*Blocks*4,
        idbytes=size_t(nq)*Cap*4;
    const size_t device=poolbytes+512+querybytes+stepbytes+2*scorebytes+2*idbytes+scratch+9*512;
    const size_t host=poolbytes+querybytes+stepbytes+2*scorebytes+2*idbytes+512+size_t(last)*8+size_t(Cap)*8+65536;
    require(device+host < size_t(512)*1024*1024,"512 MiB component live-byte admission");
    // Includes host mirrors + expanded-cell oracle + guarded device allocations.
    // oneMKL/driver's internal memory is separately observed by Main, not claimed
    // bounded by this explicit allocation sum.
    std::cout << "explicit_device_bytes="<<device<<" host_budget_bytes="<<host
              <<" scratch_bytes="<<scratch<<" nq="<<nq<<" last_n="<<last<<" kind="<<kind<<'\n';
    std::vector<float> pooled(size_t(Blocks)*128,std::numeric_limits<float>::quiet_NaN()),
        query(size_t(nq)*512),dead(128,0),ref(size_t(nq)*Blocks),cand(ref.size());
    std::vector<int32_t> steps(size_t(nq)*kernels::kStepCount),ri(size_t(nq)*Cap,-777),ci(ri.size(),-777);
    const int64_t live=last/4, active=live+1;
    for (int64_t b=0;b<live;++b)
        std::fill_n(pooled.data()+b*128,128,key_value(b,kind));
    for (int64_t qi=0;qi<nq;++qi) {
        const auto n=last-nq+qi+1;
        auto st=steps.data()+qi*kernels::kStepCount;
        st[kernels::kStepPos]=int32_t(n-1); st[kernels::kStepNKv]=int32_t(n);
        st[kernels::kStepNBid]=int32_t(n/4); st[kernels::kStepWidth]=int32_t(std::min(n,Cap));
        for (int h=0;h<4;++h) std::fill_n(query.data()+qi*512+h*128,128,query_value(qi,h,kind));
    }
    auto rt=runtime_for(); auto& q=rt->compute();
    Buffer p(rt,poolbytes),d(rt,512),x(rt,querybytes),st(rt,stepbytes),r(rt,scorebytes),
        c(rt,scorebytes),rid(rt,idbytes),cid(rt,idbytes),work(rt,scratch);
    q.memcpy(p.as<float>(),pooled.data(),poolbytes); q.memcpy(d.as<float>(),dead.data(),512);
    q.memcpy(x.as<float>(),query.data(),querybytes); q.memcpy(st.as<int32_t>(),steps.data(),stepbytes);
    prefill::QsaScoreGemm gemm; gemm.bind(work.as<void>(),scratch,Blocks,256);
    const auto shape=kernels::qsa_real_shapes();
    auto run=[&](bool candidate,bool timed,int sample) {
        auto out=candidate?c.as<float>():r.as<float>(); auto ids=candidate?cid.as<int32_t>():rid.as<int32_t>();
        q.fill(out,Sentinel,size_t(nq)*Blocks); q.fill(ids,int32_t(-777),size_t(nq)*Cap); rt->wait();
        const auto start=std::chrono::steady_clock::now();
        if (candidate) {
            gemm.prepare(p.as<float>(),d.as<float>(),last,shape,&q);
            gemm.scores(x.as<float>(),st.as<int32_t>(),nq,active,out,&q); gemm.invalidate();
        } else kernels::qsa_block_scores(p.as<float>(),d.as<float>(),x.as<float>(),st.as<int32_t>(),nq,Blocks,shape,out,&q,active);
        kernels::qsa_block_topk(out,st.as<int32_t>(),nq,Blocks,Cap,shape,ids,&q,active); rt->wait();
        const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        if (timed) std::cout<<"sample="<<sample<<" arm="<<(candidate?"candidate":"fp32")<<" selection_ns="<<ns<<'\n';
    };
    // First call is reported separately from warmed samples. Measurement covers
    // conversion + 4 products + postprocessing + top-k, with same-queue waits
    // solely at diagnostic timing boundaries. Production helper has no waits.
    run(false,true,-1); run(true,true,-1);
    for (int s=0;s<samples;++s) {
        if (s%2) { run(true,true,s);run(false,true,s); }
        else { run(false,true,s);run(true,true,s); }
    }
    q.memcpy(ref.data(),r.as<float>(),scorebytes); q.memcpy(cand.data(),c.as<float>(),scorebytes);
    q.memcpy(ri.data(),rid.as<int32_t>(),idbytes); q.memcpy(ci.data(),cid.as<int32_t>(),idbytes); rt->wait();
    size_t diffs=0,flips=0,valid=0,flip_queries=0,shared_cells=0,selected_cells=0; double maxabs=0,maxrel=0;
    for (int64_t qi=0;qi<nq;++qi) {
        const auto ss=steps.data()+qi*kernels::kStepCount;
        const int64_t n=ss[kernels::kStepNKv],nb=n/4,width=ss[kernels::kStepWidth];
        for (int64_t b=0;b<Blocks;++b) {
            auto a=ref[size_t(qi*Blocks+b)],v=cand[size_t(qi*Blocks+b)];
            if (b>nb) { require(a==Sentinel && v==Sentinel,"future score was written"); continue; }
            ++valid;
            if (std::bit_cast<uint32_t>(a)!=std::bit_cast<uint32_t>(v)) ++diffs;
            if (std::isfinite(a) && std::isfinite(v)) {
                maxabs=std::max(maxabs,std::abs(double(a)-v));
                maxrel=std::max(maxrel,std::abs(double(a)-v)/std::max(1e-30,std::abs(double(a))));
            }
            if (b==nb) require(a==(n%4?1e9f:0.f) && v==a,"dead/tail score semantics");
            if (kind=="exact" || kind=="tie" || kind=="overflow" || kind=="nonfinite") {
                // Constant rows/heads: high-precision independent dot has no
                // non-associativity here; all source values are exact powers
                // of two multiples. Overflow arm exercises FP16 GPU fallback.
                double truth=0;
                for (int h=0;h<4;++h) truth+=std::max(0.0,128.0*double(b==nb?0:key_value(b,kind))*query_value(qi,h,kind));
                if (b==nb && n%4) truth+=1e9;
                require(a==float(truth) && v==float(truth),"independent exact score oracle");
            }
        }
        auto expected_ref=independent_ids(n,width,ref.data()+qi*Blocks);
        auto expected_cand=independent_ids(n,width,cand.data()+qi*Blocks);
        size_t shared=0; int64_t ai=0,bi=0;
        while (ai<width && bi<width) {
            auto a=ri[size_t(qi*Cap+ai)],b=ci[size_t(qi*Cap+bi)];
            if (a==b) {++shared;++ai;++bi;} else if (a<b) ++ai; else ++bi;
        }
        shared_cells+=shared; selected_cells+=size_t(width); flip_queries+=(shared!=size_t(width));
        for (int64_t j=0;j<Cap;++j) {
            const auto k=size_t(qi*Cap+j);
            if (j>=width) { require(ri[k]==-777 && ci[k]==-777,"unused ID slot changed"); continue; }
            require(ri[k]==expected_ref[size_t(j)] && ci[k]==expected_cand[size_t(j)],"cell-wise top-k oracle");
            if (ri[k]!=ci[k]) ++flips;
        }
    }
    std::cout<<"PASS shape/causal/tail/future/tie/guard/component-oracle; valid_scores="<<valid
             <<" score_bit_diffs="<<diffs<<" max_abs="<<maxabs<<" max_rel="<<maxrel
             <<" selected_slot_diffs="<<flips<<" changed_queries="<<flip_queries<<" shared_cells="<<shared_cells<<" selected_cells="<<selected_cells<<"; precision differences are evidence, not model-quality acceptance\n";
    if (action=="--validate") {
        // Exercise one prepare shared by multiple selection batches, followed
        // by mutated live pooled/dead input and a new prepare. Constant exact
        // data makes this independent of dtype GEMM rounding/shape dispatch.
        auto check_epoch=[&](float key,float tail,bool split) {
            for (int64_t b=0;b<live;++b) std::fill_n(pooled.data()+b*128,128,key);
            std::fill(query.begin(),query.end(),1.f); std::fill(dead.begin(),dead.end(),tail);
            if (live) q.memcpy(p.as<float>(),pooled.data(),size_t(live)*128*4);
            q.memcpy(x.as<float>(),query.data(),querybytes); q.memcpy(d.as<float>(),dead.data(),512);
            q.fill(c.as<float>(),Sentinel,size_t(nq)*Blocks); q.fill(cid.as<int32_t>(),int32_t(-777),size_t(nq)*Cap);
            gemm.prepare(p.as<float>(),d.as<float>(),last,shape,&q);
            const int64_t batch=split && nq>1 ? (nq+1)/2 : nq;
            for (int64_t off=0;off<nq;off+=batch) {
                const int64_t count=std::min(batch,nq-off);
                const int64_t launched=steps[size_t((off+count-1)*kernels::kStepCount+kernels::kStepNBid)]+1;
                gemm.scores(x.as<float>()+off*512,st.as<int32_t>()+off*kernels::kStepCount,count,launched,c.as<float>()+off*Blocks,&q);
                kernels::qsa_block_topk(c.as<float>()+off*Blocks,st.as<int32_t>()+off*kernels::kStepCount,
                                      count,Blocks,Cap,shape,cid.as<int32_t>()+off*Cap,&q,launched);
            }
            gemm.invalidate();
            bool refused=false;
            try {gemm.scores(x.as<float>(),st.as<int32_t>(),nq,active,c.as<float>(),&q);}
            catch (const std::invalid_argument&) {refused=true;}
            require(refused,"invalidated arena epoch was accepted");
            q.memcpy(cand.data(),c.as<float>(),scorebytes);q.memcpy(ci.data(),cid.as<int32_t>(),idbytes);rt->wait();
            for (int64_t qi=0;qi<nq;++qi) {
                const int64_t n=steps[size_t(qi*kernels::kStepCount+kernels::kStepNKv)],nb=n/4,width=std::min(n,Cap);
                for (int64_t b=0;b<Blocks;++b) {
                    float truth=b>nb?Sentinel:(b==nb?512.f*tail:512.f*key);
                    if (b==nb && n%4) truth+=1e9f;
                    require(cand[size_t(qi*Blocks+b)]==truth,"prepared-key reuse or fresh pooled/dead epoch failed");
                }
                auto expected=independent_ids(n,width,cand.data()+qi*Blocks);
                for (int64_t j=0;j<Cap;++j)
                    require(ci[size_t(qi*Cap+j)]==(j<width?expected[size_t(j)]:-777),"epoch top-k/tail/unused IDs failed");
            }
        };
        check_epoch(1.f,2.f,true);
        check_epoch(3.f,4.f,false);
        std::cout<<"PASS cached-key split batches, invalidate refusal, mutated pooled/dead fresh epoch\n";
    }
    p.guards(q);d.guards(q);x.guards(q);st.guards(q);r.guards(q);c.guards(q);rid.guards(q);cid.guards(q);work.guards(q);
    std::cout<<"PASS all nine allocation redzones\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<'\n';return 1; }
