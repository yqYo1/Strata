// Private source-only synthetic qualifier; no fork source modification.
// Calls the pristine XeStrata public API, with independently decoded half bits.
#include "strata/core/runtime.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void need(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
size_t mul(size_t a, size_t b) {
    need(b == 0 || a <= std::numeric_limits<size_t>::max()/b, "size overflow");
    return a*b;
}
double half(uint16_t bits) {
    const unsigned exp = (bits >> 10) & 31u, frac = bits & 1023u;
    need(exp != 31, "nonfinite source half");
    const double magnitude = exp ? std::ldexp(double(1024u + frac), int(exp)-25)
                                 : std::ldexp(double(frac), -24);
    return bits & 32768u ? -magnitude : magnitude;
}
uint16_t operand(size_t a, size_t b, size_t group, bool weight) {
    // Finite normal dyadics, exponent fields9..13, full10-bit significands.
    // Different row/column/expert patterns expose grouping/stride/transposition.
    const uint64_t v = uint64_t(a)*131u + uint64_t(b)*71u + uint64_t(group)*977u + (weight ? 29u : 43u);
    return uint16_t(((9u + (v/7u)%5u) << 10) | ((v*37u + v/11u)&1023u) | ((v ^ (v/13u))&1u ? 32768u : 0u));
}
uint64_t hash(const void* data, size_t bytes) {
    uint64_t value = 14695981039346656037ull;
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i=0; i<bytes; ++i) { value ^= p[i]; value *= 1099511628211ull; }
    return value;
}
std::string quote(std::string text) {
    need(text.size() <= 4096, "device text bound");
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c >= 32 && c < 127) out += char(c);
        else { char b[7]; std::snprintf(b,sizeof(b),"\\u%04x",unsigned(c)); out += b; }
    }
    return out + "\"";
}
struct Plan {
    int64_t N=0, K=0, max_rows=0;
    size_t stride=0, rows=0;
    std::vector<int32_t> bounds;
};
Plan plan(const std::string& role, const std::string& which) {
    Plan p;
    need(role == "GU" || role == "Down", "role must be GU or Down");
    p.N = role == "GU" ? 1280 : 2560; p.K = role == "GU" ? 2560 : 640;
    std::vector<int32_t> counts;
    if (which == "1") counts={1};
    else if (which == "127") counts={127};
    else if (which == "128") counts={128};
    else if (which == "129") counts={129};
    else if (which == "257") counts={257};
    else if (which == "sparse") counts={0,1,129,257,0};
    else if (which == "empty") counts={0,0,0};
    else throw std::runtime_error("case must be 1/127/128/129/257/sparse/empty");
    p.bounds.push_back(0);
    for (int32_t n : counts) {
        need(n >= 0 && n <= 257 && p.bounds.back() <= 387-n, "row bound");
        p.bounds.push_back(p.bounds.back()+n); p.max_rows=std::max(p.max_rows,int64_t(n));
    }
    // Empty groups are exercised by a real launch with no active rows.
    p.max_rows=std::max<int64_t>(1,p.max_rows); p.rows=size_t(p.bounds.back());
    p.stride=mul(size_t(p.N),size_t(p.K))+32; //64-byte expert padding, deliberately not dense stride
    need(counts.size() <= 5 && p.rows <= 387 && p.stride*2%64 == 0, "fixed capacity");
    for (size_t e=0;e<counts.size();++e)
        need(p.bounds[e] <= p.bounds[e+1] && p.bounds[e+1]-p.bounds[e] <= p.max_rows, "monotone/max_rows");
    return p;
}
[[noreturn]] void unsafe(const char* why) {
    // Fatal receipt avoids allocation/throwing while completion is unknown.
    std::fputs("{\"passed\":false,\"unsafe_completion\":true,\"error\":\"",stderr);
    for(size_t i=0;why[i] && i<4096;++i) {
        const unsigned char c=static_cast<unsigned char>(why[i]);
        if(c=='"' || c=='\\') { std::fputc('\\',stderr); std::fputc(c,stderr); }
        else if(c>=32 && c<127) std::fputc(c,stderr);
        else std::fprintf(stderr,"\\u%04x",unsigned(c));
    }
    std::fputs("\"}\n",stderr);
    std::fflush(nullptr); std::_Exit(90); // no unwind past possibly live host or device buffers
}
int run(const Plan& p, const std::string& role, const std::string& which, bool cpu) {
    constexpr size_t hg=32, fg=16, bg=16; // each data pointer starts64 bytes after allocation
    const size_t G=p.bounds.size()-1, xcount=mul(std::max<size_t>(1,p.rows),size_t(p.K));
    const size_t wcount=mul(G,p.stride), ycount=mul(std::max<size_t>(1,p.rows),size_t(p.N));
    const size_t xbytes=mul(xcount+2*hg,sizeof(uint16_t)), wbytes=mul(wcount+2*hg,sizeof(uint16_t));
    const size_t ybytes=mul(ycount+2*fg,sizeof(uint32_t)), bbytes=mul(G+1+2*bg,sizeof(int32_t));
    const size_t device_bytes=xbytes+wbytes+ybytes+bbytes;
    need(device_bytes < 40*1024*1024 && device_bytes*3 < 128*1024*1024, "fixed memory budget");
    need(half(0x3c00)==1 && half(0xbc00)==-1 && half(1)==std::ldexp(1.0,-24) &&
         std::signbit(half(0x8000)) && half(0)==0, "half decode controls");
    if(cpu) {
        std::printf("{\"passed\":true,\"gpu_executed\":false,\"role\":%s,\"case\":%s,\"groups\":%zu,\"rows\":%zu,\"max_rows\":%lld,\"device_bytes\":%zu}\n",
                    quote(role).c_str(),quote(which).c_str(),G,p.rows,(long long)p.max_rows,device_bytes);
        return 0;
    }
    // All source/readback host storage is allocated before any device commands.
    // On a GPU exception the same-scope catch exits without destroying these buffers.
    std::vector<uint16_t> X(xcount+2*hg,0xa55a), W(wcount+2*hg,0xa55a), xr(X.size()), wr(W.size());
    std::vector<uint32_t> Y(ycount+2*fg,0x7fc12345u), yr(Y.size());
    std::vector<int32_t> B(G+1+2*bg,0x12345678), br(B.size());
    std::copy(p.bounds.begin(),p.bounds.end(),B.begin()+bg);
    for(size_t e=0;e<G;++e) {
        for(int32_t r=p.bounds[e];r<p.bounds[e+1];++r)
            for(size_t k=0;k<size_t(p.K);++k) X[hg+size_t(r)*size_t(p.K)+k]=operand(size_t(r),k,e,false);
        for(size_t n=0;n<size_t(p.N);++n)
            for(size_t k=0;k<size_t(p.K);++k) W[hg+e*p.stride+n*size_t(p.K)+k]=operand(n,k,e,true);
    }
    void* dx=nullptr; void* dw=nullptr; void* dy=nullptr; void* db=nullptr;
    try {
        auto& rt=strata::core::Runtime::get(); auto& q=rt.compute();
        need(q.has_property<sycl::property::queue::in_order>(), "compute queue not ordered");
        need(q.get_context()==rt.context() && q.get_device()==rt.device(), "runtime context/device");
        const bool available=strata::kernels::xmx_available(strata::kernels::XmxType::f16);
        const std::string path=strata::kernels::gemm_path(strata::kernels::XmxType::f16);
        const auto subgroups=q.get_device().get_info<sycl::info::device::sub_group_sizes>();
        std::printf("{\"kind\":\"capability\",\"reported_path\":%s,\"xmx_available\":%s,\"queue_in_order\":true,\"device\":%s,\"driver\":%s,\"subgroup16\":%s}\n",
                    quote(path).c_str(),available?"true":"false",quote(q.get_device().get_info<sycl::info::device::name>()).c_str(),
                    quote(q.get_device().get_info<sycl::info::device::driver_version>()).c_str(),
                    std::find(subgroups.begin(),subgroups.end(),size_t(16))!=subgroups.end()?"true":"false");
        std::fflush(stdout);
        std::fputs("{\"kind\":\"bounds\",\"bounds\":[",stdout);
        for(size_t i=0;i<p.bounds.size();++i) std::printf("%s%d",i?",":"",p.bounds[i]);
        std::puts("]}"); std::fflush(stdout);
        if(!available || path!="XMX") {
            rt.finish(); std::puts("{\"passed\":false,\"xmx_qualified\":false,\"reason\":\"actual runtime selected fallback; grouped XMX not run\"}"); return 3;
        }
        dx=sycl::aligned_alloc_device(256,xbytes,q); dw=sycl::aligned_alloc_device(256,wbytes,q);
        dy=sycl::aligned_alloc_device(256,ybytes,q); db=sycl::aligned_alloc_device(256,bbytes,q);
        need(dx&&dw&&dy&&db,"USM allocation");
        const auto pointer_ok=[&](void* ptr) { return sycl::get_pointer_type(ptr,rt.context())==sycl::usm::alloc::device; };
        need(pointer_ok(dx)&&pointer_ok(dw)&&pointer_ok(dy)&&pointer_ok(db),"same-context device buffers");
        auto* xp=static_cast<uint16_t*>(dx)+hg; auto* wp=static_cast<uint16_t*>(dw)+hg;
        auto* yp=reinterpret_cast<float*>(static_cast<uint32_t*>(dy)+fg); auto* bp=static_cast<int32_t*>(db)+bg;
        need(strata::kernels::xmx_gemm_ok(xp,wp,yp,p.N,p.K,p.N),"actual pointer/shape admission");
        q.memcpy(dx,X.data(),xbytes); q.memcpy(dw,W.data(),wbytes); q.memcpy(dy,Y.data(),ybytes);
        const auto sent=q.memcpy(db,B.data(),bbytes);
        // Non-null exact runtime compute stream preserves production asynchronous launch.
        // No new queue, copy-queue commands, profiling or per-kernel timing.
        strata::kernels::xmx_gemm_grouped(xp,wp,int64_t(p.stride),yp,bp,int(G),p.max_rows,p.N,p.K,&q);
        q.memcpy(xr.data(),dx,xbytes); q.memcpy(wr.data(),dw,wbytes); q.memcpy(yr.data(),dy,ybytes);
        const auto received=q.memcpy(br.data(),db,bbytes);
        rt.finish(q); // existing production runtime watchdog/async checks and one controlled drain
        need(sent.get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::complete &&
             received.get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::complete,"copy events incomplete");
        need(xr==X && wr==W && br==B,"complete sources/bounds/guards changed");
        for(size_t i=0;i<fg;++i) need(yr[i]==Y[i] && yr[fg+ycount+i]==Y[fg+ycount+i],"output prefix/tail guards");
        if(p.rows==0) need(yr==Y,"empty-group output changed");
        const double u32=std::ldexp(1.0,-24), u64=std::ldexp(1.0,-53);
        const double steps=double(2*p.K), gamma32=steps*u32/(1-steps*u32);
        const double gamma64=double(p.K)*u64/(1-double(p.K)*u64);
        double maxabs=0,maxratio=0,maxrelative=0,sumerr2=0,sumref2=0,sumabsmax=0;
        uint64_t checked=0,failed=0,nonfinite=0;
        for(size_t e=0;e<G;++e) for(int32_t r=p.bounds[e];r<p.bounds[e+1];++r) for(size_t n=0;n<size_t(p.N);++n) {
            double ref=0,sumabs=0;
            for(size_t k=0;k<size_t(p.K);++k) {
                const double product=half(X[hg+size_t(r)*size_t(p.K)+k])*half(W[hg+e*p.stride+n*size_t(p.K)+k]);
                ref+=product; sumabs+=std::abs(product);
            }
            const double actual=std::bit_cast<float>(yr[fg+size_t(r)*size_t(p.N)+n]);
            const double allowed=(gamma32+gamma64)*sumabs+u32*std::abs(ref);
            const double error=std::abs(actual-ref);
            ++checked;
            if(!std::isfinite(actual)) ++nonfinite;
            if(!std::isfinite(actual) || error>allowed) {
                if(failed<8) std::fprintf(stderr,"{\"kind\":\"difference\",\"group\":%zu,\"row\":%d,\"column\":%zu,\"reference\":%.17g,\"actual_bits\":%u,\"allowed\":%.17g}\n",e,r,n,ref,yr[fg+size_t(r)*size_t(p.N)+n],allowed);
                ++failed;
            }
            if(std::isfinite(actual)) {
                maxabs=std::max(maxabs,error); maxratio=std::max(maxratio,allowed?error/allowed:error);
                if(ref!=0) maxrelative=std::max(maxrelative,error/std::abs(ref));
                sumerr2+=error*error; sumref2+=ref*ref; sumabsmax=std::max(sumabsmax,sumabs);
            }
        }
        need(checked==mul(p.rows,size_t(p.N)),"reference/output reconciliation");
        rt.finish(); // drain all production-owned lanes before successful frees
        for(void* pointer : {dx,dw,dy,db}) rt.free(pointer);
        dx=dw=dy=db=nullptr;
        std::printf("{\"kind\":\"qualification\",\"passed\":%s,\"xmx_qualified\":%s,\"performance_eligible\":false,\"model_quality_qualified\":false,\"role\":%s,\"case\":%s,\"groups\":%zu,\"rows\":%zu,\"max_rows\":%lld,\"N\":%lld,\"K\":%lld,\"w_stride_half_elements\":%zu,\"device_bytes\":%zu,\"checked\":%llu,\"failed\":%llu,\"nonfinite\":%llu,\"max_absolute_error\":%.17g,\"max_nonzero_reference_relative_error\":%.17g,\"normalized_rms\":%.17g,\"max_error_over_bound\":%.17g,\"gamma32_2K\":%.17g,\"max_sumabs\":%.17g,\"sources_bounds_guards_equal\":true,\"copy_events_complete\":true,\"source_X_fnv1a64\":%llu,\"source_W_fnv1a64\":%llu,\"output_fnv1a64\":%llu}\n",
                    failed?"false":"true",failed?"false":"true",quote(role).c_str(),quote(which).c_str(),G,p.rows,
                    (long long)p.max_rows,(long long)p.N,(long long)p.K,p.stride,device_bytes,
                    (unsigned long long)checked,(unsigned long long)failed,(unsigned long long)nonfinite,maxabs,maxrelative,
                    sumref2?std::sqrt(sumerr2/sumref2):0,maxratio,gamma32,sumabsmax,
                    (unsigned long long)hash(X.data(),xbytes),(unsigned long long)hash(W.data(),wbytes),(unsigned long long)hash(yr.data(),ybytes));
        return failed?1:0;
    } catch(const std::exception& e) { unsafe(e.what()); }
      catch(...) { unsafe("unknown GPU exception"); }
}
}
int main(int argc,char** argv) {
    try {
        std::string role,which; bool cpu=false;
        for(int i=1;i<argc;++i) {
            const std::string a=argv[i];
            if(a=="--cpu-preflight") { need(!cpu,"duplicate cpu flag"); cpu=true; }
            else if(a=="--role" || a=="--case") {
                need(i+1<argc,"missing CLI value"); auto& target=a=="--role"?role:which;
                need(target.empty(),"duplicate CLI flag"); target=argv[++i]; need(target.size()<=16,"CLI bound");
            } else throw std::runtime_error("unknown CLI flag");
        }
        need(!role.empty()&&!which.empty(),"require --role GU|Down --case 1|127|128|129|257|sparse|empty [--cpu-preflight]");
        const Plan p=plan(role,which); // all parsing/shapes/invariants before any runtime use
        return run(p,role,which,cpu);
    } catch(const std::exception& e) {
        std::fprintf(stderr,"{\"passed\":false,\"gpu_executed\":false,\"error\":%s}\n",quote(e.what()).c_str()); return 2;
    }
}
