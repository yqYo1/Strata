// Synthetic BF16 boundary mechanics only. No model quality/performance claim.
// Run each mode in a separate process. Configure runtime variables before exec.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/layer_major_boundary_audit.hpp"
#include "strata/prefill/layer_major_plan.hpp"
#include "strata/prefill/draft_uploads.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/runtime.hpp"
#include <bit>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace p = strata::prefill;
namespace d = strata::prefill::detail;
namespace sb = strata::sycl_backend;
static std::shared_ptr<sb::Runtime> rt;
static void need(bool b, const char* message) {
    if (!b) throw std::runtime_error(message);
}
// Fail stop with DMA operands live if submission/completion cannot be proved.
static void copy_checked(void* dst, const void* src, size_t bytes) {
    if (!bytes) return;
    try { rt->wait(rt->compute().memcpy(dst, src, bytes)); }
    catch (...) { sb::cleanup_fail_stop("BF16 fixture copy failed with owners live"); }
}
static bool drain() noexcept {
    try { rt->wait(); return true; } catch (...) { return false; }
}
static void fatal() noexcept { sb::cleanup_fail_stop("BF16 fixture queue retirement failed"); }

struct Case { uint32_t input; uint16_t packed; };
// Literal integer oracle independent of the helper: even/odd retained-bit ties,
// both signs, finite overflow, all relevant subnormal edges, sNaN/qNaN payloads.
static constexpr Case cases[] = {
    {0x00000000,0x0000},{0x80000000,0x8000},
    {0x3f800000,0x3f80},{0xbf800000,0xbf80},
    {0x3f808000,0x3f80},{0x3f818000,0x3f82},
    {0xbf808000,0xbf80},{0xbf818000,0xbf82},
    {0x7f800000,0x7f80},{0xff800000,0xff80},
    {0x7f7f0000,0x7f7f},{0x7f7f7fff,0x7f7f},
    {0x7f7f8000,0x7f80},{0xff7f8000,0xff80},{0x7f7fffff,0x7f80},
    {0x00010000,0x0001},{0x80010000,0x8001},
    {0x00008000,0x0000},{0x80008000,0x8000},
    {0x00008001,0x0001},{0x80008001,0x8001},
    {0x00018000,0x0002},{0x80018000,0x8002},
    {0x00000001,0x0000},{0x80000001,0x8000},
    {0x007fffff,0x0080},{0x807fffff,0x8080},
    {0x00800000,0x0080},{0x80800000,0x8080},
    {0x7f800001,0x7fc0},{0xff800001,0xffc0},
    {0x7fffffff,0x7fff},{0xffffffff,0xffff},
    {0x7fc12345,0x7fc1},{0xffa10001,0xffe1}
};

static void conversion() {
    const size_t n = std::size(cases); // Deliberately odd scalar count.
    std::vector<float> input(n), output(n);
    std::vector<uint16_t> actual(n);
    sb::Allocation x(rt,n*4,sb::MemoryKind::Device), y(rt,n*2,sb::MemoryKind::Device),
                   z(rt,n*4,sb::MemoryKind::Device);
    d::DraftUploadGuard lease(drain,fatal); lease.arm();
    auto rejects=[](auto&& call) {
        bool rejected=false;
        try { call(); } catch (const std::invalid_argument&) { rejected=true; }
        need(rejected,"invalid boundary conversion must reject before enqueue");
    };
    rejects([&]{p::residual_pack_bf16(x.as<float>(),y.as<uint16_t>(),-1,&rt->compute());});
    rejects([&]{p::residual_unpack_bf16(y.as<uint16_t>(),z.as<float>(),-1,&rt->compute());});
    rejects([&]{p::residual_pack_bf16(x.as<float>(),y.as<uint16_t>(),INT64_MAX,&rt->compute());});
    rejects([&]{p::residual_unpack_bf16(y.as<uint16_t>(),z.as<float>(),INT64_MAX,&rt->compute());});
    rejects([&]{p::residual_pack_bf16(x.as<float>(),x.as<uint16_t>(),int64_t(n),&rt->compute());});
    rejects([&]{p::residual_unpack_bf16(z.as<uint16_t>(),z.as<float>(),int64_t(n),&rt->compute());});
    // The API must reject an unordered queue before submitting a kernel.
    sycl::queue unordered(rt->context(),rt->device());
    auto unordered_drain=[&]() noexcept {
        try { unordered.wait_and_throw(); return true; } catch (...) { return false; }
    };
    d::DraftUploadGuard unordered_lease(unordered_drain,fatal); unordered_lease.arm();
    rejects([&]{p::residual_pack_bf16(x.as<float>(),y.as<uint16_t>(),int64_t(n),&unordered);});
    rejects([&]{p::residual_unpack_bf16(y.as<uint16_t>(),z.as<float>(),int64_t(n),&unordered);});
    need(unordered_drain(),"unordered rejection queue drained"); unordered_lease.retire();
    uint64_t nan_inputs=0, signed_zeros=0, infinities=0, finite_overflows=0,
             subnormal_results=0, underflows_to_zero=0;
    for (size_t i=0;i<n;++i) {
        input[i]=std::bit_cast<float>(cases[i].input);
        need(strata::kernels::bf16_from_f32(input[i])==cases[i].packed,"literal host BF16 pack oracle");
        const uint32_t magnitude=cases[i].input&0x7fffffffu;
        const uint16_t packed_magnitude=cases[i].packed&0x7fffu;
        nan_inputs+=magnitude>0x7f800000u;
        signed_zeros+=magnitude==0;
        infinities+=magnitude==0x7f800000u;
        finite_overflows+=magnitude<0x7f800000u && packed_magnitude==0x7f80u;
        subnormal_results+=packed_magnitude>0 && packed_magnitude<0x0080u;
        underflows_to_zero+=magnitude>0 && magnitude<0x7f800000u && packed_magnitude==0;
    }
    copy_checked(x.data(),input.data(),n*4);
    p::residual_pack_bf16(x.as<float>(),y.as<uint16_t>(),int64_t(n),&rt->compute());
    p::residual_unpack_bf16(y.as<uint16_t>(),z.as<float>(),int64_t(n),&rt->compute());
    copy_checked(actual.data(),y.data(),n*2); copy_checked(output.data(),z.data(),n*4);
    for (size_t i=0;i<n;++i) {
        need(actual[i]==cases[i].packed,"literal device BF16 RNE/NaN pack bits");
        need(std::bit_cast<uint32_t>(output[i])==uint32_t(cases[i].packed)<<16,"exact BF16 expansion");
    }
    // All 65,536 encodings, including signaling NaNs, must expand by bit shift.
    // Do not use FP arithmetic/comparisons on the restored NaN values.
    std::vector<uint16_t> codes(65536);
    std::vector<float> decoded(codes.size());
    sb::Allocation packed(rt,codes.size()*2,sb::MemoryKind::Device),
                   restored(rt,codes.size()*4,sb::MemoryKind::Device);
    // These new owners follow the first lease. A second lease must retire them
    // before unwinding them on any later failure.
    d::DraftUploadGuard exhaustive_lease(drain,fatal); exhaustive_lease.arm();
    for (size_t i=0;i<codes.size();++i) codes[i]=uint16_t(i);
    copy_checked(packed.data(),codes.data(),codes.size()*2);
    p::residual_unpack_bf16(packed.as<uint16_t>(),restored.as<float>(),int64_t(codes.size()),&rt->compute());
    copy_checked(decoded.data(),restored.data(),decoded.size()*4);
    for(size_t i=0;i<codes.size();++i)
        need(std::bit_cast<uint32_t>(decoded[i])==uint32_t(i)<<16,"all BF16 expansion encodings exact");
    need(drain(),"conversion final checked drain"); exhaustive_lease.retire(); lease.retire();
    std::cout << "{\"version\":1,\"arm\":\"bf16_boundary_storage\",\"mode\":\"conversion\","
        "\"residual_dtype\":\"bf16\",\"residual_element_bytes\":2,\"pack_cases\":" << n <<
        ",\"nan_inputs\":" << nan_inputs << ",\"signed_zeros\":" << signed_zeros <<
        ",\"infinities\":" << infinities << ",\"finite_overflows\":" << finite_overflows <<
        ",\"subnormal_results\":" << subnormal_results << ",\"underflows_to_zero\":" << underflows_to_zero <<
        ",\"pack_oracle\":\"literal_bits_and_bf16_from_f32_RNE_NaN_preserving\","
        "\"unpack_oracle\":\"all_65536_exact_shift16\","
        "\"invalid_geometry_or_queue_rejections\":8,"
        "\"expansion_cases\":65536,\"packed_bits_equal\":true,\"restored_bits_equal\":true,"
        "\"status\":\"complete\",\"quality_qualified\":false,\"performance_qualified\":false,"
        "\"physical_262144_qualified\":false}\n";
}

static void store(const std::string& mode) {
    constexpr int64_t rows=19, chunk=4, width=10240, layers=48;
    const int64_t prefix=mode=="fullgpu"?rows:mode=="fullhost"?0:8;
    constexpr uint64_t row_bytes=width*2;
    const uint64_t budget=(uint64_t(prefix)+(prefix<rows?chunk:0))*row_bytes;
    const auto plan=d::LayerMajorResidualPlan::make(rows,chunk,width,budget,prefix,d::LayerMajorResidualDtype::bf16);
    // The caller supplies immutable environment before exec, never setenv here.
    const char* selected=std::getenv("STRATA_PREFILL_BOUNDARY_ROWS");
    need(selected && std::strcmp(selected,"0,3,4,7,8,15,16,18")==0,"fixture requires documented row selection");
    need(!std::getenv("STRATA_PREFILL_BOUNDARY_ROWS_ABSOLUTE"),"fixture uses relative rows");
    d::LayerMajorBoundaryAudit audit(rows,0,width,layers,chunk,plan.dtype);
    // Every host buffer outlives the checked lease, including audit Copy targets.
    std::vector<uint8_t> host(size_t(plan.host_bytes));
    std::vector<float> input(size_t(chunk*width)), output(input.size());
    std::vector<uint16_t> observed(input.size());
    sb::Allocation gpu, stage;
    if(plan.gpu_bytes) gpu=sb::Allocation(rt,size_t(plan.gpu_bytes),sb::MemoryKind::Device);
    if(plan.staging_bytes) stage=sb::Allocation(rt,size_t(plan.staging_bytes),sb::MemoryKind::Device);
    sb::Allocation live(rt,input.size()*4,sb::MemoryKind::Device);
    d::DraftUploadGuard lease(drain,fatal); lease.arm();
    d::LayerMajorBoundaryAudit::Copy copy=[](void* dst,const void* src,size_t bytes,bool device) {
        if(device) copy_checked(dst,src,bytes); else std::memcpy(dst,src,bytes);
        return true;
    };
    uint64_t packs=0,unpacks=0,packed_scalars=0;
    for(int64_t layer=0;layer<layers;++layer) {
        for(int64_t row=0;row<rows;row+=chunk) {
            const int64_t count=std::min(chunk,rows-row);
            const auto span=plan.span(row,count);
            auto* base=span.device?gpu.as<uint8_t>():host.data();
            auto* location=base+size_t(span.offset_bytes);
            uint16_t* packed=span.device?reinterpret_cast<uint16_t*>(location):stage.as<uint16_t>();
            const size_t n=size_t(count*width);
            if(layer) {
                if(!span.device) copy_checked(packed,location,size_t(span.bytes));
                p::residual_unpack_bf16(packed,live.as<float>(),int64_t(n),&rt->compute()); ++unpacks;
                need(audit.input(layer,row,count,live.as<float>(),span.device,copy),"audit restored input");
                copy_checked(output.data(),live.data(),n*4);
                for(size_t i=0;i<n;++i) {
                    const auto& c=cases[(size_t(row*width)+i+size_t(layer-1))%std::size(cases)];
                    need(std::bit_cast<uint32_t>(output[i])==uint32_t(c.packed)<<16,"restored row bit oracle");
                }
            }
            // An explicit synthetic layer result injects every edge case again;
            // no model operators or model quality are being simulated.
            for(size_t i=0;i<n;++i)
                input[i]=std::bit_cast<float>(cases[(size_t(row*width)+i+size_t(layer))%std::size(cases)].input);
            copy_checked(live.data(),input.data(),n*4);
            need(audit.source(layer,row,count,live.as<float>(),copy),"audit output source");
            if(layer+1==layers) continue; // Final callback remains FP32.
            need(packed && (span.device || span.bytes<=plan.staging_bytes),"packed stage bounds");
            p::residual_pack_bf16(live.as<float>(),packed,int64_t(n),&rt->compute()); ++packs;
            if(!span.device) copy_checked(location,packed,size_t(span.bytes));
            need(drain(),"stored boundary checked drain");
            need(audit.stored(layer,row,count,location,span.device,copy),"audit stored bits");
            copy(observed.data(),location,size_t(span.bytes),span.device);
            for(size_t i=0;i<n;++i)
                need(observed[i]==cases[(size_t(row*width)+i+size_t(layer))%std::size(cases)].packed,"stored row bit oracle");
            packed_scalars+=n;
        }
    }
    need(packs==235 && unpacks==packs && packed_scalars==47*19*10240,"all non-final boundary chunks exactly once");
    need(drain(),"store final checked drain"); audit.complete(); lease.retire();
    std::cout << "{\"version\":1,\"arm\":\"bf16_boundary_storage\",\"mode\":\"" << mode <<
        "\",\"residual_dtype\":\"bf16\",\"residual_element_bytes\":2,\"rows\":19,\"chunk\":4,\"layers\":48,"
        "\"row_bytes\":" << plan.row_bytes << ",\"residual_total\":" << plan.total_bytes <<
        ",\"residual_gpu_rows\":" << plan.gpu_rows << ",\"residual_gpu\":" << plan.gpu_bytes <<
        ",\"residual_host\":" << plan.host_bytes << ",\"residual_staging\":" << plan.staging_bytes <<
        ",\"pack_calls\":" << packs << ",\"unpack_calls\":" << unpacks <<
        ",\"packed_scalars\":" << packed_scalars <<
        ",\"packed_bits_equal\":true,\"restored_bits_equal\":true,\"status\":\"complete\","
        "\"quality_qualified\":false,\"performance_qualified\":false,\"physical_262144_qualified\":false}\n";
}
int main(int argc,char** argv) {
    try {
        need(argc==2,"usage: sycl_prefill_bf16_storage_test conversion|fullgpu|fullhost|mixedstore");
        const std::string mode=argv[1];
        need(mode=="conversion"||mode=="fullgpu"||mode=="fullhost"||mode=="mixedstore","invalid fixture mode");
        rt=sb::runtime_for(); // No process environment mutation after this point.
        if(mode=="conversion") conversion(); else store(mode);
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
