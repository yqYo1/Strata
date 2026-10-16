#include "strata/prefill/layer_major_boundary_audit.hpp"
#include <cassert>
#include <fstream>
#include <sstream>

using Audit = strata::prefill::detail::LayerMajorBoundaryAudit;
using Dtype = strata::prefill::detail::LayerMajorResidualDtype;
static unsigned cases = 0;
template<class F> void rejects(F action) {
    bool caught=false;
    try { action(); } catch (const std::exception&) { caught=true; }
    assert(caught); ++cases;
}
static std::string read(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    assert(f.good());
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
// Independent literal BF16 pack oracle. The source has both signs, RNE ties,
// finite overflow, subnormal/underflow edges and retained/small NaN payloads.
struct Bf16Case { uint32_t input; uint16_t packed; };
static constexpr Bf16Case bf16_cases[]={
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
static constexpr size_t bf16_case_count=sizeof(bf16_cases)/sizeof(bf16_cases[0]);
static size_t phase_count(const std::string& log,const std::string& phase) {
    const std::string prefix="phase="+phase+" ";
    std::istringstream lines(log); std::string line; size_t count=0;
    while(std::getline(lines,line)) if(line.compare(0,prefix.size(),prefix)==0) ++count;
    return count;
}
static bool placement_record(const std::string& log,const std::string& phase,
                             int row,const char* placement) {
    const std::string prefix="phase="+phase+" ";
    std::istringstream lines(log); std::string line;
    while(std::getline(lines,line))
        if(line.compare(0,prefix.size(),prefix)==0 &&
           line.find(" relative="+std::to_string(row)+" ")!=std::string::npos &&
           line.find(std::string(" storage=bf16 source=")+placement+" ")!=std::string::npos &&
           line.find(" memcmp=equal ")!=std::string::npos) return true;
    return false;
}
static bool phase_difference(const std::string& log,const std::string& phase) {
    const std::string prefix="phase="+phase+" ";
    std::istringstream lines(log); std::string line;
    while(std::getline(lines,line))
        if(line.compare(0,prefix.size(),prefix)==0 &&
           line.find(" memcmp=different ")!=std::string::npos) return true;
    return false;
}
int main(int argc, char** argv) {
    assert(argc==2);
    const std::string prefix=argv[1];
    auto path=[&](const std::string& name) {
        const std::string value=prefix+name;
        assert(::setenv("STRATA_PREFILL_BOUNDARY_PATH",value.c_str(),1)==0);
        return value;
    };
    assert(::setenv("STRATA_PREFILL_BOUNDARY_ROWS","0,1",1)==0);
    ::unsetenv("STRATA_PREFILL_BOUNDARY_ROWS_ABSOLUTE");
    Audit::Copy copy=[](void* d,const void* s,size_t n,bool) { std::memcpy(d,s,n); return true; };
    // Raw FP32 payload includes signed zero, subnormals, infinities and NaN payloads.
    // Compare actual bytes, independently of hash and floating comparison behavior.
    const uint32_t bits[]={0,0x80000000u,1,0x80000001u,0x3f800000u,0x7f800000u,
        0xff800000u,0x7fc12345u,0x7f812345u};
    for (bool half:{false,true}) {
        const Dtype dtype=half?Dtype::fp16:Dtype::fp32;
        const auto output=path(half?".half":".float");
        std::string expected;
        {
            Audit audit(2,100,10240,48,1,dtype);
            std::vector<float> r(10240),saved(20480);
            std::vector<uint16_t> packed(20480);
            for (int l=0;l<48;++l) for (int row=0;row<2;++row) {
                if (l) {
                    if (half) for (int i=0;i<10240;++i)
                        r[i]=strata::kernels::f32_from_f16(packed[row*10240+i]);
                    else std::memcpy(r.data(),saved.data()+row*10240,r.size()*4);
                    assert(audit.input(l,row,1,r.data(),false,copy));
                }
                for (int i=0;i<10240;++i) {
                    if (half) r[i]=float(l+row)+float(i%97)/1024;
                    else std::memcpy(&r[i],&bits[(i+l+row)%9],4);
                }
                expected.append(reinterpret_cast<const char*>(r.data()),r.size()*4);
                assert(audit.source(l,row,1,r.data(),copy));
                if (l!=47) {
                    std::memcpy(saved.data()+row*10240,r.data(),r.size()*4);
                    for (int i=0;i<10240;++i) packed[row*10240+i]=strata::kernels::f16_from_f32(r[i]);
                    const void* data=half?static_cast<void*>(packed.data()+row*10240):static_cast<void*>(saved.data()+row*10240);
                    assert(audit.stored(l,row,1,data,false,copy));
                }
            }
            audit.complete();
        }
        const auto log=read(output);
        assert(log.find("status=complete boundary_memcmp=equal\n")!=std::string::npos);
        assert(log.find("memcmp=different")==std::string::npos);
        assert(phase_count(log,"output_before_store")==96);
        assert(phase_count(log,half?"stored_RNE_packed_bits":"stored")==94);
        assert(phase_count(log,half?"next_input_decoded_expected":"next_input")==94);
        assert(log.find(half?"storage=fp16 ":"storage=fp32 ")!=std::string::npos);
        assert(phase_count(log,"stored_bf16_RNE_packed_bits")==0);
        assert(phase_count(log,"next_input_bf16_exact_expected")==0);
        assert(read(output+".rows.f32")==expected);
        ++cases;
        rejects([&] { Audit duplicate(2,100,10240,48,1,dtype); });
    }
    // CPU-only placement-label simulation: Copy always reads host memory.
    // This checks audit contracts, not GPU access, queue retirement or a model.
    // All environment switching remains in this CPU-only native fixture; there
    // is no runtime/provider initialization or runtime environment mutation.
    for (const char* mode_name:{"fullhost","fullgpu","mixedstore"}) {
        const std::string mode(mode_name);
        const auto output=path(".bf16-"+mode);
        std::string expected;
        {
            Audit audit(2,100,10240,48,1,Dtype::bf16);
            std::vector<float> r(10240);
            std::vector<uint16_t> packed(20480),literal_packed(10240);
            std::vector<uint32_t> literal_expanded(10240);
            for (int l=0;l<48;++l) for (int row=0;row<2;++row) {
                const bool device=mode=="fullgpu" || (mode=="mixedstore" && row==0);
                if (l) {
                    for (int i=0;i<10240;++i) {
                        r[i]=strata::kernels::f32_from_bf16(packed[row*10240+i]);
                        const auto& oracle=bf16_cases[(size_t(i)+size_t(l-1)+size_t(row))%bf16_case_count];
                        literal_expanded[size_t(i)]=uint32_t(oracle.packed)<<16;
                    }
                    assert(std::memcmp(r.data(),literal_expanded.data(),r.size()*4)==0);
                    assert(audit.input(l,row,1,r.data(),device,copy));
                }
                for (int i=0;i<10240;++i) {
                    const auto& oracle=bf16_cases[(size_t(i)+size_t(l)+size_t(row))%bf16_case_count];
                    std::memcpy(&r[i],&oracle.input,4);
                    literal_packed[size_t(i)]=oracle.packed;
                }
                expected.append(reinterpret_cast<const char*>(r.data()),r.size()*4);
                assert(audit.source(l,row,1,r.data(),copy));
                if(l!=47) {
                    for(int i=0;i<10240;++i)
                        packed[row*10240+i]=strata::kernels::bf16_from_f32(r[i]);
                    assert(std::memcmp(packed.data()+row*10240,literal_packed.data(),literal_packed.size()*2)==0);
                    assert(audit.stored(l,row,1,packed.data()+row*10240,device,copy));
                }
            }
            audit.complete();
        }
        const auto log=read(output);
        assert(log.find("boundary_audit_v2 ")!=std::string::npos);
        assert(log.find("storage=bf16 pack_oracle=bf16_from_f32_RNE_NaN_quiet_preserve_sign_payload unpack_oracle=f32_from_bf16_exact_shift16\n")!=std::string::npos);
        assert(log.find("status=complete boundary_memcmp=equal\n")!=std::string::npos);
        assert(log.find("memcmp=different")==std::string::npos);
        assert(phase_count(log,"output_before_store")==96);
        assert(phase_count(log,"stored_bf16_RNE_packed_bits")==94);
        assert(phase_count(log,"next_input_bf16_exact_expected")==94);
        assert(phase_count(log,"stored_RNE_packed_bits")==0);
        assert(phase_count(log,"next_input_decoded_expected")==0);
        for(int row=0;row<2;++row) {
            const bool device=mode=="fullgpu" || (mode=="mixedstore" && row==0);
            const char* placement=device?"GPU":"RAM";
            assert(placement_record(log,"stored_bf16_RNE_packed_bits",row,placement));
            assert(placement_record(log,"next_input_bf16_exact_expected",row,placement));
        }
        assert(read(output+".rows.f32")==expected); ++cases;
        rejects([&] { Audit duplicate(2,100,10240,48,1,Dtype::bf16); });
    }
    const auto invalid_dtype=path(".invalid-dtype");
    rejects([&] { Audit x(2,100,10240,48,1,static_cast<Dtype>(99)); });
    assert(!std::ifstream(invalid_dtype).good());
    assert(!std::ifstream(invalid_dtype+".rows.f32").good());
    // A BF16 packed-bit fault must remain a completed numerical mismatch.
    // Restore the corrupted store too, so both store and next-input gates see it.
    assert(::setenv("STRATA_PREFILL_BOUNDARY_ROWS","0",1)==0);
    const auto bf16_corrupt=path(".bf16-corrupt");
    {
        Audit audit(1,0,10240,48,1,Dtype::bf16);
        std::vector<float> row(10240); std::vector<uint16_t> packed(10240);
        for(int l=0;l<48;++l) {
            if(l) {
                for(int i=0;i<10240;++i) row[i]=strata::kernels::f32_from_bf16(packed[i]);
                assert(audit.input(l,0,1,row.data(),false,copy));
            }
            for(int i=0;i<10240;++i) {
                const auto& oracle=bf16_cases[(size_t(i)+size_t(l))%bf16_case_count];
                std::memcpy(&row[i],&oracle.input,4);
            }
            assert(audit.source(l,0,1,row.data(),copy));
            if(l!=47) {
                for(int i=0;i<10240;++i) packed[i]=strata::kernels::bf16_from_f32(row[i]);
                if(l==0) packed[2]^=1; // A finite value, so the fault is unambiguous.
                assert(audit.stored(l,0,1,packed.data(),false,copy));
            }
        }
        audit.complete();
    }
    const auto bf16_corrupt_log=read(bf16_corrupt);
    assert(bf16_corrupt_log.find("status=complete boundary_memcmp=different\n")!=std::string::npos);
    assert(phase_difference(bf16_corrupt_log,"stored_bf16_RNE_packed_bits"));
    assert(phase_difference(bf16_corrupt_log,"next_input_bf16_exact_expected")); ++cases;
    path(".reject");
    for (const auto* csv:{"","0,","0,0","-1","2"," 0","+0","0,,1","9223372036854775808"}) {
        ::setenv("STRATA_PREFILL_BOUNDARY_ROWS",csv,1);
        rejects([&] { Audit x(2,0,10240,48,1,Dtype::fp32); });
    }
    std::string excessive;
    for (int i=0;i<26;++i) { if(i) excessive+=','; excessive+=std::to_string(i); }
    ::setenv("STRATA_PREFILL_BOUNDARY_ROWS",excessive.c_str(),1);
    rejects([&] { Audit x(26,0,10240,48,1,Dtype::fp32); });
    ::setenv("STRATA_PREFILL_BOUNDARY_ROWS","0",1);
    for (const int64_t n:{int64_t(0),int64_t(262145),INT64_MAX})
        rejects([&] { Audit x(n,0,10240,48,1,Dtype::fp32); });
    for (const int64_t p:{int64_t(-1),int64_t(262144),INT64_MAX})
        rejects([&] { Audit x(1,p,10240,48,1,Dtype::fp32); });
    for (const int64_t chunk:{int64_t(0),int64_t(8193)})
        rejects([&] { Audit x(1,0,10240,48,chunk,Dtype::fp32); });
    rejects([&] { Audit x(1,0,10239,48,1,Dtype::fp32); });
    rejects([&] { Audit x(1,0,10240,47,1,Dtype::fp32); });
    ::unsetenv("STRATA_PREFILL_BOUNDARY_PATH");
    rejects([&] { Audit x(1,0,10240,48,1,Dtype::fp32); });
    path("/missing/directory");
    rejects([&] { Audit x(1,0,10240,48,1,Dtype::fp32); });
    ::setenv("STRATA_PREFILL_BOUNDARY_ROWS_ABSOLUTE","1",1);
    ::setenv("STRATA_PREFILL_BOUNDARY_ROWS","99",1);
    rejects([&] { Audit x(1,100,10240,48,1,Dtype::fp32); });
    ::setenv("STRATA_PREFILL_BOUNDARY_ROWS","262143",1);
    path(".absolute");
    { Audit x(1,262143,10240,48,1,Dtype::fp32); rejects([&] { x.complete(); }); }
    ::unsetenv("STRATA_PREFILL_BOUNDARY_ROWS_ABSOLUTE");
    ::setenv("STRATA_PREFILL_BOUNDARY_ROWS","0",1);
    path(".sequence");
    {
        Audit x(1,0,10240,48,1,Dtype::fp32); std::vector<float> row(10240);
        rejects([&] { x.input(1,0,1,row.data(),false,copy); });
        rejects([&] { x.source(48,0,1,row.data(),copy); });
        Audit::Copy fail=[](void*,const void*,size_t,bool) { return false; };
        assert(!x.source(0,0,1,row.data(),fail)); ++cases;
        rejects([&] { x.complete(); });
    }
    // A completed but corrupted store is reported as different, never coverage failure alone.
    const auto corrupted=path(".corrupt");
    {
        Audit x(1,0,10240,48,1,Dtype::fp32); std::vector<float> row(10240),store(10240);
        for (int l=0;l<48;++l) {
            if (l) assert(x.input(l,0,1,row.data(),false,copy));
            assert(x.source(l,0,1,row.data(),copy));
            if (l!=47) {
                store=row; if(l==0) store[31]=1;
                assert(x.stored(l,0,1,store.data(),false,copy));
            }
        }
        x.complete();
    }
    assert(read(corrupted).find("status=complete boundary_memcmp=different\n")!=std::string::npos); ++cases;
    std::printf("PASS boundary audit CPU contracts cases=%u raw FP32 bits exact; no GPU qualification\n",cases);
}
