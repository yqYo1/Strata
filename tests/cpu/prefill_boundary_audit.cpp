#include "strata/prefill/layer_major_boundary_audit.hpp"
#include <cassert>
#include <fstream>
#include <sstream>

using Audit = strata::prefill::detail::LayerMajorBoundaryAudit;
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
        const auto output=path(half?".half":".float");
        std::string expected;
        {
            Audit audit(2,100,10240,48,1,half);
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
        assert(read(output+".rows.f32")==expected);
        ++cases;
        rejects([&] { Audit duplicate(2,100,10240,48,1,half); });
    }
    path(".reject");
    for (const auto* csv:{"","0,","0,0","-1","2"," 0","+0","0,,1","9223372036854775808"}) {
        ::setenv("STRATA_PREFILL_BOUNDARY_ROWS",csv,1);
        rejects([&] { Audit x(2,0,10240,48,1,false); });
    }
    std::string excessive;
    for (int i=0;i<26;++i) { if(i) excessive+=','; excessive+=std::to_string(i); }
    ::setenv("STRATA_PREFILL_BOUNDARY_ROWS",excessive.c_str(),1);
    rejects([&] { Audit x(26,0,10240,48,1,false); });
    ::setenv("STRATA_PREFILL_BOUNDARY_ROWS","0",1);
    for (const int64_t n:{int64_t(0),int64_t(262145),INT64_MAX})
        rejects([&] { Audit x(n,0,10240,48,1,false); });
    for (const int64_t p:{int64_t(-1),int64_t(262144),INT64_MAX})
        rejects([&] { Audit x(1,p,10240,48,1,false); });
    for (const int64_t chunk:{int64_t(0),int64_t(8193)})
        rejects([&] { Audit x(1,0,10240,48,chunk,false); });
    rejects([&] { Audit x(1,0,10239,48,1,false); });
    rejects([&] { Audit x(1,0,10240,47,1,false); });
    ::unsetenv("STRATA_PREFILL_BOUNDARY_PATH");
    rejects([&] { Audit x(1,0,10240,48,1,false); });
    path("/missing/directory");
    rejects([&] { Audit x(1,0,10240,48,1,false); });
    ::setenv("STRATA_PREFILL_BOUNDARY_ROWS_ABSOLUTE","1",1);
    ::setenv("STRATA_PREFILL_BOUNDARY_ROWS","99",1);
    rejects([&] { Audit x(1,100,10240,48,1,false); });
    ::setenv("STRATA_PREFILL_BOUNDARY_ROWS","262143",1);
    path(".absolute");
    { Audit x(1,262143,10240,48,1,false); rejects([&] { x.complete(); }); }
    ::unsetenv("STRATA_PREFILL_BOUNDARY_ROWS_ABSOLUTE");
    ::setenv("STRATA_PREFILL_BOUNDARY_ROWS","0",1);
    path(".sequence");
    {
        Audit x(1,0,10240,48,1,false); std::vector<float> row(10240);
        rejects([&] { x.input(1,0,1,row.data(),false,copy); });
        rejects([&] { x.source(48,0,1,row.data(),copy); });
        Audit::Copy fail=[](void*,const void*,size_t,bool) { return false; };
        assert(!x.source(0,0,1,row.data(),fail)); ++cases;
        rejects([&] { x.complete(); });
    }
    // A completed but corrupted store is reported as different, never coverage failure alone.
    const auto corrupted=path(".corrupt");
    {
        Audit x(1,0,10240,48,1,false); std::vector<float> row(10240),store(10240);
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
