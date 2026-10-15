// Main compiles/runs serially after applying source.diff to an isolated child.
// No driver, SYCL library, provider, device queue or tensor allocation.
#include "strata/core/external_decode_profile.hpp"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    using namespace strata::core::external_profile;
    const char* mode=argc>1 ? argv[1] : "ok";
    int verifier=0, compute=0, copy=0;
    std::string err;
    request(true);
    for (int attempt_number=1; attempt_number<=8; ++attempt_number) {
        attempt(&verifier);
        const int T=attempt_number==1 ||
            (attempt_number==8 && std::strcmp(mode,"wrongT")==0) ? 1 : 4;
        const bool warm=!(attempt_number==8 && std::strcmp(mode,"cold")==0);
        // Actual verifier ordinal deliberately differs from decode ordinal 8.
        Window w(&verifier,T,34603+attempt_number,34+attempt_number,
                 &compute,&copy,48,warm,true);
        if (!w.before_launch(err)) { std::fprintf(stderr,"%s\n",err.c_str()); return 10; }
        if (attempt_number==8 && std::strcmp(mode,"failure_after_start")==0) return 11;
        // The fixture has no device work: this call stands for BOTH existing
        // successful device stream drains; it cannot certify actual coverage.
        if (!w.drained(err)) { std::fprintf(stderr,"%s\n",err.c_str()); return 12; }
        if (attempt_number==8 && std::strcmp(mode,"failure_after_drained")==0) return 13;
        if (!w.finish(err)) { std::fprintf(stderr,"%s\n",err.c_str()); return 14; }
    }
    // Neither later windows nor a second request may re-arm the diagnostic.
    request(true);
    for (int i=0;i<8;++i) attempt(&verifier);
    Window extra(&verifier,4,35000,99,&compute,&copy,48,true,true);
    if (!extra.before_launch(err) || !extra.drained(err) || !extra.finish(err)) return 15;
    std::puts("FIXTURE_DONE");
    return 0;
}
