// Independently authored for Strata; repository LICENSE applies.
// Real schema2 producer for the Python reader contract;no GPU or timing test.
#include "strata/sycl/decode_service_profile.hpp"

namespace p = strata::sycl_backend::decode_service_profile;
namespace {
uint64_t ticks = 0;
p::Tick sample_clock() noexcept { ++ticks; return {ticks * 1000, ticks * 100, true}; }
}

int main() {
    {
        p::Scope scope(true, 1, stdout, sample_clock);
        for (int region = 0; region < p::region_count; ++region)
            p::measure(static_cast<p::Region>(region), [] {
                p::NativeWait outer;
                p::NativeWait nested;
            });
        { p::NativeWait unclassified; }
        scope.finish(true);
    }
    const auto before = ticks;
    {
        p::Scope scope(false, 2, stdout, sample_clock);
        p::measure(p::staging, [] { p::NativeWait wait; });
    }
    if (ticks != before || p::active) return 1;
    { p::Scope scope(true, 3, stdout, sample_clock); scope.finish(true); }
    return p::active || std::ferror(stdout) ? 1 : 0;
}
