// New project fixture, repository LICENSE. Main alone builds/runs it.
#include "strata/core/allocated_fields.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <set>
#include <stdexcept>

int main() {
    const std::array<size_t, 5> sizes{16, 8, 8, 4, 4};
    // Fail each field both by null return and by synchronous exception. The
    // fixture allocator deliberately dirties memory, making zeroing observable.
    for (int throwing = 0; throwing != 2; ++throwing) {
        for (int fail = 0; fail <= 5; ++fail) {
            int issued = 0, freed = 0;
            std::set<void*> live;
            auto fields = strata::core::detail::allocate_zeroed_fields(sizes,
                [&](size_t bytes) -> void* {
                    if (issued++ == fail) {
                        if (throwing) throw std::runtime_error("injected allocation rejection");
                        return nullptr;
                    }
                    void* p = std::malloc(bytes);
                    assert(p); std::memset(p, 0xa5, bytes); live.insert(p); return p;
                },
                [&](void* p) { assert(live.erase(p) == 1); ++freed; std::free(p); });
            if (fail < 5) {
                assert(live.empty() && freed == fail);
                for (void* p : fields) assert(!p);
            } else {
                assert(live.size() == 5 && freed == 0);
                for (size_t i = 0; i < fields.size(); ++i) {
                    for (size_t b = 0; b < sizes[i]; ++b) assert(((unsigned char*) fields[i])[b] == 0);
                    assert(live.erase(fields[i]) == 1); std::free(fields[i]);
                }
            }
        }
    }
}
