// Main-authored lifetime/status fault injection; repository LICENSE applies.
#include "strata/prefill/selection_dump.hpp"
#include <cstring>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const std::string mode = argv[1];
    const int32_t source[] = {1, 2, 3, 4};
    int waits = 0, copies = 0;
    std::string error;
    auto copy = [&](void* dst, const void* src, size_t bytes) {
        ++copies;
        if (mode == "throw") throw std::runtime_error("injected partial submission");
        if (mode == "submit-fail") return false;
        std::memcpy(dst, src, bytes); return true;
    };
    auto drain = [&] { ++waits; return mode != "drain-fail"; };
    auto fatal = [] { std::_Exit(73); };
    const bool ok = strata::prefill::detail::selection_dump(argv[2], 0, 32,
        mode == "geometry" ? -1 : 2, 2, source, copy, drain, fatal, error);
    if (mode == "success") return ok && copies == 1 && waits == 1 && error.empty() ? 0 : 1;
    if (mode == "geometry") return !ok && copies == 0 && waits == 0 ? 0 : 1;
    if (mode == "submit-fail" || mode == "throw") return !ok && copies == 1 && waits == 1 ? 0 : 1;
    if (mode == "open-fail" || mode == "file-fail") return !ok && copies == 1 && waits == 1 ? 0 : 1;
    return 1; // drain-fail must terminate inside fatal(), before unwinding host.
}
