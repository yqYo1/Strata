// CPU-only, no GPU calls. Proposed fixture; not built or run by its author.
// Compile after integration with: c++ -std=c++17 -Iinclude tools/test_mtp_expert_source.cpp -o /tmp/mtp-expert-sha
#include "strata/core/mtp_expert_source.hpp"
#include <cstdio>
#include <string>

using strata::core::detail::MtpExpertSha256;
static std::string hex(const std::array<uint8_t, 32>& d) {
    static constexpr char chars[] = "0123456789abcdef";
    std::string s;
    for (auto c : d) { s += chars[c >> 4]; s += chars[c & 15]; }
    return s;
}
int main() {
    // SHA256 empty/abc/448-bit and million-'a' standard known-answer vectors.
    struct Case { std::string input; const char* expected; };
    const Case cases[] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"},
    };
    for (const auto& c : cases) for (size_t chunk : {size_t(1), size_t(55), size_t(56), size_t(63), size_t(64), size_t(65), size_t(8192)}) {
        MtpExpertSha256 h;
        for (size_t off = 0; off < c.input.size();) {
            const auto n = std::min(chunk, c.input.size() - off);
            h.update(reinterpret_cast<const uint8_t*>(c.input.data() + off), n);
            off += n;
        }
        if (hex(h.finish()) != c.expected || hex(h.finish()) != c.expected) {
            std::fprintf(stderr, "SHA256 KAT failed: bytes=%zu chunk=%zu\n", c.input.size(), chunk); return 1;
        }
        // A finalized snapshot must not mutate the source for a second hash.
        const uint8_t extra = 0;
        h.update(&extra, 1);
        if (hex(h.finish()) == c.expected) return 2;
    }
    std::puts("PASS MTP expert source SHA256 known-answer streaming vectors");
    return 0;
}
