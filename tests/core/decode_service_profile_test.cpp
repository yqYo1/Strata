// Independently authored for Strata; repository LICENSE applies.
// Host-only accounting contract. Compile directly with -std=c++17 -Iinclude.
#include "strata/sycl/decode_service_profile.hpp"
#include <stdexcept>
#include <string>

namespace p = strata::sycl_backend::decode_service_profile;
namespace {
uint64_t ticks = 0;
p::Tick fake_clock() noexcept { ++ticks; return {ticks * 1000, ticks * 100, true}; }
p::Tick failed_clock() noexcept { return {}; }
std::string contents(std::FILE* file) {
    std::fflush(file);
    std::rewind(file);
    std::string result;
    char bytes[1024];
    for (size_t n; (n = std::fread(bytes, 1, sizeof(bytes), file));) result.append(bytes, n);
    return result;
}
bool has(const std::string& text, const char* token) {
    return text.find(std::string(" ") + token + " ") != std::string::npos;
}
}
int main() {
    std::FILE* file = std::tmpfile();
    if (!file) return 1;
    int cases = 0;
    auto check = [&](bool ok, const char* name) {
        if (!ok) std::fprintf(stderr, "decode service contract failed: %s\n", name);
        return ok;
    };
    {
        p::Scope scope(true, 7, file, fake_clock);
        const int result = p::measure(p::layer_completion, [] {
            p::NativeWait outer;
            p::NativeWait nested; // A nested adapter wait is counted once.
            return 42;
        });
        { p::NativeWait unclassified; }
        scope.finish(true);
        scope.finish(false); // Idempotent; no second output.
        const auto text = contents(file);
        if (!check(result == 42 && p::active == nullptr &&
            has(text, "request=7") && has(text, "available=1") && has(text, "complete=1") &&
            has(text, "wall_ns=7000") && has(text, "cpu_ns=700") &&
            has(text, "unclassified_wall_ns=4000") && has(text, "unclassified_cpu_ns=400") &&
            has(text, "layer_wait_calls=1") && has(text, "layer_wait_wall_ns=3000") &&
            has(text, "layer_wait_wait_calls=1") && has(text, "layer_wait_wait_wall_ns=1000") &&
            has(text, "unclassified_wait_calls=1") &&
            text.find('\n') == text.size() - 1, "exact partition and nested-wait suppression")) return 1;
        ++cases;
    }
    std::fclose(file);
    auto fixture = [&](auto body, const char* required, p::Clock clock = fake_clock) {
        std::FILE* out = std::tmpfile();
        if (!out) return false;
        { p::Scope scope(true, 8, out, clock); body(scope); }
        const auto text = contents(out);
        const bool ok = has(text, required) && p::active == nullptr;
        std::fclose(out);
        return ok;
    };
    if (!check(fixture([](auto& s) {
            { p::Part outer(p::cpu_pool); p::Part nested(p::staging); }
            s.finish(true);
        }, "available=0"), "overlapping caller regions rejected")) return 1;
    ++cases;
    if (!check(fixture([](auto& s) {
            { p::Scope nested(true, 9); }
            s.finish(true);
        }, "available=0"), "nested requests rejected")) return 1;
    ++cases;
    if (!check(fixture([](auto& s) { s.finish(true); }, "available=0", failed_clock),
            "unavailable clocks rejected")) return 1;
    ++cases;
    if (!check(fixture([](auto&) {
            try { p::measure(p::cpu_pool, [] { throw std::runtime_error("fixture"); }); }
            catch (const std::runtime_error&) {}
        }, "pool_throws=1"), "exceptions counted and TLS retired")) return 1;
    ++cases;
    if (!check(fixture([](auto&) {}, "complete=0"), "unfinished requests stay incomplete")) return 1;
    ++cases;
    const auto before = ticks;
    file = std::tmpfile();
    if (!file) return 1;
    int value = 3;
    { p::Scope off(false, 10, file, fake_clock);
      p::measure(p::staging, [&]() -> int& { p::NativeWait wait; return value; }) = 4; }
    if (!check(ticks == before && value == 4 && contents(file).empty() && p::active == nullptr,
            "disabled observation reads no clock and preserves reference return")) return 1;
    std::fclose(file);
    ++cases;
    std::printf("decode service profile: %d host-only accounting cases passed\n", cases);
    return 0;
}
