// Independently authored for Strata; repository LICENSE applies.
// Host-only check of the actual cpu_wait_profile::blocking bridge.
// No SYCL runtime, device work, timing benchmark or model qualification.
#include "strata/sycl/cpu_wait_profile.hpp"
#include <string>

namespace p = strata::sycl_backend::decode_service_profile;
namespace w = strata::sycl_backend::cpu_wait_profile;
namespace {
uint64_t ticks = 0;
p::Tick fake_clock() noexcept { ++ticks; return {ticks * 1000, ticks * 100, true}; }
struct Output {
    std::FILE* file = std::tmpfile();
    ~Output() { if (file) std::fclose(file); }
    std::string text() {
        if (!file || std::fflush(file) != 0) return {};
        std::rewind(file);
        std::string result;
        char bytes[1024];
        for (size_t n; (n = std::fread(bytes, 1, sizeof(bytes), file));) result.append(bytes, n);
        return std::ferror(file) ? std::string{} : result;
    }
};
struct Legacy {
    w::State state;
    w::State* previous = w::active;
    explicit Legacy(bool enabled = true) { w::active = enabled ? &state : nullptr; }
    ~Legacy() { w::active = previous; }
};
bool has(const std::string& text, const char* token) {
    const auto pos = text.find(token);
    const auto end = pos + std::char_traits<char>::length(token);
    return pos != std::string::npos && (pos == 0 || text[pos - 1] == ' ') &&
        end < text.size() && (text[end] == ' ' || text[end] == '\n');
}
bool closed(const Legacy& legacy) {
    return !legacy.state.invalid && legacy.state.depth == 0 &&
        p::active == nullptr && w::active == &legacy.state;
}
}

int main() {
    if (p::active || w::active) return 1;
    int cases = 0;
    {
        Output out;
        if (!out.file) return 1;
        Legacy legacy;
        {
            p::Scope scope(true, 11, out.file, fake_clock);
            p::measure(p::layer_completion, [] {
                w::blocking(w::compute_queue, [] {
                    w::blocking(w::event, [] {});
                });
            });
            scope.finish(true);
        }
        const auto text = out.text();
        if (!closed(legacy) || legacy.state.count[w::compute_queue] != 1 ||
            legacy.state.count[w::event] != 0 || !has(text, "available=1") ||
            !has(text, "wall_ns=5000") || !has(text, "cpu_ns=500") ||
            !has(text, "layer_wait_wall_ns=3000") || !has(text, "layer_wait_cpu_ns=300") ||
            !has(text, "layer_wait_wait_calls=1") || !has(text, "layer_wait_wait_wall_ns=1000") ||
            !has(text, "layer_wait_wait_cpu_ns=100") ||
            !has(text, "unclassified_wall_ns=2000") || !has(text, "unclassified_cpu_ns=200")) return 1;
        ++cases;
    }
    {
        Output out;
        if (!out.file) return 1;
        Legacy legacy;
        bool caught = false;
        {
            p::Scope scope(true, 12, out.file, fake_clock);
            try {
                p::measure(p::expert_retirement, [] {
                    w::blocking(w::transfer_queue, [] { throw 37; });
                });
            } catch (int value) { caught = value == 37; }
            scope.finish(true);
        }
        const auto text = out.text();
        if (!caught || !closed(legacy) || legacy.state.count[w::transfer_queue] != 1 ||
            legacy.state.throws[w::transfer_queue] != 1 || !has(text, "available=1") ||
            !has(text, "copy_retire_throws=1") || !has(text, "copy_retire_wait_throws=1")) return 1;
        ++cases;
    }
    {
        Output out;
        if (!out.file) return 1;
        Legacy legacy;
        const auto before = ticks;
        int calls = 0;
        {
            p::Scope off(false, 13, out.file, fake_clock);
            w::blocking(w::direct_queue, [&] { ++calls; errno = EIO; });
        }
        if (errno != EIO || !closed(legacy) || ticks != before || calls != 1 ||
            legacy.state.count[w::direct_queue] != 1 || !out.text().empty()) return 1;
        ++cases;
    }
    {
        Output out;
        if (!out.file) return 1;
        Legacy legacy(false);
        const auto before = ticks;
        int calls = 0;
        { p::Scope off(false, 14, out.file, fake_clock);
          w::blocking(w::compat_queue, [&] { ++calls; }); }
        if (p::active || w::active || ticks != before || calls != 1 || !out.text().empty()) return 1;
        ++cases;
    }
    {
        Output out;
        if (!out.file) return 1;
        Legacy legacy;
        { p::Scope scope(true, 15, out.file, fake_clock);
          w::blocking(w::event, [] {});
          scope.finish(true); }
        const auto text = out.text();
        if (!closed(legacy) || legacy.state.count[w::event] != 1 || !has(text, "available=1") ||
            !has(text, "wall_ns=3000") || !has(text, "unclassified_wall_ns=3000") ||
            !has(text, "unclassified_wait_calls=1") ||
            !has(text, "unclassified_wait_wall_ns=1000")) return 1;
        ++cases;
    }
    if (p::active || w::active) return 1;
    std::printf("decode wait bridge: %d host-only cases passed\n", cases);
    return 0;
}
