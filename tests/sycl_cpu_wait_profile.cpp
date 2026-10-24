// Authored for Strata; repository LICENSE (MIT) applies.
// CPU-only gates for observer accounting; no SYCL runtime or GPU work.
#include "strata/sycl/cpu_wait_profile.hpp"
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace hp = strata::sycl_backend::cpu_wait_profile;
static_assert(!std::is_copy_constructible_v<hp::Host>);
static_assert(!std::is_move_constructible_v<hp::Host>);
static_assert(std::is_nothrow_destructible_v<hp::Host>);
static_assert(std::is_nothrow_destructible_v<hp::Wait>);
static_assert(!std::is_copy_constructible_v<hp::StatusScope>);
static_assert(!std::is_copy_constructible_v<hp::StatusPart>);
static_assert(std::is_nothrow_destructible_v<hp::StatusPart>);

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void work() {
    volatile unsigned value = 1;
    for (unsigned i = 0; i < 100000; ++i) value = value * 1664525u + 1013904223u;
}
void begin(const char* name) {
    std::fprintf(stderr, "cpu fixture case: %s\n", name);
}
int main() try {
    begin("disabled");
    {
        hp::Scope scope(false);
        hp::Host host(hp::moe_gather);
        hp::Wait wait(hp::compat_queue);
        hp::StatusScope status;
        hp::StatusPart part(hp::status_precheck);
        require(hp::active == nullptr, "disabled observer changed TLS");
    }
    begin("valid");
    {
        hp::Scope scope(true);
        require(hp::active != nullptr, "missing active observer");
        for (int k = 0; k < hp::host_kinds; ++k) {
            hp::Host host(static_cast<hp::HostKind>(k));
            work();
            if (k == hp::moe_status) {
                hp::StatusScope status;
                { hp::StatusPart part(hp::status_precheck); work(); }
                {
                    hp::StatusPart part(hp::status_drain);
                    hp::Wait wait(hp::compat_queue);
                    hp::Wait nested(hp::direct_queue);
                    work();
                }
                status.complete();
            }
            host.finish();
            host.finish();
        }
        // A child thread must not attribute its work to the caller's scope.
        std::thread child([] {
            require(hp::active == nullptr, "observer TLS leaked across threads");
            hp::Host host(hp::moe_gather);
            work();
        });
        child.join();
        const auto& s = *hp::active;
        require(!s.invalid && !s.host_depth && !s.depth, "valid accounting rejected");
        for (int k = 0; k < hp::host_kinds; ++k)
            require(s.host_count[k] == 1 && !s.host_throws[k] &&
                    hp::duration(s.host_cpu[k] - s.host_wait_cpu[k]) &&
                    hp::duration(s.host_wall[k] - s.host_wait_wall[k]), "host partition mismatch");
        require(s.count[hp::compat_queue] == 1 && !s.count[hp::direct_queue] &&
                s.host_wait_cpu[hp::moe_status] == s.cpu[hp::compat_queue] &&
                s.host_wait_wall[hp::moe_status] == s.wall[hp::compat_queue], "wait subtraction mismatch");
        require(s.status_scopes == 1 && s.status_completed == 1 &&
            s.status_calls[hp::status_precheck] == 1 && s.status_calls[hp::status_drain] == 1 &&
            s.status_wait_cpu[hp::status_drain] == s.cpu[hp::compat_queue] &&
            s.status_wait_wall[hp::status_drain] == s.wall[hp::compat_queue] &&
            hp::duration(s.status_scope_cpu - s.status_cpu[hp::status_precheck] - s.status_cpu[hp::status_drain]) &&
            !s.status_depth && !s.status_part_depth, "status accounting mismatch");
        scope.complete(true);
    }
    require(hp::active == nullptr, "scope did not restore TLS");
    begin("exception");
    {
        hp::Scope scope(true);
        try {
            hp::Host host(hp::moe_gather);
            hp::Wait wait(hp::compat_queue);
            work();
            throw 7;
        } catch (int) {}
        require(hp::active->host_throws[hp::moe_gather] == 1 &&
                hp::active->throws[hp::compat_queue] == 1 &&
                !hp::active->host_depth && !hp::active->depth, "exception ownership mismatch");
        scope.complete(false);
    }
    begin("nested_host");
    {
        hp::Scope scope(true);
        { hp::Host outer(hp::source_copy); hp::Host inner(hp::gemm_submit); work(); }
        require(hp::active->invalid && !hp::active->host_depth, "nested host was accepted");
    }
    begin("host_inside_wait");
    {
        hp::Scope scope(true);
        { hp::Wait wait(hp::compat_queue); hp::Host host(hp::moe_status); work(); }
        require(hp::active->invalid && !hp::active->host_depth && !hp::active->depth,
                "host inside wait was accepted");
    }
    begin("nested_scope");
    {
        hp::Scope outer(true);
        auto* original = hp::active;
        { hp::Scope inner(true); require(hp::active == original, "nested scope replaced TLS"); }
        require(original->invalid, "nested scope was accepted");
    }
    require(hp::active == nullptr, "final TLS not restored");
    begin("status_nested_scope");
    {
        hp::Scope scope(true);
        { hp::StatusScope outer; hp::StatusScope inner; work(); }
        require(hp::active->invalid && !hp::active->status_depth, "nested status scope accepted");
    }
    begin("status_nested_part");
    {
        hp::Scope scope(true);
        {
            hp::StatusScope status;
            hp::StatusPart outer(hp::status_precheck), inner(hp::status_queue);
            work();
        }
        require(hp::active->invalid && !hp::active->status_part_depth, "nested status part accepted");
    }
    begin("status_part_inside_wait");
    {
        hp::Scope scope(true);
        {
            hp::StatusScope status;
            hp::Wait wait(hp::compat_queue);
            hp::StatusPart part(hp::status_drain);
            work();
        }
        require(hp::active->invalid && !hp::active->status_depth && !hp::active->status_part_depth,
            "status part inside wait accepted");
    }
    begin("status_exception");
    {
        hp::Scope scope(true);
        try {
            hp::StatusScope status;
            hp::StatusPart part(hp::status_precheck);
            work(); throw 7;
        } catch (int) {}
        require(!hp::active->invalid && hp::active->status_scopes == 1 && !hp::active->status_completed &&
            hp::active->status_throws[hp::status_precheck] == 1 &&
            !hp::active->status_depth && !hp::active->status_part_depth, "status exception ownership mismatch");
        scope.complete(false);
    }
    std::puts("fixture_status=PASS");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "fixture_status=FAIL error=%s\n", error.what());
    return 1;
}
