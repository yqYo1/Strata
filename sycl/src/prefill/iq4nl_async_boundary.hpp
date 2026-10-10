// Private opt-in boundary: caller has successfully completed the existing wait.
#pragma once
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
namespace strata::prefill::detail {
[[noreturn]] inline void iq4nl_unknown_completion(const char* stage) {
    std::fprintf(stderr, "FAIL_STOP,IQ4NL,%s,completion_unknown,no_USM_unwind\n", stage);
    std::fflush(stderr);
    std::_Exit(2);
}
inline bool iq4nl_completed_boundary(sycl::queue& q, std::string& err) {
    try { q.throw_asynchronous(); return true; }
    catch (const std::exception& e) {
        err = std::string("prefill: IQ4NL completed compute queue async error: ") + e.what();
    } catch (...) { err = "prefill: IQ4NL completed compute queue async error: non-standard exception"; }
    return false;
}
// Used where completion cannot be inferred. Never unwind live USM on wait failure.
inline bool iq4nl_wait_boundary(sycl::queue& q, std::string& err) {
    try { q.wait(); } catch (...) { iq4nl_unknown_completion("wait"); }
    return iq4nl_completed_boundary(q, err);
}
}
