#pragma once
#include <exception>
#include <iostream>

namespace strata {
// A printing-only async handler consumes errors and lets wait_and_throw return
// successfully. Report every error, then propagate the first to the caller.
template <class Errors> void rethrow_sycl_errors(const Errors& errors) {
    std::exception_ptr first;
    for (const auto& error : errors) {
        if (!error) continue;
        if (!first) first = error;
        try { std::rethrow_exception(error); }
        catch (const std::exception& e) { std::cerr << "asynchronous SYCL error: " << e.what() << '\n'; }
        catch (...) { std::cerr << "asynchronous SYCL error: unknown exception\n"; }
    }
    if (first) std::rethrow_exception(first);
}

template <class Finish> int finish_sycl_serve(Finish finish) {
    try {
        finish();
        std::cerr << "strata SYCL shutdown: all queues completed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "strata SYCL shutdown failed: " << e.what() << '\n';
    } catch (...) {
        std::cerr << "strata SYCL shutdown failed: unknown exception\n";
    }
    return 1;
}
} // namespace strata
