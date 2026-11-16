#pragma once
// Main-authored ordinary-heap fixture owner; repository LICENSE.
#include "strata/sycl/checked_cleanup.hpp"
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace strata::prefill::test_detail {
class PrefillHostPages {
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
    bool read_only_ = false;
public:
    explicit PrefillHostPages(std::size_t bytes) {
        const long page = ::sysconf(_SC_PAGESIZE);
        if (page <= 0 || !bytes || bytes > (4u << 20) || bytes >
            std::numeric_limits<std::size_t>::max() - (std::size_t(page) - 1))
            throw std::invalid_argument("prefill fixture page allocation bounds");
        bytes_ = ((bytes + std::size_t(page) - 1) / std::size_t(page)) * std::size_t(page);
        if (::posix_memalign(&data_, std::size_t(page), bytes_) != 0 || !data_)
            throw std::bad_alloc();
    }
    PrefillHostPages(const PrefillHostPages&) = delete;
    PrefillHostPages& operator=(const PrefillHostPages&) = delete;
    ~PrefillHostPages() noexcept {
        if (read_only_ && ::mprotect(data_, bytes_, PROT_READ | PROT_WRITE) != 0)
            sycl_backend::cleanup_fail_stop("prefill fixture restoring heap page protection");
        std::free(data_);
    }
    void read_only() {
        if (read_only_ || ::mprotect(data_, bytes_, PROT_READ) != 0)
            throw std::runtime_error("prefill fixture read-only page protection");
        read_only_ = true;
    }
    void* data() const { return data_; }
    std::size_t bytes() const { return bytes_; }
};
} // namespace strata::prefill::test_detail
