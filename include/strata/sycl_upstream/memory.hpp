#pragma once
#include "strata/sycl_upstream/runtime.hpp"
#include <optional>

namespace strata::sycl_upstream {
// Owns allocations and registration metadata in one Runtime context. Runtime
// must outlive Memory. Registered storage remains owned by the caller, who must
// keep each backing arena mapped until all its registrations are released:
// adjacent logical ranges can share a coarser native mapping. Release waits
// for prior Runtime work; all users must submit through that Runtime. Callers
// must not submit new work using a pointer concurrently with its release.
// External imports require a driver that releases external mappings correctly;
// Intel builds before 39758 are rejected before registering caller storage.
class Memory {
public:
    enum class Kind { device, host, registered_host };
    struct Info { void* base; size_t bytes; Kind kind; bool read_only; };
    struct Stats { size_t device_bytes=0, host_bytes=0, registered_bytes=0, imported_bytes=0, imported_ranges=0; };
    explicit Memory(Runtime&);
    ~Memory();
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    void* allocate_device(size_t bytes);
    void* allocate_host(size_t bytes);
    void free_device(void*);
    void free_host(void*);
    void register_host(void*, size_t bytes, bool read_only = false);
    void unregister_host(void* original_pointer);
    void* device_alias(void* host_pointer) const;
    std::optional<Info> info(const void*) const;
    bool overlaps(const void*, size_t bytes) const;
    Stats stats() const;
    struct Available { size_t free, total; };
    // Physical free memory from the device driver, and SYCL global memory
    // capacity. No ledger subtraction or presumed allocation success.
    Available available() const;
    size_t page_size() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::sycl_upstream
