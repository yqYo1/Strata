#define DPCT_COMPAT_RT_VERSION 12080
// src/core/expert_cache.cpp - R4's slot storage and residency table.  Read the
// header first.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include "strata/sycl_allocation.hpp"
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/expert_cache.hpp"

#if !defined(STRATA_USE_HIP)
   // #533: the virtual memory management types (the functions come through the
   // runtime's entry points)
#endif

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <utility>
#include <memory>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#include <cstring>

namespace strata::core {

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_expert_profile: cannot open " + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " is too short to hold a header";
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " does not start with STRP";
        return false;
    }
    const uint32_t version = hdr[0], nl = hdr[1], ne = hdr[2], want = hdr[3], n_ranked = hdr[4];
    if ((int64_t) nl != n_layers || (int64_t) ne != n_expert) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is %ux%u but this model is %lldx%lld - it is a profile for a "
                      "different artifact", path.c_str(), nl, ne, (long long) n_layers, (long long) n_expert);
        err = buf;
        return false;
    }
    if (n_ranked > want) {
        std::fclose(f);
        err = "read_expert_profile: the header claims more ranked pairs than slots";
        return false;
    }
    ranked.assign(n_ranked, {0, 0});
    std::vector<uint16_t> raw((size_t) n_ranked * 2);
    if (n_ranked > 0 && std::fread(raw.data(), 2, (size_t) n_ranked * 2, f) != (size_t) n_ranked * 2) {
        std::fclose(f);
        err = "read_expert_profile: the ranked list is truncated";
        return false;
    }
    std::fclose(f);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = (int32_t) raw[(size_t) i * 2], e = (int32_t) raw[(size_t) i * 2 + 1];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "read_expert_profile: pair %u is (layer %d, expert %d), out of range",
                          i, l, e);
            err = buf;
            return false;
        }
        ranked[(size_t) i] = {l, e};
    }
    slots = (int64_t) want;
    (void) version;   // a future format bumps it; the layout check above is what protects this reader today
    return true;
}

std::vector<std::pair<int32_t, int32_t>> rank_learned_profile(int64_t n_layers, int64_t n_expert,
                                                              const std::vector<uint8_t>& resident,
                                                              const std::vector<double>& heat,
                                                              const std::vector<std::pair<int32_t, int32_t>>& prior) {
    const size_t n = (size_t) (n_layers * n_expert);
    std::vector<int64_t> prior_rank(n, INT64_MAX);
    for (size_t r = 0; r < prior.size(); ++r) {
        const auto [l, e] = prior[r];
        if (l >= 0 && l < n_layers && e >= 0 && e < n_expert) {
            int64_t& pr = prior_rank[(size_t) (l * n_expert + e)];
            if (pr == INT64_MAX) pr = (int64_t) r;
        }
    }
    std::vector<int64_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = (int64_t) i;
    auto res = [&](int64_t i) { return (size_t) i < resident.size() && resident[(size_t) i] != 0; };
    auto ht = [&](int64_t i) { return (size_t) i < heat.size() ? heat[(size_t) i] : 0.0; };
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        if (res(a) != res(b)) return res(a);
        if (ht(a) != ht(b)) return ht(a) > ht(b);
        if (prior_rank[(size_t) a] != prior_rank[(size_t) b]) return prior_rank[(size_t) a] < prior_rank[(size_t) b];
        return a < b;
    });
    std::vector<std::pair<int32_t, int32_t>> ranked(n);
    for (size_t r = 0; r < n; ++r)
        ranked[r] = {(int32_t) (order[r] / n_expert), (int32_t) (order[r] % n_expert)};
    return ranked;
}

bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                          const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    if (n_layers <= 0 || n_expert <= 0 || n_layers > 65535 || n_expert > 65535) {
        err = "write_expert_profile: the model's layout does not fit the format";
        return false;
    }
    std::vector<int32_t> table((size_t) (n_layers * n_expert), -1);
    std::vector<uint16_t> pairs;
    pairs.reserve(ranked.size() * 2);
    for (size_t r = 0; r < ranked.size(); ++r) {
        const auto [l, e] = ranked[r];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            err = "write_expert_profile: a ranked pair is out of range";
            return false;
        }
        table[(size_t) (l * n_expert + e)] = (int32_t) r;
        pairs.push_back((uint16_t) l);
        pairs.push_back((uint16_t) e);
    }
    const uint32_t hdr[5] = {1u, (uint32_t) n_layers, (uint32_t) n_expert, (uint32_t) ranked.size(),
                             (uint32_t) ranked.size()};
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = "write_expert_profile: cannot create " + tmp;
        return false;
    }
    // the format is little-endian (make_profile.py's "<"): so is every machine this engine runs on
    bool ok = std::fwrite("STRP", 1, 4, f) == 4 && std::fwrite(hdr, 4, 5, f) == 5 &&
              (pairs.empty() || std::fwrite(pairs.data(), 2, pairs.size(), f) == pairs.size()) &&
              std::fwrite(table.data(), 4, table.size(), f) == table.size();
    ok = (std::fclose(f) == 0) && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);   // replaces an existing file (MoveFileEx / rename(2))
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        err = "write_expert_profile: cannot write " + path;
        return false;
    }
    return true;
}

ExpertCache::~ExpertCache() {
    try { close(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "ExpertCache: cleanup failed: %s\n", e.what());
    }
    catch (...) { std::fprintf(stderr, "ExpertCache: cleanup failed: unknown exception\n"); }
}

// ---- #533: the segmented arena (--vram-elastic).  The driver API's virtual memory functions, looked up through the
// runtime (no link against the driver library): one address range for the whole arena, backed by physical segments,
// and the tail's segments unmapped / mapped again later.  Nothing here runs unless a segment size was set.
#if 0   // SYCL port: no driver virtual memory management (--vram-elastic is CUDA-only, #533)
namespace {
struct Vmm {
    int(CUDAAPI *device_get)(int *, int) = nullptr;
    int(CUDAAPI *attribute)(int *, CUdevice_attribute, int) = nullptr;
    int(CUDAAPI *granularity)(
        size_t *, const dpct::experimental::mem_prop *,
        sycl::ext::oneapi::experimental::granularity_mode) = nullptr;
    int(CUDAAPI *reserve)(dpct::device_ptr *, size_t, size_t, dpct::device_ptr,
                          unsigned long long) = nullptr;
    int(CUDAAPI *address_free)(dpct::device_ptr, size_t) = nullptr;
    int(CUDAAPI *create)(dpct::experimental::physical_mem_ptr *, size_t,
                         const dpct::experimental::mem_prop *,
                         unsigned long long) = nullptr;
    int(CUDAAPI *release)(dpct::experimental::physical_mem_ptr) = nullptr;
    int(CUDAAPI *map)(dpct::device_ptr, size_t, size_t,
                      dpct::experimental::physical_mem_ptr,
                      unsigned long long) = nullptr;
    int(CUDAAPI *unmap)(dpct::device_ptr, size_t) = nullptr;
    int(CUDAAPI *set_access)(dpct::device_ptr, size_t,
                             const dpct::experimental::mem_access_desc *,
                             size_t) = nullptr;
    bool ok = false;
};

template <class F> bool entry(const char *name, F &f) try {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
#if DPCT_COMPAT_RT_VERSION >= 12050
    /*
    DPCT1007: Migration of cudaGetDriverEntryPointByVersion is not
    supported.
    */
    const dpct::err0 e = cudaGetDriverEntryPointByVersion(
        name, &p, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != 0 || q != cudaDriverEntryPointSuccess || p == nullptr) {
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        (void)0;
        return false;
    }
    f = reinterpret_cast<F>(p);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

const Vmm& vmm() {
    static const Vmm v = [] {
        Vmm x;
        x.ok = entry("cuDeviceGet", x.device_get) && entry("cuDeviceGetAttribute", x.attribute) &&
               entry("cuMemGetAllocationGranularity", x.granularity) && entry("cuMemAddressReserve", x.reserve) &&
               entry("cuMemAddressFree", x.address_free) && entry("cuMemCreate", x.create) &&
               entry("cuMemRelease", x.release) && entry("cuMemMap", x.map) && entry("cuMemUnmap", x.unmap) &&
               entry("cuMemSetAccess", x.set_access);
        return x;
    }();
    return v;
}

dpct::experimental::mem_prop device_prop(int dev) {
    dpct::experimental::mem_prop prop{};
    prop.type = 0;
    prop.location.type = 1;
    prop.location.id = dev;
    return prop;
}

// one segment: a physical allocation mapped at `va`, readable and writable by this device
bool map_segment(const Vmm &v, int dev, dpct::device_ptr va, size_t bytes,
                 unsigned long long &handle) {
    const dpct::experimental::mem_prop prop = device_prop(dev);
    dpct::experimental::physical_mem_ptr h = 0;
    if (v.create(&h, bytes, &prop, 0) != 0) return false;
    if (v.map(va, bytes, 0, h, 0) != 0) {
        v.release(h);
        return false;
    }
    dpct::experimental::mem_access_desc access{};
    access.location.type = 1;
    access.location.id = dev;
    access.flags =
        sycl::ext::oneapi::experimental::address_access_mode::read_write;
    if (v.set_access(va, bytes, &access, 1) != 0) {
        v.unmap(va, bytes);
        v.release(h);
        return false;
    }
    handle = (unsigned long long) h;
    return true;
}
}  // namespace
#endif

namespace {
using SyclPhysicalMem = sycl::ext::oneapi::experimental::physical_mem;
namespace vm = sycl::ext::oneapi::experimental;

bool map_sycl_cache_segment(uint8_t* address, size_t bytes,
                            unsigned long long& handle, std::string& err) {
    const char* operation = "allocate";
    try {
        auto& q = dpct::get_in_order_queue();
        auto physical = std::make_unique<SyclPhysicalMem>(q, bytes);
        operation = "map";
        if (physical->map(reinterpret_cast<uintptr_t>(address), bytes,
                          vm::address_access_mode::read_write) != address) {
            err = "ExpertCache: a physical segment mapped at the wrong address";
            return false;
        }
        static_assert(sizeof(uintptr_t) <= sizeof(handle));
        handle = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(physical.release()));
        return true;
    } catch (const std::exception& e) {
        err = std::string("ExpertCache: cudaMalloc failed: SYCL physical segment ") + operation +
              " (" + std::to_string(bytes) + " bytes at " +
              std::to_string(reinterpret_cast<uintptr_t>(address)) + "): " + e.what();
        return false;
    }
}
} // namespace

bool ExpertCache::open_segmented(uint64_t want, std::string& err) {
    const auto cleanup = [&] {
        try { release_segmented(); }
        catch (const std::exception& e) { err += std::string("; cleanup also failed: ") + e.what(); }
        catch (...) { err += "; cleanup also failed: unknown exception"; }
    };
    try {
        auto& q = dpct::get_in_order_queue();
        if (!q.get_device().has(sycl::aspect::ext_oneapi_virtual_mem)) {
            err = "ExpertCache: this SYCL device does not support virtual memory";
            return false;
        }
        const uint64_t gran = vm::get_mem_granularity(q.get_device(), q.get_context());
        if (!gran || want > UINT64_MAX - gran ||
            static_cast<uint64_t>(seg_req_) > UINT64_MAX - gran) {
            err = "ExpertCache: invalid virtual memory granularity or size";
            return false;
        }
        seg_ = static_cast<int64_t>((static_cast<uint64_t>(seg_req_) + gran - 1) / gran * gran);
        // Arc B570 rejects a 5 MiB physical allocation even though its advertised
        // minimum granularity is 64 KiB (6 MiB and 64 MiB succeed). Use complete
        // segments, including the tail, rather than assuming every multiple of
        // the advertised minimum can back physical memory of this size.
        if (want > UINT64_MAX - static_cast<uint64_t>(seg_)) {
            err = "ExpertCache: segmented size overflow";
            return false;
        }
        const uint64_t total = (want + seg_ - 1) / seg_ * seg_;
        // Physical allocation and virtual reservation granularities are
        // distinct contracts, even when this single-device context agrees.
        const uint64_t context_gran = vm::get_mem_granularity(q.get_context(), vm::granularity_mode::minimum);
        if (!context_gran || total % context_gran) {
            err = "ExpertCache: segmented reservation is not aligned to the context granularity";
            return false;
        }
        base_ = reinterpret_cast<uint8_t*>(vm::reserve_virtual_mem(total, q.get_context()));
        reserved_ = total;
        for (uint64_t at = 0; at < total; at += static_cast<uint64_t>(seg_)) {
            segs_.push_back(0);
            seg_size_.push_back(static_cast<int64_t>(std::min<uint64_t>(seg_, total - at)));
        }
        for (size_t i = 0; i < segs_.size(); ++i) {
            if (!map_sycl_cache_segment(base_ + i * static_cast<uint64_t>(seg_),
                                         seg_size_[i], segs_[i], err)) {
                cleanup();
                return false;
            }
            mapped_segs_ = static_cast<int64_t>(i + 1);
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("ExpertCache: segmented allocation failed: ") + e.what();
        cleanup();
        return false;
    }
}

void ExpertCache::release_segmented() {
    auto& q = dpct::get_in_order_queue();
    if (base_) dpct::get_current_device().queues_wait_and_throw();
    for (size_t i = 0; i < segs_.size(); ++i) {
        if (!segs_[i]) continue;
        vm::unmap(base_ + i * static_cast<uint64_t>(seg_), seg_size_[i], q.get_context());
        delete reinterpret_cast<SyclPhysicalMem*>(static_cast<uintptr_t>(segs_[i]));
        segs_[i] = 0;
    }
    if (base_ && reserved_) vm::free_virtual_mem(reinterpret_cast<uintptr_t>(base_), reserved_, q.get_context());
    segs_.clear(); seg_size_.clear(); mapped_segs_ = 0; reserved_ = 0; base_ = nullptr;
}

int64_t ExpertCache::mapped_bytes() const {
    if (segs_.empty()) return base_ != nullptr ? full_bytes() : 0;
    int64_t b = 0;
    for (int64_t i = 0; i < mapped_segs_; ++i) b += seg_size_[(size_t) i];
    return b;
}

int64_t ExpertCache::slots_within(int64_t bytes) const {
    if (bytes >= full_bytes()) return slots_;
    if (bytes <= 0) return 0;
    if (off_.empty()) return blob_ > 0 ? bytes / blob_ : 0;
    // off_[i + 1] is slot i's end: count the slots whose end is at most `bytes`
    return (int64_t) (std::upper_bound(off_.begin() + 1, off_.end(), (uint64_t) bytes) - (off_.begin() + 1));
}

bool ExpertCache::shrink(int64_t keep_bytes, std::string& err) {
    if (segs_.empty()) {
        if (keep_bytes != 0) { err = "a non-segmented cache can only be released completely"; return false; }
        try {
            dpct::get_current_device().queues_wait_and_throw();
            if (base_) sycl::free(base_, dpct::get_in_order_queue());
            base_ = nullptr; live_slots_ = 0;
            return true;
        } catch (const std::exception& e) { err = e.what(); return false; }
    }
    try {
        int64_t keep = 0, at = 0;
        while (keep < static_cast<int64_t>(segs_.size()) && at < keep_bytes)
            at += seg_size_[static_cast<size_t>(keep++)];
        if (keep >= mapped_segs_) return true;
        dpct::get_current_device().queues_wait_and_throw();
        auto& q = dpct::get_in_order_queue();
        for (int64_t i = mapped_segs_ - 1; i >= keep; --i) {
            vm::unmap(base_ + i * static_cast<uint64_t>(seg_), seg_size_[i], q.get_context());
            delete reinterpret_cast<SyclPhysicalMem*>(static_cast<uintptr_t>(segs_[i]));
            segs_[i] = 0;
            mapped_segs_ = i;
        }
        live_slots_ = slots_within(mapped_bytes());
        return true;
    } catch (const std::exception& e) {
        live_slots_ = slots_within(mapped_bytes());
        err = std::string("ExpertCache: releasing a SYCL segment failed: ") + e.what();
        return false;
    }
}

bool ExpertCache::grow(int64_t want_bytes, std::string& err) {
    if (segs_.empty()) {
        if (want_bytes < full_bytes()) { err = "a non-segmented cache must be restored completely"; return false; }
        try {
            if (!base_) base_ = sycl::malloc_device<uint8_t>(static_cast<size_t>(full_bytes()), dpct::get_in_order_queue());
            if (!base_) { err = "ExpertCache: decode cache reallocation failed"; return false; }
            live_slots_ = slots_;
            return true;
        } catch (const std::exception& e) { err = e.what(); return false; }
    }
    int64_t at = mapped_bytes();
    const uint64_t target = want_bytes >= full_bytes() ? reserved_ : static_cast<uint64_t>(std::max<int64_t>(want_bytes, 0));
    while (mapped_segs_ < static_cast<int64_t>(segs_.size()) &&
           static_cast<uint64_t>(at + seg_size_[mapped_segs_]) <= target) {
        const size_t i = static_cast<size_t>(mapped_segs_);
        if (!map_sycl_cache_segment(base_ + i * static_cast<uint64_t>(seg_), seg_size_[i], segs_[i], err)) {
            live_slots_ = slots_within(mapped_bytes());
            return false;
        }
        at += seg_size_[i];
        ++mapped_segs_;
    }
    live_slots_ = slots_within(mapped_bytes());
    return true;
}

#if defined(STRATA_USE_HIP)
bool ExpertCache::ensure_blocking_staging(std::size_t bytes, std::string& err) {
    if (bytes <= blocking_staging_bytes_) return true;
    void* next = nullptr;
    const cudaError_t status = cudaHostAlloc(&next, bytes, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        err = std::string("ExpertCache: HIP blocking staging allocation: ") + cudaGetErrorString(status);
        return false;
    }
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
    blocking_staging_ = static_cast<uint8_t*>(next);
    blocking_staging_bytes_ = bytes;
    return true;
}
#endif

bool ExpertCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert,
                       int64_t blob_bytes, std::string &err) try {
    close();
    if (n_slots <= 0) {
        err = "ExpertCache: n_slots must be positive";
        return false;
    }
    if (n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        err = "ExpertCache: n_layers, n_expert and blob_bytes must all be positive";
        return false;
    }

    const uint64_t want = (uint64_t) n_slots * (uint64_t) blob_bytes;

    // ---- **THE ALLOCATION IS CHECKED AGAINST THE CARD, NOT AGAINST THE REQUEST.**
    //
    // `cudaMalloc` failing is the easy case. The one that matters is a machine where the weights already own
    // most of VRAM: the cache then takes what is left and `slots()` would report the number ASKED FOR while
    // `device_slot()` walks off the end. So the free-VRAM figure is read and compared BEFORE the allocation,
    // and the two numbers are named in the refusal.
    size_t free_b = 0, total_b = 0;
    /*
    DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    if (DPCT_CHECK_ERROR(
            dpct::get_current_device().get_memory_info(free_b, total_b)) == 0) {
        if ((uint64_t) free_b < want) {
            char buf[320];
            std::snprintf(buf, sizeof buf,
                          "ExpertCache: %lld slots x %lld B = %.2f GiB, but only %.2f GiB of VRAM is free "
                          "(%.2f GiB of %.2f GiB total). Lower --expert-cache.",
                          (long long) n_slots, (long long) blob_bytes, (double) want / 1073741824.0,
                          (double) free_b / 1073741824.0, (double) (total_b - free_b) / 1073741824.0,
                          (double) total_b / 1073741824.0);
            err = buf;
            return false;
        }
    }

    if (seg_req_ > 0) {   // #533: --vram-elastic: physical segments behind one address range (zeroed below)
        if (!open_segmented(want, err)) return false;
    } else if (DPCT_CHECK_ERROR(
                   base_ = (uint8_t *)strata::checked_usm(sycl::malloc_device(
                       (size_t)want, dpct::get_in_order_queue()))) != 0) {
        base_ = nullptr;
        char buf[256];
        std::snprintf(
            buf, sizeof buf, "ExpertCache: cudaMalloc(%.2f GiB) failed: %s",
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            (double)want / 1073741824.0, dpct::get_error_string_dummy(0));
        err = buf;
        return false;
    }
    // Zeroed so a slot read before it is filled is a DETERMINISTIC wrong answer rather than whatever the
    // allocator handed back.  A stale block of a previous process's memory would still sum to finite floats.
    if (DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memset(base_, 0, (size_t)want).wait()) !=
        0) {
        err = "ExpertCache: cudaMemset of the slot arena failed";
        close();
        return false;
    }

    residency_.assign((size_t) (n_layers * n_expert), kNotResident);
    slots_ = n_slots;
    live_slots_ = n_slots;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    blob_ = blob_bytes;
#if defined(STRATA_USE_HIP)
    if (!ensure_blocking_staging((std::size_t) blob_, err)) {
        close();
        return false;
    }
#endif
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    // R4.2g: each layer starts at the bottom of its own range.  Built here rather than lazily so `admit`
    // stays allocation-free on the token path.
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool ExpertCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
                             std::string& err) {
    if (slot_bytes.empty()) { err = "ExpertCache: no slots"; return false; }
    int64_t mx = 0;
    std::vector<uint64_t> off(slot_bytes.size() + 1, 0);
    for (size_t i = 0; i < slot_bytes.size(); ++i) {
        // 256-byte aligned slots, so every blob starts where the kernels' vector loads expect it
        off[i + 1] = off[i] + ((uint64_t) slot_bytes[i] + 255) / 256 * 256;
        mx = slot_bytes[i] > mx ? slot_bytes[i] : mx;
    }
    // one allocation of the summed size, through the uniform path's checks: n "slots" of 1 byte
    if (!open((int64_t) off.back(), n_layers, n_expert, 1, err)) return false;
    slots_ = (int64_t) slot_bytes.size();
    live_slots_ = slots_;
    blob_ = mx;
    off_ = std::move(off);
#if defined(STRATA_USE_HIP)
    if (!ensure_blocking_staging((std::size_t) blob_, err)) {
        close();
        return false;
    }
#endif
    // #369: each layer's cursor at the bottom of its own range, as open() seeds it - open() above ran on byte-sized
    // "slots", so its seeds are not slot indices
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
    return true;
}

void ExpertCache::close() {
#if defined(STRATA_USE_HIP)
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
    blocking_staging_ = nullptr;
    blocking_staging_bytes_ = 0;
#endif
    off_.clear();
    if (!segs_.empty()) {
        release_segmented();
    } else if (base_ != nullptr) {
        sycl::free(base_, dpct::get_in_order_queue());
        base_ = nullptr;
    }
    residency_.clear();
    slots_ = 0;
    live_slots_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    blob_ = 0;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    layer_next_.clear();
}

/// R4.2g.  Layer `l` owns `[l*q, (l+1)*q)` with `q = slots_ / n_layers_`; the LAST layer takes whatever is
/// left over, so the ranges always cover `0..slots_` exactly and no slot is orphaned by the division.
void ExpertCache::layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const {
    lo = 0;
    hi = 0;
    if (n_layers_ <= 0 || slots_ <= 0 || layer < 0 || layer >= n_layers_) return;
    const int64_t q = slots_ / n_layers_;
    lo = layer * q;
    hi = (layer == n_layers_ - 1) ? slots_ : (layer + 1) * q;
}

int32_t ExpertCache::slot_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    return residency_[(size_t) (layer * n_expert_ + expert)];
}

int32_t ExpertCache::admit(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    const size_t at = (size_t) (layer * n_expert_ + expert);
    if (residency_[at] != kNotResident) return residency_[at];
    // R4.2g: THE PER-LAYER PATH.  Same "no eviction" rule, but the ceiling is this layer's own range rather
    // than one counter shared by all 48 - which is what confined the measured hit rate to 2.97%.
    if (per_layer_) {
        if (layer_next_.empty()) return kNotResident;
        int64_t lo = 0, hi = 0;
        layer_slot_range(layer, lo, hi);
        if ((int64_t) layer_next_[(size_t) layer] >= hi) return kNotResident;   // this layer's quota is full
        residency_[at] = layer_next_[(size_t) layer]++;
        ++admitted_;
        return residency_[at];
    }
    if (next_free_ >= slots_) return kNotResident;   // full: no eviction, deliberately - see the header
    residency_[at] = (int32_t) next_free_;
    return (int32_t) next_free_++;
}

uint8_t* ExpertCache::device_slot(int32_t slot) {
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

const uint8_t* ExpertCache::device_slot(int32_t slot) const {
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

bool ExpertCache::fill_slot(int32_t slot, const uint8_t *host_blob,
                            void *stream, std::string &err, int64_t bytes) try {
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot: slot " + std::to_string(slot) + " is outside 0.." +
              std::to_string(slots_ - 1);
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot: the host blob is null";
        return false;
    }
    /*
    DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    const dpct::err0 e =
        DPCT_CHECK_ERROR(strata::q_of(stream)->memcpy(dst, host_blob, n));
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        err = std::string("ExpertCache::fill_slot: ") +
              dpct::get_error_string_dummy(e);
        return false;
    }
    ++fills_;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool ExpertCache::fill_slot_blocking(int32_t slot, const uint8_t *host_blob,
                                     std::string &err, int64_t bytes) try {
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot_blocking: slot outside the arena";
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot_blocking: the host blob is null";
        return false;
    }
#if defined(STRATA_USE_HIP)
    // Bound HIP's pageable-source staging to one expert instead of repeatedly
    // registering regions of the mmap. The blocking copy completes before reuse.
    if (!blocking_staging_ || n > blocking_staging_bytes_) {
        err = "ExpertCache::fill_slot_blocking: HIP staging buffer is too small";
        return false;
    }
    std::memcpy(blocking_staging_, host_blob, n);
    const cudaError_t e = cudaMemcpy(dst, blocking_staging_, n, cudaMemcpyHostToDevice);
#else
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    const dpct::err0 e =
        DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(dst, host_blob, n).wait());
#endif
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        err = std::string("ExpertCache::fill_slot_blocking: ") +
              dpct::get_error_string_dummy(e);
        return false;
    }
    ++fills_;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool ExpertCache::fill_slot_queued(int32_t slot, const uint8_t *host_blob,
                                   std::string &err, int64_t bytes) try {
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr || host_blob == nullptr) {
        err = dst == nullptr ? "ExpertCache::fill_slot_queued: slot outside the arena"
                             : "ExpertCache::fill_slot_queued: the host blob is null";
        return false;
    }
    /*
    DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    const dpct::err0 e =
        DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(dst, host_blob, n).wait());
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        err = std::string("ExpertCache::fill_slot_queued: ") +
              dpct::get_error_string_dummy(e);
        return false;
    }
    ++fills_;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool ExpertCache::sync_queued(std::string &err) try {
    const dpct::err0 e = DPCT_CHECK_ERROR(dpct::get_in_order_queue().wait());
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        err = std::string("ExpertCache::sync_queued: ") +
              dpct::get_error_string_dummy(e);
        return false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool ExpertCache::verify_slot(int32_t slot, const uint8_t *host_blob,
                              std::string &err, int64_t bytes) try {
    const int64_t nb = bytes > 0 && bytes <= blob_ ? bytes : blob_;
    const uint8_t* src = device_slot(slot);
    if (src == nullptr) {
        err = "ExpertCache::verify_slot: slot outside the arena";
        return false;
    }
    // `cudaMemcpy` and not `cudaMemcpyAsync`: this is a startup check, and a check that can be read before it
    // has happened is not a check.  It also synchronises the fills queued before it, which is what makes the
    // comparison meaningful.
    std::vector<uint8_t> got((size_t) nb);
    const dpct::err0 e = DPCT_CHECK_ERROR(
        dpct::get_in_order_queue().memcpy(got.data(), src, (size_t)nb).wait());
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        err = std::string("ExpertCache::verify_slot: ") +
              dpct::get_error_string_dummy(e);
        return false;
    }
    if (std::memcmp(got.data(), host_blob, (size_t) nb) != 0) {
        size_t first = 0;
        while (first < (size_t) nb && got[first] == host_blob[first]) ++first;
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "ExpertCache::verify_slot: slot %d differs from the arena at byte %llu (of %lld)",
                      (int) slot, (unsigned long long) first, (long long) blob_);
        err = buf;
        return false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::core
