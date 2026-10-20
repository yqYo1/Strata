// src/core/expert_cache.cpp - R4's slot storage and residency table.  Read the header first.
#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <utility>
#include <cstring>
#ifdef STRATA_ENABLE_SYCL
#include "strata/sycl/checked_cleanup.hpp"
#endif

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

ExpertCache::~ExpertCache() { close(); }

#if defined(STRATA_USE_HIP) || defined(STRATA_ENABLE_SYCL)
bool ExpertCache::ensure_blocking_staging(std::size_t bytes, std::string& err) {
    if (bytes <= blocking_staging_bytes_) return true;
    void* next = nullptr;
    const cudaError_t status = cudaHostAlloc(&next, bytes, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        err = std::string("ExpertCache: blocking staging allocation: ") + cudaGetErrorString(status);
        return false;
    }
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
    blocking_staging_ = static_cast<uint8_t*>(next);
    blocking_staging_bytes_ = bytes;
    return true;
}
#endif

bool ExpertCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                       std::string& err) {
    return open_impl(n_slots, n_layers, n_expert, blob_bytes, err, true);
}

bool ExpertCache::open_uninitialized(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                                     std::string& err) {
    const uint64_t limit = std::min<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max(),
                                             std::numeric_limits<int64_t>::max());
    if (n_slots <= 0 || n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0 ||
        uint64_t(n_slots) > limit / uint64_t(blob_bytes) ||
        uint64_t(n_layers) > limit / sizeof(int32_t) / uint64_t(n_expert)) {
        err = "ExpertCache: uninitialized cache geometry overflows addressable storage";
        return false;
    }
    return open_impl(n_slots, n_layers, n_expert, blob_bytes, err, false);
}

bool ExpertCache::open_impl(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                            std::string& err, bool initialize, std::vector<uint64_t> offsets,
                            std::vector<int64_t> slot_sizes) {
    close();
    if (n_slots <= 0 || n_slots > std::numeric_limits<int32_t>::max()) {
        err = "ExpertCache: n_slots must fit a positive int32 slot index";
        return false;
    }
    if (n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        err = "ExpertCache: n_layers, n_expert and blob_bytes must all be positive";
        return false;
    }

    const uint64_t limit = std::min<uint64_t>(std::numeric_limits<int64_t>::max(),
                                              std::numeric_limits<size_t>::max());
    if ((offsets.empty() && uint64_t(n_slots) > limit / uint64_t(blob_bytes)) ||
        (!offsets.empty() && (offsets.size() != size_t(n_slots) + 1 ||
                             slot_sizes.size() != size_t(n_slots) || offsets.back() > limit))) {
        err = "ExpertCache: slot arena geometry overflows its byte representation"; return false;
    }
    if (uint64_t(n_layers) > limit / uint64_t(n_expert) ||
        uint64_t(n_layers) * uint64_t(n_expert) > residency_.max_size() ||
        uint64_t(n_layers) > layer_next_.max_size()) {
        err = "ExpertCache: residency geometry overflows its host representation"; return false;
    }
    const uint64_t want = offsets.empty() ? uint64_t(n_slots) * uint64_t(blob_bytes) : offsets.back();

    // ---- **THE ALLOCATION IS CHECKED AGAINST THE CARD, NOT AGAINST THE REQUEST.**
    //
    // `cudaMalloc` failing is the easy case. The one that matters is a machine where the weights already own
    // most of VRAM: the cache then takes what is left and `slots()` would report the number ASKED FOR while
    // `device_slot()` walks off the end. So the free-VRAM figure is read and compared BEFORE the allocation,
    // and the two numbers are named in the refusal.
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
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

    if (cudaMalloc((void**) &base_, (size_t) want) != cudaSuccess) {
        base_ = nullptr;
        char buf[256];
        std::snprintf(buf, sizeof buf, "ExpertCache: cudaMalloc(%.2f GiB) failed: %s",
                      (double) want / 1073741824.0, cudaGetErrorString(cudaGetLastError()));
        err = buf;
        return false;
    }
    // Zeroed so a slot read before it is filled is a DETERMINISTIC wrong answer rather than whatever the
    // allocator handed back.  A stale block of a previous process's memory would still sum to finite floats.
    if (initialize && cudaMemset(base_, 0, (size_t) want) != cudaSuccess) {
        err = "ExpertCache: cudaMemset of the slot arena failed";
        close();
        return false;
    }

    residency_.assign((size_t) (n_layers * n_expert), kNotResident);
    slots_ = n_slots;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    blob_ = blob_bytes;
    off_ = std::move(offsets);
    slot_sizes_ = std::move(slot_sizes);
#if defined(STRATA_USE_HIP) || defined(STRATA_ENABLE_SYCL)
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

bool ExpertCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
                             std::string& err) {
    if (slot_bytes.empty()) { err = "ExpertCache: no slots"; return false; }
    if (slot_bytes.size() > size_t(std::numeric_limits<int32_t>::max())) {
        err = "ExpertCache: sized slot count exceeds int32 indices"; return false;
    }
    const uint64_t limit = std::min<uint64_t>(std::numeric_limits<int64_t>::max(),
                                              std::numeric_limits<size_t>::max());
    int64_t mx = 0;
    std::vector<uint64_t> off(slot_bytes.size() + 1, 0);
    for (size_t i = 0; i < slot_bytes.size(); ++i) {
        // 256-byte aligned slots, so every blob starts where the kernels' vector loads expect it
        if (slot_bytes[i] <= 0 || uint64_t(slot_bytes[i]) > limit - 255) {
            err = "ExpertCache: nonpositive or overflowing sized slot payload"; return false;
        }
        const auto aligned = (uint64_t(slot_bytes[i]) + 255) / 256 * 256;
        if (aligned > limit - off[i]) {
            err = "ExpertCache: sized slot arena sum overflows"; return false;
        }
        off[i + 1] = off[i] + aligned;
        mx = slot_bytes[i] > mx ? slot_bytes[i] : mx;
    }
    // Keep real slot indices throughout allocation; a multi-GiB byte arena is
    // not a cache with billions of one-byte slots or int32 cursor seeds.
    return open_impl(int64_t(slot_bytes.size()), n_layers, n_expert, mx, err, true,
                     std::move(off), slot_bytes);
}

void ExpertCache::close() {
#ifdef STRATA_ENABLE_SYCL
    if (scratch_state_) {
        strata::sycl_backend::checked_cleanup([&] {
            // Refuse the close before waiting/freeing when the loan is live.
            scratch_state_->close();
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("cache scratch: parent completion unknown");
        }, [] {}, "cache scratch parent closed before verified retirement/refill");
    }
#endif
#if defined(STRATA_USE_HIP) || defined(STRATA_ENABLE_SYCL)
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
    blocking_staging_ = nullptr;
    blocking_staging_bytes_ = 0;
#endif
    off_.clear();
    slot_sizes_.clear();
    if (base_ != nullptr) {
        const auto released = cudaFree(base_);
#ifdef STRATA_ENABLE_SYCL
        if (scratch_state_ && released != cudaSuccess)
            strata::sycl_backend::cleanup_fail_stop("cache scratch parent free failed");
#else
        (void) released;
#endif
        base_ = nullptr;
    }
    residency_.clear();
    slots_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    blob_ = 0;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    layer_next_.clear();
#ifdef STRATA_ENABLE_SYCL
    scratch_state_.reset();
#endif
}

#ifdef STRATA_ENABLE_SYCL
void ExpertCache::enable_scratch_loan_tracking() {
    if (!base_ || bytes() <= 0) throw std::logic_error("cache scratch: missing parent allocation");
    if (!scratch_state_) scratch_state_ = std::make_shared<detail::CacheScratchState>(this, base_, uint64_t(bytes()));
}
CacheScratchLease ExpertCache::prepare_scratch_loan(int32_t first_slot) {
    if (!scratch_state_ || first_slot < 128 || first_slot >= slots_)
        throw std::invalid_argument("cache scratch: loan requires enabled parent and128-slot prefix");
    if (!base_ || bytes() <= 0 || blob_ <= 0 ||
        (!off_.empty() && off_.size() != size_t(slots_) + 1) ||
        (off_.empty() && uint64_t(first_slot) > uint64_t(bytes()) / uint64_t(blob_)))
        throw std::invalid_argument("cache scratch: invalid parent geometry");
    const uint64_t offset = off_.empty() ? uint64_t(first_slot) * uint64_t(blob_) : off_[size_t(first_slot)];
    const uint64_t total = uint64_t(bytes());
    if (offset >= total) throw std::invalid_argument("cache scratch: tail outside parent");
    const auto generation = scratch_state_->prepare(offset, total - offset);
    return CacheScratchLease(scratch_state_, generation);
}
#endif

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

bool ExpertCache::fill_slot(int32_t slot, const uint8_t* host_blob, void* stream, std::string& err, int64_t bytes) {
    size_t n = 0;
    if (!fill_extent(slot, bytes, n, err)) return false;
#ifdef STRATA_ENABLE_SYCL
    std::optional<detail::CacheScratchState::Writer> writer;
    if (!scratch_fill_reserve(slot, n, writer, err)) return false;
#endif
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
    const cudaError_t e = cudaMemcpyAsync(dst, host_blob, n, cudaMemcpyHostToDevice,
                                          (cudaStream_t) stream);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_blocking(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    size_t n = 0;
    if (!fill_extent(slot, bytes, n, err)) return false;
#ifdef STRATA_ENABLE_SYCL
    std::optional<detail::CacheScratchState::Writer> writer;
    if (!scratch_fill_reserve(slot, n, writer, err)) return false;
#endif
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot_blocking: slot outside the arena";
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot_blocking: the host blob is null";
        return false;
    }
#if defined(STRATA_USE_HIP) || defined(STRATA_ENABLE_SYCL)
    // Bound pageable-source staging to one expert instead of repeatedly
    // registering regions of the mmap. The blocking copy completes before reuse.
    if (!blocking_staging_ || n > blocking_staging_bytes_) {
        err = "ExpertCache::fill_slot_blocking: staging buffer is too small";
        return false;
    }
    std::memcpy(blocking_staging_, host_blob, n);
    const cudaError_t e = cudaMemcpy(dst, blocking_staging_, n, cudaMemcpyHostToDevice);
#else
    const cudaError_t e = cudaMemcpy(dst, host_blob, n, cudaMemcpyHostToDevice);
#endif
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot_blocking: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_queued(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    size_t n = 0;
    if (!fill_extent(slot, bytes, n, err)) return false;
#ifdef STRATA_ENABLE_SYCL
    std::optional<detail::CacheScratchState::Writer> writer;
    if (!scratch_fill_reserve(slot, n, writer, err)) return false;
#endif
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr || host_blob == nullptr) {
        err = dst == nullptr ? "ExpertCache::fill_slot_queued: slot outside the arena"
                             : "ExpertCache::fill_slot_queued: the host blob is null";
        return false;
    }
    const cudaError_t e = cudaMemcpyAsync(dst, host_blob, n, cudaMemcpyHostToDevice, (cudaStream_t) 0);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot_queued: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::sync_queued(std::string& err) {
    const cudaError_t e = cudaStreamSynchronize((cudaStream_t) 0);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::sync_queued: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool ExpertCache::fill_extent(int32_t slot, int64_t requested, size_t& bytes, std::string& err) const {
    if (!base_ || slot < 0 || slot >= slots_ || requested < 0) {
        err = "ExpertCache: invalid slot or negative payload size"; return false;
    }
    const int64_t payload = slot_sizes_.empty() ? blob_ : slot_sizes_[size_t(slot)];
    const int64_t count = requested ? requested : payload;
    const auto capacity = off_.empty() ? uint64_t(blob_) : off_[size_t(slot) + 1] - off_[size_t(slot)];
    if (count <= 0 || count > blob_ || uint64_t(count) > capacity) {
        err = "ExpertCache: payload exceeds this slot's extent"; return false;
    }
    bytes = size_t(count); return true;
}

bool ExpertCache::verify_slot(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    size_t nb = 0;
    if (!fill_extent(slot, bytes, nb, err) || !host_blob) {
        if (!host_blob) err = "ExpertCache::verify_slot: the host blob is null";
        return false;
    }
    const uint8_t* src = device_slot(slot);
    if (src == nullptr) {
        err = "ExpertCache::verify_slot: slot outside the arena";
        return false;
    }
    // `cudaMemcpy` and not `cudaMemcpyAsync`: this is a startup check, and a check that can be read before it
    // has happened is not a check.  It also synchronises the fills queued before it, which is what makes the
    // comparison meaningful.
    std::vector<uint8_t> got((size_t) nb);
    const cudaError_t e = cudaMemcpy(got.data(), src, (size_t) nb, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::verify_slot: ") + cudaGetErrorString(e);
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

}  // namespace strata::core
