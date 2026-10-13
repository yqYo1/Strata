// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "strata/core/phase_timing.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {
// Successful attach/detach establishes mapped/unmapped state. A false result
// provides no post-error state guarantee: retain physical, virtual and backend
// ownership until process teardown, without retrying unmap or implicit recovery.
struct PhaseCacheBackend {
    struct Caps {
        bool vmm = false, free_known = false;
        uint64_t granularity = 0, maximum = 0, free = 0;
        uint64_t device_minimum = 0, context_minimum = 0;
    };
    virtual ~PhaseCacheBackend() = default;
    virtual bool caps(Caps&, std::string&) noexcept = 0;
    virtual void* reserve(uint64_t, std::string&) noexcept = 0;
    virtual bool release(void*, uint64_t, std::string&) noexcept = 0;
    virtual void* physical(uint64_t, std::string&) noexcept = 0;
    virtual bool attach(void*, void*, uint64_t, std::string&) noexcept = 0;
    virtual bool detach(void*, uint64_t, std::string&) noexcept = 0;
    virtual void destroy_physical(void*) noexcept = 0;
    virtual bool drain(std::string&) noexcept = 0;
    virtual bool write(void*, const void*, uint64_t, std::string&) noexcept = 0;
    virtual bool read(void*, const void*, uint64_t, std::string&) noexcept = 0;
};
std::shared_ptr<PhaseCacheBackend> make_phase_cache_gpu_backend(int ordinal);

// Opt-in persistent-serve owner. No session/settings/mode
// changes. Source/ranked-profile identity and native format validation are caller
// admission requirements. A quiesce callback MUST commit verifier/adaptive work
// and exclude all concurrent readers; backend drain alone cannot prove that.
class PhaseCacheOwner {
public:
    struct Timings { PhaseTiming quiesce, map, unmap, h2d, d2h, compare, publish, invalidate, close, physical, destroy, drain, reserve, release, residency_h2d, residency_d2h, residency_compare; };
    const Timings& timings() const { return timings_; }
    Timings& transfer_timings() { return timings_; }
    using Quiesce = std::function<bool(std::string&)>;
    using Publish = std::function<bool(const int32_t*, size_t, std::string&)>;
    static constexpr uint64_t guard_bytes = uint64_t{512} << 20;
    explicit PhaseCacheOwner(std::shared_ptr<PhaseCacheBackend> backend);
    ~PhaseCacheOwner();  // Unclosed native owners fail-stop; never implicit unsafe free.
    PhaseCacheOwner(const PhaseCacheOwner&) = delete;
    PhaseCacheOwner& operator=(const PhaseCacheOwner&) = delete;
    bool reserve(const std::vector<int64_t>& layer_blobs, uint32_t experts,
                 uint64_t virtual_cap, std::string& error, uint32_t extras_per_layer = 2);
    // Requires actual quiescence and retirement of retained readers.
    bool begin_prefill(const Quiesce& retired_readers, const Publish&, std::string& error);
    bool finish_prefill(const Quiesce&, std::string& error);
    bool prefill_leased() const { return base_ && prefill_leased_ && !poisoned_ && !readers_ready_ && mapped_slots() == 0; }
    bool map_base(const Quiesce&, std::string& error);
    bool map_extras(const Quiesce&, std::string& error);
    bool fill_checked(uint32_t layer, uint32_t expert, uint32_t slot,
                      const void* immutable_blob, uint64_t bytes, std::string& error);
    bool publish(const Publish&, std::string& error);
    bool invalidate_extras(const Publish&, std::string& error);
    bool unmap_extras(const Quiesce&, std::string& error);
    // release_readers must retire every retained cache/table/graph reference.
    // Unknown attach/detach state prevents all further cleanup and retains the
    // backend/context, physical handles and virtual range until process teardown.
    // Failed release retains the virtual range; known-unmapped cleanup may retry.
    bool close(const Quiesce& release_readers, std::string& error);
    uint32_t capacity() const { return static_cast<uint32_t>(offsets_.size()); }
    uint32_t mapped_slots() const;
    uint64_t mapped_bytes() const;
    uint64_t retained_physical_bytes() const;
    uint64_t reserved_bytes() const { return reserved_; }
    bool poisoned() const { return poisoned_; }
    bool open() const { return base_ != nullptr; }
    bool readers_ready() const { return base_ && !poisoned_ && readers_ready_; }
    const uint64_t* capacity_offsets() const { return offsets_.data(); }
    const int32_t* host_residency() const { return published_.data(); }
    size_t residency_size() const { return published_.size(); }
    void* base() const { return base_; }
    void* device_slot(uint32_t slot) const;
    uint64_t slot_bytes(uint32_t slot) const;
    uint64_t slot_stride(uint32_t slot) const;
    struct Mapping { uint64_t offset, bytes; bool owns_physical, mapped, uncertain; };
    std::vector<Mapping> mapping_snapshot() const;
    std::vector<uint32_t> layer_slots(uint32_t layer) const;
    int32_t vacant_slot(uint32_t layer) const;
    int32_t resident_slot(uint32_t layer, uint32_t expert) const;
private:
    struct Extent { uint64_t offset = 0, bytes = 0; void* physical = nullptr; bool attached = false, maybe_attached = false; };
    bool fail(const char*, std::string&);
    bool drain(std::string&);
    bool quiesce(const Quiesce&, std::string&);
    bool map_extent(size_t, std::string&);
    bool budget(uint64_t, std::string&);
    bool detach_extent(size_t, std::string&);
    bool extra(uint32_t slot) const { return slot >= layers_ * 3; }
    uint32_t slot_layer(uint32_t slot) const;
    bool mapped(uint32_t slot) const;
    bool upload(const Publish&, std::string&);
    Timings timings_;
    std::shared_ptr<PhaseCacheBackend> backend_;
    void* base_ = nullptr;
    uint32_t layers_ = 0, experts_ = 0, extras_per_layer_ = 2;
    uint64_t reserved_ = 0, granularity_ = 0, maximum_ = 0;
    uint64_t recommended_ = 0, device_minimum_ = 0, context_minimum_ = 0;
    bool poisoned_ = false, readers_ready_ = false, prefill_leased_ = false;
    std::vector<Extent> extents_; // baseline then one bounded-extra extent/layer
    std::vector<uint64_t> offsets_, strides_, blobs_;
    std::vector<int32_t> filled_, staged_, published_;
    std::vector<uint8_t> verification_; // survives failed/pending D2H until checked close
};
}  // namespace strata::core
