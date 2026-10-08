#pragma once

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

// Private opt-in prototype. All producers and consumers must use the same
// in-order queue. The owner outlives them and waits before freeing USM.
class ExpandedFp16LayerCache {
public:
    static constexpr size_t gu_elements = 1280 * 2560;
    static constexpr size_t d_elements = 2560 * 640;
    static constexpr size_t elements_per_expert = gu_elements + d_elements;
    static constexpr size_t bytes_per_expert = elements_per_expert * sizeof(sycl::half);
    static constexpr size_t reserve_bytes = 256 * 1024 * 1024;
    static_assert(sizeof(sycl::half) == sizeof(uint16_t));

    struct Entry {
        uint16_t* gu = nullptr;
        uint16_t* down = nullptr;
        int slot = -1;
        bool hit = false;
    };

    ExpandedFp16LayerCache() = default;
    ExpandedFp16LayerCache(const ExpandedFp16LayerCache&) = delete;
    ExpandedFp16LayerCache& operator=(const ExpandedFp16LayerCache&) = delete;
    ~ExpandedFp16LayerCache() noexcept {
        try { reset(); }
        catch (const std::exception& e) {
            // A failed wait gives no permission to free memory still in use.
            std::fprintf(stderr, "prefill FP16 cache: retaining USM after failed wait: %s\n", e.what());
        }
    }

    static int requested(int experts) {
        const char* text = std::getenv("STRATA_PREFILL_FP16_CACHE_EXPERTS");
        if (!text || !*text) return 0;
        errno = 0;
        char* end = nullptr;
        const long value = std::strtol(text, &end, 10);
        if (errno || end == text || *end || value < 0 || value > experts) {
            std::fprintf(stderr, "prefill FP16 cache: invalid expert count; using uncached path\n");
            return 0;
        }
        return static_cast<int>(value);
    }

    void initialize(sycl::queue& queue, int experts, int request, size_t free_bytes) {
        if (request <= 0 || experts <= 0 || free_bytes <= reserve_bytes) return;
        if (!queue.has_property<sycl::property::queue::in_order>()) return;
        try {
            const size_t max_allocation = queue.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
            const size_t slots = std::min({static_cast<size_t>(request), static_cast<size_t>(experts),
                                          (free_bytes - reserve_bytes) / bytes_per_expert,
                                          max_allocation / bytes_per_expert});
            if (!slots) return;
            // Prepare bookkeeping before allocating USM; no throwing allocation
            // follows a successful device allocation.
            expert_slots_.assign(static_cast<size_t>(experts), -1);
            ready_.assign(slots, 0);
            sycl::half* allocation = sycl::malloc_device<sycl::half>(slots * elements_per_expert, queue);
            if (!allocation) return;
            queue_ = &queue;
            data_ = allocation;
            capacity_ = static_cast<int>(slots);
        } catch (const sycl::exception& e) {
            std::fprintf(stderr, "prefill FP16 cache: allocation unavailable; using uncached path: %s\n", e.what());
        } catch (const std::bad_alloc&) {
            std::fprintf(stderr, "prefill FP16 cache: host allocation unavailable; using uncached path\n");
        }
    }

    bool enabled() const { return data_ != nullptr; }
    int capacity() const { return capacity_; }
    size_t bytes() const { return static_cast<size_t>(capacity_) * bytes_per_expert; }

    Entry acquire(int64_t layer, int expert) {
        if (!data_ || expert < 0 || static_cast<size_t>(expert) >= expert_slots_.size()) return {};
        if (layer != layer_) {
            // Layer-major processing finishes every old layer on queue_ before
            // replacing the source weights. Reuse is also ordered on queue_.
            std::fill(expert_slots_.begin(), expert_slots_.end(), -1);
            std::fill(ready_.begin(), ready_.end(), 0);
            used_ = 0;
            layer_ = layer;
        }
        int& slot = expert_slots_[static_cast<size_t>(expert)];
        if (slot < 0) {
            if (used_ == capacity_) { ++uncached_; return {}; }
            slot = used_++;
            ++admissions_;
        }
        Entry entry;
        sycl::half* start = data_ + static_cast<size_t>(slot) * elements_per_expert;
        entry.gu = reinterpret_cast<uint16_t*>(start);
        entry.down = reinterpret_cast<uint16_t*>(start + gu_elements);
        entry.slot = slot;
        entry.hit = ready_[static_cast<size_t>(slot)] != 0;
        if (entry.hit) ++hits_;
        return entry;
    }

    void publish(const Entry& entry) {
        // Call only after BOTH unchanged dequant producers were enqueued.
        // Later consumers are ordered by the same queue; no host load of USM.
        if (entry.slot >= 0) ready_[static_cast<size_t>(entry.slot)] = 1;
    }

    void report() const {
        std::fprintf(stderr, "strata prefill FP16 cache: capacity %d, bytes %llu, admitted %llu, hits %llu, uncached %llu, skipped dequant launches %llu\n",
                     capacity_, static_cast<unsigned long long>(bytes()),
                     static_cast<unsigned long long>(admissions_), static_cast<unsigned long long>(hits_),
                     static_cast<unsigned long long>(uncached_), static_cast<unsigned long long>(2 * hits_));
    }

    void reset() {
        if (!data_) return;
        queue_->wait_and_throw();
        sycl::free(data_, *queue_);
        data_ = nullptr;
        capacity_ = used_ = 0;
    }

private:
    sycl::queue* queue_ = nullptr;
    sycl::half* data_ = nullptr;
    std::vector<int> expert_slots_;
    std::vector<uint8_t> ready_;
    int capacity_ = 0, used_ = 0;
    int64_t layer_ = -1;
    uint64_t admissions_ = 0, hits_ = 0, uncached_ = 0;
};
