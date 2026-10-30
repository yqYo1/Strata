#pragma once

// Authored for Strata; repository LICENSE applies. Prefill-only proposal.
// The caller lends immutable ordinary/host-USM source storage and a private
// device bank. Neither source storage nor bank ownership is transferred here.
#include "strata/sycl/launch.hpp"
#include "strata/sycl/cpu_wait_profile.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <vector>

namespace strata::prefill::detail {

// Two slots suffice to overlap the next upload with the current expert's GPU
// consumers. There is no general-cache publication, host tensor arithmetic,
// overfetch or extra device/host allocation. This is a one-pass schedule only;
// repeated-window residency policies remain a separate choice.
class ExpertCopyPipeline {
public:
    static constexpr size_t depth = 2, experts = 512;
    struct Lease {
        uint8_t* data;
        int32_t expert;
        size_t ordinal;
        sycl::event uploaded;
    };

    ExpertCopyPipeline(void* bank, size_t bank_bytes, size_t destination_stride,
            const void* source, size_t source_bytes, size_t source_stride,
            size_t blob_bytes, const std::vector<int32_t>& order, void* stream)
        : runtime_(sycl_backend::runtime_for_device_span(bank, bank_bytes,
                       sycl_backend::queue_for(stream))),
          compute_(sycl_backend::queue_for(stream)), copy_(runtime_->make_queue(true)),
          bank_(static_cast<uint8_t*>(bank)), source_(static_cast<const uint8_t*>(source)),
          stride_(destination_stride), source_stride_(source_stride), bytes_(blob_bytes),
          order_(order) {
        // Admission is complete before start() submits the first copy. A
        // constructor failure therefore never leaves an unowned DMA source.
        if (!stream || !source || !bytes_ || !stride_ || stride_ % 256 ||
            !source_stride_ || bytes_ > stride_ || bytes_ > source_stride_ ||
            stride_ > bank_bytes / depth || source_stride_ > SIZE_MAX / experts ||
            source_bytes != experts * source_stride_ || order_.empty() || order_.size() > experts)
            throw std::invalid_argument("expert copy pipeline source/bank extent");
        if (!compute_.has_property<sycl::property::queue::in_order>() ||
            !copy_.has_property<sycl::property::queue::in_order>() ||
            compute_.get_context() != copy_.get_context() || compute_.get_device() != copy_.get_device())
            throw std::invalid_argument("expert copy pipeline queue/context contract");
        sycl_backend::validate_spans({{bank, bank_bytes}}, {{source, source_bytes}}, 1);
        if (sycl::get_pointer_type(source, compute_.get_context()) == sycl::usm::alloc::device)
            throw std::invalid_argument("expert copy pipeline source must be CPU storage");
        int32_t previous = -1;
        for (const auto expert : order_) {
            if (expert <= previous || expert < 0 || size_t(expert) >= experts)
                throw std::invalid_argument("expert copy pipeline needs unique ascending experts");
            previous = expert;
        }
        runtime_->check();
    }
    ~ExpertCopyPipeline() { drain_noexcept(); }
    ExpertCopyPipeline(const ExpertCopyPipeline&) = delete;
    ExpertCopyPipeline& operator=(const ExpertCopyPipeline&) = delete;

    void start() {
        if (started_ || closed_) throw std::invalid_argument("expert copy pipeline already started/closed");
        started_ = true;
        try {
            for (size_t i = 0; i < std::min(depth, order_.size()); ++i) enqueue(i, {});
        } catch (...) { drain_noexcept(); throw; }
    }

    Lease acquire(size_t ordinal) {
        if (!started_ || closed_ || acquired_ || ordinal != next_ || ordinal >= order_.size() ||
            !uploaded_[ordinal % depth])
            throw std::invalid_argument("expert copy pipeline acquire order");
        runtime_->check();
        const auto ready = *uploaded_[ordinal % depth];
        // A device dependency, not a host event wait. Every consumer is queued
        // after this barrier in the same in-order compute queue.
        compute_.ext_oneapi_submit_barrier({ready});
        acquired_ = true;
        return {bank_ + (ordinal % depth) * stride_, order_[ordinal], ordinal, ready};
    }

    void consumed(size_t ordinal) {
        if (!started_ || closed_ || !acquired_ || ordinal != next_)
            throw std::invalid_argument("expert copy pipeline release order");
        // This marker follows ALL current expert readers, not just its first
        // dequant/GEMM. Copy i+2 may overwrite its slot only after this marker.
        const auto used = compute_.ext_oneapi_submit_barrier();
        try {
            if (ordinal + depth < order_.size()) enqueue(ordinal + depth, used);
            ++next_; acquired_ = false;
        } catch (...) { drain_noexcept(); throw; }
    }

    void retire() {
        if (!started_ || acquired_ || next_ != order_.size())
            throw std::invalid_argument("expert copy pipeline incomplete retirement");
        drain(); closed_ = true;
    }
    size_t copies() const { return copies_; }
    uint64_t uploaded_bytes() const { return uint64_t(copies_) * bytes_; }

private:
    void enqueue(size_t ordinal, std::optional<sycl::event> previous_reader) {
        auto* destination = bank_ + (ordinal % depth) * stride_;
        const auto* source = source_ + size_t(order_[ordinal]) * source_stride_;
        // The copy queue is in order. Initial copies have no compute
        // dependency; later copies depend only on an ALREADY submitted reader
        // marker. Consequently no dependency cycle is introduced on unwind.
        uploaded_[ordinal % depth] = copy_.submit([destination, source, bytes = bytes_, previous_reader](sycl::handler& handler) {
            if (previous_reader) handler.depends_on(*previous_reader);
            handler.memcpy(destination, source, bytes);
        });
        ++copies_;
    }
    void drain() {
        sycl_backend::cpu_wait_profile::blocking(sycl_backend::cpu_wait_profile::direct_queue, [&] {
            copy_.wait_and_throw(); compute_.wait_and_throw();
        });
        runtime_->check();
    }
    void drain_noexcept() noexcept {
        if (closed_) return;
        try { drain(); closed_ = true; } catch (...) { std::_Exit(86); }
    }
    std::shared_ptr<sycl_backend::Runtime> runtime_;
    sycl::queue compute_, copy_;
    uint8_t* bank_;
    const uint8_t* source_;
    size_t stride_, source_stride_, bytes_, next_ = 0, copies_ = 0;
    std::vector<int32_t> order_;
    std::array<std::optional<sycl::event>, depth> uploaded_;
    bool started_ = false, acquired_ = false, closed_ = false;
};
} // namespace strata::prefill::detail
