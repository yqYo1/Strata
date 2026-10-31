#pragma once

// Authored for Strata; repository LICENSE applies. Prefill-only proposal.
#include "strata/sycl/launch.hpp"
#include "strata/sycl/compat/cuda_runtime.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::prefill::detail {

// One ascending scan of immutable, packed CPU expert blobs into a borrowed
// private device bank. The caller owns source/bank/stream through its checked
// drain, including cancellation and partial enqueue failure. No allocation,
// host tensor packing, additional queue, cache publication or host wait occurs
// here. The source must be private, not an externally mutable shared mapping.
//
// Ordinal i uses slot i%capacity, identically for every group size. At most
// 2/4/8 experts are uploaded before their readers (1 is the serial control).
// Groups stop at bank wrap and source gaps are never copied. Overwrites follow
// EVERY preceding reader in the SAME
// in-order queue. consumed() must follow the expert's final submitted reader.
// Multiple windows needing permanent-cache hits keep their existing scheduler.
class ExpertUploadBatch {
public:
    static constexpr size_t experts = 512;
    struct Lease { uint8_t* data; int32_t expert; size_t ordinal; };

    ExpertUploadBatch(void* bank, size_t bank_bytes, size_t destination_stride,
            const void* source, size_t source_bytes, size_t source_stride,
            size_t blob_bytes, const std::vector<int32_t>& order, void* stream,
            size_t group_experts)
        : bank_(static_cast<uint8_t*>(bank)), source_(static_cast<const uint8_t*>(source)),
          bytes_(blob_bytes), count_(order.size()), stream_(stream), group_(group_experts) {
        // Validate all geometry/order before the first enqueue. Reject padded
        // layouts rather than copying gaps or packing bytes on the CPU.
        if (!stream || !bank || !source || !bytes_ || bytes_ % 256 ||
            destination_stride != bytes_ || source_stride != bytes_ ||
            bytes_ > SIZE_MAX / experts || source_bytes != experts * bytes_ ||
            bank_bytes < bytes_ || !count_ || count_ > experts ||
            (group_ != 1 && group_ != 2 && group_ != 4 && group_ != 8))
            throw std::invalid_argument("expert upload batch source/bank extent");
        capacity_ = std::min(experts, bank_bytes / bytes_);
        int32_t previous = -1;
        for (size_t i = 0; i < count_; ++i) {
            if (order[i] <= previous || order[i] < 0 || size_t(order[i]) >= experts)
                throw std::invalid_argument("expert upload batch needs unique ascending experts");
            order_[i] = previous = order[i];
        }
        auto& queue = sycl_backend::queue_for(stream);
        if (!queue.has_property<sycl::property::queue::in_order>())
            throw std::invalid_argument("expert upload batch requires in-order consumers");
        sycl_backend::validate_device_span(bank, bank_bytes, queue);
        sycl_backend::validate_spans({{bank, bank_bytes}}, {{source, source_bytes}}, 1);
        if (sycl::get_pointer_type(source, queue.get_context()) == sycl::usm::alloc::device)
            throw std::invalid_argument("expert upload batch source must be CPU storage");
    }
    ExpertUploadBatch(const ExpertUploadBatch&) = delete;
    ExpertUploadBatch& operator=(const ExpertUploadBatch&) = delete;

    // Injectable enqueue only for an explicitly scoped mechanism measurement.
    // Production uses the unchanged compat copy below. A false/throwing enqueue
    // poisons this schedule: no retry or publication after ambiguous submission.
    template<class Enqueue>
    Lease acquire(size_t ordinal, Enqueue&& enqueue) {
        require_active();
        if (acquired_ || ordinal != next_ || ordinal >= count_)
            throw std::invalid_argument("expert upload batch acquire order");
        try {
            if (ordinal == uploaded_) {
                const size_t end = std::min({count_, ordinal + group_,
                                             ordinal + capacity_ - ordinal % capacity_});
                for (size_t first = ordinal; first < end;) {
                    size_t last = first + 1;
                    while (last < end && order_[last] == order_[last - 1] + 1) ++last;
                    const size_t length = last - first;
                    if (!enqueue(bank_ + (first % capacity_) * bytes_,
                                 source_ + size_t(order_[first]) * bytes_, length * bytes_, stream_))
                        throw std::runtime_error("expert upload batch enqueue failed");
                    ++copy_calls_; uploaded_ += length;
                    first = last;
                }
            }
        } catch (...) { poisoned_ = true; throw; }
        acquired_ = true;
        return {bank_ + (ordinal % capacity_) * bytes_, order_[ordinal], ordinal};
    }
    Lease acquire(size_t ordinal) {
        return acquire(ordinal, [](void* dst, const void* src, size_t bytes, void* stream) {
            return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                                   static_cast<cudaStream_t>(stream)) == cudaSuccess;
        });
    }
    void consumed(size_t ordinal) {
        require_active();
        if (!acquired_ || ordinal != next_)
            throw std::invalid_argument("expert upload batch release order");
        ++next_; acquired_ = false;
    }
    // Submission closure only. The outer owner must still perform its checked
    // queue drain before retiring/rebinding source, bank, stream or consumers.
    void finish_submissions() {
        require_active();
        if (acquired_ || next_ != count_)
            throw std::invalid_argument("expert upload batch incomplete submissions");
        finished_ = true;
    }
    size_t capacity() const { return capacity_; }
    size_t group_size() const { return group_; }
    size_t copy_calls() const { return copy_calls_; }
    size_t uploaded_experts() const { return uploaded_; }
    uint64_t uploaded_bytes() const { return uint64_t(uploaded_) * bytes_; }

private:
    void require_active() const {
        if (finished_ || poisoned_)
            throw std::invalid_argument("expert upload batch closed/poisoned");
    }
    uint8_t* bank_;
    const uint8_t* source_;
    size_t bytes_, count_, capacity_ = 0, next_ = 0, copy_calls_ = 0, uploaded_ = 0;
    void* stream_;
    size_t group_;
    bool acquired_ = false, finished_ = false, poisoned_ = false;
    std::array<int32_t, experts> order_{};
};
} // namespace strata::prefill::detail
