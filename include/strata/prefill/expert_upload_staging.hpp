#pragma once
// Main-authored; repository LICENSE applies. Long-prefill transfer preparation,
// not model arithmetic. Every host slot has its own copy-completion event.
#include "strata/sycl/launch.hpp"
#include "strata/sycl/cpu_wait_profile.hpp"
#include <array>
#include <cstring>
#include <cstdlib>
#include <optional>

namespace strata::prefill::detail {

class ExpertUploadStaging {
public:
    static constexpr size_t slots = 4, guard = 256, maximum_blob = 16 << 20;
    ExpertUploadStaging(const void* device_owner, size_t device_bytes,
                        size_t maximum_bytes, void* stream)
        : runtime_(sycl_backend::runtime_for_device_span(device_owner, device_bytes,
                       sycl_backend::queue_for(stream))),
          queue_(sycl_backend::queue_for(stream)), capacity_(checked_capacity(maximum_bytes)),
          stride_((capacity_ + 2 * guard + 4095) / 4096 * 4096),
          storage_(runtime_, slots * stride_, sycl_backend::MemoryKind::Host) {
        if (!stream) throw std::invalid_argument("expert staging requires explicit queue");
        std::memset(storage_.data(), 0xa5, storage_.size());
    }
    ~ExpertUploadStaging() { drain_noexcept(); }
    ExpertUploadStaging(const ExpertUploadStaging&) = delete;
    ExpertUploadStaging& operator=(const ExpertUploadStaging&) = delete;

    bool enqueue(void* destination, const void* source, size_t bytes, void* stream) {
        if (!stream || !source || !bytes || bytes > capacity_ ||
            sycl_backend::queue_for(stream) != queue_)
            throw std::invalid_argument("expert staging source/size/queue mismatch");
        sycl_backend::validate_device_span(destination, bytes, queue_);
        sycl_backend::validate_spans({{destination, bytes}, {storage_.data(), storage_.size()}},
                                    {{source, bytes}}, 1);
        if (sycl::get_pointer_type(source, queue_.get_context()) == sycl::usm::alloc::device)
            throw std::invalid_argument("expert staging requires a CPU-readable source");
        if (next_ == std::numeric_limits<uint64_t>::max() ||
            copied_bytes_ > std::numeric_limits<uint64_t>::max() - bytes)
            throw std::overflow_error("expert staging accounting overflow");
        runtime_->check();
        try {
            const size_t index = next_ % slots;
            // A copy event protects the host slot. Its later GPU readers use
            // the destination, so they do not extend the host slot's lifetime.
            if (pending_[index]) runtime_->wait(*pending_[index]);
            auto* host = storage_.as<uint8_t>() + index * stride_ + guard;
            // Subregions are compiled only into the diagnostic fixture. A
            // normal engine keeps its existing outer, nonnested Host region.
            {
#if defined(STRATA_SYCL_EXPERT_UPLOAD_ATTRIBUTION)
                sycl_backend::cpu_wait_profile::Host preparation(sycl_backend::cpu_wait_profile::source_copy);
#endif
                std::memcpy(host, source, bytes);
            }
            auto event = [&] {
#if defined(STRATA_SYCL_EXPERT_UPLOAD_ATTRIBUTION)
                sycl_backend::cpu_wait_profile::Host submission(sycl_backend::cpu_wait_profile::upload_submit);
#endif
                return queue_.memcpy(destination, host, bytes);
            }();
            pending_[index] = event;
            sycl_backend::command_profile::observe(stream, event);
            ++next_; copied_bytes_ += bytes;
            return true;
        } catch (...) {
            // An enqueue/event-ownership failure must not unwind a live DMA
            // source. A failed checked drain terminates while owners live.
            drain_noexcept();
            throw;
        }
    }
    size_t host_bytes() const { return storage_.size(); }
    uint64_t copies() const { return next_; }
    uint64_t copied_bytes() const { return copied_bytes_; }
    void check_guards() {
        wait_bound_queue();
        const auto* host = storage_.as<uint8_t>();
        for (size_t slot = 0; slot < slots; ++slot)
            for (size_t part = 0; part < 2; ++part)
                for (size_t i = part ? guard + capacity_ : 0; i < (part ? stride_ : guard); ++i)
                    if (host[slot * stride_ + i] != 0xa5)
                        throw std::runtime_error("expert staging host guard changed");
    }
private:
    static size_t checked_capacity(size_t bytes) {
        if (!bytes || bytes > maximum_blob)
            throw std::invalid_argument("expert staging finite capacity");
        return bytes;
    }
    void wait_bound_queue() {
        // Runtime::wait covers only its compute/transfer queues. Production
        // may bind another make_queue stream in this same context.
        sycl_backend::cpu_wait_profile::blocking(sycl_backend::cpu_wait_profile::direct_queue,
            [&] { queue_.wait_and_throw(); });
        runtime_->check();
    }
    void drain_noexcept() noexcept {
        try { wait_bound_queue(); } catch (...) { std::_Exit(86); }
    }
    std::shared_ptr<sycl_backend::Runtime> runtime_;
    sycl::queue queue_;
    size_t capacity_, stride_;
    sycl_backend::Allocation storage_;
    std::array<std::optional<sycl::event>, slots> pending_;
    uint64_t next_ = 0, copied_bytes_ = 0;
};
} // namespace strata::prefill::detail
