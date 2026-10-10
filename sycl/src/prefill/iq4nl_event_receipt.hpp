// Qualification-only receipt of the exact existing submit event; no public API.
#pragma once
#ifdef STRATA_SYCL_PREFILL_IQ4NL_EVENT_RECEIPT
#include <sycl/sycl.hpp>
#include "strata/sycl_queue.hpp"
#include <cstdint>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
namespace strata::kernels::detail {
enum class Iq4nlArm : unsigned { generic, private_down };
struct Iq4nlReceipt {
    std::optional<sycl::event> event;
    Iq4nlArm arm{};
    int type{};
    int64_t n{};
    sycl::queue* queue{};
    const void* src{};
    uint16_t* dst{};
    uint64_t scope_id{}, serial{};
};
struct Iq4nlReceiptSink {
    Iq4nlReceipt* slots;
    size_t capacity, count=0;
    uint64_t scope_id, serial=0;
    bool invalid=false;
    void retain(const sycl::event& event, Iq4nlArm arm, int type, int64_t n,
                void* stream, const void* src, uint16_t* dst) noexcept {
        const uint64_t call=++serial;
        if(count==capacity) {invalid=true;return;}
        auto& slot=slots[count++];
        try {
            slot.arm=arm;slot.type=type;slot.n=n;slot.queue=strata::q_of(stream);
            slot.src=src;slot.dst=dst;slot.scope_id=scope_id;slot.serial=call;
            slot.event.emplace(event);
        } catch(...) {invalid=true;}
    }
};
// Defined once in the kernel TU: fixture and library share this TLS routing.
Iq4nlReceiptSink*& iq4nl_receipt_sink() noexcept;
class Iq4nlReceiptScope {
public:
    explicit Iq4nlReceiptScope(Iq4nlReceiptSink& sink) {
        if(iq4nl_receipt_sink()) throw std::logic_error("nested IQ4NL receipt scope");
        if(!sink.slots||!sink.capacity||sink.count||sink.serial||sink.invalid)
            throw std::logic_error("IQ4NL receipt sink must be fresh");
        for(size_t i=0;i<sink.capacity;++i)
            if(sink.slots[i].event.has_value())
                throw std::logic_error("IQ4NL receipt slots must be empty");
        iq4nl_receipt_sink()=&sink;
    }
    ~Iq4nlReceiptScope() {iq4nl_receipt_sink()=nullptr;}
    Iq4nlReceiptScope(const Iq4nlReceiptScope&)=delete;
    Iq4nlReceiptScope& operator=(const Iq4nlReceiptScope&)=delete;
};
// The inactive path does not bind/retain an event or resolve the queue again.
// Submit exceptions still reach the fixture's existing fail-stop boundary.
template<class Submit>
void iq4nl_receipt_submit(Submit&& submit, Iq4nlArm arm, int type, int64_t n,
                          void* stream, const void* src, uint16_t* dst) {
    auto* sink=iq4nl_receipt_sink();
    if(!sink) {submit();return;}
    auto event=submit();
    sink->retain(event,arm,type,n,stream,src,dst);
}
} // namespace strata::kernels::detail
#endif
