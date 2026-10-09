#pragma once
// CPU-only test double. It deliberately does not implement or validate SYCL.
#include <cstddef>
#include <cstring>
#include <stdexcept>
namespace sycl {
struct queue;
struct event { queue* owner; void wait_and_throw(); };
struct queue {
    bool initial_error=false, event_error=false, submit_error=false, drain_error=false;
    unsigned initial_waits=0, copies=0, event_waits=0, drains=0, completions=0;
    void* destination=nullptr; const void* source=nullptr; size_t bytes=0; bool pending=false;
    void wait_and_throw() { ++initial_waits; if(initial_error) throw std::runtime_error("initial-error"); }
    event memcpy(void* dst, const void* src, size_t n) {
        if(pending) throw std::logic_error("fake pending copy overwritten");
        ++copies; destination=dst; source=src; bytes=n; pending=true;
        if(submit_error) throw std::runtime_error("submit-error");
        return {this};
    }
    void complete() { if(pending) { std::memcpy(destination,source,bytes); pending=false; ++completions; } }
    void wait() { ++drains; if(drain_error) throw std::runtime_error("drain-error"); complete(); }
};
inline void event::wait_and_throw() {
    ++owner->event_waits;
    if(owner->event_error) throw std::runtime_error("event-error");
    owner->complete();
}
}
