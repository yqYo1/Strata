#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace strata {
// Keep the actual asynchronous operation alive until its source can be reused.
// Only CPU threads touch this bookkeeping. Query does not submit a host task or
// a GPU marker; a failed query must never acknowledge the pending operation.
template <class Event> struct EventCompletionState {
    std::mutex mutex;
    std::optional<Event> event;
    uint64_t generation = 0;
    std::atomic<uint64_t> completed{0};
};
template <class Event>
using EventCompletion = std::shared_ptr<EventCompletionState<Event>>;
template <class Event> EventCompletion<Event> make_event_completion() {
    return std::make_shared<EventCompletionState<Event>>();
}
template <class Event>
void record_event_completion(const EventCompletion<Event>& state, uint64_t value, Event event) {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (value <= state->generation)
        throw std::logic_error("DMA completion generations must increase");
    // The caller has already acquired the previous completion before filling
    // this buffer. Retaining the event preserves its runtime reference lifetime.
    state->event = std::move(event);
    state->generation = value;
}
template <class Event, class Query>
bool event_completed(const EventCompletion<Event>& state, uint64_t value, Query query) {
    if (state->completed.load(std::memory_order_acquire) >= value) return true;
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->completed.load(std::memory_order_acquire) >= value) return true;
    if (!state->event || state->generation < value || !query(*state->event)) return false;
    // Serialize both publication and status queries: an older query cannot
    // acknowledge a newer operation or regress the cached completion value.
    state->completed.store(state->generation, std::memory_order_release);
    return true;
}
} // namespace strata
