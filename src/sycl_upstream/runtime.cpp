#include "strata/sycl_upstream/runtime.hpp"
#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace strata::sycl_upstream {
namespace {
// An explicit dependent command keeps host-task dependencies in SYCL's
// scheduler. ext_oneapi_submit_barrier(deps) blocked the submitting host on
// B570 when a dependency followed a held host task; see the runtime evidence.
sycl::event fence(sycl::queue& q, const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task<class RuntimeFence>([] {});
    });
}
bool complete(const sycl::event& e) {
    return e.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
}
}
struct Runtime::Impl {
    struct Queue {
        sycl::queue q;
        bool nonblocking;
        std::optional<sycl::event> tail;
        Queue(const sycl::context& c, const sycl::device& d, bool nonblocking):
            q(c, d, [](sycl::exception_list list) { if (list.begin() != list.end()) std::rethrow_exception(*list.begin()); },
              sycl::property::queue::in_order{}), nonblocking(nonblocking) {}
    };
    sycl::device device;
    sycl::context context;
    std::mutex mutex;
    Stream next = 1;
    std::map<Stream, std::unique_ptr<Queue>> streams;
    std::vector<std::unique_ptr<Queue>> retired;
    explicit Impl(const sycl::device& d): device(d), context(d) {
        streams.emplace(0, std::make_unique<Queue>(context, device, false));
    }
    Queue& get(Stream stream) {
        const auto it = streams.find(stream);
        if (it == streams.end()) throw std::invalid_argument("invalid runtime stream");
        return *it->second;
    }
    std::vector<sycl::event> dependencies(Stream stream) {
        auto& target = get(stream);
        std::vector<sycl::event> deps;
        auto add = [&](const Queue& q) { if (q.tail) deps.push_back(*q.tail); };
        if (stream == 0) {
            for (const auto& [id, q] : streams) if (id && !q->nonblocking) add(*q);
            // Destruction retires a stream asynchronously; its preceding
            // blocking work remains ordered until it has actually completed.
            for (const auto& q : retired) if (!q->nonblocking) add(*q);
        } else if (!target.nonblocking) add(get(0));
        return deps;
    }
    sycl::event enqueue(Stream stream, const Submit& submit) {
        auto& target = get(stream);
        auto deps = dependencies(stream);
        if (!deps.empty()) target.tail = fence(target.q, deps);
        try { target.tail = submit(target.q); }
        catch (...) {
            // A callback might throw after a partial batch has been submitted.
            // Keep that work in the dependency chain before propagating failure.
            target.tail = fence(target.q);
            throw;
        }
        return *target.tail;
    }
    std::vector<sycl::event> snapshot(Stream stream) {
        auto deps = dependencies(stream);
        if (get(stream).tail) deps.push_back(*get(stream).tail);
        return deps;
    }
};
Runtime::Runtime(const sycl::device& d): impl_(std::make_unique<Impl>(d)) {}
Runtime::~Runtime() {
    try { synchronize_device(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "SYCL runtime teardown: %s\n", e.what());
        std::terminate();
    }
}
sycl::context Runtime::context() const { return impl_->context; }
sycl::device Runtime::device() const { return impl_->device; }
Runtime::Stream Runtime::create_stream(bool nonblocking) {
    std::lock_guard lock(impl_->mutex);
    // Reclaim completed retired queues at a stream-management boundary, not at
    // every kernel launch. No wait is performed for unfinished queues.
    std::erase_if(impl_->retired, [](const auto& q) { return !q->tail || complete(*q->tail); });
    const Stream id = impl_->next++;
    impl_->streams.emplace(id, std::make_unique<Impl::Queue>(impl_->context, impl_->device, nonblocking));
    return id;
}
void Runtime::destroy_stream(Stream stream) {
    if (!stream) throw std::invalid_argument("cannot destroy the legacy stream");
    std::lock_guard lock(impl_->mutex);
    (void) impl_->get(stream);
    impl_->retired.push_back(std::move(impl_->streams.at(stream)));
    impl_->streams.erase(stream);
}
sycl::event Runtime::enqueue(Stream stream, const Submit& submit) {
    std::lock_guard lock(impl_->mutex);
    return impl_->enqueue(stream, submit);
}
sycl::event Runtime::copy(Stream s, void* dst, const void* src, size_t bytes) {
    return enqueue(s, [=](sycl::queue& q) { return q.memcpy(dst, src, bytes); });
}
sycl::event Runtime::memset(Stream s, void* dst, int byte, size_t bytes) {
    return enqueue(s, [=](sycl::queue& q) { return q.memset(dst, byte, bytes); });
}
sycl::event Runtime::host_function(Stream s, std::function<void()> fn) {
    return enqueue(s, [fn = std::move(fn)](sycl::queue& q) {
        return q.submit([&](sycl::handler& h) { h.host_task([fn] { fn(); }); });
    });
}
void Runtime::record(Event& event, Stream stream) {
    std::lock_guard lock(impl_->mutex);
    if (event.context_ && *event.context_ != impl_->context)
        throw std::invalid_argument("event belongs to a different runtime context");
    event.completion_ = impl_->enqueue(stream, [](sycl::queue& q) { return fence(q); });
    event.context_ = impl_->context;
}
void Runtime::wait_event(Stream stream, const Event& event) {
    std::lock_guard lock(impl_->mutex);
    if (event.context_ && *event.context_ != impl_->context)
        throw std::invalid_argument("cross-context event wait is not implemented");
    // Copy the event value now: a later record() must not retarget this wait.
    const auto snapshot = event.completion_;
    impl_->enqueue(stream, [=](sycl::queue& q) {
        return snapshot ? fence(q, {*snapshot}) : fence(q);
    });
}
bool Runtime::query(Stream stream) {
    std::lock_guard lock(impl_->mutex);
    for (const auto& event : impl_->snapshot(stream)) if (!complete(event)) return false;
    return true;
}
bool Runtime::query(const Event& event) {
    std::lock_guard lock(impl_->mutex);
    return !event.completion_ || complete(*event.completion_);
}
void Runtime::synchronize(Stream stream) {
    std::vector<sycl::event> events;
    { std::lock_guard lock(impl_->mutex); events = impl_->snapshot(stream); }
    sycl::event::wait_and_throw(events);
}
void Runtime::synchronize(const Event& event) {
    std::optional<sycl::event> snapshot;
    { std::lock_guard lock(impl_->mutex); snapshot = event.completion_; }
    if (snapshot) snapshot->wait_and_throw();
}
void Runtime::synchronize_device() {
    std::vector<sycl::event> events;
    { std::lock_guard lock(impl_->mutex);
      for (const auto& [id, entry] : impl_->streams) if (entry->tail) events.push_back(*entry->tail);
      for (const auto& entry : impl_->retired) if (entry->tail) events.push_back(*entry->tail); }
    sycl::event::wait_and_throw(events);
}
} // namespace strata::sycl_upstream
