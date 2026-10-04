#include "strata/sycl_upstream/runtime.hpp"
#include <cstdio>
#include <algorithm>
#include <thread>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>
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
namespace graph_api = sycl::ext::oneapi::experimental;
struct Runtime::Capture {
    sycl::context context;
    graph_api::command_graph<graph_api::graph_state::modifiable> graph;
    Stream root;
    std::thread::id owner = std::this_thread::get_id();
    std::vector<Stream> streams;
    bool active = true, invalid = false;
    Capture(const sycl::context& context, const sycl::device& device, Stream root):
        context(context), graph(context, device), root(root), streams{root} {}
};
struct Runtime::Graph::State {
    sycl::context context;
    graph_api::command_graph<graph_api::graph_state::executable> executable;
    size_t nodes;
    std::optional<sycl::event> last_submission;
    bool retained = false;
    State(const sycl::context& context, const Capture& capture):
        context(context), executable(capture.graph.finalize()), nodes(capture.graph.get_nodes().size()) {}

};
size_t Runtime::Graph::node_count() const { return state_ ? state_->nodes : 0; }
size_t Runtime::GraphDefinition::node_count() const { return capture_ ? capture_->graph.get_nodes().size() : 0; }
std::vector<Runtime::NodeType> Runtime::GraphDefinition::node_types() const {
    std::vector<NodeType> types;
    if (!capture_) return types;
    const auto nodes = capture_->graph.get_nodes();
    types.reserve(nodes.size());
    for (const auto& node : nodes) {
        using Native = graph_api::node_type;
        switch (node.get_type()) {
            case Native::empty: case Native::ext_oneapi_barrier: types.push_back(NodeType::empty); break;
            case Native::kernel: types.push_back(NodeType::kernel); break;
            case Native::memcpy: types.push_back(NodeType::memcpy); break;
            case Native::memset: types.push_back(NodeType::memset); break;
            case Native::host_task: types.push_back(NodeType::host); break;
            case Native::subgraph: types.push_back(NodeType::subgraph); break;
            default: types.push_back(NodeType::unsupported); break;
        }
    }
    return types;
}
struct Runtime::Impl {
    struct Queue {
        sycl::queue q;
        bool nonblocking;
        std::optional<sycl::event> tail;
        std::optional<sycl::event> captured_tail;
        std::shared_ptr<Capture> capture;
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
    std::vector<std::shared_ptr<Graph::State>> launched_graphs;
    void reclaim_graphs() {
        std::erase_if(launched_graphs, [](const auto& state) {
            return state.use_count() == 1 && (!state->last_submission || complete(*state->last_submission));
        });
    }
    Impl(const sycl::device& d, const sycl::context& c): device(d), context(c) {
        streams.emplace(0, std::make_unique<Queue>(context, device, false));
    }
    Queue& get(Stream stream) {
        const auto it = streams.find(stream);
        if (it == streams.end()) throw std::invalid_argument("invalid runtime stream");
        return *it->second;
    }
    bool invalidate_captures(bool blocking_only) {
        bool any = false;
        for (const auto& [id, q] : streams) if (q->capture && (!blocking_only || !q->nonblocking)) {
            q->capture->invalid = true;
            any = true;
        }
        return any;
    }
    std::vector<sycl::event> dependencies(Stream stream) {
        auto& target = get(stream);
        std::vector<sycl::event> deps;
        auto add = [&](const Queue& q) {
            if (q.capture) {
                q.capture->invalid = true;
                throw std::invalid_argument("implicit legacy dependency on a capturing stream");
            }
            if (q.tail) deps.push_back(*q.tail);
        };
        if (stream == 0) {
            if (invalidate_captures(true))
                throw std::invalid_argument("implicit legacy dependency on capturing streams");
            for (const auto& [id, q] : streams) if (id && !q->nonblocking) add(*q);
            // Destruction retires a stream asynchronously; its preceding
            // blocking work remains ordered until it has actually completed.
            for (const auto& q : retired) if (!q->nonblocking) add(*q);
        } else if (!target.nonblocking) add(get(0));
        return deps;
    }
    sycl::event enqueue(Stream stream, const Submit& submit) {
        auto& target = get(stream);
        if (target.capture) {
            if (target.capture->invalid) throw std::invalid_argument("capture is invalidated");
            try { target.captured_tail = submit(target.q); }
            catch (...) { target.capture->invalid = true; throw; }
            return *target.captured_tail;
        }
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
    void require_executing(Stream stream) {
        if (get(stream).capture) {
            get(stream).capture->invalid = true;
            throw std::invalid_argument("cannot query/synchronize a capturing stream");
        }
    }
    void require_executed_event(const Event& event) {
        if (event.captured_) {
            if (auto c = event.capture_.lock()) c->invalid = true;
            throw std::invalid_argument("captured event has no external completion state; record it again after capture");
        }
    }
    void detach(const std::shared_ptr<Capture>& capture) {
        capture->graph.end_recording();
        capture->active = false;
        for (auto id : capture->streams) {
            get(id).capture.reset();
            get(id).captured_tail.reset();
        }
    }
    void abort_all() {
        for (auto& [id, queue] : streams) {
            auto capture = queue->capture;
            if (capture) detach(capture);
        }
    }
    std::vector<sycl::event> snapshot(Stream stream) {
        require_executing(stream);
        auto deps = dependencies(stream);
        if (get(stream).tail) deps.push_back(*get(stream).tail);
        return deps;
    }
};
Runtime::Runtime(const sycl::device& d): Runtime(d, sycl::context(d)) {}
Runtime::Runtime(const sycl::device& d, const sycl::context& c): impl_(std::make_unique<Impl>(d,c)) {}
Runtime::~Runtime() {
    try { prepare_teardown(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "SYCL runtime teardown: %s\n", e.what());
        std::terminate();
    }
}
void Runtime::prepare_teardown() {
    {
        std::lock_guard lock(impl_->mutex);
        impl_->abort_all();
    }
    synchronize_device();
}
sycl::context Runtime::context() const { return impl_->context; }
sycl::device Runtime::device() const { return impl_->device; }
Runtime::Stream Runtime::create_stream(bool nonblocking) {
    std::lock_guard lock(impl_->mutex);
    // Reclaim completed retired queues at a stream-management boundary, not at
    // every kernel launch. No wait is performed for unfinished queues.
    std::erase_if(impl_->retired, [](const auto& q) { return !q->tail || complete(*q->tail); });
    impl_->reclaim_graphs();
    const Stream id = impl_->next++;
    impl_->streams.emplace(id, std::make_unique<Impl::Queue>(impl_->context, impl_->device, nonblocking));
    return id;
}
void Runtime::destroy_stream(Stream stream) {
    if (!stream) throw std::invalid_argument("cannot destroy the legacy stream");
    std::lock_guard lock(impl_->mutex);
    impl_->require_executing(stream);
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
    // Captured records represent internal graph dependencies, not exported
    // completion/timing states. Upstream's stage timer also excludes capture.
    const bool timed = event.timing_ && !impl_->get(stream).capture;
    event.completion_ = impl_->enqueue(stream, [timed](sycl::queue& q) {
        return timed ? sycl::ext::oneapi::experimental::submit_profiling_tag(q) : fence(q);
    });
    event.context_ = impl_->context;
    event.capture_ = impl_->get(stream).capture;
    event.captured_ = bool(impl_->get(stream).capture);
}
Runtime::Timing Runtime::elapsed_time(const Event& start, const Event& end) {
    std::lock_guard lock(impl_->mutex);
    if (!start.timing_ || !end.timing_ || !start.completion_ || !end.completion_ ||
        start.captured_ || end.captured_ || start.context_ != impl_->context ||
        end.context_ != impl_->context)
        return {TimingStatus::invalid};
    if (!complete(*start.completion_) || !complete(*end.completion_))
        return {TimingStatus::not_ready};
    const auto a = start.completion_->get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto b = end.completion_->get_profiling_info<sycl::info::event_profiling::command_end>();
    const double ns = b >= a ? double(b - a) : -double(a - b);
    return {TimingStatus::ready, float(ns / 1e6)};
}
Runtime::Event Runtime::snapshot_event(const Event& event) {
    std::lock_guard lock(impl_->mutex);
    if (event.captured_) throw std::invalid_argument("cross-runtime captured event wait is not implemented");
    return event;
}
void Runtime::wait_event(Stream stream, const Event& event) {
    std::lock_guard lock(impl_->mutex);
    if (event.context_ && *event.context_ != impl_->context)
        throw std::invalid_argument("cross-context event wait is not implemented");
    auto& target = impl_->get(stream);
    if (event.captured_) {
        auto capture = event.capture_.lock();
        if (!capture || !capture->active || capture->invalid) {
            if (target.capture) target.capture->invalid = true;
            throw std::invalid_argument("event is not in a live valid capture");
        }
        if (!stream || (target.capture && target.capture != capture)) {
            capture->invalid = true;
            if (target.capture) target.capture->invalid = true;
            throw std::invalid_argument("cannot merge captures or capture the legacy stream");
        }
        if (!target.capture) {
            capture->streams.push_back(stream);
            try { capture->graph.begin_recording(target.q); }
            catch (...) { capture->streams.pop_back(); capture->invalid = true; throw; }
            target.capture = capture;
            target.captured_tail.reset();
        }
    } else if (target.capture && event.completion_) {
        target.capture->invalid = true;
        throw std::invalid_argument("external event waits inside capture are not implemented");
    }
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
    impl_->require_executed_event(event);
    return !event.completion_ || complete(*event.completion_);
}
void Runtime::synchronize_native(sycl::event event) {
    // On the installed SYCL runtime, waiting on an unenqueued command behind
    // a host task holds Scheduler's read lock while it waits. A concurrent
    // addCG() needs its write lock, so a CPU event waiter can prevent the next
    // DMA/kernel from being submitted. Nonblocking status queries avoid this
    // lock retention; the final completed-event wait preserves async errors.
    while (!complete(event)) std::this_thread::yield();
    event.wait_and_throw();
}
void Runtime::synchronize(Stream stream) {
    std::vector<sycl::event> events;
    { std::lock_guard lock(impl_->mutex); events = impl_->snapshot(stream); }
    for (const auto& event : events) synchronize_native(event);
}
void Runtime::synchronize(const Event& event) {
    std::optional<sycl::event> snapshot;
    { std::lock_guard lock(impl_->mutex); impl_->require_executed_event(event); snapshot = event.completion_; }
    if (snapshot) synchronize_native(*snapshot);
}
void Runtime::synchronize_device() {
    std::vector<sycl::event> events;
    { std::lock_guard lock(impl_->mutex);
      if (impl_->invalidate_captures(false))
          throw std::invalid_argument("cannot synchronize a device with active captures");
      for (const auto& [id, entry] : impl_->streams) if (entry->tail) events.push_back(*entry->tail);
      for (const auto& entry : impl_->retired) if (entry->tail) events.push_back(*entry->tail); }
    for (const auto& event : events) synchronize_native(event);
    std::lock_guard lock(impl_->mutex);
    impl_->reclaim_graphs();
}
void Runtime::check_memory_operation() {
    std::lock_guard lock(impl_->mutex);
    bool invalid = false;
    for (const auto& [id, q] : impl_->streams)
        if (q->capture && q->capture->owner == std::this_thread::get_id()) {
            q->capture->invalid = true;
            invalid = true;
        }
    if (invalid) throw std::invalid_argument("memory operation on a thread with an active capture");
}
void Runtime::begin_capture(Stream stream) {
    std::lock_guard lock(impl_->mutex);
    if (!stream) throw std::invalid_argument("legacy stream capture is not supported");
    auto& target = impl_->get(stream);
    if (target.capture) throw std::invalid_argument("stream already capturing");
    if (!impl_->device.has(sycl::aspect::ext_oneapi_limited_graph))
        throw std::invalid_argument("device has no SYCL graph support");
    impl_->reclaim_graphs();
    auto capture = std::make_shared<Capture>(impl_->context, impl_->device, stream);
    capture->graph.begin_recording(target.q);
    target.capture = std::move(capture);
    target.captured_tail.reset();
}
Runtime::CaptureStatus Runtime::capture_status(Stream stream) {
    std::lock_guard lock(impl_->mutex);
    auto capture = impl_->get(stream).capture;
    return !capture ? CaptureStatus::none : capture->invalid ? CaptureStatus::invalidated : CaptureStatus::active;
}
bool Runtime::capturing(Stream stream) { return capture_status(stream) != CaptureStatus::none; }
void Runtime::abort_capture(Stream stream) {
    std::lock_guard lock(impl_->mutex);
    auto capture = impl_->get(stream).capture;
    if (!capture || capture->root != stream || capture->owner != std::this_thread::get_id())
        throw std::invalid_argument("capture must end on its origin stream and thread");
    impl_->detach(capture);
}
Runtime::GraphDefinition Runtime::end_capture_definition(Stream stream) {
    std::lock_guard lock(impl_->mutex);
    auto& target = impl_->get(stream);
    auto capture = target.capture;
    if (!capture || capture->root != stream || capture->owner != std::this_thread::get_id())
        throw std::invalid_argument("capture must end on its origin stream and thread");
    bool joined = true;
    try {
        // A finite DAG is fully joined iff its only leaf is the origin's
        // final node. Inspect actual native graph edges in linear time; a
        // full-window capture can contain many thousands of nodes.
        const auto nodes = capture->graph.get_nodes();
        if (!nodes.empty()) {
            if (!target.captured_tail) joined = false;
            else {
                const auto last = graph_api::node::get_node_from_event(*target.captured_tail);
                for (const auto& node : nodes)
                    if (node.get_successors().empty() && node != last) joined = false;
            }
        }
    } catch (...) { impl_->detach(capture); throw; }
    impl_->detach(capture);
    if (capture->invalid) throw std::invalid_argument("capture was invalidated");
    if (!joined) throw std::invalid_argument("capture has an unjoined stream branch");
    return GraphDefinition(std::move(capture));
}
Runtime::Graph Runtime::instantiate(const GraphDefinition& definition) {
    std::lock_guard lock(impl_->mutex);
    if (!definition.capture_ || definition.capture_->context != impl_->context)
        throw std::invalid_argument("invalid graph definition or context");
    return Graph(std::make_shared<Graph::State>(impl_->context, *definition.capture_));
}
Runtime::Graph Runtime::end_capture(Stream stream) {
    auto definition = end_capture_definition(stream);
    if (!definition.node_count()) throw std::invalid_argument("capture recorded zero nodes");
    return instantiate(definition);
}
sycl::event Runtime::launch(Graph& graph, Stream stream) {
    std::lock_guard lock(impl_->mutex);
    if (!graph.state_ || graph.state_->context != impl_->context)
        throw std::invalid_argument("invalid graph or graph context");
    auto& target = impl_->get(stream);
    if (target.capture) {
        target.capture->invalid = true;
        throw std::invalid_argument("nested graph launches during capture are not implemented");
    }
    auto& state = *graph.state_;
    // Retain before submission, including allocation-failure paths. Dropping
    // the public Graph handle must not wait for a pending launch. Reclaim at
    // management boundaries after completion, never on the hot replay path.
    if (!state.retained) {
        impl_->launched_graphs.push_back(graph.state_);
        state.retained = true;
    }
    const auto previous = state.last_submission;
    auto event = impl_->enqueue(stream, [&](sycl::queue& q) {
        return q.submit([&](sycl::handler& h) {
            // Replays of one executable graph cannot overlap, even when they
            // are launched on different streams.
            if (previous) h.depends_on(*previous);
            h.ext_oneapi_graph(state.executable);
        });
    });
    state.last_submission = event;
    return event;
}
sycl::event Runtime::upload(Graph& graph, Stream stream) {
    std::lock_guard lock(impl_->mutex);
    if (!graph.state_ || graph.state_->context != impl_->context)
        throw std::invalid_argument("invalid graph or graph context");
    if (impl_->get(stream).capture) {
        impl_->get(stream).capture->invalid = true;
        throw std::invalid_argument("graph upload during capture is not implemented");
    }
    auto& state = *graph.state_;
    if (!state.retained) {
        impl_->launched_graphs.push_back(graph.state_);
        state.retained = true;
    }
    // Native command buffers and graph kernels were prepared by finalize()
    // during instantiate(), before any queued commands can execute. Keep the
    // upload's stream and executable ordering without replaying the graph.
    const auto previous = state.last_submission;
    auto event = impl_->enqueue(stream, [previous](sycl::queue& q) {
        return previous ? fence(q, {*previous}) : fence(q);
    });
    state.last_submission = event;
    return event;
}
} // namespace strata::sycl_upstream
