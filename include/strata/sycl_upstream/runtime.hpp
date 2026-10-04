#pragma once
#include <sycl/sycl.hpp>
#include <functional>
#include <memory>
#include <optional>

namespace strata::sycl_upstream {
// Per-device execution domain. Stream 0 has CUDA legacy-default ordering;
// named streams are blocking unless explicitly created nonblocking.
// This is the runtime core, not a drop-in cuda_runtime.h implementation.
class Runtime {
    struct Capture;
public:
    using Stream = uint64_t;
    using Submit = std::function<sycl::event(sycl::queue&)>;
    class Event {
        friend class Runtime;
        std::optional<sycl::event> completion_;
        std::optional<sycl::context> context_;
        std::weak_ptr<Capture> capture_;
        bool captured_ = false;
    };
    class Graph {
        friend class Runtime;
        struct State;
        std::shared_ptr<State> state_;
        explicit Graph(std::shared_ptr<State> state): state_(std::move(state)) {}
    public:
        Graph() = default;
        Graph(Graph&&) = default;
        Graph& operator=(Graph&&) = default;
        Graph(const Graph&) = delete;
        Graph& operator=(const Graph&) = delete;
        size_t node_count() const;
        // Releasing Graph returns without waiting for pending replays. The
        // runtime retains the executable until completion; buffers referenced
        // by recorded commands remain caller-owned.
    };
    explicit Runtime(const sycl::device&);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    sycl::context context() const;
    sycl::device device() const;
    Stream create_stream(bool nonblocking = false);
    // Returns before pending work completes; the domain retains its queue.
    void destroy_stream(Stream);
    // Callback must only submit work to this queue, return its last event, and
    // never reenter this Runtime. Do not save/use the queue outside the callback.
    sycl::event enqueue(Stream, const Submit&);
    sycl::event copy(Stream, void* dst, const void* src, size_t bytes);
    sycl::event memset(Stream, void* dst, int byte, size_t bytes);
    sycl::event host_function(Stream, std::function<void()>);
    void record(Event&, Stream);
    void wait_event(Stream, const Event&);
    bool query(Stream);
    bool query(const Event&);
    void synchronize(Stream);
    void synchronize(const Event&);
    void synchronize_device();
    // Allocation/registration are unsafe on the thread owning an active
    // thread-local capture. Called by the memory component before mutation.
    void check_memory_operation();
    // Thread-local capture: end on the originating stream and host thread.
    // Waiting on a captured event enrolls another stream; all branches must
    // join the origin before end_capture. Empty or invalid captures fail.
    void begin_capture(Stream);
    bool capturing(Stream);
    Graph end_capture(Stream);
    void abort_capture(Stream);
    sycl::event launch(Graph&, Stream);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::sycl_upstream
