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
        bool timing_;
        std::optional<sycl::event> completion_;
        std::optional<sycl::context> context_;
        std::weak_ptr<Capture> capture_;
        bool captured_ = false;
    public:
        explicit Event(bool timing = false): timing_(timing) {}
    };
    class GraphDefinition {
        friend class Runtime;
        std::shared_ptr<Capture> capture_;
        explicit GraphDefinition(std::shared_ptr<Capture> capture): capture_(std::move(capture)) {}
    public:
        GraphDefinition() = default;
        GraphDefinition(GraphDefinition&&) = default;
        GraphDefinition& operator=(GraphDefinition&&) = default;
        GraphDefinition(const GraphDefinition&) = delete;
        GraphDefinition& operator=(const GraphDefinition&) = delete;
        size_t node_count() const;
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
    Runtime(const sycl::device&, const sycl::context&);
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
    enum class TimingStatus { ready, not_ready, invalid };
    struct Timing { TimingStatus status; float milliseconds = 0; };
    // Nonblocking. Requires two recorded, uncaptured timing-enabled events.
    // Timestamp queries are made only after both completion states are ready.
    Timing elapsed_time(const Event& start, const Event& end);
    void wait_event(Stream, const Event&);
    // Copy an uncaptured event under its owner's mutex for a wait in another
    // runtime sharing this context. Captured dependencies stay in their owner.
    Event snapshot_event(const Event&);
    bool query(Stream);
    bool query(const Event&);
    void synchronize(Stream);
    void synchronize(const Event&);
    void synchronize_device();
    // Teardown only, with no concurrent callers: discard unfinished capture
    // definitions, then wait for previously submitted work. Owners of memory
    // and pageable-copy staging call this before destroying that storage.
    void prepare_teardown();
    // Allocation/registration are unsafe on the thread owning an active
    // thread-local capture. Called by the memory component before mutation.
    void check_memory_operation();
    // Thread-local capture: end on the originating stream and host thread.
    // Waiting on a captured event enrolls another stream; all branches must
    // join the origin before end_capture. Invalid captures fail.
    void begin_capture(Stream);
    enum class CaptureStatus { none, active, invalidated };
    CaptureStatus capture_status(Stream);
    bool capturing(Stream);
    // A definition can instantiate independent executable graphs. The existing
    // end_capture convenience method performs both operations and rejects an
    // empty capture; definitions expose zero nodes to the host caller's check.
    GraphDefinition end_capture_definition(Stream);
    Graph instantiate(const GraphDefinition&);
    Graph end_capture(Stream);
    void abort_capture(Stream);
    sycl::event launch(Graph&, Stream);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::sycl_upstream
