#include "strata/sycl/compat/cuda_runtime.h"
#include "strata/sycl/launch.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <vector>

namespace strata::sycl_backend::compat {
namespace ex = sycl::ext::oneapi::experimental;
using ModGraph = ex::command_graph<ex::graph_state::modifiable>;
using ExecGraph = ex::command_graph<ex::graph_state::executable>;
struct EventState {
  std::mutex mutex;
  std::optional<sycl::event> event;
  bool timing = true, captured = false;
};
struct Event {
  std::shared_ptr<EventState> state = std::make_shared<EventState>();
};
struct Node {
  ex::node value;
};
struct Graph {
  ModGraph value;
  std::vector<Node> nodes;
  std::vector<std::weak_ptr<EventState>> events;
  Graph(const sycl::context &c, const sycl::device &d) : value(c, d) {}
};
struct GraphExec {
  ExecGraph value;
  std::vector<std::weak_ptr<EventState>> events;
  std::mutex mutex;
  std::optional<sycl::event> last;
  explicit GraphExec(Graph &g) : value(g.value.finalize()), events(g.events) {}
};
namespace {
thread_local cudaError_t last_error = cudaSuccess, detail_code = cudaSuccess;
thread_local std::string detail;
std::mutex registry_mutex;
struct Stream {
  std::unique_ptr<sycl::queue> queue;
  std::unique_ptr<Graph> recording;
};
std::map<sycl::queue *, Stream> streams;
struct Memory {
  std::unique_ptr<Allocation> allocation;
  MemoryKind kind;
};
std::map<void *, Memory> allocations;
cudaError_t fail(cudaError_t code, std::string message) {
  last_error = detail_code = code;
  detail = std::move(message);
  return code;
}
template <class F> cudaError_t attempt(F body) {
  try {
    runtime_for()->check();
    body();
    return cudaSuccess;
  } catch (const std::bad_alloc &e) {
    return fail(cudaErrorMemoryAllocation, e.what());
  } catch (const std::invalid_argument &e) {
    return fail(cudaErrorInvalidValue, e.what());
  } catch (const std::exception &e) {
    return fail(cudaErrorUnknown, e.what());
  }
}
void require(bool value, const char *what) {
  if (!value)
    throw std::invalid_argument(what);
}
sycl::queue &queue(cudaStream_t stream) {
  auto &q = queue_for(stream);
  require(q.get_context() == runtime_for()->context(),
          "Strata SYCL engine currently uses runtime device 0 context");
  return q;
}
void wait_all() {
  std::vector<sycl::queue> pending;
  {
    std::lock_guard lock(registry_mutex);
    for (auto &[p, s] : streams)
      pending.push_back(*p);
  }
  std::exception_ptr error;
  for (auto &q : pending)
    try {
      q.wait_and_throw();
    } catch (...) {
      if (!error)
        error = std::current_exception();
    }
  try {
    runtime_for()->wait();
  } catch (...) {
    if (!error)
      error = std::current_exception();
  }
  if (error)
    std::rethrow_exception(error);
}
std::optional<sycl::event> event_of(cudaEvent_t event) {
  require(event, "null SYCL event");
  std::lock_guard lock(event->state->mutex);
  return event->state->event;
}
cudaError_t allocate(void **out, size_t bytes, MemoryKind kind) {
  if (out)
    *out = nullptr;
  return attempt([&] {
    require(out && bytes,
            "allocation requires an output pointer and positive size");
    auto a = std::make_unique<Allocation>(runtime_for(), bytes, kind);
    void *p = a->data();
    std::lock_guard lock(registry_mutex);
    allocations.emplace(p, Memory{std::move(a), kind});
    *out = p;
  });
}
cudaError_t release(void *ptr, MemoryKind kind) {
  if (!ptr)
    return cudaSuccess;
  return attempt([&] {
    wait_all();
    std::unique_ptr<Allocation> allocation;
    {
      std::lock_guard lock(registry_mutex);
      auto it = allocations.find(ptr);
      require(
          it != allocations.end() && it->second.kind == kind,
          "free requires the original pointer and matching allocation kind");
      allocation = std::move(it->second.allocation);
      allocations.erase(it);
    }
  });
}
} // namespace
} // namespace strata::sycl_backend::compat
using namespace strata::sycl_backend;
using namespace strata::sycl_backend::compat;
cudaError_t cudaGetDeviceCount(int *out) {
  return attempt([&] {
    require(out, "null device count");
    *out = int(Runtime::devices().size());
  });
}
cudaError_t cudaGetDevice(int *out) {
  return attempt([&] {
    require(out, "null device ordinal");
    *out = 0;
  });
}
cudaError_t cudaSetDevice(int id) {
  if (id != 0)
    return fail(
        cudaErrorNotSupported,
        "the SYCL engine adapter currently supports one GPU (ordinal 0)");
  return attempt([] { (void)runtime_for(); });
}
cudaError_t cudaInitDevice(int id, unsigned, unsigned) {
  return cudaSetDevice(id);
}
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *out, int id) {
  return attempt([&] {
    require(out, "null device properties");
    const auto d = runtime_for(id)->device();
    *out = {};
    const auto name = d.get_info<sycl::info::device::name>();
    std::strncpy(out->name, name.c_str(), sizeof(out->name) - 1);
    out->totalGlobalMem = d.get_info<sycl::info::device::global_mem_size>();
    out->multiProcessorCount =
        d.get_info<sycl::info::device::max_compute_units>();
    // No CUDA compute capability is invented for an Intel device.
  });
}
cudaError_t cudaDeviceGetAttribute(int *out, cudaDeviceAttr attr, int id) {
  return attempt([&] {
    require(out, "null device attribute");
    const auto d = runtime_for(id)->device();
    switch (attr) {
    case cudaDevAttrClockRate:
      *out = int(d.get_info<sycl::info::device::max_clock_frequency>()) * 1000;
      break;
    case cudaDevAttrMultiProcessorCount:
      *out = d.get_info<sycl::info::device::max_compute_units>();
      break;
    case cudaDevAttrMaxSharedMemoryPerBlockOptin:
      *out = int(d.get_info<sycl::info::device::local_mem_size>());
      break;
    default:
      throw std::invalid_argument("unknown device attribute");
    }
  });
}
cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
  return attempt([&] {
    require(free && total, "null memory query");
    const auto d = runtime_for()->device();
    require(d.has(sycl::aspect::ext_intel_free_memory),
            "SYCL memory planning needs the free-memory query");
    *free = d.get_info<sycl::ext::intel::info::device::free_memory>();
    *total = d.get_info<sycl::info::device::global_mem_size>();
  });
}
cudaError_t cudaMalloc(void **out, size_t bytes) {
  return allocate(out, bytes, MemoryKind::Device);
}
cudaError_t cudaHostAlloc(void **out, size_t bytes, unsigned flags) {
  if (flags & ~(cudaHostAllocPortable | cudaHostAllocMapped))
    return fail(cudaErrorInvalidValue, "unsupported host allocation flags");
  return allocate(out, bytes, MemoryKind::Host);
}
cudaError_t cudaMallocHost(void **out, size_t bytes) {
  return cudaHostAlloc(out, bytes, 0);
}
cudaError_t cudaFree(void *p) { return release(p, MemoryKind::Device); }
cudaError_t cudaFreeHost(void *p) { return release(p, MemoryKind::Host); }
cudaError_t cudaHostRegister(void *, size_t, unsigned) {
  return fail(cudaErrorNotSupported, "SYCL does not register arbitrary host "
                                     "mappings; use bounded host-USM staging");
}
cudaError_t cudaHostUnregister(void *) {
  return fail(cudaErrorNotSupported,
              "arbitrary host registration is unavailable in SYCL");
}
cudaError_t cudaHostGetDevicePointer(void **out, void *ptr, unsigned flags) {
  return attempt([&] {
    require(out && ptr && flags == 0, "invalid host pointer alias request");
    require(sycl::get_pointer_type(ptr, runtime_for()->context()) ==
                sycl::usm::alloc::host,
            "only host USM has a mapped SYCL alias");
    *out = ptr;
  });
}
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t bytes,
                            cudaMemcpyKind, cudaStream_t st) {
  return attempt([&] {
    require(!bytes || (dst && src), "null copy pointer");
    if (bytes)
      queue(st).memcpy(dst, src, bytes);
  });
}
cudaError_t cudaMemcpy(void *dst, const void *src, size_t bytes,
                       cudaMemcpyKind) {
  return attempt([&] {
    require(!bytes || (dst && src), "null copy pointer");
    wait_all();
    if (bytes)
      runtime_for()->wait(queue(nullptr).memcpy(dst, src, bytes));
  });
}
cudaError_t cudaMemcpy2DAsync(void *dst, size_t dp, const void *src, size_t sp,
                              size_t width, size_t height, cudaMemcpyKind,
                              cudaStream_t st) {
  return attempt([&] {
    require(
        width <= dp && width <= sp &&
            (!height || (dp <= SIZE_MAX / height && sp <= SIZE_MAX / height)),
        "invalid 2D copy extents");
    require(!width || !height || (dst && src), "null 2D copy pointer");
    auto &q = queue(st);
    if (width)
      for (size_t i = 0; i < height; ++i)
        q.memcpy(static_cast<uint8_t *>(dst) + i * dp,
                 static_cast<const uint8_t *>(src) + i * sp, width);
  });
}
cudaError_t cudaMemsetAsync(void *dst, int value, size_t bytes,
                            cudaStream_t st) {
  return attempt([&] {
    require(!bytes || dst, "null memset pointer");
    if (bytes)
      queue(st).memset(dst, value, bytes);
  });
}
cudaError_t cudaMemset(void *dst, int value, size_t bytes) {
  return attempt([&] {
    require(!bytes || dst, "null memset pointer");
    wait_all();
    if (bytes)
      runtime_for()->wait(queue(nullptr).memset(dst, value, bytes));
  });
}
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *out, unsigned flags) {
  return attempt([&] {
    require(out && !(flags & ~cudaStreamNonBlocking),
            "invalid stream creation arguments");
    auto q = std::make_unique<sycl::queue>(runtime_for()->make_queue(true));
    auto ptr = q.get();
    std::lock_guard lock(registry_mutex);
    streams.emplace(ptr, Stream{std::move(q), {}});
    *out = ptr;
  });
}
cudaError_t cudaStreamCreate(cudaStream_t *out) {
  return cudaStreamCreateWithFlags(out, 0);
}
cudaError_t cudaStreamDestroy(cudaStream_t st) {
  if (!st)
    return cudaSuccess;
  return attempt([&] {
    queue(st).wait_and_throw();
    std::lock_guard lock(registry_mutex);
    auto it = streams.find(st);
    require(it != streams.end() && !it->second.recording,
            "destroy requires an owned non-recording stream");
    streams.erase(it);
  });
}
cudaError_t cudaStreamSynchronize(cudaStream_t st) {
  return attempt([&] {
    queue(st).wait_and_throw();
    runtime_for()->check();
  });
}
cudaError_t cudaStreamQuery(cudaStream_t st) {
  bool ready = false;
  const auto e = attempt([&] {
    ready = queue(st).ext_oneapi_empty();
    if (ready) {
      queue(st).throw_asynchronous();
      runtime_for()->check();
    }
  });
  return e == cudaSuccess && !ready ? cudaErrorNotReady : e;
}
cudaError_t cudaDeviceSynchronize() {
  return attempt([] { wait_all(); });
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *out, unsigned flags) {
  return attempt([&] {
    require(out && !(flags & ~cudaEventDisableTiming),
            "invalid event creation arguments");
    auto e = std::make_unique<Event>();
    e->state->timing = !(flags & cudaEventDisableTiming);
    *out = e.release();
  });
}
cudaError_t cudaEventCreate(cudaEvent_t *out) {
  return cudaEventCreateWithFlags(out, 0);
}
cudaError_t cudaEventDestroy(cudaEvent_t e) {
  delete e;
  return cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t st) {
  return attempt([&] {
    require(e, "null event");
    auto &q = queue(st);
    const auto event = q.ext_oneapi_submit_barrier();
    std::lock_guard lock(registry_mutex);
    auto it = streams.find(&q);
    const bool captured = it != streams.end() && it->second.recording;
    {
      std::lock_guard elock(e->state->mutex);
      e->state->event = event;
      e->state->captured = captured;
    }
    if (captured)
      it->second.recording->events.push_back(e->state);
  });
}
cudaError_t cudaEventQuery(cudaEvent_t e) {
  bool ready = true;
  const auto result = attempt([&] {
    const auto event = event_of(e);
    if (event) {
      ready = event->get_info<sycl::info::event::command_execution_status>() ==
              sycl::info::event_command_status::complete;
      if (ready)
        runtime_for()->wait(*event);
    }
  });
  return result == cudaSuccess && !ready ? cudaErrorNotReady : result;
}
cudaError_t cudaEventSynchronize(cudaEvent_t e) {
  return attempt([&] {
    if (const auto event = event_of(e))
      runtime_for()->wait(*event);
  });
}
cudaError_t cudaStreamWaitEvent(cudaStream_t st, cudaEvent_t e,
                                unsigned flags) {
  return attempt([&] {
    require(flags == 0, "invalid event wait flags");
    if (const auto event = event_of(e))
      queue(st).ext_oneapi_submit_barrier({*event});
  });
}
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t begin,
                                 cudaEvent_t end) {
  return attempt([&] {
    require(ms && begin && end, "invalid elapsed-time arguments");
    require(
        begin->state->timing && end->state->timing && !begin->state->captured &&
            !end->state->captured,
        "elapsed time is unavailable for disabled or graph-internal events");
    const auto a = event_of(begin), b = event_of(end);
    require(a && b, "unrecorded profiling event");
    runtime_for()->wait(*b);
    const auto
        ta = a->get_profiling_info<sycl::info::event_profiling::command_end>(),
        tb = b->get_profiling_info<sycl::info::event_profiling::command_end>();
    require(tb >= ta, "profiling event order is reversed");
    *ms = float(double(tb - ta) * 1e-6);
  });
}
cudaError_t cudaLaunchHostFunc(cudaStream_t st, void (*fn)(void *),
                               void *data) {
  return attempt([&] {
    require(fn, "null host callback");
    queue(st).submit([&](sycl::handler &h) { h.host_task([=] { fn(data); }); });
  });
}
cudaError_t cudaStreamBeginCapture(cudaStream_t st, cudaStreamCaptureMode) {
  return attempt([&] {
    require(st, "capture requires an explicit stream");
    auto &q = queue(st);
    std::lock_guard lock(registry_mutex);
    auto it = streams.find(st);
    require(it != streams.end() && !it->second.recording,
            "capture requires an owned idle stream");
    auto graph = std::make_unique<Graph>(q.get_context(), q.get_device());
    graph->value.begin_recording(q);
    it->second.recording = std::move(graph);
  });
}
cudaError_t cudaStreamEndCapture(cudaStream_t st, cudaGraph_t *out) {
  return attempt([&] {
    require(st && out, "invalid end-capture arguments");
    std::lock_guard lock(registry_mutex);
    auto it = streams.find(st);
    require(it != streams.end() && it->second.recording,
            "stream is not recording");
    auto &g = it->second.recording;
    g->value.end_recording(*st);
    for (auto n : g->value.get_nodes())
      g->nodes.push_back(Node{n});
    *out = g.release();
  });
}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *out, cudaGraph_t graph,
                                 unsigned long long flags) {
  return attempt([&] {
    require(out && graph && flags == 0, "invalid graph instantiation");
    *out = new GraphExec(*graph);
  });
}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *out, cudaGraph_t graph,
                                 cudaGraphNode_t *, char *, size_t) {
  return cudaGraphInstantiate(out, graph, 0ull);
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t st) {
  return attempt([&] {
    require(graph, "null executable graph");
    auto &q = queue(st);
    std::lock_guard lock(graph->mutex);
    auto event = q.submit([&](sycl::handler &h) {
      if (graph->last)
        h.depends_on(*graph->last);
      h.ext_oneapi_graph(graph->value);
    });
    graph->last = event;
    for (auto &weak : graph->events)
      if (auto state = weak.lock()) {
        std::lock_guard elock(state->mutex);
        state->event = event;
      }
  });
}
cudaError_t cudaGraphUpload(cudaGraphExec_t graph, cudaStream_t st) {
  return attempt([&] {
    require(graph, "null executable graph");
    (void)queue(st); /* finalization prepared the executable graph */
  });
}
cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  delete graph;
  return cudaSuccess;
}
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graph) {
  if (!graph)
    return cudaSuccess;
  const auto result = attempt([&] {
    if (graph->last)
      runtime_for()->wait(*graph->last);
  });
  delete graph;
  return result;
}
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes,
                              size_t *count) {
  return attempt([&] {
    require(graph && count, "invalid graph node query");
    if (nodes)
      for (size_t i = 0; i < std::min(*count, graph->nodes.size()); ++i)
        nodes[i] = &graph->nodes[i];
    *count = graph->nodes.size();
  });
}
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType *out) {
  return attempt([&] {
    require(node && out, "invalid graph node type query");
    switch (node->value.get_type()) {
    case ex::node_type::kernel:
      *out = cudaGraphNodeTypeKernel;
      break;
    case ex::node_type::memcpy:
      *out = cudaGraphNodeTypeMemcpy;
      break;
    case ex::node_type::memset:
    case ex::node_type::memfill:
      *out = cudaGraphNodeTypeMemset;
      break;
    case ex::node_type::host_task:
      *out = cudaGraphNodeTypeHost;
      break;
    default:
      *out = cudaGraphNodeTypeEmpty;
      break;
    }
  });
}
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t,
                                         cudaKernelNodeParams *) {
  return fail(cudaErrorNotSupported,
              "SYCL graph kernel parameter introspection is unavailable");
}
cudaError_t cudaGetLastError() {
  const auto e = last_error;
  last_error = cudaSuccess;
  return e;
}
cudaError_t cudaPeekAtLastError() { return last_error; }
const char *cudaGetErrorString(cudaError_t e) {
  if (e == detail_code && !detail.empty())
    return detail.c_str();
  switch (e) {
  case cudaSuccess:
    return "success";
  case cudaErrorInvalidValue:
    return "invalid argument";
  case cudaErrorMemoryAllocation:
    return "allocation failed";
  case cudaErrorNotReady:
    return "not complete";
  case cudaErrorNotSupported:
    return "unsupported SYCL operation";
  default:
    return "SYCL runtime error";
  }
}
