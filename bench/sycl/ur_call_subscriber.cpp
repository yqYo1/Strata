// Independently authored Strata diagnostic; repository LICENSE applies.
// XPTI/UR declarations come from the pinned Intel LLVM SDK, not another engine.
// Preload before provider initialization; also name this same ELF in
// XPTI_SUBSCRIBERS. No SYCL/UR API calls, GPU work or environment writes here.
#include <xpti/xpti_trace_framework.h>
#include <unified-runtime/ur_api.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <type_traits>
#include <unistd.h>

namespace {
int log_fd = -1;
pthread_mutex_t output_lock = PTHREAD_MUTEX_INITIALIZER;
uint64_t sequence = 0;
constexpr uint64_t record_limit = 65536;
constexpr int row_capacity = 2048;

[[noreturn]] void fail() noexcept {
  constexpr char message[] = "strata UR subscriber: incomplete diagnostic\n";
  (void)!write(STDERR_FILENO, message, sizeof(message) - 1);
  _exit(86);
}

uint64_t raw_ns() noexcept {
  timespec now{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &now)) fail();
  return uint64_t(now.tv_sec) * 1000000000ULL + uint64_t(now.tv_nsec);
}

// No buffered I/O, allocation or exception propagation from a callback.
void emit(const char* row, int size) noexcept {
  if (log_fd < 0 || size <= 0 || size >= row_capacity) fail();
  if (pthread_mutex_lock(&output_lock)) fail();
  if (sequence++ >= record_limit) fail();
  int offset = 0;
  while (offset < size) {
    const ssize_t written = write(log_fd, row + offset, size - offset);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) fail();
    offset += int(written);
  }
  if (pthread_mutex_unlock(&output_lock)) fail();
}

bool api_name(const char* name) noexcept {
  if (!name) return false;
  for (unsigned i = 0; i < 160; ++i) {
    const unsigned char c = name[i];
    if (!c) return i > 2 && name[0] == 'u' && name[1] == 'r';
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_')) return false;
  }
  return false;
}

// Snapshot argument scalars from XPTI's exact generated UR parameter records.
// Never dereference a handle or a USM address. Read optional output pointers,
// work-size arrays and wait lists only at a successful end, after validation.
// Failed calls must not read potentially invalid application pointees.
struct Metadata {
  char text[1280] = "{";
  size_t used = 1;
  bool first = true;
  void number(const char* key, uint64_t value) noexcept {
    int n = std::snprintf(text + used, sizeof(text) - used, "%s\"%s\":%" PRIu64,
                          first ? "" : ",", key, value);
    if (n < 0 || size_t(n) >= sizeof(text) - used) fail();
    used += size_t(n); first = false;
  }
  template<class T> static uint64_t address(T value) noexcept {
    return uint64_t(reinterpret_cast<uintptr_t>(value));
  }
  template<class T> void handle(const char* key, const T* value) noexcept {
    if (!value) fail();
    number(key, address(*value));
  }
  template<class T> void output(const char* key, T* const* value, bool success) noexcept {
    if (!value) fail();
    number(key, success && *value ? address(**value) : 0);
  }
  template<class T> void array(const char* key, const T* values, uint32_t count) noexcept {
    if (count > 32 || (count && !values)) fail();
    int n = std::snprintf(text + used, sizeof(text) - used, "%s\"%s\":[", first ? "" : ",", key);
    if (n < 0 || size_t(n) >= sizeof(text) - used) fail();
    used += size_t(n); first = false;
    for (uint32_t i = 0; i < count; ++i) {
      uint64_t value;
      if constexpr (std::is_pointer_v<T>) value = address(values[i]);
      else value = uint64_t(values[i]);
      n = std::snprintf(text + used, sizeof(text) - used, "%s%" PRIu64, i ? "," : "", value);
      if (n < 0 || size_t(n) >= sizeof(text) - used) fail();
      used += size_t(n);
    }
    if (used + 2 >= sizeof(text)) fail();
    text[used++] = ']'; text[used] = 0;
  }
  template<class T> void dependencies(const T* p, bool success) noexcept {
    if (!p->pnumEventsInWaitList || !p->pphEventWaitList) fail();
    number("wait_count", *p->pnumEventsInWaitList);
    if (success) array("waits", *p->pphEventWaitList, *p->pnumEventsInWaitList);
    output("produced_event", p->pphEvent, success);
  }
  template<class T> void geometry(const T* p, bool success) noexcept {
    handle("kernel", p->phKernel);
    if (!p->pworkDim) fail();
    number("work_dim", *p->pworkDim);
    if (!success) return;
    const uint32_t dim = *p->pworkDim;
    if (!dim || dim > 3 || !p->ppGlobalWorkSize || !p->ppGlobalWorkOffset || !p->ppLocalWorkSize) fail();
    array("global", *p->ppGlobalWorkSize, dim);
    array("offset", *p->ppGlobalWorkOffset, *p->ppGlobalWorkOffset ? dim : 0);
    array("local", *p->ppLocalWorkSize, *p->ppLocalWorkSize ? dim : 0);
  }
  template<class T> void graph_dependencies(const T* p, bool success) noexcept {
    dependencies(p, success);
    if (!p->pnumSyncPointsInWaitList || !p->ppSyncPointWaitList || !p->ppSyncPoint || !p->pphCommand) fail();
    number("sync_count", *p->pnumSyncPointsInWaitList);
    if (success) array("sync_waits", *p->ppSyncPointWaitList, *p->pnumSyncPointsInWaitList);
    number("sync_point", success && *p->ppSyncPoint ? **p->ppSyncPoint : 0);
    output("command", p->pphCommand, success);
  }
  const char* finish() noexcept {
    if (used + 2 > sizeof(text)) fail();
    text[used++] = '}'; text[used] = 0; return text;
  }
};

template<class T> const T* arguments(const xpti::function_with_args_t* data, const char* name) noexcept {
  if (!data->args_data || std::strcmp(data->function_name, name)) fail();
  return static_cast<const T*>(data->args_data);
}

void metadata(Metadata& m, const xpti::function_with_args_t* data, bool success) noexcept {
#define ARGS(ID, type, name) case UR_FUNCTION_##ID: { const auto* p = arguments<ur_##type##_params_t>(data, name);
#define END break; }
#define QUEUE m.handle("queue", p->phQueue)
#define GRAPH m.handle("graph", p->phCommandBuffer)
  switch (data->function_id) {
    ARGS(QUEUE_CREATE, queue_create, "urQueueCreate")
      m.handle("context", p->phContext); m.output("created_queue", p->pphQueue, success); END
    ARGS(QUEUE_RETAIN, queue_retain, "urQueueRetain") QUEUE; END
    ARGS(QUEUE_RELEASE, queue_release, "urQueueRelease") QUEUE; END
    ARGS(QUEUE_FINISH, queue_finish, "urQueueFinish") QUEUE; END
    ARGS(EVENT_RETAIN, event_retain, "urEventRetain") m.handle("event", p->phEvent); END
    ARGS(EVENT_RELEASE, event_release, "urEventRelease") m.handle("event", p->phEvent); END
    ARGS(EVENT_WAIT, event_wait, "urEventWait")
      if (!p->pnumEvents || !p->pphEventWaitList) fail();
      m.number("wait_count", *p->pnumEvents);
      if (success) m.array("waits", *p->pphEventWaitList, *p->pnumEvents); END
    ARGS(USM_DEVICE_ALLOC, usm_device_alloc, "urUSMDeviceAlloc")
      m.handle("context", p->phContext); m.number("bytes", *p->psize); m.output("allocation", p->pppMem, success); END
    ARGS(USM_HOST_ALLOC, usm_host_alloc, "urUSMHostAlloc")
      m.handle("context", p->phContext); m.number("bytes", *p->psize); m.output("allocation", p->pppMem, success); END
    ARGS(USM_SHARED_ALLOC, usm_shared_alloc, "urUSMSharedAlloc")
      m.handle("context", p->phContext); m.number("bytes", *p->psize); m.output("allocation", p->pppMem, success); END
    ARGS(USM_FREE, usm_free, "urUSMFree") m.handle("context", p->phContext); m.handle("allocation", p->ppMem); END
    ARGS(ENQUEUE_KERNEL_LAUNCH, enqueue_kernel_launch, "urEnqueueKernelLaunch")
      QUEUE; m.geometry(p, success); m.dependencies(p, success); END
    ARGS(ENQUEUE_KERNEL_LAUNCH_WITH_ARGS_EXP, enqueue_kernel_launch_with_args_exp, "urEnqueueKernelLaunchWithArgsExp")
      QUEUE; m.geometry(p, success); m.dependencies(p, success); END
    ARGS(ENQUEUE_USM_MEMCPY, enqueue_usm_memcpy, "urEnqueueUSMMemcpy")
      QUEUE; m.handle("dst", p->ppDst); m.handle("src", p->ppSrc); m.number("bytes", *p->psize);
      m.dependencies(p, success); END
    ARGS(ENQUEUE_USM_FILL, enqueue_usm_fill, "urEnqueueUSMFill")
      QUEUE; m.handle("dst", p->ppMem); m.number("bytes", *p->psize); m.number("pattern_bytes", *p->ppatternSize);
      m.dependencies(p, success); END
    ARGS(ENQUEUE_EVENTS_WAIT, enqueue_events_wait, "urEnqueueEventsWait") QUEUE; m.dependencies(p, success); END
    ARGS(ENQUEUE_EVENTS_WAIT_WITH_BARRIER, enqueue_events_wait_with_barrier, "urEnqueueEventsWaitWithBarrier")
      QUEUE; m.dependencies(p, success); END
    ARGS(ENQUEUE_EVENTS_WAIT_WITH_BARRIER_EXT, enqueue_events_wait_with_barrier_ext, "urEnqueueEventsWaitWithBarrierExt")
      QUEUE; m.dependencies(p, success); END
    ARGS(COMMAND_BUFFER_CREATE_EXP, command_buffer_create_exp, "urCommandBufferCreateExp")
      m.handle("context", p->phContext); m.output("created_graph", p->pphCommandBuffer, success); END
    ARGS(COMMAND_BUFFER_RETAIN_EXP, command_buffer_retain_exp, "urCommandBufferRetainExp") GRAPH; END
    ARGS(COMMAND_BUFFER_RELEASE_EXP, command_buffer_release_exp, "urCommandBufferReleaseExp") GRAPH; END
    ARGS(COMMAND_BUFFER_FINALIZE_EXP, command_buffer_finalize_exp, "urCommandBufferFinalizeExp") GRAPH; END
    ARGS(COMMAND_BUFFER_APPEND_KERNEL_LAUNCH_EXP, command_buffer_append_kernel_launch_exp, "urCommandBufferAppendKernelLaunchExp")
      GRAPH; m.geometry(p, success); m.graph_dependencies(p, success); END
    ARGS(COMMAND_BUFFER_APPEND_USM_MEMCPY_EXP, command_buffer_append_usm_memcpy_exp, "urCommandBufferAppendUSMMemcpyExp")
      GRAPH; m.handle("dst", p->ppDst); m.handle("src", p->ppSrc); m.number("bytes", *p->psize);
      m.graph_dependencies(p, success); END
    ARGS(COMMAND_BUFFER_APPEND_USM_FILL_EXP, command_buffer_append_usm_fill_exp, "urCommandBufferAppendUSMFillExp")
      GRAPH; m.handle("dst", p->ppMemory); m.number("bytes", *p->psize); m.number("pattern_bytes", *p->ppatternSize);
      m.graph_dependencies(p, success); END
    ARGS(ENQUEUE_COMMAND_BUFFER_EXP, enqueue_command_buffer_exp, "urEnqueueCommandBufferExp")
      QUEUE; GRAPH; m.dependencies(p, success); END
    default: break; // Explicitly unclassified; downstream coverage may not assume metadata.
  }
#undef ARGS
#undef END
#undef QUEUE
#undef GRAPH
}

void callback(uint16_t type, xpti::trace_event_data_t*,
              xpti::trace_event_data_t*, uint64_t instance,
              const void* user_data) noexcept {
  const auto* data = static_cast<const xpti::function_with_args_t*>(user_data);
  const bool begin = type == xpti::trace_function_with_args_begin;
  if (!data || !api_name(data->function_name) ||
      (!begin && type != xpti::trace_function_with_args_end)) fail();
  int32_t result = -1;
  static_assert(sizeof(ur_result_t) == sizeof(result));
  if (begin ? data->ret_data != nullptr : data->ret_data == nullptr) fail();
  if (!begin) std::memcpy(&result, data->ret_data, sizeof(result));
  char status[32] = "null";
  if (!begin) std::snprintf(status, sizeof(status), "%" PRId32, result);
  Metadata details;
  metadata(details, data, !begin && result == UR_RESULT_SUCCESS);
  char row[row_capacity];
  const int size = std::snprintf(row, sizeof(row),
      "{\"kind\":\"%s\",\"pid\":%ld,\"tid\":%ld,\"instance\":%" PRIu64
      ",\"function_id\":%u,\"name\":\"%s\",\"raw_ns\":%" PRIu64 ",\"status\":%s,\"metadata\":%s}\n",
      begin ? "begin" : "end", long(getpid()), syscall(SYS_gettid), instance,
      data->function_id, data->function_name, raw_ns(), status, details.finish());
  emit(row, size);
}

__attribute__((constructor)) void load() noexcept {
  const char* name = std::getenv("STRATA_UR_CALL_LOG");
  if (!name || name[0] != '/' || std::strlen(name) >= 4096) fail();
  log_fd = open(name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (log_fd < 0) fail();
  char row[512];
  emit(row, std::snprintf(row, sizeof(row),
      "{\"kind\":\"load\",\"pid\":%ld,\"raw_ns\":%" PRIu64 "}\n", long(getpid()), raw_ns()));
  // The FD and plain-data state intentionally live until process termination.
  // No destructor can race a late framework callback. This does not qualify
  // ordinary runtime teardown; the existing validation reference stays held.
}

[[noreturn]] void environment_write() noexcept {
  constexpr char message[] = "strata UR subscriber: environment mutation blocked\n";
  (void)!write(STDERR_FILENO, message, sizeof(message) - 1);
  _exit(87);
}
} // namespace

extern "C" int setenv(const char*, const char*, int) noexcept { environment_write(); }
extern "C" int unsetenv(const char*) noexcept { environment_write(); }
extern "C" int putenv(char*) noexcept { environment_write(); }
extern "C" int clearenv() noexcept { environment_write(); }

extern "C" XPTI_CALLBACK_API void xptiTraceInit(unsigned, unsigned, const char*,
                                                const char* stream) {
  if (!stream || std::strcmp(stream, "ur.call")) return;
  const auto id = xptiRegisterStream(stream);
  if (xptiRegisterCallback(id, xpti::trace_function_with_args_begin, callback) !=
          xpti::result_t::XPTI_RESULT_SUCCESS ||
      xptiRegisterCallback(id, xpti::trace_function_with_args_end, callback) !=
          xpti::result_t::XPTI_RESULT_SUCCESS) fail();
  char row[512];
  emit(row, std::snprintf(row, sizeof(row),
      "{\"kind\":\"subscribe\",\"stream\":\"ur.call\",\"pid\":%ld,\"raw_ns\":%" PRIu64 "}\n",
      long(getpid()), raw_ns()));
}

extern "C" XPTI_CALLBACK_API void xptiTraceFinish(const char* stream) {
  if (!stream || std::strcmp(stream, "ur.call")) return;
  char row[512];
  emit(row, std::snprintf(row, sizeof(row),
      "{\"kind\":\"finish\",\"stream\":\"ur.call\",\"pid\":%ld,\"raw_ns\":%" PRIu64 "}\n",
      long(getpid()), raw_ns()));
}
