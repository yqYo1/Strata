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
#include <unistd.h>

namespace {
int log_fd = -1;
pthread_mutex_t output_lock = PTHREAD_MUTEX_INITIALIZER;
uint64_t sequence = 0;
constexpr uint64_t record_limit = 65536;

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
  if (log_fd < 0 || size <= 0 || size >= 512) fail();
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
  char row[512];
  const int size = std::snprintf(row, sizeof(row),
      "{\"kind\":\"%s\",\"pid\":%ld,\"tid\":%ld,\"instance\":%" PRIu64
      ",\"function_id\":%u,\"name\":\"%s\",\"raw_ns\":%" PRIu64 ",\"status\":%s}\n",
      begin ? "begin" : "end", long(getpid()), syscall(SYS_gettid), instance,
      data->function_id, data->function_name, raw_ns(), status);
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
