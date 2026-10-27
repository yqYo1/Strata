// Main-authored diagnostic for the existing QSA fixture. Repository LICENSE.
// This observes a fresh successful UR launch, not a historical timed image.
#include <unified-runtime/ur_api.h>
#include <level_zero/ze_api.h>
#include <xpti/xpti_trace_framework.h>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr size_t binary_limit = 4u << 20;
constexpr size_t module_limit = 8u << 20;
constexpr size_t name_limit = 2048;
struct Arm {
  char name[name_limit]{};
  ur_kernel_handle_t kernel{};
  ur_native_handle_t native{};
  unsigned launches{};
  bool captured{};
  bool extracting{};
};
Arm arms[3];
pthread_mutex_t state_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
thread_local bool querying = false;
int directory = -1, log_fd = -1;

[[noreturn]] void fail(const char* message, uint32_t status = 0) noexcept {
  char row[256];
  const int n = std::snprintf(row, sizeof(row),
      "native kernel capture: %s status=0x%x\n", message, status);
  if (n > 0 && size_t(n) < sizeof(row)) (void)!write(2, row, size_t(n));
  _exit(124);
}
void require(bool good, const char* message) noexcept { if (!good) fail(message); }
void ur_ok(ur_result_t result, const char* message) noexcept {
  if (result != UR_RESULT_SUCCESS) fail(message, uint32_t(result));
}
void ze_ok(ze_result_t result, const char* message) noexcept {
  if (result != ZE_RESULT_SUCCESS) fail(message, uint32_t(result));
}
void put(int fd, const void* data, size_t bytes) noexcept {
  const auto* p = static_cast<const char*>(data);
  while (bytes) {
    const ssize_t n = write(fd, p, bytes);
    if (n < 0 && errno == EINTR) continue;
    require(n > 0 && size_t(n) <= bytes, "write");
    p += n; bytes -= size_t(n);
  }
}
template<size_t N> void emit(const char (&row)[N], int n) noexcept {
  require(n > 0 && size_t(n) < N, "record overflow");
  require(pthread_mutex_lock(&log_mutex) == 0, "log lock");
  put(log_fd, row, size_t(n));
  require(pthread_mutex_unlock(&log_mutex) == 0, "log unlock");
}
bool valid_name(const char* name) noexcept {
  if (!name) return false;
  const size_t n = strnlen(name, name_limit);
  if (n < 4 || n == name_limit || std::strncmp(name, "_ZTS", 4)) return false;
  for (size_t i = 0; i < n; ++i) {
    const char c = name[i];
    if (!(c == '_' || (c >= '0' && c <= '9') ||
          (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) return false;
  }
  return true;
}

ur_program_handle_t device_contract(ur_queue_handle_t queue, ur_kernel_handle_t kernel, size_t arm) noexcept {
  ur_device_handle_t queue_device = nullptr, program_device = nullptr;
  ur_program_handle_t program = nullptr;
  uint32_t devices = 0;
  ur_ok(urQueueGetInfo(queue, UR_QUEUE_INFO_DEVICE, sizeof(queue_device), &queue_device, nullptr),
      "queue device");
  ur_ok(urKernelGetInfo(kernel, UR_KERNEL_INFO_PROGRAM, sizeof(program), &program, nullptr),
      "kernel program");
  ur_ok(urProgramGetInfo(program, UR_PROGRAM_INFO_NUM_DEVICES, sizeof(devices), &devices, nullptr),
      "program device count");
  require(devices == 1, "sole program device required for v2 native getter");
  ur_ok(urProgramGetInfo(program, UR_PROGRAM_INFO_DEVICES, sizeof(program_device), &program_device, nullptr),
      "program device");
  require(queue_device && queue_device == program_device, "queue/program device differs");
  ur_platform_handle_t platform = nullptr;
  ur_ok(urDeviceGetInfo(queue_device, UR_DEVICE_INFO_PLATFORM, sizeof(platform), &platform, nullptr),
      "queue platform");
  ur_backend_t backend{};
  ur_ok(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND, sizeof(backend), &backend, nullptr),
      "queue backend");
  require(backend == UR_BACKEND_LEVEL_ZERO, "Level Zero backend required");
  ur_native_handle_t native_device = 0, native_driver = 0;
  ur_ok(urDeviceGetNativeHandle(queue_device, &native_device), "native device");
  ur_ok(urPlatformGetNativeHandle(platform, &native_driver), "native driver");
  require(native_device && native_driver, "native device/driver absent");
  ze_device_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  ze_ok(zeDeviceGetProperties(reinterpret_cast<ze_device_handle_t>(native_device), &props),
      "device properties");
  require(props.type == ZE_DEVICE_TYPE_GPU && props.vendorId == 0x8086 &&
      props.deviceId == 0xe20c && !(props.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE),
      "qualified physical B570 required");
  auto driver = reinterpret_cast<ze_driver_handle_t>(native_driver);
  uint32_t count = 0;
  ze_ok(zeDriverGetExtensionProperties(driver, &count, nullptr), "extension count");
  require(count && count <= 256, "extension inventory budget");
  ze_driver_extension_properties_t extensions[256]{};
  const uint32_t capacity = count;
  ze_ok(zeDriverGetExtensionProperties(driver, &count, extensions), "extension inventory");
  require(count <= capacity, "extension inventory grew");
  uint32_t version = 0;
  for (uint32_t i = 0; i < count; ++i) {
    require(strnlen(extensions[i].name, sizeof(extensions[i].name)) < sizeof(extensions[i].name),
        "unterminated extension name");
    if (!std::strcmp(extensions[i].name, ZE_GET_KERNEL_BINARY_EXP_NAME))
      version = extensions[i].version;
  }
  require(version >= ZE_KERNEL_GET_BINARY_EXP_VERSION_1_0, "kernel binary extension unsupported");
  char row[512];
  emit(row, std::snprintf(row, sizeof(row),
      "{\"kind\":\"device\",\"pid\":%ld,\"arm\":%zu,\"program_devices\":%u,"
      "\"same_queue_device\":true,\"vendor\":%u,\"device\":%u,\"binary_extension_version\":%u}\n",
      long(getpid()), arm, devices, props.vendorId, props.deviceId, version));
  return program;
}

void module_binary(ur_program_handle_t program, size_t arm, uint64_t instance) noexcept {
  // The same sole program device was bound to the successful launch above.
  // Offline qualification must still match its ELF text to the kernel bytes;
  // a module handle/name alone is not that proof.
  ur_native_handle_t native = 0;
  ur_ok(urProgramGetNativeHandle(program, &native), "native program");
  require(native != 0, "native module absent");
  auto module = reinterpret_cast<ze_module_handle_t>(native);
  size_t bytes = 0;
  ze_ok(zeModuleGetNativeBinary(module, &bytes, nullptr), "module size");
  require(bytes && bytes <= module_limit, "module binary budget");
  auto* binary = static_cast<uint8_t*>(std::malloc(bytes));
  require(binary != nullptr, "module allocation");
  const size_t capacity = bytes;
  ze_ok(zeModuleGetNativeBinary(module, &bytes, binary), "module copy");
  require(bytes == capacity, "module size changed");
  char file[32];
  const int length = std::snprintf(file, sizeof(file), "kernel%zu.zebin", arm);
  require(length > 0 && size_t(length) < sizeof(file), "module filename");
  const int fd = openat(directory, file,
      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  require(fd >= 0, "module open");
  put(fd, binary, bytes);
  require(close(fd) == 0, "module close");
  std::free(binary);
  char row[512];
  emit(row, std::snprintf(row, sizeof(row),
      "{\"kind\":\"module\",\"pid\":%ld,\"arm\":%zu,\"instance\":%" PRIu64
      ",\"program\":\"%p\",\"native_module\":\"%p\",\"file\":\"%s\",\"bytes\":%zu}\n",
      long(getpid()), arm, instance, static_cast<void*>(program),
      static_cast<void*>(module), file, bytes));
}

template<class Params>
void launch(const Params* p, uint64_t instance, const char* api) noexcept {
  require(p && p->phKernel && p->phQueue && p->pworkDim &&
      p->ppGlobalWorkSize && p->ppLocalWorkSize, "launch parameters");
  Arm* arm = nullptr;
  require(pthread_mutex_lock(&state_mutex) == 0, "state lock");
  for (auto& a : arms) if (a.kernel && a.kernel == *p->phKernel) arm = &a;
  if (!arm) {
    require(pthread_mutex_unlock(&state_mutex) == 0, "state unlock");
    return; // Other fixture kernels are outside this diagnostic.
  }
  require(!arm->extracting, "concurrent selected-kernel extraction");
  arm->extracting = true;
  const Arm previous = *arm;
  require(pthread_mutex_unlock(&state_mutex) == 0, "state unlock");
  const size_t index = size_t(arm - arms);
  require(*p->pworkDim == 1 && *p->ppGlobalWorkSize && *p->ppLocalWorkSize &&
      **p->ppLocalWorkSize == 128 && **p->ppGlobalWorkSize >= 128 &&
      **p->ppGlobalWorkSize % 128 == 0, "selected launch geometry");
  require(previous.launches < 16, "selected launch budget");

  // The synchronous callback runs before this successful UR call returns to
  // the fixture. Only read the still-owned kernel; never release/retain it,
  // alter its arguments, synchronize a queue, or request a different JIT mode.
  // No observer state lock is held across a driver call or file operation.
  querying = true;
  ur_program_handle_t program = nullptr;
  if (!previous.captured) program = device_contract(*p->phQueue, *p->phKernel, index);
  ur_native_handle_t native = 0;
  ur_ok(urKernelGetNativeHandle(*p->phKernel, &native), "UR native handle");
  require(native && (!previous.captured || native == previous.native), "native identity changed");
  auto kernel = reinterpret_cast<ze_kernel_handle_t>(native);
  char name[name_limit]{};
  size_t name_bytes = sizeof(name);
  ze_ok(zeKernelGetName(kernel, &name_bytes, name), "native name");
  require(name_bytes > 0 && name_bytes <= sizeof(name) &&
      name[name_bytes - 1] == 0 && !std::strcmp(name, arm->name), "native name differs");
  if (!previous.captured) {
    ze_kernel_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    ze_ok(zeKernelGetProperties(kernel, &props), "native properties");
    require(props.numKernelArgs == 18 && props.requiredSubgroupSize == 16,
        "selected native argument/subgroup contract");
    size_t bytes = 0;
    ze_ok(zeKernelGetBinaryExp(kernel, &bytes, nullptr), "binary size");
    require(bytes && bytes <= binary_limit, "binary size budget");
    auto* binary = static_cast<uint8_t*>(std::malloc(bytes));
    require(binary != nullptr, "binary allocation");
    const size_t capacity = bytes;
    ze_ok(zeKernelGetBinaryExp(kernel, &bytes, binary), "binary copy");
    require(bytes == capacity, "binary size changed");
    char file[32];
    const int length = std::snprintf(file, sizeof(file), "kernel%zu.gen", index);
    require(length > 0 && size_t(length) < sizeof(file), "binary filename");
    const int fd = openat(directory, file,
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    require(fd >= 0, "binary open");
    put(fd, binary, bytes);
    require(close(fd) == 0, "binary close");
    std::free(binary);
    module_binary(program, index, instance);
    char row[4096];
    emit(row, std::snprintf(row, sizeof(row),
        "{\"kind\":\"capture\",\"pid\":%ld,\"arm\":%zu,\"instance\":%" PRIu64
        ",\"ur_kernel\":\"%p\",\"native_kernel\":\"%p\",\"name\":\"%s\","
        "\"file\":\"%s\",\"bytes\":%zu,\"arguments\":%u,\"subgroup\":%u,"
        "\"local_descriptor\":%u,\"private_descriptor\":%u,\"spill_descriptor\":%u}\n",
        long(getpid()), index, instance, static_cast<void*>(previous.kernel),
        static_cast<void*>(kernel), name, file, bytes, props.numKernelArgs,
        props.requiredSubgroupSize, props.localMemSize, props.privateMemSize, props.spillMemSize));
  }
  querying = false;
  require(pthread_mutex_lock(&state_mutex) == 0, "state lock");
  require(arm->kernel == previous.kernel && arm->launches == previous.launches && arm->extracting,
      "kernel state changed during extraction");
  arm->native = native;
  arm->captured = true;
  ++arm->launches;
  arm->extracting = false;
  const unsigned ordinal = arm->launches;
  require(pthread_mutex_unlock(&state_mutex) == 0, "state unlock");
  char row[512];
  emit(row, std::snprintf(row, sizeof(row),
      "{\"kind\":\"launch\",\"pid\":%ld,\"arm\":%zu,\"instance\":%" PRIu64
      ",\"api\":\"%s\",\"queue\":\"%p\",\"global\":%zu,\"local\":%zu,\"ordinal\":%u}\n",
      long(getpid()), index, instance, api, static_cast<void*>(*p->phQueue),
      **p->ppGlobalWorkSize, **p->ppLocalWorkSize, ordinal));
}

void callback(uint16_t type, xpti::trace_event_data_t*, xpti::trace_event_data_t*,
              uint64_t instance, const void* raw) noexcept {
  if (querying || type != xpti::trace_function_with_args_end) return;
  const auto* d = static_cast<const xpti::function_with_args_t*>(raw);
  require(d && d->function_name && d->args_data && d->ret_data, "callback packet");
  if (*static_cast<const ur_result_t*>(d->ret_data) != UR_RESULT_SUCCESS) return;
  if (d->function_id != UR_FUNCTION_KERNEL_CREATE &&
      d->function_id != UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH &&
      d->function_id != UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH_WITH_ARGS_EXP) return;
  if (d->function_id == UR_FUNCTION_KERNEL_CREATE) {
    require(!std::strcmp(d->function_name, "urKernelCreate"), "create API identity");
    const auto* p = static_cast<const ur_kernel_create_params_t*>(d->args_data);
    require(p->ppKernelName && *p->ppKernelName && p->pphKernel &&
        *p->pphKernel && **p->pphKernel, "create parameters");
    require(pthread_mutex_lock(&state_mutex) == 0, "state lock");
    for (auto& arm : arms) {
      const bool match = !std::strcmp(*p->ppKernelName, arm.name);
      require(!arm.kernel || arm.kernel != **p->pphKernel || match, "reused UR identity");
      if (!match) continue;
      require(!arm.kernel || arm.kernel == **p->pphKernel, "recreated selected kernel");
      if (!arm.kernel) arm.kernel = **p->pphKernel;
    }
    require(pthread_mutex_unlock(&state_mutex) == 0, "state unlock");
  } else if (d->function_id == UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH) {
    require(!std::strcmp(d->function_name, "urEnqueueKernelLaunch"), "launch API identity");
    launch(static_cast<const ur_enqueue_kernel_launch_params_t*>(d->args_data), instance, d->function_name);
  } else {
    require(!std::strcmp(d->function_name, "urEnqueueKernelLaunchWithArgsExp"), "with-args API identity");
    launch(static_cast<const ur_enqueue_kernel_launch_with_args_exp_params_t*>(d->args_data), instance, d->function_name);
  }
}

__attribute__((constructor)) void load() noexcept {
  const char* root = std::getenv("STRATA_NATIVE_CAPTURE_ROOT");
  require(root && root[0] == '/', "absolute private output root required");
  const char* subscriber = std::getenv("STRATA_NATIVE_CAPTURE_SUBSCRIBER");
  const char* subscribers = std::getenv("XPTI_SUBSCRIBERS");
  require(subscriber && subscriber[0] == '/' && !std::strchr(subscriber, ':') &&
      !std::strchr(subscriber, ';') && !std::strchr(subscriber, ',') &&
      subscribers && !std::strcmp(subscribers, subscriber), "sole capture subscriber required");
  directory = open(root, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  require(directory >= 0, "output directory");
  struct stat st{};
  require(fstat(directory, &st) == 0 && st.st_uid == getuid() &&
      (st.st_mode & 0777) == 0700, "private owned output directory required");
  for (size_t i = 0; i < 3; ++i) {
    char key[32];
    const int n = std::snprintf(key, sizeof(key), "STRATA_NATIVE_KERNEL_%zu", i);
    require(n > 0 && size_t(n) < sizeof(key), "environment key");
    const char* name = std::getenv(key);
    require(valid_name(name), "exact mangled kernel allowlist required");
    std::strcpy(arms[i].name, name);
    for (size_t j = 0; j < i; ++j) require(std::strcmp(name, arms[j].name), "duplicate kernel allowlist");
  }
  log_fd = openat(directory, "captures.jsonl",
      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  require(log_fd >= 0, "capture log");
  char row[128];
  emit(row, std::snprintf(row, sizeof(row), "{\"kind\":\"load\",\"pid\":%ld}\n", long(getpid())));
  // Plain-data state and FDs intentionally live until native process exit.
}
} // namespace

extern "C" XPTI_CALLBACK_API void xptiTraceInit(unsigned, unsigned, const char*, const char* stream) {
  if (!stream || std::strcmp(stream, "ur.call")) return;
  const auto id = xptiRegisterStream(stream);
  require(xptiRegisterCallback(id, xpti::trace_function_with_args_end, callback) ==
      xpti::result_t::XPTI_RESULT_SUCCESS, "subscriber registration");
  char row[128];
  emit(row, std::snprintf(row, sizeof(row), "{\"kind\":\"subscribe\",\"pid\":%ld}\n", long(getpid())));
}
extern "C" XPTI_CALLBACK_API void xptiTraceFinish(const char*) {}
