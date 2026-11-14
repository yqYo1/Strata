// Main-authored host-only mutation controls; repository LICENSE applies.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <thread>
#include <xpti/xpti_trace_framework.h>
#include <unified-runtime/ur_api.h>

namespace {
unsigned simulated_calls = 0;
void notify(uint32_t id, const char* name, void* args, ur_result_t result = UR_RESULT_SUCCESS) {
  xpti::function_with_args_t call{};
  call.function_id = id; call.function_name = name; call.args_data = args;
  const auto stream = xptiRegisterStream("ur.call");
  const auto instance = xptiGetUniqueId();
  if (xptiNotifySubscribers(stream, xpti::trace_function_with_args_begin, nullptr, nullptr,
                            instance, &call) != xpti::result_t::XPTI_RESULT_SUCCESS) std::exit(4);
  call.ret_data = &result;
  if (xptiNotifySubscribers(stream, xpti::trace_function_with_args_end, nullptr, nullptr,
                            instance, &call) != xpti::result_t::XPTI_RESULT_SUCCESS) std::exit(5);
  ++simulated_calls;
}
int metadata_fixture() {
  // Fabricated scalar handles never enter UR or Level Zero. Failed operations
  // use an invalid output/list pointee to verify the observer never reads it.
  xptiFrameworkInitialize();
  if (xptiInitialize("ur.call", 0, 1, "host-fixture") != xpti::result_t::XPTI_RESULT_SUCCESS) return 6;
  auto context = reinterpret_cast<ur_context_handle_t>(uintptr_t(0x100));
  auto queue = reinterpret_cast<ur_queue_handle_t>(uintptr_t(0x101));
  auto event = reinterpret_cast<ur_event_handle_t>(uintptr_t(0x201));
  auto graph = reinterpret_cast<ur_exp_command_buffer_handle_t>(uintptr_t(0x301));
  auto kernel = reinterpret_cast<ur_kernel_handle_t>(uintptr_t(0x401));
  auto command = reinterpret_cast<ur_exp_command_buffer_command_handle_t>(uintptr_t(0x501));
  ur_queue_handle_t* queue_out = reinterpret_cast<ur_queue_handle_t*>(uintptr_t(1));
  ur_queue_create_params_t qc{}; qc.phContext = &context; qc.pphQueue = &queue_out;
  notify(UR_FUNCTION_QUEUE_CREATE, "urQueueCreate", &qc, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  queue_out = &queue;
  notify(UR_FUNCTION_QUEUE_CREATE, "urQueueCreate", &qc);
  size_t bytes = 1028;
  void* memory = reinterpret_cast<void*>(uintptr_t(0x100000)); void** memory_out = &memory;
  ur_usm_device_alloc_params_t alloc{}; alloc.phContext = &context; alloc.psize = &bytes; alloc.pppMem = &memory_out;
  ur_usm_free_params_t free{}; free.phContext = &context; free.ppMem = &memory;
  ur_queue_release_params_t qr{&queue}; ur_queue_finish_params_t finish{&queue};
  ur_event_release_params_t er{&event}; ur_event_retain_params_t retain{&event};
  uint32_t dim = 1, zero = 0;
  size_t extent[] = {257}; const size_t* global = extent; const size_t* absent = nullptr;
  const ur_event_handle_t* waits = nullptr; ur_event_handle_t* event_out = &event;
  ur_enqueue_kernel_launch_with_args_exp_params_t launch{};
  launch.phQueue = &queue; launch.phKernel = &kernel; launch.pworkDim = &dim;
  launch.ppGlobalWorkSize = &global; launch.ppGlobalWorkOffset = &absent; launch.ppLocalWorkSize = &absent;
  launch.pnumEventsInWaitList = &zero; launch.pphEventWaitList = &waits; launch.pphEvent = &event_out;
  notify(UR_FUNCTION_USM_DEVICE_ALLOC, "urUSMDeviceAlloc", &alloc);
  notify(UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH_WITH_ARGS_EXP, "urEnqueueKernelLaunchWithArgsExp", &launch);
  notify(UR_FUNCTION_EVENT_RETAIN, "urEventRetain", &retain);
  notify(UR_FUNCTION_EVENT_RELEASE, "urEventRelease", &er);
  notify(UR_FUNCTION_EVENT_RELEASE, "urEventRelease", &er);
  notify(UR_FUNCTION_QUEUE_FINISH, "urQueueFinish", &finish);
  notify(UR_FUNCTION_USM_FREE, "urUSMFree", &free);
  notify(UR_FUNCTION_QUEUE_RELEASE, "urQueueRelease", &qr);
  // Recycled numerical queue/allocation/event handles get distinct lifetimes.
  notify(UR_FUNCTION_QUEUE_CREATE, "urQueueCreate", &qc);
  notify(UR_FUNCTION_USM_DEVICE_ALLOC, "urUSMDeviceAlloc", &alloc);
  ur_exp_command_buffer_handle_t* graph_out = &graph;
  ur_command_buffer_create_exp_params_t gc{}; gc.phContext = &context; gc.pphCommandBuffer = &graph_out;
  notify(UR_FUNCTION_COMMAND_BUFFER_CREATE_EXP, "urCommandBufferCreateExp", &gc);
  uint32_t point_value = 1; ur_exp_command_buffer_sync_point_t* point_out = &point_value;
  const ur_exp_command_buffer_sync_point_t* points = nullptr;
  ur_exp_command_buffer_command_handle_t* command_out = &command;
  ur_command_buffer_append_kernel_launch_exp_params_t append{};
  append.phCommandBuffer = &graph; append.phKernel = &kernel; append.pworkDim = &dim;
  append.ppGlobalWorkSize = &global; append.ppGlobalWorkOffset = &absent; append.ppLocalWorkSize = &absent;
  append.pnumSyncPointsInWaitList = &zero; append.ppSyncPointWaitList = &points; append.ppSyncPoint = &point_out;
  append.pnumEventsInWaitList = &zero; append.pphEventWaitList = &waits; append.pphEvent = &event_out;
  append.pphCommand = &command_out; event_out = nullptr;
  notify(UR_FUNCTION_COMMAND_BUFFER_APPEND_KERNEL_LAUNCH_EXP, "urCommandBufferAppendKernelLaunchExp", &append);
  ++point_value; kernel = reinterpret_cast<ur_kernel_handle_t>(uintptr_t(0x402));
  command = reinterpret_cast<ur_exp_command_buffer_command_handle_t>(uintptr_t(0x502));
  notify(UR_FUNCTION_COMMAND_BUFFER_APPEND_KERNEL_LAUNCH_EXP, "urCommandBufferAppendKernelLaunchExp", &append);
  ur_command_buffer_finalize_exp_params_t gf{&graph};
  notify(UR_FUNCTION_COMMAND_BUFFER_FINALIZE_EXP, "urCommandBufferFinalizeExp", &gf);
  ur_enqueue_command_buffer_exp_params_t replay{&queue, &graph, &zero, &waits, &event_out};
  uint32_t invalid_count = 1;
  const ur_event_handle_t* invalid_waits = reinterpret_cast<const ur_event_handle_t*>(uintptr_t(1));
  event_out = reinterpret_cast<ur_event_handle_t*>(uintptr_t(1));
  replay.pnumEventsInWaitList = &invalid_count; replay.pphEventWaitList = &invalid_waits;
  notify(UR_FUNCTION_ENQUEUE_COMMAND_BUFFER_EXP, "urEnqueueCommandBufferExp", &replay, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  event_out = &event; replay.pnumEventsInWaitList = &zero; replay.pphEventWaitList = &waits;
  for (unsigned i = 0; i < 2; ++i) {
    notify(UR_FUNCTION_ENQUEUE_COMMAND_BUFFER_EXP, "urEnqueueCommandBufferExp", &replay);
    notify(UR_FUNCTION_QUEUE_FINISH, "urQueueFinish", &finish);
    notify(UR_FUNCTION_EVENT_RELEASE, "urEventRelease", &er);
  }
  ur_command_buffer_release_exp_params_t gr{&graph};
  notify(UR_FUNCTION_COMMAND_BUFFER_RELEASE_EXP, "urCommandBufferReleaseExp", &gr);
  notify(UR_FUNCTION_USM_FREE, "urUSMFree", &free);
  notify(UR_FUNCTION_QUEUE_RELEASE, "urQueueRelease", &qr);
  xptiFinalize("ur.call"); xptiFrameworkFinalize();
  std::printf("{\"metadata_fixture\":\"PASS\",\"simulated_calls\":%u}\n", simulated_calls);
  return 0;
}

int buffer_fixture() {
  // CPU XPTI payloads only. Addresses are scalar identities,never device work.
  // Invalid pointees on rejected calls must remain unread by the subscriber.
  xptiFrameworkInitialize();
  if (xptiInitialize("ur.call", 0, 1, "buffer-host-fixture") != xpti::result_t::XPTI_RESULT_SUCCESS) return 6;
  auto context = reinterpret_cast<ur_context_handle_t>(uintptr_t(0x100));
  auto queue = reinterpret_cast<ur_queue_handle_t>(uintptr_t(0x101));
  auto buffer = reinterpret_cast<ur_mem_handle_t>(uintptr_t(0x601));
  auto event = reinterpret_cast<ur_event_handle_t>(uintptr_t(0x201));
  ur_queue_handle_t* queue_out = &queue;
  ur_queue_create_params_t qc{}; qc.phContext = &context; qc.pphQueue = &queue_out;
  notify(UR_FUNCTION_QUEUE_CREATE, "urQueueCreate", &qc);
  size_t bytes = 32, offset = 4;
  ur_mem_handle_t* buffer_out = reinterpret_cast<ur_mem_handle_t*>(uintptr_t(1));
  ur_mem_buffer_create_params_t create{}; create.phContext = &context;
  create.psize = &bytes; create.pphBuffer = &buffer_out;
  notify(UR_FUNCTION_MEM_BUFFER_CREATE, "urMemBufferCreate", &create, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  bool enabled = true; bool* enabled_out = reinterpret_cast<bool*>(uintptr_t(1));
  ur_queue_is_graph_capture_enabled_exp_params_t query{&queue, &enabled_out};
  notify(UR_FUNCTION_QUEUE_IS_GRAPH_CAPTURE_ENABLED_EXP, "urQueueIsGraphCaptureEnabledExp", &query,
         UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  ur_native_handle_t native = 0x800; ur_native_handle_t* native_out = reinterpret_cast<ur_native_handle_t*>(uintptr_t(1));
  ur_mem_get_native_handle_params_t native_query{}; native_query.phMem = &buffer; native_query.pphNativeMem = &native_out;
  notify(UR_FUNCTION_MEM_GET_NATIVE_HANDLE, "urMemGetNativeHandle", &native_query, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  buffer_out = &buffer;
  notify(UR_FUNCTION_MEM_BUFFER_CREATE, "urMemBufferCreate", &create);
  ur_mem_retain_params_t retain{&buffer}; ur_mem_release_params_t release{&buffer};
  notify(UR_FUNCTION_MEM_RETAIN, "urMemRetain", &retain);
  enabled_out = &enabled;
  notify(UR_FUNCTION_QUEUE_IS_GRAPH_CAPTURE_ENABLED_EXP, "urQueueIsGraphCaptureEnabledExp", &query);
  native_out = &native;
  notify(UR_FUNCTION_MEM_GET_NATIVE_HANDLE, "urMemGetNativeHandle", &native_query);
  uint32_t wait_count = 1;
  const ur_event_handle_t* waits = reinterpret_cast<const ur_event_handle_t*>(uintptr_t(1));
  ur_event_handle_t* event_out = reinterpret_cast<ur_event_handle_t*>(uintptr_t(1));
  const void* host_src = reinterpret_cast<const void*>(uintptr_t(0x701));
  void* host_dst = reinterpret_cast<void*>(uintptr_t(0x702));
  bool blocking = false; bytes = 8;
  ur_enqueue_mem_buffer_read_params_t read{&queue, &buffer, &blocking, &offset, &bytes, &host_dst,
                                         &wait_count, &waits, &event_out};
  ur_enqueue_mem_buffer_write_params_t write{&queue, &buffer, &blocking, &offset, &bytes, &host_src,
                                           &wait_count, &waits, &event_out};
  notify(UR_FUNCTION_ENQUEUE_MEM_BUFFER_READ, "urEnqueueMemBufferRead", &read, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  notify(UR_FUNCTION_ENQUEUE_MEM_BUFFER_WRITE, "urEnqueueMemBufferWrite", &write, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  wait_count = 0; waits = nullptr; event_out = &event;
  notify(UR_FUNCTION_ENQUEUE_MEM_BUFFER_WRITE, "urEnqueueMemBufferWrite", &write);
  uint32_t one = 1; waits = &event;
  ur_event_wait_params_t wait{&one, &waits}; ur_event_release_params_t er{&event};
  notify(UR_FUNCTION_EVENT_WAIT, "urEventWait", &wait);
  notify(UR_FUNCTION_EVENT_RELEASE, "urEventRelease", &er);
  waits = nullptr; event_out = nullptr; blocking = true; offset = 8;
  notify(UR_FUNCTION_ENQUEUE_MEM_BUFFER_READ, "urEnqueueMemBufferRead", &read);
  notify(UR_FUNCTION_MEM_RELEASE, "urMemRelease", &release);
  notify(UR_FUNCTION_MEM_RELEASE, "urMemRelease", &release);
  bytes = 16;
  notify(UR_FUNCTION_MEM_BUFFER_CREATE, "urMemBufferCreate", &create);
  native = 0x900;
  notify(UR_FUNCTION_MEM_GET_NATIVE_HANDLE, "urMemGetNativeHandle", &native_query);
  notify(UR_FUNCTION_MEM_RELEASE, "urMemRelease", &release);
  enabled = false;
  notify(UR_FUNCTION_QUEUE_IS_GRAPH_CAPTURE_ENABLED_EXP, "urQueueIsGraphCaptureEnabledExp", &query);
  ur_queue_release_params_t qr{&queue}; notify(UR_FUNCTION_QUEUE_RELEASE, "urQueueRelease", &qr);
  xptiFinalize("ur.call"); xptiFrameworkFinalize();
  std::printf("{\"buffer_fixture\":\"PASS\",\"simulated_calls\":%u}\n", simulated_calls);
  return 0;
}

int overflow_fixture() {
  // CPU callbacks only. A two-row prefix plus 80000 callbacks exceeds even
  // the default cap. Check both concurrent cap contention and late callback
  // safety; after the cap, a bogus args_data must never be dereferenced.
  xptiFrameworkInitialize();
  if (xptiInitialize("ur.call", 0, 1, "overflow-host-fixture") != xpti::result_t::XPTI_RESULT_SUCCESS) return 6;
  const auto stream = xptiRegisterStream("ur.call");
  auto work = [stream] {
    for (unsigned i = 0; i < 10000; ++i) {
      xpti::function_with_args_t call{};
      call.function_id = UR_FUNCTION_PLATFORM_GET_API_VERSION;
      call.function_name = "urPlatformGetApiVersion";
      const auto instance = xptiGetUniqueId();
      if (xptiNotifySubscribers(stream, xpti::trace_function_with_args_begin, nullptr, nullptr,
                                instance, &call) != xpti::result_t::XPTI_RESULT_SUCCESS) std::exit(4);
      ur_result_t result = UR_RESULT_SUCCESS;
      call.ret_data = &result;
      if (xptiNotifySubscribers(stream, xpti::trace_function_with_args_end, nullptr, nullptr,
                                instance, &call) != xpti::result_t::XPTI_RESULT_SUCCESS) std::exit(5);
    }
  };
  std::thread threads[4];
  for (auto& thread : threads) thread = std::thread(work);
  for (auto& thread : threads) thread.join();
  notify(UR_FUNCTION_QUEUE_CREATE, "urQueueCreate", reinterpret_cast<void*>(uintptr_t(1)));
  xptiFinalize("ur.call");
  xptiFrameworkFinalize();
  std::puts("{\"overflow_fixture\":\"PASS\",\"simulated_calls\":40001,\"application_continued\":true}");
  return 0;
}

int overflow_boundary_fixture(bool end_boundary) {
  xptiFrameworkInitialize();
  if (xptiInitialize("ur.call", 0, 1, "boundary-host-fixture") != xpti::result_t::XPTI_RESULT_SUCCESS) return 6;
  const auto stream = xptiRegisterStream("ur.call");
  if (end_boundary) {
    // cap3: load,subscribe,valid begin. The very first omitted end has invalid
    // argument data and must not be decoded, even before capture_stopped was set.
    auto context = reinterpret_cast<ur_context_handle_t>(uintptr_t(0x100));
    ur_queue_handle_t* output = nullptr;
    ur_queue_create_params_t args{}; args.phContext = &context; args.pphQueue = &output;
    xpti::function_with_args_t call{};
    call.function_id = UR_FUNCTION_QUEUE_CREATE; call.function_name = "urQueueCreate"; call.args_data = &args;
    const auto instance = xptiGetUniqueId();
    if (xptiNotifySubscribers(stream, xpti::trace_function_with_args_begin, nullptr, nullptr,
                              instance, &call) != xpti::result_t::XPTI_RESULT_SUCCESS) return 4;
    ur_result_t result = UR_RESULT_SUCCESS;
    call.ret_data = &result; call.args_data = reinterpret_cast<void*>(uintptr_t(1));
    if (xptiNotifySubscribers(stream, xpti::trace_function_with_args_end, nullptr, nullptr,
                              instance, &call) != xpti::result_t::XPTI_RESULT_SUCCESS) return 5;
  } else {
    // cap4: load,subscribe,one complete pair,then an invalid boundary begin.
    notify(UR_FUNCTION_PLATFORM_GET_API_VERSION, "urPlatformGetApiVersion", nullptr);
    notify(UR_FUNCTION_QUEUE_CREATE, "urQueueCreate", reinterpret_cast<void*>(uintptr_t(1)));
  }
  xptiFinalize("ur.call"); xptiFrameworkFinalize();
  std::printf("{\"overflow_boundary_fixture\":\"PASS\",\"boundary\":\"%s\",\"application_continued\":true}\n",
              end_boundary ? "end" : "begin");
  return 0;
}
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const char* name = argv[1];
  if (!std::strcmp(name, "metadata")) return metadata_fixture();
  if (!std::strcmp(name, "buffers")) return buffer_fixture();
  if (!std::strcmp(name, "overflow")) return overflow_fixture();
  if (!std::strcmp(name, "overflow-boundary-begin")) return overflow_boundary_fixture(false);
  if (!std::strcmp(name, "overflow-boundary-end")) return overflow_boundary_fixture(true);
  if (std::strcmp(name, "setenv") && std::strcmp(name, "unsetenv") &&
      std::strcmp(name, "putenv") && std::strcmp(name, "clearenv")) return 2;
  std::printf("mutation-case-begin:%s\n", name);
  if (std::fflush(stdout)) return 3;
  char value[] = "STRATA_MUTATION_TEST=1";
  if (!std::strcmp(name, "setenv")) setenv("STRATA_MUTATION_TEST", "1", 1);
  else if (!std::strcmp(name, "unsetenv")) unsetenv("STRATA_MUTATION_TEST");
  else if (!std::strcmp(name, "putenv")) putenv(value);
  else clearenv();
  return 99; // Returning from a write means the interposition did not protect it.
}
