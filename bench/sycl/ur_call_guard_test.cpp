// Main-authored host-only mutation controls; repository LICENSE applies.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
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
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const char* name = argv[1];
  if (!std::strcmp(name, "metadata")) return metadata_fixture();
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
