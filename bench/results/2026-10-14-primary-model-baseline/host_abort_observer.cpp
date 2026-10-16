// Linux diagnostic preload. Forward original termination/throw operations.
// This observes only the calling thread; it is not a signal handler or a
// complete-thread dump, and unwinding an already corrupt stack may fail.
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <sys/syscall.h>
#include <typeinfo>
#include <unistd.h>

namespace {
using Throw = void (*)(void *, void *, void (*)(void *));
using Terminal = void (*)();
Throw original_throw = nullptr;
Terminal original_abort = nullptr, original_stack_fail = nullptr;
thread_local bool reporting = false;

void report(const char *operation) {
  if (reporting) return;
  reporting = true;
  char line[160];
  const int length = std::snprintf(line, sizeof line,
      "strata host diagnostic: %s pid=%ld tid=%ld\n", operation,
      static_cast<long>(getpid()), syscall(SYS_gettid));
  if (length > 0 && length < static_cast<int>(sizeof line))
    (void)!write(STDERR_FILENO, line, static_cast<size_t>(length));
  void *frames[32];
  const int count = backtrace(frames, 32);
  backtrace_symbols_fd(frames, count, STDERR_FILENO);
  reporting = false;
}

__attribute__((constructor)) void resolve_originals() {
  original_throw = reinterpret_cast<Throw>(dlsym(RTLD_NEXT, "__cxa_throw"));
  original_abort = reinterpret_cast<Terminal>(dlsym(RTLD_NEXT, "abort"));
  original_stack_fail = reinterpret_cast<Terminal>(dlsym(RTLD_NEXT, "__stack_chk_fail"));
  if (!original_throw || !original_abort || !original_stack_fail) _exit(125);
  // Load the unwinder before a failure instead of first loading it in report().
  void *frame[1];
  (void) backtrace(frame, 1);
}
}

extern "C" [[noreturn]] void __cxa_throw(void *object, void *type,
                                       void (*destroy)(void *)) {
  const char *name = static_cast<const std::type_info *>(type)->name();
  if (name && std::strstr(name, "sycl")) report("SYCL throw");
  original_throw(object, type, destroy);
  _exit(126);
}

extern "C" [[noreturn]] void abort() noexcept {
  report("abort");
  original_abort();
  _exit(126);
}

extern "C" [[noreturn]] void __stack_chk_fail() {
  report("__stack_chk_fail");
  original_stack_fail();
  _exit(126);
}
