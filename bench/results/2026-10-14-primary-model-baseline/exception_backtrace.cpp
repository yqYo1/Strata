// Linux-only diagnostic preload. Observe SYCL throws without changing their
// arguments, type, destructor, catch behavior, or kernel/runtime parameters.
#include <dlfcn.h>
#include <execinfo.h>
#include <cstring>
#include <typeinfo>
#include <unistd.h>

namespace {
using Throw = void (*)(void *, void *, void (*)(void *));
Throw original = nullptr;
thread_local bool reporting = false;
__attribute__((constructor)) void resolve_throw() {
  original = reinterpret_cast<Throw>(dlsym(RTLD_NEXT, "__cxa_throw"));
  if (!original)
    _exit(125);
}
}

extern "C" [[noreturn]] void __cxa_throw(void *object, void *type,
                                       void (*destroy)(void *)) {
  const char *name = static_cast<const std::type_info *>(type)->name();
  if (!reporting && name && std::strstr(name, "sycl")) {
    reporting = true;
    constexpr char prefix[] = "strata diagnostic SYCL throw: ";
    (void)!write(STDERR_FILENO, prefix, sizeof(prefix) - 1);
    (void)!write(STDERR_FILENO, name, std::strlen(name));
    (void)!write(STDERR_FILENO, "\n", 1);
    void *frames[48];
    int count = backtrace(frames, 48);
    backtrace_symbols_fd(frames, count, STDERR_FILENO);
    reporting = false;
  }
  original(object, type, destroy);
  _exit(126);
}
