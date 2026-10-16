// Linux opt-in host-failure marker. Terminal hooks neither unwind nor symbolize
// a possibly damaged stack. The entry's return address is only a raw caller PC.
#include <cstdint>
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

void marker(const char *operation, void *caller) {
  if (reporting) return;
  reporting = true;
  char line[160];
  size_t used = 0;
  auto literal = [&](const char *s) {
    while (*s && used < sizeof line) line[used++] = *s++;
  };
  auto number = [&](uint64_t value, unsigned radix) {
    char digits[32];
    size_t count = 0;
    do {
      digits[count++] = "0123456789abcdef"[value % radix];
      value /= radix;
    } while (value);
    while (count && used < sizeof line) line[used++] = digits[--count];
  };
  literal("strata host diagnostic: "); literal(operation);
  literal(" pid="); number(static_cast<uint64_t>(getpid()), 10);
  literal(" tid="); number(static_cast<uint64_t>(syscall(SYS_gettid)), 10);
  literal(" caller=0x"); number(reinterpret_cast<uintptr_t>(caller), 16);
  literal("\n");
  (void)!write(STDERR_FILENO, line, used);
  reporting = false;
}

__attribute__((constructor)) void resolve_originals() {
  original_throw = reinterpret_cast<Throw>(dlsym(RTLD_NEXT, "__cxa_throw"));
  original_abort = reinterpret_cast<Terminal>(dlsym(RTLD_NEXT, "abort"));
  original_stack_fail = reinterpret_cast<Terminal>(dlsym(RTLD_NEXT, "__stack_chk_fail"));
  if (!original_throw || !original_abort || !original_stack_fail) _exit(125);
  void *frame[1];
  (void) backtrace(frame, 1); // only the separate ordinary C++ throw path unwinds
}
}

extern "C" [[noreturn]] void __cxa_throw(void *object, void *type,
                                       void (*destroy)(void *)) {
  const char *name = static_cast<const std::type_info *>(type)->name();
  if (!reporting && name && std::strstr(name, "sycl")) {
    marker("SYCL throw", __builtin_return_address(0));
    reporting = true;
    void *frames[32];
    const int count = backtrace(frames, 32);
    backtrace_symbols_fd(frames, count, STDERR_FILENO);
    reporting = false;
  }
  original_throw(object, type, destroy);
  _exit(126);
}

extern "C" [[noreturn]] void abort() noexcept {
  marker("abort", __builtin_return_address(0));
  original_abort();
  _exit(126);
}

extern "C" [[noreturn]] void __stack_chk_fail() {
  marker("__stack_chk_fail", __builtin_return_address(0));
  original_stack_fail();
  _exit(126);
}
