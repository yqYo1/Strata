#pragma once
#include <cstdio>
#include <cstdlib>
#include <exception>

// Declare AFTER every ordinary-host buffer used by GPU copies. Throwing out
// of the arm must not release those buffers before the process fail-stop.
// Disarm only after the stream and all normal cleanup have completed.
class TransferFailureGuard {
  bool closed_ = false;
public:
  void closed() noexcept { closed_ = true; }
  ~TransferFailureGuard() noexcept {
    if (!closed_ && std::uncaught_exceptions() != 0) {
      std::fputs("transfer: exception before checked cleanup; preserving copy buffers until process fail-stop\n", stderr);
      std::fflush(stderr);
      std::_Exit(1);
    }
  }
};
