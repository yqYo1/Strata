// Main-authored fixture-only event seam; repository LICENSE (MIT) applies.
// Generated observed selector TUs replace only finish(stream,event) calls.
// The clean production-source objects never include or call this seam.
#pragma once
#include "strata/sycl/launch.hpp"
#include <optional>
#include <stdexcept>

namespace strata::qsa_fixture {
struct Capture;
inline thread_local Capture *active_capture = nullptr;
struct Capture {
  std::optional<sycl::event> submitted;
  unsigned calls = 0;
  Capture() {
    if (active_capture) throw std::logic_error("nested selector event capture");
    active_capture = this;
  }
  Capture(const Capture &) = delete;
  Capture &operator=(const Capture &) = delete;
  ~Capture() { active_capture = nullptr; }
  sycl::event take() {
    if (active_capture != this || calls != 1 || !submitted)
      throw std::runtime_error("expected exactly one actual selector submission");
    active_capture = nullptr;
    return *submitted;
  }
};
inline void observed_finish(void *stream, sycl::event event) {
  if (active_capture) {
    ++active_capture->calls;
    active_capture->submitted.emplace(event);
  }
  sycl_backend::finish(stream, event);
}
} // namespace strata::qsa_fixture
