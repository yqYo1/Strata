// Authored for Strata; repository LICENSE (MIT) applies.
#pragma once
#include <sycl/sycl.hpp>
namespace strata::sycl_backend::command_profile {
class Trace;
extern thread_local Trace* active;
void observe_impl(Trace&, void* stream, const sycl::event&) noexcept;
inline void observe(void* stream, const sycl::event& event) noexcept {
  if (active) observe_impl(*active, stream, event);
}
} // namespace strata::sycl_backend::command_profile
