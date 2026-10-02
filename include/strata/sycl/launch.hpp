#pragma once

#include "strata/sycl/runtime.hpp"
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace strata::sycl_backend {

struct Span {
  const void *pointer;
  size_t bytes;
};
inline void validate_spans(std::initializer_list<Span> outputs,
                           std::initializer_list<Span> inputs) {
  auto valid = [](Span span) {
    const auto address = reinterpret_cast<uintptr_t>(span.pointer);
    if (!span.pointer || address % 4 || span.bytes > UINTPTR_MAX - address)
      throw std::invalid_argument("SYCL kernel requires aligned nonnull spans");
  };
  auto disjoint = [](Span a, Span b) {
    const auto ap = reinterpret_cast<uintptr_t>(a.pointer),
               bp = reinterpret_cast<uintptr_t>(b.pointer);
    if (ap < bp + b.bytes && bp < ap + a.bytes)
      throw std::invalid_argument(
          "SYCL kernel requires disjoint writable spans");
  };
  for (auto span : outputs)
    valid(span);
  for (auto span : inputs)
    valid(span);
  for (auto a = outputs.begin(); a != outputs.end(); ++a) {
    for (auto b = outputs.begin(); b != a; ++b)
      disjoint(*a, *b);
    for (auto b : inputs)
      disjoint(*a, b);
  }
}

// Existing kernel entry points carry an opaque stream. In a SYCL build this is
// a pointer to an in-order sycl::queue, never a native CUDA/Level Zero handle.
// A null stream uses the device-0 runtime and completes before returning.
inline sycl::queue &queue_for(void *stream) {
  if (!stream) {
    auto runtime = runtime_for();
    runtime->check();
    return runtime->compute();
  }
  auto &queue = *static_cast<sycl::queue *>(stream);
  if (!queue.has_property<sycl::property::queue::in_order>())
    throw std::invalid_argument(
        "Strata SYCL kernels require an in-order queue");
  return queue;
}

inline void finish(void *stream, sycl::event event) {
  if (!stream)
    runtime_for()->wait(event);
}

inline size_t checked_count(int64_t rows, int64_t cols) {
  if (rows < 0 || cols < 0 ||
      (cols && uint64_t(rows) > uint64_t(INT64_MAX) / uint64_t(cols)))
    throw std::invalid_argument("invalid SYCL tensor dimensions");
  return size_t(rows) * size_t(cols);
}

template <typename Function>
void for_each(int64_t count, void *stream, Function function) {
  if (count < 0)
    throw std::invalid_argument("negative SYCL element count");
  if (!count)
    return;
  auto event = queue_for(stream).parallel_for(
      sycl::range<1>(size_t(count)),
      [=](sycl::id<1> i) { function(size_t(i[0])); });
  finish(stream, event);
}
} // namespace strata::sycl_backend
