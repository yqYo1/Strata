// Authored for Strata; repository LICENSE (MIT) applies.
#pragma once
#include "strata/sycl/command_profile_hook.hpp"
#include <cstdint>
#include <memory>
namespace strata::sycl_backend::command_profile {
enum class Kind : uint8_t { none, upload, dequant_gu, dequant_down, gather, gu,
                            swiglu, down, combine, write, accumulate,
                            job_weights, status, gu_swiglu, down_scatter, count };
struct Meta {
  Kind kind = Kind::none;
  int32_t expert = -1, rows = 0, n = 0, k = 0, format = -1;
  int64_t source_row = -1;
  bool selected = false;
};
bool requested(); // Startup-only, default off; never changes process environment.
class Frame;
class Tag;
class Trace {
public:
  explicit Trace(void* stream, unsigned expected_frames = 4);
  ~Trace() noexcept;
  Trace(const Trace&) = delete;
  Trace& operator=(const Trace&) = delete;
  Frame begin(int64_t layer, int64_t first_row);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class Frame;
  friend class Tag;
  friend Tag fused_tag(Kind, Kind) noexcept;
  friend void observe_impl(Trace&, void*, const sycl::event&) noexcept;
};
// Only selected first windows in layers0,15,31,47 activate a trace. Planning
// reads existing CPU route metadata, never new GPU tensor readbacks.
class Frame {
public:
  Frame() = default;
  ~Frame() noexcept;
  Frame(const Frame&) = delete;
  Frame& operator=(const Frame&) = delete;
  explicit operator bool() const noexcept { return trace_ != nullptr; }
  void consider(int32_t expert, int64_t global_row, int64_t rows) noexcept;
  Tag tag(Kind, int32_t expert, int64_t global_row, int64_t rows,
          int64_t n = 0, int64_t k = 0, int32_t format = -1) noexcept;
  // Caller must have completed its existing checked window drain. On failure
  // keep event owners in Trace until the outer lease's checked retirement.
  void after_checked_drain() noexcept;
private:
  explicit Frame(Trace* trace) : trace_(trace) {}
  Trace* trace_ = nullptr;
  bool retired_ = false;
  friend class Trace;
};
class Tag {
public:
  ~Tag() noexcept;
  Tag(const Tag&) = delete;
  Tag& operator=(const Tag&) = delete;
private:
  Tag(Trace*, Meta) noexcept;
  Trace* trace_;
  Meta previous_;
  friend class Frame;
  friend Tag fused_tag(Kind, Kind) noexcept;
};
// Relabel the event already published by finish(), preserving the outer job's
// sampling choice and shape. No event registration, GPU query or wait is added.
// Inactive tracing is a no-op. An incompatible active scope invalidates Trace.
Tag fused_tag(Kind expected, Kind replacement) noexcept;
} // namespace strata::sycl_backend::command_profile
