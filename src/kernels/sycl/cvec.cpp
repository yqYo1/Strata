#include "strata/kernels/cvec.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <memory>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
Cvec description;
bool enabled = false;
struct Tables {
  Allocation direction, scale, on;
  Tables(std::shared_ptr<Runtime> rt, size_t n, size_t layers)
      : direction(rt, n * 4, MemoryKind::Device),
        scale(rt, layers * 4, MemoryKind::Device),
        on(rt, 4, MemoryKind::Device) {}
};
std::unique_ptr<Tables> tables;
void drain() {
  auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
float sum32(sycl::sub_group sg, float x) {
  for (int d = 16; d; d >>= 1)
    x += sycl::permute_group_by_xor(sg, x, d);
  return x;
}
} // namespace
const Cvec &cvec() { return description; }
bool cvec_upload(const std::vector<float> &dir, const std::vector<float> &scale,
                 int mode, int first, int last, int64_t width, int64_t hc,
                 std::string &err) {
  try {
    if (width <= 0 || width > 4096 || hc <= 0 || hc > INT32_MAX ||
        (mode != 0 && mode != 1) || scale.empty() ||
        dir.size() != checked_count(scale.size(), width) || first < 0 ||
        last < first || size_t(last) >= scale.size() ||
        !std::all_of(
            dir.begin(), dir.end(), [](float v) { return std::isfinite(v); }) ||
        !std::all_of(
            scale.begin(), scale.end(),
            [](float v) { return std::isfinite(v); }))
      throw std::invalid_argument(
          "invalid SYCL control-vector shape or values");
    drain();
    auto rt = runtime_for();
    auto next = std::make_unique<Tables>(rt, dir.size(), scale.size());
    int on = 1;
    auto &q = rt->compute();
    q.memcpy(next->direction.data(), dir.data(), dir.size() * 4);
    q.memcpy(next->scale.data(), scale.data(), scale.size() * 4);
    rt->wait(q.memcpy(next->on.data(), &on, 4));
    Cvec desc;
    desc.dir = next->direction.as<float>();
    desc.s = next->scale.as<float>();
    desc.on = next->on.as<int>();
    desc.mode = mode;
    desc.first = first;
    desc.last = last;
    desc.n_embd = width;
    desc.hc = hc;
    desc.steered.resize(scale.size());
    for (size_t i = 0; i < scale.size(); ++i)
      desc.steered[i] = scale[i] != 0;
    tables = std::move(next);
    description = std::move(desc);
    enabled = true;
    return true;
  } catch (const std::exception &e) {
    err = e.what();
    return false;
  }
}
bool cvec_replicate(std::string &) {
  return true;
} // Engine execution uses runtime device zero.
void cvec_set_enabled(bool value) {
  if (!description.loaded() || enabled == value)
    return;
  drain();
  int on = value;
  auto rt = runtime_for();
  rt->wait(rt->compute().memcpy(tables->on.data(), &on, 4));
  enabled = value;
}
bool cvec_enabled() { return description.loaded() && enabled; }
void cvec_apply(float *R, int64_t layer, int64_t T, int64_t r_ld,
                const float *bo, int64_t bo_ld, const float *inj,
                int64_t inj_ld, bool write, void *stream) {
  if (!description.loaded() || T == 0)
    return;
  const int n = int(description.n_embd), hc = int(description.hc),
            mode = description.mode;
  if (layer < 0 || size_t(layer) >= description.steered.size() || T < 0 ||
      r_ld < int64_t(n) * hc || (write && (bo_ld < n || inj_ld < hc)))
    throw std::invalid_argument("invalid SYCL control-vector apply geometry");
  const float *dir = description.dir + size_t(layer) * n;
  const float *scale = description.s + layer;
  const int *on = description.on;
  size_t count = checked_count(T, r_ld);
  if (count > SIZE_MAX / 4)
    throw std::invalid_argument("control-vector span overflow");
  validate_spans({{R, count * 4}}, {{dir, size_t(n) * 4}, {scale, 4}, {on, 4}});
  if (write)
    validate_spans({{R, count * 4}}, {{bo, checked_count(T, bo_ld) * 4},
                                      {inj, checked_count(T, inj_ld) * 4}});
  auto &q = queue_for(stream);
  if (q.get_context() != runtime_for()->context())
    throw std::invalid_argument("control vectors belong to SYCL device zero");
  auto event = q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> part(8, h);
    h.parallel_for(
        sycl::nd_range<1>(checked_count(T, hc) * 256, 256),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          size_t row = it.get_group_linear_id(), t = row / hc;
          int c = row % hc, tid = it.get_local_linear_id(), lane = tid % 32;
          float *s = R + t * r_ld + size_t(c) * n;
          bool steer = *on && *scale != 0;
          if (!steer && !write)
            return;
          float gate =
              write ? 2.f / (1.f + sycl::exp(-inj[t * inj_ld + c] / float(hc)))
                    : 0;
          float x[16], dot = 0;
          for (int k = 0; k < 16; ++k) {
            int d = tid + k * 256;
            if (d < n) {
              float v = s[d];
              if (write)
                v = sycl::fma(bo[t * bo_ld + d], gate, v);
              x[k] = v;
              if (steer && mode == 0)
                dot = sycl::fma(v, dir[d], dot);
            }
          }
          if (steer && mode == 0) {
            dot = sum32(it.get_sub_group(), dot);
            if (!lane)
              part[tid / 32] = dot;
            it.barrier(sycl::access::fence_space::local_space);
            dot = sum32(it.get_sub_group(), lane < 8 ? part[lane] : 0.f) *
                  (*scale);
          }
          for (int k = 0; k < 16; ++k) {
            int d = tid + k * 256;
            if (d < n) {
              float v = x[k];
              if (steer)
                v = mode == 0 ? sycl::fma(-dot, dir[d], v) : v + dir[d];
              s[d] = v;
            }
          }
        });
  });
  finish(stream, event);
}
} // namespace strata::kernels
