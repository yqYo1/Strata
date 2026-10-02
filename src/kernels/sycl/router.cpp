#include "strata/kernels/router_top10.hpp"
#include "strata/sycl/launch.hpp"

#include <limits>

namespace strata::kernels {

void router_top10(const float *logits, int tokens, int experts, int k, int *ids,
                  float *weights, void *stream) {
  if (tokens < 0 || experts <= 0 || k < 1 || k > 64 || k > experts)
    throw std::invalid_argument("invalid SYCL router geometry");
  if (!tokens)
    return;
  auto &queue = sycl_backend::queue_for(stream);
  const auto local_bytes =
      size_t(experts) * (sizeof(double) + sizeof(float)) + sizeof(double);
  if (local_bytes >
      queue.get_device().get_info<sycl::info::device::local_mem_size>())
    throw std::invalid_argument("SYCL router exceeds available local memory");
  constexpr size_t width = 128;
  const float negative_inf = -std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  auto event = queue.submit([&](sycl::handler &handler) {
    sycl::local_accessor<double> exponentials(experts, handler),
        total(1, handler);
    sycl::local_accessor<float> probabilities(experts, handler);
    handler.parallel_for(
        sycl::nd_range<1>(size_t(tokens) * width, width),
        [=](sycl::nd_item<1> item) {
          const size_t token = item.get_group_linear_id();
          const size_t lane = item.get_local_linear_id();
          const auto group = item.get_group();
          float maximum = negative_inf;
          for (size_t e = lane; e < size_t(experts); e += width)
            maximum = sycl::fmax(maximum, logits[token * experts + e]);
          maximum =
              sycl::reduce_over_group(group, maximum, sycl::maximum<float>());
          for (size_t e = lane; e < size_t(experts); e += width)
            exponentials[e] = sycl::exp(double(logits[token * experts + e]) -
                                        double(maximum));
          sycl::group_barrier(group);
          if (!lane) {
            double sum = 0;
            for (int e = 0; e < experts; ++e)
              sum += exponentials[e];
            total[0] = sum;
          }
          sycl::group_barrier(group);
          const float inverse = float(1.0 / total[0]);
          for (size_t e = lane; e < size_t(experts); e += width)
            probabilities[e] = float(exponentials[e] * inverse);
          sycl::group_barrier(group);
          for (int rank = 0; rank < k; ++rank) {
            float best = negative_inf;
            for (size_t e = lane; e < size_t(experts); e += width)
              best = sycl::fmax(best, probabilities[e]);
            best = sycl::reduce_over_group(group, best, sycl::maximum<float>());
            int index = experts;
            for (size_t e = lane; e < size_t(experts); e += width)
              if (probabilities[e] == best)
                index = sycl::min(index, int(e));
            index = sycl::reduce_over_group(group, index, sycl::minimum<int>());
            if (!lane) {
              ids[token * k + rank] = index < experts ? index : -1;
              weights[token * k + rank] = index < experts ? best : nan;
              if (index < experts)
                probabilities[index] = negative_inf;
            }
            sycl::group_barrier(group);
          }
          if (!lane) {
            double sum = 0;
            for (int rank = 0; rank < k; ++rank)
              sum += double(weights[token * k + rank]);
            sum = sycl::fmax(sum, 0x1p-14);
            for (int rank = 0; rank < k; ++rank)
              weights[token * k + rank] =
                  float(double(weights[token * k + rank]) / sum);
          }
        });
  });
  sycl_backend::finish(stream, event);
}

bool router_top10_variant(const float *, int, int, int, int *, float *, void *,
                          int) {
  return false;
}
} // namespace strata::kernels
