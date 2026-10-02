#include "strata/kernels/gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/sycl/runtime.hpp"
#include <cfenv>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> runtime;
template <typename T> struct Buffer {
  sycl_backend::Allocation allocation;
  size_t count;
  explicit Buffer(size_t n)
      : allocation(runtime, n * sizeof(T), sycl_backend::MemoryKind::Device),
        count(n) {}
  T *data() { return allocation.as<T>(); }
  void upload(const std::vector<T> &values) {
    if (values.size() != count)
      throw std::logic_error("upload count mismatch");
    runtime->wait(
        runtime->compute().memcpy(data(), values.data(), allocation.size()));
  }
  std::vector<T> read() {
    std::vector<T> values(count);
    runtime->wait(
        runtime->compute().memcpy(values.data(), data(), allocation.size()));
    return values;
  }
};
void close(float actual, double expected, double tolerance, const char *label) {
  if (!std::isfinite(actual) ||
      std::abs(actual - expected) > tolerance * (1 + std::abs(expected)))
    throw std::runtime_error(std::string(label) +
                             ": actual=" + std::to_string(actual) +
                             ", expected=" + std::to_string(expected));
}

void recurrence(int width, int hk, int hv, bool native) {
  const int n = width * hv;
  std::vector<float> initial(width * n);
  // The CPU oracle stores [head,column,row], independently of the GPU's
  // [row,head,column] layout. Nonuniform data reveals a transposed state.
  std::vector<double> state(initial.size());
  for (int h = 0; h < hv; ++h)
    for (int j = 0; j < width; ++j)
      for (int i = 0; i < width; ++i) {
        const float value = float(std::sin(i * .13 + j * .37 + h * .71) * .01);
        initial[(i * hv + h) * width + j] = value;
        state[(h * width + j) * width + i] = value;
      }
  Buffer<float> gpu_state(initial.size()), q(width * hk), k(width * hk), v(n),
      gate(hv), beta(hv), output(n);
  gpu_state.upload(initial);
  for (int step = 0; step < 6; ++step) {
    std::vector<float> queries(width * hk), keys(width * hk), values(n),
        gates(hv), betas(hv);
    for (int h = 0; h < hk; ++h) {
      double qsum = 0, ksum = 0;
      for (int i = 0; i < width; ++i) {
        queries[h * width + i] =
            float(std::sin(i * .19 + h * .41 + step * .17));
        keys[h * width + i] = float(std::cos(i * .23 + h * .73 + step * .31));
        qsum += double(queries[h * width + i]) * queries[h * width + i];
        ksum += double(keys[h * width + i]) * keys[h * width + i];
      }
      for (int i = 0; i < width; ++i) {
        queries[h * width + i] =
            float(queries[h * width + i] / std::sqrt(qsum + 1e-6) /
                  (native ? 1. : std::sqrt(double(width))));
        keys[h * width + i] =
            float(keys[h * width + i] / std::sqrt(ksum + 1e-6));
      }
    }
    for (int h = 0; h < hv; ++h) {
      gates[h] = -.01f * (h + step + 1);
      betas[h] = .1f + .02f * (h % 9);
      for (int j = 0; j < width; ++j)
        values[h * width + j] = float(std::sin(j * .3 + h * .5 + step));
    }
    q.upload(queries);
    k.upload(keys);
    v.upload(values);
    gate.upload(gates);
    beta.upload(betas);
    const GdnShapes shape{width, hk, hv};
    if (native)
      native_gdn_step(gpu_state.data(), q.data(), k.data(), v.data(),
                      gate.data(), beta.data(), output.data(), shape,
                      &runtime->compute());
    else
      gdn_step(gpu_state.data(), q.data(), k.data(), v.data(), gate.data(),
               beta.data(), output.data(), shape, &runtime->compute());
    const auto actual_output = output.read(), actual_state = gpu_state.read();
    for (int h = 0; h < hv; ++h)
      for (int j = 0; j < width; ++j) {
        auto *column = state.data() + (h * width + j) * width;
        const int source = h % hk;
        double projected = 0;
        for (int i = 0; i < width; ++i) {
          column[i] *= std::exp(double(gates[h]));
          projected += column[i] * keys[source * width + i];
        }
        const double delta = (values[h * width + j] - projected) * betas[h];
        double result = 0;
        for (int i = 0; i < width; ++i) {
          column[i] += keys[source * width + i] * delta;
          result += column[i] * queries[source * width + i];
          close(actual_state[(i * hv + h) * width + j], column[i], 3e-6,
                "GDN state");
        }
        if (native)
          result /= std::sqrt(double(width));
        close(actual_output[h * width + j], result, 3e-6, "GDN readout");
      }
  }
  std::cout << "GDN " << (native ? "native" : "generic") << " S=" << width
            << " heads=" << hk << '/' << hv
            << ": six changing state updates PASS\n";
}

void convolution(bool native, int channels) {
  std::vector<float> history(channels * 3 + 1), weights(channels * 4),
      input(channels), sentinel(channels + 1, 321.f);
  for (size_t i = 0; i < history.size(); ++i)
    history[i] = float(std::sin(i * .1));
  for (size_t i = 0; i < weights.size(); ++i)
    weights[i] = float(std::cos(i * .17) * .2);
  history.back() = 789.f;
  Buffer<float> h(history.size()), w(weights.size()), x(input.size()),
      raw(sentinel.size()), activated(sentinel.size());
  h.upload(history);
  w.upload(weights);
  raw.upload(sentinel);
  activated.upload(sentinel);
  for (int step = 0; step < 5; ++step) {
    for (int c = 0; c < channels; ++c)
      input[c] = float(std::sin(c * .23 + step * .41));
    x.upload(input);
    if (native)
      native_gdn_conv_silu(h.data(), x.data(), w.data(), raw.data(),
                           activated.data(), channels, 4, &runtime->compute());
    else
      gdn_conv_step(h.data(), x.data(), w.data(), raw.data(), channels, 4,
                    &runtime->compute());
    const auto result = raw.read();
    const auto silu = native ? activated.read() : std::vector<float>{};
    for (int c = 0; c < channels; ++c) {
      double expected = double(input[c]) * weights[c * 4 + 3];
      for (int t = 0; t < 3; ++t)
        expected += double(history[c * 3 + t]) * weights[c * 4 + t];
      close(result[c], expected, 2e-6, "GDN convolution");
      if (native)
        close(silu[c], expected / (1 + std::exp(-expected)), 2e-6,
              "GDN conv SiLU");
      history[c * 3] = history[c * 3 + 1];
      history[c * 3 + 1] = history[c * 3 + 2];
      history[c * 3 + 2] = input[c];
    }
    if (h.read() != history || result.back() != 321.f ||
        (native && silu.back() != 321.f))
      throw std::runtime_error("convolution history or canary mismatch");
  }
  std::cout << "GDN convolution channels=" << channels << " native=" << native
            << ": history/canaries PASS\n";
}

void normalization(bool native) {
  constexpr int width = 128, rows = 3;
  std::vector<float> input(width * rows), z(input.size()), gamma(width);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = i < width * 2 ? float(std::sin(i * .13)) : 0.f;
    z[i] = float(std::cos(i * .19) * 3);
  }
  for (int c = 0; c < width; ++c)
    gamma[c] = .25f + c * .003f;
  Buffer<float> x(input.size()), gate(z.size()), weight(gamma.size()),
      y(input.size());
  x.upload(input);
  gate.upload(z);
  weight.upload(gamma);
  if (native)
    native_gdn_l2_norm(x.data(), rows, width, 1e-6f, &runtime->compute());
  else
    gdn_l2_norm(x.data(), rows, width, 1e-6f, &runtime->compute());
  auto result = x.read();
  for (int row = 0; row < rows; ++row) {
    double sum = 0;
    for (int c = 0; c < width; ++c)
      sum += double(input[row * width + c]) * input[row * width + c];
    for (int c = 0; c < width; ++c)
      close(result[row * width + c],
            input[row * width + c] / std::sqrt(sum + 1e-6), 2e-6, "GDN L2");
  }
  x.upload(input);
  if (native)
    native_gdn_out_norm(x.data(), gate.data(), weight.data(), y.data(), rows,
                        width, 1e-6f, &runtime->compute());
  else
    gdn_out_norm(x.data(), gate.data(), weight.data(), y.data(), rows, width,
                 1e-6f, &runtime->compute());
  result = y.read();
  for (int row = 0; row < rows; ++row) {
    double sum = 0;
    for (int c = 0; c < width; ++c)
      sum += double(input[row * width + c]) * input[row * width + c];
    for (int c = 0; c < width; ++c)
      close(result[row * width + c],
            input[row * width + c] / std::sqrt(sum / width + 1e-6) * gamma[c] /
                (1 + std::exp(-double(z[row * width + c]))),
            2e-6, "GDN output sigmoid norm");
  }
  std::cout << "GDN L2 and output RMS/sigmoid native=" << native << ": PASS\n";
}

void gates(bool native) {
  const std::vector<float> input{-100.f, -20.f, -1.f, 0.f, 1.f, 20.f, 100.f};
  const std::vector<float> dt(input.size(), .25f), a(input.size(), -.5f);
  Buffer<float> x(input.size()), d(input.size()), scale(input.size()),
      y(input.size());
  x.upload(input);
  d.upload(dt);
  scale.upload(a);
  if (native)
    native_gdn_beta_gate(x.data(), input.size(), &runtime->compute());
  else
    gdn_beta_gate(x.data(), input.size(), &runtime->compute());
  auto result = x.read();
  for (size_t i = 0; i < input.size(); ++i)
    close(result[i], 1. / (1. + std::exp(-double(input[i]))), 2e-7,
          "GDN beta sigmoid");
  if (native) {
    x.upload(input);
    native_gdn_gate(x.data(), d.data(), scale.data(), y.data(), input.size(),
                    &runtime->compute());
    result = y.read();
    for (size_t i = 0; i < input.size(); ++i) {
      const float value = input[i] + dt[i];
      close(result[i],
            -.5 * (value > 20 ? value : std::log1p(std::exp(double(value)))),
            2e-6, "native GDN softplus gate");
    }
    bool rejected = false;
    try {
      native_gdn_gate(x.data(), d.data(), scale.data(), x.data(), input.size(),
                      &runtime->compute());
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("overlapping native gate output accepted");
  }
}
} // namespace

int main() {
  try {
    std::fesetenv(FE_DFL_ENV);
    runtime = sycl_backend::runtime_for();
    recurrence(17, 2, 6, false);
    recurrence(128, 2, 6, false);
    recurrence(128, 2, 6, true);
    recurrence(128, 16, 48, true);
    for (bool native : {false, true}) {
      convolution(native, 37);
      convolution(native, 10240);
      normalization(native);
      gates(native);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
