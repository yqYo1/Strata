// SPDX-FileCopyrightText: 2026 yayoi
// SPDX-License-Identifier: MIT
#include "strata/program/dense_prefill_nll.hpp"

namespace strata::program::dense_prefill_nll {
bool target_loss(const float* values, size_t vocab, size_t target, double& out) {
    if (!values || vocab == 0 || target >= vocab) return false;
    double maximum = -std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < vocab; ++j) {
        if (!std::isfinite(values[j])) return false;
        maximum = std::max(maximum, double(values[j]));
    }
    double sum = 0.0, correction = 0.0;
    for (size_t j = 0; j < vocab; ++j) {
        const double term = std::exp(double(values[j]) - maximum) - correction;
        const double next = sum + term;
        correction = (next - sum) - term;
        sum = next;
    }
    out = (maximum - double(values[target])) + std::log(sum);
    return std::isfinite(out) && out >= 0.0;
}
} // namespace strata::program::dense_prefill_nll
