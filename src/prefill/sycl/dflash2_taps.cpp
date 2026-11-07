#include "strata/prefill/dflash2_taps.hpp"
#include "strata/core/dflash2.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/launch.hpp"

#include <algorithm>
#include <exception>

namespace strata::prefill {

Dflash2PrefillTaps::~Dflash2PrefillTaps() { release(); }

void Dflash2PrefillTaps::release() {
    if (data_) {
        const core::OnDevice selected(device_);
        // cudaFree's registered-allocation release drains every registered
        // queue before retiring this owner; no borrowed queue pointer survives.
        if (cudaFree(data_) != cudaSuccess)
            sycl_backend::cleanup_fail_stop("DFlash2 prompt tap retirement");
    }
    data_ = nullptr; device_ = -1;
    first_ = end_ = captured_ = 0;
    present_.clear();
}

bool Dflash2PrefillTaps::begin(int64_t n, int device, std::string& err) try {
    if (n < 1 || n > 262144 || device < 0 || (data_ && device != device_)) {
        err = "DFlash2 prompt taps: invalid prompt/device"; return false;
    }
    const core::OnDevice selected(device);
    if (data_ && cudaDeviceSynchronize() != cudaSuccess) {
        err = "DFlash2 prompt taps: previous readers did not drain"; return false;
    }
    if (!data_ && cudaMalloc(&data_, bytes()) != cudaSuccess) {
        err = "DFlash2 prompt taps: tail buffer does not fit"; return false;
    }
    device_ = device;
    first_ = std::max<int64_t>(0, n - kTail);
    end_ = n - 1;
    present_.assign(size_t(end_ - first_), 0);
    captured_ = 0;
    return true;
} catch (const std::exception& e) {
    err = std::string("DFlash2 prompt taps: ") + e.what(); return false;
}

bool Dflash2PrefillTaps::capture(int64_t layer, int64_t pos0, int64_t rows, const float* R,
                               const uint16_t* R_bf16, void* stream, std::string& err) try {
    const auto slot = std::find(std::begin(layers), std::end(layers), layer) - std::begin(layers);
    if (slot == kLayers) return true;
    if (!data_ || !stream || pos0 < 0 || pos0 > 262144 || rows < 0 || rows > 262144 - pos0) {
        err = "DFlash2 prompt taps: invalid residual extent/owner"; return false;
    }
    const int64_t lo = std::max(pos0, first_), hi = std::min(pos0 + rows, end_);
    if (hi <= lo) return true;
    if ((R == nullptr) == (R_bf16 == nullptr)) {
        err = "DFlash2 prompt taps: exactly one residual format is required"; return false;
    }
    auto& q = sycl_backend::queue_for(stream);
    const int64_t stride = kLayers * kWidth, n = hi - lo;
    float* out = data_ + (lo - first_) * stride + slot * kWidth;
    sycl_backend::validate_device_span(out, size_t((n - 1) * stride + kWidth) * sizeof(float), q);
    // Validate the supplied owner BEFORE forming an offset pointer into it.
    const size_t used = size_t(hi - pos0) * kStreams * kWidth;
    if (R) {
        sycl_backend::validate_device_span(R, used * sizeof(float), q);
        kernels::df2_tap_rows(R + (lo - pos0) * kStreams * kWidth, out, stride,
                             kWidth, kStreams, int(n), stream);
    } else {
        sycl_backend::validate_device_span(R_bf16, used * sizeof(uint16_t), q);
        const uint16_t* input = R_bf16 + (lo - pos0) * kStreams * kWidth;
        q.parallel_for(sycl::range<2>(size_t(n), size_t(kWidth)), [=](sycl::id<2> at) {
            float sum = 0;
            for (int c = 0; c < kStreams; ++c)
                sum += sycl::bit_cast<float>(uint32_t(input[(at[0] * kStreams + c) * kWidth + at[1]]) << 16);
            out[at[0] * stride + at[1]] = sum / float(kStreams);
        });
    }
    for (int64_t pos = lo; pos < hi; ++pos) present_[size_t(pos - first_)] |= uint8_t(1u << slot);
    return true;
} catch (const std::exception& e) {
    err = std::string("DFlash2 prompt tap capture: ") + e.what(); return false;
}

bool Dflash2PrefillTaps::capture_window(const float* taps, int64_t rows, int64_t pos0, std::string& err) try {
    if (!data_ || pos0 < 0 || pos0 > 262144 || rows < 0 || rows > 262144 - pos0) {
        err = "DFlash2 prompt window: invalid extent/owner"; return false;
    }
    const int64_t lo = std::max(pos0, first_), hi = std::min(pos0 + rows, end_);
    if (hi <= lo) return true;
    const core::OnDevice selected(device_);
    auto runtime = sycl_backend::runtime_for(device_);
    constexpr int64_t stride = kLayers * kWidth;
    sycl_backend::validate_device_span(taps, size_t(hi - pos0) * stride * sizeof(float), runtime->compute());
    if (cudaMemcpy(data_ + (lo - first_) * stride, taps + (lo - pos0) * stride,
                   size_t(hi - lo) * stride * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
        err = "DFlash2 prompt window: copy/completion failed"; return false;
    }
    for (int64_t pos = lo; pos < hi; ++pos) present_[size_t(pos - first_)] = (1u << kLayers) - 1;
    return true;
} catch (const std::exception& e) {
    err = std::string("DFlash2 prompt window: ") + e.what(); return false;
}

bool Dflash2PrefillTaps::ingest(core::Dflash2Drafter& drafter, std::string& err) try {
    if (!data_ || drafter.n_embd() != kWidth || drafter.tap_layers() != std::vector<int>(std::begin(layers), std::end(layers))) {
        err = "DFlash2 prompt taps: drafter/owner mismatch"; return false;
    }
    const core::OnDevice selected(device_);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = "DFlash2 prompt taps: producer completion failed"; return false;
    }
    captured_ = 0;
    for (size_t i = 0; i < present_.size();) {
        if (present_[i] != (1u << kLayers) - 1) { ++i; continue; }
        const size_t start = i;
        while (i < present_.size() && present_[i] == (1u << kLayers) - 1) ++i;
        if (!drafter.context(data_ + start * kLayers * kWidth, int64_t(i - start), first_ + int64_t(start), err)) return false;
        captured_ += int64_t(i - start);
    }
    return true;
} catch (const std::exception& e) {
    err = std::string("DFlash2 prompt tap ingestion: ") + e.what(); return false;
}

} // namespace strata::prefill
