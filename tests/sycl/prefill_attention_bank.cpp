// Main-authored fixture. Repository LICENSE applies.
#include "strata/prefill/native16.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/compat/cuda_runtime.h"
#include "strata/sycl/runtime.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using strata::sycl_backend::Allocation;
using strata::sycl_backend::MemoryKind;
using strata::sycl_backend::runtime_for;
using strata::sycl_backend::checked_cleanup;
namespace native = strata::prefill::native16;
using Clock = std::chrono::steady_clock;
constexpr size_t N = 2560, HC = 4, D = N * HC, Guard = 32;
constexpr uint16_t Sentinel = 0x7721;

void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}

uint16_t bf16(float v) {
    const uint32_t bits = std::bit_cast<uint32_t>(v);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

float f32(uint16_t v) { return std::bit_cast<float>(uint32_t(v) << 16); }

struct Fixture {
    const size_t rows;
    const size_t cells = rows * D;
    std::shared_ptr<strata::sycl_backend::Runtime> runtime = runtime_for();
    sycl::queue& queue = runtime->compute();
    Allocation input{runtime, cells * 2, MemoryKind::Device};
    Allocation scratch{runtime, cells * 2, MemoryKind::Device};
    Allocation output{runtime, (cells + 2 * Guard) * 2, MemoryKind::Device};
    Allocation projection{runtime, rows * N * 2, MemoryKind::Device};
    Allocation injection{runtime, rows * HC * 4, MemoryKind::Device};
    Allocation embedding{runtime, rows * N * 4, MemoryKind::Device};

    Fixture(size_t row_count, bool zero_gate) : rows(row_count) {
        require(runtime->device().get_backend() == sycl::backend::ext_oneapi_level_zero && queue.is_in_order(),
                "expected in-order Level Zero stream");
        auto* r = input.as<uint16_t>();
        auto* b = projection.as<uint16_t>();
        auto* g = injection.as<float>();
        auto* e = embedding.as<float>();
        auto* y = output.as<uint16_t>();
        const size_t tensor_rows = rows, residual_cells = cells;
        queue.parallel_for(sycl::range<1>(cells + 2 * Guard), [=](sycl::id<1> item) {
            const size_t i = item[0];
            y[i] = Sentinel;
            if (i < residual_cells) r[i] = uint16_t((i % 2 ? 0xbf00 : 0x3e00) + (i * 71 % 127));
            if (i < tensor_rows * N) {
                b[i] = uint16_t((i % 3 ? 0x3e80 : 0xbe80) + (i * 13 % 127));
                e[i] = float(int(i * 19 % 123) - 61) / 64.f;
            }
            if (i < tensor_rows * HC) g[i] = zero_gate ? 0.f : float(int(i * 31 % 323) - 161) / 32.f;
        });
        drain();
    }

    ~Fixture() { checked_cleanup([&] { drain(); }, [] {}, "attention bank fixture drain"); }

    void drain() {
        require(cudaStreamSynchronize(&queue) == cudaSuccess, "attention bank checked stream drain failed");
    }

    uint16_t* bank() { return output.as<uint16_t>() + Guard; }

    void reset(bool layer_zero) {
        if (layer_zero) native::broadcast(embedding.as<float>(), scratch.as<uint16_t>(), rows, &queue);
        else queue.memcpy(scratch.data(), input.data(), cells * 2);
    }

    void launch(bool direct) {
        auto* residual = scratch.as<uint16_t>();
        if (direct) {
            native::write_prefill_attention_bank(bank(), residual, projection.as<uint16_t>(),
                                                 injection.as<float>(), rows, &queue);
        } else {
            native::write(residual, projection.as<uint16_t>(), injection.as<float>(), rows, &queue);
            require(cudaMemcpyAsync(bank(), residual, cells * 2, cudaMemcpyDeviceToDevice, &queue) == cudaSuccess,
                    "reference attention bank copy failed");
        }
    }

    std::vector<uint16_t> result() {
        std::vector<uint16_t> values(cells + 2 * Guard);
        checked_cleanup([&] {
            queue.memcpy(values.data(), output.data(), output.size());
            drain();
        }, [] {}, "fixture output snapshot");
        require(std::all_of(values.cbegin(), values.cbegin() + Guard, [](auto v) { return v == Sentinel; }) &&
                std::all_of(values.cend() - Guard, values.cend(), [](auto v) { return v == Sentinel; }),
                "direct attention overwrote the bank guards");
        return values;
    }
};

void correctness(size_t rows, bool layer_zero) {
    Fixture f(rows, rows == 1);
    f.reset(layer_zero);
    f.launch(false);
    f.drain();
    const auto reference = f.result();
    f.queue.fill(f.bank(), Sentinel, f.cells);
    f.reset(layer_zero);
    f.launch(true);
    f.drain();
    require(f.result() == reference, "attention bank BF16 output differs");
    if (rows == 1) {
        for (size_t i = 0; i < D; ++i) {
            const uint16_t r = uint16_t((i % 2 ? 0xbf00 : 0x3e00) + (i * 71 % 127));
            const size_t d = i % N;
            const uint16_t b = uint16_t((d % 3 ? 0x3e80 : 0xbe80) + (d * 13 % 127));
            require(reference[i + Guard] == bf16(f32(r) + f32(b)), "independent zero-injection oracle differs");
        }
        int refused = 0;
        auto* r = f.scratch.as<uint16_t>();
        auto* b = f.projection.as<uint16_t>();
        auto* g = f.injection.as<float>();
        for (const auto& invalid : {std::pair{r, 1}, std::pair{f.bank(), 0}, std::pair{f.bank(), 2}}) {
            try { native::write_prefill_attention_bank(invalid.first, r, b, g, invalid.second, &f.queue); }
            catch (const std::invalid_argument&) { ++refused; }
        }
        require(refused == 3, "overlap, empty shape or extent refusal missing");
        require(f.result() == reference, "refused store changed the bank");
        std::vector<uint16_t> residual(D);
        checked_cleanup([&] {
            f.queue.memcpy(residual.data(), f.scratch.data(), D * 2);
            f.drain();
        }, [] {}, "fixture source snapshot");
        for (size_t i = 0; i < D; ++i)
            require(residual[i] == uint16_t((i % 2 ? 0xbf00 : 0x3e00) + (i * 71 % 127)),
                    "direct store changed the old residual");
    }
    // The fixture destructor owns and drains the last submission even in the
    // cancellation case, the same order the model enforces with its lease.
    f.reset(layer_zero);
    f.launch(true);
    std::cout << "verified rows=" << rows << " layer_zero=" << int(layer_zero) << " bits=" << f.cells
              << " guards=1 cancellation_lease=1\n";
}

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void timing(bool direct) {
    Fixture f(8192, false);
    f.reset(false);
    f.launch(direct);
    f.drain();
    std::array<double,257> overhead{};
    for (double& time : overhead) {
        const auto a = Clock::now();
        time = elapsed_ms(a, Clock::now());
    }
    std::sort(overhead.begin(), overhead.end());
    std::cout << "timer_pair_ms=" << overhead[overhead.size() / 2] << '\n';
    for (unsigned batch = 0; batch < 5; ++batch) {
        double wall_ms = 0;
        for (unsigned rep = 0; rep < 16; ++rep) {
            f.reset(false);
            f.drain();
            const auto begin = Clock::now();
            f.launch(direct);
            f.drain();
            wall_ms += elapsed_ms(begin, Clock::now());
        }
        std::cout << "batch=" << batch << " rows=" << f.rows << " repetitions=16 d2d_bytes="
                  << (direct ? 0 : f.cells * 2 * 16) << " wall_ms=" << wall_ms << '\n';
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected correctness, copy or direct");
        const std::string_view operation(argv[1]);
        std::cout << std::fixed << std::setprecision(9) << std::unitbuf;
        if (operation == "correctness") {
            correctness(1, false);
            correctness(63, true);
            correctness(8191, false);
            correctness(8192, true);
        } else {
            require(operation == "copy" || operation == "direct", "unknown attention-store arm");
            timing(operation == "direct");
        }
        std::cout << "attention_bank_fixture=PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
