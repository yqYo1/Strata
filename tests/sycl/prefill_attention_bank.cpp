// Main-authored fixture. Repository LICENSE applies.
#include "strata/prefill/native16.hpp"
#include "strata/prefill/layer_major_plan.hpp"
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

void check_prefill_attention_bank_input(size_t rows) {
    Fixture f(rows, false);
    f.reset(false);
    f.launch(false);
    f.drain();
    const auto reference = f.result();
    auto* scratch = f.scratch.as<uint16_t>();
    auto* bo = f.projection.as<uint16_t>();
    auto* inj = f.injection.as<float>();
    for (bool direct_store : {false, true}) {
        f.queue.memcpy(f.bank(), f.input.data(), f.cells * sizeof(uint16_t));
        f.queue.fill(scratch, Sentinel, f.cells);
        std::vector<uint16_t> original;
        if (rows == 1 && !direct_store) original = f.result();
        native::write_prefill_attention_from_bank(direct_store ? f.bank() : scratch, f.bank(),
                                                  bo, inj, rows, &f.queue);
        if (!direct_store) {
            if (rows == 1) require(f.result() == original, "scratch output modified the borrowed bank");
            f.queue.memcpy(f.bank(), scratch, f.cells * sizeof(uint16_t));
        } else if (rows == 1) {
            std::vector<uint16_t> staging(f.cells);
            checked_cleanup([&] {
                f.queue.memcpy(staging.data(), scratch, staging.size() * sizeof(uint16_t));
                f.drain();
            }, [] {}, "borrowed input scratch witness");
            require(std::all_of(staging.begin(), staging.end(), [](auto value) { return value == Sentinel; }),
                    "in-place attention modified scratch");
        }
        f.drain();
        require(f.result() == reference, "borrowed attention input changed BF16 output or guards");
    }
    if (rows == 1) {
        int rejected = 0;
        for (const auto& invalid : {std::pair{f.bank() + 2, 1}, std::pair{f.bank(), 0}, std::pair{f.bank(), 2}}) {
            try { native::write_prefill_attention_from_bank(invalid.first, f.bank(), bo, inj, invalid.second, &f.queue); }
            catch (const std::invalid_argument&) { ++rejected; }
        }
        require(rejected == 3 && f.result() == reference, "borrowed attention overlap or extent check failed");
    }
    std::cout << "attention input rows=" << rows << " copy_and_direct=1 bits=" << f.cells << " guards=1\n";
}

// Exercise the lifetime of one resident window, including its partial second
// piece. The real model separately qualifies PLE/audit and host-tail dispatch.
void check_prefill_resident_window() {
    namespace detail = strata::prefill::detail;
    constexpr int64_t rows = 11, chunk = 8;
    Fixture f(rows, false);
    const auto plan = detail::LayerMajorResidualPlan::make(
        rows, chunk, D, f.cells * sizeof(uint16_t), rows, detail::LayerMajorResidualDtype::bf16);
    Allocation weights(f.runtime, D * sizeof(float), MemoryKind::Device);
    Allocation scales(f.runtime, chunk * HC * sizeof(float), MemoryKind::Device);
    Allocation normalized(f.runtime, chunk * D * sizeof(uint16_t), MemoryKind::Device);
    Allocation mixed(f.runtime, chunk * N * sizeof(uint16_t), MemoryKind::Device);
    Allocation mixed_fp16(f.runtime, chunk * N * sizeof(uint16_t), MemoryKind::Device);
    Allocation callback(f.runtime, chunk * D * sizeof(float), MemoryKind::Device);
    std::vector<float> reference, delivered(f.cells);
    // Destroy this guard before releasing any of the callback or device owners.
    struct Drain {
        Fixture& f;
        ~Drain() { checked_cleanup([&] { f.drain(); }, [] {}, "resident window retirement"); }
    } drain_on_exit{f};
    f.queue.fill(weights.as<float>(), 1.f, D);
    for (bool borrow : {false, true}) for (bool direct : {false, true}) {
        f.queue.memcpy(f.bank(), f.input.data(), f.cells * sizeof(uint16_t));
        // Publish a previous layer's FFN result. The next attention must read
        // this update, including the tail, even when scratch still holds poison.
        native::write(f.bank(), f.projection.as<uint16_t>(), f.injection.as<float>(), rows, &f.queue);
        f.drain();
        for (int64_t row = 0; row < rows; row += chunk) {
            const auto count = std::min(chunk, rows - row);
            const auto span = plan.span(row, count);
            require(span.device && span.offset_bytes == row * D * sizeof(uint16_t), "wrong bank input span");
            auto* bank = f.bank() + row * D;
            auto* scratch = f.scratch.as<uint16_t>();
            auto* injection = f.injection.as<float>() + row * HC;
            if (borrow) f.queue.fill(scratch, Sentinel, size_t(chunk) * D);
            else f.queue.memcpy(scratch, bank, span.bytes);
            const auto* input = borrow ? bank : scratch;
            native::norm(input, weights.as<float>(), 1e-6f, scales.as<float>(), normalized.as<uint16_t>(),
                         count, &f.queue);
            native::mix(input, scales.as<float>(), weights.as<float>(), normalized.as<uint16_t>(),
                        mixed.as<uint16_t>(), mixed_fp16.as<uint16_t>(), count, &f.queue);
            if (borrow)
                native::write_prefill_attention_from_bank(direct ? bank : scratch, bank, mixed.as<uint16_t>(),
                                                         injection, count, &f.queue);
            else if (direct)
                native::write_prefill_attention_bank(bank, scratch, mixed.as<uint16_t>(), injection, count, &f.queue);
            else
                native::write(scratch, mixed.as<uint16_t>(), injection, count, &f.queue);
            if (!direct) f.queue.memcpy(bank, scratch, span.bytes);
            f.drain();
        }
        // Complete FFN in the bank before the final callback restores its rows.
        native::write(f.bank(), f.projection.as<uint16_t>(), f.injection.as<float>(), rows, &f.queue);
        f.drain();
        detail::LayerMajorChunkCursor cursor{rows, chunk};
        for (int64_t row = 0; row < rows; row += chunk) {
            const auto count = std::min(chunk, rows - row);
            cursor.accept(row, count);
            native::restore(f.bank() + row * D, callback.as<float>(), count * D, &f.queue);
            f.queue.memcpy(delivered.data() + row * D, callback.data(), count * D * sizeof(float));
            f.drain();
        }
        require(cursor.complete() && cursor.calls == 2, "missing restored window piece");
        const auto bank = f.result();
        for (size_t i = 0; i < f.cells; ++i)
            require(std::bit_cast<uint32_t>(delivered[i]) == uint32_t(bank[Guard + i]) << 16,
                    "callback and final BF16 bank differ");
        if (reference.empty()) reference = delivered;
        else require(delivered == reference, "window handoff changed the final callback");
    }
    std::cout << "attention window rows=11 pieces=8+3 arms=4 checked_final_callback=1\n";
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
        require(argc == 2, "expected correctness, input-correctness, copy or direct");
        const std::string_view operation(argv[1]);
        std::cout << std::fixed << std::setprecision(9) << std::unitbuf;
        if (operation == "correctness") {
            correctness(1, false);
            correctness(63, true);
            correctness(8191, false);
            correctness(8192, true);
        } else if (operation == "input-correctness") {
            for (size_t rows : {1, 63, 7905, 8191, 8192}) check_prefill_attention_bank_input(rows);
            check_prefill_resident_window();
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
