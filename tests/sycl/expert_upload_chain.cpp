// Main-authored; repository LICENSE applies. A production upload/dequant/GEMM
// dependency probe, not the model scheduler or a model throughput benchmark.
#include "strata/prefill/expert_bank_upload.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/native16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <ggml.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace sb = strata::sycl_backend;
namespace pd = strata::prefill::detail;
namespace pk = strata::kernels;
constexpr int N = 2560, F = 640, G = 128, Experts = 8;
constexpr uint16_t Sentinel = 0xa5a5;
void need(bool v, const char* message) { if (!v) throw std::runtime_error(message); }
uint16_t half(float v) { return std::bit_cast<uint16_t>(_Float16(v)); }
float unhalf(uint16_t v) { return float(std::bit_cast<_Float16>(v)); }
float unbf(uint16_t v) { return std::bit_cast<float>(uint32_t(v) << 16); }
uint16_t bf(float v) {
    uint32_t b = std::bit_cast<uint32_t>(v);
    return uint16_t((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}

struct Buffer {
    sb::Allocation owner;
    size_t count;
    Buffer(std::shared_ptr<sb::Runtime> rt, size_t n)
        : owner(rt, (n + 2 * G) * 2, sb::MemoryKind::Device), count(n) {}
    uint16_t* data() { return owner.as<uint16_t>() + G; }
    std::vector<uint16_t> read(sycl::queue& q, sb::Runtime& rt) {
        std::vector<uint16_t> v(count + 2 * G);
        try { rt.wait(q.memcpy(v.data(), owner.data(), owner.size())); }
        catch (...) { std::_Exit(86); } // Do not retire a possibly live DMA target.
        for (int i = 0; i < G; ++i)
            need(v[i] == Sentinel && v[count + G + i] == Sentinel, "device tensor guard");
        return {v.begin() + G, v.end() - G};
    }
};

struct Chain {
    std::shared_ptr<sb::Runtime> rt = sb::runtime_for();
    sycl::queue q = rt->make_queue(true);
    pk::NativeExpertLayout layout;
    int rows;
    size_t stride;
    std::vector<uint8_t> slab;
    strata::core::ExpertCache cache;
    std::unique_ptr<pd::PrivateExpertBank> bank;
    std::unique_ptr<pd::ExpertBankSlots> slots;
    std::unique_ptr<pd::ExpertUploadStaging> staging;
    Buffer x, gu_weights, down_weights, gu, hidden, down;
    std::vector<uint16_t> input;
    strata::prefill::Gemm gemm;
    uint64_t uploads = 0, hits = 0;

    Chain(int type, int t, bool host_staging)
        : layout(pk::native_expert_layout(type, 20, N, F)), rows(t),
          stride((layout.bytes + 256 + 255) / 256 * 256), slab(512 * stride, 0xa5),
          x(rt, size_t(t) * N), gu_weights(rt, size_t(2) * F * N),
          down_weights(rt, size_t(N) * F), gu(rt, size_t(t) * 2 * F),
          hidden(rt, size_t(t) * F), down(rt, size_t(t) * N), input(size_t(t) * N) {
        need(q.has_property<sycl::property::queue::in_order>(), "in-order queue required");
        need((type == 22 && layout.bytes == 1971200) ||
             (type == 23 && layout.bytes == 2662400), "real model payload geometry");
        const int block_bytes = type == 22 ? 82 : 136;
        for (int e = 0; e < Experts; ++e) {
            std::mt19937 rng(12301 + e * 17 + type);
            auto* p = slab.data() + size_t(e) * stride;
            for (size_t i = 0; i < layout.bytes; ++i) p[i] = uint8_t(rng());
            for (size_t i = 0; i < layout.down_off; i += block_bytes) {
                const uint16_t d = half(float(int(rng() % 257) - 128) / 262144.f);
                std::memcpy(p + i, &d, 2);
            }
            for (size_t i = layout.down_off; i < layout.bytes; i += 18) {
                const uint16_t d = half(float(int(rng() % 257) - 128) / 262144.f);
                std::memcpy(p + i, &d, 2);
            }
            const uint16_t subnormal = 1;
            std::memcpy(p, &subnormal, 2);
        }
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = half(float(int((i * 37 + 11) % 257) - 128) / 2048.f);
        std::string error;
        if (!cache.open_uninitialized(3, 1, 512, int64_t(stride), error)) throw std::runtime_error(error);
        bank = std::make_unique<pd::PrivateExpertBank>(cache, stride);
        slots = std::make_unique<pd::ExpertBankSlots>(bank->layout().slots);
        if (!gemm.init_external(&q, nullptr, 0, nullptr, 0, error)) throw std::runtime_error(error);
        if (host_staging) staging = std::make_unique<pd::ExpertUploadStaging>(
            cache.device_slot(0), size_t(cache.bytes()), layout.bytes, &q);
        // All allocations/host owners exist before the first asynchronous use.
        try {
            q.memset(cache.device_slot(0), 0xa5, size_t(cache.bytes()));
            for (auto* b : {&x, &gu_weights, &down_weights, &gu, &hidden, &down})
                q.memset(b->owner.data(), 0xa5, b->owner.size());
            q.memcpy(x.data(), input.data(), input.size() * 2);
            drain();
        } catch (...) { drain_noexcept(); throw; }
    }
    ~Chain() { drain_noexcept(); }
    void drain() { q.wait_and_throw(); rt->check(); }
    void drain_noexcept() noexcept { try { drain(); } catch (...) { std::_Exit(86); } }
    void job(int expert, bool stage) {
        need(!stage || bool(staging), "staging owner missing");
        const auto ticket = pd::stage_private_expert_bank(*slots, *bank,
            slab.data(), slab.size(), layout.bytes, expert, &q, stride,
            stage ? staging.get() : nullptr);
        uploads += ticket.upload; hits += !ticket.upload;
        const auto* p = bank->device_slot(ticket.slot);
        pk::iq_dequant_gu_f16(layout.gu_type, p, p + layout.up_off, F, N, gu_weights.data(), &q);
        pk::iq_dequant_f16(layout.d_type, p + layout.down_off, int64_t(N) * F, down_weights.data(), &q);
        need(!gemm.try_gu_swiglu16(x.data(), gu_weights.data(), hidden.data(), rows),
             "probe must exercise the actual unfused production branch");
        gemm.f16_out16(x.data(), gu_weights.data(), gu.data(), rows, 2 * F, N, true);
        strata::prefill::native16::swiglu(gu.data(), hidden.data(), rows, &q);
        gemm.f16_out16(hidden.data(), down_weights.data(), down.data(), rows, N, F, true);
    }
    void guards() {
        drain();
        if (staging) staging->check_guards();
        (void)x.read(q, *rt);
        for (auto* b : {&gu_weights, &down_weights, &gu, &hidden, &down}) (void)b->read(q, *rt);
        std::vector<uint8_t> pad(stride - layout.bytes);
        for (int s = 0; s < bank->layout().slots; ++s) {
            try { rt->wait(q.memcpy(pad.data(), bank->device_slot(s) + layout.bytes, pad.size())); }
            catch (...) { std::_Exit(86); }
            need(std::all_of(pad.begin(), pad.end(), [](uint8_t v) { return v == 0xa5; }), "bank guard");
        }
        need(x.read(q, *rt) == input, "input modified");
        if (staging) need(staging->copied_bytes() == staging->copies() * layout.bytes,
                          "staging byte conservation");
    }
    void oracle(int e) {
        const auto gw = gu_weights.read(q, *rt), dw = down_weights.read(q, *rt);
        const auto g = gu.read(q, *rt), h = hidden.read(q, *rt), d = down.read(q, *rt);
        const auto* p = slab.data() + size_t(e) * stride;
        std::vector<float> gate(size_t(F) * N), up(gate.size()), weight(size_t(N) * F);
        const auto* gt = ggml_get_type_traits(ggml_type(layout.gu_type));
        const auto* dt = ggml_get_type_traits(ggml_type(layout.d_type));
        need(gt && gt->to_float && dt && dt->to_float, "independent ggml oracle required");
        gt->to_float(p, gate.data(), int64_t(F) * N);
        gt->to_float(p + layout.up_off, up.data(), int64_t(F) * N);
        dt->to_float(p + layout.down_off, weight.data(), int64_t(N) * F);
        for (int f = 0; f < F; ++f) for (int n = 0; n < N; ++n)
            need(gw[size_t(2 * f) * N + n] == half(gate[size_t(f) * N + n]) &&
                 gw[size_t(2 * f + 1) * N + n] == half(up[size_t(f) * N + n]), "GU dequant oracle bits");
        for (size_t i = 0; i < dw.size(); ++i) need(dw[i] == half(weight[i]), "Down dequant oracle bits");
        auto dot = [&](const auto& a, const auto& w, const auto& out, int width, int cols) {
            for (int row : {0, rows / 2, rows - 1}) for (int col : {0, 1, 17, cols - 1}) {
                double sum = 0, magnitudes = 0;
                for (int k = 0; k < width; ++k) {
                    const double term = double(unbf(bf(unhalf(a[size_t(row) * width + k])))) *
                        unbf(bf(unhalf(w[size_t(col) * width + k])));
                    sum += term; magnitudes += std::abs(term);
                }
                const double expected = unbf(bf(float(sum)));
                const double ulp = std::ldexp(1., std::ilogb(std::max(std::abs(expected), 0x1p-126)) - 7);
                const double gamma = width * double(std::numeric_limits<float>::epsilon());
                const double bound = 2 * ulp + gamma / (1 - gamma) * magnitudes;
                need(std::abs(unbf(out[size_t(row) * cols + col]) - expected) <= bound,
                     "BF16 operands/FP32 accumulator GEMM oracle");
            }
        };
        dot(input, gw, g, N, 2 * F);
        for (size_t i = 0; i < h.size(); ++i) {
            const double a = unbf(g[2 * i]), b = unbf(g[2 * i + 1]);
            const double expected = std::clamp(a / (1 + std::exp(-a)) * b, -65504., 65504.);
            need(std::isfinite(unhalf(h[i])) && std::abs(unhalf(h[i]) - expected) <=
                 2e-6 + .002 * std::abs(expected), "SwiGLU FP16 boundary oracle");
        }
        dot(h, dw, d, F, N);
        for (auto v : d) need(std::isfinite(unbf(v)), "nonfinite final output");
    }
};

void correctness(int type, int rows) {
    Chain c(type, rows, true);
    std::vector<std::vector<uint16_t>> reference(17), staged(17);
    const auto before = c.slab;
    struct Drain { Chain& c; ~Drain() { c.drain_noexcept(); } } pending{c};
    for (bool stage : {false, true}) {
        c.drain(); c.slots->reset();
        auto& out = stage ? staged : reference;
        for (int i = 0; i < 17; ++i) {
            out[i].resize(size_t(rows) * N);
            c.job(i % Experts, stage);
            c.q.memcpy(out[i].data(), c.down.data(), out[i].size() * 2);
        }
        c.drain(); c.oracle(0); c.guards();
    }
    need(reference == staged, "all17 direct/staged final tensors must be bitwise identical");
    need(c.slab == before, "ordinary source slab modified");
    need(c.staging->copies() == 14 && c.uploads == 28 && c.hits == 6, "wrap/hit/miss conservation");
    std::cout << "CHECK gu=" << type << " down=20 rows=" << rows << " bytes=" << c.layout.bytes
        << " equal_outputs=17 staging_copies=14 host_slots=4 guards=checked oracle=passed\n";
}

int main(int argc, char** argv) {
    try {
        need(argc == 2, "use --smoke|--check|--bench");
        const std::string mode = argv[1];
        need(mode == "--smoke" || mode == "--check" || mode == "--bench", "unknown mode");
        const char* flag = std::getenv("STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING");
        need(flag && (std::string(flag) == "0" || std::string(flag) == "1"), "explicit startup staging0/1 required");
        const bool stage = *flag == '1';
        std::cout << std::setprecision(12);
        if (mode != "--bench") {
            if (mode == "--smoke") correctness(22, 1);
            else for (int t : {32, 640, 1280}) for (int type : {22, 23}) correctness(type, t);
        } else for (int t : {640, 1280}) for (int type : {22, 23}) {
            Chain c(type, t, stage);
            for (int i = 0; i < 32; ++i) c.job(i % Experts, stage);
            c.drain();
            for (int batch = 0; batch < 2; ++batch) {
                const auto u = c.uploads, h = c.hits;
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 256; ++i) c.job(i % Experts, stage);
                c.drain();
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                need(ms >= 100, "timing batch below prospective100ms floor");
                need(c.uploads - u == 192 && c.hits - h == 64, "timed hit/miss conservation");
                std::cout << "TIME gu=" << type << " down=20 rows=" << t << " bytes=" << c.layout.bytes
                    << " staged=" << stage << " batch=" << batch << " jobs=256 uploads=192 hits=64 ms=" << ms << '\n';
            }
            c.oracle(7); c.guards();
        }
        std::cout << "fixture_status=PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fixture_status=FAIL reason=" << e.what() << '\n';
        return 1;
    }
}
