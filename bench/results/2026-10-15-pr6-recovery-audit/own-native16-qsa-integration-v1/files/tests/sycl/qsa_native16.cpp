// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors
// Independent FP64/dyadic component oracle, not model-quality acceptance.
#include "strata/prefill/qsa_native16.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sb = strata::sycl_backend;
namespace qa = strata::prefill::native16::qsa;
namespace sk = strata::kernels;
namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
// Arithmetic IEEE encoders/decoders, independent of production bit helpers.
double even(double v) {
    const double n = std::floor(v), f = v - n;
    return n + (f > .5 || (f == .5 && std::fmod(n, 2.) != 0.));
}
uint16_t encode(double value, bool half = false) {
    const int p = half ? 10 : 7, lo = half ? -14 : -126, hi = half ? 15 : 127;
    const unsigned sign = std::signbit(value) ? 0x8000 : 0, inf = half ? 0x7c00 : 0x7f80;
    if (std::isnan(value)) return uint16_t(sign | inf | (half ? 0x200 : 0x40));
    if (std::isinf(value)) return uint16_t(sign | inf);
    double a = std::fabs(value);
    if (a == 0.) return uint16_t(sign);
    int e; std::frexp(a, &e); --e;
    const double quantum = std::ldexp(1., std::max(e, lo) - p);
    a = even(a / quantum) * quantum;
    if (a == 0.) return uint16_t(sign);
    std::frexp(a, &e); --e;
    if (e > hi) return uint16_t(sign | inf);
    if (e < lo) return uint16_t(sign | unsigned(std::ldexp(a, p - lo)));
    const unsigned exponent = unsigned(e - lo + 1);
    const unsigned mantissa = unsigned((std::ldexp(a, -e) - 1.) * double(1u << p));
    return uint16_t(sign | (exponent << p) | mantissa);
}
double decode(uint16_t bits, bool half = false) {
    const int p = half ? 10 : 7, bias = half ? 15 : 127;
    const unsigned mask = (1u << p) - 1, e = (bits & 0x7fff) >> p, m = bits & mask;
    const unsigned all = half ? 31 : 255;
    double a = e == all ? (m ? std::numeric_limits<double>::quiet_NaN() : INFINITY) :
        (e ? std::ldexp(1. + double(m) / double(1u << p), int(e) - bias) :
             std::ldexp(double(m), 1 - bias - p));
    return bits & 0x8000 ? -a : a;
}
double spacing(double value) {
    int e; std::frexp(std::fabs(value), &e);
    return std::ldexp(1., std::max(e - 1, -126) - 7);
}
template<class T> struct Buffer {
    std::shared_ptr<sb::Runtime> runtime;
    sb::Allocation memory;
    size_t count;
    Buffer(std::shared_ptr<sb::Runtime> r, size_t n) : runtime(r),
        memory(r, n * sizeof(T), sb::MemoryKind::Device), count(n) {}
    T* data() { return memory.as<T>(); }
    void put(const std::vector<T>& values) {
        require(values.size() == count, "upload size");
        try { runtime->wait(runtime->compute().memcpy(data(), values.data(), memory.size())); }
        catch (...) {
            sb::checked_cleanup([&] { runtime->wait(); }, [] {}, "QSA16 upload drain failed");
            throw;
        }
    }
    std::vector<T> get() {
        std::vector<T> values(count);
        try { runtime->wait(runtime->compute().memcpy(values.data(), data(), memory.size())); }
        catch (...) {
            sb::checked_cleanup([&] { runtime->wait(); }, [] {}, "QSA16 readback drain failed");
            throw;
        }
        return values;
    }
};
// Declare after every host/device owner, before the FIRST copy or kernel.
// A failed checked drain ends the process before any owner can unwind.
struct Drain {
    std::shared_ptr<sb::Runtime> runtime;
    sycl::queue* extra = nullptr;
    ~Drain() noexcept {
        sb::checked_cleanup([&] {
            if (extra) extra->wait_and_throw();
            runtime->wait();
        }, [] {}, "QSA16 fixture drain failed");
    }
};
template<class F> void invalid(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid interface accepted");
}
constexpr size_t G = 16;
constexpr uint16_t guard = 0x449a;

void norm_case(std::shared_ptr<sb::Runtime> r, int T, int heads, int pos0,
               int exponent, bool special = false, bool exact = false) {
    const int hs = heads == 24 ? 512 : 256, il = heads * hs + 18,
              ol = heads * 256 + 10, rows = pos0 + T + 2;
    const float eps = exact || exponent != 0 ? 0.f : 1e-5f;
    std::vector<uint16_t> x(G + T * il + G, guard), out(G + T * ol + G, guard);
    std::vector<float> w(G + 256 + G, 1234.f), c(G + rows * 32 + G, 1234.f), s(c);
    for (int d = 0; d < 256; ++d) w[G + d] = float((d % 5 + 1) / 4.);
    for (int t = 0; t < T; ++t) for (int h = 0; h < heads; ++h)
        for (int d = 0; d < 256; ++d) {
            const double v = exact ? (d % 2 ? -2. : 2.) :
                std::ldexp(double((t * 37 + h * 13 + d * 7) % 65 - 32) / 16., exponent);
            x[G + t * il + h * hs + d] = encode(v);
            if (hs == 512) x[G + t * il + h * hs + 256 + d] = encode(double(d % 17) / 8.);
        }
    if (special) x[G + il * (T - 1)] = 0x7fc1; // one normalized head diagnoses NaN
    for (int p = 0; p < rows; ++p) for (int d = 0; d < 32; ++d) {
        const double angle = double(p) * std::pow(1e7, -2. * d / 64.);
        c[G + p * 32 + d] = exact ? (p % 2 ? 0.f : 1.f) : float(std::cos(angle));
        s[G + p * 32 + d] = exact ? (p % 2 ? 1.f : 0.f) : float(std::sin(angle));
    }
    Buffer<uint16_t> dx(r, x.size()), dy(r, out.size());
    Buffer<float> dw(r, w.size()), dc(r, c.size()), ds(r, s.size());
    Drain drain{r};
    dx.put(x); dy.put(out); dw.put(w); dc.put(c); ds.put(s);
    auto run = [&] {
        qa::norm_rope_split(dx.data() + G, dw.data() + G, dy.data() + G, T, heads,
            hs, il, ol, pos0, rows, dc.data() + G, ds.data() + G, eps, &r->compute());
    };
    run();
    auto actual = dy.get();
    for (int t = 0; t < T; ++t) for (int h = 0; h < heads; ++h) {
        double sum = 0.; bool bad = false;
        for (int d = 0; d < 256; ++d) {
            const double a = decode(x[G + t * il + h * hs + d]);
            bad |= !std::isfinite(a); sum += a * a;
        }
        const double rs = 1. / std::sqrt(sum / 256. + eps);
        auto normalized = [&](int d) { return decode(x[G + t * il + h * hs + d]) * rs * w[G + d]; };
        for (int d = 0; d < 256; ++d) {
            double ref = normalized(d);
            if (d < 64) {
                const int pair = d % 32;
                const double a = normalized(pair), b = normalized(pair + 32),
                    co = c[G + (pos0 + t) * 32 + pair], si = s[G + (pos0 + t) * 32 + pair];
                ref = d < 32 ? a * co - b * si : a * si + b * co;
            }
            const size_t i = G + t * ol + h * 256 + d;
            if (bad) require(std::isnan(decode(actual[i])), "norm NaN masked");
            else if (exact) require(actual[i] == encode(ref), "dyadic norm/split/RoPE mismatch");
            else require(std::isfinite(decode(actual[i])) &&
                std::fabs(decode(actual[i]) - ref) <= 2. * spacing(ref) + 2e-6,
                "FP64 norm/split/RoPE screen");
            out[i] = actual[i];
        }
    }
    require(actual == out, "norm output guards/stride padding modified");
    require(dx.get() == x && dw.get() == w && dc.get() == c && ds.get() == s,
            "norm original inputs changed");
    invalid([&] { qa::norm_rope_split(dx.data() + G, dw.data() + G, dx.data() + G,
        T, heads, hs, il, ol, pos0, rows, dc.data() + G, ds.data() + G, eps, &r->compute()); });
    invalid([&] { qa::norm_rope_split(dx.data() + G, dw.data() + G, dy.data() + G,
        T, heads, hs, il, ol, rows, rows, dc.data() + G, ds.data() + G, eps, &r->compute()); });
    invalid([&] { qa::norm_rope_split(dx.data() + G, dw.data() + G, dy.data() + G,
        T, heads, hs, il, ol, pos0, 262144, dc.data() + G, ds.data() + G, eps, &r->compute()); });
    invalid([&] { qa::norm_rope_split(dx.data() + G, dw.data() + G, dy.data() + G,
        T, heads, hs, il - 19, ol, pos0, rows, dc.data() + G, ds.data() + G, eps, &r->compute()); });
    invalid([&] { qa::norm_rope_split(dx.data() + G, dw.data() + G, dy.data() + G,
        T, heads, hs, il, ol, pos0, rows, dc.data() + G, ds.data() + G, -1.f, &r->compute()); });
    run(); require(dy.get() == actual, "norm rejection/recovery changed result");
}

void gate_case(std::shared_ptr<sb::Runtime> r, int T) {
    const int al = 6144 + 10, ql = 12288 + 18, yl = 6144 + 26;
    std::vector<uint16_t> a(G + T * al + G, guard), q(G + T * ql + G, guard), y(G + T * yl + G, guard);
    const double av[] = {1., -2., -2., 0., 1., INFINITY, NAN, std::ldexp(1., -133), std::ldexp(1., 120)};
    const double gv[] = {0., 2., -2., INFINITY, -INFINITY, -INFINITY, 0., 0., 0.};
    for (int t = 0; t < T; ++t) for (int h = 0; h < 24; ++h) for (int d = 0; d < 256; ++d) {
        const int c = (t + h + d) % 9;
        a[G + t * al + h * 256 + d] = encode(av[c]);
        q[G + t * ql + h * 512 + d] = encode(777.); // query half is not the gate
        q[G + t * ql + h * 512 + 256 + d] = encode(gv[c]);
    }
    Buffer<uint16_t> da(r, a.size()), dq(r, q.size()), dy(r, y.size());
    sycl::context context(r->device());
    sycl::queue other(context, r->device(), sycl::property::queue::in_order{});
    Drain drain{r, &other};
    da.put(a); dq.put(q); dy.put(y);
    auto run = [&] { qa::gate(da.data() + G, dq.data() + G, dy.data() + G, T, al, ql, yl, &r->compute()); };
    run(); auto actual = dy.get();
    for (int t = 0; t < T; ++t) for (int h = 0; h < 24; ++h) for (int d = 0; d < 256; ++d) {
        const double at = decode(a[G + t * al + h * 256 + d]),
                     z = decode(q[G + t * ql + h * 512 + 256 + d]),
                     ref = at * (1. / (1. + std::exp(-z)));
        const size_t i = G + t * yl + h * 256 + d;
        if (std::isnan(ref)) require(std::isnan(decode(actual[i], true)), "gate NaN masked");
        else require(actual[i] == encode(ref, true), "gate FP64/dyadic rounded result");
        y[i] = actual[i];
    }
    require(actual == y && da.get() == a && dq.get() == q, "gate guards/strides/immutable inputs");
    invalid([&] { qa::gate(da.data() + G, dq.data() + G, da.data() + G, T, al, ql, yl, &r->compute()); });
    invalid([&] { qa::gate(da.data() + G, dq.data() + G, dy.data() + G, T, al, ql, 6143, &r->compute()); });
    invalid([&] { qa::gate(da.data() + G, dq.data() + G, dy.data() + dy.count - G,
        T, al, ql, yl, &r->compute()); });
    invalid([&] { qa::gate(da.data() + G, dq.data() + G,
        reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(dy.data()) + 1),
        T, al, ql, yl, &r->compute()); });
    // Same GPU with a distinct context: registry must reject before submission.
    invalid([&] { qa::gate(da.data() + G, dq.data() + G, dy.data() + G, T, al, ql, yl, &other); });
    run(); require(dy.get() == actual, "gate rejection/recovery changed result");
}

void append_case(std::shared_ptr<sb::Runtime> r) {
    constexpr int T = 3, ld = 526, pos0 = 3, pages = 3, physical = 2, page_size = 4;
    const size_t cache_rows = physical * 2 * page_size;
    std::vector<uint16_t> k(G + T * ld + G, guard), v(k), scales(G + cache_rows * 4 + G, 0x3555);
    std::vector<int8_t> codes(G + cache_rows * 256 + G, int8_t(85));
    std::vector<int32_t> table(G + pages + G, 123456); table[G] = 1; table[G + 1] = 0; table[G + 2] = -1;
    for (int t = 0; t < T; ++t) for (int h = 0; h < 2; ++h) for (int d = 0; d < 256; ++d) {
        double x = d % 64 == 0 ? 127. : d % 64 == 1 ? .5 :
                   d % 64 == 2 ? -1.5 : double((t * 11 + h * 7 + d) % 63 - 31);
        k[G + t * ld + h * 256 + d] = encode(x);
        v[G + t * ld + h * 256 + d] = encode(x / 2.);
    }
    Buffer<uint16_t> dk(r, k.size()), dv(r, v.size()), dks(r, scales.size()), dvs(r, scales.size());
    Buffer<int8_t> dkc(r, codes.size()), dvc(r, codes.size());
    Buffer<int32_t> dt(r, table.size());
    Drain drain{r};
    dk.put(k); dv.put(v); dks.put(scales); dvs.put(scales); dkc.put(codes); dvc.put(codes); dt.put(table);
    qa::Q8WritePools p{}; p.k_q = dkc.data() + G; p.v_q = dvc.data() + G;
    p.k_scale = dks.data() + G; p.v_scale = dvs.data() + G; p.page_table = dt.data() + G;
    auto s = sk::qsa_real_shapes(); s.page_size = page_size;
    sk::QsaPromptBf16Bounds b{pages, physical};
    auto run = [&] { qa::append_q8(dk.data() + G, dv.data() + G, T, ld, pos0, p, s, b, &r->compute()); };
    run();
    auto kc = codes, vc = codes, ks = scales, vs = scales;
    for (int t = 0; t < T; ++t) for (int h = 0; h < 2; ++h) for (int g = 0; g < 4; ++g) {
        const int pos = pos0 + t, row = (table[G + pos / page_size] * 2 + h) * page_size + pos % page_size;
        ks[G + row * 4 + g] = encode(1., true); vs[G + row * 4 + g] = encode(.5, true);
        for (int lane = 0; lane < 64; ++lane) {
            const double x = decode(k[G + t * ld + h * 256 + g * 64 + lane]);
            const double rounded = std::copysign(even(std::fabs(x)), x);
            kc[G + row * 256 + g * 64 + lane] = int8_t(rounded);
            vc[G + row * 256 + g * 64 + lane] = int8_t(rounded);
        }
    }
    require(dkc.get() == kc && dvc.get() == vc && dks.get() == ks && dvs.get() == vs,
            "typed Q8 dyadic oracle/guards/page boundary/tails");
    require(dk.get() == k && dv.get() == v && dt.get() == table, "Q8 original input immutability");
    // Missing and physically invalid pages must leave every byte unchanged.
    table[G] = -1; table[G + 1] = physical; dt.put(table); run();
    require(dkc.get() == kc && dvc.get() == vc && dks.get() == ks && dvs.get() == vs,
            "invalid Q8 page accessed");
    invalid([&] { auto bad = b; bad.physical_pages = 3;
        qa::append_q8(dk.data() + G, dv.data() + G, T, ld, pos0, p, s, bad, &r->compute()); });
    invalid([&] { auto bad = p; bad.v_q = bad.k_q;
        qa::append_q8(dk.data() + G, dv.data() + G, T, ld, pos0, bad, s, b, &r->compute()); });
    invalid([&] { qa::append_q8(dk.data() + G, dv.data() + G, T, ld, pages * page_size,
        p, s, b, &r->compute()); });
    // Recover valid mapping, diagnose NaN/Inf groups without integer UB.
    table[G] = 1; table[G + 1] = 0; dt.put(table);
    k[G] = 0x7fc1; v[G] = 0x7f80; dk.put(k); dv.put(v); run();
    const int first_row = (1 * 2) * page_size + pos0;
    const auto badks = dks.get(), badvs = dvs.get();
    require(std::isnan(decode(badks[G + first_row * 4], true)) &&
            std::isnan(decode(badvs[G + first_row * 4], true)), "Q8 nonfinite scale diagnostic masked");
    const auto badkc = dkc.get(), badvc = dvc.get();
    for (int d = 0; d < 64; ++d) require(badkc[G + first_row * 256 + d] == 0 &&
        badvc[G + first_row * 256 + d] == 0, "Q8 nonfinite group integer conversion");
    require(dk.get() == k && dv.get() == v && dt.get() == table, "Q8 special input immutability");
    k[G] = encode(127.); v[G] = encode(63.5); dk.put(k); dv.put(v); run();
    require(dkc.get() == kc && dvc.get() == vc && dks.get() == ks && dvs.get() == vs,
            "Q8 rejection/special recovery");
}
} // namespace
int main() try {
    auto runtime = sb::runtime_for();
    qa::validate_device(&runtime->compute());
    for (int T : {1, 3, 17}) for (int heads : {2, 24}) {
        norm_case(runtime, T, heads, 5, 0);
        norm_case(runtime, T, heads, 0, 0, false, true);
    }
    norm_case(runtime, 3, 24, 5, 0, true);
    norm_case(runtime, 1, 2, 0, 120);
    norm_case(runtime, 1, 24, 0, -128);
    for (int T : {1, 3, 17}) gate_case(runtime, T);
    append_case(runtime);
    runtime->wait();
    std::cout << "QSA16 local component screen PASS; attention/model/physical262144/RESTORE/performance unqualified\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "QSA16 component failure: " << e.what() << '\n';
    return 1;
}
