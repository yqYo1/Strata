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
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
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
    sb::MemoryKind kind;
    Buffer(std::shared_ptr<sb::Runtime> r, size_t n,
           sb::MemoryKind k = sb::MemoryKind::Device) : runtime(r),
        memory(r, n * sizeof(T), k), count(n), kind(k) {}
    T* data() { return memory.as<T>(); }
    void put(const std::vector<T>& values) {
        require(values.size() == count, "upload size");
        try {
            if (kind == sb::MemoryKind::Host) {
                runtime->wait();
                std::copy(values.begin(), values.end(), data());
            } else runtime->wait(runtime->compute().memcpy(data(), values.data(), memory.size()));
        }
        catch (...) {
            sb::checked_cleanup([&] { runtime->wait(); }, [] {}, "QSA16 upload drain failed");
            throw;
        }
    }
    std::vector<T> get() {
        std::vector<T> values(count);
        try {
            if (kind == sb::MemoryKind::Host) {
                // Test actual host publication after completion, without a
                // device copy that could hide a missing visibility boundary.
                runtime->wait();
                std::copy_n(data(), count, values.data());
            } else runtime->wait(runtime->compute().memcpy(values.data(), data(), memory.size()));
        }
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

struct Q8Image {
    std::vector<int8_t> k, v;
    std::vector<uint16_t> ks, vs;
    bool operator==(const Q8Image&) const = default;
};
struct Q8Cache {
    Buffer<int8_t> k, v;
    Buffer<uint16_t> ks, vs;
    Q8Cache(std::shared_ptr<sb::Runtime> r, int pages, int page_size,
            sb::MemoryKind kind = sb::MemoryKind::Device)
        : k(r, G + size_t(pages) * 2 * page_size * 256 + G, kind),
          v(r, k.count, kind),
          ks(r, G + size_t(pages) * 2 * page_size * 4 + G, kind),
          vs(r, ks.count, kind) {}
    qa::Q8MirrorPools view() { return {k.data()+G,v.data()+G,ks.data()+G,vs.data()+G}; }
    Q8Image empty() const {
        return {std::vector<int8_t>(k.count,85),std::vector<int8_t>(v.count,85),
                std::vector<uint16_t>(ks.count,0x3555),std::vector<uint16_t>(vs.count,0x3555)};
    }
    void put(const Q8Image& x) { k.put(x.k);v.put(x.v);ks.put(x.ks);vs.put(x.vs); }
    Q8Image get() { return {k.get(),v.get(),ks.get(),vs.get()}; }
};
std::pair<std::vector<uint16_t>,std::vector<uint16_t>> q8_inputs(int T, int ld) {
    std::vector<uint16_t> k(G+size_t(T)*ld+G,guard),v(k);
    for (int t=0;t<T;++t) for (int h=0;h<2;++h) for (int d=0;d<256;++d) {
        const double x=d%64==0 ? 127. : d%64==1 ? .5 : d%64==2 ? -1.5 :
            double((t*11+h*7+d)%63-31);
        k[G+size_t(t)*ld+h*256+d]=encode(x);
        v[G+size_t(t)*ld+h*256+d]=encode(x/2.);
    }
    return {std::move(k),std::move(v)};
}
// FP64 arithmetic and the independent IEEE encoders above. Dyadic test data
// avoids ambiguous FP32-division/FP16-rounding midpoints; this is not a text oracle.
void q8_oracle_cell(Q8Image& out, size_t row, const std::vector<uint16_t>& k,
                    const std::vector<uint16_t>& v, int t, int h, int ld) {
    for (int which=0;which<2;++which) for (int g=0;g<4;++g) {
        const auto& in=which ? v : k;
        auto& codes=which ? out.v : out.k;
        auto& scales=which ? out.vs : out.ks;
        bool bad=false;double maximum=0.;
        for (int d=0;d<64;++d) {
            const double x=decode(in[G+size_t(t)*ld+h*256+g*64+d]);
            bad|=!std::isfinite(x);
            if (std::isfinite(x)) maximum=std::max(maximum,std::fabs(x));
        }
        const uint16_t bits=bad ? 0x7e00 : encode(maximum/127.,true);
        const double scale=decode(bits,true);
        scales[G+row*4+g]=bits;
        for (int d=0;d<64;++d) {
            const double x=decode(in[G+size_t(t)*ld+h*256+g*64+d]);
            codes[G+row*256+g*64+d]=!bad && std::isfinite(scale) && scale>0. ?
                int8_t(std::clamp(even(x/scale),-127.,127.)) : 0;
        }
    }
}

void host_registry_case(std::shared_ptr<sb::Runtime> r) {
    sycl::context context(r->device());
    sycl::queue other(context,r->device(),sycl::property::queue::in_order{});
    sb::Allocation first(r,128,sb::MemoryKind::Host),second(r,96,sb::MemoryKind::Host),
                   device(r,64,sb::MemoryKind::Device);
    auto free_foreign=[context](void* p) {sycl::free(p,context);};
    std::unique_ptr<void,decltype(free_foreign)> foreign(sycl::malloc_host(128,context),free_foreign);
    require(bool(foreign),"foreign Host fixture allocation failed");
    Drain drain{r,&other};
    auto& queue=r->compute();void* p=first.data();void* retired=second.data();
    sb::validate_host_span(static_cast<uint8_t*>(p)+16,112,queue);
    invalid([&] {sb::validate_device_span(p,1,queue);});
    invalid([&] {sb::validate_host_span(device.data(),1,queue);});
    invalid([&] {sb::validate_host_span(p,129,queue);});
    invalid([&] {sb::validate_host_span(p,0,queue);});
    invalid([&] {sb::validate_host_span(nullptr,1,queue);});
    invalid([&] {sb::validate_host_span(p,1,other);});
    invalid([&] {sb::validate_host_span(foreign.get(),128,queue);});
    invalid([&] {sb::validate_host_span(
        reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()-3),8,queue);});
    std::vector<uint8_t> ordinary(128);
    invalid([&] {sb::validate_host_span(ordinary.data(),ordinary.size(),queue);});
    sb::Allocation moved(std::move(first));
    require(!first.data() && first.size()==0,"Host move source not empty");
    sb::validate_host_span(p,128,queue);
    second=std::move(moved);
    require(!moved.data() && moved.size()==0,"Host move assignment source not empty");
    sb::validate_host_span(p,128,queue);
    invalid([&] {sb::validate_host_span(retired,1,queue);});
    retired=device.data();device=std::move(second);
    sb::validate_host_span(p,128,queue);
    invalid([&] {sb::validate_device_span(p,1,queue);});
    invalid([&] {sb::validate_device_span(retired,1,queue);});
    sb::Allocation replacement(r,32,sb::MemoryKind::Device);
    device=std::move(replacement);
    sb::validate_device_span(device.data(),32,queue);
    invalid([&] {sb::validate_host_span(device.data(),1,queue);});
    invalid([&] {sb::validate_host_span(p,1,queue);});
}

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
    auto kc = codes, vc = codes;
    auto ks = scales, vs = scales;
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

void streamed_case(std::shared_ptr<sb::Runtime> r, int pos0) {
    constexpr int T=9,ld=526,pages=6,physical=2,page_size=4;
    auto [k,v]=q8_inputs(T,ld);const auto original_k=k,original_v=v;
    std::vector<int32_t> table(G+pages+G,123456);
    const int32_t mapping[]={1,-1,physical,-2,0,INT32_MAX};
    std::copy_n(mapping,pages,table.data()+G);
    Buffer<uint16_t> dk(r,k.size()),dv(r,v.size());Buffer<int32_t> dt(r,table.size());
    Q8Cache resident(r,physical,page_size),host(r,pages,page_size,sb::MemoryKind::Host),
            stage(r,pages,page_size);
    sycl::context context(r->device());
    sycl::queue other(context,r->device(),sycl::property::queue::in_order{});
    Drain drain{r,&other};
    dk.put(k);dv.put(v);dt.put(table);
    auto er=resident.empty(),eh=host.empty();resident.put(er);host.put(eh);stage.put(eh);
    const auto rp=resident.view(),hp=host.view(),sp=stage.view();
    const qa::Q8WritePools pools{rp.k_q,rp.v_q,rp.k_scale,rp.v_scale,dt.data()+G};
    auto shapes=sk::qsa_real_shapes();shapes.page_size=page_size;
    const sk::QsaPromptBf16Bounds bounds{pages,physical};
    auto run=[&] {
        qa::append_q8_streamed(dk.data()+G,dv.data()+G,T,ld,pos0,pools,hp,sp,
                               shapes,bounds,&r->compute());
        for (int t=0;t<T;++t) for (int h=0;h<2;++h) {
            const int pos=pos0+t,logical=pos/page_size,slot=table[G+logical];
            q8_oracle_cell(eh,(size_t(logical)*2+h)*page_size+pos%page_size,k,v,t,h,ld);
            if (slot>=0 && slot<physical)
                q8_oracle_cell(er,(size_t(slot)*2+h)*page_size+pos%page_size,k,v,t,h,ld);
        }
        require(host.get()==eh && stage.get()==eh && resident.get()==er,
                "streamed Q8 oracle/publication/guards/resident absence");
    };
    qa::validate_streamed_storage(pools,hp,sp,shapes,bounds,&r->compute());run();
    auto validate=[&](qa::Q8WritePools a,qa::Q8MirrorPools b,qa::Q8MirrorPools c,
                      sk::QsaPromptBf16Bounds d) {
        qa::validate_streamed_storage(a,b,c,shapes,d,&r->compute());
    };
    invalid([&] {validate(pools,sp,hp,bounds);});
    invalid([&] {auto x=sp;x.v_q=x.k_q;validate(pools,hp,x,bounds);});
    invalid([&] {auto x=pools;x.k_q=sp.k_q;validate(x,hp,sp,bounds);});
    invalid([&] {auto x=pools;x.page_table=reinterpret_cast<int32_t*>(sp.k_q);
        validate(x,hp,sp,bounds);});
    invalid([&] {auto x=sp;x.k_q=stage.k.data()+stage.k.count-1;validate(pools,hp,x,bounds);});
    invalid([&] {auto x=hp;x.k_scale=host.ks.data()+host.ks.count-1;validate(pools,x,sp,bounds);});
    std::vector<int8_t> ordinary(host.k.count,85);
    invalid([&] {auto x=hp;x.k_q=ordinary.data()+G;validate(pools,x,sp,bounds);});
    invalid([&] {auto x=hp;x.k_q=nullptr;validate(pools,x,sp,bounds);});
    invalid([&] {auto x=bounds;x.physical_pages=0;validate(pools,hp,sp,x);});
    invalid([&] {auto x=bounds;x.physical_pages=pages;validate(pools,hp,sp,x);});
    invalid([&] {auto x=bounds;x.logical_pages=65537;validate(pools,hp,sp,x);});
    invalid([&] {qa::validate_streamed_storage(pools,hp,sp,shapes,bounds,&other);});
    invalid([&] {qa::append_q8_streamed(reinterpret_cast<uint16_t*>(sp.k_q),dv.data()+G,
        T,ld,pos0,pools,hp,sp,shapes,bounds,&r->compute());});
    invalid([&] {qa::append_q8_streamed(dk.data()+dk.count-1,dv.data()+G,
        T,ld,pos0,pools,hp,sp,shapes,bounds,&r->compute());});
    invalid([&] {qa::append_q8_streamed(dk.data()+G,dv.data()+G,
        T,ld-15,pos0,pools,hp,sp,shapes,bounds,&r->compute());});
    invalid([&] {qa::append_q8_streamed(dk.data()+G,dv.data()+G,
        T,ld,-1,pools,hp,sp,shapes,bounds,&r->compute());});
    invalid([&] {qa::append_q8_streamed(dk.data()+G,dv.data()+G,
        0,ld,pos0,pools,hp,sp,shapes,bounds,&r->compute());});
    invalid([&] {qa::append_q8_streamed(dk.data()+G,dv.data()+G,
        8193,ld,0,pools,hp,sp,shapes,bounds,&r->compute());});
    invalid([&] {qa::append_q8_streamed(dk.data()+G,dv.data()+G,
        1,ld,pages*page_size,pools,hp,sp,shapes,bounds,&r->compute());});
    require(host.get()==eh && stage.get()==eh && resident.get()==er &&
        dk.get()==k && dv.get()==v && dt.get()==table,"streamed Q8 refusal modified an owner");
    // Nonfinite, zero, FP16-underflow and FP16-overflow groups retain the
    // existing append policy in all three stores. Recover in the same owners.
    k[G]=0x7fc1;v[G]=0x7f80;
    for (int d=64;d<256;++d) {
        const double x=d<128 ? 0. : d<192 ? std::ldexp(1.,-130) : std::ldexp(1.,120);
        k[G+d]=encode(x);v[G+d]=encode(-x);
    }
    dk.put(k);dv.put(v);run();
    k=original_k;v=original_v;dk.put(k);dv.put(v);run();
    require(dk.get()==k && dv.get()==v && dt.get()==table,"streamed Q8 immutable input/tail");
}

void streamed_full_extent_case(std::shared_ptr<sb::Runtime> r) {
    constexpr int cells=262144,T=8192,ld=526,page_size=4,pages=cells/page_size,physical=2;
    auto [k,v]=q8_inputs(T,ld);
    std::vector<int32_t> table(G+pages+G,123456);
    std::fill_n(table.data()+G,pages,-1);table[G]=1;table[G+pages-1]=0;
    Buffer<uint16_t> dk(r,k.size()),dv(r,v.size());Buffer<int32_t> dt(r,table.size());
    Q8Cache resident(r,physical,page_size),host(r,pages,page_size,sb::MemoryKind::Host),
            stage(r,pages,page_size);
    Drain drain{r};
    dk.put(k);dv.put(v);dt.put(table);
    resident.put(resident.empty());host.put(host.empty());stage.put(stage.empty());
    Q8Image expected{std::vector<int8_t>(G+size_t(T)*2*256+G,85),
        std::vector<int8_t>(G+size_t(T)*2*256+G,85),
        std::vector<uint16_t>(G+size_t(T)*2*4+G,0x3555),
        std::vector<uint16_t>(G+size_t(T)*2*4+G,0x3555)};
    for (int t=0;t<T;++t) for (int h=0;h<2;++h)
        q8_oracle_cell(expected,(size_t(t/page_size)*2+h)*page_size+t%page_size,k,v,t,h,ld);
    const auto rp=resident.view(),hp=host.view(),sp=stage.view();
    const qa::Q8WritePools pools{rp.k_q,rp.v_q,rp.k_scale,rp.v_scale,dt.data()+G};
    auto shapes=sk::qsa_real_shapes();shapes.page_size=page_size;
    const sk::QsaPromptBf16Bounds bounds{pages,physical};
    qa::validate_streamed_storage(pools,hp,sp,shapes,bounds,&r->compute());
    for (int pos=0;pos<cells;pos+=T)
        qa::append_q8_streamed(dk.data()+G,dv.data()+G,T,ld,pos,pools,hp,sp,
                               shapes,bounds,&r->compute());
    // Exercise an exact final-cell overwrite after filling every logical cell.
    qa::append_q8_streamed(dk.data()+G,dv.data()+G,1,ld,cells-1,pools,hp,sp,
                           shapes,bounds,&r->compute());
    invalid([&] {qa::append_q8_streamed(dk.data()+G,dv.data()+G,1,ld,cells,
        pools,hp,sp,shapes,bounds,&r->compute());});
    auto check=[&](const Q8Image& image,int pool_pages,bool mapped) {
        for (size_t i=0;i<G;++i) {
            require(image.k[i]==85 && image.v[i]==85 &&
                image.k[image.k.size()-1-i]==85 && image.v[image.v.size()-1-i]==85 &&
                image.ks[i]==0x3555 && image.vs[i]==0x3555 &&
                image.ks[image.ks.size()-1-i]==0x3555 && image.vs[image.vs.size()-1-i]==0x3555,
                "full-extent Q8 outside owner guard modified");
        }
        for (int p=0;p<pool_pages;++p) for (int h=0;h<2;++h) for (int c=0;c<page_size;++c) {
            const int logical=mapped ? (p==0 ? pages-1 : 0) : p;
            const int pos=logical*page_size+c,t=pos==cells-1 ? 0 : pos%T;
            const size_t row=(size_t(p)*2+h)*page_size+c,
                         ref=(size_t(t/page_size)*2+h)*page_size+t%page_size;
            for (int d=0;d<256;++d)
                require(image.k[G+row*256+d]==expected.k[G+ref*256+d] &&
                        image.v[G+row*256+d]==expected.v[G+ref*256+d],"full-extent Q8 code oracle");
            for (int g=0;g<4;++g)
                require(image.ks[G+row*4+g]==expected.ks[G+ref*4+g] &&
                        image.vs[G+row*4+g]==expected.vs[G+ref*4+g],"full-extent Q8 scale oracle");
        }
    };
    // Read the mapped Host allocation on the CPU only after a checked wait.
    const auto published=host.get();check(published,pages,false);
    const auto staged=stage.get();check(staged,pages,false);
    require(staged==published,"full-extent Q8 mirrors differ");
    check(resident.get(),physical,true);
    require(dk.get()==k && dv.get()==v && dt.get()==table,"full-extent Q8 inputs modified");
}
} // namespace
int main(int argc,char** argv) try {
    const bool full=argc==2 && std::string_view(argv[1])=="--streamed-full-extent";
    require(argc==1 || full,"usage: sycl_native16_qsa_test [--streamed-full-extent]");
    auto runtime = sb::runtime_for();
    qa::validate_device(&runtime->compute());
    host_registry_case(runtime);
    if (full) {
        streamed_full_extent_case(runtime);runtime->wait();
        std::cout << "QSA16 streamed KV producer physical262144 component PASS; attention/model/RESTORE/performance unqualified\n";
        return 0;
    }
    for (int T : {1, 3, 17}) for (int heads : {2, 24}) {
        norm_case(runtime, T, heads, 5, 0);
        norm_case(runtime, T, heads, 0, 0, false, true);
    }
    norm_case(runtime, 3, 24, 5, 0, true);
    norm_case(runtime, 1, 2, 0, 120);
    norm_case(runtime, 1, 24, 0, -128);
    for (int T : {1, 3, 17}) gate_case(runtime, T);
    append_case(runtime);
    // Cover every partial-page offset: trailing-only, both ends, leading-only,
    // and the final logical page. The oracle also checks untouched neighbours.
    for (int pos0 : {0,1,2,3,15}) streamed_case(runtime,pos0);
    runtime->wait();
    std::cout << "QSA16 local component screen PASS; attention/model/physical262144/RESTORE/performance unqualified\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "QSA16 component failure: " << e.what() << '\n';
    return 1;
}
