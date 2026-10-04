// src/kernels/qsa_prompt_attn_parity.cpp - perf-review D-1: the tensor-core prompt attention (qsa_prompt_attn.hpp)
// against the FP32 kernel it replaces (`qsa_decode_attn_batch`) and an FP64 host reference (GPU, synthetic, no model).
//
// Int8, FP16, Q4_0 and hybrid K8V4 pools with random codes, scales and queries; selections shaped like the prompt path's (the 2,051
// widest, a recent window plus older cells that drift slowly from one query to the next, so neighbours share most
// of them as they do in a real prompt; short contexts take every cell). Checks:
//   1. against FP64, the new kernel's error is no larger than a small multiple of the old kernel's (both FP32 math);
//   2. the new and old outputs agree to a relative 1e-4 of the output scale;
// then times both over a prompt chunk (the old one in batches of 32, as prefill.cpp calls it).
// HIP builds (S6): the same checks for the RDNA4 matrix-core kernel (STRATA_HIP_WMMA), skipped (77) off gfx12.
// Usage: qsa_prompt_attn_parity [context=32768] [queries=2048] [reps=5] [fallback_batch=0] [layout_variant=0]
// A positive fallback_batch compares that launch batch against the original 32-query batch, bitwise.
// Zero tests the opt-in XMX implementation (which does not accept Q4_0 K).
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_decode_attn_variant.hpp"
#include "strata/kernels/qsa_prompt_attn_xmx.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace strata::kernels {
bool qsa_prompt_attn_xmx_compact32(const float*, const QsaAttnPools&, const int32_t*, const int32_t*,
                                  int64_t, const QsaShapes&, float*, int64_t, void*);
}
namespace k = strata::kernels;

namespace {
void ck(dpct::err0 e, const char *w) {
    if (e != 0) {
        std::fprintf(stderr, "%s failed: SYCL migration error %d\n", w, (int) e);
        std::exit(2);
    }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(DPCT_CHECK_ERROR(
           d = (T *)sycl::malloc_device(h.size() * sizeof(T) + 64,
                                        dpct::get_in_order_queue())),
       "malloc");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
           d, h.data(), h.size() * sizeof(T)).wait()),
       "upload");
    return d;
}
float h2f(uint16_t b) {
    sycl::half h; *reinterpret_cast<uint16_t *>(&h) = b;
    return sycl::vec<sycl::half, 1>(h)
        .convert<float, sycl::rounding_mode::automatic>()[0];
}
uint16_t f2h(float f) {
    sycl::half h =
        sycl::vec<float, 1>(f)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    return *reinterpret_cast<uint16_t *>(&h);
}

int run(int fmt, int64_t ctx, int64_t nq, int reps, int64_t candidate_batch, int variant) {   // fmt 1 int8, 0 fp16, 2 q4_0, 3 hybrid K8V4
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t pages = (ctx + PS - 1) / PS, rows = pages * NKV * PS;
    std::mt19937 rng(1234 + fmt);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    // pools; page table a shuffled permutation (the readers must follow it)
    std::vector<int8_t> kq, vq;
    std::vector<uint16_t> ks, vs, kh, vh;
    std::vector<uint8_t> k4, v4;   // q4_0: 8 blocks of {fp16 d, 16 code bytes} per row (kv_q4.hpp)
    constexpr int64_t B4 = 18, ROW4 = 8 * B4;
    if (fmt == 2) {
        std::uniform_int_distribution<int> byte(0, 255);
        // the scales as a q4_0 pool holds them: signed (ggml's d = max / -8) and spread over two decades, so a chunk's
        // cells mix signs and magnitudes (a positive-only bound on them overflowed FP16 on real K/V)
        std::uniform_real_distribution<float> lg(std::log(0.01f), std::log(1.0f));
        auto sc4 = [&](std::mt19937& r) { return std::exp(lg(r)) * (byte(r) & 1 ? -1.0f : 1.0f); };
        k4.resize(rows * ROW4); v4.resize(rows * ROW4);
        for (auto* pool : {&k4, &v4})
            for (int64_t b = 0; b < rows * 8; ++b) {
                const uint16_t d = f2h(sc4(rng));
                std::memcpy(pool->data() + b * B4, &d, 2);
                for (int j = 0; j < 16; ++j) (*pool)[b * B4 + 2 + j] = (uint8_t) byte(rng);
            }
    } else if (fmt == 1 || fmt == 3) {
        kq.resize(rows * HD); vq.resize(rows * HD); ks.resize(rows * 4); vs.resize(rows * 4);
        for (auto& x : kq) x = (int8_t) code(rng);
        for (auto& x : vq) x = (int8_t) code(rng);
        for (auto& x : ks) x = f2h(sc(rng));
        for (auto& x : vs) x = f2h(sc(rng));
        if (fmt == 3) {
            // Signed Q4_0 V scales, using the same two-decade range as the Q4_0 case.
            v4.resize(rows * ROW4);
            std::uniform_int_distribution<int> byte(0, 255);
            std::uniform_real_distribution<float> lg(std::log(0.01f), std::log(1.0f));
            for (int64_t b = 0; b < rows * 8; ++b) {
                const uint16_t d = f2h(std::exp(lg(rng)) * (byte(rng) & 1 ? -1.0f : 1.0f));
                std::memcpy(v4.data() + b * B4, &d, 2);
                for (int j = 0; j < 16; ++j) v4[b * B4 + 2 + j] = (uint8_t) byte(rng);
            }
        }
    } else {
        kh.resize(rows * HD); vh.resize(rows * HD);
        for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
        for (auto& x : vh) x = f2h(nd(rng));
    }
    std::vector<int32_t> table(pages);
    for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    // queries at positions ctx - nq .. ctx - 1
    const int64_t cap = k::qsa_selection_width(ctx, s);
    std::vector<int32_t> ids((size_t) (nq * cap), 0), steps((size_t) (nq * k::kStepCount), 0);
    std::vector<float> q((size_t) (nq * NH * HD));
    for (auto& x : q) x = nd(rng) * 2.0f;
    std::vector<int32_t> old_cells;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t pos = ctx - nq + i, nkv = pos + 1;
        const int64_t w = k::qsa_selection_width(nkv, s);
        int32_t* sel = ids.data() + i * cap;
        steps[i * k::kStepCount + k::kStepWidth] = (int32_t) w;
        if (w == nkv) {
            for (int64_t c = 0; c < w; ++c) sel[c] = (int32_t) c;
            continue;
        }
        const int64_t recent = 512, older = w - recent;
        if ((int64_t) old_cells.size() != older) {   // the older cells: a set that drifts ~3% per query
            std::vector<int32_t> all((size_t) (nkv - recent));
            for (int64_t c = 0; c < nkv - recent; ++c) all[c] = (int32_t) c;
            std::shuffle(all.begin(), all.end(), rng);
            old_cells.assign(all.begin(), all.begin() + older);
        } else {
            std::uniform_int_distribution<int64_t> pick(0, older - 1), any(0, nkv - recent - 1);
            for (int r = 0; r < older / 32; ++r) {
                const int32_t c = (int32_t) any(rng);
                if (std::find(old_cells.begin(), old_cells.end(), c) == old_cells.end()) old_cells[pick(rng)] = c;
            }
        }
        std::vector<int32_t> v(old_cells);
        for (int64_t c = nkv - recent; c < nkv; ++c) v.push_back((int32_t) c);
        std::sort(v.begin(), v.end());
        std::copy(v.begin(), v.end(), sel);
    }
    k::QsaAttnPools pl;
    if (fmt == 2) { pl.k_q4 = up(k4); pl.v_q4 = up(v4); }
    else if (fmt == 3) { pl.k_q = up(kq); pl.k_scale = up(ks); pl.v_q4 = up(v4); }
    else if (fmt == 1) { pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs); }
    else { pl.k_pool = up(kh); pl.v_pool = up(vh); }
    // a q4_0 value: block d / 32 of the row, element j = d % 32 in the low nibble of byte j (j < 16), else the high
    // nibble of byte j - 16, minus 8, times the block's scale
    auto q4v = [&](const std::vector<uint8_t>& pool, int64_t row, int64_t d) {
        const uint8_t* blk = pool.data() + row * ROW4 + (d / 32) * B4;
        uint16_t sc;
        std::memcpy(&sc, blk, 2);
        const int j = (int) (d % 32);
        const int code = j < 16 ? (blk[2 + j] & 0x0F) : (blk[2 + j - 16] >> 4);
        return (double) (code - 8) * h2f(sc);
    };
    pl.page_table = up(table);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const float* d_q = up(q);
    float *d_old = nullptr, *d_new = nullptr, *scratch = nullptr, *d_tensor = nullptr;
    const int64_t batch = 32;
    ck(DPCT_CHECK_ERROR(d_old = (float *)sycl::malloc_device(
                            nq * NH * HD * 4, dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(d_new = (float *)sycl::malloc_device(
                            nq * NH * HD * 4, dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(
           scratch = (float *)sycl::malloc_device(
               std::max(batch, candidate_batch) * k::qsa_decode_attn_scratch_floats(cap, s) * 4,
               dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(d_tensor = sycl::malloc_device<float>((size_t) nq * NH * HD,
                                                             dpct::get_in_order_queue())), "malloc tensor ref");
    auto release_all = [&] {
    sycl::free((void *)d_ids, dpct::get_in_order_queue());
        sycl::free((void *)d_steps, dpct::get_in_order_queue());
        sycl::free((void *)d_q, dpct::get_in_order_queue());
        sycl::free((void *)d_old, dpct::get_in_order_queue());
        sycl::free((void *)d_new, dpct::get_in_order_queue());
    sycl::free((void *)scratch, dpct::get_in_order_queue());
    sycl::free((void *)pl.k_q, dpct::get_in_order_queue());
        sycl::free((void *)pl.v_q, dpct::get_in_order_queue());
        sycl::free((void *)pl.k_scale, dpct::get_in_order_queue());
        sycl::free((void *)pl.v_scale, dpct::get_in_order_queue());
    sycl::free((void *)pl.k_pool, dpct::get_in_order_queue());
        sycl::free((void *)pl.v_pool, dpct::get_in_order_queue());
        sycl::free((void *)pl.page_table, dpct::get_in_order_queue());
    sycl::free((void *)pl.k_q4, dpct::get_in_order_queue());
        sycl::free((void *)pl.v_q4, dpct::get_in_order_queue());
    sycl::free(d_tensor, dpct::get_in_order_queue());
    };
    auto old_run = [&]() {
        for (int64_t t0 = 0; t0 < nq; t0 += batch)
            k::qsa_decode_attn_batch(d_q + t0 * NH * HD, pl, d_ids + t0 * cap, d_steps + t0 * k::kStepCount, cap, s,
                                     scratch, d_old + t0 * NH * HD, std::min(batch, nq - t0), nullptr);
    };
    auto reference_run = [&]() {
        if (!k::qsa_prompt_attn_batch(d_q, pl, d_ids, d_steps, cap, s, d_tensor, nq, nullptr)) {
            std::fprintf(stderr, "original XMX CH32 refused supported pools\n");
            std::exit(2);
        }
    };
    auto new_run = [&]() {
        if (!k::qsa_prompt_attn_xmx_compact32(d_q, pl, d_ids, d_steps, cap, s, d_new, nq, nullptr)) {
            std::fprintf(stderr, "compact XMX CH32 refused supported pools\n");
            std::exit(2);
        }
    };
    if (fmt == 2) {
        const bool refused = !k::qsa_prompt_attn_batch(d_q, pl, d_ids, d_steps, cap, s, d_tensor, nq, nullptr)
                             && !k::qsa_prompt_attn_xmx_compact32(d_q, pl, d_ids, d_steps, cap, s, d_new, nq, nullptr);
        dpct::get_current_device().queues_wait_and_throw();
        std::printf("%s Q4_0 K refusal by both XMX kernels, ctx %lld, %lld queries\n",
                    refused ? "PASS" : "FAIL", (long long) ctx, (long long) nq);
        release_all();
        return refused ? 0 : 1;
    }
    old_run();
    reference_run();
    new_run();
    ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()),
       "run");
    std::vector<float> o((size_t) (nq * NH * HD)), nw(o.size());
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(o.data(), d_old, o.size() * 4)
                            .wait()),
       "down");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(nw.data(), d_new, nw.size() * 4)
                            .wait()),
       "down");
    std::vector<float> tensor(nw.size());
    dpct::get_in_order_queue().memcpy(tensor.data(), d_tensor, tensor.size() * sizeof(float)).wait();
    // 1. FP64 reference on a sample of queries
    double err_old = 0, err_new = 0, ref_scale = 0;
    for (int64_t i = 0; i < nq; i += std::max<int64_t>(1, nq / 16)) {
        const int64_t w = steps[i * k::kStepCount + k::kStepWidth];
        const int32_t* sel = ids.data() + i * cap;
        for (int64_t h = 0; h < NH; ++h) {
            const int64_t kvh = h / (NH / NKV);
            std::vector<double> sco((size_t) w);
            double mx = -1e300;
            for (int64_t c = 0; c < w; ++c) {
                const int64_t cell = sel[c], row = ((int64_t) table[cell / PS] * NKV + kvh) * PS + cell % PS;
                double a = 0;
                for (int64_t d = 0; d < HD; ++d) {
                    const double kv = fmt == 2 ? q4v(k4, row, d)
                                      : (fmt == 1 || fmt == 3) ? (double) kq[row * HD + d] * h2f(ks[row * 4 + d / 64]) : h2f(kh[row * HD + d]);
                    a += (double) q[(i * NH + h) * HD + d] * kv;
                }
                sco[c] = a / 16.0;
                mx = std::max(mx, sco[c]);
            }
            double l = 0;
            for (auto& x : sco) { x = std::exp(x - mx); l += x; }
            for (int64_t d = 0; d < HD; ++d) {
                double a = 0;
                for (int64_t c = 0; c < w; ++c) {
                    const int64_t cell = sel[c], row = ((int64_t) table[cell / PS] * NKV + kvh) * PS + cell % PS;
                    const double vv = (fmt == 2 || fmt == 3) ? q4v(v4, row, d)
                                      : fmt == 1 ? (double) vq[row * HD + d] * h2f(vs[row * 4 + d / 64]) : h2f(vh[row * HD + d]);
                    a += sco[c] * vv;
                }
                const double r = a / l;
                const size_t at = (size_t) ((i * NH + h) * HD + d);
                ref_scale = std::max(ref_scale, std::fabs(r));
                err_old = std::max(err_old, std::fabs(o[at] - r));
                err_new = std::max(err_new, std::fabs(nw[at] - r));
            }
        }
    }
    // 2. new vs old everywhere
    double diff = 0, scale = 0;
    bool finite = true;
    for (size_t i = 0; i < o.size(); ++i) {
        finite = finite && std::isfinite(o[i]) && std::isfinite(nw[i]) && std::isfinite(tensor[i]);
        diff = std::max(diff, (double) std::fabs(o[i] - nw[i]));
        scale = std::max(scale, (double) std::fabs(o[i]));
    }
    // 3. Warm both paths after the CPU oracle, then alternate measurement order.
    // Report the median of three rounds; the batch=32 control measures order/cache bias.
    reference_run();
    new_run();
    dpct::get_current_device().queues_wait_and_throw();
    auto time_run = [&](auto& launch) {
        dpct::get_current_device().queues_wait_and_throw();
        const auto begin = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) launch();
        dpct::get_current_device().queues_wait_and_throw();
        return (float) std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    };
    std::vector<float> old_times, new_times;
    for (int round = 0; round < 3; ++round) {
        if (round % 2 == 0) {
            old_times.push_back(time_run(reference_run));
            new_times.push_back(time_run(new_run));
        } else {
            new_times.push_back(time_run(new_run));
            old_times.push_back(time_run(reference_run));
        }
    }
    std::sort(old_times.begin(), old_times.end());
    std::sort(new_times.begin(), new_times.end());
    const float ms_old = old_times[1], ms_new = new_times[1];
    const bool ok1 = err_new <= std::max(4.0 * err_old, 1e-6 * ref_scale);
    const bool bits_equal = std::memcmp(tensor.data(), nw.data(), nw.size() * sizeof(float)) == 0;
    const bool ok2 = diff <= 1e-4 * scale && bits_equal;
    const bool ok = finite && ok1 && ok2;
    std::printf("%s %s ctx %lld, %lld queries: vs FP64 old %.3g new %.3g (output scale %.3g); compact vs FP32 %.3g (%.2g of "
                "scale); original XMX CH32 -> compact %.3f -> %.3f ms per chunk (%.2fx)\n",
                ok ? "PASS" : "FAIL", fmt == 3 ? "k8v4" : fmt == 2 ? "q4_0" : fmt == 1 ? "int8" : "fp16", (long long) ctx, (long long) nq, err_old,
                err_new, ref_scale, diff, diff / scale, ms_old / reps, ms_new / reps, ms_old / ms_new);
    double tensor_diff = 0;
    size_t mismatched = 0;
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor_diff = std::max(tensor_diff, (double) std::fabs(tensor[i] - nw[i]));
        if (std::memcmp(&tensor[i], &nw[i], sizeof(float)) != 0) ++mismatched;
    }
    std::printf("Compact vs original XMX CH32: %zu of %zu float bits differ; max %.9g\n",
                mismatched, tensor.size(), tensor_diff);
    release_all();
    return ok ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
    setenv("STRATA_PROMPT_ATTN_XMX", "32", 1);   // SYCL port: the batched kernel under test is the XMX one (opt-in in the engine)
#if defined(__HIP_PLATFORM_AMD__)
    // S6: on AMD the kernel under test is the RDNA4 matrix-core one (opt-in in the engine); other cards skip
    {
        int dev = 0;
        hipDeviceProp_t prop{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) return 2;
        if (std::strncmp(prop.gcnArchName, "gfx12", 5) != 0) {
            std::printf("SKIP: %s is not gfx12 (the matrix-core prompt attention is RDNA4 only)\n", prop.gcnArchName);
            return 77;
        }
#if defined(_WIN32)
        _putenv_s("STRATA_HIP_WMMA", "1");
#else
        setenv("STRATA_HIP_WMMA", "1", 1);
#endif
    }
#endif
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 32768;
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 2048;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;
    const int64_t candidate_batch = argc > 4 ? std::atoll(argv[4]) : 0;
    const int variant = argc > 5 ? std::atoi(argv[5]) : 0;
    if (variant < 0 || variant > 3 || (variant != 0 && candidate_batch == 0)) return 2;
    if (ctx < 1 || nq < 1 || nq > ctx || reps < 1 || candidate_batch < 0 || candidate_batch > 1024) {
        std::fprintf(stderr, "Invalid context, queries, repetitions, or fallback batch (0..1024)\n");
        return 2;
    }
    if (candidate_batch > 0)
        std::printf("XMX CH32 memory experiment (batch argument %lld, layout argument %d unused); require bit identity with original XMX and original FP32/FP64 accuracy\n",
                    (long long) candidate_batch, variant);
    int fails = 0;
    fails += run(1, ctx, nq, reps, candidate_batch, variant);
#if !defined(__HIP_PLATFORM_AMD__)
    if (candidate_batch > 0) {
        fails += run(3, ctx, nq, reps, candidate_batch, variant);
        fails += run(3, 1500, std::min<int64_t>(nq, 1500), reps, candidate_batch, variant);
    }
    fails += run(0, ctx, nq, reps, candidate_batch, variant);   // FP16 KV: the RDNA4 kernel takes int8 KV only
    fails += run(2, ctx, nq, reps, candidate_batch, variant);   // Q4_0 KV (mode 4)
    fails += run(2, 1500, std::min<int64_t>(nq, 1500), reps, candidate_batch, variant);
#endif
    fails += run(1, 1500, std::min<int64_t>(nq, 1500), reps, candidate_batch, variant);   // short context: the selection is every cell
    fails += run(1, 2100, std::min<int64_t>(nq, 256), reps, candidate_batch, variant);    // the identity-to-sparse edge
    std::printf("FAILURES: %d\n", fails);
    return fails;
}
