// Experimental score precision algorithm. Default OFF; prefill callers only.
#include "strata/prefill/sycl_qsa_score_gemm.hpp"
#include "strata/sycl/launch.hpp"
#include <oneapi/mkl/blas.hpp>
#include <cstdlib>

namespace strata::prefill {
namespace {
using namespace sycl_backend;
static_assert(sizeof(sycl::half) == 2 && sizeof(oneapi::mkl::bfloat16) == 2 && sizeof(float) == 4);
// A dedicated strided score product: A[nq,128], lda=512; B[N,128], ldb=128;
// C[nq,N], ldc=blocks+1. Never pass the interleaved A through Gemm::f16/bf16.
template<class Input>
void score_product(sycl::queue& queue, const uint16_t* a, const uint16_t* b,
                   float* c, int64_t rows, int64_t cols, int64_t ldc) {
    oneapi::mkl::blas::row_major::gemm(queue,
        oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
        rows, cols, int64_t(128), 1.f, reinterpret_cast<const Input*>(a), int64_t(512),
        reinterpret_cast<const Input*>(b), int64_t(128), 0.f, c, ldc);
}
template<class Input>
void convert(const float* input, uint16_t* output, int64_t count, sycl::queue& queue) {
    queue.parallel_for(sycl::range<1>(size_t(count)), [=](sycl::id<1> i) {
        reinterpret_cast<Input*>(output)[i[0]] = Input(input[i[0]]);
    });
}
} // namespace
QsaScoreMode qsa_score_mode() {
    static const QsaScoreMode frozen = [] {
        const auto mode = parse_qsa_score_mode(std::getenv("STRATA_SYCL_PREFILL_QSA_SCORE_GEMM"));
        if (mode != QsaScoreMode::off && std::getenv("STRATA_SELECT_OLD"))
            throw std::invalid_argument("QSA score GEMM conflicts with STRATA_SELECT_OLD");
        return mode;
    }();
    return frozen;
}
void QsaScoreGemm::bind(void* storage, size_t bytes, int64_t blocks, int64_t batch) {
    mode_ = qsa_score_mode();
    prepared_ = false;
    if (mode_ == QsaScoreMode::off) {
        key_ = query_ = nullptr; head_ = nullptr; return;
    }
    const size_t need = qsa_score_scratch_bytes(blocks, batch);
    if (!storage || uintptr_t(storage) % 4 || bytes < need)
        throw std::invalid_argument("QSA score GEMM arena capacity/alignment");
    key_ = static_cast<uint16_t*>(storage);
    query_ = key_ + (blocks + 1) * 128;
    head_ = reinterpret_cast<float*>(query_ + batch * 512);
    blocks_ = blocks; batch_ = batch;
}
void QsaScoreGemm::prepare(const float* pooled, const float* dead, int64_t last_n,
                         const kernels::QsaShapes& shape, void* stream) {
    prepared_ = false;
    if (mode_ == QsaScoreMode::off || !key_ || !stream ||
        shape.idx_dim != 128 || shape.idx_n_head != 4 || shape.idx_block != 4 ||
        last_n < 1 || last_n > 262144 || last_n / 4 > blocks_ - 2)
        throw std::invalid_argument("QSA score GEMM prepare geometry/queue");
    auto& queue = queue_for(stream); // Existing non-null in-order compute queue.
    live_ = last_n / 4;
    // Only finalized rows are consumed. In particular never load pooled[live_]
    // (the current tail/future slot can be uninitialized or changed by append).
    validate_spans({{key_, size_t(live_ + 1) * 128 * 2}},
                   {{pooled, size_t(blocks_) * 128 * 4}, {dead, 128 * 4}});
    pooled_ = pooled; dead_ = dead; stream_ = stream;
    if (mode_ == QsaScoreMode::fp16) {
        if (live_) convert<sycl::half>(pooled, key_, live_ * 128, queue);
        convert<sycl::half>(dead, key_ + live_ * 128, 128, queue);
    } else {
        if (live_) convert<oneapi::mkl::bfloat16>(pooled, key_, live_ * 128, queue);
        convert<oneapi::mkl::bfloat16>(dead, key_ + live_ * 128, 128, queue);
    }
    prepared_ = true;
}
void QsaScoreGemm::scores(const float* query, const int32_t* steps, int64_t nq,
                        int64_t active, float* output, void* stream) {
    if (!prepared_ || !stream || stream != stream_ || nq < 1 || nq > batch_ ||
        active < 1 || active > live_ + 1)
        throw std::invalid_argument("QSA score GEMM score geometry/epoch/queue");
    auto& queue = queue_for(stream);
    const int64_t blocks = blocks_, live = live_, ldc = blocks_ + 1;
    auto packed = query_; auto head = head_;
    auto pooled = pooled_; auto dead = dead_;
    validate_spans({{packed, size_t(nq) * 512 * 2},
                    {head, size_t(nq) * size_t(ldc) * 4},
                    {output, size_t(nq) * size_t(blocks) * 4}},
                   {{query, size_t(nq) * 512 * 4},
                    {steps, size_t(nq) * kernels::kStepCount * 4},
                    {key_, size_t(live + 1) * 128 * 2},
                    {pooled, size_t(blocks) * 128 * 4}, {dead, 128 * 4}});
    if (mode_ == QsaScoreMode::fp16) convert<sycl::half>(query, packed, nq * 512, queue);
    else convert<oneapi::mkl::bfloat16>(query, packed, nq * 512, queue);
    for (int h = 0; h < 4; ++h) {
        if (mode_ == QsaScoreMode::fp16)
            score_product<sycl::half>(queue, packed + h * 128, key_, head, nq, live + 1, ldc);
        else score_product<oneapi::mkl::bfloat16>(queue, packed + h * 128, key_, head, nq, live + 1, ldc);
        queue.parallel_for(sycl::nd_range<1>(size_t(nq * active) * 32, 32),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int64_t qi = it.get_group_linear_id() / active;
                const int64_t b = it.get_group_linear_id() % active;
                const auto st = steps + qi * kernels::kStepCount;
                const int64_t n = st[kernels::kStepNKv], nb = st[kernels::kStepNBid];
                // Match FP32 scorer write validity: future scores remain untouched.
                if (n < 0 || n > 262144 || nb != n / 4 || nb >= active || b > nb) return;
                const int lane = it.get_local_linear_id();
                auto sg = it.get_sub_group();
                float dot = head[qi * ldc + (b == nb ? live : b)];
                // Device-local numerical fallback: before ReLU, every lane sees
                // the same C cell. No host readback, atomic, wait, extra queue, or
                // second GEMM. Recompute this head using the original FP32 warp
                // arithmetic if GEMM produced NaN/Inf (including half overflow).
                const float* key = b == nb ? dead : pooled + b * 128;
                const float* q = query + qi * 512;
                if (!sycl::isfinite(dot)) {
                    dot = key[lane * 4] * q[h * 128 + lane * 4];
                    for (int d = 1; d < 4; ++d)
                        dot += key[lane * 4 + d] * q[h * 128 + lane * 4 + d];
                    for (int off = 16; off; off /= 2)
                        dot += sycl::permute_group_by_xor(sg, dot, off);
                }
                dot = sycl::select_from_group(sg, dot, sycl::id<1>(0));
                float total = (h ? output[qi * blocks + b] : 0.f) + (dot > 0 ? dot : 0.f);
                // Finite head results can overflow when added. At the final head
                // recompute the complete original score in that case, preserving
                // original NaN->zero ReLU and +Inf semantics without clamping.
                if (h == 3 && !sycl::isfinite(total)) {
                    total = 0;
                    for (int hh = 0; hh < 4; ++hh) {
                        float exact = key[lane * 4] * q[hh * 128 + lane * 4];
                        for (int d = 1; d < 4; ++d)
                            exact += key[lane * 4 + d] * q[hh * 128 + lane * 4 + d];
                        for (int off = 16; off; off /= 2)
                            exact += sycl::permute_group_by_xor(sg, exact, off);
                        total += exact > 0 ? exact : 0.f;
                    }
                }
                if (h == 3 && b == nb && n % 4) total += 1e9f;
                if (lane == 0) output[qi * blocks + b] = total;
            });
    }
}
} // namespace strata::prefill
