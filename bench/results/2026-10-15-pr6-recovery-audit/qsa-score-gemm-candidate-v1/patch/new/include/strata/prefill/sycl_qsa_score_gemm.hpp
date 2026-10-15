#pragma once
#include "strata/kernels/qsa_select.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace strata::prefill {
enum class QsaScoreMode { off, fp16, bf16 };
inline QsaScoreMode parse_qsa_score_mode(const char* text) {
    if (!text) return QsaScoreMode::off;
    if (!std::strcmp(text, "fp16")) return QsaScoreMode::fp16;
    if (!std::strcmp(text, "bf16")) return QsaScoreMode::bf16;
    throw std::invalid_argument("STRATA_SYCL_PREFILL_QSA_SCORE_GEMM: expected fp16 or bf16, or unset for OFF");
}
// Frozen on the first startup sizing/init call; changing getenv later has no effect.
QsaScoreMode qsa_score_mode();
inline size_t qsa_score_scratch_bytes(int64_t blocks, int64_t batch) {
    if (blocks < 2 || blocks > 65538 || batch < 1 || batch > 256)
        throw std::invalid_argument("QSA score GEMM scratch geometry");
    return size_t(blocks + 1) * 128 * 2 + size_t(batch) * 512 * 2 +
           size_t(batch) * size_t(blocks + 1) * 4;
}
// Non-owning, QSA-arena-only view. No constructor/destructor allocations or waits.
// bind/invalidate and prepare are explicit because GDN/MoE overwrite this arena.
class QsaScoreGemm {
public:
    void bind(void* storage, size_t bytes, int64_t blocks, int64_t batch);
    void invalidate() { prepared_ = false; }
    void prepare(const float* pooled, const float* dead, int64_t last_n,
                 const kernels::QsaShapes& shape, void* stream);
    void scores(const float* query, const int32_t* steps, int64_t nq,
                int64_t active, float* output, void* stream);
private:
    QsaScoreMode mode_ = QsaScoreMode::off;
    uint16_t *key_ = nullptr, *query_ = nullptr;
    float* head_ = nullptr;
    const float *pooled_ = nullptr, *dead_ = nullptr;
    void* stream_ = nullptr;
    int64_t blocks_ = 0, batch_ = 0, live_ = 0;
    bool prepared_ = false;
};
} // namespace strata::prefill
