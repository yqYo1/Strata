// Proposal-only IQ3_S / Q8_K, two independent output rows, ten blocks.
// Adapted from iq_single_avx2.cpp and GGML x86 quants.c (MIT).
// Included inside strata::kernels::cpu's anonymous namespace.
static_assert(QK_K == 256 && sizeof(block_iq3_s) == 110 && sizeof(block_q8_K) == 292);
static_assert(offsetof(block_iq3_s, qs) == 2 && offsetof(block_iq3_s, qh) == 66);
static_assert(offsetof(block_iq3_s, signs) == 74 && offsetof(block_iq3_s, scales) == 106);
static_assert(offsetof(block_q8_K, d) == 0 && offsetof(block_q8_K, qs) == 4);

inline __m256i iq3s_pair_grid(const uint8_t* q, uint8_t h) {
#define IQ3S_PAIR_G(k) ((int)iq3s_grid[q[k] | (((uint32_t(h) >> (k)) & 1u) << 8)])
    return _mm256_set_epi32(IQ3S_PAIR_G(7), IQ3S_PAIR_G(6), IQ3S_PAIR_G(5), IQ3S_PAIR_G(4),
                            IQ3S_PAIR_G(3), IQ3S_PAIR_G(2), IQ3S_PAIR_G(1), IQ3S_PAIR_G(0));
#undef IQ3S_PAIR_G
}

// ROWS=1 is an exact bounded tail, never a speculative load of the next row.
template<int ROWS>
__attribute__((noinline)) void iq3s_pair_dot(const uint8_t* x, size_t stride,
                                          const void* activation, float* result) {
    static_assert(ROWS == 1 || ROWS == 2);
    const __m256i mask1 = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);
    const __m256i mask2 = _mm256_setr_epi8(
        1,2,4,8,16,32,64,(char)0x80,1,2,4,8,16,32,64,(char)0x80,
        1,2,4,8,16,32,64,(char)0x80,1,2,4,8,16,32,64,(char)0x80);
    const auto* y = static_cast<const uint8_t*>(activation);
    __m256 accumf[ROWS];
    for (int r = 0; r < ROWS; ++r) accumf[r] = _mm256_setzero_ps();
    for (int b = 0; b < 10; ++b) {
        __m256i sumi1[ROWS], sumi2[ROWS];
        for (int r = 0; r < ROWS; ++r) {
            sumi1[r] = _mm256_setzero_si256();
            sumi2[r] = _mm256_setzero_si256();
        }
        const uint8_t* ay = y + size_t(b) * sizeof(block_q8_K);
        for (int j = 0; j < 4; ++j) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(ay + 4 + 64*j));
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)(ay + 4 + 64*j + 32));
            // Expose independent row work within each 64-value group. The compiler
            // may schedule these loads/lookups together; assembly must verify it.
#pragma clang loop unroll(full)
            for (int r = 0; r < ROWS; ++r) {
                const uint8_t* xb = x + size_t(r)*stride + size_t(b)*sizeof(block_iq3_s);
                const uint8_t* q = xb + 2 + 16*j;
                const __m256i grid1 = iq3s_pair_grid(q, xb[66 + 2*j]);
                const __m256i grid2 = iq3s_pair_grid(q + 8, xb[66 + 2*j + 1]);
                // Byte pointers + memcpy avoid alignment and signed-shift UB.
                int32_t signs1, signs2;
                memcpy(&signs1, xb + 74 + 8*j, 4);
                memcpy(&signs2, xb + 74 + 8*j + 4, 4);
                const __m256i sm1 = _mm256_cmpeq_epi8(_mm256_and_si256(
                    _mm256_shuffle_epi8(_mm256_set1_epi32(signs1), mask1), mask2), mask2);
                const __m256i sm2 = _mm256_cmpeq_epi8(_mm256_and_si256(
                    _mm256_shuffle_epi8(_mm256_set1_epi32(signs2), mask1), mask2), mask2);
                const __m256i ys1 = _mm256_sub_epi8(_mm256_xor_si256(sm1, q8_1), sm1);
                const __m256i ys2 = _mm256_sub_epi8(_mm256_xor_si256(sm2, q8_2), sm2);
                const uint8_t sc = xb[106 + j];
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(grid1, ys1),
                                                   _mm256_set1_epi16(2*(sc & 15) + 1));
                const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(grid2, ys2),
                                                   _mm256_set1_epi16(2*(sc >> 4) + 1));
                sumi1[r] = _mm256_add_epi32(sumi1[r], p1);
                sumi2[r] = _mm256_add_epi32(sumi2[r], p2);
            }
        }
        float dy;
        memcpy(&dy, ay, sizeof(dy));
#pragma clang loop unroll(full)
        for (int r = 0; r < ROWS; ++r) {
            uint16_t dx;
            memcpy(&dx, x + size_t(r)*stride + size_t(b)*sizeof(block_iq3_s), sizeof(dx));
            const float d = GGML_CPU_FP16_TO_FP32(dx) * dy;
            accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])), accumf[r]);
        }
    }
    for (int r = 0; r < ROWS; ++r) result[r] = hsum_float_8(accumf[r]);
}
