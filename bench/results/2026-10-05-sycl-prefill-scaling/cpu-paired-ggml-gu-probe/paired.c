#define ggml_vec_dot_iq1_m_q8_K unused_ggml_vec_dot_iq1_m_q8_K
#define ggml_vec_dot_iq1_s_q8_K unused_ggml_vec_dot_iq1_s_q8_K
#define ggml_vec_dot_iq2_s_q8_K unused_ggml_vec_dot_iq2_s_q8_K
#define ggml_vec_dot_iq2_xs_q8_K unused_ggml_vec_dot_iq2_xs_q8_K
#define ggml_vec_dot_iq2_xxs_q8_K unused_ggml_vec_dot_iq2_xxs_q8_K
#define ggml_vec_dot_iq3_s_q8_K unused_ggml_vec_dot_iq3_s_q8_K
#define ggml_vec_dot_iq3_xxs_q8_K unused_ggml_vec_dot_iq3_xxs_q8_K
#define ggml_vec_dot_iq4_nl_q8_0 unused_ggml_vec_dot_iq4_nl_q8_0
#define ggml_vec_dot_iq4_xs_q8_K unused_ggml_vec_dot_iq4_xs_q8_K
#define ggml_vec_dot_mxfp4_q8_0 unused_ggml_vec_dot_mxfp4_q8_0
#define ggml_vec_dot_nvfp4_q8_0 unused_ggml_vec_dot_nvfp4_q8_0
#define ggml_vec_dot_q1_0_q8_0 unused_ggml_vec_dot_q1_0_q8_0
#define ggml_vec_dot_q2_K_q8_K unused_ggml_vec_dot_q2_K_q8_K
#define ggml_vec_dot_q3_K_q8_K unused_ggml_vec_dot_q3_K_q8_K
#define ggml_vec_dot_q4_0_q8_0 unused_ggml_vec_dot_q4_0_q8_0
#define ggml_vec_dot_q4_1_q8_1 unused_ggml_vec_dot_q4_1_q8_1
#define ggml_vec_dot_q4_K_q8_K unused_ggml_vec_dot_q4_K_q8_K
#define ggml_vec_dot_q5_0_q8_0 unused_ggml_vec_dot_q5_0_q8_0
#define ggml_vec_dot_q5_1_q8_1 unused_ggml_vec_dot_q5_1_q8_1
#define ggml_vec_dot_q5_K_q8_K unused_ggml_vec_dot_q5_K_q8_K
#define ggml_vec_dot_q6_K_q8_K unused_ggml_vec_dot_q6_K_q8_K
#define ggml_vec_dot_q8_0_q8_0 unused_ggml_vec_dot_q8_0_q8_0
#define ggml_vec_dot_tq1_0_q8_K unused_ggml_vec_dot_tq1_0_q8_K
#define ggml_vec_dot_tq2_0_q8_K unused_ggml_vec_dot_tq2_0_q8_K
#define quantize_row_q8_0 unused_quantize_row_q8_0
#define quantize_row_q8_1 unused_quantize_row_q8_1
#define quantize_row_q8_K unused_quantize_row_q8_K
#include "/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml/src/ggml-cpu/arch/x86/quants.c"
void paired_iq3_xxs(int n,float* gate,float* up,const void* vg,const void* vu,const void* vy) {
    assert(n % QK_K == 0);
    const block_iq3_xxs* GGML_RESTRICT x_g=vg;
    const block_iq3_xxs* GGML_RESTRICT x_u=vu;
    const block_q8_K* GGML_RESTRICT y=vy;
    const int nb=n/QK_K;

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;



    __m256 accumf_g = _mm256_setzero_ps();
__m256 accumf_u = _mm256_setzero_ps();

uint32_t aux32_g[2];
uint32_t aux32_u[2];
    for(int i=0;i<nb;++i) {
const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const float d_g = GGML_CPU_FP16_TO_FP32(x_g[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3_g = x_g[i].qs;
        const uint8_t * GGML_RESTRICT gas_g = x_g[i].qs + QK_K/4;

        __m256i sumi1_g = _mm256_setzero_si256();
        __m256i sumi2_g = _mm256_setzero_si256();

        const float d_u = GGML_CPU_FP16_TO_FP32(x_u[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3_u = x_u[i].qs;
        const uint8_t * GGML_RESTRICT gas_u = x_u[i].qs + QK_K/4;

        __m256i sumi1_u = _mm256_setzero_si256();
        __m256i sumi2_u = _mm256_setzero_si256();

        for(int ib32=0;ib32<QK_K/32;ib32+=2) {
const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

            const __m256i q2_1_g = _mm256_set_epi32(iq3xxs_grid[q3_g[7]], iq3xxs_grid[q3_g[6]], iq3xxs_grid[q3_g[5]], iq3xxs_grid[q3_g[4]],
                                                  iq3xxs_grid[q3_g[3]], iq3xxs_grid[q3_g[2]], iq3xxs_grid[q3_g[1]], iq3xxs_grid[q3_g[0]]);
            q3_g += 8;
            const __m256i q2_2_g = _mm256_set_epi32(iq3xxs_grid[q3_g[7]], iq3xxs_grid[q3_g[6]], iq3xxs_grid[q3_g[5]], iq3xxs_grid[q3_g[4]],
                                                  iq3xxs_grid[q3_g[3]], iq3xxs_grid[q3_g[2]], iq3xxs_grid[q3_g[1]], iq3xxs_grid[q3_g[0]]);
            q3_g += 8;
            memcpy(aux32_g, gas_g, 8); gas_g += 8;
            const __m256i s2_1_g = _mm256_set_epi64x(signs64[(aux32_g[0] >> 21) & 127], signs64[(aux32_g[0] >> 14) & 127],
                                                   signs64[(aux32_g[0] >>  7) & 127], signs64[(aux32_g[0] >>  0) & 127]);
            const __m256i s2_2_g = _mm256_set_epi64x(signs64[(aux32_g[1] >> 21) & 127], signs64[(aux32_g[1] >> 14) & 127],
                                                   signs64[(aux32_g[1] >>  7) & 127], signs64[(aux32_g[1] >>  0) & 127]);
            const __m256i q8s_1_g = _mm256_sign_epi8(q8_1, s2_1_g);
            const __m256i q8s_2_g = _mm256_sign_epi8(q8_2, s2_2_g);
            const __m256i dot1_g  = _mm256_maddubs_epi16(q2_1_g, q8s_1_g);
            const __m256i dot2_g  = _mm256_maddubs_epi16(q2_2_g, q8s_2_g);
            const uint16_t ls1_g = aux32_g[0] >> 28;
            const uint16_t ls2_g = aux32_g[1] >> 28;
            const __m256i p1_g = _mm256_madd_epi16(dot1_g, _mm256_set1_epi16(2*ls1_g+1));
            const __m256i p2_g = _mm256_madd_epi16(dot2_g, _mm256_set1_epi16(2*ls2_g+1));
            sumi1_g = _mm256_add_epi32(sumi1_g, p1_g);
            sumi2_g = _mm256_add_epi32(sumi2_g, p2_g);

            const __m256i q2_1_u = _mm256_set_epi32(iq3xxs_grid[q3_u[7]], iq3xxs_grid[q3_u[6]], iq3xxs_grid[q3_u[5]], iq3xxs_grid[q3_u[4]],
                                                  iq3xxs_grid[q3_u[3]], iq3xxs_grid[q3_u[2]], iq3xxs_grid[q3_u[1]], iq3xxs_grid[q3_u[0]]);
            q3_u += 8;
            const __m256i q2_2_u = _mm256_set_epi32(iq3xxs_grid[q3_u[7]], iq3xxs_grid[q3_u[6]], iq3xxs_grid[q3_u[5]], iq3xxs_grid[q3_u[4]],
                                                  iq3xxs_grid[q3_u[3]], iq3xxs_grid[q3_u[2]], iq3xxs_grid[q3_u[1]], iq3xxs_grid[q3_u[0]]);
            q3_u += 8;
            memcpy(aux32_u, gas_u, 8); gas_u += 8;
            const __m256i s2_1_u = _mm256_set_epi64x(signs64[(aux32_u[0] >> 21) & 127], signs64[(aux32_u[0] >> 14) & 127],
                                                   signs64[(aux32_u[0] >>  7) & 127], signs64[(aux32_u[0] >>  0) & 127]);
            const __m256i s2_2_u = _mm256_set_epi64x(signs64[(aux32_u[1] >> 21) & 127], signs64[(aux32_u[1] >> 14) & 127],
                                                   signs64[(aux32_u[1] >>  7) & 127], signs64[(aux32_u[1] >>  0) & 127]);
            const __m256i q8s_1_u = _mm256_sign_epi8(q8_1, s2_1_u);
            const __m256i q8s_2_u = _mm256_sign_epi8(q8_2, s2_2_u);
            const __m256i dot1_u  = _mm256_maddubs_epi16(q2_1_u, q8s_1_u);
            const __m256i dot2_u  = _mm256_maddubs_epi16(q2_2_u, q8s_2_u);
            const uint16_t ls1_u = aux32_u[0] >> 28;
            const uint16_t ls2_u = aux32_u[1] >> 28;
            const __m256i p1_u = _mm256_madd_epi16(dot1_u, _mm256_set1_epi16(2*ls1_u+1));
            const __m256i p2_u = _mm256_madd_epi16(dot2_u, _mm256_set1_epi16(2*ls2_u+1));
            sumi1_u = _mm256_add_epi32(sumi1_u, p1_u);
            sumi2_u = _mm256_add_epi32(sumi2_u, p2_u);

        }


        accumf_g = _mm256_fmadd_ps(_mm256_set1_ps(d_g), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_g, sumi2_g)), accumf_g);



        accumf_u = _mm256_fmadd_ps(_mm256_set1_ps(d_u), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_u, sumi2_u)), accumf_u);


    }


    *gate = 0.25f * hsum_float_8(accumf_g);



    *up = 0.25f * hsum_float_8(accumf_u);

}

void paired_iq3_s(int n,float* gate,float* up,const void* vg,const void* vu,const void* vy) {
    assert(n % QK_K == 0);
    const block_iq3_s* GGML_RESTRICT x_g=vg;
    const block_iq3_s* GGML_RESTRICT x_u=vu;
    const block_q8_K* GGML_RESTRICT y=vy;
    const int nb=n/QK_K;

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);

    typedef union {
        __m256i  vec[2];
        uint32_t index[16];
    } index_t;



    __m256 accumf_g = _mm256_setzero_ps();
__m256 accumf_u = _mm256_setzero_ps();

index_t idx_g;
index_t idx_u;
    for(int i=0;i<nb;++i) {
const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const float d_g = GGML_CPU_FP16_TO_FP32(x_g[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs_g = x_g[i].qs;
        const uint8_t * GGML_RESTRICT qh_g = x_g[i].qh;
        const uint16_t * GGML_RESTRICT signs_g = (const uint16_t *)x_g[i].signs;

        __m256i sumi1_g = _mm256_setzero_si256();
        __m256i sumi2_g = _mm256_setzero_si256();

        const float d_u = GGML_CPU_FP16_TO_FP32(x_u[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs_u = x_u[i].qs;
        const uint8_t * GGML_RESTRICT qh_u = x_u[i].qh;
        const uint16_t * GGML_RESTRICT signs_u = (const uint16_t *)x_u[i].signs;

        __m256i sumi1_u = _mm256_setzero_si256();
        __m256i sumi2_u = _mm256_setzero_si256();

        for(int ib32=0;ib32<QK_K/32;ib32+=2) {
const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

            const __m256i idx_l_g = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs_g)); qs_g += 16;
            idx_g.vec[0] = _mm256_set1_epi32(qh_g[ib32+0]);
            idx_g.vec[1] = _mm256_set1_epi32(qh_g[ib32+1]);
            idx_g.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx_g.vec[0], idx_shift), idx_mask);
            idx_g.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx_g.vec[1], idx_shift), idx_mask);
            idx_g.vec[0] = _mm256_or_si256(idx_g.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l_g)));
            idx_g.vec[1] = _mm256_or_si256(idx_g.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l_g, 1)));

            // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
            //const __m256i q2_1_g = _mm256_i32gather_epi32((const int *)iq3s_grid, idx_g.vec[0], 4);
            //const __m256i q2_2_g = _mm256_i32gather_epi32((const int *)iq3s_grid, idx_g.vec[1], 4);
            const __m256i q2_1_g = _mm256_set_epi32(
                    iq3s_grid[idx_g.index[7]], iq3s_grid[idx_g.index[6]], iq3s_grid[idx_g.index[5]], iq3s_grid[idx_g.index[4]],
                    iq3s_grid[idx_g.index[3]], iq3s_grid[idx_g.index[2]], iq3s_grid[idx_g.index[1]], iq3s_grid[idx_g.index[0]]
            );
            const __m256i q2_2_g = _mm256_set_epi32(
                    iq3s_grid[idx_g.index[15]], iq3s_grid[idx_g.index[14]], iq3s_grid[idx_g.index[13]], iq3s_grid[idx_g.index[12]],
                    iq3s_grid[idx_g.index[11]], iq3s_grid[idx_g.index[10]], iq3s_grid[idx_g.index[ 9]], iq3s_grid[idx_g.index[ 8]]
            );

            __m256i aux256_g = _mm256_set1_epi32(signs_g[0] | (signs_g[1] << 16));
            aux256_g = _mm256_and_si256(_mm256_shuffle_epi8(aux256_g,mask1), mask2);
            const __m256i s2_1_g = _mm256_cmpeq_epi8(aux256_g, mask2);
            const __m256i q8s_1_g = _mm256_sub_epi8(_mm256_xor_si256(s2_1_g, q8_1), s2_1_g);

            aux256_g = _mm256_set1_epi32(signs_g[2] | (signs_g[3] << 16));
            aux256_g = _mm256_and_si256(_mm256_shuffle_epi8(aux256_g,mask1), mask2);
            const __m256i s2_2_g = _mm256_cmpeq_epi8(aux256_g, mask2);
            const __m256i q8s_2_g = _mm256_sub_epi8(_mm256_xor_si256(s2_2_g, q8_2), s2_2_g);

            signs_g += 4;

            const __m256i dot1_g  = _mm256_maddubs_epi16(q2_1_g, q8s_1_g);
            const __m256i dot2_g  = _mm256_maddubs_epi16(q2_2_g, q8s_2_g);
            const uint16_t ls1_g = x_g[i].scales[ib32/2] & 0xf;
            const uint16_t ls2_g = x_g[i].scales[ib32/2] >>  4;
            const __m256i p1_g = _mm256_madd_epi16(dot1_g, _mm256_set1_epi16(2*ls1_g+1));
            const __m256i p2_g = _mm256_madd_epi16(dot2_g, _mm256_set1_epi16(2*ls2_g+1));
            sumi1_g = _mm256_add_epi32(sumi1_g, p1_g);
            sumi2_g = _mm256_add_epi32(sumi2_g, p2_g);

            const __m256i idx_l_u = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs_u)); qs_u += 16;
            idx_u.vec[0] = _mm256_set1_epi32(qh_u[ib32+0]);
            idx_u.vec[1] = _mm256_set1_epi32(qh_u[ib32+1]);
            idx_u.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx_u.vec[0], idx_shift), idx_mask);
            idx_u.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx_u.vec[1], idx_shift), idx_mask);
            idx_u.vec[0] = _mm256_or_si256(idx_u.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l_u)));
            idx_u.vec[1] = _mm256_or_si256(idx_u.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l_u, 1)));

            // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
            //const __m256i q2_1_u = _mm256_i32gather_epi32((const int *)iq3s_grid, idx_u.vec[0], 4);
            //const __m256i q2_2_u = _mm256_i32gather_epi32((const int *)iq3s_grid, idx_u.vec[1], 4);
            const __m256i q2_1_u = _mm256_set_epi32(
                    iq3s_grid[idx_u.index[7]], iq3s_grid[idx_u.index[6]], iq3s_grid[idx_u.index[5]], iq3s_grid[idx_u.index[4]],
                    iq3s_grid[idx_u.index[3]], iq3s_grid[idx_u.index[2]], iq3s_grid[idx_u.index[1]], iq3s_grid[idx_u.index[0]]
            );
            const __m256i q2_2_u = _mm256_set_epi32(
                    iq3s_grid[idx_u.index[15]], iq3s_grid[idx_u.index[14]], iq3s_grid[idx_u.index[13]], iq3s_grid[idx_u.index[12]],
                    iq3s_grid[idx_u.index[11]], iq3s_grid[idx_u.index[10]], iq3s_grid[idx_u.index[ 9]], iq3s_grid[idx_u.index[ 8]]
            );

            __m256i aux256_u = _mm256_set1_epi32(signs_u[0] | (signs_u[1] << 16));
            aux256_u = _mm256_and_si256(_mm256_shuffle_epi8(aux256_u,mask1), mask2);
            const __m256i s2_1_u = _mm256_cmpeq_epi8(aux256_u, mask2);
            const __m256i q8s_1_u = _mm256_sub_epi8(_mm256_xor_si256(s2_1_u, q8_1), s2_1_u);

            aux256_u = _mm256_set1_epi32(signs_u[2] | (signs_u[3] << 16));
            aux256_u = _mm256_and_si256(_mm256_shuffle_epi8(aux256_u,mask1), mask2);
            const __m256i s2_2_u = _mm256_cmpeq_epi8(aux256_u, mask2);
            const __m256i q8s_2_u = _mm256_sub_epi8(_mm256_xor_si256(s2_2_u, q8_2), s2_2_u);

            signs_u += 4;

            const __m256i dot1_u  = _mm256_maddubs_epi16(q2_1_u, q8s_1_u);
            const __m256i dot2_u  = _mm256_maddubs_epi16(q2_2_u, q8s_2_u);
            const uint16_t ls1_u = x_u[i].scales[ib32/2] & 0xf;
            const uint16_t ls2_u = x_u[i].scales[ib32/2] >>  4;
            const __m256i p1_u = _mm256_madd_epi16(dot1_u, _mm256_set1_epi16(2*ls1_u+1));
            const __m256i p2_u = _mm256_madd_epi16(dot2_u, _mm256_set1_epi16(2*ls2_u+1));
            sumi1_u = _mm256_add_epi32(sumi1_u, p1_u);
            sumi2_u = _mm256_add_epi32(sumi2_u, p2_u);

        }


        accumf_g = _mm256_fmadd_ps(_mm256_set1_ps(d_g), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_g, sumi2_g)), accumf_g);



        accumf_u = _mm256_fmadd_ps(_mm256_set1_ps(d_u), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_u, sumi2_u)), accumf_u);


    }


    *gate = hsum_float_8(accumf_g);



    *up = hsum_float_8(accumf_u);

}

void paired_iq2_s(int n,float* gate,float* up,const void* vg,const void* vu,const void* vy) {
    assert(n % QK_K == 0);
    const block_iq2_s* GGML_RESTRICT x_g=vg;
    const block_iq2_s* GGML_RESTRICT x_u=vu;
    const block_q8_K* GGML_RESTRICT y=vy;
    const int nb=n/QK_K;

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);



    __m256 accumf_g = _mm256_setzero_ps();
__m256 accumf_u = _mm256_setzero_ps();

uint64_t aux64_g;
uint64_t aux64_u;
    for(int i=0;i<nb;++i) {
const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const float d_g = GGML_CPU_FP16_TO_FP32(x_g[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs_g = x_g[i].qs;
        const uint8_t * GGML_RESTRICT qh_g = x_g[i].qh;
        const uint16_t * GGML_RESTRICT signs_g = (const uint16_t *)(x_g[i].qs + QK_K/8);

        memcpy(&aux64_g, x_g[i].scales, 8);
        const __m128i scales8_g = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64_g >> 4, aux64_g), m4), 1), m1);
        const __m256i scales16_g = _mm256_cvtepi8_epi16(scales8_g); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        __m256i sumi1_g = _mm256_setzero_si256();
        __m256i sumi2_g = _mm256_setzero_si256();

        const float d_u = GGML_CPU_FP16_TO_FP32(x_u[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs_u = x_u[i].qs;
        const uint8_t * GGML_RESTRICT qh_u = x_u[i].qh;
        const uint16_t * GGML_RESTRICT signs_u = (const uint16_t *)(x_u[i].qs + QK_K/8);

        memcpy(&aux64_u, x_u[i].scales, 8);
        const __m128i scales8_u = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64_u >> 4, aux64_u), m4), 1), m1);
        const __m256i scales16_u = _mm256_cvtepi8_epi16(scales8_u); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        __m256i sumi1_u = _mm256_setzero_si256();
        __m256i sumi2_u = _mm256_setzero_si256();

        for(int ib32=0;ib32<QK_K/32;ib32+=2) {
const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

            const __m256i q2_1_g = _mm256_set_epi64x(iq2s_grid[qs_g[3] | ((qh_g[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs_g[2] | ((qh_g[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs_g[1] | ((qh_g[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs_g[0] | ((qh_g[ib32+0] << 8) & 0x300)]);
            const __m256i q2_2_g = _mm256_set_epi64x(iq2s_grid[qs_g[7] | ((qh_g[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs_g[6] | ((qh_g[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs_g[5] | ((qh_g[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs_g[4] | ((qh_g[ib32+1] << 8) & 0x300)]);
            qs_g += 8;

            __m256i aux256_g = _mm256_set1_epi32(signs_g[0] | ((uint32_t) signs_g[1] << 16));
            aux256_g = _mm256_and_si256(_mm256_shuffle_epi8(aux256_g,mask1), mask2);
            const __m256i s2_1_g = _mm256_cmpeq_epi8(aux256_g, mask2);
            const __m256i q8s_1_g = _mm256_sub_epi8(_mm256_xor_si256(s2_1_g, q8_1), s2_1_g);

            aux256_g = _mm256_set1_epi32(signs_g[2] | ((uint32_t) signs_g[3] << 16));
            aux256_g = _mm256_and_si256(_mm256_shuffle_epi8(aux256_g,mask1), mask2);
            const __m256i s2_2_g = _mm256_cmpeq_epi8(aux256_g, mask2);
            const __m256i q8s_2_g = _mm256_sub_epi8(_mm256_xor_si256(s2_2_g, q8_2), s2_2_g);

            signs_g += 4;

            const __m256i dot1_g  = _mm256_maddubs_epi16(q2_1_g, q8s_1_g); // blocks 2*ib32+0, 2*ib32+1
            const __m256i dot2_g  = _mm256_maddubs_epi16(q2_2_g, q8s_2_g); // blocks 2*ib32+2, 2*ib32+3

            const __m256i p1_g = _mm256_madd_epi16(dot1_g, _mm256_shuffle_epi8(scales16_g, get_scale_shuffle_k4(ib32+0)));
            const __m256i p2_g = _mm256_madd_epi16(dot2_g, _mm256_shuffle_epi8(scales16_g, get_scale_shuffle_k4(ib32+1)));
            sumi1_g = _mm256_add_epi32(sumi1_g, p1_g);
            sumi2_g = _mm256_add_epi32(sumi2_g, p2_g);

            const __m256i q2_1_u = _mm256_set_epi64x(iq2s_grid[qs_u[3] | ((qh_u[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs_u[2] | ((qh_u[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs_u[1] | ((qh_u[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs_u[0] | ((qh_u[ib32+0] << 8) & 0x300)]);
            const __m256i q2_2_u = _mm256_set_epi64x(iq2s_grid[qs_u[7] | ((qh_u[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs_u[6] | ((qh_u[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs_u[5] | ((qh_u[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs_u[4] | ((qh_u[ib32+1] << 8) & 0x300)]);
            qs_u += 8;

            __m256i aux256_u = _mm256_set1_epi32(signs_u[0] | ((uint32_t) signs_u[1] << 16));
            aux256_u = _mm256_and_si256(_mm256_shuffle_epi8(aux256_u,mask1), mask2);
            const __m256i s2_1_u = _mm256_cmpeq_epi8(aux256_u, mask2);
            const __m256i q8s_1_u = _mm256_sub_epi8(_mm256_xor_si256(s2_1_u, q8_1), s2_1_u);

            aux256_u = _mm256_set1_epi32(signs_u[2] | ((uint32_t) signs_u[3] << 16));
            aux256_u = _mm256_and_si256(_mm256_shuffle_epi8(aux256_u,mask1), mask2);
            const __m256i s2_2_u = _mm256_cmpeq_epi8(aux256_u, mask2);
            const __m256i q8s_2_u = _mm256_sub_epi8(_mm256_xor_si256(s2_2_u, q8_2), s2_2_u);

            signs_u += 4;

            const __m256i dot1_u  = _mm256_maddubs_epi16(q2_1_u, q8s_1_u); // blocks 2*ib32+0, 2*ib32+1
            const __m256i dot2_u  = _mm256_maddubs_epi16(q2_2_u, q8s_2_u); // blocks 2*ib32+2, 2*ib32+3

            const __m256i p1_u = _mm256_madd_epi16(dot1_u, _mm256_shuffle_epi8(scales16_u, get_scale_shuffle_k4(ib32+0)));
            const __m256i p2_u = _mm256_madd_epi16(dot2_u, _mm256_shuffle_epi8(scales16_u, get_scale_shuffle_k4(ib32+1)));
            sumi1_u = _mm256_add_epi32(sumi1_u, p1_u);
            sumi2_u = _mm256_add_epi32(sumi2_u, p2_u);

        }


        accumf_g = _mm256_fmadd_ps(_mm256_set1_ps(d_g), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_g, sumi2_g)), accumf_g);



        accumf_u = _mm256_fmadd_ps(_mm256_set1_ps(d_u), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_u, sumi2_u)), accumf_u);


    }


    *gate = 0.125f * hsum_float_8(accumf_g);



    *up = 0.125f * hsum_float_8(accumf_u);

}
