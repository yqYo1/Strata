from pathlib import Path
import hashlib,json
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05');out=Path(__file__).resolve().parent
source=(root/'src/kernels/cpu/iq_avx2.cpp').read_text()
pair='''// Decode independent gate/up blocks together, sharing activation loads.
// Each row keeps its original integer and FP32 accumulation order.
template <int TY, int NT>
inline void row_dot_pair(const uint8_t* gate, const uint8_t* up, int nblocks,
                          const block_q8_K* const* y, float* gres, float* ures) {
    __m256 gf[NT], uf[NT];
    for (int t = 0; t < NT; ++t) gf[t] = uf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* gb = gate + (size_t) i * Fmt32<TY>::bytes;
        const uint8_t* ub = up + (size_t) i * Fmt32<TY>::bytes;
        rows_ahead(gb + prefetch_ahead); rows_ahead(ub + prefetch_ahead);
        __m256i gi[NT], ui[NT];
        for (int t = 0; t < NT; ++t) gi[t] = ui[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                __m256i gg, gs, gc, ug, us, uc;
                Fmt32<TY>::decode(gb, j, half, gg, gs, gc);
                Fmt32<TY>::decode(ub, j, half, ug, us, uc);
                const int off = 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + off));
                    gi[t] = _mm256_add_epi32(gi[t], _mm256_madd_epi16(
                        _mm256_maddubs_epi16(gg, _mm256_sign_epi8(yv, gs)), gc));
                    ui[t] = _mm256_add_epi32(ui[t], _mm256_madd_epi16(
                        _mm256_maddubs_epi16(ug, _mm256_sign_epi8(yv, us)), uc));
                }
            }
        }
        const float gd = h2f(u16(gb)) * Fmt32<TY>::K;
        const float ud = h2f(u16(ub)) * Fmt32<TY>::K;
        for (int t = 0; t < NT; ++t) {
            gf[t] = _mm256_fmadd_ps(_mm256_set1_ps(gd * y[t][i].d), _mm256_cvtepi32_ps(gi[t]), gf[t]);
            uf[t] = _mm256_fmadd_ps(_mm256_set1_ps(ud * y[t][i].d), _mm256_cvtepi32_ps(ui[t]), uf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) { gres[t] = hsum8(gf[t]); ures[t] = hsum8(uf[t]); }
}

'''
a=source.index('template <int TY, int NT>\nvoid gu_rows(')
changed=source[:a]+pair+source[a:]
old='''        row_dot_any<TY, NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot_any<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);'''
new='''        if constexpr ((TY == 21 || TY == 22) && NT <= 2) {
            row_dot_pair<TY, NT>(blob + (size_t) r * gu_row, blob + up_off + (size_t) r * gu_row, nb, y, g, u);
        } else {
            row_dot_any<TY, NT>(blob + (size_t) r * gu_row, nb, y, g);
            row_dot_any<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        }'''
assert changed.count(old)==1;changed=changed.replace(old,new)
for name,s in [('original',source),('paired',changed)]:
 s=s.replace('namespace strata::kernels::cpu {','namespace strata::kernels::cpu_'+name+' {')
 s+='\nnamespace strata::kernels::cpu_'+name+' {\nvoid sign_bytes(uint32_t m, uint8_t* out) { _mm256_storeu_si256((__m256i*)out, sgn_vec(m)); }\n}\n'
 (out/(name+'.cpp')).write_text(s)
(out/'candidate.cpp').write_text(changed)
(out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),candidate_sha256=hashlib.sha256(changed.encode()).hexdigest()),indent=2)+'\n')
# Reuse the completed equality/streaming harness. Time the two affected formats.
s=Path('/home/yayoi/.local/state/strata-sycl/cpu-sign-probe/probe.cpp').read_text()
s=s.replace('cpu_wide','cpu_paired').replace('wide','paired')
s=s.replace('check(argc==2,"provide GGUF shard")','check(argc>=2,"provide GGUF shard"); const int benchtype=argc>2?std::atoi(argv[2]):21')
s=s.replace('streaming.empty()&&g->type==21','streaming.empty()&&g->type==(uint32_t)benchtype').replace('"no IQ3_S dataset"','"no matching benchmark dataset"')
s=s.replace('ggml_get_type_traits_cpu(GGML_TYPE_IQ3_S)','ggml_get_type_traits_cpu((ggml_type)benchtype)').replace('fn(21,streaming','fn(benchtype,streaming')
s=s.replace('\\\"nt\\\":%d,\\\"experts','\\\"type\\\":%d,\\\"nt\\\":%d,\\\"experts').replace('nt,count,repeats,round,round%2','benchtype,nt,count,repeats,round,round%2')
(out/'probe.cpp').write_text(s)
