from pathlib import Path
import hashlib,json
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05');out=Path(__file__).resolve().parent
source=(root/'src/kernels/cpu/iq_avx2.cpp').read_text()
a=source.index('template <int TY, int NT>\ninline void row_dot(')
b=source.index('// ---- IQ2_XS',a)
split=source[a:b].replace('inline void row_dot(', 'inline void row_dot_split(')
split=split.replace('''        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();''','''        __m256i acci[NT][2];
        for (int t = 0; t < NT; ++t) acci[t][0] = acci[t][1] = _mm256_setzero_si256();''')
split=split.replace('acci[t] = _mm256_add_epi32(acci[t],','acci[t][half] = _mm256_add_epi32(acci[t][half],')
split=split.replace('_mm256_cvtepi32_ps(acci[t])','_mm256_cvtepi32_ps(_mm256_add_epi32(acci[t][0], acci[t][1]))')
assert split!=source[a:b] and 'acci[t][half]' in split
changed=source[:b]+'''// Independent integer half-block sums, as in the existing IQ2_XS path.
// Modular addition changes no result; the per-block FP32 FMA order is unchanged.
'''+split+source[b:]
old='''    if constexpr (TY == 17) row_dot_iq2xs<NT>(row, nblocks, y, res);
    else                    row_dot<TY, NT>(row, nblocks, y, res);'''
new='''    if constexpr (TY == 17) row_dot_iq2xs<NT>(row, nblocks, y, res);
    else if constexpr ((TY == 21 || TY == 22) && NT <= 3) row_dot_split<TY, NT>(row, nblocks, y, res);
    else                    row_dot<TY, NT>(row, nblocks, y, res);'''
assert changed.count(old)==1;changed=changed.replace(old,new)
for name,s in [('original',source),('split',changed)]:
 s=s.replace('namespace strata::kernels::cpu {','namespace strata::kernels::cpu_'+name+' {')
 s+='\nnamespace strata::kernels::cpu_'+name+' {\nvoid sign_bytes(uint32_t m, uint8_t* out) { _mm256_storeu_si256((__m256i*)out, sgn_vec(m)); }\n}\n'
 (out/(name+'.cpp')).write_text(s)
(out/'candidate.cpp').write_text(changed)
(out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),candidate_sha256=hashlib.sha256(changed.encode()).hexdigest()),indent=2)+'\n')
s=Path('/home/yayoi/.local/state/strata-sycl/cpu-paired-gu-probe/probe.cpp').read_text().replace('cpu_paired','cpu_split').replace('paired','split')
(out/'probe.cpp').write_text(s)
s=Path('/home/yayoi/.local/state/strata-sycl/cpu-paired-gu-probe/build.sh').read_text().replace('cpu-paired-gu-probe','cpu-split-acc-probe').replace('paired.cpp','split.cpp').replace('paired.o','split.o')
(out/'build.sh').write_text(s)
