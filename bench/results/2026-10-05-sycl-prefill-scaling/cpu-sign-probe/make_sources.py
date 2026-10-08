from pathlib import Path
import hashlib,json
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05');out=Path(__file__).resolve().parent
source=(root/'src/kernels/cpu/iq_avx2.cpp').read_text()
a=source.index('    const __m128i bm =',source.index('inline __m256i sgn_vec'))
b=source.index('    const __m256i sel =',a)
wide='''    const __m256i bm = _mm256_set1_epi32((int) m);
    const __m256i bits = _mm256_shuffle_epi8(bm, _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
'''
changed=source[:a]+wide+source[b:]
for name,s in [('original',source),('wide',changed)]:
 s=s.replace('namespace strata::kernels::cpu {','namespace strata::kernels::cpu_'+name+' {')
 s+='\nnamespace strata::kernels::cpu_'+name+' {\nvoid sign_bytes(uint32_t m, uint8_t* out) { _mm256_storeu_si256((__m256i*)out, sgn_vec(m)); }\n}\n'
 (out/(name+'.cpp')).write_text(s)
(out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),candidate_sha256=hashlib.sha256(changed.encode()).hexdigest()),indent=2)+'\n')
(out/'candidate.cpp').write_text(changed)
