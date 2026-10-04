from pathlib import Path
s=Path('src/kernels/cpu/iq_single_avx2.cpp').read_text()
a=s.index('__attribute__((noinline)) void iq3s_dot(');b=s.index('\n} // namespace',a)
part=s[a:b]
part=part.replace('__attribute__((noinline)) void iq3s_dot(', 'template<int FixedN>\n__attribute__((always_inline)) inline void iq3s_dot_impl(')
part=part.replace('    assert(n % QK_K == 0);','    if constexpr (FixedN) n = FixedN;\n    assert(n % QK_K == 0);',1)
wrapper='''
__attribute__((noinline)) void iq3s_dot(int n, float* s, size_t bs,
    const void* vx, size_t bx, const void* vy, size_t by, int nrc) {
    iq3s_dot_impl<0>(n,s,bs,vx,bx,vy,by,nrc);
}

void iq3s_single_rows_2560(const uint8_t* blob, size_t row, size_t up_off,
                          const void* act, float* ff, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const auto* gate = blob + size_t(r) * row;
        float g = 0.f, u = 0.f;
        iq3s_dot_impl<2560>(2560, &g, 0, gate, 0, act, 0, 1);
        iq3s_dot_impl<2560>(2560, &u, 0, gate + up_off, 0, act, 0, 1);
        ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
}
'''
s=s[:a]+part+wrapper+s[b:]
s=s.replace('    if (type != 18 && type != 21 && type != 22) return false;', '''    if (type != 18 && type != 21 && type != 22) return false;
    if (type == 21 && n == 2560) {
        iq3s_single_rows_2560(blob,row,up_off,act,ff,r0,r1);
        return true;
    }''')
Path('/tmp/strata-sycl-goal-single-fixed-production.cpp').write_text(s)
Path('/tmp/strata-sycl-goal-single-dispatch-v4.cpp').write_text(s.replace('iq256_single_gu_rows(', 'iq256_single_gu_rows_v4('))
p=Path('/tmp/strata-sycl-goal-single-dispatch-bench.cpp').read_text().replace('bool iq256_single_gu_rows_v3(', 'bool iq256_single_gu_rows_v4(int,const uint8_t*,size_t,size_t,int,const void*,float*,int,int);\nbool iq256_single_gu_rows_v3(')
p=p.replace('iq256_single_gu_rows_v3};','iq256_single_gu_rows_v3,iq256_single_gu_rows_v4};').replace('vi=1;vi<=3','vi=4;vi<=4').replace('if(vi==1)','if(vi==4)')
p=p.replace('trial<5','trial<10').replace('it<3','it<5').replace('.count()/3','.count()/5')
Path('/tmp/strata-sycl-goal-single-fixed-bench.cpp').write_text(p)
