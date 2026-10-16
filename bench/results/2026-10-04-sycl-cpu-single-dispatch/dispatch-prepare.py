from pathlib import Path
s=Path('src/kernels/cpu/iq_single_avx2.cpp').read_text()
start=s.index('bool iq256_single_gu_rows(');stop=s.index('\n} // namespace strata::kernels::cpu',start)
body='''template<auto Dot, int FixedN = 0>
static void single_rows(const uint8_t* blob, size_t row, size_t up_off,
                        int n, const void* act, float* ff, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const auto* gate = blob + size_t(r) * row;
        float g = 0.f, u = 0.f;
        Dot(FixedN ? FixedN : n, &g, 0, gate, 0, act, 0, 1);
        Dot(FixedN ? FixedN : n, &u, 0, gate + up_off, 0, act, 0, 1);
        ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
}
bool iq256_single_gu_rows(int type, const uint8_t* blob, size_t row, size_t up_off,
                          int n, const void* act, float* ff, int r0, int r1) {
    if (type != 18 && type != 21 && type != 22) return false;
    FIXED_DISPATCH
    switch (type) {
    case 18: single_rows<iq3xxs_dot>(blob,row,up_off,n,act,ff,r0,r1); break;
    case 21: single_rows<iq3s_dot>(blob,row,up_off,n,act,ff,r0,r1); break;
    case 22: single_rows<iq2s_dot>(blob,row,up_off,n,act,ff,r0,r1); break;
    }
    return true;
}'''
fixed='''if(n==2560) {
      switch(type) {
      case 18: single_rows<iq3xxs_dot,2560>(blob,row,up_off,n,act,ff,r0,r1); break;
      case 21: single_rows<iq3s_dot,2560>(blob,row,up_off,n,act,ff,r0,r1); break;
      case 22: single_rows<iq2s_dot,2560>(blob,row,up_off,n,act,ff,r0,r1); break;
      }
      return true;
    }'''
for v in [1,2,3]:
 new=s[:start]+body.replace('FIXED_DISPATCH',fixed if v==3 else '')+s[stop:]
 if v>1:new=new.replace('__attribute__((noinline)) void','__attribute__((always_inline)) inline void')
 new=new.replace('iq256_single_gu_rows(',f'iq256_single_gu_rows_v{v}(')
 Path(f'/tmp/strata-sycl-goal-single-dispatch-v{v}.cpp').write_text(new)
