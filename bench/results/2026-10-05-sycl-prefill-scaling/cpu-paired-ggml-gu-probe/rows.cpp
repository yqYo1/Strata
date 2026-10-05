#include "strata/kernels/cpu/native_expert.hpp"
#include <cmath>
#define DECL(PREFIX, NAME) extern "C" void PREFIX##_paired_##NAME(int,float*,float*,const void*,const void*,const void*);
DECL(icx,iq3_xxs) DECL(icx,iq3_s) DECL(icx,iq2_s)
DECL(gcc,iq3_xxs) DECL(gcc,iq3_s) DECL(gcc,iq2_s)
using Pair=void(*)(int,float*,float*,const void*,const void*,const void*);
using Fmt=strata::kernels::cpu::NativeFmt;
static Pair select(int type,bool gcc) {
    if(type==18)return gcc?gcc_paired_iq3_xxs:icx_paired_iq3_xxs;
    if(type==21)return gcc?gcc_paired_iq3_s:icx_paired_iq3_s;
    return gcc?gcc_paired_iq2_s:icx_paired_iq2_s;
}
static void rows(const Fmt& f,const uint8_t* blob,const void* act,float* out,int r0,int r1,bool gcc) {
    Pair pair=select(f.gu_type,gcc);
    for(int r=r0;r<r1;++r) {
        float g=0.f,u=0.f;
        pair((int)f.n_embd,&g,&u,blob+r*f.gu_row,blob+f.up_off+r*f.gu_row,act);
        out[r]=(g/(1.f+std::exp(-g)))*u;
    }
}
void paired_icx_rows(const Fmt& f,const uint8_t* blob,const void* act,float* out,int r0,int r1) {
    rows(f,blob,act,out,r0,r1,false);
}
void paired_gcc_rows(const Fmt& f,const uint8_t* blob,const void* act,float* out,int r0,int r1) {
    rows(f,blob,act,out,r0,r1,true);
}
