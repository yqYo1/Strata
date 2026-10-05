#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
extern "C" const ggml_type_traits_cpu* __real_ggml_get_type_traits_cpu(ggml_type);
extern "C" void alternate_ggml_vec_dot_iq3_xxs_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    const bool alternate=std::atoi(argv[1])!=0;
    ggml_cpu_init();
    int changed=0;
    for(int t=0;t<GGML_TYPE_COUNT;++t) {
        auto type=(ggml_type)t;
        const auto* original=__real_ggml_get_type_traits_cpu(type);
        const auto* selected=ggml_get_type_traits_cpu(type);
        if(alternate && type==GGML_TYPE_IQ3_XXS) {
            if(selected->vec_dot!=alternate_ggml_vec_dot_iq3_xxs_q8_K ||
               selected->from_float!=original->from_float ||
               selected->vec_dot_type!=original->vec_dot_type ||
               selected->nrows!=original->nrows)return 1;
            ++changed;
        } else if(selected!=original)return 1;
    }
    std::printf("{\"all_traits_checked\":%d,\"selected_types\":%d,\"only_iq3_xxs_changed\":true}\n",GGML_TYPE_COUNT,changed);
}
