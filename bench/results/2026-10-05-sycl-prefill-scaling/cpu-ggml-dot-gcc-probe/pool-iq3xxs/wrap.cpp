#include "ggml-cpu.h"
extern "C" const ggml_type_traits_cpu* __real_ggml_get_type_traits_cpu(ggml_type);
extern "C" void alternate_ggml_vec_dot_iq3_xxs_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern "C" const ggml_type_traits_cpu* __wrap_ggml_get_type_traits_cpu(ggml_type type) {
    const auto* original=__real_ggml_get_type_traits_cpu(type);
#if defined(SELECT_IQ3XXS)
    if(type==GGML_TYPE_IQ3_XXS) {
        static const ggml_type_traits_cpu selected=[&] {
            auto copy=*original;
            copy.vec_dot=alternate_ggml_vec_dot_iq3_xxs_q8_K;
            return copy;
        }();
        return &selected;
    }
#endif
    return original;
}
