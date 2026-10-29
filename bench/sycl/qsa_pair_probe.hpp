// Main-authored component fixture interface; repository LICENSE applies.
#pragma once
#include "strata/kernels/qsa_pair_attention.hpp"
namespace pair_probe {
// The exact selected6e40 comparison template is generated from Git.
bool selected(const uint16_t *q,const strata::kernels::QsaAttnPools &pools,
    const int32_t *ids,const int32_t *steps,int64_t cap,
    const strata::kernels::QsaShapes &shape,uint16_t *out,int64_t queries,
    const strata::kernels::QsaPromptBf16Bounds &bounds,int64_t live,
    const strata::kernels::QsaBf16PackedScratch &scratch,void *stream);
}
