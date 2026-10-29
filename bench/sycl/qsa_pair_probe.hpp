// Main-authored component fixture interface; repository LICENSE applies.
#pragma once
#include "strata/kernels/qsa_pair_attention.hpp"
#include <sycl/sycl.hpp>
#include <vector>
namespace pair_probe {
// Private generated wrappers record the actual returned production events.
// Reserve before a timed batch; no tensor copies/queries/waits in capture.
extern std::vector<sycl::event> events;
bool selected(const uint16_t *q,const strata::kernels::QsaAttnPools &pools,
    const int32_t *ids,const int32_t *steps,int64_t cap,
    const strata::kernels::QsaShapes &shape,uint16_t *out,int64_t queries,
    const strata::kernels::QsaPromptBf16Bounds &bounds,int64_t live,
    const strata::kernels::QsaBf16PackedScratch &scratch,bool pack_only,void *stream);
}
