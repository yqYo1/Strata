#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::kernels::detail {

inline bool qsa_indexer_device_batch_requested(const char* value) {
    if (!value || std::strcmp(value,"0")==0) return false;
    if (std::strcmp(value,"1")==0) return true;
    throw std::invalid_argument("STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH must be 0 or 1");
}

inline size_t qsa_indexer_batch_bytes(uint64_t count,uint64_t stride) {
    if ((stride && count>uint64_t(SIZE_MAX)/stride) ||
        (stride && count>uint64_t(PTRDIFF_MAX)/stride))
        throw std::invalid_argument("native QSA device batch extent overflow");
    return size_t(count*stride);
}

// Only fixed 128-component/four-cell-block/64-RoPE geometry is admitted by
// the production adapter. This pure plan covers all host launch arithmetic.
struct QsaIndexerDeviceBatchPlan {
    int64_t n=0,p0=0,end=0,first_block=0,completed_blocks=0;
    size_t raw_bytes=0,pooled_bytes=0,block_global=0;
    bool seed_zero=false;

    static QsaIndexerDeviceBatchPlan make(int64_t n,int64_t p0,int32_t base,int64_t capacity) {
        if (capacity<=0 || capacity>INT32_MAX || base<0 || base%4 ||
            int64_t(base)+capacity>INT32_MAX || n<0 || p0<0 || p0>capacity || n>capacity-p0)
            throw std::invalid_argument("invalid native QSA device batch range/capacity/base");
        QsaIndexerDeviceBatchPlan p;
        p.n=n; p.p0=p0; p.end=p0+n; p.first_block=p0/4;
        p.completed_blocks=p.end/4>p.first_block?p.end/4-p.first_block:0;
        p.seed_zero=n>0 && p0==0;
        p.raw_bytes=qsa_indexer_batch_bytes(uint64_t(n),128*4);
        p.pooled_bytes=qsa_indexer_batch_bytes(uint64_t(capacity/4+1),128*4);
        p.block_global=qsa_indexer_batch_bytes(uint64_t(p.completed_blocks),256);
        return p;
    }
};

} // namespace strata::kernels::detail
