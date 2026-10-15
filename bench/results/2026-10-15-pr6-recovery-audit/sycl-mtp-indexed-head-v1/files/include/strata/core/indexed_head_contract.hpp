#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::core::detail {
inline size_t indexed_head_product(uint64_t a,uint64_t b) {
    const uint64_t limit=uint64_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(b && a>limit/b)throw std::invalid_argument("indexed head byte overflow");
    return size_t(a*b);
}
struct IndexedHeadShape {
    size_t weights, ids, activation, output, row;
    IndexedHeadShape(int64_t ni,int64_t full,int64_t subset,int64_t cols) {
        if(ni<=0 || ni>INT32_MAX || ni%256 || full<=0 || full>INT32_MAX ||
           subset<=0 || subset>INT32_MAX || cols<1 || cols>8)
            throw std::invalid_argument("invalid Q6_K indexed head geometry");
        row=indexed_head_product(uint64_t(ni)/256,210);
        weights=indexed_head_product(full,row);ids=indexed_head_product(subset,4);
        activation=indexed_head_product(indexed_head_product(cols,uint64_t(ni)/32),36);
        output=indexed_head_product(indexed_head_product(cols,subset),4);
    }
};
inline bool indexed_head_eligible(bool requested,int type,int64_t ni,int64_t full,size_t row) noexcept {
    return requested && type==14 && ni>0 && ni<=INT32_MAX && ni%256==0 &&
           full>0 && full<=INT32_MAX && row==uint64_t(ni)/256*210;
}
inline void validate_indexed_head_ids(const int32_t* ids,size_t count,int64_t full) {
    if(!ids || !count || count>INT32_MAX || full<=0 || full>INT32_MAX)
        throw std::invalid_argument("invalid indexed head host IDs geometry");
    for(size_t i=0;i<count;++i)if(ids[i]<0 || int64_t(ids[i])>=full)
        throw std::invalid_argument("indexed head token ID outside full vocabulary");
}
} // namespace strata::core::detail
