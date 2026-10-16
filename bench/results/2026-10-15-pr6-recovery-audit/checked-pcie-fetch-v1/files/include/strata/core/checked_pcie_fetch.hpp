#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace strata::core::detail {
// Synchronous issuer-owned state. Device/host completion callbacks never access
// this object. No CUDA/SYCL dependency, global TLS, allocation or ownership release.
struct CheckedPcieFetch {
    static constexpr size_t staging_slots=16;
    struct Error {
        int code=0, index=-1;
        const char* operation="none";
        bool detail_truncated=false;
        std::array<char,256> message{};
    } first, drain_error;
    bool drained=true, called=false;
    size_t submitted=0;
    bool failed() const noexcept {return first.code!=0;}
    void remember(int code,const char* op,int index,const char* text) noexcept {
        if(!code || failed())return;
        first.code=code;first.operation=op;first.index=index;
        const int length=std::snprintf(first.message.data(),first.message.size(),"%s",text?text:"unknown error");
        first.detail_truncated=length<0 || size_t(length)>=first.message.size();
    }
    bool reset() noexcept {
        if(!drained)return false; // Never discard unknown-completion state.
        first=Error{};drain_error=Error{};submitted=0;called=false;return true;
    }
    template<class Copy,class Notify,class Empty,class Describe>
    void submit(const void* allocation,size_t allocation_bytes,void* stage,
                int64_t slots,size_t max_blob,const uint8_t* const* src,int n,
                size_t bytes,int invalid,Copy copy,Notify notify,Empty empty,Describe describe) {
        if(failed())return;
        if(called) {remember(invalid,"duplicate_fetch",-1,"more than one fetch batch in a layer step");return;}
        called=true;
        if(n<=0) {empty();return;} // Original zero/negative-count empty contract.
        const uintptr_t base=reinterpret_cast<uintptr_t>(allocation);
        const uintptr_t dest=reinterpret_cast<uintptr_t>(stage);
        const size_t maximum=std::numeric_limits<size_t>::max();
        if(!allocation || !stage || !src || slots<=0 || size_t(slots)>staging_slots || n>slots ||
           !bytes || !max_blob || bytes>max_blob || size_t(n)>maximum/bytes ||
           base>UINTPTR_MAX-allocation_bytes || dest<base || dest>base+allocation_bytes ||
           size_t(n)*bytes>base+allocation_bytes-dest || max_blob>maximum/staging_slots ||
           allocation_bytes!=staging_slots*max_blob || (dest-base)%max_blob ||
           size_t(slots)>maximum/max_blob || size_t(slots)*max_blob>base+allocation_bytes-dest) {
            remember(invalid,"geometry",-1,"invalid PCIe staging/source/count/blob span");return;
        }
        // Validate the entire pointer batch before its first enqueue. Source
        // allocation identity/accessible byte extent remains the API's check.
        for(int i=0;i<n;++i)if(!src[i] || reinterpret_cast<uintptr_t>(src[i])>UINTPTR_MAX-bytes) {
            remember(invalid,"source_geometry",i,"invalid PCIe source pointer/span");return;
        }
        for(int i=0;i<n;++i) {
            drained=false; // Even a failed enqueue may have unknown side effects.
            const int code=copy(static_cast<uint8_t*>(stage)+size_t(i)*bytes,src[i],bytes);
            if(code) {remember(code,"copy_api",i,describe(code));return;}
            ++submitted;
        }
        drained=false;
        const int code=notify();
        if(code)remember(code,"callback_api",n,describe(code));
    }
    template<class Drain,class Describe> bool finish(Drain drain,Describe describe) {
        // Caller uses this in place of its existing checked copy-stream sync;
        // submission failure never bypasses draining the successfully queued prefix.
        const int code=drain();
        if(code) {
            drained=false;
            if(!drain_error.code) {
                drain_error.code=code;drain_error.operation="copy_stream_drain";
                const int length=std::snprintf(drain_error.message.data(),drain_error.message.size(),"%s",describe(code));
                drain_error.detail_truncated=length<0 || size_t(length)>=drain_error.message.size();
            }
            remember(code,"copy_stream_drain",-1,drain_error.message.data());
        } else drained=true;
        return drained && !failed();
    }
    bool can_consume() const noexcept {return drained && !failed();}
};
} // namespace strata::core::detail
