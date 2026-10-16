#pragma once
#include <string>
#include <utility>
#include <cstddef>
#include <cstdint>

namespace strata::core::detail {
// The backend owns all buffers and runtime handles. After prepare(), every
// retirement/detach/restore/resume operation MUST be noexcept and fail-stop on
// unknown completion. The only operation allowed to throw is the prompt itself.
// This boundary is shared by serve and standalone and host fake fixtures.
template<class Backend,class Prompt>
bool with_prefill_head_release(Backend& b,Prompt&& prompt,std::string& err) {
    static_assert(noexcept(b.retire_readers()) && noexcept(b.detach()) &&
                  noexcept(b.restore()) && noexcept(b.resume()),
                  "head transition operations must never unwind runtime owners");
    if(!b.prepare(err))return false;
    b.retire_readers();
    b.detach();
    bool ok=false;
    try { ok=std::forward<Prompt>(prompt)(); }
    catch (...) { b.restore();b.resume();throw; }
    b.restore();
    b.resume();
    return ok; // preserves original prompt failure/cancel; restore is mandatory
}
inline bool valid_prefill_head_ids(const int32_t* ids,size_t count,int64_t full) noexcept {
    if(full<=0 || full>INT32_MAX || count>uint64_t(full) || (count && !ids))return false;
    for(size_t i=0;i<count;++i)if(ids[i]<0 || ids[i]>=full)return false;
    return true;
}
// Checked state is independent of a temporarily null allocation pointer.
class PrefillHeadState {
public:
    enum class Phase { ready,prepared,retired,detached,restored };
private:
    Phase phase_=Phase::ready;
public:
    bool transition(Phase expected,Phase next) noexcept {
        const bool edge=(expected==Phase::ready && next==Phase::prepared) ||
            (expected==Phase::prepared && next==Phase::retired) ||
            (expected==Phase::retired && next==Phase::detached) ||
            (expected==Phase::detached && next==Phase::restored) ||
            (expected==Phase::restored && next==Phase::ready);
        if(phase_!=expected || !edge)return false;
        phase_=next;return true;
    }
    Phase phase() const noexcept{return phase_;}
    bool readable() const noexcept{return phase_==Phase::ready;}
};
} // namespace strata::core::detail
