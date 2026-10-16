from pathlib import Path
import difflib,hashlib,json
R=Path.cwd();D=R/'bench/results/2026-10-15-pr6-recovery-audit/prefill-head-release-v1';base={};new={}
def get(n):
 s=(R/n).read_text();base[n]=s;return s
def put(n,s):new[n]=s
helper=r'''#pragma once
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
'''
n='include/strata/core/prefill_head_scope.hpp';base[n]='';put(n,helper)
n='include/strata/core/native_head.hpp';s=get(n);s=s.replace('    bool run(const float* mixed,', '''    // Narrow SYCL prompt boundary. Caller first preflights all consumers, then
    // retires their graphs/streams before detach. Snapshot budget: 1 GiB packed
    // weights, no files; scratch and tensor metadata remain owned and unchanged.
    bool prepare_prefill_weights(std::string& err);
    void cancel_prefill_weights() noexcept;
    void detach_prefill_weights() noexcept;
    void restore_prefill_weights() noexcept;
    void resume_prefill_weights() noexcept;
    bool head_detached() const { return head_detached_; }
    bool run(const float* mixed,''',1)
s=s.replace('    void* weights_ = nullptr;','''    bool head_prepared_=false,head_detached_=false,head_release_used_=false;
    std::vector<uint8_t> head_snapshot_; // exact packed bytes, request-scoped RAM
    void* weights_ = nullptr;''',1);put(n,s)
n='src/core/native_head.cpp';s=get(n);s=s.replace('#include <exception>','#include <exception>\n#include <cstdlib>\n#include <limits>\n#ifdef STRATA_ENABLE_SYCL\n#include "strata/sycl/checked_cleanup.hpp"\n#endif')
s=s.replace('NativeHead::~NativeHead() {','''NativeHead::~NativeHead() {
#ifdef STRATA_ENABLE_SYCL
    if(head_prepared_ || head_detached_)
        strata::sycl_backend::cleanup_fail_stop("native head teardown while prompt head scope active");
    if(head_release_used_) {
        // The caller's reader lease has already drained main/verifier/MTP queues.
        if(scratch_ && cudaFree(scratch_)!=cudaSuccess)strata::sycl_backend::cleanup_fail_stop("native scratch free");
        if(weights_ && cudaFree(weights_)!=cudaSuccess)strata::sycl_backend::cleanup_fail_stop("native restored head free");
        return;
    }
#endif''',1)
s=s.replace('    if (loaded()) {','    if (head_prepared_ || head_detached_) {err="native head load forbidden during head release";return false;}\n    if (loaded()) {',1)
s=s.replace('    if (!loaded() || !mixed || !logits || !stream) {','    if (head_detached_) {err="native head is detached for prompt";return false;}\n    if (!loaded() || !mixed || !logits || !stream) {',1)
anchor='// ================================ plan v0.3 P6:'
methods=r'''bool NativeHead::prepare_prefill_weights(std::string& err) try {
#ifndef STRATA_ENABLE_SYCL
    err="head release requires SYCL";return false;
#else
    if(head_prepared_ || head_detached_ || !loaded() || !bytes_ || bytes_>(1ull<<30) ||
       bytes_>uint64_t(std::numeric_limits<size_t>::max()) ||
       bytes_!=strata::kernels::native_mmvq_weight_bytes(type_,n_in_,n_out_)) {
        err="native head release requires ready exact packed weights within 1 GiB snapshot budget";return false;
    }
    try {head_snapshot_.resize(size_t(bytes_));}
    catch(const std::exception& e){err=std::string("native head RAM snapshot: ")+e.what();return false;}
    head_prepared_=true;return true;
#endif
} catch(const std::exception& e) {err=std::string("native head prepare: ")+e.what();return false;}
void NativeHead::cancel_prefill_weights() noexcept {
#ifdef STRATA_ENABLE_SYCL
    if(!head_prepared_ || head_detached_)strata::sycl_backend::cleanup_fail_stop("invalid native head prepare cancellation");
    head_prepared_=false;std::vector<uint8_t>().swap(head_snapshot_);
#endif
}
void NativeHead::detach_prefill_weights() noexcept {
#ifdef STRATA_ENABLE_SYCL
    using strata::sycl_backend::cleanup_fail_stop;
    if(!head_prepared_ || head_detached_ || !weights_ || head_snapshot_.size()!=bytes_)cleanup_fail_stop("native head detach state");
    // Every head reader was retired by the common request boundary before here.
    if(cudaMemcpy(head_snapshot_.data(),weights_,size_t(bytes_),cudaMemcpyDeviceToHost)!=cudaSuccess)
        cleanup_fail_stop("native head D2H snapshot completion unknown");
    head_detached_=true;head_release_used_=true;
    if(cudaFree(weights_)!=cudaSuccess)cleanup_fail_stop("native head detach free");
    weights_=nullptr;
    std::fprintf(stderr,"strata prefill heads: target detached bytes=%llu source=owned-RAM D2H=%llu H2D-on-restore=%llu\n",
        (unsigned long long)bytes_,(unsigned long long)bytes_,(unsigned long long)bytes_);
#endif
}
void NativeHead::restore_prefill_weights() noexcept {
#ifdef STRATA_ENABLE_SYCL
    using strata::sycl_backend::cleanup_fail_stop;
    if(!head_prepared_ || !head_detached_ || weights_ || head_snapshot_.size()!=bytes_)cleanup_fail_stop("native head restore state");
    if(cudaMalloc(&weights_,size_t(bytes_))!=cudaSuccess)cleanup_fail_stop("native head restore allocation");
    if(cudaMemcpy(weights_,head_snapshot_.data(),size_t(bytes_),cudaMemcpyHostToDevice)!=cudaSuccess)
        cleanup_fail_stop("native head restore H2D completion unknown");
    // Full-forward remains blocked until both target and draft owners are ready.
#endif
}
void NativeHead::resume_prefill_weights() noexcept {
#ifdef STRATA_ENABLE_SYCL
    if(!head_prepared_ || !head_detached_ || !weights_)
        strata::sycl_backend::cleanup_fail_stop("native head resume state");
    head_detached_=false;head_prepared_=false;
    std::vector<uint8_t>().swap(head_snapshot_);
#endif
}

'''
s=s.replace(anchor,methods+anchor,1);put(n,s)
n='include/strata/core/mtp.hpp';s=get(n);s=s.replace('    /// Prompt-only expert retirement.', '''    // Head-only request boundary: caller serializes all public calls and keeps
    // NativeHead alive. IDs and their device allocation never change; KV-only
    // prompt graphs, arenas, logits and sampling state remain live.
    bool prepare_prefill_head(const NativeHead& head,std::string& err);
    void cancel_prefill_head() noexcept;
    void retire_prefill_head_readers() noexcept;
    void detach_prefill_head() noexcept;
    void restore_prefill_head() noexcept;
    void resume_prefill_head() noexcept;
    // Invoke before borrowed NativeHead destruction after this opt-in was used.
    void retire_head_owner_readers(cudaStream_t owner_stream) noexcept;

    /// Prompt-only expert retirement.''',1)
s=s.replace('    bool upload_experts_checked(','''    bool head_prepared_=false,head_detached_=false,head_restored_=false;
    bool head_release_used_=false,head_owner_retired_=false,head_had_subset_=false,head_storage_released_=false;
    uint64_t head_release_bytes_=0,head_full_bytes_=0;
    size_t head_row_bytes_=0;
    int head_type_=-1;
    std::vector<int32_t> head_ids_,head_map_check_; // immutable map + bounded readback scratch
    void retire_full_head_graphs() noexcept;
    bool upload_experts_checked(''',1);put(n,s)
n='src/core/mtp.cpp';s=get(n);s=s.replace('#include "strata/core/mtp.hpp"','#include "strata/core/mtp.hpp"\n#include "strata/core/prefill_head_scope.hpp"\n#include <stdexcept>',1);s=s.replace('MtpDrafter::~MtpDrafter() {','''MtpDrafter::~MtpDrafter() {
    if(head_prepared_ || head_detached_)expert_failstop("MTP teardown during prompt head release",cudaErrorUnknown);
    if(head_release_used_)retire_full_head_graphs();''',1)
for field in ('prefill_exec_','prefill_dev_exec_'):
 s=s.replace(f'for (auto& e : {field}) if (e) cudaGraphExecDestroy(e);',f'for (auto& e : {field}) if (e) {{const auto status=cudaGraphExecDestroy(e);if(head_release_used_ && status!=cudaSuccess)expert_failstop("head scope KV graph retirement",status);}}',1)
for field in ('head_logits_','dhead_','dvocab_'):
 s=s.replace(f'if ({field}) cudaFree({field});',f'if ({field}) {{const auto status=cudaFree({field});if(head_release_used_ && status!=cudaSuccess)expert_failstop("restored head allocation teardown",status);}}',1)
s=s.replace('if (cs_) cudaStreamDestroy(cs_);','if (cs_) {const auto status=cudaStreamDestroy(cs_);if(head_release_used_ && status!=cudaSuccess)expert_failstop("restored head stream destroy",status);}',1)
s=s.replace('if (expert_source_valid_ && status != cudaSuccess)','if ((expert_source_valid_ || head_release_used_) && status != cudaSuccess)',1)
# Decode entry points and every null-dhead sentinel reader are guarded.
s=s.replace('    wt_ = &wt;\n    head_ = head;\n    window_R_ = window_R;','''    if(head_release_used_) {
        if(&wt!=wt_ || head!=head_ || !head || !head->loaded() || head->head_detached() ||
           head->weight_bytes()!=head_full_bytes_ || head->row_bytes()!=head_row_bytes_ || head->type()!=head_type_) {
            err="mtp: restored head rebind changed immutable owner/format";return false;
        }
        bool readers=false;
        for(auto e:round_exec_)readers=readers || e;
        for(auto e:step_exec_)readers=readers || e;
        for(auto e:round_exec_c_)readers=readers || e;
        for(auto e:step_exec_c_)readers=readers || e;
        if(readers && window_R!=window_R_){err="mtp: restored head rebind would retarget captured window";return false;}
        window_R_=window_R;
        return true; // map and no-subset decision are immutable after first release
    }
    wt_ = &wt;
    head_ = head;
    window_R_ = window_R;''',1)

s=s.replace('uint64_t MtpDrafter::bind_bytes(uint64_t head_row_bytes, int64_t n_vocab) const {','''uint64_t MtpDrafter::bind_bytes(uint64_t head_row_bytes, int64_t n_vocab) const {
    if(head_detached_ || head_owner_retired_)throw std::logic_error("MTP head bytes requested while detached/retired");''',1)
s=s.replace('bool MtpDrafter::setup_coupled(std::string& err) {','''bool MtpDrafter::setup_coupled(std::string& err) {
    if(head_detached_ || head_owner_retired_){err="mtp: coupled setup while head detached/retired";return false;}''',1)
s=s.replace('bool MtpDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err) {','''bool MtpDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err) {
    if(head_prepared_ || head_detached_ || head_owner_retired_){err="mtp: bind during head release/retirement";return false;}''',1)
s=s.replace('    const bool full = step_row0 >= 0;','''    const bool full = step_row0 >= 0;
    if(full && (head_detached_ || head_owner_retired_ || !head_ || head_->head_detached() || !head_->loaded())) {
        err="mtp: full forward requires restored head";return false;
    }''',1)
for marker in ['bool MtpDrafter::capture_round(int T, bool coupled, std::string& err) try {','bool MtpDrafter::capture_step(int j, bool coupled, std::string& err) try {']:
 s=s.replace(marker,marker+'\n    if(head_detached_ || head_owner_retired_){err="mtp: capture requires restored head";return false;}',1)
for marker in ['    if (!experts_ || experts_detached_) { err = "mtp: draft requires restored experts"; return false; }','    if (!experts_ || experts_detached_) { err = "mtp: first draft requires restored experts"; return false; }']:
 s=s.replace(marker,'    if(head_detached_ || head_owner_retired_){err="mtp: draft requires restored head";return false;}\n'+marker,1)
s=s.replace('const bool captured = expert_recapture_timing_','const bool captured = (expert_recapture_timing_ || head_release_used_)')
s=s.replace('    if (expert_recapture_timing_)\n        expert_failstop(', '    if (expert_recapture_timing_ || head_release_used_)\n        expert_failstop(')
anchor='bool MtpDrafter::detach_prefill_experts('
methods=r'''bool MtpDrafter::prepare_prefill_head(const NativeHead& head,std::string& err) {
#ifndef STRATA_ENABLE_SYCL
    (void)head;err="head release requires SYCL";return false;
#else
    const OnDevice on_device(device_);
    int current=-1;
    if(cudaGetDevice(&current)!=cudaSuccess || current!=device_ || !g_ || max_t_!=4 || !cs_ ||
       head_prepared_ || head_detached_ || head_owner_retired_ || head_!=&head || !head.loaded() || head.head_detached() ||
       !strata::kernels::cpu::expert_layout().native || g_->n_expert!=512 ||
       n_vocab_<=0 || n_vocab_>INT32_MAX || head.type()<0 || head.row_bytes()==0 || peer_portable() ||
       coupled_draft_env() || coupled_ok_ || coupled_active_) {
        err="mtp: head release requires ready noncoupled single-device native MTP4";return false;
    }
    if(uint64_t(n_vocab_)>UINT64_MAX/head.row_bytes() || head.weight_bytes()!=uint64_t(n_vocab_)*head.row_bytes()) {err="mtp: native head full span mismatch";return false;}
    const bool subset=dhead_!=nullptr;
    if((subset && (!dvocab_ || n_dvocab_<=0 || n_dvocab_>n_vocab_)) ||
       (!subset && (dvocab_ || n_dvocab_!=0))) {
        err="mtp: inconsistent or oversized head map before release";return false;
    }
    const uint64_t rows=subset?uint64_t(n_dvocab_):0;
    if(rows>(1ull<<20)){err="mtp: retained head map exceeds 4 MiB per-vector budget";return false;}
    if(rows && head.row_bytes()>UINT64_MAX/rows){err="mtp: draft head size overflow";return false;}
    const uint64_t bytes=rows*head.row_bytes();
    if(bytes>uint64_t(SIZE_MAX) || vram_<bytes){err="mtp: draft head release size/accounting";return false;}
    if(cudaStreamSynchronize(cs_)!=cudaSuccess)expert_failstop("head map prepare did not drain",cudaErrorUnknown);
    try {
        head_map_check_.resize(size_t(rows));
        if(rows && cudaMemcpy(head_map_check_.data(),dvocab_,size_t(rows)*sizeof(int32_t),cudaMemcpyDeviceToHost)!=cudaSuccess)
            expert_failstop("head map readback completion unknown",cudaErrorUnknown);
        if(!detail::valid_prefill_head_ids(head_map_check_.data(),head_map_check_.size(),n_vocab_))expert_failstop("MTP device head map contains invalid IDs",cudaErrorUnknown);
        if(head_release_used_ && (head_had_subset_!=subset || head_ids_!=head_map_check_)) {
            expert_failstop("MTP immutable device head map changed since release",cudaErrorUnknown);
        }
        if(!head_release_used_)head_ids_=head_map_check_;
    }catch(const std::exception& e){err=std::string("mtp: bounded head map RAM: ")+e.what();return false;}
    head_had_subset_=subset;head_release_bytes_=bytes;
    head_full_bytes_=head.weight_bytes();head_row_bytes_=head.row_bytes();head_type_=head.type();
    head_prepared_=true;return true;
#endif
}
void MtpDrafter::cancel_prefill_head() noexcept {
    if(!head_prepared_ || head_detached_)expert_failstop("invalid MTP head prepare cancellation",cudaErrorUnknown);
    head_prepared_=false;
}
void MtpDrafter::retire_full_head_graphs() noexcept {
    if(cs_ && cudaStreamSynchronize(cs_)!=cudaSuccess)expert_failstop("head reader stream drain",cudaErrorUnknown);
    auto retire=[](auto& graphs){for(auto& e:graphs)if(e){
        const auto status=cudaGraphExecDestroy(e);
        if(status!=cudaSuccess)expert_failstop("head reader graph destroy",status);
        e=nullptr;
    }};
    retire(round_exec_);retire(step_exec_);retire(round_exec_c_);retire(step_exec_c_);
}
void MtpDrafter::retire_prefill_head_readers() noexcept {
    if(!head_prepared_ || head_detached_)expert_failstop("MTP head retire state",cudaErrorUnknown);
    head_detached_=true;head_release_used_=true;
    retire_full_head_graphs();
}
void MtpDrafter::detach_prefill_head() noexcept {
    if(!head_prepared_ || !head_detached_ || head_restored_ || head_storage_released_ || (head_had_subset_ && !dhead_))
        expert_failstop("MTP head detach state",cudaErrorUnknown);
    if(dhead_){
        const auto status=cudaFree(dhead_);
        if(status!=cudaSuccess)expert_failstop("MTP duplicate head free",status);
        dhead_=nullptr;vram_-=head_release_bytes_;
    }
    head_storage_released_=true;
    // dvocab/order/count, logits, coupled buffers, prompt graphs and KV stay live.
    std::fprintf(stderr,"strata prefill heads: draft detached bytes=%llu retained-map-bytes=%zu\n",
        (unsigned long long)head_release_bytes_,head_ids_.size()*sizeof(int32_t));
}
void MtpDrafter::restore_prefill_head() noexcept {
    if(!head_prepared_ || !head_detached_ || !head_storage_released_ || head_restored_ || dhead_ || !head_ || !head_->loaded() ||
       head_->weight_bytes()!=head_full_bytes_ || head_->row_bytes()!=head_row_bytes_ || head_->type()!=head_type_)
        expert_failstop("MTP head restore owner/format/state",cudaErrorUnknown);
    if(cudaStreamSynchronize(cs_)!=cudaSuccess)expert_failstop("MTP prompt completion before head restore",cudaErrorUnknown);
    if(head_had_subset_){
        if(head_map_check_.size()!=head_ids_.size() || head_ids_.size()!=size_t(n_dvocab_))
            expert_failstop("MTP retained head map size changed",cudaErrorUnknown);
        if(cudaMemcpy(head_map_check_.data(),dvocab_,head_ids_.size()*sizeof(int32_t),cudaMemcpyDeviceToHost)!=cudaSuccess)
            expert_failstop("MTP retained map restore readback",cudaErrorUnknown);
        if(head_map_check_!=head_ids_)expert_failstop("MTP retained map changed during prompt",cudaErrorUnknown);
        const auto allocated=cudaMalloc(reinterpret_cast<void**>(&dhead_),size_t(head_release_bytes_));
        if(allocated!=cudaSuccess)expert_failstop("MTP duplicate head restore allocation",allocated);
        try {
            strata::kernels::gather_rows(static_cast<const uint8_t*>(head_->weights()),int64_t(head_row_bytes_),
                dvocab_,n_dvocab_,dhead_,cs_);
        }catch(...){expert_failstop("MTP duplicate head restore gather threw",cudaErrorUnknown);}
        const auto launch=cudaPeekAtLastError();
        if(launch!=cudaSuccess)expert_failstop("MTP duplicate head restore gather launch",launch);
        const auto completed=cudaStreamSynchronize(cs_);
        if(completed!=cudaSuccess)expert_failstop("MTP duplicate head restore gather completion",completed);
        vram_+=head_release_bytes_;
    }
    head_restored_=true;
}
void MtpDrafter::resume_prefill_head() noexcept {
    if(!head_prepared_ || !head_detached_ || !head_restored_ || !head_ || head_->head_detached())
        expert_failstop("MTP head resume state",cudaErrorUnknown);
    head_prepared_=false;head_detached_=false;head_restored_=false;head_storage_released_=false;
}
void MtpDrafter::retire_head_owner_readers(cudaStream_t owner_stream) noexcept {
    if(!head_release_used_)return; // exact default-off teardown has no extra API
    if(!owner_stream || cudaStreamSynchronize(owner_stream)!=cudaSuccess)expert_failstop("borrowed target head stream teardown",cudaErrorUnknown);
    if(head_prepared_ || head_detached_)expert_failstop("borrowed head teardown during scope",cudaErrorUnknown);
    retire_full_head_graphs();head_owner_retired_=true;
}

'''
s=s.replace(anchor,methods+anchor,1);put(n,s)
n='include/strata/core/verify.hpp';s=get(n);s=s.replace('    /// One window:', '''    // Head-only retirement. Own the serialized request boundary; this preserves
    // all arenas, commit graph, session/KV/GDN/PLE and sampling state.
    bool can_release_prefill_head(const NativeHead& head,std::string& err) const;
    bool retire_prefill_head(const NativeHead& head,std::string& err);
    bool resume_prefill_head(const NativeHead& head,std::string& err);

    /// One window:''',1)
s=s.replace('    bool head_sampling_ = true;', '''    bool head_detached_=false,head_recapture_checked_=false;
    uint64_t release_head_bytes_=0;
    size_t release_head_row_=0;
    int release_head_type_=-1;
    bool head_sampling_ = true;''',1);put(n,s)
n='src/core/verify.cpp';s=get(n);s=s.replace('#include "strata/core/native_head.hpp"','#include "strata/core/native_head.hpp"\n#ifdef STRATA_ENABLE_SYCL\n#include "strata/sycl/checked_cleanup.hpp"\n#endif',1)
methods=r'''bool Verifier::can_release_prefill_head(const NativeHead& head,std::string& err) const {
#ifndef STRATA_ENABLE_SYCL
    (void)head;err="head release requires SYCL";return false;
#else
    const auto phase=diag_window_.read().phase;
    if(head_detached_ || released_.load() || head_!=&head || !head.loaded() || head.head_detached() ||
       !g_ || !cs_ || !copy_ || max_t_!=4 || next_ || lb_!=0 || le_!=g_->n_layers ||
       (phase!=VerifyPhase::inactive && phase!=VerifyPhase::completed)) {
        err="verify: head release requires idle intact single-device native MTP4 window";return false;
    }
    return true;
#endif
}
bool Verifier::retire_prefill_head(const NativeHead& head,std::string& err) {
#ifndef STRATA_ENABLE_SYCL
    (void)head;err="head release requires SYCL";return false;
#else
    using strata::sycl_backend::cleanup_fail_stop;
    const OnDevice on_device(device_);
    if(!can_release_prefill_head(head,err))return false;
    int current=-1;
    if(cudaGetDevice(&current)!=cudaSuccess || current!=device_)cleanup_fail_stop("verify head retirement owner device");
    if(!wait_commit(err))cleanup_fail_stop("verify outstanding commit completion before head detach");
    if(cudaStreamSynchronize(cs_)!=cudaSuccess)cleanup_fail_stop("verify compute drain before head detach");
    if(cudaStreamSynchronize(copy_)!=cudaSuccess)cleanup_fail_stop("verify copy/callback drain before head detach");
    head_detached_=true;head_recapture_checked_=true;
    release_head_bytes_=head.weight_bytes();release_head_row_=head.row_bytes();release_head_type_=head.type();
    auto destroy=[](cudaGraphExec_t& e){if(e){
        if(cudaGraphExecDestroy(e)!=cudaSuccess)cleanup_fail_stop("verify head graph retirement");
        e=nullptr;
    }};
    for(auto& e:exec_)destroy(e);
    for(auto& v:segments_){for(auto& e:v)destroy(e);v.clear();}
    for(auto& v:work_){for(auto& e:v)destroy(e);v.clear();}
    work_launched_=false;work_status_=cudaSuccess;reached_.store(0);
    // commit_exec_ contains only state commit operations, no output head. Its
    // event has completed; keeping it avoids destroying session-owned storage.
    return true;
#endif
}
bool Verifier::resume_prefill_head(const NativeHead& head,std::string& err) {
    if(!head_detached_ || released_.load() || head_!=&head || !head.loaded() || head.head_detached() ||
       head.weight_bytes()!=release_head_bytes_ || head.row_bytes()!=release_head_row_ || head.type()!=release_head_type_) {
        err="verify: head resume owner/format/state mismatch";return false;
    }
    head_detached_=false;return true; // windows recapture lazily; arenas remain unchanged
}

'''
s=s.replace('Verifier::~Verifier() {',methods+'Verifier::~Verifier() {',1)
s=s.replace('    g_verifiers.remove(this); // Wait for acquired callbacks before freeing any member.','''    g_verifiers.remove(this); // Wait for acquired callbacks before freeing any member.
#ifdef STRATA_ENABLE_SYCL
    if(head_detached_)strata::sycl_backend::cleanup_fail_stop("verify teardown while head detached");
    if(head_recapture_checked_) {
        if(cs_ && cudaStreamSynchronize(cs_)!=cudaSuccess)strata::sycl_backend::cleanup_fail_stop("verify restored-head teardown compute drain");
        if(copy_ && cudaStreamSynchronize(copy_)!=cudaSuccess)strata::sycl_backend::cleanup_fail_stop("verify restored-head teardown copy drain");
        auto checked=[](cudaError_t status){if(status!=cudaSuccess)strata::sycl_backend::cleanup_fail_stop("verify restored-head checked owner teardown");};
        for(auto e:exec_)if(e)checked(cudaGraphExecDestroy(e));
        for(auto& window:segments_)for(auto e:window)if(e)checked(cudaGraphExecDestroy(e));
        for(auto& window:work_)for(auto e:window)if(e)checked(cudaGraphExecDestroy(e));
        if(commit_exec_)checked(cudaGraphExecDestroy(commit_exec_));
        if(layer_done_)checked(cudaEventDestroy(layer_done_));
        if(work_done_)checked(cudaEventDestroy(work_done_));
        if(commit_done_)checked(cudaEventDestroy(commit_done_));
        if(cs_)checked(cudaStreamDestroy(cs_));
        if(copy_)checked(cudaStreamDestroy(copy_));
        if(arena_)checked(cudaFree(arena_));
        void* hosts[]={h_tok_,h_step_,h_pos_,h_commit_,h_ple_,h_out_,h_x_,h_ids_,h_w_,h_seq_,h_flag_,h_ymiss_,h_flagA_,h_plan_,h_flagB_};
        for(auto h:hosts)if(h)checked(cudaFreeHost(h));
        return;
    }
#endif''',1)
# Guard full execution, capture and diagnostics until all owners are restored.
for marker in ['bool Verifier::capture(int T, std::string& err) {','bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {']:
 s=s.replace(marker,marker+'\n    if(head_detached_){err="verify: head is detached for prompt";return false;}',1)
s=s.replace('    const OnDevice on_device(device_);\n    if (T < 1 || T > max_t_)', '    const OnDevice on_device(device_);\n    if(head_detached_){err="verify: run while head detached";return false;}\n    if (T < 1 || T > max_t_)',1)
s=s.replace('bool Verifier::commit(int n_keep, std::string& err) {','bool Verifier::commit(int n_keep, std::string& err) {\n    if(head_detached_){err="verify: commit while head detached";return false;}',1)
s=s.replace('bool Verifier::copy_logits(int t, float* host) const {','bool Verifier::copy_logits(int t, float* host) const {\n    if(head_detached_)return false;',1)
s=s.replace('    if (next_ != nullptr) return next_->window_logprobs(', '    if(head_detached_){err="verify: logits while head detached";return false;}\n    if (next_ != nullptr) return next_->window_logprobs(',1)
s=s.replace('    cudaGetDevice(&device_);   // a layer split', '    if(head_detached_){err="verify: init while head detached";return false;}\n    cudaGetDevice(&device_);   // a layer split',1)
# Restored SYCL window graph ownership must never use legacy unchecked cleanup.
s=s.replace('    if (ce != cudaSuccess) {\n        err = std::string("verify: end segment capture:', '    if (ce != cudaSuccess) {\n        if(head_recapture_checked_)strata::sycl_backend::cleanup_fail_stop("verify head recapture end segment");\n        err = std::string("verify: end segment capture:',1)
s=s.replace('    const cudaError_t ie = cudaGraphInstantiate(&exec, graph, 0);\n    cudaGraphDestroy(graph);','''    const cudaError_t ie = cudaGraphInstantiate(&exec, graph, 0);
    if(head_recapture_checked_ && ie!=cudaSuccess)strata::sycl_backend::cleanup_fail_stop("verify head segment instantiate");
    const auto destroyed=cudaGraphDestroy(graph);
    if(head_recapture_checked_ && destroyed!=cudaSuccess)strata::sycl_backend::cleanup_fail_stop("verify head source graph destroy");''',1)
s=s.replace('    if (begin_verify_capture(cs_, T) != cudaSuccess) {\n        err = "verify: begin capture failed";', '    if (begin_verify_capture(cs_, T) != cudaSuccess) {\n        if(head_recapture_checked_)strata::sycl_backend::cleanup_fail_stop("verify head recapture begin");\n        err = "verify: begin capture failed";',1)
s=s.replace('    if (!record_window(T, cs_, err) || !end_segment(T, err)) {','    if (!record_window(T, cs_, err) || !end_segment(T, err)) {\n        if(head_recapture_checked_)strata::sycl_backend::cleanup_fail_stop("verify head window recapture incomplete");',1)
# Capture exceptions after a graph API may retain handles; do not unwind.
s=s.replace('bool Verifier::capture(int T, std::string& err) {','bool Verifier::capture(int T, std::string& err) try {',1)
end=s.index('\nbool Verifier::capture_commit(')
s=s[:end].rstrip()+''' catch (...) {
#ifdef STRATA_ENABLE_SYCL
    if(head_recapture_checked_)strata::sycl_backend::cleanup_fail_stop("verify head recapture exception");
#endif
    throw;
}
\n'''+s[end:]
put(n,s)
n='src/program/generate.cpp';s=get(n);s=s.replace('#include "strata/core/native_head.hpp"','#include "strata/core/native_head.hpp"\n#include "strata/core/prefill_head_scope.hpp"\n#ifdef STRATA_ENABLE_SYCL\n#include "strata/sycl/checked_cleanup.hpp"\n#endif',1)
s=s.replace('bool run_prompt_with_mtp_experts(', 'bool run_prompt_with_mtp_experts_impl(',1)
anchor="// Windows' WDDM driver model:"
wrapper=r'''// Both warm serve and standalone pass through this request-scoped boundary.
// Default off and short segments take exactly the existing prompt/expert path.
bool run_prompt_with_mtp_experts(strata::prefill::Prefill& prompt,strata::core::MtpDrafter& mtp,
        bool supported,strata::core::NativeHead& head,strata::core::Verifier* verifier,cudaStream_t main_cs,
        const int64_t* tokens,int64_t n,int64_t pos0,std::string& err) {
    auto run=[&](){return run_prompt_with_mtp_experts_impl(prompt,mtp,supported,tokens,n,pos0,err);};
    const char* requested=std::getenv("STRATA_PREFILL_RELEASE_HEADS");
    if(!requested || std::strcmp(requested,"1")!=0 || prompt.chunk()<=0 || n<=prompt.chunk())return run();
#ifndef STRATA_ENABLE_SYCL
    (void)head;(void)verifier;(void)main_cs;
    err="head release requires SYCL";return false;
#else
    const char* lm=std::getenv("STRATA_PREFILL_LAYER_MAJOR");
    const char* cp=std::getenv("STRATA_SPEC_COUPLED");
    if(!supported || !lm || std::strcmp(lm,"1")!=0 || (cp && std::strcmp(cp,"1")==0) ||
       std::getenv("STRATA_PREFILL_LOGITS") || std::getenv("STRATA_PREFILL_LOGITS_POSITIONS") ||
       std::getenv("STRATA_PREFILL_NLL") || std::getenv("STRATA_LOGPOS")) {
        err="head release requires long layer-major single-GPU native MTP4 without split/peer/remote/coupled or logits/NLL diagnostics";
        return false; // no owner is detached on unsupported routes
    }
    struct Boundary {
        strata::core::NativeHead& head;strata::core::MtpDrafter& mtp;
        strata::core::Verifier* verifier;cudaStream_t main_cs;
        strata::core::detail::PrefillHeadState state;
        using Phase=strata::core::detail::PrefillHeadState::Phase;
        [[noreturn]] static void fatal(const char* why) noexcept {strata::sycl_backend::cleanup_fail_stop(why);}
        void move(Phase from,Phase to) noexcept {if(!state.transition(from,to))fatal("head boundary duplicate/out-of-order transition");}
        bool prepare(std::string& error) {
            if(!state.readable() || !main_cs){error="head boundary is not ready";return false;}
            if(verifier && !verifier->can_release_prefill_head(head,error))return false;
            if(!mtp.prepare_prefill_head(head,error))return false;
            try {
                if(!head.prepare_prefill_weights(error)){mtp.cancel_prefill_head();return false;}
            }catch(...){mtp.cancel_prefill_head();throw;}
            move(Phase::ready,Phase::prepared);return true;
        }
        void retire_readers() noexcept try {
            // Ordinary target projection runs on main_cs, outside token graphs.
            if(cudaStreamSynchronize(main_cs)!=cudaSuccess)fatal("main head stream retirement");
            std::string error;
            if(verifier && !verifier->retire_prefill_head(head,error))fatal("verifier rejected prepared head retirement");
            mtp.retire_prefill_head_readers();
            move(Phase::prepared,Phase::retired);
        }catch(...){fatal("head reader retirement threw");}
        void detach() noexcept {
            if(state.phase()!=Phase::retired)fatal("head detach before all graph readers retired");
            head.detach_prefill_weights();mtp.detach_prefill_head();
            move(Phase::retired,Phase::detached);
        }
        void restore() noexcept {
            if(state.phase()!=Phase::detached)fatal("head restore state");
            // Prefill::run has returned/unwound its checked local owners. Drain
            // its main stream as well as MTP's KV-only queue before reuse.
            if(cudaStreamSynchronize(main_cs)!=cudaSuccess)fatal("prompt main completion before head restore");
            head.restore_prefill_weights();mtp.restore_prefill_head();
            move(Phase::detached,Phase::restored);
        }
        void resume() noexcept try {
            if(state.phase()!=Phase::restored)fatal("head resume before complete restore");
            head.resume_prefill_weights();mtp.resume_prefill_head();
            std::string error;
            if(verifier && !verifier->resume_prefill_head(head,error))fatal("verifier head resume failed");
            move(Phase::restored,Phase::ready);
        }catch(...){fatal("head resume threw");}
        ~Boundary() noexcept {if(!state.readable())fatal("head boundary escaped without restore");}
    } boundary{head,mtp,verifier,main_cs,{}};
    return strata::core::detail::with_prefill_head_release(boundary,run,err);
#endif
}

'''
s=s.replace(anchor,wrapper+anchor,1)
s=s.replace('                    ids.data() + at, to - at, at, err);','                    native_head, &ver, main_stream, ids.data() + at, to - at, at, err);',1)
s=s.replace('                o.tokens.data(), n_batched, 0, err)) {','                native_head, nullptr, main_stream, o.tokens.data(), n_batched, 0, err)) {',1)
# NativeHead is declared after MTP. This lease therefore runs BEFORE head free,
# while the drafter still exists, including standalone's no-subset borrowed head.
s=s.replace('    strata::core::NativeHead native_head;','''    strata::core::NativeHead native_head;
    struct HeadReaderLease {
        strata::core::MtpDrafter& mtp;
        cudaStream_t stream;
        ~HeadReaderLease() noexcept {mtp.retire_head_owner_readers(stream);}
    } head_reader_lease{mtp,main_stream};''',1)
put(n,s)
# Full root-based patch; no duplicate large translation units.
patch=[];pins={}
for n,s in new.items():
 old=base[n];patch.extend(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/'+n if old else '/dev/null',tofile='b/'+n))
 pins[n]={'base_sha256':hashlib.sha256(old.encode()).hexdigest() if old else None,'proposed_sha256':hashlib.sha256(s.encode()).hexdigest()}
(D/'source.diff').write_text(''.join(patch));(D/'pins.json').write_text(json.dumps(pins,indent=2)+'\n')
# Compact helper retained for standalone host fixture include only.
(D/'prefill_head_scope.hpp').write_text(helper)
