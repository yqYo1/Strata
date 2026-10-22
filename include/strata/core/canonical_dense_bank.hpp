#pragma once
#include "strata/core/weights.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_dense_bank_policy.hpp"
#include "strata/core/canonical_bank_policy.hpp"
#include "strata/kernels/ple.hpp"
#include <cuda_runtime.h>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace strata::core {
// Only the generator's exact cudaMalloc arena variable is borrowed as owner.
// Every transition is subordinate to the ONE NativeDensePhase reader lease.
class CanonicalDenseBank {
public:
    enum class State { Full, Prepared, Bank, RestoredPending };
    using Overlay=std::map<std::string,WeightRef>;
    ~CanonicalDenseBank() {if(state_!=State::Full || bank_ || persistent_) stop("owner escaped transition");}
    CanonicalDenseBank()=default;
    CanonicalDenseBank(const CanonicalDenseBank&)=delete;
    CanonicalDenseBank& operator=(const CanonicalDenseBank&)=delete;
    State state() const {return state_;}
    uint64_t bank_bytes() const {return bank_bytes_;}
    uint64_t persistent_bytes() const {return persistent_bytes_;}
    uint64_t arena_bytes() const {return snapshot_.bytes.size();}
    const EngineSnapshot& snapshot() const {return snapshot_;}
    const Overlay& view() const {return view_;}
    const std::map<std::string,uint64_t>& persistent_inventory() const {return persistent_offsets_;}
    bool adopt(EngineSnapshot& capture,WeightTable& table,void*& arena,const NativeDense& native,std::string& err) try {
        if(table_ || !capture.complete || capture.loaded_table!=&table || capture.loaded_arena!=arena || !arena ||
           capture.bytes.size()!=table.report().arena_bytes || !detail::canonical_capture_alignment(capture.arena_alignment) || capture.rows.empty()) {err="canonical exact owner/capture mismatch";return false;}
        // Native originals must have been skipped canonically. Reject before
        // retirement rather than leaving original.data pointing to freed arena.
        for(const auto& source:native.packed_sources()) if(source.original.data || source.original.resident || capture.rows.count(source.name)) {
            err="canonical bank rejects canonical/native overlapping originals";return false;
        }
        if(native.packed_sources().empty()) {err="canonical bank requires native startup snapshots";return false;}
        for(const char* n:{"blk.1.ple_value.weight","blk.1.ple_norm_key.weight","blk.1.ple_norm_query.weight","blk.1.ple_norm_conv.weight","blk.1.ple_conv1d.weight"})
            if(!capture.rows.count(n)) {err="canonical bank requires all persistent PLE rows";return false;}
        for(int l=0;l<48;++l) for(const char* suffix:{"hc_attn_norm.weight","hc_ffn_norm.weight"}) {
            auto it=capture.rows.find("blk."+std::to_string(l)+"."+suffix);
            if(it==capture.rows.end() || it->second.original.bytes!=40960 || it->second.original.kind!=WeightKind::F32 || it->second.original.elements!=10240) {err="canonical HC96 norm ownership mismatch";return false;}
        }
        const auto& pv=capture.rows.at("blk.1.ple_value.weight").original;
        if(pv.kind!=WeightKind::Bf16InF32 || pv.bytes!=13107200) {err="canonical PLE value is not the baseline BF16 form";return false;}
        for(const char* n:{"blk.1.ple_norm_key.weight","blk.1.ple_norm_query.weight","blk.1.ple_norm_conv.weight"}) {
            const auto& norm=capture.rows.at(n).original;
            if(norm.kind!=WeightKind::F32 || norm.bytes!=40960) {err="canonical PLE norm form";return false;}
        }
        const auto pk=capture.rows.find("blk.1.ple_key.weight");
        if(pk!=capture.rows.end() && !pk->second.original.quantized() && pk->second.original.kind!=WeightKind::Bf16InF32) {
            err="canonical unquantized PLE key must be baseline BF16";return false;
        }
        std::array<uint64_t,48> layer_bytes{};
        std::array<std::map<std::string,uint64_t>,48> next_offsets;
        std::map<std::string,uint64_t> next_persistent;
        uint64_t next_persistent_bytes=0;
        std::vector<std::pair<uint64_t,uint64_t>> spans;
        for(const auto& [name,row]:capture.rows) {
            if(row.offset>capture.bytes.size() || row.original.bytes>capture.bytes.size()-row.offset) {err="canonical row extent";return false;}
            if(row.engine_fingerprint!=detail::canonical_fingerprint(capture.bytes.data()+row.offset,size_t(row.original.bytes))) {
                err="canonical captured engine payload identity mismatch";return false;
            }
            const auto* ref=table.find(name);
            if(!ref || !same(*ref,row.original) || ref->native_data) {err="canonical source/table mismatch";return false;}
            if(row.original.bytes) spans.emplace_back(row.offset,row.offset+row.original.bytes);
            const int layer=detail::canonical_layer(name);
            const bool persistent=detail::canonical_persistent(name);
            auto& offset=persistent ? next_persistent_bytes : layer_bytes[size_t(layer)];
            const auto stride=detail::canonical_stride(row.original.bytes);
            if(stride>UINT64_MAX-offset) {err="canonical packed size overflow";return false;}
            (persistent ? next_persistent : next_offsets[size_t(layer)]).emplace(name,offset);offset+=stride;
        }
        std::sort(spans.begin(),spans.end());
        for(size_t i=1;i<spans.size();++i) if(spans[i].first<spans[i-1].second) {err="canonical overlapping arena aliases";return false;}
        uint64_t next_bank_bytes=0;for(auto bytes:layer_bytes) next_bank_bytes=std::max(next_bank_bytes,bytes);
        if(!next_bank_bytes || !next_persistent_bytes) {err="canonical empty layout";return false;}
        // Exact capture moves into private ownership only after all validation.
        offsets_=std::move(next_offsets);persistent_offsets_=std::move(next_persistent);
        bank_bytes_=next_bank_bytes;persistent_bytes_=next_persistent_bytes;
        snapshot_=std::move(capture);table_=&table;owner_=&arena;
        return true;
    } catch(const std::exception& e) {err=e.what();return false;}
    bool prepare(void* stream,std::string& err) try {
        if(state_!=State::Full || !stream || !validate(false)) {err="canonical prepare identity/state";return false;}
        void* next_bank=nullptr;void* next_persistent=nullptr;
        if(!allocate(next_bank,bank_bytes_,err)) return false;
        if(!allocate(next_persistent,persistent_bytes_,err)) {checked(cudaFree(next_bank),"prepare allocation rollback");return false;}
        // Build fallible maps before upload/publication; original owners live.
        Overlay next;
        try {for(const auto& [name,offset]:persistent_offsets_) {
            auto ref=snapshot_.rows.at(name).original;ref.data=static_cast<uint8_t*>(next_persistent)+offset;next.emplace(name,ref);
        }} catch(...) {checked(cudaFree(next_bank),"prepare bank host rollback");checked(cudaFree(next_persistent),"prepare pool host rollback");throw;}
        bank_=next_bank;persistent_=next_persistent;persistent_view_.swap(next);state_=State::Prepared;
        auto queue=static_cast<cudaStream_t>(stream);
        for(const auto& [name,offset]:persistent_offsets_) upload(name,static_cast<uint8_t*>(persistent_)+offset,queue);
        checked(cudaStreamSynchronize(queue),"persistent pool completion");
        return true;
    } catch(const std::exception& e) {if(state_==State::Prepared) stop("prepare exception after reservation");err=e.what();return false;}
    void cancel_prepared() {
        if(state_==State::Full) return;
        if(state_!=State::Prepared) stop("cancel outside Prepared");
        checked(cudaFree(bank_),"cancel canonical bank");checked(cudaFree(persistent_),"cancel canonical pool");
        bank_=persistent_=nullptr;persistent_view_.clear();state_=State::Full;
    }
    void enter() {
        if(state_!=State::Prepared || !validate(false)) stop("canonical enter identity/lease");
        for(const auto& [name,row]:snapshot_.rows) {
            auto value=row.original;const auto it=persistent_view_.find(name);
            if(it!=persistent_view_.end()) value=it->second;
            else {value.data=nullptr;value.resident=false;}
            table_->table_.at(name)=value;
        }
        void* old=*owner_;*owner_=nullptr;state_=State::Bank;
        checked(cudaFree(old),"release exact canonical arena");
    }
    bool activate(int layer,void* stream,const Overlay& native,std::string& err) try {
        if(state_!=State::Bank || layer<0 || layer>=48 || !stream || !validate(true)) {err="canonical activate identity/state";return false;}
        Overlay next=persistent_view_;
        for(const auto& [name,offset]:offsets_[size_t(layer)]) {
            auto ref=snapshot_.rows.at(name).original;ref.data=static_cast<uint8_t*>(bank_)+offset;next.emplace(name,ref);
        }
        for(const auto& [name,ref]:native) if(!next.emplace(name,ref).second) {err="native/canonical overlay collision";return false;}
        auto queue=static_cast<cudaStream_t>(stream);
        checked(cudaStreamSynchronize(queue),"retire old canonical view");view_.clear();
        for(const auto& [name,offset]:offsets_[size_t(layer)]) upload(name,static_cast<uint8_t*>(bank_)+offset,queue);
        checked(cudaStreamSynchronize(queue),"active canonical upload completion");view_.swap(next);return true;
    } catch(const std::exception& e) {err=e.what();return false;}
    bool restore_full(void* stream,std::string& err) {
        if(state_!=State::Bank || !stream || !validate(true)) stop("restore canonical identity/state");
        auto queue=static_cast<cudaStream_t>(stream);checked(cudaStreamSynchronize(queue),"restore canonical readers");
        void* next=nullptr;if(!allocate(next,snapshot_.bytes.size(),err)) return false;
        checked(cudaMemcpyAsync(next,snapshot_.bytes.data(),snapshot_.bytes.size(),cudaMemcpyHostToDevice,queue),"restore original engine bytes submit");
        checked(cudaStreamSynchronize(queue),"restore original engine bytes completion");
        if(!validate(true)) stop("restore canonical prepublication identity");
        for(auto& [name,row]:snapshot_.rows) {
            auto ref=row.original;ref.data=static_cast<uint8_t*>(next)+row.offset;table_->table_.at(name)=ref;row.original=ref;
        }
        *owner_=next;snapshot_.loaded_arena=next;view_.clear();state_=State::RestoredPending;
        // Retain banks/pool until outer transaction rebinds every PLE pointer.
        return true;
    }
    void finish_restore() {
        if(state_!=State::RestoredPending || !validate(false)) stop("finish canonical restore identity");
        checked(cudaFree(bank_),"retire restored canonical bank");checked(cudaFree(persistent_),"retire restored canonical pool");
        bank_=persistent_=nullptr;persistent_view_.clear();state_=State::Full;
    }
    bool bind_ple(strata::kernels::PleWeights& w,std::string& err) const {
        if(!table_ || (state_!=State::Bank && state_!=State::RestoredPending && state_!=State::Full)) {err="canonical PLE bind state";return false;}
        const auto* key=table_->find("blk.1.ple_key.weight");
        const auto* value=table_->find("blk.1.ple_value.weight");
        const auto* nk=table_->find("blk.1.ple_norm_key.weight");const auto* nq=table_->find("blk.1.ple_norm_query.weight");
        const auto* nc=table_->find("blk.1.ple_norm_conv.weight");const auto* conv=table_->find("blk.1.ple_conv1d.weight");
        if(!key || !value || !nk || !nq || !nc || !conv || !value->data || !nk->data || !nq->data || !nc->data || !conv->data) {
            err="canonical PLE persistent/full projection absent";return false;
        }
        w.key_bf16=nullptr;w.key_codes=nullptr;w.key_scales=nullptr;
        if(key->resident && key->data) {
            if(!key->quantized()) w.key_bf16=static_cast<const uint16_t*>(key->data);
            else {w.key_codes=static_cast<const uint8_t*>(key->data);w.key_scales=reinterpret_cast<const float*>(w.key_codes+key->codes_bytes);}
        }
        w.value_bf16=static_cast<const uint16_t*>(value->data);w.norm_key=static_cast<const float*>(nk->data);
        w.norm_query=static_cast<const float*>(nq->data);w.norm_conv=static_cast<const float*>(nc->data);
        w.conv1d_f16=static_cast<const uint16_t*>(conv->data);
        return true; // native key fields are managed by existing NativeDense phase
    }
private:
    EngineSnapshot snapshot_;
    WeightTable* table_=nullptr;void** owner_=nullptr;
    State state_=State::Full;void* bank_=nullptr;void* persistent_=nullptr;
    uint64_t bank_bytes_=0,persistent_bytes_=0;
    std::array<std::map<std::string,uint64_t>,48> offsets_;
    std::map<std::string,uint64_t> persistent_offsets_;
    Overlay persistent_view_,view_;
    [[noreturn]] static void stop(const char* op) {std::fprintf(stderr,"canonical dense bank: %s; owners retained\n",op);std::fflush(stderr);std::abort();}
    static void checked(cudaError_t status,const char* op) {if(status!=cudaSuccess) stop(op);}
    static bool allocate(void*& out,uint64_t bytes,std::string& err) {
        out=nullptr;const auto status=cudaMalloc(&out,size_t(bytes));
        const auto result=detail::classify_bank_allocation(status,cudaSuccess,cudaErrorMemoryAllocation,out);
        if(result==detail::BankAllocation::uncertain) stop("uncertain allocation result");
        if(result==detail::BankAllocation::retryable_oom) {err="canonical confirmed OOM; owners retained";return false;}
        return true;
    }
    static bool same(const WeightRef& a,const WeightRef& b) {
#define SAME(f) if(a.f!=b.f) return false
        SAME(data);SAME(bytes);SAME(ne0);SAME(ne1);SAME(elements);SAME(kind);SAME(code_bits);SAME(code_bias);
        SAME(group_elems);SAME(codebook_iq4nl);SAME(has_offset);SAME(codes_bytes);SAME(scales_bytes);SAME(offset_bytes);
        SAME(src_off);SAME(src_bytes);SAME(file_id);SAME(scales_fp16);SAME(act_kind);SAME(native_data);SAME(native_q8_1);SAME(native_type);SAME(resident);
#undef SAME
        return true;
    }
    bool validate(bool bank) const {
        if(!table_ || !owner_ || snapshot_.loaded_table!=table_ || (!bank && *owner_!=snapshot_.loaded_arena) || (bank && *owner_)) return false;
        for(const auto& [name,row]:snapshot_.rows) {
            auto expected=row.original;
            if(bank) {const auto it=persistent_view_.find(name);if(it!=persistent_view_.end()) expected=it->second;else {expected.data=nullptr;expected.resident=false;}}
            const auto* ref=table_->find(name);if(!ref || !same(*ref,expected)) return false;
        }
        return true;
    }
    void upload(const std::string& name,void* dst,cudaStream_t queue) {
        const auto& row=snapshot_.rows.at(name);
        if(row.original.bytes) checked(cudaMemcpyAsync(dst,snapshot_.bytes.data()+row.offset,row.original.bytes,cudaMemcpyHostToDevice,queue),"canonical H2D submit");
    }
};
} // namespace strata::core
