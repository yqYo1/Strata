#pragma once
// Host-only accounting. No SYCL types, events, queue queries or capacity model.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

namespace strata::prefill::detail {
class RouteCensus {
public:
    static constexpr unsigned max_layers = 64, max_experts = 512;
    static constexpr size_t output_cap = 32u * 1024u * 1024u;
    // Stable receipt enums. Source 1/2 is the existing transfer-ledger join.
    enum Branch : unsigned { generic_iq = 1, generic_gguf = 2, mmq = 3 };
    enum Source : unsigned { resident = 1, pinned = 2, stager_ram = 3,
                              stager_transient = 4, stager_gguf = 5 };
    struct Meta {
        int64_t chunk = 0, T = 0, K = 0;
        unsigned layer = 0, experts = 0, branch = 0;
        int gu = 0, down = 0, H = 0, FF = 0;
        uint64_t gu_row = 0, down_row = 0, up_off = 0, down_off = 0, bytes = 0;
        uint64_t layer_offset = 0;
        int layout_version = 0;
        bool native = false, stream_all = false, mmq_built = false, fused_built = false, fused_only = false;
    };
private:
    using Clock = std::chrono::steady_clock;
    struct Cell { uint32_t ne = 0, calls = 0, gu = 0, down = 0, residence = 0; };
    // Disjoint producer lanes. Only the copy issuer writes Copy; only the
    // compute thread writes Cell/Meta. Read together ONLY after existing join.
    struct Copy { uint64_t bytes = 0; uint32_t count = 0, source = 0; };
    struct Layer { Meta meta; bool seen = false; std::array<Cell, max_experts> cells{}; };
    struct Scratch {
        std::array<Layer, max_layers> layers{};
        std::array<std::array<Copy, max_experts>, max_layers> copies{};
        std::array<std::array<unsigned, max_experts>, max_layers> planned{};
    };
    struct Bin { uint32_t ne = 0, source = 0, calls = 0; uint64_t rows = 0, bytes = 0; };
    static_assert(sizeof(Scratch) <= 2u * 1024u * 1024u, "route census scratch exceeds 2MiB budget");
    static_assert(sizeof(std::array<Bin, max_experts>) <= 16u * 1024u,
                  "route census histogram scratch exceeds 16KiB stack budget");
    std::unique_ptr<Scratch> scratch_;
    std::unique_ptr<char[]> output_;
    size_t output_size_ = 0;
    const char* reason_ = "none";
    bool valid_ = true, finished_ = false, active_chunk_ = false, copy_fault_ = false;
    int64_t requested_ = 0, position_ = 0, chunk_T_ = 0, chunk_p0_ = 0;
    unsigned lb_ = 0, le_ = 0, admitted_experts_ = 0;
    uint64_t chunks_ = 0, layers_ = 0, tokens_ = 0, calls_ = 0, rows_ = 0;
    uint64_t transfer_calls_ = 0, transfer_bytes_ = 0, routed_transfer_calls_ = 0;
    uint64_t host_ns_ = 0, copy_host_ns_ = 0, serialize_ns_ = 0;
    struct HostSpan {
        uint64_t& target; Clock::time_point start = Clock::now();
        ~HostSpan() { target += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count(); }
    };
    static bool add(uint64_t& a, uint64_t b) noexcept {
        if (b > UINT64_MAX-a) return false;
        a += b; return true;
    }
    // Constructor-pinned bounds, never changed after producer startup. The
    // copy issuer can use these without inspecting main-thread validity.
    bool admitted_id(int layer, int expert) const noexcept {
        return layer >= 0 && (unsigned)layer >= lb_ && (unsigned)layer < le_ &&
               expert >= 0 && (unsigned)expert < admitted_experts_;
    }
    template<class... Args> void append(const char* format, Args... args) noexcept {
        if (!valid_) return;
        const size_t left = output_cap-8192-output_size_;
        const int n = std::snprintf(output_.get()+output_size_, left, format, args...);
        if (n < 0 || (size_t)n >= left) { invalidate("output_cap_or_format_failure"); return; }
        output_size_ += (size_t)n;
    }
public:
    const bool on;
    RouteCensus(int64_t n, int64_t pos, int64_t lb, int64_t le, int64_t experts,
                bool unsupported_multi_gpu) noexcept
        : requested_(n), position_(pos), on(std::getenv("STRATA_PREFILL_ROUTE_CENSUS") != nullptr &&
            std::strcmp(std::getenv("STRATA_PREFILL_ROUTE_CENSUS"), "0") != 0) {
        if (!on) return;
        const char* v = std::getenv("STRATA_PREFILL_ROUTE_CENSUS");
        if (std::strcmp(v,"1") != 0) { invalidate("env_must_be_exact_0_or_1"); return; }
        if (n < 32768 || n > 262144 || pos < 0 || pos > INT64_MAX-n ||
            lb < 0 || le <= lb || le > max_layers || experts <= 0 || experts > max_experts) {
            invalidate("request_geometry_out_of_bound"); return;
        }
        lb_ = (unsigned)lb; le_ = (unsigned)le; admitted_experts_ = (unsigned)experts;
        if (unsupported_multi_gpu) { invalidate("unsupported_multi_gpu_or_pipeline"); return; }
        scratch_.reset(new(std::nothrow) Scratch);
        output_.reset(new(std::nothrow) char[output_cap]);
        if (!scratch_ || !output_) invalidate("allocation_failure");
    }
    RouteCensus(const RouteCensus&) = delete;
    RouteCensus& operator=(const RouteCensus&) = delete;
    static void fatal_receipt(const char* reason) noexcept {
        // Immutable literal receipt; safe from either producer before existing
        // no-unwind process termination. Never inspect a live producer lane.
        std::fprintf(stderr,"{\"kind\":\"prefill_route_census_receipt\",\"valid\":false,\"reason\":\"%s\",\"capacity_join\":false}\n",reason);
        std::fflush(stderr);
    }
    ~RouteCensus() { if (on && !finished_) { invalidate("request_early_return_or_exception"); finish(false); } }
    void invalidate(const char* reason) noexcept { if (on && valid_) { valid_=false; reason_=reason; } }
    void begin_chunk(int64_t p0, int64_t T) noexcept {
        if (!on || !valid_) return;
        HostSpan span{host_ns_};
        if (copy_fault_) { invalidate("copy_identity_duplicate_or_invalid"); return; }
        if (active_chunk_ || p0 != position_+(int64_t)tokens_ || T<=0 || T>8192 ||
            (uint64_t)T > (uint64_t)requested_-tokens_) { invalidate("chunk_geometry_or_coverage"); return; }
        // No issuer is alive here: the preceding chunk has been joined.
        for(unsigned i=0;i<max_layers;++i) {
            scratch_->layers[i].seen=false;
            for(unsigned e=0;e<max_experts;++e) {
                scratch_->layers[i].cells[e]=Cell{};
                scratch_->copies[i][e]=Copy{};
                scratch_->planned[i][e]=0;
            }
        }
        // copy_fault_ is request-sticky: even an out-of-range copy before the
        // next chunk or after the last fold must never be reset out of evidence.
        copy_host_ns_=0;
        chunk_p0_=p0; chunk_T_=T; active_chunk_=true;
    }
    void layer(const Meta& m, const int32_t* cnt) noexcept {
        if (!on || !valid_) return;
        HostSpan span{host_ns_};
        if (!active_chunk_ || m.layer<lb_ || m.layer>=le_ || m.experts!=admitted_experts_ || !cnt ||
            m.chunk!=chunk_p0_ || m.T!=chunk_T_ || m.K!=10 || m.bytes==0 || m.bytes>4u*1024u*1024u ||
            m.branch<generic_iq || m.branch>mmq || (m.branch==mmq && !m.mmq_built)) {
            invalidate("layer_geometry_or_unknown_branch"); return;
        }
        if ((m.native && m.branch==generic_gguf) || (!m.native && m.branch==generic_iq) ||
            (m.fused_only && !m.native)) {
            invalidate("branch_or_fused_only_native_geometry_disagreement"); return;
        }
        // fused_only is eligibility, not proof of executed fused dispatch.
        // A native generic fallback with fused_only=true remains admissible.
        if (m.native && (m.H!=2560 || m.FF!=640 || !m.gu_row || !m.down_row ||
            m.gu_row>4u*1024u*1024u || m.down_row>4u*1024u*1024u || m.up_off!=m.gu_row*(uint64_t)m.FF ||
            m.down_off!=2*m.up_off || m.bytes!=m.down_off+(uint64_t)m.H*m.down_row ||
            (!((m.gu==18||m.gu==21||m.gu==22)&&(m.down==20||m.down==42)) && !(m.gu==23&&m.down==20)))) {
            invalidate("native_geometry_or_pair_without_current_manifest_match"); return;
        }
        if(!m.native) { invalidate("unsupported_non_native_pack_geometry");return; }
        Layer& l=scratch_->layers[m.layer];
        if (l.seen) { invalidate("duplicate_layer"); return; }
        l.meta=m; l.seen=true;
        uint64_t sum=0;
        for (unsigned e=0;e<m.experts;++e) {
            if (cnt[e]<0 || cnt[e]>m.T || !add(sum,(uint64_t)cnt[e])) { invalidate("route_count_invalid"); return; }
            l.cells[e].ne=(uint32_t)cnt[e];
        }
        if (sum!=(uint64_t)m.T*(uint64_t)m.K) invalidate("routed_rows_not_T_times_K");
    }
    // Planning does not count a copy. Only actual submission finalizes source.
    void plan_source(int layer,int expert,unsigned source) noexcept {
        if(!on||!valid_)return;
        HostSpan span{host_ns_};
        if(!active_chunk_ || !admitted_id(layer,expert)) { invalidate("source_plan_identity");return; }
        unsigned& p=scratch_->planned[(unsigned)layer][(unsigned)expert];
        if(p || source<pinned || source>stager_gguf) { invalidate("source_plan_duplicate_or_invalid");return; }
        p=source;
    }
    unsigned stager_source(int layer,int expert) const noexcept {
        if(!scratch_||!admitted_id(layer,expert))return 0;
        return scratch_->planned[(unsigned)layer][(unsigned)expert];
    }
    // Called immediately AFTER the actual successful transfer submission.
    // No main validity reads/writes from issuer. active_chunk_ is published
    // before existing thread startup and stays unchanged until its existing
    // join; the issuer may therefore read this phase gate without a data race.
    void copied(int layer, int expert, uint64_t bytes, unsigned source) noexcept {
        if (!on || !scratch_) return;
        HostSpan span{copy_host_ns_};
        if (!active_chunk_ || !admitted_id(layer,expert) ||
            source<pinned || source>stager_gguf) { copy_fault_=true; return; }
        Copy& c=scratch_->copies[(unsigned)layer][(unsigned)expert];
        if (c.count || !bytes) { copy_fault_=true; return; }
        c.bytes=bytes; c.source=source; c.count=1;
    }
    void executed(int layer, int expert, bool is_resident) noexcept {
        if (!on || !valid_) return;
        HostSpan span{host_ns_};
        if (!active_chunk_ || !admitted_id(layer,expert)) { invalidate("call_identity_invalid"); return; }
        Layer& l=scratch_->layers[(unsigned)layer]; Cell& c=l.cells[(unsigned)expert];
        if (!l.seen || !c.ne || c.calls++) { invalidate("unexpected_or_duplicate_expert_call"); return; }
        c.residence=is_resident ? static_cast<uint32_t>(resident) : uint32_t{0};
    }
    void product(int layer, int expert, unsigned role) noexcept {
        if (!on || !valid_) return;
        HostSpan span{host_ns_};
        if (!active_chunk_ || !admitted_id(layer,expert) || role>1) { invalidate("product_identity_invalid"); return; }
        const Layer& l=scratch_->layers[(unsigned)layer];
        Cell& c=scratch_->layers[(unsigned)layer].cells[(unsigned)expert];
        if (!l.seen || !c.ne || c.calls!=1 || (role==0 ? c.gu!=0 : c.down!=0)) {
            invalidate("unexpected_or_duplicate_product"); return;
        }
        if (role==0) ++c.gu; else ++c.down;
    }
    // Existing chunk issuer join precedes this method. It never waits itself.
    void end_chunk() noexcept {
        if (!on || !valid_) return;
        HostSpan span{serialize_ns_};
        if (!active_chunk_ || copy_fault_) { invalidate("copy_identity_duplicate_or_invalid"); return; }
        for (unsigned i=lb_;i<le_ && valid_;++i) {
            const Layer& l=scratch_->layers[i]; const Meta& m=l.meta;
            if (!l.seen) { invalidate("missing_layer"); break; }
            std::array<Bin,max_experts> bins{}; unsigned nb=0;
            uint64_t row_sum=0, call_sum=0, copied_count=0, copied_bytes=0, routed_copies=0;
            for (unsigned e=0;e<m.experts && valid_;++e) {
                const Cell& c=l.cells[e]; const Copy& cp=scratch_->copies[i][e];
                const unsigned planned=scratch_->planned[i][e];
                if((planned!=0)!=(cp.count!=0) || (cp.count && planned!=cp.source)) { invalidate("planned_vs_executed_transfer_coverage");break; }
                if (cp.count) {
                    if (cp.bytes!=m.bytes || (!m.stream_all && !c.ne)) { invalidate("transfer_extent_or_unrouted_copy"); break; }
                    ++copied_count; if (!add(copied_bytes,cp.bytes)) { invalidate("byte_overflow"); break; }
                }
                if (!c.ne) { if(c.calls||c.gu||c.down) invalidate("unrouted_product"); continue; }
                const bool generic=m.branch!=mmq;
                if (c.calls!=1 || (generic?(c.gu!=1||c.down!=1):(c.gu!=0||c.down!=0)) ||
                    (c.residence==resident ? cp.count!=0 : cp.count!=1)) { invalidate("call_pair_or_transfer_reconciliation"); break; }
                const unsigned source=c.residence ? c.residence : cp.source;
                if(source<resident||source>stager_gguf) { invalidate("unknown_actual_source"); break; }
                if(cp.count)++routed_copies;
                bins[nb++]={c.ne,source,1,c.ne,m.bytes}; ++call_sum; row_sum+=c.ne;
            }
            if (!valid_) break;
            if(row_sum!=(uint64_t)m.T*m.K) { invalidate("layer_partition_reconciliation"); break; }
            std::sort(bins.begin(),bins.begin()+nb,[](const Bin&a,const Bin&b){return a.ne<b.ne||(a.ne==b.ne&&a.source<b.source);});
            append("{\"kind\":\"prefill_route_layer\",\"chunk_p0\":%lld,\"T\":%lld,\"K\":%lld,\"layer\":%u,\"n_expert\":%u,\"native\":%s,\"gu_type\":%d,\"down_type\":%d,\"H\":%d,\"FF\":%d,\"gu_row\":%llu,\"down_row\":%llu,\"up_off\":%llu,\"down_off\":%llu,\"blob_bytes\":%llu,\"branch\":%u,\"owner\":\"primary\",\"stream_all\":%s,\"mmq_built\":%s,\"fused_built\":%s,\"calls\":%llu,\"rows\":%llu,\"transfer_calls\":%llu,\"transfer_bytes\":%llu,\"routed_transfer_calls\":%llu,\"bins\":[",
                   (long long)m.chunk,(long long)m.T,(long long)m.K,i,m.experts,m.native?"true":"false",m.gu,m.down,m.H,m.FF,
                   (unsigned long long)m.gu_row,(unsigned long long)m.down_row,(unsigned long long)m.up_off,(unsigned long long)m.down_off,
                   (unsigned long long)m.bytes,m.branch,m.stream_all?"true":"false",m.mmq_built?"true":"false",m.fused_built?"true":"false",
                   (unsigned long long)call_sum,(unsigned long long)row_sum,(unsigned long long)copied_count,(unsigned long long)copied_bytes,(unsigned long long)routed_copies);
            bool first=true;
            for(unsigned b=0;b<nb;) {
                Bin v=bins[b++]; while(b<nb&&bins[b].ne==v.ne&&bins[b].source==v.source) {
                    ++v.calls; v.rows+=bins[b].rows; if(!add(v.bytes,bins[b].bytes)){invalidate("histogram_byte_overflow");break;} ++b;
                }
                append("%s{\"ne\":%u,\"source\":%u,\"transfer_source\":%u,\"calls\":%u,\"routed_rows\":%llu,\"logical_packed_bytes\":%llu}",first?"":",",v.ne,v.source,v.source==resident?0:v.source==pinned?1:2,v.calls,(unsigned long long)v.rows,(unsigned long long)v.bytes);first=false;
            }
            append("],\"layout_version\":%d,\"packed_layer_offset\":%llu,\"local_calls\":%llu,\"local_rows\":%llu,\"peer_calls\":0,\"peer_rows\":0,\"generic_GU_calls\":%llu,\"generic_Down_calls\":%llu,\"use_mmq\":%s,\"fused_l\":false,\"fused_nat\":false,\"fused_only\":%s}\n",
                   m.layout_version,(unsigned long long)m.layer_offset,(unsigned long long)call_sum,(unsigned long long)row_sum,
                   (unsigned long long)(m.branch!=mmq?call_sum:0),(unsigned long long)(m.branch!=mmq?call_sum:0),m.branch==mmq?"true":"false",m.fused_only?"true":"false"); ++layers_;
            if(!add(calls_,call_sum)||!add(rows_,row_sum)||!add(transfer_calls_,copied_count)||!add(transfer_bytes_,copied_bytes)||!add(routed_transfer_calls_,routed_copies))invalidate("request_count_overflow");
        }
        host_ns_+=copy_host_ns_; ++chunks_; tokens_+=(uint64_t)chunk_T_; active_chunk_=false;
    }
    void finish(bool success) noexcept {
        if (!on || finished_) return;
        finished_=true;
        if(!success)invalidate("request_failed");
        if(valid_&&copy_fault_)invalidate("copy_identity_duplicate_or_invalid");
        if(valid_&&(active_chunk_||tokens_!=(uint64_t)requested_||layers_!=chunks_*(le_-lb_)||!calls_))invalidate("request_final_coverage");
        uint64_t hash=14695981039346656037ull;
        if(valid_) { for(size_t i=0;i<output_size_;++i)hash=(hash^(unsigned char)output_[i])*1099511628211ull;
            if(std::fwrite(output_.get(),1,output_size_,stderr)!=output_size_) { invalidate("output_write_failure"); }
        }
        std::fprintf(stderr,"{\"kind\":\"prefill_route_census_receipt\",\"valid\":%s,\"reason\":\"%s\",\"scope\":\"host_accounting_only_no_capacity_prediction\",\"requested_tokens\":%lld,\"pos0\":%lld,\"chunks\":%llu,\"layers\":%llu,\"calls\":%llu,\"routed_rows\":%llu,\"transfer_calls\":%llu,\"transfer_bytes\":%llu,\"routed_transfer_calls\":%llu,\"output_bytes\":%zu,\"histogram_fnv1a64\":\"%016llx\",\"host_accumulation_ns\":%llu,\"host_serialization_ns\":%llu,\"device_commands_added\":0,\"capacity_join\":false}\n",
            valid_?"true":"false",reason_,(long long)requested_,(long long)position_,(unsigned long long)chunks_,(unsigned long long)layers_,
            (unsigned long long)(valid_?calls_:0),(unsigned long long)(valid_?rows_:0),(unsigned long long)(valid_?transfer_calls_:0),
            (unsigned long long)(valid_?transfer_bytes_:0),(unsigned long long)(valid_?routed_transfer_calls_:0),valid_?output_size_:0,
            (unsigned long long)(valid_?hash:0),(unsigned long long)host_ns_,(unsigned long long)serialize_ns_);
    }
};
} // namespace strata::prefill::detail
