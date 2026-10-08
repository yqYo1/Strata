
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "strata/kernels/qsa.hpp"
namespace strata::kernels {
constexpr int NG_HC_DIM = 10240, HD = 256, CHUNK = 64;
struct KvHostPools {};
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}
}
namespace core {
struct ModelGeometry { int64_t n_head=24,n_head_kv=2,head_dim=256,idx_q_heads=4,idx_key_dim=128,n_expert=512; };
struct QsaState { int64_t max_cells=32768; };
struct SessionState { QsaState qsa_states[1]; int qsa_primary() const { return 0; } };
}
constexpr int64_t N=2560,HC=4,D=N*HC,LR=320,K=10,NE=512,C=10240,ZV=6144,HV=48;
constexpr int64_t GEMM_SCRATCH=32ll<<20;
constexpr size_t GEMM_WS=0,MMQ_TAIL=4096;
constexpr int DQ=2,MMQ_GROUP=16;
bool gr_unfused() { return false; }
bool prompt_f16() { return true; }
bool bf16x2(bool) { return false; }
bool bf16x2_hc(bool) { return false; }
bool compact_prefill() { return true; }
bool compact_hc_prefill() { return true; }
bool fused_layout(size_t,bool) { return false; }
int64_t stream_all_min() { return 1024; }
int64_t prompt_attn_batch(size_t T) { return std::min<int64_t>(32,T); }
int ring_slots(size_t) { return 8; }
int64_t MAXBLOB() { return 1363148800/512; }
namespace mmq { size_t q8_bytes(int64_t,int64_t) { return 0; } }
namespace fused { size_t group_bytes(int64_t,int) { return 0; } size_t act_bytes(int64_t,int64_t) { return 0; } }
struct MmqPlan { bool any=false,fallback=true; size_t gu_max=0,d_max=0; };
const MmqPlan& mmq_plan() { static MmqPlan p;return p; }
struct MoeBufs { size_t gu,h,xq,hq; };
struct Prefill {
    struct Impl {
        bool f16_io=true;
        float *xn=nullptr,*lo=nullptr,*gated=nullptr,*emb=nullptr,*mixed=nullptr,*bo=nullptr,*R=nullptr,*grs=nullptr,*inj=nullptr;
        uint16_t *xn16=nullptr,*lo16=nullptr,*xn16_lo=nullptr,*lo16_lo=nullptr,*mixed_bf=nullptr,*mixed_h=nullptr,*mixed_bf_lo=nullptr;
        int32_t* steps_dev=nullptr;
    };
    static uint64_t bytes_needed_impl(const core::ModelGeometry&,const core::SessionState&,int64_t,bool);
};
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    uint64_t failed_bytes = 0;          ///< the take that could not be allocated; the failure message names it
    uint64_t granule = 0;               ///< count_only: each take rounds up to this (cudaMalloc's 2 MiB pages: owned buffers)
    bool count_only = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T *take(size_t n, bool &ok) {
        uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        if (count_only) {
            if (granule > 0) bytes = (bytes + granule - 1) / granule * granule;
            used += bytes;
            return nullptr;
        }
        std::abort();
    }
};

void take_stage(Alloc&,const core::SessionState&,const strata::kernels::QsaShapes&,strata::kernels::KvHostPools&,bool&) {}
template <typename V>
void compact_take(Alloc& a, Prefill::Impl* m, V* Prefill::Impl::*field, size_t n, bool& ok) {
    V* value = a.take<V>(n, ok);
    if (m) m->*field = value;
}
void take_compact_hc(Alloc& a, Prefill::Impl* m, size_t T, bool& ok) {
    auto take = [&](auto field, size_t n) { compact_take(a, m, field, n, ok); };
    if (gr_unfused()) take(&Prefill::Impl::xn, T * D);
    else if (m) m->xn = nullptr;
    take(&Prefill::Impl::xn16, T * D);
    take(&Prefill::Impl::lo, T * LR);
    take(&Prefill::Impl::lo16, T * LR);
    take(&Prefill::Impl::gated, T * D);
    if (bf16x2_hc(m ? m->f16_io : prompt_f16())) {
        take(&Prefill::Impl::xn16_lo, T * D);
        take(&Prefill::Impl::lo16_lo, T * LR);
    }
}
void take_compact_base(Alloc& a, Prefill::Impl* m, size_t T, bool& ok, bool hc = false) {
    float* image = a.take<float>(T * N, ok);
    if (m) m->emb = m->mixed = m->bo = image;
    auto take = [&](auto field, size_t n) { compact_take(a, m, field, n, ok); };
    take(&Prefill::Impl::R, T * D);
    take(&Prefill::Impl::grs, T * HC);
    if (!hc) take_compact_hc(a, m, T, ok);
    take(&Prefill::Impl::inj, T * HC);
    take(&Prefill::Impl::mixed_bf, T * N);
    take(&Prefill::Impl::mixed_h, T * N);
    if (bf16x2(m ? m->f16_io : prompt_f16())) take(&Prefill::Impl::mixed_bf_lo, T * N);
    take(&Prefill::Impl::steps_dev, T * strata::kernels::kStepCount);
}
uint64_t hc_set_bytes(size_t T, bool hc) {
    if (!hc) return 0;
    Alloc a; a.count_only = true; bool ok = true;
    take_compact_hc(a, nullptr, T, ok);
    return a.used;
}
uint64_t ple_set_bytes(size_t T, bool compact) {
    if (!compact) return 0;
    constexpr int64_t HD = strata::kernels::NG_HC_DIM;
    // The original region fits a complete PLE chunk. Preserve its GEMM row
    // counts, including an odd final chunk, when shrinking the other scratch.
    const uint64_t per_token = (uint64_t) (3 * HD + N + 4) * 4 +
                               (uint64_t) N * (bf16x2(prompt_f16()) ? 4 : 2) + 4096;
    return (uint64_t) T * per_token;
}
uint64_t gdn_set_bytes(size_t T, bool compact = false) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<float>(T * 2 * HV, ok); a.take<float>(T * HV, ok);
    a.take<float>(T * HV, ok); a.take<float>(T * C, ok);
    if (!compact) { a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok); }
    return a.used;
}
uint64_t qsa_set_bytes(size_t T, int64_t cap, int64_t max_blocks, int64_t sel_batch, int64_t attn_batch,
                       const strata::kernels::QsaShapes& s, bool compact = false) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * 512, ok); a.take<float>(T * 512, ok); a.take<float>(T * 12288, ok); a.take<float>(T * ZV, ok);
    a.take<float>(T * 128, ok); a.take<float>(T * 512, ok);
    if (!compact) a.take<float>(T * ZV, ok);
    a.take<uint16_t>(T * ZV, ok);
    a.take<int32_t>(T * (size_t) cap, ok);
    a.take<float>((size_t) sel_batch * (size_t) max_blocks, ok);
    a.take<float>((size_t) attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(cap, s), ok);
    return a.used;
}
MoeBufs moe_bufs(size_t T, int64_t n_expert, bool fused, bool compact = false) {
    if (compact) return {T * 1280, T * 640, 0, 0};
    if (!fused) return {T * K * 1280, T * K * 640, mmq::q8_bytes((int64_t) (T * K), N), mmq::q8_bytes((int64_t) (T * K), 640)};
    const size_t ts = (size_t) std::min<int64_t>((int64_t) T, stream_all_min() - 1);   // MMQ's last small chunk
    return {std::max(ts * K * 1280, (fused::group_bytes((int64_t) (T * K), (int) n_expert) + 3) / 4),
            std::max(ts * K * 640, (fused::act_bytes((int64_t) (T * K), 640) + 3) / 4),
            std::max(mmq::q8_bytes((int64_t) (ts * K), N), fused::act_bytes((int64_t) T, N)),
            mmq::q8_bytes((int64_t) (ts * K), 640)};
}
uint64_t moe_set_bytes(size_t T, int64_t n_expert, bool fused, bool compact = false) {
    const MmqPlan& mp = mmq_plan();
    const MoeBufs mb = moe_bufs(T, n_expert, fused, compact);
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * n_expert, ok); a.take<float>(T * K, ok); a.take<int32_t>(T * K, ok); a.take<int32_t>(T * K, ok);
    a.take<int32_t>(T * K, ok);
    if (mp.fallback) a.take<uint16_t>(T * (compact ? 1 : K) * N, ok);
    a.take<float>(mb.gu, ok);
    if (mp.fallback) a.take<uint16_t>(T * (compact ? 1 : K) * 640, ok);
    a.take<float>(T * K * N, ok); a.take<float>(T * 640, ok);
    a.take<float>(T * 640, ok); a.take<uint16_t>(T * 640, ok); a.take<float>(T * N, ok); a.take<float>(T, ok);
    if (mp.any) {
        a.take<uint8_t>(mb.xq, ok);
        a.take<float>(mb.h, ok);
        a.take<uint8_t>(mb.hq, ok);
    }
    return a.used;
}
uint64_t Prefill::bytes_needed_impl(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                    bool owned_pages) {
    // the same allocation sequence as `init`, counted
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.count_only = true;
    if (owned_pages) o.granule = 2ull << 20;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    o.take<uint8_t>(GEMM_WS, ok);
    auto f = [&](size_t n) { o.take<float>(n, ok); };
    const bool f16_io = prompt_f16();
    const bool compact = compact_prefill();
    const bool compact_hc = compact_hc_prefill();
    if (compact) {
        take_compact_base(o, nullptr, T, ok, compact_hc);
    } else {
        f(T * N); f(T * D); f(T * D); o.take<uint16_t>(T * D, ok); f(T * LR); o.take<uint16_t>(T * LR, ok);
        f(T * D); f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok); f(T * N);
        if (bf16x2_hc(f16_io)) { o.take<uint16_t>(T * D, ok); o.take<uint16_t>(T * LR, ok); }
        if (bf16x2(f16_io)) o.take<uint16_t>(T * N, ok);
        o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    }
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    const int64_t max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    o.take<uint8_t>((size_t) std::max({hc_set_bytes(T, compact_hc), ple_set_bytes(T, compact), gdn_set_bytes(T, compact),
        qsa_set_bytes(T, cap, max_blocks, 256, prompt_attn_batch(T), s, compact),
        moe_set_bytes(T, g.n_expert, fused_layout(T, true), compact)}), ok);
    for (int i = 0; i < DQ; ++i) { o.take<uint16_t>(1280 * 2560, ok); o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        o.take<int32_t>(T * K, ok);
        o.take<int32_t>((size_t) (2 * (g.n_expert + g.n_expert / MMQ_GROUP + 2)), ok);
        o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL, ok);
    }
    if (owned_pages) {
        if (ring_slots(T) > 0) o.take<uint8_t>((size_t) ring_slots(T) * (size_t) MAXBLOB(), ok);   // one allocation
    } else {
        for (int i = 0; i < ring_slots(T); ++i) o.take<uint8_t>((size_t) MAXBLOB(), ok);
    }
    f(T * N);
    f((size_t) strata::kernels::NG_HC_DIM);
    strata::kernels::KvHostPools stage;
    take_stage(o, ss, s, stage, ok);
    return o.used + (8u << 20);   // alignment slack
}
int main() {
    core::ModelGeometry g;core::SessionState ss;
    for(int64_t T:{8192,12288,16384}) {
        uint64_t accounted=Prefill::bytes_needed_impl(g,ss,T,false)-(8u<<20);
        uint64_t estimated_owned=Prefill::bytes_needed_impl(g,ss,T,true);
        auto s=strata::kernels::qsa_real_shapes();
        auto cap=strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells,s);
        uint64_t region=std::max({hc_set_bytes(T,true),ple_set_bytes(T,true),gdn_set_bytes(T,true),qsa_set_bytes(T,cap,8194,256,32,s,true),moe_set_bytes(T,512,false,true)});
        std::printf("%lld %llu %llu %llu\n",(long long)T,(unsigned long long)accounted,(unsigned long long)estimated_owned,(unsigned long long)region);
    }
}
