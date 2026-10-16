#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdarg>
#include <limits>

namespace strata::prefill::detail {
// Host issuer only. No queue, tensor, environment, callback or thread ownership.
struct LayerMajorShapeCensus {
    static constexpr int layers = 48, bins = 8193, experts = 512, topk = 10;
    static constexpr size_t output_cap = 1u << 20;
    struct Format {
        int gu_type = -1, d_type = -1;
        uint64_t gu_row = 0, d_row = 0, up_off = 0, down_off = 0, bytes = 0;
    };
    struct Record {
        Format format;
        uint64_t chunks = 0, tokens = 0, gu_calls = 0, down_calls = 0;
        uint64_t gu_rows = 0, down_rows = 0, configured_resident_gu = 0;
        uint64_t configured_resident_down = 0;
        std::array<uint64_t, bins> hist{};
    };
    // Exact issuer-ne validation for the active layer; retired only after equality.
    std::array<uint64_t, bins> gu_pending{}, down_pending{};
    std::array<Record, layers> record{};
    int active = -1, retired = 0;
    int64_t n = 0, pos = 0, chunk = 0;
    const char* failure = nullptr;
    bool finalized = false;
    LayerMajorShapeCensus(int64_t count, int64_t position, int64_t capacity) noexcept
        : n(count), pos(position), chunk(capacity) {
        if (n <= 0 || n > 262144 || pos < 0 || pos > 262144 || n > 262144-pos ||
            chunk <= 0 || chunk >= bins) reject("request_geometry");
    }
    ~LayerMajorShapeCensus() noexcept {
        if (!finalized) std::fputs("strata lm census: {\"version\":1,\"complete\":false,\"reason\":\"request_not_completed\"}\n", stderr);
    }
    LayerMajorShapeCensus(const LayerMajorShapeCensus&) = delete;
    LayerMajorShapeCensus& operator=(const LayerMajorShapeCensus&) = delete;
    void reject(const char* why) noexcept { if (!failure) failure = why; }
    bool add(uint64_t& value, uint64_t amount) noexcept {
        if (amount > UINT64_MAX-value) { reject("counter_overflow"); return false; }
        value += amount; return true;
    }
    void begin(int64_t l, Format f, const int32_t* residency, bool serialized) noexcept {
        if (failure) return;
        if (!serialized || active != -1 || l != retired || l < 0 || l >= layers || !residency ||
            f.gu_type < 0 || f.d_type < 0 || !f.gu_row || !f.d_row || !f.bytes ||
            f.gu_row > UINT64_MAX/1280 || f.d_row > UINT64_MAX/2560 ||
            f.up_off > f.bytes || f.down_off > f.bytes || f.down_off < f.up_off ||
            f.gu_row > f.up_off/640 || 640*f.gu_row > f.down_off-f.up_off ||
            2560*f.d_row > f.bytes-f.down_off) {
            reject("layer_metadata_or_serialization"); return;
        }
        for (int e=0; e<experts; ++e) if (residency[e] != e) {
            reject("configured_residency"); return;
        }
        active = int(l); record[size_t(l)].format = f;
        gu_pending.fill(0); down_pending.fill(0);
    }
    void counts(int64_t l, int64_t actual_t, const int32_t* cnt, size_t size) noexcept {
        if (failure) return;
        if (l != active || active < 0 || !cnt || size != experts || actual_t <= 0 ||
            actual_t > chunk) { reject("count_geometry"); return; }
        auto& r = record[size_t(active)];
        // Each layer must observe the actual fixed chunk sequence including its tail.
        const uint64_t left = uint64_t(n)-r.tokens;
        if (r.tokens >= uint64_t(n) || uint64_t(actual_t) != (left < uint64_t(chunk) ? left : uint64_t(chunk))) {
            reject("chunk_sequence"); return;
        }
        uint64_t rows=0;
        for (int e=0; e<experts; ++e) {
            if (cnt[e] < 0 || cnt[e] > actual_t) { reject("ne_range"); return; }
            rows += uint64_t(cnt[e]);
        }
        if (rows != uint64_t(actual_t)*topk) { reject("route_conservation"); return; }
        add(r.chunks,1); add(r.tokens,uint64_t(actual_t));
        for (int e=0; e<experts; ++e) {
            const size_t ne=size_t(cnt[e]); add(r.hist[ne],1);
            if (ne) { add(gu_pending[ne],1); add(down_pending[ne],1); }
        }
    }
    void issue(bool gu, int64_t l, int64_t ne, int expert, int cache_slot, int ring_slot) noexcept {
        if (failure) return;
        if (l != active || active < 0 || ne <= 0 || ne > chunk || expert < 0 ||
            expert >= experts || cache_slot != expert || ring_slot != -1) {
            reject("issuer_geometry_or_residency"); return;
        }
        auto& pending = gu ? gu_pending : down_pending;
        if (!pending[size_t(ne)]) { reject("issuer_ne_mismatch"); return; }
        --pending[size_t(ne)];
        auto& r=record[size_t(active)];
        add(gu ? r.gu_calls : r.down_calls,1);
        add(gu ? r.gu_rows : r.down_rows,uint64_t(ne));
        add(gu ? r.configured_resident_gu : r.configured_resident_down,1);
    }
    void end(int64_t l) noexcept {
        if (failure) return;
        if (l != active || active < 0) { reject("retire_order"); return; }
        auto& r=record[size_t(active)];
        uint64_t mass=0, rows=0, nonzero=0;
        for (size_t ne=0; ne<bins; ++ne) {
            if (gu_pending[ne] || down_pending[ne]) { reject("missing_issuer_call"); return; }
            add(mass,r.hist[ne]);
            if (ne && r.hist[ne] > UINT64_MAX/ne) { reject("counter_overflow"); return; }
            add(rows,uint64_t(ne)*r.hist[ne]);
            if (ne) add(nonzero,r.hist[ne]);
        }
        if (failure) return;
        if (r.tokens != uint64_t(n) || r.chunks != uint64_t(1+(n-1)/chunk) ||
            mass != uint64_t(experts)*r.chunks || rows != uint64_t(n)*topk ||
            nonzero != r.gu_calls || nonzero != r.down_calls || rows != r.gu_rows ||
            rows != r.down_rows || r.configured_resident_gu != nonzero ||
            r.configured_resident_down != nonzero) { reject("layer_conservation"); return; }
        active=-1; ++retired;
    }
    // Two passes: preflight serialized size before any layer record is written.
    // Only post-request formatting; each write is <=511 bytes. A failed writer
    // cannot emit a valid completion token. Collector requires all 48 records.
    struct Writer {
        FILE* out; size_t bytes=0; bool ok=true;
        void put(const char* fmt, ...) noexcept {
            if (!ok) return;
            char text[512]; va_list args; va_start(args,fmt);
            const int count=std::vsnprintf(text,sizeof(text),fmt,args); va_end(args);
            if (count < 0 || size_t(count) >= sizeof(text) || size_t(count)>output_cap-bytes) { ok=false; return; }
            bytes+=size_t(count);
            if (out && std::fwrite(text,1,size_t(count),out)!=size_t(count)) ok=false;
        }
    };
    void serialize(Writer& w) const noexcept {
        w.put("strata lm census: {\"version\":1,\"tokens\":%lld,\"position\":%lld,\"configured_chunk\":%lld,\"layers\":[",(long long)n,(long long)pos,(long long)chunk);
        for (int l=0;l<layers;++l) {
            const auto& r=record[size_t(l)]; const auto& f=r.format;
            w.put("%s{\"layer\":%d,\"gu_type\":%d,\"down_type\":%d,\"gu_row\":%llu,\"down_row\":%llu,\"up_offset\":%llu,\"down_offset\":%llu,\"blob_bytes\":%llu,",l?",":"",l,f.gu_type,f.d_type,(unsigned long long)f.gu_row,(unsigned long long)f.d_row,(unsigned long long)f.up_off,(unsigned long long)f.down_off,(unsigned long long)f.bytes);
            w.put("\"gu_tensor_bytes\":%llu,\"down_tensor_bytes\":%llu,\"configured_slots\":512,\"configured_resident\":512,\"chunks\":%llu,\"tokens\":%llu,",(unsigned long long)(1280*f.gu_row),(unsigned long long)(2560*f.d_row),(unsigned long long)r.chunks,(unsigned long long)r.tokens);
            w.put("\"gu_calls\":%llu,\"down_calls\":%llu,\"gu_rows\":%llu,\"down_rows\":%llu,\"gu_logical_nk\":[1280,2560],\"down_logical_nk\":[2560,640],\"histogram_columns\":[\"ne\",\"routed_experts\",\"gu_calls\",\"down_calls\"],\"histogram\":[",(unsigned long long)r.gu_calls,(unsigned long long)r.down_calls,(unsigned long long)r.gu_rows,(unsigned long long)r.down_rows);
            bool first=true;
            for(size_t ne=0;ne<bins;++ne) if(r.hist[ne]) {
                w.put("%s[%u,%llu,%llu,%llu]",first?"":",",unsigned(ne),(unsigned long long)r.hist[ne],
                    (unsigned long long)(ne?r.hist[ne]:0),(unsigned long long)(ne?r.hist[ne]:0)); first=false;
            }
            w.put("]}");
        }
        w.put("],\"issuer_exact_ne_verified\":true,\"hardware_cache_residency_measured\":false,\"complete\":true}\n");
    }
    void complete(FILE* out=stderr) noexcept {
        if (finalized) return;
        if (active != -1 || retired != layers) reject("request_incomplete");
        Writer preflight{nullptr}; if (!failure) serialize(preflight);
        if (!preflight.ok) reject("output_overflow");
        if (failure) {
            std::fprintf(out,"strata lm census: {\"version\":1,\"complete\":false,\"reason\":\"%s\"}\n",failure);
        } else {
            Writer writer{out}; serialize(writer);
            if (!writer.ok || std::fflush(out)!=0) reject("output_io_incomplete");
        }
        finalized=true;
    }
};
static_assert(sizeof(LayerMajorShapeCensus) <= (4u<<20), "bounded observer state");
// This lease must be declared after its owner and before Restore/checked queue
// retirement. No callback captures owner. Restore occurs on every unwind.
struct LayerMajorShapeBorrow {
    LayerMajorShapeCensus*& target;
    LayerMajorShapeCensus* previous;
    LayerMajorShapeBorrow(LayerMajorShapeCensus*& slot, LayerMajorShapeCensus* value) noexcept
        : target(slot), previous(slot) { target=value; }
    ~LayerMajorShapeBorrow() noexcept { target=previous; }
};
} // namespace strata::prefill::detail
