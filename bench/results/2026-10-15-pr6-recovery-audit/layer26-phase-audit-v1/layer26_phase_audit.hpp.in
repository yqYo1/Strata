#pragma once
// Finite diagnostic only. The outer checked queue lease owns our D2H destinations.
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>
namespace strata::prefill::detail {
class Layer26PhaseAudit {
    using File = std::unique_ptr<FILE, int(*)(FILE*)>;
    File log_{nullptr, std::fclose}, raw_{nullptr, std::fclose};
    std::array<int64_t, 6> rows_{};
    std::array<unsigned, 6> next_{};
    // No per-copy local/vector destination: this storage survives failed drains.
    alignas(64) std::array<unsigned char, 40960> scratch_{};
    std::array<int32_t, 70> grouping_{};
    size_t selected_ = 0;
    int64_t n_, pos_, chunk_;
    bool low_, completed_ = false;
    uint64_t text_bytes_ = 0, raw_bytes_ = 0, records_ = 0;
    static File exclusive(const std::string& path) {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) throw std::runtime_error("phase audit exclusive create failed: " + path);
        FILE* f = ::fdopen(fd, "wb");
        if (!f) { ::close(fd); throw std::runtime_error("phase audit fdopen failed"); }
        return File(f, std::fclose);
    }
    void line(const char* s) {
        const size_t bytes = std::strlen(s);
        if (bytes > text_limit - text_bytes_) throw std::runtime_error("phase audit text budget: incomplete");
        if (std::fwrite(s, 1, bytes, log_.get()) != bytes || std::fflush(log_.get()))
            throw std::runtime_error("phase audit text I/O: incomplete");
        text_bytes_ += bytes;
    }
public:
    using Copy = std::function<bool(void*, const void*, size_t, bool)>;
    enum Tensor : unsigned { gdn_input, gdn_out, attn_R, ffn_input_h, ffn_input_bf, ffn_input_low,
                             router_logits, router_ids, router_weights, grouping, moe_out, final_R, tensor_count };
    static constexpr uint64_t raw_limit = 8ull << 20, text_limit = 1ull << 20;
    // Separate reservation, not part of boundary audit's4MiB/48MiB/8MiB contract.
    static constexpr uint64_t host_reservation = 1ull << 20;
    static const char* name(Tensor t) {
        static constexpr const char* names[] = {"gdn_input_mixed_h", "gdn_out_bo", "attn_write_R",
            "ffn_mixed_h", "ffn_mixed_bf", "ffn_mixed_bf_low", "router_logits", "router_ids",
            "router_weights", "host_grouping", "moe_combine_bo", "final_R"};
        if (unsigned(t) >= tensor_count) throw std::invalid_argument("phase audit tensor");
        return names[unsigned(t)];
    }
    static size_t elements(Tensor t) {
        switch (t) {
        case attn_R: case final_R: return 10240;
        case router_logits: return 512;
        case router_ids: case router_weights: return 10;
        case grouping: return 70; //10 entries x {expert,slot,src,cnt,off,order_index,group_local}
        default: (void)name(t); return 2560;
        }
    }
    static size_t item_bytes(Tensor t) {
        return t == gdn_input || t == ffn_input_h || t == ffn_input_bf || t == ffn_input_low ? 2 : 4;
    }
    static const char* dtype(Tensor t) {
        if (t == gdn_input || t == ffn_input_h) return "fp16";
        if (t == ffn_input_bf || t == ffn_input_low) return "bf16";
        if (t == router_ids || t == grouping) return "int32";
        return "fp32";
    }
private:
    void shape(int64_t start, int64_t count) const {
        if (start < 0 || start % chunk_ || count <= 0 || count > chunk_ || start > n_ - count ||
            count != std::min(chunk_, n_-start)) throw std::invalid_argument("phase audit chunk");
    }
    void emit(Tensor t, size_t s, int64_t start, int64_t count, uint64_t source_offset, bool device) {
        if (next_[s] != unsigned(t)) throw std::runtime_error("phase audit duplicate/out-of-order tensor");
        const size_t bytes = elements(t) * item_bytes(t);
        if (bytes > raw_limit - raw_bytes_ || std::fwrite(scratch_.data(), 1, bytes, raw_.get()) != bytes)
            throw std::runtime_error("phase audit raw budget/I/O: incomplete");
        char buf[768];
        const int len = std::snprintf(buf, sizeof(buf),
            "record=%llu tensor=%s layer=26 row=%lld absolute=%lld chunk=%lld start=%lld count=%lld "
            "dtype=%s elements=%zu bytes=%zu offset=%llu source_offset=%llu source=%s memcmp=NA\n",
            (unsigned long long)records_, name(t), (long long)rows_[s], (long long)(pos_+rows_[s]),
            (long long)(start/chunk_), (long long)start, (long long)count, dtype(t), elements(t), bytes,
            (unsigned long long)raw_bytes_, (unsigned long long)source_offset, device ? "GPU" : "host_selected");
        if (len < 0 || size_t(len) >= sizeof(buf)) throw std::runtime_error("phase audit metadata overflow");
        line(buf);
        raw_bytes_ += bytes; ++records_; ++next_[s];
        if (!low_ && next_[s] == ffn_input_low) ++next_[s];
    }
public:
    Layer26PhaseAudit(int64_t n, int64_t pos, int64_t chunk, bool low):
        n_(n), pos_(pos), chunk_(chunk), low_(low) {
        static_assert(sizeof(int32_t) == 4 && sizeof(float) == 4);
        if (n <= 0 || n > 262144 || pos < 0 || pos > 262144 || n > 262144-pos || chunk <= 0 || chunk > 8192)
            throw std::invalid_argument("phase audit unsupported shape");
        const char* csv = std::getenv("STRATA_PREFILL_PHASE_ROWS");
        if (!csv) csv = "16384,24575,24576,32767,32768,34601";
        const char* p = csv;
        while (*p) {
            if (selected_ == rows_.size()) throw std::invalid_argument("phase audit max6 rows");
            const char* end = std::strchr(p, ','); if (!end) end = p + std::strlen(p);
            int64_t row = -1;
            const auto parsed = std::from_chars(p, end, row);
            if (parsed.ec != std::errc{} || parsed.ptr != end || row < 0 || row >= n ||
                std::find(rows_.begin(), rows_.begin()+selected_, row) != rows_.begin()+selected_)
                throw std::invalid_argument("phase audit invalid/duplicate/out-of-range row");
            rows_[selected_++] = row;
            if (!*end) break;
            p = end+1; if (!*p) throw std::invalid_argument("phase audit trailing comma");
        }
        if (!selected_) throw std::invalid_argument("phase audit empty rows");
        const char* path = std::getenv("STRATA_PREFILL_PHASE_PATH");
        if (!path || !*path || std::strlen(path)>4095) throw std::invalid_argument("phase audit requires bounded output path");
        log_ = exclusive(path); raw_ = exclusive(std::string(path)+".raw");
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "phase_audit_v1 status=incomplete layer=26 tokens=%lld pos0=%lld chunk=%lld selected=%zu low=%d "
            "raw_limit=8388608 text_limit=1048576 host_reservation=1048576 byteorder=%s "
            "grouping_fields=expert,slot,src,cnt,off,order_index,group_local\n",
            (long long)n_, (long long)pos_, (long long)chunk_, selected_, int(low_),
            [] { const uint16_t one=1; return *reinterpret_cast<const unsigned char*>(&one) ? "little" : "big"; }());
        line(buf);
        for (size_t s = 0; s < selected_; ++s) {
            std::snprintf(buf, sizeof(buf), "selection=%zu row=%lld\n", s, (long long)rows_[s]); line(buf);
        }
    }
    // Called immediately after each producer, before any shared-region overwrite.
    // copy must checked-drain the SAME compute queue, including submission errors.
    bool capture(Tensor t, int64_t layer, int64_t start, int64_t count, const void* base, const Copy& copy) {
        if (layer != 26) return true;
        shape(start, count); (void)name(t);
        if (t == grouping || (t == ffn_input_low && !low_)) throw std::invalid_argument("phase audit capture kind");
        const size_t bytes = elements(t)*item_bytes(t);
        for (size_t s = 0; s < selected_; ++s) {
            if (rows_[s] < start || rows_[s] >= start+count) continue;
            if (!base || !copy || next_[s] != unsigned(t)) throw std::runtime_error("phase audit source/sequence");
            const uint64_t offset = uint64_t(rows_[s]-start)*bytes;
            if (!copy(scratch_.data(), static_cast<const unsigned char*>(base)+offset, bytes, true)) return false;
            emit(t, s, start, count, offset, true);
        }
        return true;
    }
    // Existing host grouping remains untouched. Only selected ten-entry ledgers are serialized.
    bool capture_grouping(int64_t layer, int64_t start, int64_t count,
                          const int32_t* ids, const int32_t* slots, const int32_t* src,
                          const std::vector<int32_t>& cnt, const std::vector<int32_t>& off,
                          const std::vector<int32_t>& order) {
        if (layer != 26) return true;
        shape(start, count);
        if (!ids || !slots || !src || cnt.size()!=512 || off.size()!=513 || order.size()>512)
            throw std::invalid_argument("phase audit host grouping shape");
        for (size_t s = 0; s < selected_; ++s) {
            const int64_t row = rows_[s]-start; if (row < 0 || row >= count) continue;
            for (int k=0; k<10; ++k) {
                const size_t i = size_t(row)*10+size_t(k);
                const int32_t e=ids[i], slot=slots[i];
                if (e<0 || e>=512 || slot<0 || slot>=count*10 || cnt[size_t(e)]<=0 || cnt[size_t(e)]>count ||
                    off[size_t(e)]<0 || slot<off[size_t(e)] || slot-off[size_t(e)]>=cnt[size_t(e)] || src[slot]!=row)
                    throw std::runtime_error("phase audit host grouping bounds");
                for (int j=0; j<k; ++j) if (grouping_[size_t(j)*7]==e)
                    throw std::runtime_error("phase audit duplicate top10 ID");
                const auto it=std::find(order.begin(), order.end(), e);
                if (it==order.end()) throw std::runtime_error("phase audit expert absent from order");
                const int32_t fields[] = {e,slot,src[slot],cnt[size_t(e)],off[size_t(e)],
                    int32_t(it-order.begin()),slot-off[size_t(e)]};
                std::copy(fields, fields+7, grouping_.begin()+size_t(k)*7);
            }
            std::memcpy(scratch_.data(), grouping_.data(), elements(grouping)*4);
            emit(grouping, s, start, count, uint64_t(row)*10*4, false);
        }
        return true;
    }
    void complete() {
        const uint64_t expected = selected_*(tensor_count-(low_ ? 0 : 1));
        if (completed_ || records_!=expected || std::find_if(next_.begin(), next_.begin()+selected_,
                [](unsigned t){ return t!=tensor_count; })!=next_.begin()+selected_ || std::fflush(raw_.get()))
            throw std::runtime_error("phase audit incomplete coverage/raw flush");
        char buf[256];
        std::snprintf(buf, sizeof(buf), "status=complete coverage=complete records=%llu raw_bytes=%llu "
            "memcmp=not_compared adoption=unqualified\n", (unsigned long long)records_, (unsigned long long)raw_bytes_);
        line(buf); completed_ = true;
    }
};
} // namespace strata::prefill::detail
