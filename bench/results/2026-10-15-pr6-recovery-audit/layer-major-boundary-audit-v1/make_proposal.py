from pathlib import Path
import difflib,hashlib,json
root=Path('/home/yayoi/orca/workspaces/Strata/sycl')
out=root/'bench/results/2026-10-15-pr6-recovery-audit/layer-major-boundary-audit-v1'
header=r'''#pragma once
// Diagnostic only. No GPU ownership: caller's checked queue lease outlives us.
#include "strata/kernels/f16_bits.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
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
class LayerMajorBoundaryAudit {
    using File = std::unique_ptr<FILE, int(*)(FILE*)>;
    File log_{nullptr, std::fclose}, raw_{nullptr, std::fclose};
    std::vector<int64_t> rows_;
    std::vector<float> expected_, scratch_;
    std::vector<uint16_t> packed_, actual_packed_;
    std::vector<int64_t> generation_;
    int64_t n_, pos_, width_, layers_, chunk_;
    bool half_, mismatch_ = false;
    uint64_t text_bytes_ = 0, raw_bytes_ = 0, outputs_ = 0, stores_ = 0, inputs_ = 0;
    static constexpr uint64_t text_limit = 8ull << 20, raw_limit = 48ull << 20;
    static File exclusive(const std::string& path) {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) throw std::runtime_error("boundary audit exclusive create failed: " + path);
        FILE* f = ::fdopen(fd, "wb");
        if (!f) { ::close(fd); throw std::runtime_error("boundary audit fdopen failed"); }
        return File(f, std::fclose);
    }
    static uint64_t hash(const void* p, size_t bytes) {
        uint64_t h = 14695981039346656037ull;
        const auto* b = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < bytes; ++i) { h ^= b[i]; h *= 1099511628211ull; }
        return h; // FNV-1a64 is a locator, never proof of cross-run equality.
    }
    void line(const char* s) {
        const size_t bytes = std::strlen(s);
        if (bytes > text_limit - text_bytes_) throw std::runtime_error("boundary audit text budget: incomplete");
        if (std::fwrite(s, 1, bytes, log_.get()) != bytes || std::fflush(log_.get()))
            throw std::runtime_error("boundary audit log I/O: incomplete");
        text_bytes_ += bytes;
    }
    void record(const char* phase, int64_t layer, int64_t row, int64_t start, int64_t count,
                bool device, const void* actual, const void* expected, size_t bytes,
                const float* af = nullptr, const float* ef = nullptr) {
        const auto* a = static_cast<const unsigned char*>(actual);
        const auto* e = static_cast<const unsigned char*>(expected);
        const bool equal = !expected || std::memcmp(a, e, bytes) == 0;
        size_t first = bytes;
        if (!equal) for (size_t i = 0; i < bytes; ++i) if (a[i] != e[i]) { first = i; break; }
        double maximum = 0; uint64_t nonfinite = 0;
        if (af && ef) for (int64_t i = 0; i < width_; ++i) {
            if (!std::isfinite(af[i]) || !std::isfinite(ef[i])) ++nonfinite;
            else maximum = std::max(maximum, std::abs(double(af[i]) - double(ef[i])));
        }
        mismatch_ |= !equal;
        char buf[1024];
        const int len = std::snprintf(buf, sizeof(buf),
            "phase=%s layer=%lld chunk=%lld start=%lld count=%lld relative=%lld absolute=%lld "
            "storage=%s source=%s width=%lld bytes=%zu fnv64=%016llx expected_fnv64=%016llx "
            "memcmp=%s first_byte=%lld max_abs=%.17g nonfinite=%llu\n",
            phase, (long long)layer, (long long)(start / chunk_), (long long)start, (long long)count,
            (long long)row, (long long)(pos_ + row), half_ ? "fp16" : "fp32", device ? "GPU" : "RAM",
            (long long)width_, bytes, (unsigned long long)hash(a, bytes),
            (unsigned long long)(expected ? hash(e, bytes) : 0), expected ? (equal ? "equal" : "different") : "NA",
            equal ? -1ll : (long long)first, maximum, (unsigned long long)nonfinite);
        if (len < 0 || size_t(len) >= sizeof(buf)) throw std::runtime_error("boundary audit record overflow");
        line(buf);
    }
public:
    // copy(dst,src,bytes,device_source) must checked-drain the SAME compute queue.
    using Copy = std::function<bool(void*, const void*, size_t, bool)>;
    LayerMajorBoundaryAudit(int64_t n, int64_t pos, int64_t width, int64_t layers,
                           int64_t chunk, bool half): n_(n), pos_(pos), width_(width), layers_(layers), chunk_(chunk), half_(half) {
        if (n <= 0 || pos < 0 || width != 10240 || layers != 48 || chunk <= 0)
            throw std::invalid_argument("boundary audit unsupported shape");
        const char* csv = std::getenv("STRATA_PREFILL_BOUNDARY_ROWS");
        if (!csv) csv = "0,1,63,64,255,256,257,4095,4096,8191,8192,16383,16384,24575,24576,32767,32768,34600,34601";
        const char* mode = std::getenv("STRATA_PREFILL_BOUNDARY_ROWS_ABSOLUTE");
        const bool absolute = mode && std::strcmp(mode, "1") == 0;
        const char* p = csv;
        while (*p) {
            if (rows_.size() == 64) throw std::invalid_argument("boundary audit max 64 rows");
            const char* end = std::strchr(p, ','); if (!end) end = p + std::strlen(p);
            int64_t row = -1;
            const auto value = std::from_chars(p, end, row);
            if (value.ec != std::errc{} || value.ptr != end || row < 0 || (absolute && row < pos))
                throw std::invalid_argument("boundary audit invalid row CSV");
            if (absolute) row -= pos;
            if (row >= n || std::find(rows_.begin(), rows_.end(), row) != rows_.end())
                throw std::invalid_argument("boundary audit duplicate/out-of-range row");
            rows_.push_back(row);
            if (!*end) break;
            p = end + 1; if (!*p) throw std::invalid_argument("boundary audit trailing comma");
        }
        if (rows_.empty()) throw std::invalid_argument("boundary audit empty selection");
        // Raw output rows are a finite, exact cross-run comparison fixture.
        if (uint64_t(rows_.size()) * uint64_t(layers) * uint64_t(width) * 4 > raw_limit)
            throw std::invalid_argument("boundary audit selected raw output exceeds 48 MiB");
        expected_.resize(rows_.size() * size_t(width)); scratch_.resize(size_t(width));
        packed_.resize(size_t(width)); actual_packed_.resize(size_t(width));
        generation_.assign(rows_.size(), -1);
        const char* path = std::getenv("STRATA_PREFILL_BOUNDARY_PATH");
        if (!path || !*path) throw std::invalid_argument("boundary audit requires output path");
        log_ = exclusive(path); raw_ = exclusive(std::string(path) + ".rows.f32");
        char shape[256];
        std::snprintf(shape, sizeof(shape), "tokens=%lld pos0=%lld width=%lld layers=%lld chunk=%lld selected=%zu\n",
                      (long long)n_, (long long)pos_, (long long)width_, (long long)layers_, (long long)chunk_, rows_.size());
        line(shape);
        line("boundary_audit_v1 status=incomplete raw_order=layer_then_callback_then_CSV_selection raw_dtype=fp32 hash=FNV1a64_noncryptographic final_layer=47_not_stored first_input=embedding_not_audited text_limit=8388608 raw_limit=50331648\n");
    }
    bool source(int64_t layer, int64_t start, int64_t count, const float* r, const Copy& copy) {
        if (layer < 0 || layer >= layers_ || start < 0 || count <= 0 || count > chunk_ || start % chunk_ || start > n_ - count)
            throw std::invalid_argument("boundary audit source shape");
        char chunk_info[256];
        std::snprintf(chunk_info, sizeof(chunk_info), "phase=output_chunk layer=%lld start=%lld count=%lld width=%lld\n",
                      (long long)layer, (long long)start, (long long)count, (long long)width_);
        line(chunk_info);
        for (size_t s = 0; s < rows_.size(); ++s) {
            const int64_t row = rows_[s]; if (row < start || row >= start + count) continue;
            if (!copy(scratch_.data(), r + size_t(row-start)*width_, size_t(width_)*4, true)) return false;
            record("output_before_store", layer, row, start, count, true, scratch_.data(), nullptr, size_t(width_)*4);
            const size_t bytes = size_t(width_) * 4;
            if (bytes > raw_limit - raw_bytes_ || std::fwrite(scratch_.data(), 1, bytes, raw_.get()) != bytes)
                throw std::runtime_error("boundary audit raw I/O/budget: incomplete");
            raw_bytes_ += bytes; ++outputs_;
            float* e = expected_.data() + s*size_t(width_);
            if (layer + 1 < layers_) {
                if (generation_[s] != -1) throw std::runtime_error("boundary audit previous input not consumed");
                for (int64_t i = 0; i < width_; ++i)
                    e[i] = half_ ? strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(scratch_[size_t(i)])) : scratch_[size_t(i)];
                generation_[s] = layer;
            }
        }
        return true;
    }
    bool stored(int64_t layer, int64_t start, int64_t count, const void* store, bool device, const Copy& copy) {
        for (size_t s = 0; s < rows_.size(); ++s) {
            const int64_t row = rows_[s]; if (row < start || row >= start + count) continue;
            if (generation_[s] != layer) throw std::runtime_error("boundary audit store sequence");
            const float* e = expected_.data() + s*size_t(width_);
            if (half_) {
                for (int64_t i = 0; i < width_; ++i) packed_[size_t(i)] = strata::kernels::f16_from_f32(e[i]);
                if (!copy(actual_packed_.data(), static_cast<const uint8_t*>(store) + size_t(row-start)*size_t(width_)*2,
                          size_t(width_)*2, device)) return false;
                record("stored_RNE_packed_bits", layer, row, start, count, device,
                       actual_packed_.data(), packed_.data(), size_t(width_)*2);
            } else {
                if (!copy(scratch_.data(), static_cast<const float*>(store) + size_t(row-start)*width_, size_t(width_)*4, device)) return false;
                record("stored", layer, row, start, count, device, scratch_.data(), e, size_t(width_)*4, scratch_.data(), e);
            }
            ++stores_;
        }
        return true;
    }
    bool input(int64_t layer, int64_t start, int64_t count, const float* r, bool device, const Copy& copy) {
        for (size_t s = 0; s < rows_.size(); ++s) {
            const int64_t row = rows_[s]; if (row < start || row >= start + count) continue;
            if (generation_[s] != layer-1) throw std::runtime_error("boundary audit input sequence");
            const float* e = expected_.data() + s*size_t(width_);
            if (!copy(scratch_.data(), r + size_t(row-start)*width_, size_t(width_)*4, true)) return false;
            record(half_ ? "next_input_decoded_expected" : "next_input", layer, row, start, count,
                   device, scratch_.data(), e, size_t(width_)*4, scratch_.data(), e);
            generation_[s] = -1; ++inputs_;
        }
        return true;
    }
    void complete() {
        const uint64_t selected = uint64_t(rows_.size());
        if (outputs_ != selected*uint64_t(layers_) || stores_ != selected*uint64_t(layers_-1) || inputs_ != stores_ ||
            std::find_if(generation_.begin(), generation_.end(), [](int64_t l){return l != -1;}) != generation_.end() ||
            std::fflush(raw_.get())) throw std::runtime_error("boundary audit incomplete counts/raw flush");
        // Completion describes coverage, not a correctness or original-gate PASS.
        line(mismatch_ ? "status=complete boundary_memcmp=different\n" : "status=complete boundary_memcmp=equal\n");
    }
};
} // namespace strata::prefill::detail
'''
paths=['src/prefill/prefill.cpp','src/prefill/layer_major.inc']
base={p:(root/p).read_text() for p in paths}; mod=dict(base)
s=mod[paths[0]]
s=s.replace('#include "strata/prefill/layer_major_plan.hpp"','#include "strata/prefill/layer_major_plan.hpp"\n#include "strata/prefill/layer_major_boundary_audit.hpp"')
s=s.replace('    bool layer_major_active = false;', '    bool layer_major_active = false;\n    detail::LayerMajorBoundaryAudit* layer_major_audit = nullptr;\n    detail::LayerMajorBoundaryAudit::Copy layer_major_audit_copy;')
needle='''            } else if (cudaMemcpyAsync(m.R, hand_in_'''
insert='''                if (m.layer_major_audit && !m.layer_major_audit->input(LB, c0, T, m.R, span.device,
                                                                      m.layer_major_audit_copy)) {
                    err = "prefill boundary audit: input copy/drain failed"; return false;
                }
'''
assert needle in s;s=s.replace(needle,insert+needle); mod[paths[0]]=s
s=mod[paths[1]]
s=s.replace('    int64_t logical_chunks = 0, layer = 0;', '''    // Declare before Restore/lease: diagnostic DMA host owners survive checked retirement.
    std::unique_ptr<detail::LayerMajorBoundaryAudit> audit;
    const char* audit_switch = std::getenv("STRATA_PREFILL_BOUNDARY_AUDIT");
    if (audit_switch && std::strcmp(audit_switch, "1") == 0)
        audit.reset(new detail::LayerMajorBoundaryAudit(n, pos0, D, g.n_layers, m.T,
            plan.dtype == detail::LayerMajorResidualDtype::fp16));
    int64_t logical_chunks = 0, layer = 0;''')
s=s.replace('        bool active; const void* gpu_input;', '''        detail::LayerMajorBoundaryAudit* audit;
        detail::LayerMajorBoundaryAudit::Copy audit_copy;
        bool active; const void* gpu_input;''')
s=s.replace('            m.layer_major_gpu_input = gpu_input;', '''            m.layer_major_audit = audit; m.layer_major_audit_copy = std::move(audit_copy);
            m.layer_major_gpu_input = gpu_input;''')
s=s.replace('              m.layer_major_active, m.layer_major_gpu_input,', '              m.layer_major_audit, m.layer_major_audit_copy,\n              m.layer_major_active, m.layer_major_gpu_input,')
needle='    if (!drain()) { err = "prefill layer-major: entry queue drain failed"; return false; }'
assert needle in s
s=s.replace(needle,needle+'''
    if (audit) {
        m.layer_major_audit = audit.get();
        m.layer_major_audit_copy = [&](void* dst, const void* src, size_t bytes, bool device) {
            if (!device) { std::memcpy(dst, src, bytes); return true; }
            const auto status = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, m.cs);
            const bool retired = drain(); // check even when submission reported an error
            return status == cudaSuccess && retired;
        };
    }''')
needle='            cursor.accept(row, count);'
s=s.replace(needle,needle+'''
            if (audit && !audit->source(layer, row, count, residual, m.layer_major_audit_copy)) {
                error = "prefill boundary audit: source copy/drain failed"; return false;
            }''')
needle='''            return true;
        };
        if (!run_impl'''
assert needle in s
s=s.replace(needle,'''            if (audit && !audit->stored(layer, row, count, destination, span.device,
                                         m.layer_major_audit_copy)) {
                error = "prefill boundary audit: stored copy/drain failed"; return false;
            }
'''+needle)
s=s.replace('    lease.retire();','    if (audit) audit->complete();\n    lease.retire();')
mod[paths[1]]=s
hpath='include/strata/prefill/layer_major_boundary_audit.hpp';base[hpath]='';mod[hpath]=header
patch=''.join(''.join(difflib.unified_diff(base[p].splitlines(True),mod[p].splitlines(True),fromfile='a/'+p if base[p] else '/dev/null',tofile='b/'+p)) for p in mod)
(out/'source.diff').write_text(patch)
pins={'status':'PROPOSAL_ONLY_UNAPPLIED_UNTESTED','canonical_source_base':'current root 5fbc (parent-specified; no git command used)','base_files':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},'context_files':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['include/strata/prefill/layer_major_plan.hpp','include/strata/kernels/f16_bits.hpp']},'source_diff_sha256':hashlib.sha256(patch.encode()).hexdigest()}
(out/'pins.json').write_text(json.dumps(pins,indent=2)+'\n')
