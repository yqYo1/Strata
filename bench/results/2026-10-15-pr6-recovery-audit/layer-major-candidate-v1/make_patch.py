#!/usr/bin/env python3
"""Materialize a review patch in this directory; never edit product files."""
from pathlib import Path
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
changes = {}
sources = {}

def read(name):
    content = (ROOT / name).read_text(encoding="utf-8")
    sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    changes[name] = [content, content]
    return content

def replace(s, before, after):
    if s.count(before) != 1:
        raise ValueError(f"expected one replacement, found {s.count(before)}: {before[:100]}")
    return s.replace(before, after, 1)

name = "src/prefill/prefill.cpp"
s = read(name)
s = replace(s, '#include "strata/prefill/draft_uploads.hpp"',
    '#include "strata/prefill/draft_uploads.hpp"\n#include "strata/prefill/layer_major_plan.hpp"\n#include "strata/core/conversation_memory.hpp"')
s = replace(s, '#include <future>', '#include <future>\n#include <limits>')
s = replace(s, '    bool compact_moe_fp16 = false;',
    '    bool compact_moe_fp16 = false;\n'
    '    // Opt-in layer-major lease; normal chunk-major execution leaves these empty.\n'
    '    bool layer_major_active = false;\n'
    '    const float* layer_major_gpu_input = nullptr;\n'
    '    detail::LayerMajorResidualPlan layer_major_residual;')
s = replace(s, 'bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {',
    '#include "layer_major.inc"\n\n'
    'bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {')
s = replace(s, '        ++stats_.chunks;',
    '        if (!m.layer_major_active || LE == g.n_layers) ++stats_.chunks;')
s = replace(s, '''        if (hand_in_ != nullptr) {
            if (cudaMemcpyAsync(m.R, hand_in_ + (size_t) c0 * D, (size_t) T * D * 4, cudaMemcpyHostToDevice, m.cs) !=
                cudaSuccess) {
                err = "prefill: the layer split's hand-off upload failed";
                return false;
            }
        }''', '''        const bool residual_input = hand_in_ != nullptr || m.layer_major_gpu_input != nullptr;
        if (residual_input) {
            if (m.layer_major_active) {
                const auto span = m.layer_major_residual.span(c0, T);
                const auto* base = reinterpret_cast<const uint8_t*>(span.device ? m.layer_major_gpu_input : hand_in_);
                if (!base || cudaMemcpyAsync(m.R, base + size_t(span.offset_bytes), size_t(span.bytes),
                    span.device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                    err = "prefill layer-major: residual input copy failed"; return false;
                }
            } else if (cudaMemcpyAsync(m.R, hand_in_ + (size_t) c0 * D, (size_t) T * D * 4,
                                      cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                err = "prefill: the layer split's hand-off upload failed";
                return false;
            }
        }''')
s = replace(s, '        bool batched = hand_in_ == nullptr && m.tok_dev != nullptr;',
    '        bool batched = !residual_input && m.tok_dev != nullptr;')
s = replace(s, '        for (int64_t t = 0; hand_in_ == nullptr && !batched && t < T; ++t) {',
    '        for (int64_t t = 0; !residual_input && !batched && t < T; ++t) {')
s = replace(s, '        if (hand_in_ == nullptr) gr_broadcast(m.emb, m.R, T, m.cs);',
    '        if (!residual_input) gr_broadcast(m.emb, m.R, T, m.cs);')
s = replace(s, '        const bool stream_all = m.ring > STAGE && T >= stream_all_min() && m.src != nullptr;',
    '        const bool stream_all = !m.layer_major_active && m.ring > STAGE && T >= stream_all_min() && m.src != nullptr;')
s = replace(s, '                        if (!stream_all) {\n                            std::vector<Stager::Job> js;',
    '                        if (!stream_all && !m.layer_major_active) {\n                            std::vector<Stager::Job> js;')
s = replace(s, '                        StagerDone stager_done{stream_all ? nullptr : m.stager.get()};',
    '                        StagerDone stager_done{stream_all || m.layer_major_active ? nullptr : m.stager.get()};')
s = replace(s, '''                            if (resident) return true;
                            const int sl = stage_next;''', '''                            if (resident) return true;
                            if (m.layer_major_active) {
                                err = "prefill layer-major: temporary residency map is incomplete"; return false;
                            }
                            const int sl = stage_next;''')
s = replace(s, '        stats_.tokens += T;',
    '        if (!m.layer_major_active || LE == g.n_layers) stats_.tokens += T;')
s = replace(s, '    stats_.ms_total += ms_since(t_start);',
    '    if (!m.layer_major_active) stats_.ms_total += ms_since(t_start);')
changes[name][1] = s

name = "include/strata/prefill/prefill.hpp"
s = read(name)
s = replace(s, 'private:\n    int64_t stage_lb_',
    'private:\n'
    '    bool run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);\n'
    '    bool run_layer_major(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);\n'
    '    int64_t stage_lb_')
changes[name][1] = s

name = "include/strata/core/expert_cache.hpp"
s = read(name)
s = replace(s, '    /// Plan v0.3 P6: slots of the given sizes, back to back',
    '    /// Allocation-only temporary cache: no memset or upload is submitted. The caller must\n'
    '    /// fill every consumed slot and prove completion before publishing its residency map.\n'
    '    bool open_uninitialized(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,\n'
    '                            std::string& err);\n'
    '    /// Plan v0.3 P6: slots of the given sizes, back to back')
s = replace(s, 'private:\n#if defined(STRATA_USE_HIP)',
    'private:\n'
    '    bool open_impl(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,\n'
    '                   std::string& err, bool initialize);\n'
    '#if defined(STRATA_USE_HIP)')
changes[name][1] = s

name = "src/core/expert_cache.cpp"
s = read(name)
s = replace(s, "#include <filesystem>", "#include <filesystem>\n#include <limits>")
s = replace(s, '''bool ExpertCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                       std::string& err) {
    close();''', '''bool ExpertCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                       std::string& err) {
    return open_impl(n_slots, n_layers, n_expert, blob_bytes, err, true);
}

bool ExpertCache::open_uninitialized(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                                     std::string& err) {
    const uint64_t limit = std::min<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max(),
                                             std::numeric_limits<int64_t>::max());
    if (n_slots <= 0 || n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0 ||
        uint64_t(n_slots) > limit / uint64_t(blob_bytes) ||
        uint64_t(n_layers) > limit / sizeof(int32_t) / uint64_t(n_expert)) {
        err = "ExpertCache: uninitialized cache geometry overflows addressable storage";
        return false;
    }
    return open_impl(n_slots, n_layers, n_expert, blob_bytes, err, false);
}

bool ExpertCache::open_impl(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                            std::string& err, bool initialize) {
    close();''')
s = replace(s, '    if (cudaMemset(base_, 0, (size_t) want) != cudaSuccess) {',
    '    if (initialize && cudaMemset(base_, 0, (size_t) want) != cudaSuccess) {')
changes[name][1] = s

name = "CMakeLists.txt"
s = read(name)
s = replace(s, 'if(STRATA_BUILD_CONVERSATION_TESTS)\n  enable_testing()\n  add_executable(conversation_cache_test',
    'option(STRATA_BUILD_PREFILL_HOST_TESTS "Build CPU-only prefill orchestration contract tests" OFF)\n'
    'if(STRATA_BUILD_PREFILL_HOST_TESTS)\n'
    '  enable_testing()\n'
    '  add_executable(prefill_layer_major_plan_test src/prefill/layer_major_plan_test.cpp)\n'
    '  target_include_directories(prefill_layer_major_plan_test PRIVATE include)\n'
    '  add_test(NAME prefill_layer_major_plan_test COMMAND prefill_layer_major_plan_test)\n'
    'endif()\n\n'
    'if(STRATA_BUILD_CONVERSATION_TESTS)\n  enable_testing()\n  add_executable(conversation_cache_test')
changes[name][1] = s

patch = "".join("".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
    fromfile="a/" + name, tofile="b/" + name)) for name, (before, after) in changes.items())
(HERE / "product.patch").write_text(patch, encoding="utf-8")
(HERE / "source-base-sha256.json").write_text(json.dumps(sources, indent=2) + "\n")
print(f"Wrote {len(patch.encode())} patch bytes to {HERE / 'product.patch'}; product files unchanged")
