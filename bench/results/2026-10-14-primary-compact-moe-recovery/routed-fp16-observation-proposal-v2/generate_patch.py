#!/usr/bin/env python3
"""Generate an unapplied diagnostic diff; never write product source."""
import difflib, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'src/prefill/prefill.cpp'
EXPECTED = '484764fccef4cf58909bfe27fb01934da19dbffaca7f7be2565424d36ab86578'
original = SOURCE.read_text()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED, 'source identity changed; rebase explicitly'
candidate = original

def replace(old, new):
    global candidate
    assert candidate.count(old) == 1, ('ambiguous/missing anchor', old[:100])
    candidate = candidate.replace(old, new)

observer = r'''// Default-OFF host-only observation. No events, queue queries or device copies.
struct RoutedBlobPayload {
    uint64_t bytes = 0, calls = 0, direct = 0, staged = 0, reused = 0;
    void add(size_t nb, bool pinned, bool live) {
        bytes += (uint64_t) nb; ++calls;
        if (pinned) ++direct; else ++staged;
        if (live) ++reused;
    }
    void merge(const RoutedBlobPayload& p) {
        bytes += p.bytes; calls += p.calls; direct += p.direct;
        staged += p.staged; reused += p.reused;
    }
};
struct RoutedFp16Observation {
    bool requested = false, on = false;
    const char* decline = "not_admitted";
    // Allocated only after admission; one count per exact ne, including zero.
    std::vector<uint64_t> hist, layer_chunks;
    RoutedBlobPayload payload;
    uint64_t chunks = 0, tails = 0, layer_count = 0, active = 0, zero = 0;
    uint64_t gu_calls = 0, d_calls = 0, gu_rows = 0, d_rows = 0, resident_calls = 0;
    int64_t chunk_min = 0, chunk_max = 0, chunk_last = 0, max_ne = 0;
    int64_t tokens = 0, position = 0, configured_chunk = 0, capacity = 0;
    int64_t streamed0 = 0, dma0 = 0, resident0 = 0;
    double host0 = 0;
    void reject(const char* why) { on = false; decline = why; }
    void chunk(int64_t T) {
        if (!on) return;
        if (T <= 0 || T > capacity) { reject("actual_chunk_out_of_scope"); return; }
        if (!chunks) chunk_min = T;
        chunk_min = std::min(chunk_min, T); chunk_max = std::max(chunk_max, T);
        chunk_last = T; ++chunks; if (T < configured_chunk) ++tails;
    }
    void counts(int64_t l, int64_t T, const std::vector<int32_t>& cnt) {
        if (!on) return;
        // Validate the entire authoritative host count table before indexing.
        int64_t rows = 0;
        if (cnt.size() != 512 || l < 0 || (size_t) l >= layer_chunks.size()) {
            reject("count_table_out_of_scope"); return;
        }
        for (int32_t ne : cnt) {
            if (ne < 0 || ne > T || ne > capacity) { reject("expert_ne_exceeds_actual_chunk"); return; }
            rows += ne;
        }
        if (rows != T * K) { reject("route_row_sum_mismatch"); return; }
        ++layer_count; ++layer_chunks[(size_t) l];
        for (int32_t ne : cnt) {
            ++hist[(size_t) ne];
            if (ne) ++active; else ++zero;
            max_ne = std::max<int64_t>(max_ne, ne);
        }
    }
    void gu(int64_t ne, int slot) {
        if (!on) return;
        ++gu_calls; gu_rows += (uint64_t) ne;
        if (slot < 0) ++resident_calls;
    }
    void down(int64_t ne) { if (on) { ++d_calls; d_rows += (uint64_t) ne; } }
    void emit(bool compact, bool timing, const double* phases, double wall,
              int64_t streamed, int64_t dma, int64_t resident, double host) {
        if (!requested) return;
        if (on) {
            uint64_t hist_total = 0, hist_rows = 0;
            for (size_t ne = 0; ne < hist.size(); ++ne) {
                hist_total += hist[ne]; hist_rows += (uint64_t) ne * hist[ne];
            }
            if (!layer_count || hist_total != 512ull * layer_count || zero + active != hist_total ||
                hist[0] != zero || hist_total - hist[0] != active || gu_calls != active || d_calls != active ||
                hist_rows != gu_rows || hist_rows != d_rows)
                reject("histogram_or_call_accounting_mismatch");
        }
        // Bounds: <=8193 histogram pairs and <=512 format rows; <1 MiB.
        std::string line = "strata routed fp16 observation: {\"version\":2,\"run_return\":\"success\",\"effective\":";
        line += on ? "true" : "false";
        if (!on) {
            line += ",\"reason\":\""; line += decline; line += "\"}\n";
            std::fputs(line.c_str(), stderr); return;
        }
        auto num = [&](const char* key, uint64_t v) {
            line += ",\""; line += key; line += "\":"; line += std::to_string(v);
        };
        num("input_tokens", tokens); num("position0", position);
        num("configured_chunk", configured_chunk); num("hist_capacity_ne", capacity);
        num("compact_fp16", compact); num("routing_k", K); num("experts", 512);
        num("chunks", chunks); num("tail_chunks", tails);
        num("actual_chunk_min", chunk_min); num("actual_chunk_max", chunk_max); num("actual_chunk_last", chunk_last);
        num("layer_chunks", layer_count); num("zero_experts", zero); num("active_experts", active);
        num("max_ne", max_ne); num("gu_calls", gu_calls); num("down_calls", d_calls);
        num("gu_rows", gu_rows); num("down_rows", d_rows); num("resident_expert_calls", resident_calls);
        num("shape_gu_flops", gu_rows * 2ull * 1280 * N);
        num("shape_down_flops", d_rows * 2ull * N * 640);
        num("blob_h2d_requested_submission_payload_bytes", payload.bytes); num("blob_h2d_attempted_host_copy_calls", payload.calls);
        num("blob_h2d_attempted_direct_pinned_copy_calls", payload.direct); num("blob_h2d_attempted_stager_copy_calls", payload.staged);
        num("blob_h2d_attempts_reusing_live_slots", payload.reused);
        num("blob_h2d_attempted_copy_groups", payload.calls); // FP16: one blob per submission, no grouped gather.
        num("stats_experts_streamed_delta", streamed - streamed0);
        num("stats_experts_dma_delta", dma - dma0);
        num("stats_experts_resident_delta", resident - resident0);
        line += ",\"hist_ne_count\":[";
        bool first = true;
        for (size_t ne = 0; ne < hist.size(); ++ne) if (hist[ne]) {
            if (!first) line += ','; first = false;
            line += '[' + std::to_string(ne) + ',' + std::to_string(hist[ne]) + ']';
        }
        line += "],\"native_format_layer_chunks\":[";
        const auto& lay = strata::kernels::cpu::expert_layout();
        first = true;
        for (size_t l = 0; l < layer_chunks.size(); ++l) if (layer_chunks[l]) {
            if (!first) line += ','; first = false;
            const auto& f = lay.fmt[l];
            // Fields: layer,chunks,GU type,D type,GU matrix bytes,D matrix bytes,blob bytes,up offset,down offset.
            line += '[' + std::to_string(l) + ',' + std::to_string(layer_chunks[l]) + ',';
            line += std::to_string(f.gu_type) + ',' + std::to_string(f.d_type) + ',';
            line += std::to_string(2ull * f.n_ff * f.gu_row) + ',';
            line += std::to_string((uint64_t) f.n_embd * f.d_row) + ',';
            line += std::to_string(lay.blob_bytes((int64_t) l)) + ',';
            line += std::to_string(f.up_off) + ',' + std::to_string(f.down_off) + ']';
        }
        char b[160];
        std::snprintf(b, sizeof b, "],\"wall_ms\":%.3f,\"stats_host_staging_ms_delta\":%.3f,\"pf_timing_enabled\":%s",
                      wall, host - host0, timing ? "true" : "false");
        line += b;
        line += ",\"pf_timeline_ms_by_phase\":[";
        if (timing) for (int i = 0; i < kPfCount; ++i) {
            std::snprintf(b, sizeof b, "%s%.3f", i ? "," : "", phases[i]); line += b;
        }
        line += "],\"scope\":\"routed_fp16_only\",\"payload_semantics\":\"requested_copy_submission_bytes_not_successful_dma_or_link_traffic\","
                "\"timing_semantics\":\"existing_nonexclusive_event_mark_intervals\"}\n";
        if (line.size() > 1024 * 1024) {
            std::fputs("strata routed fp16 observation: {\"version\":2,\"run_return\":\"success\",\"effective\":false,"
                       "\"reason\":\"output_budget_exceeded\"}\n", stderr);
            return;
        }
        std::fputs(line.c_str(), stderr);
    }
};
'''
replace('struct PfTimer {', observer + 'struct PfTimer {')
admission = r'''    RoutedFp16Observation ro;
    const char* ro_flag = std::getenv("STRATA_PREFILL_ROUTED_FP16_OBSERVE");
    ro.requested = ro_flag && std::strcmp(ro_flag, "1") == 0;
    if (ro.requested) {
#if defined(STRATA_ENABLE_SYCL)
        const auto& lay = strata::kernels::cpu::expert_layout();
        const char* mmq = std::getenv("STRATA_PREFILL_MMQ");
        if (!mmq || std::strcmp(mmq, "0") != 0 || mmq_plan().any || !mmq_plan().fallback)
            ro.reject("requires_explicit_all_fp16_mmq_zero");
        else if (!m.src)
            ro.reject("requires_native_expert_source");
        else if (m.pp || core::peer_portable() || next_ || hand_in_ || LB != 0 || LE != g.n_layers)
            ro.reject("peer_or_split_stage_out_of_scope");
        else if (N != 2560 || K != 10 || g.n_embd != N || g.n_expert != 512 || ss.k != K ||
                 g.n_layers <= 0 || g.n_layers > 512 || n <= 0 || n > 262144 || pos0 < 0 ||
                 m.T <= 0 || m.T > 8192 || m.T_max < m.T || m.T_max > 8192)
            ro.reject("geometry_or_capacity_out_of_scope");
        else if (!lay.native || lay.n_expert != 512 || lay.fmt.size() != (size_t) g.n_layers ||
                 lay.bytes.size() != lay.fmt.size())
            ro.reject("requires_complete_native_layout");
        else {
            bool formats_ok = true;
            for (size_t l = 0; l < lay.fmt.size(); ++l) {
                const auto& f = lay.fmt[l];
                if (f.n_embd != N || f.n_ff != 640 || f.gu_type < 0 || f.d_type < 0 ||
                    !f.gu_row || !f.d_row || !lay.blob_bytes((int64_t) l) ||
                    lay.blob_bytes((int64_t) l) != f.bytes) formats_ok = false;
            }
            if (!formats_ok) ro.reject("native_format_out_of_scope");
            else {
                try {
                    ro.hist.resize((size_t) m.T_max + 1, 0);
                    ro.layer_chunks.resize((size_t) g.n_layers, 0);
                    ro.on = true;
                } catch (const std::bad_alloc&) { ro.reject("host_observation_allocation_failed"); }
            }
        }
#else
        ro.reject("requires_sycl_engine");
#endif
        ro.tokens = n; ro.position = pos0; ro.configured_chunk = m.T; ro.capacity = m.T_max;
        ro.streamed0 = stats_.experts_streamed; ro.dma0 = stats_.experts_dma;
        ro.resident0 = stats_.experts_resident; ro.host0 = stats_.ms_experts_host;
    }
'''
replace('    PfTimer pt;\n    PeTimer pe;', admission + '    PfTimer pt;\n    PeTimer pe;')
replace('        ++stats_.chunks;\n        pt.mark', '        ++stats_.chunks;\n        ro.chunk(T);\n        pt.mark')
replace('                        // multi-GPU: the rows of the experts the peer computes go last, as one block [rows_local, T*K)', '                        ro.counts(l, T, m.cnt);\n                        // multi-GPU: the rows of the experts the peer computes go last, as one block [rows_local, T*K)')
replace('                            m.gemm.f16(m.Xs + scratch_row * N,', '                            ro.gu(ne, slot);\n                            m.gemm.f16(m.Xs + scratch_row * N,')
replace('                            m.gemm.f16(m.Hh + scratch_row * 640,', '                            ro.down(ne);\n                            m.gemm.f16(m.Hh + scratch_row * 640,')
# Inline issuer: consumer-owned observation, immediately before existing submission branch.
replace('                if (en.job < 0) {\n                    cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes,', '                if (ro.on) ro.payload.add(bytes, en.job < 0, m.stage_live[sl]);\n                if (en.job < 0) {\n                    cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes,')
# Thread-local producer counters outlive IssuerJoin, and never access mutable ro.
replace('        std::thread issuer;\n        struct IssuerJoin', '        const bool observe_issuer_payload = ro.on;\n        RoutedBlobPayload issuer_payload;\n        std::thread issuer;\n        struct IssuerJoin')
replace('                    if (en.job < 0) {\n                        cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes,', '                    if (observe_issuer_payload) issuer_payload.add(bytes, en.job < 0, m.stage_live[sl]);\n                    if (en.job < 0) {\n                        cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes,')
replace('            stats_.experts_dma += iss_dma;\n', '            stats_.experts_dma += iss_dma;\n            if (ro.on) ro.payload.merge(issuer_payload);\n')
# Routed small-chunk staging path: both direct and stager source branches share one payload submission.
replace('                            if (pinned) {\n                                // DMA straight', '                            if (ro.on) ro.payload.add((size_t) lay.blob_bytes(l), pinned, m.stage_live[sl]);\n                            if (pinned) {\n                                // DMA straight')
# Emit only after all existing success/error checks and debug work; failed/cancelled runs have no success summary.
replace('    return true;\n}\n\n}  // namespace strata::prefill', '''    if (ro.requested)
        ro.emit(m.compact_moe_fp16, pt.on, pt.ms, ms_since(t_start), stats_.experts_streamed,
                stats_.experts_dma, stats_.experts_resident, stats_.ms_experts_host);
    return true;
}

}  // namespace strata::prefill''')
patch = ''.join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
                                  fromfile='a/src/prefill/prefill.cpp', tofile='b/src/prefill/prefill.cpp'))
(OUT / 'routed-fp16-observation.patch').write_text(patch)
identity = {'version':2, 'previous_patch_sha256':'59659923b1d6b70bd27cc6b4cdff11fd2043ccf7d471558537b3f34729097783', 'status':'unapplied_unbuilt_untested_proposal', 'head_supplied':'8d13cef8',
            'source':'src/prefill/prefill.cpp','source_sha256':EXPECTED,
            'candidate_sha256':hashlib.sha256(candidate.encode()).hexdigest(),
            'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),
            'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'full_candidate_source_retained':False}
(OUT / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
print(json.dumps(identity, indent=2))
