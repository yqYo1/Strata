#!/usr/bin/env python3
"""Create an unapplied host-only route census patch; never alter product files."""
from pathlib import Path
import difflib
import hashlib
import json
P=Path(__file__).resolve().parents[4]
D=Path(__file__).resolve().parent
name='src/prefill/prefill.cpp'
base=(P/name).read_text()
base_sha='6b4b7a71175471d7d800e40f98ec62236bf57e1cfd57d9d41f4863d4653c2104'
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(base.encode())==base_sha
observer=r"""// Host admission counts only. No capability re-query, device reads or new stream work.
struct QsaRouteObservation {
    bool requested = false, on = false;
    const char* reason = "not_admitted";
    int64_t tokens = 0, position = 0, configured_chunk = 0, capacity = 0;
    int64_t cap = 0, page_size = 0, slice = 0, layers = 0;
    // [(queries, KV mode 0..3, staged 0..1, outcome 0..2)] and fallback [(nb, mode, staged)].
    // Outcomes: matrix accepted, matrix false decline, forced bypass. Accepted means returned/submitted.
    std::vector<uint64_t> decisions, fallback_shapes, layer_chunks;
    uint64_t accepted = 0, declined = 0, forced = 0;
    uint64_t accepted_queries = 0, declined_queries = 0, forced_queries = 0;
    uint64_t fallback_calls = 0, fallback_queries = 0, expected_fallback_calls = 0;
    void reject(const char* why) { on = false; reason = why; }
    void record(bool bypass, bool taken, int64_t T, int mode, bool staged, int64_t layer) {
        if (!on) return;
        if (T <= 0 || T > capacity || mode < 0 || mode > 3 || layer < 0 || layer >= layers) {
            reject("observed_shape_out_of_scope"); return;
        }
        const int outcome = bypass ? 2 : taken ? 0 : 1;
        ++decisions[((size_t(T) * 4 + size_t(mode)) * 2 + size_t(staged)) * 3 + size_t(outcome)];
        ++layer_chunks[size_t(layer)];
        if (bypass) { ++forced; forced_queries += uint64_t(T); }
        else if (taken) { ++accepted; accepted_queries += uint64_t(T); }
        else { ++declined; declined_queries += uint64_t(T); }
        if (bypass || !taken) expected_fallback_calls += uint64_t(1 + (T - 1) / slice);
    }
    void fallback(int64_t nb, int mode, bool staged) {
        if (!on) return;
        if (nb <= 0 || nb > slice || nb > capacity || mode < 0 || mode > 3) {
            reject("fallback_shape_out_of_scope"); return;
        }
        ++fallback_shapes[(size_t(nb) * 4 + size_t(mode)) * 2 + size_t(staged)];
        ++fallback_calls; fallback_queries += uint64_t(nb);
    }
    void emit() {
        if (!requested) return;
        if (on) {
            const uint64_t expected_chunks = uint64_t(1 + (tokens - 1) / configured_chunk);
            const uint64_t expected_queries = uint64_t(tokens) * uint64_t(layers);
            if (accepted + declined + forced != expected_chunks * uint64_t(layers) ||
                accepted_queries + declined_queries + forced_queries != expected_queries ||
                fallback_queries != declined_queries + forced_queries ||
                fallback_calls != expected_fallback_calls)
                reject("decision_query_or_fallback_accounting_mismatch");
            for (uint64_t c : layer_chunks) if (c != expected_chunks) reject("qsa_layer_chunk_mismatch");
            uint64_t dc = 0, dq = 0, fc = 0, fq = 0, rows = 0;
            for (size_t i = 0; i < decisions.size(); ++i) {
                if (decisions[i]) ++rows;
                dc += decisions[i]; dq += uint64_t(i / 24) * decisions[i];
            }
            for (size_t i = 0; i < fallback_shapes.size(); ++i) {
                if (fallback_shapes[i]) ++rows;
                fc += fallback_shapes[i]; fq += uint64_t(i / 8) * fallback_shapes[i];
            }
            if (dc != accepted + declined + forced || dq != expected_queries ||
                fc != fallback_calls || fq != fallback_queries) reject("histogram_mismatch");
            if (rows > 2048) reject("bounded_output_row_budget_exceeded");
        }
        if (!on) {
            std::fprintf(stderr, "strata qsa route observation: {\"version\":1,\"run_return\":\"success\","
                "\"effective\":false,\"reason\":\"%s\"}\n", reason);
            return;
        }
        std::string line = "strata qsa route observation: {\"version\":1,\"run_return\":\"success\",\"effective\":true";
        auto num = [&](const char* key, uint64_t value) {
            line += ",\""; line += key; line += "\":"; line += std::to_string(value);
        };
        num("batched_input_tokens", tokens); num("position0", position);
        num("configured_chunk", configured_chunk); num("query_capacity", capacity);
        num("cap", cap); num("page_size", page_size); num("fallback_batch", slice); num("qsa_layers", layers);
        num("matrix_attempts_returned", accepted + declined); num("matrix_accepted", accepted);
        num("matrix_false_declines", declined); num("forced_bypasses", forced);
        num("matrix_accepted_queries", accepted_queries); num("matrix_declined_queries", declined_queries);
        num("forced_queries", forced_queries); num("fallback_calls", fallback_calls); num("fallback_queries", fallback_queries);
        line += ",\"decision_shape_counts\":[";
        bool first = true;
        for (size_t i = 0; i < decisions.size(); ++i) if (decisions[i]) {
            if (!first) line += ','; first = false;
            line += '[' + std::to_string(i / 24) + ',' + std::to_string((i / 6) % 4) + ',';
            line += std::to_string((i / 3) % 2) + ',' + std::to_string(i % 3) + ',' + std::to_string(decisions[i]) + ']';
        }
        line += "],\"fallback_shape_counts\":["; first = true;
        for (size_t i = 0; i < fallback_shapes.size(); ++i) if (fallback_shapes[i]) {
            if (!first) line += ','; first = false;
            line += '[' + std::to_string(i / 8) + ',' + std::to_string((i / 2) % 4) + ',';
            line += std::to_string(i % 2) + ',' + std::to_string(fallback_shapes[i]) + ']';
        }
        line += "],\"scope\":\"host_returned_nonzero_prefill_admissions_not_exclusive_timing\","
            "\"decline_reason_detail\":\"not_instrumented\",\"selected_cell_occupancy\":\"not_observed\"}\n";
        if (line.size() > 256 * 1024) {
            std::fputs("strata qsa route observation: {\"version\":1,\"run_return\":\"success\","
                "\"effective\":false,\"reason\":\"bounded_output_byte_budget_exceeded\"}\n", stderr);
            return;
        }
        std::fputs(line.c_str(), stderr);
    }
};
"""
needle='struct PfTimer {\n'
assert base.count(needle)==1
proposed=base.replace(needle,observer+needle)
needle='    RoutedFp16Observation ro;\n'
init=r"""    QsaRouteObservation qo;
    const char* qo_flag = std::getenv("STRATA_PREFILL_QSA_ROUTE_OBSERVE");
    qo.requested = qo_flag && std::strcmp(qo_flag, "1") == 0;
    if (qo.requested) {
#if defined(STRATA_ENABLE_SYCL)
        qo.tokens = n; qo.position = pos0; qo.configured_chunk = m.T; qo.capacity = m.T_max;
        qo.cap = m.cap; qo.page_size = s.page_size; qo.slice = m.attn_batch; qo.layers = g.n_qsa_layers();
        if (m.pp || core::peer_portable() || next_ || hand_in_ || LB != 0 || LE != g.n_layers)
            qo.reject("peer_or_split_stage_out_of_scope");
        else if (n <= 0 || n > 262144 || pos0 < 0 || pos0 > 262144 || n > 262144 - pos0 ||
                 m.T <= 0 || m.T > 8192 || m.T_max < m.T || m.T_max > 8192 ||
                 m.cap <= 0 || m.cap > 2147483647 || m.attn_batch <= 0 || m.attn_batch > m.T_max ||
                 qo.layers <= 0 || qo.layers > 512 || s.n_head != 24 || s.n_head_kv != 2 || s.head_dim != 256 ||
                 s.page_size <= 0 || s.page_size > 2147483647)
            qo.reject("geometry_or_capacity_out_of_scope");
        else {
            try {
                qo.decisions.resize((size_t(m.T_max) + 1) * 24, 0);
                qo.fallback_shapes.resize((size_t(m.T_max) + 1) * 8, 0);
                qo.layer_chunks.resize(size_t(qo.layers), 0);
                qo.on = true;
            } catch (const std::bad_alloc&) { qo.reject("host_observation_allocation_failed"); }
        }
#else
        qo.reject("requires_sycl_engine");
#endif
    }
"""
assert proposed.count(needle)==1
proposed=proposed.replace(needle,init+needle)
old=r"""                    if (old_attn || !strata::kernels::qsa_prompt_attn_batch(m.q, pools, m.sel_ids, m.steps_dev, m.cap, s,
                                                                            m.attn, T, m.cs))
                        for (int64_t t0 = 0; t0 < T; t0 += m.attn_batch) {
                            const int64_t nb = std::min(m.attn_batch, T - t0);
"""
new=r"""                    const bool fallback_attn = old_attn || !strata::kernels::qsa_prompt_attn_batch(
                        m.q, pools, m.sel_ids, m.steps_dev, m.cap, s, m.attn, T, m.cs);
                    const int observed_mode = qo.on ? (pools.k_q4 ? 2 : pools.k_q ? (pools.v_q4 ? 3 : 1) : 0) : 0;
                    if (qo.on) qo.record(old_attn, !fallback_attn, T, observed_mode, staged, qsa_index);
                    if (fallback_attn)
                        for (int64_t t0 = 0; t0 < T; t0 += m.attn_batch) {
                            const int64_t nb = std::min(m.attn_batch, T - t0);
                            if (qo.on) qo.fallback(nb, observed_mode, staged);
"""
assert proposed.count(old)==1
proposed=proposed.replace(old,new)
needle='    if (ro.requested)\n        ro.emit('
assert proposed.count(needle)==1
proposed=proposed.replace(needle,'    qo.emit();\n'+needle)
patch=''.join(difflib.unified_diff(base.splitlines(True),proposed.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
(D/'unified.patch').write_text(patch)
identity={'status':'unapplied-unbuilt-untested-unmeasured-unadopted','source_base':{name:base_sha},'proposed':{name:sha(proposed.encode())},'patch_sha256':sha(patch.encode()),'math_kernel_API_queue_event_copy_changes':False,'product_files_modified':False,'bounds':{'query_capacity':8192,'qsa_layers':512,'input_positions':262144,'host_counter_bytes_max':(8193*32+512)*8,'output_bytes':256*1024},'consumer':'Main source review then serial route observation after benchmark and PR6 requested boundary; no active run consumes this proposal'}
(D/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps(identity,indent=2))
