from pathlib import Path
import hashlib, difflib, json
P = Path(__file__).resolve().parents[4]
D = Path(__file__).resolve().parent
base = (P / 'src/program/generate.cpp').read_bytes()
expected = 'e3ca2a964806270e1d72862f979b6e262fb17a925a6027e7534c81dc45418871'
assert hashlib.sha256(base).hexdigest() == expected, 'Pinned source changed; stop and review.'
s = base.decode()
def replace(old, new):
    global s
    assert s.count(old) == 1, (old[:100], s.count(old))
    s = s.replace(old, new)
replace('    int64_t max_new = 16;', '    int64_t max_new = 16;\n    bool input_only = false;          // consume real prompt positions and emit no output IDs')
replace('        else if (a == "--max-new")', '        else if (a == "--input-only") o.input_only = true;\n        else if (a == "--max-new")')
replace('                 "  --max-new N          tokens to generate (default 16)\\n"', '                 "  --max-new N          tokens to generate (default 16)\\n"\n                 "  --input-only         standalone diagnostic; requires --max-new 0, emits no output IDs\\n"')
anchor = '    if (!have_tokens && o.serve) {   // plan v0.3 P8: requests bring their own tokens'
replace(anchor, '''    if (o.input_only && (o.max_new != 0 || o.serve || o.vision || !o.layer_split.empty() ||
        o.graph_only || o.gpu_only_full || o.gpu_stages || o.no_pool || o.no_ple ||
        o.prefill_until != 0 || !o.spec_oracle.empty() || o.spec_corrupt != 0 ||
        !o.dump_logits.empty() || !o.dump_final_r.empty() || !o.dump_layers.empty() ||
        !o.dump_halves.empty() || !o.dump_routing.empty() || !o.dump_expert_inputs.empty() ||
        !o.dump_residual.empty() || !o.dump_mixed.empty())) {
        std::fprintf(stderr, "strata generate: --input-only requires --max-new 0 and complete standalone text inference; "
                             "serve/vision/split, partial-engine probes/ablations, prefill-until, oracle/corruption "
                             "and CLI tensor/trace dumps are unsupported\\n");
        return 2;
    }
''' + anchor)
replace('    if (o.max_new <= 0 || o.max_context <= 0 || o.max_new > o.max_context ||', '    if ((!o.input_only && o.max_new <= 0) || (o.input_only && (o.max_new != 0 || o.tokens.empty())) ||\n        o.max_context <= 0 || o.max_new > o.max_context ||')
replace('    const bool native_pack = strata::kernels::cpu::expert_layout().native;', '''    const bool native_pack = strata::kernels::cpu::expert_layout().native;
    if (o.input_only && o.spec != 0 && (!native_pack || o.spec < 2 || o.native_flash_attn_short ||
        (o.tokens.size() > 1 && o.prefill_chunk <= 0))) {
        std::fprintf(stderr, "strata generate: --input-only supports spec 0 or native spec >=2; "
                             "native speculative input rejects native-flash-attn-short and multi-token input requires batched prefill\\n");
        return 2;
    }''')
replace('    int64_t pos_start = 0;\n    int64_t spec_pos', '''    int64_t input_consumed = 0, input_batched = 0, input_chunks = 0;
    auto input_receipt = [&](const char* route, bool sampled_prediction) -> int {
        if (input_consumed != n_prompt || !produced.empty()) {
            std::fprintf(stderr, "strata generate: input-only completion invariant failed\\n");
            return 1;
        }
        std::printf("INPUT_ONLY_RECEIPT {\\"schema\\":1,\\"input_ids\\":%lld,\\"consumed_input_ids\\":%lld,"
                    "\\"context_cells\\":%lld,\\"last_input_position\\":%lld,\\"last_input_id\\":%lld,"
                    "\\"batched_input_ids\\":%lld,\\"prefill_chunks\\":%lld,\\"effective_prefill_chunk\\":%lld,"
                    "\\"spec\\":%d,\\"mtp_bound\\":%s,\\"route\\":\\"%s\\",\\"head_complete\\":true,"
                    "\\"target_logits_finite\\":true,\\"state_complete\\":true,\\"emitted_ids\\":0,"
                    "\\"internal_prediction_sampled\\":%s,\\"status\\":\\"completed\\"}\\n",
                    (long long) n_prompt, (long long) input_consumed, (long long) o.max_context,
                    (long long) (n_prompt - 1), (long long) o.tokens.back(), (long long) input_batched,
                    (long long) input_chunks, (long long) o.prefill_chunk, o.spec,
                    o.mtp.empty() ? "false" : "true", route, sampled_prediction ? "true" : "false");
        return std::fflush(stdout) == 0 ? 0 : 1;
    };
    int64_t pos_start = 0;
    int64_t spec_pos''')
replace('                cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);\n                borrow = xcache.device_slot(first);', '''                const cudaError_t lend_table_status = cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
                if (o.input_only && lend_table_status != cudaSuccess) {
                    std::fprintf(stderr, "strata generate: input-only lent residency table submission failed\\n");
                    return 1;
                }
                borrow = xcache.device_slot(first);''')
replace('            cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);\n            std::fprintf(stderr, "strata generate: %zu lent slots refilled', '''            const cudaError_t refill_table_status = cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
            if (o.input_only && refill_table_status != cudaSuccess) {
                std::fprintf(stderr, "strata generate: input-only lent residency table refill failed\\n");
                return 1;
            }
            std::fprintf(stderr, "strata generate: %zu lent slots refilled''')
replace('        const strata::prefill::PrefillStats& ps = prefill.stats();', '''        const strata::prefill::PrefillStats& ps = prefill.stats();
        if (o.input_only) {
            if (ps.tokens != n_batched) {
                std::fprintf(stderr, "strata generate: input-only batched token accounting mismatch\\n");
                return 1;
            }
            input_consumed = input_batched = ps.tokens;
            input_chunks = ps.chunks;
        }''')
replace('        const bool read_logits = !o.stream_token || o.check_logits || emit_logits;', '        const bool read_logits = o.input_only || !o.stream_token || o.check_logits || emit_logits;')
replace('        int next = 0;\n        // The draw is Philox', '''        if (o.input_only) {
            if (pos != input_consumed || tok != o.tokens[(size_t) pos]) {
                std::fprintf(stderr, "strata generate: input-only token/position accounting mismatch\\n");
                return 1;
            }
            // Full logits readback above synchronized the token stream, including QSA status copies.
            if (o.native_flash_attn_short) for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
                const int64_t i = ss.qsa_ord0 + j;
                const int32_t status = ss.qsa_states[i].host_step[strata::kernels::kStepCount];
                if (status != 0) {
                    std::fprintf(stderr, "strata generate: input-only QSA status %d at layer %lld position %lld\\n",
                                 status, (long long) i, (long long) pos);
                    return 1;
                }
            }
            ++input_consumed;
            ss.ple_prev[0] = ss.ple_prev[1];
            ss.ple_prev[1] = (int32_t) tok;
            if (input_consumed == n_prompt) return input_receipt("token", false);
            tok = o.tokens[(size_t) input_consumed];
            continue;                       // no sampler or output at any prompt position
        }
        int next = 0;
        // The draw is Philox''')
replace('    if (spec_pos >= 0 && (int64_t) produced.size() < o.max_new && !ended) {', '    if (spec_pos >= 0 && (o.input_only || (int64_t) produced.size() < o.max_new) && !ended) {')
anchor = '        const int64_t pcie0 = drive.d.pcie_experts;'
replace(anchor, '''        if (o.input_only) {
            // Same native first window, actual final prompt ID, no draft and no cell beyond the context.
            if (!native_pack || spec_pos != n_prompt - 1 || input_consumed != spec_pos ||
                tok != o.tokens.back() || spec_pos >= o.max_context) {
                std::fprintf(stderr, "strata generate: input-only verifier input accounting mismatch\\n");
                return 1;
            }
            const int32_t last = (int32_t) tok;
            int32_t discarded_prediction = -1;
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            if (!ver.run(1, &last, spec_pos, &drive_pool_multi, &drive, &discarded_prediction, err) || drive.d.failed) {
                std::fprintf(stderr, "strata generate: input-only verifier failed: %s; pool failed=%d layer=%lld expert=%lld %s\\n",
                             err.c_str(), (int) drive.d.failed, (long long) drive.d.fail_layer,
                             (long long) drive.d.fail_expert, drive.d.fail ? drive.d.fail : "");
                return 1;
            }
            if (!ver.commit(1, err) || !ver.wait_commit(err)) {
                std::fprintf(stderr, "strata generate: input-only verifier commit failed: %s\\n", err.c_str());
                return 1;
            }
            std::vector<float> row((size_t) ver.vocab());
            if (row.size() != (size_t) n_vocab || !ver.copy_logits(0, row.data())) {
                std::fprintf(stderr, "strata generate: input-only verifier logits readback failed\\n");
                return 1;
            }
            for (float v : row) if (!std::isfinite(v)) {
                std::fprintf(stderr, "strata generate: input-only verifier logits are not finite\\n");
                return 1;
            }
            if (const char* fl = std::getenv("STRATA_DUMP_FIRST_LOGITS")) {
                std::FILE* f = std::fopen(fl, "wb");
                if (f == nullptr) {
                    std::fprintf(stderr, "strata generate: input-only cannot open first logits dump\\n");
                    return 1;
                }
                const bool write_ok = std::fwrite(row.data(), sizeof(float), row.size(), f) == row.size();
                const bool close_ok = std::fclose(f) == 0;
                if (!write_ok || !close_ok) {
                    std::fprintf(stderr, "strata generate: input-only cannot finish first logits dump\\n");
                    return 1;
                }
            }
            ++input_consumed;                // only after actual final-window commit and checked head readback
            drive.d.plan = nullptr;
            return input_receipt("native-verifier", true);
        }
''' + anchor)
patch = ''.join(difflib.unified_diff(base.decode().splitlines(True), s.splitlines(True), fromfile='a/src/program/generate.cpp', tofile='b/src/program/generate.cpp')).encode()
(D / 'full-input.patch').write_bytes(patch)
identity = {'status':'proposal-unapplied-no-build-no-tests', 'base_commit':'8d13cef8d9ca82887358d066d2e3d1982cb191eb', 'base_source_sha256':expected, 'patch_sha256':hashlib.sha256(patch).hexdigest(), 'derived_source_sha256':hashlib.sha256(s.encode()).hexdigest(), 'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(D / 'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
print(json.dumps(identity))
