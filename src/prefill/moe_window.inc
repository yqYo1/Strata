// Main-authored expert-major window. Saved matrix inputs remain FP16;
// router/shared scalar gate consume the original BF16 mix. Only the expert
// accumulation order and the <=8192-row GEMM job shapes change numerically.
bool Prefill::run_moe_accum_window(int64_t layer, int64_t first_row, int64_t rows, std::string& err) try {
    Impl& m = *impl_;
    if (!m.moe_window || !m.moe_window->accumulation || !m.native16_active ||
        !m.layer_major_active || m.T != 8192 || m.copy != m.cs || !m.scratch_alias ||
        !m.cache || !m.host_res || layer < 0 || layer >= m.g->n_layers ||
        rows <= m.T || rows > m.moe_window->maximum_rows || rows > detail::MoeAccumPlan::maximum_rows || first_row < 0) {
        err = "prefill MoE accumulator: unsupported entry"; return false;
    }
    auto& owner = *m.moe_window;
    if (owner.command_trace) { err = "prefill MoE accumulator: window2 command collector unsupported"; return false; }
    auto& queue = sycl_backend::queue_for(m.cs);
    const uint64_t offset = detail::layer_major_mul(uint64_t(first_row), D * 2);
    const uint64_t bytes = detail::layer_major_mul(uint64_t(rows), D * 2);
    if (!owner.bank || offset > owner.bank_bytes || bytes > owner.bank_bytes - offset ||
        !owner.mixed || !m.region || !m.emb || reinterpret_cast<uintptr_t>(m.region) % 256)
        throw std::invalid_argument("prefill MoE accumulator: backing bank/arena bounds");
    const auto plan = detail::MoeAccumPlan::make_contiguous(rows, owner.mixed_bytes, m.region_bytes);
    auto* residual = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(owner.bank) + offset);
    auto* mixed = owner.mixed;
    auto* bo = reinterpret_cast<uint16_t*>(m.emb);
    const size_t mixed_bytes = size_t(detail::layer_major_mul(uint64_t(rows), N * 2));
    sycl_backend::validate_device_span(residual, size_t(bytes), queue);
    sycl_backend::validate_device_span(m.region, size_t(plan.region_bytes), queue);
    sycl_backend::validate_device_span(mixed, mixed_bytes, queue);
    sycl_backend::validate_device_span(bo, size_t(m.T * N * 2), queue);
    sycl_backend::validate_spans({{m.region, size_t(plan.region_bytes)}, {mixed, mixed_bytes},
                                 {bo, size_t(m.T * N * 2)}}, {{residual, size_t(bytes)}});
    using Plane = detail::MoeAccumPlan::Plane;
    auto view = [&](Plane plane) -> void* {
        const auto span = plan.span(plane);
        if (span.offset > m.region_bytes || span.bytes > m.region_bytes - span.offset)
            throw std::invalid_argument("prefill MoE accumulator: logical arena extent");
        void* result = m.region + size_t(span.offset);
        sycl_backend::validate_device_span(result, size_t(span.bytes), queue);
        return result;
    };
    auto* accumulator = static_cast<float*>(view(Plane::accumulator));
    auto* weights = static_cast<float*>(view(Plane::weights));
    auto* inverse = static_cast<int32_t*>(view(Plane::slots));
    auto* source = static_cast<int32_t*>(view(Plane::source));
    auto* job_weights = static_cast<float*>(view(Plane::job_weights));
    auto* injection = static_cast<float*>(view(Plane::injection));
    auto* gathered = static_cast<uint16_t*>(view(Plane::gathered));
    auto* gu = static_cast<uint16_t*>(view(Plane::gu));
    auto* hidden = static_cast<uint16_t*>(view(Plane::hidden));
    auto* down = static_cast<uint16_t*>(view(Plane::down_job));
    auto* shared = static_cast<uint16_t*>(view(Plane::shared_chunk));
    auto* shared_gate = static_cast<float*>(view(Plane::shared_gate));
    auto* logits = static_cast<float*>(view(Plane::logits));
    auto* sgate = static_cast<uint16_t*>(view(Plane::shared_gate_input));
    auto* sup = static_cast<uint16_t*>(view(Plane::shared_up_input));
    auto* sh = static_cast<uint16_t*>(view(Plane::shared_hidden));
    auto* status = static_cast<uint32_t*>(view(Plane::status));
    const core::LayerView v(*m.wt, layer, m.native_dense_overlay);
    const core::WeightRef *wn = need(v, "hc_ffn_norm.weight", err),
                          *wd = need(v, "hc_ffn_down.weight", err),
                          *wu = need(v, "hc_ffn_up.weight", err),
                          *wi = need(v, "hc_ffn_inject.weight", err),
                          *wr = need(v, "ffn_gate_inp.weight", err),
                          *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                          *wsg = need(v, "ffn_gate_shexp.weight", err),
                          *wsu = need(v, "ffn_up_shexp.weight", err),
                          *wsd = need(v, "ffn_down_shexp.weight", err);
    if (!wn || !wd || !wu || !wi || !wr || !wgi || !wsg || !wsu || !wsd) return false;
    if (wgi->kind != core::WeightKind::Bf16InF32) {
        err = "prefill MoE accumulator: shared scalar gate kind"; return false;
    }
    if (m.layer_major_profile) m.layer_major_profile->begin(detail::LayerMajorCoarseProfile::moe, m.cs);
    if (cudaMemsetAsync(accumulator, 0, size_t(rows * N * 4), m.cs) != cudaSuccess ||
        cudaMemsetAsync(status, 0, sizeof(uint32_t), m.cs) != cudaSuccess) {
        err = "prefill MoE accumulator: initialization"; return false;
    }
    // Attention/PLE already advanced each chunk exactly once. This helper is FFN only.
    for (int64_t at = 0; at < rows; at += m.T) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        const int64_t count = std::min(m.T, rows - at);
        const auto* r = residual + at * D;
        native16::norm(r, static_cast<const float*>(wn->data), EPS, m.grs, m.xn16, count, m.cs);
        if (!bf16_proj16(m.gemm, wd, m.xn16, m.lo_b16, count, v.name("hc_ffn_down.weight"), err)) return false;
        native16::silu(m.lo_b16, m.lo16, count, m.cs);
        if (!bf16_proj16(m.gemm, wu, m.lo16, m.gated_b16, count, v.name("hc_ffn_up.weight"), err) ||
            !bf16_proj(m.gemm, wi, m.xn16, injection + at * HC, count, v.name("hc_ffn_inject.weight"), err)) return false;
        native16::mix(r, m.grs, static_cast<const float*>(wn->data), m.gated_b16,
                      m.mixed_bf, mixed + at * N, count, m.cs);
        if (!bf16_proj(m.gemm, wr, m.mixed_bf, logits, count, v.name("ffn_gate_inp.weight"), err)) return false;
        route(logits, inverse + at * K, weights + at * K, count, NE, m.cs);
        // Never reconstruct this BF16 producer from its already-rounded FP16 sibling.
        m.gemm.bf16(m.mixed_bf, static_cast<const uint16_t*>(wgi->data), shared_gate + at, count, 1, N);
    }
    owner.ids.resize(size_t(rows * K));
    if (cudaMemcpyAsync(owner.ids.data(), inverse, size_t(rows * K * 4), cudaMemcpyDeviceToHost, m.cs) != cudaSuccess ||
        cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = "prefill MoE accumulator: router readback/drain"; return false;
    }
    {
        sycl_backend::cpu_wait_profile::Host group(sycl_backend::cpu_wait_profile::host_group);
        owner.routes = detail::MoeWindowRoutes::build_accum(rows, owner.ids.data(), owner.ids.size());
    }
    if (m.layer_major_profile) {
        // Metadata already owned on the host after the existing router drain.
        // No additional tensor readback, GPU submission or inference mutation.
        // This is a planned-job census; only final request/issuer totals prove
        // that all jobs completed. The old chunk census refuses window mode.
        std::array<uint32_t, 8193> job_rows{};
        std::array<uint32_t, 11> jobs_per_expert{};
        for (const auto& job : owner.routes->jobs()) ++job_rows[size_t(job.rows)];
        for (int32_t count : owner.routes->counts())
            ++jobs_per_expert[size_t((count + 8191) / 8192)];
        std::fprintf(stderr, "strata MoE planned job census: {\"version\":1,\"layer\":%lld,\"first_row\":%lld,\"rows\":%lld,\"jobs\":%zu,\"gu_nk\":[1280,2560],\"down_nk\":[2560,640],\"gu_down_storage\":\"bf16\",\"hidden_storage\":\"fp16\",\"allocated_weight_pairs\":2,\"used_weight_pairs\":1,\"weight_pair_bytes\":9830400,\"requested_router_D2H_bytes\":%llu,\"requested_map_H2D_bytes\":%llu,\"job_rows_histogram\":[",
            (long long)layer, (long long)first_row, (long long)rows, owner.routes->jobs().size(),
            (unsigned long long)(rows*K*4), (unsigned long long)(rows*K*8));
        bool comma=false;
        for (size_t count=1;count<job_rows.size();++count) if (job_rows[count]) {
            std::fprintf(stderr,"%s[%zu,%u]",comma?",":"",count,job_rows[count]);comma=true;
        }
        std::fputs("],\"jobs_per_expert_histogram\":[",stderr);comma=false;
        for (size_t count=0;count<jobs_per_expert.size();++count) if (jobs_per_expert[count]) {
            std::fprintf(stderr,"%s[%zu,%u]",comma?",":"",count,jobs_per_expert[count]);comma=true;
        }
        std::fputs("]}\n",stderr);
    }
    if (cudaMemcpyAsync(inverse, owner.routes->original_slots().data(), size_t(rows * K * 4), cudaMemcpyHostToDevice, m.cs) != cudaSuccess ||
        cudaMemcpyAsync(source, owner.routes->source_rows().data(), size_t(rows * K * 4), cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
        err = "prefill MoE accumulator: immutable maps upload"; return false;
    }
    native16::moe_job_weights(*owner.routes, weights, inverse, job_weights, status, m.cs);
    const auto& layout = strata::kernels::cpu::expert_layout();
    const auto& format = layout.fmt[size_t(layer)];
    int32_t loaded = -1;
    for (size_t index = 0; index < owner.routes->jobs().size(); ++index) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        const auto& job = owner.routes->jobs()[index];
        if (job.expert != loaded) {
            const int32_t slot = m.host_res[size_t(layer * NE + job.expert)];
            if (slot < 0) { err = "prefill MoE accumulator: missing layer expert"; return false; }
            const auto* blob = m.cache->device_slot(slot);
            sycl_backend::validate_device_span(blob, size_t(layout.blob_bytes(layer)), queue);
            sycl_backend::validate_device_span(m.dq_gu[0], size_t(2 * 640 * N * 2), queue);
            sycl_backend::validate_device_span(m.dq_d[0], size_t(N * 640 * 2), queue);
            strata::kernels::iq_dequant_gu_f16(format.gu_type, blob, blob + format.up_off,
                format.n_ff, format.n_embd, m.dq_gu[0], m.cs);
            strata::kernels::iq_dequant_f16(format.d_type, blob + format.down_off,
                format.n_embd * format.n_ff, m.dq_d[0], m.cs);
            loaded = job.expert; ++owner.dequants; ++stats_.experts_resident;
        }
        if (job.rows < 1 || job.rows > m.T || job.global_row < 0 || job.global_row > rows * K - job.rows)
            throw std::invalid_argument("prefill MoE accumulator: job bounds");
        sycl_backend::validate_device_span(source + job.global_row, size_t(job.rows * 4), queue);
        gather_rows16(mixed, source + job.global_row, gathered, job.rows, N, m.cs);
        m.gemm.f16_out16(gathered, m.dq_gu[0], gu, job.rows, 2 * 640, N, true);
        native16::swiglu(gu, hidden, job.rows, m.cs);
        m.gemm.f16_out16(hidden, m.dq_d[0], down, job.rows, N, 640, true);
        native16::moe_accumulate(*owner.routes, index, down, source, inverse, job_weights, accumulator, status, m.cs);
        ++owner.jobs;
    }
    // The host status destination belongs to the outer checked lease, even if DMA fails.
    owner.status_observed = ~uint32_t(0);
    if (cudaMemcpyAsync(&owner.status_observed, status, sizeof(uint32_t), cudaMemcpyDeviceToHost, m.cs) != cudaSuccess ||
        cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = "prefill MoE accumulator: status readback/drain"; return false;
    }
    if (owner.status_observed) { err = "prefill MoE accumulator: invalid device route map"; return false; }
    for (int64_t at = 0; at < rows; at += m.T) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        const int64_t count = std::min(m.T, rows - at);
        if (!native_proj16(m.gemm, wsg, mixed + at * N, sgate, count, v.name("ffn_gate_shexp.weight"), err) ||
            !native_proj16(m.gemm, wsu, mixed + at * N, sup, count, v.name("ffn_up_shexp.weight"), err)) return false;
        native16::swiglu_pair(sgate, sup, sh, count, m.cs);
        if (!native_proj16(m.gemm, wsd, sh, shared, count, v.name("ffn_down_shexp.weight"), err)) return false;
        native16::moe_finish(accumulator + at * N, shared, shared_gate + at, bo, count, m.cs);
        native16::write(residual + at * D, bo, injection + at * HC, count, m.cs);
    }
    if (m.layer_major_profile) m.layer_major_profile->end(detail::LayerMajorCoarseProfile::moe, m.cs);
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) { err = "prefill MoE accumulator: HC writes drain"; return false; }
    ++owner.windows; owner.source_bytes += uint64_t(rows * K * 4);
    owner.routes.reset();
    return true;
} catch (const std::exception& exception) {
    err = std::string("prefill MoE accumulator: ") + exception.what();
    return false; // checked outer lease retains mixed/bank/host uploads through retirement
}

// Main-authored MIT Strata orchestration. No external implementation source.
// Included after the existing projection helpers; no legacy m.* aliases change.
bool Prefill::run_moe_window(int64_t layer, int64_t first_row, int64_t rows, std::string& err) try {
    Impl& m = *impl_;
    if (!m.moe_window || !m.native16_active || !m.layer_major_active || m.T != 8192 ||
        m.copy != m.cs || !m.scratch_alias || !m.cache || !m.host_res ||
        layer < 0 || layer >= m.g->n_layers || rows <= m.T || rows > 2 * m.T || first_row < 0) {
        err = "prefill MoE window: unsupported entry"; return false;
    }
    auto& owner = *m.moe_window;
    auto& queue = sycl_backend::queue_for(m.cs);
    if (uint64_t(first_row) > UINT64_MAX / (D * 2) || uint64_t(rows) > UINT64_MAX / (D * 2))
        throw std::invalid_argument("prefill MoE window: bank byte overflow");
    const uint64_t offset = uint64_t(first_row) * D * 2, bytes = uint64_t(rows) * D * 2;
    if (!owner.bank || offset > owner.bank_bytes || bytes > owner.bank_bytes - offset)
        throw std::invalid_argument("prefill MoE window: bank bounds");
    sycl_backend::validate_device_span(owner.bank, size_t(owner.bank_bytes), queue);
    auto* residual = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(owner.bank) + offset);
    sycl_backend::validate_device_span(residual, size_t(bytes), queue);
    if (!m.region || reinterpret_cast<uintptr_t>(m.region) % 256 || m.region_bytes < 998443520 ||
        !m.emb || reinterpret_cast<uintptr_t>(m.emb) % 2)
        throw std::invalid_argument("prefill MoE window: backing arena/alignment");
    sycl_backend::validate_device_span(m.region, size_t(m.region_bytes), queue);
    sycl_backend::validate_device_span(m.emb, size_t(m.T * N * 4), queue);
    // W is the maximum reservation; actual producers use `rows`. Each view has
    // a 256-byte guard reservation. This does not claim guard-canary checking.
    constexpr size_t W = 16384, J = 8192, FF = 640;
    size_t used = 0;
    auto take = [&](size_t payload) -> void* {
        const size_t reserved = ((payload + 255) & ~size_t(255)) + 256;
        if (used > m.region_bytes || reserved > m.region_bytes - used)
            throw std::invalid_argument("prefill MoE window: typed view exceeds actual region");
        void* p = m.region + used;
        used += reserved;
        sycl_backend::validate_device_span(p, payload, queue);
        return p;
    };
    auto* down = static_cast<uint16_t*>(take(W * K * N * 2));
    auto* shared = static_cast<uint16_t*>(take(W * N * 2));
    auto* weights = static_cast<float*>(take(W * K * 4));
    auto* ids_slots = static_cast<int32_t*>(take(W * K * 4));
    auto* shared_gate = static_cast<float*>(take(W * 4));
    auto* injection = static_cast<float*>(take(W * HC * 4));
    auto* gathered = static_cast<uint16_t*>(take(J * N * 2));
    auto* gu = static_cast<uint16_t*>(take(J * 2 * FF * 2));
    auto* hidden = static_cast<uint16_t*>(take(J * FF * 2));
    auto* source = static_cast<int32_t*>(take(W * K * 4));
    if (used != 998443520) throw std::logic_error("prefill MoE window: layout arithmetic mismatch");
    // Preparation views overlap Down BEFORE any expert output is produced.
    uint8_t* prep = reinterpret_cast<uint8_t*>(down);
    size_t prep_used = 0;
    auto prepare = [&](size_t payload) -> void* {
        const size_t reserved = ((payload + 255) & ~size_t(255)) + 256;
        if (reserved > W * K * N * 2 - prep_used)
            throw std::logic_error("prefill MoE window: preparation view bounds");
        void* p = prep + prep_used; prep_used += reserved;
        sycl_backend::validate_device_span(p, payload, queue);
        return p;
    };
    auto* logits = static_cast<float*>(prepare(J * NE * 4));
    auto* sgate = static_cast<uint16_t*>(prepare(J * FF * 2));
    auto* sup = static_cast<uint16_t*>(prepare(J * FF * 2));
    auto* sh = static_cast<uint16_t*>(prepare(J * FF * 2));
    auto* mixed = reinterpret_cast<uint16_t*>(m.emb);
    // `mixed` becomes BO only after the final expert has read it, same queue.
    auto* bo = mixed;
    const core::LayerView v(*m.wt, layer, m.native_dense_overlay);
    const core::WeightRef *wn = need(v, "hc_ffn_norm.weight", err),
                          *wd = need(v, "hc_ffn_down.weight", err),
                          *wu = need(v, "hc_ffn_up.weight", err),
                          *wi = need(v, "hc_ffn_inject.weight", err),
                          *wr = need(v, "ffn_gate_inp.weight", err),
                          *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                          *wsg = need(v, "ffn_gate_shexp.weight", err),
                          *wsu = need(v, "ffn_up_shexp.weight", err),
                          *wsd = need(v, "ffn_down_shexp.weight", err);
    if (!wn || !wd || !wu || !wi || !wr || !wgi || !wsg || !wsu || !wsd) return false;
    if (wgi->kind != core::WeightKind::Bf16InF32) {
        err = "prefill MoE window: HC norm/shared scalar gate kind"; return false;
    }
    if (m.layer_major_profile) m.layer_major_profile->begin(detail::LayerMajorCoarseProfile::moe, m.cs);
    // The attention/PLE body already consumed both pieces. No new token gather,
    // recurrent append or PLE advance is allowed in this FFN-only helper.
    for (int64_t at = 0; at < rows; at += m.T) {
        const int64_t count = std::min(m.T, rows - at);
        const auto* r = residual + at * D;
        native16::norm(r, static_cast<const float*>(wn->data), EPS, m.grs, m.xn16, count, m.cs);
        if (!bf16_proj16(m.gemm, wd, m.xn16, m.lo_b16, count, v.name("hc_ffn_down.weight"), err)) return false;
        native16::silu(m.lo_b16, m.lo16, count, m.cs);
        if (!bf16_proj16(m.gemm, wu, m.lo16, m.gated_b16, count, v.name("hc_ffn_up.weight"), err) ||
            !bf16_proj(m.gemm, wi, m.xn16, injection + at * HC, count, v.name("hc_ffn_inject.weight"), err)) return false;
        native16::mix(r, m.grs, static_cast<const float*>(wn->data), m.gated_b16,
                      m.mixed_bf, mixed + at * N, count, m.cs);
        if (!bf16_proj(m.gemm, wr, m.mixed_bf, logits, count, v.name("ffn_gate_inp.weight"), err)) return false;
        route(logits, ids_slots + at * K, weights + at * K, count, NE, m.cs);
        if (!native_proj16(m.gemm, wsg, mixed + at * N, sgate, count, v.name("ffn_gate_shexp.weight"), err) ||
            !native_proj16(m.gemm, wsu, mixed + at * N, sup, count, v.name("ffn_up_shexp.weight"), err)) return false;
        native16::swiglu_pair(sgate, sup, sh, count, m.cs);
        if (!native_proj16(m.gemm, wsd, sh, shared + at * N, count, v.name("ffn_down_shexp.weight"), err)) return false;
        m.gemm.bf16(m.mixed_bf, static_cast<const uint16_t*>(wgi->data), shared_gate + at, count, 1, N);
    }
    owner.ids.resize(size_t(rows * K)); // owner outlives the checked outer lease
    if (cudaMemcpyAsync(owner.ids.data(), ids_slots, size_t(rows * K * 4), cudaMemcpyDeviceToHost, m.cs) != cudaSuccess ||
        cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = "prefill MoE window: router readback/drain"; return false;
    }
    {
        sycl_backend::cpu_wait_profile::Host group(sycl_backend::cpu_wait_profile::host_group);
        owner.routes = detail::MoeWindowRoutes::build(rows, owner.ids.data(), owner.ids.size());
    }
    // Both source uploads own a full W*K plane, validated above. All maps are
    // immutable and retained until the final checked write drain below.
    namespace cp = sycl_backend::command_profile;
    auto command_trace = owner.command_trace ? owner.command_trace->begin(layer, first_row) : cp::Frame{};
    if (command_trace)
        for (const auto& job : owner.routes->jobs())
            command_trace.consider(job.expert, job.global_row, job.rows);
    {
        auto tag = command_trace.tag(cp::Kind::upload, -1, 0, rows, K, 4);
        if (cudaMemcpyAsync(ids_slots, owner.routes->slot_map().data(), size_t(rows * K * 4), cudaMemcpyHostToDevice, m.cs) != cudaSuccess ||
            cudaMemcpyAsync(source, owner.routes->source_rows().data(), size_t(rows * K * 4), cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
            err = "prefill MoE window: full route map upload"; return false;
        }
    }
    const auto& layout = strata::kernels::cpu::expert_layout();
    const auto& format = layout.fmt[size_t(layer)];
    int32_t loaded = -1;
    for (const auto& job : owner.routes->jobs()) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        if (job.expert != loaded) {
            const int32_t slot = m.host_res[size_t(layer * NE + job.expert)];
            if (slot < 0) { err = "prefill MoE window: missing layer expert"; return false; }
            const auto* blob = m.cache->device_slot(slot);
            sycl_backend::validate_device_span(blob, size_t(layout.blob_bytes(layer)), queue);
            sycl_backend::validate_device_span(m.dq_gu[0], size_t(2 * FF * N * 2), queue);
            sycl_backend::validate_device_span(m.dq_d[0], size_t(N * FF * 2), queue);
            {
                auto tag = command_trace.tag(cp::Kind::dequant_gu, job.expert, job.global_row, job.rows,
                    2 * format.n_ff, format.n_embd, format.gu_type);
                strata::kernels::iq_dequant_gu_f16(format.gu_type, blob, blob + format.up_off,
                    format.n_ff, format.n_embd, m.dq_gu[0], m.cs);
            }
            {
                auto tag = command_trace.tag(cp::Kind::dequant_down, job.expert, job.global_row, job.rows,
                    format.n_embd, format.n_ff, format.d_type);
                strata::kernels::iq_dequant_f16(format.d_type, blob + format.down_off,
                    format.n_embd * format.n_ff, m.dq_d[0], m.cs);
            }
            loaded = job.expert; ++owner.dequants; ++stats_.experts_resident;
        }
        if (job.rows < 1 || job.rows > int64_t(J) || job.global_row < 0 ||
            job.global_row > rows * K - job.rows)
            throw std::invalid_argument("prefill MoE window: expert job exceeds full source/Down plane");
        // Unlike the old gather helper, this caller validates ALL input rows,
        // full source indices, and output extents before its first enqueue.
        sycl_backend::validate_device_span(mixed, size_t(rows * N * 2), queue);
        sycl_backend::validate_device_span(source + job.global_row, size_t(job.rows * 4), queue);
        sycl_backend::validate_device_span(down + int64_t(job.global_row) * N, size_t(job.rows * N * 2), queue);
        {
            auto tag = command_trace.tag(cp::Kind::gather, job.expert, job.global_row, job.rows, N);
            gather_rows16(mixed, source + job.global_row, gathered, job.rows, N, m.cs);
        }
        {
            auto tag = command_trace.tag(cp::Kind::gu, job.expert, job.global_row, job.rows, 2 * FF, N, format.gu_type);
            m.gemm.f16_out16(gathered, m.dq_gu[0], gu, job.rows, 2 * FF, N, true);
        }
        {
            auto tag = command_trace.tag(cp::Kind::swiglu, job.expert, job.global_row, job.rows, FF);
            native16::swiglu(gu, hidden, job.rows, m.cs);
        }
        {
            auto tag = command_trace.tag(cp::Kind::down, job.expert, job.global_row, job.rows, N, FF, format.d_type);
            m.gemm.f16_out16(hidden, m.dq_d[0], down + int64_t(job.global_row) * N, job.rows, N, FF, true);
        }
        ++owner.jobs;
    }
    {
        auto tag = command_trace.tag(cp::Kind::combine, -1, 0, rows, N, K);
        native16::combine(down, ids_slots, weights, shared, shared_gate, bo, rows, rows * K, m.cs);
    }
    for (int64_t at = 0; at < rows; at += m.T) {
        auto tag = command_trace.tag(cp::Kind::write, -1, at, std::min(m.T, rows - at), D);
        native16::write(residual + at * D, bo + at * N, injection + at * HC, std::min(m.T, rows - at), m.cs);
    }
    if (m.layer_major_profile) m.layer_major_profile->end(detail::LayerMajorCoarseProfile::moe, m.cs);
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = "prefill MoE window: both HC writes drain"; return false;
    }
    command_trace.after_checked_drain();
    ++owner.windows; owner.source_bytes += uint64_t(rows * K * 4);
    owner.routes.reset(); // DMA AND all kernel readers are retired; callback/MTP can reuse arenas
    return true;
} catch (const std::exception& exception) {
    err = std::string("prefill MoE window: ") + exception.what();
    return false; // outer checked lease retires, retaining context and all upload owners
}
