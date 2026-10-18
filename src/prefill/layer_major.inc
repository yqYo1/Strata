// Included inside strata::prefill after the current chunk implementation's
// helpers. This is orchestration only; run_impl owns GDN/QSA/PLE/expert math.
bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    const char* requested = std::getenv("STRATA_PREFILL_LAYER_MAJOR");
    const char* phase_switch = std::getenv("STRATA_PREFILL_PHASE_AUDIT");
    if (phase_switch && std::strcmp(phase_switch, "1") == 0 &&
        (!requested || std::strcmp(requested, "1") || !impl_ || n <= impl_->T)) {
        err = "prefill phase audit: requires the long-prompt layer-major path"; return false;
    }
    if (activate_native_dense_layer &&
        (!requested || std::strcmp(requested, "1") || !impl_ || n <= impl_->T)) {
        err = "prefill dense bank: hook requires long layer-major request"; return false;
    }
    // Reuse only pays across chunks. Keep short chat/agent turns on the
    // original path even when long-prompt layer reuse is requested.
    if (requested && std::strcmp(requested, "1") == 0 && impl_ && n > impl_->T) {
#if defined(STRATA_ENABLE_SYCL)
        // Read on the calling thread before publishing the thread-local scope.
        const char* cpu_wait_switch = std::getenv("STRATA_PREFILL_CPU_WAIT_PROFILE");
        sycl_backend::cpu_wait_profile::Scope cpu_wait_scope(
            cpu_wait_switch && std::strcmp(cpu_wait_switch, "1") == 0);
#endif
        const bool ok = run_layer_major(tokens, n, pos0, err);
#if defined(STRATA_ENABLE_SYCL)
        cpu_wait_scope.complete(ok);
#endif
        return ok;
    }
    return run_impl(tokens, n, pos0, err);
}

bool Prefill::run_layer_major(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) try {
    err.clear();
#if !defined(STRATA_ENABLE_SYCL)
    err = "prefill layer-major: candidate requires the SYCL engine";
    return false;
#else
    if (!impl_ || !impl_->g || !impl_->ss || !impl_->wt) {
        err = "prefill layer-major: init must complete first"; return false;
    }
    Impl& m = *impl_;
    const core::OnDevice on_device(m.device);
    const auto& g = *m.g;
    auto& ss = *m.ss;
    const bool native16_on = native16::requested();
    const bool gdn_chunk_on=strata::gdn_chunk64::requested();
    const int gdn_columns = gdn_column_partitions();
    if (gdn_columns && (gdn_chunk_on || m.T > 8192)) {
        err="prefill GDN columns requires chunk<=8192 and existing GDN (chunk64 off)"; return false;
    }
    if (gdn_columns)
        std::fprintf(stderr, "strata prefill: GDN columns=%d workgroups=%d lanes=%d epilogue=GPU_FP32_same_order scratch=existing_y\n",
                     gdn_columns, 48 * gdn_columns, 512 / gdn_columns);
    if(gdn_chunk_on && !native16_on) {
        err="prefill chunk64 GDN requires native16 BF16";return false;
    }
    if(gdn_chunk_on)strata::gdn_chunk64::validate_device(m.cs);
    const int window_requested = moe_window_requested();
    const bool window_on = window_requested && !requires_prefix_checkpoints;
    const bool scratch_mixed_on = moe_scratch_mixed_requested();
    if (scratch_mixed_on && (!window_on || window_requested != 10 || !native16_on ||
        m.T != 8192 || !m.scratch_alias || m.borrowed || m.copy != m.cs)) {
        err = "prefill mixed scratch: requires mode10 native16, owned chunk8192 scratch and one queue";
        return false;
    }
    const bool command_profile_on = sycl_backend::command_profile::requested();
    if (command_profile_on && (!window_on || g.n_layers != 48)) {
        err = "prefill command profile requires an admitted MoE window and48 layers"; return false;
    }
    if (window_on && (!native16_on || m.T != 8192 || !m.scratch_alias)) {
        err = "prefill MoE window: requires native16 BF16, chunk8192 and scratch alias"; return false;
    }
    if (window_requested && requires_prefix_checkpoints)
        std::fprintf(stderr, "strata prefill: MoE window disabled for prefix-checkpoint consumer; existing route retained\n");
    const char* window_census_switch = std::getenv("STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS");
    if (window_on && window_census_switch && std::strcmp(window_census_switch, "1") == 0) {
        err = "prefill MoE window: old chunk census does not describe window GEMM jobs"; return false;
    }
    const bool score_gemm_on = qsa_score_gemm::requested();
    // This explicit opt-in is fail-closed at admission. Do not discover an
    // unsupported device after projections/KV/indexer have updated state.
    if (score_gemm_on) qsa_score_gemm::validate_device(m.cs);
    if (score_gemm_on && (m.T != 8192 || m.sel_batch <= 0 || m.sel_batch > 256 ||
                         m.max_blocks != ss.max_cells / 4 + 2 ||
                         std::getenv("STRATA_SELECT_OLD"))) {
        err = "prefill score GEMM: requires chunk8192, at most256 queries, owning session bounds and no SELECT_OLD";
        return false;
    }
    const uint64_t score_arena_bytes = detail::layer_major_mul(uint64_t(m.T), 6144 * 2);
    const uint64_t score_workspace_bytes = score_gemm_on ?
        qsa_score_gemm::workspace_bytes(m.sel_batch, m.max_blocks) : 0;
    if (score_workspace_bytes > score_arena_bytes) {
        err = "prefill score GEMM: workspace exceeds the real attn_h reservation"; return false;
    }
    const bool qsa16_on = native16::qsa::requested();
    const auto qsa_math = native16::qsa::attention_math();
    if (qsa_math != native16::qsa::AttentionMath::existing && !qsa16_on) {
        err = "prefill BF16 QSA math requires STRATA_SYCL_PREFILL_NATIVE16_QSA=1"; return false;
    }
    const bool qsa_prefix_on = qsa_math == native16::qsa::AttentionMath::bf16_prefix ||
        qsa_math == native16::qsa::AttentionMath::bf16_prefix_optimized ||
        qsa_math == native16::qsa::AttentionMath::bf16_prefix_qcache ||
        qsa_math == native16::qsa::AttentionMath::bf16_prefix_kvec4 ||
        qsa_math == native16::qsa::AttentionMath::bf16_prefix_kvvec4 ||
        qsa_math == native16::qsa::AttentionMath::bf16_prefix_ordered_exp ||
        qsa_math == native16::qsa::AttentionMath::bf16_prefix_subgroup ||
        qsa_math == native16::qsa::AttentionMath::bf16_prefix_tilepack;
    if (qsa16_on && !native16_on) {
        err = "prefill QSA16: requires STRATA_SYCL_PREFILL_NATIVE16=bf16"; return false;
    }
    const char* residual_option = std::getenv("STRATA_PREFILL_LAYER_MAJOR_R_DTYPE");
    if (native16_on && residual_option && std::strcmp(residual_option, "bf16")) {
        err = "prefill native16: residual storage must be bf16 (or leave R_DTYPE unset)"; return false;
    }
    const auto residual_dtype = native16_on ? detail::LayerMajorResidualDtype::bf16 :
        detail::layer_major_residual_dtype(residual_option);
    if (!native16_on && residual_dtype == detail::LayerMajorResidualDtype::bf16) {
        err = "prefill: BF16 residuals require STRATA_SYCL_PREFILL_NATIVE16=bf16"; return false;
    }
    const char* phase_switch = std::getenv("STRATA_PREFILL_PHASE_AUDIT");
    const bool phase_requested = phase_switch && std::strcmp(phase_switch, "1") == 0;
    if (native16_on) {
        const char* boundary = std::getenv("STRATA_PREFILL_BOUNDARY_AUDIT");
        const bool boundary_requested = boundary && std::strcmp(boundary, "1") == 0;
        if (phase_requested || boundary_requested ||
            std::getenv("STRATA_PREFILL_DUMP_R") || gr_unfused() ||
            m.xn16_lo || m.lo16_lo || m.mixed_bf_lo) {
            err = "prefill native16: old float tensor audits/dump, unfused HC and BF16x2 are unsupported"; return false;
        }
        for (int64_t l = 0; l < g.n_layers; ++l)
            if (strata::kernels::cvec().covers(l)) {
                err = "prefill native16: control vector requires the FP32 reference"; return false;
            }
    }
    if (phase_requested && residual_dtype != detail::LayerMajorResidualDtype::fp32) {
        err = "prefill phase audit: FP16 saved residual storage is outside this diagnostic"; return false;
    }
    const char* mmq = std::getenv("STRATA_PREFILL_MMQ");
    if (m.layer_major_active || stage_lb_ != 0 || stage_le_ != g.n_layers || next_ || hand_in_ ||
        m.pp || core::peer_portable() || embd_rows || !m.src || !m.compact_moe_fp16 ||
        !mmq || std::strcmp(mmq, "0") || mmq_plan().any || !mmq_plan().fallback ||
        m.copy != m.cs) {
        err = "prefill layer-major: requires full single-GPU compact FP16, native source, explicit MMQ=0, "
              "same queue, no peer/stage/hand-off/vision/reentry";
        return false;
    }
    if (!tokens || n <= 0 || n > 262144 || pos0 < 0 || ss.max_cells <= 0 || ss.max_cells > 262144 ||
        pos0 > ss.max_cells || n > ss.max_cells - pos0 || m.T <= 0 || m.T > 8192 ||
        m.T_max < m.T || m.T_max > 8192 ||
        g.n_layers != 48 || g.qsa_interval != 4 || g.n_embd != N || g.hc != HC || g.hc_lr != LR ||
        g.n_expert != NE || g.n_ff != 640 || ss.k != K ||
        g.n_head != 24 || g.n_head_kv != 2 || g.head_dim != 256 ||
        g.idx_q_heads != 4 || g.idx_key_dim != 128 || g.ssm_state_size != 128 ||
        g.ssm_k_heads != 16 || g.ssm_v_heads != HV || g.ssm_d_conv != 4 ||
        g.ssm_conv_channels != C || g.ssm_value_dim != ZV ||
        ss.layer_lo != 0 || ss.layer_hi != g.n_layers || ss.qsa_ord0 != 0 || ss.gdn_ord0 != 0 ||
        ss.qsa_alloc != 12 || ss.gdn_alloc != 36 || !ss.qsa_states || !ss.gdn_state || !m.R) {
        err = "prefill layer-major: unsupported geometry, session range, position or chunk capacity";
        return false;
    }
    for (int64_t t = 0; t < n; ++t)
        if (tokens[t] < 0 || tokens[t] > std::numeric_limits<int32_t>::max()) {
            err = "prefill layer-major: token id is outside the native int32 input range"; return false;
        }
    if(gdn_chunk_on)
        std::fprintf(stderr,"[gdn-chunk64] long-prefill-only=1 chunk=%lld arena=%zu BF16_projection_conv=1 FP32_history_state=1 experimental=1\n",
                     (long long)m.T,strata::gdn_chunk64::arena_bytes(int(m.T)));
    // Existing observers assume one full-model call. Refuse ambiguous reports
    // rather than emitting 48 apparent full-prompt measurements/checkpoints.
    for (const char* name : {"STRATA_PREFILL_ROUTED_FP16_OBSERVE", "STRATA_PREFILL_TIMING",
                             "STRATA_PREFILL_ATTN_LAYOUT", "STRATA_DBG_NAN",
                             "STRATA_STATE_HASH_GDN", "STRATA_PREFILL_PRELOAD_PLE",
                             "STRATA_PREFILL_RELEASE_CACHE", "STRATA_PREFILL_CACHE_RELEASE_FRAC",
                             "STRATA_PREFILL_CACHE_RESTORE", "STRATA_PREFILL_LAYER_MAJOR_R_INPLACE"})
        if (std::getenv(name)) {
            err = std::string("prefill layer-major: unsupported observer/legacy option ") + name; return false;
        }
    const auto& layout = strata::kernels::cpu::expert_layout();
    if (!layout.native || layout.n_layers != g.n_layers || layout.n_expert != NE ||
        layout.fmt.size() != size_t(g.n_layers) || layout.bytes.size() != layout.fmt.size() ||
        layout.offset.size() != layout.fmt.size() || !layout.max_blob) {
        err = "prefill layer-major: requires a complete native expert layout"; return false;
    }
    uint64_t previous_end = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const auto& f = layout.fmt[size_t(l)];
        const uint64_t bytes = layout.blob_bytes(l);
        const uint64_t end = detail::layer_major_add(layout.layer_offset(l), detail::layer_major_mul(NE, bytes));
        if (!bytes || bytes > layout.max_blob || bytes != f.bytes || f.n_embd != N || f.n_ff != 640 ||
            f.gu_type < 0 || f.d_type < 0 || !f.gu_row || !f.d_row ||
            layout.layer_offset(l) < previous_end || end > layout.total) {
            err = "prefill layer-major: invalid native source table/blob span"; return false;
        }
        previous_end = end;
    }
    const auto& primary = ss.qsa_states[0];
    for (int64_t i = 0; i < ss.qsa_alloc; ++i) {
        const auto& q = ss.qsa_states[i];
        if ((q.kv_mode != 0 && q.kv_mode != 1) || q.kv_hybrid || q.max_cells != ss.max_cells ||
            q.n_pages != (ss.max_cells + 3) / 4 || q.n_slots <= 0 || q.n_slots > q.n_pages ||
            (q.kv_mode == 0 && q.n_slots != q.n_pages) || !q.page_table ||
            q.kv_mode != primary.kv_mode || q.kv_q4 != primary.kv_q4 ||
            q.kv_int8 != primary.kv_int8 || q.kv_rot != primary.kv_rot ||
            q.idx_pooled_rows < ss.max_cells / 4 + 2) {
            err = "prefill layer-major: requires uniform full-resident or mode-1 streamed native KV"; return false;
        }
        if (q.kv_mode == 1) {
            const bool storage = q.kv_q4 ? q.host.k_q4 && q.host.v_q4 && m.stage.k_q4 && m.stage.v_q4 :
                q.kv_int8 ? q.host.k_q && q.host.v_q && q.host.k_scale && q.host.v_scale &&
                            m.stage.k_q && m.stage.v_q && m.stage.k_scale && m.stage.v_scale :
                            q.host.k_pool && q.host.v_pool && m.stage.k_pool && m.stage.v_pool;
            if (!storage || !m.ident_table) {
                err = "prefill layer-major: full-logical-range host/stage KV is unavailable"; return false;
            }
        }
    }
    // QSA16 v1 deliberately supports a resident text INT8 cache only. Reject
    // every unsupported layer/device/domain BEFORE this request submits work.
    if (qsa16_on) {
        native16::qsa::validate_device(m.cs);
        // The legacy OLD switch is presence-based, including a value of 0.
        if (std::getenv("STRATA_PROMPT_ATTN_OLD")) {
            err = "prefill QSA16: leave STRATA_PROMPT_ATTN_OLD unset"; return false;
        }
        const char* mapping = std::getenv("STRATA_SYCL_PROMPT_K_CELL_MAJOR");
        if (mapping && std::strcmp(mapping, "0")) {
            err = "prefill QSA16: requires prompt Mapping0"; return false;
        }
        const auto tab = strata::kernels::rope_table_for(strata::kernels::rope_scaling());
        if (strata::kernels::mrope_table() || !tab.cos) {
            err = "prefill QSA16: requires text positions and STRATA_ROPE_TABLE=1"; return false;
        }
        const auto shapes = strata::kernels::qsa_real_shapes();
        for (int64_t i = 0; i < ss.qsa_alloc; ++i) {
            const auto& q = ss.qsa_states[i];
            if (q.kv_mode != 0 || !q.kv_int8 || q.kv_rot || q.kv_q4 || q.kv_hybrid || q.host.present() ||
                q.n_slots != q.n_pages || tab.cos != q.cos_tab || tab.sin != q.sin_tab ||
                tab.max_pos != q.max_cells) {
                err = "prefill QSA16: requires resident unrotated INT8 K/V and the owning session RoPE tables";
                return false;
            }
            sycl_backend::validate_device_span(q.cos_tab, size_t(q.max_cells) * 32 * sizeof(float),
                                               sycl_backend::queue_for(m.cs));
            sycl_backend::validate_device_span(q.sin_tab, size_t(q.max_cells) * 32 * sizeof(float),
                                               sycl_backend::queue_for(m.cs));
            // Owner n_slots sizes physical pools, n_pages sizes the table.
            // max_cells is NOT used to guess allocated physical pages.
            if (!strata::kernels::qsa_prompt_attn_bf16_validate(m.qsa_q16, core::qsa_attn_pools(q),
                    m.sel_ids, m.steps_dev, m.cap, shapes, m.qsa_attn16, m.T,
                    {q.n_pages, q.n_slots}, m.cs)) {
                err = "prefill QSA16: typed attention capability/geometry unsupported"; return false;
            }
        }
    }
    // This route never submits a Stager job. Do not enter over a live previous
    // job, and do not use its unchecked finish()/event waits as a retirement proof.
    if (!m.stager || m.stager->active.load(std::memory_order_acquire) != 0) {
        err = "prefill layer-major: host stager is not idle"; return false;
    }
    const uint64_t stager_head = m.stager->head.load(std::memory_order_acquire);
    if ((stager_head & 0xffff) < ((stager_head >> 16) & 0xffff)) {
        err = "prefill layer-major: host stager still has claimable jobs"; return false;
    }
    const auto started = Clock::now();
    const auto old_stats = stats_;
    const int32_t initial_prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    int32_t final_prev[2] = {initial_prev[0], initial_prev[1]};
    for (int64_t t = 0; t < n; ++t) { final_prev[0] = final_prev[1]; final_prev[1] = int32_t(tokens[t]); }

    // Preserve decode cache, target/draft workspaces and captured graph addresses.
    // All new allocations are additional. Headroom is finite and explicit.
    constexpr uint64_t gpu_headroom = 512ull << 20, host_headroom = 8ull << 30;
    const uint64_t stride = detail::layer_major_add(layout.max_blob, 255) / 256 * 256;
    const uint64_t host_slab_bytes = detail::layer_major_mul(NE, stride);
    const char* bank_option = std::getenv("STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS");
    const int64_t requested_slots = bank_option ? detail::layer_major_requested_rows(bank_option) : 0;
    if (requested_slots < 0 || requested_slots > NE) {
        err = "prefill expert bank: slots must be decimal0..512"; return false;
    }
    const uint64_t map_bytes = detail::layer_major_mul(detail::layer_major_mul(g.n_layers, NE), sizeof(int32_t));
    // Exact transient live-prefix storage: two BF16 tensors plus one INT32
    // diagnostic per [logical cell,KV head]. Reserve cumulatively with the
    // expert cache and residual plan before submitting this request's work.
    const auto qsa_page = strata::kernels::qsa_real_shapes().page_size;
    const uint64_t prefix_cells = detail::layer_major_mul(
        (pos0 + n + qsa_page - 1) / qsa_page, qsa_page);
    const uint64_t prefix_tensor_bytes = qsa_prefix_on ?
        detail::layer_major_mul(detail::layer_major_mul(prefix_cells, 2 * 256), 2) : 0;
    const uint64_t prefix_flag_bytes = qsa_prefix_on ? detail::layer_major_mul(prefix_cells, 2 * 4) : 0;
    const uint64_t prefix_bytes = detail::layer_major_add(
        detail::layer_major_mul(prefix_tensor_bytes, 2), prefix_flag_bytes);
    // Half/head scratch borrows attn_h only BEFORE gate overwrites it. Qf is
    // still a gate INPUT and must not be reused. Only sticky status is additional.
    const uint64_t score_status_bytes = score_gemm_on ? 256 : 0;
    const uint64_t other_reserved = detail::layer_major_add(prefix_bytes, score_status_bytes);
    size_t free_gpu = 0, total_gpu = 0;
    const cudaError_t memory_status = cudaMemGetInfo(&free_gpu, &total_gpu);
    if (memory_status != cudaSuccess || free_gpu > total_gpu) {
        err = "prefill layer-major: extra expert cache (" + std::to_string(host_slab_bytes) +
              " bytes), BF16 prefix (" + std::to_string(prefix_bytes) + " bytes), score status (" +
              std::to_string(score_status_bytes) + " bytes) plus 512 MiB headroom exceeds known free VRAM (" + std::to_string(free_gpu) + ")";
        return false;
    }
    const auto max_alloc = sycl_backend::queue_for(m.cs).get_device()
        .get_info<sycl::info::device::max_mem_alloc_size>();
    const auto bank_plan = detail::ExpertBankPlan::make(n, m.T, D, stride, free_gpu,
        other_reserved, gpu_headroom, max_alloc,
        detail::layer_major_requested_rows(std::getenv("STRATA_PREFILL_LAYER_MAJOR_R_GPU")),
        residual_dtype, requested_slots,
        window_on && window_requested == 10 && native16_on && scratch_mixed_on);
    const auto& plan = bank_plan.residual;
    const uint64_t cache_bytes = bank_plan.device_bytes;
    const uint64_t gpu_reserved = detail::layer_major_add(cache_bytes, other_reserved);
    // Keep residual placement fixed for the first comparison. Admit an owned
    // FP16 mixed bank only from the remaining budget, retaining 512 MiB.
    int64_t accumulation_rows = 0;
    uint64_t accumulation_bytes = 0;
    detail::MoeAccumPlan borrowed_mixed_plan;
    if (window_on && window_requested >= 8 && plan.gpu_rows > m.T) {
        const int64_t count = std::min<int64_t>(int64_t(window_requested) * m.T, plan.gpu_rows);
        const uint64_t bytes = detail::layer_major_mul(uint64_t(count), N * 2);
        const uint64_t fixed = detail::layer_major_add(gpu_reserved,
            detail::layer_major_add(plan.gpu_bytes, detail::layer_major_add(plan.staging_bytes, gpu_headroom)));
        std::fprintf(stderr, "strata MoE accumulator admission: requested_mode=%d rows=%lld gpu_prefix_rows=%lld shared_arena_bytes=%llu free_gpu_bytes=%llu fixed_including_headroom_bytes=%llu mixed_fp16_bytes=%llu\n",
            window_requested, (long long)count, (long long)plan.gpu_rows,
            (unsigned long long)m.region_bytes, (unsigned long long)free_gpu,
            (unsigned long long)fixed, (unsigned long long)bytes);
        if (fixed <= free_gpu && (scratch_mixed_on || bytes <= free_gpu - fixed)) {
            if (scratch_mixed_on) {
                // These are independently owned persistent scratch allocations,
                // idle only inside the FFN helper after its attention drain.
                auto owns = [&](const void* pointer) {
                    return std::find(m.owned.begin(), m.owned.end(), pointer) != m.owned.end();
                };
                if (!owns(m.R) || !owns(m.region) || !owns(m.emb)) {
                    err = "prefill mixed scratch: direct Prefill owners required"; return false;
                }
                borrowed_mixed_plan = detail::MoeAccumPlan::make_scratch_borrowed(count,
                    detail::layer_major_mul(uint64_t(m.T), D * 4), m.region_bytes);
                accumulation_rows = count; // no additional mixed allocation
            } else {
                (void)detail::MoeAccumPlan::make_contiguous(count, bytes, m.region_bytes);
                accumulation_rows = count; accumulation_bytes = bytes;
            }
        }
    }
    if (window_on && window_requested >= 8 && !accumulation_rows)
        std::fprintf(stderr, "strata prefill: MoE window%d not admitted by remaining VRAM/GPU-prefix budget; effective_mode=2\n", window_requested);
    if (command_profile_on && window_requested >= 8 && !accumulation_rows) {
        err = "prefill command profile: requested accumulator window was not admitted"; return false;
    }
    if (bank_plan.staged() && (!accumulation_rows || plan.gpu_rows != n ||
        accumulation_rows != std::min<int64_t>(detail::ExpertBankPlan::window_rows, n))) {
        err = "prefill expert bank: every FFN piece must use the all-GPU accumulator"; return false;
    }
    // One owned host upload slab removes every source pointer lease before DMA.
    // Also count the cache's small blocking staging allocation and two maps.
    uint64_t host_need = detail::layer_major_add(plan.host_bytes, host_slab_bytes);
    host_need = detail::layer_major_add(host_need, stride);
    host_need = detail::layer_major_add(host_need, detail::layer_major_mul(2, map_bytes));
    if (window_on) host_need = detail::layer_major_add(host_need,
        accumulation_rows ? 16ull << 20 : 2ull << 20); // IDs, immutable routes/jobs/inverse
    if (command_profile_on) host_need = detail::layer_major_add(host_need, 128ull << 10);
    if (!core::conversation_memory_admit(core::conversation_available_memory(), host_need, host_headroom)) {
        err = "prefill layer-major: residual/upload storage exceeds known available RAM minus 8 GiB headroom";
        return false;
    }
    if (should_stop && should_stop()) { err = "cancelled"; return false; }
    core::ExpertCache layer_cache;
    std::unique_ptr<uint8_t[]> upload(new uint8_t[size_t(host_slab_bytes)]);
    std::unique_ptr<uint8_t[]> rows(plan.host_bytes ? new uint8_t[size_t(plan.host_bytes)] : nullptr);
    std::vector<int32_t> residency(size_t(g.n_layers * NE), -1);
    auto free_device = [](uint8_t* p) { if (p) (void) cudaFree(p); };
    std::unique_ptr<uint8_t, decltype(free_device)> device_rows(nullptr, free_device);
    std::unique_ptr<uint8_t, decltype(free_device)> prefix_storage(nullptr, free_device);
    std::unique_ptr<uint8_t, decltype(free_device)> mixed_storage(nullptr, free_device);
    std::unique_ptr<uint8_t, decltype(free_device)> score_status_storage(nullptr, free_device);
    // DMA destination precedes the checked lease: even a failed copy/drain
    // cannot end its lifetime before the lease's fail-stop retirement.
    uint32_t score_status_observed = 0;
    // Allocated only for an FP16 host tail. One in-order queue serializes
    // H2D -> unpack -> model -> pack -> D2H, and the outer lease owns it all.
    std::unique_ptr<uint8_t, decltype(free_device)> fp16_stage(nullptr, free_device);

    // Declare before Restore/lease: diagnostic DMA host owners survive checked retirement.
    std::unique_ptr<detail::LayerMajorBoundaryAudit> audit;
    const char* audit_switch = std::getenv("STRATA_PREFILL_BOUNDARY_AUDIT");
    if (audit_switch && std::strcmp(audit_switch, "1") == 0) {
        // Even the parser's maximum64 rows, metadata and scratch fit below4MiB.
        // Account this optional host owner before allocation, retaining8GiB.
        if (!core::conversation_memory_admit(core::conversation_available_memory(),
                detail::layer_major_add(host_need, 4ull << 20), host_headroom)) {
            err = "prefill boundary audit: diagnostic storage exceeds RAM headroom"; return false;
        }
        audit.reset(new detail::LayerMajorBoundaryAudit(n, pos0, D, g.n_layers, m.T,
            plan.dtype == detail::LayerMajorResidualDtype::fp16));
    }
    // This owner precedes Restore and the checked lease, just like boundary audit.
    // Reserve BOTH optional owners cumulatively; no file/raw bytes are held in RAM.
    std::unique_ptr<detail::Layer26PhaseAudit> phase_audit;
    if (phase_requested) {
        uint64_t need = detail::layer_major_add(host_need, audit ? 4ull << 20 : 0);
        need = detail::layer_major_add(need, detail::Layer26PhaseAudit::host_reservation);
        if (!core::conversation_memory_admit(core::conversation_available_memory(), need, host_headroom)) {
            err = "prefill phase audit: additional diagnostic storage exceeds RAM headroom"; return false;
        }
        phase_audit.reset(new detail::Layer26PhaseAudit(n, pos0, m.T, m.mixed_bf_lo != nullptr));
    }
    // Optional owner precedes borrowed pointer, Restore and checked lease.
    // Failure to qualify/allocate the observation does not change inference.
    std::unique_ptr<detail::LayerMajorShapeCensus> census;
    const char* census_switch = std::getenv("STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS");
    if (census_switch && std::strcmp(census_switch, "1") == 0) {
        uint64_t need = detail::layer_major_add(host_need, audit ? 4ull << 20 : 0);
        need = detail::layer_major_add(need, phase_audit ? detail::Layer26PhaseAudit::host_reservation : 0);
        need = detail::layer_major_add(need, sizeof(detail::LayerMajorShapeCensus));
        if (core::conversation_memory_admit(core::conversation_available_memory(), need, host_headroom))
            census.reset(new (std::nothrow) detail::LayerMajorShapeCensus(n, pos0, m.T));
        if (!census) std::fputs("strata lm census: {\"version\":1,\"complete\":false,\"reason\":\"host_observation_unavailable\"}\n", stderr);
    }
    detail::LayerMajorShapeBorrow census_borrow(m.layer_major_census, census.get());
    const char* profile_switch = std::getenv("STRATA_PREFILL_LAYER_MAJOR_PROFILE");
    const bool profile_requested = profile_switch && std::strcmp(profile_switch, "1") == 0;
    const char* gdn_profile_switch = std::getenv("STRATA_PREFILL_LAYER_MAJOR_GDN_PROFILE");
    const bool gdn_profile_requested = gdn_profile_switch && std::strcmp(gdn_profile_switch, "1") == 0;
    if (gdn_profile_requested && !profile_requested) {
        err = "prefill GDN profile: requires STRATA_PREFILL_LAYER_MAJOR_PROFILE=1"; return false;
    }
    std::unique_ptr<detail::LayerMajorCoarseProfile> profile;
    if (profile_requested) {
        uint64_t need = detail::layer_major_add(host_need, audit ? 4ull << 20 : 0);
        need = detail::layer_major_add(need, phase_audit ? detail::Layer26PhaseAudit::host_reservation : 0);
        need = detail::layer_major_add(need, census ? sizeof(detail::LayerMajorShapeCensus) : 0);
        need = detail::layer_major_add(need, 1ull << 20); // fixed rows, at most twenty small compat event owners
        if (!core::conversation_memory_admit(core::conversation_available_memory(), need, host_headroom)) {
            err = "prefill layer-major profile: diagnostic storage exceeds RAM headroom"; return false;
        }
        profile.reset(new detail::LayerMajorCoarseProfile);
        profile->gdn_detail = gdn_profile_requested;
        if (gdn_profile_requested)
            for (int l = 0; l < 48; ++l)
                if (!core::is_qsa_layer(g, l)) profile->rows[l].gdn_expected = uint64_t(1 + (n - 1) / m.T);
    }
    detail::LayerMajorCoarseBorrow profile_borrow(m.layer_major_profile, profile.get());
    // Read only: callers configure the native process environment before exec.
    // A short warmup never enters run_layer_major and allocates no observer.
    const char* qsa_profile_switch = std::getenv("STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE");
    const bool qsa_profile_requested = qsa_profile_switch && std::strcmp(qsa_profile_switch, "1") == 0;
    std::unique_ptr<detail::LayerMajorQsaProfile> qsa_profile;
    if (qsa_profile_requested) {
        if (!profile_requested) {
            err = "prefill QSA profile: requires STRATA_PREFILL_LAYER_MAJOR_PROFILE=1"; return false;
        }
        uint64_t need = detail::layer_major_add(host_need, audit ? 4ull << 20 : 0);
        need = detail::layer_major_add(need, phase_audit ? detail::Layer26PhaseAudit::host_reservation : 0);
        need = detail::layer_major_add(need, census ? sizeof(detail::LayerMajorShapeCensus) : 0);
        need = detail::layer_major_add(need, 2ull << 20); // coarse + fine fixed owners, fourteen extra events
        if (!core::conversation_memory_admit(core::conversation_available_memory(), need, host_headroom)) {
            err = "prefill QSA profile: diagnostic storage exceeds RAM headroom"; return false;
        }
        qsa_profile.reset(new detail::LayerMajorQsaProfile);
        for (int l = 0; l < 48; ++l) {
            auto& row = qsa_profile->rows[l];
            if (core::is_qsa_layer(g, l)) {
                row.expected = uint64_t(1 + (n - 1) / m.T);
                row.expected_queries = uint64_t(n);
                row.cap = m.cap;
            }
        }
    }
    // Reverse destruction: lease, Restore, this borrow, fine owner, coarse borrow,
    // coarse owner. Failed retirement aborts while all event owners remain live.
    detail::LayerMajorQsaBorrow qsa_profile_borrow(m.layer_major_qsa_profile, qsa_profile.get());
    using LmProfile = detail::LayerMajorCoarseProfile;
    int64_t logical_chunks = 0, layer = 0;
    const int64_t final_chunk_row = ((n - 1) / m.T) * m.T;
    bool stage_delivered = false;
    detail::LayerMajorChunkCursor cursor{n, m.T};

    std::unique_ptr<MoeWindowContext> window_context;
    if (window_on) window_context.reset(new MoeWindowContext);
    if (command_profile_on)
        window_context->command_trace.reset(new sycl_backend::command_profile::Trace(m.cs));
    struct PrefixBorrow {
        Impl& m; strata::kernels::QsaBf16PackedScratch previous;
        ~PrefixBorrow() noexcept { m.qsa_prefix = previous; }
    } prefix_borrow{m, m.qsa_prefix};
    struct ScoreBorrow {
        Impl& m; qsa_score_gemm::Scratch previous;
        ~ScoreBorrow() noexcept { m.score_gemm = previous; }
    } score_borrow{m, m.score_gemm};
    // Destruction order: checked drain -> field/callback restoration -> owners.
    // A failed drain aborts here, with all captures/storage still alive. Session
    // tensors are NOT rolled back by restoring these borrowed C++ fields.
    struct Restore {
        Prefill& p; Impl& m;
        const core::LayerView::Overlay* overlay;
        int64_t lb, le; const float* input;
        const core::ExpertCache* cache; const int32_t* residency;
        detail::Layer26PhaseAudit* phase;
        detail::LayerMajorBoundaryAudit* audit;
        detail::LayerMajorBoundaryAudit::Copy audit_copy;
        bool active; const void* gpu_input; const void* host_input; uint16_t* fp16_stage;
        detail::LayerMajorResidualPlan plan;
        decltype(on_chunk) chunk; decltype(on_stage_chunk) stage;
        bool native16_on; decltype(m.on_chunk16) chunk16; MoeWindowContext* window;
        ~Restore() noexcept {
            m.native_dense_overlay = overlay;
            p.stage_lb_ = lb; p.stage_le_ = le; p.hand_in_ = input;
            m.cache = cache; m.host_res = residency; m.layer_major_active = active;
            m.layer26_phase_audit = phase;
            m.layer_major_audit = audit; m.layer_major_audit_copy = std::move(audit_copy);
            m.layer_major_gpu_input = gpu_input; m.layer_major_host_input = host_input;
            m.layer_major_fp16_stage = fp16_stage; m.layer_major_residual = plan;
            p.on_chunk = std::move(chunk); p.on_stage_chunk = std::move(stage);
            m.native16_active = native16_on; m.on_chunk16 = std::move(chunk16); m.moe_window = window;
        }
    } restore{*this, m, m.native_dense_overlay, stage_lb_, stage_le_, hand_in_, m.cache, m.host_res,
              m.layer26_phase_audit, m.layer_major_audit, m.layer_major_audit_copy,
              m.layer_major_active, m.layer_major_gpu_input, m.layer_major_host_input,
              m.layer_major_fp16_stage, m.layer_major_residual, on_chunk, on_stage_chunk,
              m.native16_active, m.on_chunk16, m.moe_window};
    cudaError_t compute_drain = cudaSuccess, copy_drain = cudaSuccess;
    auto drain = [&]() noexcept {
        const auto profile_drain_start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};
        compute_drain = cudaStreamSynchronize(m.cs);
        copy_drain = m.copy == m.cs ? compute_drain : cudaStreamSynchronize(m.copy);
        if (profile) profile->add(LmProfile::wrapper_drain, profile_drain_start);
        return compute_drain == cudaSuccess && copy_drain == cudaSuccess;
    };
    detail::DraftUploadGuard lease(drain, [&]() noexcept {
        std::fprintf(stderr, "prefill layer-major: cannot prove queue retirement (compute=%d copy=%d); "
                             "aborting with temporary owners live\n", int(compute_drain), int(copy_drain));
        std::fflush(stderr); std::abort();
    });
    lease.arm();
    if (!drain()) { err = "prefill layer-major: entry queue drain failed"; return false; }
    if (profile) profile->open();
    if (qsa_profile) qsa_profile->open();
    if (audit || phase_audit) {
        m.layer26_phase_audit = phase_audit.get();
        m.layer_major_audit = audit.get();
        m.layer_major_audit_copy = [&](void* dst, const void* src, size_t bytes, bool device) {
            if (!device) { std::memcpy(dst, src, bytes); return true; }
            const auto status = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, m.cs);
            const bool retired = drain(); // check even when submission reported an error
            return status == cudaSuccess && retired;
        };
    }
    // Allocation-only cache opening is essential: no memset may fail and free
    // its allocation before the outer checked lease has retired its readers.
    if (!layer_cache.open_uninitialized(bank_plan.slots, 1, NE, int64_t(stride), err)) return false;
    if (score_gemm_on) {
        uint8_t* storage = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&storage), size_t(score_status_bytes)) != cudaSuccess) {
            err = "prefill score GEMM: status allocation failed before model work"; return false;
        }
        score_status_storage.reset(storage);
        auto& queue = sycl_backend::queue_for(m.cs);
        sycl_backend::validate_device_span(m.attn_h, size_t(score_arena_bytes), queue);
        sycl_backend::validate_device_span(storage, size_t(score_status_bytes), queue);
        // Check the actual carved topology, including still-live gate inputs.
        // attn_h is future output; Qf, q and q_idx remain independent inputs.
        sycl_backend::validate_spans({{m.attn_h, size_t(score_arena_bytes)},
                                     {storage, size_t(score_status_bytes)}},
            {{m.Qf, size_t(m.T) * 12288 * sizeof(float)},
             {m.q, size_t(m.T) * 6144 * sizeof(float)},
             {m.q_idx, size_t(m.T) * 512 * sizeof(float)},
             {m.attn, size_t(m.T) * 6144 * sizeof(float)},
             {m.sel_scores, size_t(m.sel_batch) * size_t(m.max_blocks) * sizeof(float)},
             {m.sel_ids, size_t(m.T) * size_t(m.cap) * sizeof(int32_t)},
             {m.steps_dev, size_t(m.T) * strata::kernels::kStepCount * sizeof(int32_t)}});
        for (int64_t i = 0; i < ss.qsa_alloc; ++i) {
            const auto& state = ss.qsa_states[i];
            sycl_backend::validate_spans({{m.attn_h, size_t(score_arena_bytes)},
                                         {storage, size_t(score_status_bytes)}},
                {{state.idx_pooled, size_t(state.idx_pooled_rows) * 128 * sizeof(float)},
                 {state.idx_dead, 128 * sizeof(float)}});
        }
        m.score_gemm = qsa_score_gemm::carve(m.attn_h, size_t(score_arena_bytes),
            reinterpret_cast<uint32_t*>(storage), m.sel_batch, m.max_blocks);
        // No score invocation resets this sticky word. It is outside the arena
        // reused by gate, MoE and GDN after score readers have retired in order.
        if (cudaMemsetAsync(storage, 0, 4, m.cs) != cudaSuccess) {
            err = "prefill score GEMM: status initialization failed"; return false;
        }
        std::fprintf(stderr, "strata QSA score prefill: gemm_fp16=1 arena=attn_h workspace_bytes=%llu independent_status_bytes=%llu sticky_status=1\n",
            (unsigned long long)score_workspace_bytes, (unsigned long long)score_status_bytes);
    }
    if (prefix_bytes) {
        uint8_t* storage = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&storage), size_t(prefix_bytes)) != cudaSuccess) {
            err = "prefill BF16 QSA prefix allocation failed before model work"; return false;
        }
        prefix_storage.reset(storage);
        m.qsa_prefix = {reinterpret_cast<uint16_t*>(storage),
                        reinterpret_cast<uint16_t*>(storage + prefix_tensor_bytes),
                        reinterpret_cast<int32_t*>(storage + 2 * prefix_tensor_bytes)};
    }
    if (qsa_math != native16::qsa::AttentionMath::existing)
        std::fprintf(stderr, "strata prefill: QSA BF16 operands, FP32 online state/accumulation; "
                     "KV=%s transient_bytes=%llu; source=resident-Mode0 score_pitch=%d operand_layout=%s softmax=%s\n",
                     qsa_prefix_on ? "active-logical-prefix" : "on-the-fly",
                     (unsigned long long) prefix_bytes,
                     (qsa_math == native16::qsa::AttentionMath::bf16_prefix_optimized ||
                      qsa_math == native16::qsa::AttentionMath::bf16_prefix_subgroup ||
                      qsa_math == native16::qsa::AttentionMath::bf16_prefix_qcache ||
                      qsa_math == native16::qsa::AttentionMath::bf16_prefix_kvec4 ||
                      qsa_math == native16::qsa::AttentionMath::bf16_prefix_kvvec4 ||
                      qsa_math == native16::qsa::AttentionMath::bf16_prefix_ordered_exp ||
                      qsa_math == native16::qsa::AttentionMath::bf16_prefix_tilepack) ? 33 : 32,
                     qsa_math == native16::qsa::AttentionMath::bf16_prefix_tilepack ? "dense16-vnni" : "row-major",
                     qsa_math == native16::qsa::AttentionMath::bf16_prefix_subgroup ? "subgroup" :
                     qsa_math == native16::qsa::AttentionMath::bf16_prefix_ordered_exp ? "parallel-exp-ascending-sum" : "serial");
    if (qsa_math == native16::qsa::AttentionMath::bf16_prefix_qcache ||
        qsa_math == native16::qsa::AttentionMath::bf16_prefix_kvec4 ||
        qsa_math == native16::qsa::AttentionMath::bf16_prefix_kvvec4 ||
        qsa_math == native16::qsa::AttentionMath::bf16_prefix_ordered_exp)
        std::fprintf(stderr, "strata prefill: QSA cached_query=1 grf_policy=default source=qualified-component\n");
    if (qsa_math == native16::qsa::AttentionMath::bf16_prefix_kvec4 ||
        qsa_math == native16::qsa::AttentionMath::bf16_prefix_kvvec4 ||
        qsa_math == native16::qsa::AttentionMath::bf16_prefix_ordered_exp)
        std::fprintf(stderr, "strata prefill: QSA k_staging=typed4 parallel_rows=1 source=qualified-component\n");
    if (qsa_math == native16::qsa::AttentionMath::bf16_prefix_kvvec4 ||
        qsa_math == native16::qsa::AttentionMath::bf16_prefix_ordered_exp)
        std::fprintf(stderr, "strata prefill: QSA v_staging=typed4 layout=row-major source=qualified-component\n");
    if (plan.gpu_bytes) {
        uint8_t* p = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&p), size_t(plan.gpu_bytes)) != cudaSuccess) {
            err = "prefill layer-major: extra GPU residual allocation failed"; return false;
        }
        device_rows.reset(p);
    }
    if (plan.staging_bytes) {
        uint8_t* p = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&p), size_t(plan.staging_bytes)) != cudaSuccess) {
            err = "prefill layer-major: FP16 residual staging allocation failed"; return false;
        }
        fp16_stage.reset(p);
    }
    if (accumulation_bytes) {
        uint8_t* p = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&p), size_t(accumulation_bytes)) != cudaSuccess) {
            err = "prefill MoE accumulator: mixed allocation failed before model work"; return false;
        }
        mixed_storage.reset(p);
    }
    m.layer_major_fp16_stage = reinterpret_cast<uint16_t*>(fp16_stage.get());
    m.layer_major_active = true;
    m.native16_active = native16_on;
    m.layer_major_residual = plan;
    if (window_context) {
        if (bank_plan.staged()) {
            window_context->expert_slots.reset(new detail::ExpertBankSlots(int32_t(bank_plan.slots)));
            window_context->expert_cache = &layer_cache;
            window_context->expert_slab = upload.get();
            window_context->expert_slab_bytes = host_slab_bytes;
            window_context->expert_stride = stride;
        }
        window_context->bank = reinterpret_cast<uint16_t*>(device_rows.get());
        window_context->bank_bytes = plan.gpu_bytes;
        window_context->mixed = reinterpret_cast<uint16_t*>(mixed_storage.get());
        window_context->mixed_bytes = accumulation_bytes;
        if (scratch_mixed_on && accumulation_rows) {
            window_context->scratch_mixed = true;
            window_context->mixed = reinterpret_cast<uint16_t*>(m.R);
            window_context->mixed_bytes = borrowed_mixed_plan.primary_bytes;
            window_context->mixed_tail_offset = borrowed_mixed_plan.region_bytes;
            window_context->mixed_tail_bytes = borrowed_mixed_plan.tail_bytes;
            window_context->mixed_tail = borrowed_mixed_plan.tail_bytes ?
                reinterpret_cast<uint16_t*>(m.region + size_t(borrowed_mixed_plan.region_bytes)) : nullptr;
            std::fprintf(stderr, "strata MoE mixed banks: storage=scratch primary_bytes=%llu tail_bytes=%llu tail_offset=%llu extra_allocation_bytes=0 phase=FFN_only_after_attention_drain\n",
                (unsigned long long)borrowed_mixed_plan.primary_bytes,
                (unsigned long long)borrowed_mixed_plan.tail_bytes,
                (unsigned long long)borrowed_mixed_plan.region_bytes);
        }
        window_context->accumulation = accumulation_rows != 0;
        if (accumulation_rows) window_context->maximum_rows = accumulation_rows;
        m.moe_window = window_context.get();
        std::fprintf(stderr, "strata MoE window configuration: requested_mode=%d effective_mode=%d maximum_rows=%lld mixed_fp16_bytes=%llu\n",
            window_requested, accumulation_rows ? window_requested : 2, (long long)window_context->maximum_rows,
            (unsigned long long)accumulation_bytes);
    }
    m.cache = &layer_cache;
    m.host_res = residency.data();
    if (requested_slots)
        std::fprintf(stderr, "strata MoE expert bank: requested_slots=%lld effective_slots=%lld staged=%d host_slab_bytes=%llu device_bytes=%llu residual_gpu_rows=%lld allocation_limit_bytes=%llu general_map=private_scratch_never_published\n",
            (long long)requested_slots, (long long)bank_plan.slots, int(bank_plan.staged()),
            (unsigned long long)host_slab_bytes, (unsigned long long)cache_bytes,
            (long long)plan.gpu_rows, (unsigned long long)max_alloc);
    std::fprintf(stderr, "strata prefill: layer-major candidate, tokens=%lld chunk=%lld residual_dtype=%s "
                         "residual_element_bytes=%llu residual_total=%llu residual_gpu_rows=%lld "
                         "residual_gpu=%llu residual_host=%llu residual_staging=%llu layer_cache=%llu; "
                         "extra allocations, GPU headroom=536870912, RAM headroom=8589934592\n",
                 (long long)n, (long long)m.T, detail::layer_major_dtype_name(plan.dtype),
                 (unsigned long long)detail::layer_major_storage_bytes(plan.dtype),
                 (unsigned long long)plan.total_bytes, (long long)plan.gpu_rows,
                 (unsigned long long)plan.gpu_bytes, (unsigned long long)plan.host_bytes,
                 (unsigned long long)plan.staging_bytes, (unsigned long long)cache_bytes);
    if (plan.dtype == detail::LayerMajorResidualDtype::fp16)
        std::fprintf(stderr, "strata prefill: FP16 residual storage is experimental/unqualified; "
                             "RNE, finite overflow -> signed Inf, FP16 subnormals preserved, "
                             "underflow -> signed zero, NaN -> quiet NaN; "
                             "one pack/unpack per interlayer boundary, live residual FP32, "
                             "operator/state precision unchanged\n");
    if (native16_on)
        std::fprintf(stderr, "strata prefill: native16 BF16 HC residual/projection IO and MoE outputs; "
                             "FP32 local accumulations/router/recurrent state; PLE and final callbacks bridge FP32; "
                             "QSA tensor IO=%s, indexer/GDN internals FP32; reserved scratch arena unchanged; unqualified\n",
                     qsa16_on ? "BF16" : "FP32");
    for (; layer < g.n_layers; ++layer) {
        if (profile) profile->layer = int(layer);
        if (qsa_profile) qsa_profile->layer = int(layer);
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        if (!drain()) { err = "prefill layer-major: layer boundary drain failed"; return false; }
        if (std::getenv("STRATA_TRACE")) {
            std::fprintf(stderr, "strata trace: layer-major layer %lld of %lld: loading %lld experts\n",
                         (long long)layer, (long long)g.n_layers, (long long)NE);
            std::fflush(stderr);
        }
        std::fill(residency.begin(), residency.end(), -1);
        if (bank_plan.staged()) window_context->expert_slots->reset(); // previous layer checked above
        const auto load_start = Clock::now();
        const uint64_t bytes = layout.blob_bytes(layer);
        // Resolve/copy on the main thread. copy_blob may itself call blob(),
        // whose pointer need only survive until the next blob() call.
        sycl_backend::cpu_wait_profile::Host source_copy_profile(sycl_backend::cpu_wait_profile::source_copy);
        for (int32_t e = 0; e < NE; ++e) {
            if ((e % 32) == 0 && should_stop && should_stop()) { err = "cancelled"; return false; }
            if (!m.src->copy_blob(layer, e, upload.get() + size_t(e) * size_t(stride))) {
                err = "prefill layer-major: expert source copy failed"; return false;
            }
        }
        source_copy_profile.finish();
        if (profile) profile->add(LmProfile::source, load_start);
        const auto upload_start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};
        if (profile) profile->begin(LmProfile::upload, m.cs);
        sycl_backend::cpu_wait_profile::Host upload_cpu_profile(sycl_backend::cpu_wait_profile::upload_submit);
        for (int32_t e = 0; !bank_plan.staged() && e < NE; ++e) {
            if (!layer_cache.fill_slot(e, upload.get() + size_t(e) * size_t(stride), m.copy, err, int64_t(bytes)))
                return false;
            ++stats_.experts_streamed;
        }
        upload_cpu_profile.finish();
        if (profile) {
            profile->end(LmProfile::upload, m.cs);
            profile->add(LmProfile::upload_submit, upload_start);
            if (!bank_plan.staged()) profile->rows[layer].upload_bytes += uint64_t(NE) * bytes;
        }
        const auto upload_drain_start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};
        if (!drain()) { err = "prefill layer-major: expert upload drain failed"; return false; }
        if (profile) {
            profile->add(LmProfile::upload_drain, upload_drain_start);
            profile->retired(LmProfile::upload); // existing checked upload drain above
        }
        stats_.ms_experts_host += ms_since(load_start);
        // Publish all 512 only after the upload wait succeeds. No Stager/issuer
        // job can reference upload, rows, this map, or this temporary cache.
        if (!bank_plan.staged())
            for (int32_t e = 0; e < NE; ++e) residency[size_t(layer * NE + e)] = e;
        if (census) {
            const auto& f = layout.fmt[size_t(layer)];
            // run_impl stream_all is !layer_major_active && ...; therefore its
            // copy issuer cannot start here. Compute/count hooks run inline.
            census->begin(layer, {f.gu_type, f.d_type, f.gu_row, f.d_row,
                f.up_off, f.down_off, f.bytes}, residency.data() + layer * NE,
                m.layer_major_active && !m.pp && !next_ && m.copy == m.cs);
        }
        const auto activation_start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};
        if (activate_native_dense_layer) {
            if (!drain()) { err = "prefill dense bank: preactivation drain failed"; return false; }
            m.native_dense_overlay = nullptr;
            if (!activate_native_dense_layer(layer, m.cs, m.native_dense_overlay, err)) return false;
            if (!m.native_dense_overlay || m.native_dense_overlay->empty()) {
                err = "prefill dense bank: activation published no private view"; return false;
            }
        }
        if (profile && activate_native_dense_layer) profile->add(LmProfile::activation, activation_start);
        stage_lb_ = layer; stage_le_ = layer + 1;
        m.layer_major_host_input = layer == 0 ? nullptr : rows.get();
        m.layer_major_gpu_input = layer == 0 ? nullptr : device_rows.get();
        ss.ple_prev[0] = initial_prev[0]; ss.ple_prev[1] = initial_prev[1];
        cursor = {n, m.T};
        if (window_context) window_context->complete_piece = [&](int64_t row, int64_t count, std::string& error) {
            cursor.accept(row, count);
            if (layer + 1 != g.n_layers) return true; // bank already written; no self-copy
            const auto span = plan.span(row, count);
            if (!span.device || span.offset_bytes > window_context->bank_bytes ||
                span.bytes > window_context->bank_bytes - span.offset_bytes) {
                error = "prefill MoE window: final bank callback span"; return false;
            }
            const auto* packed = reinterpret_cast<const uint16_t*>(
                reinterpret_cast<const uint8_t*>(window_context->bank) + span.offset_bytes);
            native16::restore(packed, m.R, count * D, m.cs);
            if (!drain()) { error = "prefill MoE window: final callback bridge drain"; return false; }
            ++logical_chunks;
            return !restore.chunk || restore.chunk(m.R, count, pos0 + row, error);
        };
        on_stage_chunk = [&](int64_t done, std::string& error) {
            if (layer + 1 != g.n_layers || done != pos0 + n) return true;
            if (stage_delivered || cursor.next != final_chunk_row) {
                error = "prefill layer-major: invalid final checkpoint callback order"; return false;
            }
            stage_delivered = true;
            ss.ple_prev[0] = final_prev[0]; ss.ple_prev[1] = final_prev[1];
            return !restore.stage || restore.stage(done, error);
        };
        on_chunk = [&](const float* residual, int64_t count, int64_t position, std::string& error) {
            if (position < pos0 || position - pos0 >= n) {
                error = "prefill layer-major: output position exceeds request"; return false;
            }
            const int64_t row = position - pos0;
            cursor.accept(row, count);
            if (audit && !audit->source(layer, row, count, residual, m.layer_major_audit_copy)) {
                error = "prefill boundary audit: source copy/drain failed"; return false;
            }
            if (layer + 1 == g.n_layers) {
                ++logical_chunks;
                return !restore.chunk || restore.chunk(residual, count, position, error);
            }
            const auto store_start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};
            const auto span = plan.span(row, count);
            if (profile) {
                profile->rows[layer].store_bytes[int(span.device)] += span.bytes;
                profile->begin(LmProfile::store, m.cs, span.device);
            }
            uint8_t* destination = (span.device ? device_rows.get() : rows.get()) + size_t(span.offset_bytes);
            cudaError_t status = cudaSuccess;
            sycl_backend::cpu_wait_profile::Host store_cpu_profile(span.device
                ? sycl_backend::cpu_wait_profile::residual_store_device
                : sycl_backend::cpu_wait_profile::residual_store_host);
            if (plan.dtype == detail::LayerMajorResidualDtype::fp16) {
                uint16_t* packed = span.device ? reinterpret_cast<uint16_t*>(destination)
                                              : m.layer_major_fp16_stage;
                if (!packed || (!span.device && span.bytes > plan.staging_bytes)) {
                    error = "prefill layer-major: FP16 residual staging bounds exceeded"; return false;
                }
                // to_f16 uses shared RNE bits, without hsat/clamping. Neither
                // placement rounds twice; final-layer callbacks stay FP32.
                to_f16(residual, packed, int64_t(span.bytes / 2), m.cs);
                if (!span.device)
                    status = cudaMemcpyAsync(destination, packed, size_t(span.bytes), cudaMemcpyDeviceToHost, m.cs);
            } else {
                status = cudaMemcpyAsync(destination, residual, size_t(span.bytes),
                    span.device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost, m.cs);
            }
            store_cpu_profile.finish();
            if (profile) profile->end(LmProfile::store, m.cs);
            const auto store_drain_start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};
            if (status != cudaSuccess || !drain()) {
                error = "prefill layer-major: residual store copy/drain failed"; return false;
            }
            if (profile) {
                profile->add(span.device ? LmProfile::store_drain_device : LmProfile::store_drain_host, store_drain_start);
                profile->add(span.device ? LmProfile::store_wall_device : LmProfile::store_wall_host, store_start);
                profile->retired(LmProfile::store); // existing checked residual-store drain above
            }
            if (audit && !audit->stored(layer, row, count, destination, span.device,
                                         m.layer_major_audit_copy)) {
                error = "prefill boundary audit: stored copy/drain failed"; return false;
            }
            return true;
        };
        if (native16_on) m.on_chunk16 = [&](const uint16_t* residual, int64_t count, int64_t position, std::string& error) {
            if (position < pos0 || position - pos0 >= n) {
                error = "prefill native16: output position exceeds request"; return false;
            }
            const int64_t row = position - pos0;
            cursor.accept(row, count);
            if (layer + 1 == g.n_layers) {
                // Existing MTP/decode/NLL consumers own a float interface. Only
                // the final layer expands it; intermediate layers copy BF16.
                native16::restore(residual, m.R, count * D, m.cs);
                if (!drain()) { error = "prefill native16: final bridge drain failed"; return false; }
                ++logical_chunks;
                return !restore.chunk || restore.chunk(m.R, count, position, error);
            }
            const auto start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};
            const auto span = plan.span(row, count);
            if (profile) {
                profile->rows[layer].store_bytes[int(span.device)] += span.bytes;
                profile->begin(LmProfile::store, m.cs, span.device);
            }
            uint8_t* dst = (span.device ? device_rows.get() : rows.get()) + size_t(span.offset_bytes);
            sycl_backend::cpu_wait_profile::Host cpu_profile(span.device
                ? sycl_backend::cpu_wait_profile::residual_store_device
                : sycl_backend::cpu_wait_profile::residual_store_host);
            const auto status = cudaMemcpyAsync(dst, residual, size_t(span.bytes),
                span.device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost, m.cs);
            cpu_profile.finish();
            if (profile) profile->end(LmProfile::store, m.cs);
            const auto drain_start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};
            if (status != cudaSuccess || !drain()) {
                error = "prefill native16: BF16 store copy/drain failed"; return false;
            }
            if (profile) {
                profile->add(span.device ? LmProfile::store_drain_device : LmProfile::store_drain_host, drain_start);
                profile->add(span.device ? LmProfile::store_wall_device : LmProfile::store_wall_host, start);
                profile->retired(LmProfile::store);
            }
            return true;
        };
        if (!run_impl(tokens, n, pos0, err)) return false;
        if (!drain()) { err = "prefill layer-major: pass drain failed"; return false; }
        if (score_gemm_on && core::is_qsa_layer(g, layer)) {
            const auto copied = cudaMemcpyAsync(&score_status_observed, m.score_gemm.status,
                sizeof(score_status_observed), cudaMemcpyDeviceToHost, m.cs);
            const bool retired = drain(); // retire even if enqueue reported an error
            if (copied != cudaSuccess || !retired) {
                err = "prefill score GEMM: status copy/drain failed"; return false;
            }
            if (score_status_observed) {
                err = "prefill score GEMM: invalid arithmetic status=" + std::to_string(score_status_observed);
                return false;
            }
        }
        m.native_dense_overlay = nullptr; // checked pass drain completed
        if (!cursor.complete()) { err = "prefill layer-major: incomplete layer callbacks"; return false; }
        if (census) census->end(layer);
        if (std::getenv("STRATA_TRACE")) {
            std::fprintf(stderr, "strata trace: layer-major layer %lld of %lld: completed %lld chunks\n",
                         (long long)layer, (long long)g.n_layers, (long long)cursor.calls);
            std::fflush(stderr);
        }
        // No callback retains the per-pass cursor/layer references past this
        // checked boundary. Owners and saved caller callbacks remain leased.
        on_chunk = {}; on_stage_chunk = {}; m.on_chunk16 = {};
        if (window_context) window_context->complete_piece = {};
    }
    if (!stage_delivered || logical_chunks != 1 + (n - 1) / m.T) {
        err = "prefill layer-major: incomplete final callback sequence"; return false;
    }
    if (!drain()) { err = "prefill layer-major: final drain failed"; return false; }
    ss.ple_prev[0] = final_prev[0]; ss.ple_prev[1] = final_prev[1];
    stats_.tokens = old_stats.tokens + n;
    stats_.chunks = old_stats.chunks + logical_chunks;
    stats_.ms_total = old_stats.ms_total + ms_since(started);
    if (window_context && window_context->accumulation) {
        if (window_context->expert_slots)
            std::fprintf(stderr, "strata MoE expert bank complete: uploads=%llu requested_H2D_bytes=%llu permanent_hits=%llu permanent_slots=%d total_slots=%d final_drain=checked\n",
                (unsigned long long)window_context->staged_uploads,
                (unsigned long long)window_context->staged_upload_bytes,
                (unsigned long long)window_context->staged_hits,
                window_context->expert_slots->permanent_used(), window_context->expert_slots->slots());
        std::fprintf(stderr, "strata MoE GU fusion: fused_jobs=%llu fused_rows=%llu separate_jobs=%llu min_rows=2334\n",
            (unsigned long long)window_context->gu_fused_jobs,
            (unsigned long long)window_context->gu_fused_rows,
            (unsigned long long)(window_context->jobs - window_context->gu_fused_jobs));
        std::fprintf(stderr, "strata MoE Down fusion: fused_jobs=%llu fused_rows=%llu separate_jobs=%llu min_rows=2334\n",
            (unsigned long long)window_context->down_fused_jobs,
            (unsigned long long)window_context->down_fused_rows,
            (unsigned long long)(window_context->jobs - window_context->down_fused_jobs));
        std::fprintf(stderr, "strata MoE accumulator window: windows=%llu dequants=%llu jobs=%llu source_H2D_bytes=%llu GPU_resident_windows_only=1 operator_max=8192 maximum_rows=%lld\n",
            (unsigned long long)window_context->windows, (unsigned long long)window_context->dequants,
            (unsigned long long)window_context->jobs, (unsigned long long)window_context->source_bytes,
            (long long)window_context->maximum_rows);
    } else if (window_context)
        std::fprintf(stderr, "strata MoE window: windows=%llu dequants=%llu jobs=%llu source_H2D_bytes=%llu GPU_resident_pairs_only=1 operator_max=8192 source_uploads_per_window=1\n",
            (unsigned long long)window_context->windows, (unsigned long long)window_context->dequants,
            (unsigned long long)window_context->jobs, (unsigned long long)window_context->source_bytes);
    // Field/callback restore and then cache/residual release follow this proof.
    if (audit) audit->complete();
    if (phase_audit) phase_audit->complete();
    lease.retire();
    if (profile) {
        profile->wrapper_ms = ms_since(started);
        profile->complete = true;
        profile->close(); // checked lease has retired; status checked before emit
        profile->emit();
    }
    if (qsa_profile) {
        qsa_profile->finish(profile->wrapper_ms);
        qsa_profile->close();
        qsa_profile->emit();
    }
    if (census) census->complete();
    return true;
#endif
} catch (const std::exception& exc) {
    err = std::string("prefill layer-major: ") + exc.what();
    return false;
} catch (...) {
    err = "prefill layer-major: unknown callback or engine exception";
    return false;
}
