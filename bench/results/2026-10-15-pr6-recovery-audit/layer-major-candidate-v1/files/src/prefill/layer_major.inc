// Included inside strata::prefill after the current chunk implementation's
// helpers. This is orchestration only; run_impl owns GDN/QSA/PLE/expert math.
bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    const char* requested = std::getenv("STRATA_PREFILL_LAYER_MAJOR");
    return requested && std::strcmp(requested, "1") == 0
               ? run_layer_major(tokens, n, pos0, err) : run_impl(tokens, n, pos0, err);
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
    // Existing observers assume one full-model call. Refuse ambiguous reports
    // rather than emitting 48 apparent full-prompt measurements/checkpoints.
    for (const char* name : {"STRATA_PREFILL_ROUTED_FP16_OBSERVE", "STRATA_PREFILL_TIMING",
                             "STRATA_PREFILL_ATTN_LAYOUT", "STRATA_PREFILL_DUMP_R", "STRATA_DBG_NAN",
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
    const uint64_t cache_bytes = detail::layer_major_mul(NE, stride);
    const uint64_t map_bytes = detail::layer_major_mul(detail::layer_major_mul(g.n_layers, NE), sizeof(int32_t));
    size_t free_gpu = 0, total_gpu = 0;
    const cudaError_t memory_status = cudaMemGetInfo(&free_gpu, &total_gpu);
    if (memory_status != cudaSuccess || free_gpu > total_gpu ||
        free_gpu < detail::layer_major_add(cache_bytes, gpu_headroom)) {
        err = "prefill layer-major: extra expert cache (" + std::to_string(cache_bytes) +
              " bytes) plus 512 MiB headroom exceeds known free VRAM (" + std::to_string(free_gpu) + ")";
        return false;
    }
    const auto plan = detail::LayerMajorResidualPlan::make(n, m.T, D,
        free_gpu - cache_bytes - gpu_headroom,
        detail::layer_major_requested_rows(std::getenv("STRATA_PREFILL_LAYER_MAJOR_R_GPU")));
    // One owned host upload slab removes every source pointer lease before DMA.
    // Also count the cache's small blocking staging allocation and two maps.
    uint64_t host_need = detail::layer_major_add(plan.host_bytes, cache_bytes);
    host_need = detail::layer_major_add(host_need, stride);
    host_need = detail::layer_major_add(host_need, detail::layer_major_mul(2, map_bytes));
    if (!core::conversation_memory_admit(core::conversation_available_memory(), host_need, host_headroom)) {
        err = "prefill layer-major: residual/upload storage exceeds known available RAM minus 8 GiB headroom";
        return false;
    }
    if (should_stop && should_stop()) { err = "cancelled"; return false; }
    core::ExpertCache layer_cache;
    std::unique_ptr<uint8_t[]> upload(new uint8_t[size_t(cache_bytes)]);
    std::unique_ptr<float[]> rows(plan.host_bytes ? new float[size_t(plan.host_bytes / sizeof(float))] : nullptr);
    std::vector<int32_t> residency(size_t(g.n_layers * NE), -1);
    auto free_device = [](float* p) { if (p) (void) cudaFree(p); };
    std::unique_ptr<float, decltype(free_device)> device_rows(nullptr, free_device);

    int64_t logical_chunks = 0, layer = 0;
    const int64_t final_chunk_row = ((n - 1) / m.T) * m.T;
    bool stage_delivered = false;
    detail::LayerMajorChunkCursor cursor{n, m.T};

    // Destruction order: checked drain -> field/callback restoration -> owners.
    // A failed drain aborts here, with all captures/storage still alive. Session
    // tensors are NOT rolled back by restoring these borrowed C++ fields.
    struct Restore {
        Prefill& p; Impl& m;
        int64_t lb, le; const float* input;
        const core::ExpertCache* cache; const int32_t* residency;
        bool active; const float* gpu_input; detail::LayerMajorResidualPlan plan;
        decltype(on_chunk) chunk; decltype(on_stage_chunk) stage;
        ~Restore() noexcept {
            p.stage_lb_ = lb; p.stage_le_ = le; p.hand_in_ = input;
            m.cache = cache; m.host_res = residency; m.layer_major_active = active;
            m.layer_major_gpu_input = gpu_input; m.layer_major_residual = plan;
            p.on_chunk = std::move(chunk); p.on_stage_chunk = std::move(stage);
        }
    } restore{*this, m, stage_lb_, stage_le_, hand_in_, m.cache, m.host_res,
              m.layer_major_active, m.layer_major_gpu_input, m.layer_major_residual, on_chunk, on_stage_chunk};
    cudaError_t compute_drain = cudaSuccess, copy_drain = cudaSuccess;
    auto drain = [&]() noexcept {
        compute_drain = cudaStreamSynchronize(m.cs);
        copy_drain = m.copy == m.cs ? compute_drain : cudaStreamSynchronize(m.copy);
        return compute_drain == cudaSuccess && copy_drain == cudaSuccess;
    };
    detail::DraftUploadGuard lease(drain, [&]() noexcept {
        std::fprintf(stderr, "prefill layer-major: cannot prove queue retirement (compute=%d copy=%d); "
                             "aborting with temporary owners live\n", int(compute_drain), int(copy_drain));
        std::fflush(stderr); std::abort();
    });
    lease.arm();
    if (!drain()) { err = "prefill layer-major: entry queue drain failed"; return false; }
    // Allocation-only cache opening is essential: no memset may fail and free
    // its allocation before the outer checked lease has retired its readers.
    if (!layer_cache.open_uninitialized(NE, 1, NE, int64_t(stride), err)) return false;
    if (plan.gpu_bytes) {
        float* p = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&p), size_t(plan.gpu_bytes)) != cudaSuccess) {
            err = "prefill layer-major: extra GPU residual allocation failed"; return false;
        }
        device_rows.reset(p);
    }
    m.layer_major_active = true;
    m.layer_major_residual = plan;
    m.cache = &layer_cache;
    m.host_res = residency.data();
    std::fprintf(stderr, "strata prefill: layer-major candidate, tokens=%lld chunk=%lld residual_total=%llu "
                         "residual_gpu_rows=%lld residual_gpu=%llu residual_host=%llu layer_cache=%llu; "
                         "extra allocations, GPU headroom=536870912, RAM headroom=8589934592\n",
                 (long long)n, (long long)m.T, (unsigned long long)plan.total_bytes, (long long)plan.gpu_rows,
                 (unsigned long long)plan.gpu_bytes, (unsigned long long)plan.host_bytes,
                 (unsigned long long)cache_bytes);
    for (; layer < g.n_layers; ++layer) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        if (!drain()) { err = "prefill layer-major: layer boundary drain failed"; return false; }
        std::fill(residency.begin(), residency.end(), -1);
        const auto load_start = Clock::now();
        const uint64_t bytes = layout.blob_bytes(layer);
        // Resolve/copy on the main thread. copy_blob may itself call blob(),
        // whose pointer need only survive until the next blob() call.
        for (int32_t e = 0; e < NE; ++e) {
            if ((e % 32) == 0 && should_stop && should_stop()) { err = "cancelled"; return false; }
            if (!m.src->copy_blob(layer, e, upload.get() + size_t(e) * size_t(stride))) {
                err = "prefill layer-major: expert source copy failed"; return false;
            }
        }
        for (int32_t e = 0; e < NE; ++e) {
            if (!layer_cache.fill_slot(e, upload.get() + size_t(e) * size_t(stride), m.copy, err, int64_t(bytes)))
                return false;
            ++stats_.experts_streamed;
        }
        if (!drain()) { err = "prefill layer-major: expert upload drain failed"; return false; }
        stats_.ms_experts_host += ms_since(load_start);
        // Publish all 512 only after the upload wait succeeds. No Stager/issuer
        // job can reference upload, rows, this map, or this temporary cache.
        for (int32_t e = 0; e < NE; ++e) residency[size_t(layer * NE + e)] = e;
        stage_lb_ = layer; stage_le_ = layer + 1;
        hand_in_ = layer == 0 ? nullptr : rows.get();
        m.layer_major_gpu_input = layer == 0 ? nullptr : device_rows.get();
        ss.ple_prev[0] = initial_prev[0]; ss.ple_prev[1] = initial_prev[1];
        cursor = {n, m.T};
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
            if (layer + 1 == g.n_layers) {
                ++logical_chunks;
                return !restore.chunk || restore.chunk(residual, count, position, error);
            }
            const auto span = plan.span(row, count);
            uint8_t* destination = span.device ? reinterpret_cast<uint8_t*>(device_rows.get())
                                             : reinterpret_cast<uint8_t*>(rows.get());
            const auto status = cudaMemcpyAsync(destination + size_t(span.offset_bytes), residual,
                size_t(span.bytes), span.device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost, m.cs);
            if (status != cudaSuccess || !drain()) {
                error = "prefill layer-major: residual store copy/drain failed"; return false;
            }
            return true;
        };
        if (!run_impl(tokens, n, pos0, err)) return false;
        if (!drain()) { err = "prefill layer-major: pass drain failed"; return false; }
        if (!cursor.complete()) { err = "prefill layer-major: incomplete layer callbacks"; return false; }
        // No callback retains the per-pass cursor/layer references past this
        // checked boundary. Owners and saved caller callbacks remain leased.
        on_chunk = {}; on_stage_chunk = {};
    }
    if (!stage_delivered || logical_chunks != 1 + (n - 1) / m.T) {
        err = "prefill layer-major: incomplete final callback sequence"; return false;
    }
    if (!drain()) { err = "prefill layer-major: final drain failed"; return false; }
    ss.ple_prev[0] = final_prev[0]; ss.ple_prev[1] = final_prev[1];
    stats_.tokens = old_stats.tokens + n;
    stats_.chunks = old_stats.chunks + logical_chunks;
    stats_.ms_total = old_stats.ms_total + ms_since(started);
    // Field/callback restore and then cache/residual release follow this proof.
    lease.retire();
    return true;
#endif
} catch (const std::exception& exc) {
    err = std::string("prefill layer-major: ") + exc.what();
    return false;
} catch (...) {
    err = "prefill layer-major: unknown callback or engine exception";
    return false;
}
