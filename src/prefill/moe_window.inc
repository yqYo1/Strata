namespace {
struct PrefillMoeRouterWeights {
    const core::WeightRef *norm, *down, *up, *injection, *gate, *shared_gate;
    bool ready() const {
        return norm && down && up && injection && gate && shared_gate;
    }
};

PrefillMoeRouterWeights prefill_moe_router_weights(const core::LayerView& v, std::string& err) {
    return {need(v, "hc_ffn_norm.weight", err), need(v, "hc_ffn_down.weight", err),
            need(v, "hc_ffn_up.weight", err), need(v, "hc_ffn_inject.weight", err),
            need(v, "ffn_gate_inp.weight", err), need(v, "ffn_gate_inp_shexp.weight", err)};
}

template<class Engine>
bool produce_prefill_moe_router_piece(Engine& m, const core::LayerView& v,
        const PrefillMoeRouterWeights& w, const uint16_t* residual, int64_t rows,
        PrefillMoeRouterPiece out, std::string& err) {
    if (w.shared_gate->kind != core::WeightKind::Bf16InF32 || !w.shared_gate->data) {
        err = "prefill MoE accumulator: shared scalar gate tensor"; return false;
    }
    native16::norm(residual, static_cast<const float*>(w.norm->data), EPS, m.grs, m.xn16, rows, m.cs);
    if (!bf16_proj16(m.gemm, w.down, m.xn16, m.lo_b16, rows, v.name("hc_ffn_down.weight"), err)) return false;
    native16::silu(m.lo_b16, m.lo16, rows, m.cs);
    if (!bf16_proj16(m.gemm, w.up, m.lo16, m.gated_b16, rows, v.name("hc_ffn_up.weight"), err) ||
        !bf16_proj(m.gemm, w.injection, m.xn16, out.injection, rows, v.name("hc_ffn_inject.weight"), err)) return false;
    native16::mix(residual, m.grs, static_cast<const float*>(w.norm->data), m.gated_b16,
                  m.mixed_bf, out.mixed, rows, m.cs);
    if (!bf16_proj(m.gemm, w.gate, m.mixed_bf, out.logits, rows, v.name("ffn_gate_inp.weight"), err)) return false;
    route(out.logits, out.ids, out.weights, rows, NE, m.cs);
    // The scalar gate uses the original BF16 producer, not its FP16 sibling.
    m.gemm.bf16(m.mixed_bf, static_cast<const uint16_t*>(w.shared_gate->data), out.shared_gate, rows, 1, N);
    return true;
}

template<class Engine>
void enqueue_prefill_prefix_experts(Engine& m, MoeWindowContext& owner, int64_t layer) {
    const auto& layout = strata::kernels::cpu::expert_layout();
    if (owner.prefix_router_layer != layer || owner.prefix_experts.empty() ||
        owner.prefix_upload.experts || !owner.expert_slots || !owner.private_expert_bank ||
        owner.expert_cache != m.cache || owner.private_expert_bank->owner() != m.cache ||
        owner.private_expert_bank->layout().stride != uint64_t(layout.blob_bytes(layer)) ||
        owner.expert_source_stride != uint64_t(layout.blob_bytes(layer)) ||
        owner.expert_upload_staging || owner.command_trace || !owner.scratch_mixed ||
        !owner.prefix_router_storage || !m.layer_major_active || !m.native16_active ||
        m.layer_major_residual.gpu_rows != m.layer_major_residual.rows || m.copy != m.cs)
        throw std::invalid_argument("prefill prefix expert upload: owner, extent or queue contract");
    const auto& bank = *owner.private_expert_bank;
    auto* host_staging = owner.prefix_upload_host_storage.data() ? &owner.prefix_upload_host_storage : nullptr;
#if defined(STRATA_SYCL_PREFILL_NATIVE_COPY)
    auto* native_copy = owner.prefix_native_copy.get();
#else
    sycl_backend::native_copy::NativeCopyList* native_copy = nullptr;
#endif
    if (m.borrowed)
        sycl_backend::validate_spans({{bank.device_slot(0), size_t(bank.layout().backing_bytes)}},
            {{m.borrowed_base, size_t(m.borrowed_capacity)}}, 1);
    sycl_backend::cpu_wait_profile::Host host(sycl_backend::cpu_wait_profile::upload_submit);
    owner.prefix_upload = detail::stage_prefill_fullfit_eager(*owner.expert_slots,
        bank, owner.expert_slab, owner.expert_slab_bytes, owner.expert_source_stride,
        owner.prefix_experts, m.cs, {{m.region, size_t(m.region_bytes)},
         {owner.bank, size_t(owner.bank_bytes)}, {owner.prefix_router_storage, PrefillMoePrefixStorage::bytes},
         {m.R, size_t(m.T * D * 4)}, {m.emb, size_t(m.T * N * 4)},
         {m.dq_gu[0], size_t(2 * 640 * N * 2)}, {m.dq_d[0], size_t(N * 640 * 2)}},
        host_staging, native_copy, owner.prefix_copy != PrefillPrefixCopy::sycl,
        owner.defer_prefix_completion ? &owner.pending_prefix_upload : nullptr, owner.prefix_stage_order);
}
}  // namespace

bool Prefill::run_prefill_moe_prefix_router(int64_t layer, int64_t rows, std::string& err) try {
    Impl& m = *impl_;
    auto& owner = *m.moe_window;
    using Storage = PrefillMoePrefixStorage;
    if (!owner.prefix_router_storage || owner.prefix_router_layer != -1 || !owner.bank ||
        !m.layer_major_active || !m.native16_active || m.copy != m.cs || m.T != int64_t(Storage::rows) ||
        layer < 0 || layer >= m.g->n_layers || rows != owner.maximum_rows ||
        rows <= m.T || rows > detail::MoeAccumPlan::maximum_rows ||
        uint64_t(rows) * D * sizeof(uint16_t) > owner.bank_bytes) {
        err = "prefill MoE prefix router: invalid bank, lease or extent"; return false;
    }
    if (should_stop && should_stop()) { err = "cancelled"; return false; }
    const core::LayerView v(*m.wt, layer, m.native_dense_overlay);
    const auto weights = prefill_moe_router_weights(v, err);
    if (!weights.ready()) return false;
    owner.ids.resize(size_t(rows * K));
    const auto out = Storage::view(owner.prefix_router_storage);
    if (!produce_prefill_moe_router_piece(m, v, weights, owner.bank, m.T, out, err)) return false;
    // Existing attention store/drain precedes this producer. Keep both host
    // IDs and device outputs owned through the outer checked lease on failure.
    if (cudaMemcpyAsync(owner.ids.data(), out.ids, Storage::id_bytes, cudaMemcpyDeviceToHost, m.cs) != cudaSuccess ||
        cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = "prefill MoE prefix router: prefix readback/drain"; return false;
    }
    owner.prefix_router_layer = layer;
    if (owner.prefix_expert_upload != PrefillPrefixExpertUpload::off) {
        if (!owner.prefix_experts.empty() || owner.prefix_upload.experts)
            throw std::logic_error("prefill prefix expert upload: previous layer still owns its receipt");
        owner.prefix_experts = detail::prefill_prefix_expert_ids(owner.ids, Storage::rows, K);
        if (owner.prefix_expert_upload == PrefillPrefixExpertUpload::prefix) {
            if (should_stop && should_stop()) { err = "cancelled"; return false; }
            enqueue_prefill_prefix_experts(m, owner, layer);
        }
    }
    return true;
} catch (const std::exception& exception) {
    err = std::string("prefill MoE prefix router: ") + exception.what();
    return false;
}

// Main-authored expert-major window. Saved matrix inputs remain FP16;
// router/shared scalar gate consume the original BF16 mix. Only the expert
// accumulation order and the <=8192-row GEMM job shapes change numerically.
bool Prefill::run_moe_accum_window(int64_t layer, int64_t first_row, int64_t rows, std::string& err) try {
    Impl& m = *impl_;
    const bool split_accumulator = m.moe_window && m.moe_window->accumulator_primary_rows != 0;
    const bool relocated_job_buffers = m.moe_window && m.moe_window->relocated_job_buffers;
    if (!m.moe_window || !m.moe_window->accumulation || !m.native16_active ||
        !m.layer_major_active || m.T != 8192 || m.copy != m.cs || !m.scratch_alias ||
        !m.cache || !m.host_res || layer < 0 || layer >= m.g->n_layers ||
        rows <= m.T || rows > m.moe_window->maximum_rows ||
        rows > (relocated_job_buffers ? detail::MoeAccumPlan::long_prefill_maximum_rows :
                split_accumulator ? detail::MoeAccumPlan::split_maximum_rows :
                                  detail::MoeAccumPlan::maximum_rows) || first_row < 0) {
        err = "prefill MoE accumulator: unsupported entry"; return false;
    }
    auto& owner = *m.moe_window;
    if (relocated_job_buffers && (first_row != 0 || rows != owner.maximum_rows))
        throw std::invalid_argument("prefill relocated workspace requires one complete admitted window");
    auto& queue = sycl_backend::queue_for(m.cs);
    const uint64_t offset = detail::layer_major_mul(uint64_t(first_row), D * 2);
    const uint64_t bytes = detail::layer_major_mul(uint64_t(rows), D * 2);
    if (!owner.bank || offset > owner.bank_bytes || bytes > owner.bank_bytes - offset ||
        !owner.mixed || !m.region || !m.emb || reinterpret_cast<uintptr_t>(m.region) % 256)
        throw std::invalid_argument("prefill MoE accumulator: backing bank/arena bounds");
    if (relocated_job_buffers && !split_accumulator)
        throw std::invalid_argument("prefill relocated workspace requires split accumulator ownership");
    const auto plan = relocated_job_buffers ? detail::MoeAccumPlan::make_long_prefill_scratch(rows,
        owner.mixed_bytes, m.region_bytes) :
        split_accumulator ? detail::MoeAccumPlan::make_split_scratch(rows,
        owner.mixed_bytes, m.region_bytes) :
        owner.scratch_mixed ? detail::MoeAccumPlan::make(rows, owner.mixed_bytes,
            owner.mixed_tail_bytes, owner.mixed_tail_offset) :
            detail::MoeAccumPlan::make_contiguous(rows, owner.mixed_bytes, m.region_bytes);
    auto* residual = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(owner.bank) + offset);
    auto* mixed = owner.mixed;
    auto* bo = reinterpret_cast<uint16_t*>(m.emb);
    const size_t mixed_bytes = size_t(plan.primary_bytes);
    sycl_backend::validate_device_span(residual, size_t(bytes), queue);
    sycl_backend::validate_device_span(m.region, size_t(plan.region_bytes), queue);
    sycl_backend::validate_device_span(mixed, mixed_bytes, queue);
    sycl_backend::validate_device_span(bo, size_t(m.T * N * 2), queue);
    sycl_backend::validate_spans({{m.region, size_t(plan.region_bytes)}, {mixed, mixed_bytes},
                                 {bo, size_t(m.T * N * 2)}}, {{residual, size_t(bytes)}});
    auto* mixed_tail = plan.tail_bytes ? owner.mixed_tail : nullptr;
    if (split_accumulator) {
        if (!owner.scratch_mixed || !owner.accumulator_tail ||
            owner.accumulator_primary_rows != plan.accumulator_primary_rows ||
            owner.accumulator_tail_bytes != plan.accumulator_tail_bytes() ||
            owner.extra_workspace != reinterpret_cast<uint8_t*>(owner.accumulator_tail) ||
            owner.extra_workspace_bytes != (relocated_job_buffers ? plan.extra_bytes :
                plan.accumulator_tail_allocation_bytes()))
            throw std::invalid_argument("prefill split accumulator: owner/plan mismatch");
        sycl_backend::validate_device_span(owner.extra_workspace,
            size_t(owner.extra_workspace_bytes), queue);
        sycl_backend::validate_spans({{owner.extra_workspace, size_t(owner.extra_workspace_bytes)}},
            {{m.region, size_t(m.region_bytes)}, {mixed, mixed_bytes},
             {bo, size_t(m.T * N * 2)}, {residual, size_t(bytes)}});
        sycl_backend::validate_spans({{owner.extra_workspace, size_t(owner.extra_workspace_bytes)}},
            {{m.cache->device_slot(0), size_t(m.cache->bytes())},
             {m.dq_gu[0], size_t(2 * 640 * N * 2)}, {m.dq_d[0], size_t(N * 640 * 2)}});
    } else if (owner.accumulator_tail || owner.accumulator_tail_bytes ||
               owner.extra_workspace || owner.extra_workspace_bytes || relocated_job_buffers)
        throw std::invalid_argument("prefill ordinary accumulator: unexpected split owner");
    if (owner.scratch_mixed) {
        if (!m.mixed_scratch_owners_ready() ||
            mixed != reinterpret_cast<uint16_t*>(m.R) ||
            owner.mixed_bytes > detail::layer_major_mul(uint64_t(m.T), D * 4) ||
            owner.mixed_tail_offset < plan.region_bytes || owner.mixed_tail_offset > m.region_bytes ||
            owner.mixed_tail_bytes > m.region_bytes - owner.mixed_tail_offset ||
            (owner.mixed_tail_bytes && owner.mixed_tail !=
                reinterpret_cast<uint16_t*>(m.region + size_t(owner.mixed_tail_offset))))
            throw std::invalid_argument("prefill mixed scratch: owner/phase layout mismatch");
        if (plan.tail_bytes) {
            sycl_backend::validate_device_span(mixed_tail, size_t(plan.tail_bytes), queue);
            sycl_backend::validate_spans({{mixed_tail, size_t(plan.tail_bytes)}},
                {{m.region, size_t(plan.region_bytes)}, {mixed, mixed_bytes},
                 {bo, size_t(m.T * N * 2)}, {residual, size_t(bytes)}});
        }
        if (std::getenv("STRATA_TRACE"))
            std::fprintf(stderr, "strata MoE mixed owners: phase=entry layer=%lld first=%lld rows=%lld primary=%p primary_bytes=%llu region=%p region_bytes=%llu tail=%p tail_offset=%llu tail_bytes=%llu residual=%p residual_bytes=%llu bo=%p queue=%p direct_owners=%d previous_attention=checked_drain\n",
                (long long)layer, (long long)first_row, (long long)rows,
                static_cast<void*>(mixed), (unsigned long long)plan.primary_bytes,
                static_cast<void*>(m.region), (unsigned long long)m.region_bytes,
                static_cast<void*>(mixed_tail), (unsigned long long)owner.mixed_tail_offset,
                (unsigned long long)plan.tail_bytes, static_cast<void*>(residual),
                (unsigned long long)bytes, static_cast<void*>(bo), m.cs, int(!m.borrowed));
        if (m.borrowed && std::getenv("STRATA_TRACE"))
            std::fprintf(stderr,"strata MoE cache loan: generation=%llu layout_generation=%llu layer=%lld first=%lld rows=%lld used=%llu capacity=%llu\n",
                (unsigned long long)m.cache_scratch_lease.generation(),
                (unsigned long long)m.lease_layout_generation,(long long)layer,(long long)first_row,(long long)rows,
                (unsigned long long)m.borrowed_used,(unsigned long long)m.borrowed_capacity);
    }
    auto mixed_piece = [&](int64_t at, int64_t count) {
        const auto span = plan.mixed_piece(at, count);
        auto* base = span.tail ? mixed_tail : mixed;
        if (!base) throw std::invalid_argument("prefill mixed scratch: missing logical bank");
        return reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(base) + size_t(span.offset));
    };
    using Plane = detail::MoeAccumPlan::Plane;
    auto view = [&](Plane plane) -> void* {
        const auto span = plan.span(plane);
        const bool extra = span.storage == detail::MoeAccumPlan::Storage::extra;
        auto* base = extra ? owner.extra_workspace : m.region;
        const auto capacity = extra ? owner.extra_workspace_bytes : m.region_bytes;
        if (!base || (extra && !relocated_job_buffers) || span.offset > capacity ||
            span.reserved > capacity - span.offset || span.bytes > span.reserved)
            throw std::invalid_argument("prefill MoE accumulator: logical arena extent");
        void* result = base + size_t(span.offset);
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
    const auto router_weights = prefill_moe_router_weights(v, err);
    const core::WeightRef *wsg = need(v, "ffn_gate_shexp.weight", err),
                          *wsu = need(v, "ffn_up_shexp.weight", err),
                          *wsd = need(v, "ffn_down_shexp.weight", err);
    if (!router_weights.ready() || !wsg || !wsu || !wsd) return false;
    if (m.layer_major_profile) m.layer_major_profile->begin(detail::LayerMajorCoarseProfile::moe, m.cs);
    if (cudaMemsetAsync(accumulator, 0, size_t(plan.span(Plane::accumulator).bytes), m.cs) != cudaSuccess ||
        cudaMemsetAsync(status, 0, sizeof(uint32_t), m.cs) != cudaSuccess) {
        err = "prefill MoE accumulator: initialization"; return false;
    }
    if (split_accumulator && cudaMemsetAsync(owner.accumulator_tail, 0,
            size_t(owner.accumulator_tail_bytes), m.cs) != cudaSuccess) {
        err = "prefill split accumulator: tail initialization"; return false;
    }
    int64_t first_missing = 0;
    if (owner.prefix_router != PrefillMoePrefixRouter::off) {
        using Storage = PrefillMoePrefixStorage;
        if (owner.prefix_router_layer != layer || first_row != 0 || rows != owner.maximum_rows ||
            !owner.prefix_router_storage || owner.ids.size() != size_t(rows * K)) {
            err = "prefill MoE prefix router: stale or missing prefix"; return false;
        }
        const auto prefix = Storage::view(owner.prefix_router_storage);
        if (cudaMemcpyAsync(mixed_piece(0, m.T), prefix.mixed, Storage::mixed_bytes, cudaMemcpyDeviceToDevice, m.cs) != cudaSuccess ||
            cudaMemcpyAsync(injection, prefix.injection, Storage::injection_bytes, cudaMemcpyDeviceToDevice, m.cs) != cudaSuccess ||
            cudaMemcpyAsync(weights, prefix.weights, Storage::weight_bytes, cudaMemcpyDeviceToDevice, m.cs) != cudaSuccess ||
            cudaMemcpyAsync(shared_gate, prefix.shared_gate, Storage::shared_gate_bytes, cudaMemcpyDeviceToDevice, m.cs) != cudaSuccess) {
            err = "prefill MoE prefix router: publish saved FFN outputs"; return false;
        }
        first_missing = m.T;
    } else owner.ids.resize(size_t(rows * K));
    // Attention and PLE already advanced each chunk exactly once. The optional
    // prefix producer supplies its outputs once, with the original tail order.
    for (int64_t at = first_missing; at < rows; at += m.T) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        const int64_t count = std::min(m.T, rows - at);
        if (!produce_prefill_moe_router_piece(m, v, router_weights, residual + at * D, count,
                {mixed_piece(at, count), injection + at * HC, inverse + at * K,
                 weights + at * K, shared_gate + at, logits}, err)) return false;
    }
    const bool census_requested = detail::moe_upload_census_requested();
    if (census_requested) owner.prefix_weight_audit.resize(size_t(m.T * K));
    if (cudaMemcpyAsync(owner.ids.data() + first_missing * K, inverse + first_missing * K,
            size_t((rows - first_missing) * K) * sizeof(int32_t), cudaMemcpyDeviceToHost, m.cs) != cudaSuccess ||
        (census_requested && cudaMemcpyAsync(owner.prefix_weight_audit.data(), weights,
             owner.prefix_weight_audit.size() * sizeof(float), cudaMemcpyDeviceToHost, m.cs) != cudaSuccess) ||
        cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = "prefill MoE accumulator: router readback/drain"; return false;
    }
    if (census_requested) {
        const auto ids_hash = core::fnv1a64(reinterpret_cast<const uint8_t*>(owner.ids.data()), owner.ids.size() * sizeof(int32_t));
        const auto weight_hash = core::fnv1a64(reinterpret_cast<const uint8_t*>(owner.prefix_weight_audit.data()),
                                              owner.prefix_weight_audit.size() * sizeof(float));
        std::fprintf(stderr, "strata MoE router census: layer=%lld first=%lld rows=%lld ids=%016llx prefix_weights=%016llx\n",
                     (long long)layer, (long long)first_row, (long long)rows,
                     (unsigned long long)ids_hash, (unsigned long long)weight_hash);
    }
    {
        sycl_backend::cpu_wait_profile::Host group(sycl_backend::cpu_wait_profile::host_group);
        owner.routes = relocated_job_buffers ?
            detail::MoeWindowRoutes::build_long_prefill_accum(rows, owner.ids.data(), owner.ids.size()) :
            split_accumulator ?
            detail::MoeWindowRoutes::build_split_accum(rows, owner.ids.data(), owner.ids.size()) :
            detail::MoeWindowRoutes::build_accum(rows, owner.ids.data(), owner.ids.size());
    }
    std::unique_ptr<detail::MoeUploadCensus> upload_census;
    if (census_requested)
        upload_census = std::make_unique<detail::MoeUploadCensus>();
    if (m.layer_major_profile || upload_census) {
        // Metadata already owned on the host after the existing router drain.
        // No additional tensor readback, GPU submission or inference mutation.
        // This is a planned-job census; only final request/issuer totals prove
        // that all jobs completed. The old chunk census refuses window mode.
        std::array<uint32_t, 8193> job_rows{};
        std::array<uint32_t, (detail::MoeWindowRoutes::max_long_prefill_rows + 8191) / 8192 + 1> jobs_per_expert{};
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
    namespace cp = sycl_backend::command_profile;
    namespace hp = sycl_backend::cpu_wait_profile;
    // Disjoint CPU regions cover every layer even when command tracing is off.
    // GU/down already time GEMM internally; do not wrap a whole job or add waits.
    auto command_trace = owner.command_trace ? owner.command_trace->begin(layer, first_row) : cp::Frame{};
    if (command_trace)
        for (const auto& job : owner.routes->jobs())
            command_trace.consider(job.expert, job.global_row, job.rows);
    {
        hp::Host host(hp::moe_map_upload);
        auto tag = command_trace.tag(cp::Kind::upload, -1, 0, rows, K, 4);
        if (cudaMemcpyAsync(inverse, owner.routes->original_slots().data(), size_t(rows * K * 4), cudaMemcpyHostToDevice, m.cs) != cudaSuccess ||
            cudaMemcpyAsync(source, owner.routes->source_rows().data(), size_t(rows * K * 4), cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
            err = "prefill MoE accumulator: immutable maps upload"; return false;
        }
    }
    {
        hp::Host host(hp::moe_job_weights);
        auto tag = command_trace.tag(cp::Kind::job_weights, -1, 0, rows, K);
        native16::moe_job_weights(*owner.routes, weights, inverse, job_weights, status, m.cs);
    }
    const auto& layout = strata::kernels::cpu::expert_layout();
    const auto& format = layout.fmt[size_t(layer)];
    detail::PrefillFullfitUploadReceipt eager_upload;
    const bool use_prefix_upload = owner.prefix_expert_upload != PrefillPrefixExpertUpload::off;
    bool verify_selected_readers = use_prefix_upload;
    if (use_prefix_upload) {
        if (owner.prefix_expert_upload == PrefillPrefixExpertUpload::ffn) {
            if (should_stop && should_stop()) { err = "cancelled"; return false; }
            enqueue_prefill_prefix_experts(m, owner, layer);
        }
        eager_upload = owner.prefix_upload;
        if (!first_missing || owner.prefix_experts.empty() ||
            eager_upload.experts != owner.prefix_experts.size() ||
            eager_upload.copy_commands != eager_upload.experts ||
            eager_upload.payload_bytes != eager_upload.experts * uint64_t(layout.blob_bytes(layer)))
            throw std::logic_error("prefill prefix expert upload: invalid layer receipt");
        owner.staged_uploads += eager_upload.experts;
        stats_.experts_streamed += eager_upload.experts;
        owner.staged_upload_bytes += eager_upload.payload_bytes;
    } else if (detail::prefill_fullfit_eager_requested() && owner.private_expert_bank &&
        owner.expert_slots && owner.expert_slots->full_fit() &&
        !owner.expert_slots->permanent_used() && !owner.expert_upload_staging) {
        if (m.layer_major_residual.gpu_rows!=m.layer_major_residual.rows ||
            !owner.scratch_mixed || m.copy!=m.cs || owner.expert_cache!=m.cache ||
            owner.private_expert_bank->owner()!=m.cache ||
            owner.private_expert_bank->layout().stride!=uint64_t(layout.blob_bytes(layer)))
            throw std::invalid_argument("prefill eager upload: owner/queue/residual contract");
        if (owner.command_trace)
            throw std::invalid_argument("prefill eager upload: command observer requires a separate plan");
        std::vector<int32_t> experts;experts.reserve(512);
        for (const auto& job:owner.routes->jobs())
            if (experts.empty() || experts.back()!=job.expert) experts.push_back(job.expert);
        {
            hp::Host host(hp::upload_submit);
            eager_upload=detail::stage_prefill_fullfit_eager(*owner.expert_slots,
                *owner.private_expert_bank,owner.expert_slab,owner.expert_slab_bytes,
                owner.expert_source_stride,experts,m.cs,
                {{m.region,size_t(m.region_bytes)},{mixed,mixed_bytes},
                 {bo,size_t(m.T*N*2)},{residual,size_t(bytes)},
                 {m.dq_gu[0],size_t(2*640*N*2)},{m.dq_d[0],size_t(N*640*2)}});
        }
        owner.staged_uploads+=eager_upload.experts;stats_.experts_streamed+=eager_upload.experts;
        owner.staged_upload_bytes+=eager_upload.payload_bytes;
        std::fprintf(stderr,"strata prefill fullfit upload: policy=eager layer=%lld first_row=%lld "
            "logical_experts=%llu accepted_copy_commands=%llu requested_H2D_bytes=%llu "
            "publication=accepted_same_queue native_completion=unchecked_until_outer_drain\n",
            (long long)layer,(long long)first_row,(unsigned long long)eager_upload.experts,
            (unsigned long long)eager_upload.copy_commands,(unsigned long long)eager_upload.payload_bytes);
    } else if (detail::prefill_eager_upload_policy() == detail::PrefillEagerUpload::demand_fit &&
               owner.private_expert_bank && owner.expert_slots && !owner.expert_slots->full_fit() &&
               !owner.expert_slots->permanent_used() && !owner.expert_upload_staging &&
               first_row == 0 && rows == m.layer_major_residual.rows) {
        // A single fully resident window has its final expert counts here.
        // Keep the normal per-expert source, validation, queue and slot publisher.
        if (m.layer_major_residual.gpu_rows != rows || !owner.scratch_mixed ||
            m.copy != m.cs || owner.expert_cache != m.cache ||
            owner.private_expert_bank->owner() != m.cache || owner.command_trace)
            throw std::invalid_argument("prefill demand-fit owner/queue/residual contract");
        const auto stride = layout.blob_bytes(layer);
        hp::Host host(hp::upload_submit);
        eager_upload.experts = detail::stage_prefill_demand_fit(owner.routes->counts(),
            *owner.expert_slots, [&](int32_t expert) {
                if (should_stop && should_stop()) throw std::runtime_error("cancelled");
                const auto ticket = detail::stage_private_expert_bank(*owner.expert_slots,
                    *owner.private_expert_bank, owner.expert_slab, owner.expert_slab_bytes,
                    stride, expert, m.cs, owner.expert_source_stride);
                eager_upload.first_upload[size_t(expert)] = true;
                return ticket;
            });
        verify_selected_readers = eager_upload.experts != 0;
        eager_upload.copy_commands = eager_upload.experts;
        eager_upload.payload_bytes = eager_upload.experts * stride;
        owner.staged_uploads += eager_upload.experts;
        stats_.experts_streamed += eager_upload.experts;
        owner.staged_upload_bytes += eager_upload.payload_bytes;
        if (m.layer_major_profile || census_requested || std::getenv("STRATA_TRACE"))
            std::fprintf(stderr, "strata prefill demand-fit: layer=%lld rows=%lld slots=%d "
                         "accepted_copies=%llu requested_H2D_bytes=%llu\n",
                         (long long)layer, (long long)rows, owner.expert_slots->slots(),
                         (unsigned long long)eager_upload.copy_commands,
                         (unsigned long long)eager_upload.payload_bytes);
    }
#if defined(STRATA_SYCL_PREFILL_NATIVE_COPY)
    if (owner.defer_prefix_completion) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        if (!use_prefix_upload || !owner.prefix_native_copy || !owner.expert_slots ||
            owner.prefix_router_layer != layer)
            throw std::logic_error("prefill deferred expert upload: first-reader ownership mismatch");
        hp::Host host(hp::upload_submit);
        detail::finish_prefill_fullfit_deferred(*owner.expert_slots, *owner.prefix_native_copy,
                                               owner.pending_prefix_upload);
    }
#endif
    int32_t loaded = -1;
    for (size_t index = 0; index < owner.routes->jobs().size(); ++index) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        const auto& job = owner.routes->jobs()[index];
        if (job.expert != loaded) {
            int32_t slot = -1;
            if (owner.expert_slots) {
                if (m.layer_major_residual.gpu_rows != m.layer_major_residual.rows ||
                    !owner.scratch_mixed || m.copy != m.cs || !owner.expert_cache ||
                    owner.expert_cache != m.cache)
                    throw std::invalid_argument("prefill expert bank: incomplete residual or queue contract");
                hp::Host host(hp::upload_submit);
                auto tag = command_trace.tag(cp::Kind::expert_upload, job.expert, job.global_row, job.rows,
                    layout.blob_bytes(layer), 1, format.gu_type);
                if (owner.private_expert_bank && owner.private_expert_bank->owner() != m.cache)
                    throw std::invalid_argument("prefill private expert bank: wrong backing cache");
                const auto ticket = owner.private_expert_bank ?
                    detail::stage_private_expert_bank(*owner.expert_slots, *owner.private_expert_bank,
                        owner.expert_slab, owner.expert_slab_bytes, layout.blob_bytes(layer),
                        job.expert, m.cs, owner.expert_source_stride, owner.expert_upload_staging.get()) :
                    detail::stage_expert_bank(*owner.expert_slots, *owner.expert_cache,
                        owner.expert_slab, owner.expert_slab_bytes, owner.expert_stride,
                        layout.blob_bytes(layer), job.expert, m.cs, owner.expert_source_stride);
                const bool eager_first=eager_upload.first_upload[size_t(job.expert)];
                if (eager_first && (ticket.upload || !ticket.permanent))
                    throw std::runtime_error("prefill eager upload: first reader lost permanent mapping");
                if (upload_census) upload_census->record(job.rows, ticket.upload || eager_first);
                slot = ticket.slot;
                if (ticket.upload) {
                    ++owner.staged_uploads; ++stats_.experts_streamed;
                    owner.staged_upload_bytes += layout.blob_bytes(layer);
                } else if (!eager_first) ++owner.staged_hits;
                eager_upload.first_upload[size_t(job.expert)]=false;
            } else slot = m.host_res[size_t(layer * NE + job.expert)];
            if (slot < 0) { err = "prefill MoE accumulator: missing layer expert"; return false; }
            const auto* blob = owner.private_expert_bank ? owner.private_expert_bank->device_slot(slot) :
                m.cache->device_slot(slot);
            sycl_backend::validate_device_span(blob, size_t(layout.blob_bytes(layer)), queue);
            sycl_backend::validate_device_span(m.dq_gu[0], size_t(2 * 640 * N * 2), queue);
            sycl_backend::validate_device_span(m.dq_d[0], size_t(N * 640 * 2), queue);
            {
                hp::Host host(hp::moe_dequant_gu);
                auto tag = command_trace.tag(cp::Kind::dequant_gu, job.expert, job.global_row, job.rows,
                    2 * format.n_ff, format.n_embd, format.gu_type);
                strata::kernels::iq_dequant_gu_f16(format.gu_type, blob, blob + format.up_off,
                    format.n_ff, format.n_embd, m.dq_gu[0], m.cs);
            }
            {
                hp::Host host(hp::moe_dequant_down);
                auto tag = command_trace.tag(cp::Kind::dequant_down, job.expert, job.global_row, job.rows,
                    format.n_embd, format.n_ff, format.d_type);
                strata::kernels::iq_dequant_f16(format.d_type, blob + format.down_off,
                    format.n_embd * format.n_ff, m.dq_d[0], m.cs);
            }
            loaded = job.expert; ++owner.dequants; ++stats_.experts_resident;
        }
        if (job.rows < 1 || job.rows > m.T || job.global_row < 0 || job.global_row > rows * K - job.rows)
            throw std::invalid_argument("prefill MoE accumulator: job bounds");
        sycl_backend::validate_device_span(source + job.global_row, size_t(job.rows * 4), queue);
        {
            hp::Host host(hp::moe_gather);
            auto tag = command_trace.tag(cp::Kind::gather, job.expert, job.global_row, job.rows, N);
            if (owner.scratch_mixed)
                native16::moe_gather_mixed(*owner.routes, index, mixed, mixed_tail,
                    plan.primary_rows, source, gathered, status, m.cs);
            else gather_rows16(mixed, source + job.global_row, gathered, job.rows, N, m.cs);
        }
        bool fused_gu = false;
        {
            auto tag = command_trace.tag(cp::Kind::gu, job.expert, job.global_row, job.rows, 2 * 640, N, format.gu_type);
            fused_gu = m.gemm.try_gu_swiglu16(gathered, m.dq_gu[0], hidden, job.rows);
            if (!fused_gu) m.gemm.f16_out16(gathered, m.dq_gu[0], gu, job.rows, 2 * 640, N, true);
        }
        if (!fused_gu) {
            hp::Host host(hp::moe_swiglu);
            auto tag = command_trace.tag(cp::Kind::swiglu, job.expert, job.global_row, job.rows, 640);
            native16::swiglu(gu, hidden, job.rows, m.cs);
        } else {
            ++owner.gu_fused_jobs;
            owner.gu_fused_rows += uint64_t(job.rows);
        }
        bool fused_down = false;
        {
            auto tag = command_trace.tag(cp::Kind::down, job.expert, job.global_row, job.rows, N, 640, format.d_type);
            fused_down = split_accumulator ?
                m.gemm.try_down_scatter16_split(*owner.routes, index, hidden, m.dq_d[0],
                    source, inverse, job_weights, accumulator, owner.accumulator_tail,
                    plan.accumulator_primary_rows, status) :
                m.gemm.try_down_scatter16(*owner.routes, index, hidden, m.dq_d[0],
                    source, inverse, job_weights, accumulator, status);
            if (!fused_down) m.gemm.f16_out16(hidden, m.dq_d[0], down, job.rows, N, 640, true);
        }
        if (!fused_down) {
            hp::Host host(hp::moe_accumulate);
            auto tag = command_trace.tag(cp::Kind::accumulate, job.expert, job.global_row, job.rows, N);
            if (split_accumulator)
                native16::moe_accumulate_split(*owner.routes, index, down, source, inverse, job_weights,
                    accumulator, owner.accumulator_tail, plan.accumulator_primary_rows, status, m.cs);
            else
                native16::moe_accumulate(*owner.routes, index, down, source, inverse, job_weights, accumulator, status, m.cs);
        } else {
            ++owner.down_fused_jobs;
            owner.down_fused_rows += uint64_t(job.rows);
        }
        ++owner.jobs;
    }
    if (verify_selected_readers &&
        std::any_of(eager_upload.first_upload.begin(), eager_upload.first_upload.end(),
            [](bool pending) { return pending; }))
        throw std::logic_error("prefill eager expert upload: selected expert without a reader");
    // The host status destination belongs to the outer checked lease, even if DMA fails.
    owner.status_observed = ~uint32_t(0);
    {
        hp::StatusScope status_profile;
        {
            hp::Host host(hp::moe_status);
            auto tag = [&] {
                hp::StatusPart part(hp::status_tag);
                return command_trace.tag(cp::Kind::status, -1, 0, rows);
            }();
            if (cudaMemcpyAsync(&owner.status_observed, status, sizeof(uint32_t), cudaMemcpyDeviceToHost, m.cs) != cudaSuccess ||
                cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = "prefill MoE accumulator: status readback/drain"; return false;
            }
        }
        // Existing drain retires selected expert events before shared work.
        {
            hp::StatusPart part(hp::status_after_drain);
            command_trace.after_checked_drain();
        }
        {
            hp::StatusPart part(hp::status_branch);
            if (owner.status_observed) { err = "prefill MoE accumulator: invalid device route map"; return false; }
        }
        status_profile.complete();
    }
    for (int64_t at = 0; at < rows; at += m.T) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        const int64_t count = std::min(m.T, rows - at);
        auto* input = mixed_piece(at, count);
        if (!native_proj16(m.gemm, wsg, input, sgate, count, v.name("ffn_gate_shexp.weight"), err) ||
            !native_proj16(m.gemm, wsu, input, sup, count, v.name("ffn_up_shexp.weight"), err)) return false;
        native16::swiglu_pair(sgate, sup, sh, count, m.cs);
        if (!native_proj16(m.gemm, wsd, sh, shared, count, v.name("ffn_down_shexp.weight"), err)) return false;
        float* accumulator_chunk = nullptr;
        if (split_accumulator) {
            // Do not form accumulator+at*N for a tail row: even forming an
            // out-of-allocation pointer would violate the C++ pointer contract.
            const auto piece = plan.accumulator_piece(at, count);
            auto* base = piece.tail ? owner.accumulator_tail : accumulator;
            accumulator_chunk = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(base) + size_t(piece.offset));
        } else accumulator_chunk = accumulator + at * N;
        native16::moe_finish(accumulator_chunk, shared, shared_gate + at, bo, count, m.cs);
        native16::write(residual + at * D, bo, injection + at * HC, count, m.cs);
    }
    if (m.layer_major_profile) m.layer_major_profile->end(detail::LayerMajorCoarseProfile::moe, m.cs);
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) { err = "prefill MoE accumulator: HC writes drain"; return false; }
    if (upload_census && (!upload_census->emit_after_checked_drain(stderr, layer,
            first_row, rows, format.gu_type, format.d_type, layout.blob_bytes(layer)) ||
        !upload_census->emit_first_routes_after_checked_drain(stderr, layer, first_row, *owner.routes))) {
        err = "prefill MoE census: output failed after checked drain"; return false;
    }
    if (owner.scratch_mixed && std::getenv("STRATA_TRACE"))
        std::fprintf(stderr, "strata MoE mixed owners: phase=retired layer=%lld first=%lld rows=%lld route_status=0 shared_HC=checked_drain callbacks=after_return\n",
            (long long)layer, (long long)first_row, (long long)rows);
    ++owner.windows; owner.source_bytes += uint64_t(rows * K * 4);
    if (use_prefix_upload) {
        const auto ids_hash = core::fnv1a64(reinterpret_cast<const uint8_t*>(owner.prefix_experts.data()),
            owner.prefix_experts.size() * sizeof(int32_t));
        std::fprintf(stderr, "strata MoE prefix expert upload layer: layer=%lld experts=%llu "
            "requested_H2D_bytes=%llu ids_hash=%016llx readers=checked\n",
            (long long)layer, (unsigned long long)eager_upload.experts,
            (unsigned long long)eager_upload.payload_bytes, (unsigned long long)ids_hash);
        ++owner.prefix_upload_layers;
        owner.prefix_upload_experts += eager_upload.experts;
        owner.prefix_upload_bytes += eager_upload.payload_bytes;
        owner.prefix_upload_host_bytes += eager_upload.host_staged_bytes;
        owner.prefix_experts.clear();
        owner.prefix_upload = {};
    }
    if (first_missing) {
        owner.prefix_router_layer = -1;
        ++owner.prefix_router_completed;
    }
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
