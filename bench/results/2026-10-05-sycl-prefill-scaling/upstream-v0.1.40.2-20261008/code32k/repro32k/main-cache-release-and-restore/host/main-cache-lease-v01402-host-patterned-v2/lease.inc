    struct CacheLease {
        core::ExpertCache* cache = nullptr;
        dpct::queue_ptr queue = nullptr;
        const uint8_t* address = nullptr;
        int64_t mapped = 0, slots = 0, kept = 0;
        std::vector<uint8_t> saved;
        struct RamSpan { int64_t offset; const uint8_t* source; size_t bytes; };
        std::vector<RamSpan> ram_spans;
        size_t restore_bytes = 0, span_bytes = 0;
        size_t remapped_experts = 0;
        bool verify_weights = false;
        const char* mode = "snapshot";
        std::function<bool(const uint8_t*, std::string&)> refresh;
        bool active = false;
        double suspend_ms = 0, restore_ms = 0, graph_ms = 0;
        bool suspend(const core::ExpertCache* original, core::ExpertSource* source,
                     const int32_t* residency, dpct::queue_ptr q,
                     const std::function<bool(std::string&)>& retire,
                     decltype(refresh) callback, std::string& error) {
            if (!original || !original->bytes()) return true;
            if (!retire || !callback) {
                error = "prefill cache release requires graph retirement and refresh callbacks";
                return false;
            }
            const auto t = Clock::now();
            cache = const_cast<core::ExpertCache*>(original);
            queue = q; mapped = cache->mapped_bytes(); slots = cache->slots();
            refresh = std::move(callback);
            address = cache->device_slot(0);
            double fraction = 1;
            if (const char* value = std::getenv("STRATA_PREFILL_CACHE_RELEASE_FRAC")) {
                char* end = nullptr;
                fraction = std::strtod(value, &end);
                if (end == value || *end || !std::isfinite(fraction) || fraction < 0 || fraction > 1) {
                    error = "prefill cache release fraction must be between 0 and 1";
                    return false;
                }
            }
            const int64_t want = static_cast<int64_t>(std::ceil(mapped * (1 - fraction)));
            const int64_t segment = cache->segment_bytes();
            if (!cache->segmented() && fraction != 1) {
                error = "prefill ordinary cache reallocation requires full release";
                return false;
            }
            kept = cache->segmented() ? std::min(mapped, ((want + segment - 1) / segment) * segment) : 0;
            const int64_t payload = cache->bytes();
            restore_bytes = static_cast<size_t>(std::max<int64_t>(0, payload - kept));
            if (const char* value = std::getenv("STRATA_PREFILL_CACHE_RESTORE")) mode = value;
            if (const char* value = std::getenv("STRATA_PREFILL_CACHE_VERIFY")) verify_weights = value[0] == '1';
            if (std::strcmp(mode, "snapshot") && std::strcmp(mode, "ram")) {
                error = "prefill cache restore must be snapshot or ram";
                return false;
            }
            q->wait_and_throw();
            if (!std::strcmp(mode, "ram")) {
                // Require an existing complete RAM arena. Do not silently turn a
                // RAM comparison into reads from GGUF or a transient file tier.
                auto* arena = dynamic_cast<core::ArenaExpertSource*>(source);
                if (!arena || !arena->mapped()) {
                    error = "prefill RAM cache restoration requires a complete resident expert arena";
                    return false;
                }
                const auto& layout = strata::kernels::cpu::expert_layout();
                for (int64_t l = 0; l < layout.n_layers; ++l)
                    for (int64_t e = 0; e < layout.n_expert; ++e) {
                        // Adaptive swaps update the driver's table, not the
                        // ExpertCache's original admission map. Restore the
                        // experts currently resident, including after swaps.
                        const int32_t slot = residency ? residency[l * layout.n_expert + e] : cache->slot_of(l, e);
                        if (slot < 0 || slot >= slots) continue;
                        if (verify_weights && residency && slot != cache->slot_of(l, e)) ++remapped_experts;
                        const int64_t begin = cache->device_slot(slot) - address;
                        const int64_t end = begin + static_cast<int64_t>(layout.blob_bytes(l));
                        if (begin < 0 || end > payload || end > cache->bytes_of(slot + 1)) {
                            error = "prefill cache resident payload is outside its allocation";
                            return false;
                        }
                        if (end <= kept) continue;
                        const uint8_t* blob = arena->blob(l, e);
                        if (!blob) { error = "prefill cache resident expert is absent from RAM"; return false; }
                        const int64_t first = std::max(begin, kept);
                        const size_t bytes = static_cast<size_t>(end - first);
                        ram_spans.push_back({first, blob + first - begin, bytes});
                        span_bytes += bytes;
                    }
                if (verify_weights && restore_bytes) {
                    std::vector<uint8_t> before(restore_bytes);
                    q->memcpy(before.data(), address + kept, before.size()).wait_and_throw();
                    for (const auto& span : ram_spans)
                        if (std::memcmp(before.data() + span.offset - kept, span.source, span.bytes)) {
                            error = "prefill RAM cache restoration disagrees with a current GPU expert payload";
                            return false;
                        }
                }
            } else if (restore_bytes) {
                saved.resize(restore_bytes);
                q->memcpy(saved.data(), address + kept, restore_bytes).wait_and_throw();
            }
            // Complete all consumers and retire their backend allocation
            // references before any physical backing can be destroyed.
            dpct::get_current_device().queues_wait_and_throw();
            if (mapped > kept && !retire(error)) return false;
            active = true; // A partial shrink also needs restoration on exit.
            if (!cache->shrink(kept, error)) return false;
            suspend_ms = ms_since(t);
            return true;
        }
        bool restore(std::string& error) {
            if (!active) return true;
            const auto t = Clock::now();
            if (!cache->grow(mapped, error)) return false;
            if (cache->slots() != slots || (cache->segmented() && cache->device_slot(0) != address)) {
                error = "prefill cache restoration changed its slots or address";
                return false;
            }
            if (!saved.empty())
                queue->memcpy(cache->device_slot(0) + kept, saved.data(), saved.size()).wait_and_throw();
            else if (restore_bytes) {
                // Copy directly from the already resident immutable RAM blobs.
                // No second RAM image, page faults for it, or device-to-host copy.
                if (span_bytes != restore_bytes)
                    queue->memset(cache->device_slot(0) + kept, 0, restore_bytes);
                for (const auto& span : ram_spans)
                    queue->memcpy(cache->device_slot(0) + span.offset, span.source, span.bytes);
                queue->wait_and_throw();
            }
            if (verify_weights && restore_bytes) {
                std::vector<uint8_t> after(restore_bytes);
                queue->memcpy(after.data(), cache->device_slot(0) + kept, after.size()).wait_and_throw();
                if (!saved.empty()) {
                    if (after != saved) { error = "prefill snapshot cache payload restoration failed"; return false; }
                } else {
                    for (const auto& span : ram_spans)
                        if (std::memcmp(after.data() + span.offset - kept, span.source, span.bytes)) {
                            error = "prefill RAM cache payload restoration failed";
                            return false;
                        }
                }
            }
            if (refresh && mapped > kept) {
                const auto graph_start = Clock::now();
                if (!refresh(cache->device_slot(0), error)) return false;
                graph_ms = ms_since(graph_start);
            }
            restore_ms = ms_since(t);
            active = false;
            return true;
        }
        ~CacheLease() {
            if (!active) return;
            try {
                std::string error;
                if (restore(error)) return;
                std::fprintf(stderr, "prefill: cannot restore decode cache: %s\n", error.c_str());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "prefill: cannot restore decode cache: %s\n", e.what());
            }
            // Captured kernels must never replay against unbacked cache memory.
            std::terminate();
        }
    };
