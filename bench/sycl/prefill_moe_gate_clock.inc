// Host clocks for the checked prefill MoE status-to-first-Gate path.
// This file is included by the diagnostic build in the prefill translation unit.
class PrefillMoeGateClock {
    using Stamp = sycl_backend::cpu_wait_profile::Stamp;
public:
    enum Boundary {
        copy_begin, wait_begin, wait_end, projection_begin, dequant_begin, dequant_end, boundaries
    };

    PrefillMoeGateClock() noexcept {
        const char* setting = std::getenv("STRATA_PREFILL_MOE_GATE_CLOCK");
        enabled = setting && std::strcmp(setting, "1") == 0;
    }

    void mark(Boundary boundary) noexcept {
        if (!enabled) return;
        if (seen != boundary) {
            ordered = false;
            return;
        }
        stamps[seen++] = sycl_backend::cpu_wait_profile::stamp();
    }

    void emit(int64_t layer, int64_t first, int64_t rows, int64_t chunk) const noexcept {
        if (!enabled) return;
        bool valid = ordered && seen == boundaries && stamps[0].valid;
        double wall[boundaries - 1]{}, cpu[boundaries - 1]{};
        for (int i = 1; i < boundaries; ++i) {
            wall[i - 1] = stamps[i].wall - stamps[i - 1].wall;
            cpu[i - 1] = stamps[i].cpu - stamps[i - 1].cpu;
            valid = valid && stamps[i].valid &&
                    sycl_backend::cpu_wait_profile::duration(wall[i - 1]) &&
                    sycl_backend::cpu_wait_profile::duration(cpu[i - 1]);
        }
        std::fprintf(stderr,
            "strata prefill MoE gate clock: layer=%lld first=%lld rows=%lld chunk=%lld valid=%d "
            "copy_wall=%.9f copy_cpu=%.9f wait_wall=%.9f wait_cpu=%.9f "
            "handoff_wall=%.9f handoff_cpu=%.9f projection_wall=%.9f projection_cpu=%.9f "
            "dequant_wall=%.9f dequant_cpu=%.9f\n",
            (long long)layer, (long long)first, (long long)rows, (long long)chunk, int(valid),
            wall[0], cpu[0], wall[1], cpu[1], wall[2], cpu[2], wall[3], cpu[3], wall[4], cpu[4]);
    }

    bool enabled = false;
private:
    Stamp stamps[boundaries]{};
    int seen = 0;
    bool ordered = true;
};
