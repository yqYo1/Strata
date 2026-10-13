// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "strata/core/phase_timing.hpp"
#include <functional>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <cstdint>
#include <string>
#include <exception>
namespace strata::core {
inline bool phase_cache_opt_in(const char* value) { return value && std::strcmp(value, "1") == 0; }
inline bool phase_cache_requested() {
    static const bool requested = phase_cache_opt_in(std::getenv("STRATA_PHASE_DECODE_CACHE"));
    return requested;
}
inline bool phase_prefill_capacity_admitted(bool phase, bool opt12k, bool lease, bool alias,
                                            bool automatic, int64_t requested, bool layer_major = false) {
    if (!phase) return !opt12k;
    return !automatic && (opt12k ? !layer_major && lease && alias && requested == 12288 : requested == 8192 ||
        (layer_major && lease && alias && requested == 4096));
}
inline bool phase_actual_chunk_admitted(int64_t capacity, int64_t count, int64_t pos, int64_t extent) {
    return capacity > 0 && count > 0 && count <= capacity && pos >= 0 &&
           pos <= extent && count <= extent - pos;
}
inline bool phase_requested_add(uint64_t bytes, uint64_t& total) {
    if (bytes > std::numeric_limits<uint64_t>::max() - total) return false;
    total += bytes;
    return true;
}
inline bool phase_request_kind(const std::string& line) {
    return line == "QUIT" || line.rfind("GEN ", 0) == 0;
}
// Observed caller state, not a checkpoint-policy bit. The persistent session's
// buffers remain owned; only a completed, non-resumable sequence may precede a
// fresh GEN, which performs the ordinary session_zero before reading token 0.
struct PhaseFreshState {
    bool capacities_zero = false, healthy_baseline = false, session_geometry = false;
    bool live = false, checkpoint = false, parked = false, reuse = false;
    bool slot = false, incoming = false, from_live = false, split = false, failed = false;
    int64_t resume = 0, turn_at = -1;
};
inline bool phase_fresh_admission(const PhaseFreshState& state) {
    return state.capacities_zero && state.healthy_baseline && state.session_geometry &&
           !state.live && !state.checkpoint && !state.parked && !state.reuse && !state.slot &&
           !state.incoming && !state.from_live && !state.split && !state.failed &&
           state.resume == 0 && state.turn_at == -1;
}
// Actual caller orchestration, shared with CPU fault fixtures. Callbacks are the
// actual caller/owner operations; this gate never invents a GPU quiescence proof.
class PhaseServeCycle {
public:
    using Operation = std::function<bool(std::string&)>;
    Operation quiesce, retire_workspace, grow_fill_publish, invalidate_detach, reinit_workspace;
    Operation retire_readers;
    Operation release_head, reload_head;
    Operation begin_prefill, finish_prefill;
    bool begin_gen(std::string& error) {
        PhaseLoggedCall timing("begin_gen");
        if (begin_prefill && !release_head)
            return fail("phase cycle: cache lease requires head lease", error);
        if (static_cast<bool>(begin_prefill) != static_cast<bool>(finish_prefill))
            return fail("phase cycle: incomplete cache lease callbacks", error);
        if (static_cast<bool>(release_head) != static_cast<bool>(reload_head))
            return fail("phase cycle: incomplete head lease callbacks", error);
        if (failed_ || active_ || decoding_ || requests_ == std::numeric_limits<uint64_t>::max()) return fail("phase cycle: overlapping/failed GEN", error);
        if (retired_) {
            if (!call(quiesce, "quiesce", error) || !call(retire_readers, "graph_retire", error) ||
                !call(invalidate_detach, "invalidate_detach", error) ||
                (release_head && !call(release_head, "head_release", error)) ||
                (begin_prefill && !call(begin_prefill, "begin_prefill", error)) || !call(reinit_workspace, "reinit_workspace", error)) return false;
            retired_ = false;
        }
        if (requests_ == 0 && release_head) {
            if (!call(quiesce, "quiesce", error) || !call(retire_readers, "graph_retire", error) ||
                !call(release_head, "head_release", error) ||
                (begin_prefill && !call(begin_prefill, "begin_prefill", error)) || !call(reinit_workspace, "reinit_workspace", error)) return false;
        }
        active_ = true;
        ++requests_;
        return timing.result(true);
    }
    bool successful_prompt(std::string& error) {
        PhaseLoggedCall timing("successful_prompt");
        if (failed_ || !active_ || retired_) return fail("phase cycle: invalid prompt boundary", error);
        if (!call(quiesce, "quiesce", error) || !call(retire_workspace, "retire_workspace", error)) return false;
        retired_ = true;
        if (finish_prefill && !call(finish_prefill, "finish_prefill", error)) return false;
        if (reload_head && !call(reload_head, "head_reload", error)) return false;
        if (!call(grow_fill_publish, "grow_fill_publish", error)) return false;
        active_ = false;
        decoding_ = true;
        return timing.result(true);
    }
    bool complete_gen(bool success, std::string& error) {
        PhaseLoggedCall timing("complete_gen");
        if (!success || failed_ || active_ || !decoding_)
            return fail("phase cycle: cancelled/nonfinite/failed GEN cannot complete", error);
        decoding_ = false;
        return timing.result(true);
    }
    bool workspace_retired() const { return retired_; }
    uint64_t requests() const { return requests_; }
    bool failed() const { return failed_; }
private:
    bool call(const Operation& operation, const char* name, std::string& error) {
        PhaseLoggedCall timing(name);
        if (!operation) return fail("phase cycle: actual operation callback missing", error);
        try {
            if (operation(error)) return timing.result(true);
            failed_ = true;
            return false;
        } catch (const std::exception& exception) { return fail(exception.what(), error); }
    }
    bool fail(const char* message, std::string& error) { failed_ = true; error = message; return false; }
    bool active_ = false, decoding_ = false, retired_ = false, failed_ = false;
    uint64_t requests_ = 0;
};
}  // namespace strata::core
