#!/usr/bin/env python3
"""Fabricate a review-only patch from pinned source bytes. Never writes product files."""
from pathlib import Path
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
original = {}
proposed = {}

def edit(path, before, after):
    if path not in original:
        original[path] = (ROOT / path).read_text()
        proposed[path] = original[path]
    if proposed[path].count(before) != 1:
        raise RuntimeError(f"source anchor is not unique: {path}: {before[:80]!r}")
    proposed[path] = proposed[path].replace(before, after, 1)

header_path = "include/strata/kernels/cpu/stager_affinity.hpp"
original[header_path] = ""
proposed[header_path] = '''// Bounded Linux affinity helper for the opt-in SYCL Stager experiment.
// CPU IDs above 63 are rejected instead of truncating the caller's allowed set.
#pragma once
#include <cerrno>
#include <cstdint>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace strata::kernels::cpu::stager_affinity {
inline int read_mask(uint64_t& mask) {
    mask = 0;
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    const int rc = pthread_getaffinity_np(pthread_self(), sizeof allowed, &allowed);
    if (rc) return rc;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        if (cpu >= 64) { mask = 0; return EOVERFLOW; }
        mask |= uint64_t{1} << cpu;
    }
    return mask ? 0 : EINVAL;
#else
    return ENOTSUP;
#endif
}

// Called only by a new worker with a mask captured from its issuer before pinning.
// Kernel cpuset restrictions still apply. A narrowed effective mask is an error.
inline int apply_mask(uint64_t requested, uint64_t& effective) {
    effective = 0;
    if (!requested) return EINVAL;
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    for (int cpu = 0; cpu < 64; ++cpu)
        if (requested & (uint64_t{1} << cpu)) CPU_SET(cpu, &allowed);
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof allowed, &allowed);
    const int read_rc = read_mask(effective);
    if (rc) return rc;
    if (read_rc) return read_rc;
    return effective == requested ? 0 : ERANGE;
#else
    return ENOTSUP;
#endif
}
inline int name_worker() {
#if defined(__linux__)
    return pthread_setname_np(pthread_self(), "strata-stager");
#else
    return ENOTSUP;
#endif
}
} // namespace strata::kernels::cpu::stager_affinity
'''

edit("include/strata/core/session.hpp",
     "    bool pinned = false;\n\n    /// Allocates the buffers",
     "    bool pinned = false;\n    uint64_t stager_affinity_mask = 0; ///< opt-in only: validated allowed mask before the host pin\n\n    /// Allocates the buffers")
edit("src/core/session.cpp", '#include "strata/kernels/cpu/pool.hpp"',
     '#include "strata/kernels/cpu/pool.hpp"\n#include "strata/kernels/cpu/stager_affinity.hpp"')
edit("src/core/session.cpp", "    const std::vector<int> cores = strata::kernels::cpu::physical_cores(false);\n",
'''    stager_affinity_mask = 0;
#if defined(STRATA_ENABLE_SYCL)
    const char* stager_policy = std::getenv("STRATA_SYCL_STAGER_AFFINITY");
    const bool restore_stager = stager_policy && std::strcmp(stager_policy, "1") == 0;
    uint64_t before_pin = 0;
    if (restore_stager) {
        const int rc = strata::kernels::cpu::stager_affinity::read_mask(before_pin);
        // Existing issuer teardown uses a positive signed saved mask. Decline CPU63 too,
        // before changing affinity, rather than changing that unrelated restoration contract.
        if (rc || (before_pin & (uint64_t{1} << 63))) {
            err = "SessionLoopScratch: stager-affinity pre-pin capture refused (rc=" +
                  std::to_string(rc ? rc : EOVERFLOW) + ")";
            free();
            return false;
        }
    }
#endif
    const std::vector<int> cores = strata::kernels::cpu::physical_cores(false);
''')
edit("src/core/session.cpp", "        pinned = true;\n    }\n    return true;\n}\n\nvoid SessionLoopScratch::free()",
'''        pinned = true;
    }
#if defined(STRATA_ENABLE_SYCL)
    if (restore_stager) {
        uint64_t effective = 0;
        const int rc = strata::kernels::cpu::stager_affinity::read_mask(effective);
        if (rc || cores.empty() || cores[0] < 0 || cores[0] >= 64 || pinned_core <= 0 ||
            static_cast<uint64_t>(pinned_core) != before_pin ||
            effective != (uint64_t{1} << cores[0])) {
            err = "SessionLoopScratch: stager-affinity requires a verified issuer pin and saved mask";
            free();
            return false;
        }
        stager_affinity_mask = before_pin;
    }
#endif
    return true;
}

void SessionLoopScratch::free()''')
edit("src/core/session.cpp", "void SessionLoopScratch::free() {\n", "void SessionLoopScratch::free() {\n    stager_affinity_mask = 0;\n")

edit("include/strata/prefill/prefill.hpp",
     "              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0);",
'''              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0,
              uint64_t stager_affinity_mask = 0);
    // Experimental STRATA_SYCL_STAGER_AFFINITY=1 requires an owned pre-pin mask from
    // SessionLoopScratch. No saved mask or unsupported platform: init fails explicitly.''')
edit("src/prefill/prefill.cpp", '#include "strata/kernels/cpu/expert.hpp"',
     '#include "strata/kernels/cpu/expert.hpp"\n#include "strata/kernels/cpu/stager_affinity.hpp"')
edit("src/prefill/prefill.cpp", '#include <future>', '#include <future>\n#include <exception>')
edit("src/prefill/prefill.cpp", "    int device = 0;\n\n    bool init(size_t blob_bytes, int nthreads) {",
'''    int device = 0;
    struct WorkerPolicy { uint64_t effective = 0; int rc = 0, name_rc = 0; };
    std::vector<WorkerPolicy> worker_policy;
    std::atomic<int> policy_ready{0};

    bool init(size_t blob_bytes, int nthreads, uint64_t allowed_mask, std::string& err) {''')
edit("src/prefill/prefill.cpp", "        for (int t = 0; t < nthreads; ++t) threads.emplace_back([this] { work(); });\n        return true;",
'''        if (!allowed_mask) {
            // Keep the default worker startup exactly as before.
            for (int t = 0; t < nthreads; ++t) threads.emplace_back([this] { work(); });
            return true;
        }
        worker_policy.resize((size_t) nthreads);
        threads.reserve((size_t) nthreads);
        try {
            for (int t = 0; t < nthreads; ++t) threads.emplace_back([this, t, allowed_mask] {
                auto& policy = worker_policy[(size_t) t];
                policy.name_rc = strata::kernels::cpu::stager_affinity::name_worker();
                policy.rc = strata::kernels::cpu::stager_affinity::apply_mask(allowed_mask, policy.effective);
                policy_ready.fetch_add(1, std::memory_order_acq_rel);
                if (policy.rc == 0) work();
            });
        } catch (const std::exception& e) {
            // No jobs have been published. Wake/join successful workers before returning.
            { std::lock_guard<std::mutex> lk(mu); quit = true; }
            cv.notify_all();
            for (auto& t : threads) t.join();
            threads.clear();
            err = std::string("prefill: stager-affinity worker creation failed: ") + e.what();
            std::fprintf(stderr, "%s\\n", err.c_str());
            return false;
        }
        while (policy_ready.load(std::memory_order_acquire) != nthreads) std::this_thread::yield();
        bool good = true;
        char entry[160];
        std::snprintf(entry, sizeof entry, "prefill: stager-affinity policy=pre-pin-allowed requested=0x%llx workers=%d",
                      (unsigned long long) allowed_mask, nthreads);
        std::string summary(entry);
        for (int t = 0; t < nthreads; ++t) {
            const auto& policy = worker_policy[(size_t) t];
            good = good && policy.rc == 0;
            std::snprintf(entry, sizeof entry, " worker%d={effective:0x%llx,rc:%d,name_rc:%d}",
                          t, (unsigned long long) policy.effective, policy.rc, policy.name_rc);
            summary += entry;
        }
        summary += good ? " status=verified\\n" : " status=failed\\n";
        std::fprintf(stderr, "%s", summary.c_str());
        if (!good) {
            { std::lock_guard<std::mutex> lk(mu); quit = true; }
            cv.notify_all();
            for (auto& t : threads) t.join();
            threads.clear();
            err = "prefill: stager-affinity worker mask failed; no staging jobs were published";
        }
        return good;''')
edit("src/prefill/prefill.cpp", "                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes) {\n    Impl& m = *impl_;",
'''                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes,
                   uint64_t stager_affinity_mask) {
    uint64_t allowed_mask = 0;
    const char* policy = std::getenv("STRATA_SYCL_STAGER_AFFINITY");
    if (policy && std::strcmp(policy, "0") != 0) {
        if (std::strcmp(policy, "1") != 0) {
            err = "prefill: STRATA_SYCL_STAGER_AFFINITY must be 0 or 1";
            return false;
        }
#if defined(STRATA_ENABLE_SYCL) && defined(__linux__)
        if (!stager_affinity_mask) {
            err = "prefill: stager-affinity requested without a validated pre-pin allowed mask";
            return false;
        }
        allowed_mask = stager_affinity_mask;
#else
        err = "prefill: stager-affinity experiment supports Linux SYCL only";
        return false;
#endif
    }
    Impl& m = *impl_;''')
edit("src/prefill/prefill.cpp", "        if (!m.stager->init((size_t) MAXBLOB(), threads)) ok = false;",
'''        if (!m.stager->init((size_t) MAXBLOB(), threads, allowed_mask, err)) {
            if (allowed_mask) return false;
            ok = false;
        }''')
edit("src/program/generate.cpp",
     "            if (!sp.init(wt, g, ss, srcp, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes))",
'''            if (!sp.init(wt, g, ss, srcp, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes,
                         loop_scratch.stager_affinity_mask))''')
edit("src/program/generate.cpp", "                          borrow_bytes)) {\n            std::fprintf(stderr, \"strata generate: %s\\n\", err.c_str());",
     "                          borrow_bytes, loop_scratch.stager_affinity_mask)) {\n            std::fprintf(stderr, \"strata generate: %s\\n\", err.c_str());")

patch = "".join("".join(difflib.unified_diff(original[p].splitlines(True), proposed[p].splitlines(True),
    fromfile="/dev/null" if not original[p] else "a/" + p, tofile="b/" + p)) for p in sorted(proposed))
(OUT / "unified.patch").write_text(patch)
digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
identity = {
    "status": "artifact-only; not applied, compiled, or run",
    "base_branch": "feature/sycl",
    "base_revision_note": "No Git action; source-byte hashes below are authoritative. R532 cited 24158fb7f82012af0e425d10b1de8c018d9ab869, not independently asserted current HEAD.",
    "source": [{"path": p, "base_sha256": digest(original[p]) if original[p] else None,
                "proposed_sha256": digest(proposed[p]), "base_bytes": len(original[p].encode()),
                "proposed_bytes": len(proposed[p].encode())} for p in sorted(proposed)],
    "patch_sha256": digest(patch),
    "opt_in": "STRATA_SYCL_STAGER_AFFINITY=1",
    "default": "unset or 0; original thread startup and inheritance",
}
(OUT / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
print(json.dumps({"patch_sha256": identity["patch_sha256"], "product_files_written": 0}))
