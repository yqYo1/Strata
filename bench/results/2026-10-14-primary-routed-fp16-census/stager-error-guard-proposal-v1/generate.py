#!/usr/bin/env python3
"""Create proposal bytes only. No product writes, application, compilation or execution."""
from pathlib import Path
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SOURCE = "src/prefill/prefill.cpp"
HEADER = "include/strata/prefill/stage_ownership.hpp"
original = (ROOT / SOURCE).read_text()
proposed = original
changes = []

def edit(before, after, site):
    global proposed
    if proposed.count(before) != 1:
        raise RuntimeError(f"source anchor is not unique: {site}: {before[:100]!r}")
    proposed = proposed.replace(before, after, 1)
    changes.append({"site": site, "before": before, "after": after})

def checked(call, site):
    return f'STRATA_PREFILL_STAGE_CHECK({call}, "{site}")'

def check_region(begin, end, selected, site):
    global proposed
    start = proposed.index(begin)
    stop = proposed.index(end, start)
    region = proposed[start:stop]
    for call in selected:
        if region.count(call) != 1:
            raise RuntimeError(f"region anchor is not unique: {site}: {call}")
        region = region.replace(call, checked(call, site), 1)
        changes.append({"site": site, "before": call, "after": checked(call, site)})
    proposed = proposed[:start] + region + proposed[stop:]

header = '''// Successful adapter results establish the primary Stager's ownership transitions.
// Failure stops the native process; it does not establish completion or GPU recovery.
#pragma once
#include <cstdio>
#include <cstdlib>

namespace strata::prefill::stage_ownership {
template <class Status>
inline Status checked(Status status, Status success, const char* operation,
                      const char* site, const char* file, int line,
                      const char* (*describe)(Status)) {
    if (status != success) {
        std::fprintf(stderr,
                     "prefill: stage ownership failure site=%s operation=%s error=%d (%.512s) at %s:%d; aborting native\\n",
                     site, operation, (int) status, describe(status), file, line);
        std::fflush(stderr);
        std::abort();
    }
    return status;
}
} // namespace strata::prefill::stage_ownership

#if defined(STRATA_ENABLE_SYCL)
// Evaluate each API call once. Keep its thread-local error detail intact.
#define STRATA_PREFILL_STAGE_CHECK(call, site) \\
    ::strata::prefill::stage_ownership::checked( \\
        (call), cudaSuccess, #call, (site), __FILE__, __LINE__, &cudaGetErrorString)
#else
#define STRATA_PREFILL_STAGE_CHECK(call, site) (call)
#endif
'''
if (ROOT / HEADER).exists():
    raise RuntimeError("proposed new helper path already exists; rebase/review instead of overwriting")
edit('#include "strata/prefill/prefill.hpp"',
     '#include "strata/prefill/prefill.hpp"\n#include "strata/prefill/stage_ownership.hpp"', "helper include")
edit("        cudaGetDevice(&device);", "        " + checked("cudaGetDevice(&device)", "Stager::init device") + ";", "Stager device identity")
edit("        cudaSetDevice(device);", "        " + checked("cudaSetDevice(device)", "Stager::work device") + ";", "Stager worker device")
edit("                cudaEventSynchronize(dma_done[b]);", "                " + checked("cudaEventSynchronize(dma_done[b])", "Stager::work host-slot reuse") + ";", "Stager wait before overwrite/ready")
edit("        cudaEventRecord(dma_done[j % kRing], copy);", "        " + checked("cudaEventRecord(dma_done[j % kRing], copy)", "Stager::issued_one host-slot DMA") + ";", "Stager record before issued")

issue_calls = [
    "cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0)",
    "cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes, cudaMemcpyHostToDevice, m.copy)",
    "cudaMemcpyAsync(m.stage_dev[sl], hb, bytes, cudaMemcpyHostToDevice, m.copy)",
    "cudaEventRecord(m.copied[sl], m.copy)",
]
check_region("        auto issue_until =", "        // D-5: the stream is issued", issue_calls, "inline issuer stage copy")
check_region("            issuer = std::thread([&] {", "        } else if (stream_all)", issue_calls, "threaded issuer stage copy")
edit("                const core::OnDevice od(m.device);",
'''#if defined(STRATA_ENABLE_SYCL)
                STRATA_PREFILL_STAGE_CHECK(cudaSetDevice(m.device), "threaded issuer device");
#else
                const core::OnDevice od(m.device);
#endif''', "threaded issuer checked device selection")

# stage_one has the same slot wait in two distinct branches. Keep both call locations.
check_region("                            if (pinned) {\n                                // DMA straight", "                            } else {\n                                // copied to a pinned buffer", [
    "cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0)",
    "cudaMemcpyAsync(m.stage_dev[sl], b, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy)",
], "stage_one pinned copy")
check_region("                            } else {\n                                // copied to a pinned buffer", "                            m.stage_live[sl] = true;", [
    "cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0)",
    "cudaMemcpyAsync(m.stage_dev[sl], hb, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy)",
    "cudaEventRecord(m.copied[sl], m.copy)",
], "stage_one stager/copy publication")

# Device-slot consumers: preserve the existing exact set/order of waits and release events.
# No new copied-wait is added to skipped entries.
check_region("                            b.e1 = e;", "                            if (k > k0) {\n                                consumed = k;", [
    "cudaStreamWaitEvent(m.cs, m.copied[(k - 1) % (size_t) m.ring], 0)",
    "cudaEventRecord(m.used[kk % (size_t) m.ring], m.cs)",
], "fused streamed consumer lease")
check_region("                        auto flush = [&]() {", "                        // one expert's products", [
    "cudaStreamWaitEvent(m.cs, m.copied[gg_slots[gg_nslots - 1]], 0)",
    "cudaEventRecord(m.used[rel], m.cs)",
], "group gather consumer lease")
check_region("                                if (slot >= 0 && !group_gather) {", "                                if (q + 1 < MMQ_GROUP", [
    "cudaEventRecord(m.used[slot], m.cs)",
], "MMQ individual slot release")
check_region("                            const int q = (int) (j % DQ);", "                            const int64_t o0 =", [
    "cudaEventRecord(m.used[slot], m.cs)",
], "FP16 dequant slot release")
check_region("                        if (!stream_all) {\n                            size_t staged", "                        } else {\n                            // the streamed walk", [
    "cudaStreamWaitEvent(m.cs, m.copied[stage_of[j]], 0)",
], "nonstreamed consumer copy wait")
check_region("                            auto release_to =", "                            for (size_t j = 0; j < order.size(); ++j)", [
    "cudaEventRecord(m.used[sl], m.cs)",
], "streamed skipped slot release")
check_region("                            for (size_t j = 0; j < order.size(); ++j) {\n                                const int32_t e = order[j];\n                                release_to(e);", "                            release_to(m.g->n_expert);", [
    "cudaStreamWaitEvent(m.cs, m.copied[sl], 0)",
], "streamed consumer copy wait")

# Existing drains, not new waits. Stop before reclamation if completion is unknown.
edit("    if (impl_->cs) cudaStreamSynchronize(impl_->cs);",
     "    if (impl_->cs) " + checked("cudaStreamSynchronize(impl_->cs)", "Prefill::release compute drain") + ";", "release compute completion before free")
edit("    if (impl_->copy) cudaStreamSynchronize(impl_->copy);",
     "    if (impl_->copy) " + checked("cudaStreamSynchronize(impl_->copy)", "Prefill::release copy drain") + ";", "release copy completion before free")
check_region("    // (PR #121) an expert copy that failed", "    stats_.ms_total +=", [
    "cudaStreamSynchronize(m.copy)",
], "Prefill::run final copy drain")
# The immediately preceding final compute drain shares the same explicit failure boundary.
before = '''    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    // (PR #121)'''
after = before.replace("cudaStreamSynchronize(m.cs)", checked("cudaStreamSynchronize(m.cs)", "Prefill::run final compute drain"))
edit(before, after, "final compute completion before successful run")

patch = "".join(difflib.unified_diff(original.splitlines(True), proposed.splitlines(True), fromfile="a/"+SOURCE, tofile="b/"+SOURCE))
patch += "".join(difflib.unified_diff([], header.splitlines(True), fromfile="/dev/null", tofile="b/"+HEADER))
digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
(OUT / "unified.patch").write_text(patch)
identity = {
    "status": "artifact-only; unapplied, uncompiled, unrun, unadopted",
    "base_branch": "feature/sycl",
    "base_revision": "No Git action; source hashes are authoritative. R534 cited 24158fb7f82012af0e425d10b1de8c018d9ab869; current HEAD not asserted.",
    "sources": [{"path": SOURCE, "base_sha256": digest(original), "proposed_sha256": digest(proposed)},
                {"path": HEADER, "base_sha256": None, "proposed_sha256": digest(header)}],
    "patch_sha256": digest(patch),
    "guarded_transition_count": sum(1 for c in changes if "STRATA_PREFILL_STAGE_CHECK" in c["after"]),
    "behavior": "SYCL failure: flush diagnostic and abort native; success: existing operations/order; non-SYCL macro: original expression",
}
(OUT / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
(OUT / "transition-sites.json").write_text(json.dumps(changes, indent=2) + "\n")
# Source extraction is artifact fabrication, not fixture execution or product import.
def region(text, start, end):
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]
stager = region(proposed, "struct Stager {", "\n// multi-GPU: the peer GPU's share")
issuer = region(proposed, "        auto issue_until =", "        // D-5: the stream is issued")
release = region(proposed, "void Prefill::release() {", "\nnamespace {\nconstexpr int64_t GEMM_SCRATCH")
release = release.replace("void Prefill::release()", "void extracted_release(FakeImpl* impl_)", 1)
template = (OUT / "cpu_fixture_template.cpp.in").read_text()
fixture = template.replace("@STAGER@", stager).replace("@ISSUER@", issuer).replace("@RELEASE@", release)
fixture = "// Proposed source SHA256: " + digest(proposed) + "\n" + fixture
(OUT / "cpu_fixture.cpp").write_text(fixture)
print(json.dumps({"patch_sha256": identity["patch_sha256"], "guarded_transitions": identity["guarded_transition_count"], "product_files_written": 0}))
