#!/usr/bin/env python3
"""Produce an unapplied proposal; do not compile, test or write product source."""
from pathlib import Path
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PATH = "src/prefill/prefill.cpp"
BASE = "6b4b7a71175471d7d800e40f98ec62236bf57e1cfd57d9d41f4863d4653c2104"
source = (ROOT / PATH).read_text()
digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
if digest(source) != BASE:
    raise RuntimeError("Source changed; rebase and review rather than apply a stale proposal")
before = "    f(T * N); f(T * D); f(T * D); o.take<uint16_t>(T * D, ok); f(T * LR); o.take<uint16_t>(T * LR, ok);"
after = """    f(T * N); f(T * D);
    if (gr_unfused()) f(T * D);   // carve keeps xn only for gr_norm + gr_mix
    f(T * HC);                  // grs is allocated in both residual paths
    o.take<uint16_t>(T * D, ok); f(T * LR); o.take<uint16_t>(T * LR, ok);"""
if source.count(before) != 1:
    raise RuntimeError("Estimator anchor is not unique")
proposed = source.replace(before, after, 1)
patch = "".join(difflib.unified_diff(source.splitlines(True), proposed.splitlines(True),
                                    fromfile="a/" + PATH, tofile="b/" + PATH))
(OUT / "unified.patch").write_text(patch)
identity = {
    "status": "unapplied-uncompiled-unrun-unmeasured-unadopted",
    "source": PATH,
    "base_sha256": BASE,
    "proposed_sha256": digest(proposed),
    "patch_sha256": digest(patch),
    "scope": "Estimator's residual-prefix allocations only; carve and inference unchanged",
    "source_derived_cells": [],
}
for chunk in (4096, 8192):
    xn = ((chunk * 10240 * 4 + 256 + 255) & ~255)
    grs = ((chunk * 4 * 4 + 256 + 255) & ~255)
    identity["source_derived_cells"].append({
        "chunk": chunk,
        "fused_estimator_decrease_bytes": xn - grs,
        "unfused_estimator_increase_bytes": grs,
        "carve_allocation_change_bytes": 0,
    })
(OUT / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
print(json.dumps({"patch_sha256": identity["patch_sha256"], "product_files_written": 0}))
