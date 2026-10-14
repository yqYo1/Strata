#!/usr/bin/env python3
"""Create a CMake-only review diff inside this artifact directory."""
from pathlib import Path
import difflib
import hashlib
import json

here = Path(__file__).resolve().parent
root = here.parents[3]
relative = "cmake/sycl_backend.cmake"
before = (root / relative).read_text()
anchor = "add_executable(strata src/program/generate.cpp)\n"
if before.count(anchor) != 1:
    raise RuntimeError("expected one unchanged CMake insertion anchor")
insertion = """# Standalone existing matrix-Direct comparison; no production kernel/default change.
add_executable(sycl_matrix_direct_probe EXCLUDE_FROM_ALL tools/sycl/matrix_direct_probe.cpp)
target_link_libraries(sycl_matrix_direct_probe PRIVATE strata_kernels)
target_compile_options(sycl_matrix_direct_probe PRIVATE -fno-fast-math -ffp-contract=off)
"""
after = before.replace(anchor, insertion + anchor, 1)
patch = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
    fromfile="a/" + relative, tofile="b/" + relative))
(here / "product.patch").write_text(patch)
sources = [relative, "src/kernels/sycl/prompt_attention.cpp", "tests/sycl/decode_attention.cpp",
           "tests/sycl/qsa_fallback_layout.cpp", "include/strata/kernels/qsa_prompt_attn.hpp",
           "include/strata/sycl/runtime.hpp", "src/sycl/runtime.cpp",
           "include/strata/sycl/checked_cleanup.hpp"]
(here / "source-base-sha256.json").write_text(json.dumps({p: hashlib.sha256((root / p).read_bytes()).hexdigest()
    for p in sources}, indent=2) + "\n")
print(f"Wrote {len(patch.encode())} CMake-only patch bytes; product files unchanged")
