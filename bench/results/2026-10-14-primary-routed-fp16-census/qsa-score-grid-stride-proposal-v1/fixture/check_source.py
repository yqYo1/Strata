#!/usr/bin/env python3
"""UNRUN structural fixture: no product import/execute; exact baseline/body proof."""
from pathlib import Path
import hashlib

out = Path(__file__).resolve().parents[1]
root = out.parents[3]
base = (root / "src/kernels/sycl/qsa_select.cpp").read_text()
assert hashlib.sha256(base.encode()).hexdigest() == "45084756ca5e8e36085dae46581049c81366a45cdfde23e8c6f38defbba0c809"
proposed = (out / "proposed/src/kernels/sycl/qsa_select.cpp").read_text()
begin = proposed.index("  // Process-latched opt-in;")
end = proposed.index("\n  auto event = queue_for(stream).parallel_for(", begin) + 1
branch = proposed[begin:end]
without_branch = proposed[:begin] + proposed[end:]
without_branch = without_branch.replace('#include "strata/kernels/qsa_score_grid_stride.hpp"\n', '', 1)
without_branch = without_branch.replace('#include <cstdlib>\n', '', 1)
assert without_branch == base, "off kernel or unrelated source changed"
start = base.index("        const int lane = it.get_local_linear_id();")
stop = base.index("\n      });", start)
old_body = base[start:stop]
start = branch.index("            const int lane = it.get_local_linear_id();")
stop = branch.index("\n          });", start)
new_body = branch[start:stop]
assert '\n'.join(line[4:] for line in new_body.splitlines()) == old_body
assert 'std::getenv("STRATA_SYCL_QSA_SCORE_GRID_STRIDE")' in branch
assert 'grid.for_each_block(n, nb, first,' in branch
assert 'size_t(nq * grid.group_count) * 32' in branch
assert branch.count('queue_for(stream)') == branch.count('finish(stream, event)') == 1
validator = base[base.index('int64_t validate('):base.index('\nuint32_t order(')]
assert validator in (out / "fixture/extracted_validate.hpp").read_text()
print("PASS unchanged off kernel, byte-identical FP statement order, unchanged real validator; no GPU correctness claim")
