#!/usr/bin/env python3
"""Extract the ORIGINAL layout selector; refuse changed upstream reference sources."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--ggml', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
m = json.loads((root / 'docs/sycl-audit/source-inventory.json').read_text())
hashes = {r['path']: r['sha256'] for r in m['external_sources']}
for path in ('ggml/src/ggml-cuda/mmq.cuh', 'ggml/src/ggml-cuda/quantize.cu',
             'ggml/src/ggml-cuda/quantize.cuh', 'ggml/include/ggml.h'):
    if hashlib.sha256((a.ggml / path).read_bytes()).hexdigest() != hashes[path]:
        raise SystemExit('Pinned upstream source mismatch: ' + path)
s = (a.ggml / 'ggml/src/ggml-cuda/mmq.cuh').read_text()
enum = s[s.index('enum mmq_q8_1_ds_layout {'):s.index('\n};', s.index('enum mmq_q8_1_ds_layout {')) + 3]
start = s.index('static mmq_q8_1_ds_layout mmq_get_q8_1_ds_layout')
selector = s[start:s.index('\n}\n', start) + 3]
a.output.write_text('// Generated verbatim from pinned GGML mmq.cuh (MIT).\n' +
    '#include <ggml.h>\n#include <stdexcept>\n' +
    '#pragma push_macro("GGML_ABORT")\n#undef GGML_ABORT\n#define GGML_ABORT(...) throw std::invalid_argument("upstream invalid type")\n' +
    'namespace upstream_reference {\n' + enum + '\n' + selector + '\n}\n#pragma pop_macro("GGML_ABORT")\n')
