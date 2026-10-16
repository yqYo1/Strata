#!/usr/bin/env python3
"""Generate a compact proposal only. Never writes product source; no tests/builds."""
from pathlib import Path
import hashlib, json, difflib
D=Path(__file__).resolve().parent
ROOT=D.parents[3]
# Parents: recovery audit -> results -> bench -> repo.
paths=['src/prefill/prefill.cpp','src/prefill/layer_major.inc']
base={p:(ROOT/p).read_text() for p in paths}
new=dict(base)
def replace(path, old, replacement):
    assert new[path].count(old)==1,(path,old)
    new[path]=new[path].replace(old,replacement)
p=paths[0]
replace(p,'#include "strata/prefill/layer26_phase_audit.hpp"', '#include "strata/prefill/layer26_phase_audit.hpp"\n#include "strata/prefill/layermajor_shape_census.hpp"\n#include <new>')
replace(p,'    bool layer_major_active = false;', '    bool layer_major_active = false;\n    detail::LayerMajorShapeCensus* layer_major_census = nullptr;')
replace(p,'                        ro.counts(l, T, m.cnt);','                        ro.counts(l, T, m.cnt);\n                        if (m.layer_major_census) m.layer_major_census->counts(l, T, m.cnt.data(), m.cnt.size());')
replace(p,'                            ro.gu(ne, slot);','                            ro.gu(ne, slot);\n                            if (m.layer_major_census) m.layer_major_census->issue(true, l, ne, e,\n                                m.host_res ? m.host_res[size_t(l * NE + e)] : -1, slot);')
replace(p,'                            ro.down(ne);','                            ro.down(ne);\n                            if (m.layer_major_census) m.layer_major_census->issue(false, l, ne, e,\n                                m.host_res ? m.host_res[size_t(l * NE + e)] : -1, slot);')
replace(p,'        const bool threaded_issue = stream_all && issuer_on;', '        const bool threaded_issue = stream_all && issuer_on;\n        if (m.layer_major_census && threaded_issue) m.layer_major_census->reject("threaded_issuer_unsupported");')
p=paths[1]
replace(p,'    int64_t logical_chunks = 0, layer = 0;', '''    // Optional owner precedes borrowed pointer, Restore and checked lease.
    // Failure to qualify/allocate the observation does not change inference.
    std::unique_ptr<detail::LayerMajorShapeCensus> census;
    const char* census_switch = std::getenv("STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS");
    if (census_switch && std::strcmp(census_switch, "1") == 0) {
        uint64_t need = detail::layer_major_add(host_need, audit ? 4ull << 20 : 0);
        need = detail::layer_major_add(need, phase_audit ? detail::Layer26PhaseAudit::host_reservation : 0);
        need = detail::layer_major_add(need, sizeof(detail::LayerMajorShapeCensus));
        if (core::conversation_memory_admit(core::conversation_available_memory(), need, host_headroom))
            census.reset(new (std::nothrow) detail::LayerMajorShapeCensus(n, pos0, m.T));
        if (!census) std::fputs("strata lm census: {\\"version\\":1,\\"complete\\":false,\\"reason\\":\\"host_observation_unavailable\\"}\\n", stderr);
    }
    detail::LayerMajorShapeBorrow census_borrow(m.layer_major_census, census.get());
    int64_t logical_chunks = 0, layer = 0;''')
replace(p,'        stage_lb_ = layer; stage_le_ = layer + 1;', '''        if (census) {
            const auto& f = layout.fmt[size_t(layer)];
            // run_impl stream_all is !layer_major_active && ...; therefore its
            // copy issuer cannot start here. Compute/count hooks run inline.
            census->begin(layer, {f.gu_type, f.d_type, f.gu_row, f.d_row,
                f.up_off, f.down_off, f.bytes}, residency.data() + layer * NE,
                m.layer_major_active && !m.pp && !next_ && m.copy == m.cs);
        }
        stage_lb_ = layer; stage_le_ = layer + 1;''')
replace(p,'        if (!cursor.complete()) { err = "prefill layer-major: incomplete layer callbacks"; return false; }','        if (!cursor.complete()) { err = "prefill layer-major: incomplete layer callbacks"; return false; }\n        if (census) census->end(layer);')
replace(p,'    lease.retire();\n    return true;', '    lease.retire();\n    if (census) census->complete();\n    return true;')
header='include/strata/prefill/layermajor_shape_census.hpp'
new[header]=(D/'layermajor_shape_census.hpp.in').read_text()
patch=''.join(''.join(difflib.unified_diff(base.get(p,'').splitlines(True),s.splitlines(True),fromfile='a/'+p if p in base else '/dev/null',tofile='b/'+p)) for p,s in new.items())
(D/'source.diff').write_text(patch)
sha=lambda b:hashlib.sha256(b).hexdigest()
(D/'source-pins.json').write_text(json.dumps({'base':{p:sha(s.encode()) for p,s in base.items()},'proposed':{p:sha(s.encode()) for p,s in new.items()},'diff_sha256':sha(patch.encode()),'status':'PROPOSAL_ONLY_NOT_COMPILED_NOT_RUN'},indent=2)+'\n')
