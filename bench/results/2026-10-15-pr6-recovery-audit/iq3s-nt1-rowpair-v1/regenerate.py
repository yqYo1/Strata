#!/usr/bin/env python3
"""Source preparation only: pin bytes and generate an unapplied three-file diff.

No Git, compiler, test, service, GPU, network, or product-tree writes.
--init is used once to record the currently inspected source bytes.
"""
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys

A = Path(__file__).resolve().parent
P = A.parents[3]
G = Path('/home/yayoi/ghq/github.com/ggml-org/llama.cpp/.worktree/dep-ggml-3cf-xe-e32-20261011/ggml')
targets = ['include/strata/kernels/cpu/iq_avx2.hpp',
           'src/kernels/cpu/iq_single_avx2.cpp', 'src/kernels/cpu/native_expert.cpp']
sources = targets + ['src/kernels/cpu/iq_avx2.cpp', 'include/strata/kernels/cpu/native_expert.hpp',
    'CMakeLists.txt', 'third_party/ggml/ggml-common.h', 'third_party/ggml/VERSION.txt',
    'third_party/ggml/LICENSE', 'tests/core/iq_single_exact_test.cpp',
    'include/strata/kernels/cpu/expert.hpp', 'include/strata/kernels/cpu/iq_avx512.hpp',
    'include/strata/kernels/cpu/kq_avx2.hpp', 'include/strata/kernels/cpu/expert_layout.hpp',
    'docs/ARTIFACT_RETENTION.md', '.claude/skills/strata-upstream-sync/SKILL.md']
ggml_sources = ['src/ggml-common.h', 'src/ggml-quants.c', 'src/ggml-quants.h',
    'src/ggml-cpu/arch/x86/quants.c', 'src/ggml-cpu/ggml-cpu.c',
    'include/ggml.h', 'include/ggml-cpu.h', 'include/ggml-backend.h', 'include/ggml-alloc.h']
reports = ['research-index.json', 'boundary-service-main-review.ja.md',
    'research-native-gu-service-round582-v1.txt', 'decode-ledger-final.ja.md',
    'decode-format-service-v2/review.md']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def source_pins():
    result = {str(P / p): sha(P / p) for p in sources}
    result.update({str(G / p): sha(G / p) for p in ggml_sources})
    return result
if sys.argv[1:] == ['--init']:
    if (A/'pins.json').exists(): raise SystemExit('pins.json already exists; refusing to replace the review boundary')
    pins = {'status': 'proposal_only_uncompiled_unrun_unapplied',
        'git_identity': 'not queried; current source bytes are authoritative',
        'source_pin_scope': 'All proposed target source and project/header compile inputs inspected here; GGML archive source/flag closure and toolchain/system headers require Main build receipt.',
        'configured_dependency': str(G),
        'dependency_note': 'Path from committed effective-build-flags.json, not newly queried build state',
        'source_sha256': source_pins(),
        'reference_sha256': {str(A.parent/p): sha(A.parent/p) for p in reports}}
    build_ref = P/'bench/results/2026-10-14-primary-native-runtime/effective-build-flags.json'
    pins['reference_sha256'][str(build_ref)] = sha(build_ref)
    (A/'pins.json').write_text(json.dumps(pins,indent=2)+'\n')
elif sys.argv[1:]: raise SystemExit('usage: regenerate.py [--init]')
pins = json.loads((A/'pins.json').read_text())
for path, wanted in pins['source_sha256'].items():
    if sha(Path(path)) != wanted: raise SystemExit(f'source pin changed: {path}')

def once(text, anchor, replacement):
    if text.count(anchor) != 1: raise SystemExit(f'anchor count != 1: {anchor[:80]!r}')
    return text.replace(anchor,replacement,1)
old = {p: (P/p).read_text() for p in targets}
new = dict(old)
header = targets[0]
anchor = 'bool iq256_supported(int ggml_type) noexcept;'
new[header] = once(old[header],anchor,
    '// Experimental IQ3_S nt=1 row pair, packed 2560/640 GU only.\n'
    '// Caller supplies complete non-overlapping buffers; false means no writes.\n'
    'bool iq256_iq3s_rowpair_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off,\n'
    '                               int n, const void* act, float* ff, int r0, int r1);\n\n'+anchor)
kernel = targets[1]
anchor = '} // namespace\nbool iq256_single_gu_rows'
new[kernel] = once(old[kernel],anchor,(A/'rowpair_kernel.inc').read_text()+'\n'+anchor)
anchor = '} // namespace strata::kernels::cpu\n#else'
new[kernel] = once(new[kernel],anchor,(A/'rowpair_public.inc').read_text()+'\n'+anchor)
anchor = '} // namespace strata::kernels::cpu\n#endif'
new[kernel] = once(new[kernel],anchor,
    'bool iq256_iq3s_rowpair_gu_rows(int, const uint8_t*, size_t, size_t, int, const void*, float*, int, int) {\n'
    '    return false;\n}\n'+anchor)
dispatch = targets[2]
anchor = '    // Opt-in Zen 3 single-token kernels retain GGML\'s float reduction order.'
new[dispatch] = once(old[dispatch],anchor,
    '    // Probe candidate: independent output-row ILP, not multi-token weight reuse.\n'
    '    // Keep the multi-token/MT_MIN selection above, including its nt=1 override.\n'
    '    static const bool rowpair = [] {\n'
    '        const char* v = std::getenv("STRATA_IQ3S_NT1_ROWPAIR");\n'
    '        return v && v[0] == \'1\' && v[1] == \'\\0\';\n'
    '    }();\n'
    '    if (rowpair && avx2 && !cpu512 && nt == 1 && f.n_embd == 2560 && f.n_ff == 640 &&\n'
    '        f.gu_act == GGML_TYPE_Q8_K &&\n'
    '        iq256_iq3s_rowpair_gu_rows(f.gu_type, blob, f.gu_row, f.up_off,\n'
    '                                  (int) f.n_embd, act[0], ff[0], r0, r1))\n'
    '        return;\n'+anchor)
diff = ''.join(''.join(difflib.unified_diff(old[p].splitlines(True),new[p].splitlines(True),
              fromfile='a/'+p,tofile='b/'+p,n=3)) for p in targets)
(A/'source.diff').write_text(diff)
table = (P/'third_party/ggml/ggml-common.h').read_text().split(
    'GGML_TABLE_BEGIN(uint32_t, iq3s_grid, 512)',1)[1].split('GGML_TABLE_END()',1)[0]
entries = [int(x,16) for x in re.findall(r'0x[0-9a-fA-F]+',table)]
if len(entries) != 512: raise SystemExit('IQ3_S table entry count changed')
max_magnitude = max((word >> shift)&255 for word in entries for shift in (0,8,16,24))
manifest = {'source_diff_sha256': sha(A/'source.diff'),
    'iq3s_grid_entries':len(entries),'iq3s_max_grid_byte':max_magnitude,
    'int32_block_lane_abs_bound': 32*max_magnitude*128*31,
    'proposed_target_sha256': {p: hashlib.sha256(new[p].encode()).hexdigest() for p in targets},
    'artifact_sha256': {p.name:sha(p) for p in sorted(A.iterdir()) if p.is_file()
        and p.name not in {'manifest.json','source.diff','pins.json'}}}
(A/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'source_diff_sha256':manifest['source_diff_sha256'], 'product_writes':0}))
