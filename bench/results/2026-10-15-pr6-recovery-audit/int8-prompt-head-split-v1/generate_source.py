#!/usr/bin/env python3
"""Generate proposal artifacts only. Does not apply, compile or run a test."""
from pathlib import Path
import difflib
P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
A = P / 'bench/results/2026-10-15-pr6-recovery-audit/int8-prompt-head-split-v1'
path = 'src/kernels/sycl/prompt_attention.cpp'
base = (P / path).read_text()
start = base.index('template <int Mode, bool Direct, bool Native = false>\nvoid launch(')
end = base.index('// Select distinct kernels', start)
candidate = base[start:end]
def replace(old, new):
    global candidate
    if old not in candidate:
        raise RuntimeError('Source drift: ' + old[:100])
    candidate = candidate.replace(old, new)
replace('template <int Mode, bool Direct, bool Native = false>\nvoid launch(', '''// Experimental Direct-only INT8 candidate. Keep launch<> above unchanged.
// C32 preserves chunk boundaries and scalar/MAD order; parity remains unproven.
// C64 changes softmax/factor/PV chunk boundaries and requires tolerance gates.
template <int Cells>
void launch_mode1_head_split(''')
replace('  constexpr int KG = scale_groups<Mode, false>(),\n                VG = scale_groups<Mode, true>();', '''  static_assert(Cells == 32 || Cells == 64);
  constexpr int R = 8, W = 64, KG = 4, VG = 4;''')
# Only the copied candidate is rewritten. External HEADS=12 stays unchanged.
import re
candidate = re.sub(r'\bCH\b', 'Cells', candidate)
candidate = re.sub(r'\bWG\b', 'W', candidate)
replace('qhi(16 * HD, h), qlo(16 * HD, h)', 'qhi(R * HD, h), qlo(R * HD, h)')
replace('phi(VG * 16 * Cells, h)', 'phi(VG * R * Cells, h)')
replace('plo(VG * 16 * Cells, h)', 'plo(VG * R * Cells, h)')
replace('scores(16 * Cells, h)', 'scores(R * Cells, h)')
replace('maxima(16, h), sums(16, h), alpha(16, h),\n        factors(VG, h), final(Direct ? 1 : 16 * HD, h)', 'maxima(R, h), sums(R, h), alpha(R, h),\n        factors(VG, h)')
replace('size_t(queries) * shapes.n_head_kv * W, W', 'size_t(queries) * size_t(shapes.n_head_kv) * 2 * W, W')
replace('''          const size_t index = it.get_group_linear_id(),
                       batch = index / shapes.n_head_kv,
                       kvhead = index % shapes.n_head_kv;
          const int tid = it.get_local_linear_id(), subgroup = tid / 16;
          const int hb = subgroup / 4, db = subgroup % 4;''', '''          const size_t index = it.get_group_linear_id(), tile = index % 2,
                       kvhead = (index / 2) % size_t(shapes.n_head_kv),
                       batch = index / (2 * size_t(shapes.n_head_kv));
          const int tid = it.get_local_linear_id(), subgroup = tid / 16;
          const int head_base = int(tile) * R, live_heads = tile == 0 ? R : 4;
          const int db = subgroup;''')
replace('''          float qm = 0;''', '''          // Both WGs rescan ALL 12 heads. Tile-local max would alter hi/lo.
          float qm = 0;''')
replace('''          for (int i = tid; i < 16 * HD; i += W) {
            const float x = i < HEADS * HD ? q[i] * qup : 0.f;''', '''          for (int i = tid; i < R * HD; i += W) {
            const float x = i / HD < live_heads ? q[head_base * HD + i] * qup : 0.f;''')
replace('if (tid < 16)', 'if (tid < R)')
replace('for (int c0 = 0; c0 < count; c0 += Cells)', '''// int64_t also avoids final-increment overflow for count==INT_MAX.
          for (int64_t c0 = 0; c0 < count; c0 += Cells)''')
replace('batch * cap + c0 + tid', 'batch * size_t(cap) + size_t(c0) + size_t(tid)')
replace('i < 16 * Cells', 'i < R * Cells')
replace('half_code<Mode, false, Native>', 'half_code<1, false, false>')
replace('half_code<Mode, true, Native>', 'half_code<1, true, false>')
replace('scale<Mode, false>', 'scale<1, false>')
replace('scale<Mode, true>', 'scale<1, true>')
replace('''            if (subgroup < 4) {
              const int head = (subgroup / 2) * 8, cell = (subgroup % 2) * 16;''', '''            // SG0/1 cover C32 QK; SG2/3 still reach every WG barrier.
            // C64 uses all four subgroups. Each branch is subgroup-uniform.
            if (subgroup < Cells / 16) {
              const int head = 0, cell = subgroup * 16;''')
# Real heads retain the old statement order. Pad rows publish zero P/state.
replace('''            if (tid < R) {
              float m = maxima[tid];''', '''            if (tid < R) {
              if (tid < live_heads) {
              float m = maxima[tid];''')
replace('''              alpha[tid] = a;
            }
            for (int i = tid; i < Cells * VG;''', '''              alpha[tid] = a;
              } else {
                for (int c = 0; c < Cells; ++c)
                  scores[tid * Cells + c] = 0.f;
                alpha[tid] = 0.f;
              }
            }
            for (int i = tid; i < Cells * VG;''')
replace('VG * 16 * Cells', 'VG * R * Cells')
replace('i / (16 * Cells), r = (i / Cells) % 16', 'i / (R * Cells), r = (i / Cells) % R')
replace('(vg * 16 + hb * 8) * Cells + k', '(vg * R) * Cells + k')
replace('alpha[hb * 8 + r]', 'alpha[r]')
replace('sums[hb * 8 + r]', 'sums[r]')
# Replace the output-only tail, leaving division and accumulator order intact.
output_start = candidate.index('              if constexpr (Direct) {')
output_end = candidate.rindex('        });')
candidate = candidate[:output_start] + '''              if (r < size_t(live_heads))
                output[(batch * shapes.n_head + kvhead * HEADS + head_base + r) *
                           HD + (db * 4 + j) * 16 + c] = v;
            });
          }
''' + candidate[output_end:]
# Improve indentation of the unchanged scalar body inside its new live-row branch.
a = candidate.index('              float m = maxima[tid];')
b = candidate.index('              } else {', a)
candidate = candidate[:a] + ''.join('  ' + line if line.strip() else line for line in candidate[a:b].splitlines(True)) + candidate[b:]
proposal = base[:end] + candidate + base[end:]
proposal = proposal.replace('#include <cstdlib>\n', '#include <cstdlib>\n#include <cstring>\n', 1)
anchor = '''  if (direct && direct[0] == '1') {
'''
insertion = '''    if constexpr (Mode == 1) {
      const char *split = std::getenv("STRATA_SYCL_PROMPT_HEAD_SPLIT");
      // Exact new opt-in. Existing Direct prefix semantics remain below.
      if (direct[1] == '\\0' && split) {
        if (std::strcmp(split, "32") == 0) {
          launch_mode1_head_split<32>(q, p, ids, steps, cap, s, out, nq, stream);
          return;
        }
        if (std::strcmp(split, "64") == 0) {
          launch_mode1_head_split<64>(q, p, ids, steps, cap, s, out, nq, stream);
          return;
        }
      }
    }
'''
if proposal.count(anchor) != 1:
    raise RuntimeError('Dispatch insertion anchor changed')
proposal = proposal.replace(anchor, anchor + insertion, 1)
(A / 'files' / path).write_text(proposal)
(A / 'source.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True), proposal.splitlines(True), fromfile='a/' + path, tofile='b/' + path)))
