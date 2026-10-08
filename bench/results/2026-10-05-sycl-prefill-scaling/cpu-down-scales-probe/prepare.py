"""Derive a single IQ4_NL change from unchanged upstream CPU source."""
import difflib
from pathlib import Path
import sys

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = Path(sys.argv[1]).resolve()
source = (root / 'src/kernels/cpu/iq_avx2.cpp').read_text()
needle = '    const int nb = n / QK4_NL;\n'
assert source.count(needle) == 1
candidate = source.replace(needle, needle + '''    // Native down activations are read-only across rows. Preserve the exact
    // half conversion and dx * dy product; cache only the repeated conversion.
    // Larger reductions retain the original path without exceeding this array.
    float activation_scales[NT][80];
    if (nb <= 80)
        for (int t = 0; t < NT; ++t)
            for (int ib = 0; ib < nb; ++ib)
                activation_scales[t][ib] = h2f(y[t][ib].d);
''')
needle = 'dx * h2f(b.d)'
assert candidate.count(needle) == 1
candidate = candidate.replace(needle, 'dx * (nb <= 80 ? activation_scales[t][ib] : h2f(b.d))')
assert candidate.count('namespace strata::kernels::cpu {') == 1
candidate = candidate.replace('namespace strata::kernels::cpu {', 'namespace strata::kernels::cpu_scales {')
candidate = candidate.replace('namespace strata::kernels::cpu\n', 'namespace strata::kernels::cpu_scales\n')
(out / 'candidate.cpp').write_text(candidate)
(out / 'candidate.diff').write_text(''.join(difflib.unified_diff(source.splitlines(keepends=True), candidate.splitlines(keepends=True), fromfile='upstream/iq_avx2.cpp', tofile='probe/candidate.cpp')))
probe = (root / 'bench/results/2026-10-05-sycl-prefill-scaling/cpu-gcc-down-probe/probe.cpp').read_text()
probe = probe.replace('namespace strata::kernels::cpu {', 'namespace strata::kernels::cpu_scales {')
probe = probe.replace('iq4nl256_down_rows_gcc', 'iq4nl256_down_rows')
probe = probe.replace('strata::kernels::cpu::iq4nl256_down_rows', 'strata::kernels::cpu_scales::iq4nl256_down_rows')
probe = probe.replace('gcc_ms', 'candidate_ms').replace('{32, 96, 640, 768, 2560}', '{32, 96, 640, 768, 2560, 4096}')
(out / 'probe.cpp').write_text(probe)
