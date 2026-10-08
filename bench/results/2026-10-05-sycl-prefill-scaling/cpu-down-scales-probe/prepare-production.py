"""Use actual CMake scale code and reuse the complete production-pool harness."""
from pathlib import Path

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
base = root / 'bench/results/2026-10-05-sycl-prefill-scaling'
probe = Path.home() / '.local/state/strata-sycl/cpu-down-scales-probe'
pool = Path.home() / '.local/state/strata-sycl/cpu-down-scales-pool-check'
source = (probe / 'probe.cpp').read_text()
source = source.replace('namespace strata::kernels::cpu_scales {', 'namespace strata::kernels::cpu {')
source = source.replace('strata::kernels::cpu_scales::iq4nl256_down_rows', 'strata::kernels::cpu::iq4nl256_down_rows_scales')
source = source.replace('namespace strata::kernels::cpu {\nvoid iq4nl256_down_rows(',
                        'namespace strata::kernels::cpu {\nvoid iq4nl256_down_rows_scales(')
(probe / 'production-probe.cpp').write_text(source)
(pool / 'probe.cpp').write_text((base / 'cpu-gcc-pool-probe/probe.cpp').read_text())
# Replace complete strings/paths as well as isolated arm labels.
source = (base / 'cpu-gcc-pool-probe/run.py').read_text().replace('default', 'baseline').replace('gcc', 'scales')
(pool / 'run.py').write_text(source)
