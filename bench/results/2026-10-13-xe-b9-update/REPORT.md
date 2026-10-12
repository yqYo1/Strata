<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# XeStrata update observed during finalization

XeStrata advanced from da3d426 to b9f00e9 while Main finalized the official
v0.1.43 server imports. The new tree calls itself xe0.1.43 and integrates official
`7695859`. It changes 148 files, including SYCL kernels, memory, models and benchmark
setup. It is fetched into an independent ghq worktree with no tracked source
patch. The official selective server ports are already committed at 17879c7.

Both free and contrib-llvm inference executables build with the qualified source
LLVM provider. Each mode runs nine selected host CTests: eight pass and the
existing AVX-512-only expert_multi_test skips on this Ryzen 5600X. IQ3/other native
AVX2 parity, Q8_K quantization, Q2 bit-plane and pool partition tests pass.
The retargeted diagnostic controller passes malformed protocol, owner/QUIT,
pending-admission and pending-process CPU fixtures. No GPU/model operation has
run for this new revision at this preflight boundary.

R306 identifies non-atomic mapped-word polling in unrestricted one-shot session
execution. Main and R308 follow the actual ordinary serve route: forced segmented
verification waits for compute completion before CPU reads and for copies before
the next launch. The one-shot loop is below the serve return. Resident graphs,
the new interop copy engine, prompt issuer and CPU prompt sharing are disabled;
MTP, batches and pipelines are excluded. This is a bounded source admission,
not a whole-engine portable safety or runtime performance claim.

The next controlled diagnostic keeps the prior 32768-input/16-output recipe,
8192 prompt chunks, cache128, workers5 and PCIe fraction0. Requested settings and
actual READY/INFO/PP values will be recorded separately. Level Zero validation
and progress logging remain enabled. The setup-default recipe and a parameter
search are separate work: neither this diagnostic nor the old official serial
windows8 samples establish a tuned comparison. Full physical262144 input,
RESTORE, full-vocabulary parity and task quality remain open.

The original reports, source-only proposal, Main derivative, exact source and
ELF/compiler/library pins, commands, environment and individual CPU outcomes
are preserved here. Fresh same-provider GPU health, kernel/dump checks and normal
QUIT are required before and after any model attempt. No reset or service change
is authorized by this diagnostic.
