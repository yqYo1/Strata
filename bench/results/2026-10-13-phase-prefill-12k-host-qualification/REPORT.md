<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Phase prefill 12K candidate builds and CPU checks

Built free and contrib-llvm with source-built DPC++ 7.1.1. All eleven changed
files passed the no-new-findings comparison against the actual qualified parent:
clang-tidy, cppcheck, document, secret and license checks. Both owner CTests,
separate Release and ASan / UBSan checks, and server phase lifetime tests passed.
The empty-catch finding was corrected and the fixture rechecked; unchanged files inherit
the full preceding checks. Original failures remain retained, including a controller
preflight that wrongly refused an expected exit2 before any build/test began. No GPU or
model execution occurred.

Head lease, FP32 scratch aliases and explicit capacity 12288 remain default-OFF
experiments. Original GGUF payload bytes are snapshotted in CPU RAM before the
first H2D and reused on reload. Unknown copy, conversion or release state retains
owners and source and refuses retry/free. Each confirmed released pointer is
cleared separately. Lease conversion uses Runtime checked drain, stored async
error checking and the watchdog. Canonical --native raw admission was fixed.

CPU fixtures cover allocation 1 / 2 false/throw, unknown copy/conversion, same-size
source mutation, destructor fail-stop and failures before/after free. Each mode
also passed twelve actual binary startup cases,24 total. Correct native/explicit
head reaches an intentionally invalid ordinary option; pipeline, auto, mixed
capacity, missing opt-in,01 and overflow stop at their expected guards. These
fixtures stop before Runtime/model construction and prove no physical fit or DMA.

Registry v191 adds R386,R387 and the implementation report. Gated overlay remains
unimplemented. Including the current Alloc padding, removing the separate gated
allocation at stock12K would reduce requested storage by503316736 bytes,480 MiB
plus256. This is source accounting, not physical VRAM or speed. Main additionally
fixed uncertain free handling and the checked Runtime drain. Original reports and
specific findings remain retained.

Speed comparisons still use unchanged upstream bench.sh --long --runs 3. This
work adds no speed sample. Keep default OFF pending actual head/alias/12K admission,
fixed-history logits/state, full physical262144 and restore qualification. Exact
generated prose identity is not a universal quality gate.
