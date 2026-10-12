<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Pristine XeStrata update and official server integration

XeStrata advanced from da3d426 to b9f00e9 during finalization. Its engine identifies
as xe0.1.43 and integrates official v0.1.43 at `7695859`. The independent ghq
worktree is unchanged in tracked source. Necessary official server imports are
integrated at 17879c7: reasoning reserve/loop detection, held-tool SSE keepalive,
safe activity observation, opt-in replicas, resource admission and child-only
Linux CPU affinity. The selected engine inference implementation is preserved.
Server regression: 611 pass and 11 existing skips; setup: 47 pass.

Both new Xe free and contrib-llvm executables build with source LLVM7.1.1.
Each runs nine selected host CTests: eight pass and the AVX-512-only test skips
on Ryzen5600X. The new diagnostic controller's parser, pending rejection,
owned-process and normalQUIT CPU fixtures pass. The first artifact lints failed
on Markdown/hash formatting and duplicate escaped patch licensing; their failed
statuses are preserved. Typed hashes and compact duplicate-patch metadata fix
those renderings; no lint rule or runtime assertion was suppressed.

The new free executable passes same-provider integer GPU health and a 44.38s
startup. One source-controlled32768-input/16-output diagnostic completes with
prefill97544.6ms (335.928t/s), decode2214.8ms (7.224t/s), normalQUIT, empty owned
sessions, unchanged boot, no new driver fault/dump and no runtime error. Actual
settings: batched8192 prompt,144 expert slots/281MiB, pool5, PCIe fraction0,
int8 KV,32768 resident cells, physical262144 context, noMTP or batch.
Level Zero validation and tracing are enabled; this is not an optimal-speed run.
See exact individual DONE/PP/LP5, device/provider/library/source/fixture pins,
actualINFO, commands/environment, normal closure and fault evidence in receipts.

The same token input is SHA-identical to the previous da3d426 diagnostic. The
first eight generated token IDs match, then the ninth differs; reported LP5
also differs. Completion/finite-output/fault checks pass, but numerical equivalence
and model-quality preservation are not established. The historical da sample
(217.612prefill,10.411decode t/s) is retained as a single diagnostic reference,
not a ranking against this run or the curated engine. No new Xe inference port
is adopted from these numbers. Matched settings, repeated decode samples and
numerical/quality checks are needed before choosing either stage's implementation.

R306's non-atomic control polling applies to unrestricted one-shot session
execution. Main and R308 trace the selected ordinary serve return and forced
segmented verifier: compute completion precedes CPU reads, and copy completion
precedes next-segment launch. Resident graphs and the new interop copy stream
are explicitly disabled; noMTP, batches or pipelines enter the route. This
bounded source/runtime result does not qualify every engine path. R309's new
fusion requires multi-token windows. R310's mean-row MMQ selector is a future
conditional candidate, requiring route/histogram/parity evidence. Every model
comparison remains at least32768 input;8K refers only to kernel fixtures/chunks.

The first new health wrapper failed on the stats JSON key before any integer
probe/model launch: stats reports device_index and pci_bdf, not device_id. The
new wrapper checks both actual index and BDF; the original failure remains a
failure. This was not a GPU hang. No reset, unbind or service mutation occurred.

Original unpatched official v0.1.43 32K results and curated inference timings
remain historical. The official setup-default/parameter-optimized TODO, full
physical262144-input verification, RESTORE and whole-model quality remain open.
Only free whole-model GPU execution is qualified here; contrib-llvm was built
and host-tested. Compact evidence is committed; redundant closed logs and the
successful full license inventory are retired with a path/hash/reason manifest.
