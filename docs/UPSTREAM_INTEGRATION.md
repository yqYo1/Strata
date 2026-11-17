# Personal fork: upstream integration

This personal fork uses **Niko1221/Strata** as its only upstream and targets this
PC's Intel Arc B570, Ryzen 5 5600X and 128 GB RAM. CUDA/HIP inference work and
upstream contribution are outside this task. XeStrata is a performance reference
only; its code, helpers, tests and design must not be copied or adapted.

Use the project-only [upstream sync skill](../.agents/skills/strata-upstream-sync/SKILL.md)
and [AGENTS.md](../AGENTS.md). Work remains on `feature/sycl`; promote completed
work through the fork's PR while preserving measured source commit identities.

## Current checked state

The [upstream status](../bench/results/2026-10-15-pr6-recovery-audit/upstream-status.json)
records actual fetch and remote-ref observations, separately from integration.

| Item | Recorded state |
| --- | --- |
| Last actual check and fetch | 2026-11-17 20:52:28 UTC / November 18 05:52:28 JST |
| Observed upstream | `c77572806996349b49995e4f98d41b5294d36a21` (latest fetched main;no engine import) |
| Next check at the approximate 48-hour cadence | 2026-11-19 20:52:28 UTC / November 20 05:52:28 JST |
| Current upstream Python/setup/control/server/API/CLI/Web import | Partial import from `146df3c4`; full validation remains open. Later changes through `c7757280` are fetched, with the plain GPU-list ordering fix imported and tested |
| Complete native function ledger and required-function port | Incomplete |
| Unmodified, hardware-tuned upstream performance evaluation | Incomplete |
| New inference adoption from this check | None |

The import record covers 630 changed paths in its named scope, including the
current setup/model catalog, Python control and server/API/Web code, CLI
integrations, packing/tokenizer helpers and upstream host tests. It preserves
the local owned-file console shutdown repair and existing setup/benchmark
adaptations. Focused configuration, streaming, API and CLI changes have their
own host evidence; that does not complete the native function ledger or full
host validation. The interval's changed source and partial status are recorded
under `current_sync.functional_import` in the status file.

The November18JST check fetches45additional commits/121changed paths from
`baf16363` to`c7757280`;the cumulative pending interval from`146df3c4` now
contains1166commits/2849changed paths. The new interval changes serving,App,
Control,CLI and packaging functions;its native C++changes are tests only.
Boolean tool coercion,stop-drain admission,MTP/About display and persisted
drafting settings need functional ports after the frozen bank comparison.
No production inference change or performance adoption follows from this check.
The legacy ghq checkout remains divergent and untouched;fetch/ref observation
is complete,its default-mirror reconciliation remains a separate hold.

The November16JST check fetches742additional commits/2086changed paths from
`13eedf55` to`baf16363`. The cumulative unintegrated interval beyond the partial
`146df3c4` import is1121commits/2816changed paths. This is an actual ghq-clone
fetch and matching remote-main observation;the legacy checkout and native
engines are unchanged. Root LICENSE bytes match the previous observation;
individual component provenance still needs review. Two bounded source scopes
are reviewed and their useful findings are below;they do not complete the
742commit ledger or establish imports,tests or performance. Both original
research samplers refused because Main put source views inside capture trees,
exceeding256files;one also exceeds8MiB. Preserve those failures. The corrected
future layout keeps immutable inputs outside report/control captures.
DeepSeek/35B and CUDA/HIP execution remain excluded.

| Reviewed source unit | Required action and qualification |
| --- | --- |
| One Off/MTP/DFlash2 decision through plan/fetch,setup,config,launch and visible status | Required functional port. Preserve existing MTP/legacy configurations and DFlash2 opt-in;do not silently replace the local default with upstream Auto. Start from [launch plan/argv](https://github.com/Niko1221/Strata/blob/baf16363b4403ef0ed4f538f5d5ed11c43fb9535/strata_cli/launch/runner.py#L1506),[CLI drafting](https://github.com/Niko1221/Strata/blob/baf16363b4403ef0ed4f538f5d5ed11c43fb9535/strata_cli/drafting_cmd.py#L117) and [draft selection](https://github.com/Niko1221/Strata/blob/baf16363b4403ef0ed4f538f5d5ed11c43fb9535/strata_control/drafting.py#L467). The12GBIQ3_S example is at runner336-339/modelstate86-95;the old report's unrelated lines are not evidence. |
| Arc prepared-file/pack/config adaptation | Required and unqualified. Verify the native setup entrypoint,accepted engine arguments and direct config writers before porting [the launch adaptation](https://github.com/Niko1221/Strata/blob/baf16363b4403ef0ed4f538f5d5ed11c43fb9535/strata_cli/launch/runner.py#L1364). Preserve existing config and avoid unsupported layer/vision/draft downloads. The existing Arc source map below remains the adapter consumer. |
| CPU IQ3_S weights with Q8_K activations | Conditional decode candidate,not long GPU prefill. [The mapping](https://github.com/Niko1221/Strata/blob/baf16363b4403ef0ed4f538f5d5ed11c43fb9535/src/quant/types.cpp#L32) uses the AVX2 quant path. [SYCL CMake](https://github.com/Niko1221/Strata/blob/baf16363b4403ef0ed4f538f5d5ed11c43fb9535/sycl/CMakeLists.txt#L130) includes root pool/native-expert sources but does not wire that quant layer. Native-expert checks use ggml-cpu in the SYCL branch and Strata quant with STRATA_OWN_CPU_QUANT;quant parity is a separate gate. Prove local route,compiler/ISA/activation packing,parity and decode-sized performance before adoption. Reviewed quant provenance is MIT ggml/llama.cpp;retain per-component provenance checks before import. |
| Nested cgroup quota discovery | Conditional pool-capacity candidate. [The helper](https://github.com/Niko1221/Strata/blob/baf16363b4403ef0ed4f538f5d5ed11c43fb9535/include/strata/kernels/cpu/cgroup_cpu.hpp) changes quota discovery,not row math. A real local quota and performance evidence are prerequisites. |

These four pointers are the retained result of the scopes,not coverage of all
245/732available source-view files or all2086changed paths. The original
report/capture identities and refusals remain in the research index;Main read
only the short review and the corrections needed for this map. Reopen a source
scope for a concrete port/component evidence gap or a new upstream interval.

Earlier router/control/server source reviews remain partial. Required candidates
include vision context-retry and portable serve-stop bounds;dense MMQ row-id
padding still needs an actual SYCL-path safety review. The plain `--gpu`fix from
Niko`2b9df3fe71cb24cef3052c332e032ea681729abe`clears stale ordering when replacing
a saved selection,preserving explicit roles and unchanged selections. Its four
regressions and14existing GPU-role tests pass without skips. The existing
`current_sync.functional_import.incremental_function_ports.plain_gpu_order`
record remains the evidence;the new fetch does not complete its broader ports.

The Control/App Intel Arc setup fix and expert-arena shortfall/warning changes
remain required-function candidates. Fixed-order skinny BF16 GEMMs and compiled
model geometry need component-level decisions; upstream A750 timings and
default-build checksum claims are not measurements or qualification on this
B570. No new inference component from the observed interval is adopted. The
remaining source and pending review are recorded under `new_interval`; older
unfinished scopes remain open.

The [Arc setup source map](research/upstream-arc-control-setup.md) identified
that the broad import lost this fork's native setup routing. Source `b8f0a5bbf`
restores the Linux checkout entry to `tools/setup_intel.py` and sends installed/
previous native configurations through it before upstream update/migration.
All10focused CPU/mock entry/config-preservation tests pass with zero skips,
normal0and closed ownership;source/interpreter checks pass before and after.
No actual installation is qualified. Control/App/MCP setup and the CLI launcher's direct `adopt.plan` /
`adopt.write` config replacement still need their native adapters. Control cannot
invoke the source builder while its component route is pending. The
[unmodified Arc benchmark source map](research/upstream-arc-benchmark-readiness.md)
records B570 build/workload prerequisites; its helper dependency closure,
target configuration, runtime qualification and actual measurement remain open.
These source reviews did not perform another upstream fetch or import.

The current Unsloth v3.1 catalog/pinned metadata and stock DFlash2 pack are
present. The focused v3.1 setup/pins/shard-discovery and DFlash2 packing tests
passed77host cases with1skip; this does not qualify native v3.1 model/type
execution. The native DFlash2 candidate is opt-in. Its corrected32,769-token
route admission and32K-plus sustained thinking generation have been exercised;
the cache-loan comparison also closed in six fresh processes. Those measurements
charge validation/observation and do not qualify a clean, repeated DFlash2-versus-
MTP speed comparison. Task quality, physical262144/RESTORE and default adoption
remain open. Existing configurations retain their MTP selection, and the
experimental cache loan remains off by default. See `current_sync` in the status
file for the original checks and limits; component success is not model adoption.

## Remaining integration work

The status file's `function_review`, `latest_interval_review`,
`new_interval_function_slices`, `pending_interval` and `new_interval` retain the reviewed scopes,
source identities, feature decisions and unfinished deltas. They are partial
reviews, not a certified complete feature ledger. Preserve the original ported
and tested records without converting pending functions into completed imports.

Outstanding work includes the new Control/App Arc setup fix, the complete native
function ledger and actual Arc entrypoint review; disk-session, BGEN/capability, tracking/recovery/alias and
remaining API/CLI behaviors; complete host validation; and the unmodified
upstream's hardware-tuned comparison. Reconcile each applicable function
through the observed target. Required server/API/CLI/UI/setup functions must be
preserved or adapted to this SYCL engine even when inference components are
retained locally. Keep DeepSeek and 35B execution paths excluded unless the user
explicitly requests them.

Evaluate unmodified upstream with hardware-tuned settings and the maintained
`bench.sh` workloads: long input, agent session, and thinking-enabled long
generation with observed strong and weak MTP acceptance. Choose inference
changes per component and separately for prefill/decode; an overall gain does
not justify replacing a faster local part. Do not substitute ancestry or a
history-only merge for implementation and validation.

Current engine selection, performance and open correctness gates are maintained
in [SYCL_NATIVE16_PREFILL.md](SYCL_NATIVE16_PREFILL.md) and the single
[performance CSV](../bench/performance-records.csv). The October 14 source audit
and its old speed figures remain in [their original document revision](https://github.com/yqYo1/Strata/blob/0820688a4ff14947b659d96bff112ed293559ca8/docs/UPSTREAM_INTEGRATION.md);
they do not describe the current selected engines.

Default-branch promotion remains pending through PR 180. Main ruleset `24638744`
permits only squash PR merges. The normal squash attempt for `fbe45f1b1` on
November 14 UTC is refused by the active update restriction
(`update_allows_fetch_and_merge=false`). PR 180 remains open; no administrative
bypass is authorized. All 172 CSV source/evidence references are verified on the
pushed permanent primary. Before merging, verify that every original
source/evidence commit referenced by the performance CSV remains an ancestor
of the pushed, permanent `feature/sycl`, and keep automatic head deletion
disabled. Under the project workflow, the default may receive the PR content
snapshot while the primary retains those original identities; merge the updated
default back into the primary normally and verify reachability again. This changes no protection
or measured source history. The last verified default before that operation is
`05069aeedea43f778b97dce30c729739cb3bce5c`; do not claim integration until its
remote ref and clean ghq mirror are verified.
