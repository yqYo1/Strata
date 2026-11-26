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
| Last actual check | 2026-11-26 01:07:19 UTC / November 26 10:07:19 JST. Fetched main and both new releases and verified the advertised refs. |
| Observed upstream | `c234774038d1c9249cb33637d27b895d47b39786`, one commit after v0.2.22 |
| Published release | v0.2.22, at `b28ca95e3e6847f2d5e317994f253b66716deeaa` |
| Next check at the approximate 48-hour cadence | 2026-11-28 01:07 UTC / November 28 10:07 JST |
| Current upstream Python/setup/control/server/API/CLI/Web import | Partial import from `146df3c4` plus individually tested ports, including GPU-list ordering, three server/Web fixes and offline DFlash2 pack readiness. Service and app changes remain open. |
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

The November 26 check adds 321 commits and 482 changed paths after
`6a6db7ff5`. Initial Qwen-facing scopes cover agent-session prefix reuse,
the custom Responses tool keep-alive, update resume, and chat working-folder
preservation. The custom Responses keep-alive from `e83de74d7` is ported in
`26b95ce99` at the assembler's output boundary. The scripted HTTP regression
at `630fd80d3` reproduces zero keep-alives during held custom input before
the fix. Afterward, the three new cases pass, including the complete custom
argument and ordinary function streaming. Responses and write-timeout
validation records 49 passes and one optional `jsonschema` skip. The
broader run that includes the stream-ownership module reaches its 130-second
deadline and remains open for separate verification.

The retained Arc verifier uses completed GPU events and its
host expert callback. The compiled SYCL path at `968e71988` sets the
all-resident device planner to false. Keep this host fallback. Revisit the
upstream `a7eb7c6c8` admission guard if an explicit no-host mode is added. The
normalization change in `ed5bff88c` is in the root GPU source. Niko Arc
selects `kernels.dp.cpp`, and this fork's compiler census contains its own
`prefill/sycl/postops.cpp` implementation. Their numerical and runtime
comparisons are separate work. The reviewed reports are indexed in the status
file.
The release publishes Qwen SYCL engine 0.1.69 and main adds libgomp packaging.
Track the selected unmodified build and measure both default and tuned
configurations on this B570. The required-function ledger and model
performance evaluation remain open. [v0.2.21](https://github.com/Niko1221/Strata/releases/tag/v0.2.21) · [v0.2.22](https://github.com/Niko1221/Strata/releases/tag/v0.2.22).

The prior 152-commit wave through `6a6db7ff5` changed 835 paths.
Its Qwen-facing feature review covered model discovery and API schema, setup and
installer changes, conversation branching, and SYCL device-loss propagation.
Evaluate the Battlemage grouped expert path, GDN pairing and tiling, Unsloth
multi-row SYCL formats, and cold-expert CPU SIMD as individual components.
The upstream benchmark uses the unmodified source under `engine/sycl/` and
the matching `tools/bench.sh`. Its execution and hardware-tuned comparison
remain pending. The exact prior decisions and model-discovery and admission
host checks are recoverable from `8db0078e3` in the status file.
[Niko1221's v0.2.20 release](https://github.com/Niko1221/Strata/releases/tag/v0.2.20).

The Arc BMG direct kernels accept the weight formats of five of this Qwen
pack's 48 layers. They also cap each grouped expert at 32 rows and apply separate
layout and launch guards. Check those actual row groups before testing the
direct kernels. Broad grouped scheduling and GDN pairing have their own
shape and correctness checks. The next host unit covers model list and
individual retrieval, stable `created` and `owned_by`, and all three
context size fields, including loaded-state and alias coverage.

The v0.2.18 decisions remain linked under `new_interval.prior_pending_component_review`
in the status record. Canonical Q6_K gathering stays selected for this fork's
head layout. The outstanding model metadata and recovery-counter ports still
need the matching API and engine log contracts.

The weighted expert-profile builder and its 45-prompt collector come from
the checked `58ee32277` tree. The collector uses this fork's routing
callback and thinking-enabled prompt by default. Its original closed-thinking
template and device-usage mode remain selectable. All ten profile builder
and collector host tests passed at `d3f092fb9`, including both capture modes
with a mock engine. Full live corpus collection is still open.
The bundled Intel profile already matches Niko's original `8cb26e609` blob.
Its inference effect on this B570 is a separate comparison.

The November 18 JST check fetched 45 additional commits affecting 121 paths
from `baf16363` to `c7757280`. At that point the pending interval from
`146df3c4` covered 1166 commits and 2849 changed paths. That interval changes serving,App,
Control,CLI and packaging functions;its native C++changes are tests only.
The boolean tool coercion,stop-drain admission and MTP/About display fixes are
now ported with their original Niko tests after the bank comparison closed.
All33CPU/mock/JavaScript checks pass without skips,including26existing frontend
cases. The stop test lives under`serve/`for the
maintained host harness,with unchanged contents. Their exact commits and
per-function states are in`current_sync.functional_import.incremental_function_ports`.
Persisted drafting settings and the other App/Control/CLI/packaging changes
remain required ports;the full function ledger is incomplete.
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
| One Off/MTP/DFlash2 decision through plan/fetch, setup, config, launch and visible status | Required functional port. Carry the fork's DFlash2-first Auto rule for supported single-card SYCL Qwen text through these entrypoints. Preserve explicit MTP and Off choices and the existing compatibility fallback. The 12 GB IQ3_S example is at runner336-339/modelstate86-95. [Launch plan and arguments](https://github.com/Niko1221/Strata/blob/baf16363b4403ef0ed4f538f5d5ed11c43fb9535/strata_cli/launch/runner.py#L1506), [CLI drafting](https://github.com/Niko1221/Strata/blob/baf16363b4403ef0ed4f538f5d5ed11c43fb9535/strata_cli/drafting_cmd.py#L117), [draft selection](https://github.com/Niko1221/Strata/blob/baf16363b4403ef0ed4f538f5d5ed11c43fb9535/strata_control/drafting.py#L467). |
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

At `c6facfbe3`, an offline installation can use the verified standalone
340 MB DFlash2 pack when the selected executable supports its selector.
Ten focused host tests passed, including explicit old/new executable selection
and CLI error text in both human and JSON output. Clean Git snapshots of
the parent and port ran 266 and 276 drafting tests with the same Python 3.12.3.
Both had the same 8 failures and 11 errors, matched by test ID, exception
class and source function. Remaining issues include absent App/Metal fixtures,
terminal dependencies and existing launch/default/notice assertions.

The current Unsloth v3.1 catalog/pinned metadata and stock DFlash2 pack are
present. The focused v3.1 setup/pins/shard-discovery and DFlash2 packing tests
passed 77 host cases with one skip. Native v3.1 model execution still needs
qualification. Server Auto prefers DFlash2 for supported single-card SYCL Qwen
text launches with a ready pack. Use DFlash2 for new prefill and decode tuning
and MTP for explicit use, compatibility and named comparisons. The 32,769-token
route check and a corrected clean `bench.sh` process with sustained thinking
generation have passed. Two more suite processes, task quality and physical
262144 fill and RESTORE are pending. The separate six-process cache-loan
comparison used observers and the loan is still opt-in. See the [maintained SYCL
results](SYCL_NATIVE16_PREFILL.md) and `current_sync` in the status file for
scope and evidence.

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

The fork's verified default is `cc464282e`, published through PR 242.
The next promotion is open in PR 243. The selected DFlash2 source and closed
performance rows retain their original revisions in the primary history,
including the closed width-four decode attribution at `e0cb8fc25`. Check
that the remaining model comparisons can resolve their pinned inputs before
retiring the verification worktree. Keep automatic head deletion disabled.
When the protected default publishes a squash snapshot, merge the updated
default back into the primary, verify reachability again and pull the clean
ghq mirror. [Fork PR 243](https://github.com/yqYo1/Strata/pull/243).
