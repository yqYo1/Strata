# Personal fork integration status

Niko1221/Strata is this project's base and sole upstream. This fork targets
the owner's Intel Arc B570, Ryzen 5 5600X and 128 GB RAM. Upstream contribution
is not planned. CUDA/HIP inference implementation and validation are outside
the current work. XeStrata is an independent benchmark and component reference;
its complete engine, features and release workflow need not be imported.

The project-only [upstream sync skill](../.agents/skills/strata-upstream-sync/SKILL.md)
defines the recurring workflow. Import all applicable Niko non-inference
functions, including engine-side CLI and protocol behavior. Select inference
changes separately by component, format, shape and prefill/decode role, retaining
the faster validated implementation. A history-preserving merge is not source
integration or an inference adoption decision.

The primary branch is `feature/sycl`. At the 2026-10-14 audit, local and remote
branches are only `feature/sycl` and `main`, with two corresponding worktrees.
The ghq root is the clean default-branch mirror. Completed work belongs in
`feature/sycl`; update the owner's fork default through its own PR at work
boundaries, then pull the root mirror. No PR to Niko or XeStrata is intended.

## Current implementation and outstanding work

The [exact-session regression audit](../bench/results/2026-10-14-performance-regression-audit/README.ja.md)
corrects the earlier inference-preservation claim. Before Xe evaluation, this
fork's Niko-derived SYCL overlay delivered 449.7 t/s on fresh actual32K inputs
with context262144/MTP4, and an earlier layer-major CLI experiment delivered
645.8 t/s with a different65538 context and no MTP. These are not Xe results.
The selected primary implementation does not contain the later measured
overlay's compact workspace or attention-layout selector. Old code/report
commits remain reachable, but preservation merges did not integrate them.
Current118.5 t/s is a conservative stock baseline, not preservation of the
previous best configuration. Restore and compare the prior source components
and effective settings before treating this older source family as the tuning
reference. Method/configuration differences leave causal attribution open.

The audited primary source is `5bf0742913584a343975ea7f792c97cefac488fe`.
Its engine lineage is this fork's Niko 0.1.38-based SYCL/CPU implementation,
with selected fixes and shared service changes. The configured root CMake
selects `cmake/sycl_backend.cmake`, root `src/program/generate.cpp`, root core,
SYCL prefill/kernels and CPU kernels. The separate SYCL overlay in fork `main`
and XeStrata's engine are different source paths.

Fork default `main` remains `beddab2fd6b61631d73b39b7484bc9a585ebe1de`
(0.1.39 lineage). Default promotion is outstanding. The observed Niko revision
`296b59d2f73c8f04a805df29c2fa97655adace78` / 0.1.44.1 is an update observation,
not a completed evaluation or integration.

| Area | Current disposition and next requirement |
| --- | --- |
| Serial chat/completions, Anthropic, Responses, streaming, setup/UI | Existing shared service functions and host tests are present. A complete feature-by-feature comparison through the observed Niko revision remains unfinished. |
| Single-GPU parallel requests | Server/setup can emit `--batch`, but the active primary engine lacks the corresponding independent request slots and batch protocol. Port the contract to this engine; speculative rows are not request slots. |
| Elastic VRAM | Server/API configuration exists, but the active primary engine lacks elastic CLI/cache protocol and ownership adaptation. Static VRAM reserve does not provide elasticity. |
| Responses effort and setup reruns | Configured effort propagation and preservation of an installed model/configuration require focused source adaptations and tests. |
| Latest images, frontend and recovery changes | Image generation differs from existing vision input. Applicable service/config/runner behavior and subsequent frontend/recovery changes still require an exact delta ledger and hardware/backend qualification. |
| Primary prefill and decode | The Oct4 root-native components are present, but the later pre-Xe measured overlay is not selected. Stock actual32049-input median PP118.5/24-output decode3.4 is slower than historical configurations. Reconcile and restore qualified components/settings, then compare matched long-input runs; do not count ancestry as adoption. |
| XeStrata layer-major/bulk and paired IQ3_S GU | Optional, unadopted inference candidates. Reference-fork measurements do not establish a benefit in the primary implementation. |
| Maximum context | Configured 262144 capacity and shorter successful runs do not prove full physical-input consumption, near-boundary generation or RESTORE. Relevant state/scheduling changes need those checks. |

This table is a source disposition summary, not certification that every Niko
feature has been enumerated or integrated. The known CLI/protocol gaps need
resolution before promoting this implementation as the completed integration.
Promotion does not require achieving the performance targets first.

The [primary audit](../bench/results/2026-10-14-primary-model-baseline/README.ja.md)
and its [research registry](../bench/results/2026-10-14-primary-model-baseline/research-index.json)
retain source provenance, individual measurements and corrections. Read them
when continuing work; do not infer performance or adoption from the displayed
version or ancestry alone.
