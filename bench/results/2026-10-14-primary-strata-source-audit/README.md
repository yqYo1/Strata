# Primary branch and implementation lineage audit

Niko1221/Strata is the base and sole upstream of this personal fork. The target is this PC's Intel Arc B570,
Ryzen 5 5600X and RAM. Upstream contributions are not planned; CUDA/HIP inference work is outside scope.
XeStrata is a benchmark/design/component reference, with no mandatory whole-engine or feature parity.

The audited primary `feature/sycl` is `4fa478f50f8da396e651e3ae862f8b1849d49b54` (version marker 0.1.38).
Fork `main` is `beddab2fd6b61631d73b39b7484bc9a585ebe1de` (0.1.39).
Recent reference experiments at `28a7b9ae7ca5eba28eecd65f6453a7479c04fce8` use a Xe-derived 0.1.40.2 family
with selected Niko 0.1.43 non-inference changes. They are a different source family.
See [branches.json](branches.json) for exact refs, worktrees, dirty files and merge preview.
Version markers and ancestry do not prove functional or inference adoption.

Recent 667–694 token/s results describe that reference family, not current primary performance.
No fresh matched primary benchmark was performed. [reference-measurements.json](reference-measurements.json)
retains all closed official-client samples and their limits: actual input 32049, output 24, no reuse.
Bulk adoption remains held because the control's decode also fell. A prefill benefit does not qualify decode.
Physical 256K endpoints, RESTORE and whole-model quality remain open.

[primary-component-source-pins.json](primary-component-source-pins.json) pins actual runtime, prompt/decode,
CPU/SYCL, cache/source and CLI/server/setup files. This audit changes no inference source.
[decisions.json](decisions.json) separates named optional candidates from their missing qualification.
A whole Xe kernel-family replacement is not component adoption.

Primary and main have 143/285 unique commits from their common ancestor. The ordinary merge preview has seven
conflicting files. Responses API, runtime configuration, parallel requests, VRAM controls, UI, setup and
C++ CLI/protocol behaviors require explicit adaptation and tests while preserving primary inference.
[cli-option-inventory.json](cli-option-inventory.json) records quoted option literals, not semantic parity.
The R447 functional/dependency/test map and R448 component audit are in [research-index.json](research-index.json).

The project skill now lives in actual `feature/sycl` at
[SKILL.md](../../../.claude/skills/strata-upstream-sync/SKILL.md), exposed through relative `.agents/skills`.
There is no user-wide install or dependency on an experimental worktree's absolute path.
The skill requires all Niko non-inference functions and inference choices per component and prefill/decode role;
Xe is an optional reference. Work proceeds on `feature/sycl`, children start from its current commit, actual
completed results land there, own-fork main updates use PRs, and finished branches retire after preservation.
A history-preservation merge is archival, not product-code integration or inference adoption.

No GPU run or fresh upstream remote check occurred in this audit. Preservation/cleanup and deliberate main
functional reconciliation precede default promotion and the next Niko update evaluation.
Do not execute the known GPU-failure primary revision before correction and bounded checks.
The [runtime contract notes](runtime-contract-notes.md) distinguish existing checked waits and normal
stream drains from concrete failure/exit ownership questions; they do not claim a newly proven hang cause.

R441–R448 are retained with exact text and hashes, distinguishing proposals from Main decisions.
R448 read an older README; later completed quiet reference measurements are included here. Its prefill path
spelling is corrected to actual `src/prefill/sycl/` files. R446's read-only planning questions do not revoke
existing user authorization. Compact [receipts](receipts) preserve host/ASan/UBSan and reference build proof,
commands, environment, source and termination; none qualifies primary/model/default-OFF/full-context behavior.
The failed older metadata publication remains failed. See [controller.json](controller.json) for generation.
