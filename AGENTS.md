# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

At session start, resume and after context compaction, reread this file and the
current committed decisions relevant to the task before launching work. Check
the actual branch, worktree, pending changes and owned in-flight jobs; do not
rely on conversational memory or create a separate resume checkpoint. Check the
existing job's receipt before resuming a measurement or starting another one.
Apply the retention rules below to existing artifacts as well as newly generated
ones. At a work boundary, reconcile new user instructions with this file and the
relevant maintained policy document. Keep durable rules here and per-run state
in compact committed results; do not turn this file into a chronological log.

This is a personal fork based on **Niko1221/Strata**, optimized for this PC's
Intel GPU, CPU and RAM. Niko1221/Strata is the only upstream. Because of the
user's license constraint, **XeStrata is a performance reference only**. Do not
copy, port, adapt, or use XeStrata code as the basis for a reimplementation,
including helpers, diagnostics, tests and archived source proposals. Audit
existing derived code and remove it from the active tree while preserving
independent safety fixes and compact provenance/results in Git. Changing names
or removing copyright notices does not remove a derivation. Other engines may
be implementation references only after their provenance and license are checked.
Record source provenance for delegated proposals before integration, including
helpers and tests. Newly authored project files follow the repository's `LICENSE`;
do not introduce another license through a copied header or template. Hold an
uncertain proposal out of the active implementation until its origin is resolved.
Upstream
contributions are not currently planned. CUDA/HIP implementation and validation
are outside this work's scope.

- The primary branch is **`feature/sycl`**. Use its existing worktree. Create a
  child from its current commit only when isolation is needed; read-only research
  needs no branch.
- Current tuning priority (2026-10-15): concentrate on the dedicated long-input
  prefill path until repeated long-input measurements exceed **600 tok/s**;
  continue toward **1000 tok/s**. The longer-term decode target is **70 tok/s**;
  neither target is achieved by a component microbenchmark. Keep short-input
  prefill through the decode weight/cache layout available. Defer its dispatch
  threshold and new decode
  optimization until the prefill milestone; this does not defer long-prefill
  chunk-size, placement or scheduling tuning. Preserve shared state and the
  physical 262144-context correctness requirements.
- Prefer GPU-local long-prefill data and control when added GPU work costs less
  than the removed CPU work, transfers and synchronization. Measure the whole
  dependency path. FP16 and BF16 storage both use two bytes; select format from
  range, numerical/task evidence and actual performance rather than assuming
  FP16 is universally better. Preserve the existing FP32 reference.
- Current precision work (2026-10-16): implement native 16-bit hidden-state
  producer/consumer kernels using checked SGLang/vLLM/Intel design references.
  Converting saved residuals to 16-bit and restoring FP32 before each operator
  alone does not satisfy this work. Keep necessary local FP32 accumulation,
  reductions, router decisions and recurrent state distinct from tensor storage;
  record remaining FP32 boundaries, provenance and numerical/performance gates.
- Numerical reference policy (2026-10-16): prioritize the actual SGLang/vLLM
  execution path for this model when selecting long-prefill algorithms and
  precision. Compare storage, accumulation, reductions, rounding locations,
  normalization constants, routing and persistent state. A matching validated
  SGLang/vLLM numerical contract is acceptable; disagreement with llama.cpp or
  a more precise FP32 oracle alone is not a reason to reject it. Keep existing
  FP32/ggml references for diagnostics, rather than imposing their extra precision
  as the adoption standard. Source flags alone do not prove the implemented
  contract. Record revised criteria prospectively and preserve original failed
  receipts; retain bounds, lifetime, fault and physical-context correctness gates.
- Keep long-prefill inference math on the GPU and minimize CPU work beyond
  transfer preparation, launch control and validation. Audit existing CPU
  grouping/index operations and library fallback or emulation paths explicitly;
  a GPU queue alone does not prove a library performs all arithmetic there.
  Attribute thread CPU time inside GPU waits separately from host computation
  and submission. One saturated CPU thread alone does not establish CPU latency
  as the bottleneck, because completion polling can also consume a full core.
- Check Niko upstream approximately every **48 hours** during ongoing work,
  using the project-only upstream sync skill. Record the actual check time,
  exact revision and integration status separately. Continue required functional
  imports and component-level inference decisions; do not claim a new check
  merely by rereading a previous observation. No background scheduler is implied.
- Integrate completed child source, tests, skills and reports into the actual
  `feature/sycl` tree, then push. Update the fork's remote default through a PR
  at each completed boundary; the ghq root remains the default mirror and only pulls.
  Use a merge commit for PRs containing original source revisions referenced by
  measurements, and verify those revisions remain reachable from the retained
  default branch before retiring their branches.
- Pin delegated source proposals to an immutable commit or a fully hashed source
  snapshot. Generate the patch against that same base, even if the main worktree
  changes during implementation. Review the complete patch and preservation of
  existing features before applying it; do not replace live files from stale mirrors.
- Retire finished remote/local child branches and worktrees after integration.
  Preserve branch-only reports and needed source as reachable original commits
  before deletion. Keep unfinished children only with a concrete task and review point.
- History preservation is not code integration or inference adoption. Record
  candidate origin, actual implementation, measurements and per-component decisions.
- Import Niko's non-inference functions as requested. Choose inference changes
  per component, including prefill and decode separately; retain the faster
  existing fork implementation. Measure unmodified upstream without local
  patches, including a configuration tuned for this hardware; default settings
  alone are insufficient. A prefill gain does not justify a decode regression.
  XeStrata remains excluded from implementation adoption even if its
  measurements are faster.
- Trace the configured SYCL build's actual source resolver and entrypoint.
  A feature in the upstream root CUDA generator does not establish that its
  Arc engine implements it. Record existing Arc capabilities and the remaining
  shared CLI/API port obligations separately; neither requires a CUDA/HIP engine import.

### Measurement and GPU execution

- Identify the limiting hardware resource before choosing an optimization.
  Relate stage timings to actual PCIe bytes/bandwidth and overlap, VRAM traffic
  and bandwidth, GPU arithmetic/occupancy, CPU work and RAM bandwidth as relevant.
  Use measured hardware ceilings and compatible shapes/precision to estimate
  the possible gain; record an unmeasured ceiling as unknown. Kernel service
  time, GPU utilization and critical-path wall time are different quantities.
  Use a profiler when attribution is unclear, then collect clean speed samples
  separately. A stage percentage alone does not establish hardware saturation.
- Follow [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Use the pinned upstream
  `bench.sh` for standard comparisons and the same maintained extension on both
  engines for added workloads. Adoption evidence includes the long-input and
  agent-session benchmarks, plus long generation with thinking enabled. Prefill
  comparisons need at least **32K actual input tokens**; **64K or longer** is
  acceptable for the 1000 tok/s target. Short inputs and component fixtures are
  useful diagnostics but do not replace long-input model measurements.
- Keep thinking enabled and provide context/output room for thought and an
  actual answer. Long-generation evidence needs at least about **1K generated
  tokens**, including thinking. Include both MTP-friendly and MTP-weak workloads;
  confirm their classification from observed draft acceptance rather than their
  names. A short prefill-only output does not qualify decode performance.
- Compare matched configurations and workloads, recording actual read/reused
  tokens, placement, chunk size, cache state and MTP behavior. Separate first-use
  cost from subsequent requests and independent process repetitions. Repeat
  comparisons when variability prevents a decision, including decode alone when
  needed. Keep individual samples and spread; do not dismiss a slower result as
  noise without repeated evidence. Check past verified best results and their
  source/configuration when investigating a regression.
- Qualify the advertised **262144-token physical context**, including filling
  it completely, tail/boundary cases, prefill-to-decode state and RESTORE paths.
  No output room at a completely full context is expected; crashes, invalid
  accesses and corrupted state are not. A smaller configured context or a
  successful component test does not establish this qualification. Generated
  text need not be identical; use the numerical/task policy above.
- Main serializes builds, tests, GPU jobs, profiling and cleanup. Launch owned
  processes with finite time/output budgets. A tool observation timeout does
  not establish that its process stopped: inspect the existing handle and
  ownership evidence before retrying. Do not launch a duplicate experiment.
- Before rerunning code implicated in a GPU hang, audit its bounds, USM/context
  lifetime, queue/event dependencies, collective participation and graph replay
  assumptions against the actual SYCL/Level Zero contract. Correct or isolate
  the suspected trigger first, then use a bounded small traced/validated check
  before a long model run, as specified in the retention document. Distinguish
  a host-process stall from a device fault; absence from `xpu-smi ps` alone is
  insufficient. Record fresh driver faults and process closure. Keep recovery
  actions separate from inference tests and consider other GPU clients before
  any reset/unbind; a headless process is not proof that it is independent.

### Parallel research

- Create research workers through **Orca**, using its bundled `orchestration`
  and `orca-cli` guides, rather than Codex's built-in subagent tools. Repeat
  bounded, independent **gpt-6-luna** tasks throughout tuning while Main
  continues implementation, analysis and serialized measurement. Respect
  actual concurrency/resource limits; aim for roughly five researchers only
  when capacity permits, otherwise fill the available slots.
- Shift scopes and share `bench/results/2026-10-15-pr6-recovery-audit/research-index.json`
  and relevant previous reports in every assignment. Prioritize this engine's
  measured bottlenecks, but also investigate general SYCL and transferable
  non-LLM GPU/ASIC optimization, including original technical blogs and papers.
- Label each research assignment **decision support** or **knowledge
  collection**. Main must not read knowledge-collection report bodies merely
  because they arrived. Keep them in the indexed file library for retrieval
  when a concrete implementation question needs them. Inspect only relevant
  sections of decision-support reports as needed.
- Specify a finite question and report budget for each fresh session. Research
  scopes must be broad enough to compare alternatives, applicability and
  counterevidence, while remaining one coherent investigation. Do not split
  into trivial fact lookups or fill reports with known context. Treat report
  length as a working budget, allowing enough room for substantive findings;
  review novelty and depth after completion before choosing the next scope.
  Apply scope corrections to new tasks, without steering running researchers.
  Research may write only its named report, with sources, applicability and evidence
  limits; it must not run builds, tests, GPU jobs, profilers, process inspection,
  Git commands or product edits. Do not interrupt or steer a running researcher.
- Delegate completed report review for scope, depth, redundancy and indexing to
  a **native Codex subagent using gpt-6-luna with max reasoning**, rather than
  Orca. This is the explicit exception to Orca research dispatch. The reviewer
  reads reports, pins their hashes, records corrections and retrieval metadata,
  and returns a short operational digest and next-scope recommendations to Main.
  Main does not load the original collection reports or a long technical digest.
  A coordinating subagent between Main and researchers is optional. Do not steer
  active researchers; apply scope changes to fresh bounded sessions.
- Before dispatching a recommended scope, check its short eligibility metadata.
  Require `schedulable_now=true` and recorded evidence for its prerequisites.
  Keep conditional scopes with missing prerequisites in the index for later;
  do not start a session that can only report that its prerequisite is absent.
- Keep findings in files for review; commit only substantive reusable findings
  or information supporting a current decision. Do not commit every completed
  report or its dispatch/spec/review paperwork automatically. Normal Git history
  is the archive for closed work; prune the active library and index rather than
  retaining stale reference chains. Research is not implementation adoption.
  Main implements and exclusively manages testing and
  GPU work. Following the user's 2026-10-16 resource-limit instruction, do not
  launch new **gpt-6.1-sol** workers through Orca or native Codex. Already running
  workers may finish without intervention. Continue the specified **gpt-6-luna**
  research and native **gpt-6-luna max** report review; do not substitute another
  expensive implementation model.

Use the project-only [strata-upstream-sync skill](.agents/skills/strata-upstream-sync/SKILL.md).
Its files live in `.claude/skills/strata-upstream-sync`, exposed by the relative
`.agents/skills` link. Do not install it in user-wide skill directories.

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Measurement and diagnostic artifacts

Follow [docs/ARTIFACT_RETENTION.md](docs/ARTIFACT_RETENTION.md) for every measurement and investigation.

- Apply retention rules to tracked files, untracked files, external state,
  build outputs and project caches. Before keeping information, name its
  realistic future consumer and review point, and check whether reachable Git
  history already supplies it. Completing a run, producing a debug output or
  reviewing a report does not by itself justify keeping another active file.
  Reduce unnecessary information rather than bundling, compressing or moving
  it elsewhere to make a directory look smaller.
- Set SYCL, Level Zero, driver, profiler and experiment environment variables in
  the native child's environment before `exec`. Keep the process environment
  immutable after runtime/provider initialization. For comparisons, use separate
  owned processes or an explicit test parameter; do not switch arms with
  `setenv`/`unsetenv` while runtime or collector threads may read the environment.
  Review legacy fixtures that switch variables before reusing them. Floating-point
  state operations such as `std::fesetenv` are separate from process environment variables.

- Commit maintained source and compact decisions with the configuration,
  individual samples and correctness/fault evidence actually needed to assess
  them. Do not commit every debug output, build receipt or process lifecycle.
  Retire superseded active files to reachable Git history. Resume from current
  documentation and committed decisions, without separate progress backups.
  Promote reusable fixtures and tooling into maintained source locations;
  do not keep duplicate source proposals and one-run controllers beside results.
- Keep the latest known-good engine build, plus only named imminent comparison
  generations (normally one build, at most two for A/B). Record the comparison
  consumer and retirement point in the current compact results. An unnamed
  possible future comparison does not justify another generation. Do not keep
  one build for each closed experiment.
  Retire the extra generation when its comparison ends; creating a new build
  must not silently extend the lifetime of every previous experiment's binary.
- Before building a measurement executable or running a fixture, commit the
  exact source, local changes, generated inputs and build recipe/options; pin
  compiler/dependency versions and environment.
  Verify those committed inputs match the ones used for the measured executable.
  Record the source revision and binary hash with the result. A nearby commit,
  source path or binary hash alone does not establish recoverability.
- Before retiring an old executable, library, object/native dump or build tree,
  retrieve and verify its required source/recipe from reachable Git history.
  Read the recorded revisions and compare the recovered inputs with the build's
  source identities; the existence of a commit or report alone is insufficient.
  Reconcile any uncommitted measured inputs first; do not delete their last copy
  while claiming Git can restore them. Remove comparison generations when the
  comparison closes. Installed toolchains/runtime dependencies and model weights
  are separate assets, outside this old-build cleanup rule.
- After a run closes, delete successful full API traces, repeated polling/synchronization logs, raw
  profiler timelines and duplicate kernel windows once their useful evidence is recorded and verified.
- Keep one representative diagnostic per distinct unresolved failure mechanism, with the relevant
  error, progress, stack/registers and driver dump. Keep a new instance only when it adds evidence;
  a failure status or the same signal name alone does not decide retention.
- Keep full traces only when a recorded question requires the complete history. Record the owner,
  byte budget, consumer and next review point. Review again when that question is resolved or superseded.
- Keep raw tensor/session files only for an explicitly named current comparison/RESTORE consumer or
  unresolved numerical failure. Retire superseded captures after recording comparisons and hashes.
- Cleanup must not change original result status, weaken validation, touch an active run or remove the
  last currently required fixture. For tracked deletions record the reachable
  source commit and selection reasons; use Git's diff for paths/blob hashes
  instead of duplicating the file list. Record explicit path/hash/reason for
  untracked/external deletions. Serialize cleanup with tests/builds/GPU work.
  Store compact useful evidence in Git, and necessary large captures outside Git.
- At each work boundary, reconcile the actual files with these rules, retire
  closed comparisons and transient output, and commit the useful results and
  deletion decisions. Record any retained exception's current consumer and next
  review point; do not report cleanup complete while silently carrying old
  comparisons forward. Check file counts and bytes as well as individual sizes;
  benchmark directories must remain browsable. A large directory is a reason
  to review information value, not to split the same unnecessary files.
