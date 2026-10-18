# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC with one graphics card and system RAM, on Windows or Linux. Its C++ GPU engine (`src/`, `include/`)
has CUDA/HIP backends and this fork's SYCL backend. The project also has a Python server with an OpenAI- and
Anthropic-compatible API and a web app (`serve/`), and a one-click installer (`setup.py`, started by
`START-HERE.bat` / `setup.sh`).

## This fork's scope

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
Upstream contributions are not currently planned. CUDA/HIP implementation and
validation are outside this work's scope.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

### Session and action checks

These are durable operating rules for Main and delegated work. Use the
`git-workflow` skill for Git operations and the project-only upstream sync skill
for upstream evaluation; the user's project-specific rules here take precedence
over generic workflow examples.

At session start, resume and after context compaction, perform these checks in
order before launching work:

1. Reread this file and [the artifact retention policy](docs/ARTIFACT_RETENTION.md).
   Apply the latest user instructions; a conversation summary is not a substitute
   for the maintained rules.
2. Recover the validated build, clean speed reference, selected experiment,
   pending validation and provenance holds from
   [the maintained prefill decisions](docs/SYCL_NATIVE16_PREFILL.md) and their
   linked compact committed results. Use Git for older evidence; do not load the
   full session log or research library unless a concrete question requires it.
3. Verify that state against the actual branch, worktree, pending changes and
   owned in-flight jobs. Include retained builds and artifacts: confirm their
   named consumers still exist and carry unresolved cleanup/provenance work
   forward. Inspect an existing job's receipt and process handle
   before starting or resuming a measurement. Resume a live job through its
   existing handle; conversational memory and a stale receipt are insufficient.
4. Choose the next action from the table below. Check the last actual upstream
   check time and eligible research scopes as part of the recurring work loop,
   while keeping Main's implementation and serialized measurement moving.

**Repeat the applicable check immediately before each action.** These are
prerequisites, not items to explain away in a later report. A startup read does
not replace them. Use this table to find the detailed rule without rereading
historical logs or the research library wholesale.

| Action | Required check | Detailed rules |
| --- | --- | --- |
| Choose an optimization | Identify the measured limiting resource, proposed mechanism and comparison that can accept or reject it. Distinguish an attribution improvement from a throughput gain. | [Tuning](#git-integration-and-tuning), [measurement](#measurement-and-gpu-execution) |
| Build | Commit the exact engine, benchmark, controller, generated inputs and recipe first; pin compiler, dependencies and environment. Identify retained generations across worktrees and preserve the validated fallback before overwriting it. | [Measurement](#measurement-and-gpu-execution), [artifacts](#measurement-and-diagnostic-artifacts) |
| Run or compare | Verify committed inputs against the build record and actual executable hash. Check job ownership and the frozen comparison plan; keep first-use, subsequent-request and independent-process samples distinct. | [Measurement](#measurement-and-gpu-execution), [artifacts](#measurement-and-diagnostic-artifacts) |
| Integrate | Check provenance and decide inference parts individually, including prefill and decode. Preserve required non-inference functions. History preservation and component success alone do not qualify a model route. | [Scope](#this-forks-scope), [integration](#git-integration-and-tuning) |
| Receive research | Check its classification and delegate review to native Codex gpt-6-luna with max reasoning. Main reads the short review digest; retrieve collection-report sections only for a concrete question. | [Parallel research](#parallel-research) |
| Retain a file or build | Name the realistic next consumer and review point; check compact results and reachable Git history first. Preserve individual comparison samples without automatically retaining raw logs, reports or another build generation. | [Artifacts](#measurement-and-diagnostic-artifacts), [retention policy](docs/ARTIFACT_RETENTION.md) |
| Delete artifacts | Confirm the consumer ended and useful evidence is preserved. For a build, retrieve and compare its original source/recipe from Git against the exact final executable's inputs. Check active ownership and the last required fixture. | [Artifacts](#measurement-and-diagnostic-artifacts), [deletion records](docs/ARTIFACT_RETENTION.md#cleanup-and-deletion-records) |
| Close a work boundary | Commit and push useful changes; integrate completed work into feature/sycl and the fork default through a PR. Verify remote refs and pull the clean ghq default mirror. Check actual retained paths/generations, retire finished branches, comparison builds and raw output, and record concrete remaining holds. | [Git workflow](#git-integration-and-tuning), [artifacts](#measurement-and-diagnostic-artifacts) |

An uncommitted measurement snapshot is not qualified retroactively by a later
commit. A nearby revision, similarly named build root or matching binary hash
alone does not prove recoverable source. Keep an unresolved exception pending
with its concrete remaining action and next review point in the existing compact
result; a new experiment or context change does not clear it. Do not report a
boundary complete from a local commit or history-only merge alone.

Measurement controllers must enforce the source/build/executable checks before
launch, and stop on missing evidence or a mismatch. Check all measurement inputs,
including helpers and generated workloads; unrelated pending edits outside those
inputs do not belong in the measurement commit. Never bypass a failed check by
refreshing recorded hashes. Close the affected run and commit the changed inputs
and plan before building or running again.

When the user changes a durable operating rule, update its affected section and
relevant maintained policy document before the next action it governs. If the
file is frozen by an active run, apply the instruction immediately and update
the file at the first safe boundary. Later user instructions supersede older
ones; replace contradictory rules instead of appending a second version.
Keep durable rules here and per-run state in compact committed results, without
a chronological policy log, separate resume checkpoint or duplicate policy copy.
Update the existing rule and its canonical detail document together; do not
append another checklist or report for each reminder. Apply retention rules to
existing artifacts as well as new ones.

At each work boundary, check the latest instructions against these rules and
the actual completed work before reporting completion. Carry pending validation,
unfinished upstream ports and provenance holds forward in their existing
maintained results with a concrete next action. A status request, policy edit or
context compaction does not complete or abandon the ongoing tuning experiment.

Preserve the distinction between a proposal, a tested component and a qualified
model route when resuming or reporting. An unfinished experiment or research
proposal does not become adopted during a context change. Read historical detail
only for a concrete gap. Pass applicable rules and the report-index location to
every delegated task; Main remains responsible for integration and testing.

### Git, integration and tuning

- The primary branch is **`feature/sycl`**. Use its existing worktree. Create a
  child from its current commit only when isolation is needed; read-only research
  needs no branch.
- Concentrate current tuning on the dedicated long-input prefill path toward
  **1000 tok/s**, with repeated long-input results above **600 tok/s** as the
  first milestone. Record milestone evidence in results, not as changing state
  in this policy. The longer-term decode target is **70 tok/s**;
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
- Plan long-prefill VRAM separately from decode placement. Account for the
  largest resident layer and kernel workspace before budgeting KV and hidden
  states. Compare releasing decode-only weights/cache, partial retention and
  rebuilding from RAM by their total transfer, synchronization and reconstruction
  cost; do not assume keeping decode allocations resident is always cheaper.
- Implement native 16-bit hidden-state
  producer/consumer kernels using checked SGLang/vLLM/Intel design references.
  Converting saved residuals to 16-bit and restoring FP32 before each operator
  alone does not satisfy this work. Keep necessary local FP32 accumulation,
  reductions, router decisions and recurrent state distinct from tensor storage;
  record remaining FP32 boundaries, provenance and numerical/performance gates.
- Prioritize the actual SGLang/vLLM
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
  default branch before retiring their branches. Do not squash, rebase away or
  force-push measured source revisions that retained results reference; preserve
  their original commit identities and recoverable inputs.
- Pin delegated source proposals to an immutable commit or a fully hashed source
  snapshot. Generate the patch against that same base, even if the main worktree
  changes during implementation. Review the complete patch and preservation of
  existing features before applying it; do not replace live files from stale mirrors.
- Retire finished remote/local child branches and worktrees after integration.
  Preserve branch-only reports and needed source as reachable original commits
  before deletion. Keep unfinished children only with a concrete task and review point.
- History preservation is not code integration or inference adoption. Record
  candidate origin, actual implementation, measurements and per-component decisions.
- Import all Niko non-inference functions, including server, API, CLI, web UI,
  settings and setup. Track any unfinished port explicitly. Choose inference changes
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
  Distinguish requested transfer bytes from measured bus traffic, job counts
  from arithmetic volume, and overlapping queue intervals from additive wall
  time. Do not add a printed aggregate to its constituent samples. Dense-model
  throughput and active-parameter counts provide context, but a ceiling needs
  the actual layer operations, tensor shapes, precision and data movement.
  Device-specific compiler flags and binaries are allowed for this PC; record
  them with the measured source and build recipe.
- For host-side attribution, use disjoint accounting regions and report CPU
  time inside GPU waits, transfer preparation, submission and unclassified
  time separately. Check nesting, completeness and exception/error counts;
  an invalid or incomplete profile does not qualify an attribution. Report
  observer settings and overhead limits, and collect clean speed separately.
  If a profiler cannot collect under the current permissions, preserve that
  failed capability result and use bounded source instrumentation where useful.
  Keep unavailable hardware metrics unknown; a profiler refusal does not resolve
  the bottleneck or justify stopping independent tuning work.
  Adding counters or moving time from unclassified into a named category is
  attribution progress, not a speedup. Trace the complete called wrapper,
  including preflight checks, queue lookup, observers, internal waits and
  post-checks, before assigning its time to the operation named in a log.
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
- Freeze the source, executable, helpers, fixtures, generated workloads and
  configuration pinned by an active measurement or A/B plan. Review planned
  edits against those inputs first; documentation outside the pinned inputs may
  proceed. If an input must change, close the existing run honestly and commit
  a new plan before using it. Never update hashes to disguise a changed arm.
- Before rerunning code implicated in a GPU hang, audit its bounds, USM/context
  lifetime, queue/event dependencies, collective participation and graph replay
  assumptions against the actual SYCL/Level Zero contract. Correct or isolate
  the suspected trigger first, then use a bounded small traced/validated check
  before a long model run, as specified in the retention document. Distinguish
  a host-process stall from a device fault; absence from `xpu-smi ps` alone is
  insufficient. Record fresh driver faults and process closure. Keep recovery
  actions separate from inference tests and consider other GPU clients before
  any reset/unbind; a headless process is not proof that it is independent.
  Keep the human-facing recovery entrypoint a shell script with a short, direct
  invocation; normal recovery must not require a long list of options.
  Once the trigger is sufficiently understood and a verified avoidance path
  allows safe execution, return to tuning rather than extending the incident
  investigation without a concrete unresolved question. Keep the separate Orca
  hang investigation outside the inference investigation.
- The user intentionally stopped `llama-server-qwen3embed.service`; leave it
  stopped. Do not restore it as measurement cleanup.

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

### Project references

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
  At each work boundary, check the actual retained paths against these rules;
  writing a policy or deletion plan alone does not complete cleanup. Record
  remaining consumers, missing recovery evidence and the next bounded action
  in the existing compact result. Remove stale paths/index entries after
  verified deletion, without creating another progress or inventory file.
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
  one build for each closed experiment. Count generations across this project's
  worktrees, output directories and renamed copies, rather than allowing the
  same quota separately for each location.
  Keep project-generated executables, libraries, objects and JIT/native dumps
  outside Git, including the retained latest generation. Commit their necessary
  source, recipe and compact measurement evidence instead.
  A newer build becomes the retained known-good build only after its applicable
  validation passes; a failed or unfinished build does not replace it.
  Before reusing its build directory or output filename, preserve and hash-check
  the last validated executable and any required same-build fixture at a named
  external fallback path. Count this retained generation in the same quota;
  record its source/recipe and qualification consumer in the existing result.
  Retire the fallback only after replacement qualification or comparison closure
  explicitly selects the validated generation to keep.
  Retire the extra generation when its comparison ends; creating a new build
  must not silently extend the lifetime of every previous experiment's binary.
- Before building a measurement executable or running a fixture, commit the
  exact source, local changes, generated inputs and build recipe/options; pin
  compiler/dependency versions and environment.
  Verify those committed inputs match the ones used for the measured executable.
  Record the source revision and binary hash with the result. A nearby commit,
  source path or binary hash alone does not establish recoverability.
  Include benchmark/controller changes and generated workloads in this check;
  committing them later does not retroactively qualify a different measured
  snapshot. Preserve and reconcile the original inputs when a mismatch is found.
- Before retiring an old executable, library, object/native dump or build tree,
  retrieve and verify its required source/recipe from reachable Git history.
  Read the recorded revisions and compare the recovered inputs with the build's
  source identities; the existence of a commit or report alone is insufficient.
  A receipt hash matching a binary proves artifact continuity, not recoverable
  source. Review eligibility labels and cross-root links against that exact
  final build; do not transfer proof from a seed, similarly named or later build.
  Reconcile any uncommitted measured inputs first; do not delete their last copy
  while claiming Git can restore them. Record a temporary provenance hold and
  review point if recovery is unresolved; do not present it as a reusable build.
  Name the exact missing inputs, current consumer and next bounded recovery
  action. A hold is not a reason to keep unrelated logs, caches or duplicate
  outputs, and retaining an orphan binary does not resolve its missing source.
  Remove comparison generations when the comparison closes.
  Installed toolchains/runtime dependencies and model weights
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
  Closing a comparison includes deciding which validated build remains and
  retiring the extra generation after provenance verification. Use the existing
  compact result for this decision; do not create a separate progress backup,
  build archive or permanent cleanup inventory. If source recovery is blocked,
  keep an explicit provenance hold rather than deleting the last original copy.
  Distinguish such unresolved legacy holds from qualified comparison builds.
  Record review findings and missing recovery steps in the current result;
  retaining the latest engine does not establish that older roots were retired.
