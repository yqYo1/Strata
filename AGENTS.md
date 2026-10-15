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

This is a personal fork based on **Niko1221/Strata**, optimized for this PC's
Intel GPU, CPU and RAM. Niko1221/Strata is the only upstream. XeStrata and other
engines are optional benchmark and implementation references; importing their
whole engines, feature sets or release workflows is not required. Upstream
contributions are not currently planned. CUDA/HIP implementation and validation
are outside this work's scope.

- The primary branch is **`feature/sycl`**. Use its existing worktree. Create a
  child from its current commit only when isolation is needed; read-only research
  needs no branch.
- Current tuning priority (2026-10-15): concentrate on the dedicated long-input
  prefill path until repeated long-input measurements exceed **600 tok/s**;
  continue toward **1000 tok/s**. Keep short-input prefill through the decode
  weight/cache layout available. Defer its dispatch threshold and new decode
  optimization until the prefill milestone; this does not defer long-prefill
  chunk-size, placement or scheduling tuning. Preserve shared state and the
  physical 262144-context correctness requirements.
- Check Niko upstream approximately every **48 hours** during ongoing work,
  using the project-only upstream sync skill. Record the actual check time,
  exact revision and integration status separately. Continue required functional
  imports and component-level inference decisions; do not claim a new check
  merely by rereading a previous observation. No background scheduler is implied.
- Integrate completed child source, tests, skills and reports into the actual
  `feature/sycl` tree, then push. Update the fork's remote default through a PR
  at each completed boundary; the ghq root remains the default mirror and only pulls.
- Retire finished remote/local child branches and worktrees after integration.
  Preserve branch-only reports and needed source as reachable original commits
  before deletion. Keep unfinished children only with a concrete task and review point.
- History preservation is not code integration or inference adoption. Record
  candidate origin, actual implementation, measurements and per-component decisions.
- Import Niko's non-inference functions as requested. Choose inference changes
  per component, including prefill and decode separately; retain the faster
  existing fork implementation. Evaluate reference-fork ideas under the same gates.
- Trace the configured SYCL build's actual source resolver and entrypoint.
  A feature in the upstream root CUDA generator does not establish that its
  Arc engine implements it. Record existing Arc capabilities and the remaining
  shared CLI/API port obligations separately; neither requires a CUDA/HIP engine import.

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

- Set SYCL, Level Zero, driver, profiler and experiment environment variables in
  the native child's environment before `exec`. Keep the process environment
  immutable after runtime/provider initialization. For comparisons, use separate
  owned processes or an explicit test parameter; do not switch arms with
  `setenv`/`unsetenv` while runtime or collector threads may read the environment.
  Review legacy fixtures that switch variables before reusing them. Floating-point
  state operations such as `std::fesetenv` are separate from process environment variables.

- Commit source, commands, environment, individual measurement samples, correctness/fault results and
  decisions at each work boundary. Resume from committed reports, without separate progress backups.
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
  last required fixture. Keep a path/hash/reason deletion manifest; serialize cleanup with tests/builds/GPU
  work. Store compact evidence in Git, and large necessary captures outside Git.
