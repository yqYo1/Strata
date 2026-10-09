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

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Measurement and diagnostic artifacts

Follow [docs/ARTIFACT_RETENTION.md](docs/ARTIFACT_RETENTION.md) for every measurement and investigation.

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
