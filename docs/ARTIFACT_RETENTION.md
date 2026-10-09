# Measurement and diagnostic artifact retention

These rules apply to Strata development measurements, correctness checks and
failure investigations, including files outside the checkout. Review retention
when a run closes, when a failure is resolved, before changing the reference
version, and at each experiment boundary. A file needs a current purpose;
neither a successful run nor a failed run justifies keeping every log.

## Evidence to keep

Commit a compact report and its structured results in `bench/results/`. Preserve
the original results for each independent measurement, including rejected and
failed samples. Averages alone cannot support a variance comparison. Record:

- Source revision/diff, binary hash, compiler/runtime/driver versions, hardware,
  command, effective environment, input identity/hash and actual context/chunk sizes.
- Individual prefill, cold decode and restored decode samples, measurement order,
  elapsed time and the settings that distinguish each arm. Keep phases separate.
- Correctness checks and their scope, output/token/logprob or tensor fingerprints,
  observed differences, fault checks and the final decision with its limitations.
- For a supervised process: boot ID, owned PID/start time, deadlines, normal or
  abnormal exit, cleanup outcome, and kernel-journal boundaries/fault findings.
- Reproduction commands, research references, open questions and the next step.

Do not rewrite a failed result into a pass during cleanup. A lost or retired
capture is not a new correctness result. Keep the original structured receipt
byte-identical when other records pin its hash; a separate manifest records which
historical paths have been retired. Do not weaken a test to accommodate deletion.

| File type | Keep | Retire after review |
| --- | --- | --- |
| Measurement results | All individual samples and correctness/fault outcomes used by the comparison | Duplicate copies of the same committed report; terminal console output already represented exactly in the result |
| Normal model stderr/protocol | Required progress/completion, validation and exact comparison evidence, until extracted and verified | Full successful API calls, every polling/sync iteration, duplicated output/token lists |
| Failure logs | The error and relevant progress/API window for one representative of each distinct unresolved mechanism | Repeated instances with the same cause and no new evidence; unrelated head/tail output |
| Debugger/driver captures | Relevant stacks, registers, instructions, driver dump and kernel fault entries with timestamps/boot/cursors | Duplicate captures that add no changed stop location, source/configuration or causal evidence |
| GPU health probe | Exact PASS/failure, binary/device/runtime/environment, exit/cleanup and fresh fault-check receipt | Every successful API call from every repeated healthy probe |
| Kernel journal | Per-run boundaries, command status, complete fault-relevant entries and filter definition/counts; a representative full timeline where needed | Repeated cumulative windows containing the same events or unrelated firewall/service messages |
| Profiler | Useful kernel/phase/transfer summaries, collection settings and interpretation limits | Raw event timelines after the summary is verified, unless a named open question still consumes them |
| Build output | Source/binary/toolchain/recipe identity, actual compile/flag proof and relevant warnings/errors | Repeated successful command listings once the recipe and proof are preserved |
| Tensor/session files | Current comparison fixtures and actual RESTORE inputs; the first useful divergent capture for an unresolved numerical failure | Superseded versions, redundant successful candidate captures, repeated known failure tensors |
| Resume notes | The current state and next actions in a committed report | External progress checkpoints and copies of already committed reports |

Same signal names, such as SIGABRT, do not prove the same cause. Compare the
stack, driver finding, source/configuration and reproduction before grouping
failures. Preserve counts and changed conditions in structured results even when
the corresponding verbose logs are discarded. A new stop location, driver
signature, rare numerical difference or changed lifecycle may warrant another
representative. Do not keep every repeat by default.

## Bound collection before launch

Use a private output directory, finite process deadline and a supervised log
budget. The default total raw text-log budget is **64 MiB per diagnostic job**.
If a job needs more, record the reason and finite replacement budget before
launch. Record a budget-triggered stop as an incomplete diagnostic, not a pass;
do not silently discard required history while continuing to claim a complete
trace. Do not run cleanup, compression, builds or other tests concurrently with
a GPU measurement.

For small first checks, capture flushed UR/Level Zero API entries/results. For a long model check, prefer parameter validation,
warning/error logging and `STRATA_TRACE=1` without the additional UR tracing
layer. Save the exact effective environment. Detailed logging and validation
affect timing; obtain speed samples from separate clean processes.

Retained failure excerpts normally fit within **256 KiB per distinct case**.
Select the relevant error/progress window; do not preserve a large arbitrary
tail of repeated `NOT_READY` polling. If the complete allocation/command history
is necessary, keep one complete trace as an explicit exception. Record in the
experiment's retention manifest:

- The concrete unresolved question and why a smaller excerpt is insufficient.
- The responsible experiment, exact consumer, path, size/hash and byte budget.
- The next review point, such as completion of allocation-lifetime analysis or
  replacement by a minimal reproducer. Remove the trace once that purpose ends.

Compression alone is not a reason to retain an unnecessary log. For a needed
large trace, prefer one lossless compressed copy. Verify the decoded byte count
and SHA256 against the original before removing the raw file. Keep raw and
archive hashes in the manifest. Do not put giant traces, tensor/session captures
or driver binaries into Git.

## Current fixtures and cleanup

Each retained large capture must name the controller/comparator or unresolved
failure that consumes it. Keep one canonical reference set for the currently
tested version, including the full physical context boundary when required.
An inference session file is a test input, not a progress checkpoint. After
updating consumers and verifying replacement fixtures, retire the superseded
set. A filename containing `baseline`, `failure` or `full256k` is insufficient
evidence of a current purpose.

Before deleting a file, confirm that its run is closed and its owned processes
are absent, that useful evidence is safely recorded, and that no current check
requires it. Use the experiment's exclusive measurement lock. Verify file
identity/hash and retained evidence before unlinking. Record the path, original
size/hash, retained replacement and reason. Never alter source, models, service
state or GPU state as part of log cleanup. Do not destroy home-wide snapshots
or rewrite shared Git history merely to reclaim project-file space.

Commit and push the policy, reports and deletion manifest at natural work
boundaries. Git is the durable resume record; avoid a second tree of progress
backups. A current working copy needed by a running controller may stay, but it
is not an independent archival copy. Report removed logical/file allocation
separately from actual available space, since snapshots can retain old extents.
