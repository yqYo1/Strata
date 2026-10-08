# Copy-engine comparisons, 2026-10-07

With copy offload disabled, retaining and releasing the immutable MTP decode
weights produce identical complete heads, output IDs and printed logprobs for
all four requests. This holds both with and without the post-prefill state
diagnostic. All six new model processes exit 0, complete 24 requests and have
no new xe fault in their captured kernel windows. A failed mathematical
checker is distinct from a GPU hang. See [receipts](summary.json) and the
[independent complete comparison](complete-comparison.json).

These measurements extend the [earlier comparisons](../post-reboot-retirement/README.md).
Their inference that the defect was confined to weight release was too narrow:
retained weights also reproduce an incorrect repeat when state-dump copies
are enabled. This does not identify a specific faulty kernel or prove the
driver alone is responsible.

## Measured conditions

The machine is Arc B570 10 GiB, Ryzen 5 5600X, 128 GiB RAM, kernel
7.0.0-38-generic, stock NEO 26.31.39395.14, IGC 2.41.5, loader 1.32.0 and
oneAPI 2026.1.1. Boot ID is `0a908c16-e292-4299-80ff-082da39f4bb9`.
No package change, reset, rebind, service interruption or reboot was performed.
Every model run disables direct submission and persistent SYCL cache, uses
UR V2, GDB with the same strict first-fault script, context 128, normal MTP
with spec 4, forced layer-major prefill in chunks of 32, and checkpoint reuse.

The unchanged production candidate is built from commit
`9fbf844fdb971d82f00cffbe1b9a6ce9077c6333`. Its frozen executable SHA-256 is
`79a4b363d33f66f41d910be6274e609b4eb73f62afb0dc49bca72bf2a538d223`.
The same four requests are original, repeat, one changed prompt token, and
original again; each emits four IDs. Each dumped head has 248,320 finite
floats. Equality checks read every head byte, not only selected tokens.

| Control | Copy offload | Post-prefill state dump | Engine exit | Repeat validation |
| --- | --- | --- | --- | --- |
| Recreate graphs but retain physical backing | Default | No | 0 | Fail |
| Retain weights and graphs | Default | Yes | 0 | Fail |
| Retain weights and graphs | Disabled | Yes | 0 | Pass |
| Release physical backing and recreate graphs | Disabled | Yes | 0 | Pass |
| Retain weights and graphs | Disabled | No | 0 | Pass |
| Release physical backing and recreate graphs | Disabled | No | 0 | Pass |

The physical-backing control is a private diagnostic build: only `mtp.cpp.o`
differs in its engine archive. It omits the two shrink calls and reports zero
released physical bytes, retaining graph retirement, RAM restoration and byte
verification. Its generated source, build commands, archive comparison and
hashes are retained. It still fails equality, so physical destruction alone
does not explain all observed errors. This variant is not a production change.

Both actual release runs with copy offload disabled perform six release/restore
pairs. Each releases 939,524,096 physical bytes (896 MiB) and restores
931,016,700 immutable RAM payload bytes at the same addresses. All enabled
before/after weight-byte checks pass. Within each retained/released pair, every
request's IDs, printed logprobs and complete head are identical. Original,
repeat and restored heads also match within each passing run.

## Persistent state precedes the wrong decode result

The private checker uses the existing `STRATA_PREFILL_DUMP_STATE` hook. It
stores 119,009,864 bytes per request and parses 66 length-prefixed payloads:
six global state parts, then five parts for each of twelve attention layers.
This adds synchronization and device-to-host copies, so it is a diagnostic
that can alter timing; it is not an unmodified performance measurement.

In the retained/default-copy repeat, recurrent rows 0 through 20 are identical,
while rows 21 through 35 differ. Saved attention KV and pooled state match
through layer 27 and differ from layer 31. PLE state, dead flags and block
positions agree. The [recurrent comparison](model-state-retained/gdn-repeat-comparison.json)
and [attention comparison](model-state-retained/qsa-repeat-comparison.json)
therefore locate visible state divergence before layer 28's recurrent update.
They do not distinguish attention, MoE or their preceding synchronization.
The state is already different before the first decode window, weakening an
explanation confined to decode graph replay.

With `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1`, all 66 payloads match on the
retained repeat. All 66 also match between retained and released executions
for each corresponding request. The independent comparison rehashes the
actual private payloads before accepting their recorded digests. Removing the
state diagnostic still produces matching complete heads for the actual
retained/released pair. The changed-prompt head differs from the earlier
default-copy retained run; this establishes consistency within the tested
configuration, not an independent CPU golden result or preservation of every
earlier, inconsistent result.

## Return to capacity validation and tuning

Copy offload disabled and direct submission disabled are the experimental
configuration for the next tuning checks. They are not evidence that every
hang is prevented or that default runtime settings are safe. Weight release
remains off by default. These GDB runs provide no performance comparison.

The capacity controller now accepts a complete environment file instead of
inheriting an obsolete runtime from historical receipts. It explicitly sets
spec confidence clipping off, bounds request writes, reply waits, total engine
lifetime and QUIT/TERM/KILL cleanup, drains stdout during shutdown, and records
the PID plus start time and actual exit. A surviving process blocks a new run;
the memory observer also has a bounded join. Its
[25 CPU stub cases](capacity-controller-cpu/summary.json) pass, including full
cardinality in CLI and serve, refused over-capacity requests, a clipped final
verify window, blocked input, missing READY, ignored QUIT, output during QUIT,
nonzero exit and complete environment replacement. These CPU cases are not
GPU capacity validation.

The actual candidate's CLI 262,144-cell run is in progress, with a 262,142-token
prompt and two output slots. Normal-MTP serve still needs both full-capacity
requests, including its clipped two-cell tail and healthy reuse after rejected
requests. Neither actual full-capacity stage is recorded as passed here.
Raw binary state/head dumps, binaries and archives remain in private persistent
state at `/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007`;
the repository retains text receipts, controllers, comparisons and digests.
