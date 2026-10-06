# Post-reboot model comparisons, 2026-10-07

The current graph-retirement/host-upload candidate executes on the GPU after
an external reboot. All four model processes complete all four requests and
exit 0. No new xe fault appears in any captured kernel window. **The release
experiment still fails mathematical validation**; a failed checker here does
not mean the GPU hung. See [summary](summary.json) and the unchanged per-run
receipts, raw stdout, GDB and engine logs.

The machine is Arc B570 10 GiB, Ryzen 5 5600X, 128 GiB RAM, kernel
7.0.0-38-generic. New boot ID is `0a908c16-e292-4299-80ff-082da39f4bb9`.
Stock NEO remains 26.31.39395.14 and the loader 1.32.0; installed IGC is now
2.41.5. The agent did not reboot, change these packages, reset/rebind the GPU,
or stop a service. The earlier failed health receipt belongs to another boot.

Candidate commit is `9fbf844fdb971d82f00cffbe1b9a6ce9077c6333`; the frozen
executable SHA-256 is
`79a4b363d33f66f41d910be6274e609b4eb73f62afb0dc49bca72bf2a538d223`.
All tests use UR V2, persistent SYCL cache off and **direct submission off**.
The small health prerequisite also disables copy offload. The model comparisons
use default copy offload and GDB with the retained strict first-fault script.
They are diagnostic runs, not performance benchmarks.

## Actual model behavior

Each process uses context 128, normal MTP (`--spec 4`), forced prefill, prefill
chunk 32 and four requests: the original 37-token prompt, its repeat, a prompt
with one changed token, and the original again. Each request emits four IDs
and four finite printed logprob rows. All 64 emitted IDs agree across the
four processes. The full first-window target head contains 248,320 finite
floats per request; the comparison checks all bytes, not just the chosen ID.

| Run | Engine exit | New xe faults | Weight release/restore pairs | Validation |
| --- | --- | --- | --- | --- |
| Retained, checkpoint reuse | 0 | 0 | 0 | Pass |
| Released, checkpoint reuse | 0 | 0 | 6 | Fail: repeat and changed prompt differ |
| Released, reuse disabled | 0 | 0 | 4 | Fail: repeat differs |
| Retained, reuse disabled | 0 | 0 | 0 | Pass |

The release experiment discards all six MTP graph arrays before physical
destruction and recreates graphs after restoration. Each pair releases
939,524,096 physical bytes and restores 931,016,700 payload bytes from its
immutable RAM source. All enabled before/after weight-byte checks pass.

The independent [head comparison](head-comparison.json) finds all 248,320
floats different on the cached repeat (maximum absolute difference
1.9000349044799805) and changed prompt (1.6389579772949219). In the cold
comparison, only the repeat differs, again across all floats, with maximum
absolute difference 0.2670884132385254. The first and final original requests
match their corresponding retained controls bit for bit. Both retained
controls also reproduce the original head exactly on their own repeats.

Disabling prompt/conversation reuse does not eliminate the defect. This
narrows the next tuning check to the release/recapture path; it does not yet
identify whether allocation replacement, graph recreation or their state
interaction causes the wrong result. Weight-byte equality is insufficient
to accept this optimization. It remains off by default.

## Checker coverage and provenance

The checker now has optional full-head dumps and a cold-state comparison.
When dumping, it rejects missing/nonfinite heads and compares repeated heads
byte for byte in addition to IDs and printed logprobs. Its
[eleven CPU protocol cases](checker-cpu-full-head/record.json) pass. In
particular, changing only the last head value while preserving all printed
IDs/logprobs is rejected, as are a truncated head and a NaN. Failure cleanup
still records the ordinary engine exit separately from checker failure.

Historical checker snapshots are retained: `serve-check-head-only.py` was
used by the first cached pair; `serve-check-head-uncached.py` by the cold pair.
The current checker/test snapshots include the later full-head repeat check.
Per-run receipts preserve original absolute paths and controller hashes.
Full binary head dumps and the executable remain in private persistent state
at `/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007`;
the manifest and comparison retain their digests.

These short tests do not prove prevention of every GPU hang, recovery without
reboot, default direct-submission behavior, cancellation/error restoration,
or performance. Neither CLI nor normal-MTP serve has consumed all 262,144
context cells on this candidate. Those capacity checks remain required before
accepting the optimized configuration.
