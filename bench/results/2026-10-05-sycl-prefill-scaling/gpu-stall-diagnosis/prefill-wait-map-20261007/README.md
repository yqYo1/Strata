# Source location of the observed prefill completion wait

The actual [legacy full-MTP failure](../legacy-l0-capacity-20261007/README.md)
stopped with a prefill return PC of `0x52ef0b` in lambda 15. On 2026-10-07,
an offline compilation added line tables to the same prefill translation
unit. It did not execute GPU work or replace the production object or engine.

All 146 executable CPU sections and their canonical relocation targets matched
the production object. The production object digest also matched the frozen
engine's build receipt. The lambda's symbol matched in both objects; its
201-byte size matched the frozen executable. The observed PC is 75 bytes
from the executable's function base, `0x52eec0`.

Mapping that offset and its preceding byte identifies `prefill.cpp:4298`:
the compute queue wait before `on_stage_chunk` and `on_chunk`. Thus this
failure was waiting for queued work before the residual callback ran. The
mapping does not identify which preceding operation failed to complete.
The complete main/watchdog snapshot is in `observed-failure.mi.txt`.

The source also limits the next investigation:

- This was layer-major processing with a 1,024-token chunk. It loads all
  512 experts of the current layer and waits for the copy queue before
  processing that layer's chunks. The routed expert-copy ring is inactive
  for these resident experts.
- The intermediate rows are in a host buffer. The same compute queue
  uploads a chunk before calculation; the download callback waits before
  the host range is reused. The observed stop precedes that download.
- `STRATA_PF_STEP_SYNC` only runs between layers within one `run_impl` or
  in streamed-K/V branches. This one-layer, resident-K/V case reaches
  neither boundary, so that setting would not locate this stopped work.
- The existing `STRATA_PREFILL_SYNC` waits at every phase mark instead.
  Its log is emitted after waiting for earlier work and names the next
  phase; a printed mark is not proof that the named phase's later kernels
  completed. A separate logged short check must establish output parity
  before using it on the long request.

`record.json` preserves compile arguments, source/object/engine digests,
CPU section and canonical relocation digests, and the line-table result.
The complete relocation arrays remain in the digested private receipt.
`sources/` preserves the mapper and the exact source. Binary objects and
executables remain private. No prevention or throughput result is claimed.
