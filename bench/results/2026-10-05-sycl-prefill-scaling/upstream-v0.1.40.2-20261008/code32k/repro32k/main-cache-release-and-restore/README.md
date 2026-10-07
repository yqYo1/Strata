# Main-cache release and restoration controls

Arc B570 10 GiB / Ryzen 5 5600X / 128 GiB RAM, kernel 7.0.0-38,
NEO 26.31.39395.14 and oneAPI 2026.1.1. This checkpoint retains completed
correctness controls; clean timing remains pending. No speed claim or
production adoption follows from diagnostic/state-capture durations.

The same private scheduling binary
e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323
reads the same 32,768-token code-review fixture, with 8192-token chunks,
int8 KV and normal MTP4 generating 64 greedy tokens. Compact storage 2,
attention layout 1 / batch 32 / subgroup 32, layer-major 2, 32,767 GPU
residual rows, in-place scratch reuse and MTP decode-only release are common.
All arms request 128 main-cache experts and use 64 MiB physical segments.
The release fractions are 0, 0.5 and 1; full release additionally compares
immutable-RAM restoration and a GPU-to-RAM snapshot. No implicit-counter
override or MKL CNR is set. Exact argv and environment remain with each run.

The [source review](analysis/source-review.json) checks ordered queue drains,
current residency, tail bounds, complete immutable-RAM sources, partial
failure rollback, stable virtual addresses and graph retirement before
unmapping. The main decoder/verifier cache pointers are refreshed only after
bytes are restored and checked. Verifier recapture records nodes rather than
executing a model window. Temporary layer-cache/residual buffers are reclaimed
before restoring decode weights. The explicit release experiment selects
segmented allocation without the prohibited interactive --vram-elastic flag.
Startup INFO's vram_elastic field reports whether the cache is segmented;
it is not evidence that the interactive resize command was enabled.

CPU tests extract the actual compiled private CacheLease struct. The initial
19 ASan/UBSan cases cover mixed profile order, a segment boundary inside an
expert, partial/full/control RAM or snapshot, adaptive residency, invalid
fractions/sources/spans, payload mismatch, partial-shrink rollback and
grow/refresh retries. A second set uses a distinct pattern at every byte;
all 19 pass. Its pinned negative control removes the intra-expert RAM offset
and is rejected by the actual pre-unmap payload check. These CPU resource
stand-ins do not prove SYCL mapping, GPU arithmetic or absence of runtime UB.

Completed three-run arms at this checkpoint: main-vmm-kept-ram, main-vmm-half-ram, main-vmm-full-ram.
Each has a first logged/validated 32K request and two fresh state/head captures.
All completed requests match every one of the 66 main-prefill state parts,
all 248,320 first-head floats (993,280 bytes), all 64 IDs and every logprob
with the logged default. MTP acceptance/offered counts match 43/66; this is
not a comparison of every internal MTP tensor. Owned processes exit normally,
complete cleanup and record no new xe fault.

The [captured messages](captured-release-messages.json) distinguish physical,
logical-tail and occupied payload bytes. Half release unmaps 201,326,592 B,
with 139,460,608 logical tail bytes and 130,731,008 occupied expert bytes;
the retained/released boundary may split a slot. Both sides of restoration
check the entire occupied payload. Full-RAM/snapshot and the clean matched
kept/half-RAM/full-RAM/full-snapshot/reverse comparison are pending unless
their completed controls appear above. No captured duration enters a speed
comparison. Graph recapture overhead will be included in clean prefill time.

Full 262,144-cell occupancy/repeat/restore/clipped-tail/refusal/later-valid
gates remain open. A full 262,143-row residual image alone would require
10,737,377,280 B at the observed 40,960 B/row, before KV/dense/cache/scratch.
The fixed 32,767-row setting is not a full-context default. Dynamic prefix
budgeting and the older second-prefill allocation/rollback failure remain
unresolved. PP 1000 / TG 70 has not been achieved.

Source hashes, executed controllers, protocol, environment, journal and
ownership receipts are public. Large API/state/head payloads remain private
with exact bytes and SHA256. Production code/executable, main branch,
services, packages and global settings are unchanged; no reset/rebind/reboot.
