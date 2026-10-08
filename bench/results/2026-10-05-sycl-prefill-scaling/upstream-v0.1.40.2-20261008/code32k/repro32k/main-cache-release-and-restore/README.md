# Main-cache release and restoration controls

Arc B570 10 GiB / Ryzen 5 5600X / 128 GiB RAM, kernel 7.0.0-38,
NEO 26.31.39395.14 and oneAPI 2026.1.1. This result retains twelve complete
correctness controls followed by eight clean32K timing jobs. Diagnostic and
state-capture durations are excluded; the candidate remains unadopted.

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

Completed three-run arms: main-vmm-kept-ram, main-vmm-half-ram, main-vmm-full-ram, main-vmm-full-snapshot.
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
check the entire occupied payload. The full-snapshot arm also passes all
three state/head/output captures and complete payload verification. No captured
duration enters a speed comparison. The graph callback is included in clean prefill time. These fresh serve
processes do not warm verifier graphs at startup: release arms capture all
legal window sizes at restoration, while kept backing captures on first use.
Initial capture/kernel loading is part of this comparison, so it does not
isolate repeated recapture. See the [later source/API audit](../larger-compact-chunk-12k/README.md).

Eight fresh clean jobs run kept/half-RAM/full-RAM/full-snapshot and reverse.
Payload checks, validation, debug/API logs, state/head dumps, profiler,
transfer profiling and extra waits are absent. The owned GDB/PTY observer
is common. All20 jobs match output and MTP counts, exit normally and record
no new xe fault.

| Setting | Run | Prefill token/s | Decode token/s |
| --- | --- | ---: | ---: |
| Kept backing | 1 | 443.277 | 16.708 |
| Kept backing | 2 | 443.282 | 17.064 |
| Kept backing | mean | 443.280 | 16.886 |
| Half / RAM | 1 | 443.918 | 17.914 |
| Half / RAM | 2 | 443.595 | 17.490 |
| Half / RAM | mean | 443.757 | 17.702 |
| All / RAM | 1 | 433.249 | 16.802 |
| All / RAM | 2 | 433.361 | 17.916 |
| All / RAM | mean | 433.305 | 17.359 |
| All / snapshot | 1 | 442.411 | 17.776 |
| All / snapshot | 2 | 442.078 | 17.884 |
| All / snapshot | mean | 442.245 | 17.830 |

The [clean sequence](analysis/clean-sequence.json) preserves durations, means
and relative changes. The [release messages](clean-release-messages.json)
separate physical backing, logical payload, suspend/restore and graph time.
Two repetitions per setting do not establish small differences beyond
observed variation or attribute decode variation to a prefill-only setting.
These measurements hold fixed the 8192-token chunk and residual geometry;
release alone does not test the benefit of enlarging chunks with freed VRAM.
A later logged 32K/12288-chunk attempt has enough measured capacity but fails
state/head/output equality and is rejected before clean timing.

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
