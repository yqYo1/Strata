# Processing order and GPU residuals on 32K input

Arc B570 10 GiB, Ryzen 5 5600X and 128 GiB RAM; kernel 7.0.0-38,
NEO 26.31.39395.14, oneAPI 2026.1.1, default implicit counter conversion and
no MKL CNR. All conditions use the unchanged private scheduling executable
e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323,
the same 32,768-token code-review fixture, context 33024, 8192-token chunks,
int8 KV, normal MTP4, 64 greedy output tokens, expert cache 128, five CPU
workers, pcie 0, FIRST 0/RING 8 and no prompt/conversation caching.

COMPACT2 reuses phase storage after its last consumer on the in-order queue.
QSA layout 1 retains subgroup 32, workgroup 256 and ordered arithmetic, while
reconstructing query operands through scalar loads from transposed local
storage. Batch remains 32. Three fresh 32K controls first check that combination.
Startup INFO freeVRAM increases from 830 MiB to 2220 MiB; this is a startup snapshot,
not a peak-usage measurement. The
[source review](analysis/compact-layer-v01402-source-review.json) records
allocation/carving agreement, scratch bounds, uniform barriers and the
inactive key-head alternative. These reviews do not prove all runtime UB absent.

Layer-major 2 then retains a full layer's 512 experts in a separate temporary
cache while processing all four chunks. Three more exact 32K controls check
this order with residual rows in RAM. Its logged row image is 1,342,136,320 B,
temporary layer cache 1,363,148,800 B. Main decode cache and MTP weights remain
backed for this arm. Expert copies repeat once per layer instead of once per
layer/chunk, but RAM residual upload/download adds transfers. Source-derived
copy avoidance is not a measured PCIe bandwidth or DMA duration.

A third configuration keeps all 32,767 batched residual rows on GPU, reuses
the original 8192-row scratch prefix in place, and allocates the remaining
24,575 rows separately. MTP decode-only expert/head weights are temporarily
unmapped after queues drain and graphs are retired; persistent dense weights,
state and KV remain backed. Restoration remaps the same virtual addresses,
copies the immutable RAM payload, waits before readiness and recaptures graphs
when needed. The main cache remains backed. The
[additional review](analysis/layer-gpu-release-v01402-source-review.json)
checks row/tail partitions, stable MTP graph inputs and reclamation ordering.
Three fresh 32K controls enable full byte verification before/after each lease.

All nine captured controls match all 66 main-prefill state parts, all 248,320
first-head floats (993,280 bytes), 64 IDs and all logprobs with the completed default
control. Every first use has flushed UR/Level Zero logs and parameter validation.
These timings are excluded. The [MTP protocol counts](mtp-count-parity.json)
also match 43 accepted/66 offered in every captured/clean request; this compares
counts, not every internal MTP tensor.

Only after all gates, six fresh clean jobs run default/RAM/GPU/GPU/RAM/default.
No diagnostics, validation, byte verification, state/head dumps, profiler,
transfer timing or extra waits are enabled. The uninterrupted owned GDB/PTY
observer is common. Every clean job matches all 64 IDs/logprobs. All 15 jobs
exit normally, complete owned cleanup and record no new xe fault.

| Setting | Run | Prefill token/s | Decode token/s |
| --- | --- | ---: | ---: |
| processing-default | 1 | 408.208 | 17.156 |
| processing-default | 2 | 408.360 | 17.053 |
| processing-default | mean | 408.284 | 17.104 |
| layer2-ram-layout1 | 1 | 333.979 | 16.853 |
| layer2-ram-layout1 | 2 | 342.246 | 17.058 |
| layer2-ram-layout1 | mean | 338.113 | 16.955 |
| layer2-gpu-release-layout1 | 1 | 446.315 | 17.389 |
| layer2-gpu-release-layout1 | 2 | 398.872 | 17.253 |
| layer2-gpu-release-layout1 | mean | 422.594 | 17.321 |

The [clean sequence](analysis/layer-processing-v01402-clean-sequence.json)
contains each duration, ordering and relative change. The
[release/restore messages](clean-release-messages.json) preserve measured
mapping/copy/graph-drop timings in both clean GPU-row repetitions. These
measurements apply to this model, hardware and configuration, not an
unmodified-upstream equivalence claim. Two repetitions per condition do not
establish small changes beyond observed variation or attribute decode variation
to a prefill-only flag.
The GPU-row pair differs by about 11% in prompt throughput. Its two-run mean
therefore does not establish a reproducible improvement over the default.

The candidate is private and unadopted. Full 262,144-cell normal-MTP
occupancy/repeat/restore/clipped-tail/refusal/later-valid gates remain open;
the older second-prefill memory failure is not solved by this 32K check.
Snapshot-versus-RAM and partial main-cache release are separate pending
comparisons. PP 1000/TG 70 is not declared achieved. No production executable,
reset/rebind/reboot, service, package or global setting changes.

Public files retain exact argv/environment, fixture, protocol, source hashes,
executed controllers, ownership, journal and state/head hashes. Large API,
state and head payloads remain private with byte counts and SHA256 hashes.
