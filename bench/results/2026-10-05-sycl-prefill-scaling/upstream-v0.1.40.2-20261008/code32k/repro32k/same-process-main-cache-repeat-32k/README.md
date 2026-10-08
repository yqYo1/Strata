# Same-process main-cache release: nine full 32K captures

Arc B570 10 GiB, Ryzen 5600X, 128 GiB RAM, unchanged private e82fc5 executable.
Each of three fresh processes reads the same 32,768-token code-review input
three times and generates 64 tokens with normal MTP4. Context is 33,024;
chunk 8,192; int8 KV; all 32,767 residual rows stay on GPU. Complete main
cache and MTP decode-only payloads are released/restored from immutable RAM.
Main expert slots use the matched 64 MiB segmented allocation. Prompt and
conversation reuse are disabled: every request reports resume0 and rereads
the entire input. The fixed dump paths must be absent before each request;
each new head/state is renamed and preserved after the reply.

The [sequence](sequence/record.json) passes all nine requests. Every raw
66-part main state and all 248,320 head floats match the hard control, as do
all 64 IDs/logprobs and MTP acceptance/offered counts43/66. All three engine
processes exit0 normally without forced cleanup, survivors or new xe faults.
One process has flushed UR/LevelZero validation/API logs; two have those
logs off but still have state/head capture, complete payload checks and phase
tracing. None of these durations is clean performance evidence. Every
performance comparison must use at least32,768 input tokens, and must keep
first-use loading/capture separate from later full-input rereads.

The [34 CPU comparator checks](host-check/v2/record.json) cover actual four-cell
snapshot pages, both heads' live/tail cells, all pooled rows including the
moving spare, wrong metadata and full262,144 occupancy. V1's page-offset unit
error is caught before GPU execution and preserved as a CPU negative. V2 may
exclude only rounded future KV cells, retaining raw hashes and exact ranges;
no exclusion is needed in these nine GPU captures. At full262,144 occupancy
there are no future cells to exclude. These host checks are not a proof of
SYCL mapping or absence of runtime UB.

The [UR reference analysis](analysis/graphs/record.json) verifies that the
second and third main releases each release680 old command buffers before
unmapping the main cache. MTP suspension releases6 more. Main restoration
creates/finalizes680 buffers each time. One non-cache commit buffer remains
through prefill. All UR buffer references reach zero at normal process exit.
The [native handle analysis](analysis/native/record.json) separately tracks
successful LevelZero create/destroy calls, handle reuse, device-USM free and
freeExt, physical mapping/unmapping and requested sizes. All tracked native
objects and mappings reach zero at exit. Native command-list counts remain
704 across later cache releases even while UR buffers are released, so
counting LevelZero objects alone does not demonstrate retained live graphs.
The source review also checks queue drain before unmap and recapture after
restoring stable addresses and complete payloads.

API logging is not required for the reported-free reduction: the two
non-API capture processes have identical markers, within64 KiB of the logged
process at matching later phases. Their MiB values are:

| Phase | First read | Second read | Third read |
| --- | ---: | ---: | ---: |
| Before main release | 2153.496 | 1131.254 | 1120.297 |
| After main/MTP release | 3433.500 | 2411.258 | 2400.301 |
| Prefill complete, temporary cache retained | 1089.609 | 151.059 | 140.289 |
| Temporary buffers released | 3349.613 | 2411.070 | 2400.301 |
| Main restore and verifier warm | 2128.289 | 2016.363 | 2010.852 |
| MTP restored | 1232.281 | 1120.359 | 1114.848 |

At matching prefill entry, read2 has1,022.242 MiB less free than read1;
read3 has10.957 MiB less than read2. Initial main warm creates61 modules,
61 kernels,680 regular command lists and requests1,114,112 B of device USM.
Later main warm creates no new native module/kernel/list/device-USM objects;
its384 MiB remapping accompanies reported-free drops of394.707 and389.449
MiB. Requested sizes and native counts are not resident heap sizes: these
observations do not isolate pooling, runtime-private allocations, residency
or granules, prove a leak, or establish steady state after more requests.

This closes only the captured full-RAM32K already-used-graph repeat gate.
Matched quiet kept-versus-released repeats and full262,144-cell occupancy,
repeat, restore, clipped tail, refusal and later-valid gates remain open.
The fixed32K GPU-row allocation is not a full-context default. Production
binaries/defaults remain unchanged; no reset, rebind, reboot, service, package
or global change occurs. Large API/state/head files remain private with exact
hashes. This candidate remains unadopted and the PP1000/TG70 goal stays active.
