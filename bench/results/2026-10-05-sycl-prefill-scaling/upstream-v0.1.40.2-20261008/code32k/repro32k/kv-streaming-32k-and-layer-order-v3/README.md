# RAM-authoritative KV and processing order: repeated32K checks

Measured on Arc B57010GiB, Ryzen5600X and128GiB RAM with the pinned
oneAPI2026.1.1/UR0.12/NEO26.31.39395.14 stack. Production c88f94d and
accepted e82fc5 binaries remain unchanged; both candidates are private.

The v3 KV source adds an atomic page-table read paired with the existing
atomic claim, and inherits atomic shared-hit metadata. Host/device counter
representations and16-byte copies use memcpy; resolve requires host-USM,
WG1024 and SG32 support before submission. Four unused root properties stay
removed. Earlier accepted mode0 runs did not activate this route, so these
possible conflicting accesses are not attributed as their hang cause.

The layer-major candidate replaces its mode0-only rejection with mode0/1
support requiring owned identity staging and disabled KV prefetch. Existing
in-order prefix uploads, appends, attention and host mirrors remain unchanged.
It runs one layer over all chunks, using the already allocated full-context
stage. It keeps RAM residuals (R_GPU0), full main-cache RAM restoration and
the previously validated MTP decode lease. No staging-prefix reuse optimization
is introduced here.

Each configuration passes three fresh32768-token/64-output requests in one
logged/validated process: all66 used main-state parts, all248320 first-head
floats,64 IDs/logprobs and MTP43/66 match the accepted control. First raw state
matches too; later differences are confined to unwritten rounded-page future
cells. The pooled spare row is compared. Both exit0 without forced cleanup,
survivors or new xe faults. This does not prove full262144-cell occupancy.

The first source-authority guard rejects comparison of pinned compiled
prefill with production source before GPU execution. Its preserved negative
and corrected review explicitly use the accepted actual-event-completion
overlay. The private receipts record the compile/link inputs and object
replacement rather than inferring an executable from current worktree code.

Quiet comparison uses ABBA fresh processes and three complete32768-token
reads per process. Context262144, resident32768,8192 chunks, int8 KV,
cache128 requested, workers5, pcie0, normalMTP4, greedyGEN64, FIRST0/RING8,
own stage and prefetch0 are common. Layer-major includes its RAM-residual
and main/MTP lease choices; this is a configuration comparison, not an
isolated transfer or graph metric. Every reply still matches64IDs/logprobs,
MTP43/66 and resume0. State/head/payload verification, traces, profilers,
API logs and extra phase waits are absent. Health probes run before requests.

First-request numbers average two reads after READY; startup model loading
is not part of the engine's prompt timer. Repeated numbers use four full
rereads, converting their mean prompt/decode durations to rates.

| Configuration | First request PP/TG (tok/s) | Repeated full reads PP/TG (tok/s) |
| --- | ---: | ---: |
| Chunk-major, main/MTP kept | 393.43 / 16.54 | 427.49 / 18.48 |
| Layer-major, RAM residuals, main/MTP RAM leases | 324.63 / 17.94 | 429.45 / 17.92 |

Diagnostics and captures are excluded from performance. Individual readings,
source/build/owned-process receipts and protocol logs are retained; huge API
logs, state images and binaries remain private with hashes.

Full262144 occupancy through cell262143, repeated full prefill, restored
state after another session, clipped speculative tails, capacity refusals
and a later valid>=32768 request remain required. No production adoption,
reset/rebind/reboot, service, package, runtime or global change is made.
