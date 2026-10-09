# CPU wrapper qualification and native dispatch investigation

Root built and executed the production CPU wrapper comparison. All nine
environment profiles passed: 18 fresh processes, 48 local cases per process,
and nine complete T/H output pairs equal byte for byte. The histogram engine
also built with the same actual production compiler flags. Its four fresh 32K
requests completed with exact math, valid counters and no new GPU fault. No
speed improvement or adoption is claimed. The separate C repeated-full-context
numerical rejection remains open.

## Actual CPU evidence

| Check | Result | Scope |
| --- | --- | --- |
| Matched T/H CPU target builds | Pass, 68.988 s | Both actual SYCL CPU libraries, no GPU target execution |
| Compiled flag comparison | Pass, 34 objects | All T/H flags match; 33 library objects match old T production flags |
| Fresh-process wrapper matrix | Pass, 5.872 s | 864 local cases, nine whole-file bitwise pairs |
| H full engine build | Pass, 215.638 s | No executable/GPU/model invocation by build |
| H/T actual full target flags | Pass | 115 objects, 114 sources, no missing/different flags |
| H controller v2 CPU preflight | Pass, 2.795 s | Receipt/source/reference/ownership admission only |
| Histogram parser CPU v2 | Pass, 23 cases | Two legal examples and 21 malformed/configuration/count cases |
| H four fresh 32K requests in one process | Pass, 365.275 s | Math/counters/health, normal exit0, no forced cleanup/survivors |

T qualification commit is `61a70650a004d27ec622046b3d4d6499e7012ca0`;
H qualification is `87b50497bc4bac7b40113f181ad51b47e64ba7ae`. Both carry the
identical two-file test/CMake patch. The engine H commit remains
`96bd5bb4e054f6ddcf677fd96e161b499a013fd7`. Its actual binary SHA256 is
`d04ce73bdbc9debc2ae199940f3f26d8eac249f9f4edd2133eccb522031010a3`.

The CPU target links the actual `strata_kernels_cpu`, including the migrated
SYCL IQ entry and common native wrapper. Each profile runs in a fresh process
because dispatch controls are cached. Profiles are default, IQ MT minimum
1/2/3, disabled IQ256, disabled IQ4NL, gather 0/1, and IQ3S MT1. GU types
18/21/22/23 and Down20/42 cover NT1..8, full rows, partitioned rows, interior
ranges, a fresh empty interval, guards and inactive tokens. Independent ggml
or double-Q2 references retain the existing 1e-5 bound; the largest observed
reference relative error was 1.699252517e-7. No bound was relaxed. Each complete
canonical output is 1,107,104 bytes; root verifies framing and every finite
payload before the complete T/H equality check. All processes exited normally
with code 0, with no forced cleanup or survivors.

These are legal synthetic packed weights and activations. Root verified that
their H=2560/FF=640 dimensions coincide with actual routed-expert geometry from
the header, loader's blob-length check and installed pack. They do not exercise
real model weights, routing, pool worker concurrency, output/state or full
physical-context lifecycle. Changing MT thresholds legitimately changes
accumulation bits across profiles; equality is required between T and H for
the same profile, not between different profiles or NT widths. Zen3 cannot
enter the IQ3S special Intel/VNNI MT1 route merely by forcing gather.

One parser-test launch failed in its receipt writer because `__file__` is a
string and the hash helper expected a Path. It returned code 1 without a
qualification receipt. Its original source and separate failed execution record
are retained. The corrected v2 test passes all 23 cases; it does not rewrite
the original failure or claim that the first run passed.

## Separate implementation and research work

Two Luna researchers completed shifted scopes while root built and tested.
Round30 found an upstream IQ panel candidate and documented limitations of the
existing prefill phase timers. Round31 mapped the panel's actual activation ABI,
scratch and task ownership, and inspected the repeat's QSA11 source. Round32
established the configured native NT ceiling and independently reviewed the
new histogram controller. Prior report directory, registry and committed
decisions were supplied on every assignment; root did not intervene mid-task.
Round33 investigated the actual NT1 fallback and the earlier changed GDN
ordinal while root owned the live GPU diagnostic. Both reports returned before
this boundary. Registry v30 is the immutable earlier snapshot with Sol running;
v31 records the completed source, tests and model diagnostic. There are now
71 retained Luna reports, including eight from rounds30–33.

Sol 6.1 implemented the separate owned H controller while root built H. Sol
executed no build/test/GPU operation. Root preserved its original v1 source and
fixed a reference to a historical admission field absent from the new build
receipt. V2 instead reads the separately pinned original T build, checks new
H's closed owners, and rejects native NT above two for the frozen spec4/split
configuration. Root owns all CPU/model execution and the common measurement
lock. The independent reviewer found no remaining producer-order/request-binding
or caller-count arithmetic blocker; this is source review, not runtime proof.

## Research decisions and limits

The pinned ggml already contains the IQ panel implementation, but Strata's
direct `vec_dot`/native row calls bypass its graph-level `MUL_MAT_ID` dispatch.
The upstream expert threshold is eight. With this exact `--spec 4 --spec-split`
configuration, each verifier group passes at most two tokens to the native pool;
`--pool-tasks 6` divides output rows and does not combine token groups. Thus the
NT8 panel is inapplicable to current decode traffic. No prototype is justified
from a synthetic NT8 success. Root also rejects Round30's suggestion to relax
bitwise numerical admission; Round31 explicitly preserves the old model gates.
[Upstream implementation](https://github.com/ggml-org/llama.cpp/commit/85c55223caf0a2ad0d1d88e5a73ab3fe36107867).

Current C full256K receipts lack phase timing, so their long prefill time cannot
be attributed to QSA. Existing timestamps include queue gaps and waits and do
not give per-layer/chunk pure kernel times. They are diagnostic leads, not an
additive critical-path breakdown. Source inspection of QSA11 found no concrete
defect; raw rows24..27 feed pooled block24582. A changed upstream activation
remains a hypothesis requiring intermediate captures. Round33 maps changed
final GDN ordinal7 to model layer9 and float index1 within its correct-sized
state slice. Reset and scratch inspection found no concrete defect. Final
state cannot establish the first transient difference. C adoption/performance/
full lifecycle stay false.

## Closed model check and next action

H's bounded diagnostic uses one owned process with four fresh 32K requests
A/B/A/B and 64 outputs each, existing tasks6 and unchanged production settings.
It is not four fresh processes or a clean performance comparison. First head,
all66 live state parts, IDs, logprobs and MTP compare against the independently
qualified integrated baseline. Histogram counts describe native caller jobs,
not every routed expert, actual task fragments, DRAM traffic or worker tails.
Exactly one histogram and phase report must belong to each request's stderr
interval. Fresh health/fault checks, normal QUIT, owned cleanup and no survivors
remain mandatory. All four requests passed these gates and repeated fixtures
had exactly repeated counts. The process exited normally with code0; no signal,
new fault, forced cleanup or owned survivor occurred. The original receipt SHA256
is `d007ae7b10003b3bb485dd74042cb9d592a696b4f60061c1f295e22013a0e66d`.

| Request | GU NT1 jobs | GU NT2 jobs | NT1 job fraction | NT1 routed-token fraction |
| --- | ---: | ---: | ---: | ---: |
| A0 | 26,748 | 6,928 | 79.43% | 65.88% |
| B1 | 29,982 | 8,312 | 78.29% | 64.33% |
| A2 | 26,748 | 6,928 | 79.43% | 65.88% |
| B3 | 29,982 | 8,312 | 78.29% | 64.33% |

Every populated cell has NT1 or NT2. GU/Down caller totals match exactly:
A has33,676 jobs/40,604 routed token entries; B has38,294/46,606. NT1 selects
ggml and NT2 IQ256 for GU. Root's read-only disassembly of the exact executed
H binary confirms AVX2 `ymm vpmaddubsw` instructions in all four named ggml
IQ/Q8K dot symbols. This corrects a possible inference that the ggml label means
generic scalar execution; it is not a runtime sample or bottleneck attribution.
Round33 found no distinct justified NT1 proposal after excluding existing work.
Its blanket repacking exclusion is too broad: lossless layout changes can
preserve weight values and remain possible under the existing gates. Also,
`--pool-tasks6` must not be described as six measured row tasks; it controls row
scheduling separately from the five workers.

All four individual logged prompt/decode and CPU-phase times are preserved in
the receipt. Diagnostic GU elapsed spans109.347–118.652 ms/window and Down
40.638–49.099; these nested host intervals are not additive critical-path times.
No comparison with clean performance is made. Fresh health passed three rounds
of16,384 exact words on BDF05 B570. Original kernel-query boundaries/statuses,
fault results and relevant entries are extracted before unrelated raw logs retire.

After actual exposure, prioritize a supported NT1/NT2 mechanism; keep prefill,
cold decode and restored decode decisions separate. Any speed claim needs
uninstrumented repeated fresh processes, and adoption still needs the full
physical262144 context lifecycle. Continue the bounded C capture investigation
under fresh health/fault admission using this newly closed H cursor; successful instrumented captures alone
cannot clear its earlier uninstrumented rejection.

Compact individual cases, commands, environment, actual compiler flags, failed
records and research are the durable Git record. Successful synthetic outputs
and repeated build progress have no remaining consumer after comparison and
verified extraction. Their retirement will be serial, after committed evidence,
with path/hash/reason identities. Required real reference/session/failure tensors
remain outside Git for their named current consumers.
