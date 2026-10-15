# R809 oneDNN BF16 qualification controller

This is a bounded source-only proposal for Main. All controller execution,
compilation, linking, syntax checks, tests, shared-library loads, GPU work and
profiling are **UNRUN**. The worker performed static file/header/ELF reads and
authored only this README, `main_controller.py` and `manifest.json`. The package
does not adopt the R804 component into the engine or change the Gemm ABI, its
68-case tests, CMake, services, packages, environment or existing proposals.

The current Main quality cohort owns all build/test/hardware activity. Main must
review the entire package and wait for that cohort's normal closure before
executing this controller. The package is currently blocked by actual unresolved
static dependencies under the authorized environment; no execution is suggested
until Main resolves that qualification boundary explicitly.

## Static findings and remaining blockers

The inspected oneDNN is `/opt/intel/oneapi/dnnl/2026.0/lib/libdnnl.so.3.11`,
version 3.11.4, reporting source revision
`0291f89430882d350585d276cc0d0eda623906e9`. Its SHA256 is
`3add4417acc7093ccdcc0dfe7f20d19cded96073f4bafd65a704ef2e72aafd19`.
R804's original manifest, complete proposal, static evidence and public headers
are pinned by file size, SHA256 and resolved path. The manifest also pins the
28-file resolved static ELF dependency closure and the exact v2 external inputs.
This is static dependency evidence, without loading oneDNN or calling its APIs.

oneDNN needs `libsycl.so.9`; the qualified official LLVM runtime's actual ELF
soname is also `libsycl.so.9`. This establishes soname agreement only. C++ ABI,
public API acceptance and real runtime compatibility remain UNRUN.

Four oneDNN dependencies do not resolve through the permitted fixed library
search paths or the inspected current loader cache/default directories:

- `libtbb.so.12`
- `libsvml.so`
- `libimf.so`
- `libintlc.so.5`

The permitted search path is the qualified official LLVM `install/lib`, existing
`/opt/intel/oneapi/mkl/2026.1/lib`, and the explicitly allowed extra
`/opt/intel/oneapi/dnnl/2026.0/lib`. oneDNN has no RPATH/RUNPATH supplying the
missing files. The controller records the exact unresolved edges and fails its
static dependency gate before compiling or executing the component. It does
not install, copy, symlink, preload, add compiler/TBB library directories, swap a
provider/backend/runtime, search for another oneDNN, or silently repair pins.
Changing the dependency boundary requires Main's explicit review and a newly
pinned proposal; this package cannot become qualified through an implicit retry.

There may be further blockers when Main eventually runs a reviewed compatible
package: public API rejection, a new dynamically loaded DSO outside the pinned
allowlist, unsupported strict attributes, numerical/classification failures,
unexercised scratch rejection, owner or driver failures. These are actual failed
or incomplete qualification outcomes. They do not become successful skips.

## Immutable inputs and direct build

Only the fixed v2 build at
`/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-own-native16-prefill-build-root-v2/build`
is used. Its qualified archive source revision is
`9c1a2020897a9451ec616a6bb036a661de9d86b0`.
`own-native16-prefill-build-v2.json` is pinned to SHA256
`743c6591123ac695325d20c984170e7801b9ef783fc44844c68995c091c96408`.
The unchanged 68-GEMM PASS receipt `own-native16-gemm-correctness-v2.json` is
pinned to `db4222cb981ff7124b7ab45d081d782730d7d3537e6f6e3c61917e990dbf7630`.
All frozen source pins from the build receipt are checked before and after the
job and each owned command. Report-only changes need no HEAD equality; a changed
compiled source fails the gate. No old v1 archive is used.

`manifest.json` contains the actual v2 fixture compile entry, final Ninja link
stanza and linker rule, plus complete direct compile/link argument templates.
Build expands those pinned recipes without running Ninja/CMake. It separately
compiles R804's original component CPP and a generated private qualified fixture,
then directly links those two objects with the exact existing v2 archives and
libraries and the pinned existing oneDNN. The extra proposal include directory
and oneDNN public include directory are explicit. All objects, generated source,
diff, caches, temporary files and executable live in a fresh private B root.
The existing CMake build and original R804 proposal remain read-only.

The complete qualified build/execution environment comes from the v2/R803
receipts. The sole library-search addition is the permitted existing oneDNN lib
directory. The child gets a complete environment before exec; `os.environ` and
running child environments are never mutated. The official LLVM installation
under a directory named XeStrata is explicitly permitted toolchain use only.
No XeStrata implementation, helper, diagnostic, test or source template is used.

## Private fixture safety transform

The original R804 fixture SHA256 is
`bf896b2ca07b2615f3dc2dac6958abf50e418efb1a2737dbb602627778c885e7`.
The generated private fixture SHA256 is
`525eea30a1970111d22f117327bbde6f03fbcc1c6a7dc7700f41111d77965dea`;
its exact unified diff SHA256 is
`44c23e71247cc541d70d244bc1b53cce56b8afc122117bd05ed91145cb704883`.
The manifest contains all ordered literal anchors/replacements and the complete
diff. Each anchor must have its exact expected count, and generated bytes and
diff must match those hashes. Unexpected input fails without an approximate
patch or original-proposal edit.

In `numeric()`, all device buffers and upload host vectors precede a checked
Runtime drain guard, and the guard precedes every enqueue. In `negatives()`,
the tiny/host allocations and the foreign raw USM owner are also declared before
that guard and before the first upload. The raw foreign allocation remains
unregistered, preserving that negative test. Its owner drains before freeing.
Upload/download helpers additionally put a checked queue guard after their
host-vector lifetime is established, protecting the download helper's local
vector when its wait throws. Failed drains print an error and abort before
uncertain storage destruction; errors are never swallowed. No guard calls
`g.wait()` after the component is closed.

The lifecycle fixture already declares `g` after the buffers and upload vectors.
Its retained local queue is drained by the component before those owners unwind;
the independent and automatic component objects also precede their buffers in
destruction order. That submit/close/destructor coverage is preserved. The
normal checked `Allocation::release()` is additional protection, but cannot by
itself protect an upload host vector that unwinds first.

The transform also emits loaded DSO paths using `dl_iterate_phdr`, identifies the
actual library containing `dnnl_version` through `dladdr`, and prints process and
submitting-thread user/system CPU seconds. It preserves all original numerical
cases, guards, negative checks and result markers. It does not add a GEMM,
change the oracle or require universal exact RNE. R804's faithful BF16 API remains
separate from Main's unchanged exact-RNE GEMM tests.

## Main-owned modes and gates

Every invocation requires Main's reviewed `--manifest-sha256`, a fresh bounded
`--run-id`, the exact latest normally closed passing receipt through `--prior`
and `--prior-sha256`, and `--confirm-latest-closed-prior`. Latestness is a caller
assertion; the controller never guesses it from filenames/globs. The generic
receipt gate accepts a receipt containing `commands`, including the natural
quality-screen receipt when Main has finished it. It checks passing/complete/
inactive state, boot identity, individual normal owner closures, fault/dump
state and absence of every prior PID/start identity. It never edits a receipt
or treats an original failed cohort as a passing gate. A no-GPU build receipt
can qualify as the preceding owner under the same checks.

- `build`: qualification gates, static dependency gate, two serial private
  compilations and one direct link. No GPU component is run. Its exact executable
  consumer, source transform, object/binary hashes and commands are recorded.
- `smoke`: requires that private `--build-record` and `--build-sha256`; runs one
  small `--smoke` process with
  `UR_ENABLE_LAYERS=UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION` and flushed
  `UR_LOG_TRACING=level:info;flush:info;output:stderr`. It requires a real kernel
  launch trace, original successful output, exact attributes, actual selected
  implementation and mandatory loaded-library identity proof.
- `check`: additionally requires a closed passing `--smoke-record` and
  `--smoke-sha256` for this exact manifest/binary and proof. It runs `--negative`,
  `--numerics`, `--chain`, `--lifecycle`, and `--scratch`, each in a separate
  bounded process, strictly serially. All original completion markers are
  required, including all 20 numerical cases. If scratch is actually zero and
  the original fixture prints `NOT_EXERCISED`, that evidence stays unchanged and
  the overall qualification remains failed/unestablished. No API unsupported
  result or skipped coverage is converted into PASS.

For each mode, Main can construct the invocation from the following argument
form after its prerequisite review. Angle-bracket values here are documentation,
not literal arguments or an invitation to execute during the active cohort:

```text
/usr/bin/python3 main_controller.py build --run-id <fresh-label> \
  --manifest-sha256 <reviewed-package-hash> \
  --prior <latest-closed-receipt> --prior-sha256 <its-exact-hash> \
  --confirm-latest-closed-prior
```

Use the same required arguments for `smoke`/`check`, adding the build receipt/hash
and, for `check`, the smoke receipt/hash. There is no benchmark/model mode,
automatic sequence, fallback, latest-glob, retry or parallel execution.

## Ownership, evidence and interpretation limits

The existing pinned `bench/results/2026-10-14-niko-main-service-port/direct_owner.py`
is imported unchanged (SHA256
`a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f`).
All work uses B's `owned-v0141-measurement.lock`. Owner records boot/PID/start,
anchored session, deadlines, normal/abnormal exit, reaping, survivors and owned
cleanup. Only that Owner may signal its pinned owned pidfds. No unrelated
signals, PID adoption, dump removal, GPU reset or service action exists here.
Same-UID no-model-owner scanning occurs only when Main executes the controller;
the source-only worker never performs it. It complements the common lock and
prior identity checks, without claiming to disprove uncooperative foreign work.

Initial journal collection uses a fresh `journalctl -k -b -n 0` cursor, with
current owner/dump checks. Each native case gets its own before/after fault
window, and closure gets a final window even after a normally exited failure.
The xe device and all existing devcoredumps are inspected read-only. Source,
package, receipt, archive, external and library pins are checked before/after;
each successful native process must enumerate only pinned loaded libraries,
including the exact dnnl/SYCL/Level Zero adapter, and identify dnnl's symbol owner.
An unknown loaded DSO fails; it is never adopted automatically.

Whole build/smoke/check budgets are 600/480/1200 seconds. Individual compile/link
commands have at most 300 seconds; native cases 180 seconds; journal checks 20
seconds. Owned RSS limits are 32 GiB for build and 8 GiB for native checks. The
aggregate raw text budget is 64 MiB, private job budget 512 MiB, original fixture
stdout budget 256 KiB and compact durable receipt budget 512 KiB. Owner enforces
additional CPU, file and identity bounds. Commands and cleanup are serialized.

Active/private evidence stays in `B/main-r809-onednn-<run-id>-<mode>/record.json`.
Only after collection and ownership closure complete does the controller create
one new compact A receipt `own-native16-onednn-<mode>-<run-id>.json`. A closed
failed diagnostic can have `complete=true` and `passed=false`; its original FAIL
and exit status stay intact and cannot satisfy a following gate. Incomplete
ownership/fault/pin collection leaves only private evidence for Main review.
There is no file cleanup. Main alone may retire successful full traces after
verifying the extracted evidence and recording path/hash/reason deletion
manifests. No large raw traces enter this source package or Git.

Actual selected implementation, strict attributes, scratch/primitive accounting,
original stdout, CPU time, loaded identities and fault results are evidence for
bounded component qualification. Process/thread CPU deltas start before Runtime
creation and end after the fixture's final Runtime check. They include setup,
JIT, submission and waits, excluding loader startup, evidence collection and
final Runtime destruction. Owner also records full command wall time. These
deltas do not isolate host arithmetic from completion polling. A GPU
engine/queue, UR launch, implementation name or aggregate scratch
count does not prove XMX use, all-GPU arithmetic, zero internal FP32 storage or
speed. These remain explicit unknowns. Model quality, model performance and full
physical 262144 context correctness are **UNESTABLISHED**; no benchmark is run.

Shared research metadata is
`bench/results/2026-10-15-pr6-recovery-audit/research-index.json`; the manifest
records its observed hash and relevant metadata scope. It is not an execution
pin because Main owns ongoing index updates. No knowledge-collection report
body was read. All newly authored files follow the repository's MIT LICENSE;
oneDNN remains an external Apache-2.0 dependency.

## Main review after source archive 709235b0

Main reviewed the full controller and private fixture transform. Four installed
oneDNN dependencies are now explicitly pinned and available after the existing
LLVM/MKL/DNNL paths: TBB 2023.1 and compiler 2026.1 math DSOs. All previously
resolved sonames, including libsycl/UR/IGC, must remain identical. No package,
driver, runtime provider or system configuration is changed. The actual mapped
DSOs still require the immutable allowlist proof. Earlier unresolved-dependency
statements describe the archived source proposal, not an executed failure.

An actual zero-scratch primitive leaves scratch-rejection coverage explicitly
NOT_EXERCISED. It does not become PASS for that coverage, and it does not by
itself fail the other qualification checks. API/runtime/numerical/owner errors
remain failures. No product or Gemm test was weakened; execution is still UNRUN
at this review boundary.

The first Main build is an unchanged FAIL in `own-native16-onednn-build-v1.json`:
the qualified SYCL event wait method is non-const. The next private fixture
changes its local event to non-const and requires the original failed CPU-only
job's exact closure; GPU execution was false. Effective compile/link templates
remain in this manifest. Redundant raw v2 recipe copies are recoverable from
archive709235b0 and the pinned v2 receipt.
