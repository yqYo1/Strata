# Bounded repeated-prefill capture (source only)

Base: `6cdf81078128488e5797afb9e92824b7ebe51124`. The frozen C full-context
receipt SHA256 is `a3413ac4c3f20ae593d8507a7db0ac405cd7fab2de286547460e6ba82b0d9404`:
normal exit, healthy, but the second fresh full request failed exact math.
The localization receipt SHA256 is
`96f908d51cb3fe535c892e38550fb55c6f41a35a2aa54e24f1e8c52f4dffcb23`.
First changed persisted QSA K/V/scales start at cell 98328, ordinal 2;
this does not establish the first changed compute operation. R13/R14 research
and registry v17 were read; no root cause is established.

Set `STRATA_PREFILL_REPEAT_CAPTURE` to a controller-owned private binary file.
Use a fresh path inside a private directory, preserve the exact environment,
and keep all ordinary run ownership, deadline and numerical/fault gates.
The file is append-only, requires a regular file owned by the effective user,
permissions without group/other access, one link, and no final-path symlink.
No file is opened or device operation issued for nonoverlapping chunks.
Unset disables all diagnostic allocations, file IO, queue waits and copies;
only the enable/row guards remain. Source/math/kernel dispatch is unchanged.

The only physical rows captured are `[98304,98336)`, intersected with the chunk.
Source layers 0 through 12 capture R after PLE and before the half loop, after
attention/GDN half and after MoE half, including any control-vector write.
R width is the existing `D=10240=4*2560`. QSA layers 3,7,11 additionally capture
Kc/Vc after the append submissions (the float quantization inputs are unchanged
by append), raw indexer projection, normalized/rotated query and q_idx,
active score blocks, device step fields and selected IDs, and attention output
after inverse rotation before gate/output projection. Score rows are copied
inside their score batch before scratch reuse, each exactly `n_bid+1` blocks;
IDs are exactly the step's active width. Padding is never read.

Each record has 12 native-endian uint64 words followed by a 32-byte zero-padded
phase string and 4-byte elements. Words are magic `0x5354524152505431`, version
1, source layer, p0, chunk T, physical row first, row count, element count,
type (1=float32,2=int32), contiguous row width, process record ordinal starting
at zero, and header size 128. Record ordinal continues across requests. The
controller must associate request transitions with its protocol records;
there is no independent request identifier in this diagnostic file.

Both existing file contents and new records count against 128 MiB. A failed
shape, file open/ownership, write or budget check fails the diagnostic request
with an explicit error. Partial last records indicate incomplete diagnostics.
Files remain scoped to the process, with close checked at exit; close failure
is reported on stderr. Each successful record reports count/layer/phase/row/
bytes. No broad KV/state copy, full-buffer hash or profiler was added.

The opt-in queue drain and blocking CPU copies can mask timing-dependent bugs.
Host copy destinations remain alive through event completion and queue drain
on an asynchronous error. Captures may also perturb expert issuer overlap.
An equal capture is not proof that an ordinary unsynchronized run is exact.
The fixed row window may miss an earlier transient difference outside it;
final GDN states cannot alone establish where that difference originated.

Validation: source inspection and `git diff --check` only. No build, test,
selftest, GPU run, model invocation or profiler was performed. Main owns all
serial validation. Exact compile/runtime behavior, budget sufficiency for the
chosen request sequence, output parsing and capture parity remain untested.
No traces or frozen reference files were modified or deleted.

Retention: future capture owner/controller is the root experiment. Consumer:
first-versus-repeat comparison of these compute edges for the unresolved
full-context mismatch. Binary budget 128 MiB/process. Keep outside Git and
review after edge localization or superseding diagnostic; retain hashes,
comparison results and deletion manifest. This report is the Git resume record.
