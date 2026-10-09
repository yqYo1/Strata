# Native CPU wrapper differential harness

Source only / unbuilt / untested. This implementation branches from frozen H
`96bd5bb4e054f6ddcf677fd96e161b499a013fd7`. Frozen T is
`7d0105f2a942ac72ca20fef69d4c2e7960653de3`. Root owns all serial builds and
execution. No configure, build, test, syntax/help/version query, target, model,
GPU, profiler, service, cleanup, Git network operation or push was performed.
Source-writing Python only edited CMake text; it did not import or run a test.
The original H, T and reader worktrees remain unchanged.

The opt-in `STRATA_SYCL_NATIVE_WRAPPER_PARITY=ON` CMake option adds
`native_wrapper_parity`, linked only to the actual SYCL project's
`strata_kernels_cpu` and its public ggml/Threads dependencies. It does not link
`strata_kernels`, `strata_core`, an engine or a GPU parity target. The existing
SYCL project still applies its global compiler/link flags; the harness creates
no SYCL device or queue. Build only this target. The option defaults to OFF and
requires native experts and Linux for its private output sink.

Source resolution correction: neither frozen H nor T contains
`sycl/src/kernels/cpu/native_expert.cpp`. Their existing
`strata_resolve(_cpu_native ...)` therefore selects
`src/kernels/cpu/native_expert.cpp` for the production wrapper and
`sycl/src/kernels/cpu/iq_avx2.cpp` for the migrated IQ entry point. That migrated
file includes the common kernel and optionally substitutes the GCC IQ2_S groups.
This harness links that exact resolved production library. It does not alter
resolution or reproduce resolver predicates as an oracle. A generic top-level
`iq_avx2_parity` build bypasses this SYCL library mapping and is not the requested
qualification. The source report's earlier generic-target recommendation is
superseded by this target.

The harness includes the existing `src/kernels/cpu/iq_avx2_parity.cpp` with its
main renamed, reusing its `fill_blob`, `Acts`, ggml-trait access, relative-error
and bitwise row helpers. The renamed original main is never invoked. Its source
SHA-256 is `fb4e0b0629e749b816c4ec71da836f383330cf93aed49a11d03607937e1232d9`.
The source and SYCL CMake were identical between T and H before this patch; the
same two-file harness/target patch applies to both. Root must use separate new
T/H qualification worktrees, leaving frozen sources untouched, apply this exact
patch to both and assert identical harness and CMake delta hashes. Do not copy
H's wrapper or resolver into T.

The 48 cases call `native_gu_rows` for GU 18/21/22/23,
`native_down_rows` for Down 20, and the pool's actual Q2 direct entry
`q2_rows_any` for Down 42, each at NT 1 through 8. Production native activation
quantizers and `act_quant_any` consume the reused deterministic finite-float
fixtures. ActQ is checked for the expected 20 chunks, finite nonnegative scales,
finite correction values, and no -128 codes. Encoded weights use the existing
legal synthetic block generator with valid FP16 scales. This does not qualify
externally manufactured arbitrary Q8 bytes.

For each exact type/NT tuple the harness checks:

- All full, partitioned, direct-reference and independent-reference outputs are
  finite; sentinel padding and inactive tokens remain untouched.
- A full wrapper call equals three disjoint calls over `[0,1)`,
  `[1,rows/2+3)`, `[rows/2+3,rows)` bitwise. A separate interior call over
  `[7,rows-9)` matches those full rows bitwise and leaves other rows untouched;
  an empty call at `rows/2+3` changes nothing.
- Full wrapper outputs match ggml dots with the same quantized activations and
  SwiGLU formula, or for Q2 a double sum over the same ActQ/scales/codes, at the
  pre-existing aggregate relative L1 bound `1e-5`.
- Full wrapper outputs also match the independent direct AVX2 scalar-variant
  entry for that exact NT at `1e-5`. This second reference bypasses the wrapper;
  it does not duplicate branch selection. Direct AVX2 calls require detected
  usable AVX2. No AVX-512 or AVX-VNNI feature is simulated.

The tolerance uses the existing `iq_avx2_parity` reference metric. Applying it to
GU's final SwiGLU is a proposed additional gate, not a measured pass; if root
observes a rejection it must retain and investigate it rather than relaxing the
bound to achieve admission. NT1 output is never demanded to equal NT2 output:
one-token ggml and grouped kernels can legitimately round differently. The
required T/H bitwise comparison instead pairs the same type, NT, inputs,
compiler settings and fresh-process environment.

The reused synthetic dimensions are H=2560 and FF=640, with block-aligned
reductions and every legal NT. They cover wrapper control flow and row arguments;
this is not full-model-shape qualification, worker scheduling, histogram count
reconciliation, performance, GPU math or an end-to-end decode test. KQ controls
and types 12/7/8 are excluded until this base gate passes. Setting the histogram
environment alone does not exercise pool accumulation in this standalone test.

## Root execution recipe (not executed here)

Under root's existing exclusive measurement lock, finite CPU supervisor, fresh
private output directories and no concurrent build/test/GPU work, reuse the
preserved T configuration. Its receipt is
`decode-pool-phase-timing-v0141-private-build-v2/record.json` under
`/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007`;
compile_commands SHA-256 was
`8569f31555066098f9922ac74dc15b7010162f4b1cfbacc4edb2d695dd83d636`,
CMakeCache SHA-256
`09a6a41f91f30d7f55ecd2afbfc0c91b4816d7e72c9aa3ba4b96bac04673fdac`.
H has no harness build result yet. Root must record fresh T/H configuration,
reachable object flags and harness/library/binary hashes; no original-build
receipt qualifies this new executable automatically.

A representative configure/build recipe for each patched worktree is below.
Root substitutes fresh absolute source/build/output paths and checks the local
pinned ggml checkout at `3cf03257f219afbe7334045ff7c6a06ac68c627d` first. Reuse
its existing checkout instead of FetchContent/network activity. Preserve
oneAPI 2026.1, Release, precise float policy and existing per-source
`-mavx2 -mfma -mf16c` flags. No new march/ISA setting is introduced. Existing
ggml native settings remain those of the recorded production configuration.

```sh
/usr/bin/cmake -S /absolute/patched-source/sycl -B /absolute/fresh-build -G Ninja \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx \
  -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DSTRATA_GGML_DIR=/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_SYCL_PARITY=OFF \
  -DSTRATA_SYCL_NATIVE_WRAPPER_PARITY=ON -DSTRATA_IQ2S_GCC=OFF \
  -DSTRATA_SYCL_AOT= -DSTRATA_SYCL_SPIN_MAX= -DSTRATA_NATIVE_POOL_TASK_FACTOR=0
/usr/bin/cmake --build /absolute/fresh-build --target native_wrapper_parity --parallel 4
/absolute/fresh-build/native_wrapper_parity --out /absolute/private/fresh-output.bin
```

The build graph must show the resolved SYCL CPU library and ggml only. Capture
stdout/stderr, each of the 48 case outcomes, normal exit, closed output size/hash,
CPU brand/effective flags, uname/boot identity, deadline/owned process identity
and no-survivor result in root's compact receipt. Zen3 should report actual
AVX2 usable, AVX-512 unavailable and AVX-VNNI unavailable; refuse unsupported
AVX2 instead of reporting a skipped pass. Record any affinity root applies.

Use a fresh process per profile because production settings are cached statics.
Start with all relevant controls unset and histogram unset/off, then pair T/H
with identical isolated overrides: `STRATA_IQ_MT_MIN=1`, `=2`, `=3`,
`STRATA_NO_IQ256=1`, `STRATA_NO_IQ4NL=1`,
`STRATA_IQ256_GATHER=0`, `=1`, and `STRATA_IQ3S_MT1=1`.
The IQ3S special-case profile requires unset IQ_MT_MIN, present IQ3S_MT1,
usable AVX2, IQ256 enabled, no AVX-512, nonzero gather setting and
`cpu_gather_fast()`. The latter requires GenuineIntel plus AVX-VNNI and is false
on Zen3: forcing gather alone cannot qualify that special selection. An optional
pair with `STRATA_IQ3S_MT1=1 STRATA_IQ256_GATHER=1` checks the flag combination
without claiming that Zen3 can enter the special branch. The harness records
actual gather capability/setting, kernel variant and public GU thresholds. Keep Q2 bitplane and KQ unset, and use no FORCE_ISA/fake-feature
matrix. Preserve other production controls identically and record them.

Every local gate must pass and each matched T/H raw output must be byte-identical
before qualifying this narrow wrapper gate. Root can use `cmp` and SHA-256 on
closed outputs, then localize a mismatch by the fixed case header and first raw
word. A program exit 0 means only its local gates passed: the summary explicitly
records `T_H_bitwise_compared=0`. The external root comparison establishes the
T/H result. Stop on any failed local gate or differential mismatch; do not infer
a pass from matching output hashes alone when either local gate failed.

## Output and retention contract

`--out` creates a fresh owned 0600 regular single-link file with O_EXCL,
no symlink in any opened parent component, and an owned private final parent.
Dot path components are rejected. Writes are capped at 2 MiB per process,
checked for errors, and final file mode/ownership/link/size and close result are
checked. It never overwrites or deletes an artifact. A successful complete
output is exactly 1,107,104 bytes. Failed/partial outputs remain failed evidence.

The format is canonical little endian: eight uint32 file-header words
`0x53575031,1,48,2560,640,8,0,0`; then 48 cases in GU types 18,21,22,23 followed
by Down types 20,42, NT increasing 1..8 inside each type. Each case has six
uint32 words: phase (GU=1, Down=2), type, NT, row count, float-word count,
reserved zero. Payload contains every full wrapper output as raw IEEE binary32
words in token-major then row-major order. Payload is neither rounded text nor
just a hash. Partition/reference arrays are gate evidence, not substituted
output. The complete output contains 1,105,920 payload bytes plus 1,184 framing
bytes. At most 2 MiB raw output and root's finite text-log budget are admitted per
process; keep small binaries outside Git. Root can set a 180-second execution
supervisor initially; this is a proposed bound, not a measured duration.

Owner/consumer: root native wrapper CPU qualification, matched T/H output
comparison and first mismatch localization. Next review: root closes every
profile receipt, commits compact source/configuration/case/differential results,
then retires successful duplicate raw outputs under the shared lock with a
path/hash/reason manifest. Keep one representative required failure output per
distinct unresolved mismatch. This implementation performs no cleanup.

Evidence read: registry v23 (SHA-256
`5dfcd721be2d40077ef584264f05ba370c118f5f94aad44c3117fb7733f19e2b`),
CPU phase R7, GU/task R17, histogram R18/R20, Zen3 R21/R25, R24 reader audits,
R26 fixture/stale-content audits and R27 wrapper design/final reader audit.
The original C full-repeat math gate stays false and its rejection is not cleared.
H is not adopted, full lifecycle is not qualified and no speed claim is made.
