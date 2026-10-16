BF16 residual boundary storage proposal v2, 2026-10-15

V1 omitted the existing compiled CPU boundary fixture's boolean constructor
callers. V2 corrects those callers and adds CPU BF16 audit coverage. V1 stays
byte-identical, including its proposal, fixture, consumer and manifests.

`source.diff` is a complete replacement proposal against the actual unchanged
root. It contains all nine V1 file changes plus the change to
`tests/cpu/prefill_boundary_audit.cpp`. Apply V2 alone, after Main rechecks every
base SHA in `source-manifest.json`; do not apply V1 followed by V2. Every root
base/proposed SHA is recorded. The two fixture sources are also supplied for
review; the existing large engine sources are represented only by the diff.

Only this V2 proposal directory was written. No root sources, source pins, V1
files, historical checkers or receipts were changed. No test (including host),
compiler, build, GPU, process inspection, runtime environment operation, Git
mutation or cleanup was executed. This is source review and payload generation,
not validation. Main's concurrent GPU profile continues to use unchanged root
sources.

All V1 implementation semantics remain: FP32 default/reference; exact
case-sensitive fp32/fp16/bf16 selection; both two-byte modes share checked
planning/admission and generic staging ownership; BF16 uses shared RNE
`bf16_from_f32` and exact `f32_from_bf16` expansion. FullGPU, host and mixed
paths pack once at each non-final boundary and restore once into the unchanged
FP32 live residual. Final callbacks, arithmetic, operators, weights, KV and
recurrent-state types are unchanged. The SYCL activation `to_bf16` remains
unchanged and is not the BF16 boundary primitive. Phase audit remains FP32-only.

The CPU fixture now passes explicit `LayerMajorResidualDtype` values in every
constructor, including all existing validation and failure cases. Existing
FP32/FP16 output/raw-byte, rejection, sequence, failed-copy and corrupted-store
checks remain. New assertions pin their historical phase names. No existing
expectation was weakened.

New CPU-only BF16 coverage uses 35 independent literal input/packed-bit pairs
for ties-even, both signed zeros/infinities, finite overflow, positive/negative
subnormals and underflow, and signed signaling/quiet NaN payloads. Each pack
must match literal bytes; each restore must match the literal BF16 code shifted
left 16 by byte comparison. Two selected rows traverse all 48 layers and 47
boundaries in each of fullhost/fullgpu/mixedstore label simulations. Every arm
requires 96 output records, 94 packed records and 94 restored-input records,
exact selected FP32 raw bytes, correct BF16 metadata and the expected RAM/GPU
placement labels. These labels simulate placement on the CPU: its Copy callback
uses host memcpy throughout, and this is not GPU/queue correctness evidence.
An invalid enum must reject before log/raw file creation. A deliberately
corrupted BF16 packed value must produce both stored/input differences and
retain `status=complete boundary_memcmp=different`.

CPU fixture environment switches remain confined to its existing standalone
CPU-only process. It initializes no SYCL, Level Zero or driver provider. The
GPU fixture still requires all experiment/runtime variables in its native child
environment before exec and never mutates that environment after initialization.

Actual caller/build review (read-only):

| Source or build entry | Finding and V2 action |
| --- | --- |
| `src/prefill/layer_major.inc` | The only existing production constructor site. V1/V2 pass `plan.dtype`. |
| `tests/cpu/prefill_boundary_audit.cpp` | The only existing root fixture containing constructor calls, through the `Audit` alias. All boolean constructor arguments become explicit enums in V2. |
| `src/prefill/prefill.cpp` | Holds audit pointer/Copy members; no additional constructor. Includes `layer_major.inc`. |
| `include/strata/prefill/layer_major_boundary_audit.hpp` | Constructor changes from bool to explicit enum and rejects invalid enum before files. |
| `tests/sycl/prefill_bf16_storage.cpp` | New proposed fixture from V1, unchanged; its constructor passes `plan.dtype`. |
| `cmake/sycl_backend.cmake` | Directly compiles `prefill.cpp` and SYCL `postops.cpp`; adds the GPU BF16 fixture/conversion target in this proposal. |
| Root `CMakeLists.txt` | `STRATA_BUILD_PREFILL_HOST_TESTS` registers `prefill_layer_major_plan_test`. It does not register the CPU boundary fixture. `tests/CMakeLists.txt` is absent in this checkout. |
| `apply_build_boundary_service_v1.py` | Existing A recipe source compiles the actual CPU boundary fixture explicitly with C++17, `-UNDEBUG`, warnings-as-errors, normally and ASan/UBSan. V2 CPU additions retain C++17 compatibility. This historical recipe remains untouched. |
| `apply_build_phase_audit_v1.py` | Existing A recipe source explicitly compiles the same actual CPU fixture with C++20 and `-UNDEBUG`. It remains untouched. |

The caller search covered root `src`, `include`, `tests` and `tools` for the
class and `Audit` alias constructors. Historical proposal source snippets are
evidence and remain unmodified. Main should use a new pinned build/controller
receipt rather than rerun a historical apply script with old patch/source pins.

Audit metadata compatibility review:

| Arm | Stored phase | Restored-input phase | Metadata |
| --- | --- | --- | --- |
| FP32 | `stored` | `next_input` | `storage=fp32`, identity converters |
| FP16 | `stored_RNE_packed_bits` | `next_input_decoded_expected` | `storage=fp16`, existing FP16 helpers |
| BF16 | `stored_bf16_RNE_packed_bits` | `next_input_bf16_exact_expected` | `storage=bf16`, BF16 RNE/NaN-preserving pack and exact expansion |

`boundary_audit_v2` adds explicit dtype/converter metadata. Raw output remains
FP32 with the same ordering, byte budgets, phase semantics and completion line.
The existing `check_boundary_audit_v1.py`,
`check_boundary_audit_phase_off_v2.py` and `check_phase_audit_v1.py` parse
key/value phase records rather than require the v1 header prefix. Their
FP32/FP16 store/input names therefore remain available. They do not consume the
new BF16 phase names and must not be used to qualify BF16. Their original
nonfinite checks also remain unchanged. The new consumer is explicitly for
synthetic BF16 mechanics and accepts `boundary_audit_v2` plus BF16 names.

Main's remaining serial gates, none executed here:

1. Recheck V2 base SHA values, integrate only V2, compile the actual CPU fixture
   using `-UNDEBUG` and warnings-as-errors, and run its original/new contracts
   plus ASan/UBSan. Use a fresh private output prefix; the audit uses exclusive
   file creation. Keep original outcomes unchanged.
2. Run the extended host planner, existing exhaustive BF16 bit-helper fixture,
   existing FP16/postops regressions, and the SYCL build. The planned 262144
   arithmetic is not physical maximum-context correctness/admission evidence.
3. Run GPU fixture modes conversion/fullgpu/fullhost/mixedstore separately and
   serially, with owned-process closure, fresh faults and actual compiler/
   runtime/device/source/binary identities. V1's README documents arguments,
   environment, finite budgets and closure schema. Use this directory's copied
   consumer and V2 manifest/diff hashes; its `compiled_source_sha256` must also
   include the added actual CPU fixture source. The copied GPU fixture/consumer
   sources are byte-identical to V1; only the consumed proposal identities differ.
4. Run actual-model BF16 boundary and placement checks using separate BF16
   receipts. Then matched natural-history selected full-vocabulary KL and
   teacher-forced target NLL, paired task/retrieval quality at intended 32K/64K,
   physically admitted 262144-context correctness, RESTORE, and clean matched
   end-to-end prefill/decode performance. Quality, performance, RESTORE and
   physical-maximum gates are unchanged. BF16 remains experimental/unqualified;
   FP32 remains default. 600/1000 tok/s are goals, not results.

No run artifacts were generated. Preserve the compact proposal, source and
identities for Main's next idle review. Future raw CPU/GPU audit files have the
named mechanics comparator as consumer; retire them only after closed-run
comparison is recorded and verified, under the measurement lock with a
path/hash/reason manifest. No cleanup is authorized during the live profile.
