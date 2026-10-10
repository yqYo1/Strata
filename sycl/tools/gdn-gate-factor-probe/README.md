# Private GDN gate-factor numerical screen

Source-only, unbuilt and unrun. This standalone SYCL tool is a fail-fast necessary
screen for moving gate exp into its producer. A pass does not qualify recurrence
contexts, compiler/backend contexts, model outputs, full256K, or adoption.
No production source, public gate representation or selector changes.

The pinned snapshot `gdn-gate-consumers-cd353f30` manifest is
`1ac15ccad1b17ca2913f17abdf40a4989f567a39568cb27bf74721af9572dcfb`;
`kernels.dp.cpp` is
`cc5254f688b9138fc71800884c853211771067c0f07a0a44a237fd45b59b4bfa`.
Geometry HV48/HK16/S128 is from its constants at line46. Producer gate and beta
expressions copy lines456–469 and sigmoid lines84–86. Baseline writes FP32 loggate
and beta, then separately submitted scalar/SG16/SG32 readers call native exp on
the stored bits. Candidate computes float loggate and exp in the same producer,
while also storing loggate and beta. No volatile/store-reload trick guarantees
that candidate exp sees a physical roundtrip. This difference is the question.
All loggate/beta/factor comparisons are bitwise, no tolerance/native substitution.
SG readers are independent indexing/lowering screens, not recurrence contexts.

Default finite inputs include negative/near-zero synthetic log-gates and the
producer v below/at/above20 using exact adjacent FP32 values; artificial finite
large-magnitude v crosses underflow/subnormal regions. Those are not established
observed model ranges. Optional `--include-artificial-nonfinite` replaces two
head classes with +Inf and a qNaN; these are explicitly artificial/out-of-domain,
and differences still produce nonzero exit. NaN behavior is not portable evidence.
Inputs depend deterministically on token/head, dt0, ssm_a negative. No payloads.

Fifteen lengths0/1/3/5/15/16/17/127/128/129/8191/8192/3/8192/8192 exercise empty,
tails, guarded active prefix and large-small-large/repeat-reset reuse. Outputs are
reset to bit canaries each case, inactive tails and both guards checked. Inputs
including guards are immutable and bit checked. Full8K repeats must retain the
combined output fingerprint. All buffer extents checked; allocations below16MiB,
max8192*48 active elements and bounded15cases/75stages. Additional host input
snapshot is bounded. Source caps are not process RSS/deadline guarantees.

Requires an Intel Arc B570 LevelZero GPU, device SG16/SG32 and WG128; CPU/OpenCL are refused.
Owner must select and pin the intended GPU, compiler/backend/options; selection
is checked by backend, Intel vendor and B570 device name, recorded with driver version. Explicit bounded nonthrowing sticky async handler,
in-order queue, flushed STAGE before every submitted/validation boundary, and
wait_and_throw after every stage. Any stage exception/async error terminates via
_Exit2 without unwinding/freeing possible in-flight endpoints. Parent must retain
only its owned process session and driver diagnostics and enforce wall deadline.
There is no recovery, reset, retry or process-killing code in this tool. On fully
drained numerical/guard errors, ordinary RAII frees allocations; success exits0
only after all cases/final flush. Root owns any error-session handling. SYCL waits
have no intrinsic wall timeout: owner supervision is mandatory.

Root-only unexecuted build/run recipe:

```
cmake -S "$W/sycl/tools/gdn-gate-factor-probe" -B "$PRIVATE_BUILD" \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$PRIVATE_BUILD" --target gdn_gate_factor_probe --parallel 2
"$PRIVATE_BUILD/gdn_gate_factor_probe"
# separate explicitly labelled robustness screen:
"$PRIVATE_BUILD/gdn_gate_factor_probe" --include-artificial-nonfinite
```

Root runs under shared serial lock, exact owned-session/wall/CPU/RSS/FSize/NOFILE
limits and retained LevelZero diagnostic environment. Suggested stdout+stderr
cap1MiB, wall120s, RSS1GiB; root reviews actual limits before execution. No timings
are emitted. STAGE/CASE/CASE_PASS/FULL_REPEAT_HASH and terminal counts are finite;
first difference reports case/T/index/field/baseline/candidate uint32bits/domain.
Source asserts15cases/7,970,640comparisons/three full8K repeats before its terminal.
Require these counts and clean normal0/flush plus fault/closure checks; a parseable
terminal alone cannot establish process success.

Build uses IntelLLVM C++20, -fsycl/-fp-model=precise, default SG32, per_kernel split
and the production JIT correctly-rounded-divide/sqrt backend option, without
fast/relaxed changes. Root must verify all actual inherited configuration/link
flags, target backend and production-match compile options; CMake guard is not a
complete arbitrary flag auditor. Native exp accuracy/range/context agreement is
implementation-defined. Necessary remaining gates are each real serial/columns/
pipeline/keyhead/quad context, actual generated device code, exact state every
token/final state and normalized output, real reachable inputs, carry/multiple
layers/repeated prompts/physical262144, errors and model math. Convolution and
postnorm/state are absent. Earlier rejected CPU candidate remains rejected; this
screen cannot establish speed or adoption.
