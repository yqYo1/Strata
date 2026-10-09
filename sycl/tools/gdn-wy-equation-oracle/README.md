# Conditional GDN chunkwise WY equation oracle

Source-only mathematical prototype. No build/test/run was performed by the implementation agent. This standalone portable C++20 target has no production translation unit, parent SYCL project, ggml, model reader, runtime/device dependency or dispatch. It cannot establish model parity, speed, adoption or full lifecycle qualification. Root owns all execution and final mathematical review.

The equation input is already post-convolution Q/K/V, natural-log decay gate `g` and already-sigmoid beta. The oracle does not construct gates from weights/logits, sigmoid beta again, convolve, normalize Q/K as the production conv/L2 kernel does, post-normalize/gate/project recurrence output, convert it to FP16, or run a model. Its synthetic Q/K are normalized to keep fixtures bounded; they are not captured model data. Conv history, output RMSNorm/gamma/sigmoid(z), projections and residual/MoE paths are outside scope.

For each value head h, Q/K head is `h % 16`. All48 value heads are covered. State is indexed `[key_row,value_head,value_column]`, matching the production logical state layout. For key vector k, value v, query q, beta b and a=exp(g):

```
p = S_old^T k
d = b (v - a p)
S_new = a S_old + k d^T
o = S_new^T q / sqrt(K)
```

`serial()` independently iterates tokens with ordinary ordered projections/state updates. `wy()` does not call it or perform a tokenwise state recurrence. Within a live chunk, set G_i=sum(g_0..g_i), P_i=exp(G_i), and define the causal triangular matrix with unit diagonal:

```
L_ij = beta_i exp(G_i-G_j) (k_i^T k_j), j < i
L_ii = 1
L_ij = 0, j > i
L U = diag(beta) V
L W = diag(beta P) K
D = U - W S_start
```

These two forward substitutions are independent of the incoming state. Expanding the original recurrence yields:

```
S_i = P_i S_start + sum(j<=i) exp(G_i-G_j) k_j D_j^T
o_i = [P_i S_start^T q_i
       + sum(j<=i) exp(G_i-G_j) (q_i^T k_j) D_j] / sqrt(K)
S_end = P_last S_start
        + sum(j live) exp(G_last-G_j) k_j D_j^T
```

To derive L, insert the expansion for S_(i-1) into `d_i=beta_i(v_i-a_i S_(i-1)^T k_i)` and collect earlier d_j on the left. Thus `L D=diag(beta)V-diag(beta P)K S_start`, giving D=U-W S_start. The final state is carried into the next chunk, never reinitialized. Ratios use exp(log-prefix differences), avoiding a division by a possibly small prefix product; the fixed synthetic gate ranges remain finite. No FLA code or serializer is used. Head indexing remains modulo16 rather than FLA's contiguous three-value grouping.

The fixed suite has100 positive fixture cases plus one wrong-map negative discriminator:

- Cheap shapes K4/V3, all16/48 heads, chunks1/8/16/32/64, zero and nonzero initial state. Lengths0, C-1, C, C+1 and2C+1 (deduplicated atC1), plus T129 for the smaller chunks; C64 already covers129. This includes empty live input, partial tails and multiple handoffs, including129 single-token handoffs.
- Beta0 with decay, g0, beta1 and beta0/g0 identity controls at2C+1, both initial-state kinds. Beta0 final states also match independently calculated analytic `exp(sum g) S_initial`; beta0/g0 state identity is checked directly.
- Two literal K1/V1 cases atT3/C1 andT3/C8, all heads, k=q=1, g=-ln2, beta=.5, initial state3, v=(2,4,-1). Independently written expected outputs/states are1.75,2.4375,.109375. These distinguish gate placement and use of pre-update versus post-update state.
- Only two full-width16/48-head K128/V128 representatives: T17/C16 nonzero state and T65/C64 zero state. Maximum executed T is129; this is not physical262144 execution.
- A deliberate wrong scalar mapping `value_head / 3` on a distinct-head K4/V3/T17/C8 nonzero fixture must differ from modulo16 by more than1e-6. This is a synthetic negative discriminator threshold, not a model output tolerance.

Each chunk uses fresh live-only synthetic input buffers; unused C-live slots are NaN. Output tails and all buffer boundaries are finite sentinels, checked unchanged. Live output prefixes, states and WY scratch must stay finite. Scalar and WY update exactly T live tokens, with a checked `T*48` logical head-update count; padded input and output slots remain untouched. Only final states cross chunks. No all-token state history or all-chunk SxS state array is stored.

Double scalar-versus-WY output and entire carried state are compared at every chunk handoff with the fixed synthetic-fixture bound `max_abs_error <= 1e-10 * (1 + max_abs_reference)`. Analytic controls use that same fixture bound. The bound checks this finite normalized synthetic matrix identity; it is not an existing Strata tolerance, a captured/model acceptance gate, a permitted production arithmetic change, or a threshold to adopt a candidate. FP32 serial-versus-WY state/output differences and double/FP32 output differences are characterization only. No FP32 numerical acceptance threshold is introduced; finite/sentinel checks still apply. Independent literal controls constrain the common scalar reference rather than trusting WY agreement alone.

The frozen production source inspected is `sycl/src/prefill/kernels.dp.cpp`, SHA256 `3a5077a284048b4fde24a4146e3b02e8d84d259480f755e4b0e94cfc52e54f1c`. Its default pipelined column kernel maps `head%16`, uses `sycl::native::exp`, computes four32-row FMA projection partials and combines `red[0]+red[1]+red[2]+red[3]`, computes `(v-g*kv)*beta`, updates with FMA, computes four FMA output partials, then multiplies by `sycl::rsqrt(128)`. Software-pipelined loads preserve token order. CPU serial FP32 here uses standard exp/sqrt, ordinary noncontracted multiply/add, a single ordered K-row sum, and synthetic input casts. It is NOT a bitwise emulation of that kernel: exp/rsqrt approximations, FMA contraction and reduction grouping can each differ. WY additionally changes prefix-decay and triangular/matrix accumulation order. No CPU result clears any prior model failure or supplies GPU parity evidence.

PHYSICAL_COUNTS emits arithmetic only atT262144 for C1/8/16/32/64: number of sequential chunks, one FP32 state(48*128*128*4=3145728B), one-head WY scratch, one-chunk input and output bytes, and hypothetical all-chunk saved-state bytes. Scratch is processed one head at a time and reused; counts exclude guards, duplicate comparison buffers, allocator padding, production's existing T-sized GDN scratch, conv history, launches and library/backend requirements. They are not a device peak measurement or a demonstrated feasible/high-throughput GPU design. AtC16/32/64 the hypothetical all-chunk saved states alone are48/24/12GiB; that array is never allocated here.

Root-only recipe, unexecuted:

```
cmake -S "$WORACLE/sycl/tools/gdn-wy-equation-oracle" \
  -B "$ORACLE_BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ORACLE_BUILD" --target gdn_wy_equation_oracle --parallel 2
"$ORACLE_BUILD/gdn_wy_equation_oracle" \
  > "$PRIVATE_RUN/oracle.csv" 2> "$PRIVATE_RUN/oracle.stderr"
```

The target requires C++20, strict FP policy and no fast math; GNU/Clang/IntelLLVM use `-fno-fast-math -ffp-contract=off`, MSVC `/fp:strict`. Other compilers require explicit flag review. Root reviews actual compile flags to exclude config-specific/global overrides and source/binary/compiler identities, then runs under the shared serial lock with finite deadline/CPU/RSS/AS/FSize/NOFILE limits. Suggested wall120s, RSS/AS256MiB (adjust explicitly for sanitizer/loader needs), stdout1MiB/stderr1MiB. Fixed shapes keep peak working memory conservatively below128MiB and far below1GiB; only bounded current-chunk data/output and four current comparison states exist. No timing/statistical/performance samples are collected. Root may separately build sanitizer diagnostics; those do not become model parity or performance runs.

CSV: META; CASE header and100 rows (kind/geometry/T/C/initial/chunks/live_head_updates/double output+state maxabs/FP32 output+state maxabs/double-to-FP32 output deltas/analytic maxabs/status); NEGATIVE wrong-map rejected; PHYSICAL_COUNTS five rows; RESULT aggregate. Normal success requires exit0 and complete owner-verified outputs. A RESULT line alone does not establish successful final flush/process closure. Errors return1; stdout errors are detected after final flush. Parent owns authoritative closed receipts and all acceptance interpretation.

Read R33/35/36/81/82. R82 SHA256 `6e19978d99197a1bd91b0ea6630adf3046978a7e2b46693c6980478eb59f1006`. Preserve existing conv/session ownership, all-state/model/logits/logprob/IDs/MTP/repeat gates, original C/D/r5 failures, current P qualification status and runtime configuration. No production source/kernel/environment/GPU dispatcher was changed. No service v1/calibration or collector files were touched. This prototype does not request or imply a GPU implementation.

The causal query-key matrix `exp(G_i-G_j) q_i^T k_j` is computed once per head/chunk and reused across value columns. This avoids repeating the K-row dot product V times in the full-width CPU fixtures. PHYSICAL_COUNTS one-head scratch therefore includes two C-by-C matrices (L and causal QK), W[C,K], U/D[C,V] and G[C]. They are released/reused at each chunk; no matrices for all heads/chunks/tokens accumulate.
