# GDN chunk64 fixture and experimental long-prefill route

The independently authored MIT implementation is in
`src/prefill/sycl/gdn_chunk64.cpp` and
`include/strata/prefill/gdn_chunk64.hpp`. This directory owns its maintained
fixture. `STRATA_SYCL_PREFILL_GDN_CHUNK64=1` enables it only inside native BF16
long layer-major prefill; the default, short-input and decode paths keep the
existing GDN implementation. The current model comparison establishes no
material speed gain, so leave the flag off for performance work.

## Storage and state contract

The loaded model has16 key groups,48 value heads, dimension128, conv4 and up to
8192 tokens per call. Preserve its existing tiled head order `head %16` and
FP32 state layout `[r,head,c]`. A source-grouped head `s=3*k+j` would map to
`l=16*j+k`, or `l=16*(s%3)+s/3`; this relation alone does not prove a converter's
weight-axis history. No weight/head permutation is performed here.

BF16 projection QKV goes through local FP32 convolution/SiLU, a BF16 round,
local FP32 Q/K normalization and direct BF16 storage. BF16 z feeds the final
gate. Gates/beta and persistent carry stay FP32. Shared conv history remains
FP32 `[10240,3]`, including the existing history supplied by short/decode work.
The recurrence uses BF16 activation/scratch and a full64 FP32 triangular solve
rounded once to a BF16 inverse. Its output is genuine FP16 for the existing
ssm_out projection, not BF16 bits reinterpreted as FP16. This schedule is not
qualified as the SGLang XPU block16 numerical contract.

The arena planner is shared by Prefill and the fixture. QKV storage is reused
only after convolution/history update for lower-KKT and a disjoint inverse.
Normalized input is dead before the carry history overwrites its storage;
W/U consumes vb before FP16 output overwrites vb. Only those two exact same-base
recurrence aliases are allowed. Partial/other aliases are rejected, and complete
preflight precedes any conv-history mutation. All lifetimes use one in-order
queue; allocations and DMA spans remain owned through checked retirement.

At8192, the shared arena uses1,148,718,848 bytes, fitting the existing
1,180,698,624-byte GDN reservation. It does not reduce that reserved model region.
The previous disjoint1,467,486,720-byte layout shrank the model chunk to6144 and
failed MoE-window admission; the original failed result remains in Git at
`305dcc0a`. These numbers exclude external state/history, weights, KV, residuals
and GEMM scratch.

## Verification and current decision

At `ed1a9693`, the producer/smoke/boundary/general checks passed for1/3,
63/64/65/127/128/129 and8192, including independent CPU conv formulas, guards,
input immutability, nonzero history, repeated and split calls, shared-versus-
disjoint equality and rejection before mutation. Conv formula differences were
zero on the bounded synthetic inputs. The dyadic fixture ignores only the sign
of zero in BF16 readout/FP16 output; every nonzero value still needs exact bits.
General FP64-formula errors remain diagnostics, not model-quality tolerances.
The full physical262144 model and RESTORE paths are unqualified.

The earlier8192 resident component used one warmup and five rotating samples per
arm. Median queue intervals were23.428125ms for the existing FP32 recurrence,
21.436146ms for the initial chunk64 schedule,21.097604ms with retained inverse
coefficients and18.266458ms with16-column carry slices. The last change requests
14 rather than28KiB carry SLM and holds16 rather than32 state values per workitem.
Its22.0% component reduction does not establish occupancy or whole-model gain.
The measured source `af6f7eda` and individual samples remain reachable through
result commit `ec97c3d3`.

A diagnostic39,260-input official bench.sh check at engine source `5c59fdd4`
completed with8192 chunks and no new fault. PP546.2/decode14.1 tok/s used tracing
and validation and is not a clean performance denominator. The subsequent clean
matched same-binary comparison used four78,451-input requests per process,
no prefix reuse, thinking on and24 generated tokens:

| Route | First PP, tok/s | Next three PP, tok/s | Median of next three |
| --- | ---: | --- | ---: |
| Existing GDN | 682.9 | 725.9,719.5,725.3 | 725.3 |
| Chunk64 | 677.4 | 726.8,719.6,725.8 | 725.8 |

The0.07% median difference is smaller than the observed request spread. Both
processes closed normally without fresh GPU faults. Short-output decode medians
were14.1 tok/s for both; this does not qualify sustained decode. The compact
`native-gdn-model-current.json` records every candidate sample and the pinned
Git baseline. Quality/framework equivalence, agent sessions, sustained thinking
output, full context/RESTORE and default adoption remain open.

## Rebuild and repeat

Commit exact source and recipes before building. Main serializes builds, tests,
model jobs, profiling and artifact retirement under the shared measurement lock.
The maintained owner/guards enforce finite budgets, immutable first-check
tracing/validation settings, runtime identities and checked process closure.

```sh
python3 bench/sycl/gdn_chunk64/qualify.py build
python3 bench/sycl/gdn_chunk64/qualify.py smoke
python3 bench/sycl/gdn_chunk64/qualify.py boundaries
python3 bench/sycl/gdn_chunk64/qualify.py general
python3 bench/sycl/gdn_chunk64/qualify.py producer-smoke
python3 bench/sycl/gdn_chunk64/qualify.py producer
python3 bench/sycl/gdn_chunk64/qualify.py producer-large
python3 bench/sycl/gdn_chunk64/qualify.py component-check
python3 bench/sycl/gdn_chunk64/qualify.py component
```

The helper refuses changed inputs or an existing generation; retire a closed,
superseded private build through its two-stage `retire` operation after verifying
its original source/recipe from Git. Current compact fixture evidence is
`gdn-chunk64-current.json`. Do not keep successful full traces/build transcripts.

`python3 tools/sycl/build_engine.py` builds the comparison engine after checking
committed project/dependency inputs. For a model check, use
`python3 tools/sycl/bench_engine.py prepare check`, commit the compact plan, then
`run check`; clean `base`/`candidate` modes follow the same sequence. Existing
run roots are never overwritten. The24-token cap is a prefill screen; use
[the full benchmark contract](../../../docs/BENCHMARKS.md) for adoption.
