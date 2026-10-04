# MMQ stream-K partition and fixup port

The routed product now implements the pinned GGML `mmq.cuh` stream-K work
partition and separate fixup operation, in addition to full-K tiling. This closes
that component's missing scheduling algorithm; it does not establish complete
engine or CUDA-device parity.

## Processing correspondence

For each workgroup, the continuous K-block interval starts at
`group * (tiles * blocks_per_row) / groups`. The start and stop are rounded down
within their tile to a `256 / quant_block_width` boundary, exactly as in pinned
`mul_mat_q`. A workgroup can have no work, part of one tile, several full tiles,
or a tile suffix followed by full tiles and a final prefix.

The tile index is decoded in upstream stream-K order: activation tile first,
then expert, then weight-row tile. Every segment reaching the end of its tile
writes the destination. A final segment ending inside a tile writes its
workgroup's private partial-sum slot. There are no atomic output additions.

The second kernel follows `mul_mat_q_stream_k_fixup`: the workgroup which wrote
a tile's final segment walks predecessor workgroups in reverse order, skips
empty intervals, adds their scratch values in that order, and adds the total to
the destination once. Empty/padded expert tiles are checked before reading scratch;
unwritten scratch for those tiles is never consumed.

The host planner reproduces `launch_mul_mat_q`'s NVIDIA 90% tile-efficiency rule:
full tiles when efficient, otherwise one group per supplied compute unit. The
B570 test passes SYCL's reported `max_compute_units=160`. This mapping is explicit;
it is not a claim that an Intel compute unit and a CUDA SM have equal throughput
or that this setting is optimal. The tile remains the documented Intel 16×64 tile.
A diagnostic override forces other group counts for validation.

`MmqPlan` reports scratch bytes. The caller supplies scratch, retaining it until
the returned event completes. The API queues the main kernel and then fixup on
an in-order queue and returns the final event. It allocates no scratch and waits
for no work internally; an engine context can retain/reuse the scratch as upstream
does. Plan/geometry consistency and the original `< 2^30` continuous-block-index
bound are checked before launching.

## Evidence

On Intel Arc B570, Linux, oneAPI 2026.1.1, Level Zero, 2026-10-04:

- The previous 95 product fixtures now run under five policies: full tiles,
  normal selection, and forced group counts 3, 7 and 257.
- 475 cases pass: 433,260 numerical comparisons, 935,825 output guard checks,
  unchanged input weights, and leading/trailing scratch guards for every case.
- 365 cases allocate scratch and launch the fixup kernel. Some have no actual
  partial tile because their K dimension is only one iteration; the directed
  checks below explicitly exercise nonempty partial sums and empty predecessors.
- Small fixtures compare every output numerically. The original five large
  fixtures (now 25 policy cases) check all active outputs for finite written
  values and sample numerical values across tiles/boundaries; they do not compare
  every large output numerically.
- Maximum observed error divided by the reference sum of absolute products is
  `1.19116e-7`, with the unchanged `3e-6 * L1 + 1e-6` acceptance bound.
- The quantizer's separate 546-case byte comparison also passes.

A directed Q8_0 fixture checks addition order exactly, rather than accepting any
result within that tolerance. Its four 256-element chunks contribute
`A, -A, 3, 4`, where `A = 127 * 2^25`. One, two or three groups yield exactly 7;
four, five, seven, 160 or 257 groups yield exactly 4 under the upstream reverse
fixup order. All eight results match exactly. The larger group counts also force
empty workgroup intervals between nonempty ones. Forward-order partial addition
would give 7 in the four-piece case, so this fixture distinguishes the order.

JIT and `bmg_g21` AOT source/binary hashes and logs are in
[mmq-stream-k-results.json](mmq-stream-k-results.json). Test runtime is correctness
validation, not a PP/TG benchmark.

## Remaining scope

The automatic CUDA tile-size/configuration choice has not been mapped to Intel;
this port retains its explicitly documented Intel tile. The full adapter/scratch
pool, gather/SwiGLU operations, optional MMQ formats, and engine runtime are still
being ported/audited. No CUDA device execution, CUDA fast-math/FTZ comparison, or
full-model benchmark has been performed for this new baseline. The original
upstream host/CPU source files and root build remain unchanged.
