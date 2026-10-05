# Production RAM PLE correctness on the B570

`controller.py --proof-only` runs the existing production `--ple-io ram`
option on the validated AOT binary. It processes 8,087 input tokens at
chunk 4,096, compact HC scratch, forced layer traversal, ring 16 and the
first 4K residual rows in VRAM. All 248,320 finite head floats, every
residual row and all persistent-state bytes match the previous original
and AOT proofs. The [summary](summary.json) preserves all comparison hashes.
This proof uses the same explicitly selected Q2_0 table shard as those
initial AOT proof cases; future long tests use the original native-model
shard, matching their controls. The three engine source hashes match the
[AOT source snapshot](../aot-validation/initial/source/). The profiling
helper's updated source is saved here.

The full table is loaded at startup. `mlock` fails because the session's
memlock limit is smaller than the table, so the upstream implementation
touches every page. The log explicitly reports loaded, not locked.
Startup table loading takes 42.3 seconds. The entire process takes about
95.0 seconds. Sampled peak engine RAM is 74.104 GiB, VRAM is 7.176 GiB,
and no process swap is observed. Sampling is at 1 Hz and can miss peaks.
A separate embedding server remains resident on this GPU; this is a
correctness proof, not an exclusive-GPU performance comparison.

The [subsequent long-input measurements](../isolated-long-validation/README.md)
run after the service pause and record warm startup costs. Do not use
this proof's state-download timing as production throughput or exclude
startup loading without reporting it.
