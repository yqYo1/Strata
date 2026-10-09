# CPU GDN/WY equation oracle qualification

Root reviewed the separate source-only Sol oracle and the independent Luna R85
review, then built it with GNU g++13.3.0 using no fast math and FP contraction off.
Release and ASan+UBSan variants each passed100 deterministic synthetic cases,
including partial chunks, zero/nonzero/carry states and all48 value heads with
head%16 key mapping. The deliberately wrong head/3 mapping is distinguished.
All per-case output/state errors and guard results are in all100-case-results.csv;
identical stdout from both binaries is stored once and linked by receipt hashes.

Maximum double output/state absolute differences were9.992007e-16 and
1.554312e-15; maximum FP32 characterization differences were5.960464e-7 and
9.834766e-7. The double1e-10(1+maximum reference) bound belongs only to this
synthetic equation identity. FP32 differences are recorded, not accepted as a
new production tolerance. The program does not emulate SYCL native exp/rsqrt,
FMA or four-partial GPU reduction order, convolution, post-norm/FP16, model
heads/logprobs/MTP or throughput. It is not model or GPU parity evidence.

The physical262144 rows are checked arithmetic workspace counts, not a
full-context allocation or lifecycle run. The WY algorithm never materializes
all chunk states in this CPU program; full-model 262144 validation remains a
separate mandatory gate before production adoption.

The first controller failed the sanitizer compile because its2MiB per-file
size limit killed cc1plus while emitting sanitized assembly. The failure receipt
and exact compiler error remain. A new controller allowed32MiB only during
compilation; unchanged runtime limits, inputs, exact source hashes and math
bounds then passed both variants in21.014s. ASan execution needs a large sparse
shadow mapping; that stage has512MiB polled resident-memory, CPU/wall/output
bounds and fixed-size inputs while its virtual-address cap is unlimited. All
owned children exited normally, with no cleanup or survivors. No GPU runtime or
model was opened. The root supervisors, actual compiler commands/imports,
source hashes and failed/success receipts are committed. No derived binary or
tensor is copied into Git.
