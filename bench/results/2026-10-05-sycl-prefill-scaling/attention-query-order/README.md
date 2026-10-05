# Attention workgroup query order on Arc B570

`STRATA_PREFILL_ATTN_LAYOUT=4` retains layout 3's subgroup 16, transposed
shared query tiles, 64-cell split and original merge. It exchanges the query
and KV-chunk workgroup dimensions so neighboring queries use the innermost
dimension. Selection indices and per-query scratch remain separate. The
floating-point operations and reduction order remain the same. The default
layout is still 0.

The standalone [study record](study/run.json) uses an Intel Arc B570 10 GB,
oneAPI DPC++ 2026.1 and compute runtime 26.35.39758.10. One immutable binary
runs layouts 3, 4, 4, 3, each with batch 128. The printed reference time is
for the original batch 32; compare the candidate columns across layouts
to isolate the workgroup order. After warming both paths, each case uses
three alternating timing rounds of five calls and reports their median.

All 32 cases have finite outputs and agree bit for bit with the original
batch-32 implementation. The independent FP64 reference error is unchanged.
They cover FP16, INT8, Q4_0 and hybrid K8V4; short context and an
identity-to-sparse selection transition; and 513 queries processed as four
128-query batches plus a one-query tail.

The two FP16 samples at context 8,192 decrease from 11.867/11.869 ms to
11.175/11.181 ms per 513-query chunk, a 5.81% reduction in the median.
This is a synthetic attention measurement. It does not measure overall
model prefill throughput. The [kernel](study/probe-kernel.cpp) and
[driver](study/probe-driver.cpp) retain the original arithmetic and reference
checks; source and binary hashes are in the record.

The integrated JIT build passes [all 30 registered tests](jit-ctest.log),
with no skipped tests. The new [query-order test](jit-query-order-parity.log)
checks all eight format/context cases using 129 queries, including the
one-query tail, against the original batch 32. The frozen model-test binary
has SHA-256 `de59fb937d2446c153b1bfe94a4e58c220fec9ee003afa86739da75973e1076c`.
