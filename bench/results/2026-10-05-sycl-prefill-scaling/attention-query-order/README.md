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

The first [resident model comparison](resident-initial/summary.json) fixes
batch 128, cache 128, context/chunk 8,192, top-k tuning and GDN tuning for
both layouts. Each length has one checked warmup and one measured request.
All eight requests reread the full prefix, resume at zero and process one
chunk; their complete finite 248,320 first logits and generated IDs match
the accepted reference. Expert bytes are identical between layouts at each
length: 45,402,470,400 at 4,096 and 46,989,286,400 at 8,087.

| Input tokens | Layout 3 attention interval | Layout 4 attention interval | Layout 3 resident prompt wall | Layout 4 resident prompt wall |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 882.25 ms | 820.64 ms | 14,663.7 ms | 14,616.5 ms |
| 8,087 | 1,996.63 ms | 1,884.28 ms | 18,246.3 ms | 18,147.4 ms |

These single samples have transfer and phase markers enabled. The attention
interval includes kernel-launch gaps; resident prompt wall also includes
request setup. They support the observed attention change but do not
establish a repeated end-to-end speed improvement. Warmups are retained in
the raw records and excluded from this table.

The [normal CLI confirmation](paired-cli-wall/summary.json) alternates layouts
3/4, 4/3, 3/4 and reverses the input-length order for the middle pair. Both
paths use the same frozen JIT binary, batch 128, cache 128, context/chunk
8,192, top-k tuning and GDN tuning. Phase/transfer/preload markers are off;
model loading and generation are outside the prefill timer. No compilation
or other GPU work overlaps these runs. Persistent SYCL caching is on, with
no extra CLI warmup; the first run is not claimed to have every module
prewarmed.

| Input tokens | Layout 3 median wall | Layout 4 median wall | Layout 3 overall prefill | Layout 4 overall prefill | Median wall reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 13,498.4 ms | 13,442.9 ms | 303.44 tok/s | 304.70 tok/s | 0.41% |
| 8,087 | 17,034.7 ms | 16,924.2 ms | 474.74 tok/s | 477.84 tok/s | 0.65% |

Each median uses three runs. Between the two lengths, the median wall
increase is 3,536.3 ms for layout 3 and 3,481.3 ms for layout 4: 1,128.58
and 1,146.41 additional tokens/s. These incremental rates are not overall
prefill throughput. All twelve complete finite first heads and generated
IDs match the accepted baseline bit for bit, with exactly one prompt chunk.

The [normal MTP short-prompt check](normal-mtp-serve/run.json) passes first,
repeat, other-input and restored-checkpoint requests. All return IDs
`[760, 10849, 88851, 2272]`; repeat/restored logprobs agree exactly. This
server check measures correctness, not a speed improvement.

The [long normal MTP check](long-mtp-serve/summary.json) repeats this test for
both layouts with a 1,025-token input. The first prompt is processed as
1,018 checkpoint tokens followed by six tokens, with total prefill 1,024.
The first complete finite head matches bit for bit between layouts (SHA-256
`95435da8b4d1c0ee4f84ff295b5b4ee5d02f791b233d19dd0bd2ebb969d66e2e`).
All eight requests return IDs `[40, 3172, 1151, 539]` with equal logprobs.
Repeat/restored requests resume at checkpoint 1,018 and read seven tokens.
