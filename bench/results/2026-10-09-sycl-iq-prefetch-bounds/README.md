# Bound AVX2 IQ prefetch addresses

The IQ row kernels formed `block + distance` and a second hint 64 bytes later
without checking the matrix allocation. Non-faulting hardware prefetch does
not make out-of-array C++ pointer arithmetic valid. This source-only experiment
checks remaining bytes before forming either address. Gate/Up pass the complete
matrix bounds; other row paths use their assigned row range. Interior hints keep
the existing T0 policy and default distance 2048. Dot products, quantization,
float accumulation and dispatch are unchanged. Removing the bounds feature
reproduces both entire original sources byte for byte.

The environment parser now accepts a complete representable decimal integer;
invalid, overflowing and nonpositive distances disable prefetch. It does not
use atoi on overflowing input. Empty row ranges return before forming bounds.

The audit did not prove that the current resident arena crossed its backing
allocation or that this issue caused a GPU fault. The retained audit also has
native-Q8 producer/line-reference corrections in the shared report registry;
use source symbols when reviewing those claims.

This change has not been compiled or tested and is not adopted. Root will check
actual helper/parser boundary cases and native CPU row output parity, build all
115 engine objects with the qualified flags, and then run logged 32K correctness
and separate repeated decode timing. Physical 262144-cell/session qualification
is required before adoption. No performance improvement is claimed.


## CPU boundary validation

The actual parser and hint helper extracted from this source passed 24 separate
ASan/UBSan processes on the Ryzen 5600X. Across 745920 cases, the address oracle
checked 582238 hint targets with volatile reads. Logical bounds include empty
ranges, final bytes, page/cacheline boundaries and assigned-row ends, with
prefix/suffix sentinels outside the allowed range. Environment cases cover the
unset default, valid distances, negative/zero values, INT_MAX, overflow, huge
integers, leading zeros, spaces, plus signs and trailing junk. Every target was
strictly inside its supplied logical range; interior distance and T0 policy
were preserved. These are CPU helper checks, without GPU/model execution or
whole-kernel math/performance qualification.

See [CPU receipt](host-test-receipt-v1.json),
[actual-source harness](actual-hint-host-test-v1.cpp),
[controller](run-hint-host-test-v1.py) and
[independent source review](independent-source-review-round6.txt).
The independent report has a SHA spelling error in its header: the actual
rows.inl SHA is `290e479093fcdcd3ba162f448bc9f422a7d85efbcd1917a7f11423937adb43c5`.
The original report is preserved. No causal link to a GPU fault is established.
Full native row, 32K/session/full 256K and performance checks remain pending.
