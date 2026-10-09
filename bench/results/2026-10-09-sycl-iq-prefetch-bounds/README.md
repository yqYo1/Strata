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
