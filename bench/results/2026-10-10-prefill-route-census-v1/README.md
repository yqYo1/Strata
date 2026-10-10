# Default-off prefill route census v1 — rejected CPU admission

The Sol source-only handoff was reviewed, then the main agent attempted its
predeclared CPU contract with g++ -O2 -Wall -Wextra -Werror. The exact unmodified
header failed compilation normally: an enum/non-enum conditional in executed().
No test case or SYCL/GPU/model execution occurred. The closed failed receipt and
complete 699-byte compiler error are preserved. The empty compile stdout is
represented exactly by its size/hash in the receipt, rather than copied here.

Static review additionally found that IDs within max64/max512 but outside the
request layer range or actual expert population can be written yet excluded from
end_chunk folding. This is a source finding, not an executed reproducer. The
root-owned driver predeclares outside-layer-copy as an invalid-receipt case.
A private v2 fix was assigned to the same separate Sol implementer. Main owns
all subsequent CPU/SYCL/GPU qualification. Original handoff and failed result
remain unchanged; no census or performance optimization has been adopted.
