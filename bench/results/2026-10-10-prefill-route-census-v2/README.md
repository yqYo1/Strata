# Default-off prefill route census v2 — limited main qualification

Private Sol v2 changes only the host header. The reviewed v1 caller/diff is
byte-identical and retained at the canonical paths in source/identities.json.
No production tree was edited, no optimization adopted and no model run made.

The main agent compiled the exact v2 header with g++ -O2 -Wall -Wextra -Werror,
then ran nine fresh CPU cases: absent env, exact0, bad env, four-chunk resident
generic, threaded mixed stream_all with an unrouted copy, MMQ with no generic
products, duplicate call, resident copy, and a copy outside the admitted layer.
Every case passed its predeclared receipt/hash/count or rejection gate. This
is selected host accounting coverage, not all cap/identity/fault combinations
and not a real >=32K model execution. The v1 rejected compile is unchanged.

The main agent also compiled the actual modified SYCL caller using the precise
O3/subgroup32/per-kernel/sequential-MKL production recipe, pinned frozen includes,
and an exact v2 header overlay. Quoted relative includes resolve from the frozen
original prefill directory. Command/environment/dependencies/object hash and
normal owned-session closure are in the TU receipt. No link, GPU execution,
model correctness/default-off parity, actual route joins or full262144 inference
was performed. The external object is retained for the next linked candidate
qualification consumer; it is not committed. All builds/tests were serialized.
