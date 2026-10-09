# Exact static prefix-quota count oracle

The fixed profile order can have increasing captured-count marginals. Greedy
slot assignment can miss the optimum. This pure CPU module uses exact dynamic
programming under an explicit total of at most128 uniform MAXBLOB slots, then
breaks score ties by minimum L1 distance from the supplied default and by
lexicographic order. It does not read files or change the inference engine.

The separate gpt-6.1-sol implementation agent wrote source only. Root reviewed
it and ran eight CPU tests under the shared exclusive measurement lock. Tiny
exhaustive enumeration is independent of the DP recurrence and covers
nonconcave prefixes, ties and all feasible small default vectors. Literal tests
include the greedy19-versus-optimum109 example, strict malformed/uint64-overflow
rejection, zero/full budgets, and maximal48-layer geometry with hand-calculated
answers. All tests passed with exit0 in.353s, with no cleanup or survivor.

Source SHA2566025cbef7718d3f26f086f7de03cd65370b6edf4dce511c434357bc6c79f0220;
fixture SHA2560554c59c6f69940db5316103786213c3e9ce635105108b92b98a3c3933bcd8cb.
Original root receipt SHA2567325fb8e6d1495e9db50785e78e81a71247a73f38216e7ddc3a7d7051fd896a1.

This proves the bounded arithmetic selector, not captured-route preparation or
a live policy. Counts are not service time, traffic or predicted latency. Any
later route-to-prefix adapter needs pinned data/provenance and a frozen
objective; live cache changes additionally need separate32K/full262144 math,
phase-specific timing, repeated clean processes and independent holdout inputs.
The original C/D full-context failures and all adoption gates remain unchanged.
