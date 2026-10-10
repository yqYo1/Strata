# Source handoff

Synthetic GDN component timing mode for32768-token carried prefixes and3..9 paired repetitions. Source-only unbuilt and unrun. Timer covers actual gates+conv+recurrence+norm submissions and queue completion; setup, warmup, input copies, readback/hash checks excluded. It measures component service under correctness-check cache conditions, not whole-model prefill, decode, or physical262144 capacity. Default noargs/host-only/prefix contracts require root requalification.

A separate gpt-6.1-sol implemented the source. Main owns all serial builds/tests, faults and final numerical/performance decisions. Research uses frozen copies and remains read-only. No test or benchmark was run for this source boundary.
