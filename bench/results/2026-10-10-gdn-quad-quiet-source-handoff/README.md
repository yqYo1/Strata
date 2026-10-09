# Quiet GDN component timing source handoff

A separate gpt-6.1-sol added opt-in >=32768 paired carried-prefix service timing to the parity fixture. Root reviewed the complete report and source diff; Luna R102 independently found no endpoint-lifetime or paired-state/teardown defect. This commit is source only: new mode has not been built or run.

Kernel arithmetic, headers, CMake/FP flags, selector and defaults are unchanged. Each sample sums host steady-clock intervals including per-chunk admission, direct recurrence, norm and completion wait; transfers/readbacks/checks/digests are untimed. It is not a whole-model, default-selector or kernel-only timing. Both warmup orders and alternating measured orders are recorded, with full bitwise paired state/output checks after each chunk.

Root corrects the recipe path and DPCT check() wording; that helper is a no-op. wait_and_throw() is the actual completion/error boundary. Original Sol report is preserved unchanged. Root owns all serialized build/runtime/statistical qualification.
