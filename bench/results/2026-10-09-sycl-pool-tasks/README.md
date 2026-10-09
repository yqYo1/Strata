# CPU pool task-count option for SYCL

The common CPU pool already supports an explicit row-task count, but the SYCL
CLI did not pass one. This change wires `--pool-tasks N` to that existing argument.
It accepts 0 through 4096. Zero uses the existing automatic policy; positive
counts are capped by the rows in each batched Gate/Up and Down phase.

No task-count option still passes zero, as the previous constructor did through
its default argument. Removing the feature from `generate.cpp` reproduces the
entire original source byte for byte. Common pool kernels, quantizers, task
partitioning, synchronization and model state are unchanged. The main agent
performed this static review after the separate implementation agent finished.
The agent report is archived here; the agent ran no builds or tests.

This is an unqualified tuning option. No speed improvement or production
adoption is claimed. The main agent will build with the existing task-factor
setting zero, run input validation and numerical checks, then compare separate
32K cold and restored decode runs before physical 256K qualification. No GPU
work or builds run concurrently with another engine test. The old nonzero
CMake task-factor substitution is stale in this upstream version and is outside
this CLI change.
