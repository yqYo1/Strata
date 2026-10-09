# Cache pair diagnostic source review

Separate Sol6.1 implementation from frozen H96bd5. Opt-in host counters only;
no cache policy or tensor arithmetic change. Root reviewed all five changed
files and the completed report. No build, test or GPU run has executed this
source. Source-only commit is not adoption or qualification.

The reported pair callback count includes hits and CPU routes and is not the
native CPU-job counter. Exact DONE deltas and unchanged host residency must
reconcile before a complete frame. A cancellation currently needs an explicit
request-failure flag at the reporting call; root will correct that before
execution. The original Sol v1 source/report remains in Git for review.

Root alone schedules tests under the shared measurement lock after all owned
GPU work is closed. Full model math, physical262144 lifecycle and clean repeated
performance gates remain separate.
