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

## Root correction and CPU validation

Root corrected cancelled/non-normal-finish requests before execution, then ran
the portable fixture under AddressSanitizer/UndefinedBehaviorSanitizer: v1
passed in1.105s. The independent Luna review identified uncaught invalid-env
exceptions and cancellation being labelled as a diagnostic error. Root now
handles the strict parser's runtime_error locally in main before GPU setup
(source returns2), and reports request completeness separately from dispatch
failure. The fixture checks normal cancellation gives complete0/error0 and
genuine failures give complete0. Root also added independent duplicate-route
and split-callback accounting examples.

After these source changes, v2 passed in1.056s with normal compile/fixture exits0,
no forced cleanup and no GPU/model access. Both original receipts and recipes
are preserved. The compiler, flags, exact source/binary hashes and sanitizer
environment are in those receipts. This fixture validates the accumulator and
bounds, not production main/callbacks or controller parsing. The actual engine
has not been built or run, and the executable's invalid-env exit still needs
its production check after that build. No timing neutrality or adoption is
claimed. Sol is separately preparing the owned controller/parser; root alone
will qualify it and run the model.
