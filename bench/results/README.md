# Measurement records

For the SYCL implementation, commit reviewed measurement summaries (`run.json`),
documentation and verification source code. A summary should identify hardware,
software, model, settings, measured samples and validation scope.

Keep generated execution logs, per-trial JSON, comparison CSV files, profiling
captures, binary dumps and workstation configs outside Git. Changing a log's
extension to `.txt` does not make it source. The SYCL patterns in `.gitignore`
exclude these captures; prompt token fixtures remain available to verification
code. Do not bypass these patterns with `git add -f`.

The original local SYCL captures from 2026-10-02 and 2026-10-03 are preserved at
`~/.local/state/strata-sycl/measurement-archive/6ddaadd/`. They retain their
repository-relative paths. Removing them from tracking does not delete those
local copies. Summaries record measured results; the raw captures are local
supporting material rather than repository content.
