# Mixed IQ3_S validation and tuning on B570

`run.json` is the reviewed summary of model provenance, kernel checks,
CPU-reference checks, parameter screens, three complete paired cache-layout
measurements and real HTTP validation. Hardware, flags, input fixtures and
output hashes identify the scope. It includes all 19 persistent-process samples;
slower parameter settings remain in the record.

These measurements predate the upstream sync/rebase. The CPU-reference label
`sycl_cache_off` was corrected to `sycl_cache_auto`: `--expert-cache 0` with a
profile selected automatic sizing and allocated 2,530 slots. The numbers were
not changed. A true cache-off check and post-rebase validation are recorded in
the [upstream integration summary](../2026-10-03-sycl-upstream-rebase/README.md).

The implemented formats are IQ2_S, IQ3_XXS and IQ3_S; this model uses all three
alongside existing Q2_0, IQ4_NL and IQ4_XS expert paths. Source commits are
`e21c48d` (mixed format support) and `8d3b3f0` (per-layer slot sizing).
Independent checks live in `tests/sycl/mmvq.cpp` and
`tests/sycl/host_engine.cpp`; `tools/sycl/llama_logits.cpp` drives the pinned
CPU reference. The dump comparators in `tools/sycl/` check finite inputs and
exactness at the stated scopes.

Decode rates exclude startup and prefill. Repeated requests reuse prompt tokens
and retain the expert cache. Background workloads were active. These are short
fixtures on one B570/5600X workstation, not broad quality scores or a matched
comparison against another GPU. The storage optimization's bitwise check fixes
placement; automatic placement is a separate configuration and can change
continuations.

Generated logs, trial JSON, float dumps, model files and workstation configs are
not repository artifacts. Local captures and run-specific drivers are preserved
at `~/.local/state/strata-sycl/measurement-archive/2026-10-03-iq3_s/`.
They use the recorded local data and temporary paths; update those paths when
reproducing on another machine. Follow the
[measurement storage policy](../README.md).

See the [detailed implementation report](../../../docs/SYCL_STATUS_2026-10-03.md)
for the selected preset, startup command, measured results and upstream gaps.
