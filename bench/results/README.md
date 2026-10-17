# Measurement results

Follow [the retention policy](../../docs/ARTIFACT_RETENTION.md). Keep the current
comparison's useful measurements and decisions, not every intermediate artifact.
Maintained test/benchmark source belongs in `tests`, `bench/sycl` or `tools`.

For completed or superseded work, normal reachable Git history is sufficient.
Remove stale reports, rejected patches, source backups, generated receipts and
one-run controllers from the active tree. Do not preserve them by compressing,
bundling or moving them into more archive directories. Historical documentation
links should name the original Git revision.

Keep individual samples and validation limits for comparisons used by current
decisions. Successful full traces, repeated environment/maps, process lifecycle
records and routine compiler output are temporary diagnostics. A useful research
report needs a concrete retrieval reason; completion alone is insufficient.

Before adding a result, state which information a real consumer needs. Before
keeping an old file, ask whether restoring its original commit would suffice.
See the [2026-10-18 review](../../docs/ARTIFACT_RETENTION_AUDIT_20261018.md) for the
archive boundary and this cleanup's information-level decisions.
