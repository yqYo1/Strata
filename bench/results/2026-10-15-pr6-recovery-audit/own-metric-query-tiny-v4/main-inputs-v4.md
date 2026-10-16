# Main additive controller correction v4

Main retains the exact v3 C++ status patch, collector identity, status/CSV/report-validity readers, catalog schema and numerical scope byte-identically. The actual v5 collector was built with the reviewed v3 package; Main final source/build seals for v4 explicitly retain that actual build context and prove the patch/header bytes are unchanged. No rebuild or relabel of the actual candidate is performed.

The original frozen manifest stores some map values as {realpath,bytes,sha256} and others with an additional path field. V3 incorrectly compares those dictionaries with full artifact pins by raw dictionary equality. V4 accepts only that serialization difference: exact map-key path, optional existing path, realpath, byte count and SHA256 must match; unknown/missing keys or identity differences hold. The original local pin remains in place. There is no new runtime ELF enrollment or byte refresh.

The controller canonical package and fresh output directory are v4. All other v3 source/build/catalog/health/native ownership, immutable environment, finite budget, exact API and compromised-report gates remain required. Historical failures remain unchanged. Main tests the changed host pin equivalence before any native execution.
