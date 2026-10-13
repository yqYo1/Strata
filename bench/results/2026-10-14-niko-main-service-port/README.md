# Primary source audit and Niko feature adaptation

The base and sole upstream is Niko1221/Strata. This is a personal fork for the
Arc B570, Ryzen 5 5600X and 128 GB RAM. XeStrata is an optional benchmark,
architecture and component reference. No upstream contribution or CUDA/HIP
inference execution/implementation occurred.

See the [detailed Japanese report](README.ja.md) for the branch audit, feature
matrix, source changes, failed attempts and remaining qualification.

The primary is `feature/sycl`, starting at `36cc848a0eb68b96a1ed2ec007c5d86cfb3b13a7`.
Its version marker remains 0.1.38. Fork main is pinned at
`beddab2fd6b61631d73b39b7484bc9a585ebe1de` (0.1.39); these are functional source
imports from that existing tree, not a fresh upstream update or a full merge.
The remaining layer-major reference branch is Xe-derived, not primary.
Historical 667–694 token/s reference results do not measure current primary.

Main ported server/API/UI/setup/MCP/tokenizer/draft/profile functions, adapted
setup to native host paths/builds, and added the trailing-role checkpoint CLI.
Server tests: 268, seven skips, no failures with optional jsonschema available.
Setup and supporting tools: 295, four skips, no failures; individual subtests
are recorded. No installer, GPU, model or speed benchmark was executed.

Allocation teardown now fail-stops before free/unwind when checked retirement
fails. Graph-exec deletion occurs only after its checked wait succeeds. The
production cleanup helper and prompt checkpoint have normal and ASan/UBSan host
proof; CMake/CTest and SYCL Werror syntax checks passed. These do not establish
the historical hang cause, all runtime safety, physical 256K state or RESTORE.

The [inference hash comparison](inference-preservation.json) verifies existing
kernel/prefill/core/spec files remain unchanged except an `override` annotation.
There is no whole Xe engine adoption. Existing inference performance still needs
a fresh matched measurement.

The first 48 setup failures remain failed in their original receipts. Fresh
processes reproduced 47 on unchanged main: 46 Linux golden normalization cases
and a Windows fixture's executable name. Expectations were not regenerated;
fixture semantics were corrected. The remaining case required the MCP source
dependency. Guard, dependency-install, syntax and controller failures are also
preserved, rather than rewritten by the final successful checks.

Original JSON receipts are stored byte-identically in gzip with a
[hash index](receipts-index.json). Source/commands/environment, all individual
results, ownership/exit, short relevant diagnostics and research are committed.
The source/evidence commit is `51ad49e895c7dce75c0fb74b254255d504f22704`.
All 211 intermediate source pins were matched, including four saved exact source
variants and 12 controller recipes in `06da66fe052934be4d1f66ca755458edc905bf7a`.
After checking committed gzip against the originals, Main serially retired 12
closed job roots, 100 files and 7,509,335 logical bytes. The path/hash/reason
[manifest](deletion-manifest.json.gz) and [receipt](retention-results.json) retain
that action. Historical B paths in immutable receipts are now read through the
committed gzip/manifest. Source fixtures, owner, compiler and active test dependency
consumer remain available; closed CPU build/binaries have source/recipe/hash proof.
The test-only optional dependency target has a named follow-up consumer and a
32 MiB budget; the serving venv was unchanged.

Native BGEN/BGENI/BSTOP/BYIELD, elastic VRAM and remaining generator/build feature
adaptations are pending. Python mocks do not qualify those engine behaviors.
Default promotion and the next Niko update check wait for that reconciliation;
archival ancestry is not functional integration. Performance adoption remains
per component and separately for prefill/decode, using unchanged official
bench.sh/client and sufficiently long actual token counts.
