# Long-prompt numerical reference collector

`long_prompt_reference_v1.py` prepares an independent numerical reference on
the qualified 869 baseline binary for the complete frozen 36,004/37,462-token
inputs. It uses the original argv: that binary has no `--pool-tasks` option.
It cannot provide a matched performance comparison with the task-6 candidate.
Only root runs it, with the shared measurement lock and bounded supervision.

Root's nine independent pure rule fixtures passed on 2026-10-10 in 0.103s.
Receipt `long-prompt-reference-rules-cpu-validation-v1/record.json`, SHA-256
`6ad0c044946ffcb77071ae4f9f203448dd89e54dea2f5d675ee3271fd65717df`.
They check variable input endpoints, early stop, malformed protocol/counters,
IDs/all LP candidates, repeated numerical identity and capture geometry.
They do not qualify the collector's PTY, debugger or model execution.

The actual CPU preflight failed before GPU/model/output launch in 0.454s:
`baseline clean exact HEAD`. Its pinned build commit is `1eb89482`, while the
clean worktree now has HEAD `23268953`. Receipt
`long-prompt-reference-cpu-preflight-v1/record.json`, SHA-256
`fe55dd547d351e5a8487e5422b7a1d099a6abd0c481bcc121cdee0cc8fe73033`.
Both receipts are under the shared `post-reboot-tuning-20261007` directory.
The original source and failure remain preserved. A later collector must
resolve source identity with an exact reviewed build-input closure, rather
than waive provenance or alter the baseline worktree to satisfy a HEAD check.

The separate Luna review also identifies setup after output-directory creation
outside the finalization handler. Protect that setup in the next collector
so failure has a terminal receipt and cannot strand an unexplained directory.
The collector additionally names a SYCL override for `native_expert.cpp` that
is absent in the actual source tree; use the actual resolved production path
when verifying the build inputs.

No new long-input numerical reference, matched speed measurement or candidate
full-262144 lifecycle has been established. Existing C/D rejections remain
unchanged. The next revision and all tests/model runs remain root-owned.
