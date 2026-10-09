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


## Revision 3 CPU qualification

`long_prompt_reference_v3.py` preserves the numerical rules and original 869
argv. It obtains the translation units from the recorded `ninja -t commands
strata` target graph: 115 compile rules and 114 unique sources. The CMake
compilation database contains only GGML entries and is not a project TU list.
The retained-object dependency superset includes 241 project and 52 GGML paths,
including both compiled `.inl` files, regardless of extension. Git blobs from
the recorded build commits are compared to current files. A generated GGML
version header is separately pinned for this run and checked again at exit;
this does not reconstruct historical compiler or system-header hashes.

Root's eight v3 admission fixtures passed in 4.272s, and the actual collector
CPU preflight passed in 15.022s. The latter compared 807 project and 1,362 GGML
conservative inputs. The 352 build-to-current changed paths are outside that
closure: 351 archived bench files and one server launcher. Revision 2's pure
fixtures passed but its real preflight failed before model execution because
the sparse compilation database omitted the project native source. Both
failures remain recorded; no earlier status has been upgraded.

Source, commands, receipts and the exact compiled-input proof are committed in
`bench/results/2026-10-10-long-prompt-admission/`. CPU admission does not qualify
PTY/model execution, establish the new long-input reference, or demonstrate
performance or candidate full-context correctness. The next root-owned model
run has a unique output path ending in `reference-r3`.
