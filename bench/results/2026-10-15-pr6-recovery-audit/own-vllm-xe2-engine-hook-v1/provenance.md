# R821 source provenance and review boundary

All newly authored hook/build/docs/metadata changes follow Strata's repository
MIT LICENSE, whose source SHA256 is
949b9b0aada7476006469e58ee4ffb0e0f68c2d9137e90a356757b61a09d37f0.
There is no copied third-party header/template in those new files. Existing
Strata interfaces, queue behavior, allocation registry and prefill call sites
are the implementation inputs for integration. No XeStrata source, helper,
test, diagnostics or archived proposal was read or used. No compiler/library
was invoked; no library contents were read or modified for this task.

## Exact imported component

The following files are byte-identical copies, with no adaptation in this task,
from committed `../own-vllm-xe2-native16-v1` (R810):

| Product path suffix under src/prefill/sycl/vllm_xe2 | SHA256 | License |
| --- | --- | --- |
| native16.hpp | eec1581b369759e4fd6c8ea7f2012cf9518bd4341c01e19362cc10b3553b066b | Strata MIT |
| native16.cpp | d8b361cabea29344b48aff5a9db41303f3ea715de47f27bc57f29f85c5f3aa2d | Strata MIT |
| xe_gemm_bsd.hpp | 114210ceb61f769d1ba7717690b43027a74f07cf7bdb0cb82d7ea3cd28dbd1c0 | Intel 2025 BSD-3 |

The R810 adapter's original standalone-only comments remain byte-identical;
R821 adds the actual production connection outside those files. Their original
math, shapes, type dispatch, guards, async API and tile policies are unchanged.
The Intel BSD notice in xe_gemm_bsd.hpp is retained in full and remains distinct
from MIT project files. Main must not relabel the derived tile MIT.

The actual core derives via R810 from official vLLM XPU kernels
6b30c9b92cc525c32729df988b5f918e7c5661a5,
`csrc/xpu/grouped_gemm/xe_2/gemm_xe2.hpp` lines 95-247, with original full-file
SHA256 359b1f04dfab3ad0850945b5240ec9b16be02b60fb7ab5bcc10d5cf254af99b4.
Its retained coordinate/block-copy/prefetch/reorder/DPAS/split-barrier/typed-store
body is the approved R810 adaptation. R821 has not reconstructed it from another
engine, copied a policy header or imported a grouped scheduler/Torch interface.
The package imports no R810 fixture or second Runtime implementation.

The implementation and license facts above were read from R810 README,
manifest, native16.hpp/cpp, CMakeLists, provenance and xe_gemm_bsd.hpp, exactly the
assigned local component. No external online research was needed or performed.
This is not a new upstream Niko observation, library survey or performance result.

## External dependency

Intel sycl-tla remains external at revision
87f6850680a580654b9ea2c80dbc01aeb36ad231. The existing path is
/home/yayoi/ghq/github.com/intel/sycl-tla/.worktree/strata-native16-87f.
R821 does not inspect, download, clone, update, compile or vendor that library.
`cmake/vllm_xe2_tla_headers.json` is an own metadata projection of R810's
`source_pins.tla_headers`, retaining path, exact hash and byte length for all
207 conservative transitive headers. CMake checks consumed header identity
without Git; it does not claim a checkout's unrelated source is verified.
The external closure's R810 license inventory is 192 BSD-3 and 15 Apache-2.0
WITH LLVM-exception files. Main retains the original external notices and
applies their distribution requirements, including Intel/NVIDIA notices and
Apache/LLVM terms. The production doc acknowledges both families; it does not
substitute one root label for every file's license.

R810's own CMake recipe supplies the exact relevant flag facts and defines;
R821 build glue is newly authored around production targets. It deliberately
links strata_sycl_runtime once. Its whole-engine backend GRF256 scope is a known
review concern; actual acceptance and image effects are not established.

## Base and information hygiene

All seven touched production files and inspected prefill.cpp match source pins
from own-native16-prefill-build-v2.json at
9c1a2020897a9451ec616a6bb036a661de9d86b0. The patch's before/after hashes are
recorded in manifest.json. No project Git command was used and no later HEAD
identity is asserted. No stale mirror was used to replace a live file. All
writes are inside own-vllm-xe2-engine-hook-v1.

research-index.json was consulted only for current metadata:
current_tuning_priority and native16_activation_pipeline, plus its key list.
No collection report body, R818 live code or R810/R816 test/build output was
opened. The source receipt was read only for revision/source identity. No
claims about Main's evolving active qualification result are made here.

Only required Orca lifecycle/check/heartbeat/completion communication was used;
no subagents, interagent steering, external messages, build, GPU job, test,
profiler, process inspection, syntax check, Git command or cleanup was run.
The package's SHA256SUMS is a source artifact seal, not a runtime validation.
Main's review and all execution remain pending.
