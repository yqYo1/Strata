# Unused SYCL workspace and restoration memory checks

The SYCL prefill GEMM wrapper does not pass the migrated cuBLAS workspace
to oneMKL. Setting its owned workspace payload to zero removes 32 MiB;
the generic allocator's 256-byte padding remains. Standalone `Gemm::init`
and the shared attention/MoE workspace are unchanged. Trace-only memory
queries now record cache release, temporary allocations and release,
verifier graph recapture and MTP weight restoration. They report a failed
restore before the existing lease destructor retries it. Model arithmetic,
kernel geometry and expert residency settings are unchanged.

On 2026-10-07, the B570/Ryzen 5600X short normal-MTP check completed four
requests: first, repeat, different and restored. Every generated ID,
printed logprob and complete finite 248,320-float head matched the released
control. Six verified MTP weight release/restore pairs completed. The
candidate exited normally without forced cleanup, surviving children or
new xe fault/reset messages. It used context 128, chunk 32, compact mode 2,
layer-major mode 2, five CPU workers and phase synchronization. Direct
submission, persistent cache and V2 copy offload were disabled.

The logged owned-prefill allocation was 139,840,000 bytes, compared with
173,394,432 in the preceding typed-K/V candidate check: exactly 33,554,432
bytes less. Both logged 29,428,736 bytes of shared workspace. The private
candidate `3f3ed052…` replaced only the prefill object in its archive. Its
177.06-second diagnostic is not throughput. The full API log remains
private; its digest, entry/result counts, progress and final 64 KiB are
archived. The only recorded non-success API results are 148,424 unsupported
graph-capture queries, separately counted in `api-log.json`.

The normal build succeeded and produced `75b66b8f…`. The whole-executable
comparison with the private candidate failed because the embedded source
filename differs. Recompiling the private object with only file-path
remapping reproduced the normal executable exactly. All 147 indexed
executable sections and their canonical relocations then matched the normal
object. Before remapping, executable bytes differed only at 22 filename
length constants (108 versus 103); the string section matched after replacing
that filename. Whole object files still differ in metadata. The original
hash-assertion failures and the successful assessment are both retained.
This is an offline build check; an independent production-path GPU check
is not claimed.

Two small allocation-pressure probes also ran with full Level Zero/UR
logs and parameter checks. Each retained 8,832 MiB of device USM anchors,
released 1,280 MiB of virtual-memory physical backing, allocated and freed
1,300 MiB of temporary USM, and restored the backing in 8 MiB segments.
The temporary allocation was either a single buffer or twenty 64 MiB
buffers plus 20 MiB. Per-process resident VRAM returned to 9,047,364 KiB
after freeing it, then reached 10,358,084 KiB after successful restoration
in both cases. Every segment's test word matched. Both exited normally
with no new xe faults. These probes did **not** reproduce the full-model
restore failure; they omit model kernels and captured graphs. The optional
driver-cache-off and UR-pool-off cases were not run.

The installed NEO version's corresponding
[USM manager](https://github.com/intel/compute-runtime/blob/26.31.39395.14/shared/source/memory_manager/unified_memory_manager.cpp)
can retain eligible freed allocations, and ordinary USM allocation failure
can trim this cache before retrying. Its
[cache limit](https://github.com/intel/compute-runtime/blob/26.31.39395.14/shared/source/memory_manager/unified_memory_manager.h)
is 256 MiB per serviced allocation, excluding the single 1,300 MiB buffer.
The
[physical allocation method](https://github.com/intel/compute-runtime/blob/26.31.39395.14/level_zero/core/source/context/context.cpp)
does not perform that SVM-cache trim. This source audit motivates measuring
the actual restore points; it does not establish the full-model failure's
cause. Source URLs, SHA-256 digests and verified Git blob IDs are saved in
`allocator-source-audit/record.json`.

The earlier full-prefill run's 8 MiB MTP physical allocation failure is
recorded in the [phase-sync evidence](../phase-sync-capacity-20261007/README.md).
The new candidate's full-cell CLI and normal-MTP capacity checks remain
pending. Short parity and the pressure probes do not establish full-context
capacity, clean throughput or general GPU-stall prevention.
