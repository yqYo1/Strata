# Real model check of the layer progress trace

On 2026-10-07, the layer-range trace build completed four real normal-MTP
requests through the owned GDB/PTY controller on this host's Arc B570 and
Ryzen 5 5600X. The process exited normally. No new xe fault/reset messages
were recorded, and neither the engine nor its debugger survived cleanup.

This was a correctness diagnostic with complete API tracing and validation,
not a throughput measurement. It does not establish 256K capacity or prevention
of the earlier intermittent submission stall.

## Configuration and results

The executable was `strata-prefill-layer-trace-candidate`, SHA-256
`b81a7d6fbc1c6d524cf3b64e196f5866c91ec56c76e460a66b63b28ce3aed461`.
It adds layer ranges to prefill's chunk-start messages. The preceding CLI
window trace remains in this build; neither trace changes model arithmetic.

The test used context 128, prefill chunks of 32, speculative length 4,
normal MTP, compact prefill 2, layer-major mode 2, draft-weight release and
verification, direct submission disabled, and V2 copy offload disabled.
The complete argv and selected runtime/tuning environment are in
`owned-layer-trace-short/record.json`.

For all four requests (`first`, `repeat`, `other`, `restored`), all four output
IDs, all printed logprobs and the complete 248,320-float finite head matched
the preceding released-weight control byte for byte. That control's executable
SHA-256 was
`79a4b363d33f66f41d910be6274e609b4eb73f62afb0dc49bca72bf2a538d223`.
The head hashes and reference record hash are preserved in the result.
Six draft-weight releases and six restores were recorded. All recorded
`equality` fields are true, exit code is zero, and all cleanup flags are false.

The caller supplied `diagnostic_environment()` before launch. It enabled
Level Zero API entry and successful return logging, parameter validation,
flushed UR tracing and Strata progress. The engine's stderr was separate
from GDB/MI and the PTY's serving protocol. The diagnostic took 169.73 seconds;
this elapsed time must not be compared to clean benchmark timings.

The complete 1,045,333,042-byte stderr remains in private state. Its SHA-256 is
`46e558a8017e9532b986508fdb8c0c92276b04a96dd35bb93772e6eb6967e589`.
`api-log.json` preserves the digest, 1,098,028 API entry lines, 948,177 successful
result lines, Strata progress and the final 64 KiB. Its 148,424 other logged
results were graph-capture queries returning `ZE_RESULT_QUERY_FALSE`
(`0x78000023`), not a device failure. Full head binaries and the complete host
kernel journal also remain private; only relevant xe/B570 journal rows are
archived here.

## Build and source receipts

`sources.json` maps the exact controller, runtime helper, debugger helper and
prefill source hashes recorded during this test to their archived copies.
The executable's build receipts are in `build-receipts/`.

The first three build attempts failed because of the compiler/toolchain
library search configuration. The successful archive/link used an unset
`LD_LIBRARY_PATH`, an explicit oneMKL root and compiler/MKL `LIBRARY_PATH`.
All 148 object digests stayed unchanged during that final archive/link; the
CPU native-pool task factor remained zero. The failed archiver reconfiguration
did change a ggml object and is recorded as such. These receipts do not claim
that all preceding failed attempts left every object unchanged.

`manifest.json` covers the archived files. Raw debugger prompts are retained
unchanged, including their original trailing spaces.
