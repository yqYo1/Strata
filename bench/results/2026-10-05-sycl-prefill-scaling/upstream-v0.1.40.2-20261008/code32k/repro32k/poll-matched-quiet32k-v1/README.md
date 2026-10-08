# Matched 32K quiet comparison of host polling backoff

Ryzen 5 5600X, 128 GiB RAM and Arc B570 with 10 GiB VRAM; kernel 7.0.0-38, NEO 26.31.39395.14 and oneAPI 2026.1.1. Both binaries use the integrated v0.1.40.2 engine with context capacity 262144, chunk 8192, int8 KV, 32768 resident tokens, cache 128, five CPU workers and MTP 4. The candidate only changes the prefill host wait after 32 failed event readiness checks to request a 10 us sleep. The [prior numerical and source/link gate](../device-profile-rejection-and-poll32k-v1/README.md) describes the exact build.

Run order was DD5 control, polling candidate, polling candidate, DD5 control. Each fresh process read A/B/A/B, every input 32768 tokens and each output 64 tokens. All sixteen reads reported RESUME 0 and REUSED 0, and every output ID, logprob and visible MTP count matched the completed DD5 gate. All processes exited through normal QUIT, with no forced cleanup, surviving owned process or new kernel fault. Runtime validation logs, profiler/preload, state/head/payload dumps and extra phase waits were disabled; model/debugger protocol records were retained. No other model, GPU health test, build or heavy local analysis ran concurrently.

Each first process read is reported separately from its later full reads. It includes process JIT/capture warmup and is not a cold disk/system measurement. A later first use of input B belongs to the already-running process and is retained as its own read slot in summary.json. Rates below use total tokens divided by total engine time, not the arithmetic mean of rates.

| Reads | Count | Control PP tok/s | Poll PP tok/s | PP change | Control TG tok/s | Poll TG tok/s | TG change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First process read | 2 per mode | 393.713 | 390.224 | -0.886% | 16.200 | 16.369 | +1.042% |
| Subsequent full reads | 6 per mode | 427.510 | 427.225 | -0.067% | 16.486 | 16.373 | -0.686% |

This small sample shows no useful prefill gain from the polling change. The raw decode range and per-read-slot results are retained in summary.json; two processes per mode do not establish a decode benefit or regression. CPU utilization and GPU/PCIe overlap were not measured. This is not evidence that transfer waits dominate, that the pending native counter problem is fixed, or that the performance target is reached. The candidate is not adopted.

The complete physical 256K sequence still has to pass before adoption: all 262144 cells, repeated fresh full prefill, actual disk restoration, clipped speculative tail, capacity refusal, then a valid fresh 32K input. Earlier incomplete or instrumented runs remain excluded from these speed figures. Main and production binaries were unchanged; no reset, rebind, reboot, service, package or global setting was changed.

The exact terminal receipts, actual argument/environment identities, raw protocol/MI/engine logs, input token files, supervisor and executed versioned controllers are included. Archived paths retain the original machine paths; the manifest identifies the copied files. Historical controller comments describing the diagnostic ancestor are not evidence of instrumentation: the quiet controller's explicit environment checks and terminal receipts describe this execution.
