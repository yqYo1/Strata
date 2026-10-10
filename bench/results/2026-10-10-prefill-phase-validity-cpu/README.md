# Fail-closed opt-in prefill phase telemetry

Existing timing could silently omit failed profiling queries or unsigned-wrap reversed endpoints. This diagnostic now validates queue admission, completion, endpoint order and marker/query reconciliation, suppressing all totals on invalid samples. Unknown completion or asynchronous failure exits without unwinding model storage. No extra per-mark waits; source outside PfTimer is byte-identical to7d0105 and disabled timing performs no new queue/event operations.

Root-owned GPU-free actual-source extraction contract passed all normal, equal-endpoint, failure, fatal no-unwind and disabled/SYNC cases. One compiler unused-argc warning is preserved. Production SYCL compilation and actual-device model validation are pending; this is not an adopted optimization or clean performance result.
