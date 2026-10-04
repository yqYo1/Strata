# Upstream Arc integration on B570

Upstream `6f32ec070f23ced9f50e704d854d775da52591ab` (0.1.39), tested on an
Intel Arc B570 10 GB, Ryzen 5 5600X (6 cores/12 threads), 128 GB installed
RAM and Linux 7.0.0-38. Compiler: oneAPI DPC++ 2026.1.1. Selected compute
runtime: 26.35.39758.10, SYCL driver string `1.17.39758+10`. The pinned ggml
revision is `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

| Check | Result |
|---|---|
| JIT kernel/snapshot tests | 25/25, 32.92 s |
| BMG-G21 AOT kernel/snapshot tests | 25/25, 33.52 s |
| Setup tests | 6/6 |
| Native expert checks on real IQ3_S shard weights | All 48 layers pass, including AVX2 rows against ggml vec_dot |
| Captured vs eager split verifier, PCIe share 0.55 | First 248,320 logits bit identical, finite; generated IDs match |
| Genuine host USM staging vs the preceding DMA implementation | First logits bit identical |
| JIT vs AOT, same split verifier and staging | First logits bit identical; generated IDs match |
| Persistent requests and parked conversation restoration | Same four output IDs and logprobs; repeated/restored requests reuse 30 prompt tokens |

The upstream port required core API corrections to build against the same
revision's shared headers. Its grouped S2 loop contracted an inline product
that the old loop rounded separately; preserving that rounding passes the
original bit comparison. A snapshot fixture paired host USM allocation with
`free`; it now uses `sycl::free`. The IQ registration passed `--selftest` as
a fixture directory. PLE fixtures were absent, and its null-stream rejection
case had become a valid default queue during migration. Both tests now run
with their original reference checks and bounds. See
[fixture generation](../../../sycl/tools/PARITY.md).

The real model initially stopped at layer 1 because its CPU/GPU flag handshake
never completed. The verifier now captures the original mixer and MoE phases
separately and schedules CPU expert work between events. CPU plans and output
rows are explicitly copied to device buffers. The pageable resident expert
arena is reported accurately; a small genuine host USM buffer stages its PCIe
share. Streaming sources retain their original eligibility rules. The
one-token prompt's position-zero sentinel is also fixed. Device checks query
SYCL aspects and a real executable kernel bundle instead of CUDA compatibility
versions or placeholder kernel information.

The tested model is the local Qwen3.8-Flash-Next IQ3_S native pack and original
GGUF shards, with PLE and the existing MTP draft. Most diagnostic runs use
600 cache slots, five CPU pool workers, context 128, speculative window 4 and
an eight-slot prompt ring. Exact arguments, binary hashes, environment,
outputs and logs are in [run.json](run.json). PLE uses real artifact weights
with a CPU graph oracle and deterministic synthetic inputs. Its sparse
`dense.bin` is a test fixture, not a model pack.

These are correctness checks. Early model runs overlap compilation, and the
four/eight-token runs do not establish throughput. PP 1000 / TG 70 are still
optimization targets; this record makes no claim that they were reached.

Re-run the persistent check from the repository root with the selected oneAPI
and driver environment:

```sh
STRATA_PREFILL_RING=8 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  python3 bench/results/2026-10-05-sycl-upstream-arc/serve_check.py \
  --engine build-sycl-upstream-aot/strata --out /tmp/strata-arc-serve-check.json
```

The model root defaults to `~/.local/share/strata-sycl`; `--model-root` changes
it. The check enables a 256 MiB conversation cache and requires positive
`RESUME` counts on both repeated and restored requests. Its CPU-only fixture
generator and all 25 tests are described in the linked parity instructions.
