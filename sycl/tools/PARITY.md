# Arc parity fixtures

The IQ and PLE tests need reference files. Missing files are test failures.
Use Python with numpy and the pinned llama.cpp checkout's `gguf-py` on
`PYTHONPATH`. No model download is needed when the Q2_0 shards already exist.

```sh
python tools/iq_fixture.py --out /tmp/strata-iq-fixtures --seed 7 --rows 32 --cols 256
cmake -S sycl -B build-sycl \
  -DSTRATA_IQ_FIXTURE_DIR=/tmp/strata-iq-fixtures \
  -DSTRATA_PLE_FIXTURE_DIR=/tmp/strata-ple-fixtures
cmake --build build-sycl --target ple_graph_oracle
python sycl/tools/ple_fixture.py --gguf /path/to/Q2_0-shard1.gguf \
  --oracle build-sycl/ple_graph_oracle --out /tmp/strata-ple-fixtures
STRATA_PLE_GGUF=/path/to/Q2_0-shard2.gguf \
  ctest --test-dir build-sycl --output-on-failure --timeout 180
```

PLE uses the original artifact's Q2_0 key, BF16 value, norms and convolution
weights. `ple_graph_oracle` follows the pinned ggml CPU graph with those
weights dequantized to F32, as the existing parity test requires. It records
both tokens' key, value, gate, gated value, normalization, convolution and
result. The GPU test retains its original bounds and allocation guards.

The generator's sparse `dense.bin` holds only the three regions the test
reads at its original oracle offsets. It is a diagnostic fixture, not a
model pack. `metadata.json` records the seed and source tensor and capture
hashes. `ple_graph_oracle` requires `STRATA_NATIVE_EXPERTS=ON`.
