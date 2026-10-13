# Main-only full-context diagnostic

`--full-main-context-probe` supplies exactly 262144 input IDs at position zero to production `Prefill::run`, with MTP disabled and zero generated tokens. Ordinary generation and serve admission are unchanged. Runtime qualification is pending; no speed or prediction-quality result is claimed.

Use a native pack and `--native`, exactly 262144 IDs in `--tokens-file`, `--max-context 262144 --max-new 0`, a positive fixed `--prefill-chunk`, `--no-prefill-borrow`, and enabled PLE. MTP/speculation, serving, batching, GPU splits, borrowing, elastic/growing storage and dumps are rejected. No terminal successor ID is supplied. The pending target configuration is int8, streaming mode 1, resident 32768.

Success requires gapless completed chunk coverage `[0,262144)`, exact PrefillStats input and chunk counts, and the final physical cell 262143. Main-only snapshot APIs save QSA authoritative KV/indexer, GDN/conv and PLE state. Reports distinguish full host-backed KV from bounded resident slots. Physical RAM admission happens before prefill and again before capture.

The diagnostic verifies all captured authoritative KV bytes, checks running/index floats and explicit quantization scales for finite values, poisons state, requires both KV and running-state negative checks, restores in place, and compares every KV payload/fingerprint and running checkpoint. Saved logical ID count is reported separately from processed input count. It performs no inference after restore and fail-stops if completion is uncertain. Packed q4 payload scales and fp16 KV values do not receive a separate numerical scan; payload equality still covers their bytes.

A PASS establishes this input-processing and state lifecycle diagnostic only. It does not establish prediction quality, ordinary full-capacity generation, or performance. `bench.sh` is unchanged. The host-only `full_context_probe_test` exercises gaps, overlaps, extent overflow, exact chunks and the final single cell. Main owns pending build, CPU tests and hardware qualification.
