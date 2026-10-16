# Main-only V3 combined-bank serial qualification (not executed)

Main first reviews source-review.md, merge-provenance.json and retained-bank-invariants.json. The sole patch base is the exact immutable ../prefill-native-dense-phase-v2/proposed overlay, pinned by its source-hashes.json; do not derive a new base from live production. Never apply physical-context-consume-v2/source.diff wholesale. Assemble bank-v2's proposed files first and overlay V3/proposed, or use a reachable Main commit after matching every pinned relevant base hash. V3 source.diff is incremental from that bank state, not from pre-bank production. Match the audited external SYCL facade/runtime/kv_stream hashes too; stop and rebase explicitly if they differ. Wait for active pinned builds/measurements to close before Main applies anything. Build the actual configured SYCL source resolver and all ABI callers into a new owned directory; verify no bank source/guard/callsite was removed. All build/test/GPU management is Main-only.

Compile/run host_fixture.cpp with proposed/include only in a Main-owned directory. Its HOST_ONLY outcome validates actual helper policy/coverage/counter arithmetic, never a GPU/full-context result. Negative actual CLI gates must separately preserve ordinary --max-new0 rejection and ordinary prompt+new reservation, reject target ring/reused maps/invalid resident values and competing selected-logit/NLL/verify-logit/NaN diagnostics, and reject serve/vision/split/peer/remote/partial-prefill/suffix-default routes before consuming target state. Actual mode is checked after normal initialization, not inferred solely from the flags.

Use the existing Main supervised finite-deadline controller with one immutable pre-exec environment per child, exclusive measurement lock, source/binary/compiler/driver/hardware identity, exact token-file SHA256, PID/start/boot and normal/abnormal exit, owner retirement and fresh kernel-journal boundaries. Bound raw stdout+stderr to64MiB; no full API trace or tensor dump is needed. No build/test/cleanup/GPU in parallel. Budget-triggered stops are incomplete.

Use actual configured generator and known-valid native model/static expert profile/cache/MTP/PLE settings (thits.d_res required). Three sequential gates:

1. Exact512 input/context, --kv-resident32768, --consume-prompt --max-new0 --spec4 --mtpPATH --suffix-draft0 --prefill<supported chunk> --no-token-graph. This is ACTUAL MODE0 because512cells fit; record that honestly. Example deadline180s. Expected terminal511/QSA block origins508, no output/draft/adaptation. The same producer is used, but this is not the streamed gate.
2. Genuine small streamed input32768/context32768, --kv-resident20480 and otherwise valid pinned configuration. Expect actual mode1, resident20480 per QSA, real map call0→1/overflow0 and exact one selected prior host/slot packed block comparison per layer. A finite deadline such as600s must be declared before launch. Do not replace this with512 and pretend streaming occurred.
3. Required physical target: exact input262144, --max-context262144 --kv-resident32768, same explicit consume shape. Expect twelve actual mode1 layers,32768 resident cells/65536 logical pages, target terminal pos262143, QSA block origins262140, commit1/wait success and output0. Declare a finite full gate deadline such as3600s. Never request context262145, force full resident KV or add a separate full KV owner to make this pass.

Exact command spelling uses the existing CLI's spaces, for example:

```
<GENERATOR> <VALID_NATIVE_CACHE_ARGS> --tokens-file <INPUT> --max-context 262144 --kv-resident 32768 --consume-prompt --max-new 0 --spec 4 --mtp <RT> --suffix-draft 0 --prefill <CHUNK> --no-token-graph
```

Keep ordinary streaming Prefill staging settings, residual dtype, QSA dispatch and model flags pinned; do not change environment after provider init. Existing one-layer Prefill KV staging is unchanged, not an added full target KV pool. MTP's separate draft ring/state and callback input coverage do not establish target state positions. The receipt does not certify all selected blocks or numerical correctness.

After each child closes, run the independent checker serially. It requires explicit expected ACTUAL mode/resident cells:

```
python3 <V3>/verify_receipt.py --stdout <OUT512> --tokens-file <IN512> --expected-count 512 --context 512 --exit-code <STATUS> --kv-resident 32768 --expect-kv-mode 0 --expect-resident-cells 512
python3 <V3>/verify_receipt.py --stdout <OUT32768> --tokens-file <IN32768> --expected-count 32768 --context 32768 --exit-code <STATUS> --kv-resident 20480 --expect-kv-mode 1 --expect-resident-cells 20480
python3 <V3>/verify_receipt.py --stdout <OUT262144> --tokens-file <IN262144> --expected-count 262144 --context 262144 --exit-code <STATUS> --kv-resident 32768 --expect-kv-mode 1 --expect-resident-cells 32768
```

STRATA_KV_MAIN_OFF or any actual resident fallback must fail the mode1 checker, even with correct input/context and a zero-output receipt. Check actual resolver calls/misses/lookups/overflow, selected prior probe block<terminal page, forward/reverse slot agreement and exact format payload bytes (8192F16/4224INT8/2304Q4). These bytes are checked in the real producer; the checker only validates their receipt and independent input identity. Normal process exit0, owner retirement and absence of fresh GPU faults are independent mandatory gates. Existing manual cleanup free-status handling remains its current limitation.

Separate owned processes should test asynchronous commit and STRATA_COMMIT_SYNC=1, and direct/fallback MTP callback paths if required. Fault/negative cases include reused map counters, invalid/overflowed resolver state, missing host backing, malformed sampled map, actual host/slot mismatch, failed commit/wait/copy and callback coverage errors. No uncertain completion may emit a success receipt or resume decode; preserve operation-specific diagnostic excerpts rather than classifying by signal alone. No adaptation thread/draft/next target window/position is allowed after the terminal commit.

Keep compact receipts plus all individual outcomes, source/binary/environment/input hashes and fault/exit receipts. Task quality, full-state numerical parity and legal-context RESTORE-and-continue remain separate tests; this zero-output producer cannot legally consume another target row at262144.


Combined bank arms must retain the existing startup opt-in before exec. First qualify consume with bank switch0/unset, then separate immutable-environment bank1 processes on an admitted long layer-major/MMQ0/native compact route. Keep STRATA_PREFILL_RELEASE_HEADS unset/0: bank-v2 still explicitly rejects head-release1, so do not infer combined support from a separate head-only run. Record actual bank activation eligibility (N−1>chunk) and require the source-preserved outer wrapper to restore Full owners before the terminal verifier. Schema3 additionally requires native_dense_residency0 (Full) and native_dense_overlay_rows0; these observations alone do not prove bank activation or physical memory recovery. The real physical32768-residency gate cannot be replaced by full resident KV or by a successful512 resident fallback. Bank snapshot payload and existing streamed Prefill staging retain their own resource costs; the observer adds no GPU allocation.
