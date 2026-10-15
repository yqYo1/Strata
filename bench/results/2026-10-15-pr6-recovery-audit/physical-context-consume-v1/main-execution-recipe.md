# Main-only bounded serial qualification (unexecuted)

Read README and pin source.diff/source-hashes.json first. Main must wait for the current fixed-source measurement to close, then apply into an isolated source/build using the actual configured SYCL include resolver/compiler/options and build the real generator. Do not apply bankv2 implicitly; source base is current production. Rebuild the generator and MTP header callers, verify compile/link identity, and preserve the source patch/binary SHA256 plus compiler/runtime/driver/hardware metadata. No CUDA/HIP qualification is requested.

Host fixture example (Main chooses a private owned output directory):

```
c++ -std=c++20 -I<PROPOSAL>/proposed/include <PROPOSAL>/host_fixture.cpp -o <OWNED>/host-consume
<OWNED>/host-consume
```

Its HOST_ONLY message is only a policy/coverage result, never a GPU/full-context pass. Actual CLI negative tests must separately check --consume-prompt with serve, vision, spec!=4, missing MTP/native, suffix default/nonzero, prefill0/prefill-until, split/peer/remote, token graph, KV streaming, noPLE, dump-logits/selected-logits/NLL and other refused diagnostics. Require admission exit2 before target-state consumption. Verify ordinary --max-new0 still rejects, ordinary legal generation remains unchanged, and ordinary input512+new1/context512 still rejects.

Prepare valid token-ID files with exactly512 and262144 IDs, preserving the tokenization/model identity and exact file SHA256. Use existing Main-owned supervised controller: one immutable pre-exec child environment per arm, owned PID/start/boot identity, finite wall deadline, <=64MiB total raw stdout/stderr, exclusive GPU measurement lock and independent fresh kernel-journal boundaries. The producer suppresses the full prompt list. Initial small512 diagnostic deadline can be180s; full262144 gets a predeclared finite budget appropriate to the admitted configuration (e.g.3600s), never an unbounded child. Budget stop is incomplete, not pass. No cleanup/build/test/GPU work in parallel.

Use the actual configured generator entrypoint and the existing known-valid native model/cache/MTP/PLE/KV configuration. The mandatory diagnostic arguments are:

```
<GENERATOR> <EXISTING_NATIVE_MODEL_AND_CACHE_ARGS> --tokens-file <INPUT512> --max-context 512 --consume-prompt --max-new 0 --spec 4 --mtp <MTP_RT> --suffix-draft 0 --prefill <CHUNK> --no-token-graph
```

Set CHUNK to an actually supported size such as128/512/8192 according to the established kernel route and available memory; record the initialized batched_chunk_capacity (request sizing may lower it). Pass real --native, the existing static expert profile/cache resources required for thits.d_res, enabled PLE and fully resident target KV. Rejects are not measurement passes. Leave layer-major/residual/QSA-batch options at the separately pinned comparison configuration; all options/environment must be immutable before exec.

Require native MTP4 use, exactly one terminal receipt, successful target T1 at511 using input[511], commit1/wait success, zero completion, no MTP draft or adaptation thread, target QSA12 origins508, PLEprev=input[-2:], full contiguous prefill[0,511) and successful MTP callback input-pair span separately. Require native/head/MTP attachment observations and subsequent normal owned process exit0, complete owner retirement and no fresh GPU fault. The receipt alone does not establish exit/driver health. Main must inspect checked vs unchecked teardown limitations; successful process exit is distinct from per-free status evidence.

Run the independent receipt checker after the child has closed (not in parallel with a measurement):

```
python3 <PROPOSAL>/verify_receipt.py --stdout <STDOUT> --tokens-file <INPUT512> --expected-count 512 --context 512 --exit-code <ACTUAL_STATUS>
```

After the small gate passes, repeat the same producer with exact input262144/max-context262144 and an admitted supported chunk, without increasing context capacity or output reservation. Expected prefill span[0,262143), terminal pos262143 using final *prompt* ID, QSA completed-block origins262140, completion0. Check actual buffers/configuration and all individual correctness/fault outcomes. The full physical result requires normal process exit and fresh health evidence; 262144 host fixture simulation or QSA component-only test does not qualify it.

Separate processes should exercise commit-async and STRATA_COMMIT_SYNC=1 while keeping their environments fixed. Run direct Prefill::draft_kv and its ordinary MTP fallback as separately configured valid arms if both are needed. The distinct MTP receipt is callback input coverage, not all-window retained KV. Exercise head/expert release only under its own admitted long-layer-major settings and independently verify existing restore helpers; this proposal does not add dense bank adoption.

Negative/fault gates: short/duplicate/gapped callback input rejected before MTP use, first-window identity/T/position mismatch rejected before ver.run, failed run/commit/wait or uncertain observer completion must not emit success. Use separate owned subprocess fault injection, preserve exact failure operation/error/progress and owner survival. Do not treat the signal name alone as success. No subsequent target window/draft/adapt launch may occur after a valid terminal commit.

Record compact receipts, input/output/source/binary hashes, effective environments, exit/fault results and each failed/incomplete arm. No full tensors or successful full API traces are required. External SHA256/input identity and FNV receipt consistency are complementary. Task-quality and real RESTORE-and-continue remain separate legal-context tests; a next target token at262144 is out of bounds, and this zero-output mode makes no continuation claim.
