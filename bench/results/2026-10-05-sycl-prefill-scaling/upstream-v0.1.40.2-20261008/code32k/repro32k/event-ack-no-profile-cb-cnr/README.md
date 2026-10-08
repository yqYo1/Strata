# Nonprofiling private copy queue: three32K state/head comparisons

The private binary4c184a9dfe0632ca1e79b8dfe4273b21f9eee0a420ee48ab1b28c07d57269a9b
retains the actual memcpy event for source reuse and creates an owned in-order
copy queue in the compute queue's context/device. Copy profiling is requested
only when STRATA_PREFILL_TRANSFER_TIMING is set. Compute queue properties and
GPU arithmetic are unchanged. Implicit counter conversion is disabled and
MKL_CBWR=AUTO remains set as in the previous production CNR control.

The logged first use and two fresh unlogged state/head processes each read the
same32,768-token code fixture with8192-token chunks, int8 KV and normal MTP4.
All three complete64finite-logprob outputs, exit normally and leave no owned
inferior/debugger. No new xe fault is recorded. All66prefill state parts,
all248,320first-head float bytes, all64IDs and every protocol logprob match
one another and the production CNR logged control. The [sequence](sequence.json)
passes both full comparisons. All timings are excluded from speed comparisons
because these runs capture full state/head and the first also logs API calls.

This is a narrow mechanism experiment, not an adopted queue implementation.
The [source audit](source-audit.json) identifies that this directly owned queue
is absent from device_ext::_queues: existing global waits and default-queue
sync_barrier no longer include it. Normal run/relayout/release have explicit
copy waits, but that does not validate every cache/graph retirement path.
drain_pipeline itself only joins successors. A separate full private rebuild
uses a registered copy-queue factory to preserve the global-wait contract.
Neither full262,144-cell serving nor stall prevention is proved here.

The initial source build fails because an initializer-list comma reaches the
single-argument DPCT_CHECK_ERROR macro. The second private build adds the needed
parentheses and passes compile/link and the identical CPU ring/lifetime suite,
including ASan/UBSan. The first controller preparer fails before writing either
GPU controller because it expects one build-receipt path occurrence, where
there are two; its v2 replaces both. Those executed versions are preserved.

Raw multi-GiB API logs and583,631,432-byte state files stay in private state
storage. Their byte counts and SHA256 digests are in private-artifacts.json.
The archived records include exact argv/environment, fixture and binary hashes,
full protocol output, pre/post kernel journal probes and ownership cleanup.
Production sources/objects/binary remain unchanged by the private builds.
