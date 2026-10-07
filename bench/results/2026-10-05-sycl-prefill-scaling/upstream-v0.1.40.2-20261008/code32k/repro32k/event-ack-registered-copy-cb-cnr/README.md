# Registered nonprofiling copy queue: matched32K state/head controls

On Arc B57010GiB / Ryzen5 5600X /128GiB RAM, kernel7.0.0-38,
NEO26.31.39395.14, oneAPI2026.1.1, the private binary
f02213fc440a1f11b64857fceaa028ab43eeee7fdfb40e5cec33abca411a9705
retains actual DMA events until source reuse and requests profiling for its
registered in-order copy queue only when transfer timing is requested.
Compute queue properties, GPU arithmetic and explicit release waits remain
unchanged. The factory inserts into device_ext::_queues, so device-wide waits
and sync_barrier on the default queue still include this copy queue. This
avoids the global-wait gap in the earlier directly owned queue experiment.

All translation units are rebuilt against the same changed header. The final
configure argv matches the production build, including the C/C++ compilers,
IQ2_S GCC groups, shared pinned ggml checkout and precise floating-point flags.
The [matched-build receipt](build/event-ack-registered-copy-v01402-matched-build/record.json)
compares the selected Strata/GGML/compiler settings with no differences and
verifies production source/header/binary hashes are unchanged. The completion
header is byte-identical to the previously sanitized CPU ring/lifetime suite.

The first logged check and two fresh unlogged state/head captures all use the
same32,768-token fixture,8192-token chunks, int8 KV and normal MTP4.
All three complete64finite-logprob outputs, normal exit and owned cleanup;
no new xe fault is recorded. All66prefill state parts, all248,320first-head
float bytes, all64IDs and every protocol logprob match one another and the
production CNR logged control. The [sequence](sequence.json) passes both
comparisons. Implicit counter conversion remains disabled and MKL_CBWR=AUTO
remains set as in that control; these settings alone previously failed.

These are equality checks, not clean timing jobs. State/head dumping is enabled
in all three, and the first also captures UR/Level Zero diagnostics. No speed
claim is based on these times. A stable32K comparison does not prove general
stall prevention, default-environment reproduction or full262,144-cell serving.
The candidate remains private and unadopted while those gates are open.

The first full source archive omitted third_party/ggml headers and fails before
linking. The second includes those headers but leaves CMake's C compiler at
system cc, which rejects -fp-model=precise. Both complete failures are preserved.
The third selects icx and completes the full rebuild, but its default IQ2_S
GCC option is off; before any GPU use, the configuration-matching controller
preserves that binary then reruns the exact production configure argv and
incrementally rebuilds. Its snapshot hash remains in the matching receipt.
Only the matched build is used by the GPU controllers. Earlier generated
v2/v3 model controllers are not executed. No package/service/reset/rebind or
reboot change is made by these controllers.

Private multi-GiB API logs,583,631,432-byte state files, compiler logs over1MiB
and the earlier binary snapshot have byte counts and SHA256 hashes in
private-artifacts.json. Public records retain exact argv/environment, fixture
and binary hashes, full protocol output, kernel journal and ownership cleanup.
Raw excerpts retain their original bytes. Manifests verify archived contents.
