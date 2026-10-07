# Lazy verifier full-context check on 2026-10-07

The private lazy verifier candidate completes its first normal-MTP request
with 262,140 input and four output tokens on B570 10 GiB / Ryzen 5600X /
128 GiB RAM, oneAPI 2026.1.1 and the recorded NEO/Level Zero versions.
The windows [262139,1], [262140,4], [262141,3] execute through KV cell
262143. All four printed logprobs and the complete 248,320-float head are
finite. One verified MTP release/restore pair completes. The separate
eager comparison has not yet validated this full head.

The next 262,142-input request fails before the temporary layer allocation
checkpoint. Main-cache restoration in `CacheLease` destruction then fails
to allocate a 64 MiB physical segment and calls terminate. GDB captures
SIGABRT there. Both owned processes are removed during forced cleanup;
no new xe fault/reset is recorded. The following logged H2D/kernel/D2H
probe passes all three rounds of 16,384 exact words without a reset.

The [memory checkpoints](memory-checkpoints.txt) show 1,414,582,272 bytes
free before the first successful temporary allocation, which consumes
1,363,152,896 bytes. After the second main/MTP release, only
1,244,880,896 bytes are free. Source review supports an insufficient
whole-layer allocation budget, but the original return error is obscured
by failed destructor restoration. The specific allocation retaining the
extra memory is not identified. Lazy capture alone is insufficient for
the complete repeated-capacity test.

The clipped-tail, overflow-refusal and later-valid-request gates do not
complete. No diagnostic duration is accepted as clean throughput, and
the production verifier is unchanged. [Assessment](assessment.json),
terminal [record](full-run/record.json), GDB crash state, bounded log tails
and full private-log hashes preserve the distinction between first-request
success and suite failure. The [next eager controller](sources-used/run_full_eager_verifier_serve.py)
uses the existing eager-window setting and unchanged 3f executable;
its pending full result is not included here. Each manifest covers every
archived file except itself; binaries and full large logs stay private.
