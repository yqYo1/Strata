# Visible speculative commits and the full-capacity gate

All performance comparisons use at least32,768 input tokens. The earlier
[quiet processing-order comparison](../kv-streaming-32k-and-layer-order-v3/README.md)
separates first requests from repeated full reads. These new logged/validated
runs are correctness checks and contribute no performance numbers.

On the same ArcB57010GiB / Ryzen5600X /128GiB host, the direct-pointer-assignment
candidate matches all64 output IDs/logprobs, all248320 first-head float bytes,
all66 live prefill-state parts and MTP43/66 of the accepted32K control. Its
SAVE then reports32833 consumed tokens. The saved IDs contain the complete
32768-token prompt, all64 displayed outputs and an additional invisible token.
The correct consumed prefix ends before the final displayed output, at32831.

The serial serve loop commits every accepted verifier row before applying
max_new or EOS to output. The pipeline serve and CLI speculative loops have
the same ordering. A private source candidate clips the accepted prefix to
the remaining output count and the first EOS before commit. VerifierT,
numerical rows, kernels and queues stay unchanged. Twelve CPU cases with
ASan/UBSan cover output limits, EOS and the reproduced final four-row window.
Source, object/link receipts and the complete candidate diff are retained.

The fixed private executable passes the initial32K GPU gate: complete
head/used state/64 IDs/logprobs remain identical. The offered-draft count
remains66; visible accepted drafts become41 because the two invisible rows
are no longer committed or counted. SAVE now contains exactly32831 IDs,
equal to prompt plus outputs except the last. A32832-token continuation
reports RESUME32831 and generates64 further tokens. See the immutable
[initial snapshot](initial32k-v4-snapshot.json), whose parent process continues
the capacity test; it does not claim whole-process exit or cleanup.

The first harness configuration used the default turn boundary with PC1,
splitting the final32K chunk at32761+6. It kept output IDs but changed the
state/head/logprobs, so that comparison was rejected and exited normally.
The corrected configuration disables turn/root boundaries and checkpoints
only after the existing262139-token final full-input chunk. A later harness
assumed an optional MTP-count field in an old record and failed before GEN.
Its corrected fixture/reference checks run on CPU before GPU startup. These
terminal negatives and owned cleanup are retained; neither is reported as
a xe fault or as a valid performance result. Read-only health checks pass.

The session reader hashes every tensor byte. Only LRU stamps, file offsets
and derived checksums are omitted from semantic equality. Synthetic tests
detect a changed last physical KV byte, truncation and invalid counts. The
reader's initial trailer assumption is preserved alongside the correction:
the actual codec writes payload checksum before end magic. The corrected
reader parses/hashes the actual saved file; engine RESTORE must validate
its original checksum and compatibility before changing device state.

The full-capacity controller is running separately. It requires two fresh
262140-input/4-output reads through physical cell262143, interposed32K input
and RESUME0 on each full read, all13 saved main/MTP KV images through rounded
cell262143, real32K live-state restoration, full-prefix restoration and a
clipped two-row tail, capacity refusals, and a later complete32K control.
Every input is at least32768 tokens. Diagnostic timing is excluded from speed.
Full262144 occupancy/repeat/restore has not yet passed and the candidate is
not adopted. Production c88f94d and accepted e82fc5 binaries are unchanged.

The pure unpatched upstream32K baseline remains in its earlier archive.
No reset, rebind, reboot, service, package or global setting is changed.
Large API/state/session payloads and binaries remain private with hashes.
The captured compiled overlay, not current worktree source, is authoritative.
