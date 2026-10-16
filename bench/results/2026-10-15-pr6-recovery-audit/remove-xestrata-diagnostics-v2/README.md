# Remove provenance-unknown GU observer after v1

Source-only supplementary removal. Fixed base is commit
`0f816385a0b2ed4cc856202ebd9d7c4e53a44a63` overlaid by the exact v1 proposed
hashes/deletions embedded in `pins.json`. Modified inputs were read from hash-checked
v1 proposal copies, other inputs from `git show`; live source was not used or edited.
The v1 originals were not changed. R717 records the independent origin as unknown;
this patch removes the observer rather than claiming an acceptable origin.

Deletes `include/strata/core/cpu_gu_reuse_observer.hpp` and
`tests/cpu/cpu_gu_reuse_observer.cpp`, including every type, field and method in them.
Removes CpuGuReuseObserver declaration, ExpertDispatch::cpu_gu_reuse, observer include,
gu_observer pointer, gu_job_expert mapping, gu_completed_rows counter, touch/completed
hooks and diagnostic-only envelope/blob checks. Actual expert identity/group creation,
activation/job assignment, native dispatch and v1's independent exception fail-stop
are unchanged. Pool parking/completion and MTP code are untouched. The GU diagnostic
has no replacement. CMake/tools/src/include/tests search of the immutable base found
no other active consumer; historical bench consumers are Main's separate retirement.
No build, test, model, GPU, fixture, environment mutation or service operation occurred.
No compilation, functional, safety or performance success is claimed.

## Temporary consumers and permanent record

`after-v1.diff` contains deleted source verbatim because unified deletion needs it.
It and `temporary-proposed/` are temporary Main application/review consumers and must
not be committed as permanent artifacts. Do not blindly copy proposals onto live files.
After Main applies/reviews and records the results, remove both using Main's serialized
artifact-retention procedure. Record actual path/hash/reason in its deletion manifest.
Original deleted code remains available through the immutable base commit and paths;
no redundant deleted-file mirrors were created.

Keep only README.md, pins.json and prepare.py as compact permanent artifacts here,
plus Main's compact application/validation/deletion records. pins.json records each
modified/deleted input hash, proposed hash or deletion, and the complete v1 base overlay.
The patch SHA identifies the temporary consumer after its deletion. prepare.py regenerates
from v1 temporary proposals while available; optionally pass a reachable post-v1 commit
as its sole argument, whose modified source blobs must match pinned post-v1 hashes.
For future regeneration Main may preserve that exact post-v1 tree as a reachable commit;
this recipe rejects mismatched inputs and never reads live product source.

## Main recipe (not executed here)

At a serialized boundary compare pins against the actual post-v1 source; adapt any
concurrent changes by full patch review. Check/apply after-v1.diff only against that
reviewed base. Search active source for CpuGuReuseObserver, cpu_gu_reuse, gu_observer,
gu_job_expert, gu_completed_rows and the removed header/test names. Historical reports
must not be counted as active caller failures or as fresh validation.
Compile configured SYCL source resolver/entrypoint and CPU objects; run Main's existing
independent CPU pool/native-layout, lifetime/drain and model checks, with short/long
input and MTP/coupled completion/failure cases as applicable. Preserve the independent
native-call abort before borrowed owners unwind. Record new outcomes and all limits;
old diagnostics measurements do not validate this removal or imply a speed improvement.
