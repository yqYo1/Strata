Layer-major K8V8 prefix reuse candidate v1, 2026-10-15

This source-only proposal skips repeated host-to-stage copies of a QSA layer's
already appended KV prefix during ascending layer-major prefill. It is disabled
by default. No root source or source pin was changed, and no compiler, build,
test (including host), GPU, process/environment/private-state inspection, Git
mutation or cleanup was executed. Main owns integration and serial validation
after its live profile closes.

`source.diff` is against the actual unchanged root. `source-manifest.json`
records every base/proposed SHA. The new production helper and host fixture are
also supplied as source for review; large existing sources appear only in the
diff. `artifact-sha256.json` records proposal and read-only input identities.
BF16 storage v2 is a separate unapplied proposal; both touch prefill sources,
the public header and CMake. Do not apply them together without deliberate
composition and new source pins. No BF16 change is included here.

`STRATA_PREFILL_LAYER_MAJOR_KV_REUSE=1` enables this candidate only for the
existing long-input, single-GPU SYCL layer-major route with uniform mode-1
K8V8, full-logical-range identity stage, and the same compute/copy queue. The
parser accepts only absent/0 (off) or exact 1. Unsupported flag values, short or
ordinary routes, full-resident/other KV formats, existing peer/split/vision
restrictions, missing stores and overlapping scratch are rejected before the
request mutates session state or submits work. With the flag absent/0, ordinary
chunk-major, short-input and full-resident behavior stays available. The flag is
read once in `Prefill::run` and passed down; there is no environment mutation or
per-layer/per-chunk flag read.

At the first chunk of each QSA layer, copy exactly the existing
`ceil(pos0/4)` full pages from that layer's authoritative host backing. For
pos0=0 this is a zero-byte seed decision, not a DMA. The existing append writes
each new K/V code and FP16 scale to the active full-context stage, authoritative
host backing and every resident slot exactly as before. Later chunks of the
same layer retain that stage prefix and omit the redundant copy. There is no
quantization, arithmetic, indexer, RoPE, GDN, attention or residual dtype change.
All twelve host/resident mirrors and allocations remain live. This introduces
no layer commit, host spill, decode restore protocol or capacity saving.

The new request-local `LayerMajorKvPrefixReuse` is the actual production
scheduling helper. It checks the expected QSA layer/index, ascending exact
chunks, final partial chunk, physical-context arithmetic bounds, first-page
seed extent and completed layer sequence. It describes same-in-order-queue
submissions, not a GPU completion or session-validity marker. `Impl` borrows it
only inside the existing checked outer lease. After each layer, the existing
run/pass drain and callback cursor must succeed before the helper permits the
next QSA owner. Restore removes the pointer before the local object dies. Any
new call seeds anew, including after cancellation, an error or a restored
session; no persistent stage generation is trusted.

Partial first pages retain the previous behavior: the seed copies a whole page,
then append overwrites every newly valid cell. Later partial pages remain in
place while append fills their new cells. Attention's valid positions and
identity table are unchanged. The bound 262144 is preserved; host planning and
this source proof do not establish physical maximum-context correctness.

The successful opt-in request emits one compact summary: seed decisions,
reused chunks, copied/avoided page counts and logical copy bytes. A K8V8 page
is 4224 bytes for 4 cells, 2 KV heads, K+V codes and FP16 scales. These are logical
counts of submitted paths, not elapsed time, measured transfer efficiency,
capacity recovery or throughput. No kernel instrumentation, new synchronization
or asynchronous/double-queue path is added.

Why the actual final-layer callbacks preserve the active stage:

| Actual source | Write ownership and proposed protection |
| --- | --- |
| `Prefill::carve` in `src/prefill/prefill.cpp` | Allocates `m.region` for operator scratch, then later calls `take_stage`. The stage is after scratch in the parent allocation, or owns separate allocations under the existing stage-own option. It is not the QSA/MoE shared operator scratch. |
| Candidate admission in `layer_major.inc` | Checks complete K8V8 K/V code and scale spans, plus identity-table span, against `m.region` using checked address arithmetic. This rejects an overlap before work. No new GPU allocation is required. |
| `Prefill::draft_kv` | Its temporary arrays are carved exclusively from `m.region`; GEMM scratch/workspace was carved before the operator/stage regions or has separate owners. It appends to the drafter's own `QsaState` and host backing; it never passes `m.stage` to its append. Existing checked completion/host-upload leases remain unchanged. |
| `MtpDrafter::prefill` in `src/core/mtp.cpp` | The fallback uses drafter-owned Rin/token/step/position buffers, graph pointers and KV state on its own queue, and checks completion before returning. It receives residual rows, not the target stage pointer. It does not borrow `m.stage`. |
| CLI callback at `generate.cpp` around 6435; serve callback around 4961 | Both use `draft_kv` with its own-pass fallback. Optional selected-head logits use a local `ss.block` copy and head scratch; checkpoints save canonical session/host state. None receives or writes the target's private `m.stage`. |
| Last target layer 47 | Its external callback executes between chunks, but all the above scratch/owner paths are separate from the retained stage. The next layer 47 chunk therefore sees its prior K/V prefix. This remains a required runtime comparison gate. |
| Reentrant layout/state operations | While this candidate borrows the helper, public run/init/relayout/set_peer calls reject; reset/set_stage throw before changes. This prevents callback reentry or layout rebinding from invalidating the stage. `set_stage` moves from an inline body to a checked out-of-line definition. All guards are inactive outside this opt-in lease. |

The original object's lifetime/thread-safety contract still applies; concurrent
mutation or destruction of the same Prefill object during its callback is not
an authorized access path. The candidate does not export any new stage pointer.
All actual `m.stage` uses in the current prefill source were reviewed: carve,
QSA seed/append and QSA attention. The separately named `m.stage_dev` expert
ring is a different owner.

The configured resolver remains `cmake/sycl_backend.cmake`, which compiles
`prefill.cpp`; it includes `layer_major.inc` after the helper definitions. The
new header is included by that translation unit. Root CMake's existing
`STRATA_BUILD_PREFILL_HOST_TESTS` option additionally registers
`prefill_layer_major_kv_prefix_reuse_test`. There is no CUDA/HIP implementation
or validation in this candidate.

The host fixture uses the production helper, not a second implementation of
its scheduling policy. It covers strict parsing, first seed versus reuse,
monotonic chunks, duplicate/wrong layer/reentry, partial first/final pages,
incomplete/full layer boundaries, a fresh call after a partial request, checked
span overlap/overflow, and planned 262144 extents with chunks 1/4096/8192. An
opaque encoded-cell simulation compares every valid retained-stage cell against
the old copy-every-chunk control across all twelve layer changes and a nonzero
partial-page start. This is host scheduling evidence only when Main executes
it; it does not replace actual K8 code/scale/attention checks.

Remaining Main gates, none executed here:

1. Recheck actual base hashes, review the callback/layout guards and integrate
   this candidate alone with new source pins. Compile/run the new host fixture
   and existing orchestration fixtures, then the configured SYCL build. Do not
   use an old receipt's pinned compiler/source fields for the changed binary.
2. Run matched flag 0/flag 1 component and model checks with identical K8V8
   mode 1, stage-own choice, residual format/placement, chunks, tokens and state.
   Require exact stage/host/resident code and scale bytes, indexer/page state,
   sampled residuals, full selected logits and target/decode state. Exercise all
   twelve QSA layers, more than two chunks, odd chunk/tail sizes and starts at
   0/1/3/4/5, including a nonzero restored prefix.
3. Specifically compare layer 47 between-chunk callbacks with batched MTP, its
   own-pass fallback and selected-head/checkpoint observers. Check the opt-in
   summary matches expected prefix-copy counts and that the stage is unchanged
   by callback scratch work. No timing qualification follows from counts alone.
4. Exercise unsupported-route rejection, attempted callback reentry/rebinding,
   stop/cancel after seed and reused chunks, failed seed copies/queue drains,
   and subsequent reset/RESTORE. The scheduler must not survive a call, and no
   failed or partially mutated baseline session becomes a successful one.
   Existing lease fail-stop behavior must retain owners on unprovable drains.
5. Check a physically admitted 262144-context final row and decode/RESTORE
   handoff. Then obtain clean matched repeated long-input prefill timings,
   decode/regression checks, actual memory/runtime identities and fresh driver
   fault closure. No precision, quality, capacity, RESTORE, maximum-context or
   speed qualification is claimed here; 600/1000 tok/s remain goals.

This is the first exact dataflow step toward GPU-local long prefill, not a
replacement for the separate full weight/KV/hidden residency design in
R678/R679. Retaining mirrors intentionally avoids that design's new commit and
restore protocol; it also retains its existing allocation costs. End-to-end
measurement must decide whether the removed prefix transfers matter.

Retention: no run artifacts were produced. Keep this compact diff, helper,
fixture, commands/gates and identities until Main's next idle review. Future
measurements must retain individual matched outcomes and fault closure. Retire
successful full traces/raw tensors only after their named comparator records
verified compact evidence, under the measurement lock with path/hash/reason
manifests. No cleanup is performed during Main's live profile.
