# Optional fused HC scratch proposal

2026-10-14 JST. Implementation artifact only. **Unapplied, unbuilt, untested, default off.** Main owns every source application, build, test and GPU run after the current modelpair closes. This artifact does not establish numerical equality, allocation success, VRAM savings, speed or adoption.

The proposal changes only `src/prefill/prefill.cpp`. Exact `STRATA_PREFILL_HC_SCRATCH_COMPACT=1` moves `lo`, `lo16`, `gated` and the conditional HC `lo16_lo` into `max(HC,GDN,QSA,MoE)` scratch. It shares the outer `emb` take with `bo`. `grs`, `xn16`, conditional `xn16_lo`, `inj`, `mixed`, `mixed_bf`, `mixed_h` and conditional `mixed_bf_lo` remain outer, distinct takes. QSA `q` and `attn` remain distinct. No run-loop math, dispatch, server, CPU, queue, metadata, expert staging or KV ownership changes are proposed.

## Identity and provenance

Main supplied native identity `6361e706` and primary-Niko origin; these are context supplied by Main, not a newly inspected binary or Git receipt. The source was read directly from the actual current tree. No Git command, source pin, branch or worktree operation was performed.

| Artifact | SHA256 |
| --- | --- |
| Original `src/prefill/prefill.cpp` | `484764fccef4cf58909bfe27fb01934da19dbffaca7f7be2565424d36ab86578` |
| Derived source, in memory only | `105c59f9999b333ca86d49c29173fc31c380448253c895073340e7587cf0d916` |
| Unified patch | `336d41211501a1fbc0fc694c17b5c896a7a61884268fa8f44886d482b06f18bf` |

Files beside this report:

- `fused-hc-scratch.patch`: unified diff against the original SHA above.
- `identity.json`: source/patch identities, status, consumer and review point.
- `byte-plan.json`: allocator arithmetic for capacity T=1,17,4096,8192 and BF16X2 modes 0,1,2; these are calculated plans, not executed samples.
- `generate.py`: deterministic generation recipe. It reads only the product source, refuses a different source hash, derives the candidate in memory, and writes only the patch and small proposal records. It does not persist a full candidate or apply a patch.

Generation command, from the workspace root:

```sh
python3 bench/results/2026-10-14-primary-compact-moe-recovery/fused-hc-proposal/generate.py
```

Inputs reviewed: project `strata-upstream-sync` skill, `docs/ARTIFACT_RETENTION.md`, the pre-Xe model-compare registry and its recorded corrections, current recovery R508 and R509, and R510 at `/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/research-20261009/research-fused-hc-compact-restoration-round510-v1.txt`. Current source, the SYCL prefill postops/GEMM, launch queue contract, native embedding gather wrapper, model geometry and Prefill callback/reset API were read for the lifetime proof. The forbidden record, HF_TOKEN, broad environment dumps and private run logs were not accessed. All writes are under this proposal directory; no product source, build, test, GPU work, cleanup or Git mutation was performed.

## Admission and fallback

`hc_scratch_plan` is shared by `carve` and `bytes_needed`. Only an exact string `1` requests activation. Absent, `0`, `01`, `true` and other values use the original allocation sequence. A present but non-exact value reports `requested=0`, `effective=0`, reason `value-not-exact-1`.

Effective requests require a SYCL build, fused GR, BF16X2 mode 0/1/2, and all current fixed ModelGeometry dimensions: N=2560, HC=4, LR=320, K=10, 48 layers, QSA interval 4, 512 experts, FF=640, current GDN and QSA/indexer dimensions. Other backends, `STRATA_GR_UNFUSED=1`, invalid BF16X2 modes, differing geometry or an unrepresentable HC byte plan decline with a reason. Nothing forces GR_UNFUSED on or off.

Before any new HC product is formed, the plan checks T against the padded sum bound for HC plus the shared `bo` take in both `size_t` and `uint64_t`, and against `INT64_MAX / D`. The code introduces no INT32 launch-grid restriction. Existing SYCL GEMM/postop dimension and span validation remains untouched; this proposal does not repair pre-existing malformed-input/overflow behavior elsewhere in the allocator. Declined requests preserve existing behavior rather than silently claiming those original inputs are safe.

There is no admission dependency on `compact_moe_fp16`, MMQ or fused-expert dispatch. HC off/on can therefore be paired independently with MoE compact off/on. The same helper carves and counts the HC takes, including allocator padding and optional low buffers; bounded region carving checks the resulting HC byte count against the plan. GDN/QSA/MoE count and carve sequences stay unchanged.

## Lifetime proof from the actual path

The following line references refer to the original source hash above.

**Fused carry.** At 1767–1776 each half norms only when `normed` is false, projects down, applies SiLU, projects up and inject, and mixes into the outer mixed representations. At 2657–2673 `gr_write_norm_rs` can both write `R` and produce the next half's `grs`, `xn16` and optional low part. Those values survive into the next HC read. The proposal leaves them outside every shared phase span. `inj` is also outer because it is consumed with `bo` only after attention or MoE completes. The `normed` flag and the PLE/control-vector exceptions remain byte-for-byte unchanged in the run loop.

**New phase scratch.** `lo` is consumed by SiLU; `lo16` and optional `lo16_lo` are consumed by the HC up projection; `gated` is consumed by `gr_mix_r`. All these reads are submitted before the GDN/QSA/MoE body of the same half uses the phase region. The next half begins its own HC projections after the preceding residual write. The shared HC set keeps its simultaneously live buffers disjoint using the same padded take sequence in count and carve. It does not alias the HC set's components with one another.

**Queue order.** All these HC helpers and projections receive `m.cs`. SYCL `queue_for` requires an in-order queue for explicit streams, and the implicit runtime queue is also in order. The GEMM uses that same queue. Consequently no new fence or queue is needed between the HC last read and the following phase writes. Current `m.copy=m.cs` and R509's host handoff fixes are untouched; the HC lifetime argument does not propose a second copy queue.

**Embedding and block output.** At 1418–1469 a stage either uploads incoming residual rows directly into `R`, or gathers/uploads ordinary/image token embeddings into `m.emb` and broadcasts them into `R`. That broadcast is the final read of `m.emb` in the main prefill run. The first `bo` writes occur later in GDN output (1799), QSA output (1997), or MoE combine (2631), after that read on the same queue. `bo` is consumed by the residual write at the end of each half before a later chunk can overwrite the embedding span. For incoming layer-split rows the embedding span is unused, so sharing it introduces no additional read.

**PLE.** Asynchronous gathering and upload use `ple_emb_host[2]` and `m.ple_emb`, not `m.emb`. The PLE block at 1703–1756 runs before the layer's HC read. Its batched local temporaries already use the idle phase region; they are consumed before HC overwrites that region. The original write/norm dispatch prevents a carried norm across the PLE modification of `R`. Neither PLE's outer embedding span nor its host buffers/events are changed.

**Callbacks, drafter and reuse.** The chunk/stage callback boundaries at 2695–2734 synchronize compute and expose `R` or the chunk position. They never expose `m.emb` or `m.bo`. `draft_kv`, when called from `on_chunk`, uses local embedding/HC temporaries carved from the phase region and the supplied final residual rows; it never reads these two main-path members. Main HC temporaries have died by that boundary, and the next chunk initializes its own HC temporaries. If the callback leaves ordered work queued, the next chunk's submissions still follow it on `m.cs`. A callback that races an unrelated queue with private scratch would already violate existing scratch ownership; no new callback contract is introduced.

`relayout` at 899–922 drains compute and copy before recarving the borrowed region. `reset` calls `release`, which drains the queues before freeing registered owned allocations, then installs a new Impl. The alias pointer itself is never registered as another owned allocation: only the original embedding take or shared-region base is registered. No double free is introduced. Early-return consumers must continue using the existing release/reset/relayout lifetime contract.

**R510 mixed/bo correction.** R510 says GDN/QSA write `bo` while `mixed` remains needed by the layer's later MoE. The source instead performs a separate HC read for `half=0` and `half=1`, and each HC read rewrites all mixed representations. Thus attention-half `mixed` is not the same live value as the FFN-half input. That stated later-MoE rationale does not prove aliasing unsafe. An independent full review would need to cover each phase's direct FP32 consumers, converted mixed buffers, asynchronous peer/expert use, diagnostics and all output producers. This proposal deliberately keeps `mixed` and all converted forms distinct and makes no unsafe/safe adoption claim about a future mixed/bo alias.

## Byte plan and estimator limits

Every take costs `A(n,b)=align256(n*b+256)`. At capacity T=4096:

| BF16X2 mode | HC phase bytes | `bo` take saved | Outer takes removed | `bytes_needed` reduction if phase does not grow |
| --- | ---: | ---: | ---: | ---: |
| 0 or 2 | 175,637,248 | 41,943,296 | 217,580,544 | 217,514,752 |
| 1 (HC low enabled) | 178,258,944 | 41,943,296 | 220,202,240 | 220,136,448 |

The new phase is `max(old_phase,HC_phase)`; actual carve reduction is `outer_removed - phase_growth`. The padded region take's additional 256 bytes cancel because both phase sizes are already 256-byte multiples. R510's supplied old phase of 968,510,976 bytes is larger than either HC plan; **if that is the same actual configuration**, growth is zero and the corresponding carve reductions are the table's outer-removal values (approximately 207.5/210 MiB). This supplied phase value is not newly measured here. The HC change leaves the GDN candidate in the max, so at the fixed geometry the HC set is smaller than GDN's set even when MoE is compact; no MoE-specific reduction is claimed by this artifact.

The current `bytes_needed` differs from normal fused carve: it conservatively counts an FP32 `xn` take, omits the separate `grs` take, and passes `src=true` to fused expert layout sizing whereas carve uses the actual source presence. The default and declined branch preserves this original sequence exactly. The effective branch retains the conservative `xn` allowance but explicitly counts `grs`. Therefore its requested-size reduction is `carve_reduction - A(T*HC,4)`, or 65,792 bytes less than carve reduction at T=4096. This is intentional opt-in accounting, not estimator/carve equality. The existing 8 MiB slack and all GEMM/KV/staging counts remain. Main must record requested arena bytes and actual carved bytes separately, with actual source/layout inputs.

The carve-time diagnostic reports exact request/effective status, decline reason, T, HC phase bytes, old/new phase bytes, outer removal, phase growth, chunk carved bytes, full allocator accounted bytes and `ok`. It runs in init/relayout, outside `Prefill::run` timing. It prints only when the new variable is present. The unchanged MoE diagnostic can identify the independent MoE arm. These are plans/accounting values, not measured resident VRAM.

## Main's next boundary

Before applying, require the original source SHA above and rebase explicitly if Main's live source changed. Do not apply while the current owned modelpair is running. First review the proposed default/declined allocation order and compile through the configured source resolver. Then qualify the four independent HC/MoE arms in separate processes with identical source, geometry, chunk, queue, attention and expert settings. Admission, requested-versus-carved bytes and per-component phase maxima must be recorded for each arm.

Necessary qualification includes fused carry across both halves and layer transitions, BF16X2 modes 0/1/2, non-exact flag values/unfused/geometry declines, capacity and actual tail T=1/17, owned and borrowed relayout/reset, ordinary and image embeddings, PLE scalar/batched paths, chunk callbacks with the drafter, matrix/scalar QSA with distinct q/attn, and resident/streamed expert paths. Use R508's independent off/on residual/logit and full-context/restore procedure, retaining individual correctness and fault receipts before performance. Ring 8 is routed-only and does not qualify threaded streaming or R509's skip-order mechanism. Keep skip-fix effects separate from the two compaction toggles.

The current consumer is Main's serialized post-modelpair review and qualification. Review this proposal before source application and again at the four-arm qualification boundary. No duplicate full source or raw traces were created, so no retirement action or deletion manifest is needed for this artifact generation. Main can commit the compact proposal and resulting decisions at its next authorized work boundary; this agent has not committed or pushed them.
