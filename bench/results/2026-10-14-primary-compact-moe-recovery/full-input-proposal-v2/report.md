# Full-input zero-output proposal v2

Status: unapplied proposal. No product-source write, Git operation, build, test,
GPU/profiler operation, service change, cleanup, private log/environment read or
secret record access. Only this directory was written. The in-memory derived
source is never saved. `generate_proposal.py` pins and checks the current source
SHA256, then writes a unified patch and `identity.json`; it is an artifact
constructor, not a compiler or test. Parent supplied HEAD is
`8d13cef8d9ca82887358d066d2e3d1982cb191eb`; no independent Git inspection was done.

Origin: R512 and current Niko1221/Strata personal fork implementation. No Xe code
or CUDA/HIP engine is imported. Consulted R512, the pre-Xe model-compare,
performance-regression-audit and compact-MoE-recovery canonical READMEs, project
strata-upstream-sync skill and artifact-retention policy. Historical 262140+4,
262143+1 and small compact/tail/ring checks do not qualify 262144+0.

The configured SYCL resolver is `cmake/sycl_backend.cmake:103,107`: root
`src/program/generate.cpp`, root `src/prefill/prefill.cpp` and its SYCL component
sources. Existing model loading, normal session pricing and initialization at
`generate.cpp:2662,2666`, KV/page/QSA/GDN/PLE allocation, native head and MTP
loading, expert source/cache and token graph setup remain the execution route.
No allocation or position table is resized or mocked. Context remains exactly
`o.max_context`; input-only contributes zero output reservation.

The proposed flag requires explicitly `--input-only --max-new 0`, nonempty real
input IDs fitting context. Normal output validation/behavior remains unchanged.
It accepts smaller inputs for preliminary checks; Main's full gate must demand
exactly 262144 IDs and context262144, last input position262143 and emitted0.

At source 6246, `n_batched` remains `n_prompt-1`. The existing MTP on-chunk callback
indexes the following real prompt ID at 6298; it cannot safely consume the final
row without a successor contract. All earlier prompt IDs still pass through
normal Prefill::run; its reported token count is checked, then saved before
SYCL `prefill.reset`. Cache lending, queued refill, refill synchronization and
PLE predecessor reconstruction remain in their existing order; diagnostic mode
also checks both previously unchecked lending/refill residency-table copies.

Ordinary spec0 uses the existing token loop, layers, target head and full logits
readback/finite scan. It checks available native-short-attention QSA host status
before completing a position. It never invokes the sampler, including when
prefill is disabled. It updates the actual two-token PLE window, counts each
completed real position, teacher forces the next input and returns after the
last real input. Prompt and output lists are not dumped on successful completion.

Native spec>=2 (including configured MTP4) still bypasses the token loop and
initializes/binds the existing Verifier with native head, sampling, split and
PCIe configuration. The diagnostic runs exactly the normal first T=1 window
at `n_prompt-1` with the actual final input ID, checks pool failure, commits one
real input, waits for the commit, reads the full target head and checks finite.
It does not silently set spec0 or unbind MTP. `Verifier::run` synchronizes compute
and expert-copy streams (verify.cpp:1400-1404) and internally samples a prediction
through its existing head path; that prediction is discarded and never appended
or fed. No draft call, adaptation thread, refill stream/event or position advance
is started. `commit(1)` updates ordinary GDN/QSA/PLE state (including PLE window).
The target-model state is complete; this does not claim that MTP has a final
no-successor draft cell, because drafting requires a predicted successor and is
intentionally not performed. Existing batched MTP input cells and verifier
binding remain intact.

Native-verifier first-logits environment diagnostic (`STRATA_DUMP_FIRST_LOGITS`)
is supported in the new verifier branch with
checked open/write/close. The ordinary spec0 token route does not implement this
environment dump; its full readback and finite scan do not produce a capture.
CLI dump flags remain rejected, so token-route head comparisons need a separately
reviewed capture extension. The existing complete-window SYCL verifier capture
already checks its own copy/open/write/close. CLI dump/capture paths are rejected
before they open files, so early success returns do not bypass their closes.
The ordinary and verifier success returns retain existing automatic object
cleanup, including prefill/verifier/session owners and checked SYCL allocation
cleanup; there is no synthetic 'all queues completed' message. The receipt is
printed before destructors execute. It is therefore not sufficient on its own:
Main must additionally require normal native/process-owner exit0, no cleanup/API
error, same boot and fresh driver/kernel fault0. Error returns never print the
receipt. This proposal adds no cleanup replacement and no signal suppression.

Unsupported combinations explicitly rejected: positive/nonzero max-new with
flag, empty input, serve, vision, layer split, graph-only, gpu-only-full,
gpu-stages, no-pool, no-PLE, partial `prefill-until`, oracle/corrupt drafts, all
CLI logits/residual/mixed/layer/half/routing/expert-input/final-R dumps. Existing
logits-stride>1 validation also rejects it. Spec1 and non-native speculative
modes are rejected; native speculative multi-token inputs require batched
prefill and reject native-flash-attn-short (verifier lacks that token-path status
contract). Native spec0 keeps the existing platform support restrictions. Other
existing configuration validations still apply. Help remains available.

The single compact JSON receipt uses observed prefill token/chunk statistics,
sequential actual completion counts, actual input/context and final input ID,
effective prefill chunk/spec, actual successful MTP prefill/verifier binding, route,
synchronized head and target-state completion, finite full target logits, output0
and whether the verifier internally
sampled a discarded prediction. It does not assert binary/compiler/backend
identity; Main's enclosing receipt must pin those using actual configured build
and resolver identities. Effective spec may be expanded by existing suffix-draft
policy (e.g. spec4 to6); Main must preserve/record both requested spec, mtp-max-t,
suffix-draft and effective spec instead of relabeling configuration.

Main's serial review/validation obligations after the active run closes:

- Verify source and patch SHA256 against `identity.json`, apply/review, build
  through the actual configured root resolver; record binary/toolchain identities.
- Small 1-ID, 2-ID and multi-token inputs; spec0 with and without batched prefill;
  native MTP4 with the same existing configuration, split on/off where required.
- A chunk+tail case (e.g. 130 real inputs/chunk64 =>129 batched as64/64/1,
  final real token at129), with compact OFF/ON and appropriate numerical control.
- Negative cases: zero max-new without flag, flag with positive max-new, empty
  input, over-context input and each rejected mode. Confirm diagnostic never
  emits output IDs and normal output still samples as before.
- Separate normal near-boundary262143 input/max-new1/context262144 arm; compare
  first target head against matching diagnostic where its dump is available.
- Actual262144 input/max-new0/context262144, same native MTP4 model and relevant
  compact configurations. Verify fixture count/hash and actual receipt262144,
  batched262143, last262143, head/state complete, emitted0; allocation admission,
  accepted chunk, clean process/owner exit, checked cleanup and fresh fault status.
- Failures/readback/finite/commit/cleanup checks must remain failures; full receipt
  alone never establishes quality, restore safety or performance adoption.

No measurement or runtime pass is claimed. Owner/consumer: Main's serial full
boundary implementation review; next review: patch application and small-input
validation after active benchmark closure. Retain this compact proposal/identity
in Git at the next authorized boundary; no full candidate source or raw traces
were generated.

V2 receipt correction: `mtp_bound` is now the OR of booleans set only after the
standalone batched `mtp.bind` and/or native verifier `mtp.bind` return success.
Separate `mtp_prefill_bound` and `mtp_verifier_bound` expose those two actual
events. They initialize false before the standalone prompt path. A one-token
spec0 route cannot set either flag: it skips batched prefill and verifier setup.
The existing source also clears/ignores `--mtp` when spec<2 at generate.cpp:2721;
CLI presence is never counted as binding. A one-token native spec>=2 route can
set verifier-bound only, while a multi-token native MTP route can set both.
Bind failure returns before setting its respective flag and never emits a
completion receipt. No runtime checks were run for these cases.

Schema2 replaces ambiguous `state_complete` with `target_state_complete` and
adds `final_mtp_draft_cell_processed:false`. This deliberately says that target
GDN/QSA/PLE completion and MTP binding do not prove a no-successor final MTP
draft cell. The verifier's prediction is still internally sampled/discarded,
with emitted IDs0 and no final drafting call. V1 files remain untouched.
