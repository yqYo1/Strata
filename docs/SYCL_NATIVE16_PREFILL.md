# SYCL prefill and decode decisions

This page records current selections, the evidence needed for the next decision,
and open qualification gates. Closed experiments and detailed samples remain
in Git history and the linked cumulative result; this is not a run log. The
closed chronology is recoverable from the [original maintained decision at
commit `3abcceba08b25a632f4e26bde7551039120ac42a`](https://github.com/yqYo1/Strata/blob/3abcceba08b25a632f4e26bde7551039120ac42a/docs/SYCL_NATIVE16_PREFILL.md).

## Current selections

- The Main-authored exact selected-expert copy-run experiment is **closed and
  not adopted**. Correctness/byte/lifetime checks pass,but both prospective
  performance gates fail in each clean campaign. Maximum-run evidence43d1a03e
  records ordinary384 wall371.933503->517.427586ms and packed512
  175.042611->275.450563ms. The separately planned cap16 campaign at7ba50b1f
  closes six quiet independent processes ABBAAB with per-expert clocks off:
  ordinary384 medians371.899923->380.143548ms,paired candidate/reference ratios
  1.026201/0.991949/1.107934;packed512 155.988031->152.884204ms,ratios
  1.057468/0.956912/0.980102. The lower packed median does not satisfy its
  all-three-pairs gate. All samples are eligible,normal0,owners closed,no fresh
  fault;neither campaign permits a model trial. Four component rows are in the
  CSV,with original individual samples/decisions in the linked result.
  The V1clock-calibration failure00ac4489 remains failed. V2at316a03df
  completes15profile processes with conserved disjoint wall/current-thread
  buckets;later reader submission can include runtime waits. Requested bytes,
  API counts and these timings do not establish physical bus traffic,CPU math
  or hardware saturation. The native review corrects the earlier report's false
  payload mismatch and arithmetic;no other reachable defect was found there.
  The failed prototype-only run APIs,modes,tests and controllers are retired to
  original sourcef98dd238/controller0a029517. The original single-ticket overflow
  check and independent generic-builder fixes remain;production prefill is
  unchanged. After2366Git source/dependency and727compiler/24link-input recovery,
  the final fixture/raw products are retired:143files,45865160regular content
  bytes. No symlink-target bytes are counted. The coherent engine is unchanged.
  Native Luna-max review finds no reachable clean-controller/admission defect.
  The consumed research transactions and duplicate copy-only private receipts
  are retired;compact provenance/corrections remain in the research index.
  The earlier GCC V1provenance failure remains unqualified.
- The historical partial-relink **engine dcb8efd4** and GU component **cd99a46e** are on a
  **class-definition coherence hold**. The archive combined the new `Gemm`
  member declaration from `213dd970` with eleven original archive members,
  including a translation unit using the old class definition. Layout and
  existing-symbol equality did not establish C++ ODR coherence. Keep those
  partial-build artifacts out of further runtime qualification and target/adoption claims.
  Preserve the original numerical, normal-exit, fault and timing observations;
  this finding does not attribute any GPU hang or speed difference to the defect.
  Correcting evidence is `gu_class_definition_coherence_20261112` at
  `32bde1b0dba0b5c7f83529a0e182e8d83497afc6` in the linked result.
  The fresh coherent full build from **14508acb** now closes normally with
  137 compilation objects/134 unique sources,1470 compiler inputs and29 external
  link inputs,with zero uncovered inputs. Its five actual `Gemm` includers use
  the same header and SYCL class condition;no native object/archive was reused.
  Evidence: `coherent_prefill_native_v1` at
  `cca8ac8d9426769c995e99e4af0aa7d5b624ff11`. The new engine and GU fixture are
  byte-identical to dcb8efd4/cd99a46e respectively;source/build provenance still
  distinguishes them. This does not retroactively qualify the historical mixed
  build. Fresh `coherent_prefill_device_v1` now closes all11stages at
  `5788e16a61278e3d42117313041b1be9beb1a777`:host/policy/decode-module checks,
  traced submission,phase and real canonical/packed heads,QSA/producer small
  and262144-cell boundaries,and GU smoke/44numerical cases. All native children
  close normally0with quiet parameter validation and no fresh fault;GU hidden
  outputs have0FP16ULP difference. No model or speed qualification follows.
  The coherent actual-model `coherent_prefill_model_v3` now passes the thinking-on
  32769/82/32769-input handoff with32outputs per request,evidence at
  `9c678595231882266623373cb84e4b65071c53dd`. All three requests close normally0;
  both long entries verify actual wide GU/Down selection,typed QSA/bulk ranges,
  cache bindings,MTP expert restoration and decode recapture. All15full asset
  hashes match,with quiet validation,closed ownership and no fresh fault.
  V1's wrong Python and V2's wrong action name remain recorded prelaunch failures;
  neither started a model. V3uses the qualified model venv and `run` interface.
  This is observed route/state evidence,not a clean-speed or sustained-TG result.
  The fresh same-build `coherent_prefill_speed_v1` now closes six independent
  `bench.sh` processes/eighteen thinking-on exact24-output requests at32K/64K,
  evidence `f9662f7abfebb1c5f20e7e9e4677fad8f1f5e695`. All processes close
  normally0,with exact fresh counts,actual GU/Down/placement checks,quiet
  validation,unchanged boot/driver,no fresh fault and all15final asset hashes.
  Warmed32K process medians are871.305/871.574/872.318tok/s;the aggregate is
  37.5975s prefill and37.741s client TTFT. Warmed64K medians are
  1006.412/1006.832/1008.257tok/s;the aggregate is65.0923s and65.2565s TTFT.
  Thus the prospectively defined600PP repetition gate passes at both lengths,
  and the1000PP gate passes for warmed64K. First requests remain separate:
  aggregate32K/64K PP is676.896/865.934 and TTFT48.555/75.840s. All18individual
  samples remain in the linked result;four summary rows are in the CSV.
  The warmed adjacent slope is0.839074707ms/token;two lengths do not validate
  a fitted law or physical fixed cost. No cross-build gain,sustainedTG,task/agent,
  physical262144/RESTORE,ordinary teardown or model adoption is established.
  The exact-copy component campaign described above is now closed and rejected
  for performance;its original source/evidence is retained through the merge.
  No model trial or inference adoption follows. The failed production selector,
  prototype fixture helpers/modes and ended standalone build are retired.
  Migrate obsolete97fda/old-fixture
  reader dependencies and retire ended generations before another engine build;
  preserve56cad9 and a347only for their still-named phase comparisons.
  Normal successful test/model logs and caches are retired:151files,20896310bytes;
  both current test/model roots retain only their records and model admission
  still passes. The compact decision is under
  `retained_builds_review_20261103.coherent_closed_raw_retirement_20261112`.
  The closed speed cohort separately retires354successful raw/cache files
  (62679346bytes)and two consumed review transactions(7660bytes),after original
  source/recipe recovery and owner closure. Configs/plans/receipts/runtime inputs,
  compact samples and native builds remain;see its`artifact_retirement`field.
  Old partial-build and legacy provenance cleanup remains open;no global
  retirement is claimed.
  The existing physical full-context diagnostic and its host fixture are also
  built;actual model fill and ordinary provider teardown remain pending.
  The prospective [coherent_native_reuse.py](../tools/sycl/coherent_native_reuse.py)
  now verifies2361original source/dependency files,all1470actual compiler and29
  external link inputs,the consistent137-object/134-source build and all12ELFs,
  plus unchanged probe and closed qualification evidence. Its source-only V2
  passes;V1's host/GPU-schema refusal remains failed. It permits only the exact
  uncompiled generic-builder delta and named committed observers;old fallback
  ELF bytes are not inputs of this new reader. This is no new GPU/model result
  or permission to retire unresolved legacy generations. Use a fresh committed
  plan for small decode-PCIe resource tradeoffs. The first finite1/16/EARLY0
  trial closes three no-fork copy fixtures and three thinking-on82input/32output
  requests,normal0/quiet/closed/no freshfault. Each model request submits and
  completes375ordinary-host-view copies/768896000requested bytes;these are not
  physical traffic or bandwidth measurements. V1keeps an advancement hold:
  native review finds missing strict host-view equality,request/footer pairing
  and durable preflight failure/no-retry handling. Preserve its original
  execution/counts;the corrected43a4controller now passes its separate live
  32769/82/32769-input handoff at8a7283b8,with32thinking outputs each,all15final
  asset hashes,normal0/quiet/closed/no freshfault. Each long request has400
  submitted/completed/private-host-view/GPU-miss groups and819404800requested
  bytes;the short request has375/768896000. All3prompt/copy/stdout joins pass,
  with exact extents,drafts and rounded clocks. Host corrupt-stream controls
  used82-token geometry;the actual long run establishes long-geometry binding.
  Native Luna-max review confirms the3controller corrections;the oldV1hold
  stays intact. This is route/state execution evidence,not clean speed,
  numerical/task quality,physical262144/RESTORE or adoption.
  At this boundary,2984ended raw/cache/research files/links are retired after
  original controller/native-source recovery and actual owned-process closure;
  regular content bytes are591846816,with symlink targets and physical disk
  reclamation not inferred. The long-V2record remains for its named imminent
  comparison;oldV1duplicate private state and consumed research transactions
  are removed. Detailed row histograms/process-memory snapshots stay in the
  original8a7283b8Git evidence,while current useful job/format/upload totals
  and individual request/copy results remain. All12coherent ELFs are unchanged.
  The `coherent_decode_pcie_speed_v3` comparison is now **closed** at controller
  `e2ce32a8`,using unchanged coherent14508/dcb8. Six independent official
  bench.sh processes run R,C,C,R,R,C:finite0versus1/16missed-expert offload,
  EARLY0,static1256slots,5workers,spec4,8192throughput BF16/vLLM and the same
  nominal MTP callbacks/quiet validation. Each has four warmups followed by
  thinking-enabled code/prose long generation. All12long requests have
  33211/33281fresh input tokens and at least2815output tokens. All six native
  and benchmark processes close normally0,owners close,validation stays quiet,
  boot/driver remain unchanged,no fresh fault occurs,and all15final full model
  hashes match. Individual samples and the original decision remain in the
  linked result under that key;`final_summary`holds the three-process aggregates.

  | Workload / arm | Generated | PP tok/s | Client TTFT s | TG tok/s (range) | Draft acceptance |
  | --- | --- | --- | --- | --- | --- |
  | Code / reference0 | 2815 | 675.780 | 50.609 | 15.037 (15.008..15.086) | 87.009% |
  | Code / candidate1/16 | 3602 | 669.976 | 50.893 | 14.409 (14.405..14.586) | 87.007% |
  | Prose / reference0 | 6823 | 848.150 | 39.401 | 9.073 (9.045..9.104) | 33.935% |
  | Prose / candidate1/16 | 8911 | 844.758 | 39.551 | 9.424 (9.383..9.485) | 39.149% |

  Code is the first long request after warmups;prose is the subsequent long
  request with a different task. Request order/content are confounded. The
  candidate/reference TG ratios are0.959847/0.955140/0.970032for code
  (geometric mean0.961653)and1.041889/1.030666/1.045350for prose(1.039283).
  Thus code fails the repeated-gain gate and prose passes it. Both observed
  strong-code/weak-prose classes pass in every process. This is a repeatable
  observed-route prose gain,not a global adoption or fixed-work attribution:
  generated lengths differ,and prose draft acceptance changes. Keep the0
  code reference;dedicated-prefill preservation,agent/task,physical262144/tails/
  RESTORE,ordinary-provider teardown and arbitrary-client nonfinite-parser
  repair remain open before adoption. The actual limiting hardware resource
  remains unknown. Candidate code/prose requested copies are129390259200/
  404578022400bytes per process,not measured bus traffic or saturation.

  V1's host-state mutation refusal and V2's missing presence-enabled callback
  failure remain failed in their original evidence. V3prospectively separates
  short warmup observations from the strict long read-1cells/8192chunk contract;
  all6request/footer/API/copy joins pass per process. Original Main launch/
  cleanup wrapper errors and their verified reconciliations also remain in the
  compact result. After pushed samples at`b54cbd5df`and original-source recovery,
  final routine output is retired:69files/links,13085550regular content bytes.
  All6measured sessions are empty,the current reader passes after retirement,
  and all12ELFs/generated inputs remain unchanged. Prior retirements remain
  in that same result,without renewing their consumers. Four aggregate CSV
  rows reference the closed evidence and remain in the cumulative CSV.

  The completed EARLY audit establishes a finite-DMA gate,not a controlled
  deterministic hold. Its earlier completed-before-admission failure stays
  failed;synthetic F-K state does not qualify production capture or same-build
  grouped numerics. Keep EARLY0until a new bounded fixture/runtime admission.
  Corrected CPU-representation/cache/worklist/source-audit findings remain in
  the research index;their consumed report/snapshot/review payloads are retired.
  The Nov11template root has a separate ownership/helper-consumer hold;reconcile
  its original Task before pruning ended payload. Legacy19reader/deployment
  and source-recovery holds remain open. No new engine generation was created.
  The CPUcompact ownership and actual-production-dispatch numerical component
  now pass,with original source/results merged through`938829829`. V4at`9071db9a6`
  checks768real-weight serial cases across Down20/42,1280disjoint threaded rows,
  phase/NT/alias guards and exact IQ3paired/Q8_K output bits. This is component
  parity,not model/task/RSS or performance qualification. The fresh hot/streaming
  fixture at`20a202d28`reuses the source-qualified CPUarchives;its separate128case
  admission per format also passes. The committed V2plan`d27a245f0`now closes
  12fresh processes/144eligible timing batches,min157.451ms,at evidence`762d2ff47`.
  Lookup+GU gains pass all3matched pairs in all8Down20/42/thread/pattern cells.
  Geometric-mean speedups are2.33..2.41for1worker reuse,2.21..2.25for1worker
  stream32,2.62..2.68for6worker reuse,and1.82..1.86for6worker stream32. Per32experts,
  allocation/encoding/freeze is171..195ms and known view storage54898632bytes;
  raw bytes remain resident. These are synthetic full640-row worker loops,not
  production-pool partial-row scheduling or observed model reuse. Cross-worker
  L3reuse is possible;logical bytes/time is not physical DRAM traffic or saturation.
  All individual samples/setup/RSS and per-cell amortization remain in the linked
  CPUresult. Eight component rows bring the CSV to352;model PP/TG remain blank.
  The new source-owned production bridge has passed its fresh CPU whole-pool
  numerical gate;the generate/core engine wiring remains unbuilt and has no
  model qualification. `STRATA_CPU_COMPACT_GU_MIB=0`or unset keeps native GU;
  an explicit budget prepares all eligible immutable resident host layers before
  pool/READY,or keeps native fallback. Source coordinates are captured before
  CPUjob grouping loses expert identity. Leases span synchronous pool readers;
  raw blobs still provide Down. Only Decode/NT1 with matching GU format selects
  cached math;prefill,NT>1,misses and unsupported providers retain native GU.
  The codec arithmetic is moved unchanged from the owned component into the
  CPUmodule;Cache now hides storage behind an opaque API. Source`c61ee3e7b`and
  plan`e2fd9e722`fresh-build11CPUunits,24GGMLunits and the fixture,covering627
  compiler inputs and22external links. All96whole-pool cases pass across
  Down20/42,including GU/hidden quantization/Down bit equality,partial rows,
  97/193job batches,lease retirement,misses and phase/NT/budget refusals. Both
  children close normally0;no GPU work runs. See`compact_production_pool_v1`
  in the CPUresult. This is numerical evidence,not production-pool speed.
  Existing keys/misses repeat every32/3jobs,so a wrong96job-offset implementation
  could escape these data;the source uses the correct offset. The next timing
  admission must use nonperiodic keys/misses across batches and independently
  enforce host/worker topology,with correctness outside timing. Reuse only this
  coherently built generation;rebuild all affected translation units for any
  further class change. No new model/TTFT/RSS or physical262144 claim.
  The completed source reviews correct six-worker stream32 to16experts per
  worker/32aggregate and distinguish singleton affinity from topology admission.
  Main observed distinct host cores0..5 outside the old controller;the new pool
  controller pins and rechecks topology. Old raw samples/gain gates remain a CPU
  screen,not cold-DRAM or model evidence. The original V1missing-header refusal
  remains failed. The retained engine ELF and its prefill/decode paths are
  unchanged;new source is opt-in and model/performance validation is pending.
  The whole-pool timing source`cd156b4ba`and plan`3cbf1c597`include
  lookup/lease handoff,GU,host hidden quantization,Down and synchronous waits.
  They close12synthetic cells per format:1worker or5workers+host,1/6/32jobs,
  reuse or stream32. All12native processes close normally0with quiet output;
  432batches are eligible,min139.603578ms. The new240nonperiodic numeric cases
  pass exact bits/counters and detect deliberate wrong96job-offset leases in
  both worker geometries. The controller verifies all35reused archive members
  against the original compiled objects;no old CPUgeneration is consumed.
  Its measure-index interface does not enforce order;the original12start/end
  receipts establish R,C,C,R,R,Cand no overlap. Existing hash/link/library-name
  admission is not full ELFheader/interpreter qualification. See the CPUresult's
  `compact_production_pool_timing_v1`for individual samples and original gates.
  Twenty-three of24cells gain in all3pairs. The passing five-workers+host cells
  show1.206..2.083xgeometric-mean speedup. Down42/32jobs/reuse has ratios
  0.959739/1.295959/1.067272and mean1.099022;the slower first pair fails its
  prospective repeated-gain gate. The global model-trial gate stays failed:
  no cache-enabled model trial or adoption underV1. The newV2source focuses on
  four32job cells:Down20/42,reuse/stream32,5workers+host. Calibration now reaches
  2s and each of3batches must reach1s. It enforces closed previous-arm receipts
  before RCCRRCadvancement and checks ELFclass/machine/type/interpreter.
  Its prospectively declared combined component gate requires all4fresh cells
  to improve in all3pairs plus the original23passing cells,with the same CPU
  archives. V1remains failed. Commit the new plan before execution;no model
  qualification follows from these component checks. Source`b3032cc18`and
  plan`7f8b40ae4`build timing ELF`7d67f886`;its506compiler/26link inputs,
  all35archive members and ELFheader/interpreter admission pass. Both fresh
  120case nonperiodic checks pass,normal0and closed. The RCCRRCspeed cohort
  is now closed under`compact_production_pool_timing_v2`:12native processes,
  72eligible batches,min1.942018409s,last calibration at least2.047953854s.
  Three of4cells gain in all3pairs:Down20/reuse1.294353x,Down42/reuse1.269931x,
  and Down42/stream321.258734x geometric means. Down20/stream32 has ratios
  0.934068/1.150250/1.197623(mean1.087670),so the all-four/combined component
  gate fails. Preserve both V1andV2failures;no cache-enabled model trial is
  allowed by these campaigns. GU wall improves in the medians,while unchanged
  Down math can take longer with the larger GU representation. This does not
  establish physical RAM/cache traffic or saturation. The next bounded
  mechanism inspection finds the optimized integer sums already reassociated
  into multiple series:the actual c61compact object has295instructions in
  pair_dot and no YMMstack operands. The source-only accumulator rewrite was
  therefore not implemented. A separate Main-authored kernel prototype stores
  eight precomputed odd scales in138bytes rather than134(+2.9851%),removing
  their runtime nibble/2*s+1work. Its FPblock-FMA order,reduction,std::exp and
  codes are unchanged;the production134-byte codec/pool stays untouched.
  Source`a2afd41f8`and plan`bad6016c4`now build the standalone kernel
  ELF`c92a494b`:442compiler/23link inputs and full ELFadmission pass.
  All768real-layer/synthetic-Q8_K full/partial/empty-row cases match bits,
  and both wrong-scale controls are detected,normal0quietclosed. The frozen
  six-arm single-core comparison is now closed under`compact_expanded_scale_kernel_v1`:
  12native processes,108eligible batches,min1.806970283s,normal0quietclosed.
  None of6cells improves in all3pairs;all lose in the second pair. Geometric
  mean speedups are0.989345..1.023826,so the prototype is not adopted and does
  not permit a production-pool follow-up. Static pair-dot instructions fall
  295to258with no YMMstack operands,but unchanged vector arithmetic and changed
  block stride/alignment do not establish a cycle,IPC or resource improvement.
  Preserve all individual samples and failed gates;do not repeat until lucky.
  Pushed evidence`78cdb8bb5`and source/recipe recovery at`81361126d`allow
  retirement of its standalone ELF/object/raw root:57files,1101251regular
  bytes. Ten original project pins,291GGMLpins and442compiler/23link inputs
  match. Three consumed temporary retirement inventories are also removed.
  Both native reviews now close with no candidate-specific valid-Q8_K defect.
  The117project/7GGMLpins match their immutable snapshot. Synthetic-128tests
  establish common PSIGNBwrap parity,not mathematical signed-dot correctness;
  production Q8_K quantization uses[-127,127]. The standalone prototype,
  controller and plan are retired to originala2afd41f8/bad6016c4Git. The next
  small-batch pool build consumes none of their bytes. A compact conditional
  [Zen3 instruction reference](research/zen3-iq3-gu-instructions.json)retains
  reviewed constraints/sources for later retrieval,without adopting its methods.
  After462original project revision/path and28GGMLpin checks,the four ended
  Orca source/report scopes and four completed native review transactions are
  retired:2267files/links,460966525regular content bytes. Corrected facts and
  the8473-byte conditional reference remain;Main did not load collection bodies.
  Legacy GPU/IR/deployment and Nov11template ownership holds remain separate.
  Prioritize the original small-job134-byte gains. The unprivileged perf counter probe exits1
  at paranoid4;no counters or privilege changes. The host is5600X,
  family25/model33/stepping0 with32MiBL3;1/6/32-expert kernel working sets
  do not establish physical cache/DRAMtraffic. The actual generate/core wiring
  and12eallocation repair remain unbuilt. Native-reviewed source facts correct
  Down42raw blob versus460800BDown accounting and the claim of saving bytes by
  splitting already packed nibbles. Startup interposition is not GPU-free;
  allocation-fault qualification remains conditional on its admitted setup.
  Separately,source`a4430656fe54b289f85301f53c3b7a6f166a330e`implements
  the optional production selection as an unbuilt
  candidate capped at6jobs per complete Decode batch. Both the real dispatcher
  and pool entry use the same policy;larger batches retain native GU and per-job
  NT1/format guards remain. This starts from the original repeated small-job
  gains rather than adopting the unstable32-job result. It does not enable the
  cache by default or change the134-byte codec. Rebuild all affected CPU/core
  units and commit fresh numerical/timing criteria before evaluating it. The
  old97/193-job cached-offset negative control becomes unreachable under this
  policy;replace it with an eligible-batch wrong-expert control while retaining
  large-batch native-fallback,phase,NTand span-bound checks. No speed/model
  qualification or reinterpretation of the old failed campaigns follows.
  The new maintained CPUcontroller fresh-builds11CPU,24GGMLand two fixture
  objects for this policy,without borrowing the old archives. Its prospective
  gate checks240regular cases per layer,60forced-valid-lease controls,the
  eligible6-job wrong-expert control,independent one-batch phase/NToracles,
  short-span refusal and real Cache-owner retirement. Speed uses20cells per
  layer:1or5workers,1/3/6/7/32jobs,reuse/stream32,in RCCRRCorder. Each cached
  shape is decided independently;blocked shapes must select native math and
  retain separate nonregression samples. Source81914a76e and plan0de6e20a8
  now pass the fresh37-object build,637compiler-input/22external-link census
  and both CPU-only ELFchecks. Both layers pass240regular/60forced pool cases,
  independent phase/NToracles and384GUcases;all four native check children
  exit0and close normally. The original frozen RCCRRCcohort now closes12native
  processes/720eligible batches,min0.463212660s. Twenty-three of24cached cells
  improve in all3pairs,with geometric-mean gains1.340898..1.842588x. Down20,
  5workers+host,1job,stream32fails:R/C1.471468,0.842961,1.100006(mean1.109135).
  Preserve that failure;no blanket<=6policy or model adoption follows. The16
  blocked7/32cells remain native-fallback diagnostics,with geometric means
  0.944585..1.041501;their slower samples also remain. Main evaluates the
  declared paired gates;the controller's successful receipt alone does not.
  Both native reviews verify113project+7GGMLpins,find no reachable new policy/
  lifetime defect,and correct the report's cached-cell count and negative-n
  claim:the helper checks negative n,but the pool runtime case uses0only.
  Core/generate and allocation-fallback qualification remain separate.
  Start the next prospective model policy from passing production shapes:
  3/6jobs pass both formats/patterns at5workers+host;1job cannot be selected
  globally and2/4/5lack clean speed samples. Prioritize a coherent actual engine
  build and model distribution/RAM/TTFT/long-generation gates rather than
  repeating this CPUcohort. The real dispatcher source now offers cache leases
  only for3/6jobs in the5worker+host pool;other shapes keep native GU. The pool's
  <=6bound remains independent. This narrower source is now compiled in the
  fresh engine below,but has no actual-model qualification;the CPUtiming
  source remains81914a76e.
  The new full-engine controller source89be7ce36 now prepares
  `compact_cpu_model_native_v1`in the linked fused-service result. Its recipe
  starts from the validated14508full build,recompiles all affected units and
  rechecks all eight passing3/6job production shapes from original07ea14c4c
  evidence. Keep the14508GPUfallback and current CPUfixtures;legacy reader/IR/
  deployment holds remain separate. CPUresult and native build result are
  frozen during this build. No GPUexecution or model qualification yet.
  Configure/host-control/planning children now close normally0. The fresh
  configuration plans138object compilations,including compact_gu.cpp and
  both actual generate entrypoints. Commit this configured stage before
  compilation. The full build now passes:138planned compiler objects from
  135unique sources,1473actual compiler inputs,29external links,no uncovered
  inputs and all12ELFidentities. Five Gemm header consumers share the same
  SYCLdefinition;all are fresh,without reused native objects. Engine source
  89be7ce36 has SHAe7431db9;ten fixtures match the14508reference bytes,but no
  GPU/model result is inherited by that fact alone. All five build children
  close normally0. The new explicit native observer retains the bounded
  eleven-stage checks before actual-model handoff. Fresh host and submission
  stages now pass,normal0and closed owners. All eleven fresh native stages now
  close successfully,with no fresh driver fault. GU has44correctness cases,
  maxFP16ULP0,exactBF16capture,intact guards and immutable inputs. Attention
  and producer exercise262144cells as components,not a full model fill.
  The new candidate-only model observer keeps32769/82/32769thinking inputs
  and32outputs each,with16GiB CPUcache budget,5workers+host and unchanged
  spec4/8192/static1256placement. It requires8704prepared entries and actual
  supported-layer3/6pool exposure,with80GiB RSS cap and77GiB available-memory
  admission(old model peak52.016GiB). This is allocation/state/route diagnosis;
  per-job cached-math coverage,clean speed,task and whole262144model/RESTORE
  qualification remain separate. Its committed diagnostic now closes all three
  requests and the native process normally0,without a fresh GPUfault. The8704
  entries use14932662352known bytes;setup is44.809973s and peakRSS70997741568B.
  The original selected-shape gate fails:867supported-layer pool calls all
  have7..36jobs,so none can use the measured3/6policy. The three request-owned
  CPUcensuses conserve jobs/rows and pass error checks;their aggregate NTbins
  and charged pool times remain compact,not clean-speed or saturation evidence.
  Final quiet-validation/asset/source checks after the route predicate were
  not reached. Keep the screen failed;do not claim cached-math/model adoption.
  Next freeze a representative7..36job/mixed-NT CPUcomparison before widening
  selection,and qualify the new full engine with cache0independently. Keep
  the14508GPUfallback until that migration qualifies.
  The independent cache0plan at168c76904now passes all32769/82/32769input
  requests and96generated tokens,including complete normal0/closed ownership,
  quiet parameter validation,15assets before/after and no fresh GPUfault.
  No optional CPUcache is prepared. Peak owned RSS is55855677440B. This
  qualifies the historical coherent89be7engine for this functional handoff only;
  no matched clean-speed,sustained TG/task or physical262144/RESTORE adoption follows.
  Its ended build,native-check and cache0raw roots are now retired after exact
  original Git source/recipe recovery and final-input/12ELF checks:359files,
  220404252regular content bytes. The original build/native/model outcomes are
  recoverable atbe45a249e/6652e07b5/13645ffde;compact results keep their limits
  and deletion proof. The fresh baseline joint-census successor uses14508seed,
  not these old bytes. Keep14508as the clean measured reference. The earlier cache-on exposure
  screen remains failed and is not superseded by the independent cache0pass.
  The CPU-only scheduling component at sourcea841051e2 now closes a fresh
  36-object build,180numeric cases per Down20/42format and12RCCRRCnative
  processes/432eligible batches. Fourteen of24cells gain in all3pairs;all eight
  mixed-NT24/36job cells pass with geometric means1.018636..1.083121. All12job
  cells fail;two NT1stream64cells also lose a pair. The unchanged native math
  completes GU/quant/Down on one claiming worker. Serial quantization is only
  0.19..0.48percent of reference service;the joint scheduling/locality effect
  does not establish physical CPU/RAM saturation or model TG. Original samples
  and failures are committed at1d207e4fa;24CSVpoints index them. Merge d67570b63
  integrates the original child identities into pushed feature/sycl. Core/generate
  still do not select it. Next commit a shape-scoped model policy and fresh full
  build,preserving prefill and14508clean fallback. The finished child is now
  retired remotely/locally with its worktree;its extra CPUfixture and normal
  output are deleted(118files/links,14945636bytes)after203project/291GGMLpins,
  635compiler inputs,36objects,22links and original recipe recovery. The new
  production selector at sourcee2a305b74 remains opt-in via
  STRATA_CPU_DECODE_WHOLE=1;default0and all prefill keep row scheduling.
  Its fresh36-object CPUbuild3d00f203passes635compiler/22link admission.
  Both formats now pass180direct-method plus360actual-selector cases,with
  exact2selected whole calls/format,four lease/span controls,two wrong-input
  controls and intact guards/inputs. Both native processes close normally0.
  The actual-entry comparison now closes12native processes/432eligible batches.
  Its original eight-selected-cell gate fails:Down20/24mixed/reuse loses two
  pairs,andDown20/36mixed/stream64 loses one. Six cells pass all3pairs;all four
  Down42cells have geometric means1.052688..1.095976. Preserve144selected
  wall/CPU samples and all failures;16forced-row cells pass dispatch/work gates
  but supply no adoption claim. Default0stays selected;the gated model trial is
  blocked. Next scope Down42parts prospectively with changed source/plan/new
  evidence,preserving prefill and14508clean reference. The closed selector
  fixture/output are retired after205project/291GGMLpins,635compiler inputs,
  36objects,22links and original source/recipe recovery(118files,14965957bytes).
  The new prospective selector excludesDown20;both formats retain numerical
  coverage,but fresh timing focuses on four Down42mixed24/36cells in sixRCCRRC
  processes. Its scoped qualifier preserves the original failed cohort and all
  consumed source/sample/gate fields without freezing unrelated result rows.
  Source5660b2f41now builds36fresh CPU/GGML/fixture objects with635compiler
  inputs/22links,ELF49296435. Both formats pass540regular cases plus controls;
  selected whole calls are exactly0forDown20and2forDown42. Both native checks
  close normally0. The frozen four-cell cohort now closes six fresh processes/72eligible
  batches(min0.452564786s):only1of4cells gains in all3pairs. Paired gains for24reuse
  are0.998965/1.010517/0.983845;24stream1.003947/1.006470/0.994736;
  36reuse1.063607/1.075312/0.914179;36stream1.046199/1.064652/1.026876.
  The original performance gate fails;keep the production row default and do not
  enable an optimized model trial. A separate
  representativeness check finds that its36job global mix has47..52token-expert
  rows/call(mean48.9375),above the40-entry limit of this spec4/topk10target.
  The24job mix has30..36rows and fits that bound,but that fixture did not
  establish the actual joint distribution. Preserve stress timings without
  treating36job gains as model evidence. The full build is a baseline selector0joint-census
  observer(`tools/sycl/build_cpu_joint_census.py`),not an optimized model trial.
  Its12-target plan pins1199project files atd3578a0ae and requires
  selector0/census1before native exec. Fresh configuration closes normally0
  with138objects/135unique sources;the complete fresh build now closes normally0.
  ELFf59e7f44and all12fixtures have1473compiler/29external-link inputs with
  zero uncovered inputs;all five Gemm consumers have the same class definition.
  No speed result is inherited;its fresh native/model checks are recorded below.
  Native V1preparation passed source/build admission but failed before any child:
  the new controller omitted one zero-argument context factory before saving its
  explicit baseline environment. Keep V1failed;the corrected controller uses a
  fresh device-v2root/plan with the same f59e7f44engine. No GPU was submitted.
  V2preparation now closes normally0and pins explicit row/cache0fixture settings;
  the11-stage plan now passes54reader controls,11native CPUrefusals,phase/census/
  physical-context host controls and current-head IR identity. All11stages now
  close normally0,with quiet validation,no fresh GPUfault,and44GU numerical
  cases at0FP16ULP plus exact BF16capture. Fresh QSA/producer262144-cell checks
  pass;physical model fill remains unqualified. The next committed controller,
  `tools/sycl/cpu_joint_census_model.py`,keeps row/cache0and checks each target
  callback against its actual traced width*topk10,census,resident membership
  and old pool clocks. One literal positive and five rejection controls pass,
  including the prior47..52row/spec4volume error. The actual32769/82/32769-input,
  thinking-on32-output model plan is prepared with explicit selector0/cache0,
  census1and immutable safety settings. Two positive and13negative host placement
  controls pass. All three actual requests now close normally0with quiet
  validation,all15assets before/after,no fresh fault and complete state handoff.
  All1440target CPUcallbacks conserve actual traced width/topk10,NT,format,
  restored membership and the old pool clocks. NT1is75.0508percent of30029CPUjobs;
  GU18accounts for38.8463percent of observed pool wall and8845NT1jobs. Its
  actual decode batches are7..33(Down20)and9..33(Down42),with zero3/6job calls;
  those old small-cache selection shapes still do not occur. Peak owned RSS
  is56047382528bytes.
  Pool wall per32output sample is772.924/843.415/942.614ms,not sustained-TG
  or hardware-saturation evidence. Ninety bounded representative joint cases
  retain explicit coverage;full prefill histograms and GPUcycle counters are
  omitted. The separate Main-authored RAMread fixture now closes nine fresh
  processes/27eligible batches with a1GiB anonymous read-only buffer and distinct
  physical-core affinities. Three-process median logical rates are31.8424GB/s
  on1core,42.2734on3cores and39.3035on6cores;process ranges are31.5284..31.9334,
  38.5364..42.6838and36.4578..40.2513GB/s respectively. Minimum batch wall is
  0.802038s;estimated timer overhead is at most0.002180percent and timed minor/
  major faults are0. Worker CPU/wall medians are0.9992/2.9503/5.7191. Source
  `7f7c64676` and ELF`7a2f3e48` are pinned under`ram_read_ceiling_v4`in the
  CPUresult. These are application continuous-read references,not physical
  DRAMtraffic or random-expert ceilings. The model pool's logical21..26GB/s
  includes arithmetic,quantization,dispatch and waits;the comparison does not
  establish its saturation or justify changing its worker count. The prior
  nine-process V3observations retain their missing-observer-check hold.
  Target-verifier CPUjobs in the32-output diagnostics visit0.62609..0.67167
  logical GB/output(draft/refill/prefill excluded):unchanged
  traffic would require43.83..47.02GB/s at70TG. Dividing the measured6-core or
  3-core continuous-read rates by those bytes gives illustrative58.5..67.5TG,
  only if every blob requires one DRAMread and effective bandwidth matches this
  fixture. Physical traffic,LLCreuse and sustained strong/weak-context dependence
  remain unknown;this is not a hard ceiling. Reducing bytes per accepted output,
  increasing GPU-resident expert work and improving verification reuse remain
  important alternatives alongside CPUmath improvements.
  Next test actual legal mixed-NT shapes with selective NT1cached-GU and unchanged
  native NT2..4processing;also obtain1K+strong/weak long-census coverage.
  The source-only `BatchPolicy::MixedNt` proposal now gives that CPUcomparison
  an explicit per-call policy. Default model callers retain`SmallJobs`;the new
  policy is Decode-only,at most40jobs,totalNT<=40and eachNT1..4. Existing per-job
  NT1/format/lease guards remain;unsupported shapes and Prefill keep native GU.
  Source`761de6d1`now builds all36CPU/GGML/fixture objects freshly,with635actual
  compiler and22external-link inputs,ELF`3d7f6388`. Both class-header consumers
  share the same header and definitions. Both actual Down20/42formats pass
  152regular/legacy cases and14special controls,including actual NT1route
  accounting,NT2..4parity,NT5..8fallback and cache-owner retirement. This permits
  only the CPUcomponent comparison;no model route is enabled. Its frozen cohort
  now closes all12fresh processes/216eligible batches,normal0/ownership closed.
  All12Down/shape/pattern cells gain in all3matched pairs;geometric-mean gains
  are1.16024..1.82795. For example,Down20mixed25stream64 falls2.00557->1.47264ms
  per complete pool call;Down42mixed13stream64 falls1.14435->0.98865ms. Minimum
  batch wall is0.397586s,final calibration minimum0.501367s and maximum observer
  fraction0.00000372819. All216individual samples and paired spreads remain
  under`mixed_nt_cache_v1`in the CPUresult. The rotating64expert sets overlap
  between adjacent calls;this is not DRAM-cold or serial48-layer model coverage.
  Cache payload adds460800logical bytes per NT1job. This qualifies a separate
  model-plan preparation,not model speed or adoption. The owned model
  adapter now offers explicit `STRATA_CPU_COMPACT_GU_POLICY=mixed` alongside a
  nonzero cache budget;`small` and cache0stay default. It checks totalNT<=40,
  NT1..4and the measured5worker+host configuration before leasing NT1GU,with
  optional census binding counts. A repeated-expert MAXTwrite guard is added
  independently. No new model/PP/TG evidence follows;see`mixed_nt_model_v1`. Audit the actual activation
  format/range and cache selection,RAM/setup,handoff and full-context boundaries
  in that plan;preserve both prefill strategies and cache0/MTPdefaults. Do not
  partial-relink old class consumers. Closed RAMfixture/raw generations are
  retired after original Git/source/recipe recovery:44files,364453regular bytes;
  all27individual samples remain in the CPUresult. The455-row performance CSV
  now includes three RAMreference points and twelve mixed-cache comparison points.
  Keep14508clean speed reference and3d7fCPUfixture. The ended c0dc/c858small,
  492whole-stress and f59observer roots are now retired after original Git/recipe/
  final-input recovery:390files,231322303regular bytes. Useful actual census,
  failed gates and original samples remain;see`mixed_nt_prebuild_retirement_v1`.
  The fresh explicit model adapter now builds coherently froma1d8c29ec:139objects/
  136unique sources,1474actual compiler inputs/29external links,13ELFs including
  the actual-dispatch bounds fixture. Enginee8a1e54d is built,not yet runtime/model
  qualified. ExpertDispatch/ExpertPool have10/12fresh header consumers with their
  matching class conditions. Next run the separately committed native checks;
  no model PP/TG is inherited. Its first actual-dispatch host fixture exits
  SIGILL(-4),with no stdout/stderr and closed ownership,before any GPUstage.
  That fixture incorrectly selected the AVX512-only canonical Q2_0path on this
  AVX2PC. The target-model native IQ3path is separate. Preserve this failure and
  use matching native geometry in a new committed V2qualification;no old result
  is relabelled. Original2405source/dependency and1474compiler/29link inputs
  are recovered before retiring the ended V1build/host roots:211files,214621561
  regular bytes. The corrected native fixture retains all six positive/negative
  cases. Fresh V2from393d88e01now compiles139objects/136unique sources,1474
  compiler inputs/29links and13ELFs,with zero uncovered input. Its engine is
  byte-identical e8a1e54d;the corrected bounds fixture differs and still needs
  runtime qualification. The V2host stage now passes all six actual native
  MAXT/overflow cases,six invalid-policy refusals and the existing54reader/11native
  controls,with normal/expected exits and closed owners. No GPUwas submitted
  by that host stage. All11V2native stages now close normally0,with quiet
  validation,no fresh fault,44GUcases at0FP16ULP and exact BF16capture. QSAand
  producer262144-cell boundaries pass;actual full-context model fill is still
  unqualified. This admits preparation of a new bounded model-cache plan,not
  model speed/adoption. Source1826e44b0adds the explicit native-proof and
  `cache_policy/bound_nt1`observer adapter. Two literal positives and17negative
  parser controls pass,including NT4native fallback and independent per-NT
  conservation. The prepared`cpu_mixed_cache_model_v1`plan uses16GiBcache and
  thinking32769/82/32769inputs with32outputs each. Host2positive/13negative
  checks and the actual model route now pass,normal0/closed ownership,no fresh
  fault and15full asset hashes verified. All8704entries across17layers are
  admitted:14932662352known bytes,44.545736s setup,70997975040peak owned RSSbytes.
  Actual NT1bindings are2844/3157/2844,with independent NT/entry/membership
  conservation and prefill/decode/cache handoff. Poolwall790.041/771.382/772.646ms
  is unmatched against the older baseline,not a speedup. This charged diagnostic
  is not sustainedTG/cleanPP or adoption. Next compare repeated bench.sh long
  thinking workloads before independent prefill/agent/task/full-context gates.
  The frozen`cpu_mixed_cache_speed_v1`plan now specifies six independent
  bench.sh processes inR,C,C,R,R,Corder,each with4warmups and thinking-enabled
  code/prose1K+ generation. Only cache budget0/16GiBdiffers;common census/trace
  validates actual timed bindings and charges its observer cost. Collector
  controls pass3positive/9negative cases,including a zero-window one-token
  warmup. Host admission and all six processes remain pending. This is a
  mechanism screen before clean timing/adoption. See`mixed_nt_model_v1`;
  clean14508PP/TG remains the speed reference.
  Legacy holds remain separate.
  This baseline observer is not an optimized model adoption. The consumed V1private
  preparation transaction is retired after exact controller recovery. Use the
  verified actual layer/format/jobs/NTcorpus for the next prospective policy.
  Physical saturation and whole-model speed remain unqualified.
  See`compact_small_batch_pool_v1`in the
  CPUresult. The metadata contains17eligible layers/8704experts:encoded block
  payload alone would add13.906GiB while retaining raw weights. This arithmetic
  excludes index/Entry/allocator overhead and does not prove model RAMadmission.
  The superseded pool.cpp,old timing controller and two ended plans are removed
  from the active tree:four files,91353bytes,recoverable respectively from
  c61ee3e7b,b3032cc18,e2fd9e722 and7f8b40ae4. Their original pins are verified
  against pushed permanent history. The fresh successor is now selected for
  CPUauxiliary numerics. Originalc61source/recipe recovery checks38project,
  291GGML,627compiler,36object and22external-link inputs. Its old CPUroot,
  closed speed receipts/output and both consumed research/review transactions
  are deleted:1256files/links,248238592regular bytes. Current source/build/
  executable/numeric admission passed at that boundary. The selected3d7fCPU
  fixture now supplies the coherent engine/model qualification;the superseded
  small CPU pair is retired as recorded above.
  Corrected research facts remain indexed;legacy GPU/IR/deployment and Nov11
  ownership holds are separate. This is not global cleanup.
  The old plan/source are recoverable from`3cbf1c597`/`cd156b4ba`. After42project
  and291GGMLsource-pin recovery,the new committed consumer retires the old
  V1fixture's13files/3728035bytes;V2admission passes after deletion. Keep current
  history references for the original CPUarchives/numeric8c2e;their bytes are
  retired above. No V1timing byte is consumed.
  Actual limiting hardware resource remains unknown.
  The native review finds two conditional allocation-fallback gaps in generate:
  collecting resident views and sizing dispatch metadata can throw outside the
  helper's catch. The source now catches preparation allocation failure and
  prepares both dispatch arrays before publishing any cache/source binding.
  Failure releases the optional cache and keeps native GU;PREPARED precedes
  final READYbinding. The repair is in the coherent89be7full build and normal
  cache preparation completes above;allocation-fault injection remains pending.
  Normal math has no observed defect. RAM/TTFT,prefill preservation and model/
  physical262144qualification remain pending. The old leaf loop's individual
  samples are retired from this current result to original`762d2ff47`Git;
  its eight summaries/CSV and current whole-pool samples remain.
  Closed old lifetime/numeric/admission/leaf roots are now physically retired:
  92files,17763285regular content bytes,after42project revision/path pins,
  291GGMLpins,625compiler/22link inputs and34old archive members are recovered
  and checked. Original individual samples and failures remain in Git. The new
  timing admission passes after deletion. The integrated CPUchild at`762d2ff47`
  is retired remotely,locally and as a worktree;all original revisions remain
  ancestors of the pushed permanent primary. Current fresh pool archives and
  the coherent GPUengine remain. Legacy GPUfixture/IR/deployment holds stay separate.
  Successful whole-pool output and four consumed source/report/review
  transactions are also retired:2294files/links,463517579regular content bytes.
  Recovery verifies492project revision/path pins and28research GGMLpins;
  compact432samples/order evidence and corrected facts stay committed. The
  current reader passed after deletion. After its focused repeat closed and
  evidence71eaf4c58was pushed,42project revision/path and291GGMLpins,
  506compiler/26link inputs and35archive members were recovered. The ended
  V2timing fixture/raw root is now retired:53files,3866348regular bytes.
  New standalone scale admission passes without those bytes;the current CPU
  numeric archives and coherent GPUfallback remain unchanged. The CSV
  now contains380points,including24V1and4V2whole-pool cells with failed gates marked.
  Preserve the coherent **dc645 / engine97fda** fallback and its build-proof
  readers, **611a / engine56cad9** for the matched32K/64Kprefill comparison,
  and **221e / enginea347** for independent sustained-decode preservation.
  The obsolete9af and552b build roots are physically retired;
  their exact source/recipe recovery and deletion evidence are recorded in
  `retained_builds_review_20261103.latest_phase_transition` in the linked result.
  Historical retention instructions below describe their original experiments;
  they do not renew ended binary consumers.
- The closed same-binary `gu_small_prefill_compare_v1` compares immutable
  GUlarge/auto policies through `bench.sh`:12fresh processes across32K/64K,
  three thinking-enabled requests with24outputs per process. First use and
  subsequent-request medians stay separate. All36requests close normally0,
  with exact counts,actual dispatch checks,quiet validation,no fresh fault and
  all15final asset hashes. Both lengths fail the prospectively committed
  all-pairs-gain gate;do not adopt compact fusion or advance it through further
  model gates. Retain GUlarge as the policy for the coherent rebuild. Model timing does not
  consistently reproduce the compact component gain.
  At32K,the warmed large/auto aggregate PP is **870.042/872.650**;at64K it is
  **1006.223/1006.552**. The three64Klarge process medians are
  **1007.834,1006.223,1005.917 tok/s**;aggregate prefill time is **65.1317s**
  and client TTFT **65.2995s**. First-use large PP is678.422at32Kand862.122at64K.
  The cohort evidence at `ef1d787134b15d13ebdc13559175b20a5a45601b` and the CSV
  retain all samples and separate scopes,including the earlier one-process
  milestone. These historical mixed-build observations remain held from target/
  adoption qualification;the new coherent cohort independently reproduces
  warmed64K above1000. Neither cohort establishes a gain over56cad9.
  The32K-to64K warmed adjacent slopes are0.838257/0.841042ms per token for
  large/auto;two lengths cannot validate a fitted time law or a physical fixed
  cost. Parameter validation and the retained-reference preload are charged;
  ordinary teardown,agent/task quality,sustained TG and physical262144/RESTORE
  remain unqualified. Hardware saturation is still unknown. Both attempted
  preparations of the subsequent owned-engine comparison failed before any
  native launch;their original failures remain recorded. That comparison now
  requires the coherent rebuild while preserving the independent a347decode path.
- The historical **479a / engine552b** clean117489-token throughput-route result
  is three independent processes, warmed subsequent-three medians
  **1044.845, 1047.067, 1047.336 tok/s** (median 1047.067); first-use results
  are separately **965.892, 968.399, 969.346 tok/s**, with client TTFT
  **121.812, 121.489, 121.375 s**. This is a same-engine clean revalidation,
  not a matched gain over a retired engine, general model adoption, sustained
  decode, task quality, resource saturation or full-context qualification.
  The warmed aggregate is **112.2077 s prefill and 112.385 s client TTFT**.
  Source/result: `cache_map_prefill_speed117489_v1.decision` in
  [`fused-moe-service-current.json`](../bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json);
  cumulative row in [`performance-records.csv`](../bench/performance-records.csv).
- The closed adaptive8 comparison uses **a330 / engine429b** as the positive-swap
  candidate alongside the552b prefill fallback. Its bounded184-swap/32K-loan
  correctness entry and second-event-record refusal both pass. The six-process
  code cohort is closed:exact median TG is **10.1410 -> 10.7173tok/s**
  (static0 -> adaptive8),with all three paired TG ratios above one. Prompt/TTFT
  nonregression does not pass. The separate cap2/prose cohort is also closed:
  exact median TG is **8.3610 -> 10.5548tok/s**,with three favorable pairs.
  Its prompt/TTFT nonregression also fails. Cap2 acceptance does not establish
  a fixed4 weak-MTP class. Broader qualification remains pending;no inference
  adoption,new best TG or70TG follows. The e1b011 successor now passes its
  synthetic32-update positive entry and checked32-record-refusal. The new
  clean8/32comparison is closed with six regular-code processes and three
  opposite-order pairs on33210fresh input tokens. Exact median TG improves
  **11.79608->13.38632tok/s**;all three paired ratios exceed one
  (1.140356/1.131540/1.134581;geometric mean1.135486). Cache hits rise24.0->41.1%;
  generation is3016/3459tokens and acceptance83.8772/86.8056%. All six close
  normally0with quiet validation,closed owners,no fresh fault and unchanged
  boot/driver;all15model assets pass final full hashing. Prompt-time ratios
  0.991177/1.029460/1.007754and TTFT ratios0.991928/1.029290/1.008306fail
  their independent nonregression gates. This qualifies a repeated decode
  gain in this workload,not phase preservation,fixed-work attribution,full
  model adoption or70TG. The separate cap2/prose cohort now also closes all
  six processes normally0with quiet validation,closed owners,unchanged boot/
  driver,no fresh fault and all15final model-asset hashes. On33278fresh input
  tokens,exact median TG improves10.46625->10.72590;the three paired ratios
  are1.027211/1.023297/1.022849(geometric mean1.024450). Cache hits rise47.6->53.5%;
  generation is9854/7433tokens and acceptance63.1997/62.3062%at cap2.
  Prompt-time ratios0.983862/0.990009/1.016017and client-TTFT ratios
  0.984159/0.990145/1.016449fail both independent nonregression gates.
  This qualifies the repeated focused decode gain,not phase preservation,
  adoption,a fixed4weak-MTP class or70TG. Preserve the552b prefill fallback.
  The subsequent regular-code32/96cohort also closes all six processes with
  normal0,quiet validation,no fresh fault and15final asset hashes. Median TG
  improves12.61363->13.40847;all three paired ratios are favorable,with geometric
  mean1.051973. Prompt/TTFTnonregression gates fail independently. Select96only
  for the focused CPU leaf resource comparison,not prefill or whole-route
  adoption;preserve the552b fallback. Varied-prose32/96remains unprepared.
- The earlier **05a9 (`05a9b45c0a3a266779d4757a855b45a3bb42a62e`) / engine a99aa1**
  decode comparison is closed with six processes and three pairs:
  code median TG **13.2594 -> 13.5052** tok/s and weak-MTP prose
  **8.6795 -> 8.8299** tok/s (reference -> candidate). All three paired TG ratios
  exceed one in each workload; their geometric means are **1.017949 / 1.016802**.
  All 12 long requests pass, with normal exits, quiet validation, no fresh fault
  and final hashes for all 15 model assets. This qualifies reproducible decode
  gain, not the 70 tok/s target, inference adoption, dedicated prefill/TTFT
  preservation, agent/task quality, prompt-verify or physical-context behavior.
  Code prompt/TTFT directions remain mixed. Follow
  `verify_window_generation_speed_v1.decision`, evidence commit
  `dd685c75fd6e77be17f826672bcd3486a70cbdc0`, and the CSV for exact samples.
  The dedicated PP/TTFT comparison below is now closed; 117K preservation is
  inconclusive. Keep the fallback. The next CPU-service observer uses this same
  recoverable 05a9 source. The closed a99 binary is retired after source/recipe
  recovery and consumer updates; the 479a fallback remains available.
  Sampled-cache execution still needs its sampler/device admission and a newly
  qualified engine; this does not adopt the candidate's prefill path.
- `native_prefill_preservation_v1` failed during the first 65537-input request
  of the matched comparison
  of **552b versus a99aa1**, not just Q6 on/off within a99aa1. Source
  `tools/sycl/compare_native_prefill.py` fixes 12 processes: three opposite-order
  pairs at each of 65537 and 117489 exact input tokens, four fresh requests per
  process, first use separate from the subsequent-three median. Both arms use
  the throughput route and 20ms pool spin; the historical 1047 PP result used
  the pool default and is context, not the matched reference. Head retirement
  stays manually 0 at 64K and 1 at 117K in both arms. PP and client TTFT are
  separate decisions; the `ttft` strategy itself is not tested by this plan.
  Native inputs/fixtures and all 15 model assets were checked at preparation.
  The inherited 32K MTP checker incorrectly required every callback to be
  batched: at 64K, the declared 32768-cell MTP window leaves a 63-cell boundary
  that intentionally takes the token path. The request finished but the end
  gate stopped the benchmark; supervisor cleanup closed all owned identities,
  with no fresh driver fault. Keep V1 failed and its bounded parser fixture.
  A new prospective V2 must recognize only that source-derived boundary and
  reject other unexpected fallbacks before collecting another comparison.
- `native_prefill_preservation_v2` is closed with the same two engines,
  cases and runtime safety. It explicitly fixes the existing 32768 MTP window
  in both argument lists and checks the source-derived callback boundary.
  Five explicit shapes, 25 corrupt variants and the actual failed long-request
  window pass offline controls. All 15 model assets were hashed again; fresh
  frontend probes pass for both benchmark and server. All 12 processes and
  48 requests pass with normal exits, quiet validation, closed owners and
  no fresh fault; all 15 model-asset hashes pass again at final closure.
  All three completed 64K pairs support warmed prompt-time and
  client-TTFT nonregression separately: candidate/reference prompt-time ratios
  are 0.998945/0.995090/0.998678 and TTFT ratios are
  0.998943/0.993840/0.999426. Reference/candidate process-median PP is
  991.250/992.747 tok/s and client TTFT is 66.258/66.161 s. Each arm has three
  independent processes and nine subsequent requests; first use remains separate.
  All three 117K pairs are also complete. Their candidate/reference warmed
  prompt-time ratios are 0.995565/1.002483/1.001587, and TTFT ratios are
  0.995503/1.002465/1.001551. One pair favors the candidate and two favor the
  reference, so neither nonregression nor a consistent slowdown is established.
  Reference/candidate aggregate warmed PP is 1048.981/1048.455 tok/s and client
  TTFT is 112.180/112.225 s, each from three process medians and nine warmed
  requests. First-use PP remains separate: reference
  966.354/959.964/969.472 and candidate 968.437/967.808/968.894 tok/s.
  Successful execution does not satisfy the 117K performance-preservation gate;
  no inference adoption or sustained decode qualification follows from these
  short-output checks.
  V1 remains failed.
- The tuned 8192 DFlash2 route admission passes in `drafter_long_admission_v5`
  with one fresh native process and two exact 32,769-token thinking requests.
  Both read all tokens with zero reuse, generated 24, and accepted 18/24 drafts;
  PP is 584.7/709.2 tok/s and client TTFT is 56.193/46.395 s. The actual route
  is 2,046 DFlash2 tail rows with `cold_prefix=0`, 384 staged slots, 32K
  residuals on GPU, and 1,579 MiB free at startup against a 1,536 MiB floor.
  Validation is quiet; the process exits 0, all owners close, all 15 model assets
  pass final full hashing, and no fresh fault appears. These traced samples
  establish bounded 32K route admission only; they do not qualify clean speed,
  sustained decode, resource saturation, task quality or physical-context
  behavior. Preserve `drafter_long_admission_v4` as failed: its checker expected
  `cold_prefix=1` although both fresh resume-0 requests reported 0. The corrected
  V5 source/plan and both outcomes are in the linked result.
- Retain two explicit prefill strategies, `--prefill-strategy ttft` and
  `--prefill-strategy throughput`, within the same binary; selection is manual.
  Keep decode independently selected.
  Maintain the short-input path and avoid environment mutation after runtime
  initialization. No automatic length threshold is selected.
- The candidate's BF16 residual/interlayer storage does not make every hidden
  or intermediate tensor 16-bit. At 117488 rows, the HC bank's BF16 payload is
  `117488 * (4 * 2560) * 2 = 2406154240` bytes. Conditional PLE processing and
  final float callbacks still materialize chunk-sized FP32 HC residuals
  (`05a9b45c:src/prefill/prefill.cpp` and `src/prefill/layer_major.inc`).
  This HC payload differs from the MoE accumulator. Its FP32 split begins at
  65,536 of 117,488 rows; at width 2,560, its full payload extent is
  1,203,077,120 bytes, including a 531,988,480-byte FP32 tail
  (source `05a9b45c`, [accumulator plan](../include/strata/prefill/moe_accum_plan.hpp)). Keep local FP32
  accumulation/reductions, routing decisions/weights, and recurrent/online
  state distinct from storage dtype. A prospectively validated SGLang/vLLM
  numerical contract is acceptable; FP32/ggml remains diagnostic and does not
  impose its extra precision. Flags alone do not prove native-16 storage.
  In the current`591c2fbe3`build,the8192-row planner accounts for2,311,569,920
  scratch bytes,including a1,180,698,624-byte shared region. Source review at
  `src/prefill/prefill.cpp:1224`confirms that this region is the maximum of
  GDN,QSA and MoE layouts. Shrinking only a nonmaximum layout saves no allocation.
  Storage views also need individual checks:the old`gated`float reservation
  contains two disjoint BF16 arrays,and`Kc`contains both raw and rotated BF16
  keys,so neither reservation has a spare half. The chunk-sized FP32`R`still
  serves conditional PLE and float callbacks. The existing planner also covers
  the non-layer-major path used when rows do not exceed the chunk,even with
  BF16 enabled. A tighter long-prefill layout therefore needs an explicit
  producer/consumer contract while preserving that short-input path. These
  allocation facts establish no timing or hardware-saturation result.
- Q6 specialization is a decode-only candidate detail. In
  `q6_byte_load_component_v2`, 30/35 geometry/column combinations pass the
  performance screen: each candidate/reference median time ratio is below one
  in all three independent processes. These are the **30 eligible shapes**.
  Both arms include the byte-read safety repair; this measures column
  specialization, not the repair's effect. Separate `q6_columns_component_v4`
  reports 28/35 performance-eligible shapes. The
  explicit `STRATA_SYCL_DECODE_Q6_COLUMNS` selector (default `0`), phase/token
  cache ownership, and prompt-verification reference path are in
  `verify_window_policy_host_v3`. These
  component results do not qualify model TG or adoption; follow also
  `verify_window_generation_speed_v1`.
- Q6 native-half scale widening remains a **decode-only candidate**. The closed
  component at source `4947b7746dafb0efde7da9bb2ead49dc89790d89`, evidence
  `3e6b836a438b419dcd43e2eab68ba3cc3f7ea798:q6_native_half_component_v3`,
  passes32 numerical/guard/refusal cases and retains1050 timing samples from
  three independent processes.27/35 geometry/column cells are faster in every
  process;paired geometric-mean time reductions range0.435%..25.313%,median
  12.937%. These are resident-operand public-entry timings including queue
  completion under parameter validation,not model TG or saturation. The CSV
  retains all35 component result points and the closed model-comparison aggregates.
  The actual5c30-cell incumbent intersects26 winners;512x4 and all other
  ineligible cells retain their previous path. Native half requires columns1,
  centering0,explicit decode phase and an immutable default-off selector.
  Prompt/prefill selection stays unchanged. FP32 accumulation/reduction remains.
  The203-input UBSan policy fixture at `517a285094bffa8cd49168282f4ef813d9cbfbdd`
  passes;the source and closed result remain recoverable after its build retirement.
  The full model constructor at `ec4ab9a0edfaf33e7bf84c43778c0cc157dc0139`
  recovers the original566-file5c source and selectively composes567inputs,
  preserving all prefill,CPU and runtime bytes. Its1452 actual compiler inputs,
  recipe and six ELFs are checked;engine SHA256 is
  `cb7dd5aad4cb613dd98a1347e91eb75a157397f88040f18d5e26cf99056b0e15`.
  Its ended compiled products are now retired for the optional decode-gate
  replacement below;its measured source/recipe/results remain recoverable.
  Follow `q6_native_half_model_build_v1`. The old5c generation and normal
  build/host/device output are retired after original-source recovery;the
  immutable shared5c record remains a required provenance input.
  Both same-build CPU28-case observer modes and traced submission10,Q6-half32,
  phase-cache16 device gates pass with normal exits,strict before-main
  validation,closed owners,unchanged boot/driver,and no fresh fault. Original
  device evidence is `q6_native_half_model_device_v1` at source
  `2b1c8f6fe17893e47b8e1722f88d06866f5e650d`. The `ece575451` model-controller
  correction preserves the device/parser/global AST,all104 other controller
  pins,and all native/fixture/recipe/provider checks;it does not create a new
  GPU-fixture validation. Its CPU controls accept the real model-only change
  and reject six changed device/parser/import/definition controls. The unchanged
  Q6 snapshot reader passes2 positive and17 corrupt cases.
  The first spec8 half0 model entry remains failed:the unchanged scratch lease
  requires spec4,so native exit2/benchmark exit1 occurred before inference.
  Original protocol observation remains incomplete;independent reconciliation
  confirms all four owned identities stopped. No fresh GPU fault was recorded.
  Both corrected **spec4/scratch-lease1** entries now pass:two actual32769-input,
  thinking-on requests per arm,identical bodies and317source pins,the same
  native ELF/configuration/placement and only the immutable half selector
  differing after private-path normalization. Both frontends pass;native exits0,
  parameter validation is quiet,owners close,all15final model-asset hashes match,
  and no fresh fault occurs. Follow `q6_native_half_model_entry_arm0_v2` and
  `q6_native_half_model_entry_arm1_v2`. Both observed1/4-token decode windows
  complete their replays;the2560x248320 Q6 head selects columns in half0 and
  native half in half1. These are live-capture cumulative observations,not
  per-request deltas or retired history. Short-read0 does not exercise prompt
  verification. The actual model contains129Q6 tensors across seven matrix
  geometries;the521472000-byte head metadata is not physical bus traffic.
  **The closed model comparison below qualifies a repeated TG gain.** Main's
  `tools/sycl/q6_native_half_generation_speed.py` at `24b271b60` fixes the
  six-process0/1/1/0/0/1 comparison with thinking-on v5 code/prose. Its
  source is outside the original317model inputs;all317 remain unchanged.
  Its CPU decision/cache/parser controls pass via `python3 -B
  tools/sycl/q6_native_half_generation_speed.py controls`;no GPU is executed.
  Closed entry output is retired:165files,28364861bytes,with explicit hashes
  and original outcomes preserved in `q6_native_half_model_entry_output_retirement_v1`.
  Their useful outcomes remain committed. At generation closure,the next MTP
  admission needs only the full cohort record and original half0 record;
  obsolete arm configs/plans,provider clones,host output and the duplicate
  half1 receipt are retired. All567native sources and15selected executables
  are unchanged.
  The generation plan at source `03f5e0192` froze318source inputs,
  all15model-asset hashes and one invariant for the six normalized arm plans.
  Follow `q6_native_half_generation_speed_v1`:three counterbalanced adjacent
  pairs,thinking-on code/prose,at least32K fresh input and1K generated tokens,
  answer-body/actual-MTP/route/fault/ownership gates,and separate PP/TTFT.
  Both generation frontend probes and all six processes/twelve long requests
  now pass:33212/33282fresh input and3170/7275generated tokens including thinking.
  Reference/candidate median code TG is13.453906/13.710960tok/s,and prose is
  8.811746/8.980914tok/s. The three exact candidate/reference ratios are
  1.017834/1.017903/1.026035 for code and1.012887/1.023539/1.025999 for prose;
  geometric-mean gains are2.058%/2.079%. Each workload passes the predeclared
  all-three-pairs-faster screen. MTP acceptance stays82.44%/38.92% in every
  sample and all requests produce actual answers. All four paused gates and
  six cache/IQ3 footers pass per process,selected Q6 head replays complete,
  native exits are0,validation is quiet,ownership closes and no fresh fault
  occurs. All15model-asset hashes pass at final closure. This qualifies a
  sustained decode gain in this common audit/runtime envelope,not70tok/s or
  inference adoption. The32K prompt-time and client-TTFT ratios also favor the
  candidate in all three pairs;these are separate observations,not dedicated
  long-prefill preservation. The next width comparison may use half1 in both
  arms while preserving the fallback and the remaining gates.
  Closed evidence is `1b6e71f60e9590419857c9fffb8dab1ce19ce7b1`,key
  `q6_native_half_generation_speed_v1`;four CSV points retain reference/candidate
  code/prose aggregates with that evidence commit and the native source separately.
  Keep separate64K/117K PP/TTFT,handoff,
  agent/task,physical262144/tails/RESTORE and ordinary-provider-teardown gates.
  Preserve the validated552b fallback and current same-binary candidate;these
  are15 selected executables across two generations,not global cleanup.
- Two reviewed current-source decisions constrain the next decode work. The
  Q6 geometry bridge finds10/12observed T1/T4 cells applicable to the component
  winners;its synthetic public-entry timer does not measure graph-replay service
  or a physical bottleneck. A bounded post-cohort T4 attribution still needs
  lossless,unambiguous UnitTrace joins/counts and disjoint host/wait accounting.
  UnitTrace is not on PATH,but the existing `unitrace-indexed-collector-v1`
  tool is built from MIT Intel/PTI `4b6d74294cdc72766c00a9211fe2e4261ee94244`
  at source `e57b5ee7fce1de6391c793f1014c44cae3b85d9c`. Its launcher/library
  hashes still match6faac1e7/e7626aed. The host-only indexed-aggregation change
  and460compiler inputs are recorded;the local Level Zero/IGA headers already
  exist. Do not infer a missing build from PATH or acquire duplicate dependencies.
  Reuse the owned `unitrace_launch.py` admission with explicit Sysman0;the stock
  launcher forces Sysman1. Closed compatibility-graph and model-service checks
  exist for9922/b310,as detailed below;the research index's earlier static-only
  implementation note is not their final outcome. They do not qualify ec4's
  different fixture/engine ELFs. Reuse the maintained IQ3 collector/reader
  recipe with an explicit current-build basis instead of rebuilding the tool
  or repeating an already closed source investigation. Host build success alone
  does not establish current collector or model qualification.
  Main's new `tools/sycl/q6_half_collector.py` supplies that current-build
  active-fixture binding,with the existing launcher/readers/owner controls.
  It gates execution on all six frozen MTP-width processes closing,and checks
  the ec4 engine,fixture,providers,source and immutable runtime envelope afresh.
  The inspected no-argument compatibility fixture has62kernel executions
  (1direct+12first+24host-handoff+24cross-queue+1late);the submission-only mode
  is a different case. Preparation now pins114exact project/controller inputs
  at `b3b5387a95db2ddfadb8be612a07bdfb56923c2c`,the unchanged current engine/
  fixture,existing launcher/collector and runtime safety. The current-ELF check
  now closes successfully under `q6_native_half_collector_active_v1`:all62
  expected kernels have their names and unique native host-flow joins,with249
  GPU and770host events in the271452-byte timeline. Both raw-clock gates pass;
  before-main validation is accepted,native exit is0,owned processes close,
  boot/driver stay unchanged and no fresh fault occurs. The controller also
  exits0 after final source/provider checks and named-session removal. This
  qualifies the small active-from-exec collector only;pause/resume windows,
  model coverage,observer cost,physical traffic and resource saturation still
  require their separate prospective checks.
  Main's new `tests/sycl/compat_window.cpp` source-prepares the next pause/resume
  control using the actual CUDA-named SYCL adapter and checked graph-completion
  path. It records a two-kernel graph while collection is active,then executes
  2active,3paused,3resumed and2paused replays,with fully drained external gates.
  Expected coverage is4prefix+6resumed kernels out of20executed;names,unique
  host-flow joins,raw-clock containment and excluded gap/suffix work still need
  qualification. Every replay checks unsigned32 numerical results and both
  allocation guards. No environment changes or device spin loops are used.
  Both it and `tools/sycl/compat_window_build.py` are outside the frozen320/
  native567inputs. The new builder prepares four serial SYCL compilations of
  this fixture and the three exact selected runtime sources,then one link.
  It requires all six width processes and the current active62-kernel control
  to close first,recoverable native composition,the complete original compiler/
  package image and actual header/link closure. The build is now prepared
  from `7a1706d114e2cdd8af5abd533882747412d59ea5`,with117exact source/controller
  inputs,the original6239-file compiler/package image and four compile commands
  followed by one link. All five compiler commands exit0 with closed owners,
  but V1 fails its final closure check before any device execution. The driver
  transforms all four SYCL objects into private host-link inputs,so requiring
  their original names in GNU ld's trace is incorrect. Preserve V1's failed
  outcome. V2 source `087edccfd6f04398434d968cb2615595b2c952ca` now prepares
  the same four units with117pinned inputs and the same compiler image,adding
  verbose driver evidence and object hashes before/after link. It requires the
  original-object inputs at that driver boundary as well as ld's transformed
  inputs. V2 now builds successfully:all five commands exit0 with closed owners,
  739actual input paths are qualified,including four unchanged driver objects,
  2720dependency entries,42host-link entries and five derived private inputs.
  The289696-byte fixture SHA256 is
  `cb8cbb6596829b5488b29170a35be0936227f145d4cc5a2d2ce8bfaddf1f6116`.
  Compilation does not qualify its device or collector behavior. The source-only
  `tools/sycl/compat_window_check.py` now prepares the three separate smoke,
  prefix-only and resumed-window controls with the existing owned launcher,
  strict validation and original providers. It checks20actual numerical/guard
  kernels,4/10recorded named kernels,unique host joins and excluded gap/suffix
  intervals. Preparation at `d15525f286862d35da76cc0b3549ba7c0f068e09` now
  passes two positive and seven corrupt reader controls without GPU execution.
  The first separately owned smoke case now passes all20kernel numerical/guard
  checks,normal native0,strict before-main validation,closed ownership,unchanged
  boot/driver and no fresh fault. Prefix-only also passes:20kernels execute
  correctly,but only the four named prefix kernels are recorded,with unique
  native host-flow joins and all GPU events inside the active raw-clock
  interval. The paused gap/window/suffix work is excluded;native exit0,
  validation,ownership and no-fresh-fault gates pass. The resumed-window case
  also closes successfully:20kernels execute,exactly4prefix+6resumed named
  kernels are recorded,with unique native host joins and all16GPU operations
  inside active raw-clock intervals. The paused six-kernel gap and four-kernel
  suffix are excluded;the41069-byte trace has199host events. All three cases
  pass numerical/guard,normal-exit,strict validation,closed-owner,unchanged
  boot/driver and no-fresh-fault gates. This qualifies pause/resume only at
  fully drained fixture gates after active graph construction. It does not
  qualify asynchronous toggling during in-flight model work,model coverage,
  observer overhead or physical resource saturation. The next model plan must
  preserve that distinction rather than assuming arbitrary timed toggles are
  lossless. Keep its current model-attribution consumer and source/recipe
  recovery separate from the disposable fixture's binary/raw retirement.
  The closed V1 failure is recoverable at `730726403c58e3ca40176469cd1ef9e7c933a799`;
  its superseded full result is removed from the active JSON. The closed V1
  outputs and successful62-kernel
  control's raw trace/JIT/logs are retired after original117/114controller and
  567native-input/recipe recovery:31files,1391544bytes,manifest SHA256
  `7282ab4cfb8b0c5cdb61031430a3f8b0653898495e9a52bad429a477a32eb89d`
  in external `compat-v1-and-active-collector-output-retirement-20261106.json`.
  The V2 build and all three checks are closed. After exact117/118controller
  recovery,the selected567native composition/recipe and all six native ELF
  checks,58files/1271414bytes of disposable V2 executables,objects,dependencies,
  successful traces/logs/JIT output and the superseded status-only IPC diagnostic
  are now retired. External manifest
  `compat-window-and-superseded-ipc-output-retirement-20261106.json` has SHA256
  `60bbe477260e4fde02006b1a4d7e04f5f30105be6ce262d985812aaca28e485e`.
  Keep only the required V2 build/check records and both selected engine
  generations. This fixture is not another engine generation or current model
  attribution.
  The optional native decode-gate diagnostic is closed. Main-authored source
  `a388c63f70b1f33363b6e8e51293f6989791ba09`changes only the selected generator
  plus its default-disabled host gate header. Its45-case host fixture passes;
  all five native component fixtures are byte-identical to the previously
  qualified generation. The actual519e engine reaches0/8/16/24completed rounds,
  emits128bounded tokens from1024inputs,and exits0 with normally closed ownership.
  Its Python3.12 controller then segfaults in `PyObject_GC_Del`during post-run
  processing. The exact statement/cause remains unknown and that run stays failed.
  All original processes are absent and no new GPU fault is recorded.
  The separate CPU-only replay,source
  `5e18110fcc3bcabaf1fc45c946fcb257cf4a29e1`,now exits0. It verifies the saved
  protocol,four gates,token hash,strict pre-main UR evidence and completed T1/T4
  Q6 heads,recomputes all six resource snapshots,and fully hashes all15model
  assets again. Follow `decode_gate_model_entry_v3_postcheck_v1`,original closed
  evidence `3ceff37475b397f9154bf95040cf7497fff7afcc`. This verifies retained
  native gate evidence without another GPU run;it does not fix Python,qualify
  clean speed/task quality,ordinary provider teardown or physical262144/RESTORE.
  The superseded INFO and RESUME controller failures and the host-crash outcome
  are archived unchanged at `5e18110fcc3bcabaf1fc45c946fcb257cf4a29e1`;their
  duplicate per-version configurations are removed from the active result.
  Between rounds8..16 and16..24,process CPU use is5.0907/5.2786core equivalents
  and CCS busy-cycle fractions are0.543462/0.562392. All six boundary snapshots
  show requested client VRAM fully resident;decode reaches10577268736bytes.
  CPU includes polling,GPU busy includes stalls,and intermediate paging is
  unmeasured. These short synthetic observations establish no bandwidth ceiling,
  physical saturation or clean PP/TG. Next revalidate the previously faster owned
  positive-swap cache path to reduce CPU misses;its payload/map/actual dispatch
  and subsequent occupied scratch-loan/refill checks still need native execution.
  The ended519e build/source/temp/cache and superseded protocol-failure raw output
  are retired after recovering568native inputs,110controls,9311third-party source
  inputs and the exact recipe from pushed Git:814files/182274867bytes.
  External manifest`decode-gate-diagnostic-retirement-20261106.json`has SHA256
  `30b82fba1a35ab73723a49259c4b1b964f182241023fae64fe7fbe7331bd3f07`.
  Keep the552b fallback and all nine ELF outputs,original required records,and
  one minimal host-Python failure diagnostic. No519e compiled generation remains.
  The protocol reader is now maintained in`native_request_protocol.py`,with
  four passing CPU tests for the imminent positive-swap entrypoint. The closed
  `decode_gate_entry.py`,`inspect_decode_gate_entry.py`and`profile_decode_gates.py`
  CLIs are retired(3files/45567bytes);recover them from
  `c7f50a3456656d05c3d7fd35e58cc16fdfa2544e`.
  Their paths are outside both current build/admission input graphs. Keep the
  separate minimal host-Python failure diagnostic;its cause remains unresolved.
  The completed Orca Luna/native Luna-max writer review finds no supported
  bounded live-reduction route:the pinned PTI writer writes one JSON document
  through a buffered stream,and the owned reducer loads the whole file before flow joins.
  `UniLogger` also calls `tellp()` in its emptiness/finalization contract;
  non-seekable FIFO behavior and reader-failure handling remain unqualified.
  The closed24-output trace's348285412bytes are the only measured size datum.
  Temporary report/review files under `.cache/research/2026-11-06/unitrace-stream-window`
  (SHA256 e879f20b/241894d5)serve only the post-cohort current-collector design
  decision;curate the needed source findings and retire those bodies there.
  Current-ELF qualification and finite output budgets still apply.
  The old tool receipt's Xe-named loader path is historical
  `original_assets.preload.loader.path` metadata. No such path occurs in its460
  compiler inputs or system-only ELF dependency list. The current control
  constructs loader expectations from its selected native validation probe;
  actual current provider mappings pass the closed active-fixture gate.
  New Orca research remains on hold. The bounded workspace-profile status
  probe exits0 but reports `runtime.reachable=false`,disconnected and
  `stale_bootstrap`;this is not a successful worker IPC check. Main's ordinary
  status independently reaches the ready runtime. No new researcher,task,
  dispatch or terminal was launched from that failed capability result.
  A new bounded exact-socket probe distinguishes the failure:metadata is
  readable and names the expected0600/UID1000Unix socket,but `socket(AF_UNIX,
  SOCK_STREAM)` itself returns EPERM before `connect()` can run. The target
  PID probe returns ESRCH in that sandbox. This does not test the socket path's
  connect permission or Orca RPC. Keep its small diagnostic under external
  `orca-narrow-ipc-connect-probe-20261106` for the current narrow sandbox-policy
  question,with a256KiB total budget and review after that question. The older
  status diagnostic is superseded and retired. A fresh
  finite native Luna-max review identifies a macOS-only Unix-socket proxy
  limitation in the local Codex0.162.1binary,consistent with the Linux denial.
  Read-only follow-up now finds no environment store in either the default
  Orca data root or its mapped active-profile root. Installed Orca1.4.229has
  a paired WebSocket RPC client,but its supported `serve` path binds all
  interfaces and starts a separate runtime sharing the desktop profile lock.
  `--pairing-address` changes only the advertised address;it cannot provision
  an in-place loopback-only listener. The raw-json-v1 hook is notification,
  not status/lifecycle RPC. No app,profile,listener or worker sandbox changed.
  The installed-source decision is curated at
  `b1d255bde0c282b494c44aa7bdf8311e4ade0547`and here. Three superseded
  coordination bodies are retired after curation(12115bytes);external
  manifest`orca-transport-review-retirement-20261107.json`has SHA256
  `0d866c8b5f0cfdb3b686bfb10afdc09957694f9c7704aeb0251e5d69841b009f`.
  Its installed-source references are `cli/specs/serve.js`and
  `cli/runtime/serve-orcad-launch.js`in the immutable Orca1.4.229Nix package.
  Keep dispatch pending until a supported narrow worker transport exists;
  review on an Orca transport/version change,not another identical socket or
  missing-store probe. Worker-side connectivity and lifecycle settlement must
  still pass before research resumes. Main's tuning continues independently.
  On the current greedy linear MTP route,adding only a second token candidate
  duplicates the existing fallback correction rather than adding useful accepted
  continuation. Retain linear MTP;reopen only for a concrete branch-aware proposal
  and state/rollback design. This does not reject tree-aware verification in
  general. Exact source/report/review pins,limits and the temporary administrative
  index-edit exposure are curated in the
  [research index](../bench/results/2026-10-15-pr6-recovery-audit/research-index.json),
  keys `q6_replay_component_bridge_20261106` and
  `qwen38_weak_mtp_sibling_source_feasibility_20261106`. No new speed claim follows.
  The reviewed current fixed2/cache-lease delta has a conditional source pass,
  not runtime or speed qualification. Keep `--spec 4`,change only the planned
  `--mtp-max-t 2` selector,and hold min-p/suffix controls. The new controller must
  require prefill borrowing and a positive single-cache tail loan,then verify
  actual T1/T2 routing,graph-reader retirement/recapture,borrower lifetime and
  state handoff before model speed collection. Loan-driven graph retirement
  applies even without optional dense/head release. Use the existing fixed2
  short-route evidence only within its original scope. Corrected source anchors
  and review pins are in `mtp2_current_cache_lease_source_feasibility_20261106`
  in the research index. Fix the input workloads and classify strong/
  weak MTP from the matched fixed4 reference;report fixed2 acceptance and yield
  separately because reducing draft depth changes the aggregate's denominator.
  A candidate crossing the reference's acceptance-class threshold must not
  retrospectively exclude its speed sample. Commit final criteria and controls
  before this new experiment;do not change the active Q6 comparison's rules.
  Main's bounded-entry and generation controller at
  `19b43827ae503e58123fa43aaf4492a40213f285` is now integrated into feature/sycl
  by merge `4c24b6931e9090538cb400153856704f9b33eb00`. The pushed merge preserves
  original source/test commits;the completed child branch/worktree is retired.
  It preserves spec4 allocation,
  compares explicit cap4/cap2 with min-p/suffix disabled,and checks actual
  traced widths against completed live Q6 captures. It requires a positive
  occupied tail loan,checked resident-prefix/refill payload,ordered owner
  retirement/restoration,and renewed loan/capture identities across requests.
  It explicitly admits only the tested startup-config frontend delta plus its
  new controller/test inputs;fresh frontend probes and both model entries are
  required. Full host-only controls now pass at the declared19bsource:
  lease1positive/12negative,
  source1positive/3negative,and width4positive/24negative,exit0,without native/GPU
  execution. Exact command in the child was `env PYTHONDONTWRITEBYTECODE=1
  PYTHONNOUSERSITE=1 prlimit --as=1073741824 --cpu=60 --fsize=65536 -- timeout
  --signal=TERM --kill-after=5s 90s /home/yayoi/.local/share/strata-sycl/venv/bin/python3
  -B tools/sycl/check_mtp_width_model.py controls`. Integration followed the
  pushed closed Q6 evidence. These parser controls alone do not qualify the
  fresh frontend/model entries recorded below,or a new native build or speed.
  The later source adds a separate `speed` mode for six fresh4/2/2/4/4/2
  processes with the fixed v5 thinking-on long workloads. It requires both
  bounded entries first,disables entry-only trace/payload audits,and records
  individual TG,PP,TTFT and depth-specific acceptance. Only fixed4 establishes
  the MTP class;all three speed ratios must exceed one for a per-workload gain.
  Generation/decision controls also pass:changed candidate acceptance denominator
  and small gains remain eligible,a slower pair and a misclassified reference
  are distinguished. The actual generated cap-pair invariant,immutable entry
  template,preparation-time common-half selection and disabled entry observers
  pass;missing selection,changed runtime safety and prefill drift are rejected.
  The reused protocol controls also pass,including2positive/17negative Q6 parser
  cases,5invalid cache footers,5invalid pre-generation histories,9invalid workload
  contracts and4disabled-observer cases. These are CPU-only controls,not new
  native/device/model or speed qualification.
  The fixed4 entry is now prepared at `826df22f565805cbb30c4be0966f9e764256f680`,
  key `mtp_width_model_entry_cap4_v1`:320frozen inputs,all15model assets,
  the unchanged cb7/ec4 generation and common half1. Only the declared startup
  frontend delta and controller/test additions extend the original317inputs.
  Both fresh frontend probes now pass with the new startup import:benchmark
  144modules/18code maps and server191modules/26code maps,normal exits and
  closed ownership. The fixed4 model entry also closes successfully:two actual
  32769-input/24-output thinking-on requests,normal native0,quiet validation,
  closed owners,unchanged boot/driver,all15final model hashes and no fresh fault.
  Both requests lend1124slots starting at132 of1256,with2321664000B capacity
  for2319958528B needed. The native audit checks all1256resident expert payloads
  (2575206400B),including the unlent prefix,after refill;masked/ready admission
  and owner phase ordering pass. Loan generations2/3 and disjoint capture IDs
  3/4 then5/6 prove renewal;each request has oneT1 and sevenT4 completed replays
  matched to its eight traced rounds. Follow `mtp_width_model_entry_cap4_v1`.
  These logged short-output checks qualify this reference route/lifetime only;
  no clean or sustained TG,task/full-context or ordinary-teardown claim follows.
  The cap2 entry is prepared at `e2fc5f7050ec17fee0353a4b9eb6fdb9d8e1166f`,
  key `mtp_width_model_entry_cap2_v1`,with the same320source pins,15model assets,
  native binary and normalized configuration/environment except the cap.
  Both fresh cap2 frontend probes pass:benchmark144modules/18code maps and
  server191modules/26code maps,normal exits and closed owners. Its bounded model
  check also closes successfully:two32769-input/24-output thinking-on requests,
  the same request-body hashes as cap4,normal native0,quiet validation,closed
  owners,unchanged boot/driver,all15final model hashes and no fresh fault.
  Both requests return the same1124-slot loan and checked1256-slot payload as
  cap4. Each completes oneT1 and twelveT2 replays;loan generations2/3 and
  disjoint capture IDs3/4 then5/6 renew across requests. Width/lifetime checks
  pass;this is neither sustained TG nor recurrent-state/task/full-context
  qualification. The sustained comparison is now prepared at
  `83909a4eb2cb17a418e505edec46f7c5a14e3052`,key
  `mtp_width_generation_speed_v1`:320frozen inputs,the same15model assets and
  six native ELFs,common half1,and cap order4/2/2/4/4/2. Only the cap differs
  after private-path normalization. Four official warmups precede each process's
  fixed thinking-on v5 code/prose requests. Trace,payload audit,CPU attribution,
  DECODE_TIMING and profiler are disabled;IQ3/Q6 audits and runtime safety stay
  common. Classify MTP only from fixed4;keep fixed2 acceptance at its changed
  denominator and retain every speed sample. Each workload needs all three
  paired TG ratios above one for a repeated gain. Both fresh generation frontend
  probes and all six processes now pass with both long requests eligible per process,
  normal native exits,quiet validation,closed owners,unchanged boot/driver and
  no fresh fault. Actual T1/T4 or T1/T2 and selected Q6 heads complete;all four
  paused gates[4,5,5,6] and six cache/IQ3 footers pass per process. The original
  individual samples,body hashes and route evidence stay in the cumulative
  result. Fixed4 establishes MTP classes at acceptance.824388/.389236;keep
  fixed2's changed denominator and different generated lengths.
  All15model-asset hashes match again at final closure. Cap4/cap2 median TG is
  13.811709/12.247815tok/s for code and8.985282/10.647958 for prose,each from
  three independent processes. The three cap2/reference ratios are
  .886770/.874191/.899623 for code and1.238236/1.178238/1.188583 for prose.
  Geometric-mean change is-11.320%/+20.140%;all three prose pairs are faster
  and all three code pairs slower. The per-workload prose gain qualifies;
  the aggregate both-workloads gain flag is false and must not discard that
  separately qualified option. The existing same binary exposes cap2/4 through
  `--mtp-max-t`;keep cap4 for the strong-MTP reference and cap2 as the faster
  weak-MTP comparison option. Automatic selection and inference adoption remain
  unqualified. Prose prompt/TTFT directions are mixed,with cap2 slower in two
  pairs;dedicated64K/117K PP/TTFT preservation,handoff,agent/task,
  physical262144/tails/RESTORE and ordinary provider teardown remain open.
  The cap4 reference's completed target windows are1xT1+913xT4 for code and1xT1+3357xT4 for prose:
  3653/13429token positions,or1.152366/1.845911per generated token. Dividing
  those extents by whole decode time gives15.916143/15.840919positions/s.
  This descriptive work normalization supports checking rejected verification
  work with cap2;it excludes separate draft work and is neither a target-only
  service timer nor FLOPs,physical traffic or a limiting-resource measurement.
  Cap2 completes3825/10363positions,or1.064866/1.231053per generated token and
  13.042286/13.081261positions/s of whole decode. Both workloads process fewer
  verification positions per output,but positions per whole-decode second also
  fall. Keep round overhead,draft work,shape-dependent compute and weight reuse
  unresolved until source-matched attribution;the ratios do not identify hardware saturation.
  A second normalization uses completed target rounds `R` and positions `P`.
  Whole-decode milliseconds per round are251.110941/153.307266 for code and
  252.454199/152.875704 for prose (cap4/cap2). Generated tokens per round are
  3.468271/1.877679 and2.166468/1.624469. Thus cap2/cap4 cost ratios
  .610516/.605558 combine with yield ratios.541388/.749824 to reproduce the
  opposing TG directions. This is accounting,not an independent causal finding.
  At evidence `ef5d26fc18bbc6510c7a65c7b2d9a93f5c3a9363`,two-point descriptions
  are `D_ms = 55.394057*R + 48.969404*P` for code and
  `D_ms = 53.271936*R + 49.806690*P` for prose. They interpolate the first pair
  exactly by construction;linearity and physical fixed/per-token costs are
  unproven. Separate draft work,scheduling,first/tail work and both devices
  remain inside `D_ms`. Process2's held-out prediction errors relative to
  measured whole-decode time are-1.246454%/+0.205901% for code/prose. These are
  repeatability checks at an already observed width,not validation of linearity,
  unseen widths or different contexts. Process3's errors are+0.174541%/+5.308524%:
  prose shows a larger variation that the first-pair description does not explain.
  Process4's errors are-1.242242%/+4.703693%;process5's final held-out errors
  are+0.189086%/+0.505151%,without refitting or changing cohort decisions.
  This completes same-width repeatability checks;the larger cap4 prose residuals
  remain unexplained and no unseen-width/context prediction is qualified.
  Successful raw output/JIT from all six processes is retired after pushing
  their samples and final decision through `dcb8ba8cff3d6618c19a8874a218112832b0ded3`,
  recovering all320original inputs from Git,and checking closed ownership:
  414files and78934883bytes (69files per process). Native-source recovery also matches
  all567materialized inputs and the exact recipe through
  `build_q6_native_half_model.py:projection`:this build is an explicit historical
  composition,not the plain ec4 controller-commit tree. Required inputs match
  before/after retirement;the next consumer still requires full unchanged native
  admission. Keep all12generated config/plan files and global cohort/provider
  inputs for the current-ELF collector and subsequent fixture admission. External deletion manifests
  `mtp-width-generation-process0-output-retirement-20261106.json` and
  `mtp-width-generation-process1-output-retirement-20261106.json` have respective
  SHA256 `7e8d87edafaa5208fc46ad26f7747a1e2de975195782f8d9514cb3244dd6c0e4`
  and `a07ae9e6ab4e024f74219b97ecd80539f2b4439519736c0636e6da6ee22028d3`.
  Process2's `mtp-width-generation-process2-output-retirement-20261106.json`
  has SHA256 `c01e2b4b9441cf9f705af788a323adcceb066bb0c682b57bf70b204aff00d370`.
  Process3's `mtp-width-generation-process3-output-retirement-20261106.json`
  has SHA256 `f1ef8414a6320a559843aedd2b952cdcde75e465fcc178fa5cfbd4921f8008b2`.
  Process4's `mtp-width-generation-process4-output-retirement-20261106.json`
  has SHA256 `9cc8d14ac348c0b795d2aef498fecb71155dc74fdb348ee43c28e68414a1d351`.
  Process5's `mtp-width-generation-process5-output-retirement-20261106.json`
  has SHA256 `c30da282359f9723a1640ea75758b54f502ff9e82cfbe1602611d9d9efdb2e50`.
- Current ec4 adaptive publication updates the private `ExpertCache` residency
  snapshot as well as host/device tables. The older20f2 stale-private-map finding
  is not a current-source defect. Closed static-swap0 model entries and long
  generation remain separate evidence;they do not qualify positive replacement.
  The next bounded positive-swap plan must exercise actual replacements,identity
  and payload agreement,checked completion/publication,and a subsequent occupied
  scratch loan/refill before clean speed comparisons. Preserve the checked
  refill-failure control and runtime safety;do not require identical generated
  text or turn the old answerless prose failure into a pass.
  Main's `AdaptiveResidencyAudit` passes its41-case host fixture with normal0
  and closed owners. It checks independent host/device/private
  identities,real replacement tuples,old/new payload callbacks and checked
  completion before publication;failures and callback exceptions prevent reuse.
  `tools/sycl/adaptive_residency_host.py` pins the full original compiler image
  and111exact project/controller inputs for this finite host-only compile/test
  at the closed process boundary. Actual compiler inputs match the image;
  the ELF has no GPU-provider dependency. The frozen320-input cohort and both
  engine generations are unchanged. A host pass cannot qualify real copies,positive model swaps,
  native lifetime or speed;native integration remains the next consumer.
  Its one-use executable and successful output are retired after consuming the
  host test and recovering all111inputs/recipe from pushed Git:10files,170369bytes.
  Follow `adaptive_residency_audit_host_v1` for the source,result and deletion pins.
  `tools/sycl/adaptive_residency_sources.py` source-prepares the native serving
  attachments against the exact b4423da4 generator and tested assertion header.
  It adds only an opt-in,single-GPU/full-RAM-arena diagnostic:pre-mask victim
  bytes,three map views,successful refill-record generation,checked completion,
  incoming bytes and publication. The legacy record counter advances only in
  failure-injection mode;the attachment uses a separate success-generation stamp.
  Its source now also attaches an opt-in check after the actual window pool:
  conserve routed GPU/CPU entries,verify every GPU plan slot/pointer/destination
  and every CPU job's blob/token/output mapping,and count replaced incoming
  GPU hits and outgoing CPU misses. The callback runs after pool completion and
  before the next adaptive worker starts;previous workers are joined before the
  next window. Audit mode requires the existing full-payload scratch-loan checks,
  then checks all three maps and decode readiness after a post-swap occupied
  loan/refill. Final audit-only publication drains outstanding replacements and
  emits compact cumulative coverage. No core/kernel source changes are planned.
  The extended transformation now constructs568native files successfully at
  source`9541e37c621990f67a7c2a1d6f0838ce8a6ee204`. Only the generator and audit
  header differ;all566other files match. Generated generator SHA256 is
  `5049d431471250de41cef0d89becb696ab26529b2dab331192f1ced97179c648`.
  `tools/sycl/build_adaptive_audit_model.py`now source-prepares its separate
  native build using the original six targets,full compiler/package/SDK images,
  recovered third-party inputs and the physically verified519e retirement.
  It preserves552b and all nine fallback outputs. Preparation now passes from
  builder source`a330eb7fbd300d6f2755be51f8419e33ab3fd89e`,pinning568native
  inputs and the original full compiler/package/SDK/dependency images. Commit
  the prepared plan before compilation. The build now closes normally:
  configure/compile/dependency commands exit0,1453actual compiler inputs and
  the original optimization/contraction flags pass. Engine SHA256 is
  `429b69f2984b06db36fde39eb4f1a0e39b8e4dbec406169b35a3c96db78106bd`.
  All five component fixture ELFs are byte-identical to the original cb7 outputs.
  The new `adaptive_native_admission.py`source checks explicit reuse of the
  original real-pool28+28and device10/32/16checks with unchanged providers,
  validation probe,boot/driver and recoverable original controls. This is not
  another GPU test and cannot qualify the new generator's adaptive attachments.
  This reuse check now passes at source`c7f50a3456656d05c3d7fd35e58cc16fdfa2544e`:
  all206original control inputs are recovered from Git,the five actual fixture
  ELFs and84providers match,and validation-probe/boot/driver identities are unchanged.
  Follow`adaptive_residency_native_build_v1.component_reuse`. Native positive
  replacement,actual dispatch and post-positive loan/refill remain the next gates.
  The new Main-authored`check_adaptive_native_entry.py`source prepares one direct
  native process with1024/256then32769/256synthetic requests. Positive8swaps
  every8rounds,all map/payload/actual-dispatch checks and the occupied-loan audit
  are fixed before exec;other placement/kernels/safety stay unchanged. It uses
  the maintained wire reader without replacing Interactive.poll,and preserves
  native closure before post-processing. Its ordered reducer has one complete
  prospective fixture and16corrupt generation/tuple/order/coverage cases;all
  controls now pass at source`b9c952b147b3e6be996ce30366534b0541ca32c3`without
  GPU execution. Preparation also passes with all15model assets fully hashed
  and the exact engine429b/environment/two requests pinned. Commit this plan,
  then run once. The native entry now closes successfully:both256-output
  requests complete with normal native/controller0,quiet strict validation,
  completed T1/T4 Q6 heads,closed ownership,unchanged final15assets/boot/driver
  and no fresh GPU fault. Across23published groups,all184actual swaps pass
  victim/incoming payload,three-map,completion/publication and real GPU-plan/
  CPU-job conservation. Request1has7213incoming GPU entries and53outgoing CPU
  entries;request2cumulative counts are19004/399. The32Kprefill lends/refills
  1124occupied slots with matching payload/maps/decode readiness,and incoming
  GPU dispatch increases after that loan. Follow`adaptive_residency_native_entry_v1`.
  This establishes the bounded positive path and post-loan dispatch,not model
  quality,clean PP/TG,physical262144/RESTORE or inference adoption. Checked
  record-failure injection is now closed in a separate native control before the
  clean static-versus-adaptive bench.sh comparison. The Main-authored
  `check_adaptive_record_failure.py` uses the
  same429b engine/configuration and one1024-input/max256-output request. The
  existing native injection calls the real compatibility API with a null event
  on the second record,after accepted positive copies;the first event remains
  intact. Require generation1 published,generation2 masked but never published,
  exact adaptive-refill ERR without DONE,and owned native exit1 without cleanup,
  quiet validation,unchanged final assets/boot/driver and no fresh GPU fault.
  At source`31cdb841de048f95bf919944a19992cf3c51ec8d`,its one valid reducer
  fixture and21corrupt variants pass. The committed native plan now passes:
  generation1publishes8swaps;generation2has8accepted copies and a masked map,
  but its real record returns1 and no second publication occurs. The request
  emits45tokens,then the exact ERR without DONE;native exits1 as expected with
  normal closed ownership and no supervisor cleanup. Strict validation is quiet,
  all15final model hashes match,boot/driver remain unchanged and no fresh GPU
  fault appears. Follow`adaptive_record_failure_native_v1`. This expected error
  control supplies no throughput,task,ordinary-teardown or full-context claim.
  Preserve the fast552b prefill
  fallback and assess its PP/TTFT independently of any decode gain.
  After pushing the useful result at`dca625eac0fdec4c2b47cbd524c0ca73a3c3399f`
  and recovering all115controller inputs,the normal raw streams,poll receipt,
  empty validation log and temporary/JIT products are retired:59files/13590344
  bytes. External manifest`adaptive-native-entry-success-output-retirement-20261106.json`
  has SHA256`5031d7415120d69fea490918d56aaf3dcf35479d1e156e7b8f25bd0388506226`.
  Keep the original record and both552b/429b generations for the imminent
  clean comparison;the separate Python host-fault diagnostic is unchanged.
  The expected-refusal control's successful full streams,poll receipt and JIT
  are also retired after source recovery:50files/12562814bytes,with all15
  comparison ELF outputs unchanged. Its external manifest is
  `adaptive-record-failure-success-output-retirement-20261106.json`,SHA256
  `6406c2d021bde38c029f35e80b46c4864070447ba5eba2698807e5acff013150`.
  The Main-authored`compare_adaptive_generation.py`at source
  `7378e89dc0baa63f1acde7c8a2e95d4ad9eee442`compares same429b static0/adaptive8
  through maintained bench.sh. Each focused workload has six fresh processes
  R/C,C/R,R/C,four official warmups and one thinking-on32K+/1K+output request
  with an actual answer. Code uses cap4;prose uses cap2 in a separate cohort.
  Heavy correctness observers are off;qualified runtime safety and light route
  footers stay. The332pinned inputs include the initial expert profile,which
  the unlaunched V1 omitted. V1 was cancelled before any model/frontend run;
  its original preparation is recoverable at`3b7756d0241e02c509ab5cdcc45aa5a7752297b1`.

  The V2 code cohort is now closed. All six processes pass their actual body,
  native-width/head,clock,journal,validation and ownership gates,with normal0,
  unchanged boot/driver and no fresh GPU fault. All15final asset hashes match.
  Every long request reads33211fresh tokens;reference/candidate generate
  3157/3293tokens with acceptance.837597/.820028. Cache hits are10.5%/24.3%.
  Supplementary exact TG was declared before any candidate at
  `65c192bdea6e6d5d8066af1a95df9e2fc06be214`as generated_tokens*1000/decode_ms;
  preserve the original one-decimal controller decision separately.
  Reference exact TG is7.294485/10.141007/10.324376tok/s and candidate is
  10.717266/10.630499/11.553421. Their medians are10.141007/10.717266;
  paired candidate/reference ratios are1.469229/1.048269/1.119043,geometric
  mean1.198955. All three favor the candidate,without a minimum gain floor.
  The original rounded ratios also pass;no sample is excluded. The wide
  reference spread remains unexplained,so do not treat19.90%as a general gain.
  Candidate/reference prompt-time ratios are.972723/1.000189/1.040591 and
  TTFT ratios are.971933/1.000336/1.039978. Both independent nonregression
  decisions fail;median PP/TTFT alone cannot conceal the two slower pairs.
  Dedicated64K/117K preservation,agent/task quality,physical262144/RESTORE
  and ordinary provider teardown remain unqualified. The previous13.811709
  code result remains above this cohort's candidate median. Its input count
  differs by one and its actual placement-profile file was not separately
  pinned,although current and ec4Git profile bytes match. Keep this historical
  decrease unresolved and Main's further work on tuning/current attribution.
  The independent cap2/prose cohort is now prepared at source
  `a6a8e863ebf41c04f173eb3d916b36d1be752320`,with the same332pinned inputs,
  six-process ordering and15asset hashes. Its supplementary exact-TG plan
  is declared before even the first reference. Both fresh benchmark/server
  frontend probes pass. The six-process prose cohort now closes normally,
  with quiet validation,closed owners,unchanged boot/driver,no fresh fault
  and all15final asset hashes matching. Reference exactTG is8.073058/8.749193/
  8.360982,candidate is10.554848/10.609532/10.105522tok/s;their medians
  are8.360982/10.554848. Three paired TG ratios are1.307416/1.212630/1.208652,
  geometric mean1.242075;all exact and original-rounded pairs favor
  adaptive8. Candidate/reference prompt ratios are0.950961/1.013803/1.010724
  and TTFT ratios0.950443/1.013293/1.009552. Both independent
  nonregression decisions fail;retain all samples and the original outcomes.
  Every request reads33279fresh tokens;reference/candidate generate8286/10147
  with6.8%/46.4%cache hits. Actual cap2/heads/body/clock/journal gates pass.
  Cap2acceptance uses its own denominator and establishes no fixed4 weak class.
  Follow`adaptive_generation_varied_prose_precision_analysis_v1`. Keep552b as
  the fast prefill fallback;these focused gains do not qualify whole-model
  adoption,other lengths,agent/task,physical262144/RESTORE,ordinary provider
  teardown or70TG.
  The next bounded32-swap entry is source-prepared in
  `tools/sycl/check_adaptive_budget_entry.py`. Its CPU reducer now passes a
  valid32-group fixture and22corrupt variants at source
  `409ad93c554ef984f7522c4aecff75ef84fc9909`;the committed20-second locked
  command exits0 in0.726s. The native32entry is now prepared with all15asset
  hashes. Its first execution fails the qualification:three full32-copy groups
  publish96swaps with three-map/payload checks,but the fourth group hits the
  unchanged64MiB cumulative auditor limit before prepare/mask/copy. Native
  emits ERR;the supervisor terminates it(exit-15). The original normal-exit
  and ownership gates remain failed,although the child is reaped,the session
  is empty and pinned identities are gone. Boot/driver stay unchanged with no
  fresh GPU fault. Actual layer blobs are1,510,400..2,662,400bytes,so32copies
  can total48,332,800..85,196,800bytes. This is an auditor-cap refusal,not
  a clean32speed result. Next retain64MiB per-expert bounds but qualify a
  separate256MiB aggregate cap,which covers96times the actual maximum blob;
  keep32and its full publication/dispatch/loan/refusal gates. The independent
  `tools/sycl/adaptive_payload_sources.py`now prepares that one-header projection
  from the original299218/0a47Git source. Its legacy41-case fixture projection
  replaces only the obsolete80MiB overflow case with five valid64MiB extents;
  the separate source-only16-case`tests/core/adaptive_payload_budget_test.cpp`
  covers actual32/96 maximum-blob groups,exact256MiB and plus-one limits,
  per-expert bounds,checked-event failure,last-payload refusal and all three
  partial-publication views. These callbacks check metadata/order,not real
  payload bytes or GPU copies. The private projection now passes57host cases at
  source`ab23c478b74b1c319e13063eb6f6fa95605691c5`,with117source/control
  inputs and exact two-fixture recipes. Its new header SHA starts8387ea06c7;
  the public header and all frozen cohort inputs remain unchanged. The owned
  `tools/sycl/adaptive_payload_host.py`check closes normally with six0exit
  commands,exact41+16receipts and closed owners. Actual compiler/header/link
  closures cover207/204files and neither ELF links a GPU provider. Follow
  `adaptive_payload_audit_host_v1`. No successor native build or32-copy GPU
  qualification follows from these CPU checks. Its two one-use CPU ELFs and
  successful output are retired after recovery of117original sources and all
  four projected inputs:21files/321081bytes removed,all15engine/fixture
  outputs preserved. Keep only its record for the imminent successor header
  gate;regenerate the retired private inputs from Git rather than reusing them.
  After the frozen resource cohort closes,qualify the successor
  native header,positive32publication/dispatch/occupied loan and second-record
  refusal before preparing an8/32speed comparison. The successor builder
  `tools/sycl/build_adaptive_payload_model.py`now prepares that exact568-file
  composition:only the audit header changes,all567other inputs and the recipe
  remain fixed. Its first preparation at source`be2d75dab`fails after the
  template publishes its own new result:the post-step incorrectly rereads the
  entire result as a still-committed input. No compiler or GPU runs. Preserve
  `adaptive_payload_native_build_v1`as failed. The separate V2root/key captures
  validated predecessor/header evidence before publication and then requires
  the exact expected output with every other qualification field unchanged.
  It includes two valid and six corrupt publication controls. The build
  post-step and positive-entry preparation use the same ordering fix;no input
  hash is refreshed. V2preparation now closes normally0at source`bbd345c63`,
  with568inputs,only header8387changed,the correct429b predecessor receipt,
  the two/six publication controls passing,and no predecessor products in its
  external build inputs. V2compilation now closes normally0for all three
  commands and six targets,with1453covered actual compiler inputs,checked
  header/link/flags and all five fixture ELFs byte-identical to429b. The new
  engine SHA starts`e1b01116cc`;its native source remains`bbd345c63`. Only the
  audit header changes. Fresh same-build observer0/1CPU pool checks now each
  pass28cases with identical checksums,normal0and closed owners. Component
  reuse now passes;native32positive and its refusal control also pass.
  No GPU was executed
  by compilation or these CPU checks. It gates preparation
  on all six resource processes closing. Allow one temporary third comparison
  generation only through successor native32positive/refusal qualification:
  keep552b and429b's15outputs for fallback and exact five-fixture comparison,
  then prospectively migrate consumers and retire429b before clean8/32trials.
  Retire the successor instead if qualification fails. This is a bounded
  replacement window,not another retained generation per experiment.
  `tools/sycl/adaptive_payload_admission.py`binds the successor's
  actual sources/recipe/header and fresh28+28CPU parity to five identical
  fixture ELFs and the original10/32/16device checks. Reuse now passes at
  source`a1cf63b67`,including the live429b chain and original control recovery;
  later readers pin its consumed metadata/Git inputs rather
  than requiring retired429b ELFs. Runtime/probe bytes,boot and driver still
  need to match. This is unchanged-component identity qualification,not fresh
  GPU execution. The separate native32positive entry now closes normally.
  `tools/sycl/check_adaptive_payload_entry.py`now binds
  the unchanged full32copy/dispatch/occupied-loan reader and supervisor to the
  successor at a fresh`adaptive_payload32_native_entry_v1`key/root. It requires
  the new header/component proof and keeps the original64MiB trial failed.
  Its model plan is now prepared at source`4062acfba`on the compiled e1b011
  successor,with all15asset hashes and the unchanged32-group/22-corrupt reader
  controls passing. Both synthetic1024/256and32769/256requests now close
  normally0,with21full32-copy generations/672swaps and checked three-map/
  payload publication. Incoming GPU observations increase20616to47016and
  outgoing CPU observations332to2026;one post-positive1124-slot loan/refill
  also passes. Actual T1/T4 Q6/heads,closed owners,quiet validation,unchanged
  final15assets/boot/driver and no fresh fault pass. Follow
  `adaptive_payload32_native_entry_v1`;this is correctness,not speed or broader
  model qualification. Its receipt SHA starts`3ba342f1d887`;freeze these bytes
  and the compiled/reuse receipt for the next consumer. The companion
  `tools/sycl/check_adaptive_payload_failure.py`also rebinds the existing
  full32second-record control to that closed successor entry at a separate
  `adaptive_payload32_record_failure_v1`key/root. It requires a real first
  publication,32queued second copies and checked refusal with no second
  publication,normal expected1/closed owners/no fault. Its plan is now prepared
  at source`0c56a1f48`,with all15assets checked,the immutable closed32positive
  receipt and the unchanged two32-group/86-corrupt reader controls. The native
  control now closes as expected:record1succeeds and publishes32swaps;record2
  returns status1after32queued copies,with its generation masked/unpublished.
  The native exits1normally without supervisor termination,DONE is absent,
  45wire tokens are retained as extent/hash,and final15assets/boot/driver,
  strict validation,ownership and no-fresh-fault checks pass. Its receipt SHA
  starts`b50481aa8dfa`;keep it immutable. The inherited opening`engine429b`
  plan label is stale;structured pins and the actual executable check bind
  this run to e1b011. This expected error is a correctness control,not speed.
  `tools/sycl/compare_adaptive_payload_generation.py`
  now reuses the original same-ELF8/32decision/benchmark runner with successor
  positive/refusal qualifications. Prior settings/frontend selection supply no
  old-engine qualification;refs bind only the successor controls and tested
  Niko frontend. Code cap4/prose cap2 remain separate three-pair cohorts,exact
  TG is primary,and prompt/TTFT remain independent. Its regular-code plan is
  now prepared at source`bc1413ec2`,with all15assets checked and six actual
  configurations8/32,32/8,8/32on e1b011. All native/runtime/header/qualification
  reads succeed with the old429b paths absent. Both fresh benchmark/server
  frontend probes now pass normally0with closed owners at source`869568e8a`,
  still with the old paths absent. The held old products are now retired.
  All six regular-code processes close with native/benchmark0,quiet
  validation,closed owners,unchanged boot/driver and no fresh fault. Each
  reads33210fresh tokens after four official warmups,then measures its first
  long request with thinking;this is neither engine first use nor a subsequent
  long request. All15model-asset hashes pass again at final closure.

  | Process | Swaps | Generated | PP tok/s | Client TTFT s | TG tok/s | Cache hit % | Draft acceptance % |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 0 | 8 | 3016 | 542.03329 | 61.446 | 11.73872 | 24.0 | 83.8772 |
  | 1 | 32 | 3459 | 546.85840 | 60.950 | 13.38632 | 41.1 | 86.8056 |
  | 2 | 32 | 3459 | 541.39795 | 61.533 | 13.34774 | 41.1 | 86.8056 |
  | 3 | 8 | 3016 | 557.34756 | 59.782 | 11.79608 | 24.0 | 83.8772 |
  | 4 | 8 | 3016 | 556.63571 | 59.834 | 11.83684 | 24.0 | 83.8772 |
  | 5 | 32 | 3459 | 552.35295 | 60.331 | 13.42985 | 41.1 | 86.8056 |

  Exact three-process median TG is11.79608/13.38632for8/32,with ranges
  11.73872..11.83684/13.34774..13.42985. The three TG ratios are
  1.140356/1.131540/1.134581(geometric mean1.135486),so the preregistered
  repeated-gain gate passes. Prompt-time ratios are0.991177/1.029460/1.007754
  and client TTFT ratios0.991928/1.029290/1.008306;both independent
  nonregression gates fail. Median PP is556.63571/546.85840and client TTFT
  59.834/60.950s. Preserve the dedicated552b prefill fallback and keep the
  decode candidate separate;this result does not permit whole-route adoption.
  Logical missed lookups per generated token decrease414.971->313.727;
  these are not CPU kernel jobs or physical traffic. Different generation
  lengths and acceptance preclude fixed-work attribution. Individual clocks,
  counts and routes remain in the linked compact result under
  `adaptive_payload_generation_regular_code_v1`. The separately prepared
  cap2/prose cohort uses the same executable and fixed8-round cadence;its
  acceptance cannot establish a fixed4 weak-MTP class. Agent/task,quality,
  prompt-verify,physical262144/RESTORE and ordinary teardown remain pending.
  The cap2/prose cohort prepared at`c632a0421`now closes all six processes.
  Its346source inputs,four immutable configuration controls,small exact gain,
  slower-pair retention and independent phase decisions pass. Fresh frontend
  probes pass at`d79bd3448`. The fixed8/32,32/8,8/32process order uses the same
  e1b011engine,8192chunk,placement,runtime safety and observer settings.
  Every measured first long request follows four official warmups and reads
  33278fresh tokens;these are not engine-first-use or subsequent-long samples.
  Native16/GDN3,384slots,non-split,T1/T2head captures and MTP mode2with5callbacks
  are observed. All six native/benchmark exits are0with quiet validation,
  closed owners,unchanged boot/driver and no fresh fault. All15model assets
  pass final full hashing. Each8arm generates9854tokens including thinking
  with7217answer characters;each32arm generates7433with6940answer characters.

  | Prose process | Swaps | Generated | PP tok/s | Client TTFT s | TG tok/s | Cache hit % | Draft acceptance % |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 0 | 8 | 9854 | 550.51912 | 60.604 | 10.46625 | 47.6 | 63.1997 |
  | 1 | 32 | 7433 | 559.54897 | 59.644 | 10.75104 | 53.5 | 62.3062 |
  | 2 | 32 | 7433 | 571.33929 | 58.474 | 10.72590 | 53.5 | 62.3062 |
  | 3 | 8 | 9854 | 565.63109 | 59.056 | 10.48170 | 47.6 | 63.1997 |
  | 4 | 8 | 9854 | 557.66611 | 59.820 | 10.45067 | 47.6 | 63.1997 |
  | 5 | 32 | 7433 | 548.87472 | 60.804 | 10.68946 | 53.5 | 62.3062 |

  Exact TG is generated_tokens*1000/decode_ms. Three-process medians are
  10.466246/10.725899for8/32;ranges are10.450674..10.481704and
  10.689458..10.751041. The paired TG ratios1.027211/1.023297/1.022849
  have geometric mean1.024450and pass the repeated-gain gate without a gain
  floor. Prompt-time ratios0.983862/0.990009/1.016017and client-TTFT ratios
  0.984159/0.990145/1.016449fail their independent nonregression gates.
  Median PP is557.666108/559.548971and client TTFT59.820/59.644s;these
  aggregates do not override the preregistered pair decisions. Logical missed
  lookups per generated token decrease308.137->275.271;these are not CPU jobs
  or physical RAM/PCIe bytes. Different generation lengths and acceptance
  preclude fixed-work attribution. This closes the focused decode comparison,
  not phase preservation,adoption or a fixed4weak-MTP class. Both workloads'
  repeated32over8TGgains support the separately gated96candidate;preserve
  the552b prefill fallback and independent phase selection.
  Follow`adaptive_payload_generation_varied_prose_v1`.
  The next96-copy positive entry is source-prepared in
  `tools/sycl/check_adaptive_payload96_entry.py`;its CPU controls now pass at
  `df426f83c`. Its positive plan is now prepared at`8380579f7`,with125source
  inputs and all15model assets verified. Its actual native entry now closes
  normally0with quiet validation,closed owners,unchanged boot/driver and no
  fresh fault;all15model assets pass final full hashing. Both1024/256and
  32769/256requests finish at the declared length. All22groups contain96
  replacements(2112total),with full payload/host/private/device map equality,
  incoming-GPU/outgoing-CPU dispatch and occupied post-positive loan/refill.
  Its first request publishes12full groups,so the planned second-record
  refusal seed is eligible. Adaptive H2D requests total4312166400bytes,
  min/max190592000/201369600per group;these are requested bytes,not physical
  bus traffic or bandwidth. Preserve the immutable positive receipt SHA256
  `83ba7a590d31533b69696ea958365fd1f061f8048f7468a1624a9bbf79b5c7cb`.
  No96speed,task quality,ordinary teardown or inference adoption follows.
  After recovering its125inputs from pushed Git,59successful raw/JIT files/
  13769854bytes are retired. Only the immutable positive record remains;
  continued admission passes and all21protected receipt/build assets are
  unchanged. Manifest:`adaptive-payload96-entry-v1-output-retirement-20261107.json`,
  SHA256`11cd26a6c7a52d3d62525af3a7c523d2fa0c88aa12cd4cb41f2402284856a856`.
  It reuses the unchanged e1b011engine and ordered
  owned supervisor,requires both current8/32cohorts closed,and does not enable
  a performance trial. The pinned48-layer native index has max blob2662400B;
  even96maximum blobs total255590400B(243.75MiB),within the unchanged256MiB
  diagnostic cycle cap. The new CPU parser will check all48arena offsets and
  that bound before exec. Its positive reader now source-prepares an ordered
  join of each`ADAPT_SUBMIT`count/byte counter with its masked/published group
  and the incoming experts' actual layer sizes. It keeps only group/copy counts,
  total/min/max requested bytes and empty-update count;no per-layer arrays.
  Zero updates must request zero bytes with no unpublished group. Pure CPU
  cases add mixed-size/partial groups,missing/duplicate/late requests,incorrect
  counts/bytes,integer overflow and invalid layer bounds. These CPU cases pass.
  This measures expert-copy requests,not initial placement,map/audit traffic,
  physical PCIe bytes,transfer duration or saturation. Require two full96-copy
  publications,both dispatch
  roles and occupied post-positive loan/refill. The same file now source-prepares
  the separate expected-exit1 control through`refusal-prepare`/`refusal-run`.
  It requires actual positive96closure and proof that its first1024/256seed
  produced two full96groups,then uses the unchanged owned refusal supervisor.
  Require real first record0/publication,second null-record status1 after96
  queued copies,second map unpublished,noDONE,no supervisor kill,quiet
  validation and closed owners/no fresh fault. Its byte bound is the pinned
  model's96-group maximum. Source-prepared CPU controls cover second-group
  publication,malformed status/count/order and oversized payload;these CPU cases pass.
  This separate exact96second-record refusal remains mandatory before any96
  speed comparison. Its actual plan is now prepared at`a93e4edcc`,with125
  committed inputs,all15model assets checked and the exact positive receipt
  pinned. Only the pre-exec failure-injection selector/private paths differ;
  the expected native exit is1with no supervisor kill,noDONE and no second
  publication. The actual control now closes with the declared normal native1,
  no supervisor kill,quiet validation,closed owners,unchanged boot/driver,no
  fresh fault and all15final asset hashes. First record0publishes96;second
  record1follows96queued copies and keeps generation2masked/unpublished.
  The wire emits45tokens thenERR,noDONE. First requested H2D is191334400bytes,
  below the pinned model bound. Preserve its immutable receipt SHA256
  `300b500dc856304c961fa7dbaf12e1e37caf9d71ed43c48fbcc4c9ced3410cce`.
  Both actual96controls now pass;this permits a separately committed speed
  plan,not a96speed or inference-adoption claim. Its50raw/JIT files/
  12579134bytes are retired after125-input Git recovery;both96receipt bytes
  and all22protected assets are unchanged,with continued admission passing.
  Manifest:`adaptive-payload96-record-failure-v1-output-retirement-20261107.json`,
  SHA256`6111e9bf523c14d603995631981876b3e488f7b0876f9588a9b3cfd256a771ac`.
  Keep the original32
  reader and all current cohort/native inputs unchanged. Its ordered/index
  adversarial controls completed normally0after both cohorts closed:
  `/usr/bin/timeout -k 2s 20s /usr/bin/prlimit --as=1073741824 --cpu=10 --fsize=65536 /usr/bin/python3 -X faulthandler -B tools/sycl/check_adaptive_payload96_entry.py controls`.
  The125-input source signature and28ordered,7index-bound,14copy-request and
  213refusal negative cases are recorded under`adaptive_payload96_reader_host_v1`.
  The CPU result alone establishes no96native safety,performance or adoption;
  the closed positive entry above supplies only its specifically observed
  payload/map/dispatch/loan qualification. The separate actual96refusal has now
  closed as recorded above.
  The compiled source audit matches the recorded private generator,cache and
  audit header. Swaps and pending metadata are dynamic vectors;the pending
  capacity is reserved before accepted refill copies. The audit reserves96
  entries and bounds them by1256slots,24576map rows,per-layer identity and
  payload size. Cache publication validates all slots before changing its
  private map. The real refill event must complete successfully before final
  host/device publication. The full-RAM Arena route does not use the separate
  RAM-complement victim-exchange path. These source checks permit the bounded
  diagnostic,not actual96qualification or a new lifetime/performance claim.
  The same-ELF32/96regular-code cohort is now closed at
  `adaptive_payload96_generation_regular_code_v1`,prepared at`a9d072acf`.
  All six fresh processes in32/96,96/32,32/96order close normally0with quiet
  validation,closed owners,unchanged boot/driver and no fresh fault. All15model
  assets pass final full hashing. Each process has four official warmups then
  one first v5thinking-on33208fresh-input long request,with16Koutput allowance.
  Exact median TG improves12.613630->13.408466;all three paired ratios exceed
  one(1.068515/1.028556/1.059266;geometric mean1.051973). The printed TG decision
  is independently favorable. Within each arm,generated tokens(3477/3111),
  draft acceptance(2462/3048and2231/2643),output/reasoning hashes and cache
  counts agree across all three processes. Cache hits rise37.4->44.5%.
  Different output lengths and acceptance between arms preclude fixed-work
  attribution or assigning the entire5.20%paired TGgain to placement.
  Prompt-time ratios0.992150/1.026572/1.003363and client-TTFT ratios
  0.992116/1.027145/1.001150fail their independent nonregression gates.
  Their arm medians are60.0553->59.8823s prompt and60.233->60.046s TTFT;
  those medians do not override the predeclared paired outcomes. This qualifies
  a focused repeated decode gain,not prefill preservation,whole-route adoption,
  quality,agent,ordinary teardown,physical262144/RESTORE or70TG.
  Select96for the regular-code leaf-only resource comparison;keep the552b
  prefill fallback independently. The varied-prose32/96cohort is not prepared.
  Under frozen first-T1/then-T4,drafts/cache lookups imply1017/882verifier
  windows for32/96. Whole decode time per inferred window is
  272.311/262.925,271.046/271.871and270.091/263.059ms in the three pairs;
  output/window is3.419/3.527. These source/count-derived whole intervals are
  not traced kernel service,physical traffic or a saturation measurement.
  Processes0..3normal outputs are retired with350Git inputs recovered,
  all15ELFs/six immutable receipts unchanged and full continued admission
  passing. Detailed samples and retirement manifests remain in the linked
  service result. Processes4and5normal output(138files/26191897bytes)and the
  consumed scope-review body(5294bytes)are now retired after source recovery.
  All15ELFs/six immutable receipts stay unchanged;full before/after admission
  passes. Manifest:`adaptive-payload96-generation-regular-code-v1-final-outputs-retirement-20261107.json`,
  SHA256`1fbfd0957bf95418443a09aa6f1baba278fff0fd77ae62f540fef97fb7861fc1`.
  Generated config/plan remain for admission. The qualified129-input controls remain at
  `adaptive_payload96_generation_reader_host_v2`;source is`8f9636060`.
  The completed CPU command is:
  `/usr/bin/timeout -k 2s 20s /usr/bin/prlimit --as=1073741824 --cpu=10 --fsize=65536 /usr/bin/python3 -X faulthandler -B tools/sycl/compare_adaptive_payload96_generation.py controls`.
  This source is outside the closed prose cohort's346inputs. Retire it if96admission fails
  without a named follow-up consumer.
  All six prose processes each have69successful raw/JIT files
  (13136414/13122266/13122277/13136451/13136498/13122251bytes)retired after recovering346Git inputs
  and pushing compact results. The closed-cohort recovery also reproduces
  all12generated config/plan files byte-for-byte,the105-key environment and
  model-asset inventory from the recorded source/recipe. The remaining17files/
  126943bytes,including successful frontend logs,are retired;the root no
  longer exists. The next96reader passes without that root;all15ELFs/four
  immutable receipts are unchanged. Original samples/decisions remain at
  `c2e317112`;metadata manifest:
  `adaptive-payload-generation-varied-prose-v1-metadata-retirement-20261107.json`,
  SHA256`3ad29bc3a863745c63a00c20c8e88c0486d88a54aeda9c8db7299cc9a7f2cd21`.
  Per-process manifests:
  `adaptive-payload-generation-varied-prose-v1-process0-output-retirement-20261107.json`,
  SHA256`e8e7ef533b03366618eb612cc3cf7030197a8047a2152a8f4eeed705008de0cc`,and
  `adaptive-payload-generation-varied-prose-v1-process1-output-retirement-20261107.json`,
  SHA256`376194f62f6bd5335fa022a8a1a354584d06b587f01b60d177f5f4ae9505dfa8`,and
  `adaptive-payload-generation-varied-prose-v1-process2-output-retirement-20261107.json`,
  SHA256`2def8272f5fbaba656fb67dbcbd8d3c330522dcb5b3a22bdd2d67608d9bba506`,and
  `adaptive-payload-generation-varied-prose-v1-process3-output-retirement-20261107.json`,
  SHA256`ccde4bb997c82d2a9b1c8a56c42e262913b6da9b26957d20465726b1ebe4577b`,and
  `adaptive-payload-generation-varied-prose-v1-process4-output-retirement-20261107.json`,
  SHA256`941ce371cd176ae564cdef6d61ed4dcf780e324536920e36b06947e9e079e8e0`,and
  `adaptive-payload-generation-varied-prose-v1-process5-output-retirement-20261107.json`,
  SHA256`395f6c8f763d77923a1c19876cab5d088c275e3f7b971b194f59c44b29024d88`.
  For the closed regular-code cohort,all six processes each have69successful
  raw/JIT files retired(13095666/13096255/
  13096306/13095722/13095457/13096142bytes),after recovering all346source inputs
  from Git and pushing their compact results. Admission passes after all six
  deletions. The subsequent closed-cohort recovery reproduces all twelve
  generated config/plan files byte-for-byte,the105-key environment and the
  model-asset inventory from the recorded346-input source/recipe and unchanged
  qualification records. No maintained reader needs this closed code root.
  Its remaining17files/126927bytes,including successful frontend logs,are
  retired;the root no longer exists. Original samples and final decisions
  remain recoverable at`3fc4e9094`. The metadata deletion manifest is
  `adaptive-payload-generation-regular-code-v1-metadata-retirement-20261107.json`,
  SHA256`0213a61500f490d8aa7e3c9aec4efbfd0c228673e417ebf746c2528a6022b58d`.
  The provenance hold is resolved;all15ELFs and four immutable native/
  control receipts are unchanged. The external deletion manifests are
  `adaptive-payload-generation-regular-code-v1-process0-output-retirement-20261107.json`,
  SHA256`0741507908c4976b8bed7b6cf1ecc5364b4fbe37a4fbc526da51522b2c5a85e7`,
  `adaptive-payload-generation-regular-code-v1-process1-output-retirement-20261107.json`,
  SHA256`ae43e20567d57e3c1da3e9bf5713f084642bd891fbaa35edc8dc6f94f3941258`,
  `adaptive-payload-generation-regular-code-v1-process2-output-retirement-20261107.json`,
  SHA256`0a9081498ca80b097ab20df3b9597ddaec7d0676a369283d2721697a878cb7bc`,
  `adaptive-payload-generation-regular-code-v1-process3-output-retirement-20261107.json`,
  SHA256`258a63edc119b877a078d0c20f5142f85720a07ac410301752da8c9476071076`,
  `adaptive-payload-generation-regular-code-v1-process4-output-retirement-20261107.json`,
  SHA256`5c8e36b486e52069dea54e0332eec361a62ae2da548667e57250710f4d9cca26`,
  and`adaptive-payload-generation-regular-code-v1-process5-output-retirement-20261107.json`,
  SHA256`56fe7a3bf95437488a2aca75fe56fb330c7c9668f86e33cb69ea12a3cd62ad4a`.
  Before its preparation,the following CPU-only control is
  preregistered at a closed GPU boundary:
  `/usr/bin/timeout -k 2s 20s /usr/bin/prlimit --as=1073741824 --cpu=10 --fsize=65536 /usr/bin/python3 -X faulthandler -B tools/sycl/compare_adaptive_payload_generation.py controls`.
  Require actual0,quiet stderr,four immutable configurations,small exact gain,
  slower-pair retention and independent phase verdicts;the CLI pins its full
  transitive source closure before controls. These controls now pass at source
  `7e2177511bce6cfd975dc7b28c6b9c3585a1f781`:127inputs verified,actual0,
  quiet stderr,four configurations and all exact/slower/phase checks. Follow
  `adaptive_payload_generation_reader_host_v1`;no GPU/model execution occurred.
  Keep native build/positive/refusal
  receipt bytes frozen when later gates consume their hashes:record raw-output
  retirement separately rather than rewriting those receipts or refreshing refs.
  Actual successor preparation and both frontend probes passed with429b's
  original paths absent before its retirement;the first clean model process
  also now closes normally without those products.
  The positive32entry's59successful output/JIT files(13634939bytes)are retired
  after recovery of121original inputs,with21ELFs and both consumed receipt
  hashes unchanged. Keep only its immutable record. The deletion manifest is
  `adaptive-payload32-positive-output-retirement-20261107.json`,SHA256
  `c2566566c666d7d5721b621a6b9edf9726b1f2153d79d7ab4646eec818ca0b94`.
  The expected-error32control's50output/JIT files(12567346bytes)are also
  retired after124-input recovery,with all three receipt hashes and21ELFs
  unchanged. Its manifest is
  `adaptive-payload32-refusal-output-retirement-20261107.json`,SHA256
  `8aa1d60a7c3ec44aadf4a81fe2146a0c22e252ba7b15a20c10c3491764386cf9`.
  The ended429b generation's568native inputs,112controllers,recipe and actual
  compiler inputs are recovered from reachable Git. After real successor
  preparation and both frontend probes passed with the original paths absent,
  its748files/163050896bytes are deleted and the temporary hold no longer
  exists. All four receipt hashes and15fallback/successor ELFs are unchanged.
  Keep429b's immutable record for scoped provenance;its compiled tree is gone.
  Follow`retained_builds_review_20261103.payload_successor_replacement`.
  At the closed build/CPU boundary,the failed V1preparation's568projected
  inputs,115controls and recipe are recovered;its duplicate private source is
  retired. The old64MiB entry's116inputs and the SSE tests'352inputs are also
  recovered. Their unneeded output is retired with638files/24966960bytes
  removed and all21comparison ELFs preserved. Only the567-byte cap-refusal
  diagnostic remains for the new32trial. Full SSE input inventory is archived
  at`af613119f`;the current row retains its two changed source hashes.
  The static resource profiler is
  independently eligible on unchanged429b;its census disables adaptation.
  Vectors are dynamic/reserved before submission and publication waits for
  the checked real event. The diagnostic keeps
  the same429b native engine,runtime safety,prefill layout
  and every8round interval,and blocks preparation/execution until both frozen
  code/prose cohorts close. The reducer requires a full32-copy generation,
  both replacement dispatch roles,three-map/payload publication and a later
  occupied loan/refill. These native conditions remain unqualified. The initial
  placement profile is separately Git-pinned. Larger-budget refusal,clean
  performance and broader model gates remain pending;no third engine build.
  The companion`tools/sycl/check_adaptive_budget_failure.py`is now source-only:
  it reuses the existing owned second-record supervisor,requires a closed
  positive32entry,and checks two full32-copy generations with only the first
  published. The20-second locked CPU-only controls now pass at source
  `5943022da76d6ee75b6b9af6cf5dd71dd73c40ff`:two full32groups and86corrupt
  variants,actual0 in0.576s,without a native engine or GPU runtime. Follow
  `adaptive_budget32_failure_reader_host_v1`. Native expected-exit1 refusal
  remains unexecuted. The frozen speed-cohort inputs are unchanged.
  The follow-up`tools/sycl/compare_adaptive_budget_generation.py`is source-prepared.
  Its CPU-only controls pass at`a70edb59a3ca6909c90178339b6bd2986dad4774`:actual0
  in0.594s,quiet stderr,four budget/width configurations,small exact gain,one
  slower pair and independent TG/prompt/TTFT verdicts. Follow
  `adaptive_budget_generation_reader_host_v1`;no model comparison is prepared
  or executed. It prospectively compares adaptive8/32on the same429b engine,
  every8rounds,with separate cap4/code and cap2/prose cohorts. Preparation
  requires both current cohorts closed,the required streamed-tool frontend
  port tested,and the real32-copy positive/second-record refusal controls.
  Exact generated-count/decode-time TG is primary,with the printed decision
  retained;all three pairs must improve without a minimum percentage floor.
  Prompt/TTFT remain independent. No new native generation or model execution
  follows from preparing this source;run its CPU controls at a serialized
  closed-process boundary before preparing the comparison.
  The CPU-only command is preregistered as
  `/usr/bin/timeout -k 2s 20s /usr/bin/prlimit --as=1073741824 --cpu=10 --fsize=65536 /usr/bin/python3 -X faulthandler -B tools/sycl/compare_adaptive_budget_generation.py controls`.
  Its internal measurement lock must be free;require actual0,quiet stderr,
  four configuration controls and independent TG/prompt/TTFT decisions. This
  reader control may run between closed processes;do not run it during a GPU job.
  The retrieved review's stale-map condition is resolved for the current
  bounded8-swap route:actual compiled inputs match its generator/cache/header
  identities,`adaptive_res_upload`updates`xcache.replace_residency(host_res)`
  before device publication,and the existing23-generation/184-swap native
  entry observes incoming GPU7213->19004,outgoing CPU53->399and one post-positive
  refill. The checked second-record refusal is also closed. This is the
  current-binary gate for extending the budget,not a proof for32swaps itself.
  The source-level request reader is extracted to
  `tools/sycl/native_request_protocol.py`. All four CPU regression tests pass
  at source`67e7e2154e69968d1a5a31ef00ac2d9cd7769f84`via
  `python3 -X faulthandler -B -m unittest discover -s tests/core -p test_native_request_protocol.py -v`.
  They cover1/1024/32769-input extents,the actual RESUME/PP omission,malformed
  order/counters/output and geometry bounds. The bounded native checks above
  now exercise actual dispatch,positive copying,post-positive occupied loan/refill
  and checked record refusal. These CPU reader tests alone do not supply those
  native results. Task quality,physical262144/RESTORE and ordinary provider
  teardown remain open. The cb7 and519e diagnostic generations are retired;
  retain552b and429b for the current comparisons.
  The source report in `.cache/research/2026-11-06/adaptive-cache-revalidation`
  remains on HOLD as written:its unsupported swap0/MTP2 claims and incorrect
  `generate.cpp:1559` anchor are not adopted. Its pins and bounded-control design
  were checked by native Luna-max review;neither is runtime qualification.
  Keep the report/review only until the post-width positive-swap plan consumes
  the applicable design. Research-index edits remain deferred during the cohort.
- The exact ec4 HC/interlayer source audit now supplies static262144-index
  evidence,not physical-context qualification. At full `n=262144`,the HC tensor
  has2684354560elements;its BF16 store ends at5368709120bytes. `checked_count`
  bounds the `size_t` count below `INT64_MAX` before the `for_each` handoff;
  per-chunk `T` remains distinct from full `n`. Chunk/window planning and
  preflight failure paths do not prove actual allocation or final-slot writes.
  Keep the existing MoE/expert-bank planner fixtures distinct from runtime
  admission,and `validate_spans` distinct from complete lifetime/alias proof.
  Ordinary serve rejects zero output and reserves `n+max_new+8` context slots.
  Its prefill loop uses exclusive end `n-1`;the final input position is processed
  by the first T1 verifier window inside the generation loop. A private full-fill
  diagnostic must explicitly run,commit and drain that T1 window at262143
  before suppressing output/drafts;merely setting `max_new=0` skips the final
  position. The reviewed q6 engine-entry report remains HOLD until this is
  explicit. The read-only e1b011 audit confirms the same serve gap in its
  recorded private source:zero-output rejection at6250,`p=n-1`at6940and the
  generation loop at7025;standalone`n_batched`at7500also excludes the last
  token. Its generate.cpp and layer_major.inc bytes differ from the public
  tree,so a future diagnostic patch must target the recorded compiled
  composition. After`cur=ids`at6370,the current`read_windows`reads`cur[q+t+1]`for its MTP
  next-token input. Extending its upper bound to`n`would read`cur[n]`;
  its normal bound stops at`n-1`. The private consume-only diagnostic must
  use the generation-side final T1 run/commit and checked completion,without
  inventing a next token or drafting an output. Define and verify target and
  draft valid extents separately. This source audit supplies no physical-fill execution.
  Keep its report/review in
  `.cache/research/2026-11-06/full-context-engine-entry-design` only for the
  post-cohort implementation plan;ordinary API admission remains unchanged.
  The separate HC-address review accepts its static source report with the
  count/notation/fixture corrections above:
  report SHA256 `7664086617755171588e42d0ff5c7d4b6fd172af4628144246b50c12f995f0ff`,
  review `df6db1aa0898e50ddcf18cc3c8dca493ff41f7617692e5fab8bbb1a84d7d348e`.
  Worker/terminal closure is confirmed;worker-side startup IPC chronology is
  unverified. Full allocation/fill,tails,handoff,RESTORE,safety,speed and adoption
  remain open. Keep the bounded report/review only until the post-cohort runtime
  plan has consumed its source/fixture matrix in
  `.cache/research/2026-11-06/full-context-current-source`;defer research-index edits while
  the current comparison's transitive reader freezes that file.

## Prefill envelope and scaling

The 117489 result above is the current repeated warmed PP reference. A separate
same-engine curve has one process per length and describes actual routes, not
an adoption law. On the same **layer-major, bank-384, non-split** route,
19665..78445 actual input tokens are descriptively affine. With `n` actual read
tokens, `T_prompt[s] ≈ 13.0863 + 0.00082309*n` and
`TTFT[s] ≈ 13.1867 + 0.00082431*n`.
The prompt fit's maximum residual is 1.58%; leave-one-length-out mean/max
relative errors are 1.30% / 2.86% (TTFT 1.38% / 3.03%).
This is an empirical same-route description only: the 1286..8193
chunked route has a different fit, 117489 uses split, and route/chunk boundaries
change the curve. Do not interpret the intercept as a physical fixed cost,
extrapolate outside the measured interval, infer a limiting resource, or assume
one fixed-plus-linear law. See `cache_map_prefill_curve_v7.decision` and
`.interpretation` in the linked service result.

The slower same-engine 117K clean/observed comparison pair, **918.3721 / 943.8786
tok/s**, remains below the 1047.0672 tok/s reference; environmental variability
is unresolved. Retrieve its exact samples/configuration from the [original
maintained record at commit
`3abcceba08b25a632f4e26bde7551039120ac42a`](https://github.com/yqYo1/Strata/blob/3abcceba08b25a632f4e26bde7551039120ac42a/docs/SYCL_NATIVE16_PREFILL.md);
the cumulative CSV does not contain both values. The earlier adaptive-cache
model comparison recorded **15.529 tok/s** within a failed cohort: its code
sample qualified locally; the prose benchmark failed its answer gate and no
repeat decision exists. Preserve it as a failed observation via the CSV row
(`cache_map_generation_speed_v1.processes[1].samples[0]`), never as adopted
speed.
The separate **32K-output feasibility probe also failed**: its code sample was
15.5924 tok/s, but prose consumed all 32768 generated tokens in thought with
zero answer characters. It is not an unexecuted next experiment or a repeated
speed gain. Preserve `cache_map_generation_room_v1.failure_diagnosis` and the
original evidence at commit `a24593e1f84402244d8660914bb5e80587bcdc7d`;
do not repeat it or merely escalate its output allowance. Before another cache
comparison, choose and commit a distinct answer-eligible workload/sampling plan
and verify it on the safety-corrected native source. Keep thinking enabled;
do not waive the answer gate or assume sampling caused the previous failure.

## Decode and resource evidence

Keep prefill and decode kernels, placement, scheduling and selectors independent.
The completed 05a9 comparison changes only its decode selector. Required
adoption evidence still includes both observed MTP-strong
and MTP-weak long generation, dedicated PP/TTFT preservation, long-input and
agent-session benchmarks, prompt-verify behavior, and the prefill-to-decode
handoff. A component pass or one request does not qualify a route.

CPU census reports logical packed-blob/job work and nested pool wall; pool wall
includes scheduling and waits. GPU Down service and GPU-reach waits are service
attribution, not useful CPU work or critical-path wall time. Neither useful CPU
arithmetic nor physical RAM/PCIe transactions, VRAM traffic/bandwidth, hardware
ceilings or saturation are established. Keep requested transfer bytes distinct
from measured physical traffic.

The CPU-only read calibration at source
`b3f2d7f31fbd836008930c2354c0a6479a6caea5`now closes:one1GiB anonymous
4KiB-page mapping,CPU0first touch,one warm pass and five seven-pass samples
at each1/2/4/6/12pinned-thread condition. All worker checksums/affinities pass,
with zero timed minor/major faults and every batch above100ms. Actual assembly
shows eight32-byte reads per256-byte block,eight accumulators,a real reloading
pass loop and vzeroupper;compile/test exit0 and owners close. Median payload
GB/s is24.5573/33.7208/33.5026/33.6126/32.1893;the observed peak is34.6618.
This ordered sequential pattern plateaus around33.5..33.7GB/s from2to6threads.
It does not qualify physical DRAM transactions,an absolute hardware ceiling,
random-expert behavior,CPU arithmetic saturation or a model bottleneck. The
older30.642GB/s single-thread result is a different pattern and remains valid.
Follow`cpu_read_capacity_v1`;consume its calibration with actual format/NT
service rather than converting historical logical job bytes to a measured limit.
The auxiliary ELF,dependency output and successful streams are retired after
pushed results and recovery of all7original sources/the committed recipe. Its
one-use source/controller are archived at the source commit above;normal Git
history restores them. Both inference engine generations remain protected.
The best-median two-thread point is also in the cumulative CSV.

The earlier source-matched decode-resource comparison reused the429b
engine's profile-ON build and existing leaf,census and disjoint caller
observers. It kept CPU leaf sums,caller regions,nested wait CPU and draft
work separate and used static0because the compiled
`CpuWorkCensus` snapshots slots at construction and rejects`residency_changed`
in`record`,and its Python reader requires the initial profile membership.
The429b compiled/current header hashes match. Enabling these existing checks
on adaptive8/32does not qualify dynamic attribution;that needs a prospective
membership-aware contract rather than suppressing the rejection.
The Main-authored`tools/sycl/profile_decode_resources.py`provides that
static plan on the same429b engine,with cap4/code and cap2/prose
in separate off/on,on/off,off/on cohorts. CPU leaf/census/decode/trace remain
common;only caller profiling changes. Its new T1/Tcap reader binds traces,
work and rounded clocks directly,without changing the frozen T1/T4 reader.
The preregistered CPU controls now pass at source
`7e4537e42ff32bb2eb3f5b6fe313f8b4d107d4aa`:both widths,24corrupt timing
cases,paired-overhead controls and four immutable environment configurations;
actual0 in0.577s. Follow`decode_resource_reader_host_v1`. Both clean speed
cohorts are now closed. The regular-code resource plan is prepared at source
`afdf7c96670ecbfaf5b761dd6017afc1e9cf040b`,with all15model-asset hashes and
341committed source inputs checked. Follow`decode_resource_regular_code_v1`.
Fresh benchmark/server frontend probes pass at source`54802134745c35f35c8df144edad3e562f5d2227`.
All six regular-code processes now close normally0,with five joined resource
records each,quiet validation,closed owners,no fresh GPU fault and all15final
model-asset hashes matching. Every long request reads33211fresh tokens and
generates3157,with matching aggregate logical work across all three pairs.
Decode times span225.6214..298.9372s on unchanged code. Caller-on/off latency
ratios are0.977564/1.068187/0.989841,geometric mean1.011081. The second pair
exceeds the predefined1.05bound,so incremental caller-cost qualification fails.
Keep all samples;these data do not assign6.8%causal overhead or qualify total
observer cost,clean speed or hardware saturation. The cap2/prose resource plan
has not been prepared or run.

Every request performs1207577CPU expert jobs,78.13%atNT1,and2.463TB logical
blob visits;these are not measured RAM traffic. Process5's3157-token decode
takes225.6214s. Its rounded stage clocks are105.147s CPU pool,67.914s GPU-reach
wait and30.231s draft;the GPU wait includes its wrapper and does not establish
GPU arithmetic saturation. Six leaf threads accumulate512.287CPU seconds.
The main leaf slot64has the native PID and86.617CPU seconds,inside the
caller's220.132CPU seconds. Caller plus the five worker leaves accounts for
645.802CPU seconds after removing that overlap;it excludes parked-worker spins
and other runtime threads. The pool source confirms host participation with
`drain(-1)`. The main thread performs useful work and completion waiting.

A fixed-work sensitivity model for this last process is
`T_hyp[s]=105.147/r_pool+67.914/r_reach+30.231/r_draft+22.3294`.
Each`r`reduces only that interval with all work and other intervals fixed.
Halving pool time gives18.244tok/s;zero pool time gives26.205tok/s. These are
conditional arithmetic,not measured gains or physical hardware ceilings.
A changed cache can alter work,overlap and other intervals,so it needs fresh
model samples. The independent audit-cap successor can now proceed through
source preparation and bounded qualification;resource closure does not turn
the failed observer-cost decision into a pass.
For the current e1b011generation,the recorded profile-ON recipe and the
recorded/private/public CPU census header,leaf-service header and pool source
still agree exactly. This preserves existing instrumentation source,not a new
current-engine resource profile or an observer-cost pass. The native Luna-max
library review finds no existing current-engine evidence separating CPU
arithmetic,cache effects and RAM service. Main consumed only its short digest;
the review is indexed as`cpu_memory_working_set_retrieval_review_20261107`.
Before selecting a CPU working-set optimization,obtain current operation/format/
NT attribution with a qualified observer-cost comparison. The unchanged static
census still rejects adaptive residency changes;do not suppress that guard.
The current leaf profile is independent of that census:pool construction reads
`STRATA_CPU_SERVICE=1`,and the compiled generator snapshots/writes its separate
per-request GU/HQ/DOWN format/NT counters. Main verified its header and pool
against both the e1b011build record and compiled private source. A new bounded
leaf-only plan can therefore test the current adaptive route without rebuilding
the engine or enabling the incompatible static census. It must declare its
limited scope and compare observers off/on in immutable pre-exec environments;
this is not a way to disable a census failure or claim dynamic residency
conservation. No such current-engine profile or cost comparison has run.
The source-prepared`tools/sycl/adaptive_cpu_leaf_evidence.py`reuses the existing
version2record grammar and joins each completed request to its independent
native CPU-job/entry footer. HQ calls derive final-job counts;GU/HQ/DOWN rows,
NT-weighted entries and thread/bin clocks must conserve them. This does not
independently identify each format bin or prove published-map identity,physical
traffic,saturation or observer cost. At source`b71fdacaf`,all six CPU test
groups in`tools/test_adaptive_cpu_leaf_evidence.py`pass with zero skips,normal0
and closed owners;the host plan pins37inputs and the qualified interpreter.
Follow`adaptive_cpu_leaf_reader_host_v2`. The superseded five-group proof and
its raw-output retirement are archived at`8327c62f8`;no maintained reader uses
that old result. No leaf-only adaptive model profile or cost comparison has run.
These tools introduce no new native engine.
The Main-authored`tools/sycl/profile_adaptive_cpu_leaves.py`implements
separate leaf-off/on,on/off,off/on cohorts on the same native engine. Each
workload requires its closed32/96cohort,then uses96only with a repeated
TGgain,otherwise32. That is a focused decode choice,not route adoption.
Only the pre-exec CPU leaf flag differs;static census,caller/full trace and
decode timing remain off,with runtime safety unchanged. Bind the four warmup
records before the long request and all five after normal exit. Matching
aggregate CPU/token/draft work and long-output hashes plus all three paired
latency ratios<=1.05qualify the leaf observer's cost only. A work/hash mismatch
is inconclusive for that cost,not a task-quality failure. Concurrent leaf CPU
sums still cannot distinguish arithmetic from memory stalls or prove a critical
path. Retain operation totals,format/NT job counts and top3GU/Down bins for the
next named working-set calibration. At source`4547d9e6e`,the controller CPU
controls pass normally0with133inputs,including on/off leaf fixture binding,
summary conservation,eight immutable environment/configuration cases,and
refusal of changed work,cache budget or reused process ownership. Both compiled
chunk diagnostics pass in each arm;unknown prompt records and seven nonserial
configurations are refused. Follow`adaptive_cpu_leaf_controller_host_v2`;
the superseded control is archived at`b2c37754a`. No adaptive leaf model profile or
observer-cost result follows. The completed CPU command is:
`/usr/bin/timeout -k 2s 20s /usr/bin/prlimit --as=1073741824 --cpu=10 --fsize=65536 /usr/bin/python3 -X faulthandler -B tools/sycl/profile_adaptive_cpu_leaves.py controls`.
The returned Orca source audit and native Luna-max review identify a legitimate
prompt-chunk diagnostic that both readers incorrectly counted as a prompt
footer. The Main-authored repair accepts only the two exact compiled diagnostic
grammars and retains malformed-footer rejection;all six reader groups pass,
including the new positive/negative diagnostic cases. Revised on/off controller
controls also pass. The controller now pins
parallel1,no batch/slots,one synchronous client,and all five private compiled
producer/resolver hashes. Quiescence remains a caller assertion supported by
that serialized path. Public generator/resolver bytes differ from the compiled
snapshot;do not substitute them. The current reader/controller CPU gates pass.
The regular-code plan is now prepared at`ad3723d20`,with358source inputs,
all15model assets fully hashed and focused decode budget96selected from the
closed32/96decision. Follow`adaptive_cpu_leaves_regular_code_v1`.
The prepared invariant,serial admission and five compiled producer/resolver
hashes are pinned. No native code changes. Fresh current-source benchmark and
server frontend probes both pass with closed owners. All six processes
now close normally0with quiet validation,closed owners,unchanged boot/driver
and no fresh fault. Every process has33209fresh-input/3227generated tokens,
five bound native/client records and a checked four-warmup prefix. The work and
long-output signatures match. Processes0/3/4are leaf-off,with exact TG13.567158/
13.338442/12.514679;processes1/2/5are leaf-on,with13.195703/13.028172/12.356003. Their full individual
PP,time,TTFT and work samples remain in the compact result. These inputs differ
from the earlier32/96cohort;no cross-cohort speed gain follows.
All three opposite-order leaf-on/off per-token latency ratios are
1.028149645/1.023815340/1.012841992,each within1.05with matching work;
geometric mean1.021581972. The observer-cost gate passes. All15model assets
pass final full hashing. The slower fifth/sixth samples remain included;no
sample replacement or changed comparison criteria. This qualifies the charged
leaf observer's bounded cost,not a speedup,critical path or hardware saturation.
The first charged leaf sample totals398.843CPU seconds across six threads:
GU265.234,HQ0.726and DOWN132.883. Of846250CPU expert jobs,728497(86.085%)
are NT1;989999entries give1.170entries/job. All three leaf-on samples have
identical format/NT job counts;GU takes65.62..66.50%of their CPU leaf time.
The top GU bins are NT1
IQ3_XXS/IQ4_NL,IQ2_S/IQ4_NLand IQ3_S/IQ4_NL. These concurrent CPU sums
include memory stalls and do not establish critical path,physical traffic or
hardware saturation. For these matched CPU jobs,the sum of original full-blob
extents is1728439500800bytes,or535.62MB per generated token. If each job reads
its full blob once,70tok/s requires37.49GB/s. Actual cache reuse and DRAM traffic
remain unmeasured;this is not a hard ceiling. Use the named current working-set
calibration after
the complete cost comparison;the previous calibration used an older
compiler/reference and is not qualified unchanged for this engine.
The CPU calibration closed at`8547cb65d`;recover`cpu-roofline-current.json`
from that commit. It uses
the recovered568-file e1b011native composition and its actual icx/icpx2026.2
C/C++recipe,with service/census observers off,IQ2/IQ3pairs enabled,the IQ3
audit and20ms pool spin. Three matching current NT/format distributions bind
all846250jobs. The matrix covers seven formats,groups6/28 and1/2/3/6compute
threads. All56real-weight numerical cases pass NT1..4 and mixed-NT pool/serial
bit equality,finite outputs,inactive lanes,canaries and actual affinity. This
is same-route consistency,not independent model/task quality.

The original V3 timing at49b2d4821 remains failed:all112expert cells finish
560eligible samples,but its old scanner rejects2threads before allocation.
The separately planned V4 continuation reuses exactly those expert samples
and56checks after recovering261original controller/recipe inputs and checking
all634actual compiler inputs,link inputs,ELF,model slices and environment.
The original individual samples remain recoverable at that immutable commit;
that closed result keeps their identity and compact analysis. V4 newly builds
only the public scanner. It supports1..6/12threads with allowed-affinity,
range and duplicate checks;scan/checksum/partition math stays unchanged.
At source d7a92b074,all eight host controller tests pass without skips. Its
262-input plan builds with627covered actual compiler inputs and no GPU runtime
link. Four bounded invalid-CLI cases pass with the expected1and exact errors.
All four new timing children close normally0with20eligible samples.

Requested-byte256MiB checksum-read medians for1/2/3/6threads are
20.413/25.890/24.297/24.028GB/s. The expert synthetic marginal mixture scales
1->6compute threads by4.248x for reuse-group6,3.144x for stream-group6,and
3.386/3.289x for reuse/stream group28. Six-thread weighted per-job wall time
is106.556/139.844us for reuse/stream group6,and136.915/140.256us for group28.
The plan records the exact weighting/median formulas. Expert-cell CV spans
0.44..39.74%;all samples remain included. These are independent synthetic
format/NT marginal processes,not actual layer/access-trace latency. Requested
blob visits/read rates are not physical DRAM/PCIe traffic,hardware saturation
or a70TG ceiling. Pool waiting,arithmetic/unpack,scheduling and memory effects
remain mixed;early read-rate flattening and expert thread scaling alone do
not prove the actual limiting resource. These observations motivated the closed
compact,gather and activation-LUT comparisons below. Current-model GPU counters,
physical traffic and critical path remain unqualified. The calibration is not
a new model speedup,adoption or device execution.

The packed98B gather experiment is closed and **rejected for performance**.
All524288integer controls,eight real-weight/threaded checks and24timing
processes/240samples pass execution and numerical checks,but all eight
raw/gather geometric-mean speed ratios are below one(0.79342..0.94404).
No production inference path changed. Exact source41167d33b,recipe and samples
are recoverable from the [closed gather result](https://github.com/yqYo1/Strata/blob/dfe6118a12f2d946fc968b3aca8c290f84b2aedf/bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-current.json).

The earlier134B compact-view experiment preserved native integer/FMA output
and improved one-thread GU by1.71..1.83x and six-thread reuse by1.24..1.29x.
Neither six-thread stream cell qualified stable improvement;raw+view storage
and0.354..0.481s allocation/encoding costs remain charged in its original
[comparison](https://github.com/yqYo1/Strata/blob/286b4a132b960a3a633511173846b57ebef32cee/bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-current.json).
This component result does not establish production critical-path benefit,
physical RAM traffic or model adoption. Its original source is313e5c4bc.

The isolated child`experiment/cpu-compact-cache`pins Main's bounded,
frozen compact-GU ownership component at`09ca4f1613d5e4f0ca5be80fb5dc0e8f691e34c3`.
It restores only the original owned kernel and adds known-byte admission,
independent leases and explicit decode/NT1guards. Primary engine inputs are
unchanged. The ownership gate below passes;independent numeric/performance
and production integration remain pending. Its maintained plan is
`bench/sycl/cpu_compact_hot/README.md`on that child. Keep the child for real-block
dispatch and hot/cold timing,then merge useful original commits into
`feature/sycl`or retire rejected source. Legacy reader/build holds remain open.
The completed source audit at09ca4f16confirms no source index/encode-publication
rollback defect,but retains serialized publication/model-role preconditions,
known-payload versus peak-RSS limits,and actual paired-versus-GGML numeric
admission. Its12project/2GGMLpins match;the IQ3paired implementation itself
was outside that audit's pins. Main has committed the ownership-only controller
at`ca8f8cb8e275b5e0554f0a12a8b0b34963cc3d96`and generated recipe/budgets at
`d65bb75bca80dad90931bb33a5697214637e0911`. Its serialized CPU-only build and
ownership check now pass at pushed child`8a05b016c674903ae9be27ffdf68f156acbfa0e1`:
all432actual compiler inputs are covered,only standard CPU libraries are linked,
and both owned stages close normally0with quiet stderr. The lease survives
cache/raw retirement;known-byte/phase/type/bounds and six disjoint row readers
pass. The fixture ELF is`0676de3990b4980ee723829cf74748d9263289852068044f96ed074ed64a95c5`.
This is ownership consistency,not independent arithmetic,strict peak RSS or
speed. Keep the child for real-block dispatch and hot/cold timing,then integrate
useful original commits and retire it. No source audit or ownership marker
qualifies arithmetic,physical RAM or model performance. The next numeric plan
must pin the actual paired helper in`src/kernels/cpu/iq_single_avx2.cpp`and its
`iq_avx2.hpp`declaration,then exercise production dispatch with explicit startup
flags rather than assuming that the paired reference was selected.

The block-local activation-LUT screen is also closed and **rejected**.
Source3406abc97 and ELF6144036d passed nine controller tests,633actual compiler
input checks,2097152table entries,4194304scaled chunks and14refusals. Both real
IQ3_XXS formats pass bit equality/finite/guards across150experts,four activation
vectors and E1/E10sharing. The CPU-only twelve-process screen retains120whole
call samples,all>=1s,with native dispatch and LUT build/block/output counts
verified. Table generation,accumulator clearing and GU output are timed.

| Same-input experts | Raw/LUT speed ratio, reuse | Raw/LUT speed ratio, stream |
| --- | --- | --- |
|1|0.24527|0.18078|
|10|0.19310|0.19076|

All three pairs lose in every cell. E10stream median time is2.69266ms native
versus14.06488ms LUT;paired geometric-mean LUT/native time ratio is5.24228.
This rejects the512KiB slab prototype;it does not establish the physical cause,
all LUT algorithms' performance or any model regression. Samples and exact
source/recipe references are in the [closed LUT result](https://github.com/yqYo1/Strata/blob/fe88dddb2a48922bf30140611cef9c7b92217e0d/bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-current.json).
Actual routing caps same-token reuse at10experts;28jobs do not imply28-way reuse.
NT2..4or model trials are not advanced from this failed screen.

The compact,gather and LUT auxiliary generations are physically retired after
original source/recipe and636/637/633actual compiler-input recovery. Together
with closed output and a consumed review,333files/43379170B are removed;all15
inference ELFs remain unchanged. Rejected source,targets,controllers,fixtures
and the obsolete plan are removed from the active tree. Restore them from the
recorded original commits;the current result keeps compact deletion evidence.

The exact current bbd345/e1b011 generation now passes the small active-from-exec
collector check under `adaptive_payload_collector_active_v1`,using its df574bb8
same-build fixture and the existing indexed collector. All62expected kernels
have names and unique native host-flow joins;the274446-byte timeline contains
249GPU and770host events within the owned raw-clock bounds. Native exit is0,
parameter validation is quiet,owners/session close,boot/driver are unchanged
and no fresh fault is recorded. The read-only host readiness check also closes
with no failures. This qualifies only the small compatibility-graph control.

The current-payload V3model diagnostic exits native0with closed owners and no
fresh fault,but remains **failed**:the clean-speed environment omitted
`STRATA_TRACE`,so the required actual prefill-policy marker is absent. Its
post-failure caller evidence independently conserves request2:3.39840s wall,
3.02440s main-thread CPU,1.44720s nested native waits consuming1.40538s CPU.
Disjoint capture/layer-wait/pool regions take0.68150/1.04063/0.74124s. These
short24-output cold-request observations do not qualify sustained attribution,
observer cost,complete device coverage,critical path or a new speed result.
The original failure is retained under `adaptive_payload96_device_service_v3`.

The fresh V4plan explicitly enables actual-window/progress records and persists
RAW/MONOTONIC pairs as sampled,including on later reduction failure. Qualify
its bounded thinking-on MTP4 window and observer cost before interpreting the
dependency/critical path. Prior b310/429b/cb7model evidence does not qualify
this ELF. Join host work/waits/submission and GPU events without adding
overlapping services. Physical RAM/VRAM/PCIe traffic and saturation remain
unknown;existing Luna-reviewed references cover the method,so no duplicate
survey is selected. Keep prefill independence,physical262144/RESTORE and
task-quality gates open.

The selected GPU cache already retains original packed native expert weights.
The immutable48-layer pack and compiled per-layer plan give1256slots and
2575206400bytes(2.39835GiB),matching the runtime2.40GiB header;47layers own
26slots and the last owns34. Slot fills copy raw blobs and the captured native
hit path passes their device pointers plus per-layer formats directly to the
SYCL grouped kernels. Q8_1 names activation arithmetic,not expanded weight
storage. This closes the source question,not a new speedup or saturation claim.
Varied-prose
preparation still requires its own closed32/96cohort.
Raw/JIT output from all six closed processes and the consumed cache review are
retired: 415 files / 78,704,633 bytes. Each closure recovered all 358 Git inputs,
preserved all 15 ELFs/six immutable receipts and passed full continued admission.
Only generated config/plan remain in those six process roots. Per-process
manifests/hashes and original results remain in the compact result;the current
CPU representation retrieval review has its separate named consumer in the
research index.
The completed preparation command is:
`/usr/bin/timeout -k 15s 900s /usr/bin/python3 -X faulthandler -B tools/sycl/profile_adaptive_cpu_leaves.py regular-code prepare`.
After committing the plan,run the same command with`host`instead of`prepare`.
After committing normal frontend closure,run indices0..5in declared order,
each using`/usr/bin/timeout -k 15s 3600s /usr/bin/python3 -X faulthandler -B tools/sycl/profile_adaptive_cpu_leaves.py regular-code INDEX`.
Commit each closed process before the next;stop on the first failed gate.
The six consumed reader/audit/review files(30333bytes)are retired after37-input
Git recovery and before/after source/runtime admission. Compact findings and
the deletion manifest remain in the current host result/research index.
For a later full adaptive census,the reviewed design refreshes membership on
the request owner only after the adaptive worker joins and refill publication
succeeds,latches a published-map epoch per window,and rejects masked/in-flight
state. Preserve final-job NT/footer conservation. The existing library is
sufficient for that design;follow`adaptive_cpu_census_retrieval_review_20261107`.
The selected e1b011 path multiplies usage counts by 0.7 after each adaptation;
neither arm overrides this default. At every eight verifier rounds, the score
half-life is about 15.55 rounds. The generated-token interval varies with
acceptance. The minimum candidate count is 2 and replacement advantage is 1.5.
Changing cadence also changes recency and count scaling, so it needs a new
policy comparison. This full-RAM ArenaExpertSource route leaves
FileExpertSource's complement exchange inactive. Normal adaptive payload
copies are incoming H2D; victim D2H belongs to the separate RAM-complement
path, and audit readbacks are observers. The clean SYCL_EXPERT_COPY footer
excludes adaptive uploads, so its zero submitted bytes do not establish zero
bus traffic. Physical traffic and saturation remain unmeasured.
The read-only capacity audit uses the same recorded native inputs.
`NativeDense::load`uploads the original packed GGUF matrices and keeps its
optional layer-bank snapshots in owned RAM;that owner is not a second full
FP16 expansion in VRAM. This finding covers NativeDense,not the whole process.
`layer.cpp::kv_plan`selects streamed mode1when`--kv-resident`is below the
configured context. The original8/32cohort's QSA16 branch requires mode0,
INT8,all pages resident and no host backing;its context/residency remains131072.
The independently authored streamed-QSA16 extension passes its small/full262144
producer components,the same-build native prerequisites,and actual typed-QSA
32769/82/32769model handoffs at logical262144/resident20480/owned stage. Its
separate clean32K`bench.sh`baseline now passes all four requests:subsequent-three
median514.2599PP/clientTTFT63.863s in one process. Nine fastest552b prefill
choices remain disabled in that admission configuration. Their separate
`qsa16_stream_fast_entry_v1` composition now passes actual long/short/long
route,packed-bank/cache-loan,MTP expert restoration and decode recapture
checks on the same native build. Its clean five-length `bench.sh` campaign
now closes all20thinking-on24-output requests,with normal0,closed owners,
quiet validation,no fresh fault and15final model hashes. Subsequent-three
medians are794.9319/876.5926/916.5706/934.2008/946.8245PP at actual
32769/49153/65537/81921/98305inputs;clientTTFT is41.357/56.218/71.662/
87.866/104.007s. Each length uses one fresh process,with first-use samples
separate. All residual rows stay on GPU. The96K128-slot split accumulator
route stays separate from the ordinary384-slot32..80Kfit. The separately
committed57345-input point also passes all four requests and15final hashes,
with911.2347PP/63.084s clientTTFT subsequent median. Only the observer changed;
688other source hashes,configuration and native binary remain exact. On five
compatible lengths,quadratic has0.956%mean/1.780%maximum whole-length held-out
prompt error versus affine1.618%/3.908%;first and warmed timing scopes stay
separate. Treat this as bounded interpolation,not a physical cost law. The
current-build collector V2 now passes its exact1/37/24kernel-class and
host-flow checks,normal0,quiet validation,closed ownership and no fresh fault.
The matched32Kmodel clean/profile/clean bracket now closes under
`qsa16_stream_fast_profile_v2`:six requests,normal0,quiet validation,closed
owners,no fresh fault and15final hashes pass. Profile overhead1.071..1.092
passes the prospective1.2screen. The second observed request has29.775s of
GPU kernel union,10.152s M2D copy service and negligible observed copy/compute
overlap;main CPU has21.977s inside waits and20.304s outside,including0.074s
row grouping. Physical traffic,saturation and the limiting resource remain
unknown. The first streamed KV bulk-mirror candidate now passes separate
small/full off/on producer oracles,including physical262144fill and last-cell
overwrite,at component source`a4162bf1a396c8633bcc0f34f743b1c772b2fdb5`.
Its full engine is built from`611a0fd6fecc4593e17160c294103cd4464a292b`
(SHA`56cad9eb45f8f5fc…`);all five fresh native checks close at`65cff97e1`.
The fresh actual thinking-enabled32769/82/32769input handoff also closes at
`f79a5cd2145efb91ff85445710bbb31f62f00e61`,with32outputs per request,
normal0,closed owners,quiet validation,no fresh fault and15final model hashes.
The two long-prefills publish96bulk calls,384copy commands and830472192
requested bytes,with packed-bank/cache-lease binding,MTP expert restoration
and independent decode recapture checked. This diagnostic charges validation
and logging. The separate clean same-binary comparison closes at
`f0b08ded893c9ed253322b330fd825c96234357c`:eight fresh processes in
32Koff/on/on/off then64Koff/on/on/off order,three thinking-on requests per
process. All24original samples have exactly24outputs and zero reuse;all
processes exit0with closed owners,quiet validation,no fresh fault and15final
model hashes. Median-of-process subsequent-two PP is790.5512/827.4974at
32769inputs(off/on,+4.6735%)and914.7170/970.5080at65537(+6.0993%).
Corresponding clientTTFT is41.5905/39.74375and71.811/67.687seconds. Both
independent opposite-order pairs favor bulk at each length. First requests
stay separate:633.3361/650.3534PP at32Kand803.5971/847.9418at64K.
Clean bulk selection is inferred from actual typed-QSA entry,immutable
environment observed at both live gates and exact deterministic dispatch;
per-launch command counts and physical bus traffic are not measured here.
The shared runner accepts1..24outputs,so Main checked every retained count
before advancing and after closure;passed alone was insufficient. Add the
strict24gate before another cohort. The single CSV retains all eight points.
Two-length adjacent slopes are0.9215233ms/token off and0.8523079on;they
are local differences,not a fitted law or physical fixed-cost estimate.
The old multi-length curve/profile remains separately scoped. Current
source-view mode1is observed in all24requests;the old4.5512%source-view
result must be crosswalked before any duplicate experiment. Hardware
saturation remains unknown;post-bulk append service is now measured below.
The bulk flag remains
default off;1000PP,sustained decode,task/agent,full262144model/RESTORE and
adoption remain open. The remaining cost is now profiled on this faster path;
expert-copy preparation/submission is a separate candidate. The current
bulk32Kclean/profile/clean bracket now passes all6requests,observer ratio
screen,quiet validation,normal closure,no fresh fault and15final hashes.
Second-request prefix service is6.796s versus append0.0838s;copy/kernel
overlap is only0.0064s. Main outside-wait CPU is20.518s,with upload-submit
4.678s and GEMM-submit3.670s;grouping and source copying are negligible.
These are diagnostic intervals,not physical saturation or additive wall
costs. The cached-Q/vector-KV subgroup component now passes11numerical cases
(including262144live cells)and108paired timings across3fresh processes at
5dcdec7e0:the3shape median time reductions are0.6120%,0.5359%and0.9736%,
geomean ratio0.99292646. This is the same bba952e33 mechanism,not a new one;
its older<=0.95speed gate remains failed. The new prospectively committed
reproducibility gate passes without a minimum gain. No model PP/TG or
adoption follows. The explicit startup selection
`STRATA_SYCL_PREFILL_QSA_MATH=bf16-prefix-cached-subgroup`now connects
long-prefill scratch admission,dispatch and diagnostics;defaults and decode
source are unchanged. The fresh full build at`cf7ad7bc4`now produces
engine`9af65516...`and seven same-build fixtures,with131unique sources,
132selected object commands,1465actual compiler inputs and zero uncovered
inputs. Its9-stage native qualification passes at`82e8ce35a`:54CPU reader
cases,11CPU-only refusals,submission10/phase16,canonical and packed-head
numeric/restore checks,11attention oracle cases including262144live cells,
and off/on producer full-fill plus final-cell overwrite. All8GPU stages
close normally without fresh faults. At`81948ba22`,the same engine also passes
the actual thinking-enabled32769/82/32769input sequence,32outputs each,with
logical262144/resident20480cells. Both long entries confirm cached-Q/typed4/
subgroup dispatch and96bulk calls;MTP restores twice and the live decode
head records27C4replays. All15model assets match afterward,normal0/closed
ownership and no fresh fault. This is a validation-charged route/handoff
diagnostic,not clean PP/TG,task quality or physical262144model/RESTORE
qualification. Full-context,task/agent and sustained-decode gates remain open. The clean
same-binary comparison now closes at `d085a3b2977d4375df0947184cb4d91c3740ff20`:
12 fresh processes,32K ABBAAB then64K ABBAAB,three thinking-on requests each.
All36 requests have exactly24 outputs and zero reuse;all owners exit0,with
quiet validation,no fresh faults and15 final model hashes. Bulk stays1 and
expert-upload staging0 in both arms;only QSA math and private paths differ.

| Actual input | Request scope | Ordered PP | Cached PP | Ordered client TTFT s | Cached client TTFT s |
| --- | --- | ---: | ---: | ---: | ---: |
| 32769 | First request;3process median | 547.9051 | 553.2183 | 60.013 | 59.503 |
| 32769 | Subsequent2;median of3process medians | 685.9587 | 687.9200 | 47.9985 | 47.8385 |
| 65537 | First request;3process median | 746.3441 | 730.2417 | 88.108 | 89.961 |
| 65537 | Subsequent2;median of3process medians | 847.9029 | 850.6085 | 77.5140 | 77.2765 |

At each length one of the three warmed pairs is slower with cached QSA.
The prospectively declared repeated-gain gate therefore fails at both lengths;
no model/default adoption or advancement follows. Keep the selectors explicit
and preserve independent decode. All individual samples and paired ratios are
in `qsa_cached_subgroup_prefill_compare_v1`;the CSV retains all eight aggregate
first/warmed points. Local adjacent slopes are0.9009384/0.8975922ms/token,
not a fitted law or physical fixed cost. Older56cad9/827.5-and970.5PP evidence
remains a separate retained reference,not a performance claim for this build.
The initial `moe_job_geometry_v1` observer is failed at
`237f6390fc64081d8a02bc4c6ba9f00f24807eee`:the timing-coupled
`STRATA_PREFILL_LAYER_MAJOR_PROFILE=1` run stops at the announced layer18/GDN
stage until the480-second watchdog aborts the native process(-6). Planned
censuses cover layers0..17 only;zero requests complete. The original failed
ownership result remains false;the native/wrapper are gone and the owned
benchmark session is independently observed empty. No fresh driver fault is
recorded;GPU health and the stall's exact cause are unproved. No64K run/retry. The source review corrects the timeline:the layer18 trace follows a successful entry drain;it does not identify the subsequent blocked operation. Separate GDN-subinterval logging was not established.
The successor adds `STRATA_SYCL_PREFILL_MOE_CENSUS=1` independently of GPU
timing. Its host-only counters use existing router data after checked drains;
they add no GPU event, readback, wait or inference math. Native source
`dc645342a8e23f8fd2d76396caa6604b2f234bd7` produces engine ELF `97fda7ba`.
Its fresh nine native/component gates pass at `c734c4fc8`, including selected
decode-module identity and attention/producer 262144-cell boundary checks.
Those component checks do not qualify full-model 262144/RESTORE or sustained TG.

The actual thinking-enabled 32769/82/32769-token model sequence now closes at
`46bb1a8313cd89563676dbc9c9742b5564f16b62`: normal exit0, closed owners,
unchanged model/runtime identities and no fresh driver fault. Timing observation
is off; the host census is on. Each long request completes 48 planned and 48
accepted frames, with identical histograms: 22320 accepted expert uploads,
zero cache hits and 45,841,868,800 requested weight bytes. There are 22479 jobs,
131 split experts and 15,728,640 routed-row assignments. Splitting adds jobs,
not another weight upload for that expert. First expert jobs below128 rows
account for 9474 uploads (42.45%) and 19,470,156,800 requested bytes (42.47%),
while all sub128-row jobs account for only353715 row assignments (2.25%).
These are logical requested bytes and a token-tiled diagnostic workload, not
physical PCIe traffic, saturation, clean model speed or a bench.sh census.
Representative bench.sh32K/64K geometry and resource attribution remain open.

The compact16x64 GU/SwiGLU adapter comparison closes at
`d6bbdccf892a5293c38e5c8f2c6a7cb25513b03a`. Fixture source is
`116ba612755e3890a41de0b79b7ff88f421e49f6`; it links the exact current
native archives/images without rebuilding or changing engine dispatch.
Smoke plus20 compact/wide numerical cases pass; every observed difference is
0FP16ULP, BF16 GEMM captures are exact and input/guard checks pass.
The first timing attempt remains failed:8192 repeats did not reach the
declared150ms calibration floor. The new committed plan increases only the
finite repeat cap to262144, preserving150ms calibration and100ms sample floors.
Three fresh processes each complete four rotated matched pairs per shape,
with parameter validation, no API/profiler trace and no GPU timing events.

| Rows | Fused/separate median paired ratios, by process | Component follow-up |
| --- | --- | --- |
| 1 | 0.59174, 0.58872, 0.68659 | Eligible |
| 16 | 0.54409, 0.46521, 0.51260 | Eligible |
| 64 | 0.88536, 0.82526, 0.86605 | Eligible |
| 128 | 1.00226, 0.98035, 0.97052 | Gain gate failed |
| 256 | 0.93813, 0.98902, 0.97351 | Eligible |
| 640 | 0.91017, 0.97602, 0.89904 | Eligible |
| 1280 | 0.96566, 0.96653, 0.96583 | Eligible |

Both measured arms in that first comparison use the compact adapter.
Production auto dispatch uses compact below128 rows and wide at128 and above;
those earlier larger-row ratios do not compare against production.

The follow-up production-auto comparison closes at
`45efc75dc1888aa502b3245c597bd4c6abe96e67`, fixture source
`e07ce328f9deffbf356e048b0ac4fd984382f2ea`. Baseline calls the actual
`Gemm::try_gu_swiglu16/f16_out16` and native SwiGLU with immutable auto-tile
selection. Candidate calls the compact fused adapter plus common finish;
its future production binding is still excluded. Smoke and32numerical cases
pass with exact BF16 captures and0FP16ULP observed error. Three fresh processes
each complete four rotated pairs at12row lengths, with the same150mscalibration/
100mseligibility floors, validation, no API/profiler trace and no GPU timers.
The repeated-gain gate passes only at1,16,32,64,65and96rows:per-process ratios
span0.522..0.915. At127rows two processes have ratios just above1, so that gate
fails. At128..1280the compact candidate loses against the production wide
baseline (ratios1.487..2.199). All12points, including failed candidates, are
in the CSV; useful individual pairs stay in the committed result.

The prefill-only candidate is implemented at `a231eb7b3`, with an explicit
per-job eligibility overload at `be0dff683`. Startup
`STRATA_SYCL_PREFILL_GU_SWIGLU=auto` selects compact fusion for1..96rows
and retains existing wide fusion at2334or more. Rows97..2333 retain the
separate path;existing0/large selection is preserved. The new mode requires
native16 BF16,vllm and auto tile and is not selected by existing configurations.

The actual-production comparison closes at
`d5ea1cb8e7b7e8feaa89c0af93df9a02e82ef509`, wrapper/fixture source
`213dd970f757460d15b13c994ff17a8f42c88d18`. The private archive replaces
only `gemm16.cpp.o`;all11other members,original archives/images and nine
baseline ELFs are verified. The new admission treats historical ELFs as
provenance,checks the actual nine outputs and passes2positive/23negative CPU
controls. No original failure is relabeled. Both timed arms now call actual
production binding,validation and finish under immutable startup configuration;
only per-job compact eligibility changes. Every actual fused/fallback outcome
is asserted. Smoke and44numeric cases pass,with0FP16ULP observed,including
95/96/97,2333/2334and8192rows. Three processes complete4rotated pairs at18shapes.

All changed shapes1,16,32,64,65,95and96pass the repeated-gain gate. M1 process
ratios are0.58767/0.58403/0.61874;M96ratios are0.84620/0.89288/0.87049.
Median per-call times across process medians are24.157to14.296us atM1and
39.572to34.569us atM96. The unchanged-route controls use the same selected
algorithm but show timing spread,including slower127-row samples in all
three processes;timing equivalence and model/phase preservation are unproved.
The two arms use distinct output allocations,so do not attribute that spread
to a production-code regression or dismiss it as noise. All18points and original
pairs are retained through the CSV and pinned result.

The model relink is now built at controller source`53bd39540`,engine
SHA`dcb8efd4698d4c9c6de913cb5f92e2229e945b242f24bd1fca2998e4cf89e6d1`.
Relinking original objects first reproduces the exact97fda baseline;only the
qualified GUv5 archive changes in the candidate. The Gemm class layout and
old declarations match after removing the one added overload. Eight exact
original fixtures are reused because none links the changed archive. Fresh
nine-stage native/phase/head/attention/producer gates all pass under
`gu_small_model_device_v1`,including262144component fill and last-cell overwrite,
normal closures and no fresh fault. These fixtures do not exercise changed
GU math;the separate44-case production-GU evidence remains its prerequisite.
The first relink controller's post-link bookkeeping failure remains failed
at`2985c1153`;its unused outputs are retired,not adopted retroactively.

The actual thinking-enabled32769/82/32769model handoff now closes under
`gu_small_model_entry_v1`at controller source`588630940`,with32outputs each,
23/27,24/27and23/27accepted/offered drafts,normal0,quiet validation,closed
ownership,no fresh fault and15final model hashes. The validation/census-charged
native PP readings are573.4first and792.6after the short request;these single
diagnostic readings are not clean speed or an improvement claim.

A source-backed posthoc analysis finds existing actual returned-boolean
aggregates in`moe_window.inc`and`layer_major.inc`;new GPU instrumentation is
unnecessary for the next narrow GU comparison. In each long request,GU has
10224fused jobs/7359289rows and12255separate jobs. These exactly match8530
compact jobs/246809rows plus1694large jobs/7112480rows from the observed plan;
Down's1694fused jobs match the large subset. Both complete planned/upload
censuses equal the original46bb reference,so their duplicate histograms are
replaced with a pinned reference and compact counts. This is aggregate dispatch
evidence,not a per-job trace,time attribution or model adoption. The legacy
GU summary's`min_rows=2334`label describes the old wide threshold and must not
be interpreted as excluding the separately logged compact1..96policy.

The fresh`gu_small_prefill_compare_v1`plan is prepared at source`6be38874d`;
current-source frontend checks pass. It uses12fresh processes,32K ABBAAB then
64K ABBAAB,three thinking-on bench.sh requests per process,exactly24outputs,
first use separate from subsequent2. Both arms use dcb8;only immutable startup
GUlarge/auto differs. Existing GU-minus-Down aggregates identify actual compact
jobs under the exact same wide eligibility,without adding kernel observers.
Parameter validation remains charged. All3paired warmed prompt-time ratios
must favorauto at both lengths to advance;keep mixed samples and separate
TTFT/first-use outcomes. A fresh repetition plan may revisit small inconclusive
gains. Reuse the preceding final15asset hashes only after identity checks and
hash all15again at cohort closure. The added Niko role-vector JSON is an
explicit uncompiled input in the prospective native reader;old hashes are not
refreshed. No GPU comparison sample has qualified yet.

Next run the frozen comparison and independently preserve sustained decode. Preserve the fastest
56cad9prefill reference and a347decode reference. The obsolete9af/552b roots
are now retired:867files/245903239bytes,with original9af Git inputs and all558
552b composed inputs recovered from Git and the original recipe. The552b
Git-only reconstruction descriptor is in the retained-build result;no private
origin receipt is required. Full-model262144/RESTORE
and task gates remain open. Dequant,gather,PCIe and handoff are excluded from
these component intervals;no model PP/TG gain or default adoption is claimed.

Global staging remains off after its failed component gate;
1280-row component gains alone do not qualify a selective model policy or
physical PCIe/CPU saturation. The native source audit closes the old helper-
snapshot gap and finds no ordered-exp/decode replacement;it does not identify
a cause for the unmatched older/current throughput difference.
The old two component ELFs remain a bounded proof-reader dependency until
that consumer is updated to this qualified successor;retire them at the current
model-entry/comparison boundary. The current97fda fallback temporarily joins
56cad9 and the independent a347decode reference;select candidate or56cad9
at matched phase closure rather than expanding the normal retention quota. See
[the resource decision](RESOURCE_PROFILING.md) for exact identities,samples,
limits and pending profiler/full-context/task gates. This does not replace the
historical552b117Kevidence or adopt a production resident/cache policy.
For a future capacity screen,the pinned main-QSA state-arena formula gives
an exact948699136-byte(904.75MiB)reduction from131072to65536configured cells
under resident Mode0/K8V8. The12QSA layers have2KV heads of256elements,
64-element INT8 scale groups,128-element pooled keys every4cells,page size4,
and one shared64-dimension RoPE table. The change comprises792MiB KV codes/
scales,96MiB pooled keys,0.75MiB page tables and16MiB RoPE;
`delta_bytes=(12*(1056+128+1)+256)*65536`. Original compiled/public inputs
agree for`layer.cpp`,`session.cpp`,`layout.hpp`,`qsa.hpp`and`kv_q8.hpp`;
the pack confirms QSA layers3,7,...47. The pinned native index gives
884044800bytes(843.09082MiB)of packed expert weights for nine extra slots
per layer(432slots),whereas ten cost936.76758MiB. This is only state-arena
and packed-weight arithmetic:other allocations,metadata,fragmentation,actual
admission,cache hit improvement and speed remain untested. Do not use free
memory reported during prefill owner release as decode headroom. This is a
separate smaller-context candidate,not a new current configuration or a
replacement for physical262144qualification. Finish the current cohort and
96candidate gates before selecting another capacity experiment.
The MTP residency source census is pinned to the actual e1b011 build's
`bbd345c632e2d472e47ad3a1e36f2cdae2740d5e` inputs. Original Git bytes for
`src/core/mtp.cpp`,its header,`expert_routing.cpp`and SYCL compatibility
code match the compiled snapshot and public files. MTP owns a separate512-expert
bank of707788800bytes(675MiB);the target's`--expert-cache`does not size it.
Each full MTP forward is one row:GPU top10 routing feeds`moe_group_resident`,
which forms`base+expert_id*BLOB`pointers for grouped S2 kernels. It does not
return IDs to the CPU for grouping. Round/chain graphs capture that bank;
the existing long-prefill release drains its stream and destroys all four
decode-graph families before free,then restores the full bank before decode.
A partial decode bank therefore needs explicit address indirection and a
qualified miss/owner transaction;changing allocation size alone is invalid.
The next MTP residency decision needs bounded route frequencies,graph-reader
and owner coverage,actual free/max-allocation admission,and copy/wait costs
against a full-bank reference. A full-bank reload time alone does not establish
a residency gain or physical PCIe traffic. No partial-bank candidate
is implemented or qualified;finish the current8/32and candidate96gates first.
All six successful raw outputs are retired after source recovery and committed
evidence. The final69files/13871094bytes and17finished configuration/frontend/
record files/183538bytes are removed;the resource root no longer exists.
All341original inputs were recovered and all15engine/fixture ELFs preserved.
The successor consumes this cohort's compact committed result only.
The earlier caller V3 failed its third pair,so it does not qualify current
long-generation overhead. Current `perf_event_paranoid=4` remains a collection
constraint;no permission change or PMU collection was made. Preserve unknown
physical RAM/PCIe traffic and GPU saturation rather than inferring them from
these host clocks. This source/recipe inspection is not a new profile or gain.

Current historical result pointers are
`cache_map_decode_census_v1.decision`, `cache_map_worker_scaling_v1.decision`,
and `cache_map_decode_gpu_service_v2`; diagnostic observations are not
clean speed comparisons.

The compact, originally failed `cache_map_decode_gpu_service_v2` capture is
useful only for selecting a new question. Its 2.154 s interior window observes
342.910 ms summed Q6K service (868 launches) and 76.589 ms mapped-row copy
service (316 launches); these overlapping services are not critical-path costs.
That earlier 479a, short-output configuration does not establish today's
long-decode ranking, and its original artifact-cap failure remains unchanged.
Both independently authored row-uniform copy comparisons are closed. The
direct-entry screen (`mapped_row_copy_component_v1`,source`7f6c38b37`,plan
`507b96cee`) has no qualifying host-USM cell;its device-only milestone remains
in the CSV. Its source/recipe are recoverable and its ended build/raw root is
already retired. Keep the original failed host-cell gates.

The production-graph screen (`mapped_row_copy_graph_v1`,source`85f33092f`,plan
`18eadae2b`,ELF`53b7cbca…`) passes eight CPU controls,80small and1120full device
cases. Its five actual translation units cover716compiler and24external link
inputs,with zero uncovered/generated inputs. Metadata changes,bit copies,
NaNs/signed zeros,odd widths/alignment and guards pass on both placements.
All three fresh timing processes close normally0,with quiet before-main
validation,closed ownership,unchanged boot/driver and no fresh fault. All600
samples report8192successful ordinary graph replays and one kernel node;the
minimum batch is323.626ms against the10ms floor. Each cell has five samples per
variant/process,with forward,reverse and mixed orders. None of the16host-USM
candidate cells improves in all three processes. Keep production unchanged;
this gate does not admit a model trial. Two device-only cells improve at one
bank/20hits,with candidate/reference geometric means0.948168and0.836202,but
they do not match the model's host source. These instrumented component results
do not qualify clean timing,model gain,physical traffic or saturation. Keep
the useful individual samples and original decisions in the compact result.
The ended fixture/build/raw root is retired after exact1281-file archive/recipe
recovery and716compiler/24link binding checks. Unused native proposals,target,
fixture and comparison CLI are removed;only Python input/provenance helpers
used by`device_fp64_probe`and`measure_q6_columns`remain. Their import does not
requalify old inputs after a source change.

The compressed-Q6 C4 component at
`src/kernels/sycl/q6_dpas_component.cpp` closes its V7 comparison.
Q6/Q8_1 quantization and signed per16 scales are unchanged;two integer DPAS
halves use a separate uint16 interleave. Its pinned reference is MIT GGML
`34ca79122`,not XeStrata;packing/integration is authored here.
Source`f97893576704e6679c89ea6c4613d912665a0824`,native ELF`7b2b8775…`:
16CPU readers,12CPU pack/shape cases,6small+4full+12timing numerical checks pass.
Five GPU processes exit0,close normally,keep validation quiet and add no fault.
All60timed samples meet their floor;130560successful one-node graph replays
include submission/completion. The shortest timed batch is423.4972ms.
The3process orders are reference/candidate,candidate/reference,reference/candidate.
The linked C4 SPIR-V image has two reachable static GenX DPAS calls;that is
packaged-code evidence,not final ISA,arithmetic utilization or dynamic call count.

| K/N/columns | Reference median ms/replay | Candidate median ms/replay | Paired geometric time ratio | Decision |
| --- | ---: | ---: | ---: | --- |
|2560/248320/4|11.089957|1.684034|0.15260586|6.55283x component speedup;eligible for head-only model proposal|
|2560/640/4|0.120803|0.144796|1.20673003|20.673% slower;keep existing small-matrix path|

Each value is the median of three process medians,each based on five samples
per arm;the paired ratio uses all three processes. Full-head process medians
are11.097119/11.089957/11.084126ms versus1.682505/1.684034/1.710992ms.
Parameter-validation and retained-loader observers are charged. No clean
model PP/TTFT/TG gain,physical traffic/saturation,task quality or262144-context
qualification follows. Individual samples,conditions and decisions are in
`q6_dpas_component_v7` and the performance CSV.
The two-gate check observes21mapped files before Runtime and34after it.
The forced Level Zero driver appears only after device initialization.
This resolves the premature-provider check;V6's original refusal remains failed.

The canonical and candidate full-head payloads are each521472000bytes;
both together plus inputs/outputs admit1050901760Device bytes.
The maximum single-allocation report is10132807680bytes. Full-head pack time
is about0.15s;fixture uploads include both layouts,not one model upload or
complete cold startup. The ended1348-file/62073223-byte raw/build root is
retired after exact1283-source/archive,8controller,800compiler/24link recovery
and original-reader replay of every timing sample. The closed CLI and its
census/timing controls are archived;the unchanged linked-image reader and six
CPU controls remain for model integration. The next consumer uses source and
fresh checks rather than the old executable.
The target-head proposal remains default off. Its standalone native owner
fixture now passes at native source afbd954c0,controller 484d403e6 and
ELF81340dc8;closed evidence is 0840eccd2:q6_head_owner_v2.
STRATA_SYCL_DECODE_Q6_DPAS_HEAD=1 requires exact Q8_1 and
STRATA_PREFILL_RELEASE_HEADS=1. Only decode C4 at2560x248320 selects it.
Canonical rows and weight_bytes/row_bytes stay unchanged for C1,MTP and DFlash2.
A separate521472000-byte device layout and equal immutable RAM packing are
owned together. Both device layouts retire for long prefill;canonical D2H is
521472000bytes and combined restoration H2D is1042944000bytes. The packed
layout is never copied back from GPU. Combined target RAM backing/snapshot
1042944000bytes fits the1GiB budget.

Fourteen CPU reader/parser/acceptance controls and five native CPU-only owner/
switch cases pass. Two fresh GPU processes close normally0 with quiet
validation,unchanged boot/driver,no fresh fault and full real-shard agreement.
Eight numerical cases check993280outputs each,543distributed/head-tail scalar
oracle rows per case,C1 and prompt-reference paths,and guards. Maximum packed/
reference absolute difference is4.17232513428e-7. Four total retire/overwrite/
restore cycles remove the old addresses from the live registry,overwrite the
reclaimed capacity,and restore every canonical/packed byte correctly. The24
ordinary graph replays use two nodes each. C4's linked SPIR-V image is exactly
the previously qualified e9f6543e image,two reachable static DPAS calls.
This proves the fixture-owned graphs and NativeHead APIs,not the production
Verifier's lease or full model. V1's overall failed result remains failed:
the caller omitted host_probe_accepted;the V2 acceptance-contract regression
and fresh processes resolve that controller defect without a native rebuild.

The whole-model build uses primary source `4c2c824e7` and the selected 90f
recipe, preserving the independent phase guards. The measured e1b tree has
24 native-file differences from current; its measured `iq3s_dot` body is
byte-identical to current. This source comparison does not prove engine parity.
The head fixture links the production `strata_engine` objects. Build evidence
`cc05a1034` records 127 translation units, 1,461 compiler inputs, one generated
input and 29 external link inputs, with zero uncovered inputs. Engine
`0b6633cd` and fixture `d59bfea1` both contain the qualified `e9f6543e` C4 image.
Fourteen CPU controls pass; the retained reader bodies are AST-identical to
`484d`.

Device evidence `0e6c9385e` passes five CPU owner/switch processes, 10 submission
and 16 phase-cache checks, and separate ordinary/packed real-head processes.
Each head arm passes four numerical cases, two retire/overwrite/restore cycles
and 12 graph replays. Packed maximum absolute error remains 4.17232513428e-7.
All four GPU processes exit 0 with closed owners, quiet validation and no fresh
fault. Full shard hashes agree for each head process. This qualifies the
production-linked fixture; model speed remains unqualified.

Real-model evidence `873907a1e` closes two fresh head0/1 processes. Each runs
thinking-enabled 513/82/513-token inputs and 32 outputs per request. Both arms
offer and accept MTP drafts, exit 0 with quiet validation and no fresh fault,
and preserve all 15 model assets. The Verifier uses two decode-C4 capture IDs
per process: 2 and 6. Their cumulative completed replay maxima are 17 and 9,
giving 26 head replays per arm; the intermediate snapshot of 9 belongs within
the later 17. Only packed decode C4 selects DPAS. Prompt and C1 retain their
canonical paths. Both long requests retire and restore the target head.
Initial free VRAM is 1,678 MiB ordinary versus 1,181 MiB packed, with the same
1,256 expert slots. Each ordinary/packed transition restores 521,472,000 versus
1,042,944,000 H2D bytes. These placement and ownership observations do not
qualify 32K speed, phase preservation, task quality or the full context.
The next consumer is matched long-model speed and resource evaluation.
That comparison now records source `67a0c9cf3`, engine `0b6633cd`, through
the current `bench.sh` frontend. Reference process0 closes normally with
32K inputs and 2,860/7,627 thinking-inclusive outputs, validating the intended
strong/weak MTP classes. Its individual samples are at
`c99e5febf:q6_head_generation_speed_v4`. Packed process1 exits1 before READY:
the extra521472000device bytes leave2.09GiB free for a2.40GiB expert cache.
Evidence `5a70c1f35` preserves the failed cohort and successful reference;
no timed candidate request or new GPU fault occurred. Keep packed head off.
The direct-canonical Q6 C4 component now closes at native source `cdbd68796`,
ELF `dbecb5c6`, with controller admission `26ad7807e`. It shares the same
canonical allocation with the reference, adding zero device/host weight bytes.
Twenty-two CPU reader/census controls, 12 native guards, six small and four
full-shape numerical cases pass. Five GPU processes exit0 with closed owners,
quiet validation and no fresh fault; maximum absolute difference is6.10352e-5.
The independent scalar oracle checks543 distributed full-head rows per seed.
Three fresh R/C,C/R,R/C processes retain60 eligible timed batches. Full-head
reference process medians are 11.068782,11.089001,11.076291ms,
versus 5.225984,5.208117,5.257649ms canonical,
for a geometric 2.117949x component gain. N640 is slower in all three
processes and remains rejected. Individual samples and the original decision
are in `q6_canonical_component_v4`; timing includes graph submission/completion
and UR parameter validation, excluding setup/upload/JIT. This shared-image
source/recipe reuses the original native artifact; it inherits no GPU result.
The source now selects this kernel only for the opt-in full-head decode-C4
NativeHead trial. It removes the extra packed host/device owners and uses the
canonical allocation;C1,prompt and other head paths remain unchanged. Selection
is immutable after load and stays default-off. The Verifier snapshots that
selection with the owner metadata across retirement,restore and lazy recapture.
The engine and fixtures now compile from `d1744729f`:127TUs,1461compiler inputs,
one generated input and29external link inputs are covered. All28CPU controls
pass. The earlier nested-path reader refusal is preserved in Git. V3 then
refuses raw device-image identity before any GPU run:the fixture retains
`250b825f` but the engine embeds `5148340a`,both26028bytes/1187instructions,
with the kernel entry order reversed. The original V3 refusal remains failed.
The prospective V4 admission keeps those exact ELFs and native source identity,
with no rebuild or inherited GPU result. It requires36CPU controls and a whole
selected-module comparison preserving every instruction,literal,annotation and
function-body order while normalizing only IDs and global/function declaration
order. Unknown instructions,modes,decorations or ambiguous IDs reject admission.
This is not final-ISA or model-speed equivalence. Production-linked device
evidence `51fbdc381` passes five CPU guards and four GPU processes:10 submission,
16 phase-cache cases,and reference/canonical real-GGUF heads. Each head passes
four numerical cases,two retire/overwrite/restore cycles and12graph replays;
canonical maximum absolute difference is4.17232513428e-7. All children exit0,
close normally,keep validation quiet and add no fresh fault. Model qualification
remains separate.
That test now requests actual131072-context/KV-resident allocation and1256cache
slots,with8192prefill chunks but initially small513/82/513requests. Passing it
will qualify placement and repeated handoff,not full-context processing or
speed. V2 reaches READY131072/cache1256 but the observer stops it before GEN:
`INFO kv_resident=0` means the main KV is fully resident;the observer incorrectly
requires131072. The owned child is SIGTERM/reaped/session-empty with no fresh
fault;the original failed/incomplete-observation outcome remains unchanged.
The prospective V3 fixes this wire contract and adds owned DRM-memory snapshots.
Its controller-only admission requires the same four exact engine/fixture ELFs,
native compiler inputs/recipe and prior GPU proof;only the two model/cohort
observer files may change. V3 closes reference0 normally with three complete
513/82/513-input,32-output thinking requests,then fails its recapture check.
Both long inputs are below the8192chunk,so `uses_layer_major` correctly does
not retire the head:decode-C4 capture2 remains live,with cumulative completed
replays14/22/36 and no detachments. No candidate request runs. Ownership is
closed and no fresh fault occurs. The original failed status and its erroneous
`model_executed=false` flag remain,with observed execution recorded separately.
V4 now closes both arms at the same context/cache/chunk settings with
10197/82/10197-input,32-output thinking requests. Each native process exits0,
closes normally and adds no fresh fault;all15model assets retain their hashes.
Each arm retires/restores exactly521472000target bytes twice. Decode-C4
capture IDs2/2/6 carry cumulative replay counts10/18/10:28actual replays per
arm,not38. Only canonical candidate decode C4 records DPAS;prompt/C1 retain
their original routes. This qualifies bounded placement/ownership,not speed,
task quality or physical262144/RESTORE. Owned DRM snapshots after the requests
show about9.91GiB resident-vram0 and1.84/1.94GiB resident-gtt;these per-client
allocations do not identify the live tensor placement or physical traffic.
Use the
existing `~/.local/share/strata-sycl/venv/bin/python3` for these tokenizer/template
controllers;the system Python lacks `regex`. The packed kernel/exports remain temporarily only to preserve the exact
qualified shared image through this comparison;review their removal afterward.
No model PP/TG,task,
physical-context or default adoption follows from the component result.
A smaller expert cache would be a separate placement comparison rather than a
repair of the failed packed same-placement cohort.
The new default-off mode2 instead tests the faster owned packed representation
as the sole GPU head owner. Its fixed geometry needs the same521472000bytes,
not a second copy. C1 and partial-column consumers must be made format-aware,
and MTP can project the shared full head then gather its subset logits instead
of materializing223227900bytes of Q6 subset weights. That would trade the
subset weights for425196bytes of one-row logits;the proposed arithmetic and
net222802704-byte saving require fresh fixture/model checks and matched timing.
The source now implements this proposal for MTP4/spec4,including format-aware
C1..4 projection,immutable vocabulary IDs,one-row GPU-logit gathering and the
existing graph retirement/byte-exact restore boundary. C4 keeps the original
packed kernel;C1..3 use exact-width DPAS repeat counts and bounded input/output
spans. The installed oneAPI2026.2 `esimd/xmx/dpas.hpp` and
[Intel's API source](https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/ext/intel/esimd/xmx/dpas.hpp)
define that width contract;no SDK implementation is copied. All changes are
authored here from the owned Q6 path. The fixture adds independent real-GGUF
scalar/envelope checks for partial widths and a three-process C4 graph timing
plan;fixture-only canonical reference memory is reported separately.
The existing0/1canonical choices and dedicated long-prefill path remain the
reference. DFlash2 still uses0/1;the experimental mode2 explicitly refuses its
unadapted row-major binding. Current packed-primary component and model evidence
follows below; V4 above remains scoped to the originald174native.
The fresh sole-packed source6dd74eab7 passes36CPU controls and compiles all
four ELFs,including engine1db9bcae and head fixturee5aa53f2. Its127TUs,
1461compiler inputs,one generated header and29external link inputs reconcile.
The pre-GPU structural reader then refuses opcode70,which appears once in each
53528-byte/2443-instruction selected module. Khronos's pinned grammar identifies
it as `OpInBoundsPtrAccessChain`;the old reader supported only opcode67's
non-inbounds form. The failed build gate remains failed,no GPU ran,and a fresh
prospective reader/control admission is required for these exact ELFs.
That V2 admission now passes40CPU controls,including pointer ID/declaration
reordering,distinct inbounds semantics,changed element edges and malformed/
undefined operands. Both full2443-instruction selected modules compare equal
after only the declared ID/declaration normalization. Native source remains
6dd74eab7 and all four ELFs/actual inputs/recipe stay unchanged;the reader
admission is f518f10c8. No GPU qualification or performance is inherited.
Fresh device evidence now passes43CPU readers and seven native CPU refusals,
then five GPU processes:submission10,phase16,and reference0/canonical1/
sole-packed2 real-GGUF owners. Every head passes four numerical cases,two
retire/overwrite/restore cycles and12graph replays. Mode2also passes eight
C2/C3 cases,independent scalar checks,unchanged Q8 inputs/inactive outputs and
byte-exact packed storage checks. All native children exit0,close normally,
keep validation quiet and add no fresh fault;full shard hashes agree per arm.
This qualifies the head fixture,not the new MTP/model route or speed.
The212.5MiB calculated saving applies to decode/startup:the old materialized
MTP subset was already retired during long prefill. Computing a full head for
MTP instead of its106299-row subset adds arithmetic,so its net latency still
needs matched model evidence. Keep the faster target-head path eligible for
separate use if MTP's new subset strategy loses;do not reject both together.
The fresh three-process C4 fixture timing now closes at controller328735038,
native6dd74eab7 and fixturee5aa53f2. It compares quantization plus projection
and two-node graph submission/completion,with parameter validation charged
and full API tracing removed. Reference process medians are12.216543,
12.215415,12.227409ms versus1.682780,1.686541,1.694870ms packed.
All30samples meet their declared floor;the paired geometric ratio is
0.1381412065,or7.238970x. Three native processes exit0 with closed ownership,
quiet validation,no fresh fault and unchanged full shard hashes.
This reference is generic `native_mmvq`,the prompt-window route. The dedicated
decode-columns entry is different,so this ratio qualifies that component
comparison only. Preserve the earlier independent6.55283x columns comparison
as separate evidence;the next same-engine0/2 model trial must exercise the
actual selected decode path and MTP subset,then measure phase preservation and
long/agent/strong-and-weak thinking workloads. Neither ratio establishes TG.
The same-engine 0/2 model gate closes at controller `043650224`, native
`6dd74eab7` / ELF `1db9bcae`. Each arm completes 10197/82/10197-input,
32-output thinking requests with context 131072, fully resident INT8 KV,
1256 cache slots and chunk 8192. Both retire/restore 521472000 target bytes
twice; the canonical MTP subset retires 223227900 bytes while the shared packed
owner retires 0 bytes and retains its 425196-byte map. Each arm completes 28
actual decode-C4 replays after coalescing repeated capture snapshots. All six
requests offer and accept drafts. Both native exits are 0 with closed ownership,
quiet validation, no fresh fault and unchanged hashes for all 15 model assets.
This qualifies bounded placement and handoff. Speed, task quality and physical
262144/RESTORE remain open. The bench.sh comparison uses six fresh
head0/2 processes with the same strong/weak long-thinking bodies, judging
prefill, TTFT and sustained TG separately.
The six-process 0/2 cohort now closes: all six native exits are0, owners close,
validation is quiet, no fresh GPU fault occurs, and the final full15-asset hash
check passes. Thinking remains enabled. Every arm has the same body, output,
reasoning hashes and draft counts; exact output equality was not required.

| Workload / input / output | Reference TG median [min,max] | Candidate TG median [min,max] | Paired geometric mean C/R |
| --- | --- | --- | --- |
| regular-code /33208 /2975 |12.672177 [12.647078,12.681181] |13.354971 [13.326907,13.406964] |1.054955 |
| varied-prose /33278 /7961 |7.562103 [7.527015,7.603605] |8.021696 [8.017009,8.024354] |1.060396 |

All three TG ratios exceed1 for each workload: code1.053881/1.053754/1.057233,
prose1.066074/1.060156/1.054986. Observed acceptance is0.861847/0.348578 in
every sample, confirming strong/weak classification. Each arm has three fresh
processes with one sample per workload. Speeds use actual token counts and
unrounded native intervals; all individual samples remain in the result.

| Workload | Median PP R/C | Median prompt seconds R/C | Median client TTFT seconds R/C |
| --- | --- | --- | --- |
| regular-code, first long request |580.070640 /579.572546 |57.2482 /57.2974 |57.373 /57.438 |
| varied-prose, after code |684.835377 /697.171134 |48.5927 /47.7329 |48.768 /47.880 |

Code prompt-time ratios are1.006725/1.004763/0.989495 and TTFT ratios
1.006885/1.004705/0.988970: preserve the mixed result and pending phase gate.
Prose prompt-time ratios are0.975358/0.987319/0.982306 and TTFT ratios
0.975494/0.987366/0.981791. This comparison establishes sustained TG gains
in its common parameter-validation/audit envelope, not qualified observer cost,
physical saturation or default adoption. Dedicated long-prefill/length,
agent/task and physical262144/RESTORE checks remain open. Mode2 remains
an opt-in development candidate.
After recovery of all634 original benchmark inputs from `f1e2f622d`, the closed
generation/frontend raw logs, journals and caches are retired:427files/77802912bytes.
All12 speed samples and their original outcomes remain in Git. The cohort's
`artifact_retirement` records the deletion manifest; its private record and runtime
manifest remain for the next qualification consumer. No engine or model asset is deleted.
All six processes also have identical requested-transfer counters
for each long prefill. These are logical native copy requests, not measured
PCIe traffic or bandwidth:

| Workload / FFN rows | Expert uploads / H2D bytes | Router-ID D2H bytes | Two route-map H2D bytes |
| --- | --- | --- | --- |
| regular-code /33207 |19368 /39552998400 |63757440 |127514880 |
| varied-prose /33277 |19048 /38936601600 |63891840 |127783680 |

Native source `6dd74eab7:src/prefill/moe_window.inc` reads `rows*K*4` router
bytes back before CPU grouping, then uploads two distinct maps of that size.
The printed `source_H2D_bytes` counter advances only once per window; it counts
one map, not both maps or all H2D transfers. The expert-bank byte counter
advances separately after accepted enqueue. All48windows complete per request.
Thus the router/map round trip requests191272320/191675520bytes, versus
39552998400/38936601600expert-weight H2D bytes. Do not add the one-map counter
again to the two-map total. These values exclude other model copies and give
no copy latency, CPU-grouping cost, overlap or bus-saturation measurement.
They support the next long-prefill transfer/synchronization attribution;
byte volume alone cannot decide whether moving grouping to the GPU wins.
The six-shape Q6 C4 fixture/controller at `d003ba0b6` is integrated into
`feature/sycl` by merge `7b07921ba`. Its completed child branch/worktree are
retired locally and remotely after verifying all original commits against the
pushed primary. It keeps the actual production generic/columns references and
adds no inference dispatcher. Fixture `8b4d0ca9`, built from `7b07921ba`, covers
all14TUs/801compiler inputs/24external links with0uncovered and passes its
selected canonical-DPAS image reader. Its31 native CPU bounds/policy cases
pass with normal0 and closed ownership. The original V1 build/parser/import
results are archived at `491953c77:q6_dense_c4_component_v1` in the result file.
Controller `491953c77` now admits those unchanged fixture/census/recipe bytes
as `q6_dense_c4_component_v2`. Compiled bytes and31CPU cases are reused;
The original native identity is preserved through that admission.
It corrects the global trace-error regex that would reject the mandatory
constructor canary. A unique verified pre-main receipt must attest expected37/46;
other constructor errors and every subsequent UR/ZE error remain failures.
Fresh3positive/10corrupt parser cases,2positive/8negative trace cases and both
module-import orders now pass. The first small full-trace GPU check also passes:
four shapes/two seeds give8numerical cases and4C1 refusals, with guards and
immutable inputs. Both native provider gates attest the actual process and
environment; exit0, quiet validation, closed ownership and no fresh fault are
confirmed. Full six-shape numerics also pass:12cases,6C1 refusals, finite/guard
checks and unchanged inputs, with normal0, closed ownership and no fresh fault.
The original build/CPU outcomes are unchanged. All three timing processes now
close with normal0, quiet validation, closed ownership and no fresh fault.
All180 eligible samples remain in `q6_dense_c4_component_v2`; each process has
five samples per arm and shape, in R/C,C/R,R/C process order.

| Q6 input/output shape, C4 | Reference median ms | Candidate median ms | Geometric mean C/R time | Component decision |
| --- | --- | --- | --- | --- |
|2560x512 |0.113327 |0.110934 |1.120199 |Keep generic incumbent; one process loses |
|6144x2560 |0.336407 |0.231145 |0.709908 |Candidate wins in3/3processes |
|2560x6144 |0.356240 |0.196059 |0.520624 |Candidate wins in3/3processes |
|2560x10240 |0.587055 |0.282678 |0.472114 |Candidate wins in3/3processes |
|2560x12288 |0.683111 |0.325297 |0.483854 |Candidate wins in3/3processes |
|2560x248320 |11.079762 |5.229101 |0.471740 |Wins against columns; keep faster packed model head |

Intervals are medians of the three process medians; the geometric mean uses
each process's paired time ratio. The smallest shape's ratios are
0.934407/1.536808/0.978880: its lower aggregate candidate median does not pass
the predeclared all-three gain rule. Preserve that loss. This fixture repeatedly
accesses one prequantized matrix and charges graph submission/completion and
parameter validation; it proves no model throughput or physical saturation.
The current packed full head already has a separate approximately1.69ms result,
so this approximately5.23ms canonical candidate must not replace it. Advance
only the four winning non-head shapes into a fresh production trial, keeping
prompt/C1, the smallest shape and packed head on their existing paths. Model,
prefill-preservation, agent/task and physical262144/RESTORE gates remain open.
The closed standalone fixture, objects and successful raw logs are retired:
79files/44806457bytes, after recovering the original native/admission and
dependency sources and verifying the exact recipe and compiler/link census.
The compact result keeps every sample and original outcome; its retirement
entry identifies the external deletion manifest and retained source/census
receipts. The qualified engine and required same-build fixtures are unchanged.
The source-only prefill proposal at `2ca460fa3ec74f6be7c9760216ac4b4044b987ab`
is integrated into `feature/sycl` by merge `f3e1941a3597b4611d8d0e0835da25978083c766`.
`STRATA_PREFILL_HEAD_HOST_BACKING=1` retains the immutable initial upload's
exact canonical/packed bytes in RAM. It transfers the existing packed host
allocation into the owner, adds no GPU weight allocation, and skips the per-prompt
target D2H snapshot; restore still uploads the same bytes after reader retirement.
The flag defaults to 0, retaining the current snapshot path. For this head the
trade is 521472000 permanent RAM bytes for 521472000 fewer requested D2H bytes
per long transition, plus removal of request-scoped snapshot allocation. Neither
copy latency nor a PP/TTFT gain is measured. The ownership fixture source adds
checks for both policies' actual RAM extent, byte equality and repeated
retire/overwrite/restore; it remains unbuilt and unexecuted. The completed source
child branch/worktree is retired locally and remotely after verifying its
original identity on the pushed primary. A fresh four-ELF build, CPU refusals, traced
small ownership checks, matched model handoff and repeated PP/TTFT/TG comparison
are required; do not reuse the old same-build qualification for this changed owner.
The independent dense trial is default-off at
`STRATA_SYCL_DECODE_Q6_DPAS_DENSE=1`. It selects only the four winning non-head
C4 decode shapes, requires exact Q8_1/columns1/centering0/native-half0, and leaves
prompt, other widths, the smallest shape and the packed head on their incumbents.
`Q6_DPAS_DENSE` records per-shape static capture counts separately from the
unchanged four-arm `Q6_SHAPE` schema. Neither path adds device weights.
`tools/sycl/build_q6_dense_model.py` prepares a fresh private candidate with the
engine, three production-linked GPU fixtures and a CPU phase/shape/cache-policy
fixture. It preserves the1db engine/four fixtures and records50host controls,
complete actual compiler/generated/link inputs and selected linked-kernel
identity. Compare dense0/1 with backing fixed0, then backing0/1 with decode
selection fixed. No prior GPU or model qualification is inherited; both
mechanisms need independent numerical/lifetime and repeated phase-performance
checks. Existing phase/feature generations and legacy provenance holds remain
separate; compilation alone does not qualify this candidate.
The current V3 build is `q6_dense_head_backing_build_v3`, native source
`221e95e9ad890383deb418d0a865ee32c99168b6`, engine `a347a9f8`. All50host
controls pass. The actual census covers128translation units,1462compiler files,
one generated input and29external links with0uncovered. Engine/head selected
kernel structures match under the checked ID/declaration-order normalization.
All five binaries are fresh; compilation and owned closure pass, with no GPU
execution at that build step. The compiled CPU policy check also passes:four
selected cells,prompt/width isolation and cache retirement,normal0/closed
ownership. The fresh GPU fixture and bounded actual-model routing cohorts
below now pass;broader model performance/task/full-context qualification remains
pending.
Before its first CPU policy execution, controller-only admission `84e7deb51` corrects
the output budget to include the existing144MB build tree:retain the builder's
finite4GiB total/512MiB file ceilings and120s/1GiB RSS/1MiB text bounds.
Native sources, recipe, census and all five ELF bytes must match the original
V3 build; its native revision remains221e and no GPU result is inherited.
The unchanged1db engine and its four fixtures remain the fallback.
The original unbuilt V1 plan was superseded to fix split-window audit width;
V2 failed compilation because its CPU policy target lacked matching SYCL mode.
V3 adds target-scoped `-fsycl` at compile/link without relaxing Werror. Original
V1/V2 outcomes remain in their result keys, with full records archived at
`261b7d1b3`. Their ended receipts/partial objects/logs are retired after original
source/recipe recovery:101files/36103460bytes. The successful V3 build/host/policy
raw output is also retired:13files/6686539bytes. Current five ELFs,census,
generated/configured inputs and the live planned-command input remain;source,
numeric/CPU results and original failure statuses are unchanged.
`tools/sycl/check_q6_dense_model.py` is the sole new observer admitted against
that exact V3 build. It reuses the maintained real-head supervision and strict
constructor/body trace reader without changing native or existing reader bytes.
The prospective device order is50CPU reader cases and11native CPU-only refusals,
submission10,phase-cache16,then independent head0/1/2 processes each with
backing0and1. Every head process hashes both real shards before/after and checks
the actual process/provider/environment at both gates. Keep all six sets of
numerical/transition samples and original faults;commit every closed stage and
stop on the first failure. This is fixture qualification,not a dense-dispatch
model trial or a PP/TTFT/TG claim.
That device cohort now closes successfully:50CPU readers,11native CPU-only
refusals,submission10,phase-cache16 and all six head-mode/backing processes.
The eight GPU processes exit0 with closed ownership,quiet validation,no fresh
fault and unchanged boot/driver. The six real-head processes hash both shards
before/after;24numerical cases,16partial-column cases,12restore cycles and72
graph replays pass,max paired absolute error4.17232513428e-7.
Post-hoc UR trace counts in `copy_trace_observation` show two fewer successful
521472000-byte memcpy requests for each format's backing1 process across its
two transitions:1042944000fewer requested bytes. These include fixture validation
copies;the removed D2H direction follows the unchanged source branch,not a
physical-traffic measurement. All transition clocks remain diagnostic samples,
one process per layout/policy,two cycles;no short-interval speed ratio is qualified.
After closure and recovery of the original906e85fc5controller inputs,61successful
diagnostic files/6531467bytes are retired. Only the compact private record and
path/hash/reason retirement manifest remain in that device root. All individual
numerical/transition samples and original outcomes are preserved;the current
a347/five fixtures,configured inputs and1db reference remain for the next model
comparison. Other named phase/feature consumers and legacy holds are unchanged.
The four-combination dense0/1/backing0/1 model handoffs below now pass with
head2fixed,checking real C4dense routing and prefill/short/decode state. Next
collect independent matched long/agent/strong-and-weak thinking speed.
The observer is `tools/sycl/q6_dense_model_entry.py`:four fresh
processes in0/0,1/0,0/1,1/1order,with the existing10197/82/10197thinking-input
handoffs and32outputs per request. It preserves the original221e native source
and five exact ELFs;only the committed device/model observers extend the build
admission. Require identical selected call counts and identical common canonical
phase/width tables after excluding the four selected cells. Record differing
optional MTP widths without demanding identical generated text. Commit each
closed stage;stop on the first failed gate. These short outputs qualify routing
and ownership only,not sustained generation speed.
The original observer at`cc1f28bf4`fails during Python import before preparing
the cohort:the dense builder exposes`primary.RESULT`,not`RESULT`. That alias is
corrected before preparation;the failed import has no model/native/GPU execution.
At controller source`413961faa`,all four actual-model processes now pass:
12thinking-on requests and8target-head transitions,normal0/closed owners,
quiet validation,no fresh fault,unchanged boot/driver,and all15final model-asset
hashes. Every request produces32tokens with positive MTP offers/acceptance.
The selected C4static counts remain40/14/22/5;the small generic cell remains21.
Prompt1/4and decode1/4canonical tables match across all four settings after
excluding those selected C4cells. Packed target/MTP ownership is unchanged.
Backing1reports521472000permanent host bytes,0requested target D2H per long
transition,and the same521472000requested H2D-on-restore bytes as backing0.
Restoration still uploads the full target head. These are
logical allocation/copy extents,not measured physical PCIe traffic. This closes
the bounded actual routing/handoff gate;model speed,agent/task,length scaling,
physical262144/RESTORE and default adoption remain pending. Next use the
maintained official long/agent and strong/weak thinking suite in fresh matched
processes to assess the mechanisms independently.
After recovering all291original controller inputs from413961faa and verifying
the four normal closed owners,228successful diagnostic/cache files/42019543bytes
are retired. Only the private record and path/hash/reason manifest remain.
Post-retirement admission passes with the exact source,five ELFs,census,recipe,
runtime,generator and model identities retained. The model-generator venv is
required for this check;the system-Python cleanup admission refused before any
manifest/deletion because`regex`was absent. Original model outcomes are unchanged.
The next speed observer is `tools/sycl/q6_dense_model_speed.py`. It admits only
that additional committed observer against the original1129model/build inputs;
native221e,five ELFs,census and recipe remain unchanged. Head2/backing0 and the
closed tuned128K/1256/spec4/8192/384bank/adapt96/lease1configuration stay fixed.
V2explicitly admits the mandatory historical V1control and five fresh
processes in the same0/1,1/0,0/1pair order. Each uses the same official
`bench.sh --long --long-generation --think --runs 1`and unchanged workload-id,
including the six-turn agent session. Only this observer changes;all native/
client/workload sources and the timed environment/launch helpers must match.
The V2private reader accepts current v5metadata and compares actual common
canonical windows without requiring a Verifier-prompt capture. The original
v4helper and V1failure remain unchanged. Actual18request reanalysis,corrupt
reader controls and two fresh frontend probes precede candidate execution. Keep all paired TG/PP/TTFT samples and actual MTP classes;
the nominal32Kcase is supplemental and added generation inputs must meet the
strict32768fresh-token requirement. V2now passes four route-positive cases,
13corrupt-route refusals,11corrupt-v5metadata refusals and the actual18request
reader. Fresh benchmark/server probes pass144/230modules and18/27code maps,
with normal closed owners. No candidate speed result follows from these CPU
gates. The mandatory historical control0and five fresh processes1..5now
complete all108requests,normal native/bench0,closed ownership,quiet validation,
no fresh fault and unchanged boot/driver. All15model assets pass final full
hashing. Fixed non-agent request inputs match;agent history follows each arm's
own generated answers. Candidate code TG is13.8/14.0/14.5versus paired controls
12.2/12.2/12.5;prose is8.6/9.0/9.0versus8.5/8.6/8.7. All three paired TG
ratios exceed one in both workloads,with geometric means1.146169/1.030819.
Agent-followup rate ratios are also favorable in all three pairs,GM1.078221.
Code/prose generation is3044/6937tokens in every candidate and3656/7950in
every control. Code acceptance is2186/2577versus2529/3384;prose3688/9750
versus4227/11169. Both arms confirm strong/weak MTP from all three samples.
These are model-workload gains with different generated work,not isolated
kernel attribution or70TG. The common actual canonical tables match for
decode1/4,and the independent mode10/BF16long-prefill checks pass.

Prefill and TTFT remain separate. On the strict33217/33287fresh-input cases,
candidate/reference PP geometric means are0.998543/0.973261,prompt-time
1.001419/1.027492,and clientTTFT1.001268/1.027758. Preserve every slower
sample;the decode preservation screen does not establish prefill preservation.
The nominal32088-input case is supplemental,below the strict32768minimum.
Use dense1only as the next focused decode-resource candidate;keep the552b
prefill fallback and independent selections. Physical limiting resource remains
unknown. Qualify CPU_SERVICE observer-off/on on the exact a347 binary before
attribution;host backing1is a separate unmeasured model-speed candidate.
Task quality,physical262144/tails/RESTORE,ordinary provider teardown and full
inference/default adoption remain open. All individual samples and decisions
are in `q6_dense_dpas_official_v2.cohort`;the performance CSV retains the earlier
partial milestones and now indexes the closed comparison and14.5TGpoint.
After original-source recovery,359closed-process1..5logs/journals/cache files,
ended configuration inputs and frontend probe outputs(65843981bytes)are retired
with a path/hash manifest and passing final admission. Only the private V2record
and deletion manifest remain in that cohort root. V1raw remains consumed
by the historical-admission reader;the next resource controller must use the
closed committed evidence before that temporary raw hold can retire.

V1remains failed:its postprocessor expected metadata v4from the unchanged v5
producer,and a second predicate wrongly required a Verifier-prompt capture
under `--short-read 0`. All18requests/native/bench exits completed normally;
the later independent source/five-ELF/census/recipe/runtime/asset audit passes.
V2uses that mandatory control only through an explicit committed reanalysis
proving unchanged timed native/client/workload/environment helpers. It does
not turn the original failed V1into a success. The original outcome,late audit
and historical-control proof remain in the committed result and original
source;the closed V2reader replaces neither their criteria nor their history.
The original `58a15c29b:q6_head_packed_timing_v1` retains all21 transition
samples, unchanged in the current result. Across its three processes/six
cycles, snapshot preparation takes0.283010..0.306318s (median0.290381),
detach0.103158..0.112146s (median0.108827), and restore0.137620..0.198068s
(median0.144641). Preparation includes the request-scoped RAM vector's allocation
and initialization; detach includes both D2H and free. Host backing removes
the former allocation and D2H, but still frees/restores GPU weights. These are
whole fixture-wrapper intervals, not pure copy latency, bandwidth or model gains.
All30timing samples and the named21transfer-wrapper samples remain in Git.
The ended device/timing private roots,normal traces and caches are retired:
53files/10392061bytes after recovering both original controller manifests.
Native admission still passes with the current engine/four fixtures,census and
configured recipe retained. The resolved V2/V3 model-observer detail is now
archived at its original Git commits;its failed flags remain explicit.
The last standalone canonical ELF/census is also retired:two files/
11727623bytes. Its exact737-source native revisioncdbd68796,800compiler/
24link inputs,finaldbecb5c6 ELF and full original recipe are re-recovered and
reconciled before deletion. The complete old samples remain at0ccc4ed45;
current1db admission still passes. This removes the ended standalone generation
without altering the retained5a6/90f/552b/e1b consumers or the56legacy holds.
The0b engine/fixtures and closed V2/V3/V4 model diagnostic roots are now
retired:354files/183373323bytes. Exact1119project and1145dependency source
inputs are recovered from their original commits;the original recipe and
single generated header are committed with the retirement proof. All1461
compiler and29external link identities and four final ELFs are reconciled.
The replacement5a6 engine/fixtures/configuration stay live for the imminent
sole-packed comparison;90f/552b/e1b retain their separate consumers. This
cleanup does not resolve the56legacy ELF provenance holds.
Both standalone roots are retired after recovering all 1,285 original source
files, the exact recipe, 803 compiler and 24 link inputs, and the final ELF.
The deletion removes 1,366 files / 67,207,656 bytes. Another 146 successful
log/cache files / 33,126,177 bytes are retired from the closed production
checks. Original failed/successful results remain distinct in Git. The current
model admission still passes after both deletions. Preserve the current engine
and required fixtures/configuration for the imminent long comparison, plus
the independent 90f/552b/e1b consumers and separate legacy provenance holds.

The traced wrapper samples show268..323ms preparing the RAM snapshot and
133..165ms versus289..345ms restoring reference/packed heads. These are two
samples per arm,not clean performance or physical bandwidth. The current
immutable initial-RAM backing above is now implemented and passes fresh device
fixtures;its actual PP/TTFT/TG effect remains unmeasured. Deferred packed upload
remains unimplemented. Neither is adopted for model inference.
The bounded real-model Verifier/head handoff above passes. Clean matched model
speed,agent/long-thinking,phase/length behavior,task quality and physical262144/
RESTORE remain open.
The6.55283x component ratio is not a model-throughput result.

The current device-service diagnostic targets `9922 / b310` through
`tools/sycl/profile_iq3_decode.py`, using the maintained owner, frontend and
trace readers.
Its fresh active-from-exec current-native control is closed and passes all62
named kernels and unique native host-flow joins,normal exit,quiet validation,
closed ownership and no fresh fault. The274401-byte trace contains249 GPU and
770 host events. It retains the original small control budgets;the old
`479a / 552b` control is not inherited. Both current-source frontend probes
and all16 guarded-window reader tests pass. The model capture is now closed:
one warmup and one 32769-input, 24-output, thinking-on diagnostic request pass,
with exact IQ3 footer IDs1/2 and the immutable selector. Native exit is0,
validation is quiet, ownership is closed, no fresh fault appears and all15
final model-asset hashes match. The matching arm's closed model-entry body and
prompt time provide the observer comparison. Its qualified environment has
`STRATA_CPU_WORK_CENSUS=0` and `STRATA_DECODE_TIMING=0`;the latter still enables
decode timing because the native code checks variable presence. It differs
from the current long comparison,which removes that variable. Remove the old
profiler's mandatory CPU-census extraction rather than enabling an unrelated
observer. The disjoint-clock header is not compiled into b310. Interior client
clock guards and active collection through normal exit pass. The prompt-time
ratio is0.86594, below the frozen1.5 limit; this does not demonstrate a profiler
speedup or bound its decode overhead. The guarded client interior is2.066245s.
Observed GPU-event union is939.793ms, including7084us of copy service;
Q6 column kernels contribute247.197ms and mapped-row copy kernels91.437ms of
summed service. Host API union is856.244ms, including event and command-list
synchronization; API wall time is not CPU work. These intervals can overlap.
The1.126452s without observed GPU events is not proof of device idle, because
complete device coverage is unqualified. Physical PCIe/VRAM traffic, hardware
saturation and the limiting resource remain unknown.
This first capture also retains the matched short-entry default pool-spin
policy,rather than the long cohort's20ms setting. Treat it as collection/route
admission; any representative long-generation attribution needs its own frozen
workload and matched runtime/observer envelope. Keep the compact request, clock,
route and service evidence in `decode_iq3_device_service_v1`. The verified raw
timeline, successful logs and private compiler cache are retired after307
original Git source blobs were recovered and matched. Next isolate the audit's timed effect with the
same-binary comparison below before interpreting the gap from the faster clean
reference. No new inference engine was built for this capture.

The Q6 expression compiler comparison is now closed and passes at source
`198b403f68a9683aac6158ebe30fb93ef56a9905`,with its prebuild recipe at
`70b1e7d90`. Both owned CPU operations exit0;655 actual include files match the
prebuild compiler/Linux-header images,and source,Python,qualified libraries,
completed-cohort signature and final ownership checks pass. No GPU operation
or device query occurred. The full original recipe/outcome is recoverable at
`572973470`;the maintained compact decision is `q6_expression_codegen_v4` in
the linked service result.
The installed2026.2 compiler emits binary LLVM,then its translator produces
internal SPIR-V text. The module declares Physical64/OpenCL and has both bodies
in each independent kernel family. Earlier textual-LLVM,optional text-backend
and missing-header attempts remain failed,with original source/results in Git.

The two expression pairs differ in portable SPIR-V. Public signed-byte
`dot_acc` retains private byte-array storage,an extraction loop and expanded
scalar products;this does not demonstrate native DP4A or a speed advantage.
Keep the shared multi-format `packed_dot` helper unchanged and prioritize the
separate centering candidate. After the common six-bit-per-byte mask,the
reference retains a four-iteration subtract/clamp/pack loop;the candidate's
centering core is one add and one XOR. Its address/load/store instructions
remain. The generated module is not native B570 ISA:later IGC optimization
may erase either difference,so static instruction shape is not a speed claim.

The independently authored [centering helper](../include/strata/kernels/q6_packed_center.hpp)
uses `(word + 0x60606060u) ^ 0x80808080u`. Each reconstructed Q6 byte is0..63;
adding96 cannot carry between bytes,and XOR128 yields value minus32's byte
representation. Compilation passes the fixture's all64 repeated-byte values
and per-lane mixed-word assertions. These are compile-time algebra checks,
not a device numerical gate. The retired expression fixture had no production caller and omitted
full Q6 reconstruction,scales,column reductions and surrounding optimization.

The complete Q6-kernel centering candidate now passes its bounded B570
comparison. The explicit
`native_q6_k_centered_columns_mmvq` entry and existing column entry instantiate
the same checks,geometry,byte-safe loads,dot/scaling and reduction code with
only the Q6 centering expression selected at compile time. Generic MMVQ,prompt
products and columns5..7 retain the scalar reference. The source-only
`STRATA_SYCL_DECODE_Q6_CENTERING` selector defaults to0 and requires the existing
decode-column selector and exact layout. Only the18eligible shapes below can
select the centered entry. Prompt reads retain their independent reference;
phase/token cache metadata tracks centered calls separately,and a separate
centering receipt preserves the existing phase-log format. This integration is
now built into the candidate engine below,but is not qualified for model speed.
At source `dc2299e1f`,the real C++ policy/cache test passes under GCC13 UBSan:
30original column selections,18centering selections,the other17centering
cells,prompt/type/layout/disabled/unmeasured exclusions,phase/T roundtrip,
four invalid keys,and interrupted/complete metadata retirement. All174 actual
header inputs match the prebuild dependency image. This host test does not
compile or execute the full verifier or a GPU/model route.

The next model constructor,`tools/sycl/q6_centering_sources.py`,recovers all563
b310 inputs from their original Git compositions and matches the recorded map.
A plain9922 checkout differs in30files. Only seven Q6/phase files change in
the new564-file candidate;prefill,CPU kernels,generator,placement and runtime
remain byte-identical. `tools/sycl/build_q6_centering_model.py` prepares a
replacement generation with the original optimization recipe and explicit
GCC13 header selection. Retire closed b310 products after exact source/recipe
recovery and keep552b before compiling. Fresh device/model gates remain required.
V1 preparation stops before compiler/GPU execution because it published its new
result before the inherited committed-input check. Preserve that failure;
V2 performs all admission checks before publishing its prospective result and
corrects the inherited compact recipe-helper label. V2 preparation now passes
with101controller inputs,564native files and6239toolchain members pinned.
The closed b310 products and failed V1 preparation are retired after563native,
1449actual compiler and9311external dependency source inputs are recovered and
matched. The9validated552b outputs remain. Only the b310 metadata bridge remains
for replacement qualification;the next build does not consume its retired ELF.
The V2 build now closes successfully at source92832da86,including the full
verifier and Q6 helper. All1450actual compiler inputs and29external linker
inputs match their frozen images. Five outputs are hashed;the new engine is
`a1b0b7e336495bde2482fcbe13b070cba303d49fac09a134f610699c9eaf1616`.
Configure,compile and dependency children exit0 with closed ownership.
No GPU/model execution or model-speed qualification follows from this build.
Fresh same-build host qualification now passes all four exact0,pool0,exact1,
pool1 children:384single/80pair bitwise/canary controls and28real-pool cases
per arm,with the original `e5d7a50a8e7d5ac5` checksum. All children exit0,
stderr is empty,and ownership closes. Model gates remain pending.
The fresh traced GPU gate also passes submission10 and phase-cache16windows/
12captures on this generation. Both native exits are0,strict pre-main validation
proofs pass,ownership closes,boot/providers remain unchanged,and no fresh fault
appears. This synthetic gate does not execute model Q6 arithmetic or full
record_window. The next32K model-entry controller joins each new centering
receipt to its original phase/T/column receipt and actual IQ3 request footer;
its two valid and11corrupt receipt controls pass. No model-speed result yet.
Both centering0/1 entries now pass fresh benchmark/server probes and two32769-input,
24-output thinking-on requests per process,with313identical source inputs and
all15model assets pinned. Only the immutable centering selector differs.
Both arms capture decode T1/T4 with82column calls:centering0 selects zero centered
calls;centering1 selects68/46 respectively. Their matched request bodies and
positive IQ3 request2/3 footers preserve eligible/selected row equality.
Both native exits are0,ownership closes,validation is quiet and no fresh fault
appears. These diagnostic samples are not clean speed,sustained decode or
all18model-shape evidence. The clean long-generation comparison is now closed:
six fresh processes in centering0/1/1/0/0/1 order,with four official warmups and
thinking-on code/prose requests per process. Both actual MTP regimes need three
paired TG gains;keep every sample and judge PP/TTFT independently. All15model
assets pass fresh hashes,and fresh benchmark/server frontend probes pass.
All six processes close with twelve eligible long requests,native exit0,
quiet validation,closed ownership,no fresh fault and all15final model hashes.
Reference/candidate median TG is13.3054/13.4934 for code and8.7744/8.7840
for prose. Code's three paired TG ratios exceed one,with geometric mean1.012237;
prose's ratios are0.991553/1.012267/1.004646. The original first prose failure
keeps the whole centering performance gate false;no centering adoption follows.
The next diagnostic therefore selects centering0. Preserve all independent
PP/TTFT samples and MTP classes. Evidence commit
`327c34aab1af2d116df5e8455aa74541acf7dd34` and the two CSV rows hold the
closed comparison;the clean speed reference is unchanged.

The next bounded conversion candidate is the independently authored
[`q6_f16_widen_clz.hpp`](../include/strata/kernels/q6_f16_widen_clz.hpp).
It replaces the binary16 subnormal normalization loop with a nonzero32-bit
`clz` argument and integer selects. There is no production caller. The
[`host screen`](../bench/sycl/f16_widen_clz.cpp) covers all65536patterns against
an independent exact dyadic oracle,including zero signs and raw NaN payloads.
The GCC13 UBSan host screen now passes all65536patterns with zero reference or
candidate mismatches. All139actual source/header inputs match their prebuild
images,both preprocessor and final MD lists agree,and all3owned children exit0
with quiet stderr and closed ownership. Checksum is `8c8d4b0224882325`;see
`q6_f16_widen_clz_host_v1` in the linked result. The standalone SYCL comparison
and offline SPIR-V inspection below are now closed. Production-flag numerical
checks,actual Q6-entry timing,final GPU ISA and model impact remain pending
before any Q6-only integration. Keep the original shared converter and active
measurement inputs frozen. This host pass itself is not GPU,model or speed evidence.
The one-use host executable and raw outputs are now retired after source/recipe
recovery;the compact result retains its139actual compiler inputs and checksum.

The prospective [device screen](../bench/sycl/f16_widen_clz_device.cpp) compares
the existing converter,CLZ and `sycl::bit_cast<sycl::half>` followed by a numeric
float conversion. All65536encodings run at both even and odd byte addresses.
Each arm uses the production-style two-byte copy,with read-only input and output
guards,and the host-only dyadic oracle. Reference/CLZ require exact bits for
every encoding. Native requires exact finite values,signed zeros,subnormals
and infinities;NaNs require classification and sign,with quiet/signaling and
payload changes recorded separately. A failing arm remains failed;the aggregate
passes only if all3contracts pass. No timing or production integration exists.
The native half representation and supported-device requirement follow the
[SYCL specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)
and [scalar-type reference](https://github.khronos.org/SYCL_Reference/iface/scalar-types.html);
fp16 availability alone does not establish preservation of subnormals.

The standalone [controller](../bench/sycl/f16_widen_clz_device.py) V1 preparation
failed before publishing a plan or creating its root. Its call to the legacy
preload qualifier selects an old Xe-SDK/driver inventory,including four changed
or absent system/GPU libraries. All84provider files of the current qualified
model match;this is a controller admission mismatch,not a new GPU fault or a
change during the closed comparison. No compiler or GPU process ran. Preserve
`q6_f16_widen_device_v1` as failed. V2 now reuses the exact current
same-build host/device/preload proof and sets EXPECTED_LOADER before native exec;
V2 preparation now passes with112source/control pins and the exact current
library,official loader and84provider files. The standalone executable builds
with705actual compiler dependencies and24covered external link inputs;all3
compile/link children exit0 with closed ownership. All6device arm/offset cases
now pass. Reference/CLZ have zero bit mismatches;native has zero finite,zero,
subnormal,Inf,NaN-class/sign/payload mismatches and quiets exactly1022signaling
NaNs at each offset. Guards and read-only input pass. The native child exits0
with the strict current-loader pre-main proof,closed ownership,unchanged boot
and no fresh fault. This qualifies the standalone contracts,not Q6-entry timing,
model math/speed,integration or ordinary runtime teardown. See
`q6_f16_widen_device_v2`;V1 remains failed.
The saved device bitcode also passes offline SPIR-V inspection:each of the
three template arms exposes a wrapper/direct entry pair. Native has one
`FConvert` per entry;CLZ has one `clz`;the reference retains its integer
switch/branch/phi path. See `q6_f16_widen_device_IR_v1`. Clang textual IR output
is unsupported in this installation;the initial translator parser's three-entry
assumption was corrected to six. This completes the current code-choice
inspection,not final GPU-ISA,timing or production-flag qualification. Native
half remains a prospective Q6-only candidate after the current model service
attribution. No converter or engine change is adopted from static IL alone.
The standalone executable,device IR and receipt are now retired after recovering
the original6038 source/recipe and checking all705 compiler inputs. Restore that
source when a production-flag numerical comparison is selected;no closed-test
binary is required by the current diagnostic model admission.
The Main-authored,unselected `native_q6_k_native_half_columns_mmvq` entry now
offers that comparison inside the existing Q6 column kernel. It changes only
Q6/Q8_1 scale widening,retaining packed bytes,FP32 accumulation and reduction
order. Generic/prompt products and columns5..7 keep the reference;no production
caller selects the entry. Its per-call fp16 capability check is part of the
candidate's eventual entry timing. The maintained Q6 fixture's `--native-half`
mode compares this entry with the existing columns entry;smoke inputs add signed
zero,subnormal,normal and max-finite weight scales and a subnormal Q8_1 scale.
This is source preparation only:it has not been compiled or run. The current
model cohort used its immutable old source image,whose326control inputs remain
unchanged. Its fifth process nevertheless fails the separate transitive
active-tree Q6 guard after this edit;the failed cohort is preserved below.
After the closed CPU IQ3_S comparison,commit the
production-flag build/numerical and paired-entry timing plan before executing
this candidate. The standalone conversion result does not qualify this kernel,
and final GPU ISA,model impact and decode-only selection remain pending. The
new code derives from this fork's existing Q6 kernel and own standalone SYCL
conversion,with no XeStrata implementation input. Review its retained entry and
fixture mode when this named Q6 comparison closes.
Its Orca gpt-6-luna source audit and native Luna-max review now give a
conditional source pass. The entry remains unselected;decode-only ownership is
not established. The retained5c5e compile database confirms `-ffp-contract=off`
and the production optimization flags;native-half compilation and full-entry
subnormal/numerical/timing checks remain pending.
The maintained [Q6 controller](../tools/sycl/measure_q6_columns.py) now prepares
this comparison instead of the closed centering experiment. It requires the
closed six-process CPU IQ3_S analysis and scoped committed Q6 source review,
then pins the actual5c5e production recipe,compiler/package image and84qualified
providers before a fresh build. The common inactive command-profile changes
are recorded separately;other GPU translation units match5c5e. Its
`prepare`, `build`, `smoke` and `speed` transitions require each prior result
committed. One traced32-case edge/guard smoke precedes three independent
35-case entry-cost processes,five paired batches per arm and a10ms batch floor.
A host-only2048-pair steady-clock screen is outside both arms;its positive
median must be at most0.1% of the fastest batch. The new fixture supplement
does not change kernels or numerical workloads. Require every process's
candidate/reference median time ratio below1 for each eligible shape,without
a percentage-gain floor. V1 preparation failed before creating its root or
building:Main compared the build-controller commit tree instead of the actual
materialized model sources. V2 also failed before its root/build because Main
used the compact source signature as the full source map. V3 loads the original
full record,checks its566-input signature against committed evidence and uses
those actual hashes. Both failed outcomes remain failed. At source
`4947b7746dafb0efde7da9bb2ead49dc89790d89`,all15host controls pass and V3
preparation closes with the exact source/production recipe and toolchain image.
Its13translation units and link now finish with normal0,closed ownership and
all822actual compiler/link inputs covered. The actual project dependencies
change only the candidate entry/declaration and shared inactive profile
source/header,plus the fixture. Executable SHA256 is
`67d942b16b456c8595028a73b71b0aad9f7e62563357ead216089b90a66adc8a`.
Its exact-B570 traced smoke now passes all32cases,including finite half-scale
edges,subnormal Q8_1 scales,columns5..7 reference fallback and invalid-column
refusals. Output bits,scalar/Q8-byte oracles,guards and read-only inputs pass;
the child exits0 with quiet validation,closed ownership,unchanged inputs/boot
and no fresh driver fault. All three independent speed processes now pass
35cases each,with1050individual timing samples and20ns median host clock-pair
screens. **27of35shape/column combinations are faster in all three processes**;
their geometric-mean time reductions range from0.44% to25.31%,median12.94%.
All five larger model geometries pass every measured column count;the640-row
geometry passes none,and512rows pass only4/8columns. Keep the other8combinations
on the reference,including mixed directions and the640-row/4-column outlier.
All numerical,source/provider,ownership and fresh-fault gates pass. This is
full public-entry wall time with resident data in the parameter-validation
envelope,not clean model TG/GPU saturation. Native-half remains unselected;
caller/phase policy,actual GPU ISA and model impact remain pending. After exact
recovery of382project inputs,822actual compiler/link inputs and14build commands,
the closed auxiliary build and consumed converter-review bodies are retired:
457files,51361158bytes. All1050samples and35decisions remain unchanged;the
source/fixture/parser have the imminent phase/caller comparison as consumer.
The named552b fallback and5c5e diagnostic-model14ELFs remain intact. A fresh
Orca Luna phase/caller audit now runs from this measured evidence;Main has not
read its report body or selected a model path.
The standalone converter's `prepare`, `build` and `check` actions require the original cohort's
closed decision,and each next action requires the prior plan/result committed.
The fixture/controller remains outside the original benchmark's314inputs.
It preserves per-arm semantic failures,checks both byte offsets and mismatch
conservation,and uses the already qualified traced negative-validation probe.
Separate runtime/fixture compilations and linking record actual MD/host-link
inputs. Live ownership uses a bounded separate receipt,without rewriting the
whole cumulative result on every poll. The source audit and native Luna-max
review found that population checks alone accepted contradictory signaling-NaN
quiet counts. The controller now requires quieted signaling NaNs to change the
quiet bit,and reconciles mismatch/hash counters;its negative controls include
both integer arms and contradictory native records. The positive control still
requires native NaN quieting to be accepted. All3valid and11corrupt-report
parser controls pass during V2 preparation;this is not device qualification.
Commit the generated-input/recipe pins before building this fixture. Reuse the
qualified92832source `src/sycl/runtime.cpp` and its headers,then the new helper
and fixture;this needs no third inference engine. The intended compiler is the
qualified icpx2026.2/GCC13 image,with `-std=c++20 -O3 -fsycl
-fsycl-id-queries-range=size_t --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13
-fno-fast-math -ffp-contract=off -Wall -Wextra -Werror`. These conservative
semantic flags do not qualify the later fused production build's lowering.
Use the exact B570/LevelZero/in-order runtime and immutable qualified safety
environment,with tracing/validation and180s per compiler child/60s device budgets,
64MiB raw output per job. Freeze actual toolchain/provider/dependency identities
and source-to-executable admission in that controller before execution.
Inspect the same artifact's device lowering after the semantic result;commit
repetitions,timer floor/overhead and speed criteria before a separate clean
timing experiment. The device fixture has not been compiled or run.

All six frozen model-audit processes are closed. The first fixture attempt
remains failed:its descriptive-name check rejected the runtime's `(R)`/`(TM)`
decorations before any kernel enqueue. Its source,recipe and diagnosis are
recoverable from `9be2ef797`;the superseded executable and output are retired.
V2 uses exact LevelZero,vendor8086,devicee20c and PCI0000:05:00.0 identity;
the descriptive name is printed. No inference engine changed.

At source `f18b8e548`,V2 builds with the actual b310 optimization recipe and
explicit GCC13 headers;all822 actual include/link inputs match the prebuild
image,including `linux-libc-dev`. The `--centering` fixture passes its256
compile-time helper cases,32traced device smoke cases and all35actual-model
shape/column combinations in each of three fresh timing processes. Full
output-bit,read-only input,guard,Q81-byte and scalar-dot checks pass outside
timing. Native exits are0,validation is quiet,owners close,and no fresh fault
appears. Five paired batches per arm meet the10ms floor.
Exactly18of35shapes meet the prospective all-three median-time-ratios-below1
performance gate;the other17keep the reference. See
`q6_centering_component_v2.decision` and individual samples in the linked result.
For the2560x248320 vocabulary head,the three-process geometric-mean time ratios
are0.995707 for one column and0.991889 for two columns. These are resident,
validation-instrumented,full-dispatch dependent-batch results,not isolated-call
latency or model TG. Only individually eligible shapes may enter a new
decode-only model comparison. Prefill remains independently selected.
Retire the temporary fixture after source/recipe recovery and remote result
verification. Physical PCIe/VRAM traffic and the limiting resource remain
unqualified;this tests arithmetic sensitivity at unchanged logical data volume.
Model speed,prefill/TTFT preservation,task quality,and physical262144/RESTORE
qualification remain separate gates.
The source-only implementation/controller audits at `f7a96d864` and their
native Luna-max reviews are closed. They confirm the valid-domain centering
seam and same-geometry arms;the 32smoke cases and retained paired samples are
correct. The controller follow-up binds every ordered build argv to committed
state,persists PID/start-tick/session ownership during execution,and retains
fault/input/owner post-checks on error,including a terminal failure if the final
owner guard fails. The integer-width assumption is explicit. These source
corrections pass13Q6 host controls and5shared diagnostic-controller regressions
at source `4b0d9fb4ccfeb976c8c663c814ff91d79512830d`,using
`python3 -B -m unittest discover -s tests/core -p test_q6_columns.py -v` and
`test_device_fp64_probe.py` respectively. These controls use mocks and bounded
local files;no real toolchain inventory,compilation or device execution occurred.
The all-six model-audit gate remains unchanged. The source's masked word is at most
`0x3f3f3f3f`,so the reported unsigned-to-int concern is inapplicable here.

The compiler-only fixture/controller,separate result and generated outputs are
retired after exact source recovery and remote-evidence verification. Their
deletion manifest also closes process3 raw output and the consumed source-review
transaction. Keep the compact lowering decision for the next whole-kernel
comparison. The helper remains the unselected current candidate.
The compiled verifier's `window_mmvq` can select columns for GDN QKV/gate/output,
QSA key/value/query/output,and the final vocabulary head. Its generic kernel
name and247.197ms aggregate do not attribute that service to the vocabulary
head alone. A later shape/resource discriminator must keep those callers
distinct. Any favorable dot expression must be scoped inside Q6's dot path;
changing the shared `packed_dot` helper would also change other formats.
This source check uses the actual build-record verifier bytes
`ce53f5c2b56e6ace16000d807c84e3f6307e58ff14e1baa8e7168c143114e22c`:
its dispatcher and all eight call-site contexts match the current tree.
The complete verifier file differs because later observer source is not part
of b310; whole-file source equality is not claimed or needed for this map.

The earlier CPU-only [mixed-format calibration](../bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-current.json)
at source `6e40afc1b` already records five samples per cell and achieved
256 MiB read references of 30.642 GB/s with one thread and 29.403 GB/s with
six. Its 28-job stream cells scale 2.69..4.10 times from one to six compute
threads. For six of seven formats the six-job stream/reuse latency ratio is
1.014..1.098; IQ4_XS is 1.563. These are prior-source calibration sensitivities,
not physical DRAM traffic, a bandwidth ceiling, current model CPU service or
proof of saturation. Reuse this evidence when selecting the next resource
question rather than repeating the same calibration without a changed question.
Read-only `lscpu -J` on 2026-11-04 identifies a Ryzen 5 5600X with six physical
cores, 12 SMT threads and 32 MiB L3; the earlier 256 MiB read buffer is eight
times that L3 capacity. Configured DIMM speed and physical RAM bandwidth remain
unknown; the current user cannot read the DMI table, and no ceiling is inferred.
Read-only PCI sysfs inventory at2026-11-06 13:51UTC resolves the B570 path as
`00:01.2/01:00.0/02:02.0/03:00.0/04:01.0/05:00.0` (domain0000).
Read `vendor`,`device`,`class`,`current_link_speed`,`current_link_width`,
`max_link_speed`,`max_link_width` in the resolved endpoint directory and every
parent;no Sysman,driver setting,collector or workload change was made.
AMD00:01.2/01:00.0 and Intel03:00.0(e2ff) report current16GT/s x4,max16GT/s x8;
AMD02:02.0(57a3) reports current/max16GT/s x4. Intel04:01.0(e2f0) and the
05:00.0(e20c) GPU endpoint report current/max2.5GT/s x1. Do not use that endpoint
report as the external host-link ceiling. Intel's pinned
[BMG helper](https://github.com/intel/compute-runtime/blob/724697d73af22685918fcad4575c6bc267589ca5/level_zero/sysman/source/shared/linux/product_helper/xe2_hpg_core/bmg/sysman_product_helper_bmg.cpp#L1288)
selects an upstream-connected topology;its
[card-bus selector](https://github.com/intel/compute-runtime/blob/724697d73af22685918fcad4575c6bc267589ca5/level_zero/sysman/source/shared/linux/zes_os_sysman_imp.cpp#L160)
walks two parents,which identifies03:00.0 here. This is source-supported topology
interpretation,not verification of the installed Sysman implementation.
The observed16GT/s x4 path gives `16*4*128/130/8 = 7.876923` decimalGB/s per
direction before packet/protocol overhead,sharing and host-copy costs,using
[Intel's encoding calculation](https://github.com/intel/compute-runtime/blob/724697d73af22685918fcad4575c6bc267589ca5/level_zero/sysman/source/api/pci/sysman_pci_imp.cpp#L22).
This is an encoding-only link bound,not achieved transfer bandwidth or model
bus traffic/saturation. Use it to bound the next serialized transfer measurement;
the current critical resource remains unresolved.
The observer build's actual GGML x86 quantizer compile command already uses
`icx -O3 -march=native -fno-associative-math`. Static disassembly of
`ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/x86/quants.c.o`, SHA256
`b8942d84db7db78a72afeb43896206c0196a8e49e205601f860897106bd8a72e`,
finds AVX2 `vpsignb`, `vpmaddubsw`, `vpmaddwd` and vector FMA in
`ggml_vec_dot_iq3_xxs_q8_K`. Missing vector code is therefore not the premise
for the next pair experiment. This is static code evidence, not measured
dispatch frequency, arithmetic throughput or saturation.

The current native-expert inventory has 17 IQ3_XXS, 20 IQ2_S, ten IQ3_S and one
IQ4_XS GU layers, each with 512 experts. Its `native_experts.txt` SHA256 is
`d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d`.
For a possible future exact 4-bit IQ3_XXS CPU view, 128 code bytes, two scale
bytes and four group-scale bytes make 134 bytes per 256-value block, versus
98 in the original packed form. At 2560x640 per GU matrix, all 8704 affected
experts would need `17*512*2*640*10*134 = 14929100800` additional bytes
(13.9038 GiB). The original GU payload is 10918297600 bytes; both would total
25847398400 bytes. The current raw arena remains necessary for GPU cache fill
and fallback, so the proposed CPU view adds its full size, rather than only the
4.01 GB representation-size difference. Down, activations, other formats and
allocator overhead are outside these GU payload totals. No such view, owner,
conversion or allocation admission is implemented or qualified. Compare its
conversion/reuse cost only after a named component plan; the immediate pair
candidate below avoids this expansion.

The decode-only `service_profile.hpp` observer separates GU, hidden
quantization and Down by format, token count and worker, with nested wall and
thread-CPU clocks. Its replacement build, real-pool off/on parity, small device
checks and short model/census entry are now closed. Representative long-decode
collection is closed: all six processes and twelve long requests pass, and all
three paired incremental observer-cost bounds qualify as detailed below.
The retired a99 generation's measured snapshot used two pool files from
`6e40afc1b`, rather than the plain Git `05a9` pool files. The observer hooks
are applied to those recovered bytes while preserving the eight
candidate-specific files; a checkout of the recipe's commit alone would not
recover the complete measured source snapshot.
The observer constructor is now retargeted to that complete 560-file snapshot,
with the checked output-v2 delta. Its source check passes at
`1e791e80bde6366c37469a07476f4518594bc1a9`: all 563 projected files are checked,
and reversing the seven observer/build/test files recovers all 560 measured
inputs byte for byte. The eight Q6/phase files remain preserved. Follow
`cache_map_cpu_service_sources_v1.retarget_v2` for the source-only proof.
The replacement builder targets that snapshot and preserves the 479a fallback.
Its source preparation passes at `8e2499937969dbfa1dcd950b95025d9712037a83`;
follow `verify_window_cpu_service_build_v2`. It keeps the existing build settings and enables only
the observer. The four targets are the engine, real CPU-pool fixture, submission
fixture and phase-cache fixture. The separately qualified pure writer fixture
stays excluded because `STRATA_BUILD_TESTS` remains OFF; the target-list
correction is checked separately from the unchanged source constructor.
The closed a99 comparison outputs and superseded unbuilt observer archive are
now physically retired: 1293 files and 141950711 bytes, with original source,
recipe and compile evidence checked before deletion. The original immutable
build receipt remains a source-only input to the replacement; its deleted ELF
is not reused as qualification. Follow
`retained_builds_review_20261103.current_replacement_retirement` for the manifest
and remaining consumers. All nine fallback ELFs and the research-pinned
runtime/configuration inputs remain intact. The observer build is now closed
with actual compiler/link input coverage and four hashed outputs, from source
`8e2499937969dbfa1dcd950b95025d9712037a83`. The real-pool off/on checks both
pass 28 cases with identical output checksums and normal closed processes.
The same-build small device gate passes. No measured CPU service or speed gain
follows from those host/device checks alone. The
`tools/sycl/cpu_service_device.py` reuses the maintained small submission/phase
runner in a private context and requires that exact real-pool parity. Its
prospective preparation does not execute a GPU kernel.
V2 preparation failed before GPU execution: the supervising process allowed a
1 MiB file while preparation writes the maintained result JSON (about 5 MiB).
The exact committed prefix was verified, the complete committed state restored,
and the closed exit-1 failure retained. V3 preparation passes at controller
`1ef6e18ec5e266ed6ced67b704f6cf08aa038314`, with a 16 MiB file cap and
explicit result-size admission; raw output retains its 4 MiB private budget.
The normal exit-0 process is closed with quiet stderr. The exact same-build
submission/phase-cache plan is prepared under
`verify_window_cpu_service_device_v3`. Both device cases now pass: submission
has ten checks, and phase-cache covers sixteen windows with twelve captures.
Both processes exit normally with closed owners, checked validation, unchanged
boot/providers and no fresh driver fault. These synthetic payloads do not
qualify model arithmetic, measured CPU service, observer overhead or ordinary
provider teardown. The short model/census entry below uses that same build.
The Main-authored `tools/sycl/cpu_service_model_entry.py` now defines that entry
using the maintained model supervisor in a private context. It requires the
closed device gate, unchanged twelve-case reader qualification and three
complete model/census/service joins, including the warmup. Its preparation now
passes at controller `40c4efeaec6427a81db5181840b27fabddf38086`, with all fifteen
model assets hashed and the current source/provider/configuration pinned.
The owned preparation exits normally in 111.821 s, with quiet stderr and no
GPU/model execution. Both current-source frontend probes now pass with normal
exit and closed owners. The actual model entry now passes: one short warmup
and both fresh 32769-input/24-output thinking-on requests bind complete v2
service records to their independent censuses and native/API clocks. All three
requests conserve rows, jobs, entries, format/NT bins and six thread slots;
the benchmark/native exit normally, validation is quiet, all fifteen final
model hashes match, owners close and there is no fresh fault.
For the two 32K requests, summed leaf CPU time is 3.310/3.091 s, while nested
pool wall is 0.676/0.650 s. GU consumes 69.33/70.06% of that CPU time, Down
30.49/29.75%, and hidden quantization 0.185/0.190%; NT1 is 72.88/72.36% of
CPU jobs. These concurrent leaf sums include memory stalls and observer cost;
they are not request critical-path fractions or hardware saturation evidence.
The different 74-token warmup remains separate. Follow
`verify_window_cpu_service_model_entry_v1.cpu_service`; no representative
long-generation mix, observer overhead or clean-speed gain is qualified.
The Main-authored `tools/sycl/cpu_service_overhead.py` closes the repeated
comparison: three alternating off/on pairs in six independent processes, each
with both fixed thinking-on long-generation workloads. Common census, decode
clocks and trace settings stay identical. A five-percent per-pair latency bound
is interpreted only when aggregate logical work matches and both actual MTP
classes are observed; changed work is inconclusive and keeps every sample.
Preparation passes at controller `dab5a31d91a6c724e944ab921915526ec144556e`:
the six generated plans have one invariant configuration/runtime signature,
all fifteen model assets match, and the bounded decision controls pass.
Both current-source frontend probes pass. All six processes pass six native/API
census joins apiece and both thinking-on workloads, with normal exits, closed
owners, quiet validation and no fresh fault. All fifteen final model-asset
hashes match. Logical work and actual MTP classes match in all three pairs.
On/off per-output latency ratios are 1.002853/1.017201/1.026476 for code and
1.008799/1.016185/1.017022 for prose, each below the predeclared 1.05 bound;
geometric means are 1.015464/1.013995. This qualifies incremental CPU_SERVICE
cost with common observers retained, not clean speed or model adoption.
In the three on processes, GU is 68.94..68.99% of code leaf CPU and
69.72..69.74% for prose. IQ3_XXS NT1 GU, the selected component, is
20.27..20.29% / 16.15..16.16% of total leaf CPU; it is not the entire GU stage.
The first on process has summed leaf CPU 520.298/1686.105 s for 3170/7275 outputs.
If those instrumented CPU costs per output stayed fixed, six concurrent leaf
threads would have optimistic aggregate budgets of 36.6/25.9 output/s before
any other work. This conditional accounting is not a physical CPU ceiling or
clean-speed prediction: observer cost, CPU frequency, memory stalls, cache
residency and the MTP/NT mix still matter. Reaching70 requires lower CPU-service
cost or less CPU work, alongside GPU and transfer improvements.
The 69–70% GU figure is a share of summed leaf CPU time, not of total decode
latency. In the three on processes, nested pool wall is 44.53..45.01% of code
decode and 41.23..41.70% of prose decode. Removing all recorded pool wall in
the first two on processes while holding everything else fixed would yield only
about 24.0–24.1 code or 14.8 prose output/s
(`generated_tokens*1000/(decode_ms-pool_ms)`). This is conditional accounting,
not a physical ceiling: pool includes waits and can overlap device progress,
and changing it may affect other scopes. CPU tuning alone under that fixed-rest
assumption does not reach 70; retain the whole-decode question alongside the
CPU component. Before the final pair, a prospective retention addendum keeps
the already-emitted outer/nested timing footer for processes 4/5's long requests.
No instrumentation, workload, ordering, acceptance or stopping rule changes.
Those rounded host clocks can distinguish the next draft/verify/host question;
they do not measure GPU arithmetic service, physical traffic or saturation.
The final off process's retained rounded clocks show code/prose
total 257.72/247.17 ms per window: CPU pool 113.21/100.88, GPU-reach wait
84.50/84.77, and draft 33.63/33.53. In the final on process these are
264.54/251.37 total,119.07/104.81 pool,84.51/84.64 wait and33.63/33.56 draft.
These are nested host scopes, not additive device service. Trace the called
wait wrapper before attributing its cost to GPU arithmetic.
The next diagnostic source adds `STRATA_SYCL_DECODE_SERVICE_PROFILE=1`, fixed
before native exec. It partitions the main thread's decode-through-final-commit
interval into capture/cache lookup, staging, complete layer-wait wrappers,
pool calls, expert-copy retirement, graph submission and final stream drains.
Nested native blocking calls are recorded inside each region; do not add them
to the containing region. Draft/commit and other uncovered work stay explicitly
unclassified. One integer-nanosecond record per request replaces per-call logs.
The host-only `decode_service_profile_test.cpp` checks exact conservation,
nested waits, invalid nesting/clocks, exceptions and disabled observation.
The original schema1 host-only fixture compiles with GCC13.3/O2 and passes all seven cases,
with normal exits, quiet stderr and closed owners;168 actual dependency files
are fingerprinted. See `decode_service_host_contract_v1`. This does not compile
or qualify the diagnostic engine. The separate Main-authored
[`decode_wait_bridge.cpp`](../bench/sycl/decode_wait_bridge.cpp) adds five
prospective host cases through the actual `cpu_wait_profile::blocking` function:
nested suppression with both ledgers,exception propagation,disabled decode
observation with the legacy ledger still active,both disabled,and an
unclassified native wait. The maintained
[`decode_service_contract.py`](../tools/sycl/decode_service_contract.py) now
prepares V2 for the current schema2 ledger and actual bridge:compile/run the
seven ledger cases,then compile/run the five bridge cases in four sequential
owned children. Each compilation has a60s/1GiB budget,each fixture15s/128MiB,
with quiet stderr,exact footers,pinned source/compiler/package identities and
actual repository dependencies checked before execution. All seven ledger and
five actual wait-bridge cases now pass,with normal0,quiet stderr and closed
owners for all four children. The actual dependency maps contain168 and197
files,respectively. See `decode_service_host_contract_v2`;the original V1 result
stays unchanged. This is host accounting qualification,not model attribution.
Both one-use binaries
retire after the compact result is committed;the diagnostic host gate consumes
their source/recipe and results. The post-centering review confirms that the
92832/a1b0 projection lacks both this profile call and header;an environment
flag cannot enable it. After the frozen Q6 cohort closes,explicitly add the
profile to that source projection and qualify a diagnostic build. The prospective
[`decode_service_sources.py`](../bench/sycl/decode_service_sources.py) now
recovers the564native inputs and applies the pinned96b9 observer delta:
generator,Verifier and native-wait bridge,plus the header and host fixture.
The pinned1c42b4700 delta then names the three final synchronization regions
`final_compute_wait`, `final_copy_wait` and `sampling_wait`,with profile schema2.
It requires all other bytes and the complete reverse projection to remain
unchanged,and selects the original cohort's qualified arm. The source-only
contract at b5941dd8 now passes:566 projected inputs,only the five declared
changes,exact reverse recovery of all564 native inputs,and12 corrupt-receipt
rejections. The actual cohort is accepted with its original failed centering
performance gate,so the diagnostic selects arm0. See
`decode_service_source_contract_v2`,which also preserves the earlier unavailable
CLI observation and reverse-composition correction. No diagnostic root,build,
GPU/model execution or hardware attribution follows. Keep the Orca
attempt failed:its own inbox/completion calls used a mistyped terminal handle;
Main verified its final turn,abandoned the attempt and closed that exact external
terminal. The technical review remains a source inspection. The former
`head_sampling` region covers only the wait after `sample_tokens`;the schema2
names now identify that and the two final stream waits explicitly. These remain
caller wall/CPU wait wrappers,without device-arithmetic attribution. No new
diagnostic executable or runtime qualification follows from the naming fix.
Enabling CPU observers is a pending model-controller contract,not a build-recipe
defect. The prospective
[`build_decode_service_model.py`](../bench/sycl/build_decode_service_model.py)
reuses the owned native build supervisor and pins both bench/tools import roots,
the exact projected source,recipe,toolchain,compiler and link inputs. Its build
source gate now binds the closed schema2 host result and requires both actual
profile/wait headers plus the ledger fixture to match the projected bytes.
Its retirement
gate binds the two short model entries,the original six-process decision and
standalone widening V2 to their exact committed closure hashes. It requires
all those a1b0 consumers closed and that generation recovered/retired,
while preserving552b. The closed a1b0 build/source products are now retired:
all564 original inputs,the exact recipe and230 compiled vendor sources were
recovered from Git;all1450 actual compiler inputs matched their recorded hashes.
The final process's raw output,the two one-use host fixtures and closed audit
transports are also retired,884files/186722689bytes in total. All nine552b
outputs and the three configured deployment outputs remain unchanged. See
`retained_builds_review_20261103.decode_service_model_replacement` for the
manifest and exact closed-consumer bindings. The diagnostic now builds all five
targets with1451 actual compiler inputs and29 covered external link inputs;
all configure,compile and dependency children exit0 with closed ownership.
Both profile/wait headers and the actual production flags pass their native
recipe checks. Engine hash is `5c5e25d04e8ea8a8abcd9d26ed8f70a32d78b8aa93a89a49e7232ec8f47f6d76`;
the build source is70169d388,distinct from the later admission controller.
The [same-build checks](../bench/sycl/decode_service_checks.py) now pass all six
owned host children:exact/pool with IQ3 pairing0/1,then real-pool service-observer1
with pairing0/1. All arithmetic/canary/dispatch gates pass,and all four pool
checksums equal `e5d7a50a8e7d5ac5`;all children exit0 with quiet stderr and
closed ownership. See `decode_service_model_host_v1`. Both Python import roots
are pinned without mutating a shared module. The bounded traced submission10
and phase16 GPU gates now both pass,with exact payload/guards,normal0,strict
current-loader pre-main proof,closed owners,unchanged boot/provider/driver and
no fresh fault. See `decode_service_model_phase_device_v1`. These synthetic
checks do not execute the model or qualify observer overhead. Compare
profile on/off in separate same-binary processes with the existing CPU leaf
service/census enabled in both,checking clock resolution/read overhead,depths,
wall/CPU conservation,nested waits and observer effects. Keep clean speed
separate. This can distinguish CPU service from wait/submission/copy envelopes;
physical RAM/PCIe traffic remains unknown. The measured engine/source remain
unchanged;no model attribution,physical-resource ceiling or speedup follows yet.
The new [schema2 reader](../tools/sycl/decode_service_evidence.py) requires exact
request joins,complete field sets,unsigned counters,closed clean scopes,integer
caller conservation and nested-wait containment. Its
[host contract](../bench/sycl/qualify_decode_service_reader.py) compiles a real
C++ producer using the same header as5c5e25d0,covering all nine regions,nested
wait suppression,disabled observation and an empty caller interval,then corrupts
the emitted records. The contract now passes with two actual C++ records,
disabled-observer absence,all nine regions,exact clock conservation and28
rejected corrupt variants. Both owned children exit0 with quiet stderr and
closed ownership;83 actual compiler dependencies are pinned. See
`decode_service_reader_contract_v1`. The first
[model entry](../bench/sycl/decode_service_model_entry.py) now passes:one
thinking-on short warmup and two fresh32769-input/24-output requests,with all
three caller/leaf/census/native/API joins,normal0,closed owners,quiet validation,
unchanged boot and all15 final model hashes,no fresh fault. See
`decode_service_model_entry_v1`,controller source844e82e99 and native source70169d388.
The two long-input short-output caller intervals are1.814..1.826s,including
0.970..0.972s in nested native waits;caller CPU inside those waits is0.960..0.971s.
Pool wall is0.691..0.695s. Unclassified wall is0.250..0.252s,of which
0.233..0.234s is itself native wait;do not add nested waits to the containing
regions. The warmup separately spends4.163s in capture/setup. These records
show that high caller CPU includes substantial wait-API time,not that useful
CPU arithmetic or GPU hardware is saturated. Worker CPU can overlap caller
CPU and must not be added blindly. The separate core0
[clock calibration](../bench/sycl/qualify_decode_clock_cost.py) now passes five
50000-read bands with the same production header:GCC13/O2,reported monotonic
and thread-CPU resolution1ns,observed clock-pair read means0.558..0.792us,
including loop/check cost. All individual bands are retained in
`decode_clock_cost_v1`;both owned children exit0 with quiet stderr and closed
ownership. This is backend-read calibration,not production inlining cost,a
worst-case bound or whole-observer overhead. A same-binary profile0/1 comparison
remains pending before representative long-generation attribution. This short
diagnostic does not qualify clean speed,physical traffic,agent/task,
physical262144/RESTORE or adoption.
The [long caller-profile controller](../bench/sycl/decode_service_overhead.py)
defines six independent same-5c5e processes,off/on,on/off,off/on,with CPU leaf
service/census enabled in both arms. Each has four official warmups and both
fixed v5 thinking-on long-generation tasks. It requires completed prefixes4/5,
all six caller/leaf/census joins or disabled caller absence,and unchanged
clock/source/provider evidence. Each paired per-output latency ratio must be
at most1.05 with matching logical work and observed strong/weak MTP classes;
changed work remains inconclusive without discarding samples. Keep compact
caller regions/nested waits and worker operation totals,not full leaf histograms.
V3 preparation now passes at controller0b950da2e:all15 model assets and326
source inputs are pinned,and all six generated arms have one invariant
configuration/runtime signature. V1/V2 failed during preparation before any
native execution;their original failures remain recorded. Both current-source
frontend probes now pass with normal0 and closed owners. The first off/on pair
now passes both long requests per process:3170/7275 generated tokens,actual MTP
acceptance0.8244/0.3892,all six census/leaf joins and strict completed prefixes4/5.
The off process emits no caller record;the on process has six complete conserved
caller records. Native/benchmark exit0,owners close,validation is quiet and no
fresh fault is recorded in either process. The first on/off per-output latency
ratios are1.005990/1.003774,with matching logical work in both workloads. The
second on/off pair also passes both long requests,all six caller/leaf/census
joins or disabled-caller absence,strict completed prefixes4/5,normal exits,
quiet validation and fault/owner checks. Its on/off per-output latency ratios
are0.987694/0.991769,again with matching logical work. The fifth process now
fails its end-of-run source qualification:the new,uncompiled native-half entry
changed both active-tree Q6 files,which the historical reconstruction compares
outside the326control-pin map. Both long requests completed,native/benchmark
exit0,six leaf/census joins and disabled caller absence pass,and all566immutable
native source inputs/five outputs remain unchanged. There is no fresh fault.
Keep that process and V3 failed;do not refresh hashes,retry it or launch the
sixth process under this plan. The third pair and three-pair overhead decision
remain unqualified. Future model plans must freeze these transitive active-tree
comparisons as well. The first on
caller's code/prose wall intervals are237.918/834.038s,including pool regions
109.089/355.207s,layer waits77.883/287.238s and unclassified31.729/116.189s.
Nested native waits total119.421/439.711s;their caller CPU is118.554/435.462s.
These nested values must not be added to their containing regions or worker CPU.
The off observer-envelope TG13.4038/8.7556 is not clean speed. The declared
overhead ratios use reported decode time,which ends before final wait_commit,
caller-scope finish and profile/footer writes;they do not bound complete
end-to-end or final output-formatting cost. A closed-process
retention addendum keeps only four already-emitted rounded host timing scopes
and window counts to compare draft time with future caller-unclassified waits;
it changes no execution or decision criterion.
No engine rebuild or GPU fault follows from the V2 preparation failure.
Useful CPU service still does not prove RAM bandwidth or arithmetic saturation.
For planning only,the first two current on samples give a conditional
`outputs / (caller_wall - pool_wall)` of24.606..24.614 tok/s for code and
15.193..15.206 for prose. This sets only the observed disjoint pool region to
zero while fixing all other times;it is neither measured speed nor a physical
ceiling or prediction of changed overlap. In that approximation,CPU-pool
optimization alone does not reach70 tok/s. Keep GPU-reach/final waits and draft
work as separate next targets,and recompute after the final pair qualifies.
The six closed CPU-service processes also agree exactly on logical work volume
for each workload. Dividing `logical_job_blob_bytes` by generated tokens gives
the following target-verifier demand, including rejected verification rows but
excluding separate MTP draft work. GB below means decimal10^9 bytes.

| Workload | Resident expert entry share | CPU logical blob GB/output | At70 output/s,logical GB/s |
| --- | ---: | ---: | ---: |
| Code,33212 input /3170 output | 10.8501% | 0.786020 | 55.0214 |
| Prose,33282 input /7275 output | 6.7215% | 1.049027 | 73.4319 |

The numerator is one full native expert blob per unique CPU job,not one per
token entry:the existing jobs already group repeated experts across NT1..4.
Code/prose have1221484/3743079 CPU jobs;78.1594%/63.6290% are NT1 expert jobs,
which does not mean that the verifier itself mostly runs one-token windows.
Resident-entry share measures expert placement,not GPU compute share;dense,
attention and recurrent work are outside that count. Source is
`verify_window_cpu_service_overhead_v1.processes[*].cpu_census.long_requests`
at commit `c74d1539cceec80c13e9356ff73d54dfa4545587`.
The reviewed packed-row reconstruction uses `H=2560,FF=640`,
`GU=2*FF*ggml_row_size(gu_type,H)` and
`Down=H*ggml_row_size(down_type,FF)`. Summing every format/NT job bin separates
code GU/Down coverage into0.465002/0.321018 GB per output and prose into
0.620522/0.428505 GB per output. Type23 is IQ4_XS;its per-job GU/Down sizes are
1740800/921600 bytes. These are unique-per-job tensor coverage,not per-token
load-instruction counts;generic NT>1 paths may load weights repeatedly.
The census is from the8e249/6e073 observer,distinct from the479a/552b cache-map
census and the later9922/b310 IQ3 generation. The latter's builder record has
build-only scope;its separate host,GPU and actual-model gates passed later.
These logical bytes are not measured DRAM/PCIe traffic or a physical lower
bound:cache reuse,partial reads and actual memory access still matter. The
55/73 GB/s demand exceeds the prior achieved30.642 GB/s read reference,which
is itself not a hardware ceiling. Before treating a wider CPU weight view as
the next model route,compare its extra bytes with compatible read capacity
and evaluate fewer CPU jobs through expert placement/reuse. Neither cache
placement nor arithmetic is established as the limiting physical resource yet.
Read-only PMU inventory on2026-11-05 finds no `amd_df`,`amd_l3` or `amd_umc`
under `/sys/bus/event_source/devices`;the CPU and software PMUs are present.
This establishes current registration only,not hardware support or permission
to collect events. No perf counter collection or permission change was made.
Before testing `STRATA_IQ_SINGLE_EXACT=1`, the worktree now confines that
single-token GU selector to `NativeGuPhase::Decode`, preserving the prefill
dispatch. This source-only containment is outside the frozen observer build;
its compiled qualification and matched model comparison remain pending.
The next IQ3_S NT1 GU component is `iq3s_pair_gu_rows`,currently unselected by
production. IQ3_XXS and IQ2_S already have paired NT1 selections in the caller
comparison;IQ3_S still evaluates gate/up separately. This source candidate
shares activation loads and masks,keeps packed weights unchanged,and preserves
each output's block FMA and horizontal reduction. Its integer lane bound is
1904640. The512-entry official GGML `iq3s_grid` has maximum byte15 at commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`(MIT);the new scheduling is Main-authored
from this project's existing GGML arithmetic and independent IQ3_XXS pair,with
no XeStrata input. The optional `iq_single_exact_test --iq3s-pair` now defines
80 additional synthetic oracle/canary cases,all512 grids,256 high/sign bytes,
16 scale nibbles,FP16 edge scales and invalid/overlapping layout refusals.
The original default and IQ3_XXS dispatch test modes retain their scope.
The CPU-only component now builds from source00eb08ece with196project,
1376official GGML and12179external pins. All628actual compiler inputs and22
external link inputs are covered;three CPU ELFs use the same freshly compiled
expert archive. The actual public-wrapper/dot AVX2/FMA call-chain gate passes.
All six same-build correctness children exit0 with quiet stderr and closed
ownership:384single/80XXS cases,the new80IQ3_S cases,old dispatch off/on and both
28-case pool selectors. Both pool checksums remain `e5d7a50a8e7d5ac5`.
Production selection and model speed remain unqualified.
All six timing children have now exited0 with240samples and2016real-model GU
cases checked before/after each. V1's final aggregation nevertheless fails:
Main's result curation removed `receipt.direct_identity`,which the unchanged
decision function needs for independent-process proof. Keep V1 failed. Its
original full owner records and hashed raw outputs supply the separate offline
`analyze` action. That analysis now passes at source99123b122:all240original
samples,628compiler inputs,original source/build/model identities and six actual
owner/argv/environment/exit records match. It restores identities in a copy and
uses the unchanged three-pair decision,without another native execution or a
V1 parent pass. Four cells qualify:640rows/prequantized reuse/stream time ratios
are0.992100/0.987010,640rows/packing-included stream is0.991277,and128rows/
prequantized reuse is0.990867 (geometric means). The other four cells have mixed
pair directions,including a1.005874mean for128rows/packing-included stream;
do not advance those cells. The next eligible CPU question is a separately
planned640-row production entry,after the queued GPU component. The earlier
qualified service cohort attributes only13.44..13.48%of code worker CPU and
9.60..9.63%of prose worker CPU to IQ3_S NT1 GU. These are worker-time shares,
not additive wall intervals or a model-speed prediction;this component does not
cover the whole GU bottleneck. The auxiliary build/raw output,ended caller
controls and consumed IQ3 source/protocol report bodies are now retired after
exact source/recipe recovery. All240samples and the failed parent remain at
`4cb47ff93933d509788afca106291b1caec12e54`;the active result keeps its terminal
marker and separate analysis. The14inference ELFs,short-entry admission and
current Q6 report/inputs remain unchanged. This is scoped cleanup,not global
retirement of old project artifacts.
The Orca gpt-6-luna source audit and native gpt-6-luna max report review now
accept this as conditional source-level decision support:all three assigned
source hashes match and no necessary source correction was identified.
That source review itself did not establish compiler/ISA or numerical
correctness;the actual CPU qualification is recorded above. Its report/review
bodies are retired;the research index keeps the useful source pins,limits and
no-duplicate-scope decision.
After the caller comparison's failed closure,commit a CPU-only build/comparison
plan and qualify this fixture before timing. Use actual IQ3_S/IQ4_NL model
weights,128/640-row ranges,six-expert reuse/252-expert streaming and separate
prequantized/activation-inclusive cells in three opposite-order independent
pairs. Freeze exact inputs,recipe,budgets,timing eligibility and decision before
collection. Keep the IQ3_XXS production-entry comparison separate.
The CPU-only `cpu_iq3s_pair` target now reuses the maintained comparison source
with an explicit compile-time direct-component mode. Its prospective
`bench/sycl/cpu_roofline/iq3s_pair.py` controller defines the eight-cell matrix,
six correctness children,actual-symbol ISA check and three opposite-order
pairs. Neither timed arm selects a production dispatcher;the reference is the
linked GGML two-dot GU. Source,plan preparation,compilation and execution remain
separate stages;no new benchmark has run alongside the caller cohort.
The prospective component admission now uses the already closed six-process
CPU-service/census cohort and requires the later caller attempt to be inactive,
ownership-closed and honestly terminal,including its recorded source failure.
A passing caller-observer overhead result is not a prerequisite for this direct,
observer-disabled CPU arithmetic comparison. Its numerical,actual-input/ISA,
timer,three-pair and production-selection gates remain unchanged. This does not
qualify the failed caller cohort or its overhead;representative caller
attribution remains provisional. Commit this revised source before preparation.
Its separate protocol audit and native gpt-6-luna max review identify an ISA
evidence gap before any build:arithmetic checks must bind to the noinline
`iq3s_pair_dot`,with the public wrapper's call checked separately. The source
gate now records that candidate call chain and per-symbol checks,including
compiler variants or outlined chunks;unreachable symbols cannot satisfy it.
This corrects prospective admission,not an observed binary or numerical failure.
A direct-component gain must precede production-entry/pool dispatch and model
qualification. Retire the component if it does not justify that next comparison.
The earlier `iq3xxs_pair_gu_rows` IQ3_XXS NT1 GU candidate shares activation
loads between gate/up, keeps packed weights
unchanged and preserves each output's block FMA and final float reduction.
Its integer lane bound is 7872512; byte loads avoid unaligned scale/sign objects.
The maintained single-exact fixture now defines GGML parity, padded/unaligned
input, output-canary, invalid-layout, overlap and overflow checks for this
component. Its V2 same-build correctness fixture and direct single-core
component comparison now pass as detailed below. The helper
initially needed decode entry/pool qualification before a model trial;the
subsequent production and model evidence is recorded below. That evidence
does not qualify the new IQ3_S component.
The independent source audit uses the frozen scalar oracle at
`/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/cpu-service-verify-window-v2/source/src/kernels/cpu/iq_single_avx2.cpp`,
SHA256 `8579db5f02086cc660d376a77e00a9285e038f6757a53e8b2a8ef6d79eb2074f`.
Its `iq3xxs_dot` body is byte-identical to the candidate's scalar reference.
The refusal fixture now snapshots the actual weight and activation allocations,
so aliased output cases check those targets as well as the ordinary output.
It also defines a separate misaligned-output refusal. These source changes
still need the committed compiler/ISA and correctness gate before timing.
The prospective CPU-only `cpu_iq3_pair` benchmark uses 252 actual IQ3_XXS/IQ4_NL
experts from one pinned model layer. It compares the existing decode NT1
`native_gu_rows` entry with the direct pair entry, including that helper's
layout checks, at 640 and 128 rows. Each shape has six-expert reuse and 252-expert
stream access, with prequantized activation or one production activation
quantization per six experts. Even the 128-row stream's requested GU footprint
is about 60.3 MiB, above this CPU's 32 MiB L3; this does not prove DRAM traffic or
saturation. Both entries keep the packed weights unchanged. All 252 experts and
four deterministic activations need finite, bit-equal reference outputs before
and after timing; every sample also checks its timed outputs against an untimed
reference. Five samples per cell target 0.35 s with a 0.25 s eligibility floor,
four warm passes and timer cost below 0.1% of the fastest warm pass. Model
loading, hashing, allocation and reference checking stay outside timed work;
the pack-inclusive cells include the specified activation setup. The eventual
frozen plan must pin the full build/fixture/source inputs and alternate three
independent reference/candidate pairs. No minimum gain is imposed; compare all
three paired median directions separately for each cell. This is a direct
single-core component experiment, not a production-pool route or model gain.
The required observer cohort is now closed. V1 compiles, but its linker-input
admission fails before any correctness or timing execution, as detailed below.
The CPU-only CMake project also exposes `cpu_iq_single_exact_test`, reusing the
maintained correctness fixture and the same freshly compiled `expert_cpu`
library as the benchmark. This permits the compiler/ISA and correctness gate
without linking or rebuilding a GPU engine. V2 pins the complete linker inputs
before rebuilding and now passes the new fixture.
The benchmark emits deterministic weight/activation input fingerprints before
timing; require them to match across all six processes in addition to the
pinned model-range hashes. FNV fingerprints are consistency checks, not
cryptographic source provenance. The inspected CPU reference dependency is
official `ggml-org/llama.cpp` commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`; its CPU subtree is
`c38dca42a07f0aa703bdbe3e55639ad56c91dc1e`, and the selected working CPU,
headers and CMake paths have no diff from that revision. The checkout's old
directory name is not implementation provenance.
The prospective CPU recipe uses a fresh external
`cpu-iq3-pair-component-v2/build`, icpx/icx 2026.2, CMake Release/Ninja with
compile commands, and this CPU-only project with its pinned GGML directory.
Build only `cpu_iq_single_exact_test` and `cpu_iq3_pair`, with four compile jobs;
keep the known-good engine and active observer generation unchanged.
Before execution, commit the exact argv/environment and complete transitive
project, compiler and dependency identities in the compact result. Check the
actual compile/link inputs and CPU-only configuration before the fixture.
`bench/sycl/cpu_roofline/iq3_pair.py` now defines that serialized lifecycle:
prepare/commit, build/commit, correctness/commit, then six separately committed
processes. Its prospective input map covers committed project/GGML bytes,
official SDK and host linker inputs, generated version header and tools. It
requires actual compile/link coverage and CPU-only linkage, the same archive,
384 original plus 80 paired fixture cases, and complete real-model validation
before/after each process. Preparation now passes at source
`229fe7376dfeeb4c478b322934ce6fda882506cf`: 188 committed project files,
1376 official GGML files and 12172 external inputs are frozen, including the
exact generated header, tools, environment and 252-expert model-range hash.
V1 configure, compile and dependency collection all exit0 with closed owners;
compilation takes20.035s. Actual linker-input admission then fails on
`/usr/lib/x86_64-linux-gnu/libm.so.6`. Static inspection of the complete trace
finds four additional missing runtime objects: libc, libgcc_s, libstdc++ and
the loader. The pinned `.so` linker scripts do not cover these selected ELF
targets. V1 remains failed, and neither unadmitted executable may feed a test.
V2 uses a fresh root and queries/hashes all five targets and their package
versions before compilation, preserving the shared helper's existing contract
for the retained engine. The same80 pair cases now explicitly cover all128
sign selectors,16 scale nibbles,256 grid indices and signed FP16
zero/subnormal/normal/max-finite scales, with coverage checks. The arithmetic
candidate and production dispatch are unchanged. V2 preparation passes at
`39963e5ab`, with188 project,1376 official GGML and12177 external inputs,
including all five runtime targets. V2 compilation now passes, with complete
actual compiler/link input coverage, two hashed CPU ELFs, the same freshly
compiled expert archive, and only pre-pinned CPU runtime libraries. All five
owned build/inspection commands exit0 with closed owners. The same-build
correctness fixture exits0 and passes384 single plus80 paired cases, including
the explicit sign/grid/scale and FP16-edge coverage, bit parity and refusal
targets. Its process closes with quiet stderr. All six component processes
and240 individual samples now pass, with identical actual weight/activation
fingerprints and2016 real-model comparisons before and after each process.
All eight cells favor the pair in all three independent pairs. For640 rows,
candidate/reference geometric time ratios are0.9474/0.9461 without activation
packing (reuse/stream), and0.9460/0.9408 with packing. For128 rows they are
0.9490/0.9466 and0.9479/0.8947. The last cell includes one slower reference
process (63.38us/expert versus roughly53..54us); all samples remain recorded.
This is typically about5% lower direct-component time, not model gain, RAM
saturation or production adoption. This motivates the explicitly decode-only
production-entry comparison below; no model speed is qualified.
After original V1 source/recipe recovery from pushed Git history, its failed
unqualified generation and V2's closed normal build/correctness output are
retired:89 files,20090636 bytes. The failed outcome remains intact. After the
direct comparison closes, V2's two CPU executables, archive and remaining
outputs are also retired:83 files,19278175 bytes. Its original source/recipe
and all240 samples remain at `ee5a8e41307a97bdb7155004e4bc9c2a33af5e33`;
the validated GPU fallback is unchanged.
The source now offers default-off `STRATA_IQ3_PAIR=1` only at the decode NT1
production entry, below the unchanged multi-token dispatch. Prefill keeps its
existing selection. `STRATA_IQ3_PAIR_AUDIT=1` uses separate per-thread IQ3
eligible/selected row counters; the legacy IQ2 audit retains its own fields.
No process environment changes between phases. The `cpu_iq3_pair_dispatch_v3`
build now passes from source `1ec110452cdd70c425bbc31edd48bbd0421563ca`,
with three hashed CPU ELFs, checked actual compiler/link inputs and the same
fresh expert archive. No GPU runtime is linked. All four correctness children
now close normally with quiet stderr. Each selector passes384 single and80
paired bitwise GGML cases and80 audited production-entry calls; decode selects
0/10240 paired rows for selector0/1 while prefill selects none. Both real-pool
fixtures pass28 cases covering repeated prefill/decode, seven GU/Down format
pairs, NT1..4 and the scratch-batch boundary; their full output checksums both
equal `e5d7a50a8e7d5ac5`. All six timing processes and240 samples now pass,
including2016 real-model output cases before and after each process and
actual dispatch/eligible/selected-row counts for every timed batch. All eight
cells improve in all three independent pairs. Candidate/reference geometric
time ratios are0.9390/0.9425 (640 rows,prequantized,reuse/stream),0.9385/0.9627
(640 rows,packing included),0.9417/0.9257 (128 rows,prequantized),and
0.9504/0.9619 (128 rows,packing included):3.7..7.4% lower time. Both arms use
the actual production entry with the same audit counters enabled. This is
single-core entry timing; the pool fixtures qualify phase/output behavior,
not pool speed. `decode_iq3_pair_model_build_v1` now compiles from controller
source `9922f8a5a`, with563 recovered source files,1449 actual compiler inputs,
29 frozen linker inputs and five hashed outputs. Its first host gate fails
at the observer artifact cap:the launcher counts the entire compiled tree
against a1MiB host-output budget. The child exits0 and is reaped with an empty
session,but its stdout was not recovered;no numerical pass follows. Preserve
that failure and use a fresh host-v2 controller/root with the same binaries.
The two actual CPU units have O3/znver3;the fixture body is generic O3.
An auxiliary assertion mistakenly requiring znver3 on the fixture remains
recorded as failed;it was not part of the frozen native build acceptance.
The fresh `decode_iq3_host_v2` gate now passes all four same-build children
with the original exact384/80 and28-pool-case criteria;both pool checksums
match `e5d7a50a8e7d5ac5`. All native outputs are unchanged. V1 stays failed;
no GPU or model speed qualification follows. Only the six
qualified CPU/fixture files and checked per-request IQ3 route-output patch
`7969ccb9f` differ from the prior fixed-Q6 observer source; GPU math, prefill,
placement and graph/runtime sources remain byte-identical. The same-build
`decode_iq3_phase_device_v1` gate now passes submission10 and phase-cache16
windows/12 captures, with strict traced validation, normal exits, closed owners
and no fresh driver fault. Arithmetic/pool and small device prerequisites pass.
The IQ3-off model entry now also passes its two frontend probes, warmup and
two fresh32769-input/24-output thinking-on requests. Actual eligible NT1 rows
are1400960/1378560 and selected paired rows are0/0, with normal native exit,
quiet validation, all15 final asset hashes, closed owners and no fresh fault.
IQ3-on now passes the same probes and both short requests, with identical
calls/eligible rows and selected paired rows1400960/1378560. Both selectors
close normally with all15 final asset hashes, quiet validation and no new fault.
The six-process/twelve-request long-generation comparison is closed.
Candidate/reference exact TG ratios are1.016142/1.014124/1.000177 for code and
1.029070/1.052548/1.049583 for prose;their geometric means are1.010122/1.043681.
Reference/candidate arm medians are10.235157/10.391146 code and
6.690531/6.885026 prose tok/s. Each request generates3170/7275 tokens with MTP
acceptance0.82439/0.38924. All twelve requests satisfy their extent,answer and
actual-route gates;all six native exits0,quiet validation,closed ownership
and no fresh fault. All15 final model-asset hashes match. Candidate selected
NT1 rows221271040/566117760 equal eligible rows;reference selected rows are0.
This qualifies the predeclared repeated IQ3 TG gain in both observed MTP
classes. Prompt-time and TTFT directions are mixed. Dedicated prefill/TTFT,
agent/task,physical262144/RESTORE and inference adoption remain unqualified.
Both arms retain the common IQ3 audit,whose incremental cost is not separately
measured;these are not a matched comparison against the older clean TG values.
The older clean Q6 cohort and this IQ3 cohort do have identical CLI arguments,
v5 request-body hashes,actual input/generated lengths and MTP acceptance.
Their native/pool sources and common IQ3 audit differ. Unmeasured observer
cost does not explain the lower absolute rates by itself. Keep the faster
13.5052/8.8299 clean reference and obtain a matched clean-envelope comparison
before replacing it. The bounded source/settings audit is closed and reviewed.
`tools/sycl/iq3_audit_speed.py` defines the next same-b310 comparison:fixed
IQ3 selector1 and audit1/0 in three counterbalanced fresh-process pairs,using
the same two v5 workloads. V1 preparation stopped before creating a run root
or executing a GPU process: its legacy admission expected only one added
controller, while four other later tools were also in the current benchmark
snapshot. All original302 short-admission source hashes still matched.
Preserve that failure. The prospective V2 revalidates the closed303-source
cohort at its original revision, then pins the four declared later tools
separately without allowing any original source change. It also records
preparation failures before model execution. V2 preparation now verifies all307
caller/tool pins, all15 model assets and six immutable process plans. Its source
and environment controls pass without native execution. Both fresh frontend
probes pass and the V2 controller is ready. The report-only Luna audit and native
Luna-max review are closed. Main checked the additionally identified compiled
`native_expert.cpp` against the build record and actual b310 ELF. Its immutable
flag accepts exactly"1". The treatment includes GU-dispatch call accounting
across native formats and IQ3-specific row accounting; it does not isolate the
cost of IQ3 counter instructions. The source caveat is cleared without a
controller or native rebuild. All six processes (audit1,0,0,1,1,0) now close
with all12long requests eligible,native exit0,quiet validation,closed ownership,
no fresh fault and all15final model-asset hashes matching. Audit-on/off code TG
medians are13.486251/13.472547 tok/s;prose medians are8.802234/8.852377 tok/s.
The three audit-off/on paired TG ratios are0.989403/0.999270/0.988585 for code
and0.998553/1.005697/1.003632 for prose;their geometric means are0.992407459
and1.002622575. Neither workload meets the predeclared all-three-off-faster
screen. Thus this contrast does not support audit disabling as a reproducible
speed improvement or explain the earlier lower absolute speed. It does not
establish that counters are free or that enabling them causally accelerates
code. Preserve the earlier repeated clean reference and individual CSV peaks.
Audit-off timed IQ3 route telemetry is unavailable;an audit-on replay cannot
qualify it. This measures the enabled-audit effect on timed speed,not total
observer wall cost or the cause of the cross-build gap. Request snapshots and
footer lie outside decode_ms. Disabled compiled profile object/layout,guards
and request scaffolding remain a separate unmeasured confound. The enabled-audit
contrast is closed;do not extend cross-build diagnosis without a concrete new
optimization decision. Main continues the complete Q6 centering implementation
and serialized component validation;all-six closure now permits preparation.
The diagnostic preset's
`STRATA_DECODE_TIMING=0` is presence-enabled by the measured generator and emits
timing in both short arms; the new speed controller omits that variable.
These short diagnostics do not qualify sustained TG. The preload retains a
runtime reference until exit, so ordinary
runtime teardown and model speed remain unqualified. The closed observer and
CPU-only comparison outputs are now retired after original Git source/recipe
recovery and consumer checks:859 files,160869493 bytes. All240 samples and
the original CPU recipe remain at `cf0a6b175261f013099965dcb9ca19d1e840ad2d`;
the observer source is reconstructed by the pinned constructor. Protect479a.
No model gain or adoption follows from the component result or prepared build. The closed V2 generation was retired before
building the three V3 CPU-only targets; keep the validated GPU fallback unchanged.
The observer cohort is closed and its successful full output is
retired (408 files,84331126 bytes), with both final requests' clocks preserved.
The benchmark's 252 full blobs alone occupy 548352000 bytes, exceeding a
512 MiB process cap before library/scratch overhead. Its prospective measurement
cap is therefore 1 GiB; the small correctness fixture keeps its 512 MiB cap.
Prospective build limits are 300 s wall,1200 s CPU,8 GiB RSS,1 MiB text and384 MiB
private artifacts; the correctness child gets 60 s wall/CPU,512 MiB RSS and64 KiB
text. Normal exit0 plus supported ISA, all numerical/refusal cases and closed
owned processes are required before timing. A skip77 is not a correctness pass.
The reviewed retarget contract preserves the candidate's Q6/phase behavior and
requires checked observer-output completion: a truncated record or failed
write/flush cannot qualify attribution. The source review supplies no measured
CPU service or observer-overhead result.
The source-only output-v2 fix adds a versioned completion trailer, checks every
write and the final flush, and propagates failure at the generator call site.
The retargeted engine passes this short model entry, with representative
long-generation attribution and overhead still unqualified. The validated
fallback and recorded PP reference remain unchanged.
The reader now requires version 2 and an exact completion trailer, including
row counts and ordered thread slots; its host plan and result are in
`cache_map_cpu_service_sources_v1.reader_v2_host`. All 12 tests pass at source
`73642e6324b3403e196339dafb5aa5096538fa42`, including all 15 truncated prefixes
with an otherwise complete independent census. The owned host process exits
normally with no cleanup or survivors. This cannot qualify native write/flush
handling or observer overhead. The writer's host-only check passes at source `49c20a41c` using
`/usr/bin/g++ -std=c++17 -O2 -pthread -DSTRATA_CPU_SERVICE_PROFILE=1 -Iinclude tests/core/cpu_service_profile_test.cpp -o /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/cpu-service-output-v2/profile_test`,
then running that fixture without `--clock-screen`. GCC is Ubuntu 13.3.0-6ubuntu2~24.04.1,
executable SHA256 `1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769`;
development packages are libstdc++ 13.3.0-6ubuntu2~24.04.1,
libc 2.39-0ubuntu8.9 and linux-libc 6.8.0-147.147. Pin the three project inputs
(header, test and native_expert.hpp) to the committed source before execution.
Budgets: 60 s compile, 30 s fixture, 1 GiB address space and 64 KiB text.
The fixture passes buffered flush and unbuffered write failure through
`/dev/full`, preexisting/null stream refusals and the exact completion counts;
no model, engine-call-site execution or observer-cost claim follows.

The prospective next cache feasibility test uses the [official thinking-mode
sampling parameters](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): temperature
1, top-p .95 and top-k 20 (also in the [pinned generation configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/generation_config.json)).
Keep the existing greedy benchmark available and choose the new profile
explicitly. Retain the 16K total output allowance, uncapped thought/xhigh and a
prospectively fixed positive seed; do not combine a new output cap with sampling
or compare it with old greedy TG as a cache gain. The compiled-source sampler
and uncoupled-MTP audit is closed: greedy proposals with independently sampled
verifier draws are source-consistent. Its RNG transform is custom, not standard
Philox4x32-10, and non-finite logits silently select fallback tokens. Pin those
contracts and resolve invalid-logit handling before sampled qualification;
parity fixtures sharing the custom RNG are not independent RNG evidence.
The reviewed compiled path submits the same generic `launch<false>` sampler
for both runtime modes; FP64 exp/CDF operations remain in that body even with
greedy parameters, including graph capture. The installed B570 SYCL
`aspect::fp64` and actual FP64 dispatch remain unverified. Intel's
[B580 architecture entry](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html)
does not establish B570's installed capability. After this
frozen cohort, commit a bounded exact-device query and arithmetic/exp plan
before sampler or sampled-model execution. A later precision change must
name the selected production path and invalid-logit policy. SGLang's
[sampler entry](https://github.com/sgl-project/sglang/blob/27aedf7f93cc7ac3bff43e7150e42a3c0baa6957/python/sglang/srt/layers/sampler.py)
selects distinct default, deterministic, seeded and XPU behavior. Pin its
actual kernel, input dtype and non-finite settings for comparison.
The Main-authored [exact-device diagnostic](../tools/sycl/device_fp64_probe.cpp)
is source-prepared only, without a build or device execution. It separates a
no-kernel capability query from 64-element FP64 arithmetic and exponential
checks. It requires the exact Level Zero B570 at `0000:05:00.0`, fixed runtime
safety and an in-order queue; an absent `aspect::fp64` refuses arithmetic before
submission. Arithmetic must preserve a `2^-40` increment that FP32 loses;
exponentials over -63..0 need relative error at most `1e-12`, checked finite
outputs, unchanged source and allocation guards. These are bounded capability
and numerical checks, not sampler equivalence, native arithmetic throughput,
absence of software emulation or model qualification. The diagnostic deliberately
skips provider teardown after completed copies and freed allocations.
The Main-authored [owned controller](../tools/sycl/device_fp64_probe.py)
is source-prepared, without compilation or device execution. At source
`4851ec726c605b13a6aef69b54a55709eca75937`, fourteen host-only controls pass:
syntax, an actual premature-preparation refusal without creating its output
root, a closed-cohort positive fixture, and refusals for incomplete/active/open
cohorts, wrong process count, failed or incomplete processes, nonzero exits,
fresh faults and open ownership. No device is queried by these controls.
It refuses preparation until all six IQ3 audit comparison processes
close successfully. At that boundary, use `prepare`, commit its exact
source/compiler/header/provider identities and plan, then use `build`, `query`,
`arithmetic` and `exp`, committing each transition before the next. Use query,
arithmetic, then exp in separate processes, once each; stop on the first
unsupported capability, error, timeout, validation message or fresh fault.
The source follow-up adds a complete versioned compiler-tree and named host
toolchain-package image, including membership and symlink changes. It fixes
GCC selection to the recorded13installation and checks actual compiler header
dependencies and host-link trace paths against that prebuild image. Private
compiler temporaries are derived inputs; their disposable names are not retained.
The image has a180-second traversal/hash deadline,16 GiB unique-payload limit
and8 MiB manifest limit; it is not collected during the live model comparison.
The shared package image now includes the full `linux-libc-dev` package after
Q6's closed compiler check found20 actual headers missing from the inherited
SDK/host-package image. This source correction has not been executed for the
unprepared FP64 diagnostic and does not change its earlier outcomes.
An exclusive operation receipt now precedes cursor,Owner and directory setup,
so an early setup failure is terminal and preserves the no-retry decision.
At source `fdd2de5a1`, all five host regression tests pass via
`python3 -B -m unittest discover -s tests/core -p test_device_fp64_probe.py -v`.
They cover added/changed toolchain members, symlink retargeting, unpinned actual
compiler inputs and terminal receipts for cursor/Owner setup failures. They do
not build the diagnostic or query the device.
API trace substrings establish activation and expected symbols; query's lack
of a kernel launch follows from its pinned C++ branch, not parsed API-call counts.
Keep the existing compiler/runtime without emulation flags. Prospective caps:
180 seconds/4 GiB RSS for compilation, 60 seconds/1 GiB per device child,
1 MiB text per command and 16 MiB total in a fresh private output root.
CPU/file limits are per-process RLIMIT values. Summed RSS and present artifact
bytes are sampled session bounds with exit checks; these are not strict cgroup
quotas or cumulative-write limits.
Retain the small diagnostic executable only until its recorded capability
consumer ends; preserve source and compact outcomes, then retire normal traces.
Benchmark source `1f39425ddea4dd98e172d0033f78fb00a8586675` adds the explicit
profile. Its 73 host tests finish with 72 passes and one optional real-tokenizer
fixture skipped. They check wire/identity and legacy behavior, not GPU execution.
Bounded sampler/device and actual draft-route validation must precede a sampled
model trial; no such execution or sampling-causality claim is established yet.

## Qualification and operating constraints

Before adoption, retain the model prompt-verify, task/agent quality and numerical
checks, failure ownership and graph/allocation retirement, sustained decode,
and **physical 262144-token full fill**, final tails/boundaries,
prefill-to-decode handoff and RESTORE gates. Component address tests or a
smaller context do not satisfy these gates. An API/CLI admission refusal that
reserves output room does not exercise the final physical 262144-token slots;
record the engine entrypoint and actual processed extent for full fill.
Ordinary provider/runtime teardown is not qualified by the retained pre-exec
validation reference through process exit.

Preserve the qualified runtime safety settings and immutable pre-exec
SYCL/Level Zero environment in
[`measurement_environment.json`](../tools/sycl/measurement_environment.json).
No provider-teardown claim follows from this setting. Keep exactly one validated
fallback plus a named imminent candidate while that comparison is active; do
not infer global cleanup from the current-build inventory. Keep unresolved
legacy source/recipe recovery holds separate from these two comparison
executables, with the missing inputs, consumer, next bounded recovery action
and review point recorded in the retained-build result. The current97fda fallback
retains nine ELF artifacts;56cad9 and a347 have the named imminent prefill and
decode comparisons above. The obsolete9af/552b roots are physically deleted
after exact original source/recipe recovery and current-consumer checks. The
117K552b milestone remains in Git/CSV without retaining its unused executable.
Closed a1b0,519e,429b and superseded f633 outputs are also retired. Earlier
adaptive/DFlash2 records and the56legacy ELF provenance holds remain separate
reconciliation work;their old retention instructions do not renew a consumer.
Review actual remaining paths against those recorded holds before the next
generation or collector/model boundary. This is not global cleanup.
Other configured
deployment outputs and qualified validation/profiling tools have separate named
consumers. The standalone widening executable,IR and receipt are retired after
numerical/IL choice;successful host/device/build output and superseded source
reviews are also retired. This is not global cleanup:the configured deployments'
original source/recipe recovery and validated migration remain independent holds.

Niko1221/Strata remains the only upstream. The maintained
[`upstream-status.json`](../bench/results/2026-10-15-pr6-recovery-audit/upstream-status.json)
records the last actual check at **2026-11-11 19:42:07 UTC**,
`c1ff07f1a1817594811c960039571c6e268fe89a`(`v0.2.13.1-152-gc1ff07f1`).
The latest interval adds151commits/303changed paths beyond504b1433. The interval
is fetched, with only the plain GPU-list ordering fix and help from Niko
2b9df3fe imported so far;four regressions and14existing GPU-role tests pass.
The current
630-path functional import still targets
`146df3c468739147f520ac3816de806f71232571`(`v0.2.12`):shared Python/setup/
control/server/API/Web/CLI source is present with partial validation. Native
disk-session SAVE/RESTORE,BGEN/capability and desktop ports,the whole function
ledger,full host validation and hardware-tuned unmodified-upstream evaluation
remain incomplete. Bounded router/control/server source reviews cover selected
changes in the latest151commits;the complete functional/component ledger is
still pending. The earlier Arc-setup/readiness review covers504b1433only.
The next actual check
is due around **2026-11-13 19:42:07 UTC**.
**DeepSeek and35B execution support remain excluded until the
user explicitly requests them, including future syncs.**

Unsloth v3.1 catalog/pinned metadata and the stock DFlash2 Q4 pack are present.
The current optional feature-validation generation is compiled at
`591c2fbe3116c7e6172566646b5ded4a0526b761`(engine SHA`90f7e158…`).
Its original measured child commits are merged into the pushed primary;the
finished branch/worktree and superseded`5ecb4435`/f633 build are retired.
The original arithmetic/ring and short target-model evidence below belongs to
`f50df1f3c81272b14d994e4d8fcad19703ad0b7e`(former engine SHA`d0f42948…`).
The new model plan checks the shared native inputs and five unchanged component
fixtures against that original source;new head ownership is checked separately below.
Both serve and standalone now allocate at least7DFlash2 result slots,and the
span API refuses undersized outputs before submission. MTP sizes/math remain
unchanged. The rebuilt kernel/ring fixtures and retained mask fixture pass
16kernel/prompt,7small stock-drafter,5ring/lineage and12attention-mask cases.
The small cases include13undersized token/probability spans with untouched
outputs/ring state,all7active widths with guards/inactive zeros,and omitted
probabilities. Ring rewind/parked/fresh replay and the0/1poisoned-value mask
oracle remain valid component evidence;synthetic target pieces do not qualify
full target-context behavior or task quality.

The identical82ID,thinking-enabled32output target-model recheck now passes:
24of24drafts accepted,normal child exit0,reaped/session empty,no fresh GPU
fault. Actual INFO placement,pre-main validation,38file-backed executable ELF
mappings and all15model assets before/after execution are checked. The two
bracketed executable VMAs(`[vdso]`,`[vsyscall]`)are counted separately;this is
not full executable-address-space qualification. The original V1completed GEN
then aborted-6from writing7outputs into4slot callers;it remains failed in Git
and the V2previous-failure record. The new run resolves that specific host-heap
mechanism for these gates,without making the raw cold/observed rate eligible
for a speed comparison. Legacy preload-reuse's changed-zlib refusal also stays
unchanged;the new admission recovers the original probe source/recipe and
checks six later current-runtime pre-main proofs.

Native default MTP and legacy explicit-MTP configs remain selected. The host
Auto policy can choose DFlash2 on a fresh supported configuration;the current
candidate is not installed/default-adopted. Both fast phase references remain.
The new single-card spec4 path separates target-head/verifier retirement from
optional MTP duplicate-head ownership. It checks that no DFlash2 owner is live
before release,preserves the MTP transition order,and keeps short-input decode
placement available. DFlash2 width1,expert-release and cache-scratch conflicts
are rejected before pack/GPU initialization;cache scratch remains MTP-only.

The host fixture passes29cases/845checks at controller source`930bab9a7`,with
214compiler and28link inputs covered,no sanitizer output and normal closed
children. The new engine is compiled;this alone does not qualify native ownership.
The first small model gate remains failed at evidence`8868b857`:it inherited
8192-only tuning into a256-token chunk and the native mixed-scratch guard refused
the first513-token prompt. Its three pre-GPU rejection cases pass. The owned
child was terminated-15,reaped and its session emptied;qualified normal ownership
remains false. No fresh GPU fault occurred.

The new V2plan at controller`cec2e237b`explicitly disables those six shape-specific
options,retaining ordinary full512expert-bank/native16 behavior. It uses the same
f633engine,not a rebuilt or differently labelled executable. Three processes
exercise DFlash2 release0,DFlash2 release1 and MTP release1. Each receives
513/82/513thinking-enabled input tokens and32output tokens per request;positive
draft acceptance,actual long/short routing,head transition counts,restored decode
state,quiet validation and normal owner closure are required. All nine requests
pass,with positive accepted drafts and normal exit0 for all three model children.
The three pre-GPU refusals also pass. Actual route/head
transitions,pre-main validation,38file-backed ELF mappings per child and all15
model-asset hashes before/after are checked;all owners close and no fresh GPU
fault occurs. DFlash2 frees521472000target-head bytes per long transition;MTP
also frees223227900duplicate-head bytes. The phase/free-memory samples are
retained in the gate;these are not physical PCIe-byte or saturation measurements.
This is a small ownership diagnostic,not the tuned8192performance route or a
>=32Kcomparison. Closed stage/CLI logs,private compiler caches and the host
fixture outputs are retired after original-source/recipe recovery:188files,
43151075bytes. The original result records and phase engines remain;post-retirement
metadata admission passes without new GPU work. The path/hash manifest is in
commit`44e71e7fb`.

Measure total detach/restore cost and matched end-to-end benefit before claiming
a speed gain;the released bytes and phase/free-memory readings alone do not
establish one.
Then collect pinned bench.sh long/agent and thinking long-generation evidence.
Physical262144/RESTORE,task quality,ordinary-provider teardown and default adoption
remain open. See`dflash2_native_port`,`dflash2_model_check`,
`dflash2_prefill_heads_check`and`validation_environment_reuse`in the linked status
for exact identities,outcomes and remaining functional ports.
The host drafter adapter at`388eb6f18`passes20CPU/mock cases with no skips,normal
exit0and closed ownership(evidence`d69950af2`). SYCL Auto sees config-implied
or explicit`--vision`and uses available MTP/off;named DFlash2 fails with a
text-only reason before either encoder or model starts. Explicit MTP/off and
legacy configs retain their selection,and text Auto remains available.
An interpreter script without capability markers is unknown rather than proven
unsupported;the actual`engine_once.py`retains named DFlash2/file-vocab arguments.
An inspectable native without drafter flags still selects MTP. Explicit config
`family`identifies the supported model independently of its public display alias;
an unsupported family cannot borrow a supported display name.
The shared asset lookup has a local`dflash2`symlink to the already verified
stock pack beside the current model;no weights moved or duplicated.
The named drafter bench.sh route now passes diagnostics with 32,769 input tokens
and 24 generated tokens. Settings are 65,536 INT8 context/resident cells, 256
expert slots, 8192-token chunks, owned prompt buffers, a 384-slot packed prefill
bank and native16 BF16/GDN16. Thinking stays on; reused tokens are zero.
MTP at evidence `1d065ac07` has first/subsequent PP 591.5/713.6 tok/s,
prompt time 55.3988/45.9220 s and client TTFT 55.578/46.139 s. DFlash2 at
`e73050a42` has PP 584.7/709.2 tok/s, prompt time 56.0484/46.2078 s and
TTFT 56.193/46.395 s. Both native and benchmark owners exit zero, reap and empty
their sessions, with no fresh GPU faults. Actual 8192-token chunks, GPU residuals
(671,088,640 bytes), mode10 checked drain and head retirement are observed.
Startup free memory is 526 MiB for MTP and 1579 MiB for DFlash2. DFlash2 uses
2046 tap rows and cold_prefix=0 with resume=0; it loads/binds again for each
fresh long prompt. The native engine is unchanged f633/source `5ecb4435f`;
the corrected controller source is `6c4ec648a`.

These use parameter validation and stage tracing. They do not qualify clean
paired speed, sustained TG, task quality, rollover/split, physical 262144/RESTORE
or default adoption.
The highest retained phase references and performance CSV remain unchanged.
First-to-next prompt differences here are 9.4768 s (MTP) and 9.8406 s (DFlash2),
observed at one length; they do not identify a fixed cost or fit a new length law.
The current throughput-limiting hardware resource remains unqualified.
The corrected matched >=1K thinking-output screen is closed in
`drafter_generation_v2` at result commit `92c6e784b` and controller/source commit
`3b24ea716`. It uses the existing f633 engine and compares one fresh MTP process
then one DFlash2 process on the same 33,213-token input with thinking enabled.
Both produce an answer, finish normally with quiet validation, close all owners,
retain unchanged model assets, and record no fresh GPU fault. The complete
request body hashes match; this does not require identical generated text. MTP reports
585.2 PP / 56.921 s TTFT / 11.1 TG tok/s for 2,955 output tokens at 85.67%
acceptance; DFlash2 reports 579.9 PP / 57.427 s TTFT / 12.1 TG tok/s for 3,905
output tokens at 85.0% acceptance. The actual route checks identify MTP4 callback
coverage versus 2,046 DFlash2 tail rows with the MTP reader absent. This is an
eligible one-request-per-arm screen: stage and parameter-validation observers
remain charged, output lengths differ, and there is no repeatability, task
quality, clean-speed, whole-request-latency, or adoption result. The request
census shows zero expert-copy submissions and 1,192,027 / 1,567,836 CPU unique
misses with 3.8% / 3.7% printed cache hit rates; these are logical work counters,
not physical traffic, host service time, or proof of the limiting resource.

The checked-loan implementation is `dflash2_checked_cache_loan_candidate_v1`
from source commit `969e4891d`;its measured child is integrated and retired. Its narrow,
default-off policy uses the existing checked cache-loan lifecycle; provenance
records the original owned cache lifecycle and reviewed `aa5df5bd` snapshot, with
no XeStrata source. The host fixture passes 38 cases/854 checks under ASan/UBSan
at evidence `3c7ccce8f`, with 214 compiler and 28 linker inputs covered. Candidate
engine 90f and its five same-build fixtures compile from source/recipe `591c2fbe3`
(evidence `f2d679316`). The original precompile controller-key failure remains
failed at `ed10b6a69`; its fixed prospective attempt is separate. Closed host
outputs are retired after exact source/recipe recovery (`adfd78b57`). Actual
compiler/linker census covers 1461 frozen inputs, one owned generated header and
29 external linker files with zero uncovered inputs. The V3 CLI matrix passes
12 normal exit2 cases with same-PID before-main validation and no device/model
calls: eight policy refusals and four CPU-only missing-pack inspections. V1's
metadata budget stop and V2's incorrect expectation about SYCL's default
no-token-graph flag remain failed. All five fixture ELFs match f633 bytes;this
alone does not reuse native qualification. The same-build DFlash2 GEMM/tap
fixture now passes all16cases at controller source `54c8b4f99` and plan
`d4b373923`,with38actual executable mappings verified,normal exit0,reaped/empty
ownership and no fresh GPU fault. Pre-main parameter validation and full API
tracing remain enabled. This does not execute the changed main cache-loan policy
or qualify all five fixtures. The remaining same-build cohort now passes53cases
at controller`c6f915a48`/plan`8c4afccfa`:19pack,10submission,7small-ring,
5wrap/rewind/parked-restore and12mask controls,all normal0/reaped/empty and no
fresh fault. V1remains failed because Main's controller did not consume the
submission fixture's final two protocol lines;V2corrects that reader. All five
fixture binaries have now executed. The bounded owned-versus-checked-borrowed
model gate now passes: controller`66fac0b8c`,plan`1e91cedb4`,candidate90f,
4096context,1256expert slots and256chunk. Two fresh processes complete all8
short/long/short/live-prefix requests with32/32/1/32thinking outputs. The third
one-output request creates the exact82-token live prefix;prompt-cache1/every0
avoids checkpoint copies. Both actual routes and2target-head retirements per
process are verified. Each borrowed long request lends87slots,then verifies all
1256resident expert payloads(2,575,206,400bytes)against RAM and publishes the
original slots ready before decode. Both owners exit0,reap and empty their
sessions;38mapped ELFs and before-main validation match;no fresh fault occurs.
This is a charged payload/ownership diagnostic,not speed or task quality.
The subsequent long admission also passes at controller`7b4425b22` and
plan`c730aeb9d`:two fresh90f processes,65536INT8context,8192chunk,384packed bank
and two exact32769-token thinking requests each. Owned256slots starts with
1579MiBfree;checked-borrowed1536starts with1300MiBfree. Both keep all671088640
residual bytes on GPU and the384bank checked drain. Each borrowed long prompt
loans/refills1092slots;both native and benchmark owners exit0,reap and empty,
with no fresh fault. First/subsequent PP is582.8/703.8tok/s owned versus
575.2/714.4borrowed;clientTTFT56.401/46.743s versus57.146/46.034s. Per-run
request bodies match. Direction changes between first/subsequent and there is
only one process per placement,so speed remains undecided. Output24does not
qualify sustained TG. Descriptive expert-bank counters report the same
requestedH2Dbytes in both arms;they do not measure wire traffic or saturation.
The matched sustained-output cohort is closed at controller`31a16936a` and
original plan`f0d81312a`:six fresh ABBAAB processes,three per placement,two
thinking workloads each. All native/benchmark exits are0,owners reap and empty,
validation is quiet,no fresh fault occurs,and all15final model-asset hashes
match. Final process evidence is`e040c3929`. Request bodies match within each
workload,with33213code and33283prose read tokens,reuse0. The table gives
three-process medians,owned256/borrowed1536. Rates use actual counts and engine
intervals;individual printed rates remain intact in the compact result.

| Request scope | Generated tokens | PP(tok/s) | ClientTTFT(s) | TG(tok/s) | Draft acceptance |
| --- | --- | --- | --- | --- | --- |
| Code:first long after four warmups | 3905/3465 | 579.260/571.522 | 57.528/58.295 | 12.180/11.992 | 85.000%/77.756% |
| Prose:subsequent long | 9870/8430 | 667.973/679.002 | 50.008/49.166 | 7.032/7.295 | 32.486%/31.849% |

Pairs0/1,3/2,4/5give borrowed/owned code TG ratios
0.972313/1.004487/0.982865(geometric mean0.986465). A repeatable code gain
does not pass;PP and clientTTFT are slower in all three pairs. Prose TG ratios
are1.039395/1.040639/1.020333(geometric mean1.033414);PP and clientTTFT also
improve in all three pairs. This is a repeated observed-route gain for the
second-long prose case,not a reason to adopt the placement for code. Both
placements satisfy the predeclared strong-code/weak-prose draft classes in all
three processes. Workload and request order are confounded;this plan cannot
attribute the prefill difference to content alone.

The final borrowed footer reconciles GPU hits+CPU entries with
target_rows*48layers*10experts. CPU jobs per emitted token fall3.686%for code
and7.497%for prose;these are logical work counts,not physical traffic or
hardware saturation,and generated paths differ. The borrowed planner records
the same8192-row layout at startup and again after smaller warmup layouts;
its reinitialization cost is unmeasured. Keep that first-long/subsequent
distinction. Each borrowed long prompt loans/refills1092slots.

Select90f as the retained feature-validation generation,with the loan still
default-off;this does not adopt its whole inference route. The measured child
is integrated with its original commits. Consumer migration,original-input
recovery and physical retirement of f633,the finished child checkout/branch
and closed private cohort metadata are complete. Admission still passes.
Preserve552b/e1b as independent phase guards. Validation/stage observers remain
charged;clean speed,agent/task,length scaling,physical262144,tails and RESTORE
remain pending. The physical limiting resource remains unknown.
The production-adapter fixture now builds at`f4f83be4f`(ELF`5046a535…`),with
711compiler and24external link inputs checked. Direct and active construction
pass. Paused construction remains failed at`4d1528859`:native computation exits0,
but the collector omits all eight kernels. The separate active-from-exec mode at
controller`6aa32694f`passes three CPU controls and its same-ELF device check,
retaining all eight kernels/five copies with quiet validation,closed ownership
and no fresh fault. This enables a bounded model profile,not a hardware
saturation or observer-overhead claim. The subsequent exact32K model screen
is closed under`dflash2_device_service_v2`:all six requests,native/benchmark
exits,owner closures,quiet validation and15final asset hashes pass. Its whole
observer screen fails:decode ratios1.516427/1.515178 exceed the declared1.5,
although prompt/clientTTFT ratios pass. Preserve this failed admission;the
observed Q6/mapped-copy rankings select bounded component questions without
qualifying normal-run attribution or physical saturation. The inference engine
is unchanged. See`docs/RESOURCE_PROFILING.md`for samples and limits.
Standard long/agent comparisons and independent prefill/decode selection remain
pending. About96% of expert lookup entries fall to the CPU in this256-slot
diagnostic layout; the requested expert
H2D-copy counters are zero. This motivates the capacity comparison,without
establishing CPU/RAM saturation or measured PCIe wire traffic.

The first startup-identity failure remains failed at`c70ef71ae`. Its superseded
driver is retired from the active tree and recoverable at`2a7b98908`;
`tools/sycl/compare_drafters.py` contains the corrected stable-field check.
The source-audit assignment returned the wrong task response without a report
and is failed,not a qualification. Both affected research terminals are closed;
the available cache-loan snapshot and its completed native review remain
usable decision support. See the research index for the exact lifecycle limits
and the required isolated/read-only Task preflight before the next round.

The count-only OpenAI route, removed by the bulk server import, is restored at
`5db3ede7c`. At frontend/benchmark source `c9f4ce45d`, 69 CPU/mock tests pass with
zero skips (normal zero exit and closed ownership, evidence `aaa109d17`). The
isolated real-tokenizer test also passes 73729/73730/81921/81922-token extents
before native launch after its import-order fix at `cecb75ab6`; inference is
not called. Optional Anthropic SDK
client tests remain unqualified because that interpreter lacks the SDK.
Original failed V1/V2/V3/V4 cohorts remain failed in the compact results and
linked Git evidence: alias fallback, missing count API/VRAM chunk fallback,
isolated fixture imports, and Main's incorrect cold_prefix=1 assertion. Fresh
cohorts resolve those admissions without relabelling their outcomes.
Disk sessions,BGEN/capabilities,native v3.1 model
behavior and the full bulk-import suite remain open.
The following earlier host-component results refer to their original pinned
inputs;they do not qualify the later bulk Python imports.
After both frozen cohorts closed,the required streamed-tool closing-tag
behavior from Niko`3bbf0f399e2a1b3d38642eca4342721fc7994201`was adapted to
the older fork parser at`f7784635b`,preserving early tool/argument streaming
and its scan optimization. An ambiguous final close is bounded by EOF,a later
call or1500lookahead characters. At tested source`44ad93dfe`,six new literal
tests,11existing boundary/terminator tests and the full147server CPU/mock
tests pass with no skips and normal closed owners. Follow
`upstream_streamed_tool_tags_host_v2`. The original V1 remains failed:its
2GiB virtual-space cap prevented new threads;the separately committed V2
uses8GiB AS with the same2GiB RSS and other criteria. An owned32-thread
snapshot shows2,512,637,952virtual bytes but65,536,000resident bytes. This
corrects the test envelope;it does not fix the telemetry daemon lifetime or
qualify GPU inference. Fresh frontend admission is still required before
new model comparisons. The required stalled-SSE fix from Niko`c08fd2ec`is
ported in the shared`make_handler._sse`:socket writes time out after60s
by default,configured by`STRATA_STREAM_WRITE_TIMEOUT_S`with0preserving
unlimited writes. It uses the existing disconnect cancellation/turn-release
path across all three APIs. The Main-authored`serve/test_stream_write_timeout.py`
adds four CPU/mock real-socket cases for unread clients,next requests and
limit0. At source`1c2956b22`,these four tests,the147server and20Responses
regression tests all pass with zero skips,normal0and closed owners. Follow
`upstream_sse_write_timeout_host_v1`;the existing shared host owner pinned352
source inputs and the qualified interpreter manifest. This is transport/CPU
validation,not GPU or inference qualification. The other four reviewed server concerns,
remaining functional ports,whole function ledger and vanilla performance remain
incomplete.
The Niko`b776e1cd`console-backpressure function is now connected at source
`f3115189c`:the existing no-argument`main()`uses`decoupled_console`,with
`STRATA_BLOCKING_CONSOLE=1`preserving the upstream tester bypass. The module
origin is MIT and matches this fork's LICENSE. The Main POSIX owned-descriptor
repair avoids holding Python's original BufferedWriter lock at interpreter
shutdown;the original buffered-child failure remains failed under
`upstream_nonblocking_console_module_host_v1`,and the nine-case module repair
passes underV2. Source/test identities and original outcomes remain in Git.

The new`upstream_nonblocking_console_entry_host_v1`pins396actual source/template/
controller inputs and the qualified Python runtime. All13console-entry/module,
147server mock and20Responses cases pass,zero skips,normal0and closed owners.
Unread buffered/unbuffered child entry exits normally;the explicit blocking
bypass still blocks its control,and both CLI-help modes pass. The source
inventory is recoverable from committed plan9256be028;compact results keep
useful source/count/closure evidence. This qualifies wiring and CPU/mock
compatibility,not actual model startup,GPU inference or performance. Obtain
fresh bounded actual frontend/model startup before the next model comparison.
Upstream queue`max_bytes`counts characters,not encoded bytes;no encoded-byte
memory bound is claimed. Other required functional imports remain incomplete.
The configuration parsing helpers are ported and
pass five focused tests. Actual server startup wiring at
`92f7b07227bae2117087cdf270b9d03c41d5ebd8` passes eight helper/integration
tests and is now merged into feature/sycl at `4c24b6931`;its completed child
is retired and original commits remain reachable from the pushed primary.
Fresh bounded model/frontend checks now pass for both fixed-width routes;
their sustained comparison and broader model qualification remain pending.
Required tracking/recovery/alias/API/CLI imports and hardware-tuned unmodified
upstream measurements remain incomplete. Source and evidence are published on
`feature/sycl`. Default integration is pending through PR180 under main's
squash-only rule24638744. All92CSV source/evidence commits are verified ancestors
of the pushed permanent primary and automatic head deletion is disabled. The
project workflow permits a default content snapshot while preserving all those
original identities on `feature/sycl`. The ordinary squash attempt at `514111ac9`
was also rejected: the active ruleset restricts updates, with
`update_allows_fetch_and_merge=false`. Admin merge permission for this concrete
PR is pending; no protection bypass or rule change was performed. After an
authorized merge, merge the updated default back normally and verify the original
identities again. The last
verified default before that operation is`05069aeedea43f778b97dce30c729739cb3bce5c`.

## Expert upload chain decision

The closed `expert_upload_chain_v2` result tests the existing four-slot Host-USM
staging through upload, GU/Down dequant, native16 GU, SwiGLU and Down. Six fresh
processes use ABBAAB, with two 256-job batches per shape/process; all 48 batches
exceed 100 ms. Parameter validation is charged in both arms. These are synthetic
N2560/F640 three-slot dependency probes, not model throughput or the384-slot scheduler.

| Expert rows | GU / Down type | Paired geometric-mean staged/direct wall time |
| --- | --- | --- |
| 640 | IQ2_S / IQ4_NL | 0.943; one of three pairs slower |
| 640 | IQ4_XS / IQ4_NL | 1.100; all three pairs slower |
| 1280 | IQ2_S / IQ4_NL | 0.758; all three pairs faster |
| 1280 | IQ4_XS / IQ4_NL | 0.890; all three pairs faster |

The all-shape gain criterion fails. Keep global staging disabled; do not advance
that flag to a model trial. The1280-row results justify investigating an
actual-job-size policy, not assuming the same gain at other rows or fused jobs.
Map the production job geometry before that new experiment. The independent
cached-QSA candidate remains eligible for its own32K/64K model comparison.

V1 failed because its CPU oracle BF16-rounded FP16 inputs. The actual path uses
FP16 inputs, FP32 accumulation and BF16 output. The corrected source
`3efcb16e2` passes both formats at32/640/1280 rows, all17direct/staged output pairs
per case, independent dequant/dot/activation checks and guards; no fresh GPU fault.
Original V1 and older byte-only V5 failures remain unchanged. The compact result
keeps all individual timings; full source/recipe is recoverable from its pinned
Git evidence. Closed fixture binaries, raw traces and research transaction/source
copies are retired. No model PP/TG/TTFT, saturation or default adoption follows.

## Evidence pointers

The cumulative user-requested index is
[`bench/performance-records.csv`](../bench/performance-records.csv) is the
cumulative index of superior results and useful comparison milestones. Each
row identifies its original evidence/native commits,metrics and qualification;
retrieve detailed samples from that pinned evidence. Keep first-use,warmed,
independent-process and observer scopes distinct. The device-only row-copy
component point leaves model PP/TTFT/TG blank and does not qualify host-USM use.
Detailed current evidence and frozen plans are in
[`fused-moe-service-current.json`](../bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json)
under `cache_map_prefill_speed117489_v1`, `cache_map_prefill_curve_v7`,
`cache_map_decode_census_v1`, `cache_map_worker_scaling_v1`,
`verify_window_generation_speed_v1`, `dflash2_device_service_v2`,
`mapped_row_copy_component_v1`, and `retained_builds_review_20261103`.
Earlier closed phases are retrievable from the baseline Git revision above;
consult them only for a concrete comparison or provenance recovery. Follow
[`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md) for retained builds and cleanup.
