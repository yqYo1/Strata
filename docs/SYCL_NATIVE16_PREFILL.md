# SYCL prefill and decode decisions

This page records current selections, the evidence needed for the next decision,
and open qualification gates. Closed experiments and detailed samples remain
in Git history and the linked cumulative result; this is not a run log. The
closed chronology is recoverable from the [original maintained decision at
commit `3abcceba08b25a632f4e26bde7551039120ac42a`](https://github.com/yqYo1/Strata/blob/3abcceba08b25a632f4e26bde7551039120ac42a/docs/SYCL_NATIVE16_PREFILL.md).

## Current selections

- Keep staging all selected expert spans before native append. The 65537-token
  staging-order comparison at `fe6d81002` keeps other inputs fixed. Subsequent
  medians are 982.1 tok/s for all-at-once and 980.0 tok/s for per-span.
  Per-span won one of the three warmed pairs. Keep early native copy issue and
  first-reader completion as separate, interacting choices. Earlier issue
  alone increased warmed prompt time by 0.51%. With deferred completion fixed,
  earlier issue reduced time by 9.42%. The combination reduced time by 9.76%.
  See `prefill_prefix_expert_schedule_timing_v1` and
  `prefill_prefix_expert_host_stage_timing_v1` for the fixed reference arms.
- The 32769-token same-binary observer bracket (`bcacbe020`,
  `prefill_prefix_expert_stage_profile_v2`) closes three processes and six
  thinking-enabled requests with matched routes/work, normal exits and no
  fresh faults. Only the full host API trace was removed from the earlier
  instrumentation; warm prompt ratios are 1.229/1.218
  and client TTFT ratios 1.230/1.222
  against the two clean controls, above the 1.20x gate. Clean warm PP is
  860.9/852.8 tok/s and TTFT 38.353/38.603 s.
  Keep the profiled request as diagnostic evidence. Its warmed H2D/kernel interval
  overlap is 5.25 s, CPU time in known waits
  is 21.10 s, and CPU row grouping is
  0.089 s. The original
  V1 cleanup failure remains at `b9117a05c`; physical ownership and session
  cleanup were verified at `4053aaa12`. Physical saturation and the per-layer
  critical path remain unmeasured.
- The CPU-wait-only bracket at `002f1105d`, `prefill_prefix_expert_cpu_wait_v2`,
  closes three processes and six 32769-input thinking-on requests. The ELF,
  routing, payload and draft counts match. Prompt and TTFT observer ratios
  are 0.992–1.010 against both clean controls, within the 1.20 limit.
  Clean warm PP is 879.4/879.6 tok/s and
  client TTFT is 37.447/37.512 s. All processes exit
  normally with unchanged driver and no fresh fault. V1 preparation refused
  a changed decision-document pin at `2566f8ac5`. V2 verified all historical
  build controls from Git, the committed document delta and the exact live
  compiler, dependency, fixture and executable files.
  The warm calling thread accrued 34.55 CPU s.
  Compatibility-queue waits account for 18.78 CPU s
  across 1069 calls. Row grouping took
  0.090 CPU s.
  Inside 48 MoE status scopes, the four-byte D2H enqueue wrapper took
  3.702 s wall and the subsequent explicit drain took
  4.68 ms. These are nested calling-thread times.
  The wrapper may wait for earlier GPU work. The CPU profile does not measure
  the wire service or show that reducing CPU cycles will reduce prompt latency.
- Status readback placement closes at `d99bacc3f`
  (`prefill_status_destination_v5`). The component invokes the exact
  compatibility memcpy and stream drain from the retained runtime archive.
  Both ordinary and host-USM destinations pass status values, bounds and
  checked cancellation. Six counterbalanced timing processes close 60
  eligible batches and two smoke processes close normally with no new fault.
  Same-context host USM stays alive through retirement.
  [SYCL USM allocation rules](https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html).
  With a drained producer, the three paired full-time ratios are
  0.778/0.888/0.855 (geometric mean 0.839).
  With queued synthetic producers, they are 0.652/2.086/1.268
  (geometric mean 1.199). The comparison charges one USM allocation per
  48 queued copies and includes both enqueue and explicit drain. Host USM
  shifts substantial time to the explicit drain. Queued results miss the
  repeated-gain gate. Keep ordinary placement on the model route.
  The V4 clock-floor failure is retained, including the original 4.06–9.97 ms
  queued batches. V5 increases queued repetitions from 48 to 384 while keeping
  the 10 ms floor. V1–V3 are separate pre-GPU dependency/launch refusals.
- A same-ELF queue-audit experiment at `f37718d09` refused coverage.
  Hardware and software perf collection were unavailable to this account.
  The CPU-only loader canary passed both caller addresses and a propagated C++
  exception. The actual native ELF verifies wait return PC 0x72762b and
  caller slot +272 from the direct PLT call and its unwind row, and stderr was
  the owned regular file. [GNU libc audit callbacks](https://snapshots.sourceware.org/glibc/trunk/latest/manual/html_node/Audit-API-Callbacks.html).
  The unloaded 32K control reached 881.2 tok/s warm with 1,069 compatibility
  waits and 18.71 CPU s inside them. The first audited request collected
  1,034 of 1,069 waits, below the predeclared 98% count floor. All 1,034 resolve
  to direct callers and carry 18.555 of 18.586 scoped wall seconds. The gate
  stopped the subsequent requests. Physical termination of all four recorded
  owners and absence of a fresh GPU fault were verified at `ac2b46bbc`.
  Keep the original failure and first-use-only diagnostic.
- The dominant observed caller at 0x4b400a is the attention store/drain in
  `Prefill::run_impl`. Its 192 waits total 17.182 wall seconds and 17.135 CPU
  seconds in that diagnostic. The return and error edge point to
  `prefill MoE window: attention store/drain`. This drain also covers already
  queued attention and input retirement. The 17.182 seconds are service at
  the synchronization boundary. The warm profile and the amount of removable
  wall time remain unmeasured.
  Next isolate the BF16 attention residual store. Today the in-place write
  updates the old chunk residual in scratch and a separate D2D copy saves
  it into the window bank. Compare it with an out-of-place attention write
  that reads the same original residual from scratch and writes the same
  rounded BF16 value directly to the validated bank span. The current
  in-place write reads its destination, so simply retargeting that pointer
  could read the wrong value. Preserve the existing stream drain, host
  token/step/PLE retirement, prefix-router order, callbacks, cancellation,
  partial-tail handling and decode. For the observed four chunks × 48 layers,
  the separate copy requests about 30 GiB of device-to-device payload.
  The device component at `fe515ce995169764eb19316abf987b4426ffa9b4` checks
  exact BF16 bits for 1/63/8191/8192 rows, including broadcast initialization,
  guard regions, invalid spans and checked retirement. Each timed interval
  includes 16 stores and their drains. The three copy/direct process
  medians in ms are 52.197/36.225, 52.059/38.887 and 52.196/36.618
  (geometric mean time ratio 0.714). The copy arm requests 2.5 GiB per
  interval, and the direct arm requests zero. All seven processes retired
  without a fresh GPU fault. These intervals start with ready inputs.
  The coherent build adds only the four attention-store overlays to the
  verified stage-all image. The 32769-input numerical comparison
  (`7b134c044`) uses the same ELF in both arms. Four requests retain
  identical routing digests, selected experts and draft counts. Direct
  writes replace all 192 copy sites and remove 30 GiB of requested D2D
  bytes per request.
  At 65537 fresh input tokens, the clean same-ELF comparison
  (`b973b13dd1408aeae56ef17e7ff5536d57bf6654`) retains six processes and
  18 samples with thinking on. Direct writes remove 60 GiB of requested
  D2D bytes per request. All three warmed pairs improve, with prompt-time
  ratios 0.99301/0.99268/0.99358. The median of warmed process medians is
  977.8→984.1 tok/s and client TTFT is 67.226→66.788 s. First-use TTFT
  improves in two pairs and regresses in one. All processes exit normally
  without a fresh GPU fault. The 982.1 tok/s old stage-all result remains
  historical context.
  The same-build standalone capacity comparison (`e5408e1c2`) passes both
  arms at exactly 262144 physical input tokens. Both process 262143 batched
  rows, then commit the last token in one checked verifier window, with zero
  output and a checked refill drain. The physical route uses fallback mode 2
  with 16384-row windows, 114688 GPU-resident BF16 rows, and 147455 host-backed
  rows. Direct writes cover all 672 eligible attention pieces and remove
  105 GiB of requested D2D copies. Both 16385-input controls emit the same
  four greedy fixture IDs. All four native cases close normally with quiet
  validation, no fresh GPU fault, and all 15 final asset hashes matched.
  The original setup failure at `eb11bc3db` precedes any child launch.
  The same-ELF native server RESTORE comparison (`f2a044c7f`) passes all
  four independent processes and twelve requests. After switching to an
  unrelated 33-token conversation, both arms restore 32769 or 253919 saved
  tokens and read 8209 fresh tokens. The largest completed prompt is 262128.
  The snapshot byte verifier passes main KV for 12 layers, draft KV, and
  running state. Each paired request matches active fingerprints, prompt
  publications and its single greedy fixture token. The unrelated request
  uses the decode-cache window without lending expert slots. Both longer
  requests use layer-major prefill and recover the checked 1256-slot decode
  cache before generation. All native owners close normally without a
  fresh fault. The original host peer refusal (`1a2115365`) and the
  obsolete reader assumption about the unrelated request (`3f3687175`)
  remain recorded.
  Keep the direct path opt-in for the agent-session and repeated clean
  thinking-enabled long-generation comparisons. Exercise the observed
  high/low MTP-acceptance cases, keep decode settings fixed in both arms,
  and separate clean speed from the native RESTORE diagnostics.
  The first copy process completes all 18 requests and exits normally.
  Its v4 result reader rejects the official v5 suite at `a246cf27f`.
  A post-closure check reconciles the v5 request journal and native counters,
  including six agent turns and both sustained requests. Code and prose
  read 33211/33281 tokens, generate 2815/6823 tokens including thinking, and
  reach 12.35/7.21 tok/s at 87.0%/33.9% draft acceptance. These are single
  copy-arm diagnostics. The original failed gate stays recorded.
  The v5 comparison stops during its first copy-arm process at the 16098-token
  prompt, before it reaches agent or long generation. Layers 0 through 3
  published checked expert-reader receipts. The watchdog reported 480 seconds
  without progress at layer 4, offset 8192, and the native process terminated
  with SIGABRT. Five CPU expert workers were parked. The first kernel fault
  record follows the final engine log write by 4.83 seconds and is within
  27 ms of the wrapper recording native exit. Xe recorded a CCS reset.
  This timing leaves the initial stall cause unresolved. The identical
  v1 request completed. Keep the 512-slot early native-copy path out of
  subsequent runs pending isolation. After the automatic reset, the ordinary
  SYCL device fixture checked copied data in two GPU consumers without a
  reboot. The ordinary-queue model check (`prefill_ordinary_bank_ragged_v1`)
  also passed six requests across the historical engine and the current
  copy/direct stores. Each processed 8192 + 7905 batched rows, the final
  prompt token, and 24 generated tokens. These synthetic inputs reproduce
  the stalled shape with different prompt bodies. In each current request,
  the 96 direct writes removed 15,823,994,880 requested D2D bytes versus
  copying. All three owners exited normally without a fresh Xe fault.
  The 384-slot setting sizes the common allocation for the largest expert
  stride. All three arms used the same 1,022,361,600-byte bank.
  At this shape, 24 layers fit 512 expert slots, 14 fit 469, nine fit 438
  and one fits 384. This check admits the separate clean 64K comparison.
  The same-period 384-slot comparison at `ecac777fc` closed nine
  independent processes and 27 exact 65537-token requests. Each process
  used the same three fresh bodies, with 24 output tokens and thinking
  enabled. All native exits were zero and no fresh GPU fault appeared.
  Warmed rates use the second and third prompt times in each process.

  | 384-slot route | Warmed process rates (tok/s) | Median PP | First-request TTFT (s) | Warmed TTFT (s) |
  | --- | --- | ---: | ---: | ---: |
  | Historical ELF | 883.0/883.6/862.4 | 883.0 | 86.948 | 74.467 |
  | Current ELF, copy output | 902.0/899.3/958.7 | 902.0 | 84.791 | 72.871 |
  | Current ELF, direct output | 902.0/876.5/896.6 | 896.6 | 87.738 | 73.289 |

  Paired warmed direct/copy time ratios were 1.00004/1.02603/1.06925. The direct path
  avoided 60 GiB of requested GPU-local copies per request. Its
  repeated-gain condition failed on ordinary384. Retain copy on this
  route. The earlier 512-slot warmed gain remains scoped to that route;
  the faulted early native-copy configuration stays out of subsequent runs.
  The historical ELF previously reached 1006.8 tok/s. Its three new
  processes used identical request hashes, expert work, requested H2D
  volume and MTP counts. Replacing the old measurement root in the five
  path-valued settings of the original environment template reproduces the
  new template hash. The period difference remains unattributed.
  The ordinary official suite closed six independent processes at
  `f6bb38a61` with exit zero, closed owners and no fresh GPU fault. It checked
  the identical 16098-token request, six-turn agent sessions, and thinking-on
  sustained generation. Code read 33211 tokens and generated 2815 at 87.01%
  draft acceptance. Prose read 33281 and generated 6823 at 33.93%.
  Direct/copy paired TG geometric means were 1.00786 for code, 1.01896
  for prose and 1.02267 for agent follow-ups. The first code pair was 0.99328.
  All three comparisons passed the predeclared decode-preservation gate with
  matched work counts. The separate ordinary384 long-prefill speed gate
  above still selects copy. The CSV retains each configuration and workload.
  In the copy arm the median engine decode time per output token was 84.75 ms
  for code (2036 of 2340 offered drafts accepted) and 142.33 ms for prose
  (3442 of 10143). In this fixed-width configuration the first verifier
  window takes one token and each later window offers three drafts. The
  counters imply 781 code windows and 3382 prose windows, producing 3.604
  and 2.017 output tokens per window. Final EOS handling accounts for two
  and one extra verified positions respectively. Total engine decode time
  divided by windows was about 305.5 ms for code and 287.1 ms for prose.
  The 70 tok/s target at these observed output yields allows 51.5 and
  28.8 ms per window. Those totals include the work around verification.
  Measure verify, commit and draft on these same workloads before assigning
  the gap. The older 3157-output/900-window executable cannot supply that split.
  Use `bench/sycl/ordinary_decode_round_cost.py` on the resident-input build
  with both BF16 copy paths selected. Run the same official thinking-on suite
  in clean, timing, timing, clean order. The timing arm enables only the
  existing request-end decode footer. Bind its window count, verifier rows,
  CPU expert entries and phase clocks to the native request and client result.
  Compare the two pairs for each workload. Treat the clocks as representative
  only with equal work and paired per-token times within a factor of 1.10.
  GPU-reach wait and CPU-pool wall time are nested inside verification.
  These counters cannot by themselves identify VRAM, RAM or CPU saturation.
  The first instrumented process at `866182a40` closed with the same
  generated-token, draft and CPU expert counts as the preceding clean process. Code/prose
  verification took 287.77/262.44 ms per window, including GPU-reach
  waits of 128.33/125.17 ms. The host interval following GPU reach was
  141.45/117.95 ms. Inside it, 131.73/108.66 ms elapsed around the expert
  dispatcher's pool run. In source `1bf54ec32`, the serve footer prints
  `drive.d.ms_run` for the latter field. The dispatcher times from just
  before pool invocation through its return and any peer/remote completion.
  These wall clocks do not measure worker CPU cycles or RAM bandwidth.
  Draft generation alone took 28.65/28.51 ms. At the observed acceptance,
  70 tok/s leaves only 22.84 ms (code) or 0.31 ms (prose) for all other
  work if draft time stays fixed. For this MTP-weak workload measure the
  cost of proposing drafts and the emitted tokens per verification together;
  reducing CPU expert time alone cannot recover that budget at fixed yield.
  Both matched pairs closed at `bf5f94144`. The two clean code results
  were 11.312/11.368 tok/s for 2815 output tokens and prose 6.844/6.604
  for 6823. Clocked/clean per-token ratios were 0.994/1.003 for code
  and 0.988/0.956 for prose. The second clocked process gave 289.07/263.32
  ms for verification, 133.35/109.80 ms within the expert pool and
  28.73/28.54 ms for drafting. Both workloads met the 1.10 observer gate
  with matched work in both pairs. All four native processes exited zero
  with no new fault. These observations qualify attribution on this route.
  For each measured process and workload, let `t` be total engine decode ms
  per window and `y` the actual emitted tokens divided by windows. Holding
  work and every other interval fixed, halving the expert-pool interval `p`
  gives `1000*y/(t-p/2)`, or 14.34–14.37 tok/s for code and 8.51–8.52 for
  prose. Removing that interval entirely gives about 19.5 and 11.1 tok/s.
  The per-layer GPU-reach wait ends before the pool callback starts in
  `src/core/verify.cpp` at the measured source `1bf54ec32`. Even setting both
  wall intervals to zero in the same calculation gives 63.73–63.98 tok/s
  for code and 35.21–35.28 for prose. These are fixed-work sensitivities.
  Halving both intervals yields 19.26–19.32 and 11.55–11.58 tok/s on
  the same samples. Halving draft generation as well yields 20.86–20.92
  and 12.58–12.61 tok/s. These calculations keep the observed yields of
  3.604 and 2.017 tokens per window. Planning, input quantization and job
  preparation together account for 8.17–8.42 ms per window, at most 2.82%
  of its measured time. Use the queued segment clocks to distinguish
  execution of the verifier's layer graph from the delay in reaching it.
  Keep reductions in routed work and changes in accepted tokens per window
  in the same decode comparison.
  Moving experts changes how much other work the wait hides and may change
  the draft yield. Measure both after a ranking change. Reaching 70 at the
  observed yields also requires reducing the remaining verification/draft
  time or producing more accepted tokens per window.
  In the three completed CPU-service observer processes at `c74d1539c`,
  the single-token IQ4_NL Down bins are 20.61–20.64% of summed leaf CPU
  time for code and 15.12–15.15% for prose. These concurrent worker clocks
  motivate a bounded component test on the measured Zen 3 host. The
  denominator is CPU time within the instrumented pool's leaves.
  The decode-only paired-row candidate at `5c1052eb8` reused Q8_0 input loads
  for two adjacent IQ4_NL Down output rows. The CPU build verified 618 inputs,
  and all 60 bitwise-output and guard cases passed. The six processes at
  `34506f209` closed with 30 eligible pairs. Median candidate/reference times
  were 0.9874/0.9817/0.9980 for one-expert reuse and 0.9869/1.0191/0.9726
  for rotation through 128 experts. The rotation arm missed the three-gain
  gate. The original plan and linked build evidence remain at `e4a9cc023`
  and `a0ec64a8e`. The rejected selector and fixture were removed from the
  active tree. After recovering the exact inputs and checking the archives and
  ELF, the private build and output were removed (103 files, 16,085,717 bytes).
  Test the existing request's MTP confidence floor independently of expert
  placement. The older six-process width comparison at `dcb8ba8cf` found
  that fixed width two beat four on prose in every pair and lost on code in
  every pair. The B570's short-output threshold comparison also favored 0.9
  over 0.5 but generated only 128 tokens per request. For the ordinary stock
  1256-slot layout, `bench/sycl/decode_confidence.py` keeps width four
  available and compares confidence floors 0 and 0.9. First check actual
  verification widths and accepted positions on two fresh 65537-token
  prompts against the same engine's fixed-width trace. If widths shorten,
  run three balanced pairs of complete official thinking suites. Determine
  the strong/weak class from the reference and report the candidate's
  changed draft denominator, generation length, time to answer and TG.
  Preserve fixed-input prefill placement, PP and client TTFT independently.
  The reviewed `mtp-confidence-work-contract-20261121` audit confirms that
  draft-0 runs before the confidence check. Assess later draft work and
  target verifier width separately. Define `A_i` as the event that draft `i`
  is offered and accepted, and `s_i` as its probability given acceptance of
  the preceding draft prefix. Before EOS or output clipping, expected yield
  is `1 + sum_i product_{j=1..i} s_j`. Use the measured emitted count after
  clipping. The pilot records widths and acceptance by depth. Request-mean
  observer phase clocks and clean speed remain separate measurements.
  The decode-only committed-prefix candidate changes one other piece of work.
  After the target commits `a+1` rows of a window of width `T`, the MTP round
  ordinarily computes K/V for all `T` and then recomputes the selected row in
  its full forward. With `STRATA_SYCL_MTP_DECODE_COMMITTED_PREFIX=1`, ordinary
  uncoupled MTP4 selects the existing round graph for `a+1` rows. The
  full forward, subsequent draft chain and host synchronization stay the same.
  From the two clocked processes in `ordinary_decode_round_cost_v1` at
  `bf5f94144`, code offered 2340 and accepted 2036 drafts. Prose offered
  10143 and accepted 3442. Exclude the last verifier window, which generates
  no further drafts. Of the original 3117 and 13521 K/V catch-up rows, this
  can skip 303–304 (code) and 6699–6701 (prose). The ranges allow for the
  unknown last acceptance using its two and one unemitted positions.
  Elapsed savings need measurement because narrower matvecs can reread the same
  weights, and the first occurrence of each width can capture another graph.
  The `4d2efd356` translation unit passes the retained production SYCL
  syntax check with all 754 included files matching the recorded compiler
  inputs. The later linked build supplies the live state and timing checks.
  Qualify the context-boundary fix independently. The CLI admits `n + max_new`
  equal to the physical capacity. With a native pack, `n = capacity - 2`,
  width four and two requested outputs, the initial one-row verification fits.
  The next width-four window can fail the CLI range check before emitting the
  second output. Clip the final width after suffix lookup. Bound MTP catch-up,
  the standalone first draft and every subsequent draft step by the drafter's
  own `max_cells`, before staging or capture. The reference CLI has enough room
  for this width-four case's draft steps, so reproduce its refusal on a small
  exact-fit request. On the fixed image require both outputs, an actual
  truncated verifier window, and quiet validation. The 262142-input/2-output
  request and separate exact-262144 no-output fill use the same width-four
  head configuration. Qualify width eight with its supported head separately.
  Keep the safety fix in both prefix-selection arms.
  Both patch orders on the measured `fde0cdd11` image produce the same two
  source files. The fixed generator hash starts `81fe4b074a`. The MTP hashes
  start `243078ea8c` with the fix alone and `33f2e63708` with both changes.
  With both changes, the pinned `e69d184fc` syntax check passes for the MTP
  engine unit, normal generator and exact-fill diagnostic, with 754, 804 and
  805 matched includes respectively.
  The original CLI at `4ac41d827` completes 16384 batched rows on the
  16385-input, 16387-capacity request and exits at the expected refusal at
  position 16385. Validation and ownership close without a fresh driver fault.
  `bench/sycl/decode_mtp_prefix_build.py` projects only the two changed units
  onto the measured prefill source. The build at `59914a469` closes with 137
  fresh objects, 1474 verified compiler inputs, zero uncovered inputs and all
  twelve linked targets. The normal CLI ELF begins `cdc551c029` and the
  exact-fill diagnostic begins `5fe3a8a5f6`. The independent
  decode fallback and the measured prefill reference stay available through
  this comparison. The original fixed-context trial's same-build host check
  and width-four smoke passed. The smoke processed 16384 batched rows, started
  its first MTP catch-up at position 16384 and emitted both requested tokens
  in two verifier windows, with the last width clipped to two. The width-eight
  case exits 1 at initialization with `experimental sole-packed head requires
  MTP4` from `MtpLayer::bind`, before processing the prompt. The trial stopped
  there. Both processes closed with unchanged driver and no fresh faults.
  The head mode is `STRATA_SYCL_DECODE_Q6_DPAS_HEAD=2`. Width eight needs a
  separate check with the ordinary native head and its own allocation budget.
  The validation preload retains a provider reference through process exit.
  The continuation checks the archived width-four smoke and both original
  long-case commands, inputs and environments. Only the private fixture and
  output paths change. The width-four `near-four` run at `c559907fe` confirms
  262141 batched rows for a 262142-token input, two outputs, one accepted
  draft and actual terminal width two. The CLI exits normally and the trial
  closes without a fresh driver fault. The exact-fill trial consumes 262143
  batched rows and one final committed row, reaches physical position 262144
  with zero outputs, and passes the commit and refill waits. The 114688 GPU
  rows and 147455 host rows match the original placement. All 15 model assets
  match after normal exit, with closed ownership and no fresh driver fault.
  This qualifies physical fill on the fixed width-four configuration with
  prefix selection disabled. Live prefix RESTORE and clean speed still need
  their own evidence.
  The reviewed `decode-mtp-committed-width-numerics-20261121` audit finds a
  distinct Q8_0 matvec dispatch for one row. Row strides and per-row fused
  GR stay fixed across widths. Compare the earlier accepted K/V cells and
  scales separately from the selected cell overwritten by the full one-row
  forward. Check the draft ID and probability with the same staged inputs.
  The reviewed `decode-mtp-selected-row-kv-20261121` source audit supports
  rewriting the selected cell in the full forward after omitting its earlier
  K/V write. A separate change would have to retain the selected residual and
  token staging or remap the selected row explicitly. Shrinking those inputs
  along with the K/V pass leaves index `a=3` outside the staged rows. Measure
  the existing `a+1` route first, then test the selected-row change separately.
  `bench/sycl/decode_mtp_prefix_restore.py` pins the post-capacity live
  RESTORE comparison. It binds each catch-up line to its verifier window and
  actual commit, checks the draft and target state over the committed extent,
  and requires a shorter eligible catch-up at both lengths. A differing state
  fingerprint triggers a fixed-history numerical comparison before timing.
  The host trace parser accepts both planned shapes and rejects excess
  catch-up, skipped commit positions, wrong published extent and a terminal
  draft call. The first traced baseline restored 32769 tokens, read 8209 more
  and generated all eight requested tokens. It accepted all six drafts, with
  widths one, four and four and catch-up widths one and four. The campaign
  stopped at the no-shortening-opportunity check before the candidate arm.
  The server exited zero and ownership closed without a fresh driver fault.
  The V2 comparison uses the same native build and an 8209-token fresh
  suffix ending in the existing varied-prose task. The pinned model chat
  template renders the suffix with thinking enabled. The shorter resumed
  input has 40978 tokens; the near-capacity one has 262064, allowing 64 output
  tokens plus 16 spare positions. Both matched pairs generated 64 tokens,
  accepted 36 of 81 drafts and exercised 19 shorter catch-up windows. Each
  pair reduced MTP K/V catch-up from 105 to 62 rows with matching requests,
  accepted positions, committed state fingerprints and RAM snapshot checks.
  All four native processes exited zero with closed ownership and no fresh
  GPU fault. See `decode_mtp_prefix_restore_v2` in the linked service result.
  Retired 280 files and 54,045,800 bytes of private driver caches and process
  streams from the closed V1 and V2 RESTORE trials. Their outcomes match
  pushed commits `7a479bc5b` and `76bd5586e`. The seven local campaign and
  execution receipts retain their verified hashes. The speed suite reads the
  committed outcome and creates its own private caches.
  The six processes in `decode_mtp_prefix_generation_v1` closed normally
  with no fresh GPU fault and verified final model assets. The thinking-on
  official suite matched all request histories, token counts, draft counts
  and fixed-input prefill dispatch. Code TG rose in two of three pairs,
  with a geometric-mean gain of 1.073%. Prose rose in one pair, with a
  geometric-mean loss of 1.113%. Code's fixed-input client TTFT was higher
  in two pairs. Keep the original MTP route as the default. The linked
  decision retains individual TG, PP and TTFT pairs and the separate
  boundary and RESTORE evidence. Recheck the retained coherent decode
  reference with the same official prelude before choosing the next
  verifier experiment.
  Prioritize reducing the work inside MTP over deferring its two intermediate
  host waits. At `1532659f2`, the timestamps of 32 B570 graph replays cover
  ten complete uncoupled T4 chains and the first two graphs of an eleventh.
  Within each complete window, sum the two differences between the next
  graph's start and the preceding graph's end. Those two gaps together took
  0.140–0.201 ms (median 0.156 ms), within a median 31.365 ms from the
  round's start to the final step's end. The gaps account for 0.51% across
  these ten spans. The observer was active on one historical prose request.
  Its 33.33 ms of host drain per window includes time the GPU spent executing.
  Revisit deferred submission if a current same-queue trace shows a larger
  inter-graph gap. [Original event tuples](https://github.com/yqYo1/Strata/blob/1532659f22488169459040c4a510ca7160bb33fd/bench/results/2026-10-15-pr6-recovery-audit/mtp-model-observer-current.json)
  and [SYCL profiling timebase](https://github.khronos.org/SYCL_Reference/iface/event.html#sycl-info-event-profiling).
  At the fixed 1256-slot decode placement, resident experts handled 143,358
  of 1,498,080 routed entries for code (9.57%) and 366,480 of 6,492,000
  for prose (5.65%). For a placement comparison, measure held-out hits and
  unique CPU jobs per layer and verifier window. Several entries can share
  a job. An increase in hits alone supplies no elapsed-time estimate.
  The same two clocked processes at `bf5f94144` bound the logical expert
  weight volume represented by these jobs even without per-layer job counts. Code
  issued 1,074,995 unique CPU jobs over 3,121 verified rows, prose 3,718,375
  over 13,525. Use the measured pack's `native_experts.txt` (SHA256 prefix
  `d9ac2dfa3ee6`). Its 48 layers have seven expert blob sizes between
  1,510,400 and 2,662,400 bytes. Each layer can issue at most ten CPU jobs
  per verified row. Assign the total jobs to the smallest blobs first up to
  that per-layer limit for a lower bound, then to the largest first for an
  upper bound. Count a reused expert once per job. Dividing by the actual
  generated tokens gives 0.741–0.828 GB/output for code and 1.026–1.209
  GB/output for prose. At 70 tok/s these imply 51.9–58.0 and 71.8–84.6
  GB/s of logical weight reads. The highest measured median for the 1 GiB
  continuous RAM read on this host was 42.27 GB/s with three cores. If each
  job fetched its full blob from RAM at that rate, even the lower bounds
  would require reducing the bytes per output by 18.5% and 41.1% respectively.
  Cache hits, repeated weight loads and nonsequential access change that
  comparison. Recompute the bounds with the new job counts and draft yield
  after changing the expert ranking or speculative width. Physical DRAM
  traffic and the cost of GPU verification are separate measurements.
  The fixed-width verifier imposes a second limit on either placement.
  At width four a verifier window can emit at most four tokens. In the two
  clocked `bf5f94144` runs, even perfect draft acceptance and zero time
  outside verification would give only 13.84–13.90 tok/s for code and
  15.19–15.24 for prose if verification retained its measured duration
  (`4000 / verify_ms`). At 70 tok/s a four-token window allows 57.14 ms
  in total. Verification alone therefore needs at least a 5.04–5.06×
  reduction in time for code and 4.59–4.61× for prose under those favorable
  assumptions. With the observed yields of 3.604 and 2.017 tokens per
  window the required factors rise to 5.59–5.61× and 9.11–9.14×.
  For the placement and confidence experiments record the new verifier
  duration and yield together. Faster draft generation alone cannot close
  this gap at the measured target-verifier cost.
  The two completed on-mode caller profiles in `decode_service_overhead_v3`
  put per-layer copy retirement plus the final copy wait at 0.232–0.239 ms
  per verifier window across code and prose. This is about 0.09% of each
  profiled request's caller interval. The campaign's final overhead gate
  remains failed. Separately, the code request at `55265f9e3` spent 0.242%
  of decode in the resident completion-query wrappers. Keep the checked
  copy retirement and prioritize the larger layer-graph intervals.
  [Caller profiles](../bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json),
  [query profile](../bench/results/2026-10-15-pr6-recovery-audit/decode-query-profile-current.json).
  Revalidate the cooperative KV resolver on the measured fast prefill image.
  Its older same-ELF decode comparison `decode_kv_speed_v1` gained 6.77%
  on code and 7.00% on prose. Its 64K warmed prefill preservation check
  remains failed (0.48% longer).
  `bench/sycl/decode_kv_fast_prefill_build.py` projects only the measured
  decode phase selector and the owned cooperative translation unit from
  `644412c8e` and `93b964cae` onto the exact `fde0cdd11` build image.
  The projected delta is six files. The existing scalar KV translation unit
  retains SHA256 prefix `4473c66af0` and the other engine source bytes match.
  The `fde0cdd11` source label denotes the composed build recorded by
  `prefill_large_gather_model_source.py`. Reconstructing its 1171 files
  from the recorded Git inputs produced source SHA256 prefix
  `f35b6b1a3f` and recipe prefix `ee8be44d38`, both equal to the retained
  ELF's build record. All twelve measured-source files in the two Orca
  reviews match that reconstruction. Their original manifests are preserved.
  The retained `14508acbf` composition matches the eight source files used
  by the verifier projection and the context fix. Apply only the patch from
  `178d532^` to `178d532` to `src/core/mtp.cpp` and `src/program/generate.cpp`.
  On both bases, the resulting hashes start `243078ea8c` and `81fe4b074a`.
  Applying that patch before or after the six-file KV projection produces
  identical bytes for all nine checked paths, including the unchanged scalar
  resolver. The MTP shortening change stays independent. Qualify the
  boundary-only recipe first. The fast prefill builder exposes
  `prepare-boundary`, `configure-boundary` and
  `compile-boundary` for the two-file control on `fde0cdd11`. Its ordinary
  `prepare` requires that committed build and its completed boundary checks,
  then adds the KV layer. In `bench/sycl/decode_kv_fast_prefill_capacity.py`,
  run `boundary-prepare`, `boundary-host`, `boundary-small-four`,
  `boundary-near-four` and `boundary-full` in order. Reuse the original
  MTP4 inputs and environment. Both exact-fit cases require two outputs and
  a terminal width of two from the two-window, one-draft receipt. The
  full case requires the width-one commit and drain at 262144 with zero
  outputs. Bind all four results to the same exact boundary-only build.
  Standalone boundary admission reads the earlier exact-fit and physical-fill
  outcomes from their original commits. It binds them to the recorded
  fast-prefill build and checks the live runtime, model assets and new
  executables. The closed MTP test executables and the phase-naming reference
  are historical inputs for these standalone cases.
  The host input check on `c316a2666` reproduced all three token-body and
  environment hashes using the retained runtime. It refused a changed boot
  identity and a different parent executable hash. The new boundary executable
  still needs its own build and device outcomes.
  The build projection recovers the coherent foundation from its original
  receipt and Git sources. Preserve its executables for live comparisons and
  readers that still validate those bytes. The source reconstruction at
  `6bbcbf627` reproduced 1171
  files for the boundary control and 1172 for the cooperative build. Both
  contain the same two context edits and the unchanged scalar resolver.
  The planned fresh object counts are 137 and 139. The existing model
  admission still reaches `coherent_inputs` and `stock.check_build`, which
  validate the original composed image and its retained inputs. The replay
  and cache cohorts have ended. Check the remaining live readers before
  retiring an old build or opening the next timing cohort.
  Admission of the committed fast-prefill build reads the original retained
  decode identity from its build result. It still validates the fast-prefill
  source, recipe, compiler inputs, executables and same-build fixtures.
  The phase-naming executable was a build-time fallback. Retain coherent
  `14508acbf` for decode and `fde0cdd11` for fast prefill during the
  boundary comparison. Use the boundary-only build as the single temporary
  third generation. Select and retire a superseded reference at the next
  boundary before adding a cooperative build.
  The old phase-naming and MTP-prefix build outputs were retired after
  recovering both original build recipes and compiler inputs. The cleanup
  removed 1573 files and links with 410910775 regular-file bytes. The
  original build receipts remain for source recovery through the boundary
  comparison. See `retained_builds_review_20261103.boundary_candidate_reference_transition`.
  The standalone boundary build passes all four checks at `04b277311`.
  The 16385- and 262142-input cases each produce two outputs with final
  width two. The full input processes 262143 batched rows and commits the
  final slot at 262144 with a width-one window and a checked refill drain.
  It emits zero tokens. All three GPU processes exit normally without a
  fresh fault, and the final model asset hashes match.
  The 65537-token boundary timing at `ea0f97045` closes six independent
  processes and 18 matched fresh requests. Route, draft counts, provider
  identities and final model asset hashes match. All six native processes
  close without a fresh GPU fault. The median paired first-use prompt time
  and client TTFT fall 1.289% and 1.180%. The warmed medians of the paired
  time ratios are 1.000733 and 1.000881. These exceed the predeclared
  preservation limits. The process-median warmed rates are 873.147 and
  876.725 tok/s for retained and boundary, with lower rates than the
  earlier 65537-token cohort on the retained executable. Use the paired
  ratios for the preservation decision. The cause of the period difference
  remains unresolved.
  The PLE attribution candidate enables `STRATA_SYCL_PREFILL_PLE_WAIT_PROFILE=1`
  for SYCL layer-major prefill. Its layer-one record includes the position,
  rows, table mode and completed gathers after the existing queue drains.
  The future retrieval clock covers the caller's exposed gather wait. The
  separate host interval also includes upload and PLE work submission.
  For the syntax check of `fcf8acbac`, apply its single source diff to a
  private copy of the verified boundary build's prefill translation unit.
  Keep the original `strata_prefill` compiler, flags and environment. Add
  the original unit directory as a quoted include directory. Substitute
  syntax-only mode and a private dependency file for the object output.
  Verify that every dependency matches the recorded compile or external
  inputs, with the private translation unit as the sole overlay. The check
  passed with 799 verified dependencies, overlay SHA-256
  `903bc778ffbad603014a6e173d9fe3908b4f42534832185ccdf3612a61e1bbf9`
  and compile database SHA-256
  `36c68b081b50699928315c206928b12f1485c5a11525e08ddd7664cace4e73c5`.
  After a native build, bind the records to request boundaries and measure
  observer overhead with the same executable.
  The split-prefill accumulator can check its two fixed dequantization
  workspace ranges once per nonempty window with
  `STRATA_SYCL_PREFILL_SPLIT_SCRATCH_CHECK_ONCE=1`. The workspace allocations
  and queue belong to the invocation through its checked drain. Each expert
  blob and job source range still has its own check. The default checks both
  workspace ranges on every new expert. The first matched 117489-token pair
  executed 18860–18900 dequantizations across 48 windows per request. This
  bounds the removed registry lookups at 37624–37704 calls per request. Measure
  their host cost before assigning any change in request time to the lock.
  Project this change onto the exact boundary image as the sole
  `src/prefill/moe_window.inc` overlay. For syntax verification, copy the
  original prefill translation unit into the same private directory as the
  modified include, then use the pinned compiler, flags and original source
  directory as a quoted include directory. Compare dependency hashes against
  the original compile and external inputs, with the one include as the only
  changed input. This projection uses the original PLE implementation.
  For the same-image experiment, qualify checked numerical output, actual
  split placement, route counts, cancellation and the decode handoff first.
  Use the established four fresh 117489-token requests and three independent
  pairs in off/on, on/off, off/on order. Capture prompt time and client TTFT
  for the first request and warmed median separately. Hold the scalar decode
  resolver, aligned KV reuse and expert dispatch fixed, with PLE timing
  disabled in both arms.
  Count a warmed gain when at least two pairs improve and their median ratio
  is below one. Record first-use TTFT separately and verify phase preservation
  before adoption. Preserve the actual new build inputs and sample spread.
  Run the independent 117489-token comparison with
  `decode_kv_fast_prefill_preservation.py boundary`. Keep three matched
  pairs and separate first-use and warmed prompt time and client TTFT.
  Both arms use scalar decode KV and the selected prefill configuration
  for the length. A new 65K plan must establish preservation before the
  cooperative comparison can advance. Keep the completed 65K outcome.
  After qualifying both lengths, use the boundary control as the prefill
  reference for the cooperative comparison. Prove the original fast-prefill
  source and recipe against its executable before retiring it. The previous
  build record supplies source projection and historical workload admission
  for the cooperative candidate. The paired timing still checks the exact
  live fast-prefill executables and fixtures. Keep its build record for
  reconstruction after that output is retired.
  Rebuild the complete engine because the verifier headers change. Add the
  existing `ae74403d1` planner fixture as a separate build target linked to
  the same `strata_kernels` archive. Run its traced bounded smoke before its
  full oracle/graph validation. The fixture exercises the physical last cell,
  invalid tails and the planned 2/4-query shapes. Commit both results before
  starting the model diagnostic.
  Before timing, `bench/sycl/decode_kv_fast_prefill_restore.py` checks
  both settings at each fixed MTP width 2 and 4 on the candidate binary,
  plus the scalar width-3 fallback on the shorter case.
  It uses the established 32769- and 253919-token fresh prefixes, an
  unrelated 33-token switch and 8209 fresh tokens after live RESTORE.
  A final distinct 33-token request returns from the restored decode to
  the short scalar path in the same process.
  The first two requests emit one token. The restored request has an
  eight-token output cap, leaving room for the last verifier window even
  when it commits more rows than it emits. Bind each verifier position,
  width and capture to the same native reply. Compare the actual committed
  state, snapshot payload and expert-cache restoration against the off arm.
  Treat this bounded state and route check separately from exact 262144-token
  fill and from both-phase performance.
  Use `bench/sycl/decode_kv_fast_prefill_capacity.py` for the standalone
  physical fill on the candidate's linked diagnostic. Replay the original
  16385-token smoke and 262144-token input with aligned prefill KV reuse
  fixed in all four cases. Only the decode KV selector changes within a
  pair. Require the previous residual placement, KV prefix counts and host
  mirror writes. At capacity, require 262143 batched rows and the checked
  scalar width-one commit and drain of the final cell with zero output.
  The leased server's exact-full protocol and RESTORE remain separate gates.
  Keep expert placement, logical context, resident KV and copy geometry fixed.
  The cooperative timing uses three counterbalanced blocks per length with
  the boundary control, KV off and KV on. Off/boundary measures the linked
  change with scalar KV fixed. On/off measures the decode selector on the
  same executable. On/boundary measures their combined impact. Bind all
  three to the earlier paired boundary timing and keep its result separate.
  Keep a later direct check of the complete change against the retained
  phase references before adoption. Hold the measured prefill selection
  fixed at KV reuse and vector gather for 65537, and KV reuse with scalar
  gather for 117489. Require the original bodies, read counts, expert work,
  draft counts, and actual decode capture route. Keep first and warmed
  prefill and client TTFT ratios separate. After both length gates pass,
  `bench/sycl/decode_kv_fast_prefill_generation.py` compares the decode
  selector off/on in three independent pairs on the same new executable.
  Keep the stock 1256-slot decode placement and the 65K KV reuse/vec8
  choices fixed. Run official long, six-turn agent and both thinking-enabled
  long-generation workloads in each process. Verify the captured width-four
  selection and the actual prefill dispatch. Retain TG both with and without
  matching generated and draft counts, and assess agent preservation only
  for matching histories and work. Measuring the new and retained binaries
  on the sustained suite is still a separate preservation check.
  Require prefill and TTFT preservation in both comparisons and higher TG in
  at least two off/on pairs with a geometric mean ratio above one for each
  sustained workload. Close on a state, identity, ownership or fault failure.
  This projection has source evidence only. Its build waits until a comparison
  generation retires and the current serialized measurements close.
  To distinguish layer-graph execution from the host's wait on the same
  event, sample the ordinary, unsplit 48-layer target verifier at width four.
  Set `STRATA_SYCL_DECODE_SEGMENT_CLOCK_REQUEST` before engine startup to
  the native request ID of one sustained workload. The SYCL server instantiates
  profiled layer graphs for eligible decode shapes. Only eligible windows
  9–12 of the selected request query clocks. The first graph includes input
  staging and the first layer's pre-router work. Interior segments span the
  preceding layer's post-router work and the next layer's pre-router work.
  Segment 48 contains the final layer's post-router work and output head.
  Shared/resident and PCIe expert graphs retain their original instantiation.
  The host reads the reusable event after its existing completion wait and
  before the next launch replaces it. Report the device execution envelope,
  submit-to-start delay, host wait and CPU expert callback wall time separately.
  Their sum would double-count overlapping time. The B570 command service
  profile at `6afcc0ddb` retains 17 submit-clock inversions. SYCL's
  event contract orders submit before start. Treat such device returns as
  evidence that queue-delay attribution is unavailable for those segments.
  [B570 command samples](../bench/results/2026-10-15-pr6-recovery-audit/moe-window-census-current.json),
  [SYCL event timing](https://github.khronos.org/SYCL_Reference/iface/event.html#sycl-info-event-profiling),
  [DPC++ graph profiling](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc#queue-properties).

  For a new model timing plan use `bench/sycl/decode_segment_envelopes.py`.
  Retain all inverted or zero-valued submit readings and leave the whole-window
  delay unknown unless all 49 submit comparisons qualify. Require complete
  ordered start/end and checked native completion in every case. Sum the
  49 layer-graph envelopes and the 48 gaps
  between consecutive layer graphs. They must reconcile with the first
  start to final end duration. The gaps can contain expert graphs and CPU
  dependencies. The same-position wall controls keep their local overhead
  gate. The queued source check retains its original pinned reader.
  First check the two compiled translation units against the retained build's
  production SYCL recipe and exercise the reader on ordered and damaged rows.
  For model evidence use a coherent reference and candidate build, then
  separate same-binary wall/profiled/profiled/wall processes for each
  thinking-enabled code and prose workload. Keep the current ordinary capture,
  stock expert placement and T4 decode settings. Check native request IDs,
  accepted/offered drafts, emitted tokens and expert work between each pair.
  Set the same request selector to `wall:<id>` in the wall controls. They
  retain ordinary graphs and time just the four matching verifier windows
  from before capture lookup through successful completion. Compare each
  window by absolute position and width, charging timestamp queries and raw
  formatting to the profiled arm. Require each matched window and the whole
  sustained per-token time within 1.10× in both directions. The aggregate
  request rate alone can conceal an expensive observer on four windows.
  Require all 196 consecutive valid start/end samples per profiled process,
  normal closure and no new Xe faults. Inspect each transition's graph and host time
  separately. A failed overhead screen leaves the clocks diagnostic only.
  This observer has no model timing or speed claim yet.
  The decode MoE combine candidate changes the post-router path for ordinary
  unsplit, batched T4 verification at width 2560 with ten routed experts.
  The previous short device capture at `current_decode_device_service_v1`
  observed 133–177 ms of mapped-row copy service in two 2.46–2.82 s windows.
  Its decode observer overhead failed the predeclared screen. Use those
  observations to select this experiment. Keep the clean verifier clocks
  above as the request-level reference.
  With `STRATA_SYCL_DECODE_MOE_COMBINE_MAPPED=1`, the reduction reads each
  expert result from the existing mapped CPU output or from the GPU output
  according to the published route. It keeps the original FP32 expert order
  and shared-expert addition. The fused reduction needs the row selection to
  remove the assembled tensor. Test this as one replacement for copy, add
  and reduction while expert placement, PCIe scheduling, target verification
  width and MTP all stay fixed. This removes two recorded kernel nodes per
  eligible layer and the logical write and read of the assembled tensor.
  That tensor traffic is 39,321,600 bytes per 48-layer T4 window. The same
  mapped CPU result values must still reach the GPU.
  The retained SYCL build compiles `src/program/generate.cpp` into the
  actual `strata` target. Its `--native` initialization enables the existing
  native reduction before graph capture. The mapped projection keeps that
  initializer and selects its new reduction for ordinary, unsplit decode
  windows with four rows. Compare the selector on and off with the same
  `--native` arguments. Confirm eligible capture and replay in the model test.
  Build `sycl_decode_moe_combine_test` and the native engine against the
  same kernel objects and production flags. Preserve the retained measured
  engine and its fixtures through the comparison. First run the component's
  `validate` arm with bounded runtime tracing and parameter validation.
  Both captured graphs must match the ordered FP32 CPU reference and guards
  for 0, 4, 10, 20, 39 and 40 routed GPU rows. Reuse each captured graph with
  two different publications of the CPU rows. Poison unused rows so that
  selection failures show up in the output. Inject one duplicate
  destination into the candidate and require its token to report invalid
  output while other tokens receive the ordinary CPU contribution.
  Run three fresh component processes in time-forward, time-reverse,
  time-forward order with the qualified GPU runtime settings and parameter
  validation. Collect API tracing only in the correctness process.
  Compare four counterbalanced pairs of 512 completed graph replays at each
  of 4, 10 and 20 GPU rows. Republish only the CPU miss rows between completed
  replays. Retain both launch-through-completion and publication-through-
  completion clocks. Require each graph batch to reach 10 ms and charge
  at most 1% for the measured empty timing bracket. Advance the T4 path
  only if both candidate medians are faster at all three hit counts in
  all three processes, with normal teardown and no fresh fault. Use these
  local intervals to decide whether to test the native server.
  Before clean model timing, verify traced T4 dispatch, CPU callback and
  graph completion at the normal handoff, near the end of context and after
  RESTORE. Production performance requires
  three independent matched pairs of the official long, agent, and both
  sustained thinking workloads with placement and width fixed. Preserve
  prefill PP and client TTFT and screen controlled decode separately.
  The mapped-row fusion has no speed or device correctness result yet.
  A separate same-ELF decode screen covers the empty PCIe expert projection.
  The first 30 measured requests in the 65537-token comparison reported zero
  expert-copy bytes and groups at `--pcie-frac 0`. For the same prompt body,
  their CPU expert jobs and entries were identical in all ten processes.
  The compiled verifier at `12ea4b002` still records the inline zero-count
  projection. The grouped kernels return before loading weights, but their
  graph also records a scratch clear, SwiGLU and activation quantization.
  Its early scheduler records the same projection separately and launches it
  only for a positive count. The controller at `d4c210196` holds Arena copy,
  overlap, the input and expert work fixed in two opposite-order pairs.
  Check the zero-copy route on the short startup request first. Retain graph
  capture, prefill, client TTFT and native decode clocks per long request.
  The early selector also affects prompt verification. A gain warrants a
  decode-only capture and replay change, followed by sustained generation.
  Before training another ranking, compare the shipped Niko Intel profile
  to the current ordinary profile (source `8cb26e609`, data already in
  this fork at `97e9c9847`). On the actual 1256-slot layout, layers
  0–46 have 26 slots each and layer 47 has 34. Of the occupied
  (layer, expert) pairs, 400 are common and 856 change. This comparison
  holds capacity fixed. Niko's reported gain used a B70 at a different
  capacity. The earlier local code-trained ranking reduced code latency
  by 11.18% but increased prose latency by 1.24% in the one-pair
  diagnostic (`cache-window-rank-current.json`). Compare the Intel
  profile on both workloads and check the prefill loan and restored map.
  The stock and Intel layouts both pass the 32769-token and 253919-token
  RESTORE controls at `f67440687`. The near-capacity suffix reads 8209 new
  tokens, reaches 262128 input tokens and generates one token. All four native
  processes exit normally with no fresh Xe faults. In that restored T1
  window Intel uses 416 CPU expert jobs versus 426 with stock. With the short
  restored prefix it uses 425 versus 414. Use the three independent matched
  full-suite pairs for the sustained generation decision.
  In the first sustained pair at `5ed90bf00`, both arms use a one-row first
  window followed by width four. The native counters start after cache
  refill. The table uses the same pack-based traffic bound.

  | Workload | Windows, stock → Intel | Unique CPU jobs, stock → Intel | Engine ms/window | Output/window | Logical GB/output |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Code | 781 → 816 | 1,074,995 → 1,124,557 | 270.318 → 269.042 | 3.604 → 3.577 | 0.741–0.828 → 0.748–0.835 |
  | Prose | 3382 → 3199 | 3,718,375 → 3,506,954 | 254.990 → 255.045 | 2.017 → 1.965 | 1.026–1.209 → 1.049–1.238 |

  Jobs per window move by +0.12% for code and −0.29% for prose. The shorter
  prose response reduces total jobs and raises logical bytes per output.
  The Intel arm's lower bound implies 52.36 GB/s for code and 73.45 GB/s for
  prose at 70 tok/s, assuming each job reads its packed blob once. Physical
  traffic needs separate evidence. All three long/agent/generation pairs
  closed at `3e432e20f`. Intel won zero TG pairs on either sustained workload.
  Its paired geometric mean TG was 0.98758 times stock for code and 0.93350
  for prose. Both observed MTP classes qualified. Agent histories differed
  in all three pairs, so the follow-up rate does not qualify preservation.
  Keep stock placement for the next isolated comparison. Measure verifier
  wall time and emitted tokens per window together when changing draft width
  or host-expert work.
  A later comparison can keep the winning profile fixed and vary its
  per-layer quotas. Removing `--expert-cache-per-layer` also changes the
  meaning of `--expert-cache` for this native pack. In the measured SYCL
  source the global allocator admits ranked slots up to the lesser of free
  VRAM after the reserve and the requested count times the largest blob.
  The per-layer 1256-slot layout uses 2,575,206,400 bytes. With sufficient
  free VRAM, global 1256 would use about 768 MB more with either shipped
  ranking, changing prefill headroom at the same time. Requesting global
  967 instead caps it at 2,574,540,800 bytes. With the stock ranking this
  funds 1241 slots (2,573,849,600 bytes, 1021 common pairs with per-layer
  stock). With the Intel ranking it funds 1244 slots (2,573,772,800 bytes,
  1097 common pairs with per-layer Intel). These counts use the pack's
  256-byte-aligned sizes and the complete STRP order.
  An unmodified global ranking at 967 fails the 8192-token scratch budget.
  The observed per-layer loan starts at slot 132; dropping one slot leaves
  2,319,692,800 bytes, below the required workspace. Keeping the first 128
  global slots leaves only 2,308,224,000 (stock) or 2,301,670,400 (Intel).
  Use `bench/sycl/decode_profile_tail.py` to permute the admitted set before
  GPU qualification. It preserves every decode member and the first excluded
  pair. With 26 (stock) or 32 (Intel) replacements among the first 132 members,
  the tail from 132 has exactly the reference 2,321,664,000 bytes and its
  first slot is exactly 1,971,200 bytes. The planner therefore selects the
  same 8192-token workspace whenever the full computed cache is admitted.
  The unborrowed experts and the bytes that must be refilled per layer can
  differ. At smaller admissions, recompute both budgets. Before timing, verify
  actual admission, occupied slot identities, scratch loan/refill range and
  both near-full RESTORE and short decode-cache handoff. Measure the change
  in unique CPU jobs and actual throughput on both workloads.
  Use `bench/sycl/decode_global_profile_restore.py` after both shipped
  per-layer rankings pass their RESTORE comparison. Keep the same native ELF
  and the 1500 MiB reserve. The predicted 1241/1244 global slots must be
  reported by the server, including the primary-device slot count. Read back
  every expert payload and check all 48 per-layer membership masks against the
  original ranked prefix after the fresh prompt, the 33-token switch and the
  resumed conversation (payload readback applies to the two loans).
  Compare the fresh and restored prefill bank, row counts, KV copies and
  expert jobs to the measured ordinary controls. Lower free VRAM at
  startup can reduce the global admission even for an explicit count.
  Keep that outcome separate from an 8192-token scratch shortage.
  The 8192-token scratch loan in both original RESTORE fixtures begins
  at slot 132 and covers 1124 slots (2,321,664,000 bytes). On this
  per-layer path, both the slot ranges and the slot widths depend only
  on layer and quota. At the same split the loan therefore has the same
  byte extent with either ranking. The expert identities do change.
  Only 48 of the 132 unborrowed experts are in both prefixes, and only
  three occupy the same slot. Across the full cache, 27 occupy the same
  slot. These are predictions from the two committed ranking files.
  `bench/sycl/decode_profile_restore.py` checks the actual 48-layer
  decode membership against each selected file after a long prompt, a
  33-token conversation switch and the 8209-token restored suffix.
  The diagnostic reads back the borrowed and retained expert payloads
  and checks the loan extent on both the 32769 and 253919 input cases.
  Keep its observed CPU jobs and GPU hits separate from clean decode
  timing. Exact generated-token equality across different memberships
  is not a correctness criterion.
  After both profiles pass those state checks, use
  `bench/sycl/decode_profile_generation.py` with the same executable. Run the
  official thinking-enabled long, agent and sustained-generation suite in three
  independent stock/Intel pairs, reversing the second pair. Only the expert
  ranking changes. Check the fixed fresh prompt bodies, prefill dispatch, prompt
  times and client TTFT on both generation workloads. Compare sustained TG
  and MTP yield on code and prose, and keep agent request-history identity
  separate from its achieved follow-up rate. Require higher TG in at least two
  pairs and a geometric mean ratio above one on each sustained workload.
  Report prefill and agent preservation separately before selecting a ranking.
  The eight traced state/RESTORE processes at `4775de3b0` expose a separate
  first-token cost on the resident-input executable (`b35d66a4`). On the
  first one-token decode per process, the verifier's graph capture clock
  covered 4.771–5.030 s of the 5.033–5.222 s decode (94.1–96.5% within
  each request). The intervening 33-token prompt reused the graph. After
  RESTORE the same one-token window took 28–52 ms to capture, in a total
  decode of 156–256 ms. The clock in the compiled `Verifier::capture`
  includes recording and finalizing segments, diagnostic output and the free
  VRAM query. Driver compilation time within it is unknown. Measure these
  parts before moving first-request work into startup. Use the separate
  sustained-generation suite above for the steady decode question.
  The resident-input candidate at `1bf54ec32` is integrated into
  `feature/sycl` for validation. It lets eligible prefill layers read the preceding layer's
  BF16 bank directly and traces cumulative reads and copied bytes after
  the existing layer drain. Decode is outside this change. The model
  controller at `6e6a37930` checks the expected 46 of 48 layers in each
  bank-backed arm and compares state and RESTORE across all four input
  and output combinations. Its 253919-token prefix plus 8209-token
  continuation reaches 262128 tokens and exercises a 16-row tail;
  the final physical context slots need separate coverage. The server reader
  at `f7e0ddeb0` checks the effective 512-slot bank for the mixed-residency
  prefix and the staged 384-slot bank for its fully resident suffix. Both
  use ordinary SYCL copies. The standalone controller at `b89af0bff` checks
  all four input/output choices at 16385 and exactly 262144 tokens, including
  the engine's checked final-slot marker and zero output at capacity.
  The coherent engine build at `bea6f8199` produced 11 executables from 133
  translation units. The host upload plan passed. The Level Zero traced
  BF16 fixture at `b92d822d7` passed for 1, 63, 7905, 8191 and 8192 rows
  on the B570. A two-piece window (8 + 3 rows) also passed for all four
  input/output combinations, including the final callback. Validation was
  quiet and the process exited normally without a fresh Xe fault.
  The four 32769-token server arms passed with identical state, output and
  RESTORE data, including the intervening 33-token cache-layout request and
  the 8209-token continuation. Direct input eliminated 30,870,077,440 bytes
  of requested GPU-local copies on the fresh prompt and direct output
  eliminated another 32,212,254,720 bytes. The combined arm eliminated both.
  All four owners closed with zero native exit and no fresh Xe fault.
  At 253919 tokens all four arms passed at `4775de3b0`. Each placed 114688
  batched rows on the GPU and 139230 in RAM. On the fresh prompt, direct
  input avoided 108,045,271,040 requested D2D bytes and direct output
  avoided 112,742,891,520. On the 8209-token restored suffix those values
  were 7,732,592,640 and 8,068,792,320 bytes. The combined arm eliminated
  both copies in each request. Active state, output hashes, publications,
  decode-cache loan and restored payload matched through 262128 tokens.
  All eight server processes closed normally with no fresh Xe fault, and
  the final 15 model asset hashes matched. The separate standalone
  final-slot check is recorded below.
  The four-way comparison from `4d9b176e5` closed at `73d9394a9`.
  Each of 16 independent clean processes handled three fresh 65537-token
  prompts. Every request produced 24 tokens and accepted 16 of 21 drafts.
  Across the four counterbalanced blocks, warmed process median rates were
  888.26 tok/s for copy, 885.30 for direct input, 891.71 for direct output,
  and 888.00 for both. Direct output shortened warmed prefill time against
  copy in every block by 0.30–1.11%. Median client TTFT was 73.70 s for
  direct output and 73.99 s for copy. Direct input and the combined change
  had mixed results. Keep direct output as a candidate on this executable.
  The earlier direct-output loss on a different executable remains recorded.
  The short outputs leave sustained decode open. The standalone capacity
  run exercised the final slot independently of these timing samples.
  Also revalidate the historical 117489-token workload on this ordinary
  build with the configured context kept at 262144. At `792ec1ee5` the
  reference clocked 1045.73–1051.06 PP across three warmed process medians
  (median 1048.98). It used context and resident KV both 131072; the current
  control uses 262144 and 20480 respectively. Both request split mode 10 and
  384 expert slots. In the historical reference, increasing the input from
  65537 to 117489 tokens increased the median warmed expert-bank upload
  request from 37.173 to 38.541 GB. That reduces the requested bank bytes
  per input token from 567.2 to 328.0 kB. These counts cover this bank only.
  Compare the four current requests with their historical counterparts by
  request hash, expert-bank bytes, jobs, fused Down jobs, routing-map bytes
  and actual draft offers and accepts. All twelve historical requests produced
  24 tokens with 16 of 21 drafts accepted. Check each body against its own
  reference before attributing a time difference to the engine.
  Record any differences separately from the speed decision. The historical
  117K route rebuilt 223,227,900 draft-head bytes; the current route expects
  zero. Equal expert work would still leave that handoff difference.
  For 117488 batched rows the production planner instead
  tries 128 slots and one split accumulator window. With the current model,
  it budgets 2,406,154,240 bytes of BF16 residual, 531,988,736 bytes of extra
  accumulator including its guard, a 340,787,200-byte expert bank, 241,555,328
  bytes of QSA prefix, 256 bytes of score status and 512 MiB of headroom.
  The total is 4,057,356,672 bytes before admission. In the three 64K control
  requests at `880c93b14`, the same formula with its actual 384-slot bank
  and no extra accumulator reproduces the logged fixed budget exactly.
  Those requests reported 188–195 MB of free VRAM beyond the projected 117K
  need. The split uses 897,762,560 bytes of aligned planes and guards
  followed by a 265,994,240-byte mixed-input tail inside the existing
  1,180,698,624-byte shared arena, leaving 16,941,824 bytes. The same
  plane calculation reproduces the measured 888,410,624-byte tail offset
  at 64K. The largest proposed individual allocation (the residual)
  is below the reported 10,132,807,680-byte single-allocation limit.
  The four traced requests at `5c760036d` admitted this layout with
  4,245,643,264–4,262,002,688 bytes free at the planner check. Each
  completed 48 GPU windows, including the split tail guard and queue drain.
  Their request hashes, expert jobs, fused Down jobs, requested expert
  and row-map bytes, and draft offers/accepts matched all three historical
  processes body for body. The ordinary route restored zero draft-head
  bytes. The traced native process exited normally without a fresh Xe fault.
  At the same 262144 capacity, raising resident K8V8 from 20480 to
  262144 cells adds `12*(262144-20480)*1056 = 3,062,366,208` pool
  bytes across the main attention layers. The mode-1 residency-map
  allowance from `kv_stream_map_bytes(5120)+6*16` is 1,230,720 bytes
  across the twelve layers. Even crediting that entire allowance and
  the measured 276,824,064-byte prefill stage leaves about 2.78 GB
  extra, before other layout changes. The observed 117K prefill plan's
  margin above its fixed requirement was only 188,286,592–204,646,016
  bytes. Reproducing the historical all-resident layout at full capacity
  requires changing another major allocation or its lifetime. Use the
  one-layer KV reuse comparison to isolate transfer cost first.
  Increasing the batched rows from 65,536 to 117,488 multiplies them by
  1.793. The ordered QSA loop grows from 100,638,720 to 181,683,840
  tiles (1.805 times): each of 12 layers and two KV heads processes
  `ceil(min(position + 1, 2051) / 32)` tiles per fresh query. Counting
  padded QK and PV matrix work gives 52.76 and 95.25 TFLOP respectively.
  In contrast the median warmed expert-bank payload increases only from
  37.17 to 38.54 GB (1.037 times). These different scaling rates motivate
  measuring both lengths. Kernel time and physical traffic at 117K remain
  open; the tile count is derived from the executed geometry.
  In each of the four traced 117K requests, the streamed KV quantizer
  published 1,488,807,936 bytes to the host. Each of 12 attention layers
  issued 14 calls for 8192 rows and one for 2800 rows, with four copies
  per call (720 copies in all). At the previously measured 6.56 GB/s
  device-to-host rate for queued 64 MiB transfers, the same payload
  would take 0.227 s. The two warmed 64K PTI windows at `c40c8c1bd`
  recorded 384 D2H commands and 830,472,192 bytes each, exactly the
  count and volume expected from eight chunks per layer. Their device
  service was 147.960 and 147.956 ms (5.61 GB/s). At that rate the
  117K payload would take 0.265 s. The PTI summary groups commands by
  direction; it supplies no per-producer timing or 117K queue-wait time.
  A change to host publication would need to explain a larger scheduling
  or synchronization cost to recover several seconds from this route.
  There is a separate transfer in the other direction. Before each
  streamed QSA chunk, `kv_stage_from_host` reloads every page before
  its starting position into the shared stage. For a fresh position-zero
  request, chunks of 8192, 12 layers and 1056 bytes per cell, the code
  requests 2,906,652,672 bytes at 64K and 10,899,947,520 at 117K.
  This follows from the executed positions and copy lengths. The full
  64K PTI window labeled 344 copies / 3,577,741,312 bytes as H2D;
  source arithmetic alone cannot assign individual events to this caller.
  The opt-in `STRATA_SYCL_PREFILL_KEEP_KV_STAGE=1` skips that reload
  for contiguous, page-aligned chunks of the same nonfinal layer when its
  typed Q8 writer has already populated the private stage on the same
  queue. Each new layer or request starts without a reusable prefix.
  Host publication continues after each chunk, including the partial tail.
  External layer taps and the final output-callback layer keep ordinary
  reloads. On this MTP route the 11 eligible layers could save 308 copy
  commands / 2,664,431,616 bytes at 64K and 616 / 9,991,618,560 bytes
  at 117K. At the three measured SYCL host-to-device process medians
  (6.292–6.485 GB/s, `prefill_copy_engine_v6`), the removable payload
  corresponds to 0.41–0.42 s at 64K and 1.54–1.59 s at 117K. If that
  transfer time is fully exposed and other work is unchanged, the 117K
  control would still need about 5.1–5.2 s to reach 1000 tok/s. The
  calibration used 256 MiB host-USM copies. Actual prefix copy sizes,
  submission costs and overlap can change the saving in either direction.
  The official dispatch check at `c6eda71cb` verified both lengths across
  eight processes, two requests each. The per-layer reload counters and
  observed append positions imply 336 prefix copies (2,906,652,672 bytes)
  for the 65537-token control and 28 with aligned reuse
  (242,221,056). At 117489 the counts were 672 (10,899,947,520 bytes)
  and 56 (908,328,960). Scalar and vector gathers produced the same copy
  totals. For the two 117489-token workloads, 7441 and 7434 jobs met the
  gather threshold in every arm. The observed dispatch matches the volume
  model at both lengths. Compare untraced request wall times separately.
  In these 16 traced requests, compare the native batched read, expert
  loan and checked refill intervals with the committed per-request prompt
  time. At 65537 tokens the batch occupied 98.87–99.50% of prompt time.
  Refilling 1124 slots took 388.1–877.9 ms. At 117489 tokens the batch
  occupied 99.03–99.69% and refill took 386.6–1125.8 ms. Lending took
  at most 0.6 ms. At 117489 the three intervals together left 3.9–4.2 ms
  of prompt time unassigned. The batch includes
  PCIe transfers, CPU waits and GPU kernels. Clean A/B timing and hardware attribution remain separate.
  The 117489-token comparison closed at `568d015ba` with 16 clean native
  processes and four fresh prompts per process. Across four process medians
  the warmed rates were 1000.36 tok/s for ordinary reload, 1039.64 for KV
  reuse, 1014.31 for vector gather and 1024.16 for both. KV reuse with
  scalar gather won all four pairs against reload (2.409–6.581 s less).
  Median paired prompt time fell 3.77%; median client TTFT fell from
  117.636 to 113.214 s. All four warmed KV rates exceeded 1000; the first
  long prompts were 931.66–957.54 tok/s. At 117489 tokens, 1000 tok/s
  allows 117.489 s of prompt time. Across the four KV-reuse processes,
  reaching that rate on the first long request would require saving
  5.209–8.618 s, or 50–73% of the observed first-to-subsequent prompt-time
  gap in each process. Use this budget to interpret the driver-cache results
  alongside the separately measured startup and client TTFT.
  Vector gather alone also beat reload in four pairs. Combining it with reuse
  won two pairs against each individual change. Select KV reuse alone for
  the split-route length curve and decode-preservation suite.

  The 64 native footer records in that same comparison each report 128
  slots, 127 permanent slots and zero permanent hits. Each request uploads
  18,860–18,900 blobs and requests 38.507–38.589 GB. The route plan groups
  all jobs for an expert consecutively and the split caller uses the same
  dequantized weights for those jobs. A single ordered scratch slot would
  free 338,124,800 bytes of the 340,787,200-byte bank. The comparison uses
  the ordinary source view and the same in-order queue.
  Mode 10 now accepts an explicitly smaller bank through the existing
  slot setting. `bench/sycl/prefill_split_bank_build.py` projects the one-file
  change onto the measured 117K source and recipe. At a serialized boundary,
  `bench/sycl/prefill_split_bank_compare.py` compares actual 128-slot and
  one-slot admission with aligned KV reuse and scalar gather fixed. The first
  two processes check the route, guards, response digests and decode cache
  restoration. Three fresh process pairs then compare first and subsequent
  prefill and TTFT separately, with identical blobs, requested bytes, expert
  jobs and draft counts. Check full-context and sustained decode before use.
  The earlier ordinary-RAM lane lost its component comparison with one
  dependency marker per expert. Evaluate batching, queue and timing changes
  separately from this capacity change.

  The 65537-token comparison closed at `84c883393` with 16 independent
  processes and 48 fresh requests. The subsequent-request rates were
  984.54 tok/s with ordinary reload, 980.22 with KV reuse, 984.42 with
  vector gather and 1009.33 with both. KV reuse alone won one paired block
  and vector gather alone won two. The combination won all four blocks
  against each single change and the reference. Median paired prompt time
  fell 2.32% against the reference and warmed client TTFT went from
  66.739 to 65.097 s. First long-request TTFT medians were 77.332 and
  75.976 s. Three of the four warmed combined process rates reached 1000.
  Select the combination for validation on the ordinary 384-slot route at
  this length. The 117489-token split-route comparison selects KV reuse.
  Run `bench/sycl/prefill_ordinary_generation.py` for the 65K combination
  on the measured executable with the stock decode placement fixed. Its three
  matched pairs cover the official long case, agent-session and both
  thinking-enabled sustained workloads. The 65K prefill-only speed comparison
  and final-slot checks remain separate evidence.
  The 9,991,618,560 fewer requested H2D bytes would take 1.54–1.59 s at
  the 256 MiB calibration rate. The larger observed time differences
  leave copy issue, synchronization, actual copy size, overlap and
  run variation to distinguish before assigning the gain to PCIe bandwidth.
  The Q8 prefix helper issues four direct SYCL queue copies per prefix in
  `src/kernels/sycl/kv_stream.cpp`. The compatibility memcpy counters miss
  this call path. Use request-scoped direct call and event observations for
  the 8192–114688-cell prefixes, keeping scalar gather and clean timing fixed.
  The first six independent processes through `6b5e834e1` took 9.93–12.84 s
  longer on their first long prompt than on their subsequent median. All six
  used private cache directories with `SYCL_CACHE_PERSISTENT=0`, and all wrote
  the same 53 `.l0_cache` files (10,461,608 bytes) in `neo_compiler_cache`.
  SHA256 over sorted relative path, NUL and raw file SHA256 for each entry was
  `12438bf9e0c669cb1a93c09e27fc82b83e51f10677ac7673e1c5e68d094e3414`.
  In each process the same 12 entries (938,752 bytes) have last-write times
  between the first long request's start and first output. The short chat
  warm-up precedes this interval. Align filesystem and client clocks using
  the last request's monotonic end and journal mtime. Moving that offset by
  ±0.5 s or using the observed realtime-minus-monotonic offset retains all
  12 identities. The 15.5–16.6 s span between first and last writes also
  includes the intervening model work. A seed made by an actual short-only
  process can isolate reuse of the remaining entries if its inventory first
  matches the 41 entries seen before the long request.
  The 12 device images have 91 distinct kernel names in their ELF
  `.ze_info`; 90 include Strata, among them GDN recurrence, MoE row gather
  and QSA score GEMM. Image membership establishes no per-kernel duration.
  A read-only comparison of the closed `84c883393` and `568d015ba`
  cohorts found the same 53 names and sizes in all 32 private caches. The
  base and selected process in each cohort also matched the full SHA256
  above and the `config.file` size of 10,461,608. The selected 65537
  and 117489 cases wrote the same 12 image names during their first long
  requests, using the journal alignment and its ±0.5 s check. Across the
  16 processes at each length, median first-minus-own-subsequent prompt
  time was 10.768 and 10.522 s, with ranges of 8.721–13.408 and
  8.896–13.942 s. This makes the shorter measured workload a candidate
  for producing a complete seed on this exact engine and provider.
  The retained `fde0cdd11` executable embeds the three native ELF images
  from its linked Xe2 object (recorded SHA256 prefix `f89830ee88`). Its 39
  named kernel text sections have zero exact name overlap with the 91
  in the 12 first-long cache images. The present AOT object therefore
  provides no exact named coverage for this cache set. Module selection
  and compile time still require runtime evidence.
  The timed provider manifest identifies Level Zero GPU 1.17.39758 (native
  library SHA256 prefix `52e29a60`) and IGC 2.41.5 (`40badd8f`).
  The installed NEO package is 26.35.39758.17-1~24.04~ppa1. Its matching
  upstream release disables the cache if an explicit directory does not
  exist. For a restart comparison, pre-create different empty private
  directories, or populate them from whole completed cache snapshots including
  `config.file`, anew for each process. Set `NEO_CACHE_PERSISTENT=1` and
  `NEO_CACHE_DIR` before exec in every arm. Verify hits in a separate
  `NEO_CACHE_STATS=1` run; keep stats off in clean samples. Time readiness,
  first long-prompt TTFT, prompt time and a subsequent fresh request separately.
  Keep SYCL disk caching off; the older image encoder failed when it was
  enabled. Cache creation alone does not attribute latency to compilation.
  [Release cache implementation](https://github.com/intel/compute-runtime/blob/c9fe9ecbe5c137657c34f2ae0bea35123afe35ac/shared/source/compiler_interface/default_cache_config.cpp),
  [NEO cache format](https://github.com/intel/compute-runtime/blob/c9fe9ecbe5c137657c34f2ae0bea35123afe35ac/programmers-guide/COMPILER_CACHE.md),
  [SYCL cache design](https://github.com/intel/llvm/blob/fbccb466cc122c294a6c746dcf0d37e45ff70f3c/sycl/doc/design/KernelProgramCache.md),
  [earlier Strata cache experiments](SYCL_IMPLEMENTATION.md).
  On the selected 117489-token arm, `bench/sycl/prefill_driver_cache.py`
  takes a whole short-warm-up cache from the producer before the first long
  request, and its full cache after exit. Three separate diagnostic processes
  compare driver hits and misses with empty, short and full seeds. If both
  effects are observed, time the three seeds in three balanced blocks, with
  a fresh process and two fresh long prompts per case. Compare seed preparation,
  server readiness, short warm-up, first long prefill and TTFT, and subsequent
  prefill and TTFT separately. A clock favors a seed when at least two of three
  paired times improve and their median ratio is below one. For readiness use
  the candidate's upper and reference's lower observation bounds.
  The three diagnostic processes at `42466f2ee` pass the cache-hit gate.
  Before and during the first long prompt, the empty cache misses 41 and
  12 images. The short seed hits the first 41 and misses the next 12. The
  full seed hits all 53. First client TTFT in these three processes is
  130.091, 127.958 and 126.286 s. The repeated clean timing blocks still
  determine the size and reproducibility of the effect.
  In the first clean block, empty/short/full first-long TTFT is
  128.110/127.393/126.936 s and subsequent TTFT is
  117.096/116.198/117.558 s. All six requests read 117489 fresh tokens and
  match output and draft counts. The 1.174 s first-long TTFT saving with
  the full seed is one block of the three-block decision.
  The eighth process refused at prelaunch before opening a native owner.
  The only differing control was `AGENTS.md` from the older gather build.
  All seven completed processes and the original plan are preserved at
  `1acdbbbe7`. Closed gather build admission now recovers its pinned control
  bytes from the original commit and verifies the actual build as before.
  A fresh cache comparison needs a separate committed plan.
  After the 117489-token four-arm comparison closes, use `bench/sycl/prefill_split_length_curve.py` on its
  fastest eligible arm with the same executable and actual 128-slot placement.
  Measure six lengths from 94209 through 117489, plus 98306. At 98305 the
  98304 batched rows fill twelve chunks. One more input row opens a thirteenth.
  On this route the extra chunk requests 48 additional prefix copies totalling
  1,245,708,288 bytes with ordinary KV reload, or four copies totalling
  103,809,024 bytes with aligned reuse. These counts follow from the twelve
  QSA layers reading 98304 preceding cells at 1056 bytes per cell, with only
  the final layer reloading when reuse is enabled.
  Take each length in two independent processes in reverse orders, with the
  first request and the median of two subsequent fresh requests separate.
  Compare affine, quadratic and n log n fits to prefill time and client TTFT
  by leaving out a whole length. Hold 98306 outside the smooth fits and
  report its change from 98305 with the actual expert jobs and transfer counts.
  A time difference across these two bodies alone cannot isolate chunk setup.
  `bench/sycl/prefill_split_generation.py` compares the baseline and the selected
  KV policy over three independent pairs (AB, BA, AB). Keep the stock decode
  layout, thinking, MTP and native binary fixed.
  Each process runs the official long, agent and two sustained generation
  workloads. Bind fresh prompt identities, executed gather and KV work,
  generation lengths and MTP counts before comparing sustained decode and
  agent follow-up rates. The standard sustained prompts have about 33K actual
  input tokens and exercise the ordinary route. Check split handoff with a
  separate matched `split-v1-96k` comparison in
  `bench/sycl/prefill_split_continuation.py`. Three pairs keep the stock decode
  settings and scalar gather fixed. Check count-preflight bodies, actual
  128-slot admission, split tail guard/drain, MTP restore and same-request
  decode footers. The code and prose workloads each need at least 1024 output
  tokens with thinking. Judge the MTP classes from measured acceptance and
  report warmup and the two long requests in their original order. Physical
  context fill and RESTORE keep their separate qualification.
  For a fresh request with `n-1` batched tokens, let
  `c = ceil((n-1)/8192)`. An ordinary attention layer reloads
  `1056 * 8192 * c*(c-1)/2` bytes of prefix: every chunk rereads all
  its predecessors. This term grows in steps even when the tensor work
  grows smoothly with input length. Multiply by twelve for ordinary
  reload and by one for aligned reuse on this MTP route. At 253919 tokens
  (31 chunks), `116a0e295` observed the control's 48,271,196,160 bytes
  and 3,217,648,896 bytes of separate D2H publication. At `924f752c7`
  aligned reuse matched the predicted 4,022,599,680 H2D bytes and
  reduced the copy count from 1440 to 120. The same D2H publication
  continued. The difference of 44,248,596,480 requested bytes corresponds
  to 6.82–7.03 s at the calibration rates. On the restored 8209-token
  suffix both arms requested 6,539,157,504 bytes in 96 copies. All
  three request states and the cache restoration payload matched; both
  processes closed with no fresh fault. The clean request-time effect
  remains unmeasured.
  A separate QSA score-stage experiment reuses the FP16 pooled index keys
  within one SELECT loop. The append for the complete chunk precedes selection
  on the same queue. The ordinary scorer reconverts the growing pooled key
  prefix for each batch of up to 256 queries. With 8192-row chunks, let
  `a=(last_position+1)/4+1` using integer division. Skipping a previous
  batch's `a` live key rows avoids `a*128*(4+2)` logical read/write bytes,
  excluding cache and compression effects. Summed within each chunk over 12
  QSA layers this gives 18,725,658,624 bytes at 65537 fresh tokens and
  60,019,273,728 bytes at 117489. Neither number measures DRAM traffic.
  Each new chunk, layer or restored request starts at zero; the existing
  workspace and sticky validation word remain owned by the in-order queue.
  All original scores, explicit IDs, query conversion, dead-key handling,
  selection mapping, attention, residual and decode placement are preserved.

  Compare the separately pinned source `a29048486` with its cold pack on
  the measured ordinary image via `bench/sycl/prefill_retained_score_keys.py`.
  The fixture uses the parent radix4, score GEMM, query8 and head1 settings.
  Check two source epochs, ragged/equal active extents, zero padding, stale
  keys, guards, invalid extents and sticky status on a small traced sequence
  and a late 256/256/91/1/1-query sequence. Compare every score and selected
  ID across arms. Then time the 2800-row final chunk of the 117489 request
  with eleven score+selection calls, a 65538-word score stride and one final
  drain. Each of three clean processes runs five alternating pairs of twelve
  chunks per arm. Require every interval to exceed 100 ms, timer control at
  most 1%, exact control IDs and lower median paired wall time in all three
  processes before model integration. Stop on a fault, mismatch or owner
  failure. At `328b7589d` all three processes passed. The median of their
  median intervals for twelve chunks fell from 473.720 to 462.839 ms.
  The paired process ratios were 0.97703, 0.97703 and 0.97685. The
  separate correctness sequences matched every score and selected ID.
  Each timed arm matched all selected IDs. The next fixture at `0cfa87d2d`
  appends 2299 cells with the production device batch indexer, snapshots
  its incomplete four-cell block, and appends 14 more cells. After restoring
  the snapshot it appends different keys. Each epoch overwrites the borrowed
  score arena and runs four sliced score and sparse-selection calls on the
  indexer's in-order queue. Require exact scores and IDs against a fresh
  pack, valid guards, and a detectable finite score mismatch when a stale
  extent is deliberately carried across the overwrite. The boundary build
  reuses the measured archive and compiles only the new fixture. Retain
  that archive through the check, then recover its build inputs before
  retirement. This tests the indexer/SELECT boundary. The full server
  transition, final slot, model PP, TTFT and sustained decode remain open.
  The current KV/gather comparison keeps its original executable.

  The independent prefill GDN experiment at `f4e47cd02` moved the decay
  calculation into the gate producer. At 8192 rows the joint producer and
  recurrence interval for sixteen repetitions fell from 291.187 to 287.166 ms.
  The paired process ratios were 0.98615, 0.98642 and 0.98619. The numerical
  fixture and all three timing comparisons passed. Test this change on its
  own before combining it with QSA or KV reuse in a model comparison.
  On the measured `fde0cdd11` source image, applying the score-key and
  decay deltas in either order produces the same shared prefill caller
  (`43d1dfbf35344028f517e0b8ac827941c0d130fdd0112434c4bd5930866422fd`).
  Reversing either patch recovers the other component's measured caller.
  The combined 1171-file source fingerprint is
  `027b2f5b32e267aed193f942f1d7b7e6249340a4c99258f8fd735820875d9b78`
  using the original `a29048486` and `18d0b137e` fixtures. This source
  check leaves compilation, four-way state/dispatch and model speed open.
  `bench/sycl/prefill_score_decay_model.py` prepares the complete model
  from that same measured KV/gather parent. Its ten changed compile inputs
  include the two production changes and the physical-context diagnostic
  at `1919720d9`. The component timing fixtures stay at their independently
  measured commits. The source-only projection at `def2dd8ee` precedes
  the layer counters. Applying the score and GDN counter changes at
  `e88604bca` and `063f2e392` in either order gives the same caller.
  The lookahead at `fa4b9855a` rejects a diagnostic that cannot borrow before
  removing a parked conversation. Including this check, the 1171-file
  model image hash is
  `481eff69cf893772d41ec4c9478af9f991da722df1468db01d287da68e66e71b`.
  Build admission requires both three-process component gains and the real
  indexer append/restore/overwrite check. Compare the four score/decay
  combinations with KV reload and scalar gather fixed. The baseline on
  the new binary must reproduce the old dispatch, state and decoder
  handoff. In both traced and clean samples, explicitly set both selectors
  in every process. After successful launches each layer reports its actual
  packed/reused live key rows or the GDN recurrence rows for each path.
  The controller's `dispatch` check derives expected counts from the
  request positions. At 117488 batched rows, each of 12 attention layers
  processes 459 score slices. Full packing visits 6,756,935 live rows
  per layer and retention visits 244,427. The avoided FP32 reads and FP16
  writes across these layers total 60.019 GB of logical work, excluding
  padding and repeated GEMM reads. Each of the 36 GDN layers must account
  for all 117488 recurrence rows in 15 chunks. These are predictions for
  the upcoming model run; the component evidence remains separate.
  Use `bench/sycl/prefill_score_decay_restore.py` to compare all four
  arms on the measured KV-reload/scalar-gather route. Each process
  starts with 32769 or 253919 fresh tokens, issues the independent
  33-token decode-layout request, then restores the original conversation
  with 8209 new tokens. Check the original state and decoder handoff at
  `b633f7d69`, then compare each candidate with the new binary's control.
  On the 8208 batched rows of the near-capacity continuation, the 33 score
  slices process 2,130,684 live key rows per layer when repacked in full.
  Retaining keys still packs 131,060 because scratch expires at the chunk
  boundary. The short cache-window request must produce no dedicated
  score or decay dispatch. These checks precede the clean timing runs.
  There is a second prefix operation entirely on the GPU. Before ordered-exp
  attention, the pack kernel dequantizes K and V for all live pages and writes
  BF16 operands and bad-scale flags into request scratch. The opt-in
  `STRATA_SYCL_PREFILL_PACK_KV_SUFFIX=1` retains complete packed pages only
  across consecutive chunks of the same owned layer. The first chunk after
  entry or RESTORE packs the full extent. A partial page is always rewritten
  with both heads and zero padding. The final callback and external taps use
  the original full pack. This switch can be compared with KV H2D reuse off.
  At 65537 and 117489 fresh input tokens, the 11 eligible layers could skip
  5,046,272 and 18,923,520 head-rows. Each row writes 1028 bytes of packed
  K/V and flags. This would remove 5.19 and 19.45 GB of logical device writes;
  it does not lower the attention consumer's reads. The 14.38-second 64K
  kernel interval measured the consumer. Time the producer and clean request
  separately before assigning a gain to this change.
  Use `bench/sycl/prefill_packed_suffix.py` to compile the caller and run
  `sycl_qsa_bf16_prefix_test --packed-suffix` with the coherent model's
  compiler and runtime. Its fixture replaces one kernel archive member and
  checks both full and suffix work against the same executable. Compare all K/V words, flags, padding, output and
  guards against a full pack at page size four, with invalid scales and
  aligned and nonaligned starts. Check source refusal on invalid retained
  extents. Then compare full/suffix with scalar gather and KV reload fixed,
  using layer-major native16, owned staging (`STRATA_KV_STAGE_OWN=1`) and
  the same copy/compute queue (`STRATA_SYCL_PREFILL_SAME_QUEUE=1`). Confirm
  that the queue and stage actually selected meet the ownership check. With
  the reload policy fixed, subsequent chunks reload the corresponding host
  KV prefix. Packed reuse depends on those earlier host cells staying
  unchanged during the same layer pass. Require nonzero retained pages on
  eligible multi-chunk layers, the expected written/retained counts, equal
  model work, unchanged restored state, cancellation and the physical final slot.
  Collect the clean 64K/117K first and warmed prompt times and TTFT in
  alternating order and recheck sustained decode before adoption.
  Before timing, compare device and host state on a fresh prompt, a nonzero
  restored prefix, interrupted/restarted prefill, the tail and a physical
  262144 fill. Check the external-tap fallback and decode handoff. Time
  this change with the original attention kernel and MoE gather in both
  arms. Neither traffic savings nor a component pass qualifies model PP.
  The isolated change at `fea0f3c79` passed the SYCL compiler's syntax
  check against the measured 117K translation unit using all 798 original
  includes (`15c02eabc`). This check generated no model binary.
  All four ordinary input/output-copy combinations passed the standalone
  262144-token capacity check at `0f5e11521`. Each processed 262143
  batched rows, consumed the physical final slot and returned zero output
  tokens. All four native processes exited normally without a new Xe fault.
  Positions in each 48-layer trace account for 1488 KV-prefix H2D copy
  requests (51,489,275,904 bytes) and 1632 host-mirror D2H copies
  (3,321,876,096 bytes). The prefix count uses the executed positions and
  original reload path. Bus service and overlap were not measured.
  The 253919-token mixed-host path follows a different planner branch.
  The standalone ordinary copy control at `9c94b7206` filled all 262144
  physical slots. Of its 262143 batched rows, 114688 stayed on device
  and 147455 used a 3,019,878,400-byte BF16 host residual bank. Each of
  47 layer boundaries writes that host portion and reads it back. The
  corresponding copy requests total 141,934,284,800 bytes in each direction.
  This is a calculation from the admitted split and the executed BF16
  copy path. The measured 539.401 seconds include diagnostics; they do
  not isolate transfer time or qualify throughput.
  A separate scheduling experiment could run all 48 layers on successive
  chunk-aligned token ranges when the ordinary placement spills residual
  rows. Compare the avoided residual transfers with repeated dense and
  expert weight uploads and prefix processing. Check actual per-range
  placement, predecessor tokens, KV and recurrent state, ordered callbacks,
  cancellation and the final slot before timing. Keep this experiment
  independent of the two-switch KV and MoE gather comparison.
  The isolated candidate uses `STRATA_SYCL_PREFILL_SPLIT_HOST_RESIDUAL=1`.
  It checks actual host spill after the ordinary placement decision and
  before allocating the request workspace. It runs up to twelve original
  8192-row chunks through all layers per sweep. A would-be final sweep of
  at most one chunk receives two preceding chunks. At 262143 batched rows
  the resulting range sizes are 98304, 98304 and 65535. Each is admitted
  again against current VRAM, including the prefix through its absolute
  end. Short inputs, explicitly set GPU row counts, tapped/checkpointed
  requests and whole-pass command/layer profilers follow their original route.
  Use an isolated coherent build for the comparison; both the gather and
  KV-reload changes stay out of it. First check the host planner, SYCL
  translation unit and actual unsplit dispatch. Trace fresh, restored and
  cancelled requests across the cut, the final slot and the decode handoff.
  Check actual residual placement, expert bank size, uploads, chunk order,
  canonical KV/GDN/PLE and output/draft state. A completed sweep does not
  authorize a retry on partially updated state. After those checks, run
  thinking-on fresh 253919-token prompts in four independent process pairs,
  ordered AB, BA, BA, AB (A ordinary, B split), with first use and subsequent
  requests separate. Use the same bodies, allocation policy and decode setup.
  Require a lower median paired warmed prefill time and improvement in at
  least three pairs. Record TTFT and reconnect the sustained decode checks
  before selecting the route. Stop on a state, placement, driver or ownership
  error. Evaluate the 64K/117K KV and gather work independently.
  Source `0e7cf8107` passed the isolated compiler and host check at
  `cfe0ebd08`. The measured ordinary prefill translation unit compiled
  in syntax-only mode against 799 verified inputs. The host planner
  executable passed the existing lease/order tests and every input length
  from 98305 through 262144 using 267 verified compiler inputs. This
  leaves actual sweep dispatch, model state and timing open.
  The clean comparison at `9c03b44d0` completed three independent
  processes with the same ordinary copy route, decoder placement, 262144
  context and four historical thinking-on requests. The last three prompt
  times in each process gave median rates of 946.48, 945.88 and 929.10
  tok/s. Across processes, the median was 945.88 tok/s, 124.211 s of
  prefill and 124.485 s of client TTFT. First use had a median of
  869.91 tok/s and 135.298 s TTFT. At the warmed median, reaching 1000
  tok/s requires saving 6.722 s per request. The 64K copy control reached
  888.26 tok/s with a different bank size. Treat these as two route-specific
  observations when characterizing length dependence. The short outputs
  in this cohort leave sustained decode to its own workload.

- The optional `STRATA_SYCL_PREFILL_READ_ATTENTION_BANK=1` candidate reads
  the previous layer's resident BF16 rows directly in the native attention
  norm, mix and residual write. Layer zero, the PLE bridge at layer one when
  active, host-backed rows and non-window pieces keep their existing
  inputs. Native16 admission rejects the legacy float boundary and phase
  audits before window execution. Decode uses its existing route. The window's
  checked queue drain still precedes bank reuse and callbacks. A window
  can end with a partial second device span. After the attention stores
  drain, the window FFN updates the bank in place and drains before
  `complete_piece` advances the layer cursor. The final layer restores
  each completed BF16 span into the float callback buffer.
  For 65536 batched rows, each eligible layer avoids 1.25 GiB of requested
  D2D input copies. With PLE active, 46 eligible layers would avoid 57.5 GiB.
  Verify actual bytes and placement before using this estimate. This
  candidate has no measured speed or model qualification yet.
  Use the retained ordinary 384-slot configuration after the existing
  comparison closes. First verify BF16 results and guards at 1, 63, 7905,
  8191 and 8192 rows, including partial overlap refusal and both output
  destinations. The same-build fixture also exercises two resident pieces
  of 8 and 3 rows, prior FFN publication, norm/mix, both attention stores,
  final FFN update and callback restoration in all four arms. It uses
  synthetic projections. Actual model dispatch, cancellation and fallback
  still need separate evidence. In the numerical runs verify the unborrowed
  PLE layer, mixed host tail and disabled selector against their copied-input
  controls. Check the existing audit admission refusal separately. Use the
  cache-enabled native verification route for active state hashes: the
  clean benchmark disables the prompt cache and cannot emit that evidence.
  The model controller takes the traced 48-layer input counters from
  `1bf54ec32` at the checked layer drains. It requires zero eligible
  reads through the PLE layer and the expected bytes at every later
  layer. Equal final totals can hide an erroneous earlier read.
  Exercise the four combinations on independent native servers at
  32769 and 253919 initial tokens. After switching to 33 unrelated
  tokens, restore the original and process 8209 fresh tokens, including
  a 16-row tail in the batched suffix. Match active state, publications
  and restored decode-cache payload for every arm. These cases leave
  output room; the physical final slot has its own test.
  Then compare the four combinations of copied/borrowed input
  and copied/direct output on the same coherently rebuilt engine. Check
  per-request routes, numerical fingerprints and both copy counters before
  measuring clean 65537-token requests. Use four fresh-process blocks with
  orders ABCD, BDAC, CADB, DCBA, with A=copied input/copied output,
  B=borrowed input/copied output, C=copied input/direct output and
  D=borrowed input/direct output. Each process makes three exact requests.
  Evaluate input-only (B/A), output-only (C/A), combined (D/A) and
  incremental input (D/C) using paired warmed prompt times. A repeatable
  gain requires all four block ratios below one for the stated contrast.
  Keep first-use timings, client TTFT and target attainment separate.
  Stop on a safety, route, workload, validation or ownership failure.
  Adoption also requires a physical 262144 fill, mixed GPU/host tails,
  RESTORE and repeated matched agent and sustained-decode evidence.

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

  Retain the coherent `14508acbf` decode reference for the next verifier
  comparison. The three earlier reference processes and all six
  `decode_mtp_prefix_generation_v1` processes used the same v5 bodies, read and
  generated counts, draft counts, model assets, boot, driver and runtime
  manifest. Their expert footers report 1,074,995 CPU jobs and 1,354,722
  entries for code, and 3,718,375 jobs and 6,125,520 entries for prose in
  every process. Submitted expert-miss blobs are zero at `--pcie-frac 0`.
  The coherent reference runs the two long requests directly after four
  warmups and enables Arena copy and the snapshot marker. The prefix suite
  uses a different native revision, adds chat, prefill and agent requests
  before long generation, and disables both options. Both use effective
  MTP width four. Match the preceding work and phase settings when selecting
  the decode reference. The verifier candidate has its own context checks.
  On `14508acbf`, the Arena-copy view is acquired only when the effective
  PCIe share is positive. The two suites report zero expert-miss copies at
  share zero with early PCIe disabled. The snapshot setting prints the
  draft-prefill marker and checks RESTORE when used. The measured requests
  start fresh with no RESTORE.
  The full-suite replay `decode_reference_replay_v1` closed at `097922a15`
  with identical preceding requests, sustained work and MTP classes across
  both cohorts. Each replay process exited normally with no fresh GPU fault.
  The three-process medians were 11.394 versus 11.384 tok/s for code and
  6.898 versus 6.867 for prose. Both meet the committed development-reference
  screen. Retain coherent `14508acbf` for the next decode comparison.
  The collections used different measurement periods, and the unchanged
  replay build ranged from 10.468 to 11.534 tok/s on code. Use the decision
  for reference selection. The MTP prefix build has ended its timing role.
  Check its source recovery and remaining executable readers before retirement.

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
  all27individual samples remain in the CPUresult. The performance CSV
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
  is not sustainedTG/cleanPP or adoption. Its separate repeated bench.sh
  comparison and rejection are recorded below.
  The frozen CPU mixed-cache comparison now closes all six fresh bench.sh
  processes and twelve thinking-on long requests,normal0/closed owners,quiet
  validation/no fresh fault and all15final asset hashes. Code/prose read33211/
  33281tokens and generate2815/6823with strong.870085/weak.339347acceptance;
  requests,outputs and logical work match across arms.
  Charged reference/candidate TGmedians are
  12.61910/12.46926for code and
  7.58449/7.60636for prose.
  Candidate/reference paired TGis
  0.823831/1.006318/1.015203and
  0.911901/1.007435/1.009865;
  geometric means0.944152/0.975310.
  Both original reproducibility gates fail:cache0remains selected. Favorable later
  pairs and individual samples remain;no changed criteria or candidate adoption.
  Existing affected-format pool clocks improve in the two later pairs,but are
  not pure CPUarithmetic and do not establish hardware saturation or clean
  whole-model gain. Reference0format times were not retained;none are invented.
  The current-runtime fixture and direct/active-from-exec controls now pass
  at`c95d4670c`:665actual fixture compiler inputs,the current archive/three
  runtime objects and13retainedELFs are bound without an engine rebuild. All
  five native gates,guarded arithmetic,normal0/closed owners,quiet validation
  and no fresh fault pass in both modes. Active collection retains inner6
  kernels/3copies and whole8/5 with unique native joins. V1's shared test-mock
  failure remains failed;V2passes all nine host controls. The old fixture's
  source-mismatch refusal is preserved. The current393/e8cache0 profile now
  closes three bracketed processes at`d2016f730`:all native requests/owners and
  fault checks pass,but both decode observer ratios1.57633/1.53794 exceed the
  prospective1.5bound. The observer gate remains failed;these kernel services
  do not establish critical-path shares,hardware saturation or clean speed.
  Caller accounting includes substantial GPUcompletion polling. Scalar mapped
  row copies and GPU KVresolution are bounded mechanism hypotheses,not a
  conclusion that PCIe or CPUarithmetic is saturated. Successful raw profile
  output and the superseded V2fixture are retired at`dc3df2ef2`;compact original
  samples and the failed decision remain in`current_decode_device_service_v1`.
  The fresh mapped-row fixture now closes at`84bdffd85`:five coherent TUs,
  715compiler inputs/24external links,ten host controls,18captures/54bitwise
  replays/five refusals and six independent processes/432eligible samples.
  All processes finish0with closed owners,quiet validation and no fresh fault.
  Minimum sample is469.673ms,warmup501.188ms and calibration752.580ms;maximum
  clock-pair fraction0.000297600. Production dispatch stays unchanged.
  For20rows with five hits and fresh host production,the three paired C/R
  intervals are0.989908/0.859729/0.973510,geometric mean0.939216:the declared
  component screen passes. The10/30/40row screens fail their original
  all-three-pairs gate,including favorable10/30row geometric means;do not
  erase those outcomes or promote them from their mean alone. All individual
  samples and auxiliary cells remain under`mapped_rows_subgroup_v5`;four
  component CSVpoints link the original evidence and actual8b40f5c84source.
  This is no model PP/TG gain or physical-resource saturation claim.
  Source and original caller counters establish that all six retained32K
  attribution requests used8target windows/29rows. The first window isT1 and
  the maximum isT4,so each remaining window isT4;none exercises the qualifying
  T2/20-row shape. Do not wire it into the measured hot path on that evidence.
  The explicit MTPwindow2/4 comparison in the existing clean14508engine used
  matched long thinking/strong-and-weak-MTP workloads. Its`decode_window_speed_v3`controller
  keeps allocation spec4and the prefill cache loan in both arms;only existing
  --mtp-max-t0/2changes the actual MTPwindow. Its plan and both frontend probes
  pass with closed owners. Cap/base,copy/draft,two positive/six negative phase
  controls and the committed actual V2prompt/decode fixture pass. Its first
  candidate now stops at the prose end gate;the original cohort remains failed.
  The plan declares six fresh processes C,R,R,C,C,R,four official warmups
  and both long thinking workloads per process;no V1sample is reused. Native
  source stays14508;the controller independently recovers its original
  inputs/recipe and checks the retained12ELFs,SDK/runtime and all15
  model assets. Today's native checkout is not substituted for that build.
  V1's frontend probes and window/copy/draft host controls pass with closed
  owners. Its reference process closes normally0,
  with quiet validation,closed owners,no fresh fault and actual spec4argv.
  Code/prose have33213/33283fresh inputs,3548/5485outputs,TG14.0220/8.82839,
  and observed acceptance80.3074%/34.2952%. Both reference classes and long
  thinking/answer gates pass. V1then stops before candidate warmups:changing
  global spec to2triggers the existing cache-scratch admission refusal. Actual
  native exit is2,with no fresh fault;all four owned processes and their session
  are now absent. The original incomplete owner observation and failed trial
  remain failed,with separate later closure evidence. These are
  new fixed-id workload samples,not a matched regression comparison to the
  earlier id. Consumed V1raw output is now retired after original controller,
  native source/recipe and actual input recovery;compact samples/refusal and
  original failed status remain. V1does not become a successful comparison.
  V2passes its host gates and four real candidate short warmups,with actual
  prompt captures1/4and decode captures1/2. Its overly broad observer rejects
  the prompt4graph at the first long begin gate;no long timing starts. Native
  Source14508`src/core/verify.cpp:1585`prints prompt/decode explicitly. All
  owned processes are now absent,no fresh fault occurred,and the original
  failed observation stays failed with separate later closure evidence.
  Its consumed raw output is retired after original source/recipe recovery;
  the compact actual marker fixture remains for V3host admission.
  V3allows prompt widths1..4and requires actual selected2/4decode captures,
  checking the observed V2marker fixture before fresh model execution. No
  performance/length/thinking criterion changes. Code completes3112outputs
  at12.2716TG and90.625%acceptance. Prose spans two native calls:33283input/
  12288output and45589input/2432output;their14720native outputs differ from
  the server's14738reported tokens by18explicit server-inserted reasoning
  wrap-up IDs,as the bounded Luna review now confirms. The controller's one-native-call
  boundary assumption rejects the extra footer before the client journals its
  prose result. All owned processes are now absent,no fresh fault occurred,
  and the failed whole trial stays failed. Preserve the compact physical-call
  samples and code APIresult;no API aggregate,repeated gain or quality claim.
  The consumed V3raw output and review transaction are retired after exact
  751controller-file and original native source/recipe recovery. Before another
  cohort,bind each APIinterval to its ordered native segments and journal the
  APIsample before end-gate approval. Preserve the failed outcome;Main proceeds
  with the KVplanner component without another long MTPcohort now.
  Judge reference
  strong/weak acceptance and report candidate acceptance independently,because
  changing window width can change the observed rate. No useful T2model route
  qualified in that failed configuration comparison. The unused row-copy
  source/fixture/controller and ended13-ELF observer consumers are now retired
  to`644412c8e`;eight unchanged build/admission helpers remain in
  `tools/sycl/fixture_build.py`for the active KVwork. The original89401/e8
  source,recipes and final compiler/link inputs are now recovered and their
  ended binaries retired with the other normal superseded generations below.
  Original unexecuted V1preparation and V2protocol/V3API-name/V4timer failures
  stay in Git at245eecc93/bf67e3a0b/67d28b364/8b40f5c84. V5increases warmup/
  calibration rather than relaxing the250ms/1%bounds. No prior failure is
  retroactively qualified. Consumed normal logs and superseded V1..4fixture
  products are now retired after exact original-source/recipe/input recovery:
  134files/7432029regular bytes. Current component admission and all retained
  engine/fixture guards pass after deletion. Four obsolete active result copies
  are retired to the exact history references above;current samples stay intact.
  The closed CPUcache/whole-job and mapped-row prototypes'46source/controller
  files(724140bytes)are retired to`644412c8e`. CPUdispatch/pool/generator and
  original elementwise processing are restored from clean14508,while the
  independent native-job MAXTguard and its regression test remain. Clean14508
  remains the speed reference. No candidate clean-speed,
  dedicated-prefill,agent/task,physical262144/RESTORE or ordinary-provider
  teardown qualification follows. See mixed_nt_model_v1.
  The separate cooperative KVplanner has component qualification and an
  opt-in decode hook from`644412c8e`;default selection stays scalar. Its
  standalone timing fixture now builds from`79ed58840`:five fresh TUs,718compiler
  inputs/24external links,zero uncovered/reused objects and36passing host
  controls. Binary`1e7765b60`first passes14bounded semantics and20traced graph
  replays,then795direct oracle records,60graph replays and15physical-tail
  replays across three KVformats,plus12timing-case records with two exact
  replays each after map-only resets. Both owned processes exit0with quiet
  validation,closed ownership and no fresh fault;all34actual providers match.
  This is component numerical evidence,no model performance or physical262144
  qualification. One256-work-item group prefetches selection/table data and marks
  hits atomically;the leader preserves exact query-order miss discovery and
  the existing clock policy. It adds2KiB local scratch,no persistent allocation
  or CPUround trip. Existing serial selection/clock statements are preserved.
  The observed161.75/162.20ms service in the failed-overhead profile motivates
  a bounded component question,not a hardware-saturation or model-gain claim.
  Provenance is the owned1e196a13SYCL policy,Niko's MITf3925cf0CUDA parallel-hit
  design,and the Khronos [group-barrier](https://github.khronos.org/SYCL_Reference/iface/group-functions.html)
  and [atomic-ref](https://github.khronos.org/SYCL_Reference/iface/atomic_ref.html)
  contracts. The reference LICENSEbytes match this repository. The maintained
  owned`kv_copy_groups_bench.cpp --planner-validate`extension's independent
  CPUoracle covers all map/control/miss and payload
  bytes,three KVformats and copy geometries. New cases cover invalid gaps
  across256items and dynamic four-query/zero-width/epoch-wrap/invalid-width
  graph replays at the262143last cell. The target compiles its kernel and the
  three runtime TUs directly,without an engine/library rebuild. The committed
  recipe/controller recovers the original14508foundation and preserves its
  twelve ELFs. See`kv_planner_speed_v1`for the exact checks and limits;the prior
  numerical fixture remains recoverable from`945dea2ee`and its committed result.
  Its old build is retired after exact source/recipe/compiler/link recovery;
  the timing fixture was retained through the immediate comparison. That
  standalone generation is now retired after1222project/291ggml/906TLA source,
  718compiler/24link-input,five-object and original recipe recovery:
  37files/2823683bytes. All25fallback/candidate native ELFs are unchanged.
  The separate paired graph-time plan is committed for2/4-query INT8shapes,
  5120resident pages,full cache and zero/quarter/all selected pages missing.
  It declares R,C,C,R,R,Cfresh processes,500mswarmup,750mscalibration and three
  >=250msisolated-interval sums per cell;map resets and oracle work are outside
  timing. All three miss cells must improve in every paired process comparison
  for that width to qualify for a separate model trial. Parameter validation
  remains charged equally;API/device traces,events and profilers are absent.
  The six processes now close0with108eligible samples,quiet validation,closed
  owners and no fresh fault. Every cell improves in all three paired process
  comparisons,so both widths2/4pass the declared component gate. Paired
  geometric-mean C/Rtimes are0.388179/0.860978/0.871358forT2and
  0.372557/0.672889/0.806724forT4(zero/quarter/all selected pages missing).
  The shortest batch sum is545.887ms;maximum estimated clock fraction is
  0.000280246. Individual samples and all paired ratios remain in the result.
  The authored hook selects only`VerifyInput::decode`,INT8,65536logical/
  5120resident pages,page4,2KVheads,dim256,cap2051andT2/4. Prompt,T1and
  other shapes keep their existing path. Its pre-exec control is
  `STRATA_SYCL_DECODE_KV_COALESCED=0|1`,default0,with copy geometry0required.
  Captured phase caches retain separate scalar/cooperative call counts and
  print the actual selected calls. The new policy/cache/Verifier fields now
  have a fresh full engine build from`1eec28c77`:engine`330405370`,13ELFs,
  138fresh objects/135unique sources,1471compiler inputs,one generated input
  and29external links,zero uncovered inputs. All five actual class-header
  closures have consistent conditions;no project archive/object was reused.
  All five owned build commands close0,and28compiler-census host controls pass.
  This is build evidence only;no GPU/model execution or speed/adoption follows.
  See`decode_kv_native_v1`in the linked result. Fresh same-build
  `decode_kv_device_v1` now closes all 11 CPU/GPU stages: the six-case native
  MAXT guard, 54 reader controls and the prior owner/policy/admission checks;
  traced submission/phase/actual canonical and packed heads; small/full QSA
  and off/on KV producers, including 262144 cells and final-cell overwrite;
  and GU smoke plus 44 numerical cases. All native children close normally
  with quiet parameter validation, closed ownership and no fresh fault.
  GU has zero FP16 ULP difference and exact BF16 captures. Its successful raw
  traces, transactions and cache are retired: 95 files, 9700246 regular bytes.
  The fresh `decode_kv_model_v1` now passes the thinking-on
  32769/82/32769-input handoff with 32 generated tokens per request. Both long
  handoffs actually capture cooperative decode4 in all 12 QSA layers, with
  zero scalar calls in those captures. Prompt1/4 and decode1 retain 12 scalar
  calls and zero cooperative calls. All 15 model asset hashes match before
  and after; the native child closes normally with quiet parameter validation,
  closed ownership and no fresh fault. Width2 has not been observed in this
  model check. This qualifies the observed route/state handoff, not sustained
  TG, clean PP/TTFT, task/agent, physical model 262144/RESTORE, ordinary teardown
  or default adoption. The API journal now persists its completed response
  before end-gate approval and records pending/passed/failed gates separately.
  A refused gate preserves the response without making the cohort complete.
  Source`5228b1ee7`adds ordered native-segment accounting: the first API prompt,
  each actual read/reuse extent, output/draft totals and summed rounded clocks
  must bind. Server-inserted token lengths must be independently pinned;those
  tokens stay outside native TG. The synthetic historical-count fixture binds
  14720native outputs plus18insertions to14738API tokens,with either a reread
  or reused continuation. The73-test host suite passes with one optional real
  tokenizer fixture skipped. No actual API/GPU speed trial follows from this
  offline test. The prospective retained-build reader at`781268737`also passes
  two positive/five negative source/recipe/census/13-ELF checks, preserving the
  original admission refusal. It recovers original build evidence`6c05d9a4`
  and allows only explicitly pinned uncompiled Python deltas;it does not
  rebuild or repeat GPU qualification. The new controller`e0e916140`now wires
  these checks into`decode_kv_speed_v1`:six fresh flag0/1processes R,C,C,R,R,C,
  each with the official warmups and both thinking-enabled long workloads.
  Its76-test host suite has75passes and one optional-tokenizer skip;all15full
  asset hashes match and both actual benchmark/server frontend probes pass.
  The pinned tokenizer independently gives3close and18wrap-up IDs. The
  comparison now closes all six processes normally with both workloads
  eligible,closed ownership,quiet validation,no fresh fault and all15final
  asset hashes. Code C/R TG ratios are1.067062/1.073229/1.062957 and
  prose1.066171/1.076062/1.067888. All three pairs improve for both confirmed
  MTP classes;paired geometric-mean gains are6.774%and7.003%. Independent
  process median TG is14.059->14.991for code and8.833->9.440for prose.
  Request/output/reasoning hashes and draft counts match across the six
  processes;each API has one native segment and zero inserted tokens. This
  observation does not impose identical output as a quality standard.
  In the second code pair,
  candidate first-use prompt time54.211s exceeds reference52.0236s by4.2%,and
  TTFT55.627s exceeds53.315s by4.3%. The third code pair is also slower by
  0.56%in prompt time and1.06%in TTFT;the prose intervals do not worsen.
  Preserve this unresolved first-use result rather than dismissing it as noise.
  The dedicated first/warmed32K/64K comparison below is now closed,but
  preservation remains unresolved.
  The separate`decode_kv_prefill_v1`plan from`9269a3cef`uses twelve fresh
  processes:three adjacent flag0/1pairs per length,with each middle pair
  reversed. Every process uses three exact24-output thinking-on requests;
  first long GPU use and subsequent2 medians stay separate. TTFT is measured
  after count preflight. All eight first/warmed,length,prompt-time/TTFT median
  series must be no slower for the median screen;all paired ratios remain.
  This screen does not prove zero effect or adopt the model. The same330405
  native13-ELF generation is reused with an explicit new uncompiled observer;
  no native build or runtime-safety changes. Both actual frontend probes and
  two CPU preservation-policy tests pass normally with closed ownership.
  These controls cover ten invalid cohorts and keep first-use,warmed and
  TTFT-only regressions separate;they do not execute a model or establish PP.
  All twelve processes/thirty-six requests now close normally0,with unchanged
  boot/driver,quiet validation,closed ownership and all15final asset hashes.
  Evidence`18a7118b2`keeps the original order and criteria. The32K first/warmed
  prompt-time median C/Rratios are0.992524/0.998343,and TTFT0.992042/0.998041.
  The64K first-use prompt/TTFT ratios are0.994457/0.996003,but warmed prompt
  time66.91395->67.2345s and TTFT67.0715->67.3985s give1.004790/1.004875.
  Thus two of eight median series fail;execution success does not pass the
  preservation screen. All three paired ratios and mixed directions remain.
  Keep the measured decode gain as an eligible candidate,with its default
  flag0and preservation unresolved;no criterion change or extra cohort sample.
  After735original controller-input and native source/recipe/census checks,
  835ended raw/cache/transaction files(126546171regular bytes)are retired.
  The cross-build consumer has now ended;the24config/plan files are retired
  after exact reconstruction,keeping the receipt. All25fallback/candidate
  ELFs and original36samples/failed decision are unchanged.
  This campaign isolates the flag within330405. A matched cross-build
  prefill/TTFT comparison with the retained coherent14508fallback closes
  below with failed preservation;no whole-engine adoption follows. Its older
  `native_gdn_20261018`absolute PP and this campaign's different workload ID
  are not a matched regression comparison.
  The historical reader from`084e7eb2252f2192bd82e2b284d7c7eca414a1bf`is now
  integrated after the735-input cohort closes. Its two source/test files are
  byte-identical to the reviewed child. Both CPUtests pass under Main's
  serialized ownership,with two positive snapshot
  checks and four refusals,1470compiler/29link inputs and twelve original ELFs.
  The compact child result is`coherent_original_reader_v1`at`ad505e6a9`.
  This is historical input/artifact binding,not new GPU/model qualification.
  Original source and test-evidence commits remain through the pushed merge;
  the finished remote/local child and worktree are retired. Its three consumed
  host-output/receipt files(1337bytes)are also retired;the CPUoutcome is unchanged.
  Source`34ea625bca77eb72c9b9e756aaa8fdb09cc976f3`now prepares the separate
  `coherent_generations_prefill_v1`comparison. It uses the same twelve-case
  order,three thinking-on exact24-output requests per process and pinned
  `decode_kv_prefill_20261114`input ID as the closed flag comparison. Both
  generations use KVcoalesced0/copygroups0;only the supervised launch plan's
  native ELF/source changes. Current engine_once/server/benchmark,placement,
  safety and runtime remain identical. Original14508has no KV_PLANfooter;
  its scalar-only frozen source and closed handoff evidence apply,while
  current330405must print actual scalar decode4captures. Preserve first and
  warmed samples and all paired ratios;the same eight-series median screen
  does not adopt inference or revise the old failed flag screen. The plan
  binds both source/recipe/compiler/provider/ELF proofs and739current inputs.
  Both actual benchmark/server frontend probes and four CPUcontrols now pass
  normally with closed ownership. The two new tests bind the native launcher
  plan without bypassing engine_once,keep both environment arms at KV0and
  reject wrong generation/shape/wrapper or hidden settings;the two existing
  reducer tests preserve first/warmed and invalid-cohort refusals. This is
  source`34ea625bc`host evidence,not a rerun of the full native/test suite.
  Evidence`f26560ad7`now closes all twelve processes/thirty-six requests
  normally0,with quiet validation,closed ownership,no freshfault,unchanged
  boot/driver and all15final model-asset hashes. All three pairs per length
  retain first/warmed prompt-time and TTFT samples. The32K first/warmed
  prompt median C/Rratios are0.969932/0.992058;64K gives1.014361/1.004382.
  Corresponding TTFT ratios are0.972084/0.993217and1.014245/1.004250.
  Four of eight median series fail;all three64K paired prompt and TTFT
  directions are slower. Do not dismiss the decrease as noise or adopt the
  whole current generation. Preserve14508as the faster64K owned prefill
  reference and330405for the separately planned CPUcapacity diagnostic;
  its decode-only KVgain and failed same-binary flag screen remain distinct.
  These matched bodies differ from the older`native_gdn_20261018`speed
  reference;their absolute PP cannot establish that historical regression.
  No third engine is built. After recovering734/735/739original controller
  inputs from Git and reproducing all60generated configuration/launcher
  files,the three ended cohorts retire896files/126435673regular bytes and
  their469184-byte temporary manifest. Each root now retains only its record;
  all25ELFs,samples and original decisions are unchanged. The historical
  admission CLIs remain archive-only;the next CPUquestion consumes compact
  records. Physical262144/RESTORE,agent/task and ordinary runtime teardown
  remain unqualified;three legacy build-provenance holds are separate.
  The default flag stays0;no candidate adoption follows. Broader adoption
  gates remain open. After734original controller-input and native recipe/census
  checks,the six-process cohort retires424successful raw/cache files
  (78601580bytes)and its75376-byte temporary manifest. Its final12config/plan
  files and ended host-attempt receipt are also retired above;their hashes,
  both engines,individual samples and original outcomes remain recorded. See
  `long_generation_segment_metrology_v1`in the linked result. The model's
  successful raw output/cache is retired after source/recipe recovery:
  58 files, 11231037 regular bytes. Individual request/capture evidence and
  conservation/format/upload totals remain; 256 unused histogram bins are
  omitted from the maintained result. The corrected
  `decode_kv_policy_host_v2`check from`ba7715d00`passes all four CPUcommands:
  actual policy compilation,actual SYCL Verifier syntax,link and fixture
  execution,with874compiler/23link inputs and closed ownership. V1's incorrect
  success-marker gate remains failed. This is no engine build,GPU/model or
  ODRqualification. Both temporary host products and eight ended engine/fixture
  groups are now retired after exact Git source/recipe/compiler/link recovery:
  1210files/970017769regular bytes. Original source receipts and still-consumed
  SDK/census metadata remain;the known-good14508engine is unchanged and the
  separate1e776timing fixture is now retired as recorded above. The partial
  GU,adaptive audit and old DFlash2special-format provenance holds are also
  reconciled and retired,as detailed below and in
  `production_preparation.build_retirement.legacy_retirement`. Among these
  reviewed engine paths,only the validated reference and current candidate
  retain executable generations;no global cleanup is claimed.
  The fresh native/model handoff stages above are closed;the
  separate repeated decode speed comparison and dedicated prefill comparison
  are closed,as is the separate cross-build comparison;both failed
  preservation results remain unresolved. The component alone did not qualify model TG;the
  completed model campaign above supplies the workload-scoped gain.
  Actual request overlap/hit distribution and physical bus traffic remain
  unknown. The161ms observer service is still
  only a mechanism hypothesis;the limiting physical resource remains unknown.
  The source-bound`decode_resource_demand_v1`derives the current workload's
  CPUdemand from closed`4ee20baa4`counters and the pinned48-layer expert layout.
  CPUentries per generated token are499.085for code and892.989for prose,
  including verified drafts;VRAMlookup fractions are11.427%and5.630%.
  At70TG,the same work distribution needs171.717/307.245dense-equivalent
  GMAC/s on the CPU. Assuming one whole expert-blob traversal per distinct
  CPUjob,the layer-size bounds give40.717..71.773/56.742..100.020GB/s.
  These are logical demand estimates,not measured DRAM traffic,ISAthroughput
  or a hardware ceiling;per-layer job weights and cache reuse are unresolved.
  A bounded native Luna-max review of four existing reports confirms that
  their older source/service data do not qualify current14508/1eec resource
  limits. The clean5-versus1pool-worker comparison closes with three
  independent processes per arm;it tests scheduling/critical-path
  contribution,not arithmetic versus DRAM. In the current1eec native layout,
  `expert_pool_dispatch`also sends T1through`expert_pool_dispatch_multi`.
  `run_split_multi_native`uses`3*(workers+host)`row tasks for each GU/Down
  phase,with host-side hidden quantization between them. Thus the completed
  5/1worker comparison retains host participation and has six/two potential
  compute participants;it does not exercise the canonical single-worker
  inline path. Record the actual startup worker footer and argv;changed row
  partitions/task counts prevent interpreting it as pure memory scaling.
  Source`85fb654ba4db67eef946c605a9aaaeee6230d307`and plan`72b7ea2bc`
  define`cpu_pool_capacity_v1`on the same330405engine,with scalar
  KV0/copy0in both arms. Six fresh processes use5,1,1,5,5,1workers;each runs
  the two thinking-on long-generation bodies from`decode_kv_20261114`.
  Both actual frontend probes and three CPUcontrols pass normally with
  closed ownership(no skips;unit0.003s). They check worker-only settings,
  required actual pool footer,paired directions/work conservation and
  invalid-cohort refusals. All six processes close normally0with quiet
  validation,no fresh fault,15matched model assets and absent/reaped owners.
  Evidence`1325d67fe`preserves every individual sample. For code/prose,
  5worker+host TGmedians are12.210823/7.617721with ranges7.563689..12.237382
  and4.732473..7.718794;1worker+host medians are7.983899/4.985458with ranges
  7.143063..8.003741and4.874395..5.042066. MTPacceptance stays0.803074/0.342952,
  with3548/5485native outputs and matching request hashes and CPUentries/jobs
  in every process. The three paired1/5TG ratios are0.652419,0.584978,1.058180
  for code and0.631497,0.654455,1.065419for prose. Both directions are mixed:
  the original all-three gate is inconclusive,despite geometric means0.739166/
  0.760777. Retain the slowed third reference;do not discard it as noise or
  claim CPUarithmetic,DRAM or scheduling saturation. Four CSVrows index the
  two worker counts and two bodies. This is no native rebuild,speedup or
  inference adoption,and does not revise either failed prefill screen.
  The741-source cohort froze AGENTS.md and the retention policy throughout.
  Its source/policy hashes remain unchanged in the original receipt. After
  normal closure,merge`38ecca0ee`integrates original`6b6098593`into the primary:
  phase ownership,naming and smallest-coherent-unit rules now live in AGENTS.
  The source-only child`tune/phase-kv-isolation`prepares the next two units
  without changing this cohort's primary tree. Its policy commit`6b6098593`
  carries those AGENTS rules. Unit`ae74403d1`only renames the decode KVentry
  and phase selector,with required callers/header/policy tests. The earlier
  combined`93b964cae`proposal is now split further before any measurement:
  current child`97ed15507`selects B1,the cooperative TU extraction/CMake wiring
  only. It keeps Alpha's scalar/helper source byte-exact except for removing
  the372-byte public decode wrapper now defined by the new TU
  (scalar-file SHA34418f3b0b5ae8c2b4e0a217d54413197f60246c9e76299a22298619300bde57).
  Removing the old definition,adding its independently compiled replacement
  and linking both production/fixture targets form one required link unit.
  B2,the scalar14508restoration,is a separate pending target;its original bytes
  remain recoverable with SHA4473c66af0c24663534fe2ca01e56cdb41f96c3aaae7de61eabeda8d25294696.
  Do not measure its whole-file restoration as one unexplained change:after B1,
  independently compare removal of the unreachable cooperative branch,restoring
  the victim-selection body inside the scalar kernel,and removal of its extra
  scalar adapter,then their combination. Commit the exact unit plans first.
  Scalar`qsa_kv_resolve`still serves Verifier fallback/width1as well as prefill;
  B1separates only cooperative decode. Preserve that decode scalar path through
  an explicit independent selection before changing prefill scalar processing;
  B1alone does not establish complete phase separation.
  Source-only child`73d88aa12`prepares that next independent unit,B1s,against
  B1/`25b9c5113`:decode scalar/fallback entry,explicit prompt/decode selector,
  required declarations/caller and same-fixture oracle/graph modes. Its body
  is extracted from owned1eec scalar code;prefill scalar bytes remain SHA34418f3b0b5ae8c2b4e0a217d54413197f60246c9e76299a22298619300bde57.
  B1sremains unbuilt/unqualified and must be compared after B1,not folded into
  B1's build or results. The first serialized host-only syntax screen closes
  normally0with reaped empty ownership and all five compile-time phase controls:
  `/usr/bin/g++ -std=c++17 -Iinclude -fsyntax-only tests/core/verify_window_policy_test.cpp`
  in that exact child revision,with GCC13and no generated ELF. It cannot
  qualify SYCL compilation,linkage,actual routes or
  speed;fresh bounded native/oracle/model and both-phase comparisons remain required.
  B1scovers Verifier's scalar fallback,not the separate single-token
  Layer/Session path. That legacy caller still reaches`qsa_kv_resolve`and
  must remain unchanged for a prompt-only scalar candidate. Source review AX
  corrects the earlier generic graph-route proposal:the measured`--serve`
  `--spec4`route uses Verifier,including width1decode and short prompt windows;
  long batched prefill uses separate append kernels. Do not advance a shared
  scalar-body restoration as its long-prefill arithmetic optimization. Any
  future short-prompt scalar candidate needs independent Verifier prompt
  selection while preserving decode and legacy Layer code/capture/configuration.
  The active width4cooperative TU remains reachable;only a separately proved
  dead branch after B1could be removed. High-level checkpoint/restore sources
  confirm prevalidation and transfer-failure fail-stop;post-restore readback
  remains optional. These source facts neither explain nor erase the original
  measured PPdifference or qualify a runtime/model route.
  The new decode source is extracted from the owned1eec implementation;
  bounds,barriers,ordered clock/copy policy,parameters and persistent storage
  are unchanged. The independent expert-dispatch bounds fix is preserved.
  Naming and B1source are merged into the primary with their original identities
  through`25b9c5113`(merge`07c323f9d`). B1sremains held in original73d88history and B2is
  not applied. Source integration is not an adopted model route.
  Child`25b9c5113`prepares the same builder's explicit`split`selection for B1,
  with fresh production/fixture objects and separate scalar-restoration exclusion.
  Its five source-pinned host controls pass normally0with reaped empty ownership;
  they test unit selection and refusal of an unfinished/failed naming gate,
  not the new kernel. Building B1still requires both original Aphase controls
  and retirement of the old1eecgeneration after exact source/recipe recovery.
  Main's first real B1origin check finds that its whole Git delta also includes
  three uncompiled Python additions;the original guard refuses before building.
  Prospective`94799047b`scopes this delta to native/source prefixes while keeping
  the complete native-file set and exact97edbytes checks. Its six host tests
  execute normally0,but Main's outer result aggregation fails after exit by
  treating Owner's returned tuple as an entry. Preserve that failed outer
  receipt and the separately verified unchanged-input,closed six-test execution;
  do not repeat or relabel the original failure. Independent source/recipe/build
  admission remains required and no native/GPUqualification is inherited.
  Source`0569e6924`passes exact871-native-input B1admission and pins the fresh
  build. Configuration closes normally0with28compiler-census host controls;
  the planned142objects/137unique TUs include two new decode-TUobjects for
  production and the KVfixture. Commit`0208f89eb`records configured inputs
  before compilation. All five configure/compile/census commands now close
  normally0with reaped empty ownership. All14fresh ELFs use142objects/
  137unique TUs,1474compiler inputs,one generated input and29external link
  inputs,with zero uncovered files and consistent definitions across five
  headers. Engine SHA2f83ee16d7286f83c3278ec23e342121a71a0e20b0eccd49fb4a0bfd301a67b8
  comes from0569e6924;the original build receipt SHAis
  9807e41d5d73074e03c8e553f4d2f0a977a57cfddc3b0eba7e7745aa180266e1.
  All26retained reference images remain unchanged. This is compilation only;
  all11fresh native stages now close normally0with reaped empty ownership,
  quiet validation,unchanged boot/driver and no fresh GPUfault. Both head
  modes pass four seeds,two restore cycles and12graph replays;packed head2's
  maximum pair error is4.17232513428e-7. Fresh attention/producer small/full
  and GUsmoke/numerics gates pass their original criteria. This remains
  component qualification;the fresh KV oracle and production handoff below
  now close,while both-phase performance comparisons remain pending.
  Child`2734a78bb`also prepares its standalone B1device reader. Six committed
  host controls pass normally0with reaped empty ownership:one valid census
  and five generation/fixture/source/coverage/partial-link refusals. Its
  142objects/137TUs now match the actual fresh B1build. Add this uncompiled
  observer only after that exact build's committed closure;fresh
  eleven-stage native gates,oracles,model and both-phase comparisons remain
  pending. The ended host raw root is retired(3files/50050bytes)after348
  original source inputs and the exact recipe are recovered and checked.
  The prospective reader recovers A's original compiler/class/SDKclosure and
  retains fast14508/12ELFs plus A/14ELFs,without reviving retired old binaries.
  Merge`80ab8eac2`integrates only its199uncompiled observer/control lines and
  preserves original2734/73d88identities in permanent history. Every1193B1
  build input remains byte-exact;the held B1simplementation is excluded.
  The new B1device plan pins1194source files and11fresh stages,without
  inheriting GPUresults. The completed child has no unique commits,PRs,
  pending files or ELFconsumer;its remote/local branch and worktree are now
  removed after verifying pushed permanent ancestry. The PR6verification
  child remains unfinished. Future
  Orca accounting uses the byte-identical permanent`tools/research/orca_scope.py`.
  The next66-line`check_split_kv_oracle.py`adapter reuses the existing KV
  oracle/graph driver,with explicit B1native selection and a fresh result root.
  Commit its source and new smoke/full plan before execution;retain the
  original14direct/20graph and795full/60graph/15physical-tail/12timing-shape
  criteria. It adds no native build or inference setting and inherits no
  old oracle/model/performance result.
  Its fresh traced smoke and quiet full oracle now close normally0with
  unchanged boot/driver,no fresh fault and absent owned native processes:
  14direct smoke cases,20smoke graph replays,795exact full records,60full
  graph replays,15physical-tail graph cases and12timing-shape checks.
  Runtime inventory contains34files with zero missing/unknown mappings.
  These are same-build cooperative/scalar numerical gates;actual production
  handoff now also closes normally0at controller`c184f1eb8`:thinking-on
  32769/82/32769inputs and32outputs each,all48layers twice,two actual
  cooperative-decode4captures and scalar-only short prefill. Placement,
  GU/Down/QSA work conservation,MTPrestore/recapture,15full asset hashes,
  quiet validation,closed ownership and unchanged boot/driver all pass
  without fresh faults. Full first-job histograms are curated out after
  validation;their work totals and hashes remain. Both-phase performance
  remains pending. The final cell in a
  component fixture does not qualify a262144-token model fill.
  After exact source/recipe/census/ELF admission and closed-owner checks,
  its successful native/oracle/model raw roots are retired:172files,
  23298472regular bytes. Curated committed gates,shared build/runtime
  manifests and all40engine/fixture images remain;do not resume the closed
  raw-parser controllers.
  PR#180publishes this closed validation/retirement boundary to default
  `f08f3a431`;normal merge-back`2e4147ba8`preserves every original measured
  identity in permanent`feature/sycl`,and the clean ghq default mirror is pulled.
  Source`9c2fc6005`now closes the B1-only`bench.sh`prefill comparison at
  `b800ec706`:12independent processes/36fresh requests,first/warmed32K/64K
  separate,all normal0/owners closed,no fresh faults,and15final model hashes.
  Execution passes,but the predeclared performance gate fails six of eight
  median screens. Warmed32Kfalls834.76945->827.13447tok/s and64K979.15722->
  974.05859;clientTTFT rises39.397->39.7795s and67.1455->67.4405s respectively.
  Only the64Kfirst-use prompt/TTFT median screens pass. Keep the candidate
  unselected,the faster14508reference and every original failed outcome.
  There is no naming-only image-identity exception and no B1s/B2advancement.
  Decode remains a separate prospective comparison;this prefill loss cannot
  be accepted in exchange for any decode gain. Eight CSVpoints index both arms
  and scopes. The post-closure descriptive warmed-time lines are
  `11.577405+0.000844632*n`s for A and`11.951756+0.000844266*n`s for B1,
  using actual input count`n`. These are only two-anchor descriptions,not
  validated scaling laws or measured physical setup costs;TTFT/first-use
  descriptions and individual samples remain in the result. Held-out48Kand
  longer/window-boundary checks on the retained fast path remain pending.
  After source/recipe recovery and closed-owner checks,the ended B1prefill
  and current PTIhost/probe raw roots are retired:887files/129719275bytes.
  All40retained engine/fixture ELFs are unchanged;36PPsamples,failed phase
  decision and current small-fixture clock/map evidence remain committed.
  Do not resume the closed raw-parser plans;use a new committed plan.
  Current PTItools are Main-authored and integrated through`ba5185424`;
  their original child identities remain in permanent`feature/sycl`,and the
  finished child is retired. Admission recovers the original457/460compiler
  closures,surviving launcher/collector,fast14508source/12ELFs and84current
  independent runtime assets. Small control/trace fixtures each close normally0
  with ten numerical cases,actual target/preload/provider maps,no fresh faults,
  and680GPUintervals within externally measured RAW bounds. These are safety
  and clock gates,not model attribution or overhead qualification.
  The first two current model-profile attempts remain failed:the first used
  the wrong32K/packed-draft reader;the second's clean arm closes three fresh
  32769/24thinking requests normally0,but its trace arm rejects the environment
  before a long request. Clean observations are654.0/838.4/825.5tok/s with
  clientTTFT50.251/39.225/39.893s;they are comparison observations,not a new best.
  Actual model exec-environment differences were not captured in that failure.
  Bounded Python-target observation confirms the launcher adds
  `INTEL_LIBITTNOTIFY64=libunitrace_tool.so`. Fix`152b8d51c`accounts for that
  exact value and pinned version while still rejecting unknown keys;new
  environment gate closes at`79a63ad48`with normal0/closed ownership/no faults.
  The third campaign remains failed:its trace reaches the predeclared256MiB
  file cap and exits with SIGXFSZ(-25). Source`a8ee23524`prospectively permits
  1GiB/4million events per trace;the new environment gate closes at`7ca3b66e2`.
  The fourth campaign closes at`a1f90cdff`:four independent processes in
  clean/trace/trace/clean order,three fresh32769-input/24-output thinking-on
  requests each,all normal0,closed ownership and no fresh GPUfault.
  Every paired first-use/warmed prompt,clientTTFT and total-duration ratio
  passes the committed0.9..1.1observer screen. First-use prompt ratios are
  1.058010/1.042343;warmed ratios1.081827/1.076385. Clean subsequent PP
  observations are841.2/833.5and838.7/836.2;profiled observations remain
  separate and are not clean speed/adoption evidence. Preserve all earlier
  failed outcomes;no source/pin refresh.
  Orca Luna BDand native Luna-max BEreview are closed:the original no-defect
  report disposition is corrected,and actual`bench.sh`source pin coverage is
  verified. Main reads the short review,not the collection body.
  Main-authored offline analysis`d3557debd`passes11host controls and both
  exact-trace/count/service/union conservation checks. In the four warmed
  traced requests,median observed copy interval union is10.309796s and
  kernel union28.658416s;copy/kernel overlap is about0.004s. The largest
  individual kernel is the existing cached-Q/vector-KV ordered-exp QSA
  prefix,6.799218/6.805407/6.798673/6.804860s. Copy types and top12normalized
  kernel families plus conserved tail are retained in
  `cpu-profiler-current.json:unitrace_current_request_resources_v1`.
  Unknown-memory-to-device API requests total about46.2GB per warmed request;
  PTI's`M`means unknown allocation type,not proof of host memory. These are
  requested bytes and observed service intervals,not physical PCIe/VRAM
  traffic,saturation or prefill critical-path attribution. Each request also
  contains24decode tokens. Native prefill-drain proof and hardware saturation
  remain unqualified;the limiting resource within the large kernel interval
  is unresolved. Existing provider-origin and1.576/1.538overhead failures
  remain unchanged. Do not select a native-exp2 API rename:the earlier pinned
  Intel source audit found regular FP32exp2 forwarding to native_exp2,without
  demonstrated different emitted code. Check prospective bank-slot changes
  separately from kernel changes;the current scratch contract requires8192
  chunks,so a larger chunk is not a settings-only experiment on this route.
  The trace-analysis consumer is closed and700external files/1402195439bytes
  are physically retired,including the two complete traces and the failed
  256MiBcapture. Keep the compact individual samples,corrections and resource
  evidence plus512CSVpoints. Three ended diagnostic/analyzer source files
  are retired to their original Git commits;shared seed/runtime receipts,
  surviving collector and all40native ELFs remain available for named consumers.
  PR#182publishes this boundary on fork default`11dc9765e`;normal merge-back
  `01ca08f23`preserves179original CSVsource/evidence identities on permanent
  `feature/sycl`. The clean ghq default mirror is pulled;no child branch is added.
  Main's next unit is the existing prefill bank capacity only:384versus512,
  with fast14508math,8192chunks,allGPU32Kresiduals and decode1256slots fixed.
  Source`b28ac9e03`defines six independent512/384/384/512/512/384processes,
  three fresh32769/24thinking requests each,first use separate from subsequent2.
  The original host stage fails because its synthetic512completion footer
  still says384;no native model ran. Preserve that failure. Fixture-only
  correction`56ce46ce4`uses a fresh`prefill_bank512_compare_v2`root and plan,
  whose19host controls pass. The six-process cohort closes at`6103815b9`with
  18fresh thinking-on32769/24requests,all native0,quiet validation,closed
  ownership,no fresh fault and15full model hashes matching. Candidate512/
  reference384warmed prompt-time ratios are0.999155/1.001262/0.998744;
  clientTTFT ratios0.999457/1.001176/0.998908. The all-three gain gate fails.
  Warmed medians are832.050PP/39.5355sTTFT for384and831.895/39.554for512.
  All arms request exactly the same bank bytes for each matched body;their
  process medians are35326259200bytes. This is requested APIvolume,not
  physical PCIe traffic. Keep384and reject this512change;do not start its
  gated64K/decode follow-up. No kernel,chunk,decode or default change is bundled.
  Orca Luna BFand native Luna-max BGreview are complete. All ten Taskpins
  match;the report is consumed through its146-word review digest only.
  Missing private-view/full-fit telemetry limits causal claims,without
  invalidating the checked bank/allGPUplacement. No copy reduction rejects
  that mechanism claim alone;a separate reproducible speed gain stays eligible.
  The owned worker family and terminal are closed;CPU/wall upper bounds are
  52.256379s/462.73s. Keep report/review bodies only through this cohort's
  decision. The existing research-index row now holds the curated findings
  and closure;report/review bodies and transactions have no remaining consumer.
  Their467raw files/63432307bytes are physically retired. Six ended bank/
  fixed-profile controller,test and fixture files are retired to their source
  commits;the temporary bank-reader parameters are restored exactly to their
  original bytes. The then-retained40native ELFs were unchanged;the failed B1
  generation is now retired after2390Git source/dependency,1474compiler and
  29link-input recovery and14final-ELFchecks:197files/207539980bytes.
  Retain fast14508/12ELFs and namingA/14ELFs;shared receipts remain available.
  Four individual first/warmed arm aggregates bring the CSVto516points.
  The independent QSA workgroup experiments are now **closed and rejected**.
  WG64source`f20d1b751`and WG256source`1fa754132`start from the owned WG128
  kernel,holding32cells,pitch33,cachedQ,K/Vvec4,orderedFP32softmax and BF16
  storage fixed. Required head stepping,PVpartition and8/2accumulator fragments
  form each coherent geometry unit;no placement,chunk,decode or math change is
  bundled. WG64evidence`93db6b0e3`and WG256evidence`2a2aca003`each retain108
  eligible paired batches from3independent processes. Every shape is slower in
  every process. The WG256cohort's WG128/WG256aggregate times are
  8.532513/9.300179ms(cap33gather),223.887965/268.867288ms(cap2051gather),and
  138.230953/184.714947ms(cap2051clustered). Preserve WG128;neither failed
  performance gate permits a model trial. These direct-export comparisons
  isolate geometry and include admission,prefix refresh and checked drain.
  They do not qualify production dispatch,modelPP/TG or physical model context.
  WG256executes all7stages normally0with closed owners,quiet validation and no
  new fault. Its two11-case oracle suites include all12real heads,padded heads,
  exactBF16output/rawFP32state and262144physical-cell boundaries. Exact normal/
  capture kernels admit256threads;reported maximum1024,device-local capacity
  131072B,requested dynamic local20036B and reported private0 are distinct
  quantities. The last value does not prove zero spills or occupancy.
  Original failed source/API checks remain failed. Before any GPUwork,Mainand
  Luna-max BLcorrect the WG256softmax's duplicate head ownership;the report's
  inheritedQKoverlap claim is false. The initial same-application,cross-TU
  kernel-ID lookup then fails before anyURenqueue. Its observed2026.2symptom
  matches the [Intel report](https://github.com/intel/llvm/issues/22852);
  resolving IDs beside their submissions is verified by the fresh7-stage run.
  Source/recipe and all final ELFs are recovered before retiring each ended
  generation. The final WG256component alone retires47files/8679440bytes;
  fast14508/12ELFs and namingA/14ELFs remain unchanged. Unused prototype
  exports,fixtures,controllers and adapter-only hooks are retired to their
  original Git commits;the shared oracle is restored exactly to`96d319326`.
  Six additional component CSVpoints bring the cumulative index to528rows.
  The source-derived demand model in`cpu-profiler-current.json`keeps padding
  and32-cell tail work distinct from real-head work:the2051fixture executes
  279172874240matrixMACs and has34410070016Bof logicalK/Vload upper demand.
  These are not physical traffic or saturation measurements.
  Native Luna-max BMretrieves three existing overlap/prefetch reports. Current
  layer-major admission requires`m.copy==m.cs`;bank publication relies on that
  ordering. An independent long-prefill upload/compute lane needs event-backed
  publication,reader retirement,retained owners and checked drains as one
  coherent unit,preserving math,decode and qualified runtime safety. The bounded
  Orca Luna BNsource audit and native Luna-max BOreview are now closed.
  Actual two-request evidence`prefill_transfer_route_v1`(`45260f03c`)confirms
  source-view1,staging0,owned slab0and all48private layer geometries;24layers
  are full-fit512and24remain partial. Existing layer/FFNadmission establishes
  their same-queue ordering and checked final drains.
  The metadata-only inventory`prefill_queue_capability_v4`(evidence`31413aec3`)
  now confirms ordinal0compute+copy/one physical engine and ordinal1copy-only/
  one engine,API1.17,normal0/closed owners/no fresh fault. It submits no GPUwork.
  The unchanged production Runtime's four constructor cases now return native
  immediate,in-order ordinal0/index0handles in`prefill_runtime_queues_v2`
  (`794bb2672`),with zeroURenqueue calls,normal0,quiet pre-main validation and
  closed owners. Preserve copy-offload avoidance. An explicit ordinal1copy-only
  path is API-feasible in principle and requires a separate queue-policy safety
  qualification;keeping the offload variable set does not qualify raw LevelZero
  work. The standalone copy-engine probe uses `sycl::malloc_host` host USM;
  physical page pinning is unverified. Earlier plan wording "pinned host" is
  corrected here without changing allocation,inputs or either timed arm.
  Orca Luna BZand native Luna-max CAconfirm the borrowed-handle/drain-before-free
  ordering and identify the documented interop-header order and explicit named
  provider-admission gaps. Fresh source`3d801e985`corrects those prerequisites
  for both arms;its executable is byte-identical to the original probe,so the
  header correction is not evidence of a compile/runtime failure or speedup.
  Main's separate V4→V5control changes only`ZE_ENABLE_HANDLE_LIFETIME=1→0`:
  default-context/internal1-byte allocation initialization changes from a ZE
  validation error to normal0for both ordinal0and1metadata. Parameter/thread/UR
  validation,APIlogging and qualified runtime safety remain enabled. This is
  checker compatibility evidence,not copy bandwidth or model qualification.
  The fresh`prefill_copy_engine_v6`cohort now closes all14normal0processes:
  two metadata controls,two traced256-byte checks,four1MiB/256MiB checks,and
  three alternating independent bandwidth pairs. Full source/readback/edge
  guards,35actual initialized providers,quiet validation,closed owners and
  unchanged boot/driver pass with no fresh GPUfault. Each speed process retains
  one warmup and three8×256MiBhost-complete batches;all18timed batches exceed
  the declared100msfloor. Source is`3d801e985`,ELF770f72aae9eed67e94f73c18e88e48dd2103219f456012ab17be8dc66fb9d82f.
  Evidence`a0404d96e`preserves the full cohort. Process-median requested GB/s
  are SYCL6.485465/6.292787/6.291627and copy-only5.801157/6.444878/6.259003.
  Paired copy/SYCLtime ratios1.117961/0.976401/1.005212have geometric mean
  1.031425;the predeclared independent replacement gate fails. Preserve all
  slower batches,including the3.52..6.49GB/sSYCLand3.55..6.45copy-only range.
  A queue replacement alone is not adopted or advanced to a model trial.
  These are synchronized hostUSMrequested-byte rates,not physical pinning,
  bus/VRAMtraffic,saturation,production-path cost or model PP. CPUtime covers
  submission plus completion together and does not qualify disjoint attribution.
  The current limiting-resource evidence remains the original~10.31scopy union,
  ~28.66skernel union and~4msoverlap;this probe calibrates a transfer primitive
  around6.3GB/sbut does not establish either resource's physical saturation.
  Any copy/compute overlap test needs a new minimal ownership/dependency unit
  and prospective comparison;it cannot inherit this failed speed gate.
  Exact Git source/recipe recovery for both probe images verifies355project
  files,635actual compiler inputs and25link inputs each. The closed fixture,
  one-shot controller and normal transaction output are retired to their
  original commits;122external files/3000590bytes are physically removed.
  All26independently retained phase-reference ELFs remain unchanged. The CSV
  now contains550points,including two explicitly component-only transfer rows.
  Legacy/.cache/provenance holds remain separate from this closed cohort.
  The new E1safety prerequisite at`f411a5c40`isolates copy-ready and last-reader
  tokens,explicit ordinal1work,source/bank retention,cancellation and dual-queue
  drain. The fresh single-TUfixture closes compilation with641actual compiler
  inputs/25link inputs;metadata alone is normal0with zero application enqueues,
  verified providers and closed ownership. Native Luna-max CCretracts the
  unreachable event-leak claim and distinguishes fixture gaps from production
  owner obligations. New committed host controls at`0a5513b67`expose two
  bool/inttype-confusion cases in plain local/public dictionary equality;the
  original failed outcome remains. Source`a49344a6f`uses typed JSONfingerprints
  and adds actual exception unwinding after an accepted copy. Its fresh image
  passes all three host test methods,five changed-input and two type-refusal
  cases before compilation,and all seven E1native processes at`e0609f6fa`:
  metadata,held forward,held second-reader overwrite,permanent hits,normal
  cancellation,exception unwind and invalid/foreign-token refusal. Every
  process is normal0,closed,quiet,with required providers through before-exit,
  unchanged boot/driver and no fresh fault. The declared50msheld observations,
  manual release and100mssubmission assertions pass. Its ELFis
  b36d91d9ee7a28240bd25a7cf8732a2393cb262e40942e665b61fa965c7ea09b.
  This qualifies these standalone hostUSMevent cases,not ordinary RAMviews,
  actual Runtime::compute() integration,async-fault injection,overlap speed,
  a production route,model quality or physical262144context.
  Original Git source/recipe recovery verifies357/358project files and641
  compiler/25link inputs for V1/V2. The ended metadata image and normal E1/audit
  output are physically retired:85files/2431219bytes. The final four E1files
  are now retired after exact original source/recipe/ELF recovery and fresh
  production Runtime qualification. Together with the ended CD/CEaudit output,
  source views and owned native transcript,39more files/1589907bytes are
  physically removed. All26retained phase-reference ELFs remain unchanged;
  legacy/.cache holds stay separate.
  The copied-bank E2component source is archived at`0a3295b89`
  (`bench/sycl/prefill_upload_overlap.cpp`);it is not a
  selected route:it fixes the copy primitive,payload,allocations and existing
  Xe2dense kernel while moving only the readiness wait across independent
  compute. A per-epoch reader fence limits lookahead to one expert. Its build,
  source/recipe and prospective timing plan are pinned at`d4e45762a`/
  `7b0afbd77`. Five fresh coherent TUs use the original14508Runtime/class
  definitions and dense math,925actual compiler inputs and25link inputs;
  no original archive/object is reused. E1bchanges only the fixture's queue
  factory to that production Runtime and passes all seven safety cases at
  `3782c0f31`,normal0/closed/quiet/no fresh fault. The separate E2then fixes
  that context,original dense math,copy primitive,bytes,placement and lookahead,
  changing only the readiness-wait position. Its committed comparison plan
  must retain three alternating independent-process pairs per shape. U1eager
  upload,U2aggregation and copy-only replacement keep their separate outcomes;
  no combined result qualifies those units. CD/CEsource review retracts the
  false missing-exit-provider allegation:the controller requires provider
  paths at every gate except before-runtime,including before-exit. Generic
  enqueue presence is a sanity check,not device-overlap attribution. Full cohort
  `3ee710401`closes23normal0processes:seven Runtime safety cases,four validated
  arm/shape batches and twelve clean processes. Every payload,oracle,guard,
  provider,timer,ownership and fresh-fault gate passes;all33owned build/native
  commands close. The actual timed boundary is512epochs plus dual drain,
  not a model request. At1024rows serial seconds are0.247889/0.292666/0.248008,
  overlap0.165671/0.165638/0.248944;paired ratios0.668329/0.565964/1.003774,
  geometric mean0.724111. Its all-three gain gate **fails**. At4096rows serial
  seconds are0.357760/0.357830/0.367523,overlap0.205792/0.209515/0.364216;
  ratios0.575224/0.585516/0.991001,geometric mean0.693665. Its component
  wall-time gate **passes**,including the small third-pair gain. Preserve both
  slow third overlap samples;do not relabel them as noise. The CSVrecords both
  shapes with model PP/TTFT/TGblank. Kernel/transfer critical-path attribution
  and resource saturation remain unresolved:wall-time gain alone does not prove
  actual device overlap or explain the variation. The archived
  `profile_prefill_upload_overlap.py`controller reused those two images
  through a historical source/recipe/census reader. It separates seven
  collector/event safety cases from C/T/T/Cper arm/shape attribution,with
  owned-process RAW-clock associations,exact513copy/514dense calls and
  one-lookahead dependency checks. V1passes nine host controls and zero-work
  metadata. Its forward native process is normal0with the expected10URcalls,
  two reader kernels and one256Bcopy,but the reader wrongly requires a host
  API flow endpoint in device-only capture;the cohort remains failed. A new
  observed-schema host control at`f9e82a578`reproduces that refusal(six tests,
  one expected error). Prospective V2matches the two GPUside t/s markers,
  owned native-PID category suffix and HOSTmetadata with no PMI/PMIXrank,
  as emitted by pinned PTIchromelogger.h. It claims no host API timeline.
  V2's committed plan passes all ten host controls,metadata and forward native
  checks. The next profiled reverse process exits1with
  `scratch overwrite escaped last reader`;P1interoperation fails and P2speed
  sampling does not start. Its saved LevelZero sequence spends approximately2s
  in a query/collector chain until the independent host gate auto-releases.
  This does not establish an early scratch overwrite or a GPUhang. Copy/compute
  drain succeeds,the direct child is reaped and its owned session is empty;
  record physical closure separately from the failed protocol-observation flag.
  Boot/driver remain unchanged and no fresh fault appears. The actual
  PTIevent-cache helper guards release/reset by ownership,so merely passing an
  application event to that helper does not establish event corruption. Native
  Luna-max confirms that ownership guard and retracts the source audit's
  event-reset concern. The separate ordinary-RAMreview requires same-context
  import proof and API1.16for the read-only flag. Do not reuse this failed
  collector/event combination for P2or model work. No rebuild or model route
  is implied. Preserve the failed1024gate and qualify observer cost separately;
  do not change its criteria retroactively. After this cohort and the capability
  query close,the E2images,fresh objects,source view and private census are
  retired. Their original399Git-source files,925compiler/25link/7object inputs
  and both final ELFs were verified against the original14508/d4recipe and
  pushed retained history first. Including closed profiler/capability output,
  reviewed research inputs/control/review files and two closed task-only native
  transcripts,534files/21698392bytes are physically removed. Three ended
  observer/reuse/capability tools and two tests are retired to Git;keep the
  event-lane header,two C++fixtures and E2recipe/host controls only for the next
  source-adapter unit,reviewing them at its prepare boundary. One17688Bquery
  excerpt and the9546Bordinary-RAMreport have named next consumers. All26phase
  reference ELFs remain unchanged;legacy/.cache holds remain separate. The next
  native unit must qualify its rebuilt baseline/candidate;the old component
  generation is not retained for an unspecified future comparison. Earlier normal
  output was retired after source/recipe and both final-ELF recovery:
  113files/22007248bytes,with all26phase-reference ELFs unchanged. Keep the
  individual clean samples and both failed/passed shape decisions in the compact
  current result;the complete original receipt is reachable at`3ee710401`.
  The separate zero-command capability check at
  `prefill_host_source_capabilities_v2`is normal0/closed,with verified before/after
  loader/validation/NEOproviders,unchanged boot/driver and no fresh fault. The
  actual B570 reports API1.17and standard external-system-memory mapping1.0;
  its reported single-allocation limit is10132807680bytes. This is prerequisite
  discovery only:no allocation,import,queue,copy or kernel was attempted.
  V1's reused nonzero device count was rejected by validation before device
  enumeration;its failed outcome remains. The new source-adapter unit
  `prefill_mapped_source_v1`uses fresh original14508Runtime/dense code and five
  coherent TUs at`08b7289ea`;ten host admission controls and native metadata pass.
  Its next zero-command import process exits1at the`HOST_IMPORTED`property
  check,after successful same-address,read-only allocation. Rejected mapping
  release and queue drains succeed;the process is reaped,its owned session is
  empty,and no fresh fault appears. The actual numeric type was not printed.
  Keep this31-case cohort failed;the remaining29cases and all source-cost timing
  are unrun. Full evidence is at`c35fe4199`;the current result preserves physical
  closure separately from its incomplete protocol-observation flag. The official
  API1.17specification requires`HOST_IMPORTED`,while inspected Intel source/tests
  return`HOST`;that source is not proven to match the installed package.
  Native Luna-max confirms the pinned static owner/lifetime audit,not runtime
  acceptance. The independently committed zero-command discovery at`37017156e`,
  native source`b777f52f8`,now closes normally0with quiet validation,closed owners
  and no fresh fault. A4096Bpage-aligned read-only heap map reports type0/id0
  before import,type1(`HOST`)/id1/page4096during import,and type0/id0after
  successful unimport. Its queried range has the same base and4096Bextent,and
  the original heap remains unchanged. This uses an explicit native context,
  not the production Runtime;it enqueues no application commands and qualifies
  no source admission,copy,math or model route. It does not supply the missing
  numeric value from the earlier failed process. The next eligible unit is a
  newly planned,explicit pinned-provider `HOST` property contract with positive
  owner/context/range/allocation-ID/lifetime proof;keep standard admission and
  unknown-pointer refusal intact. Requalify that contract in the actual Runtime
  before any copy,then compare source costs with readiness order fixed. No
  implementation of that compatibility contract is adopted yet. The new
  `prefill_mapped_source_v2`proposal keeps the default standard contract strict
  and opts the fixtures into a pinned B570/API1.17HOST-property contract.
  Its private lease captures type/allocation ID and rechecks them on each copy.
  A shared active-import registry rejects proofless aliases through the plain
  host-USMbranch,which otherwise also sees typeHOST. Those checks and their
  producer/consumer changes are one inseparable admission unit;timing order and
  dense math stay fixed. The zero-command Runtime case must reject the standard
  contract,then report positive HOST/id proof before any copy test. Every mapped
  timing process reports its actual import proof. Four new host controls cover
  typed historical property admission and ended-observer recovery. The new
  coherent build at`0a3295b89`passes14host methods,with928actual compiler,
  25link and seven object pins. All31native processes now close normally0,
  with quiet validation,unchanged boot/driver and no fresh fault,evidence
  `9cc26a79b`. Fifteen safety cases cover actual Runtime HOST1/id12admission,
  strict-default rejection,reader/overwrite order,reuse,cancel/unwind,
  unknown/proofless/foreign references and held-copy retirement/stale/reimport.
  Four traced serial batches validate full exact dense output,source/bank guards
  and513payload copies/514dense launches. Three independent alternating pairs
  per shape then measure clean source cost,holding math and wait order fixed:

  | Dense rows | Host-USMwall s | Mapped-Neowall s | Paired time ratios | Source-cost gate |
  | --- | --- | --- | --- | --- |
  | 1024 | .247883,.247614,.343656 | .246534,.246326,.253677 | .994560,.994800,.738172 | Pass |
  | 4096 | .356052,.367439,.357758 | .523874,.534236,.462948 | 1.471341,1.453943,1.294027 | Fail |

  Geometric-mean steady ratios are .900550and1.404111;including preparation and
  retirement gives approximately .905745and1.404121. Import is a subset of
  preparation,not an extra term. Keep every sample,including the slow1024USM
  third process and all slow4096Neo processes. The prospective gate requires
  all three steady and lifecycle ratios<=1.10;it permits only a separately
  committed1024readiness-order comparison. It is neither a speedup/adoption
  gate nor model/physical-PCIe evidence. Allocation ID is an additional
  consistency check;the header does not establish global uniqueness/nonreuse.
  Owned lifetime,range and distinct tombstone state remain essential. Keep the
  original source-build receipt for the next actual-weight-bank input comparison.
  The separately committed same-source/same-ELF1024wait-order comparison now
  closes at`98e6972a2`:eight normal0/quiet/closed processes,including two traced
  513-copy/514-dense batches and three clean independent-process pairs.
  Only`wait_ready`moves across the same dense submission. Serial times are
  .510194/.256638/.246519s;overlap .164447/.204115/.164570s. Paired ratios
  .322322/.795341/.667575 and geometric mean .555198 pass the prospective
  component gain gate;all lifecycle ratios also pass. Keep the slow first
  serial sample. V1's omitted provider metadata fails before Runtime and remains
  failed;V2adds transitive metadata admission and four passing host controls.
  This permits an independently planned actual model-weight source/selected-expert
  reader fixture,then a separate wait-order comparison;no model trial or
  production adoption follows. Both ended component ELFs/objects/source view
  and consumed research/raw output are now retired:484files/24029387bytes.
  Both independent phase-reference engines and the original private source-build
  receipt remain unchanged. The three closed controller/test files are retired
  to measured source`d1f3868e2`;keep the actual source-adapter headers and recipe/
  fixture sources for the next actual-bank producer/reader plan. The earlier
  normal source-cohort traces/build output retirement was159files/24022908bytes.
  The actual-weight proposal is now `prefill_weight_pipeline_v2`,native source
  `2187bfa5d`,plan`5c6c68938`,with six fresh TUs/943compiler inputs/25link inputs/
  eight objects. No archive/object reuse;the isolated dense AOTrecipe is unchanged.
  It extracts actual layer2experts0/1(IQ2_Sgate/up,IQ4_NLDown),then holds serial
  copying and the M1024production fallback math fixed while comparing hostUSM
  with the pinned imported heap. Both metadata processes close normal0with zero
  app work;imported HOST1IDs28/29are observed. The first USMmath batch closes
  normal0/quiet/no fresh fault,with full bitwise FP32accumulator agreement,
  all guards,1548math launches and258actual parallel-IQlaunches per role.
  The cohort now closes at`07735df23`:all10native processes pass with quiet
  validation,closed owners and no fresh fault. Both traced readers verify the
  same counts/bitwise output/guards. Clean USMtimes are
  .124446/.120681/.113829s;imported times .211941/.123120/.184089s.
  Paired imported/USMratios1.703074/1.020203/1.617239,geometric mean1.411123,
  **fail** the prospective all-three<=1.10source-cost gate,including lifecycle.
  Preserve every sample;no noise exclusion or mapped scheduling/model promotion.
  V1closed before any native build/execution because its inherited copy-only
  environment omitted the model's parallel-IQflag;V2matches flag1before exec
  and checks actual named launches. No job remains live. CQ/CRreview confirms
  the ownership seam with corrections:IQ4_NLblocks contain32values/18bytes;
  Main verifies the model's vllm/auto backend and M1024fallback call path,while
  dispatcher/gather costs remain omitted. The next independent unit starts
  from the faster,numerically verified **host-USMreference**,not the failed
  mapped candidate:one private scratch blob may be overwritten only after both
  IQreaders retire;GU/Down retain their separate dequantized workspace until
  their readers finish. Commit a new serial/lookahead plan with512jobs for
  timing headroom,holding source,math and placement fixed. This is not a relaxed
  old gate or model trial. That separate `prefill_weight_lookahead_v1`unit now
  closes at`41b00bc09`,native source`e2ef5ebe8`:six fresh TUs,941compiler inputs,
  25link inputs/eight objects and six host admission controls. One metadata and
  both traced readers plus six clean processes all close normally0/quiet/no fault;
  full bitwise accumulator/finite/status/guards and6156math/1027payload-copy API
  counts pass per traced reader,including1026actual IQlaunches per role.
  Serial clean seconds are .503431/.684353/.426261;lookahead .292253/.269679/
  .294672. Paired ratios .580523/.394064/.691294 and geometric mean .540775
  pass the predeclared component gain and lifecycle gates;no sample is excluded.
  Only the scratch upload fence moves;this is not whole-model PP/TG/TTFT or
  production adoption. `after_math`includes the initial reset-fenced copy.
  CS/CTreview validates the two-reader seam;dispatcher confirmation remains
  Main's prior context,not a newly audited callsite. Next pin actual production
  job geometry,source preparation and scratch publication/reader lifetimes;
  keep those independent units and decode preservation separately qualified.
  Hardware resource saturation remains unknown.
  The ended component generation,unrun V1preparation and consumed research/raw
  output are retired:941files/57321828bytes,after exact original source/recipe/
  final-ELF recovery. Both phase-reference engines remain unchanged. The former
  source-only receipt holds are resolved by the Git-only seed recipe reader at
  `b3b96e0ec`;closed-state admission now precedes any private receipt read.
  Eight current and11remaining source-capability host controls pass. These are
  controller-only changes;the native scheduling build still has its original
  six host controls and source`e2ef5ebe8`. Original source/recipe/final-ELF and
  both older source receipts were recovered before retiring538files/92114178
  regular-content bytes of ended native/raw/research and all three closed
  private receipts. All selected paths are absent;the Git-only seed reader and
  real closed CLIrefusal pass after deletion without changing the result.
  Both phase-reference engines remain unchanged;no new GPUqualification follows.
  The reviewed CL/CM
  audit's consumed report/view/control/closed transcript are also retired:
  40files/2033262bytes,with its correction and source hashes in the index.
  Four-Ksource-cost attribution needs its own bounded diagnostic. Compare the
  source adapter with readiness order fixed,and
  only then compare readiness order with source and compute fixed.
  After original source/recipe and both final-ELF recovery from pushed history,
  its failed temporary generation and consumed CHreport are retired:446files,
  16557507bytes. All26phase-reference executables and eight CMake probe ELFs
  remain unchanged;this is scoped cleanup,not resolution of legacy holds.
  The completed property observer and consumed CJ/CKresearch transactions,
  source views and positively closed native transcript are also retired after
  source recovery and the compact decision:41external files/2286503bytes and
  the one-question observer. Its original source/result remain in Git;the two
  independent phase-reference generations remain unchanged.
  Keep bounded host-USM staging as an independent control if import fails.
  Intel's vendor malloc/new-copy behavior does not itself establish the standard
  external-map contract;unknown pointers remain refused. No debug import flag,
  relaxed source check,production adoption or model-speed claim follows.
  Ordinary RAMsource views,actual production consumers,32K+model/decode/agent/quality
  and physical262144/RESTORE remain separate required gates. No model trial
  or engine adoption is authorized by this component result.
  The source unit now closes at`8bb58356b`,native source`1896d6a18`:
  ordinary readonly RAMthrough the complete production copy wrapper versus
  existing four-slot Host-USMstaging. Nine fresh TUs/961compiler inputs/29link
  inputs/11objects use original14508math;no partial archive reuse. All42native
  processes close normal0/quiet/no fault,including2metadata controls,10traced
  readers and30clean samples. Both split planes agree bitwise and all guards
  pass. Actual kernels confirm compact at16rows,wide at128/1024,and fused
  GU/Down at4096/8192in both arms. Each timed batch copies all512real layer2
  experts/1009254400requested bytes,with512math jobs;this is not modelPP or
  measured physical bus traffic. Three paired staged/direct ratios per shape:

  | Expert rows | Pair ratios | Geometric mean | Repeated source gain |
  | --- | --- | --- | --- |
  | 16 | .989884,1.033156,1.937132 | 1.255943 | No |
  | 128 | .985431,1.016530,1.681105 | 1.189726 | No |
  | 1024 | .754467,.996070,1.006687 | .911188 | No |
  | 4096 | 1.218028,.997216,.876361 | 1.021041 | No |
  | 8192 | 1.068731,1.027690,.981442 | 1.025333 | No |

  Keep every sample,including slow samples from both arms. All shapes also
  have at least one adapter preparation+wall+retirement loss. Keep production
  direct copying/global staging disabled;the1024geometric mean alone does not
  establish a repeated gain. Scharacterizes the whole existing adapter,not
  CPUmemcpy or wrapper overhead alone. Compare copy wrapper/primitive Qand
  readiness order Oseparately next,then their combination against the complete
  original direct-source path. No model trial/adoption,PP/TG/TTFT,decode,
  hardware saturation or physical262144/RESTORE qualification follows.
  The request-length model/held-out curve remains a separate pending check;
  these expert-row batches do not establish a model time law.
  V1plan`2a813610f`failed at fixture compilation before any executable/GPUwork
  because its compatibility header was missing;preserve that outcome. V2then
  built coherently. Its original unrun capture layout would count the1GBcorpus
  against raw capture caps;controller-only admission`d7e0a146d`replaced it with
  case-scoped capture before any GPUwork,retaining the exact native ELF.
  Eight current host controls pass. CU/CVreview confirms the actual source
  entrypoint and separate IQ/workspace-reader retirement. After exact original
  source/recipe/final-ELF recovery from the pushed primary,737regular files/
  1283624160content bytes of ended component/corpus/raw and consumed research
  are physically retired. Both phase references retain their original hashes.
  Six closed prototype roots subsequently pass exact source/recipe recovery
  and retire228files/56083842content bytes. Both phase references remain.
  Three configured deployment source/migration holds remain in the ledger;
  this is scoped cleanup.
  The CSVnow has562points,including all five Scomparisons with model metrics blank.
  See`prefill_weight_source_v2`and the retained-build ledger for actual state.
  The owned Q0copy-wrapper control now closes42native processes at source
  `d3ca4956b1bfda8ef64ef19d50827cff2abbafa4`and evidence`39d2c8ead3549d684c327d67991cbf415ce4f59a`:
  two metadata,ten traced dispatch/number/guard checks and30clean samples.
  Every child closes normally0with quiet validation,closed ownership and no
  fresh fault. Both arms keep ordinary immutable RAM,the compute queue,math,
  Runtime::check and command_profile::observe;only repeated handle/context
  lookup and status translation differ. All original samples remain in the
  compact result,including losses. Shortcut/direct geometric-mean time ratios:

  | Expert rows | Ratio | All3pairs faster |
  | --- | --- | --- |
  | 16 | 1.192071 | no |
  | 128 | 1.190276 | no |
  | 1024 | 0.883954 | yes |
  | 4096 | 1.019456 | no |
  | 8192 | 1.005329 | no |

  OnlyM1024passes its prospective repeated component-gain condition. The other
  shapes remain non-gains;the complete compatibility wrapper stays the engine
  reference. No model trial/adoption or PP/TG/TTFT/physical-context result
  follows. Limiting hardware remains unknown. A recorded outer scheduling pause
  after M1024pair0direct did not stop a native child or change order/inputs;
  preserve all independent samples. Five CSVrows bring the index to567points.
  Exact original364controller/source and398math-view files,961compiler/29link
  inputs,nine fresh TUs/11objects,recipe/final-ELF and all1GBrole bytes are
  recovered and verified. The ended Q0generation/corpus/view/raw output now
  retires691files/1279279287regular content bytes;both phase hashes remain
  unchanged. The subsequent R0/Q1/Ocohort closes below. Sresults retain
  their original1896d6a18source;the two phase references remain selected.
  The subsequent owned SYCLcopy-lane cohort is **closed and rejected as a
  combined performance candidate** at all five shapes. Original full evidence
  is`9888d6ad88f3cb9565cd64a327de74885109094f`,key
  `prefill_sycl_copy_lane_v2`;native source is
  `e5cd35836c1e0e608970eec19ad63bcf69823e4a`,admission source
  `148e6dac73b8cc964503be9decef0e6912abc8d4`. The coherent ten-TU/twelve-object
  build has962compiler and29link inputs. Both original ELFs and all inputs
  were reverified after closure. V1's wrong fixture-provider admission remains
  failed at`44e1fb7eea0b9ff4ea8eb1c1496ae1d28a613fb8`;its native child was
  terminated/reaped(-15)before Runtime/application commands. V2corrects only
  admission and reuses exact verified native bytes,not a relabelled rebuild.

  All115V2native processes pass normal0,quiet validation,positive ownership
  closure and no fresh fault:ten small lifetime cases,five zero-command math
  metadata,25traced full-math cases and75clean independent processes. Every
  clean shape uses three complete five-arm blocks,without retry/exclusion.
  The maintained result retains all75wall/lifecycle samples and20unit decisions;
  full per-process diagnostics/plans remain recoverable from the evidence above.
  Each batch has512ordinary-RAM copies/1009254400requested H2Dbytes and actual
  layer2IQ2_S/IQ4_NL weights,unchanged Gemm/split math and terminal drain inside
  timing. Bitwise output,split/tail/status/source/device guards,actual IQ/fused
  dispatch and timer eligibility pass. Matrix rows are not model input tokens.

  Candidate/reference geometric-mean batch-time ratios are below;the final
  column names only units whose **all three** predeclared ratios are below1.

  | Rows | R0dependency adapter | Q1queue separation | Oearly reader marker | Net original | Repeated unit gain |
  | ---: | ---: | ---: | ---: | ---: | --- |
  | 16 | 3.683795 | 0.961957 | 0.932996 | 2.772853 | none |
  | 128 | 2.939209 | 0.993769 | 1.005408 | 2.608187 | none |
  | 1024 | 2.133059 | 1.026150 | 0.887848 | 1.605262 | O |
  | 4096 | 1.243751 | 0.946126 | 0.997302 | 1.060399 | Q1 |
  | 8192 | 1.223881 | 0.842372 | 1.083604 | 1.087740 | Q1 |

  R0compares shortcut with the dependency/ownership adapter on compute;Q1changes
  only to a separate same-context/device,in-order/profiling-matched queue;
  Ochanges only the marker position from full math to after both IQreaders.
  Net compares the complete original compatibility path with their combination.
  Preserve every enabling loss. The isolated Q1/Ogains are follow-up evidence,
  not permission for a model trial. No new PP/TG/TTFT,decode,task/agent,
  physical262144/RESTORE,ordinary teardown or inference adoption follows.
  The selected prefill/decode engines and both phase references are unchanged.

  Closed1024-row traces show1028additional readiness/reader barriers for R0,
  with unchanged1049whole-process URmemcpy and46URevent-wait calls. The three
  Q1native immediate lists all request ordinal0/index0;this is not proof of
  physical-engine assignment. R0does not add explicit host-synchronize calls
  in that trace scope. API duration,physical bus traffic and the limiting
  CPU/GPU/bus resource remain unknown;do not infer them from counts or timing
  buckets. After exact original Git/recipe/SDK/final-ELF recovery and remote
  ancestry verification,the ended component/corpus/view/raw retires1120files,
  1696470830bytes. Both phase hashes remain unchanged. Its unused five-file
  prototype/controller is retired to the original source/admission revisions;
  neither the code nor the timings imply adoption. Two source-audit transactions
  retire70files,1699499bytes after source commits and curation. Only compact
  corrections remain;the new fixtures' committed build/run plan follows below.

  Main's next **unselected** proposal is`prefill/sycl_copy_groups.hpp`
  and`tests/sycl/prefill_sycl_copy_groups.cpp`,source
  `695565fbda8802363b2c91b49a6021ba39326b55`. The earlier audited prototype is
  recoverable at`dbb3506bc1f295f16d500a87ef1f924c2003dea3`. Both are independently
  Main-authored from the owned lane/Runtime;no external implementation is used.
  The ordering rationale is the[SYCL in-order contract](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:queue-properties)
  and Intel's[enqueue-barrier contract](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc),
  not evidence of installed-provider behavior. Separate readiness/retirement
  sequence counters and consecutive-ticket overloads let either marker vary
  on128one-expert slots. Sources/banks stay borrowed through terminal drain;
  bank reuse requires its submitted reader marker. Physical slots are layout
  width times bank count;capacity bounds submitted groups,not payloads.

  The new`prefill_sycl_copy_groups_v1`plan uses native source
  `05eceacdd807628e4b27834f7167c5672f72fccb`,with the same695565fbdheader/lifetime.
  Its fresh owned recipe creates ten-TU/twelve-object math/lifetime fixtures;
  neither retained phase reference is overwritten. The bounded third generation
  retires at comparison closure before another build. It first runs13host
  admission controls,then24small traced lifetime,9math metadata,27traced math
  and81clean processes:141native cases. Matrix rows128/1024/8192are a bounded
  lower/intermediate/fused component screen;old16/4096evidence stays separate.
  Three full nine-arm blocks use forward,reverse and a predeclared permutation.
  Isolate S(complete compatibility->queue shortcut),L(slot capacity1->128 at
  one-expert window),R(required adapter),Q(queue),then W(window1->128).
  At fixed128one-expert slots/window,compare readiness1->64with reader width1,
  and independently reader width1->64with readiness1,then both64and their
  interaction. Also compare window,ready64,readers64and both64with the complete
  original separately;a losing combination cannot reject a faster valid variant.
  These13contrasts across three shapes retain39unit decisions and three
  multiplicative interaction points. Any later64-wide/two-bank Gpartition
  remains a separately scoped bundled policy,not part of this campaign.
  Keep512individual copies,source order/bytes,full-math retirement,math and
  precision fixed;slot phase includes the same two warm copies in every arm.
  All128final slots are checked. Timed branch counters prove actual compat/
  shortcut/adapter calls and each marker's count;these scalar counters are
  included in timing,without per-expert clocks. Terminal drain is timed;
  reference/reset/validation is outside. Each batch must reach100ms,with
  maximum128clock-pair overhead<=.001batch. Every unit's gain condition remains
  all three paired ratios below1,no minimum gain. No model trial is authorized.
  Source-reviewed fixtures include widths1/2/64,partial groups,other-bank
  prequeue,reuse/refusals and accepted-copy/reader host exceptions. The ordered
  mode now consumes/readbacks its ticket after capacity refusal;four additional
  modes exercise256experts/full128-slot window/reuse and independent marker
  policies. Their occupied-bank refusals are not capacity checks. Readiness
  explicitly depends on copy completion in both placements. DG/DH and DI/DJ
  source reviews find no blocker under caller contracts. The fresh05eceacd
  build now passes13host controls and all15owned build commands,with962actual
  compiler inputs,29link inputs and12fresh objects. Both fixtures are compiled
  and their ELF hashes are pinned. All24small traced native cases now close
  normally0,with the exact declared kernel/copy counts,quiet validation,closed
  owners and no fresh fault. This includes both placements and the full128-slot
  independent-marker matrix. DK/DL and DM/DN source audits find no blocker after
  correcting the reports' slot-count,citation,readback and performance-failure
  statements. All9math metadata and all27traced math cases now pass at
  `31258714d`:all three row shapes and nine policies. Actual kernel/copy/marker
  counts,bitwise reference,split/tails and guards agree,with normal0/quiet/closed
  owners and no fresh fault. The27clean128-row samples are complete at
  `dc842728f`;none of its13unit/variant contrasts meets the predeclared condition
  of all three ratios below1. The adapter-only Rcontrast takes2.405/2.323/2.907
  times the layout batch time. This identifies a reproducible component loss,
  not its limiting hardware resource. The direct batch varies0.369/0.609/0.563s;
  preserve those individual samples rather than dismissing the variation.
  All27clean1024-row samples are complete at`fc234a435`. At fixed128slots
  and window,Aready batching ratios are0.967989/0.743473/0.985289and Bindependent
  reader batching ratios0.939587/0.636921/0.767772:both satisfy their own gates.
  Combining both is1.100071/1.052476/1.112726times B's batch,so preserve the
  distinct decisions. None of the four net variants beats the original in all
  three pairs. The full141case cohort now closes at`1496a4902`:81clean
  processes,all39unit/variant decisions and three interactions. At8192rows,
  L,AB_after_A,netWindow,netReaders andnetBoth satisfy their own three-pair
  conditions. NetWindow/Readers/Both geometric mean time ratios are
  0.918370/0.796766/0.748444. Both64uses1.255137/1.357802/1.352193s versus
  original1.809218/1.649663/1.841626s;its ratios0.693746/0.823078/0.734239
  pass at every pair. Its gain over readers64alone does not repeat in all pairs,
  and A/Bisolated gains do not repeat at8192. Preserve every individual decision;
  a winning net policy does not qualify each constituent independently.
  This leaves both64as the fastest credible large-shape component proposal,
  with window/readers64also eligible for a future independently planned test.
  Do not apply it to small/intermediate jobs or assume that synthetic512expert
  loops predict model routing,cache reuse or total throughput. All native owners
  close normally0,validation is quiet and no fresh faults occur. The CSV retains
  six useful component milestones;model PP/TTFT/TG remain blank.
  API counts alone do not attribute duration or prove that retained events cause
  the adapter loss. DQ/DRreview supports only a bounded lifetime hypothesis;
  its600sworker deadline failure remains failed. Choose one reference-retirement
  edge only after current closure and time/resource attribution;never manually
  destroy native events or release borrowed tensor storage before terminal drain.
  **Only the8192-row component net gates pass;no model qualification follows.**
  For subsequent shape selection,the original14508model evidence at
  `9c678595231882266623373cb84e4b65071c53dd` retains a conserved16-bin planned
  router census for32769input tokens and48completed windows:22479jobs and
  15728640expert-row assignments. M<128accounts for42.190489%of jobs but
  2.248859%of rows;M>=2334accounts for7.535922%of jobs and45.219930%of rows.
  These are planned-job aggregates after router drain/success,not measured
  kernel work,upload traffic or hardware saturation. Exact within-bin tails and
  row/weight-format correlation remain unavailable. The current layer2fixture
  has K=2560,FF=640;its128/1024-row traces select the actual
  `DenseKernel<half,bfloat16,Tile1>`with2052launches,which the original source
  maps towide128x128. This does not qualify a tile change or a model-wide
  arithmetic ceiling. The M/Ngeometry research scope remains conditional until
  measured stage/resource evidence makes it the next tuning candidate.
  Submission/async/drain-fault and active
  observer coverage remain explicit holds;source review is not safety,
  precision or model qualification. Next pin the actual expert-slot/cache and
  last-reader integration seam before a new one-unit plan;fullfit and recycled
  scratch slots have different ownership contracts. Do not copy the fixture's
  fixed128-slot schedule wholesale into the model. Exact final ELFs,recipe,
  366project sources,398original native sources and962/29/12compiler/link/object
  inputs now match their original Git/build records;the1GBcorpus is also rebuilt
  bit-for-bit by streaming the original model slices. The ended third component
  root is retired:1264files,1786561342bytes,plus its592705-byte temporary inventory.
  Duplicate case/environment/recipe inputs recover from`1496a4902`;all141outcomes
  and39decisions remain unchanged. Both selected phase references are retained;
  legacy provenance/fixture/cache holds remain separate. No new build starts
  until its own committed source,recipe and comparison plan are ready.
  The independently authored,unselected candidate is
  `sycl_unique_slot_copies.hpp`at`84f3d3c29`:strict sequential one-write-per-slot
  ownership permits omitting per-expert reader markers until terminal drain.
  It is not connected to a model route;no model qualification or gain is established.
  Decision-support scope`prefill_unique_slot_fullfit_source_ds_20261117`
  completed normally0 within its resource envelope;native Luna-max review
  accepts the source with corrections. Its curated index records four concrete
  fullfit/staging/publication/checked-loan invariants. The wrapper must drain and
  die inside a live cache loan;the staging buffer,not merely its original slab,
  must survive H2D. Main source`8b2d9ac46`adds a15case small GPU lifetime plan
  and8passing host admission controls. Native source`535911a2b`now passes
  all15small checks at`4eb8276e0`:both placements,128/129/512slots,ready1/64,
  refusals,terminal drain and accepted-copy host-exception retirement. All21
  build/test/native commands close normally0;no validation errors or fresh
  device faults. Four fresh TUs/four objects match719compiler/25link inputs.
  No model integration or performance gate passes from this safety fixture.
  The three-arm comparison now closes at`52bd1156b`,native source`01823c734`:
  3metadata/9traced/27clean processes,all52owned build/test/native commands
  normal0,quiet validation and no fresh fault. Source proofs admit252314112B
  versus1009254912Bbanks;full bitwise/split/tail/status/source/guards and all
  final128or512payloads pass. Every arm retires its two-job warm owner and
  creates a fresh timed owner at slot0. Capacity-only S wins all3pairs at each
  M128/1024/8192,with geometric mean time ratios0.467536/0.612318/0.916215.
  At8192,base512uses1.191739/1.242359/1.190181s versus base128's
  1.303205/1.296314/1.356206s. Single-write/omitted-reader U fails its own
  repeated-gain gate at every shape:ratios at8192are0.998692/0.962194/1.024334.
  Its winning netUnique ratios inherit the capacity change;do not adopt U from
  them. Retain all27samples/9contrasts and3CSVcapacity milestones. These Mrows
  are not model tokens;physical hardware saturation and model gain are unknown.
  Owner-management-plus-batch sums exclude GPU buffer resets,allocation and
  numerical reference. DU/DVsource review supports fairness/lifetime/count
  gates with that label correction;Main verifies39ordered case receipts.
  Its collection envelope/normal-exit failure remains failed,independent of
  the successful GPU cohort. Before automatic outcome qualification,add a
  bounded ordered-prefix-to-receipt guard under a new controller plan.
  The closed `prefill_bank_capacity_v1` model comparison at`fb8e32cd3`
  changes only requested expert-bank capacity384->512 on the coherent14508ELF.
  Bank bytes rise1022361600->1363148800(+325MiB). Three stage-progress64K
  requests admit all48fullfit packed banks and the full65536-row GPU residual.
  Then12clean processes/36requests supply three alternating matched pairs per
  32769/65537actual input length,each with three thinking-on24-output requests.
  All13processes/39declared requests close normally0,quiet,without fresh faults;
  all15final full model hashes match. Sources,math,same queue,chunk8192,decode
  placement,MTP,requests,actual arithmetic/jobs and requested H2Dbytes match.

  | Input | Warmed384 PP | Warmed512 PP | Paired512/384time ratios | Repeated-gain gate |
  | --- | --- | --- | --- | --- |
  | 32769 | 720.674 | 730.386 | 0.986702 /0.985230 /0.983671 | passes for this length |
  | 65537 | 889.430 | 891.765 | 0.951563 /1.012040 /1.005266 | fails |

  PPcolumns use the median of three process medians,each from two subsequent
  requests. Paired time geometric means are0.9852002622at32Kand0.9892487378
  at64K;these differ from ratios of aggregate PP. Keep all individual first/
  subsequent samples and client TTFT in the linked result. Warmed aggregate
  TTFT384->512is45.6485->45.043s at32Kand73.8675->73.7085s at64K.
  Thus32Kpasses its prospective1.48%paired-time improvement screen;retain512
  as a per-length development candidate for further gates. At64Ktwo eligible
  pairs worsen1.20%and0.53%;the slight aggregate gain does not pass the
  all-three-pairs rule. Preserve this failure and the original criteria. No
  global inference selection changes or sustained decode/task qualification
  follow. The earlier871..872/1006..1008warmed32K/64Kcohort remains separate.
  Eight first/warmed reference/candidate milestones are in the604-row CSV.
  Adjacent warmed time slopes are0.861033630(384)and0.873597717(512)ms/token.
  Two lengths do not validate a time law or physical fixed/per-token costs.
  Closed capacity logs,journals,receipts and job-private JIT caches are retired:
  890files/137067798B,after exact767controller and2361native/dependency
  source recovery,recipe/12ELF matching and13normal/reaped/empty owner checks.
  Initial post-cleanup admission passed with27plan/config/proof files;those
  files are now retired too,after original767source recovery and reconstruction
  of all26config/plan files and the environment/model maps from the retained
  f966reference. No capacity private root remains. The temporary349568B
  first inventory and27764Bfinal joint inventory are retired. Neither phase engine
  generation nor a model asset was removed.

  Source window_rows=81920and observed48windows/zero permanent hits explain
  why32K/64Kdo not exercise reuse across MoEwindows. Requested transfer bytes
  do not decrease;physical saturation and the gain mechanism remain unknown.
  Read-only snapshots find CPU affinity0..11,no exposed CPU quota and GT0at
  its2750MHzconfigured/rp0limit. Host-facing upstream ports report16GT/s,x4;
  leaf/internal-port2.5GT/s,x1fields alone are not an external-link ceiling.
  These snapshots do not explain the difference from historical best speeds.
  DWsource audit/DXnative Luna-max review finds no fresh-run blocker;its
  independent replay-validator caveat remains. Main reconciled omitted helper
  bodies with all767committed input pins. The existing full-context diagnostic
  is among the12coherent ELFs,and its standalone physical262144fill now passes
  below. Serve allocation/RESTORE and ordinary provider teardown remain
  unqualified. Future capacity work needs a fresh coherent-source plan;the old
  full_context_model.py pins retired6e40
  inputs and131072resident KV and must not be rerun unchanged. Standalone
  consume-prompt evidence must remain distinct from serve allocation/state
  qualification. The fresh`coherent_context_model.py`controller targets the
  retained diagnostic ELF with original BF16-prefix telemetry and resident20480.
  The seven host reader controls and same-build physical bounds/hash-order
  fixture pass with normal0,closed owners and no GPU work or fresh faults.
  V1's16385-input GPUcase exits1normally at the checked mixed-scratch guard,
  before layer-major math:the borrowed standalone buffers lack the server
  lease's checked ownership while reference MIXED_SCRATCH=1was retained.
  The owner is reaped/empty,validation activates and no fresh fault appears.
  Preserve this configuration failure;V1cannot advance to full fill.
  A fresh standalone allocation adaptation must disable borrowed mixed scratch
  or establish its owners;neither is a production speed/decode selection.
  V1's full case remains unrun;the later V2requires262143batched rows plus final checkedT1.
  The prospective V2uses lease0/mixed-scratch0/split-accumulator0together:
  the native split guard requires mixed1,and borrowed standalone storage has
  no server lease. This inseparable diagnostic unit selects the original512
  bank and independently allocated mixed storage or fallback2,while preserving
  BF16math/same queue/chunk/decode flags. Its reader checks the actual window
  tuple against native allocation formulas and requires numeric release telemetry.
  V2's nine host reader controls and same-build134bounds/9459hash cases pass
  with normal0and closed owners. Its16385-input/4-output standalone case now
  closes normally0,with quiet validation and no fresh faults;actual16384BF16
  batched rows,ordered-exp prefix,512bank,window/storage and release checks pass.
  V2then completes the physical262144-input/0-output diagnostic:262143batched
  rows plus the exact finalT1commit/wait/refill-drain marker. Native exit0,
  reaped/empty owners,quiet validation,unchanged boot/driver,no fresh faults
  and all15post-case model hashes match. TotalBF16residual5368688640Buses
  114688GPUrows/2348810240Band147455hostrows/3019878400B;the original512bank
  uses1363148800B,prefix538968064B,and actual max single allocation10132807680B.
  Contiguous mixed storage cannot fit its remaining budget,so the declared
  mode2/16384rows/0mixed-bytes fallback runs. Post-release free memory is1038MiB.
  This proves the standalone capacity subgate,not full serve allocation,
  RESTORE,numerical/task,current-decode-preservation or ordinary provider
  teardown qualification. Its synthetic traced540737.9ms is speed-ineligible;
  do not compare it with bench.sh PP or add it as a performance CSVmilestone.
  The0-draft summary displays-nan ms/round from division by0;guard that display
  at the next native-source change,without treating it as a model-state result.
  The one-variable bank comparison is closed (`prefill_bank_reuse_v4`,
  controller `1ab5f0465`, result `516d8dbc2`). It uses the original coherent
  14508 ELF, BF16 math, same queue, chunk 8192, decoder placement and MTP.
  Only requested prefill bank capacity changes between 384 and 512 slots.
  Seven processes and all 21 thinking-on requests finish normally with exit 0:
  three diagnostic requests and 18 clean samples in three alternating pairs.
  Owners are reaped and empty, validation is quiet, no fresh fault occurs,
  and all 15 final model hashes match. All 98,304 residual rows stay on GPU.
  Diagnostic admission confirms 48 full-fit 512-slot banks and 96 windows
  with at most 81,920 rows. Paired requests, arithmetic, jobs and maps match.

  | Scope | 384 PP | 512 PP | 384 client TTFT | 512 client TTFT |
  | --- | --- | --- | --- | --- |
  | First request; median of 3 processes | 818.551 | 823.326 | 120.337 s | 119.611 s |
  | Subsequent; median of 3 process medians, each from 2 requests | 906.501 | 893.041 | 108.6555 s | 110.307 s |

  Warmed 512/384 time ratios are 0.996206354, 1.009424935 and 1.018651994;
  their geometric mean is 1.008052285. Two pairs worsen by 0.942% and 1.865%,
  so the declared repeated-gain gate fails. All nine paired requests reuse
  5/10/9 more experts and request 11.648/23.296/20.9664 MB less bank H2D.
  This small reduction does not establish a speed benefit from the extra
  325 MiB bank. Requested bytes are not physical bus or saturation evidence.
  First-use time ratios, 0.990952812/0.972870418/0.995163048, improve in all
  pairs (geometric mean 0.986281162). Preserve that observation separately;
  first-use adoption needs a new prospective plan and remaining phase/model
  gates. It does not change the failed warmed-throughput decision.
  Default selection is unchanged. Sustained TG, task/agent quality, decode
  preservation, serve RESTORE and ordinary provider teardown remain unqualified.
  The earlier 32K gain, 64K failure and faster historical 871/1006 PP cohort
  stay separate; one new 96K point does not fit a physical timing law.
  Four first/warmed milestones bring the CSV to 608 rows.

  Original failures remain in their committed results: V1's two synthetic
  ordering errors; V2's 128K admission of only 98,304 GPU and 32,767 host rows;
  and V3's reader requiring a marker that native split0 does not emit.
  Their supervised termination is distinct from normal-exit qualification.
  V4 corrects only that reader/fixture contract and passes nine controls,
  including actual whole-request telemetry. EA/EB review withdraws its claimed
  missing job/map/dequant gate after inspecting the complete imported helpers.
  No other concrete reader mismatch was found.

  Ended V2/V3/research captures (221 files, 23,938,423 B), earlier bridge
  inputs (20 files, 1,575,123 B), and closed V4 output (497 files, 74,462,349 B)
  are retired after original source/input/recipe recovery. Their temporary
  inventories are retired too. The shared native/model basis admits without
  those private roots. The closed V4 CLI and experiment-specific test restore
  from `1ab5f0465`. Six maintained route controls keep only the 1,402 B actual
  excerpt they consume, alongside the shared reader and model basis for the
  named next proposal. All six pass at cleanup source `b8caac7f7`, using
  `python3 -B -m unittest tools.test_prefill_bank_route` with the pinned model
  Python and committed measurement environment. This bounded host check runs
  no GPU work and does not relabel the model source or its failed warm gate.
  Both phase reference engines and model assets remain current consumers.

  **128K bank-capacity unit completed:** `prefill_bank_placement_v2`, controller
  `f12afed85`, same native14508 ELF; original full result `da393c9b5`.
  Only requested capacity384 ->256 changes. Automatic residual/bank/source-view
  placement is its intended consequence; math,kernels,chunk8192,window10,same
  queue,split-request1(inactive),decode1256,resident20480 and MTPspec4 stay fixed.
  All nine processes/27thinking-on requests close normally: three traced
  diagnostics and six clean processes in three alternating adjacent pairs.
  Validation is quiet,no fresh fault occurs,all15final model hashes match,and
  original native identities are absent. Eight host controls and two frontend
  probes pass. V1's incorrect increased-budget negative control stays failed
  in `57df0c83c`; no GPU ran in V1. EC/ED source review and Main's additional
  provider/guard inspection find no source blocker under the actual runtime gates.

  | Scope | Requested384;actual512 | Requested256 | Client TTFT384 ->256 |
  | --- | --- | --- | --- |
  | First;median of3independent processes | 697.072 PP | 852.618 PP | 188.337 ->154.012 s |
  | Subsequent;median of3process medians,each2requests | 755.530 PP | 920.432 PP | 173.722 ->142.6845 s |

  Candidate/reference warm-time ratios are0.820842616/0.819991384/0.807385529;
  geometric mean0.816049899(18.395%less time). Every pair improves.
  Process PP ranges are751.911..761.388 and916.974..943.029 respectively.
  All nine paired mechanism checks pass. Actual reference placement is98,304GPU
  and32,767host rows;candidate all131,071BF16rows/2,684,334,080B stay on GPU.
  Its681,574,400Bbank frees650MiB,retains512MiBheadroom and73,379,584Bmore
  observed budget;actual allocation succeeds below10,132,807,680Bsingle limit.
  Both use96windows,max81,920;per-arm GU/Down/job/map conservation passes.
  Source-derived reference weightH2D is50,292,326,400B,the sum48*512*per-layer
  blob_bytes,not padded maximum stride. Candidate logged requests are
  43,611,008,000/43,839,692,800/43,663,718,400B per nonce;physical bus traffic,
  overlap and hardware saturation remain unknown. Layer-source repacking is
  removed;this does not prove absence of internal runtime staging or CPU polling.

  Performance and mechanism gates permit further model qualification;the128K
  1000PPtarget is not met,and inference/defaults are not adopted. Four CSVpoints
  retain first/warm results;prior32/64/96Kcohorts and the historical64K1006PP
  point stay separate. Do not fit a physical timing law across different routes.
  Current attribution `prefill_resource_profile_v2` is closed at
  `1dfadd0d6`:four processes/eight thinking-on24-output requests,all normal0,
  closed owners,quiet validation,no fresh fault and15final model hashes.
  Eleven host controls,the exact62-class fixture and the separate32Ksmoke
  passed. The128Kclean/profile/clean bracket bounds prompt/TTFT observer
  ratios to1.07975..1.095214,below the predeclared1.2limit. Both six-clock
  model traces match per-requestGU/Down and CPU/status conservation. The
  originalV1 remains failed;ordinary DownScatterKernel was classified as
  other. Reuse coherent14508 and the unchanged Sysman0collector;no inference
  build,math/default change or speedup/adoption is implied.
  Fresh clean first-usePPis820.8/810.3;subsequentPP887.8/884.8. Two CSVrows
  retain their process aggregates;prior128K920.432and64K1006+cohorts remain
  separate best evidence. Do not fit across these timing periods.
  In the observed128Ksubsequent client window,kernel union is114.755s;
  QSAprefix service30.450s,other kernels30.389s,andM2Dcopy service22.575s
  for55,305,417,464requested bytes. These are overlapping service/payload
  quantities,not additive wall time,wire traffic or hardware saturation.
  MainCPU153.999s comprises87.461s in known waits and66.538s outside;
  host row grouping is0.308s. Submission and polling are material;this does
  not establish a CPU critical-path limit. The actual limiting arithmetic/
  VRAM/PCIe resource and compatible ceilings remain unresolved. Next pin/read
  the dominant QSAprefix and copy paths and measure matching kernel/memory
  ceilings before selecting one independent change. The separate `transfer_refresh`
  result already retains2MiB and335544320Bordinary/Host-USM copy samples on this
  boot;match the actual called wrapper,shape and environment before reusing it
  as a ceiling or commissioning another transfer run. Keep the current-build
  five-length/held-out timing curve pending as described below;two lengths
  and mixed routes/timing periods do not qualify a physical law.
  The native14508 symbol resolves to cached-Q/vec4K/vec4V/ordered-exp,
  score stride33,32-key tiles and128-thread groups;those optimizations already
  run. V1preparation failed before tests/GPU because its model compact reader
  required an irrelevant model-assets field;V2uses a component-specific reader. A new bounded `qsa_prefix_calibration_v2` plan reuses its same-build
  fixtures to compare only ascending denominator sum versus subgroup sum,
  keeping operand staging,placement and scheduling fixed. Retain all three
  shape outcomes and108timings;the earlier32/64Kmodel gain gate remains failed.
  Component success can only permit a separately committed128Kquestion,
  not model launch/adoption or a physical bandwidth/saturation claim. No new
  native generation is needed. Independent placement/window candidates and
  their combinations require their own smallest coherent comparisons.
  V2now closes:8host controls,traced small checks,11numerical/boundary cases
  and3fresh timing processes,all normal0/quiet validation/no fresh fault.
  All108timings pass their declared floors. At cap2051,gather process ratios
  are0.994402/0.994095/0.993048(geomean0.993848),clustered0.988254/0.990184/
  0.989057(0.989165). Both large shapes pass repeated gain;cap33fails because
  its third ratio is1.001629. Preserve that loss and the old failed model gates.
  Model128Kcomparison is eligible only under a new committed plan;none ran.
  The limiting arithmetic/SLM/VRAM/cache resource remains unresolved. Same-work
  gather/clustered intervals differ materially;requested-read upper bounds
  and Intel's published380GB/s are not measured physical traffic or saturation.
  Two reviewed source references are indexed;the stride34/258proposal is
  blocked by8-byte matrix-stride requirements. All218ended private files
  (21,116,664B)and62,136Binventory are retired;shared admission/all12ELFs pass
  afterward. Source/inputs restore from48139c9ef/9b3efe10b;all108times remain
  as54ordered pairs without repeated case fields. Three CSVpoints bring the
  index to617;full result/reference evidence is2da1e8623. No newnativegeneration.
  Next derive the MoEwindow/workspace capacity from the remaining original
  layout/admission dependencies before selecting a larger-window implementation.
  Measure an independently admissible bank change at fixed window,then a window
  change at fixed admitted bank,then their combination;retain infeasible cells
  and each speed loss. Do not assume that bank128alone frees enough space or
  that131Kfits. Keep current decode and both prefill strategies unchanged.
  Current-build five-length/time/TTFT and full-model qualification remain pending.
  The next source proposal `prefill_window_workspace_v1` moves only gathered
  rows and router logits behind a distinct FP32 accumulator-tail allocation.
  At131071rows,it needs729,799,424extra bytes;existing-arena planes plus mixed
  tail need1,177,025,536B,below the source GDNregion lower bound1,180,698,624B.
  With bank1,the earlier observed free-budget scenario leaves22,492,160B
  beyond the unchanged512MiBheadroom. This is arithmetic,not actual admission.
  The ordinary/split public methods and their row limits remain unchanged.
  The opt-in long-prefill caller is built coherently in V2;component checks
  pass and model checks remain unrun. Mode16uses
  the explicitly requested bank and one admitted90113..131072row window;
  unsupported lengths/budgets retain ordinary mode10 and its host-tail/full
  context fallback. Mode10plus`STRATA_SYCL_PREFILL_MOE_RELOCATE_JOB_BUFFERS=1`
  isolates relocation at the unchanged split128bank/window for the98Karm.
  All settings are fixed before exec. The extra owner covers FP32tail,gathered
  and logits;checked outer retirement precedes its free,and all three guards
  are checked after model work. Producer/consumer spans bind the same owner.
  No arithmetic,chunk8192,QSA,decode placement or U1implementation changes.
  CPU capacity now passes all49,152row cases81921..131072,old-layout
  invariants,two-plane separation,guards/chunk maps and negative refusals.
  The preliminary CPUproof establishes capacity arithmetic;its closed output
  is retired. Fresh same-build component results are below. Next independently
  compare relocation,larger-window and combination arms;the bank-only loss remains.
  Do not drop the positive U1eager-upload component:its32/64Kgain was measured,
  and a separate period's lower absolute PP does not refute that matched gain.
  The original owned layout has source limits117488split,
  81920ordinary accumulation and8192per job,not runtime admission. The proposal
  pins original moe_window.inc and expert_bank_plan.hpp;
  assess bank-only/window-only/combined arms separately and retain infeasible
  cells. Retained native14still rejects mode16. The new source proposal has
  no model qualification. Extended route checks pass48
  cases through131072rows,stable inverse/order/no-atomic uniqueness and old
  factory refusals. Pure CPU CMake targets cover
  routes and workspace. The existing Down fixture gains explicit long-prefill
  tiny,full-span/dataflow and production-dispatch cases,including2333/2334/2335/
  8192job rows,65536bank boundary and131072final row;startup arms are separate
  processes with immutable environment. Their component safety checks pass below.
  Every affected TU is rebuilt.
  `prefill_window_build.py`pins an explicit14508-plus15owned-overlay composition,
  preserving U1and the byte-identical9849native-job capacity safety check.
  It rebuilds all objects and checks actual compiler/link inputs and the
  Gemm/MoeAccumPlan/MoeWindowRoutes definitions across producers/consumers.
  Its eight targets include CPUproofs,bank/Down safety fixtures,a matching
  resident-GEMM resource reference and the physical-context diagnostic pair.
  The initial preparation at`aeb671619`refused before creating a build root:
  two U1headers contain the later`f13b3baf`cleanup of rejected U2,while the
  device fixture still matches measured`51ea99cae`. The corrected prospective
  controller pins the retained source and verifies exactly the removal of
  the coalesce=false argument and its unreachable branch. Original U1samples
  keep their original source identity;no build/device qualification is reused.
  The bank-only cohort is closed. V1at`064ffb767`configures normally0,but is
  closed before compilation after review finds its census omitted the two
  pinned AOTfinalization/host-unbundle products. The prospective V2controller
  accounts for those products,their actual Ninja recipe,relative link paths
  and private temporary objects;missing external inputs still fail. Original
  configure success and the observer hold remain. V2at`3a195f247`now builds
  all eight targets normally0,in271.552s,with133fresh objects/130translation
  units,1468actual compiler inputs,29external link inputs and no uncovered
  dependency. All six Gemm includers and the accumulator/route producer and
  consumer definitions are coherent;both AOTfinalization products are bound.
  Engine SHA256is`b7c1d818c01264f6a6ceb17e4744a8c7a64054baa70c995787af0fbe2146638c`.
  The source is the explicit14508-plus15overlay composition,not the entire
  later tree. Owners are reaped/empty;model checks remain unrun. The live tree also
  contains independent KV/verify proposals,which must not enter a prefill
  performance comparison implicitly. Preserve the9849decode
  reference;its implementation/performance/handoff remain adoption gates.
  `prefill_window_device.py`closes all seven same-build checks at source
  `e247fbff4`:three CPUchildren and six GPUchildren,all normal0/reapedempty,
  quiet parameter validation,unchanged boot/driver and no fresh fault.
  Host coverage is48routes,49152layouts,134admission bounds and9459KVhash cases.
  Traced U1and tiny long-prefill pass;full bank checks cover0/1/127/512experts
  and cancellation. All six117489/131071/131072contiguous/irregular cases pass
  30bitwise original FP32FMA/BF16capture job checks,guards and terminal rows.
  Full weight-map and selected-job gather dataflow pass at131072irregular.
  Separate Down0/large processes each pass eight actual dispatch/fallback
  cases at2333/2334/2335/8192rows;the large path selects only at>=2334.
  Native16BF16/vllm/auto/original-split settings are fixed before exec;actual
  initialized providers and pre-main validation are verified. This is component
  evidence,not speed/adoption or physical262144qualification. Actual model
  GPUadmission,old-path coherence and each independent model comparison remain pending.
  All42closed successful build/component capture files(4,279,332B)and the
  16,868Btemporary inventory are retired. Exact build/device records,all eight
  ELFs and source/recipe evidence remain;fresh model admission passes afterward.
  `prefill_window_model.py`prospectively separates four27request cohorts:
  old-path coherence at128K,relocation at98K/split128,window length at128K/bank1,
  then the combination against128K/bank256/window10. Each has three traced
  admissions and three opposite-order clean process pairs;first/warm samples
  stay separate. Safe fixed-work coherence/layout losses remain recorded and
  permit a separate window question;only repeated fixed-bank window gain admits
  the combination. Coherence closes at source`e81f93750`:nine normal0processes
  and27thinking-on requests,quiet validation,no fresh fault and15final model
  hashes. All residual rows remain onGPU;maps,jobs,dequants,GU/Down selections,
  uploads and requested bytes match within each pair at bank256/window10.
  The three clean new/original warmed-time ratios are1.008021,1.012501and
  1.000776(geomean1.007088). The repeated-gain gate FAILS;aggregate warmed
  PPis921.087original versus915.134new. Keep all samples and first use separate.
  Do not adopt this as a replacement for the existing ordinary path or offset
  its loss with another unit's gain. Original14508remains the ordinary-prefill
  reference;legacy-path preservation/selection is an unresolved adoption hold.
  The prospective rule permits the separately scoped relocation/window questions
  after this safe fixed-work loss. It does not adopt the composed engine or
  qualify decode/task/physical-context behavior. Next compare98K/bank128/same
  window with relocation0/1,then128K/fixed bank1with window10/16. Relocation is
  prepared at`5d9b768f3`:all15model hashes and18generated configurations/plans
  are pinned;both arms use the same composed3a195fengine,bank128and window10.
  Its host checks and first32Kfallback process pass:three exact32769-read,
  thinking-on24-output requests,bank128/window10,ordinary layout/extra0,
  all32768BF16residual rows onGPU,normal0/reapedempty,quiet validation and no
  fresh fault. The first98Kreference also passes all three98305-read requests:
  98304rowsGPU,48windows,primary65536,335544576Bextra tail allocation and the
  original job-plane placement,normal0/reapedempty,quiet validation and no
  fresh fault. The relocated98Karm also passes all three requests with the
  same maps,jobs,dequants,GU/Down selections,uploads and requested bytes.
  Its394265344Bextra owner is actually admitted,and all three guards and final
  drains pass before retirement;all98304BF16residual rows remain onGPU.
  Native exit is normal0/reapedempty,validation quiet and no fresh fault.
  The full relocation cohort closes with nine normal0processes/27requests,
  quiet validation,no fresh fault and15final model hashes. All three clean
  pairs conserve maps,jobs,dequants,GU/Down selections,uploads and requested
  bytes. Relocated/original warmed-time ratios are1.012512,1.000491and1.006909
  (geomean1.006626);the repeated-gain gate FAILS. Aggregate warmed PPis942.723
  original versus935.818relocated. Keep first-use and individual samples;
  neither the placement change nor the composed engine is adopted. This
  preserves the separate bank/coherence losses. The prospective rule permits
  the next independent128K/fixed-bank1window10/16question;only its repeated
  gain permits the combination. The fixed-bank window question is prepared
  at`3a520417f`,with15model hashes and18generated configurations pinned.
  Both arms use the same coherent candidate and bank1;only window10/16and
  its required layout differ. Host probes and the first32Kfallback process
  pass:three32769-read/thinking-on24-output requests,bank1,all32768BF16rows
  onGPU,ordinary32768window/extra0,normal0/reapedempty,quiet validation and
  no fresh fault. The128Kreference also passes all three131072-read requests:
  all131071BF16rows onGPU,bank1,81920maximum rows/96windows,extra0,
  normal0/reapedempty,quiet validation and no fresh fault. The larger-window
  arm also passes all three131072-read requests:729799424Bextra owner is
  actually admitted,all131071BF16rows remain onGPU,maximum131071rows and
  48windows,three guards/final drains checked,normal0/reapedempty,quiet
  validation and no fresh fault. Jobs fall38324/38507/38286to24709/24759/24721;
  dequants fall33606/33785/33584to19079/19107/19085. Requested expert H2Dfalls
  68.54..68.95GBto38.95..39.01GB;this is payload,not measured wire traffic.
  Traced warmed time149.73605sversus136.69505sis mechanism evidence,not the
  clean repeated-gain decision. The full window cohort now closes with nine
  normal0processes/27requests,quiet validation,no fresh fault and15final model
  hashes. All three opposite-order clean pairs pass the fixed-bank window
  gain gate:time ratios0.914916/0.908997/0.904083,geomean0.909321.
  Aggregate warmed PPis864.962reference versus951.557new;new process PPis
  952.034/951.557/949.647. Median warmed client TTFTis151.8305sversus137.9815s.
  The separately scoped combination is prepared at`86493d658`:15model
  hashes and18generated inputs pinned,same coherent engine in both arms,
  bank256/window10versusbank1/window16. Host probes and the32Kfallback
  process pass:three32769-read/thinking-on24-output requests,all32768BF16rows
  onGPU,bank1,ordinary32768window/extra0,normal0/reapedempty,quiet validation
  and no fresh fault. The128Kreference also passes three131072-read requests,
  bank256/window10,all131071BF16rows onGPU,81920maximum rows/96windows,
  normal0/reapedempty,quiet validation and no fresh fault. The combined
  candidate also passes all three131072-read requests,bank1/window16,
  all131071BF16rows onGPU,729799424Bextra owner admitted,48windows and
  three guards/final drains checked,normal0/reapedempty,quiet validation and
  no fresh fault. Jobs38324/38507/38286become24709/24759/24721;requested
  expert H2D43.61..43.84GBbecomes38.95..39.01GB. Traced warmed time is
  144.01965sversus137.41725s;this is not the clean repeated-gain decision.
  The full combined cohort closes with nine normal0processes/27requests,
  quiet validation,no fresh fault and15final model hashes. All three
  opposite-order clean pairs pass the declared combined-gain gate:time
  ratios0.963803/0.950117/0.956801,geomean0.956891. Aggregate warmed PPis
  911.365reference versus952.512new;new process PPis946.737/957.637/952.512.
  Median warmed client TTFTis144.1225sversus137.9395s. This closes the four
  prospective coherence/relocation/window/combined questions,not model
  qualification. No inference route is adopted and128K1000PPis not reached.
  A further experiment needs its own committed plan. The next source question
  revalidates64Kat fixed384bank/window10:original14508JIT,coherent3a195fJIT,
  and same-image U1eager. These isolate the coherent-source unit,the upload
  unit and their net result with three opposite-order clean process pairs per
  question. Three traced admissions precede the nine clean processes;each has
  three thinking-on24-output requests. The shared JITcontrols are declared
  prospectively. Preserve every unit/first-use loss;neither a net gain nor
  1000PPalone qualifies an ordinary route or current9849decode preservation.
  The CSV's best coherent64Kevidence is`f9662f7ab`:1006.832warmed PP,three
  independent processes and65.2565swarmed client TTFT. Positive U1eager
  evidence`410f6cb8d`uses an earlier same-image cohort;its64Ktime geomean
  is0.986223,but a32Kfirst-use pair loses10.3%. These are reference evidence,
  not new same-period comparisons. Use the existing three native generations;
  no build is needed. The closed window controller retires to its original
  commits;shared workspace/dispatch readers and this source question live in
  maintained tools. Source/helper/gate tests must be committed before running;
  generated configurations must be committed before host or native execution.
  Initial source`00556af9b`preparation stops before model hashing/root creation
  or GPUexecution:the old build/device readers also pin the retired window
  controller. Preserve that failed admission. V2reuses only the exact closed
  native/component records after recovering every original control from Git,
  requiring all live inputs unchanged and allowing only that named committed
  CLIretirement. Original strict readers and all native/fixture hashes remain
  unchanged;the controller-only reuse is not a new build or GPUqualification.
  V2is prepared at`c429e16d1`:773current source inputs,15full model hashes
  and24generated JSONpayloads are pinned;seven CPUgate regressions and both
  frontend probes pass. All three traced64Kadmissions close normal0/reaped
  empty with nine exact65537-read/thinking-on24-output requests,quiet
  validation,no fresh fault and unchanged boot. All65536BF16residual rows
  stay onGPU,bank384/window10/48windows/extra0. Actual GU/Down/job/map/upload/
  requested-payload work matches across all arms. Eager logs24unique full-fit
  layers per request,9178/9179/9160individual copies and checked final bank
  drains. The full64Kcohort now closes with12normal0/reaped-empty processes,
  36exact thinking-on requests,quiet validation,no fresh fault,unchanged boot
  and15final model hashes. Actual request/job/map/upload/payload work agrees
  across all three arms. The coherent-source unit FAILS its repeated-gain gate:
  warmed-time ratios0.984350/0.984745/1.001215,geomean0.990072;the last pair
  loses0.12%. Preserve that failure and the original ordinary path. Same-image
  U1eager PASSES independently:ratios0.983994/0.987560/0.976567,geomean0.982696,
  with all three paired warmed TTFTs improved. Its third first-use pair loses
  0.53%;do not claim every first use improves. The net eager/original question
  PASSES:0.968595/0.972495/0.977753,geomean0.972940. Aggregate warmed PPis
  881.993original/890.059candidate-JIT/904.537eager;eager process PPis
  904.537/907.461/902.061. Warmed client TTFTmedians are74.501/73.8235/72.648s.
  Net improvement does not adopt the failed coherent-source ordinary route,
  current9849decode or the whole model. No arm reaches the1000PPtarget.
  Full cohort evidence is`64f5b9899`;six first-use/warmed CSVpoints bring
  the cumulative index to643without replacing earlier milestones.
  All36samples,12native identities and the unchanged unit decisions remain
  compact with six shared actual-work cases. The773original Git inputs and
  24generated JSONpayloads are recovered exactly. The closed campaign CLI
  and its four decision tests retire to Git;reusable workspace/upload
  admission readers and their three host tests remain for the next curve.
  All845closed capture files(127,754,680B)and237,945Btemporary inventory
  are retired after the pushed recovery/curation decision. Shared build,
  device and original speed-basis records remain;no native build is retired.
  Fresh post-retirement basis/device admission passes for all eight candidate
  and12original same-build executables,current decode artifact and shared
  records;three reusable host regressions pass. This executes no GPUwork and
  establishes no new model/teardown qualification.
  Historical64Krequest hashes match all three current requests;different
  timing periods do not identify a regression cause. Keep the36samples and
  each separate outcome. Same-image eager-path input-length/time/TTFT
  characterization uses16K/32K/48K/64K/80Kordinary-route
  lengths and independent held-out56K(actual reads add one final token).
  Keep bank384/window10/chunk8192fixed;16Kis diagnostic,not long-input adoption
  evidence. Commit its controller,repetition/order,approximation/held-out rule
  and stopping criteria before preparation or new execution.
  The prospective controller uses traced16K/80Kadmissions followed by clean
  64K/16K/80K/32K/48K/56Kprocesses,three thinking-on24-output requests each.
  First use and the median subsequent2remain separate. Fit affine/quadratic/
  NlogNmodels to the five declared training lengths,choose by whole-length
  LOOerror with a training-spread preference for an adequate affine model,
  then report independent56Kprediction errors.56Knever changes coefficients
  or selection. Preserve competing models within the empirical spread;one
  process per length does not establish independent repetitions or a physical
  law. Same-length traced/clean work and the preceding64Keager work must
  match exactly. No native build or source-unit adoption is implied.
  `eager_prefill_length_curve_v1`is prepared at`6d71a5f63`:775committed
  source inputs,16generated JSONpayloads and15full model hashes are checked.
  Five analytic host regressions pass,including held-out independence and
  competing-model/spread handling. Both frontend probes pass. V1stops before
  GPUexecution:the first16Kclient exits2because the controller omitted the
  existing`--prefill-short-diagnostic`flag. Preserve the failed protocol/owner
  observation;separate reconciliation confirms its sole child was reaped and
  no session member remains. No native engine starts and no curve sample is
  qualified. V2preserves the original lengths,order,math and fit criteria,
  adding only the required16Kdiagnostic argument and its real-parser test.
  All eight planned commands pass the actual CLIparser;all six host tests
  pass at`8b490fe33`. The775original Git inputs and16generated JSONpayloads
  are recovered exactly before retiring24failed-client files(173,835B)and
  the6,859Btemporary inventory. The original failed observation remains
  unchanged and recoverable at`cc9814fe9`;ownership reconciliation is separate.
  Corrected V2is prepared at`a9a90b203`:775committed source inputs,
  16generated JSONpayloads and15full model hashes are pinned again. No
  native build changes;both frontend probes pass. Both traced admissions
  close normal0/reapedempty with six exact16385/81921-read requests,
  all BF16residual rows onGPU,bank384/window10/extra0,quiet validation and
  no fresh fault. The full curve closes with eight normal0/reapedempty
  processes/24thinking-on24-output requests and15final model hashes.
  Traced/clean16Kand80Kwork and preceding64Keager work agree exactly.
  Clean warmed prefill seconds at16/32/48/64/80Kare
  31.52345/44.03335/57.73680/72.50165/88.35810;reserved56Kis64.37670.
  Actual reads add one token. With`x=actual_read_tokens/65536`,the selected
  warmed prefill approximation in seconds is
  `T=20.065975+43.561341*x+8.862286*x*x`;warmed client TTFTis
  `T=20.248736+43.537613*x+8.921143*x*x`.
  Whole-length LOOmean/max errors are0.1208/0.3474%for prefill and
  0.0376/0.0588%for TTFT. Independent56Kpredictions are64.96824/65.17530s
  versus64.37670/64.56850s observed:errors0.9189/0.9398%,larger than the
  training two-sample spread envelopes0.5338/0.4373%. Do not turn this
  descriptive fit into a precise physical law or confidence interval.
  Initial-request prefill and TTFTselect affine models:
  `T=27.970634+56.773760*x`and`T=28.121433+56.816400*x`seconds.
  Their LOOmean/max errors are3.9288/6.0379%and3.9184/6.0326%;reserved56K
  errors are1.7693/1.7685%. Preserve all three candidate models per series,
  residuals,adjacent slopes and the original selection rule. No56Kvalue
  changes coefficients or selection. One independent process per length
  characterizes this fixed ordinary route only;the fitted intercept does
  not identify a physical fixed cost and the curve must not be extrapolated
  past80Kor mixed with another placement/window route. Current64K/80KPPis
  903.938/927.148;this is not a paired gain,1000PPor model qualification.
  Full curve evidence is`90a1bc130`;12first-use/warmed length points bring
  the single CSVto655rows,with24-output TGleft unqualified.
  All24samples,eight native identities and original fit/qualification
  decisions remain with18shared actual-work cases;every route reconstructs
  exactly. All775original Git inputs and16generated JSONpayloads recover.
  The closed controller and its CLI-specific test retire to Git;five reusable
  fit and three workspace host tests pass. All565ended capture files
  (85,171,178B)and160,597Btemporary inventory are retired. Fresh static
  basis/device admission passes after retirement,without GPUwork.
  This develops the faster owned candidate while its source-selection and
  phase-preservation gates remain pending. Retain both phase references and
  the candidate for the next bounded64Kclean/observed resource attribution;
  commit its observer-cost plan and collector admission before execution.
  Its first gate is`eager_pti_probe_v1`:control then collector on the exact
  retained eager-bank fixture,4experts/256and512strides/2layers,normal and
  cancellation cases. Pin original collector/source/provider identities,
  checked drains,quiet validation,normal closure and enclosing RAW clocks;
  no model launch until that gate passes. It is prepared at`98e1ac6b3`,
  with776committed inputs and three host regressions passing. Control and
  collector both close normal0/reapedempty with both original numerical/
  cancellation cases,quiet validation and no fresh fault. The single trace
  has positive GPUintervals inside the actual enclosing RAW lifetime;
  source/fixture/collector/provider hashes remain unchanged. This qualifies
  the bounded collector probe,not model overhead,complete service/count
  coverage,physical saturation or ordinary teardown. No native rebuild.
  All776original source inputs recover exactly. The15ended probe files
  (381,237B)and4,636Btemporary inventory are retired;only its full qualification
  record remains for the immediate64Kprofile reader and retires with that consumer.
  The next model plan is four fresh64Kprocesses,clean/collector/collector/
  clean,three thinking-on24-output requests each,matched to the closed
  curve's exact bodies and bank384/window10/chunk8192/eager/throughput
  configuration. Only device collector injection varies. Keep first use
  and the subsequent2median separate;both paired ratios for prompt,client
  TTFTand total duration must remain0.9..1.1for the observer-cost screen.
  Preserve failures and perturbed observations;this is no speed-adoption
  gate. Use the qualified device-only settings,without API trace,pause,
  hardware metrics or a native rebuild. Prospective limits are1GiB/4million
  events per trace,1280MiBper process and5GiBcampaign. Analyze closed native
  runs serially with explicit CPU/RSS/time bounds. Retain kernel-family/copy
  services,requested bytes,interval unions/overlap and boundary crossings.
  Whole-request gates include24decode tokens;they do not establish a drained
  prefill critical path,physical PCIe/VRAM traffic or hardware saturation.
  The first model cohort used source `14191596d` with 783 committed inputs,
  eight generated plans and 15 model assets. Fourteen host tests and both
  frontend probes passed. The first clean process closed with 902.133 PP
  and a warmed client TTFT of 72.840 s. The first traced process closed
  natively with all three request bodies and routes verified, quiet validation
  and no fresh faults. Its host analysis was stopped because the 4 MiB output
  budget counted the existing 550,229,888-byte input trace. The cohort stopped
  at that failure. Its original outcome is retained at `8173b1aab`.
  A separate analysis run isolated output from the trace, then failed on an
  optional comparison argument. Preserve `7c1432307`. The corrected reader
  uses the verified request totals without this argument. It passed the 11
  host regression tests and recovered all three windows without GPU execution.
  In the two warmed traced requests, observed GPU intervals span 71.690 and
  73.075 s within 80.661 and 82.905 s external windows. The QSA prefix family
  takes 14.382 and 14.374 s of service. Copies classified by PTI as M2D take
  14.715 and 15.858 s for 48.338 and 48.302 GB of requested API payload.
  M means unknown memory classification. Copy/kernel overlap is 13.8 and
  14.7 ms. Kernel interval union is 55.024 and 55.136 s. The first pair puts
  warmed prompt overhead at 7.98%. The full observer-cost screen is pending.
  These observations put both transfer costs and the QSA kernel on the
  shortlist. Counter data for physical PCIe, VRAM and arithmetic limits are
  still unavailable. Keep the failed cohort and run the reverse order under
  a prospectively committed plan with isolated analysis output. The new
  trace/clean pair has three identical requests per process. Its 10% screen
  applies to the fresh pair. The earlier clean/trace timings are a descriptive
  cross-check and the original four-process qualification stays failed.
  Run the collector unchanged and preserve the same native build, parameters
  and trace limits. The separate analysis owner counts its own 4 MiB output
  tree. All 14 host checks pass at `fe99c5ac4`. The original 783 Git inputs and
  eight generated inputs recover exactly. All 162 closed capture files
  (571,700,347 bytes) and the 44,960-byte inventory have been retired.
  The fresh reversed pair closes with six exact thinking-on requests,
  both native exits zero, quiet validation, no fresh driver fault and all 15
  final model hashes checked. Clean warmed prefill is 72.433 s (904.797 PP)
  and client TTFT is 72.627 s. Fresh trace/clean ratios for first prefill,
  TTFT and total duration are 1.071699, 1.074036 and 1.084660. Warmed ratios
  are 1.070933, 1.073609 and 1.081274. All six are within the planned 10%
  envelope. The archived clean-first pair remains a descriptive cross-check.
  GPU event counts match the earlier trace for all three requests
  (229519, 229312 and 229131) with zero boundary crossings in both traces.
  Fresh warmed QSA prefix service is 14.376 and 14.383 s across 96 calls
  each. M2D service is 14.594 and 14.950 s for 48.338 and 48.302 GB of
  requested API payload. PTI explicitly classifies another 344 copies
  per request as H2D, taking 0.949 and 0.937 s for 3.578 GB each.
  The M2D source memory type remains unknown. Copy/kernel overlap is
  12.8 and 14.9 ms.
  For these 65,536 fresh batched rows, row `t` selects up to
  `min(t + 1, 2051)` cells. The 12 attention layers and two KV heads
  execute 100,638,720 tiles of 32 cells. The QK and PV matrix products
  each operate on 16 padded head rows and 256 dimensions. Counting a
  multiply-add as two operations gives 52.76 TFLOP, or 3.67 TFLOP/s
  over the measured compute-kernel service. Reading 256 BF16 values
  each for K and V per valid cell gives at most 3.252 TB of logical
  reads before cache reuse. Physical traffic and XMX utilization remain
  unmeasured. In `qsa_prefill_probability_v1`, the ordered-exp reference
  with 8,192 queries and width 2,051 took 224.05–224.21 ms on spread
  cell IDs and 139.68–139.78 ms on clustered IDs (three process medians
  per pattern). The generator changes the selected IDs, so the two inputs
  can encounter different missing pages and reuse. To isolate layout,
  measure the model's selected-row locality and compare equivalent physical
  layouts with the same logical IDs, values and reduction order. Include
  repacking in that comparison.
  To budget a V-only storage change, `H=24*sum(min(i,2051))` over
  positions `i=1..n-1` counts selected rows across both KV heads and
  twelve layers for a fresh prompt. At 117489 tokens, H is 5,732,774,712.
  Expanded BF16 V reads 512 bytes per valid row. Q8 V with four FP16
  scales would use 264 bytes if lanes share the scales, or 384 bytes
  with one scale load per four values. If all rows are valid, these two
  mappings avoid 1.422 TB or 0.734 TB of logical operand reads while
  adding 256 integer-to-float conversions, scale products and BF16
  roundings per row. At 65537 tokens the differences are 0.788 TB and
  0.406 TB. Actual DRAM bytes depend on cache reuse and coalescing.
  Keep K storage, ID order, softmax and matrix accumulation fixed in
  a comparison and include any V producer or addressing cost.
  The candidate at `146f71321` keeps the original K pack and BF16
  matrix operands. It records both the logical K row and the physical
  V row while looking up the page, using the same 256 bytes of local
  storage. Each four signed V codes share one FP16 scale; lanes convert
  the products to BF16 when populating the PV tile. Both scale checks
  remain in the original producer. The scratch allocation is unchanged.

  Run `bench/sycl/prefill_qsa_groups.py --q8-value` with the measured
  ordered-exp configuration. Compare ordinary packing, direct Q8 V with
  the original V pack still executed, and direct V with the pack omitted.
  The third arm must leave poisoned V scratch untouched. First test
  page size four, sparse/dead/tail rows, physical aliases, identity and
  remapped tables and all output bits and guards on the B570 with
  validation. A selected invalid V scale must poison exactly its head group.
  An active but unselected invalid scale in the identity table must set its
  row flag while preserving every finite result. Then run three clean processes
  with 6491 queries, 39260 live cells and selection width 2051. Use
  spread IDs on both layouts, clustered IDs on identity staging, and a
  causal identity case with 512 ascending four-cell pages and three tail
  cells per query. These synthetic selections bracket locality.
  In each case collect all six permutations of the three arms, including
  admission, packing, attention and the final drain. Require every clean
  interval to reach 10 ms and timer p99 below one percent. Record paired
  median ratios for the consumer, eliminated pack and combined change.
  Advance the combined identity-stage path only if the combined change
  lowers paired time in every process on all three identity shapes. Record the
  remapped result separately. Stop on invalid output, provider, fault,
  budget or ownership. Keep model PP, TTFT and decode checks separate.
  A separate candidate specializes selected-row lookup for the full KV
  staging pool. Its owner constructs an immutable identity page table
  and passes that table with the stage. The candidate selector requires
  the owner's table pointer to match the pool and both page counts to
  agree. All other calls use the original ordered-exp entry. The packer
  and caller-visible diagnostics keep the original implementation.
  Only the attention consumer skips its second page lookup for valid
  selected cells. The 117K geometry permits up to 5,732,774,712
  such lookups across both KV heads and 12 layers; invalid selections
  reduce that count, and physical memory traffic depends on caching.
  Compare the two loaders on the same initialized identity table and
  original K lane/workgroup order, including invalid IDs, the last live
  cell and a partial final page. Check null, unrelated and smaller
  resident tables select the original function. Require matching output
  bits, guards and three independent lower warmed medians on all three
  wide locality patterns before testing this stage-specific model route.
  The three independent processes at `3ec72cec2` passed the output and
  guard checks. With width 2051, candidate/reference time ratios were
  1.0196–1.0204 for spread IDs, 1.0275–1.0281 for clustered IDs and
  0.9988–1.0073 for four-cell groups. The three-pattern speed criterion
  failed. Keep the measured ordered-exp page lookup for the model.
  First test changing just the K loader's lane mapping with the existing
  packed layout. Four adjacent SIMD lanes read 32 contiguous bytes for
  one cell. This can reduce global address segments per subgroup but can
  increase conflicts writing transposed K to shared local memory. The
  isolated test retains QK/PV matrix operations, ordered softmax, V loads,
  publication barriers and packing. Compare exact full-output bits and
  guards on identical inputs. Measure both arms through the pack and
  drain over six alternating pairs in each of three processes. Include
  width 33 and 2,051, spread, clustered and sorted four-cell groups with
  invalid entries and missing pages. Report separate medians per shape.
  Require three lower candidate medians on both wide spread and four-cell
  maps, valid timers, quiet validation and no fresh fault before a model
  trial. These synthetic maps do not measure real cache-line traffic.
  Use the model's four-cell KV page for both arms and check its value
  in every case. A selected four-cell group occupies one page at this
  setting. The older probability fixture used 16-cell pages, including
  its 224 ms and 140 ms references; its times cannot serve as the
  four-cell baseline. Retrieve the reference interval from each pair.
  The lane comparison builds two isolated translation units against the
  measured engine's three static archives. Both arms use the original
  ordered-exp compiler flags. Run a traced three-query smoke first. For
  each timed process require all 48 samples with at least 0.5 ms per
  narrow call and 10 ms per wide call. The steady-clock median must be
  below 1 microsecond and its 99th percentile below 5 microseconds.
  Keep packing and the checked queue drain in each sample, with the full
  output comparison outside the clock. Retire the standalone fixture
  once its model-trial decision and source recovery are recorded.
  At `f7244dded` the four-lane K reader passed the traced smoke, output,
  input and guard checks. In three independent processes it shortened
  wide spread-cell time by 10.83–10.99%, but lengthened wide four-cell
  group time by 7.29–7.88% and clustered time by 2.93–3.11%. The planned
  wide spread and four-cell group gate failed. Keep the original K loader
  in the model and in the independent workgroup and KV stage trials.
  A separate QSA candidate at `f0b231aec` changes only how workgroups
  map to (query, KV head). The current launch alternates the two KV
  heads for each query. The candidate assigns the first `queries` groups
  to head 0 and the rest to head 1. Each head therefore has consecutive
  group IDs for adjacent queries sharing selected cells. The runtime
  chooses which workgroups actually execute together. Both variants use
  the original K reader, QK/PV tiles, per-query softmax order and output
  addresses. The three-query smoke exercises an odd group boundary.
  Use the same pack-to-drain clock and bit/guard checks as the lane
  trial in separate processes. On the four-cell page fixture measure the
  valid logical IDs shared with the preceding query, excluding duplicates,
  out-of-range cells and missing pages. Require at most 10% shared for
  the wide spread map, 40–50% for the sorted four-cell groups and at least
  90% for the clustered map. These are synthetic distributions. Advance
  the wide scheduling change only if all three wide maps improve in each
  of the three independent processes. The comparison at `124ffb26d`
  passed the output, overlap, guard and device checks. All three processes
  agreed on direction for each distribution. Across their paired median
  times, head-major/reference geometric means were 1.0040 for wide spread,
  1.0340 for wide clustered, 0.9606 for wide four-cell groups and 1.0175
  for width 33. The most shared IDs occurred in the clustered case, which
  slowed down. Keep the existing workgroup order on the model route. The
  four-cell gain is eligible evidence for a future locality-specific
  experiment if measured production selections warrant it.
  In each trace, 9717 D2D commands requested 128,861,904,128 bytes in
  0.7884 and 0.7880 s of GPU service. Removing BF16 input/output copies
  can also change submission or scheduling costs, but the measured D2D
  service alone is far below the 6.896 s improvement the clean 64K run
  needs. Give the separate QSA probability-reuse comparison priority after
  checking the four 32K bank combinations. Run its optional targets only
  while the GPU is idle and verify the original model and fixtures again
  before continuing the near-capacity model cases.
  In these two requests the expert-bank counters record 18,204 and
  18,182 uploads, requesting 37,173,094,400 and 37,137,228,800 bytes.
  Subtracting those totals from the corresponding M2D counters leaves
  exactly 3,013 commands and 11,164,427,264 bytes in each request.
  This is a cross-check on aggregate demand. The trace does not associate
  each copy with a call site or measure physical PCIe traffic.
  The first clean four-way BF16 block at `03a379bba` also establishes
  expert demand for layer 47 on the three 65537-token bodies. Its checked
  final bank occupancy was 260, 256 and 261 in all four processes. These
  counts are below the 383 permanent slots in the 384-slot bank, so none
  of this layer's experts needed its scratch slot. At 2,662,400 bytes per
  expert, this layer alone could issue 692,224,000, 681,574,400 and
  694,886,400 bytes before its first reader. Earlier issue keeps the same
  transfer volume. The model comparison still needs to measure its timing.
  The retained request summaries have no per-copy submission IDs or
  pointer classes to join to individual tensor uploads. The split
  accumulator submits both source and inverse int32 row maps. The two
  together request 251,658,240 bytes, about 2.25% of the difference if
  both were classified as M2D.
  The canonical weight upload comparison closed at `94a0649c8`. Both
  paths submitted exactly 1,455,214,080 bytes across the same 48 layers.
  Joining contiguous source and destination intervals reduced nonblocking
  UR copies from 672 to 217. Both traces verified addresses and ranges
  against the IQ3_S pack. Tensor readback, PLE bindings and full restore
  passed. All eight fixture processes exited normally without a new Xe fault.
  Clean per-process medians for the original tensor copies were 360.992,
  377.327 and 369.957 ms. Joined copies took 386.536, 393.459 and
  382.082 ms in the matched pairs. The three paired time ratios were
  1.07076, 1.04275 and 1.03277. Keep per-tensor activation and close the
  joined-copy candidate without a model trial. These component intervals
  measure layer activation calls. The transfer volumes are requested bytes.
  The original controller admission at `a931adbe7` stopped before any
  GPU trial. The corrected linker check matched the actual compiler-produced
  host and device inputs in the measured fixture.
  The two original IQ3_S shard directories, filtered by the native dense
  loader's matrix criteria, give 300 GPU weight owners across 48 layers,
  totaling 2,116,951,040 bytes. Per-layer packed payload ranges from
  30,423,040 to 50,918,400 bytes. Keeping layers 3 and 7 would occupy
  64,471,040 bytes during prefill and avoid 128,942,080 requested bytes
  if it eliminated both their activation copies and post-prefill reloads.
  Actual admission, graph ownership and duration for partial retention
  remain open. The layer-copy candidate above retains no extra weight owner.
  The earlier 117489-token, 1256-expert-slot run spent about 384–385 ms
  restoring all native packed owners. Its enclosing 650–817 ms also included
  canonical restoration. Use the inner interval when estimating savings
  for native owners. Neither interval establishes costs at 384 slots or
  admission at the physical context limit.
  Clean 64K needs 6.896 s less prefill time to reach 1000 PP. QSA prefix
  is the largest observed kernel family. Its service and the competing copy
  cost both warrant independent work. The next QSA component comparison
  should hold the WG128/SIMD16, 32-cell tile, typed K/V, BF16 rounding and
  ordered FP32 softmax fixed. Compare the current shared K/V local tile
  against a separate V tile with the same late gather, then compare late
  versus early V gathering with the same separate storage. The extra V
  tile costs 16 KiB per workgroup. Check the actual allocation and
  occupancy impact, zero-fill and tail rules, uniform barriers, and exact
  output and state conservation before timing. Start from the owned large
  cap-2051 gather and clustered production-shaped fixtures and include
  short/invalid/tail and full-context address cases. Commit source and
  repeated component criteria before building. The QSA timing and measured
  copy overlap do not identify a saturated hardware resource.
  The two warmed traces also isolate 20,591 and 20,568 mixed-input expert
  gather commands (2.001 and 2.002 s). In each, all 48 windows used a
  single resident input bank. The 125,829,120-byte int32 source map covers
  31,457,280 routed rows, each copying 2,560 FP16 values. The explicit
  gathers therefore request 150 GiB of input and 150 GiB of output per
  prompt, equivalent to about 161 GB/s of logical tensor work. Cache
  reuse, physical VRAM traffic and bandwidth saturation remain unknown.
  Only 4,217 jobs reached fused gate/up in either request, but they
  account for 89.0% of the routed rows. A gather-free input to those jobs
  could avoid about 133.5 GiB of intermediate writes per prompt. The
  resulting XMX reader cost and benefit to end-to-end prefill are untested.
  Child `6a4af5fd2` keeps the existing scalar gather and adds a component
  arm that copies eight FP16 bit patterns per work item. The operation
  retains invalid-row zeroing, sticky status bit 4 and four-byte base
  alignment. The new numerical cases cover one/two banks, nonzero route
  offset, 8,191-row tail and base offsets 4/8/12 modulo 16. Fixture
  `20f39e0dc` compares 64 distinct jobs per batch over the same 65,536-row
  bank and one reused destination. The synthetic job sizes of 211 and
  6,640 approximate average separate and fused jobs. Each uses spread and
  16-row clustered maps. Three independent processes alternate six pairs
  per shape. The timed interval covers checked submission through drain.
  Shape-specific advancement requires all three candidate median batch
  times to improve, with equal planned maps, numerical checks, outputs,
  normal exits, no new fault and no timer-floor failures. Keep each
  shape's individual results. Build the gather fixture with
  `bench/sycl/build_prefill_mixed_gather.py` against the verified
  resident-input build. Compile the two changed units with their original
  production and test flags. The header adds a free function. Recreate a
  private prefill archive with the new kernel object and the original
  eleven other members. Link only the fixture with that archive. Its next
  consumer is the traced numerical check followed by the independent
  scalar/vec8 timing comparison. Retire the fixture after that decision.
  At `071032b9c`, the traced 8,192-row and split 8,191-row cases passed
  bitwise, including invalid rows and slice offsets of 4, 8 and 12 bytes
  modulo 16. In three
  independent processes the 6,640-row vec8/scalar time ratios were
  0.7421–0.7422 (spread) and 0.7418–0.7466 (clustered). At 211 rows
  neither map won in all three processes. Keep scalar for small jobs.
  The opt-in `STRATA_SYCL_PREFILL_MOE_GATHER_VEC8=1` trial starts at 2,334
  rows, the lower edge of the existing large MoE job population; fusion
  selection stays independent. The default is scalar. The request footer
  records jobs and rows submitted to vec8. For the next fixture use the
  actual production selector at 2,334 and 8,192 rows on both maps. Keep
  the original three-process order, timing floor, six alternating pairs
  and bit/guard checks. Require faster median time per process for each
  of the four shapes before model timing. If it passes, verify that the
  same model binary in both arms selects the expected jobs on real inputs,
  preserves state and RESTORE, then compare clean 64K PP and client TTFT.
  Keep decode placement and kernels fixed and check sustained generation
  separately. The original component results establish no model speedup.
  The dispatch screen passed at `194e1df1f`. Across the three process
  medians at 2,334 rows, vec8/scalar ratios were 0.6999–0.7003 (spread)
  and 0.7905–0.8554 (clustered). At 8,192 rows they were 0.7589–0.7665
  and 0.7250–0.7388. All four configurations are in the performance CSV.
  The model projection at `c6bc02593` recovers all 1,171 source files
  of the resident-input engine from Git and changes only six gather inputs.
  The checked candidate image hash is `b5d2c3bc755bc6ae53062fd74679a1c0eaa7b19ec24496352206d6165fb0e599`.
  Its 12-target build recipe hashes to `4eeb51b69215bc2d2e0bea04e1676e7007e69fb119367d8532b7717714a817dc`.
  The next coherent model build includes two switches in the same binary.
  Applying `fea0f3c79` to that exact gather image changes only
  `src/prefill/prefill.cpp`. Its source image hashes to
  `466b03d720fa5dbff5a3855c154a6f8e45820d6c6b71a636570a2006789d1838`
  and the relocated 12-target recipe to
  `ee8be44d38b2badea4621a4e04259662ccc38fcc88ada6702dd9c72f79aa0fad`.
  Clean comparisons explicitly set the KV selector in every arm. At
  `fde0cdd11` each staged attention layer then reports its reload and
  reuse counts without the full stage trace. The measured prefix lengths
  and the retained GU fusion counts can therefore check both dispatches
  during timing. The four-way build adds this reporting to the pinned
  image above. Its 1171-file input digest is
  `f35b6b1a3f0c4f6317bb132caf43d01d19bc06acb26eb50ea4a8f2234e570ca2`.
  The measured parent retains its active consumers until they close. Build in
  a fresh directory after its retirement. Check both switches off against
  the parent's recorded route, expert and draft counts, KV state and output.
  Exercise the gather fixture and KV lifecycle checks before collecting speed.
  For the standalone final-slot check, use `bench/sycl/prefill_kv_capacity.py`.
  On the measured 16385-token copy control the window used separate mixed
  storage (mode 10). At exactly 262144 tokens it switched to mode 2 with
  147455 host-backed residual rows. Neither used the mixed-scratch gather.
  Keep the vec8 request on in both arms and verify this fallback. Compare
  only full KV reload against aligned reuse on the same executable. The
  32 chunks in each of 12 attention layers at physical capacity would
  request 51,489,275,904 prefix bytes with reload or 4,290,772,992 with
  reuse. Check the actual per-layer pages, host publication and final-slot
  marker. The completed standalone comparison at `5fb96bc69` matched
  both predicted volumes with 1488 prefix copies for reload and 124 for reuse.
  Both arms published 3,321,876,096 bytes to RAM through 1632 calls,
  consumed the last token with zero output, and closed without a fresh
  fault. The instrumented batched intervals were 475.300 and 491.205 s
  respectively, one sample per arm. The clean speed comparison uses
  independent processes. Server workloads separately exercise the vector
  gather and decode state handoff.
  For the leased server route the diagnostic sibling accepts
  `CONSUME_FULL <comma-separated token ids>` on stdin with MTP, the checked
  cache loan, payload audit and stage trace enabled. It requires exactly
  262144 tokens. The normal server keeps its output reserve. Park the same
  253919-token prefix with the short divergent request used below, then
  restore it with an 8225-token suffix. This leaves 8224 actual batched
  rows and one terminal verifier row. Check RESUME/RESTORE and the mixed
  bank, GU and scalar/vec8 counts in this last segment in each of the four
  KV/gather arms. Match its loan generation from mask through publication,
  audited expert payload and residency table. Require one checked terminal
  commit and drain, zero emitted or drafted tokens, a full state hash and
  DONE with zero outputs. Use different token IDs from the parking request
  for the next short GEN (e.g. 512..544) and require RESUME 0. Compare its
  QSA/GDN/PLE state and generation with that same short request in a fresh
  server. Check MTP ring invalidation or overwrite before read; STATE_HASH
  includes only the allocated draft extent. Exercise restore of the parked
  full-capacity image via one of its shorter saved checkpoints, then send
  a further independent short request and repeat the reset check. A full
  prompt cannot reuse an exact full-length prefix. An effective fallback
  or an earlier loan does not qualify mixed dispatch. Also reject ordinary
  GEN with zero output, ordinary GEN with a full prompt and positive
  output, and a short CONSUME_FULL. The diagnostic admission change at
  `fa4b9855a` rejects a selected 262143-token parked prefix before taking
  the entry: no rows remain between the prefix and the terminal verifier.
  Exercise this rejection and reuse the still-parked entry afterwards.
  The ordinary restore fixture disables turn and periodic checkpoints.
  This leaves a full parked conversation with no shorter restore point.
  `bench/sycl/prefill_terminal_server.py` sets the turn token to 4062,
  already the last ID of the original 253919-token body. Disable the
  optional root checkpoint and place the same turn token at the end of
  the full prompt. The two checkpoints now cover 253918 and 262143
  tokens without splitting the 8224-row terminal batch. Replaying the
  full prompt must reject the 262143 match before removing the image.
  A divergent ordinary prompt must then restore 253918 from that same
  262144-token image (SNAPSHOT_VERIFY upto=262144). Compare real MTP
  decode on both short reset prompts with independent ordinary-server
  processes. This tests observed state and draft behavior over those
  prompts. A general MTP ring overwrite invariant needs separate evidence.
  The leased server model and protocol checks remain open.
  Run `bench/sycl/prefill_kv_gather_restore.py` against the coherent build
  after its contiguous and ragged gather checks. All four combinations
  replay the original fresh 32769- and 253919-token prompts, switch to a
  33-token request, and restore the original state for an 8209-token
  suffix. Its first batched token starts at a non-page-aligned position.
  Check actual per-layer prefix reloads against the Q8 host mirror writes.
  With the fixed large-only GU kernel, its row threshold is the same as
  the vector gather's. Compare the independent GU and gather job and row
  counts, including the scalar route. Match state, decode-cache payload
  and output with the original engine and same-build control. The
  cancellation, external tap and physical-final-slot checks remain separate.
  On the original copy server (`4775de3b0`), replay of the traced
  32769- and 253919-token inputs found 1647/23317 and 6942/47991
  gather jobs eligible for the 2334-row threshold. Their restored
  suffixes had 115/18114 and 120/18039 eligible jobs. In those
  suffixes the original reload path requested 934,382,592 and
  6,539,157,504 prefix bytes respectively. Both suffixes start inside a
  four-token page, so the aligned candidate keeps these reloads. This
  provides a negative dispatch control and a separate optimization case.
  On the two-switch build, the fresh 32769-token request with scalar
  gather requested 622,854,144 prefix H2D bytes in 144 copies on the
  control and 51,904,512 bytes in 12 copies with aligned reuse. Both
  wrote 415,236,096 bytes to the host mirror. On the restored 8209-token
  suffix both requested 934,382,592 prefix bytes. The vec8-only arm
  matched the control's copies and used the vector gather on 1,647 jobs
  (7,937,266 rows) on the fresh prompt. The combined arm at `274931d19`
  preserved those gather counts and the KV-only arm's copy counts.
  All four arms passed the same state and handoff checks with no new fault.
  The counters measure requested copies. Clean prompt timing follows
  separately.
  A separate opt-in prefill candidate, `STRATA_SYCL_PREFILL_KEEP_KV_STAGE=partial-page`,
  reuses complete pages only after the preceding typed Q8 append in the
  same layer invocation. At a nonaligned successor it refreshes the one
  boundary page from the canonical host mirror using the existing four-plane
  range copy. Entry to a restored request still reloads the full prefix.
  With scalar gather fixed, compare `1` against `partial-page` in separate
  processes. In the second batched chunk of each restored suffix, the
  11 eligible layers should retain 10240 or 65527 complete pages per layer,
  saving 475,791,360 or 3,044,646,528 requested H2D bytes respectively.
  Both arms still issue four plane copies per eligible layer in that chunk.
  Check each layer's counts, all four boundary planes, active state, output and
  decode handoff. Require equal counts on aligned fresh prompts and full
  reload on layer entry, after cancellation and with an external tap. Verify
  the physical final slot separately. This candidate has no native timing
  or device qualification and is outside the pinned four-way build below.
  For scalar/reload (A), scalar/reuse (B), vec8/reload (C), and vec8/reuse
  (D), use the same ordinary expert copies, output-copy policy, ordered-exp
  attention, 8192 chunk, K8V8, model and decode placement. At each of
  65537 and 117489 actual tokens run independent processes in blocks
  A-B-D-C, B-C-A-D, C-D-B-A, D-A-C-B. Each arm receives the same thinking-on
  fresh prompt bodies. Separate first-use measurements from the median of
  subsequent requests. Compare B/A and C/A to isolate the two changes, then
  D/C and D/B to measure how each behaves with the other enabled. Report
  block-wise prompt-time ratios and client TTFT. Advance a throughput
  choice at a measured length if at least three of four paired prompt times
  improve and the median paired ratio is below one. Keep an inconsistent
  result open. Verify traced work and state outside the speed samples;
  stop on a changed input, ownership, fault, numerical or dispatch failure.
  Recheck sustained decode, agent sessions and full physical capacity before
  adoption. The two-switch build's completed 65K and 117K timing decisions
  are recorded above.
  Run `bench/sycl/prefill_kv_gather_speed.py` first at 117489 tokens
  (four original bodies per process), then at 65537 (three bodies).
  Each length has 16 independent processes. Validate the KV and gather
  dispatch counters on every timed request, with full tracing disabled.
  Match its body hash, placement, expert and draft work to the original
  control. Report each contrast, its four ratios and the absolute rates.
  In the two warmed 64K requests of `eager_request_profile_v2`, all
  scalar mixed gathers took 2.001 and 2.002 seconds of GPU service
  (20,591 and 20,568 calls). The count includes the small jobs that keep
  their scalar implementation. Eliminating all that service would cover
  about 29% of the 6.896-second gap from the paired clean 64K prompt to
  1000 PP if every saved second shortened the request. The model comparison
  must measure how much of this time the large jobs actually save.
  The current QSA selector emits the first one to four cells of each
  contributing block in ascending order. Only one threshold-tied block may
  be cut before the physical tail. The diagnostic 39,260-token request
  validated 6,491 rows per layer with a width of 2,051 across all 12 layers.
  One block ID per emitted group plus a row header suffices to represent
  these rows. The header gives the first rank after the threshold cut and
  the one to three skipped positions within that block. After this rank,
  adding the skipped positions before dividing by four gives the descriptor
  index directly. A last partial physical page needs no internal cut.
  The proposed 515-word row reserves 16.09 MiB at chunk size 8192;
  explicit 2051-word rows reserve 64.09 MiB. Neither this difference nor
  the smaller index reads is a measurement of free VRAM or saved K/V bytes.
  First check `bench/sycl/qsa_prefill_groups/host.cpp` against explicit IDs,
  including every internal cut position at width 2051, tails, guard and
  padding words, and ranks reaching the final physical page. Compile with
  `/usr/bin/g++-13 -std=c++17 -O2 -Wall -Wextra -Werror` (GCC 13.3.0,
  compiler SHA256 `1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769`),
  `LC_ALL=C`, the committed fixture, and a fresh private output path.
  The first GCC invocation at `1839471c1` rejected the nested padding
  lambda's reference to the locally captured sentinel. The corrected fixture
  uses the same sentinel value at file scope and keeps the original checks.
  At `e39aca1e0` the host check passed all 3,240 cases and reproduced
  3,231,810 explicit cell IDs in order. Full-width rows used 513 or 514
  group IDs within the 515-word row including its header.
  The executable SHA256 was
  `5a0a13ffbd16a5d717be7698bc023e1d740169be971364dcc84c264de8e5a357`.
  The host format check cannot qualify the selector or attention. A device
  comparison must include the real threshold emission, the same typed
  attention, exact selected-cell order, all four tail residues and measured
  producer-plus-consumer time before any model comparison.
  Source `ce7e64c4d` pairs the ordinary prefill threshold selector with
  the ordered-exp attention kernel. The candidate emits block IDs during
  the final selection scan and resolves each rank when attention reads
  it. Both arms allocate the original explicit-ID capacity. This first
  comparison isolates the representation and its indexing costs; a
  smaller allocation needs separate extent, alias and lifetime checks.
  The first component compiled at `342f02fcf`. Before device trials we
  found that its single selector submission differed from the measured
  caller's 256-query loop. Supersede its timing plan before sampling.
  In `5c02d5b0a` the identical kernels receive the real 65,538-word
  score-row stride and a separate active-block bound per slice. The last
  91 rows of the 6,491-row case follow 25 full slices. Both arms reuse
  the same synthetic score slab; timing begins after scoring.
  The closed 253919-token model process at `116a0e295` used
  `STRATA_SYCL_SELECT_RADIX4=1`, cyclic and early exit both zero, and
  no `STRATA_SELECT_OLD`. Its recorded child environment matches the
  planned SHA256 `0fa3ee921ad520ad9bd300e932c7750613536a488138646d7dc58cf9a4435066`.
  Both earlier component plans used Mapping0. The single-call variant
  compiled; neither variant supplied a device speed sample. Source
  `9639531fd` retains the model's radix4 threshold search in both arms.
  Recheck the model environment in the controller before each stage.
  At `9bab6f3c1` both revised native kernels and the sliced fixture
  compiled and linked against the traced build. The original single-call
  executable was retired after source and recipe recovery at `cc0fd8744`.
  Run the revised B570 component with `bench/sycl/prefill_qsa_groups.py`.
  Use the same in-order queue, four-cell pages and remapped/missing pages.
  Start with 16 queries at 97 cells and 259 queries at 401 cells,
  both with capacity 33. Check cross-slice output placement, empty and
  dense rows, partial ties, all tail residues, a one-word-short output and a
  stable CPU cell sort. At the diagnostic geometry, use 6,491 synthetic
  score rows ending at cell count 39,260, with capacity 2,051, and compare
  every selected ID, output bit and guard. The measured interval includes
  selection, packing, attention and completion. Run three fresh processes,
  six alternating pairs each, reversing first arm by process and round.
  Each interval must be at least 10 ms; timer p99 must be below 1% of it.
  Advance only with a lower median paired time in all three processes.
  At `4a8bc762e` every output check passed. The median of the three
  process medians for explicit IDs was 136.445 ms and for grouped IDs
  132.605 ms. The paired time ratios were 0.97083, 0.97182 and 0.96965.
  The component gate passed using the original allocation in both arms.
  This fixture cannot qualify a model fallback: if grouped attention
  declines after selection, regenerate explicit IDs before any consumer,
  including diagnostics. Score storage is reused every 256 rows, so replay
  of earlier slices also needs their scores again. Model routing is still open.
  The 117489-token ordinary control at `9c03b44d0` routed 56,394,240 rows
  on each fresh prompt (48 layers, ten experts per token). Across the four
  original bodies, jobs of at least 2334 rows carried 52,160,499–52,231,136
  rows, or 92.49–92.62% of the total, in only 7423–7441 of 23816–23841
  jobs. At 2560 FP16 values per row, the selected jobs account for
  248.72–249.06 GiB of logical gather output per request. Vec8 still writes
  that output. Only a gather-free reader could remove it.
  The mapped-input alternative now has a diagnostic at child `19999c6c1`.
  Its host mode checks the actual compact and wide TLA lane/atom
  coordinates, replicas and valid/padded rows over three of 80 K tiles.
  Its opt-in device mode compares the dense block2D load after reorder,
  the indirect FP16 read and the host address oracle, slot by slot using
  two bit tags. It exercises sorted and reversed maps, invalid rows,
  sticky status and a four-byte-aligned single bank. Run the host witness
  before the traced three-row smoke, then all shape/bit cases before
  considering full DPAS. Device build and execution are pending.
  The 65537-token workload gathers about 28 million eligible rows per
  request. Their temporary matrix writes represent 133.5 GiB of logical
  FP16 output. The reviewed `prefill-gu-indirect-load-traffic-20261120`
  scope identifies repeated reads across N workgroups and subgroup replicas
  as the main question for indexed A. After the bit test, compare dense
  materialization, indexed A and tile-local repack on the production wide
  shape. Time each arm from input availability through completed SwiGLU
  output. Use compact tiles to test the effect of more read replicas.
  The source inputs (783 for the pair and 776 for the probe), all four
  generated inputs and the final comparison decision were verified from
  Git before cleanup. The 152 capture files (571,845,791 bytes), 42,745-byte
  inventory and spent private probe record were retired. The closed launcher,
  probe and resource-analysis modes and their checks live in Git history.
  The shared workspace admission, PTI request reader and collector assets
  remained available after the resource comparison.
  The combined-window engine and seven matching fixtures have since been
  retired, together with their closed device receipts. The recovery binds
  1,166 Git files (14508 plus 15 owned overlays), 1,468 compiler inputs, the
  original recipe and 29 link inputs to the eight checked executable hashes.
  The removed generation contained 1,358 paths and 187,251,978 bytes. The
  independent original prefill and decode reference builds are retained.

  The V tile comparison compiled the unchanged ordered-exp prefix and two
  isolated derivatives. Shared local storage requests 20,036 bytes per
  workgroup. A separate V tile raises this to 36,420 bytes. The two derived
  kernels differ only in where the existing V gather runs. That earlier
  gather requires separate storage because QK still reads the shared K tile
  at that point. The separate-late arm isolates the storage change. The
  separate-early arm compared with separate-late isolates gather order.
  The first
  attempt failed before linking because the benchmark's legacy timing loop
  escaped its compile-time branch. The corrected fixture preserved both
  generated kernel hashes. The original failure is recorded at `1d598cb71`.
  The corrected build used nine fresh translation units, 725 actual compiler
  inputs and 25 link inputs. Both traced smoke processes and all 11 numeric
  cases, including full-capacity addresses, passed with bitwise output,
  checked online state, guards and no fresh fault. This is component coverage
  of the 262144 boundary.
  Three independent timing processes collected 162 intervals across three
  shapes. The table reports geometric means of the three process time ratios.

  | Selected cap and row pattern | Separate vs shared storage | Early vs late at separate storage | Early separate vs current |
  | --- | ---: | ---: | ---: |
  | 33, gather | 1.664699 | 0.986915 | 1.642917 |
  | 2051, gather | 1.671171 | 0.994635 | 1.662206 |
  | 2051, clustered | 1.801015 | 1.000032 | 1.801073 |

  The large gather gains 0.54% from early V within the separate-storage path,
  in all three processes. The other early-gather comparisons fail that
  repeated-gain check. Storage and combined comparisons lose at every shape
  in every process. Keep shared late V. The comparison shows a cost of the
  extra local storage and the changed placement within the kernel. Device
  counter evidence for achieved workgroup occupancy or memory saturation
  is still unavailable. Commit `94871b2c9` preserves the 162 samples and
  each comparison decision. The corrected fixture and recipe are reachable
  at `daca8f47e`. Verified retirement removed its 68 output files
  (12,212,498 bytes) and the temporary inventory. The original prefill and
  decode reference executable hashes were rechecked after removal.

  A separate candidate retains the two BF16 probability fragments used in
  the probability-times-V product across its four output tiles. In the
  retained 14508 reference ELF (`dcb8efd4`), the selected ordered-exp
  compute kernel is in the generic SPIR-V image. The current ordinary384
  ELF (`a3dddff2`) contains the identical 170752-byte module, SHA-256
  `e9144cf11a608a279c94d7588e7fec7787e19fcca8f5ddcdf3381c742edbadd7`.
  The probability matrix load remains inside both loops in that image.
  Its address depends on the cell offset, while the V address also depends
  on the output tile. This yields eight probability-fragment loads per
  subgroup per 32-cell tile in the intermediate code, with two distinct
  addresses. The final driver-generated instructions, register pressure
  and physical local-memory traffic are unknown. Test fragment reuse alone,
  holding workgroup size, barriers, V storage, accumulation order and
  tail handling fixed. Check actual generated code, numerical output and
  clean component time on cap-33 and cap-2051 cases before a model trial.
  The fixture and build definition at `45a1b7b77` check each normal
  export's complete fresh K/V pack before the other can rewrite it. The
  controller at `0ed10ef1d` requires the complete ordered compiler command
  for both kernels, including the device target and translator arguments,
  to match the checked original recipe apart from the two artifact paths.
  Both fixture executables built from the pinned compiler commands. The
  traced smoke and eleven numerical cases passed, including the packed
  inputs, BF16 output, all sixteen FP32 head states and 262144-cell
  addressing. Each of three untraced timing processes alternated six pairs
  on each shape, including pack, compute and drain. At cap 2051, reuse divided
  by original median time was 1.00198/1.00194/1.00065 for gather and
  1.00364/1.00317/1.00253 for clustered cells. At cap 33 the ratios were
  0.99878/1.01122/0.99390. All five GPU processes closed normally without
  a fresh Xe fault. Retain the original ordered-exp kernel for all three
  shapes. The reuse candidate did not qualify for a model trial.
  Skipping padded-head ordered softmax already had a separate 72-sample
  test at `24d666972`. It zeroed the unused probabilities and alpha once
  before the tile loop. Wall-time ratios were 1.063/1.056 at 32K and
  1.056/1.045 at 117488 cells (contiguous/strided). The current measured
  source adds only an unrelated cached-subgroup export to that original
  translation unit; the ordered-exp template and its entrypoint match.
  The November proposal to publish zeros every tile has no demonstrated
  advantage over that already tested variant. Defer it pending a distinct
  mechanism or shape-specific evidence.
  In the measured bench ELF (`eeb1a47d`), the smoke trace binds each UR
  kernel name to its launch handle. Each arm launched both a WG64 pack and
  a WG128 compute kernel (eight calls to each in the original arm, seven
  in the candidate). The WG128 entry names identify unique embedded SPIR-V
  modules: original `8f0d4d74`, 170708 bytes, and candidate `1afa03f7`,
  38816 bytes. The original loads both P and V within the nested output
  tile and cell loops. The candidate loads P in a separate cell loop before
  the output loop and still loads V in the inner loop. The proposed change
  survived translation to SPIR-V. The trace does not identify final native
  instructions or register traffic; the decision above uses measured time.

  Next evaluate the already available cached-subgroup normalizer at 128K
  on the retained 14508 engine. Its earlier cap-2051 comparisons passed
  independently for gather and clustered shapes, and its old small-cap and
  32K/64K losses remain recorded. Use a fresh matched plan at fixed bank256
  and window10. Do not combine the reduction switch with bank, window,
  upload policy, decode or prefill strategy changes.

  The `qsa_normalizer_128k_v1` plan pins the original engine, both existing
  denominator selectors and the exact previous request bodies. The
  32769-token candidate and two 131072-token progress checks precede
  three fresh clean 131072-token pairs in AB, BA, AB order. Each process has
  three thinking-on requests. Timing separates the first from the two
  subsequent requests. The route reader checks the actual denominator,
  residual and bank placement, window geometry and decode handoff. It
  records changes in jobs and requested bytes as possible downstream effects
  of the rounding difference, separately from the fixed-input speed result.
  Three host tests cover the two selectors, invalid workspace/fallback
  evidence and distinguishing a repeated gain from a same-work gain.
  Both traced 128K arms close with three requests each, checked handoff
  and no new driver fault. All 48 bank views in each reference request
  have fewer than 512 slots at the fixed bank256 capacity. Warmed
  diagnostic PP is 928.548 ordered and 921.508 subgroup. The subgroup
  arm has 123, 3 and 91 more expert jobs in the three matched requests.
  All nine processes and 27 requests finish normally with quiet
  validation, checked handoff and no new driver fault. The final run
  verifies all 15 model asset hashes. The six clean processes give
  the following 131072-token results.

  | Pair order | Warmed ordered PP | Warmed subgroup PP | Warmed time ratio | First-use time ratio |
  | --- | ---: | ---: | ---: | ---: |
  | Ordered, subgroup | 925.223 | 922.422 | 1.003037 | 0.993503 |
  | Subgroup, ordered | 926.355 | 924.583 | 1.001916 | 1.022974 |
  | Ordered, subgroup | 924.724 | 924.840 | 0.999874 | 1.008852 |

  The geometric mean of the three warmed ratios is 1.001608. Keep the
  ordered selector. The subgroup fails the repeated-gain criterion.
  Process-median warmed TTFT is 141.895 seconds ordered and 141.995
  seconds subgroup. The same job deltas recur in all three clean
  pairs, with 72.6–93.9 MB more weight upload payload requested per
  subgroup request. The fixed-input result includes this changed
  downstream work. Kernel-only attribution and hardware saturation
  remain unqualified. All original individual samples are retained.
  Full route and plan evidence is at `6bdb37b3a`. Recovery verified 776
  original inputs, all 18 generated case inputs and the exact retained
  engine and provider identities. The nine native owners were closed.
  The 635 capture files (96,703,218 bytes), temporary inventory and spent
  one-shot controller were retired. Four CSV entries capture separate
  first-use and warmed aggregates for each selector.

  The next transfer experiment will examine full-fit weight prefetch while
  the layer's attention runs. The 64K profile at `c40c8c1bd` measures
  14.6–15.9 seconds of M2D copy service and less than 15 ms of copy/kernel
  overlap. Reaching 1000 PP from its clean reference requires 6.9 seconds of
  wall-time savings. Treat the copy service as the budget to investigate.
  Prior bank512 admission at `fb8e32cd3` fits all 48 packed layers and all
  residual rows at 64K. Its independent 64K capacity comparison keeps its
  mixed outcome. Use the same bank in each proposed prefetch arm and
  recheck actual admission.
  The nine clean bank512 requests at 65537 actual input tokens requested
  37,137,228,800–37,214,720,000 bytes of demand uploads. The same native
  image and exact model assets have 50,292,326,400 bytes across all expert
  layers, as checked in `prefill_bank_minimum_v2`. Preloading every expert
  would add 13.08–13.16 GB of requested payload, or 35.14–35.42%.
  Include this work increase in the separately measured all-expert
  loading unit. If needed, compare a bounded prefetch subset with the
  same exact demand fallback and verify its useful versus unused bytes.

  A separate selection candidate follows from the distinction between
  early known demand and predicted demand. Bevy's concurrent decode work
  exposes true gate decisions before scheduling transfers. Its cache
  results also show why prefetched bytes alone cannot establish useful
  work. PDScope's preprint schedules a subset ahead and reconciles it
  with the true gate result. These are design references for this
  single-request prefill question. [Bevy, sections 2–3](https://arxiv.org/html/2611.04794v1#S3)
  [PDScope, section 4.4](https://arxiv.org/html/2610.39216v1#S4.SS4)

  The `prefill_first_demand_v1` diagnostic closes at `e3d947885`. It uses
  the recovered native window image and a host-only extension of the
  existing MoE observer. After the existing drain, read one earliest
  row per selected expert from `source_rows()[offsets()[expert]]` and
  weight it by the actual packed stride. Four thinking-on `bench.sh`
  requests compare prefix demand to all 48 layers of full-fit bank512
  JIT upload counts and bytes. All have zero private-bank hits, checked
  handoff, normal process exit and no fresh GPU fault. All 15 final model
  hashes match. The 32K and 64K requests are two independent processes,
  each with two requests. Their observer timings do not qualify speed.

  | Engine rows | Exact weights needed | Identified by row 8192 | Identified by row 16384 | Additional payload if all 512 experts load |
  | --- | ---: | ---: | ---: | ---: |
  | 32768 | 35.39–35.49 GB | 87.15–87.26% | 92.73–92.78% | 14.80–14.91 GB |
  | 65536 | 37.17–37.21 GB | 82.95–83.10% | 88.21–88.28% | 13.08–13.12 GB |

  The first 8192 rows identify about 30.9 GB at 64K. The next 8192 add
  about 1.9–2.0 GB. Start with one prefix and exact demand fallback for
  the tail. First isolate moving the existing prefix router earlier and
  prove its scratch lifetime and unchanged route IDs, weights and upload
  work. Then separately change copy issue time, copy queue and first-reader
  wait with a fixed payload. Keep the final combined comparison against JIT.
  These are requested bytes known from completed routes. Actual early
  availability, transfer and attention overlap, and physical bus traffic
  require their own measurements.

  The first-piece producer is implemented at `14107b721`. `off` uses the
  existing order. `late` runs the first piece after attention in a dedicated
  59,539,456-byte GPU allocation, copies the first IDs to RAM and merges the
  other outputs into the ordinary FFN arena. `early` runs that same producer
  after the first checked attention store and retains its outputs through the
  remaining attention pieces. Both experimental modes keep later FFN,
  selected expert uploads and the compute queue fixed. Admission requires a
  single ordinary GPU-resident full-fit window. The added storage and host
  readback live through the outer checked drain.

  `prefill_prefix_router_model_v1` ran three independent 32K processes with
  two matching thinking-enabled requests each. All six requests completed
  with 48 observed layers, normal process exits and no new driver fault.
  The Python aggregate failed when it looked for the MTP restoration count
  in a route dict that had only validated the count in the native log. The
  separate `prefill_prefix_router_replay_v1` at `36460f958` recovers the
  original handoff evidence, reruns the original comparisons and rehashes
  the 15 model assets without launching another GPU process. All full router
  ID digests, prefix FP32 gate-weight digests, job counts, expert upload
  bytes and MTP restore counts match across `off`, `late` and `early`.

  `prefill_prefix_router_timing_v1` closes at `0d3c241c4`. Three matched late
  versus early pairs use the same native ELF, 65537 actual input tokens and
  three thinking-on requests with 24 outputs per process. Per-request job
  counts, JIT upload counts, requested bytes and MTP restoration match in
  every pair. All six native processes close normally with no fresh GPU
  fault. The final pass checks all 15 model asset hashes.

  | Native early minus late | Late, early | Early, late | Late, early |
  | --- | ---: | ---: | ---: |
  | First request prefill time | -1.217 s | -1.384 s | -0.744 s |
  | Warmed prefill time | -0.120 s | -0.136 s | +2.571 s |
  | Warmed client TTFT | -0.084 s | -0.061 s | +2.581 s |

  Early/late time ratios have geometric means of 0.987075 for first-use
  prefill and 0.984473 for first-use TTFT. The warmed ratios are 1.010633
  for prefill and 1.011161 for TTFT with mixed pair directions. Median
  warmed process throughput is 894.82 tok/s late and 889.52 tok/s early.
  Preserve the repeated first-use observation and the mixed warmed outcome.
  Use the ordering change as a controlled prerequisite to test transfer
  overlap. Verify that overlap earns a repeatable net gain on the actual
  model path. A clean loss in one pair remains part of this comparison.

  Before judging the combined route, use a storage-only control that
  publishes prefix IDs back to the device map and keeps the original full
  ID readback. Compare storage, split readback and earlier dispatch as
  individual changes. For transfer overlap, separately compare copy issue
  position, copy queue and first-reader wait against fixed payloads. Physical
  context and sustained decode retain their own qualification gates.

  The same-queue production-path control fixes `STRATA_SYCL_PREFILL_MOE_PREFIX_ROUTER=early`,
  the ordinary resident host source, the private full-fit bank and the existing
  in-order compute queue. `STRATA_SYCL_PREFILL_PREFIX_EXPERT_UPLOAD` is opt-in
  with `off`, `ffn` and `prefix` modes. The two bulk-issue modes derive the
  same sorted demand set from the same first8192 rows at the same point. They
  submit the same individual copies either immediately after that readback or
  at the existing FFN boundary. Remaining experts still use JIT. Keep the
  router order, arithmetic, attention, allocation and decode policy fixed.
  First qualify the same-build CPU and small bank fixture, then check off, ffn
  and prefix with the same 32769-token requests. Require all48 layers, identical
  route/weight digests, expert-job and upload conservation and the expected
  MTP handoff. Check ffn and prefix for identical selected expert digests and
  bytes. Each selected mapping must have an actual reader. Stop on a mismatch,
  timeout or fresh GPU fault. This arm qualifies only same-queue issue timing.

  Each clean64K comparison closes three counterbalanced pairs with three
  requests per process. Expert jobs, selected payload and MTP handoff match
  where applicable. All18 processes close normally with no fresh GPU faults.

  | Candidate and reference | Warmed time ratios | Decision |
  | --- | --- | --- |
  | FFN issue versus JIT, `c6b99c36b` | 0.9726, 0.9676, 0.9553 | Repeated gain |
  | Earlier issue versus FFN issue, `907dd5a71` | 0.9810, 0.9820, 1.0290 | Mixed |
  | Earlier issue versus JIT, `760747a44` | 0.9444, 0.9643, 0.9978 | Repeated net gain |

  Preserve first-use and warmed samples separately. The measured net gain
  establishes the combination under the fixed early router and the same queue.
  Select FFN issue as the development reference for source placement based on
  its separate repeated gain. The phase references remain available after
  retiring the temporary third build.

  The separate source-placement control fixes FFN issue and switches only
  the selected copy source with `STRATA_SYCL_PREFILL_PREFIX_EXPERT_HOST_USM=0|1`.
  Use one checked request-owned host-USM allocation with the largest layer
  extent. Copy only selected payloads to their original offsets ahead of
  the same individual commands. Drain the previous layer before host-buffer
  reuse and retire accepted commands before freeing its owners. The ordinary
  source still services JIT misses. Admission includes this allocation in the
  existing RAM budget and selects the experimental source only for the
  single full-fit FFN route. The same-build smoke checks both strides,
  reuse, cancellation and unselected bytes. Verify matching 32K router,
  payload and handoff evidence before timing three counterbalanced 64K pairs.
  The closed source comparison at `9d845d7dd` uses 65537 fresh tokens and
  three requests in each of six processes. All selected IDs, copy commands,
  requested bytes and handoffs agree. Host-USM/arena warmed prefill time
  ratios are 1.07859, 1.01961 and 1.01427 (geometric mean 1.03709).
  First-use time increases 4.96% by the paired geometric mean. Median
  warmed throughput is 880.11 versus 866.50 tok/s. Retain the arena source
  for the model speed reference. The comparison measures the complete
  source change, including CPU staging, within this scheduling policy.

  Build the independent transport control with
  `STRATA_SYCL_PREFILL_NATIVE_COPY=ON`. Both native modes use the checked
  immediate-list owner from the overlap fixture. Fix FFN issue, host USM,
  selected IDs, copy extents, layouts and arithmetic. Compare same-queue
  SYCL with and without the added completion bracket, then compare fenced
  SYCL with native ordinal 0, then native ordinals 0 and 1. The bracket
  drains preceding compute and verifies copy completion before publishing
  the private map. Check actual native context, device, ordinal and queue
  flags. The outer lease retires any accepted native work before releasing
  either allocation or rebinding the layer. Smoke-test both strides,
  repeated staging, guarded bytes and cancellation before tracing the
  four 32K routes. This control holds overlap off. Test earlier issue
  and a deferred first-reader wait separately before measuring device
  overlap and the net result against the arena reference.

  The next copy schedule fixes the copy-only engine, native API, host USM and
  selected experts. Compare ordinary FFN issue and earlier prefix issue with
  immediate completion, then change the completion point at each issue site.
  Preserve one of each arm per 64K timing block and keep first use separate.
  The private bank stays unpublished until the checked native wait. On
  cancellation, the outer lease drains accepted copies before source or bank
  release. Validate both source strides, unpublished cancellation and first
  readers in the small fixture, then match all four 32K model routes.
  To establish same-layer overlap, match device intervals by request, layer and
  copy list in a separate observer run. Clean timing measures wall-time
  benefit. Both are needed to attribute a benefit to overlap, and the net
  comparison still includes host staging against the arena FFN reference.

  On native `14508acbf`, the layer boundary drains the previous readers before
  rebinding the private bank. Dense activation completes before the
  attention pass. Each attention piece stores BF16 residuals on the compute
  queue, and the first accumulator-window call follows the last piece.
  Test whether the bank can be filled in this interval with the already
  borrowed immutable weight slabs and existing packed destination. Compare
  demand-selected and all-expert loading at the same FFN boundary first.
  Then hold payload, expert IDs and placement fixed while moving issue before
  attention, moving copies to a second same-context queue with the early
  completion check fixed, and deferring that check to the first expert
  reader. Keep the net comparison against demand-selected loading. No
  copy aggregation is part of these comparisons.

  The additional queue needs its own checked retirement before source or
  bank release on success, cancellation, enqueue failure or exceptions.
  Check allocation and span separation against the attention workspace,
  publish a usable expert map only after successful copy completion and
  guard the first reader. Leave the main compute queue and decode
  ownership in place. Begin with a bounded component reproducer using
  identical payloads and actual attention work, with explicit copy-only and
  compute-only times, observed queue intervals and interference in the
  combined case. The existing copy-lane component gains and losses stay
  recorded. They establish neither this lifetime contract nor the model
  throughput effect.

  The opt-in fixture `STRATA_QSA_BENCH_FULLFIT_OVERLAP` uses the original
  ordered-exp prefix with an 8192-query, cap-2051 gather and a separate
  synthetic ordinary-RAM payload at the observed layer-2 stride. The
  512 individual copies total 1,009,254,400 requested bytes. The scheduled
  arms are compute-only, copy-only, one-queue serial, two-queue serial and
  two-queue deferred wait. The two serial arms isolate queue choice and the
  two separate-queue arms isolate wait position. Output is reset before
  each interval. Full bank bytes, guards, source immutability and exact QSA
  output are checked outside timing. The small case also checks retirement
  after an exception following an accepted copy. An opt-in RAW clock
  interval permits separate device-timeline capture. The controller
  keeps the first cycle separate from four subsequent cycles per arm. It
  checks observed copy bytes, kernel counts and interval boundaries and
  reports concurrent-to-alone device times as an interference diagnostic.
  V1 passed host checks and failed before invoking the compiler because
  the log directory collided with the precreated output directory.
  The failed attempt is preserved at `e78837070`, and its four temporary
  files are retired. V2 separates `compile/` logs from `build/` products
  and adds a host regression check. It keeps the same fixture sources,
  comparison order and decision criteria.

  V2 closes at `5e85355f5`. Four host checks pass. The fresh six-TU
  build verifies 722 compiler inputs, 25 link inputs and the final
  executable. Five small traced cases include cancellation after an
  accepted copy. The larger fixture produces 25 traced and 75 clean
  samples with exact output, guard checks, normal exits and no fresh
  GPU fault.

  | Candidate/reference time | Clean pair ratios | Geometric mean |
  | --- | --- | ---: |
  | Separate serial/shared serial | 0.999875, 0.985145, 1.034187 | 1.006194 |
  | Deferred/separate serial | 0.994444, 1.001983, 0.959401 | 0.985100 |
  | Deferred/shared serial | 0.994320, 0.987099, 0.992200 | 0.991202 |

  Both individual changes miss the three-pair gain criterion. The net
  candidate is 0.88% faster by paired geometric mean. Its copy and QSA
  device intervals overlap by zero in all five traced cycles. On the
  separate queue, deferring the wait reduces the gap from the last copy
  to the first QSA kernel from 4.22–9.76 ms to 7.1–7.5 µs. The clean
  standalone copy medians range from 226.3 to 329.3 ms, and compute
  medians range from 221.5 to 221.7 ms. The variation and trace gaps
  establish neither physical PCIe saturation nor separate engine use.
  The original prefetch advancement gate fails.

  The same-ELF `fullfit_attention_attribution_v1` captures 25 windows and
  joins 512 copy operations and two QSA kernels in each of ten selected
  windows via PTI operation flows. In the four warmed deferred-wait
  rounds the first QSA append returns 55.56–59.70 ms before the final
  copy completes. QSA starts on the device 7.17–7.53 µs after that
  completion. In the serial control the QSA append enters 4.55–5.30 ms
  after copy completion. Output, guards, validation, provider mapping,
  ownership and the fresh GPU fault check pass. These times locate the
  serialization after the observed append return. The trace gives
  logical lanes, with native list and physical-engine identity unresolved.
  The compact evidence is `820cc84db`. Exact Git source and build input
  recovery permits retirement of the temporary executable and 59 files
  totaling 20,803,077 bytes. Keep the fixture source for the next
  source-memory comparison.

  Follow the queue route before changing issue order. The earlier
  metadata run at `794bb2672` puts both `Runtime::compute()` and
  `make_queue(true)` on ordinal 0, index 0. The relevant Runtime source
  and headers, boot, driver, provider libraries and queue settings match
  the current fixture. The metadata case used a host-only build and UR
  tracing, so the ordinal result guides the next test without identifying
  which hardware engine ran each operation in the current trace.
  Hold payload, queue, QSA and the deferred readiness check fixed when
  comparing ordinary RAM with host USM through SYCL. For a native copy
  trial, use the same supported host allocation in both arms and
  compare the checked native submit/drain path at the original ordinal
  first. Compare native ordinal 0 with native ordinal 1 separately,
  including the same copy count, timing boundaries and guard checks.
  Qualify each native configuration with small bounded cases first.
  Preserve the qualified runtime safety settings and test any later
  readiness or issue-order changes against a reference using the same
  copy path. This sequence keeps source memory, submit/drain and queue
  selection independently measurable. The model gate remains closed.
  [Level Zero command-list requirements](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/command_list.html)

  `fullfit_attention_source_memory_v1` evaluates the first change at
  `1f03719b3`. The same six-TU image holds the SYCL copy entrypoint, queue,
  waits, 512 uploads and ordered-exp QSA fixed between ordinary RAM and host
  USM. A same-context pointer query verifies the source type and both source
  ends. Both small cases pass accepted-copy cancellation, content and guards.
  The ten native processes finish normally with checked ownership, quiet
  validation and zero fresh GPU faults. Keep the 150 clean samples and 50
  observer windows, with source allocation and fill outside timed intervals.
  [SYCL USM allocation and pointer query requirements](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:usm)

  | Warmed host-USM / ordinary-RAM wall time | Three clean pair ratios | Geometric mean |
  | --- | --- | ---: |
  | Copy only | 1.022765, 1.553040, 1.006185 | 1.169173 |
  | Copy and QSA, deferred wait | 0.870347, 1.122055, 1.132908 | 1.034270 |
  | QSA only | 0.999555, 1.001513, 0.998554 | 0.999873 |

  The deferred-wait source change fails the repeated-gain gate. Copy-only
  trace time favors host USM (223.1 ms versus 259.0 ms); all three clean
  copy-only pairs give the opposite ordering. QSA-only medians stay within
  221.2–222.0 ms across the six clean processes. The observed copy direction
  changes from M2D to H2D, and copy duration varies substantially more than
  QSA service. The deferred-wait trace shows 4.854 µs of overlap in the first
  host-USM cycle and zero in the four warmed cycles. Physical bus traffic,
  page pinning and the resulting physical limit remain unmeasured.

  Keep ordinary RAM selected for the model. The native comparison uses host
  USM in both arms as a supported allocation, with source contents fixed.
  Compare a checked native append/drain owner at ordinal 0 with the SYCL
  control before testing ordinal 1 on that same native owner. Preserve
  the current QSA, copy extents, wait positions and safety settings.
  The host-USM component samples exclude production staging from RAM.

  `fullfit_attention_copy_api_v1` closes at `e10c4bcb7`. Both paths use the
  same host-USM payload, destination, QSA and wait placement. The raw fixture
  owns its immediate copy list and matches the SYCL list's context, device,
  ordinal 0, index 0, asynchronous mode, in-order flag and normal priority.
  The two small cases pass with the mapped Level Zero validation provider,
  exact buffer and output checks, and checked unwind after an accepted copy.
  All ten native processes close normally, with unchanged boot and driver
  and zero fresh GPU faults. There are 120 clean samples and 40 traced
  windows. [Level Zero immediate-list synchronization](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/command_list-2.html#zecommandlisthostsynchronize)

  | Warmed native / SYCL wall time at ordinal 0 | Three clean pair ratios | Geometric mean |
  | --- | --- | ---: |
  | Copy only | 1.322876, 1.182392, 1.017979 | 1.167723 |
  | Copy and QSA, deferred wait | 1.168082, 1.068308, 1.136057 | 1.123372 |
  | Copy and QSA, serial | 1.186771, 1.358053, 1.066287 | 1.197804 |
  | QSA only | 1.002884, 1.003065, 1.001062 | 1.002337 |

  Keep the SYCL route selected. The native replacement fails the repeated
  gain gate at ordinal 0. QSA-only medians remain 221.3–222.1 ms. The two
  traces conserve all requested copies and QSA calls and show zero overlap
  in every window. Deferred-wait wall time is 442.9 ms through SYCL and
  443.3 ms through Level Zero in the observer pair. The larger clean losses
  drive the selection. The experiment establishes the component's actual
  list settings and output, with physical bus traffic and saturation
  unmeasured. Compare ordinal 0 and the copy-only ordinal 1 with the same
  native owner, allocation, copy count, QSA and completion placement. Test
  the small ordinal-1 case first, then distinguish device overlap and clean
  wall time. Keep that queue comparison independent of the failed API
  replacement and of later ordinary-RAM source admission.

  `fullfit_attention_copy_ordinal_v1` closes at `ff6be2692`. The same native
  owner, index 0, mode, flags, priority, memory, copy sizes and QSA workload
  run with copy ordinal 0 and copy-only ordinal 1. The small cases verify both
  list configurations, accepted-copy unwind, buffer guards and exact output.
  All ten native processes close normally with zero fresh GPU faults. Keep
  120 clean intervals and 40 device-trace windows.

  | Native ordinal 1 / ordinal 0 wall time | Three independent pair ratios | Geometric mean |
  | --- | --- | ---: |
  | Copy and QSA, deferred wait, warmed | 0.546616, 0.541617, 0.459654 | 0.514361 |
  | Copy and QSA, deferred wait, first cycle | 0.761036, 0.432872, 0.584085 | 0.577316 |
  | Copy only, warmed | 0.972306, 1.043566, 1.161285 | 1.056219 |
  | QSA only, warmed | 0.999932, 0.998919, 0.999763 | 0.999538 |

  The four warmed deferred windows show 220.270–221.728 ms of device
  overlap on ordinal 1 and zero on ordinal 0. The first ordinal-1 window
  has 220.288 ms. The clean deferred-wait medians are 436.8–498.4 ms for
  ordinal 0 and 229.1–256.1 ms for ordinal 1. The repeated-speed and
  repeated-overlap gates both pass. Individual copy-only timings vary
  and their geometric mean increases. The overlapping workload is the
  demonstrated gain. Device service traces leave actual PCIe traffic and
  saturation unknown.

  The independent source-layout check reads the exact model asset already
  bound to the clean reference. Its 48 contiguous layer offsets and
  48 payload lengths are divisible by the host's 4096-byte page size.
  Layers hold 773,324,800–1,363,148,800 bytes. The host's explicit huge-page
  pool has zero pages. These are layout conditions for a bounded read-only
  layer view of the anonymous RAM arena. Check the actual owner, start
  address, import scope and type in the production Runtime before issuing
  DMA. Keep the same native ordinal-1 owner and wait placement when
  comparing a borrowed layer view with host USM, and charge source
  preparation and retirement. Preserve the earlier negative mapped-source
  results on the dense readers. Once this workload's source is qualified,
  compare actual expert demand and the resulting prefill route before any
  model speed claim. [Level Zero queue descriptors](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/command_queue.html#ze-command-queue-desc-t)

  `fullfit_attention_layer_source_v1` closes at `2f5164903`. For both
  arms, the production `PinnedArena` and GGUF packer load layer 2 into an
  anonymous one-layer arena. Both use the same immutable `ResidentExpertLayer`
  geometry and copy-only ordinal 1. The reference allocates host USM and
  copies the resident layer into it. The candidate registers the existing
  read-only pages at the same address in the copy list's context. Admission
  checks the pinned B570 API 1.17 provider, extension, type, allocation ID
  and extent, followed by unknown allocation type and the original content
  after unimport. The host-only preflight caught the original Down offset
  typo. The corrected controller supplies offsets directly from the pinned
  layout and verifies the three layer-role hashes before and after the run.

  Both zero-enqueue metadata cases and both four-expert cancellation checks
  pass. All 12 native processes close normally with no fresh GPU fault.
  The copy/QSA cases verify both payloads, QSA outputs and guards. The 512-expert layer
  contributes 120 clean intervals and 40 trace windows. The borrowed
  deferred-wait trace retains 181.810–221.361 ms of overlap in all four
  warmed windows.

  | Borrowed / staged wall time | Three clean pair ratios | Geometric mean |
  | --- | --- | ---: |
  | Layer preparation, warmed deferred interval and retirement | 0.511980, 0.688965, 0.500973 | 0.561162 |
  | Warmed deferred interval alone | 0.982580, 1.421488, 0.930296 | 1.091216 |

  Borrowing takes 4.0–9.9 ms to prepare and 2.3–4.8 ms to retire. The
  measured staging path takes 212.3–259.9 ms to allocate and populate and
  12.2–13.2 ms to retire. Common GGUF loading is recorded separately.
  The source-lifecycle gate passes while the copy and QSA interval alone
  has mixed results. The stage reference allocates per process for one
  layer. Its source owner and staging lifetime are covered by the next
  comparison.

  `fullfit_attention_arena_source_v1` closes at `1b0b01c9c`. It loads the
  actual private `ArenaExpertSource` in every arm and validates borrowed
  registration and release for all 48 layers (50,292,326,400 bytes), with
  no application copy or kernel enqueues in that metadata test. The three
  four-expert tests pass cancellation after an accepted copy. All 16 native
  processes close normally with no fresh GPU fault. The full layer-2 cases
  copy 1,009,254,400 bytes and verify the payload, guards and QSA output.
  Copy ordinal 1, QSA, waits and exact transfer extent are fixed across source
  comparisons. Preserve the independent allocation and source decisions.

  | Layer preparation, warmed deferred interval and release | Three matched time ratios | Geometric mean | Gate |
  | --- | ---: | ---: | --- |
  | Reused host USM / per-layer host USM allocation | 0.693929, 0.542989, 0.492079 | 0.570226 | Pass |
  | Borrowed pages / reused host USM | 0.864334, 0.950116, 1.423323 | 1.053385 | Fail |
  | Borrowed pages / per-layer host USM allocation | 0.599786, 0.515902, 0.700388 | 0.600668 | Pass |

  Reused host USM takes 259.6–289.9 ms to allocate and prime once, separate
  from layer timing. Its ongoing preparation is 55.3–65.6 ms, versus
  39.2–44.2 ms to register the original pages. The third matched borrowed
  interval takes 357.0 ms for the warmed deferred median, versus 230.4 ms
  for reused USM. All three traced source modes show real device overlap
  (187.468–221.583 ms across the 12 warmed deferred windows). Those traces
  do not attribute the third clean-pair slowdown. Direct borrowing misses
  the prospective performance gate against the reused baseline, so keep
  it out of the model trial. Use reusable staging as the source reference
  for the next comparison. The first-route census above identifies the
  size of one exact prefix for the independent scheduling comparison.
  QSA state and the GPU bank were fixture-owned in the source comparison.
  Full-model throughput, decode and physical context remain unqualified by
  this result.
  The current-build five-length/time/TTFT curve is closed;hardware attribution
  remains pending. Do not pool mixed timing periods into a law. Preserve
  the bank/coherence/relocation losses as independent outcomes. Reduced
  jobs/dequants/requested bytes
  do not isolate the limiting arithmetic/VRAM/PCIe/submission resource.
  Full combined evidence is`570f6903e`;four first-use/warmed CSVpoints bring
  the index to637. All27samples,nine native identities and the original
  successful decision remain compact. Original771Git inputs and18generated
  payloads are recovered exactly;all635ended files(96,622,772B)and287,082B
  temporary inventory are retired. Both phase references,the coherent
  candidate and required shared records remain for the pending source/phase
  validation;select and retire a superseded generation before a new build.
  Full window evidence is`6ead5d99b`;four first-use/warmed CSVpoints bring
  the index to633. All27samples,nine native identities and the original gain
  decision remain compact. Original771Git inputs and18generated payloads
  are recovered exactly;all635ended files(96,621,178B)and288,349Btemporary
  inventory are retired. Both phase references,the coherent candidate and
  required shared records remain;fresh combined-predecessor/device admission
  passes after retirement.
  Hardware saturation,current9849decode
  preservation,task/agent,long-thinking MTPstrong/weak and physical262144/
  RESTORE qualification remain unresolved. Full relocation evidence is
  `a82b3ef93`;four first-use/warmed CSVpoints bring the index to629.
  All27samples,nine native identities and the failed decision remain compact;
  original771Git inputs and18generated configurations are recovered exactly.
  All635ended private files(96,245,904B)and287,079Btemporary inventory are
  retired. Both phase references,the coherent candidate and required shared
  records remain for the independent window/combination questions;fresh
  predecessor/device admission passes after retirement.
  Full coherence evidence is`c68086021`. Four first-use/warmed CSVpoints bring
  the index to625. All27samples,nine native identities,the failed decision and
  useful request-work/admission values remain in the compact result. Original
  771Git inputs and18generated JSONpayloads are recovered and compared exactly;
  all635closed private files(96,707,795B)and317,566Btemporary inventory are
  retired. Fresh relocation-predecessor and same-build device admission pass
  afterward. All three engine generations and the required shared records stay
  for the named next comparisons;legacy source/configuration/cache holds remain separate.
  U1jit is fixed to isolate native14coherence;its separately positive eager
  component remains eligible.
  `prefill_bank_minimum_v2`closes at`4d027b3e1`:nine normal0processes/27thinking-on
  requests,quiet validation,no fresh fault and15final model hashes. At fixed
  native14/window10/128K,the three bank1/256warmed-time ratios are1.069543,
  1.045789and1.061431(geomean1.058875):the bank-only performance gate fails.
  Aggregate warmed PPis887.605versus835.333;first-use and all samples remain
  separate. All131071BF16residual rows stay onGPU;windows/jobs/dequants/maps
  and actualGU/Down work match. The678,912,000Bbank reduction adds roughly25GB
  of requested H2Dpayload per request;physical bus traffic/saturation remain
  unknown. Four CSVpoints bring the index to621. Preserve V1's invalid
  synthetic-metadata refusal and this standalone loss. The prospective
  capacity/mechanism rule permits only the separately planned window question;
  no larger-window model is admitted before its fresh component/host gates.
  All637closed private files(96,700,824B),the240,780Btemporary inventory and
  three ended Pythoncontrols(41,865B)are retired after exact original768source,
  18generated-input and native-source/recipe recovery from pushed Git. The
  compact result retains all27samples and the original failed decision;both
  phase references and every required current C++fixture remain.
  CPUv4host/wait partition plus active kernel timeline use no coarse CUDA-event
  markers;the old marker-associated stall remains causally unresolved. Require
  exact route/count/clock/owner/conservation and<=1.2observer ratios against both
  controls before normal-route attribution. Finite3GiB/file,4GiB/job,5GiBwhole
  capture is reduced/retired after closure. This identifies service/requested
  payloads and host waits;physical bandwidth/saturation remain separate checks.
  The one-shot profile controller/test are retired;restore from90756f079.
  The equivalent ordinary-Down classifier is preserved in profile_split.py,
  with19host tests passing and exact old-wrapper-plus-base AST equivalence.
  All518ended files(2,360,701,676B)and287,550Btemporary inventory are removed;
  original source/recipe/inputs are recoverable from1dfadd0d6 on pushed feature.
  Both phase builds and the owned collector/model assets remain. Select the
  next independent change from this evidence;keep speed samples separate from observers. Numerical/
  task,agent,long-thinking generation with observed MTPstrong/weak acceptance,
  repeated decode preservation and serve RESTORE/full-context/ordinary teardown
  gates remain before adoption. The earlier standalone physical262144pass is
  reused only at its original scope;two prefill strategies remain available.
  Closed tooling restores from`f12afed85`;full inputs/receipts from`2ade7c494`.
  After original767-source/environment/generated-input recovery,710ended
  raw/JIT/research files(99,334,743B)and307,392Btemporary inventory are retired.
  The shared native/model basis admits after cleanup;both phase references
  remain. Three configured-output provenance holds,the895-file legacy cache
  and the unfinished PR6child remain separate concrete holds,no newgeneration.
  All134resolved V1/V2raw/JIT
  files(27388603B)and the60846Binventory are retired;post-cleanup admission
  passed with three temporarily retained inputs,now retired by the bridge. No V1private root remains
  and no extra native generation was created.
  Queue placement,ready batching and window remain later independent units.
  Closed math/research captures are retired:766files/1262355027B plus the
  208933Btemporary inventory. Exact final ELF,recipe,367project/398original
  sources,906TLA sources,962/29/11compiler/link/object inputs and the1GBcorpus
  reconstructed from model roles match original records;both phase references
  remain. Full comparison inputs restore from`a24162eae`/`dbee24492`;useful
  individual samples/9decisions remain active. Legacy provenance/fixture/cache
  holds stay separate and do not renew this ended third generation.
  The old9arm controller/13host controls/24case fixture restore from`05eceacd`;
  that unused24case fixture is removed from the active tree. Nine closed
  prototype files(125824B)restore from`da18ad915`:seven Unique-slot-only
  source/helper/test files without production/CMake callers,and the completed
  capacity controller/test. Their original measured commits remain reachable.
  The new`coherent_context_model.py`reads original coherent native/speed proof
  directly. Do not refresh the archived capacity runner's767pins. All27private
  inputs are retired after historical recovery and new coherent V2host/short
  admission. Independent parent-copy safety stays. The reviewed DY/DZprivate
  report/views/review/lifecycle/native capture are also retired;65joint files
  total2204021B,with corrected findings/provenance retained in the research index.
  The22.95MBsmall generation and private audit captures are now retired after
  exact final ELF,recipe,370project/398original source recovery;both phase
  references remain. Full input/15case evidence restores from`4eb8276e0`.
  Generic feasibility scopes remain closed;any new audit needs its concrete
  code and bounded plan rather than repeating the same survey. Initial source/header admission failures
  and a system-first IGCselection refusal remain failed;using the original
  model library path selects its qualified /usr/local IGCwithout a system edit.
  A separate transfer queue's existence and
  about4msobserved overlap do not prove a usable independent engine or causal
  speedup. A broader required redesign remains work to scope,not a reason to
  end tuning. The earlier U1proposal preserved the same queue and
  bytes:U1compared JITper-expert with eager per-expert upload;U2compares eager
  per-expert with contiguous-run copies,then validates their useful combination.
  `fullfit_upload_plan.hpp`and its independent byte/publication oracle remain
  experimental. CPUV2evidence`35041702b`passes6byte
  cases/10refusals,including failed-enqueue nonpublication. V1's missing launch
  guard stays held;V2checks source/compiler/ELF/consumed bytes before launch.
  The initial resource annotation's28eligible layers is corrected to24from the
  actual geometry counts;no C++input or sample changes. U1now has an owned
  long-prefill caller and native fixture:committed14508sources plus four
  explicit overlays at`51ea99cae`,excluding pending KVengine changes. The
  fresh coherent`prefill_eager_native_v1`build uses125unique TUs/126objects,
  all four production Gemm-header includers,1461compiler and29external link
  inputs,zero uncovered. Its engine SHAis
  `7ea162eac831256af28e861059641501021a5fec194871b9bcb4dcaddc1b1f9d`.
  `STRATA_SYCL_PREFILL_FULLFIT_UPLOAD=jit|eager`selects the upload position
  before process initialization;U2coalescing is not selected. The eager caller
  checks the whole bank against live FFNscratch and rejects unequal payload/
  stride bytes;source/cache owners remain under the existing checked outer
  drain. Partial,nonempty and staging banks retain their original route.
  Fresh traced smoke and quiet validation in`prefill_eager_device_v1`,evidence
  `04634f850`,pass all8cases across0/1/4/127/512experts,strided sources,full
  readback/guards,refusals,reuse and cancellation. Both native processes close
  normally0with quiet validation,verified actual initialized providers and no
  new fault. This is byte/lifetime evidence,not model performance or adoption.
  The closed`prefill_eager_model_v1`cohort,evidence`410f6cb8d`,passes two
  traced model processes plus twelve clean processes:three independent matched
  pairs at each of32Kand64K,three fresh thinking-on bench.sh requests per
  process. All42requests and14native owners close normally0with quiet validation
  and no new fault. Every paired request has identical actual job/fused-job/H2D
  counts;its request hashes also match the historical fastest cohort. U1alone
  passes its predeclared warmed PP/TTFTscreen in all three pairs at both lengths.
  Eager/JITpaired geometric-mean warmed time ratios are0.9761108569at32Kand
  0.9862228847at64K. Process-median PPis748.708→765.376at32Kand
  911.589→926.929at64K;these medians and paired geometric means are different
  summaries. Keep every first/warmed sample:the third32Kfirst eager/JITtime
  ratio is1.103185748,so no all-pair first-use gain is claimed. Local warmed
  32K→64Kslopes are0.858330ms/token(JIT)and0.851109(eager);two lengths do
  not qualify a fitted law,physical fixed cost or extrapolation. The CSVnow
  has540points,including8first/warmed arm summaries from this cohort and
  four current-period original-control summaries.
  Absolute rates remain below historical fast14508PP871.574/1006.832.
  `prefill_eager_original_control_v1`now reuses that exact retained dcb8ELF
  under the current matched frontend/configuration/workloads. Its first32K
  control closes normally at756.450warmed PP,also below its own historical
  value;its64Kcontrol closes at914.473PP. Both processes and all six requests
  finish normally0with closed owners,quiet validation,no new fault and15final
  model hashes matching. This is one process per length,not a new
  paired cross-build gain/regression claim or a cause attribution. The normalized
  shared compile commands have no differences;hardware saturation remains
  unknown. Main keeps extended source-gap analysis delegated to bounded Orca
  Luna. Preserve U1's positive same-image result and the faster historical
  reference separately. Default adoption,source replacement,repeated sustained
  decode/agent/task and fullphysical262144/tails/RESTORE remain unqualified.
  This boundary recovered the exact1165native source inputs,original recipe,
  1461compiler/29link inputs and final two ELFs from their original commits
  against the actual files before retirement. Its2540files/302703511bytes,
  including the ended candidate build,raw model/device output and BR/BS/BT
  source/review transactions,are physically retired. Useful individual samples
  and source/protocol corrections remain in compact results and Git. The two
  independently retained phase references remain unchanged:fast14508/12ELFs
  and namingA/14ELFs. Existing legacy/orphan/.cache holds remain separate.
  The four ended one-shot controllers are retired to Git;restore their original
  committed source and recipe for a newly planned reproduction. The U1C++
  helper and independent byte/lifetime oracles remain the concrete inputs for
  U2's fresh image below,including the U1individual-copy control itself.
  U2contiguous-run copying is now independently measured and **not adopted**.
  Its native snapshot is14508base plus four owned overlays at`be82383e8`,not
  the whole primary native tree. Fresh full compilation closes125unique TUs/
  126objects,1461compiler/29link inputs and all four production Gemm includers.
  The four native stages close normally0with25case executions,including
  strided fallback,clustered IDs ending511,full readback/guards and one
  1009254400-byte production-stride copy with checked cancellation. Actual33
  initialized providers match the qualified runtime;no fresh fault occurs.
  The closed`prefill_coalesced_unit_v1`cohort at`5c9ab0bae`contains14normal0
  processes/42thinking-on requests:two diagnostics,then three independent
  matched pairs at32Kand64K. All18matched clean request pairs have identical
  request hashes,other actual routes,jobs,fused jobs,logical experts and bytes.
  First32Kfull-fit copies fall8697→2507with16012825600Bunchanged;first64K
  falls9178→2275with16889804800Bunchanged. Actual source/bank strides are equal
  in all24eligible layers;the other24layers keep their original path.
  Warmed coalesced/individual time ratios are1.006195/0.993623/0.961485at32K
  and0.997323/1.039700/1.005200at64K. Geometric means are0.9869203and
  1.0139086:32Kaverage time shortens1.308%,while64Klengthens1.391%.
  Neither length passes the prospective all-three-pairs gain condition;64K
  also fails TTFTpreservation. Process-median PPis763.084→767.981at32Kand
  925.697→920.908at64K. These medians differ from paired geometric means.
  Preserve each failed/positive pair and every first/warmed sample;do not
  dismiss the observed decreases as noise. The combined/JITtrial is not
  admitted by this failed unit screen. No decode or physical-context adoption
  follows. Eight CSVpoints bring the cumulative index to548rows.
  Adjacent warmed slopes are0.850054932ms/token(individual)and
  0.869647217(coalesced);two lengths do not qualify a fitted/held-out law,
  physical fixed cost or extrapolation. APIcounts and requested bytes do not
  establish physical PCIe/VRAMtraffic,saturation or the cause of variance.
  Copy aggregation alone has not established a reproducible model speedup.
  Preserve U1's independently positive upload-order result and separate decode
  selection. U2's unused selector/planner branch,fixtures and three one-shot
  controllers are removed. Exact source/recipe/actual-input recovery precedes
  physical retirement of2373files/286027510bytes. All26phase-reference ELFs
  remain unchanged;42individual samples and6conserved request workloads stay
  in the compact result. The trimmed U1source needs a fresh coherent engine
  before further native/model qualification. The cleanup-only host/syntax
  checker now closes normally0:production SYCLfixture syntax,6independent
  byte cases and10refusals including failed-publication preservation pass.
  It runs no GPUwork and does not relabel an existing executable or establish
  model adoption.
  Fast14508and namingAphase references remain the two validated generations.
  Reducing the streaming-bank allocation or introducing an independent upload
  engine is a separate future unit,requiring its own opportunity,safety and
  ownership/dependency qualification. Neither is silently bundled here.
  Metadata and CPUtests themselves qualify neither model speed nor saturation;
  the U1speed claim comes only from its closed paired model cohort.
  Main`9849e3580`integrates`build_named_kv.py`from the reviewed child`7776f66a0`
  (original controller`1d0beb514`)for the first unit only:
  it checks the original six-file mechanical rename and complete native-file
  set/bytes,refuses the later TU split,and requires legacy retirement before
  a fresh14-target coherent build. Its source/unit/recipe/provider admission
  builds`phase_kv_naming_native_v1`. Its28host census controls and all five
  configure/compile/census commands close normally0with empty owned sessions.
  All14fresh ELFs use140fresh objects/136unique TUs,1473compiler inputs and
  29external link inputs,with zero uncovered files and consistent definitions
  across all five tracked headers. The engine SHAis
  6d2b4f7d6a79424c3da5ac6e35d7e1336016604e2f7bdb68b84031d2879903de;
  source remains9849e3580and all25phase-reference ELFs are unchanged.
  No GPU/model/performance qualification follows from compilation.
  Main`8a1d10d28`integrates the reviewed`bbc8a6c00`qualification proposal and
  prepares`phase_kv_naming_device_v1`:11fresh host/submission/phase/head/
  attention/producer/GUstages on this generation,without copying old results.
  The fresh host stage closes normally with all owned children reaped and
  empty sessions:MAXT/physical admission and hash-order guards,reader controls,
  owner/configuration refusals,phase policy and retained1eecselected-head check.
  The traced small submission stage also closes normally0with accepted
  pre-main/provider validation,no fresh fault and absent/reaped owners.
  All11fresh native stages now close normally0without fresh faults:both real
  head modes pass four seeds,two restore cycles and12graph replays each;
  packed head2maximum pair error is4.17232513428e-7. Attention/producer
  boundaries include262144component cells and GUfinishes44numerical cases
  within the original FP16ULP/BF16capture/guard criteria. All source/provider/
  ownership gates remain intact. This is native component qualification only;
  The same-build KVfixture now also closes its fresh traced smoke and full
  oracle checks normally0:14direct smoke semantics,20smoke graph replays,
  795exact full oracle records,60full graph replays,15physical-tail graph
  records and12timing-shape correctness cases. All provider/validation/owner
  and unchanged-boot gates pass without fresh GPUfaults. This is component
  correctness only;the final physical context cell in a fixture does not
  qualify a full model fill. The naming-only production handoff now also
  closes normally0with32769/82/32769thinking-on inputs and32outputs each.
  Both long entries complete all48layers,restore the decode heads/MTPstate
  and actually capture cooperative decode4;the short prefill captures only
  scalar KV. Placement,typed QSA/GU/Down and cache/drain checks,15full asset
  hashes,provider/owner/unchanged-boot checks all pass without fresh faults.
  The public model gate keeps work totals and source hashes while retiring
  full first-job row histograms after validation;it is intentionally curated,
  not input for resuming the closed full-parser reader. This diagnostic's
  32outputs do not qualify sustainedTG or clean PP/TTFT. The original
  per-unit speed comparison and broader model gates remain pending.
  After exact original source/recipe/ELFadmission and closed-owner checks,
  all three successful diagnostic raw roots are retired:169files/23168434bytes.
  Committed curated gates remain the comparison inputs;all39retained ELFs
  are unchanged. Do not resume the closed raw-parser controllers.
  The independent naming-image checks now compare the exact330405and6d2b4f
  production ELFs. V1`f6da849ab`passes36ordinary sections(22974831bytes),
  load layout and24645normalized non-file symbols;AMreview identifies its
  missing raw PT_LOADcoverage. Fresh V2`8139ae06a`also passes all four raw
  initialized load segments(22813505file bytes),normalizing only the ELF
  section-table offset at bytes40..47. The two byte totals overlap and are
  not additive. Self/refusal controls pass(1positive/6negative in V1,
  1positive/7negative in V2). Original V1and the Aperformance plan are unchanged;
  this is static compatibility evidence,not throughput or model adoption.
  Prospective comparison source`f2e3eccad`now pins separate naming-only
  prefill/decode plans against the exact1eec/330405reference at KV1in both
  arms. Prefill keeps the existing12-process32K/64Korder and first/warmed
  scopes;decode keeps six R,C,C,R,R,Cprocesses and both long-thinking bodies.
  The unchanged median screens remain separate diagnostics,including failures.
  Naming-control eligibility additionally requires the original numerical/state
  gates,exact loaded-image identity,and no clear slowdown in each three-pair
  log-time95%Student interval(df2). Timing alone does not prove equivalence
  or speedup;this prospective criterion is only for the byte-identical naming
  control and changes neither prior failed preservation result. It does not
  apply to the different B1/B2images. Prefill now closes all12processes and
  36thinking-on exact24-output requests at32K/64K,with normal0,reaped empty
  ownership,quiet validation,unchanged boot/driver,no fresh faults and15final
  asset hashes. Evidence`0ec9f0132`passes all eight original median screens and
  all eight naming-control intervals;individual slower pairs remain in place.
  Named-image first/warmed medians are609.991/765.026PP at32769inputs and
  798.745/919.334PP at65537;warmed client TTFT is43.0035/71.4715s.
  Four CSVrows index this control,not a speedup. The added descriptive
  `length_characterization`keeps both arms/first-use scopes separate:the named
  warmed two-anchor interpolation is approximately
  `T_ms(n)=14379.382+0.868334961*n`;its three block-local slopes are
  0.868335/0.861969/0.843611ms/token. Two lengths cannot validate a law or
  a held-out length,and the intercept is not a physical fixed cost. Source14508
  remains the faster credible owned reference;this control does not replace it.
  The ended CPUcapacity and naming-prefill raw roots are now retired after
  original Git/source/recipe/owner checks:1300files/205966509bytes,including
  960derived Intel GPUcache ELFs. All39retained engine/fixture images remain
  unchanged. Original samples/verdicts,CSVand source identities stay in Git;
  the shared configuration/model/environment receipt remains retained. The
  initial no-ELF cleanup refusal is preserved;the prospective cache-specific
  v2rule verifies original parents/SDK/inputs and rejects other ELF/symlinks.
  Do not resume their closed raw-parser readers or recreate old process paths.
  Sustained decode control closes all three matched pairs,including its first:
  3548/5485thinking-on native outputs at33213/33283fresh inputs give
  reference15.0378/9.4120TG and named-image14.5927/9.4278TG,
  with unchanged80.3074%/34.2952%draft acceptance. API/native clocks,
  zero inserted/reused tokens and actual scalar1/cooperative4routes match;
  normal0,quiet validation,reaped empty ownership and no fresh fault pass.
  Keep the slower code sample without dismissing it as noise. Original
  reference`da46ff31f`and candidate`2f3b83d29`remain the original first pair.
  The second pair now also closes normally0with the same counts/acceptance:
  candidate`3aa4c65e7`gives14.9646/9.4128TG and reference`bfcb3f4fc`
  gives14.9628/9.4163TG. Both directions and all individual samples remain;
  the last reference/candidate give14.9905/14.9895TG for code and
  9.4486/9.4465TG for prose. All six fresh processes close normally0with
  unchanged counts/acceptance,quiet validation,reaped empty ownership and
  no fresh fault. Preserve both original TGmedian-screen failures:reference
  medians14.9905/9.4163versus candidate14.9646/9.4278are not a speedup.
  The prospective naming-only three-pair log-time95%intervals are
  [-0.0331343,0.0531229]for code and[-0.00320874,0.00248841]for prose;
  both admit the byte-identical image control together with its original
  numerical/handoff proof. Both Aphase controls are now eligible;this is
  neither statistical equivalence from timing alone nor inference adoption.
  B1phase performance and B1s/B2remain unqualified. The ended1eecbuild is now
  retired after exact2376-source/dependency,recipe/compiler/class/SDKand
  13ELFrecovery:192files/206386091bytes. Both retained generations'26images
  stay unchanged. The first optional all-same-uid maps scan is refused by
  process permissions before deletion;its failure remains recorded. The new
  bounded plan verifies named ownership and accessible maps/fds,with four
  unrelated processes' internals unavailable and no global-absence claim.
  Keep original receipts/census/configuration and build B1alone against A
  while retaining faster14508. Compare each independent
  unit before their combination;do not fold B1s or scalar restoration into B1.
  Native16capacity review BA also keeps the current8192chunk. The optional
  vLLM adapter has additional8Kownership/GU/Down guards;raising a top-level
  limit is unsafe. A16Kouter batch over8Koperator tiles is only a held
  scheduling proposal with an unreviewed planner/cursor/residual/route seam,
  unknown workspace admission and no performance or model qualification.
  The retained cross-build configuration receipt supplies
  its frozen environment/model manifest until these comparisons close;old
  generated configurations or full diagnostic captures are not their inputs.
  Its temporary-third-generation
  consumer is the sole-change naming comparison,preserving both phase references.
  CPUcapacity is closed and the three legacy build holds are reconciled and
  retired. Qualify and compare the original units separately before combination;
  the second unit must not inherit an unpassed first-unit gate. Rebuild every
  affected TU and require actual production phase routing,component oracles,
  matched prefill/TTFT and sustained decode preservation. A split's result
  does not qualify legacy shared graph/state/placement ownership or establish
  the cause of an earlier regression. Integrate completed units into the
  primary with their original identities,then retire the finished child.
  The reused driver's`execution.json`is an Owner-closure snapshot before
  final publication;completion comes from the canonical record plus actual
  reaped/session-empty/absent-owner checks. Preserve the snapshot unchanged.
  The new admission consumes the closed
  compact records after the old generated configurations are retired.
  If CPUexpert work is limiting,compare one actual format/
  NT at fixed work over hot and rotating footprints before choosing a kernel
  change. Separate any CPUcensus/service observer and its overhead from clean
  speed. Existing decode timing counters alone are wall-time regions.
  The short review transaction`existing-decode-resource-review-r-20261114`
  is now retired(2files/7309bytes). Its GGMLtype42=Q2_0correction and the old
  workload's zero NT1cache bindings were already present in the existing
  report entries;both are checked before deletion. The consumed digest hash
  is`1ee8e7c8fc6458888c32ca9f6728f8b58c70aa6e772ebb27472847796038046b`
  and Task hash`b5bec75432645ab661c6e045535e5ce3210840f2e9b4d2efa36b8f904d75ac8b`.
  No default adoption or hardware-saturation claim follows. The opt-in hook
  has a named prefill-preservation consumer;the standalone timing fixture's
  generation is retired after exact original source/recipe recovery. Active
  retained admission does not call the archive-only component prerequisite.
  Legacy source recovery now verifies the GUbaseline's1152project inputs,
  its four213dd overlay inputs and13actual project headers;the old mixed
  Gemm definition remains ineligible. DFlash2's841project inputs and both
  dependency maps are recoverable,and its retired CLIcensus is still present.
  Adaptive's115controller and568composed native inputs plus exact recipe are
  recovered using the originalbbd345checkout. Its already-retired558-source
  prerequisite is expanded from the original Git count/hash scope;no private
  receipt is rewritten or execution qualification inferred. The temporary
  `.worktree/retire-adaptive-source`checkout's source-only consumer is closed;
  its973regular files(13916535bytes)are retired after verifying a clean tree
  and the originalbbd345commit's reachability from the pushed primary.
  The1467/698GU,1453adaptive and1461DFlash2compiler maps now bind completely
  to their recovered project/dependency sources or the original SDKmanifest,
  with zero uncovered inputs. All three29-file frozen-link maps also match
  that manifest. This is source-census/membership proof,not new execution or
  qualification. Final live checks match1227SDK/compiler/link files and all
  13original ELFs;the original GUcontroller and one-archive link recipe match
  their Git/controller records. At the serialized closure,all three ended
  build/source trees are retired:933files/links,334847897regular bytes.
  Five original provenance/census/SDKfiles remain,including the actively read
  adaptive external-inputs.json;all25fast-reference/current ELFs are unchanged.
  The455738-byte deletion inventory is itself retired after reconciliation,
  with hash8dab9fda693031823519c6c6fb1270d69ea1b7d7ac94aa46907bf817cc543bf9
  and compact decision in`production_preparation.build_retirement.legacy_retirement`.
  `legacy_remaining`is empty;the old mixed GUrelink stays runtime/speed
  ineligible. This closes these three holds,not global artifact cleanup.
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
  New Orca source research remains on hold;Main's ordinary CLI reaches the
  ready runtime. T/V/Wstay failed. Wnow proves the sacrificial read-only
  control structurally(errno30/EROFS,target absent),but Orca rejects its
  wrapped Codex as`agent_unconfigured`/`no_agent_detected`before Dispatch.
  Its owned helper closes and Main remains unchanged. No actual worker
  product/report mount,IPC,done or parent-publication gate is qualified.
  Native Luna-max ADreview corrects the earlier interface conclusion:
  Orca1.4.229`agent.launch`accepts`agentArgs`and terminal`cwd`,and the
  command builder consumes supplied args. Its supported existing-terminal
  `worker-start`can own that recognized terminal with matching worktree.
  This conditional composition is supported;cwd is not confinement.
  Current Codex0.167.1CLI/help/binary is checked and AFreview is complete.
  The independent AGTask stops before launch:no supported callable raw
  client exposing the required arguments/cwd is available in the checked
  CLI/tool surface. AHnative Luna-max review confirms that prerequisite stop;
  it does not establish global impossibility. No worker,helper,Task or
  Dispatch is created. Actual report-only writes,recognition,worker IPC and
  handle/Dispatch binding remain unqualified. Main now identifies a different
  documented custom-argv route:`terminal create --command`with plain native
  Codex,then`worker-start --terminal`on the same primary worktree. This needs
  a new finite no-source canary at a serialized measurement gap;recognition,
  sandbox enforcement and IPC are not established by the guide. Keep AG's
  original stop and do not retry it or guess RPC. Existing-terminal start
  cannot take model/effort flags;pin them in the native startup argv. The current
  custom workspace-write policy requires sandbox network access for IPC;
  AF's contrary network=false suggestion is rejected. Xtraces Tto N's supervisor,whose ancestry/license
  remain unresolved;do not reuse T/N/Wcode as an implementation basis.
  Current readiness and the two conditional source scopes are curated in
  `research-index.json`under`orca_research_wave_o_20261114.launch_review`.
  Its latest short review digest hash is
  `7eb88accd1dd710e03bc47b2320acaf31fdec4a62dd68a22fa52bfa4db62421f`;
  Wproof hash is`2567c4e8fd73fdab3f846be55b7fe99591b4d7daaae4ec64e45734bb5d00044b`.
  Curated AG/AH facts are retained in the existing index;their six ended
  transaction files(20343bytes)are retired in the serialized CPU-process gap.
  Other superseded startup/review paperwork still requires reconciliation.
  Keep only substantive reusable findings and
  the smallest needed W/origin evidence. The prior exact NTrash/
  credential-cleanup hold stays under`orca_deterministic_startup_n_20261114`.
  At that next boundary,check the superseded external
  `orca-narrow-ipc-connect-probe-20261106`for retirement using archived
  decision`b1d255bde0c282b494c44aa7bdf8311e4ade0547`;its old EPERM result
  does not describe W's recognition failure. No credential bytes are read or purged.
  No ordinary source researcher is active. Main's tuning continues independently.
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

The current coherent14508reference has repeated32K/64Kprocesses at evidence
`f9662f7abfebb1c5f20e7e9e4677fad8f1f5e695`. Its two distinct lengths support
an adjacent slope only,not a fitted law. The offline
[prefill_scaling.py](../tools/prefill_scaling.py)adapter prepares the next
characterization using the existing`prefill_scaling_bench.py`hypotheses and
whole-length holdouts. It separates native source,configuration,actual route
and first/subsequent scope,using one median per process/length before the
cross-process median. Repeated requests cannot masquerade as extra independent
lengths or overweight one process. All seven host controls pass,including
curvature/whole-length holdouts,route/source/scope separation,process weighting
and coherent-record admission. Offline replay of all18original requests gives
first-use prompt/TTFT adjacent slopes0.832302856/0.832672119ms/token and
subsequent slopes0.839074707/0.839706421ms/token. Both groups contain three
independent processes at each of two lengths;all fitted hypotheses remain
ineligible. Follow`prefill_scaling_analysis_v1`for source/data identities and
the compact analysis. This is no new timing or speed improvement.
The next current-build curve needs a separately committed matched-length/order/
repetition plan,at least five lengths on a compatible route,and first-use,
prefill time and client TTFT kept separate. Revisit the curve after placement,
chunk or scheduling changes;do not combine old and new timing periods merely
because the ELFmatches.

The historical117489result and separate same-engine curve below retain their
original source/configuration limits. That curve has one process per length
and describes actual routes,not an adoption law. On the same
**layer-major, bank-384, non-split** route,
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

The indexed MTP draft-head candidate at `a0ede5973` was tested against
the packed-head projection followed by vocabulary gather. Controller
`5c3e72146` binds the same retained `fde0cdd11` build, model shards and
ordered draft vocabulary. The linked fixture has SHA256 prefix `9e9ac48d0a`
and the compiled recipe has prefix `19ea3964f3`.
The traced smoke and all three timing processes pass the 24 exact and
three masked cases, input immutability and guards. All four retire with
normal exits and no fresh fault. Reference/candidate process medians
are 220.077/320.724, 217.753/310.100 and 234.613/300.157 ms for the
timing batch. Candidate time ratios are 1.457, 1.424 and 1.279.
The component misses the performance gate in all three processes. Keep
the existing projection and gather. Its private `decode-q6-draft-indexed-v1`
build and captures were retired (52 files, 61,262,282 bytes, tree SHA256
`80925e94c2ab6f5da1759bf0be2e2a6761f415184ce439136eaec107fda71f79`).
Before retirement, Git recovered the three source overlays and 472 controller
inputs. The four recorded commands, 797 actual input and executable hashes,
and all 48 archive members matched. The source and controller remain
reachable through `1a2f2ab37`.

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

Scope inference investigation to source code, primary technical references,
ordinary inference logs and measured results. This applies to Main, collection
and review. Exclude executable analysis, disassembly, reverse engineering and
cybersecurity research from this project work.

Before adoption, retain the model prompt-verify, task/agent quality and numerical
checks, failure ownership and graph/allocation retirement, sustained decode,
and **physical 262144-token full fill**, final tails/boundaries,
prefill-to-decode handoff and RESTORE gates. Component address tests or a
smaller context do not satisfy these gates. An API/CLI admission refusal that
reserves output room does not exercise the final physical 262144-token slots;
record the engine entrypoint and actual processed extent for full fill.
Ordinary provider/runtime teardown is not qualified by the retained pre-exec
validation reference through process exit.

Design and select prefill and decode independently. Owned inference functions
and kernels must identify their phase in their name (`prefill_`, `decode_` or
`common_`, or an explicitly phase-qualified namespace); wrappers around external
APIs carry the same distinction. Names describe the actual callers and state
ownership. Existing shared processing remains an audit item until matched
measurements justify it. A change to common processing must check both phases.
If one phase worsens, retain its previous implementation and give the improved
implementation to the benefiting phase. Do not accept the other phase's loss
as the price of that improvement.

When local separation cannot remove the effect, revisit the enclosing
architecture: shared workspaces, weights and cache placement, initialization,
runtime configuration, graph lifetime, scheduling and the state handoff.
Separate ownership and selection at the boundary that actually causes the
coupling; renaming functions alone does not resolve it. Preserve the unchanged
phase's implementation, configuration, placement, correctness and repeated
performance before adoption.

Compare each smallest coherent change separately, even when several changes
are being developed together. Include the headers, callers, producer/consumer
layout changes and tests needed for that unit to remain valid. Evaluate
independent kernels, placement changes and settings in separate matched
comparisons, holding the other units and workload fixed regardless of the extra
build/run effort. Use identical prerequisites in both arms of a dependent
comparison. In the committed plan, identify each reference and state the
invariant that makes any group of edits inseparable. Keep each unit's outcome
and check the combined implementation and relevant interactions, including
the prefill and decode gates for changes to their shared processing.

Preserve the qualified runtime safety settings and immutable pre-exec
SYCL/Level Zero environment in
[`measurement_environment.json`](../tools/sycl/measurement_environment.json).
No provider-teardown claim follows from this setting. Normally retain one validated
fallback plus a named imminent candidate while that comparison is active. When
both independent phase references are needed, follow the
[bounded third-generation rule](ARTIFACT_RETENTION.md#rebuildable-binaries).
Do not infer global cleanup from the current-build inventory. Keep unresolved
legacy source/recipe recovery holds separate from these two comparison
executables, with the missing inputs, consumer, next bounded recovery action
and review point recorded in the retained-build result. Use the current build
selection and concrete holds above, rather than an older generation's retention
instruction, before the next build or collector/model boundary. Historical
milestones remain recoverable from Git and the performance CSV after their
executables are retired. This is not global cleanup.
Other configured
deployment outputs and qualified validation/profiling tools have separate named
consumers. The standalone widening executable,IR and receipt are retired after
numerical/IL choice;successful host/device/build output and superseded source
reviews are also retired. This is not global cleanup:the configured deployments'
original source/recipe recovery and validated migration remain independent holds.

Niko1221/Strata remains the only upstream. Consult the maintained
[`upstream-status.json`](../bench/results/2026-10-15-pr6-recovery-audit/upstream-status.json)
for the last actual check, fetched revision and next due time, and
[`UPSTREAM_INTEGRATION.md`](UPSTREAM_INTEGRATION.md) for functional imports,
component decisions and outstanding validation. Fetching a revision does not
complete its functional ports or the hardware-tuned unmodified-upstream
evaluation. Keep those obligations moving alongside tuning.
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
`feature/sycl`. The published default before this U1round is
`2192aba41ac199a8522cafbe5ef60db5cdf616c0`through PR186;the earlier PR180
permission hold is no longer current. This boundary publishes source/evidence
from the permanent primary while preserving original measured commit identities.
The U1source/evidence publication is [PR187](https://github.com/yqYo1/Strata/pull/187).
Source publication does not close the unfinished functional-import or inference
adoption gates above. The current CSVcontains540points;verify its original
source/evidence ancestors and disabled automatic head deletion before default
publication,then merge the updated default back normally and pull the clean ghq mirror.

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
