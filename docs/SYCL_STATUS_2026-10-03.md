# SYCL implementation status, updated 2026-10-04

This report describes the Linux Intel Arc B570 worktree, rather than every
feature advertised by the upstream CUDA/HIP engine. Measurements use a B570
with 10 GiB VRAM, Ryzen 5 5600X, 125 GiB system RAM, oneAPI DPC++ 2026.1.1 and
Level Zero driver `1.17.39395+14`. Driver defaults were retained. Other workloads
on the workstation were not isolated. The tuning table predates the upstream
sync/rebase; post-rebase checks are recorded separately below.
The final section records the subsequent single-GPU feature ports. Earlier
performance tables remain measurements of their original builds.

## Model acquisition and coverage

The requested model is `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S,
at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`. Both GGUF shards were
verified against the repository's SHA-256 values:

| Shard | Bytes | SHA-256 |
| --- | ---: | --- |
| 1 | 54,817,524,224 | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| 2 | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

The total is 83,617,662,656 bytes. Shard 2 is identical to the already verified
Q2_0 shard 2 and shares its immutable local file through a hard link. This
avoids a second download and disk copy without modifying the Q2_0 model.
The generated native expert arena contains 50,292,326,400 bytes, or 46.84 GiB.
The model and generated packs remain outside Git.

IQ3_S is the model's file label, not the format of every tensor. Its 48 expert
layers use these gate/up and down combinations:

| Gate/up | Down | Layers |
| --- | --- | ---: |
| IQ2_S | IQ4_NL | 15 |
| IQ3_XXS | IQ4_NL | 14 |
| IQ3_S | IQ4_NL | 9 |
| IQ2_S | Q2_0 | 5 |
| IQ3_XXS | Q2_0 | 3 |
| IQ3_S | Q2_0 | 1 |
| IQ4_XS | IQ4_NL | 1 |

The native quantized GPU path supports 13 formats; IQ2_XXS, IQ2_XS and IQ1_M
are not implemented there yet. Their absence does not affect this mixed IQ3_S
GGUF, which uses none of those formats.

The new SYCL support covers IQ2_S, IQ3_XXS and IQ3_S. Existing IQ4_NL, IQ4_XS
and Q2_0 paths serve the other experts. Dense tensors also retain their original
Q4_K, Q5_K, Q6_K and other supported formats. There is no extra requantization.
The MTP drafter uses the existing separate draft pack from the same checkpoint;
its quantization need not match the target model's IQ3_S label.

## Measured IQ3_S configuration and tuning

The selected local preset uses context 512, FP16 KV, four CPU workers,
16-token prefill, the shipped expert profile, automatic per-layer GPU caching,
64 adaptive replacements every four verification rounds, MTP width four,
minimum draft probability 0.9 and PCIe share zero. It is now the default local
server config. The preceding Q2_0 preset is saved separately.

Each cache slot previously reserved the largest expert blob, even for layers
with smaller formats. The SYCL change sizes slots for their owning layer and
regenerates their ranges on a smaller allocation retry. Automatic sizing keeps
the preceding byte budget and leaves the same reserve. The measured cache grew
from 1,649 to 2,137 slots, or 29.6%, while using approximately 4.09 GiB in both
configurations. An explicit 1,649-slot compact cache uses 3.15 GiB.

Three complete uniform/compact pairs ran in alternating fresh processes. Each
served the 37-token story prompt twice, the 45-token Python merge prompt twice,
a story cancelled after its first emitted token, then eight recovery tokens.
Every main request generated 128 tokens. Median decode rates were:

| Request | Uniform slots, token/s | Layer-sized slots, token/s | Change |
| --- | ---: | ---: | ---: |
| First story | 12.27 | 12.94 | +5.4% |
| Repeated story | 14.07 | 14.37 | +2.1% |
| First Python merge prompt | 15.71 | 16.43 | +4.6% |
| Repeated Python merge prompt | 17.81 | 18.30 | +2.7% |

Timers exclude model startup and prompt processing. First requests include lazy
graph preparation. Repeated requests reuse 30/37 and 38/45 prompt tokens and
retain expert-cache contents. Background workloads were active; these modest
changes are workstation measurements, not isolated laboratory estimates. All
three processes with the same configuration retained identical output ids,
cache/reuse counts, draft counts and cancellation behavior across six requests.
Automatic cache placement differs between the two variants, so cross-variant
continuations are not asserted identical.

For the three compact processes after the screens, median startup to `READY`
was 16.35 seconds. Median first story/code prompt processing was 6.97/8.79
seconds; repeated prompt processing was 1.44/1.43 seconds. The first-prompt
figures cover only 37/45 input tokens and do not predict long-prompt speed.
The decode table excludes all of these times.

The earlier parameter screens retained 13 fresh-process samples: probability
floors 0.5/0.7/0.9/0.99, two/four/five/six workers in the recorded combinations,
and 64/128/256 adaptive replacements. Increasing replacement traffic or worker
count did not produce a consistent improvement across both prompts. The final
preset therefore retains probability 0.9, four workers and 64 replacements.
All screen and paired samples are summarized, without discarding slower runs.

IQ3_S remains slower than the earlier Q2_0 measurements on this PC: Q2_0's
repeated story/code medians were 25.46/29.70 token/s. That is an earlier model
measurement, not a matched simultaneous comparison. IQ3_S puts more expert
bytes in RAM and fewer experts in the same GPU's cache. This tuning did not
bring IQ3_S to approximately 29 token/s. No task-quality comparison between
these quantizations was performed.

The selected IQ3_S preset passed real HTTP checks for `/v1/models`, the web UI,
OpenAI chat/streaming, Anthropic messages, repeat output, completed-stream output
and cancellation recovery. Its canonical API model name is
`qwen3.8-flash-next-iq3_s-sycl`; `qwen3.8-flash-next-sycl` remains a tested alias
for existing apps. The test server stopped cleanly. The existing Q2_0 preset
also passed a separate 16-token regression check against its prior fixture.

The default local config is `~/.local/share/strata-sycl/serve-config.json`;
`serve-config-iq3_s.json` stores the same IQ3_S preset and
`serve-config-q2_0.json` preserves the preceding Q2_0 preset. From this checkout:

```sh
source /opt/intel/oneapi/setvars.sh
PYTHONPATH=tools ~/.local/share/strata-sycl/venv/bin/python -m serve.server \
  --engine strata --config ~/.local/share/strata-sycl/serve-config.json \
  --host 127.0.0.1 --port 18085
```

The OpenAI base URL is `http://127.0.0.1:18085/v1`. To run the saved Q2_0 preset,
use its config path in the same command. The measured context remains 512.
Reviewed results and validation scope are in
[`bench/results/2026-10-03-sycl-iq3s/run.json`](../bench/results/2026-10-03-sycl-iq3s/run.json).
Raw logs, generated trial files, binary dumps, model files and workstation
configs remain outside Git. Local raw captures are preserved under
`~/.local/state/strata-sycl/measurement-archive/2026-10-03-iq3_s/`.

## What runs today

| Area | Implementation | Validation boundary |
| --- | --- | --- |
| Model loading | Split GGUF reader, native quantized projections, resident CPU expert arena, on-demand PLE table reads | Q2_0 and mixed IQ3_S model files loaded on this machine |
| Decoder | Normalization, RoPE, router/top-k, shared experts, GDN recurrence, attention and sampling | GPU operator tests and real short-context generation |
| CPU/GPU experts | Profile-filled GPU cache, CPU misses with AVX2, grouped multi-token GPU experts, adaptive replacement | Real persistent text requests; mixed-format row and grouped tests |
| Dense projections | Native quantized MMVQ, with ESIMD paths for selected existing formats | Unit tests and sampled rows from actual model tensors |
| Runtime | SYCL queues/events/USM and the engine's CUDA-compatible adapter; native and segmented graph capture | Runtime, graph, ordering and large-allocation checks on B570 |
| Prompt processing | Native quantized MMQ, fused INT8 experts, oneMKL dense products and matrix/split attention | Independent operator references and real IQ3_S prompt logits; 16- and 64-token chunks |
| MTP | Draft, verification, rejection/commit and state rewind; short-window native graph recording | Persistent generation and cancellation recovery |
| Server | Existing Python server, OpenAI and Anthropic API formats, streaming and web UI | Local HTTP smoke checks; this is not a full application/agent test suite |
| KV formats | FP16, INT8, Q4 and hybrid K8V4 operator/snapshot paths | Kernel tests; measured model configuration uses FP16 and context 512 |
| Vision | CPU or SYCL image encoder, image embeddings and M-RoPE positions | One B570 ran the SYCL encoder and IQ3_S model through both image APIs, streaming and ordered two-image requests |
| Other models/platforms | Shared model-family code remains available | Coder/Swift/Unsloth checkpoints, Windows and other Intel GPUs have not been validated here |

An API endpoint working does not establish long-context behavior, image support
or agent reliability. The validated local configuration has a 512-token context.
Increasing that setting is possible in the engine, but 128K/256K context has not
been measured on this SYCL configuration.

## Numerical checks

The IQ decoder checks use the pinned ggml library's independent `to_float`
functions, not the newly added SYCL decoder as the expected result. They cover
all 1,024 IQ2_S, 256 IQ3_XXS and 512 IQ3_S codebook indices, signs, subscales,
finite FP16 scale edges, odd row counts and one/three/eight activation columns.
FP32 transfer results match ggml bits. FP16 transfers and interleaved prefill
transfers are checked against their expected representations.

The actual IQ3_S files passed the first/middle/last row check for 447 supported
tensors with three activation columns. Maximum
`|error| / (1 + sum|terms|)` was `6.59863e-8` against a scalar dequantized dot
product. All 34 existing `sycl_` tests passed after adding the three formats.

A separate CPU-only llama.cpp reference at pinned commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d` teacher-forced a 25-token arithmetic
chat. The target used FP16 KV, context 128 and four CPU workers. All 25 argmax
ids matched with automatic sizing (2,530 slots) and a fixed 1,649-slot cache.
The first condition was originally labelled cache-off incorrectly: with a
profile, `--expert-cache 0` selects automatic sizing. Its minimum/mean cosine
were 0.976301/0.997614; fixed-cache values were 0.983445/0.997201. Logits were finite but not bit-identical to the reference.
The final position predicted `4` for `What is 2+2?`.

An ordinary 32-token greedy run with MTP disabled also completed with finite
logits. Its continuation differed from the fixed-placement MTP run from token
index 11. Different CPU single-/multi-token paths and CPU/GPU arithmetic are
not established as bit-equivalent. The storage optimization was checked within
the same MTP mode and fixed placement: 64 output ids and all float32 bits in
69 full-vocabulary rows across 36 windows matched, including rejected rows.
This distinction matters when interpreting exactness claims.

CPU experts and native GPU experts use different activation quantization and
float addition orders. Changing GPU residency can therefore change a close
sampling decision. The upstream engine also documents cache-dependent rounding.
These checks do not assert that whole-model logits equal a CPU-only run, or that
IQ3_S has better task quality than Q2_0. No broad benchmark of model quality was
performed. Fixture responses are truncated at their requested token limit.

## Upstream integration snapshot (2026-10-03)

The reference is the independent, read-only `ghq` clone of
[Niko1221/Strata at `99f3dbd`](https://github.com/Niko1221/Strata/tree/99f3dbd0b21d1401b3769e0c0d963913607f380b),
committed 2026-10-03 00:21:36 UTC. Its common base with the original SYCL work
was `1678de333d0e0711bc414ad992b640e1a37dd814`: 96 upstream commits changed 81
files, with 11,050 added and 645 removed lines. Those are historical divergence
counts, not a count of missing features.

[PR #1](https://github.com/yqYo1/Strata/pull/1) imported the remaining 83 commits
into the fork's main and was merged as
`d988955e141190bc079b58ab3d3038da637ad070`. The main tree matches upstream
`99f3dbd` and the clean local main mirrors the remote. All 77 SYCL commits were
rebased onto it. The SYCL build now identifies as 0.1.38. Integration commit
`3205987` supplies the newer host interfaces while retaining SYCL's existing
GPU arithmetic and single-device boundary. The original branch was retained
locally as `backup/sycl-before-upstream-2026-10-03`.

This table records the state immediately after the rebase, before the
2026-10-04 ports in the final section.

| Area | After integration | Remaining validation or port work |
| --- | --- | --- |
| Quantized expert prefill | Shared staging changes are imported; SYCL retains FP16 expansion and oneMKL | CUDA fused INT8 experts and native MMQ still need a SYCL implementation |
| Prompt attention and CUDA kernel tuning | Upstream sources are present; SYCL keeps the validated split fallback and its own kernels | Matrix prompt attention, newer CUDA GDN/cluster optimizations are not SYCL paths |
| Second GPU expert tier | Shared peer code is imported | SYCL still accepts one selected GPU; `--peer-device` is rejected before model loading |
| Adaptive cache controls | `--adapt-decay` and `--expert-profile-save` are imported | Local preset continues to use the shipped profile; learned-file ranking was checked below |
| CPU IQ4_XS | New AVX2 multi-token kernel is imported | Real IQ3_S requests exercise this CPU; no broad quality comparison is claimed |
| Memory/startup policy | New shared reserve, loading and diagnostic changes are imported | Small-card presets and Windows low-RAM paths need their own hardware tests |
| Server recovery, diagnostics and request checks | New timeout/restart handling, Host/Origin checks, malformed-message handling and metrics are imported | Mock tests and local real-model HTTP checks pass; no full agent reliability test is claimed |
| Installer/updater | New upstream setup/update code is imported | Intel SYCL discovery, automatic build/model setup and Windows installation remain unimplemented here |

Most web/API/model-family features already existed before the sync. The SYCL
runtime and B570 kernels are fork additions. At that checkpoint, quantized
expert prefill and matrix prompt attention were the next backend priorities.
The subsequent ports and vision checks are recorded below. Multi-GPU and Intel
setup automation are excluded from that work.

## Post-rebase checks

The rebuilt engine passed all 35 CTest tests: 34 SYCL checks and the imported
expert-profile serialization test. The new grouped-expert API hints retain the
same SYCL results; scattered rows were checked on device and host USM with holes
and a canary, and rejected peer copies leave their destinations untouched.
The imported server passed 139 mock-engine tests. All 11 setup test scripts
passed in separate processes after fixing two Linux portability issues in the
fixtures: Windows archives must be tested with `strata.exe`, and executable
normalization must leave `strata-*.log` filenames intact. The unmodified upstream
setup failures are recorded in PR #1; the fixture fixes change no installer code.

Against the same pinned CPU-only reference, the 25-token arithmetic fixture was
checked again with three actual cache configurations:

| GPU expert cache | Argmax matches | Minimum cosine | Mean cosine |
| --- | ---: | ---: | ---: |
| Disabled, no profile | 25/25 | 0.976385 | 0.997371 |
| Automatic, 2,530 slots | 25/25 | 0.976301 | 0.997614 |
| Fixed, 1,649 slots | 25/25 | 0.983445 | 0.997201 |

All logits were finite. The automatic and fixed-cache dumps retained every
float32 bit across all 25 full-vocabulary rows from their pre-rebase runs.
That check is ordinary teacher-forced decode; it does not establish bitwise
MTP/ordinary equivalence or broader task quality.

One fresh persistent IQ3_S process completed two story requests, two Python
requests, cancellation after its first token and an eight-token recovery. Each
main request produced 128 tokens; recovery retained the first eight ids. The
cache still held 2,137 experts in approximately 4.09 GiB. Startup was 16.98 s;
first/repeated decode rates were 12.74/14.74 token/s for the story and
16.84/18.90 for Python. These are one integration check, not a new median or a
speedup claim. Original three-pair tuning medians above remain historical.
`--expert-profile-save` also wrote a 196,632-byte file whose complete 24,576-pair
ranking and reverse lookup were verified.

The rebuilt engine and imported server passed real HTTP checks on
`127.0.0.1:18085`: model listing, web UI, both APIs, streaming, repeat output,
old model alias and cancellation recovery. The test server stopped cleanly.
The selected local IQ3_S config and Q2_0 backup remain available at the paths
above. Those checks used context 512; the subsequent work adds vision and
larger-context validation.

Reviewed post-rebase evidence is in
[`bench/results/2026-10-03-sycl-upstream-rebase/run.json`](../bench/results/2026-10-03-sycl-upstream-rebase/run.json).
Raw logs, float dumps, per-request outputs and temporary configs remain outside
Git under `~/.local/state/strata-sycl/measurement-archive/2026-10-03-upstream-rebase/`.

## Single-GPU feature ports (2026-10-04)

The comparison used upstream `99f3dbd0b21d1401b3769e0c0d963913607f380b`,
also returned by a final remote-main check. The fork's main remains the clean
mirror at `d988955`; implementation work is on `feature/sycl`. The requested
scope excludes multi-GPU and the setup script.

| Identified gap | Implementation and validation |
| --- | --- |
| Four native expert formats | Added Q5_1, IQ2_XXS, IQ2_XS and IQ1_M to device decoding, single/grouped products and transfer decoding. SYCL now covers the same 17 native MMVQ formats as upstream CUDA. Independent pinned-GGML tests cover codebooks, signs, subscales, finite FP16 edges, affine offsets and IQ1_M's embedded scale. |
| Native quantized prefill | Added routed MMQ with padded Q8_1 activations, arbitrary row counts, row maps, strides and direct packed products for all 17 formats. Q2_0 plane gathering and native raw-block gathering are covered by independent references and guards. The SYCL path is built in; it needs no GGML CUDA kernels. |
| Fused MoE prefill | Added GPU routing counts, offsets, placement and 64-row tiles; per-token INT8 inputs; direct expert blobs; fused gate/up, SwiGLU and hidden INT8 rounding; and down output. Tests cover Q2_0 plus all 12 upstream native format pairs, empty experts, partial batches, 65-row experts, table reuse and memory guards. |
| Matrix prompt attention | Added Intel joint_matrix FP16 products with FP32 per-group scales, query/probability hi+lo parts and online softmax for FP16, INT8, Q4_0 and K8V4. Independent double references cover paging, masks, negative cell ids, empty selections, invalid widths, mixed query magnitudes and output guards. Unsupported devices/shapes retain split attention. |
| Intel GPU image encoding | Added `STRATA_VISION_SYCL` to the optional encoder build. The pinned CPU and SYCL encoders were checked independently, then the SYCL encoder and IQ3_S model passed real OpenAI/Anthropic image requests on one B570. [Build and image validation details](SYCL_IMPLEMENTATION.md#intel-gpu-image-encoder-2026-10-04). |
| Hybrid K8V4 storage | Fixed the SYCL validator's rejection of upstream's folded K/V pool arguments. Append assigns one writer to each physical output; gather accepts identical folded sources and outputs. Independent encoded-byte/scale references and memory guards cover batch append, device-step append and gather. Different inputs sharing an output remain rejected. |

The rebuilt engine passed all 38 CTest tests, including the three new MMQ,
fused-MoE and matrix-attention tests. The shared server passed 139 tests again.
The production server and setup script remain the imported upstream versions.

For the 25-token arithmetic fixture, the first target logits after prefill
were compared with the last row of the pinned CPU-only reference. All 248,320
logits were finite and the top token was `19` (`4`) in each condition:

| Prompt path | Cosine with CPU | RMSE | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| Previous FP16 expert expansion and split attention | 0.991999 | 0.26158 | 1.4774 |
| Native MMQ and split attention | 0.994525 | 0.21223 | 1.2786 |
| Native MMQ and matrix attention | 0.995078 | 0.21617 | 1.1764 |
| Fused native experts and matrix attention | 0.990063 | 0.30414 | 1.6587 |

The cache had 1649 slots, adaptation was off, context was 128, prefill was 16,
and MTP width was four at minimum probability 0.9. The fused check used
`STRATA_PF_FUSED=1`, `STRATA_PREFILL_STREAM_MIN=16` and
`STRATA_PREFILL_RING=128`; its log confirmed execution of the fused path.
These are one prompt's logits, not a broad quality or speed comparison.

At context capacity 2048 and prefill 64, four separate persistent IQ3_S
processes checked FP16, INT8, Q4_0 and K8V4 KV. Each completed six requests:
an 827-token color-retrieval prompt, its repeat, a 25-token arithmetic prompt,
the color prompt again, cancellation after one generated token, and recovery.
All four returned `blue` and `4` for the corresponding prompts. Repeat and
recovery reused 820 prompt tokens and preserved output ids; every process
exited cleanly. K8V4's initial failure led to the folded-buffer fix above.
The other three completed checks preceded that fix, which changes only folded
buffer handling. This is one retrieval fixture above 512 tokens, not a maximum
context or general long-context quality result. Resident KV streaming was off
in these model checks; the independent KV test covers FP16/INT8/Q4 eviction,
ring restoration, overflow and staging. K8V4 resident streaming remains
unsupported by upstream too.

The current text default also completed two 128-token story requests, two
128-token Python requests, cancellation after one token and an eight-token
recovery in one persistent process. Recovery matched the initial story's first
eight ids and the process exited cleanly. This used the automatic 2137-slot
cache, context 512, prefill 16, four workers, 64 adaptive replacements, and MTP
width four at minimum probability 0.9. It checks the new prompt paths with the
existing longer decode/MTP flow; it is not a new throughput median.

MMQ and fused products use portable integer SIMD arithmetic; CUDA's tensor
instructions and cluster scheduling are not reproduced on Intel. Decode and
CPU handoff retain the validated SYCL graphs/events. These execution choices
preserve the corresponding single-GPU operations but do not establish equal
CUDA/SYCL performance. The historical decode medians above have not been
remeasured for the newly changed prompt paths.

Reviewed evidence is in
[`bench/results/2026-10-04-sycl-functional-parity/run.json`](../bench/results/2026-10-04-sycl-functional-parity/run.json).
Raw logs, per-request outputs, float dumps and unsuccessful development/harness
attempts remain outside Git at
`~/.local/state/strata-sycl/measurement-archive/2026-10-04-functional-parity/`.

See [SYCL implementation and build instructions](SYCL_IMPLEMENTATION.md),
[measurement storage policy](../bench/results/README.md) and the
[model repository](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S).
