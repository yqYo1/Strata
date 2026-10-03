# SYCL implementation status, 2026-10-03

This report describes the Linux Intel Arc B570 worktree, rather than every
feature advertised by the upstream CUDA/HIP engine. Measurements use a B570
with 10 GiB VRAM, Ryzen 5 5600X, 125 GiB system RAM, oneAPI DPC++ 2026.1.1 and
Level Zero driver `1.17.39395+14`. Driver defaults were retained. Other workloads
on the workstation were not isolated.

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
| Prompt processing | Chunked prefill, FP16 expert expansion and oneMKL matrix products, split attention fallback | Short prompts, 16-token chunks; longer prompt performance remains open |
| MTP | Draft, verification, rejection/commit and state rewind; short-window native graph recording | Persistent generation and cancellation recovery |
| Server | Existing Python server, OpenAI and Anthropic API formats, streaming and web UI | Local HTTP smoke checks; this is not a full application/agent test suite |
| KV formats | FP16, INT8, Q4 and hybrid K8V4 operator/snapshot paths | Kernel tests; measured model configuration uses FP16 and context 512 |
| Vision | Existing engine image-position/embedding paths remain in source | No complete image encoder plus SYCL model run validated |
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
ids matched, both without a GPU expert cache and with a fixed 1,649-slot cache.
Cache-off minimum/mean cosine were 0.976301/0.997614; cached values were
0.983445/0.997201. Logits were finite but not bit-identical to the reference.
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

## Comparison with the latest upstream

The comparison uses an independent, read-only `ghq` clone of
[Niko1221/Strata at `99f3dbd`](https://github.com/Niko1221/Strata/tree/99f3dbd0b21d1401b3769e0c0d963913607f380b),
committed 2026-10-03 00:21:36 UTC. The common base is
`1678de333d0e0711bc414ad992b640e1a37dd814`. Upstream has 96 commits beyond that
base and changes 81 files, with 11,050 added and 645 removed lines. These are
branch-divergence counts, not 96 separately missing features. The local engine
still identifies as 0.1.34; the upstream engine identifies as 0.1.38.
The upstream changes were inspected, not merged into this worktree.

| Upstream area | Current SYCL gap | Practical implication |
| --- | --- | --- |
| Quantized expert prefill | CUDA fused INT8 experts, native MMQ and their newer streamed staging are not ported to SYCL | Prompt processing is the largest remaining port/performance gap; the current path expands experts to FP16 |
| Prompt attention | Newer upstream matrix attention optimizations are not ported; SYCL `qsa_prompt_attn_batch` requests the split fallback | Long-prompt throughput needs separate implementation and measurement |
| Second GPU expert tier | New `--peer-device`, peer cache and peer prompt work are absent; the SYCL adapter accepts ordinal 0 only | The SYCL engine currently uses one GPU |
| Adaptive cache controls | New `--adapt-decay` and `--expert-profile-save` persistence are absent | Existing adaptive replacement works, but learned residency is not saved by these newer interfaces |
| CPU IQ4_XS | New AVX2 multi-token IQ4_XS expert kernel has not been imported | The existing fallback handles this format; the requested model has one IQ4_XS gate/up layer |
| Memory/startup policy | Newer reserve heuristics, allocation diagnostics and Windows low-RAM changes are not all present | Current settings are measured on this Linux B570, not a general small-card preset |
| Server recovery/diagnostics | New silent-engine timeout/restart, startup-log diagnostics and mid-prompt cancellation accounting are not imported | Basic API tests pass, but recent upstream failure handling is missing |
| Server request protections | Latest Host/Origin checks and their new tests are not imported | This fork does not yet include those upstream request checks; the measured server binds to 127.0.0.1 |
| Server compatibility/metrics | New malformed-message handling, draft-count metrics and thinking-budget notices are not all imported | The standard API paths work, but matching the latest server behavior needs an explicit update |
| Installer/updater | Setup supports NVIDIA/AMD discovery, not the Intel SYCL build and model setup used here; newer upstream updater/setup changes are also absent | The SYCL installation remains a manual Linux workflow |

Most upstream web/API/model-family features predate the common base and already
exist in the fork. The gaps above are specific missing updates or SYCL paths;
they do not mean the whole upstream server or model loader must be rewritten.
Conversely, this worktree's SYCL runtime and optimized B570 kernels are its own
additions, not upstream features available on Intel through the standard setup.

Next port priorities are quantized expert prefill and matrix prompt attention,
then the newer server fixes and profile persistence. Multi-GPU, complete vision
validation and Windows installation require their own devices and test runs.
Measured decode tuning cannot establish those capabilities.

See [SYCL implementation and build instructions](SYCL_IMPLEMENTATION.md),
[measurement storage policy](../bench/results/README.md) and the
[model repository](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S).
