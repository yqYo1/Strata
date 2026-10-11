<!-- SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors -->
<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Community HTTP benchmarks

[日本語](COMMUNITY_BENCHMARKS.ja.md)

Send HTTP requests to a dedicated server and record its environment, individual samples and failures.

## Community reports

Reports are in `bench/results/`. Check their source, hardware, model, conditions and scope of success or failure.
Putting reports with different conditions side by side does not establish a speedup.

## Run the benchmark

Start a dedicated server first, wait for its model to load and confirm that no other requests are running.
The tool attaches with `--url`; it does not start, stop or reconfigure the server.
Requests use the conversation cache. Do not share this server with your regular applications.

```sh
./bench.sh --url http://127.0.0.1:8095 --out ./bench/results
# Without the launcher:
.venv/bin/python tools/strata_bench.py --url http://127.0.0.1:8095 --out ./bench/results
```

For authentication, use `STRATA_KEY` or the `api_key` in `--config FILE`.
Config supplies annotations and a key; it does not apply those settings to the server.
Interactive setup can show a command for another terminal when requested. The default is no.
`xestrata --bench` is the separate native engine benchmark.

| Case | Request | Report |
| --- | --- | --- |
| chat | Fixed short questions, up to 100 new tokens | decode tok/s and time to first token |
| 4K / 16K | Calibrated fresh word lists | prompt tok/s and time to first token |
| agent | system, 4 tools, 5 turns; about 1,500 words added each turn | conversation reuse, reread tokens and turn time |
| `--clients N` | N concurrent requests; requests beyond the slot count wait | total tok/s, p50/p95 latency and failures |
| `--long` | A fresh prompt of about 32,768 tokens | prompt tok/s and time to first token |

Check actual token counts in response usage. Each case runs three times by default.
`--runs` is 1–16 and `--clients` is 0–32. Use long or parallel cases within that engine's checked scope.
Duration depends on the server. HTTP text shares a 64 MiB budget; saved reports also total at most 64 MiB.
Each generation request has a 3,600-second deadline and at most 30 seconds of socket waiting.
Manage a whole-run deadline separately. EOF without `[DONE]`, request failures and budget exhaustion
are excluded from successful samples and cause a nonzero command exit.

## What to record

GPU, VRAM, CPU and RAM come from server HTTP metadata; missing values remain unknown.
Client OS, Python and checkout are separate from the server engine version.
Check that annotated config matches the server's actual settings.

- Source/binary and model/tokenizer/profile identities or hashes
- Context, KV, prefill, spec/MTP, parallel slots and environment
- Actual prompt/generated/reused tokens, individual samples, aggregates and failures
- Conditions the tool does not collect, including power limits, background load and storage

## How to measure

Exclude warm-up and calibration from aggregates, while recording their failures. Give fresh prompts unique tags.
The agent case retains earlier replies. Speed uses engine timings; latency uses the HTTP client's clock.
Match model, inputs, settings and actual token counts; repeat with varied run order when needed.

```sh
./bench.sh --url http://127.0.0.1:8095 --compare ./bench/results/OLD
.venv/bin/python tools/strata_bench.py --compare ./bench/results/OLD --compare-only ./bench/results/NEW
```

## Optional correctness checks

The tool does not verify answer correctness or full-vocabulary logit parity.
Validate fixed-input token/logit comparisons and maximum-context fit separately, and record their results.

## Submit a report

Review paths, keys and user identities in `README.md` and `results.json` before sharing.
Redaction covers known formats only. Check environment-specific information manually and preserve failures and skips.
Server logs remain with their owner under the retention policy. Keep the last diagnostic for unresolved failures.
Retire successful raw captures serially after recording and verifying their useful evidence.

## Report template

Use `not measured` / `not reported` for unmeasured values, with the following sections.

### Hardware and software

Server GPU/VRAM/CPU/RAM, client OS/Python/checkout, and engine/source/binary identities.

### Model and configuration

Input hashes, actual context/KV/prefill/spec/slots/environment, and agreement with annotated config.

### Method

Command, server owner, dedicated/idle check, warm-up, fixture, repetitions, clocks and aggregation.

### Results

Actual tokens, reuse, engine timing, client latency, median/range, failures, skips and exit status.

### Correctness and limitations

Separate correctness/fit evidence and unknown background load, power or storage conditions.
