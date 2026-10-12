<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Performance comparisons in this workspace

Use upstream's unchanged `bench.sh` and Python client for model performance comparisons.
Measure upstream, XeStrata and local candidates with the same client revision.
Tune each configuration and record the launch command, effective settings, environment
and model/binary/source identities. Do not treat upstream's default settings as its optimum.

Start a dedicated server on `127.0.0.1`, then run in upstream's worktree.
Specify the client's `--url`; the measurement owner starts and stops the server.

```sh
./bench.sh --url http://127.0.0.1:18095 --long --runs 3 --out /path/to/results
```

Use upstream's normal prompt generation, HTTP requests and aggregation.
Do not replace model speed measurements with custom GEN protocols or inputs.
Small kernel, transfer-bandwidth and profiler checks remain diagnostic measurements.

## Assessing results

`--long` adds approximately 32K input tokens. Check each request's actual input and
reused token counts rather than its label alone. Exclude samples below 32,768 tokens
from long-input adoption decisions while retaining their results. Keep short-input
and agent-session measurements as supporting results. Document any method change
needed to add 64K input.

Assess prefill and decode separately. Keep at least three individual samples,
their median and range; increase repetitions when variation prevents a decision.
For long-input decode, use the individual rows in `results.json` and keep short-chat
decode separate. The standard long-input output cap is 24 tokens, so record the
actual generated count as well.

Record GPU progress and errors during first diagnostics, and separate logging and
validation overhead from speed samples. Set each job's deadline and log budget;
verify owned-process closure and GPU state before and after. Check failed requests
in the result as well as the client's exit code. Interrupted or faulty runs cannot
become successful speed samples.

Generated-text equality is not a mandatory quality gate. Separate speed and quality;
assess candidates changing precision or arithmetic routes with fixed-history
likelihood/distributions and task scores. Retain prior custom measurements as
history, without mixing them into the new HTTP comparison. Configuring maximum
context is also distinct from testing the full physical 256K input.

Commit individual results, settings, errors and closure evidence in reports.
After useful evidence is extracted and verified, retire duplicate successful logs
with a path/hash/reason manifest.
