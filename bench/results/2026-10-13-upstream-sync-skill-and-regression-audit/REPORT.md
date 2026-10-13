<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Upstream sync skill and recovered performance history

Created and installed strata-upstream-sync in personal Codex skills. It evaluates
pristine standalone upstreams with box defaults and tuned settings,imports every
non-inference feature through a coverage table,and selects inference by role,
quantization,shape and individual component. Keep better fork components and do
not hide decode regression with prefill gains. Use upstream bench.sh,retain best
settings/call paths and individual samples,and repeat research/context checks.
Validator,document/secret/license checks and two independent synthetic forward
evaluations were completed. This is no real upstream measurement or integration.
Suggestions to make all features and requested parallel research optional were
rejected; scope,unsupported upstream,bounded search and consumed-context evidence
were clarified.

Session history led to PR#6 paired-normal-wall reread from Git:32768 inputs,8192
chunk,context65538,three alternating pairs,medians453.2935 to645.848 t/s,with full
248320-logit and output equality. This is valid historical evidence. Current stock
bench differs in input,timing,context/KV and output horizon; do not compute a direct
regression ratio or discard the old result. PR#8 rates520/533/540 remain single,
instrumented64K observations.

Old layer-major processes every input chunk per layer and reduces expert reloads.
Current root src lacks that route. The old sycl/ topic was not carried into the
root-engine lineage selected at c3aa33ae; the later v0.1.43 server port is not the
shown deletion. Preserve the insufficient-scope R394/R395 v1 reports and their
v2 corrections. Short-context IQ3_S decode18.30 and current long,no-MTP values
near8 differ in conditions; direct decode regression remains unproven.

Repeated12K completed1024,32768,32768 inputs,then refused65536 before workspace
allocation:4383187712 requested against4355325952 free bytes. Preserve passed=false
and complete observation. Following GPU/full-model hashes passed,fault count0.
This proves no64K completion,quality,full physical262144 or speed gain. Retire only
closed duplicate logs through the manifest,preserve original statuses and commit
compact evidence. Registry v193 holds948 reports. The1000/70 t/s target remains open.
