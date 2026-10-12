<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Upstream v0.1.43 server completion

Remote HEADs still match the already measured official v0.1.43 (`7695859`) and
XeStrata xe0.1.42 (`da3d426`). The existing32K pristine-source samples remain the
engine update evidence. This work closes shared server gaps found during the recheck.

The single-turn queue now sends SSE keepalive comments every10 seconds while waiting.
Cancellation and generator close leave the queue without releasing another turn.
A lazy startup or restart that never reaches READY returns503, preserving the reason,
and clears startup state even when the slot condition was never initialized.

Frontend ports retain the first order of each tool-name set, recognize spaced tool
names with their schema, read refusal blocks as text, and report ignored content parts
at debug level. The parser keeps quoted reasoning end tags intact across fragment
boundaries and accumulates long lines in bounded blocks before materialization.

Main ran the full fake-engine/HTTP regression:555 tests,544 passed,11 skipped.
The skips are existing missing tokenizer/Pillow/jsonschema cases. New queue, READY,
tool-order/name, content and parser boundary tests passed. Fresh baseline/candidate
Python lints have no new findings; existing findings remain. The first complete run's
one fixture error and the earlier lint failures remain failures in their receipts.
The SlowRelease fixture now accepts timed acquire and still forces and verifies the
original status handover race, with all original accounting assertions retained.

All318 engine/build/setup/data source files are byte-identical to0691bcdf. No model
or GPU inference was run for this Python port, and no new speed claim follows from
the fake-engine timings. The prior model binary and its measurement identities remain
separate. CUDA/HIP inference changes were not selected. Optional replica/Windows
features, fully optimized pristine parameters, physical262K and RESTORE remain outside
this qualification.

See results.json, engine-source-equivalence.json and the receipt index for exact
source/environment/command/test outcomes. The implementation report and Main corrections
are retained in registryv165. Continue performance work from the qualified model recipe;
the separately committed grouped-event diagnostic confirms its XMX path in synthetic cells.
