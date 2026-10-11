#!/bin/sh
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# Benchmark this Strata install the way it is used and write a report you can share (docs/COMMUNITY_BENCHMARKS.md).
# Attach to a dedicated server you already started: ./bench.sh --url http://127.0.0.1:8095
# It sends inference requests and changes that server's conversation cache; it never manages its lifecycle.
# Options: --clients 4 (several requests at once), --long (a 32K prompt), --compare <older results folder>.
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python tools/strata_bench.py "$@"
fi
echo "Strata is not set up here yet: run ./setup.sh first."
exit 1
