# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Validate the distinct completed host API overhead line, not GPU timing."""

import math
import re
from collections.abc import Iterable

UINT64_MAX = (1 << 64) - 1
_UINT = r"(?:0|[1-9][0-9]*)"
_MS = r"(?:0|[1-9][0-9]*)\.[0-9]{3}"
_FAMILY = "strata prefill timer"
_LINE = re.compile(
    rf"strata prefill timer overhead: host wall CPU API overhead: mark (?P<mark_ms>{_MS}) ms "
    rf"\(marker calls (?P<marker_calls>{_UINT}), new event creations (?P<new_events>{_UINT})\), "
    rf"fold (?P<fold_ms>{_MS}) ms \(fold calls (?P<fold_calls>{_UINT}), "
    rf"completed intervals (?P<intervals>{_UINT})\); not GPU service or timer calibration"
)


class ValidationError(ValueError):
    """The overhead capture or explicit enabled expectation is invalid."""


def validate(lines, *, timing_enabled):
    """Require one exact line if enabled, otherwise absence and no measurements.

    timing_enabled must reflect presence of STRATA_PREFILL_TIMING, including
    values '0' and ''. Counts are actual declared wrapper counts. Intervals are
    not timestamp endpoint API call counts. Other phase text is ignored.
    """
    if type(timing_enabled) is not bool:
        raise ValidationError("timing_enabled must be an explicit bool")
    if isinstance(lines, str):
        lines = lines.split("\n")
    if not isinstance(lines, Iterable):
        raise ValidationError("stderr must be text or an iterable of text lines")
    overhead = None
    for number, raw in enumerate(lines, 1):
        if not isinstance(raw, str):
            raise ValidationError(f"line {number}: expected text")
        line = raw.removesuffix("\n").removesuffix("\r")
        if "\n" in line or "\r" in line:
            raise ValidationError(f"line {number}: embedded line boundary")
        if _FAMILY not in line:
            continue
        if not timing_enabled:
            raise ValidationError(f"line {number}: overhead diagnostic present with timing disabled")
        if overhead is not None:
            raise ValidationError(f"line {number}: duplicate or second-run overhead diagnostic")
        match = _LINE.fullmatch(line) if len(line) <= 2048 else None
        if match is None:
            raise ValidationError(f"line {number}: malformed or nonfinite/negative overhead diagnostic")
        overhead = {}
        for key, token in match.groupdict().items():
            if key.endswith("_ms"):
                value = float(token)
                if not math.isfinite(value) or value < 0:
                    raise ValidationError(f"line {number}: invalid {key}")
            else:
                if len(token) > 20 or int(token) > UINT64_MAX:
                    raise ValidationError(f"line {number}: {key} overflow")
                value = int(token)
            overhead[key] = value
        if overhead["marker_calls"] < 1:
            raise ValidationError(f"line {number}: completed enabled diagnostic requires positive marker calls")
        if overhead["new_events"] > overhead["marker_calls"]:
            raise ValidationError(f"line {number}: new_events > marker_calls")
        if overhead["intervals"] > overhead["marker_calls"] - 1:
            raise ValidationError(f"line {number}: intervals > marker_calls - 1")
        if overhead["intervals"] and not overhead["fold_calls"]:
            raise ValidationError(f"line {number}: intervals without a fold call")
    if timing_enabled and overhead is None:
        raise ValidationError("missing completed overhead diagnostic")
    return {"capture_valid": True, "timing_enabled": timing_enabled, "overhead": overhead,
            "independent_completion_validation": False}
