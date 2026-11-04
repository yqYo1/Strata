#!/usr/bin/env python3
"""Shared strict UR full-trace reducer; repository LICENSE.

Exactly one qualified pre-main negative probe is permitted. The retired Q6
build-audit/run CLI is recoverable from Git; no old ELF or result is reopened.
"""
import re
import ur_validation as validation
from measurement_guards import require


def validation_log(text, pid, loader, traced):
    proof = validation.preload_evidence(text, pid)
    require(proof['loader'] == loader, 'wrong validation loader')
    # The qualified preload intentionally submits one null-platform query to
    # prove parameter validation before main. Full API tracing logs its expected
    # INVALID_NULL_HANDLE; every other diagnostic must still fail admission.
    boundary = text.index('strata UR validation: pid=' + str(pid) + ' phase=before-main ')
    pattern = (r'(?m)^[ \t]*<--- urPlatformGetApiVersion\(\.hPlatform = nullptr, '
        r'\.pVersion = 0x[0-9a-f]+(?: \([^\r\n]*\))?\) '
        r'-> UR_RESULT_ERROR_INVALID_NULL_HANDLE;[ \t]*$')
    expected = list(re.finditer(pattern, text[:boundary]))
    require(len(expected) == int(traced), 'wrong expected negative-probe trace count')
    if expected:
        match = expected[0]
        text = text[:match.start()] + text[match.end():]
    require(not any(word in text for word in
        ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', '[WARNING]', '[ERROR]')),
        'unexpected validation/provider diagnostic')
    return proof
