"""Bind bulky private build inputs to a compact, committed plan.

Repository LICENSE. This checks metadata, not source recovery or actual bytes.
Callers must obtain expected pins from the committed plan and separately verify
the original Git inputs, generated files, tools and environment before launch.
"""
import hashlib
import json
import re


FIELDS = ('project_inputs', 'generated_inputs', 'dependency_inputs',
          'external_inputs', 'environment', 'compiler_inputs')


def pins(record):
    if any(name not in record for name in FIELDS):
        raise RuntimeError('missing private build input field')
    return {name: hashlib.sha256(json.dumps(record[name], sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest() for name in FIELDS}


def verify(record, committed_pins):
    if not isinstance(committed_pins, dict) or set(committed_pins) != set(FIELDS) or any(
        not isinstance(h, str) or re.fullmatch('[0-9a-f]{64}', h) is None for h in committed_pins.values()):
        raise RuntimeError('incomplete committed build metadata pins')
    if pins(record) != committed_pins:
        raise RuntimeError('private build metadata differs from committed inputs')
