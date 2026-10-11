# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Strict capture validation for one local Prefill stage/run.

No GPU, model, process, or physical bandwidth validation is performed. The
completion field is a production diagnostic declaration, not independent proof.
All geometry and policy expectations must come from the caller. Optional layer
pins must cover the entire stage; without pins, only internal consistency of
the declared types/extents is checked. Strings or chunk-offset maps may specify
mode/issuer expectations, allowing an explicitly described tail policy change.
"""

import re
from collections.abc import Iterable, Mapping

UINT64_MAX = (1 << 64) - 1
INT64_MAX = (1 << 63) - 1
INT32_MAX = (1 << 31) - 1
_UINT = r"(?:0|[1-9][0-9]*)"
_TYPE = rf"(?:-1|{_UINT})"
_FAMILY = "strata prefill expert"
_ROW = re.compile(
    rf"strata prefill expert H2D: device=(?P<device>{_UINT}) "
    rf"stage=\[(?P<lb>{_UINT}),(?P<le>{_UINT})\) "
    rf"chunk_offset=(?P<offset>{_UINT}) pos=(?P<pos>{_UINT}) "
    rf"tokens=(?P<tokens>{_UINT}) layer=(?P<layer>{_UINT}) "
    r"mode=(?P<mode>routed|full_nonresident_stream) "
    r"issuer=(?P<issuer>inline|thread) format=(?P<format>native_gguf|canonical_q2_0) "
    rf"gu_type=(?P<gu_type>{_TYPE}) down_type=(?P<down_type>{_TYPE}) "
    rf"blob_bytes=(?P<blob_bytes>{_UINT}) submitted_calls=(?P<calls>{_UINT}) "
    rf"submitted_bytes=(?P<bytes>{_UINT})"
)
_TOTAL = re.compile(
    r"strata prefill expert H2D total: scope=this_prefill_stage_run "
    rf"device=(?P<device>{_UINT}) stage=\[(?P<lb>{_UINT}),(?P<le>{_UINT})\) "
    rf"pos=(?P<pos>{_UINT}) tokens=(?P<tokens>{_UINT}) chunks=(?P<chunks>{_UINT}) "
    rf"routed_calls=(?P<routed_calls>{_UINT}) routed_bytes=(?P<routed_bytes>{_UINT}) "
    rf"full_stream_calls=(?P<full_stream_calls>{_UINT}) "
    rf"full_stream_bytes=(?P<full_stream_bytes>{_UINT}) "
    rf"submitted_calls=(?P<calls>{_UINT}) submitted_bytes=(?P<bytes>{_UINT}) "
    r"completion=checked; request extents, not PCIe-exclusive bytes or DMA service"
)


class ValidationError(ValueError):
    """The capture or its explicit expected scope is invalid."""


def _integer(value, name, maximum=UINT64_MAX, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError(f"{name}: expected integer in [{minimum},{maximum}]")
    return value


def _number(token, name, maximum=UINT64_MAX):
    if len(token) > 20:
        raise ValidationError(f"{name}: integer overflow")
    value = int(token)
    return _integer(value, name, maximum, -1 if name in ("gu_type", "down_type") else 0)


def _add(a, b, name):
    return _integer(a + b, name)


def _lines(lines):
    if isinstance(lines, str):
        lines = lines.split("\n")
    if not isinstance(lines, Iterable):
        raise ValidationError("stderr must be text or an iterable of text lines")
    for number, raw in enumerate(lines, 1):
        if not isinstance(raw, str):
            raise ValidationError(f"line {number}: expected text")
        # Accept capture-reader CRLF handling, without normalizing other spaces.
        line = raw.removesuffix("\n").removesuffix("\r")
        if "\n" in line or "\r" in line:
            raise ValidationError(f"line {number}: embedded line boundary")
        yield number, line


def _policy(value, name, allowed, chunks, n, limit):
    if isinstance(value, str):
        if value not in allowed:
            raise ValidationError(f"{name}: unsupported expectation")
        return lambda offset: value
    if not isinstance(value, Mapping) or len(value) != chunks:
        raise ValidationError(f"{name}: require a string or exact chunk-offset map")
    for offset, item in value.items():
        _integer(offset, f"{name} offset", INT64_MAX)
        if offset >= n or offset % limit or not isinstance(item, str) or item not in allowed:
            raise ValidationError(f"{name}: invalid chunk-offset map entry")
    return lambda offset: value[offset]


def _layer_record(record, name, format_name):
    if not isinstance(record, Mapping) or set(record) != {"blob_bytes", "gu_type", "down_type"}:
        raise ValidationError(f"{name}: require blob_bytes, gu_type and down_type only")
    extent = _integer(record["blob_bytes"], f"{name} blob_bytes", minimum=1)
    gu = _integer(record["gu_type"], f"{name} gu_type", INT32_MAX, -1)
    down = _integer(record["down_type"], f"{name} down_type", INT32_MAX, -1)
    if format_name == "canonical_q2_0":
        if (gu, down) != (-1, -1):
            raise ValidationError(f"{name}: canonical format requires both type fields -1")
    elif gu < 0 or down < 0:
        raise ValidationError(f"{name}: native format requires nonnegative type fields")
    return {"blob_bytes": extent, "gu_type": gu, "down_type": down}


def validate(lines, *, expected_device, expected_stage, expected_pos0, expected_n,
             expected_chunk_limit, expected_mode, expected_issuer, expected_format,
             expected_layers=None):
    """Return JSON-compatible validated rows/totals, or raise ValidationError.

    expected_stage is [LB, LE). expected_layers, if supplied, is an exact
    integer-layer-keyed mapping of blob_bytes/gu_type/down_type records. n must
    be positive: production's empty run has no completed ledger. No model
    context cap or scheduler policy is inferred. Unrelated stderr is ignored;
    recognizable expert diagnostic lines must match the exact grammar.
    """
    device = _integer(expected_device, "expected_device", INT32_MAX)
    if not isinstance(expected_stage, (tuple, list)) or len(expected_stage) != 2:
        raise ValidationError("expected_stage: require [LB,LE)")
    lb = _integer(expected_stage[0], "expected LB", INT64_MAX)
    le = _integer(expected_stage[1], "expected LE", INT64_MAX)
    if lb >= le:
        raise ValidationError("expected_stage: empty or reversed stage")
    pos0 = _integer(expected_pos0, "expected_pos0", INT64_MAX)
    n = _integer(expected_n, "expected_n", INT64_MAX, 1)
    limit = _integer(expected_chunk_limit, "expected_chunk_limit", INT64_MAX, 1)
    _integer(pos0 + n, "expected end position", INT64_MAX)
    if expected_format not in ("native_gguf", "canonical_q2_0"):
        raise ValidationError("expected_format: unsupported format")
    chunks = (n - 1) // limit + 1
    width = le - lb
    expected_rows = _integer(chunks * width, "expected row count")
    mode_at = _policy(expected_mode, "expected_mode", {"routed", "full_nonresident_stream"}, chunks, n, limit)
    issuer_at = _policy(expected_issuer, "expected_issuer", {"inline", "thread"}, chunks, n, limit)
    pins = None
    if expected_layers is not None:
        if not isinstance(expected_layers, Mapping) or len(expected_layers) != width:
            raise ValidationError("expected_layers: require every stage layer exactly once")
        pins = {}
        for layer, record in expected_layers.items():
            _integer(layer, "expected layer", INT64_MAX)
            if not lb <= layer < le:
                raise ValidationError("expected_layers: layer outside stage")
            pins[layer] = _layer_record(record, f"expected layer {layer}", expected_format)
    rows = []
    layer_metadata = {}
    sums = {"routed": {"calls": 0, "bytes": 0}, "full_nonresident_stream": {"calls": 0, "bytes": 0}}
    total = None
    numeric_row = {"device", "lb", "le", "offset", "pos", "tokens", "layer", "gu_type", "down_type", "blob_bytes", "calls", "bytes"}
    for line_number, line in _lines(lines):
        if _FAMILY not in line:
            continue
        if len(line) > 2048:
            raise ValidationError(f"line {line_number}: oversized diagnostic")
        match_row = _ROW.fullmatch(line)
        match_total = _TOTAL.fullmatch(line)
        if match_row is None and match_total is None:
            raise ValidationError(f"line {line_number}: malformed expert diagnostic")
        if total is not None:
            raise ValidationError(f"line {line_number}: duplicate or second-run ledger")
        if match_row is not None:
            row = match_row.groupdict()
            for field in numeric_row:
                row[field] = _number(row[field], field, INT32_MAX if field in ("gu_type", "down_type") else UINT64_MAX)
            index = len(rows)
            if index >= expected_rows:
                raise ValidationError(f"line {line_number}: extra row")
            offset = (index // width) * limit
            layer = lb + index % width
            geometry = (row["device"], row["lb"], row["le"], row["offset"], row["pos"], row["tokens"], row["layer"])
            if geometry != (device, lb, le, offset, pos0 + offset, min(limit, n - offset), layer):
                raise ValidationError(f"line {line_number}: wrong scope, chunk geometry, layer or row order")
            if (row["mode"], row["issuer"], row["format"]) != (mode_at(offset), issuer_at(offset), expected_format):
                raise ValidationError(f"line {line_number}: wrong mode, issuer or format")
            if row["mode"] == "routed" and row["issuer"] != "inline":
                raise ValidationError(f"line {line_number}: routed issuer must be inline")
            record = _layer_record({key: row[key] for key in ("blob_bytes", "gu_type", "down_type")}, f"line {line_number}", expected_format)
            if pins is not None and record != pins[layer]:
                raise ValidationError(f"line {line_number}: mismatched external layer pins")
            if layer in layer_metadata and record != layer_metadata[layer]:
                raise ValidationError(f"line {line_number}: layer extent/type changed across chunks")
            layer_metadata[layer] = record
            product = _integer(row["calls"] * row["blob_bytes"], f"line {line_number} byte product")
            if product != row["bytes"]:
                raise ValidationError(f"line {line_number}: bytes != calls * blob_bytes")
            mode_sum = sums[row["mode"]]
            for field in ("calls", "bytes"):
                mode_sum[field] = _add(mode_sum[field], row[field], f"{row['mode']} {field} sum")
            rows.append(row)
        else:
            if len(rows) != expected_rows:
                raise ValidationError(f"line {line_number}: total before all exact layer/chunk rows")
            total = {key: _number(value, key) for key, value in match_total.groupdict().items()}
            if (total["device"], total["lb"], total["le"], total["pos"], total["tokens"], total["chunks"]) != (device, lb, le, pos0, n, chunks):
                raise ValidationError(f"line {line_number}: wrong total scope or geometry")
            expected_sums = {
                "routed_calls": sums["routed"]["calls"], "routed_bytes": sums["routed"]["bytes"],
                "full_stream_calls": sums["full_nonresident_stream"]["calls"],
                "full_stream_bytes": sums["full_nonresident_stream"]["bytes"],
                "calls": _add(sums["routed"]["calls"], sums["full_nonresident_stream"]["calls"], "combined calls"),
                "bytes": _add(sums["routed"]["bytes"], sums["full_nonresident_stream"]["bytes"], "combined bytes"),
            }
            if any(total[key] != value for key, value in expected_sums.items()):
                raise ValidationError(f"line {line_number}: inconsistent mode or combined sums")
    if total is None:
        raise ValidationError("missing completed expert H2D total; missing ledger is not zero")
    return {"scope": "this_prefill_stage_run", "capture_valid": True, "rows": rows,
            "total": total, "layer_metadata": layer_metadata, "externally_pinned_layers": pins is not None,
            "completion_declaration": "checked", "independent_completion_validation": False}
