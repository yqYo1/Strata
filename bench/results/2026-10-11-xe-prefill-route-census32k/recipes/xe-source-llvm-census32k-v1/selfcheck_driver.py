# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Finite parser fixtures only; no Runtime, GPU, model or submission mock.

Main may run with no arguments for selfchecks, or supply --actual-stderr FILE
--expected-json FILE and exactly one of --timing-enabled / --timing-disabled.
The JSON object contains transfer_ledger.validate's explicit expected_* kwargs;
layer keys and optional mode/issuer chunk-offset keys are decimal strings.
No receipt/environment/file discovery is performed. Success only validates
diagnostic capture syntax/scope/arithmetic; Main owns overall result status.
"""

import argparse
import copy
import json
from pathlib import Path

import timer_overhead
import transfer_ledger


def _fixture():
    # Synthetic small diagnostic grammar fixture, not a route/copy simulation.
    lines = [
        "unrelated engine line",
        "strata prefill expert H2D: device=0 stage=[2,4) chunk_offset=0 pos=10 tokens=2 layer=2 mode=routed issuer=inline format=native_gguf gu_type=1 down_type=2 blob_bytes=16 submitted_calls=0 submitted_bytes=0",
        "strata prefill expert H2D: device=0 stage=[2,4) chunk_offset=0 pos=10 tokens=2 layer=3 mode=routed issuer=inline format=native_gguf gu_type=3 down_type=4 blob_bytes=24 submitted_calls=1 submitted_bytes=24",
        "strata prefill expert H2D: device=0 stage=[2,4) chunk_offset=2 pos=12 tokens=1 layer=2 mode=routed issuer=inline format=native_gguf gu_type=1 down_type=2 blob_bytes=16 submitted_calls=2 submitted_bytes=32",
        "strata prefill expert H2D: device=0 stage=[2,4) chunk_offset=2 pos=12 tokens=1 layer=3 mode=routed issuer=inline format=native_gguf gu_type=3 down_type=4 blob_bytes=24 submitted_calls=1 submitted_bytes=24",
        "strata prefill expert H2D total: scope=this_prefill_stage_run device=0 stage=[2,4) pos=10 tokens=3 chunks=2 routed_calls=4 routed_bytes=80 full_stream_calls=0 full_stream_bytes=0 submitted_calls=4 submitted_bytes=80 completion=checked; request extents, not PCIe-exclusive bytes or DMA service",
        "unrelated closing line",
    ]
    expected = {
        "expected_device": 0, "expected_stage": [2, 4], "expected_pos0": 10,
        "expected_n": 3, "expected_chunk_limit": 2, "expected_mode": "routed",
        "expected_issuer": "inline", "expected_format": "native_gguf",
        "expected_layers": {
            2: {"blob_bytes": 16, "gu_type": 1, "down_type": 2},
            3: {"blob_bytes": 24, "gu_type": 3, "down_type": 4},
        },
    }
    return lines, expected


def _replace(lines, index, before, after):
    result = list(lines)
    if before not in result[index]:
        raise AssertionError("fixture mutation did not match")
    result[index] = result[index].replace(before, after, 1)
    return result


def _expect_rejection(name, callback, error_type):
    try:
        callback()
    except error_type:
        return
    raise AssertionError(f"{name}: invalid capture was accepted")


def selfcheck():
    positive = 0
    negative = 0
    lines, expected = _fixture()
    result = transfer_ledger.validate(lines, **expected)
    if len(result["rows"]) != 4 or result["total"]["bytes"] != 80:
        raise AssertionError("valid fixture result mismatch")
    positive += 1
    unpinned = {**expected, "expected_layers": None}
    transfer_ledger.validate("\n".join(lines), **unpinned)
    positive += 1
    transfer_ledger.validate([line + "\r\n" for line in lines], **expected)
    positive += 1

    canonical = [line.replace("format=native_gguf", "format=canonical_q2_0")
                 .replace("gu_type=1 down_type=2", "gu_type=-1 down_type=-1")
                 .replace("gu_type=3 down_type=4", "gu_type=-1 down_type=-1") for line in lines]
    canonical_expected = copy.deepcopy(expected)
    canonical_expected["expected_format"] = "canonical_q2_0"
    for record in canonical_expected["expected_layers"].values():
        record.update(gu_type=-1, down_type=-1)
    transfer_ledger.validate(canonical, **canonical_expected)
    positive += 1

    mixed = list(lines)
    for index in (1, 2):
        mixed[index] = mixed[index].replace("mode=routed issuer=inline", "mode=full_nonresident_stream issuer=thread")
    mixed[5] = mixed[5].replace("routed_calls=4 routed_bytes=80 full_stream_calls=0 full_stream_bytes=0",
                               "routed_calls=3 routed_bytes=56 full_stream_calls=1 full_stream_bytes=24")
    mixed_expected = {**expected, "expected_mode": {0: "full_nonresident_stream", 2: "routed"},
                      "expected_issuer": {0: "thread", 2: "inline"}}
    transfer_ledger.validate(mixed, **mixed_expected)
    positive += 1

    # A finite full-stage 32K capture shape; all-zero rows still must be present.
    large = []
    for offset in range(0, 32767, 4096):
        for layer in range(48):
            large.append(f"strata prefill expert H2D: device=0 stage=[0,48) chunk_offset={offset} pos={offset} tokens={min(4096, 32767 - offset)} layer={layer} mode=routed issuer=inline format=native_gguf gu_type=1 down_type=2 blob_bytes=16 submitted_calls=0 submitted_bytes=0")
    large.append("strata prefill expert H2D total: scope=this_prefill_stage_run device=0 stage=[0,48) pos=0 tokens=32767 chunks=8 routed_calls=0 routed_bytes=0 full_stream_calls=0 full_stream_bytes=0 submitted_calls=0 submitted_bytes=0 completion=checked; request extents, not PCIe-exclusive bytes or DMA service")
    large_expected = {**unpinned, "expected_stage": [0, 48], "expected_pos0": 0,
                      "expected_n": 32767, "expected_chunk_limit": 4096}
    if len(transfer_ledger.validate(large, **large_expected)["rows"]) != 384:
        raise AssertionError("32K capture shape mismatch")
    positive += 1

    bad_transfers = [
        ("missing ledger", []), ("missing total", lines[:5]),
        ("missing zero row", lines[:1] + lines[2:]),
        ("duplicate row", lines[:2] + [lines[1]] + lines[2:]),
        ("out of order", lines[:1] + [lines[2], lines[1]] + lines[3:]),
        ("wrong device", _replace(lines, 1, "device=0", "device=1")),
        ("wrong stage", _replace(lines, 1, "stage=[2,4)", "stage=[1,4)")),
        ("wrong offset", _replace(lines, 3, "chunk_offset=2", "chunk_offset=1")),
        ("wrong position", _replace(lines, 3, "pos=12", "pos=13")),
        ("wrong tail", _replace(lines, 3, "tokens=1", "tokens=2")),
        ("wrong layer", _replace(lines, 4, "layer=3", "layer=4")),
        ("wrong mode", _replace(lines, 1, "mode=routed", "mode=full_nonresident_stream")),
        ("wrong issuer", _replace(lines, 1, "issuer=inline", "issuer=thread")),
        ("wrong format", _replace(lines, 1, "format=native_gguf", "format=canonical_q2_0")),
        ("wrong type", _replace(lines, 1, "gu_type=1", "gu_type=5")),
        ("negative native type", _replace(lines, 1, "gu_type=1", "gu_type=-1")),
        ("type overflow", _replace(lines, 1, "gu_type=1", "gu_type=2147483648")),
        ("zero extent", _replace(lines, 1, "blob_bytes=16", "blob_bytes=0")),
        ("wrong pinned extent", _replace(lines, 1, "blob_bytes=16", "blob_bytes=17")),
        ("negative count", _replace(lines, 1, "submitted_calls=0", "submitted_calls=-1")),
        ("noncanonical count", _replace(lines, 1, "submitted_calls=0", "submitted_calls=00")),
        ("counter overflow", _replace(lines, 1, "submitted_calls=0", "submitted_calls=18446744073709551616")),
        ("product mismatch", _replace(lines, 2, "submitted_bytes=24", "submitted_bytes=25")),
        ("wrong mode sum", _replace(lines, 5, "routed_bytes=80", "routed_bytes=79")),
        ("wrong combined sum", _replace(lines, 5, "submitted_calls=4", "submitted_calls=5")),
        ("wrong chunk count", _replace(lines, 5, "chunks=2", "chunks=3")),
        ("wrong total position", _replace(lines, 5, "pos=10", "pos=11")),
        ("wrong completion", _replace(lines, 5, "completion=checked", "completion=unchecked")),
        ("wrong scope", _replace(lines, 5, "scope=this_prefill_stage_run", "scope=all_runs")),
        ("truncated total", lines[:5] + [lines[5][:-12]]),
        ("duplicate total", lines + [lines[5]]),
        ("second ledger", lines + lines),
        ("second truncated ledger", lines + ["strata prefill expert H2D: device=0"]),
        ("prefixed ledger", _replace(lines, 1, "strata prefill expert", "prefix strata prefill expert")),
    ]
    for name, capture in bad_transfers:
        _expect_rejection(name, lambda capture=capture: transfer_ledger.validate(capture, **expected), transfer_ledger.ValidationError)
        negative += 1
    _expect_rejection("unpinned changed type", lambda: transfer_ledger.validate(_replace(lines, 3, "gu_type=1", "gu_type=5"), **unpinned), transfer_ledger.ValidationError)
    negative += 1
    _expect_rejection("unpinned changed zero-row extent", lambda: transfer_ledger.validate(_replace(lines, 1, "blob_bytes=16", "blob_bytes=17"), **unpinned), transfer_ledger.ValidationError)
    negative += 1
    overflow = _replace(lines, 1, "submitted_calls=0 submitted_bytes=0",
                        "submitted_calls=18446744073709551615 submitted_bytes=18446744073709551615")
    _expect_rejection("byte product overflow", lambda: transfer_ledger.validate(overflow, **expected), transfer_ledger.ValidationError)
    negative += 1
    # Each individual product fits, but the per-mode accumulated sum overflows.
    sum_overflow = [line.replace("blob_bytes=16", "blob_bytes=1").replace("blob_bytes=24", "blob_bytes=1") for line in lines]
    sum_overflow = _replace(sum_overflow, 1, "submitted_calls=0 submitted_bytes=0",
                            "submitted_calls=18446744073709551615 submitted_bytes=18446744073709551615")
    sum_overflow = _replace(sum_overflow, 2, "submitted_bytes=24", "submitted_bytes=1")
    _expect_rejection("sum overflow", lambda: transfer_ledger.validate(sum_overflow, **unpinned), transfer_ledger.ValidationError)
    negative += 1
    combined_overflow = [line.replace("blob_bytes=16", "blob_bytes=1").replace("blob_bytes=24", "blob_bytes=1") for line in mixed]
    combined_overflow = _replace(combined_overflow, 1, "submitted_calls=0 submitted_bytes=0",
                                 "submitted_calls=18446744073709551615 submitted_bytes=18446744073709551615")
    combined_overflow = _replace(combined_overflow, 2, "submitted_calls=1 submitted_bytes=24", "submitted_calls=0 submitted_bytes=0")
    combined_overflow = _replace(combined_overflow, 3, "submitted_bytes=32", "submitted_bytes=2")
    combined_overflow = _replace(combined_overflow, 4, "submitted_bytes=24", "submitted_bytes=1")
    _expect_rejection("combined overflow", lambda: transfer_ledger.validate(combined_overflow, **{**mixed_expected, "expected_layers": None}), transfer_ledger.ValidationError)
    negative += 1
    for name, bad_expected in [
        ("missing layer pins", {**expected, "expected_layers": {2: expected["expected_layers"][2]}}),
        ("invalid expected bool", {**expected, "expected_device": True}),
        ("zero chunk limit", {**expected, "expected_chunk_limit": 0}),
        ("missing policy chunk", {**mixed_expected, "expected_mode": {0: "routed"}}),
        ("position overflow", {**expected, "expected_pos0": (1 << 63) - 1}),
    ]:
        _expect_rejection(name, lambda bad_expected=bad_expected: transfer_ledger.validate(lines, **bad_expected), transfer_ledger.ValidationError)
        negative += 1

    timer = "strata prefill timer overhead: host wall CPU API overhead: mark 1.250 ms (marker calls 4, new event creations 2), fold 2.500 ms (fold calls 2, completed intervals 3); not GPU service or timer calibration"
    phase = "strata prefill timing: existing phase text is outside this parser"
    timer_result = timer_overhead.validate([phase, timer], timing_enabled=True)
    if timer_result["overhead"]["intervals"] != 3:
        raise AssertionError("timer fixture result mismatch")
    positive += 1
    timer_overhead.validate(timer.replace("1.250", "0.000").replace("2.500", "0.000"), timing_enabled=True)
    positive += 1
    if timer_overhead.validate([phase], timing_enabled=False)["overhead"] is not None:
        raise AssertionError("disabled timing must not manufacture measurements")
    positive += 1
    bad_timers = [
        ("missing overhead", [phase]), ("duplicate overhead", [timer, timer]),
        ("truncated overhead", [timer[:-8]]),
        ("NaN", [timer.replace("1.250", "nan")]),
        ("Inf", [timer.replace("2.500", "inf")]),
        ("nonfinite decimal", [timer.replace("1.250", "9" * 400 + ".000")]),
        ("negative ms", [timer.replace("1.250", "-1.250")]),
        ("negative zero", [timer.replace("1.250", "-0.000")]),
        ("negative count", [timer.replace("marker calls 4", "marker calls -4")]),
        ("count overflow", [timer.replace("fold calls 2", "fold calls 18446744073709551616")]),
        ("events exceed markers", [timer.replace("new event creations 2", "new event creations 5")]),
        ("intervals exceed markers minus one", [timer.replace("completed intervals 3", "completed intervals 4")]),
        ("zero markers", [timer.replace("marker calls 4", "marker calls 0")]),
        ("intervals without fold", [timer.replace("fold calls 2", "fold calls 0")]),
        ("wrong suffix", [timer.replace("not GPU service", "GPU service")]),
    ]
    for name, capture in bad_timers:
        _expect_rejection(name, lambda capture=capture: timer_overhead.validate(capture, timing_enabled=True), timer_overhead.ValidationError)
        negative += 1
    _expect_rejection("disabled overhead present", lambda: timer_overhead.validate([timer], timing_enabled=False), timer_overhead.ValidationError)
    negative += 1
    _expect_rejection("disabled malformed overhead present", lambda: timer_overhead.validate(["strata prefill timer overhead:"], timing_enabled=False), timer_overhead.ValidationError)
    negative += 1
    return {"synthetic_parser_positive_cases": positive, "synthetic_parser_negative_cases": negative,
            "scope": "parser capture validation only"}


def _offset_keys(mapping, name):
    if not isinstance(mapping, dict):
        return mapping
    converted = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or len(key) > 20:
            raise ValueError(f"{name}: expected canonical decimal JSON keys")
        number = int(key)
        if str(number) != key:
            raise ValueError(f"{name}: noncanonical decimal JSON key")
        converted[number] = value
    return converted


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-stderr", type=Path)
    parser.add_argument("--expected-json", type=Path)
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--timing-enabled", action="store_true")
    timing.add_argument("--timing-disabled", action="store_true")
    args = parser.parse_args()
    supplied = args.actual_stderr is not None or args.expected_json is not None or args.timing_enabled or args.timing_disabled
    if not supplied:
        result = selfcheck()
    else:
        if args.actual_stderr is None or args.expected_json is None or not (args.timing_enabled or args.timing_disabled):
            parser.error("actual capture requires both files and one explicit timing expectation")
        config = json.loads(args.expected_json.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(config, dict):
            raise ValueError("expected JSON must be an object of explicit expected_* arguments")
        for key in ("expected_layers", "expected_mode", "expected_issuer"):
            if key in config:
                config[key] = _offset_keys(config[key], key)
        capture = args.actual_stderr.read_text(encoding="utf-8")
        result = {"transfer": transfer_ledger.validate(capture, **config),
                  "timer": timer_overhead.validate(capture, timing_enabled=args.timing_enabled)}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
