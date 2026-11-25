"""Reconcile traced DFlash2 serve windows with the native DONE counters.

The forced-width check uses confidence zero, no secondary draft limit and no
lookup drafts. The serve loop starts at the last prompt token. It commits the
anchor and accepted prefix before applying the output limit or EOS. This
reader checks that geometry independently of the captured kernel timings.
"""

from collections import Counter
import re


CONTEXT = 262144
REQUEST = re.compile(r"strata trace: request (\d+) 0")
WINDOW = re.compile(r"strata trace: window (\d+) (\d+)")

Q6_ROUTE = re.compile(
    r"strata Q6_ROUTE request=(\d+) device=0 phase=(prompt|decode) tokens=([1-8]) "
    r"capture_id=(\d+) scope=live_capture_cumulative native_half_enabled=0 complete=1 "
    r"shapes=\d+ cache_hits=\d+ replay_started=(\d+) replay_completed=(\d+)"
)
Q6_HEAD = re.compile(
    r"strata Q6_DPAS_HEAD request=(\d+) device=0 phase=(prompt|decode) tokens=([1-8]) "
    r"capture_id=(\d+) layout=canonical extra_weight_bytes=0 "
    r"canonical_table_excludes_dpas=1 calls=(\d+) complete=1"
)

Q6_HEAD_SHAPE = re.compile(
    r"strata Q6_SHAPE request=1 device=0 phase=(prompt|decode) tokens=([1-8]) capture_id=(\d+) "
    r"ni=2560 no=248320 nc=([1-8]) reference_calls=(\d+) columns_calls=(\d+) "
    r"centered_calls=0 native_half_calls=0"
)


def audit_windows(windows, response, *, prompt, output_limit, width, context=CONTEXT):
    """Check row extents, the accepted prefix and the published output count."""
    if width not in (4, 5) or not 1 <= prompt <= context or output_limit < 1:
        raise ValueError("invalid width diagnostic geometry")
    if not windows or windows[0] != (prompt - 1, 1) or response["prompt"] != prompt:
        raise ValueError("the first window must consume only the last prompt token")

    offered = 0
    for index, (position, rows) in enumerate(windows):
        expected_rows = 1 if index == 0 else min(width, context - position)
        if not 0 <= position < context or rows != expected_rows:
            raise ValueError(f"unexpected DFlash2 window at position {position} with {rows} rows")
        offered += rows - 1
    if response["offered"] != offered or not 0 <= response["accepted"] <= offered:
        raise ValueError("draft counters disagree with the traced windows")

    earlier_commits = [following[0] - current[0]
                       for current, following in zip(windows, windows[1:])]
    emitted_before_last = sum(earlier_commits)
    final_commit = len(windows) + response["accepted"] - emitted_before_last
    commits = [*earlier_commits, final_commit]
    if any(not 1 <= committed <= rows
           for (_, rows), committed in zip(windows, commits, strict=True)):
        raise ValueError("accepted prefix exceeds its verified window")

    emitted = response["generated"]
    finish = response["finish"]
    if (finish not in ("length", "stop") or
            not emitted_before_last < emitted <= min(output_limit, emitted_before_last + final_commit) or
            (finish == "length" and emitted != output_limit)):
        raise ValueError("output extent disagrees with the committed prefixes and finish reason")

    return {
        "width_counts": dict(sorted(Counter(rows for _, rows in windows).items())),
        "accepted_prefix_counts": dict(sorted(Counter(rows - 1 for rows in commits).items())),
        "final_window": list(windows[-1]),
        "final_commit": final_commit,
        "commit_end": windows[-1][0] + final_commit,
        "committed_after_last_output": emitted_before_last + final_commit - emitted,
    }


def audit_requests(text, requests, responses, *, width):
    """Join the bare native text requests and windows in stderr wire order."""
    traces = []
    for line in text.splitlines():
        if line.startswith("strata trace: request "):
            request = REQUEST.fullmatch(line)
            if request is None:
                raise ValueError("unexpected native request trace")
            traces.append((int(request[1]), []))
        elif line.startswith("strata trace: window "):
            window = WINDOW.fullmatch(line)
            if window is None or not traces:
                raise ValueError("malformed or unowned native window trace")
            traces[-1][1].append((int(window[1]), int(window[2])))

    result = []
    for (ids, budget), response, (prompt, windows) in zip(requests, responses, traces, strict=True):
        if len(ids) != prompt:
            raise ValueError("native request trace disagrees with the committed prompt length")
        result.append(audit_windows(windows, response, prompt=prompt, output_limit=budget, width=width))
    return result


def audit_head(text, width_counts, *, c5=False):
    """Join the first request's completed graph replays and head capture calls."""
    captures = {}
    heads = {}
    fallback_calls = {}
    for line in text.splitlines():
        if line.startswith("strata Q6_PACKED_HEAD "):
            raise ValueError("sole-packed target head was used")
        if line.startswith("strata Q6_ROUTE "):
            match = Q6_ROUTE.fullmatch(line)
            if match is None or match[1] != "1" or match[5] != match[6]:
                raise ValueError("incomplete or unexpected verifier replay counters")
            if match[2] == "decode":
                rows, capture, completed = int(match[3]), int(match[4]), int(match[6])
                if rows in captures or capture == 0:
                    raise ValueError("duplicate or missing verifier capture identity")
                captures[rows] = (capture, completed)
        elif line.startswith("strata Q6_DPAS_HEAD "):
            match = Q6_HEAD.fullmatch(line)
            if match is None or match[1] != "1":
                raise ValueError("incomplete or unexpected target head capture counters")
            rows, capture, calls = map(int, match.groups()[2:])
            if match[2] == "prompt":
                if calls:
                    raise ValueError("decode-only DPAS head reached a prompt capture")
                continue
            if rows in heads:
                raise ValueError("duplicate target head capture")
            heads[rows] = (capture, calls)
        elif line.startswith("strata Q6_SHAPE ") and " ni=2560 no=248320 " in line:
            match = Q6_HEAD_SHAPE.fullmatch(line)
            if match is None:
                raise ValueError("unexpected canonical head shape")
            if match[1] == "prompt":
                continue
            rows, capture, columns, reference, column_calls = map(int, match.groups()[1:])
            if columns != rows or rows in fallback_calls:
                raise ValueError("duplicate or mismatched canonical head shape")
            fallback_calls[rows] = (capture, reference, column_calls)

    if not width_counts or captures.keys() != heads.keys() or captures.keys() != width_counts.keys():
        raise ValueError("decode head and graph captures must cover every observed width")
    if len({capture for capture, _ in captures.values()}) != len(captures):
        raise ValueError("decode widths share an unexpected capture identity")
    dpas_widths = {4, 5} if c5 else {4}
    if fallback_calls.keys() != captures.keys() - dpas_widths:
        raise ValueError("canonical head shapes must cover the widths outside DPAS")
    result = []
    for rows, (capture, completed) in sorted(captures.items()):
        head_capture, calls = heads[rows]
        dpas = rows in dpas_widths
        if completed != width_counts[rows] or head_capture != capture or calls != int(dpas):
            raise ValueError("target head capture route or replay count differs from traced work")
        if not dpas and fallback_calls[rows] != (capture, int(rows == 5), int(rows == 1)):
            raise ValueError("canonical head fallback differs from the selected Q6 route")
        route = "dpas" if dpas else {1: "q6-columns", 5: "q6-reference"}[rows]
        result.append(dict(rows=rows, capture_id=capture, completed=completed, route=route))
    return result
