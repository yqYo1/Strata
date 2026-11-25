"""Measure the pinned reconciliation prompt through an owned Strata server."""

import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import strata_bench as client

from dflash2_reconciliation_fixture import build, validate_answer


INPUT = 35915
OUTPUT_CAP = 16384
WIRE_SHA256 = '06b1845151ce2fa7a79fa020108b9c44b9afb0dd77046475dd4cb1c6441aa748'
FOOTER = re.compile(
    r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms '
    r'\([0-9.]+ tok/s\), (\d+) generated in (\d+) ms \([0-9.]+ tok/s\), '
    r'drafts accepted (\d+) of (\d+), \d+ checkpoints(?: \(cancelled\))?$'
)
FIELDS = ('prompt', 'reused', 'read', 'prompt_ms_rounded', 'generated', 'decode_ms_rounded', 'accepted', 'offered')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def request_body():
    fixture = build(54)
    body = client.chat_body(fixture['messages'], OUTPUT_CAP, True)
    require(hashlib.sha256(json.dumps(body).encode()).hexdigest() == WIRE_SHA256,
            'the generated reconciliation wire body changed')
    return fixture, body


def history_row(before, after):
    seen = {row['id'] for row in before['requests']}
    new = [row for row in after['requests'] if row['id'] not in seen]
    require(len(new) == 1, 'need exactly one completed request on the private server')
    return new[0]


def native_segments(log):
    for line in log.splitlines():
        if line.startswith('strata serve: prompt '):
            match = FOOTER.fullmatch(line)
            require(match is not None, 'unexpected native request footer')
            yield dict(zip(FIELDS, map(int, match.groups()), strict=True))


def account(raw, history, segments, *, context, expected):
    sample = client.digest(raw)
    eligibility = client.long_generation.sample_eligibility(raw, sample, maximum=OUTPUT_CAP, context=context)
    require(raw['request_sha256'] == WIRE_SHA256 and sample['prompt_tokens'] == INPUT
            and sample['reused_tokens'] == 0 and sample['read_tokens'] == INPUT,
            'completed request differs from the pinned fresh input')
    require(segments and (segments[0]['prompt'], segments[0]['reused'], segments[0]['read']) == (INPUT, 0, INPUT)
            and all(s['read'] + s['reused'] == s['prompt'] <= context
                    and 0 <= s['accepted'] <= s['offered'] for s in segments),
            'native segments do not form fully accounted requests')
    generated, accepted, offered = (sum(s[k] for s in segments) for k in ('generated', 'accepted', 'offered'))
    require((history['prompt_total'], history['prompt_read'], history['reused'], history['output_tokens'],
             history['engine_generated'], history['drafts_accepted'], history['drafts_offered']) ==
            (INPUT, INPUT, 0, sample['generated_tokens'], generated, accepted, offered)
            and (sample['drafts_accepted'], sample['drafts_offered']) == (accepted, offered)
            and sample['finish_reason'] == history['finish'], 'SSE, history and native request counters disagree')
    for field in ('prompt_ms', 'decode_ms'):
        exact = history[field]
        rounded = sum(s[field + '_rounded'] for s in segments)
        require(type(exact) in (int, float) and math.isfinite(exact) and exact > 0
                and abs(exact - rounded) <= .500001 * len(segments)
                and sample[field] == round(exact, 1), 'SSE, history and rounded native clocks disagree: ' + field)

    try:
        validate_answer(raw['text'], expected)
    except ValueError as error:
        answer_ok, answer_error = False, str(error)
    else:
        answer_ok, answer_error = True, None
    native_tg = 1000 * generated / history['decode_ms']
    enough_native = generated >= client.long_generation.MIN_OUTPUT_TOKENS
    return dict(sample=sample, eligibility=eligibility, native_segments=segments, native_generated=generated,
                api_generated=sample['generated_tokens'], native_tg=native_tg,
                minimum_server_added_tokens=max(0, sample['generated_tokens'] - generated),
                native_generation_long_enough=enough_native,
                weak_drafts_observed=enough_native and eligibility['agreement_eligible'] and accepted * 5 <= offered * 2,
                reasoning_recoveries=history['reasoning_recoveries'],
                answer_correct=answer_ok, answer_error=answer_error,
                answer_sha256=hashlib.sha256(raw['text'].encode()).hexdigest(),
                thinking_sha256=hashlib.sha256(raw['reasoning_text'].encode()).hexdigest())


def collect(base, key, native_stderr, *, context):
    """Send one request. The caller owns the server and its finite job budget."""
    fixture, body = request_body()
    require(type(context) is int and INPUT + OUTPUT_CAP + client.long_generation.CONTEXT_RESERVE <= context,
            'the server context cannot hold this request')
    require(client.count_chat_body_tokens(base, key, body, timeout=30) == INPUT,
            'live tokenization differs from the pinned reconciliation extent')
    before = client.http_json(base, '/metrics?requests=all', key)
    with Path(native_stderr).open('rb') as native:
        native.seek(0, 2)
        start = native.tell()
        raw = client.chat(base, key, fixture['messages'], OUTPUT_CAP, True, timeout=2400)
        history = history_row(before, client.http_json(base, '/metrics?requests=all', key))
        # The engine writes DONE to stdout before emitting its stderr footer.
        for _ in range(20):
            native.seek(start)
            log = native.read((1 << 20) + 1)
            require(len(log) <= 1 << 20, 'reconciliation footer capture exceeded its budget')
            segments = list(native_segments(log.rpartition(b'\n')[0].decode()))
            if segments and sum(s['generated'] for s in segments) == history['engine_generated']:
                break
            time.sleep(.1)
    return account(raw, history, segments, context=context, expected=fixture['expected'])
