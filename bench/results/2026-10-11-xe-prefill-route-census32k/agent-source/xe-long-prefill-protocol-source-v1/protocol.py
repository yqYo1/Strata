# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bounded fresh GEN request protocol; not GPU completion or numerical parity."""
import math
import re

INT64_MAX = (1 << 63) - 1
MAX_CHUNKS = 1024
MAX_GENERATED = 4096
MAX_CAPTURE_BYTES = 1 << 20
MAX_LINES = 10000
MAX_LINE_BYTES = 4096
_UINT = re.compile(r'(?:0|[1-9][0-9]*)')
_FLOAT = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')
_TRACE_FAMILY = 'strata trace: prompt chunk'
_TRACE = re.compile(r'strata trace: prompt chunk ((?:0|[1-9][0-9]*)) of ((?:0|[1-9][0-9]*))')


class ValidationError(ValueError):
    """Capture or caller contract does not satisfy this fresh-request grammar."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def integer(value, name, minimum=0, maximum=INT64_MAX):
    require(type(value) is int and minimum <= value <= maximum,
            f'{name}: bounded integer required')
    return value


def uint(token, name):
    require(len(token) <= 19 and _UINT.fullmatch(token) is not None,
            f'{name}: canonical int64 decimal required')
    return integer(int(token), name)


def real(token, name, nonpositive=False):
    require(len(token) <= 128 and _FLOAT.fullmatch(token) is not None,
            f'{name}: decimal floating value required')
    value = float(token)
    require(math.isfinite(value), f'{name}: finite value required')
    require(value <= 0 if nonpositive else value >= 0, f'{name}: invalid sign')
    return value


def geometry(n_prompt, batched_n, chunk_limit, physical_context):
    integer(n_prompt, 'n_prompt', 2)
    integer(batched_n, 'batched_n', 1)
    integer(chunk_limit, 'chunk_limit', 1)
    integer(physical_context, 'physical_context', 2, (1 << 31) - 1)
    require(n_prompt <= physical_context and batched_n == n_prompt - 1,
            'fresh prompt/batched/context contract mismatch')
    chunks = (batched_n - 1) // chunk_limit + 1
    require(chunks <= MAX_CHUNKS, 'finite chunk count budget exceeded')
    # Count bounded before creating starts/ends; never range over prompt tokens.
    starts = [index * chunk_limit for index in range(chunks)]
    ends = [min(start + chunk_limit, batched_n) for start in starts]
    return starts, ends


def bounded_lines(lines):
    # Accept a whole request capture only, without silently filtering stdout.
    if isinstance(lines, str):
        require(len(lines) <= MAX_CAPTURE_BYTES, 'capture character budget exceeded')
        require(len(lines.encode('utf-8')) <= MAX_CAPTURE_BYTES, 'capture byte budget exceeded')
        lines = lines.split('\n')
        if lines and lines[-1] == '':
            lines.pop()
    out = []
    size = 0
    for raw in lines:
        require(isinstance(raw, str), 'text line required')
        require(len(raw) <= MAX_LINE_BYTES + 2, 'line character budget exceeded')
        line = raw.removesuffix('\n').removesuffix('\r')
        require('\r' not in line and '\n' not in line, 'embedded line boundary')
        width = len(line.encode('utf-8'))
        require(width <= MAX_LINE_BYTES, 'line byte budget exceeded')
        size += width + 1
        require(size <= MAX_CAPTURE_BYTES and len(out) < MAX_LINES, 'capture budget exceeded')
        out.append(line)
    return out


def fields(line, tag, count):
    parts = line.split(' ')
    require(len(parts) == count and parts[0] == tag and all(parts),
            f'exact {tag} grammar/order required')
    return parts


def validate(lines, *, n_prompt, batched_n, chunk_limit, max_generated,
             physical_context, vocabulary, output_requirement, require_lp5):
    """Validate one fresh GEN request's stdout, including all terminal fields.

    output_requirement: 'positive' or explicitly 'exhausted_zero'. The latter
    requires n_prompt==physical_context and exactly zero generated output. It
    accepts only that declared protocol contract; it does not establish that
    any binary necessarily exhausts its context at precisely this boundary.
    require_lp5 is explicit bool: true requires one complete LP5 after every T;
    false requires no LP lines. Startup, QUIT, batch, reuse, cancellation and
    lookup-chain extensions are outside this grammar and rejected.
    """
    starts, ends = geometry(n_prompt, batched_n, chunk_limit, physical_context)
    integer(max_generated, 'max_generated', 1, MAX_GENERATED)
    integer(vocabulary, 'vocabulary', 5, (1 << 31) - 1)
    require(type(require_lp5) is bool, 'explicit LP5 boolean required')
    require(output_requirement in ('positive', 'exhausted_zero'), 'explicit output requirement required')
    if output_requirement == 'exhausted_zero':
        require(n_prompt == physical_context, 'zero output contract requires physical context boundary')
    capture = bounded_lines(lines)
    require(len(capture) >= len(ends) + 3, 'truncated fresh request')
    cursor = 0
    require(fields(capture[cursor], 'RESUME', 2)[1] == '0', 'fresh RESUME0 required')
    cursor += 1
    progress = []
    previous_ms = 0.0
    for end in ends:
        p = fields(capture[cursor], 'PP', 5)
        require(uint(p[1], 'PP end') == end and uint(p[2], 'PP prompt') == n_prompt,
                'wrong PP identity/end/count/order')
        elapsed, rate = real(p[3], 'PP elapsed'), real(p[4], 'PP rate')
        require(elapsed >= previous_ms, 'PP elapsed regressed')
        previous_ms = elapsed
        progress.append({'end': end, 'elapsed_ms': elapsed, 'reported_tokens_per_second': rate})
        cursor += 1
    require(fields(capture[cursor], 'REUSED', 2)[1] == '0', 'fresh REUSED0 required')
    cursor += 1
    outputs = []
    lp_rows = []
    while cursor < len(capture) and capture[cursor].startswith('T '):
        require(len(outputs) < max_generated, 'output exceeds caller maximum')
        token = uint(fields(capture[cursor], 'T', 2)[1], 'output token')
        require(token < vocabulary, 'output token outside vocabulary')
        outputs.append(token)
        cursor += 1
        if require_lp5:
            require(cursor < len(capture), 'missing paired LP5')
            p = fields(capture[cursor], 'LP', 7)
            selected = real(p[1], 'selected logprob', nonpositive=True)
            top = []
            seen = set()
            for item in p[2:]:
                pair = item.split(':')
                require(len(pair) == 2, 'exact LP5 id:logprob required')
                ident = uint(pair[0], 'LP5 token')
                probability = real(pair[1], 'LP5 logprob', nonpositive=True)
                require(ident < vocabulary and ident not in seen, 'LP5 ID invalid or duplicate')
                require(not top or probability <= top[-1]['logprob'], 'LP5 ranking regressed')
                if ident == token:
                    require(abs(probability - selected) <= 0.000001, 'paired selected LP disagrees')
                seen.add(ident)
                top.append({'id': ident, 'logprob': probability})
            require(selected <= top[0]['logprob'] + 0.000001, 'selected LP exceeds maximum')
            # Printed logprobs have six decimal places; allow their finite rounding.
            require(sum(math.exp(item['logprob']) for item in top) <= 1.000003,
                    'LP5 probability mass exceeds one')
            lp_rows.append({'selected_logprob': selected, 'top5': top})
            cursor += 1
    require(cursor < len(capture), 'missing DONE')
    d = fields(capture[cursor], 'DONE', 16)
    require(cursor + 1 == len(capture), 'extra protocol after DONE')
    names = {1: 'generated', 2: 'prompt', 6: 'drafts_accepted', 7: 'drafts_offered',
             8: 'reused', 9: 'cache_hits', 10: 'cache_lookups', 11: 'ram_blobs',
             12: 'file_blobs', 14: 'read_n', 15: 'offloaded'}
    counts = {name: uint(d[index], name) for index, name in names.items()}
    prompt_ms, decode_ms, file_mb = (real(d[i], name) for i, name in
                                    ((3, 'prompt_ms'), (4, 'decode_ms'), (13, 'file_mb')))
    generated = counts['generated']
    require(generated == len(outputs) and generated <= max_generated,
            'DONE output count mismatch')
    require(counts['prompt'] == n_prompt and counts['reused'] == 0 and counts['read_n'] == n_prompt,
            'DONE fresh prompt/read/reuse identity mismatch')
    require(d[5] in ('length', 'stop'), 'unsupported finish reason')
    require(prompt_ms > 0 and prompt_ms + 0.55 >= previous_ms,
            'DONE prompt timing impossible versus rounded PP elapsed')
    require(counts['drafts_accepted'] <= counts['drafts_offered'],
            'impossible MTP accepted/offered counts')
    require(counts['cache_hits'] <= counts['cache_lookups'], 'cache hits exceed lookups')
    integer(counts['cache_lookups'] + counts['offloaded'], 'combined routed counters')
    integer(counts['ram_blobs'] + counts['file_blobs'], 'combined source-read counters')
    require(counts['file_blobs'] > 0 or file_mb == 0, 'file MB without file reads')
    # The final prompt token is consumed by the verifier: available verifier
    # positions are physical_context-batched_n, not an inferred chunk policy.
    capacity = physical_context - batched_n
    require(generated <= capacity, 'output exceeds physical verifier positions')
    if output_requirement == 'exhausted_zero':
        require(generated == 0 and d[5] == 'length', 'caller-declared zero-output contract mismatch')
        require(all(counts[name] == 0 for name in ('drafts_accepted', 'drafts_offered',
                'cache_hits', 'cache_lookups', 'ram_blobs', 'file_blobs', 'offloaded')),
                'decode counters with zero generated output')
    else:
        require(generated > 0 and decode_ms > 0, 'positive output and decode timing required')
        require(d[5] != 'length' or generated == min(max_generated, capacity),
                'length ended before caller or physical verifier bound')
    return {'scope': 'one_fresh_GEN_request_stdout', 'protocol_complete': True,
            'prompt_tokens': n_prompt, 'actual_batched_positions': batched_n,
            'physical_context': physical_context, 'chunk_limit': chunk_limit,
            'chunk_starts': starts, 'chunk_ends': ends, 'progress': progress,
            'generated_tokens': generated, 'output_ids': outputs, 'lp5': lp_rows,
            'finish_reason': d[5], 'counts': counts, 'prompt_ms': prompt_ms,
            'decode_ms': decode_ms, 'file_mb': file_mb,
            'output_requirement': output_requirement, 'lp5_required': require_lp5,
            'independent_gpu_completion': False, 'numerical_parity_validated': False,
            'full_model_quality_qualified': False, 'performance_eligible': False}


def validate_trace(lines, *, n_prompt, batched_n, chunk_limit, physical_context):
    """Bounded optional stderr observation; exact starts, unrelated lines allowed."""
    starts, _ = geometry(n_prompt, batched_n, chunk_limit, physical_context)
    positions = []
    for line in bounded_lines(lines):
        if _TRACE_FAMILY not in line:
            continue
        match = _TRACE.fullmatch(line)
        require(match is not None, 'malformed recognizable prompt-chunk trace')
        require(len(positions) < len(starts), 'extra trace chunk')
        start, total = uint(match[1], 'trace start'), uint(match[2], 'trace total')
        require(total == batched_n and start == starts[len(positions)],
                'trace start/order/identity mismatch')
        positions.append(start)
    require(positions == starts, 'missing trace chunks')
    return {'scope': 'host_prompt_chunk_trace', 'chunk_start_positions': positions,
            'batched_positions': batched_n, 'independent_gpu_completion': False}
