"""SOURCE ONLY phase N64/N32768 fixed8192 protocol; original protocol/window helpers remain immutable."""
import math
import re


def require(value, message):
    if not value:
        raise ValueError(message)


def windows(n, chunk):
    require(type(n) is int and n in (64, 32768) and type(chunk) is int and chunk == 8192,
            'explicit two-case phase window')
    return [(offset, min(chunk, n - 1 - offset)) for offset in range(0, n - 1, chunk)]


def validate(lines, *, n, chunk):
    schedule = windows(n, chunk)
    require(all(isinstance(line, str) and not line.startswith('ERR') for line in lines), 'no protocol ERR')
    done = [line.split() for line in lines if line.startswith('DONE ')]
    require(len(done) == 1 and len(done[0]) == 16, 'one complete DONE')
    d = done[0]
    generated = int(d[1])
    require(0 < generated <= 16 and d[5] in ('length', 'stop') and
            (d[5] != 'length' or generated == 16), 'honest bounded finish')
    require(int(d[2]) == n and int(d[8]) == 0 and int(d[14]) == n, 'fresh complete prompt/read_n')
    require(0 <= int(d[6]) <= int(d[7]), 'ordered MTP counts')
    require(all(math.isfinite(float(d[i])) and float(d[i]) > 0 for i in (3, 4)), 'finite positive times')
    for tag in ('RESUME', 'REUSED'):
        require([line for line in lines if line.startswith(tag + ' ')] == [tag + ' 0'], 'fresh ' + tag)
    pp = [line.split() for line in lines if line.startswith('PP ')]
    require([int(p[1]) for p in pp] == [offset + size for offset, size in schedule], 'exact n-1 window ends')
    require(all(len(p) == 5 and int(p[2]) == n and
                all(math.isfinite(float(x)) and float(x) >= 0 for x in p[3:]) for p in pp), 'finite complete PP')
    outputs = [line for line in lines if line.startswith(('T ', 'LP '))]
    require(len(outputs) == 2 * generated, 'complete output/LP count')
    ids = []
    for index in range(generated):
        token, lp = outputs[2 * index].split(), outputs[2 * index + 1].split()
        require(len(token) == 2 and token[0] == 'T' and 0 <= int(token[1]) < 248320, 'paired token domain')
        require(len(lp) == 7 and lp[0] == 'LP' and math.isfinite(float(lp[1])), 'finite paired LP5')
        for value in lp[2:]:
            token_id, probability = value.split(':')
            require(0 <= int(token_id) < 248320 and math.isfinite(float(probability)), 'finite top5 domain')
        ids.append(int(token[1]))
    return dict(prompt_tokens=n, actual_batched_positions=n - 1, generated_tokens=generated,
                prefill_chunk=chunk, mtp_counts=list(map(int, d[6:8])), finish_reason=d[5], output_ids=ids,
                prompt_ms=float(d[3]), decode_ms=float(d[4]), diagnostic_only=True,
                performance_eligible=False, full_model_quality_qualified=False)


def validate_trace(text, *, n, chunk):
    offsets = [offset for offset, size in windows(n, chunk)]
    pairs = re.findall(r'strata trace: prompt chunk (\d+) of (\d+)', text)
    require([(int(a), int(b)) for a, b in pairs] == [(offset, n - 1) for offset in offsets], 'exact chunk starts')
    return dict(chunk_start_positions=offsets)


def output_deltas(reference, current):
    """Record observations before enforcing strict continuation; no numerical allowance invented."""
    def output(value):
        selected = [x for x in value['request_protocol'] if x.startswith(('T ', 'LP '))]
        return dict(selected=selected, ids=[int(x.split()[1]) for x in selected if x.startswith('T ')],
                    lp=[x for x in selected if x.startswith('LP ')])
    before, after = output(reference), output(current)
    differences = []
    for index in range(max(len(before['lp']), len(after['lp']))):
        left = before['lp'][index] if index < len(before['lp']) else None
        right = after['lp'][index] if index < len(after['lp']) else None
        if left != right:
            delta = None
            same_token = index < len(before['ids']) and index < len(after['ids']) and before['ids'][index] == after['ids'][index]
            if left is not None and right is not None and same_token:
                delta = float(right.split()[1]) - float(left.split()[1])
            differences.append(dict(output_index=index, reference_lp5=left, current_lp5=right,
                                    selected_token_ids_match=same_token, selected_logprob_delta=delta))
    return dict(reference_output_ids=before['ids'], current_output_ids=after['ids'],
                output_ids_equal=before['ids'] == after['ids'], selected_logprob_top5_equal=before['selected'] == after['selected'],
                reference_generated_tokens=reference['validation']['generated_tokens'],
                current_generated_tokens=current['validation']['generated_tokens'], lp5_differences=differences,
                strict_continuation_passed=(reference['validation']['generated_tokens'] == 16 and
                    current['validation']['generated_tokens'] == 16 and before['selected'] == after['selected']),
                scope='observed exact16 diagnostic continuation only; chunk shapes can cause numerical deltas; no quality threshold')
