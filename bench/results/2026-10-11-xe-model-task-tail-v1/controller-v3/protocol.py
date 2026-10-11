"""Strict diagnostic GEN16 for two 4096-token chunks; no speed eligibility."""
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(lines):
    n, chunk, maximum = 8192, 4096, 16
    require(all(not line.startswith('ERR') for line in lines), 'no protocol ERR')
    done = [line.split() for line in lines if line.startswith('DONE ')]
    require(len(done) == 1 and len(done[0]) == 16, 'one complete DONE')
    d = done[0]
    generated = int(d[1])
    require(0 < generated <= maximum, 'bounded output count')
    require(d[5] in ('length', 'stop'), 'known finish reason')
    require(d[5] != 'length' or generated == maximum, 'honest length')
    require(int(d[2]) == n and int(d[8]) == 0 and int(d[14]) == n,
            'fresh complete prompt/read_n')
    require(0 <= int(d[6]) <= int(d[7]), 'ordered MTP counts')
    require(all(math.isfinite(float(d[i])) and float(d[i]) > 0 for i in (3, 4)),
            'positive finite phase times')
    for tag in ('RESUME', 'REUSED'):
        require([line for line in lines if line.startswith(tag + ' ')] == [tag + ' 0'],
                'fresh ' + tag)
    pp = [line.split() for line in lines if line.startswith('PP ')]
    require([int(p[1]) for p in pp] == [chunk, n - 1], 'exact two chunks')
    require(all(len(p) == 5 and int(p[2]) == n and
                all(math.isfinite(float(x)) and float(x) >= 0 for x in p[3:])
                for p in pp), 'complete finite PP')
    outputs = [line for line in lines if line.startswith(('T ', 'LP '))]
    require(len(outputs) == generated * 2, 'complete output/LP count')
    ids = []
    for i in range(generated):
        token = outputs[2 * i].split()
        lp = outputs[2 * i + 1].split()
        require(len(token) == 2 and token[0] == 'T' and
                0 <= int(token[1]) < 248320, 'token pairing/domain')
        require(len(lp) == 7 and lp[0] == 'LP' and math.isfinite(float(lp[1])),
                'finite paired LP5')
        for value in lp[2:]:
            token_id, probability = value.split(':')
            require(0 <= int(token_id) < 248320 and math.isfinite(float(probability)),
                    'finite top-vocabulary LP5')
        ids.append(int(token[1]))
    return dict(prompt_tokens=n, actual_batched_positions=n - 1,
                generated_tokens=generated, prefill_chunk=chunk,
                mtp_counts=list(map(int, d[6:8])), finish_reason=d[5],
                output_ids=ids, prompt_ms=float(d[3]), decode_ms=float(d[4]),
                diagnostic_only=True, performance_eligible=False,
                full_model_quality_qualified=False)


def validate_trace(text):
    positions = [int(x) for x in __import__('re').findall(
        r'strata trace: prompt chunk (\d+) of 8191', text)]
    require(positions == [0, 4096], 'both exact trace chunk starts')
    return dict(chunk_start_positions=positions)


def selftest():
    lines = ['RESUME 0', 'PP 4096 8192 25000 100', 'PP 8191 8192 50000 100',
             'REUSED 0']
    for _ in range(16):
        lines += ['T 123', 'LP -0.1 123:-0.1 124:-1 125:-2 126:-3 127:-4']
    lines += ['DONE 16 8192 50000 2000 length 8 12 0 774 46560 0 0 0.0 8192 0']
    assert validate(lines)['generated_tokens'] == 16
    negatives = []
    for old, new in [('PP 4096 8192 25000 100', 'PP 2048 8192 25000 100'),
                     ('REUSED 0', 'REUSED 1'),
                     (lines[-1], lines[-1].replace('8192 0', '8191 0')),
                     (lines[-1], lines[-1].replace('50000', 'nan')),
                     ('T 123', 'T 248320'),
                     ('LP -0.1 123:-0.1 124:-1 125:-2 126:-3 127:-4',
                      'LP nan 123:-0.1 124:-1 125:-2 126:-3 127:-4')]:
        altered = lines.copy()
        altered[altered.index(old)] = new
        negatives.append(altered)
    negatives += [lines[:-1], lines + [lines[-1]], lines + ['T 5'],
                  [line for line in lines if not line.startswith('PP 4096')],
                  lines + ['ERR failed']]
    for bad in negatives:
        try:
            validate(bad)
        except (ValueError, IndexError):
            pass
        else:
            raise AssertionError('Malformed protocol accepted')
    trace = 'strata trace: prompt chunk 0 of 8191\nstrata trace: prompt chunk 4096 of 8191\n'
    validate_trace(trace)
    for bad in [trace.splitlines()[0], trace + trace, trace.replace('4096', '2048')]:
        try:
            validate_trace(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed trace accepted')
    return dict(positive_protocol=1, negative_protocol=len(negatives),
                positive_trace=1, negative_trace=3)


if __name__ == '__main__':
    print(selftest())
