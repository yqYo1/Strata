"""Check lossless QSA captures against a prospective arithmetic contract.

Authored under repository LICENSE. No inference fallback or empirical quality
tolerance. The caller must qualify the target arithmetic and divide/conversion
contract independently; synthetic captures cannot establish model quality.
"""
from fractions import Fraction as F
import json
import re
import qsa_sum_interval as q

MARK = 0xdeadbeef


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite(bits):
    return (bits & 0x7f800000) != 0x7f800000


def bf16_rne(bits):
    need(type(bits) is int and 0 <= bits < 1 << 32, 'FP32 bits required')
    if not finite(bits) and bits & 0x7fffff:
        return (bits >> 16) | 0x40
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def ulp(value):
    """OpenCL absolute-error ULP at an exact finite real reference value."""
    value = abs(F(value))
    if value < q.MIN_NORMAL:
        return q.MIN_SUBNORMAL
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < q.power2(exponent):
        exponent -= 1
    return q.power2(exponent - 23)


def within(value, bounds, label):
    need(bounds[0] <= value <= bounds[1], label + ' outside derived interval')


def pair(control, candidate, *, arithmetic_mode, division_ulps, bf16_mode):
    """Validate one head in two arms; return only compact conservation counts."""
    need(arithmetic_mode == 'RNE_gradual' and bf16_mode == 'RNE',
         'independently qualified FP32/BF16 modes required')
    need(isinstance(division_ulps, F) and 0 <= division_ulps <= 4,
         'source-justified finite target division ULP bound required')
    meta = ('call', 'cap', 'width', 'tiles', 'queries', 'query', 'kv_head',
            'head', 'live_cells', 'page')
    need(control['arm'] == 'original' and candidate['arm'] == 'candidate', 'arm order')
    need(all(control[k] == candidate[k] for k in meta), 'paired workload differs')
    cap, width, tiles = (control[k] for k in ('cap', 'width', 'tiles'))
    need(0 < cap <= 2051 and control['queries'] in range(1, 4) and
         0 <= control['query'] < control['queries'] and control['page'] in (4, 16) and
         control['kv_head'] in (0, 1) and control['head'] in (*range(12), 15),
         'unplanned capture geometry')
    need(tiles == ((width + 31)//32 if 0 <= width <= cap else 0), 'live tile extent')
    capacity = (cap + 31)//32
    for record in (control, candidate):
        need(record['guards'] is True and record['normal_parity'] is True,
             'normal/capture correspondence or guards failed')
        need(len(record['state']) == tiles*4 and
             len(record['details']) == capacity*64 + 512 and
             len(record['output']) == (256 if record['head'] < 12 else 0),
             'lossless trace extent differs')
        need(all(type(v) is int and 0 <= v < 1 << 32
                 for v in record['state'] + record['details']), 'invalid raw word')
        need(all(type(v) is int and 0 <= v < 65536 for v in record['output']), 'invalid BF16')
        need(all(v == MARK for v in record['details'][tiles*64:capacity*64]),
             'unexecuted probability tile changed')
    intervals = [(F(0), F(0)), (F(0), F(0))]
    previous = [F(0), F(0)]
    changed_sums = changed_norms = 0
    for tile in range(tiles):
        states = [record['state'][tile*4:tile*4+4] for record in (control, candidate)]
        details = [record['details'][tile*64:tile*64+64] for record in (control, candidate)]
        need(states[0][:2] == states[1][:2], 'maximum/alpha changed')
        need(details[0] == details[1], 'FP32 p or actual BF16 PV operands changed')
        p = details[0][:32]
        need(all(v < 0x80000000 and 0 <= q.exact_fp32(v) <= 1 for v in p),
             'invalid probability domain')
        need(all(v == 0 for i,v in enumerate(p) if tile*32+i >= width),
             'probability past selected width')
        need([bf16_rne(v) for v in p] == details[0][32:], 'BF16 PV cast differs from RNE')
        alpha = q.exact_fp32(states[0][1])
        need(0 <= alpha <= 1, 'alpha outside online contract')
        need((previous[0] > 0) == (previous[1] > 0), 'normalizer branch diverged')
        if not previous[0]:
            need(states[0][1] == 0, 'first/non-live tile alpha must be positive zero')
        serial = F(0)
        for bits in p:
            serial = q.rn32(serial + q.exact_fp32(bits))
        need(q.exact_fp32(states[0][3]) == serial, 'original ascending FP32 fold differs')
        for arm, depth in enumerate((31, 16)):
            _, contribution = q.tile_interval(p, depth=depth, arithmetic_mode=arithmetic_mode)
            norm, total = map(q.exact_fp32, states[arm][2:])
            within(total, contribution, 'tile sum')
            need(norm == q.rn32(previous[arm]*alpha + total), 'online FMA differs from RNE')
            intervals[arm] = q.normalizer_step(intervals[arm], states[arm][1], contribution,
                                              arithmetic_mode=arithmetic_mode)
            within(norm, intervals[arm], 'online normalizer')
            need(norm >= 0 and total >= 0, 'negative normalizer/sum')
            if not norm:
                need(states[arm][2] == 0, 'normalizer zero sign')
            previous[arm] = norm
        changed_sums += states[0][3] != states[1][3]
        changed_norms += states[0][2] != states[1][2]
    base = capacity*64
    accumulators = [record['details'][base:base+256] for record in (control, candidate)]
    need(accumulators[0] == accumulators[1], 'FP32 PV accumulator changed')
    normalized = [record['details'][base+256:] for record in (control, candidate)]
    result = dict(pairs=1, tiles=tiles, changed_sums=changed_sums,
                  changed_norms=changed_norms, checked_ratios=0, changed_output_bits=0,
                  zero_heads=0, diagnostic_heads=0, padded_heads=0)
    if control['head'] >= 12:
        need(all(v == MARK for values in normalized for v in values), 'padded head normalized')
        result['padded_heads'] = 1
        return result
    for arm, record in enumerate((control, candidate)):
        if all(v == MARK for v in normalized[arm]):
            # The source writes a complete canonical diagnostic or positive-zero
            # row without executing division. Do not interpret sentinel as FP32.
            if all(v == 0x7fc0 for v in record['output']):
                result['diagnostic_heads'] += 1
            else:
                need(not previous[arm] and all(v == 0 for v in record['output']),
                     'unclassified no-division head')
                result['zero_heads'] += 1
            continue
        need(previous[arm] > 0, 'division with nonpositive normalizer')
        for a_bits, y_bits, output in zip(accumulators[arm], normalized[arm], record['output']):
            need(finite(a_bits) and finite(y_bits), 'nonfinite finite-division path')
            numerator, actual = q.exact_fp32(a_bits), q.exact_fp32(y_bits)
            reference = numerator/previous[arm]
            need(abs(actual-reference) <= division_ulps*ulp(reference),
                 'target division exceeds qualified ULP accuracy')
            if not numerator or not actual:
                need(a_bits >> 31 == y_bits >> 31, 'division zero sign differs')
            lower, upper = q.exact_ratio_interval(a_bits, intervals[arm])
            radius = division_ulps*max(ulp(lower), ulp(upper))
            within(actual, (lower-radius, upper+radius), 'normalized output')
            need(output == bf16_rne(y_bits), 'output BF16 cast differs from RNE')
            result['checked_ratios'] += 1
    need([v == MARK for v in normalized[0]] == [v == MARK for v in normalized[1]],
         'normalization/diagnostic branch changed')
    result['changed_output_bits'] = sum(a != b for a,b in zip(control['output'], candidate['output']))
    return result


def check(stdout, *, mode, page, arithmetic_mode, division_ulps, bf16_mode):
    """Parse complete owned fixture output; require exact paired coverage."""
    need(mode in ('tiny', 'component') and page in (4, 16), 'planned mode/page')
    records = [json.loads(line[len('QSA_NORM_TRACE '):]) for line in stdout.splitlines()
               if line.startswith('QSA_NORM_TRACE ')]
    end = re.findall(r'^QSA_NORM_DONE calls=(\d+) records=(\d+) tensor_oracle=PASS model_quality=UNRUN$', stdout, re.M)
    need(len(end) == 1 and len(records) == int(end[0][1]) and records, 'incomplete trace')
    need(stdout.count('QSA_NORM_ADMISSION rejections=7 output_state_detail_unchanged=1') == 1,
         'trace admission coverage')
    calls, pending, totals = {}, {}, {}
    for record in records:
        need(record['page'] == page, 'page differs')
        key = tuple(record[k] for k in ('call', 'query', 'kv_head', 'head'))
        arm = record['arm']
        need(arm in ('original', 'candidate') and arm not in pending.setdefault(key, {}),
             'unknown/duplicate arm')
        pending[key][arm] = record
        calls.setdefault(record['call'], set()).add(key[1:])
    need(set(calls) == set(range(int(end[0][0]))), 'call coverage')
    for key, arms in pending.items():
        need(set(arms) == {'original', 'candidate'}, 'missing paired arm')
        result = pair(arms['original'], arms['candidate'], arithmetic_mode=arithmetic_mode,
                      division_ulps=division_ulps, bf16_mode=bf16_mode)
        for name, value in result.items():
            totals[name] = totals.get(name, 0) + value
    for call, keys in calls.items():
        sample = next(record for record in records if record['call'] == call)
        queries = [0] if mode == 'tiny' or sample['queries'] == 1 else [0, sample['queries']-1]
        heads = [0, 15] if mode == 'tiny' else [*range(12), 15]
        need(keys == {(query, kv, head) for query in queries for kv in (0,1) for head in heads},
             'head/query/KV coverage')
    need(stdout.count('case cells=') == (1 if mode == 'tiny' else 11), 'independent tensor-case coverage')
    need(len(calls) == (2 if mode == 'tiny' else 30), 'fixture invocation coverage')
    return dict(totals, calls=len(calls), records=len(records),
                arithmetic_mode=arithmetic_mode, target_division_ulps=str(division_ulps),
                bf16_mode=bf16_mode, model_quality_qualified=False)
