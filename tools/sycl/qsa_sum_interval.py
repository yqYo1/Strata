"""Exact prospective FP32 normalizer bounds; authored under repository LICENSE.

Host validation only, never an inference fallback. No guessed reduction tree or
empirical tensor/task tolerance. Callers must independently qualify RNE/gradual
underflow, common operands, lane mapping and unchanged PV arithmetic.
"""
from fractions import Fraction as F


def power2(exponent):
    return F(1 << exponent) if exponent >= 0 else F(1, 1 << -exponent)


MIN_SUBNORMAL = power2(-149)
MIN_NORMAL = power2(-126)
MAX_FINITE = F((1 << 24) - 1) * power2(104)


def exact_fp32(bits):
    if type(bits) is not int or not 0 <= bits < 1 << 32:
        raise ValueError('binary32 bits required')
    exponent, fraction = (bits >> 23) & 255, bits & ((1 << 23) - 1)
    if exponent == 255:
        raise ValueError('nonfinite input needs the separate behavior gate')
    value = (F(fraction) * MIN_SUBNORMAL if not exponent else
             F((1 << 23) + fraction) * power2(exponent - 150))
    return -value if bits >> 31 else value


def fp32_bounds(value):
    """Directed finite binary32 endpoints of an exact rational, including FMA underflow."""
    value = F(value)
    if value < 0:
        lower, upper = fp32_bounds(-value)
        return -upper, -lower
    if not value:
        return F(0), F(0)
    if value > MAX_FINITE:
        raise ValueError('overflow outside the finite component domain')
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < power2(exponent):
        exponent -= 1
    quantum = power2(max(exponent - 23, -149))
    lower = (value // quantum) * quantum
    return lower, lower if lower == value else lower + quantum


def rn32(value):
    """Exact binary32 RNE value, without a host binary64 multiply/add round trip."""
    value = F(value)
    if value < 0:
        return -rn32(-value)
    lower, upper = fp32_bounds(value)
    if lower == upper:
        return lower
    distance = 2 * (value - lower) - (upper - lower)
    if distance < 0:
        return lower
    if distance > 0:
        return upper
    return lower if (lower / (upper - lower)).numerator % 2 == 0 else upper


def gamma(depth):
    if type(depth) is not int or not 0 < depth < 1 << 24:
        raise ValueError('proved positive addition-path depth required')
    return F(depth, (1 << 24) - depth)


def tile_interval(probability_bits, *, depth, arithmetic_mode):
    if arithmetic_mode != 'RNE_gradual':
        raise ValueError('qualify arithmetic mode; FTZ/DAZ need a different oracle')
    if len(probability_bits) != 32:
        raise ValueError('exactly 32 unique probability operands required')
    terms = [exact_fp32(v) for v in probability_bits]
    if any(v < 0 for v in terms):
        raise ValueError('positive-sum bound excludes negative operands')
    total = sum(terms, F(0))
    # Every positive partial sum is on the subnormal lattice in this case.
    if total < MIN_NORMAL:
        return total, (total, total)
    error = gamma(depth) * total
    return total, (fp32_bounds(total - error)[0], fp32_bounds(total + error)[1])


def normalizer_step(prior, alpha_bits, contribution, *, arithmetic_mode):
    if arithmetic_mode != 'RNE_gradual':
        raise ValueError('FMA FTZ/DAZ require separate mode-aware intervals')
    lower, upper = prior
    q_lower, q_upper = contribution
    alpha = exact_fp32(alpha_bits)
    if not (0 <= lower <= upper and 0 <= q_lower <= q_upper and 0 <= alpha <= 1):
        raise ValueError('nonnegative normalizer and common alpha in [0,1] required')
    # Exact fused expression; directed endpoint rounding encloses FP32 RNE FMA.
    return (fp32_bounds(lower * alpha + q_lower)[0],
            fp32_bounds(upper * alpha + q_upper)[1])


def exact_ratio_interval(accumulator_bits, normalizer):
    """Real ratio only: target F32 division and BF16 rounding still need qualification."""
    lower, upper = normalizer
    if not 0 < lower <= upper:
        raise ValueError('zero denominator/branch classification must be gated separately')
    numerator = exact_fp32(accumulator_bits)
    endpoints = (numerator / lower, numerator / upper)
    return min(endpoints), max(endpoints)
