"""Length characterization for the official bench.sh client; no native launcher."""
import math
import statistics

MINIMA = (1024, 4096, 16384, 32768, 65536, 98304)


def order(round_index):
    """Freeze order before measurements; do not optimize it from observed times."""
    if round_index % 2:
        return tuple(reversed(MINIMA))
    offset = (round_index // 2 * 2) % len(MINIMA)
    return MINIMA[offset:] + MINIMA[:offset]


def linear_fit(samples):
    """Descriptive OLS on actual counts; this alone does not identify fixed cost."""
    points = [(r.get('actual_read_tokens'), r.get('prompt_ms')) for r in samples]
    if len(points) < 3 or any(type(n) is not int or n <= 0 or
            type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for n, t in points):
        return None
    if len({n for n, _ in points}) < 3:
        return None
    xn, yt = statistics.mean(n for n, _ in points), statistics.mean(t for _, t in points)
    slope = sum((n-xn)*(t-yt) for n, t in points) / sum((n-xn)**2 for n, _ in points)
    intercept = yt - slope*xn
    residuals = [t - (intercept+slope*n) for n, t in points]
    total = sum((t-yt)**2 for _, t in points)
    return dict(intercept_ms=intercept, ms_per_token=slope,
        r_squared=1-sum(e*e for e in residuals)/total if total else None,
        max_relative_residual=max(abs(e)/t for e, (_, t) in zip(residuals, points)),
        points=[dict(actual_read_tokens=n, prompt_ms=t, residual_ms=e)
                for (n, t), e in zip(points, residuals)],
        interpretation='Descriptive only. Check route, chunk boundaries and residuals before interpreting the intercept as fixed cost; no linearity or saturation qualification.')


def approximation_models(samples):
    """Compare small hypotheses with entire input lengths left out, not runs."""
    points = [(r['actual_read_tokens']/65536, r['prompt_ms']) for r in samples]
    if len(points) < 5:
        return None

    def features(x, name):
        return [1.0, x] + ([] if name == 'affine' else
                           [x*x] if name == 'quadratic' else [x*math.log2(x)])

    def fit(rows, name):
        fs = [features(x, name) for x, _ in rows]
        width = len(fs[0])
        matrix = [[sum(f[i]*f[j] for f in fs) for j in range(width)] +
                  [sum(f[i]*t for f, (_, t) in zip(fs, rows))] for i in range(width)]
        for col in range(width):
            pivot = max(range(col, width), key=lambda r: abs(matrix[r][col]))
            if abs(matrix[pivot][col]) < 1e-12:
                return None
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            divisor = matrix[col][col]
            matrix[col] = [v/divisor for v in matrix[col]]
            for r in range(width):
                if r != col:
                    factor = matrix[r][col]
                    matrix[r] = [a-factor*b for a, b in zip(matrix[r], matrix[col])]
        return [row[-1] for row in matrix]

    result = {}
    for name, formula in [('affine', 'a+b*x'), ('quadratic', 'a+b*x+c*x^2'),
                          ('n_log_n', 'a+b*x+c*x*log2(x)')]:
        coefficients = fit(points, name)
        held_out = [fit(points[:i]+points[i+1:], name) for i in range(len(points))]
        if coefficients is None or any(c is None for c in held_out):
            continue
        predict = lambda c, x: sum(a*b for a, b in zip(c, features(x, name)))
        residuals = [t-predict(coefficients, x) for x, t in points]
        errors = [predict(c, x)-t for c, (x, t) in zip(held_out, points)]
        result[name] = dict(formula_ms=formula, x='actual_read_tokens/65536',
            coefficients_ms=dict(zip(('a','b','c'), coefficients)), residual_ms=residuals,
            leave_one_length_out_error_ms=errors,
            mean_held_out_relative_error=statistics.mean(abs(e)/t for e, (_, t) in zip(errors, points)),
            max_held_out_relative_error=max(abs(e)/t for e, (_, t) in zip(errors, points)))
    return dict(candidates=result, selected_model=None,
        interpretation='Hypotheses, not a required law. Compare held-out error against measured spread; inspect route/chunk changes and prefer a simpler adequate model. Piecewise route-specific fits may be required. Do not extrapolate outside measured lengths.')


def plan(suite):
    return dict(mode='prefill-scaling', requested_min_tokens=list(MINIMA),
        construction_target_tokens=[n*5//4 for n in MINIMA], construction_version=1,
        max_tokens=24, thinking_requested=True, workload=suite.workload_identity(),
        order=[list(order(r)) for r in range(suite.runs)],
        warmup='One short chat, then one fresh maximum-length request retained separately; measured rounds follow in the same server process.',
        freshness='Distinct length/round tags; every reported reuse count must be zero.',
        calibration='No adaptive body repair. Fit actual read counts and engine milliseconds, not construction targets.',
        limits='Short lengths diagnose fixed costs and do not qualify the >=32K prefill target. Same-process repeats are not independent-server repeats. Output cap24 does not qualify tg. Global fit may mix execution routes.')


def run(suite, error_type, stat):
    spec = plan(suite)
    cohort = dict(spec, requests=[], first_long_request=None, complete=False)
    suite.out['prefill_scaling'] = cohort
    required = max(spec['construction_target_tokens']) + 24 + 256
    if type(suite.ctx) is not int or suite.ctx < required:
        reason = f'prefill-scaling needs known context >= {required}, got {suite.ctx!r}'
        suite.out['terminal_failure'] = dict(phase='prefill-scaling/setup', error=reason)
        suite.fail('prefill-scaling/setup', dict(err=reason))
        raise error_type(reason)
    cases = {}
    for minimum in MINIMA:
        case = dict(requested_min_tokens=minimum, requests=[])
        cases[minimum] = case
        suite.out['prompts'][f'{minimum}-minimum'] = case

    def request(minimum, round_index):
        phase = f'prefill-scaling/{minimum}'
        before = len(suite.out.get('request_samples', []))
        try:
            suite.request(phase, round_index, None,
                suite.prompt_for(minimum*5//4, round_index, fixed=True), 24, True)
        finally:
            for event in suite.out.get('request_samples', [])[before:]:
                if event['phase'] != phase:
                    continue
                sample = dict(event['sample'], run=round_index, requested_min_tokens=minimum)
                if round_index == -1:
                    cohort['first_long_request'] = sample
                else:
                    cohort['requests'].append(sample)
                    cases[minimum]['requests'].append(sample)

    try:
        suite.request('warm-up/chat', None, None,
                      [dict(role='user', content='Explain what a token is in one sentence.')], 24, True)
        request(max(MINIMA), -1)
        for round_index in range(suite.runs):
            for minimum in order(round_index):
                request(minimum, round_index)
    finally:
        medians = []
        for minimum, case in cases.items():
            good = [r for r in case['requests'] if r.get('eligible') and r.get('ok')]
            case.update(eligible_runs=len(good), complete=len(good) == suite.runs)
            for out_key, sample_key in [('prompt_tps', 'prompt_tps'), ('prompt_ms', 'prompt_ms'),
                    ('ttft_s', 'ttft_s'), ('prompt_tokens', 'actual_prompt_tokens'),
                    ('read_tokens', 'actual_read_tokens'), ('reused_tokens', 'actual_reused_tokens')]:
                case[out_key] = stat(r[sample_key] for r in good)
            if good:
                # Preserve a real request's (n,T) pair rather than manufacturing
                # one by independently taking the median of n and T.
                median = sorted(good, key=lambda r: r['prompt_ms'])[len(good)//2]
                medians.append(median)
            case.pop('requests')  # Individual samples live once in cohort.requests.
        complete = all(c['complete'] for c in cases.values())
        cohort.update(complete=complete,
            eligible_runs=sum(c['eligible_runs'] for c in cases.values()),
            linear_fit=linear_fit(medians) if complete else None,
            approximation_models=approximation_models(medians) if complete else None,
            adjacent_slopes=[dict(from_read=a['actual_read_tokens'], to_read=b['actual_read_tokens'],
                ms_per_token=(b['prompt_ms']-a['prompt_ms'])/(b['actual_read_tokens']-a['actual_read_tokens']))
                for a, b in zip(medians, medians[1:]) if b['actual_read_tokens'] > a['actual_read_tokens']])
    return suite.out
