"""Bind existing decode clocks/copy counters to actual T1 or T4 routes.

Repository LICENSE. Offline evidence only; no model execution or GPU calls.
The printed clocks are nested host wall scopes, not hardware service times.
"""
import math
import re


LABELS = ('average_T', 'tokens_per_window', 'total', 'verify', 'GPU_reach_wait',
          'per_layer_host', 'plan', 'activation_quant', 'jobs', 'CPU_pool_run',
          'stage', 'commit_emit', 'draft', 'CPU_experts', 'CPU_entries', 'VRAM_hits', 'PCIe')
COPY_KEYS = ('submitted_blobs', 'submitted_bytes', 'completed_blobs', 'completed_bytes',
             'host_view_groups', 'gpu_miss_groups', 'cpu_unique_misses', 'cpu_entries')
NUMBER = r'(\d+(?:\.\d+)?)'
TIMING = (r'^strata decode timing: (\d+) windows, avg T ' + NUMBER + r', ' + NUMBER +
          r' tokens/window, ' + NUMBER + r' ms/window = verify ' + NUMBER +
          r' \(GPU-reach wait ' + NUMBER + r' \+ per-layer host ' + NUMBER +
          r' \[plan ' + NUMBER + r' actq ' + NUMBER + r' jobs ' + NUMBER + r' CPU ' + NUMBER +
          r'\] \+ stage ' + NUMBER + r'\) \+ commit/emit ' + NUMBER + r' \+ draft ' + NUMBER +
          r'; per layer-window: CPU experts ' + NUMBER + r' \(' + NUMBER +
          r' entries\), VRAM hits ' + NUMBER + r', PCIe ' + NUMBER + r'$')
COPY = r'^strata serve: SYCL_EXPERT_COPY ' + r' '.join(k + r'=(\d+)' for k in COPY_KEYS) + r'$'
HEADER = (r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, '
          r'(\d+) generated[^\n]*, drafts accepted (\d+) of (\d+),[^\n]*$')


def require(value, message):
    if not value:
        raise ValueError(message)


def analyze(text, samples, routes, layers=48, top_k=10):
    """Require one clock and copy footer per independently checked request route."""
    headers = list(re.finditer(HEADER, text, re.M))
    require(len(headers) == len(samples) == len(routes) > 0, 'request count differs')
    require(text.count('strata decode timing:') == len(headers) == len(re.findall(TIMING, text, re.M)),
            'missing/extra/malformed timing footer')
    require(text.count('strata serve: SYCL_EXPERT_COPY') == len(headers) == len(re.findall(COPY, text, re.M)),
            'missing/extra/malformed copy footer')
    result, previous = [], 0
    for i, (header, sample, route) in enumerate(zip(headers, samples, routes)):
        require(tuple(map(int, header.groups())) == tuple(sample[k] for k in
                ('prompt_tokens', 'reused_tokens', 'read_tokens', 'generated_tokens',
                 'drafts_accepted', 'drafts_offered')), 'native/client request differs')
        segment = text[previous:header.start()]
        timings = re.findall(TIMING, segment, re.M)
        tail = text[header.end():headers[i + 1].start() if i + 1 < len(headers) else len(text)]
        copies = re.findall(COPY, tail, re.M)
        require(len(timings) == len(copies) == 1, 'footer escaped completed request')
        windows = int(timings[0][0])
        values = dict(zip(LABELS, map(float, timings[0][1:])))
        counters = dict(zip(COPY_KEYS, map(int, copies[0])))
        require(windows == route['target_windows'] > 0 and
                all(math.isfinite(v) and v >= 0 for v in values.values()), 'invalid clocks/windows')
        traced = [tuple(map(int, v)) for v in re.findall(r'^strata trace: window (\d+) (\d+)$', segment, re.M)]
        require(len(traced) == windows and sum(t for _, t in traced) == route['target_rows'] and
                traced[0] == (sample['read_tokens'] - 1, 1) and
                all(t in (1, 4) for _, t in traced) and
                all(1 <= b[0] - a[0] <= a[1] for a, b in zip(traced, traced[1:])) and
                sum(t - 1 for _, t in traced) == sample['drafts_offered'], 'actual T1/T4 route differs')
        require(all(counters[k] == 0 for k in COPY_KEYS[:6]) and values['PCIe'] == 0,
                'expert-copy route differs from admitted pcie0')
        jobs, entries, rows = counters['cpu_unique_misses'], counters['cpu_entries'], route['target_rows']
        maximum_t = max(t for _, t in traced)
        require(0 < jobs <= entries <= maximum_t * jobs and entries <= rows * layers * top_k,
                'CPU job/entry conservation differs')
        for key, expected in [('average_T', rows / windows),
                              ('tokens_per_window', sample['generated_tokens'] / windows),
                              ('CPU_experts', jobs / (windows * layers)),
                              ('CPU_entries', entries / (windows * layers)),
                              ('VRAM_hits', (rows * layers * top_k - entries) / (windows * layers))]:
            require(abs(values[key] - expected) <= .00501, 'rounded work does not bind: ' + key)
        error = abs(values['total'] * windows - sample['decode_ms'])
        rounding = .005 * windows + .05
        require(error <= rounding + 1e-8, 'rounded decode clock differs from DONE/API')
        residual = values['total'] - sum(values[k] for k in ('verify', 'commit_emit', 'draft'))
        nested = values['per_layer_host'] - sum(values[k] for k in ('plan', 'activation_quant', 'jobs', 'CPU_pool_run'))
        verify_residual = values['verify'] - sum(values[k] for k in ('GPU_reach_wait', 'per_layer_host', 'stage'))
        require(min(residual, nested, verify_residual) >= -.025, 'nested clocks exceed parent')
        emitted = sample['generated_tokens']
        require(type(emitted) is int and emitted > 0, 'invalid emitted count')
        result.append(dict(target_windows=windows, target_rows=rows, emitted_tokens=emitted,
            maximum_T=maximum_t, copy=counters, GPU_routed_entry_fraction=1 - entries / (rows * layers * top_k),
            cpu_unique_jobs_per_emitted=jobs / emitted, cpu_entries_per_emitted=entries / emitted,
            target_rows_per_emitted=rows / emitted, exact_decode_tps=1000 * emitted / sample['decode_ms'],
            ms_per_window=values,
            approximate_ms_per_emitted={k: values[k] * windows / emitted for k in
                ('total', 'verify', 'GPU_reach_wait', 'per_layer_host', 'CPU_pool_run', 'stage', 'commit_emit', 'draft')},
            outer_residual_ms_per_window=residual, host_nested_residual_ms_per_window=nested,
            verify_residual_ms_per_window=verify_residual,
            clock_binding=dict(absolute_total_difference_ms=error, rounding_bound_ms=rounding)))
        previous = header.end()
    return result


def closed_pilot():
    """Replay the closed pilot before extracting its planned clock/work evidence."""
    import json
    import mtp_net_bench as pilot
    pilot.short.gate.native.guards.no_model_owners([])
    row = json.loads((pilot.ROOT / 'record.json').read_text())
    require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
            row['ownership_closed'] and not row['active'] and
            [p['cap'] for p in row['processes']] == [0, 1], 'complete closed cohort required')
    pilot.current(row)
    source = pilot.gen.git('rev-parse', 'HEAD').decode().strip()
    report = dict(source_commit=source, passed=False, inference_rerun=False)
    try:
        processes = []
        for plan, outcome in zip(row['plans'], row['processes']):
            root = pilot.Path(plan['root'])
            for name, sha in outcome['evidence_sha256'].items():
                require(pilot.digest(root / name) == sha, 'closed raw evidence changed: ' + name)
            samples, events, journal = pilot.official(root, plan['cap'])
            require(samples == outcome['requests'] and journal == outcome['journal'], 'official replay differs')
            start = pilot.native_end(root, row, plan, outcome['native_identity']['sid'])
            text = (root / 'engine.stderr').read_text()
            all_samples = [e['sample'] for e in events if e['kind'] == 'result']
            routes = pilot.request_routes(text, all_samples, plan['cap'])
            require(routes == outcome['routes'], 'actual route replay differs')
            timing = analyze(text, all_samples, routes)
            capture = [json.loads(l) for l in (root / 'client-resources.jsonl').read_text().splitlines()]
            resources = [pilot.resources.summarize_capture(capture, start['native'], start['boot_id'], s['client_interval'])
                         for s in samples]
            require(resources == outcome['resources'], 'resource replay differs')
            processes.append(dict(cap=plan['cap'], requests_replayed=len(all_samples),
                long_requests=[dict(workload=s['workload'], **v) for s, v in zip(samples, timing[-2:])]))
        report.update(passed=True, processes=processes,
            limits='Post-collection descriptive replay of predeclared clocks/work/resource evidence. Nested host wall scopes include polling, submission and overlap. Zero expert-copy bytes cover only that native route; no physical PCIe/VRAM bandwidth, hardware saturation, clean speed, repeated gain or adoption qualification.')
    except Exception as error:
        report['error'] = repr(error)
    row['analysis'] = report
    pilot.write(pilot.ROOT / 'record.json', row)
    pilot.write(pilot.RESULT, pilot.compact(row))
    print(json.dumps(dict(analysis_passed=report['passed'], error=report.get('error'))), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    import sys
    require(sys.argv[1:] == ['closed-pilot'], 'closed-pilot')
    sys.exit(closed_pilot())
