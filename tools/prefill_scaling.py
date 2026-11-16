#!/usr/bin/env python3
"""Describe fresh-prefill length scaling without mixing routes or warmup scopes.

Main-authored, Python standard library only; repository LICENSE.
This offline analysis never launches a workload or qualifies a model route.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


import prefill_scaling_bench as shared_fit


SCOPES = ('first-use', 'subsequent-request')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def finite_positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            'positive finite ' + name + ' required')


def metric_analysis(points):
    # Reuse the official bench extension's existing hypotheses/whole-length
    # holdouts. Only cohort grouping and process weighting are new here.
    rows = [dict(actual_read_tokens=n, prompt_ms=1000*y) for n, y in points]
    compared = shared_fit.approximation_models(rows)
    names = dict(affine='constant+linear', quadratic='constant+linear+quadratic',
                 n_log_n='constant+linear+nlog2n')
    candidates = []
    if compared is not None:
        for name, model in compared['candidates'].items():
            errors = [x/1000 for x in model['leave_one_length_out_error_ms']]
            residuals = [x/1000 for x in model['residual_ms']]
            held = [dict(tokens=n, observed_s=y, predicted_s=y+error, residual_s=-error)
                    for (n, y), error in zip(points, errors)]
            candidates.append(dict(name=names[name], model=model, eligible=True,
                residuals_s=residuals, held_out=held,
                fit_RMSE_s=math.sqrt(statistics.mean(x*x for x in residuals)),
                held_out_RMSE_s=math.sqrt(statistics.mean(x*x for x in errors)),
                held_out_predictions_positive=all(x['predicted_s'] > 0 for x in held),
                physical_fixed_cost_established=False))
    return dict(candidates=candidates, models_eligible=bool(candidates),
        minimum_distinct_lengths=5,
        adjacent_slopes_ms_per_token=[dict(from_tokens=a[0], to_tokens=b[0],
            slope=1000*(b[1]-a[1])/(b[0]-a[0])) for a, b in zip(points, points[1:])],
        physical_law_established=False)


def analyze(samples):
    """Use one median per process/length, then equal-weight length medians.

    Input records explicitly name source, configuration, route, first/warmed
    scope, process, fresh token count, prefill seconds and client TTFT seconds.
    Cached-context curves need a separately defined multidimensional study.
    """
    require(isinstance(samples, list) and bool(samples), 'nonempty sample list required')
    groups = defaultdict(list)
    for row in samples:
        for name in ('source_commit', 'configuration_id', 'process'):
            require(isinstance(row.get(name), str) and bool(row[name]), name + ' required')
        require(row.get('scope') in SCOPES, 'explicit first/warmed scope required')
        require(len(row['source_commit']) == 40
                and all(c in '0123456789abcdef' for c in row['source_commit']),
                'full native source commit required')
        require(isinstance(row.get('route'), (str, dict)) and bool(row['route']),
                'actual route descriptor required')
        require(type(row.get('input_tokens')) is int and row['input_tokens'] > 0,
                'positive actual token count required')
        require(type(row.get('reused_tokens')) is int and row['reused_tokens'] == 0
                and type(row.get('context_before')) is int and row['context_before'] == 0,
                'fresh zero-start-context samples required')
        for metric in ('prefill_s', 'ttft_s'):
            finite_positive(row.get(metric), metric)
        identity = {k: row[k] for k in ('source_commit', 'configuration_id', 'route', 'scope')}
        groups[canonical(identity)].append(row)
    output = []
    for identity, rows in sorted(groups.items()):
        by_length = defaultdict(lambda: defaultdict(list))
        for row in rows:
            by_length[row['input_tokens']][row['process']].append(row)
        points = []
        for n, processes in sorted(by_length.items()):
            point = dict(input_tokens=n, processes=len(processes), samples=sum(map(len, processes.values())))
            for metric in ('prefill_s', 'ttft_s'):
                values = [statistics.median(r[metric] for r in group) for _, group in sorted(processes.items())]
                point[metric] = dict(process_medians=values, median=statistics.median(values),
                    minimum=min(values), maximum=max(values))
            point['pp_from_median'] = n/point['prefill_s']['median']
            points.append(point)
        output.append(dict(identity=json.loads(identity), points=points,
            analysis={metric: metric_analysis([(p['input_tokens'], p[metric]['median']) for p in points])
                      for metric in ('prefill_s', 'ttft_s')}))
    return dict(groups=output, timing_unit='seconds',
        aggregation='median within each process/length,then median across processes;equal weight per length',
        held_out_unit='one entire token length,including every repetition at that length',
        inference_adopted=False, physical_law_established=False)


def coherent_samples(record):
    """Normalize the maintained bench.sh coherent-prefill record, not a CSV peak."""
    require(record.get('stage') == 'closed' and record.get('passed') is True
            and record.get('complete') is True and record.get('ownership_closed') is True
            and record.get('active') is False, 'closed coherent speed record required')
    args = record['configuration']['args']
    require(all(args.count(flag) == 1 and args[args.index(flag)+1] == '0'
                for flag in ('--conversation-cache-mib', '--prompt-cache')),
            'original coherent fresh-request cache contract required')
    configuration = canonical({k: record[k] for k in
        ('binary', 'environment_template_sha256', 'model_assets_sha256')})
    configuration_id = hashlib.sha256(configuration.encode()).hexdigest()
    samples = []
    for process in record['processes']:
        require(process['passed'] is True and process['clean_speed'] is True
                and process['native_exit_code'] == 0 and process['ownership_closed'] is True
                and process['native_source_commit'] == record['native_source_commit']
                and process['binary'] == record['binary'] and not process['fresh_faults'],
                'clean,closed same-source/same-executable process required')
        require(len(process['routes']) == len(process['samples']), 'one actual route per request required')
        for i, (sample, route) in enumerate(zip(process['samples'], process['routes'])):
            require(sample['actual_read_tokens'] == process['read'], 'actual token count differs')
            finite_positive(sample['prompt_ms'], 'recorded prefill milliseconds')
            window = route['fast_window']
            descriptor = {k: route[k] for k in ('actual_chunk', 'residual_dtype',
                'configured_context', 'configured_resident', 'actual_typed_QSA')}
            descriptor.update(residual_fully_on_gpu=route['residual_gpu_rows'] == route['rows'],
                staged=window['staged'], split=window['split'], prefill_slots=window['prefill_slots'])
            samples.append(dict(source_commit=record['native_source_commit'],
                configuration_id=configuration_id, route=descriptor,
                scope='first-use' if i == 0 else 'subsequent-request', process=str(process['index']),
                input_tokens=sample['actual_read_tokens'], reused_tokens=sample['actual_reused_tokens'],
                context_before=0, prefill_s=sample['prompt_ms']/1000, ttft_s=sample['ttft_s']))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--coherent-key', help='top-level coherent speed record;otherwise input is a sample list')
    args = parser.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    result = analyze(coherent_samples(data[args.coherent_key]) if args.coherent_key else data)
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
