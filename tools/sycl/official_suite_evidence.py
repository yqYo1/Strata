"""Read the current official agent and thinking suite and report work matching."""

import json
import math
import statistics

import compare_official_prefill as comparison
import generation_engine as gen
import qualify_qsa_k_model as official

require = official.require
WORKLOAD_VERSION = 5
SUSTAINED_WORKLOADS = ('regular-code', 'varied-prose')
WORK_FIELDS = ('prompt_tokens', 'read_tokens', 'reused_tokens', 'generated_tokens',
               'drafts_offered', 'drafts_accepted')


def read_complete_suite(root, workload_id):
    files = list((root / 'bench').glob('*/results.json'))
    require(len(files) == 1, 'one official result required')
    result = json.loads(files[0].read_text())
    out, params = result['results'], result['params']
    journal = official.d.client.verify_request_journal(files[0].parent)
    require(result['complete'] and not out['failures'] and not out['skipped']
            and params['runs'] == 1 and params['think'] and params['long'] and params['long_generation']
            and journal['valid'] and journal['complete'] and not journal['pending'],
            'incomplete official full suite')
    identity = result['comparison_identity']
    expected_workload = dict(workload_id=workload_id, nonce=workload_id, nonce_policy='fixed-id')
    require(result['machine']['context'] == identity['context'] == 262144
            and identity['workload'] == params['workload'] == expected_workload
            and identity['long_generation_version'] == WORKLOAD_VERSION
            and identity['client'] == official.d.client.client_fingerprint(),
            'benchmark identity, workload or context differs')
    events = [json.loads(line) for line in (files[0].parent / 'requests.jsonl').read_text().splitlines()]
    starts = {event['index']: event for event in events if event['kind'] == 'start'}
    samples = []
    for event in events:
        if event['kind'] != 'result':
            continue
        sample = event['sample']
        start = starts[event['request_index']]
        require(sample['ok'] and sample['stream_complete']
                and sample['request_sha256'] == start['request_sha256']
                and start['thinking'] is (event['phase'] != 'calibration'),
                'actual request/body/thinking differs')
        if event['phase'] != 'calibration':
            comparison.require_valid_client_ttft(sample)
        samples.append(dict(sample, phase=event['phase'], run=event['run'], turn=event['turn']))
    order = ['warm-up/chat', 'calibration', 'calibration', 'warm-up/prompt', *(['chat'] * 3),
             'prompt/4K', 'prompt/16K', *(['agent'] * 6), 'prompt/32K',
             *('long-generation/' + name for name in SUSTAINED_WORKLOADS)]
    require([s['phase'] for s in samples] == order, 'unexpected complete-suite request order')
    nominal_32k, = out['prompts']['32K']['requests']
    require(out['prompts']['32K']['target_tokens'] == 32768
            and nominal_32k['read_tokens'] == nominal_32k['prompt_tokens'] > 0
            and nominal_32k['reused_tokens'] == 0
            and nominal_32k['prompt_tps_source'] == 'engine',
            'fresh engine timing for the nominal 32K request is missing')
    generation = out['long_generation']
    require(generation['workload_version'] == official.d.client.long_generation.WORKLOAD_VERSION == WORKLOAD_VERSION
            and generation['complete'] is True and generation['diagnostic_subset'] is False
            and generation['thinking'] is True and generation['reasoning_budget_tokens'] == 0
            and generation['max_tokens'] == 16384 and generation['selected_workloads'] == list(SUSTAINED_WORKLOADS),
            'pinned v5 thinking/output/workload differs')
    rows = gen.collect_samples(generation)
    require([(s['workload'], s['run'], s['order']) for s in rows]
            == [(name, 0, index) for index, name in enumerate(SUSTAINED_WORKLOADS)]
            and all(s['controller_eligible'] for s in rows), 'ineligible or incomplete v5 sustained generation')
    for sample, row in zip(samples[-2:], rows, strict=True):
        require(all(sample[field] == row[field] for field in (*WORK_FIELDS, 'request_sha256', 'decode_ms')),
                'v5 generation/journal work evidence differs')
        sample.update(row)
    agent = out['agent']
    session, = agent['sessions']
    require(agent['complete_sessions'] == 1 and not agent['partial_sessions'] and not agent['failed_sessions']
            and official.d.client.agent_complete(session), 'incomplete official agent session')
    turns = [s for s in samples if s['phase'] == 'agent']
    require([s['turn'] for s in turns] == list(range(6))
            and all(s['decode_tps_source'] == 'engine' and math.isfinite(s['decode_ms'])
                    and s['decode_ms'] > 0 and s['reasoning_chars'] > 0 for s in turns),
            'unclocked agent session')
    for sample, turn in zip(turns, session, strict=True):
        require(all(sample[field] == turn[field] for field in (*WORK_FIELDS, 'request_sha256', 'decode_ms')),
                'agent summary and journal work evidence differ')
    rates = {s['workload']: s['generated_tokens'] * 1000 / s['decode_ms'] for s in rows}
    rates['agent_followup'] = statistics.median(s['generated_tokens'] * 1000 / s['decode_ms'] for s in turns[1:])
    return dict(samples=samples, decode_rates=rates, journal=journal)


def work_signature(samples, metric):
    if metric == 'agent_followup':
        selected = [s for s in samples if s['phase'] == 'agent' and s['turn'] > 0]
    else:
        selected = [s for s in samples if s['phase'] == 'long-generation/' + metric]
    return [tuple(sample[field] for field in WORK_FIELDS) for sample in selected]


def decide(processes):
    result = official.decision(processes)
    matched = {name: [] for name in (*SUSTAINED_WORKLOADS, 'agent_followup')}
    for first, second in zip(processes[::2], processes[1::2], strict=True):
        left, right = first['evidence']['samples'], second['evidence']['samples']
        for name in matched:
            matched[name].append(work_signature(left, name) == work_signature(right, name))
    result['matching_work_counts_by_metric'] = matched
    result['decode_preservation_with_matching_counts'] = result['preservation_screen_passed'] and all(
        all(pairs) for pairs in matched.values())
    return result
