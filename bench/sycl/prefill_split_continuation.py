"""Measure sustained decode after the actual split-prefill route on the same ELF."""

import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_split_generation as standard

clock, suite, state, require = standard.clock, standard.suite, standard.state, standard.require
WORKLOADS = suite.evidence_reader.SUSTAINED_WORKLOADS
WORK_FIELDS = suite.evidence_reader.WORK_FIELDS
ROOT = clock.model.B / 'prefill-split-continuation-v1'
KEY = 'prefill_split_continuation_v1'
THIS = str(Path(__file__).resolve().relative_to(clock.PROJECT))
INPUT_CASE = 'split-v1-96k'
WORKLOAD_ID = 'split_continuation_v1'
MAX_OUTPUT = 16384


def command(plan):
    root = Path(plan['root'])
    return ['/bin/sh', str(clock.PROJECT / 'bench.sh'), '--config', str(root / 'config.json'),
            '--port', str(plan['port']), '--long-generation-only', '--long-generation-input-case', INPUT_CASE,
            '--long-generation-max-tokens', str(MAX_OUTPUT), '--think', '--runs', '1',
            '--workload-id', WORKLOAD_ID, '--fail-fast-engine-error', '--ready-timeout', '600',
            '--out', str(root / 'bench')]


def read_complete_suite(root, workload_id):
    path, = (root / 'bench').glob('*/results.json')
    result = clock.build.previous.read(path)
    params, identity, out = result['params'], result['comparison_identity'], result['results']
    client = suite.official.d.client
    journal = client.verify_request_journal(path.parent)
    workload = dict(workload_id=workload_id, nonce=workload_id, nonce_policy='fixed-id')
    require(result['complete'] and not out['failures'] and not out['skipped']
            and params['runs'] == 1 and params['think'] and params['long_generation_only']
            and params['long_generation'] and not params['long_generation_speed_only']
            and result['machine']['context'] == identity['context'] == 262144
            and identity['workload'] == params['workload'] == workload
            and identity['long_generation_version'] == 'split-v1'
            and identity['client'] == client.client_fingerprint()
            and journal['valid'] and journal['complete'] and not journal['pending'],
            'incomplete split continuation or changed benchmark identity')
    cohort = out['long_generation']
    geometry = cohort['long_generation_input']
    require(cohort['complete'] and cohort['thinking'] and not cohort['speed_only']
            and not cohort['diagnostic_subset'] and cohort['workload_version'] == 'split-v1'
            and cohort['reasoning_budget_tokens'] == 0 and cohort['max_tokens'] == MAX_OUTPUT
            and geometry == params['long_generation_input'] == identity['long_generation_input']
            and geometry['case'] == INPUT_CASE and cohort['selected_workloads'] == list(WORKLOADS),
            'split-input geometry, thinking or generation workload differs')
    rows = suite.gen.collect_samples(cohort, MAX_OUTPUT)
    require([(s['workload'], s['run'], s['order']) for s in rows] ==
            [(name, 0, i) for i, name in enumerate(WORKLOADS)] and all(s['controller_eligible'] for s in rows),
            'both sustained generation samples must qualify')
    events = [json.loads(line) for line in (path.parent / 'requests.jsonl').read_text().splitlines()]
    requests = [e for e in events if e['kind'] in ('start', 'result')]
    suite.gen.warmup_reconciliation(requests, WORKLOADS, MAX_OUTPUT)
    preflight = [e for e in events if e['kind'] == 'count-preflight']
    require(len(preflight) == len(WORKLOADS), 'require one count preflight for each split request')
    samples = [dict(e['sample'], phase=e['phase'], run=e['run'], turn=e['turn'])
               for e in requests if e['kind'] == 'result']
    for row, sample, checked in zip(rows, samples[-len(WORKLOADS):], preflight, strict=True):
        proof = checked['proof']
        require(checked['phase'] == sample['phase'] == 'long-generation/' + row['workload']
                and checked['run'] == sample['run'] == 0 and proof['input_case'] == row['input_case'] == INPUT_CASE
                and proof['body_sha256'] == row['count_preflight_body_sha256'] == row['request_sha256']
                and proof['counted_input_tokens'] == row['counted_input_tokens'] == row['read_tokens']
                and 90114 <= row['read_tokens'] <= 117489
                and row['timing_frontend_state'] == 'post-count-preflight'
                and all(row[k] == sample[k] for k in (*WORK_FIELDS, 'request_sha256', 'decode_ms', 'prompt_ms')),
                'counted inputs, actual requests or journal clocks differ')
        clock.model.basis.comparison.require_valid_client_ttft(sample)
        sample.update(row)
    return dict(samples=samples, journal=journal,
                decode_rates={r['workload']: r['generated_tokens'] * 1000 / r['decode_ms'] for r in rows})


def routes(text, samples, arm, *, kv_reuse=None):
    standard.profiles.check_profile_startup(text, 'original')
    require(not any(mark in text for mark in ('strata trace:', 'strata loan audit:',
                                             'strata MoE prefix expert upload layer:', 'strata decode timing:')),
            'unplanned route or observer')
    heads = list(re.finditer(clock.footer.HEADER, text, re.M))
    footers = list(re.finditer(clock.footer.COPY, text, re.M))
    require(len(heads) == len(footers) == len(samples), 'one native request and expert footer per sample')
    reader = clock.model.basis.reference.context()
    results = {}
    for i, (head, footer, sample) in enumerate(zip(heads, footers, samples, strict=True)):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        require(head.end() <= footer.start() < end and tuple(map(int, head.groups())) == tuple(sample[k] for k in
                ('prompt_tokens', 'reused_tokens', 'read_tokens', 'generated_tokens', 'drafts_accepted', 'drafts_offered')),
                'native request or expert footer differs from the client')
        if not sample['phase'].startswith('long-generation/'):
            continue
        frame = text[heads[i - 1].end() if i else 0:head.start()]
        read = sample['read_tokens']
        route = reader.request_route(frame, dict(read=read, heads=1))
        window = route['fast_window']
        require(window['staged'] and window['split'] and window['prefill_slots'] == 128
                and window['maximum_window_rows'] == route['residual_gpu_rows'] == read - 1
                and route['residual_host_bytes'] == 0 and window['split_guard_checked']
                and window['staged_bank_checked_drain'] and route['MTP_expert_restore_bytes'] == 707788800,
                'actual split, tail guard, drain or MTP restoration differs')
        store = clock.model.restore.require_store(frame, dict(direct=False), dict(read=read))
        clock.model.input_receipt(frame, read - 1, store, False, trace_layers=False)
        kv = state.kv.check_clean_request(frame, 0, read - 1, 8192,
                                          reuse=arm == 'candidate' if kv_reuse is None else kv_reuse)
        decode = dict(zip(clock.footer.COPY_KEYS, map(int, footer.groups()), strict=True))
        require(all(decode[k] == 0 for k in clock.footer.COPY_KEYS[:6])
                and 0 < decode['cpu_unique_misses'] <= decode['cpu_entries'],
                'the pinned post-refill decode expert route did not run')
        results[sample['workload']] = dict(read=read, prefill_slots=128, split_guard_checked=True,
            MTP_expert_restore_bytes=route['MTP_expert_restore_bytes'], kv=kv, store=store, decode=decode,
            native_work=dict(gather=state.gather_counts(frame, False), routes=route['actual_GU_dispatch'],
                             bank_bytes=window['requested_bank_H2D_bytes']))
    require(set(results) == set(WORKLOADS), 'both split-prefill handoffs are required')
    return results


def decision(processes):
    require(tuple(p['arm'] for p in processes) == suite.ORDER
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == 6
            and all(p['complete'] and p['passed'] and p['ownership_closed'] and not p['fresh_faults']
                    and p['native_exit_code'] == 0 for p in processes), 'six independent clean processes required')
    by_workload = {n: [] for n in WORKLOADS}
    classes = {}
    for arm in ('original', 'candidate'):
        cases = {n: dict(requests=[s for p in processes if p['arm'] == arm
                                  for s in p['evidence']['samples'] if s.get('workload') == n]) for n in WORKLOADS}
        classes[arm] = suite.official.d.client.long_generation.classify_pair(cases, 3)
    for a, b in zip(processes[::2], processes[1::2], strict=True):
        reference, candidate = (a, b) if a['arm'] == 'original' else (b, a)
        for name in WORKLOADS:
            x, y = [next(s for s in p['evidence']['samples'] if s.get('workload') == name)
                    for p in (reference, candidate)]
            r, c = reference['routes'][name], candidate['routes'][name]
            same_prefill_work = r['native_work'] == c['native_work'] and r['read'] == c['read'] and r['store'] == c['store']
            same_decode_work = all(x[k] == y[k] for k in WORK_FIELDS) and r['decode'] == c['decode']
            by_workload[name].append(dict(same_prefill_work=same_prefill_work, same_decode_work=same_decode_work,
                prompt_time_ratio=y['prompt_ms'] / x['prompt_ms'], TTFT_ratio=y['ttft_s'] / x['ttft_s'],
                TG_ratio=candidate['evidence']['decode_rates'][name] / reference['evidence']['decode_rates'][name]))
    screens = {n: dict(matched_prefill=all(p['same_prefill_work'] for p in samples),
        matched_decode=all(p['same_decode_work'] for p in samples),
        TG_geomean=math.exp(statistics.mean(math.log(p['TG_ratio']) for p in samples)))
        for n, samples in by_workload.items()}
    return dict(pairs=by_workload, MTP=classes, workload_screens=screens,
        preservation_screen_passed=all(c['qualified'] for c in classes.values()) and all(
            s['matched_prefill'] and s['matched_decode'] and s['TG_geomean'] >= 1 for s in screens.values()),
        scope='Sustained thinking after actual split prefill. Physical full fill and RESTORE are separate gates.',
        adopted=False)


def experiment():
    ctx = standard.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.WORKLOAD_ID = ROOT, KEY, THIS, WORKLOAD_ID
    ctx.command, ctx.routes = command, routes
    ctx.evidence_reader.read_complete_suite, ctx.evidence_reader.decide = read_complete_suite, decision
    ctx.protocol = dict(order=list(suite.ORDER), input_case=INPUT_CASE, workloads=list(WORKLOADS),
        compare='Three same-binary pairs with fixed stock decode and scalar gather. Switch aligned KV reuse alone.',
        acceptance='Match exact counted bodies, native split placement, checked tail and handoff. Keep first and '
                   'second request clocks by workload. To pass preservation, both arms must show the expected MTP '
                   'classes. Every pair must match prefill and decode work, and each workload must have a geometric '
                   'mean TG ratio of at least one. Report TG as workload speed when decode work changes.',
        stop='Stop at first identity, route, clock, work eligibility, runtime, ownership or fault failure. '
             'No speed exclusion or automatic adoption.')
    original_environment, original_pins = ctx.environment, ctx.pins

    def environment(row, plan):
        require(row['selected_arm'] == 'reuse', 'the split throughput decision must select KV reuse alone')
        return original_environment(row, plan)

    def pins(revision):
        values = original_pins(revision)
        path = clock.PROJECT / THIS
        raw = suite.gen.git('show', revision + ':' + THIS)
        require(path.read_bytes() == raw, 'commit the split continuation controller')
        values[THIS] = clock.build.sha(path)
        return values

    ctx.environment, ctx.pins = environment, pins
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
