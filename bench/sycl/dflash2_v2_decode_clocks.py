"""Measure caller waits and verifier work on the fixed long-code DFlash2 route."""

import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_v2_generation as generation
import decode_service_evidence as caller

bank, basis, clock, suite = generation.bank, generation.basis, generation.clock, generation.suite
require, footer = generation.require, generation.clock.footer
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_v2_decode_clocks_v1'
ROOT = bank.B / KEY.replace('_', '-')
REFERENCE = 'bdd9ff4ce094eb9946bda4d2599fa1fc284e3d3a'
WORKLOAD, PHASE = 'regular-code', 'long-generation/regular-code'
ORDER = ('clean-before', 'clocks', 'clean-after')
OBSERVERS = ('STRATA_DECODE_TIMING', 'STRATA_SYCL_DECODE_SERVICE_PROFILE')
WORK = suite.evidence_reader.WORK_FIELDS + ('request_sha256', 'output_sha256', 'reasoning_sha256')
LIMIT = 1.10


def reference():
    evidence = bank.previous.historic(REFERENCE, generation.SYNC_KEY)
    process = evidence['processes'][0]
    sample, = (s for s in process['evidence']['samples'] if s['phase'] == PHASE)
    require(process['complete'] and process['passed'] and process['ownership_closed']
            and process['native_exit_code'] == 0 and process['fresh_faults'] == []
            and suite.gen.sample_eligible(sample), 'recover the closed official DFlash2 code request')
    return evidence, sample


def baseline():
    row = generation.sync_baseline()
    evidence, sample = reference()
    require(all(evidence[k] == row[k] for k in ('binary', 'native_source_commit', 'native_build_sha256',
                                               'boot', 'driver', 'runtime_manifest')),
            'the observed route must use the same recovered V2 build and runtime')
    return dict(row, request_reference=dict(commit=REFERENCE, sha256=sample['request_sha256']))


def environment(row, plan):
    env = generation.sync_environment(row, plan)
    require(not any(k in env for k in (*OBSERVERS, 'STRATA_TRACE', 'STRATA_DRAFT_TIMING'))
            and all(env.get(k, '0') == '0' for k in ('STRATA_CPU_SERVICE', 'STRATA_CPU_WORK_CENSUS')),
            'unexpected inherited timing observer')
    if plan['arm'] == 'clocks':
        env.update(dict.fromkeys(OBSERVERS, '1'))
    return env


def command(plan):
    root = Path(plan['root'])
    return ['/bin/sh', str(bank.P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(plan['port']),
            '--long-generation-only', '--long-generation-workload', WORKLOAD, '--think', '--runs', '1',
            '--workload-id', generation.WORKLOAD_ID, '--fail-fast-engine-error', '--ready-timeout', '600',
            '--out', str(root / 'bench')]


def read_subset(root, workload_id):
    path, = (root / 'bench').glob('*/results.json')
    result = bank.previous.read(path)
    out, params, identity = (result[k] for k in ('results', 'params', 'comparison_identity'))
    client = suite.official.d.client
    journal = client.verify_request_journal(path.parent)
    workload = dict(workload_id=workload_id, nonce=workload_id, nonce_policy='fixed-id')
    cohort = out['long_generation']
    require(result['complete'] and not out['failures'] and not out['skipped']
            and params['runs'] == 1 and params['think'] and params['long_generation_only']
            and cohort['complete'] and cohort['diagnostic_subset'] and cohort['selected_workloads'] == [WORKLOAD]
            and not cohort['speed_only'] and cohort['thinking'] and cohort['max_tokens'] == 16384
            and cohort['reasoning_budget_tokens'] == 0
            and cohort['workload_version'] == identity['long_generation_version'] == 5
            and identity['workload'] == params['workload'] == workload
            and identity['context'] == result['machine']['context'] == 262144
            and identity['client'] == client.client_fingerprint()
            and journal['valid'] and journal['complete'] and not journal['pending'],
            'official code subset, context or journal differs')
    observations = [json.loads(line) for line in (path.parent / 'requests.jsonl').read_text().splitlines()]
    suite.gen.warmup_reconciliation(observations, (WORKLOAD,))
    samples = [dict(e['sample'], phase=e['phase'], run=e['run'], turn=e['turn'])
               for e in observations if e['kind'] == 'result']
    sample, = suite.gen.collect_samples(cohort)
    last = samples[-1]
    _, expected = reference()
    require(sample['controller_eligible'] and (sample['workload'], sample['run'], sample['order']) == (WORKLOAD, 0, 0)
            and sample['request_sha256'] == expected['request_sha256']
            and last['phase'] == PHASE
            and all(sample[k] == last[k] for k in (*WORK, 'decode_ms', 'prompt_ms', 'ttft_s')),
            'the long code request must match its original body and current journal')
    basis.comparison.require_valid_client_ttft(last)
    fields = (*WORK, 'phase', 'run', 'turn', 'decode_ms', 'prompt_ms', 'ttft_s')
    return dict(samples=[{k: s[k] for k in fields} for s in samples], journal=journal,
                decode_rates={WORKLOAD: 1000 * last['generated_tokens'] / last['decode_ms']})


def decode_clocks(timing, sample, copy):
    windows = int(timing[0])
    observed = dict(zip(footer.LABELS, map(float, timing[1:]), strict=True))
    offered, accepted, emitted = (sample[k] for k in ('drafts_offered', 'drafts_accepted', 'generated_tokens'))
    rows = windows + offered
    jobs, entries = (copy[k] for k in ('cpu_unique_misses', 'cpu_entries'))
    layer_windows = windows * 48
    require(windows > 0 and 0 <= accepted <= offered <= 3 * (windows - 1)
            and 0 <= windows + accepted - emitted <= 4 and emitted > 0
            and all(math.isfinite(v) and v >= 0 for v in observed.values()), 'invalid adaptive verifier work or clocks')
    require(all(copy[k] == 0 for k in footer.COPY_KEYS[:6]) and observed['PCIe'] == 0
            and 0 < jobs <= entries <= min(4 * jobs, rows * 48 * 10), 'unexpected expert placement or work')
    expected = dict(average_T=rows / windows, tokens_per_window=emitted / windows, CPU_experts=jobs / layer_windows,
                    CPU_entries=entries / layer_windows, VRAM_hits=(rows * 48 * 10 - entries) / layer_windows)
    require(all(abs(observed[k] - v) <= .00501 for k, v in expected.items()),
            'rounded adaptive-window footer disagrees with executed work')
    delta = abs(observed['total'] * windows - sample['decode_ms'])
    require(delta <= .005 * windows + .050001, 'footer clock disagrees with native API decode time')
    outside = observed['total'] - sum(observed[k] for k in ('verify', 'commit_emit', 'draft'))
    require(outside >= -.02001, 'outer decode intervals exceed the request clock')
    return dict(windows=windows, target_rows=rows, target_rows_per_emitted=rows / emitted,
                wall_ms_per_window=observed, unclassified_ms_per_window=outside,
                approximate_ms_per_emitted={k: v * windows / emitted for k, v in observed.items()
                                             if k in ('total', 'verify', 'GPU_reach_wait', 'CPU_pool_run',
                                                      'stage', 'commit_emit', 'draft')},
                footer_rounding_delta_ms=delta)


def routes(text, samples, arm):
    generation.v2.continuation.standard.profiles.check_profile_startup(text, 'original')
    require('strata trace:' not in text and 'strata loan audit:' not in text
            and text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
            and not suite.gen.draft_rows(text), 'unexpected route or observer')
    headers = list(re.finditer(footer.HEADER, text, re.M))
    require(len(headers) == len(samples) and samples[-1]['phase'] == PHASE, 'native request order differs')
    for header, sample in zip(headers, samples, strict=True):
        require(tuple(map(int, header.groups())) == tuple(sample[k] for k in (
            'prompt_tokens', 'reused_tokens', 'read_tokens', 'generated_tokens', 'drafts_accepted', 'drafts_offered')),
            'native request extent or draft counters differ')
    frame = text[headers[-2].end():headers[-1].start()]
    tail = text[headers[-1].end():]
    sample = samples[-1]
    read = sample['read_tokens']
    store = generation.model.restore.require_store(frame, dict(direct=False), dict(read=read))
    require(store['gpu_rows'] == read - 1 and store['window_mode'] == 10
            and len(re.findall(r'^strata DFlash2: eager, prompt_tail_rows=2046 cold_prefix=0 vram_bytes=[1-9][0-9]*$', frame, re.M)) == 1,
            'expected ordinary DFlash2 all-GPU residual route')
    kv = generation.v2.state.kv.check_clean_request(frame, 0, read - 1, 8192, reuse=False)
    copy_fields, = re.findall(footer.COPY, tail, re.M)
    copied = dict(zip(footer.COPY_KEYS, map(int, copy_fields), strict=True))
    parsed = dict(kv=kv, attention_store=store, expert_copy=copied)
    if arm == 'clocks':
        timing, = re.findall(footer.TIMING, frame, re.M)
        ledger = caller.extract(text, list(range(1, len(samples) + 1)))[-1]
        parsed.update(clocks=decode_clocks(timing, sample, copied), caller=caller.compact(ledger))
    else:
        require('strata decode timing:' not in text, 'unexpected clocks on clean arm')
        caller.extract(text, ())
    return parsed


def decision(processes):
    samples = [p['evidence']['samples'][-1] for p in processes]
    ratios = [samples[1]['decode_ms'] / samples[i]['decode_ms'] for i in (0, 2)]
    work = [([s[k] for k in WORK], p['routes']['expert_copy']) for s, p in zip(samples, processes, strict=True)]
    matches = work[0] == work[1] == work[2]
    require(tuple(p['arm'] for p in processes) == ORDER
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == 3,
            'three independent fresh processes required')
    rates = [1000 * s['generated_tokens'] / s['decode_ms'] for s in samples]
    return dict(matched_work=matches, observed_over_clean_latency=ratios,
                bounded_attribution=matches and all(1 / LIMIT <= v <= LIMIT for v in ratios),
                clean_TG=[rates[0], rates[2]], observed_TG=rates[1],
                clean_TG_median=statistics.median((rates[0], rates[2])),
                hardware_saturation_qualified=False)


def experiment():
    ctx = generation.experiment(synchronous_commit=True)
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.environment, ctx.command, ctx.routes = baseline, environment, command, routes
    ctx.evidence_reader.read_complete_suite = read_subset
    ctx.evidence_reader.decide = decision
    ctx.protocol = dict(order=list(ORDER), workload=WORKLOAD, overhead_factor_limit=LIMIT,
        unit='Compare the compiled rounded verifier clocks and conserved caller waits as one observer arm.',
        acceptance='Use the original code request body with thinking and a visible answer, >=32768 fresh inputs, '
                   '>=1024 generated tokens and >=256 offers. Keep the synchronized DFlash2 V2 placement, '
                   'confidence 0.6 and capped width four. Vary only the two observer flags in fresh processes.',
        decision='Match emitted work, draft counts and CPU expert visits. Both observed/clean latency ratios '
                 'must be within a factor of 1.10 before interpreting the caller intervals at this workload. '
                 'Keep host wall, caller CPU within waits and overlapping work distinct. Physical bus traffic '
                 'and device arithmetic saturation require separate evidence. Keep the clean samples even if '
                 'the clocks are too intrusive or unrelated workload differences prevent attribution.',
        stop='Stop at the first source, ownership, route, eligibility, validation, fault or budget failure.')
    old_pins, old_current = ctx.pins, ctx.current

    def pins(revision):
        sources = old_pins(revision)
        for path in (THIS, str(Path(caller.__file__).resolve().relative_to(bank.P))):
            require((bank.P / path).read_bytes() == basis.validation.git('show', revision + ':' + path),
                    'changed DFlash2 attribution input: ' + path)
            sources[path] = bank.sha(bank.P / path)
        return sources

    def current(row):
        normalized = []
        for plan in row['plans']:
            env = environment(row, plan)
            normalized.append({k: v.replace(plan['root'], '<process>')
                               for k, v in env.items() if k not in OBSERVERS})
        require(all(env == normalized[0] for env in normalized), 'another immutable setting differs between arms')
        return old_current(row)

    ctx.pins, ctx.current = pins, current
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
