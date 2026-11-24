"""Attribute DFlash2 decode CPU-leaf service on the retained current engine."""

import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_kv_stage_speed as speed
import profile_adaptive_cpu_leaves as leaf_reader

generation, bank, basis = speed.generation, speed.bank, speed.basis
require, clock, suite = speed.require, generation.clock, generation.suite
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_cpu_leaves_v1'
ROOT = bank.B / KEY.replace('_', '-')
REFERENCE = 'e26315def2fb25b591ac6070b78236276ef2456d'
REFERENCE_KEY = 'dflash2_v2_decode_clocks_v1'
WORKLOAD, PHASE = 'regular-code', 'long-generation/regular-code'
ORDER = ('clean', 'leaves', 'leaves', 'clean')
WORK = suite.evidence_reader.WORK_FIELDS + ('request_sha256', 'output_sha256', 'reasoning_sha256')
OBSERVER = 'STRATA_CPU_SERVICE'
OVERHEAD_FACTOR = 1.10
PRODUCERS = ('src/program/generate.cpp', 'src/kernels/cpu/pool.cpp',
             'include/strata/kernels/cpu/pool.hpp', 'include/strata/kernels/cpu/service_profile.hpp')


def reference():
    prior = bank.previous.historic(REFERENCE, REFERENCE_KEY)
    require(prior['stage'] == 'closed' and prior['complete'] and prior['passed']
            and prior['ownership_closed'] and not prior['active']
            and all(case['passed'] and case['native_exit_code'] == 0 and not case['fresh_faults']
                    for case in prior['processes']), 'recover the closed DFlash2 code bracket')
    samples = prior['processes'][0]['evidence']['samples']
    require(len(samples) == 5 and samples[-1]['phase'] == PHASE
            and samples[-1]['read_tokens'] >= 32768, 'recover exact code request framing and body')
    return prior, samples[-1]


def baseline():
    row = speed.baseline()
    prior, sample = reference()
    require(all(row[k] == prior[k] for k in ('boot', 'driver', 'runtime_manifest', 'native_validation_probe',
                                             'dflash2_assets', 'validated_dflash2_args', 'commit_order_gate')),
            'DFlash2 build must retain the tested code route and platform')
    root = speed.build.DFLASH2_KV_ROOT
    built = bank.previous.read(root / 'record.json')
    require(built['binaries']['strata'] == row['binary']
            and '-DSTRATA_CPU_SERVICE_PROFILE:BOOL=ON' in built['recipe']['configure'],
            'the exact retained native build must enable the leaf producer')
    producer_pins = {name: built['source_files'][name] for name in PRODUCERS}
    require(all(bank.sha(root / 'source' / name) == sha for name, sha in producer_pins.items()),
            'actual native leaf producer inputs differ')
    reader = basis.validation.committed(bank.RESULT, ('adaptive_cpu_leaf_reader_host_v2',))
    require(reader['stage'] == 'closed' and reader['complete'] and reader['passed']
            and reader['ownership_closed'] and not reader['active'] and not reader['GPU_executed']
            and len(reader['tests']) == 1 and reader['tests'][0]['tests'] == 6
            and reader['tests'][0]['exit_code'] == 0 and reader['tests'][0]['skipped'] == 0,
            'recover the qualified version-two leaf reader')
    reader_paths = ('tools/sycl/adaptive_cpu_leaf_evidence.py', 'tools/sycl/cpu_service_evidence.py',
                    'tools/sycl/cpu_work_census.py', 'tools/sycl/decode_footer.py')
    require(all(bank.sha(bank.P / name) == reader['sources'][name] for name in reader_paths),
            'the qualified leaf reader inputs changed')
    return dict(row, request_reference=dict(commit=REFERENCE, sha256=sample['request_sha256']),
                leaf_producer_sha256=producer_pins, leaf_reader_sha256=bank.signature(reader))


def configuration(row, plan):
    cfg = speed.configuration(row, plan)
    leaf_reader.serialized_configuration(cfg)
    args = cfg['args']
    expected = {'--draft': 'dflash2', '--spec': '4', '--pool-workers': '5', '--pcie-frac': '0',
                '--adapt-swaps': '0', '--prefill': '8192', '--max-context': '262144', '--prompt-cache': '0',
                '--expert-cache': '1256'}
    require(all(args.count(key) == 1 and args[args.index(key) + 1] == value for key, value in expected.items()),
            'DFlash2 decode placement or serial execution differs')
    return cfg


def environment(row, plan):
    env = speed.environment(row, dict(plan, arm='retain'))
    require(not any(k in env for k in ('STRATA_TRACE', 'STRATA_DECODE_TIMING', 'STRATA_DRAFT_TIMING'))
            and all(env.get(k, '0') == '0' for k in (OBSERVER, 'STRATA_CPU_WORK_CENSUS',
                                                   'STRATA_SYCL_DECODE_SERVICE_PROFILE')),
            'unexpected inherited observer')
    env[OBSERVER] = str(int(plan['arm'] == 'leaves'))
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



def routes(text, samples, arm):
    speed.validate_model_text(text, {'arm': arm})
    require(len(samples) == 5 and samples[-1]['phase'] == PHASE
            and not any(mark in text for mark in ('strata trace:', 'strata decode timing:', 'strata decode service:')),
            'four warmups and one long request in one quiet process required')
    native, leaf = leaf_reader.evidence(text, samples, 4, arm == 'leaves')
    headers = list(re.finditer(clock.footer.HEADER, text, re.M))
    sample = samples[-1]
    require(sample['read_tokens'] >= 32768 and sample['reused_tokens'] == 0
            and sample['generated_tokens'] >= 1024 and sample['drafts_offered'] >= 256,
            'long code request did not exercise sustained DFlash2 decode')
    frame = text[headers[-2].end():headers[-1].start()]
    read = sample['read_tokens']
    store = generation.model.restore.require_store(frame, dict(direct=False), dict(read=read))
    require(store['gpu_rows'] == read - 1 and store['window_mode'] == 10
            and len(re.findall(r'^strata DFlash2: eager, prompt_tail_rows=2046 cold_prefix=0 '
                               r'vram_bytes=[1-9][0-9]*$', frame, re.M)) == 1,
            'ordinary all-GPU prompt handoff or DFlash2 tap differs')
    kv = speed.KV.check_clean_request(frame, 0, read - 1, 8192, reuse=True,
                                       tap_layers=speed.gate.TAPPED_QSA_LAYERS)
    receipt = dict(counted_requests=len(samples), expert_work=native, kv=kv, attention_store=store)
    if leaf:
        receipt['leaves'] = leaf_reader.summarize(leaf[-1])
    return receipt


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER)
            and all(p['complete'] and p['passed'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in processes), 'four closed counterbalanced code processes required')
    samples = [p['evidence']['samples'][-1] for p in processes]
    work = [
        (tuple(tuple(sample[k] for k in WORK) for sample in p['evidence']['samples']),
         p['routes']['expert_work'], p['routes']['kv'])
        for p in processes
    ]
    require(all(math.isfinite(s['decode_ms']) and s['decode_ms'] > 0 for s in samples), 'invalid decode clock')
    clean_indices, observed_indices = (0, 3), (1, 2)
    ratios = [samples[observed]['decode_ms'] / samples[clean]['decode_ms']
              for clean, observed in zip(clean_indices, observed_indices, strict=True)]
    same_counts = all(signature == work[0] for signature in work)
    same_observed_format_work = (processes[1]['routes']['leaves']['format_nt_jobs'] ==
                                 processes[2]['routes']['leaves']['format_nt_jobs'])
    rates = [1000 * s['generated_tokens'] / s['decode_ms'] for s in samples]
    clean_rates = [rates[index] for index in clean_indices]
    return dict(matched_request_and_job_counters=same_counts,
                matched_observed_format_work=same_observed_format_work, observed_over_clean_latency=ratios,
                leaf_observer_cost_qualified=same_counts and same_observed_format_work and
                    all(1 / OVERHEAD_FACTOR <= q <= OVERHEAD_FACTOR for q in ratios),
                clean_TG=clean_rates, clean_TG_median=statistics.median(clean_rates),
                observed_TG=[rates[index] for index in observed_indices],
                expert_identity_qualified=False, physical_bandwidth_qualified=False,
                hardware_saturation_qualified=False)


def experiment():
    ctx = generation.experiment(synchronous_commit=True)
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment, ctx.command = baseline, configuration, environment, command
    ctx.routes, ctx.evidence_reader.read_complete_suite, ctx.evidence_reader.decide = routes, read_subset, decision
    ctx.protocol = dict(order=list(ORDER), workload=WORKLOAD, overhead_factor_limit=OVERHEAD_FACTOR,
        unit='Change only the compiled CPU worker leaf observer on the retained DFlash2 executable and model route.',
        acceptance='Require byte-identical regular-code inputs, four warmups before the thinking-on sustained request, '
                   'and matching output and CPU job counters across two independent pairs in reverse order. '
                   'Check all native worker requests and conserve GU, hidden quantization and Down jobs against '
                   'the expert footers. Match observed format/width bins. Compare paired latency within the declared factor.',
        stop='Stop on changed sources, work framing, route, validation, ownership, budget or fault. '
             'The code subset isolates attribution; it does not qualify weak-acceptance performance or physical traffic.')
    old_pins, old_current = ctx.pins, ctx.current

    def pins(revision):
        sources = old_pins(revision)
        for name in (THIS, 'tools/sycl/profile_adaptive_cpu_leaves.py',
                     'tools/sycl/adaptive_cpu_leaf_evidence.py', 'tools/sycl/cpu_service_evidence.py',
                     'tools/sycl/cpu_work_census.py', 'tools/sycl/decode_footer.py'):
            require((bank.P / name).read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed DFlash2 CPU leaf input: ' + name)
            sources[name] = bank.sha(bank.P / name)
        return sources

    def current(row):
        normalized = []
        for plan in row['plans']:
            env = environment(row, plan)
            env.pop(OBSERVER)
            normalized.append(bank.signature({k: v.replace(plan['root'], '<process>') for k, v in env.items()}))
        require(len(set(normalized)) == 1, 'another native setting differs between leaf observer arms')
        return old_current(row)

    ctx.pins, ctx.current = pins, current
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
