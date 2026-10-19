#!/usr/bin/env python3
"""Main-owned official long-generation checks and independent clean processes.

Reuse the selected native build; never rebuild, restart a failed engine, change
runtime settings in-process or infer a decode speedup from the prefill screen.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import bench_engine as common
import measurement_guards as guards
import measurement_owner as supervision
from build_engine import blobs

P, B, A = common.P, common.B, common.A
RESULT = A / 'decode-long-generation-current.json'
PREFILL = A / 'prefill-long-input-current.json'
SETTINGS = common.SETTINGS
WORKLOAD_ID = 'native16_decode_long_20261019'
RESEARCH = A / 'research-index.json'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P)


def committed(path):
    data = path.read_bytes()
    guards.require(data == git('show', 'HEAD:' + str(path.relative_to(P))),
                   'commit the exact input first: ' + str(path))
    return json.loads(data)


def benchmark_pins(revision):
    names = git('ls-tree', '-r', '--name-only', revision, '--', 'serve', 'tools', 'bench.sh').decode().splitlines()
    names = [n for n in names if n == 'bench.sh' or
             (n.endswith('.py') and not n.startswith('tools/hip/'))]
    pins = {}
    for name, data in blobs(P, revision, names):
        guards.require((P / name).read_bytes() == data, 'uncommitted benchmark input: ' + name)
        pins[name] = hashlib.sha256(data).hexdigest()
    return pins


def sample_eligible(row):
    """Independently reject mislabeled short, incomplete or unclocked samples."""
    def integer(key, minimum):
        value = row.get(key)
        return type(value) is int and value >= minimum

    offered, accepted = row.get('drafts_offered'), row.get('drafts_accepted')
    return bool(row.get('eligible') is True and row.get('speed_eligible') is True and
        row.get('ok') is True and row.get('stream_complete') is True and
        row.get('finish_reason') in ('stop', 'length') and row.get('reused_tokens') == 0 and
        integer('prompt_tokens', 32768) and integer('read_tokens', 32768) and
        row['prompt_tokens'] == row['read_tokens'] and integer('usage_generated_tokens', 1024) and
        row['usage_generated_tokens'] <= 16384 and
        row.get('generated_tokens') == row['usage_generated_tokens'] and
        row.get('generated_tokens_source') == 'usage' and integer('output_chars', 512) and
        integer('reasoning_chars', 1) and row.get('completion_includes_thinking') is True and
        type(offered) is int and offered >= 256 and type(accepted) is int and 0 <= accepted <= offered and
        isinstance(row.get('acceptance_rate'), (int, float)) and
        math.isclose(row['acceptance_rate'], accepted / offered, rel_tol=0, abs_tol=1e-12) and
        row.get('decode_tps_source') == 'engine' and all(
            isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool) and
            math.isfinite(row[key]) and row[key] > 0 for key in ('decode_ms', 'decode_tps', 'total_s')))


def collect_samples(generation):
    # Preserve every completed/failed client row before assessing eligibility.
    rows = []
    for workload, case in generation.get('workloads', {}).items():
        for request in case.get('requests', []):
            row = dict(request, workload=workload)
            row['controller_eligible'] = sample_eligible(row)
            rows.append(row)
    return sorted(rows, key=lambda r: r.get('order', -1))


def require_complete_generation(generation, rows):
    guards.require(generation.get('workload_version') == 4 and generation.get('thinking') is True and
        generation.get('reasoning_budget_tokens') == 0 and generation.get('max_tokens') == 16384,
        'unexpected long-generation workload or thinking/output budget')
    guards.require(len(rows) == 2 and
        [(r.get('workload'), r.get('run'), r.get('order')) for r in rows] ==
        [('regular-code', 0, 0), ('varied-prose', 0, 1)] and
        all(r['controller_eligible'] for r in rows), 'ineligible or incomplete long-generation pilot')


def journal_evidence(root):
    sys.path.insert(0, str(P / 'tools'))
    from strata_bench import verify_request_journal
    paths = list((root / 'bench').glob('*/requests.jsonl'))
    guards.require(len(paths) == 1, 'missing or ambiguous official request journal')
    journal = verify_request_journal(paths[0].parent)
    rows = []
    for line in paths[0].read_text().splitlines()[:journal['complete_prefix_rows']]:
        row = json.loads(line)
        if row['kind'] == 'result' and row['phase'].startswith('long-generation/'):
            rows.append(dict(row['sample'], workload=row['phase'].split('/', 1)[1],
                run=row['run'], order=len(rows)))
    return journal, rows


def cpu_check_qualification(build):
    """Use fresh same-build evidence; do not broaden the historical bank reuse proof."""
    scope = committed(RESEARCH)['current_decode_mtp_verifier_reuse_source_audit']['cpu_observer_source_audit']
    delta = scope['runtime_smoke_plan']['runtime_asset_delta']
    for name in ('native_cpu_qualification', 'runtime_smoke', 'runtime_bank_validation'):
        item = scope[name]
        guards.require(item['complete'] and item['passed'] and item['ownership_closed'],
                       'CPU check prerequisite incomplete: ' + name)
    smoke, bank = scope['runtime_smoke'], scope['runtime_bank_validation']
    guards.require(scope['native_cpu_qualification']['source_commit'] == build['source_commit'] and
        smoke['binary'] == build['binary'] and
        smoke['build_source_commit'] == bank['build_source_commit'] == build['source_commit'] and
        smoke['runtime_asset_delta'] == bank['runtime_asset_delta'] == delta and
        bank['device_cases_passed'] == 6 and bank['provider_checks_passed'] and
        bank['provider_fixture'] == build['fixtures']['file_expert_source_test'] and
        bank['binary'] == build['fixtures']['sycl_expert_bank_test'], 'same-build runtime/bank qualification differs')
    fixtures = ('sycl_expert_bank_test', 'file_expert_source_test', 'sycl_moe_gu_swiglu_bench',
                'sycl_moe_down_scatter_bench', 'sycl_moe_accum_test')
    inputs = {}
    for name in fixtures:
        item = build['fixtures'][name]
        if name != 'file_expert_source_test':
            guards.require(item['sha256'] == build['retained_validation_reference']['fixtures'][name]['sha256'],
                           'previously qualified arithmetic fixture changed: ' + name)
        guards.require(common.digest(item['path']) == item['sha256'], 'CPU check fixture changed: ' + name)
        inputs[item['path']] = item['sha256']
    return delta, dict(live_inputs=inputs, source_commit=bank['source_commit'],
                      build_source_commit=build['source_commit'], provider_checks_passed=True, full_bank_cases=6)


def runtime_assets_unchanged(record):
    delta = record.get('runtime_asset_delta', {})
    guards.require(not delta or record['mode'] == 'cpu-check', 'proposed runtime is diagnostic-only')
    settings = json.loads(SETTINGS.read_text())
    guards.require(set(delta) <= set(settings['runtime_elfs']), 'unknown proposed runtime asset')
    for name, item in settings['runtime_elfs'].items():
        proposed = delta.get(name)
        if proposed:
            guards.require(proposed['qualified_sha256'] == item['sha256'], 'original runtime baseline changed')
        expected = proposed['candidate_sha256'] if proposed else item['sha256']
        guards.require(common.digest(name) == expected, 'runtime asset changed: ' + name)


def cpu_observations(root, log, rows):
    """Reconcile every successful native request, including unmeasured warmups."""
    files = list((root / 'bench').glob('*/requests.jsonl'))
    guards.require(len(files) == 1, 'missing CPU-observer request journal')
    entries = [json.loads(line) for line in files[0].read_text().splitlines()]
    results = [e for e in entries if e['kind'] == 'result']
    observed = [json.loads(line.partition(': ')[2]) for line in log.splitlines()
                if line.startswith('strata decode cpu: ')]
    expected_phases = ['warm-up/chat', 'calibration', 'calibration', 'warm-up/prompt',
                       'long-generation/regular-code', 'long-generation/varied-prose']
    guards.require([e['phase'] for e in results] == expected_phases and len(observed) == len(results),
                   'attempted/successful/native CPU-record counts do not reconcile')
    selected = []
    for entry, value in zip(results, observed):
        sample = entry['sample']
        guards.require(sample['ok'] and value.get('schema') == 1 and all(value.get(k) is True for k in
            ('valid', 'single_stage', 'clock_deltas_valid', 'quiescent', 'worker_jobs_fit', 'main_subsets_fit')) and
            all(type(value.get(k)) is int and value[k] == 0 for k in
                ('counter_errors', 'unwound', 'snapshot_read_errors', 'unexpected_mode_tasks')) and
            value['input_tokens'] == sample['prompt_tokens'] and
            value['generated_tokens'] == sample['generated_tokens'] and
            math.isfinite(value['wall_through_commit_ms']) and value['wall_through_commit_ms'] > 0,
            'invalid or mismatched successful-request CPU record')
        if entry['phase'].startswith('long-generation/'):
            guards.require(value['windows'] > 0 and all(type(value.get(k)) is int and value[k] > 0 for k in
                ('worker_job_calls', 'native_quant_calls', 'sync_calls', 'main_thread_ns', 'native_process_ns')),
                'long-generation CPU observation has no work')
            selected.append(dict(workload=entry['phase'].split('/', 1)[1], **value))
    guards.require([(o['workload'], o['input_tokens'], o['generated_tokens']) for o in selected] ==
        [(r['workload'], r['prompt_tokens'], r['generated_tokens']) for r in rows], 'CPU records differ from official rows')
    return selected, dict(successful_requests=len(results), valid_CPU_records=len(observed),
                          warmup_and_calibration_records_excluded=4, all_request_counts_reconciled=True)


def prepare(mode, cohort, root):
    reference = committed(PREFILL)
    summary = reference['independent_down_comparison']
    guards.require(reference['passed'] and reference['complete'] and reference['ownership_closed'] and
        not reference['active'] and summary['performance_selection_gate_passed'] and
        summary['prefill1000_target_passed'], 'selected prefill cohort must close first')
    previous = committed(RESULT) if RESULT.exists() else None
    cpu_check = mode == 'cpu-check'
    if cpu_check:
        guards.require(previous and previous['complete'] and previous['passed'] and previous['ownership_closed'] and
            previous.get('independent_process_summary', {}).get('original_benchmark_inputs_recovered') == 114,
            'close and summarize the original long-generation baseline first')
    if mode == 'clean':
        guards.require(previous and previous['complete'] and previous['passed'] and
            previous['ownership_closed'] and not previous['active'] and
            previous['mode'] == ('check' if cohort == 0 else 'clean') and
            previous['cohort'] == (0 if cohort == 0 else cohort - 1),
            'a completed eligible check/preceding clean process is required')
    build, build_receipts = common.committed_engine_build()
    if cpu_check:
        delta, proof = cpu_check_qualification(build)
    else:
        guards.require(build['binary'] == reference['binary'], 'selected engine changed')
        component = committed(A / 'moe-expert-bank-current.json')
        proof = common.expert_bank_build_proof(component['gpu'], build)
    committed(SETTINGS)
    qualifications = {str(SETTINGS): common.digest(SETTINGS)}
    if cpu_check:
        qualifications[str(RESEARCH)] = common.digest(RESEARCH)
    for key in ('ordered_exp_qualification', 'gu_qualification', 'mixed_qualification',
                'expert_bank_qualification', 'down_qualification'):
        item = reference[key]
        path = P / item['component_path']
        guards.require(common.digest(path) == item['component_sha256'], 'component qualification changed: ' + key)
        committed(path)
        qualifications[str(path)] = item['component_sha256']
    for name, expected in reference['source_view_qualification']['source_files_sha256'].items():
        guards.require(common.digest(P / name) == expected, 'reviewed source-view implementation changed')
    guards.require(not root.exists(), 'do not overwrite/retry an existing process root')
    root.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (root / name).mkdir()
    profile = P / 'data/expert-profile.bin'
    guards.require(common.digest(profile) == reference['profile_sha256'], 'selected expert profile changed')
    shutil.copyfile(profile, root / 'expert-profile.bin')
    old_root = reference['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old_root, str(root)) for k, v in reference['environment'].items()}
    for key in list(env):
        if key.startswith('UR_LOG_') or key in ('UR_ENABLE_LAYERS', 'STRATA_TRACE'):
            del env[key]
    if mode in ('check', 'cpu-check'):
        env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', STRATA_TRACE='1')
        env.update({k: v for k, v in json.loads(SETTINGS.read_text())['environment'].items()
                    if k.startswith('UR_LOG_') and k != 'UR_LOG_TRACING'})
    if cpu_check:
        env.update(STRATA_DECODE_CPU_PROFILE='1', STRATA_DECODE_TIMING='1')
    guards.require(all(env.get(k) == v for k, v in {
        'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD': '1', 'NEOReadDebugKeys': '1',
        'EnableDirectSubmission': '0', 'STRATA_SYCL_PREFILL_SAME_QUEUE': '1'}.items()),
        'preserve the qualified runtime safety settings')
    config = dict(reference['config'])
    flags = list(config['args'])
    flags[flags.index('--expert-profile') + 1] = str(root / 'expert-profile.bin')
    config.update(exe=build['binary']['path'], args=flags, log=str(root / 'engine.stderr'), env={})
    config_path = root / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = ['/bin/sh', str(P / 'bench.sh'), '--config', str(config_path), '--port', str(port),
        '--long-generation-only', '--think', '--runs', '1', '--workload-id', WORKLOAD_ID,
        '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')]
    revision = git('rev-parse', 'HEAD').decode().strip()
    pins = benchmark_pins(revision)
    if mode == 'clean':
        guards.require(hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest() ==
            previous['source_aggregate_sha256'], 'freeze benchmark sources across check and clean processes')
    plan = dict(schema=1, mode=mode, cohort=cohort, source_commit=revision,
        source_count=len(pins), source_aggregate_sha256=hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest(),
        binary=build['binary'], build_source_commit=build['source_commit'],
        engine_build_receipts=build_receipts, qualification_files=qualifications,
        bank_proof_live_inputs=proof['live_inputs'], config=config, environment=env, command=command,
        config_sha256=common.digest(config_path), profile_sha256=reference['profile_sha256'],
        prefill_reference=dict(commit=revision, path=str(PREFILL.relative_to(P)), sha256=common.digest(PREFILL)),
        preceding_generation_result=None if previous is None else dict(
            commit=revision, path=str(RESULT.relative_to(P)), sha256=common.digest(RESULT)),
        budgets=dict(wall=5400, cpu=54000, rss_cap=96 << 30, text_cap=64 << 20,
                     file_cap=64 << 20, total_cap=64 << 20),
        complete=False, passed=False, active=False, performance_adopted=False,
        sustained_decode70_qualified=False, physical262144_RESTORE_qualified=False, model_quality_qualified=False,
        purpose='First validated long-generation screen, then three independent clean processes; no optimization comparison or bottleneck claim.',
        prospective_gates='Each of both workload rows must have >=32768 fresh actual input/read tokens, >=1024 actual generated tokens including thinking, >=512 answer characters, engine clocks, normal SSE and >=256 offered drafts. Preserve all failures. Three eligible clean independent processes are needed to assess throughput/spread and observed strong>=60%/weak<=40% acceptance. A single process or traced check cannot qualify70tok/s.',
        limits='Current decode placement, spec4, suffix0 and safety settings unchanged. Standard client warmup/calibration precedes regular-code then varied-prose; there are no repeated same-workload warm samples within a process. Requested payload/call counts are not physical traffic. Physical full-context, quality, agent-session and RESTORE gates remain open.')
    if cpu_check:
        plan.update(runtime_asset_delta=delta,
            same_build_bank_qualification={k: v for k, v in proof.items() if k != 'live_inputs'},
            purpose='First parameter-validated official long-generation CPU observation on the proposed runtime assets. Both original strong/weak workloads and all six successful native requests must reconcile. No clean speed or resource bottleneck attribution before matched observer-overhead evidence.',
            prospective_gates=plan['prospective_gates'] + ' Exactly six valid native CPU records (four warmup/calibration and both long requests), no invalid clocks/counters, with matched input/output counts through final wait_commit. Preserve every failed/partial request. Runtime safety settings and qualified baseline hashes remain unchanged; only explicit proposed assets are admitted for this diagnostic.',
            limits=plan['limits'] + ' Parameter validation, stage logging, CPU clocks and wall observers enabled: this is not clean throughput or observer-overhead evidence. RAM/PCIe bytes, GPU service, physical full context and broad quality remain unqualified. Previous clean samples belong to the older runtime and do not isolate observer overhead.')
    (root / 'plan.json').write_text(json.dumps(dict(plan, source_sha256=pins), indent=2) + '\n')
    RESULT.write_text(json.dumps(plan, indent=2) + '\n')
    print('Prepared ' + str(RESULT.relative_to(P)) + '; commit it before run.')


def run(mode, cohort, root):
    record = committed(RESULT)
    guards.require(record['mode'] == mode and record['cohort'] == cohort and not record['complete'], 'commit a fresh matching plan')
    plan = json.loads((root / 'plan.json').read_text())
    guards.require({k: v for k, v in plan.items() if k != 'source_sha256'} == record, 'external plan differs')
    guards.require(benchmark_pins(record['source_commit']) == plan['source_sha256'], 'benchmark inputs changed')
    for files in (plan['engine_build_receipts'], plan['qualification_files'], plan['bank_proof_live_inputs']):
        common.recorded_files_unchanged(files)
    guards.require(common.digest(root / 'config.json') == plan['config_sha256'] and
        common.digest(root / 'expert-profile.bin') == plan['profile_sha256'], 'generated inputs changed')
    build = json.loads((B / 'main-native-gdn-engine/record.json').read_text())
    for source in [build['project'], *build['dependencies']]:
        for name, expected in source['source_sha256'].items():
            guards.require(common.digest(Path(source['root']) / name) == expected, 'native source changed: ' + name)
    runtime_assets_unchanged(record)
    guards.require(common.digest(plan['binary']['path']) == plan['binary']['sha256'], 'native executable changed')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
    cursor = next(x.split(': ', 1)[1] for x in before.splitlines() if x.startswith('-- cursor: '))
    owner = None

    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (root / 'record.json').write_text(json.dumps(dict(record,
            owned_commands=owner.commands if owner else []), indent=2) + '\n')

    owner = supervision.Owner(root / 'owned', persist)
    try:
        entry, _, _ = owner.run('bench', plan['command'], plan['environment'], P, **plan['budgets'])
        record.update(complete=supervision.closed(entry), exit_code=entry.get('exit_code'))
        results = list((root / 'bench').glob('*/results.json'))
        guards.require(len(results) == 1, 'missing official benchmark result; inspect the verified request journal')
        suite = json.loads(results[0].read_text())
        generation = suite['results'].get('long_generation', {})
        record['requests'] = collect_samples(generation)
        record['client_classification'] = generation.get('classification')
        record['client_failures'] = suite['results'].get('failures', [])
        journal, journal_rows = journal_evidence(root)
        record['request_journal'] = journal
        guards.require(journal['valid'] and journal['complete'], 'official request journal is incomplete or invalid')
        guards.require([r['request_sha256'] for r in journal_rows] ==
            [r['request_sha256'] for r in record['requests']], 'journal and official result requests differ')
        end = json.loads((root / 'engine-exit.json').read_text())
        record.update(engine_exit_code=end['exit_code'], native_identity=end['native'])
        require_complete_generation(generation, record['requests'])
        guards.require(entry.get('exit_code') == end['exit_code'] == 0, 'benchmark/native process failed')
        log = (root / 'engine.stderr').read_text(errors='replace')
        guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', log), 'runtime failure')
        if mode == 'cpu-check':
            observations, counts = cpu_observations(root, log, record['requests'])
            record.update(cpu_observations=observations, cpu_record_reconciliation=counts,
                          observer_overhead_qualified=False, hardware_bottleneck_qualified=False)
        runtime_assets_unchanged(record)
        record['passed'] = True
    except BaseException as error:
        record['error'] = str(error)
        try:
            journal, rows = journal_evidence(root)
            record['request_journal'] = journal
            if not record.get('requests'):
                record['requests'] = [dict(r, controller_eligible=sample_eligible(r)) for r in rows]
        except Exception as journal_error:
            record['journal_recovery_error'] = str(journal_error)
        if (root / 'engine.stderr').exists():
            record['error_excerpt'] = (root / 'engine.stderr').read_text(errors='replace')[-8192:]
    finally:
        record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True)
        record['fresh_faults'] = [x for x in after.splitlines() if re.search(
            r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', x)]
        record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        try:
            record['driver_after'] = guards.driver_guard()
        except Exception as error:
            record['driver_guard_error'] = str(error)
            record['passed'] = False
        record['passed'] = bool(record['passed'] and record['ownership_closed'] and
            record['boot_unchanged'] and not record['fresh_faults'])
        persist()
        RESULT.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record.get(k) for k in ('mode', 'cohort', 'complete', 'passed', 'ownership_closed', 'error')}))
    return 0 if record['passed'] else 1


def summarize():
    """Use all three original independent clean processes; no GPU execution."""
    current = committed(RESULT)
    guards.require(current['mode'] == 'clean' and current['cohort'] == 2 and
        'independent_process_summary' not in current, 'commit the final clean process before summary')
    head = git('rev-parse', 'HEAD').decode().strip()
    path = str(RESULT.relative_to(P))
    revision = git('log', '-1', '--format=%H', '--', path).decode().strip()
    records = []
    for _ in range(4):
        subprocess.run(['git', 'merge-base', '--is-ancestor', revision, head], cwd=P, check=True)
        record = json.loads(git('show', revision + ':' + path))
        guards.require(record['complete'] and record['passed'] and record['ownership_closed'] and
            not record['active'] and record['exit_code'] == record['engine_exit_code'] == 0 and
            not record['fresh_faults'] and record['boot_unchanged'] and
            record['request_journal']['valid'] and record['request_journal']['complete'],
            'failed, incomplete or faulted generation process')
        records.append((revision, record))
        if record['preceding_generation_result'] is not None:
            revision = record['preceding_generation_result']['commit']
    records.reverse()
    guards.require([(r['mode'], r['cohort']) for _, r in records] ==
        [('check', 0), ('clean', 0), ('clean', 1), ('clean', 2)], 'wrong generation process order')
    guards.require(len({(r['native_identity']['pid'], r['native_identity']['start_ticks'])
        for _, r in records}) == 4, 'independent native processes required')
    common_inputs, source_set = None, None
    process_rows = []
    for receipt_commit, record in records:
        root = record['environment']['STRATA_MEASUREMENT_ROOT']

        def canonical(value):
            if isinstance(value, str):
                return value.replace(root, '$RUN')
            if isinstance(value, list):
                return [canonical(v) for v in value]
            if isinstance(value, dict):
                return {k: canonical(v) for k, v in value.items() if
                        not k.startswith('UR_LOG_') and k not in ('UR_ENABLE_LAYERS', 'STRATA_TRACE')}
            return value

        pins = benchmark_pins(record['source_commit'])
        guards.require(len(pins) == record['source_count'] and
            hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest() ==
            record['source_aggregate_sha256'], 'original benchmark source cannot be recovered')
        source_set = pins if source_set is None else source_set
        guards.require(pins == source_set, 'generation benchmark sources changed')
        command = canonical(record['command'])
        command[command.index('--port') + 1] = '$PORT'
        identity = dict(binary=record['binary'], build_source=record['build_source_commit'],
            config=canonical(record['config']), environment=canonical(record['environment']),
            profile=record['profile_sha256'], command=command,
            request_hashes=[r['request_sha256'] for r in record['requests']])
        common_inputs = identity if common_inputs is None else common_inputs
        guards.require(identity == common_inputs and len(record['requests']) == 2 and
            all(sample_eligible(r) for r in record['requests']), 'unmatched or ineligible generation inputs')
        process_rows.append(dict(mode=record['mode'], cohort=record['cohort'],
            receipt_commit=receipt_commit, source_commit=record['source_commit'],
            samples=[{k: r[k] for k in ('workload', 'generated_tokens', 'decode_tps', 'decode_ms',
                'total_s', 'drafts_offered', 'drafts_accepted', 'acceptance_rate')} for r in record['requests']]))
    workloads = {}
    for name, strong in (('regular-code', True), ('varied-prose', False)):
        rows = [next(r for r in record['requests'] if r['workload'] == name) for _, record in records[1:]]
        speeds = [r['decode_tps'] for r in rows]
        rates = [r['acceptance_rate'] for r in rows]
        workloads[name] = dict(independent_process_tps=speeds, median_tps=statistics.median(speeds),
            sample_sd_tps=statistics.stdev(speeds), acceptance_rates=rates,
            proposed_regime_confirmed=all(r >= .60 if strong else r <= .40 for r in rates),
            decode70_speed_target_passed=all(t >= 70 for t in speeds))
    current['independent_process_summary'] = dict(analyzer_source_commit=head, processes=process_rows,
        workloads=workloads, strong_weak_pair_confirmed=all(w['proposed_regime_confirmed'] for w in workloads.values()),
        decode70_speed_target_passed=all(w['decode70_speed_target_passed'] for w in workloads.values()),
        check_excluded_from_clean_statistics=True, original_benchmark_inputs_recovered=len(source_set),
        GPU_executed=False, original_outcomes_unchanged=True,
        limits='Three independent clean processes, one request per workload in each after standard client warmup/calibration; not three subsequent same-workload requests. No MTP speedup attribution without a matched no-MTP arm. Broad quality,agent-session,physical262144 and RESTORE remain separate unqualified gates.')
    RESULT.write_text(json.dumps(current, indent=2) + '\n')
    print(json.dumps(current['independent_process_summary']))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'summarize'))
    parser.add_argument('mode', choices=('check', 'clean', 'cpu-check'), nargs='?', default='check')
    parser.add_argument('cohort', type=int, choices=(0, 1, 2), nargs='?', default=0)
    args = parser.parse_args()
    guards.require(args.mode == 'clean' or args.cohort == 0, 'only one initial validation process')
    root = B / (('decode-long-generation-v2-' if args.mode == 'cpu-check' else 'decode-long-generation-v1-') +
                args.mode + '-' + str(args.cohort))
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        if args.action == 'summarize':
            return summarize()
        guards.driver_guard()
        if args.action == 'prepare':
            prepare(args.mode, args.cohort, root)
            return 0
        return run(args.mode, args.cohort, root)


if __name__ == '__main__':
    sys.exit(main())
