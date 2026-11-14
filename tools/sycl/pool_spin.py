#!/usr/bin/env python3
"""Versioned prose pilot and same-engine cache comparison; repository LICENSE.

Main-authored from the fork's owned benchmark supervision. The completed timing/
cache diagnostic is recoverable at49170c397; its raw logs have no live consumer.
Native safety settings and the fixed ELF remain. A failed v4 run is preserved;
the new task/gates are qualified prospectively before repeated comparison.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import select
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import mtp_prefill as generation

model, comparison, shared, build = generation.model, generation.comparison, generation.shared, generation.build
P, B, RESULT = generation.P, generation.B, generation.RESULT
PILOT_ROOT = model.adaptive.ROOT / 'cache-generation-speed-v2-pilot'
PILOT_KEY = 'adaptive_cache_generation_speed_v2_pilot'
FAILED_ROOT = model.adaptive.ROOT / 'cache-generation-speed-v1'
FAILED_KEY = 'adaptive_cache_generation_speed_v1'
DIAGNOSTIC_ROOT = model.adaptive.ROOT / 'cache-generation-v1'
DIAGNOSTIC_KEY = 'adaptive_cache_generation_v1'
SOURCE = 'tools/sycl/pool_spin.py'
CHANGED = {SOURCE, 'tools/long_generation_bench.py', 'tools/strata_bench.py',
           'tools/test_strata_bench.py'}
WORKLOAD_ID = 'cache_generation_v5_20261102'
MODE, ROOT, KEY = 'pilot', PILOT_ROOT, PILOT_KEY
WORKLOADS, ARMS = ('varied-prose',), (('reference', 0),)
OBSERVERS = ('STRATA_TRACE', 'STRATA_PREFILL_CPU_WAIT_PROFILE', 'STRATA_DECODE_TIMING',
             'STRATA_TRACE_ADAPT', 'STRATA_DRAFT_TIMING', 'STRATA_SYCL_TEST_ADAPT_RECORD_FAILURE')
require, digest, write, compact = generation.require, generation.digest, generation.write, generation.compact


def select_mode(mode):
    global MODE, ROOT, KEY, WORKLOADS, ARMS
    require(mode in ('pilot', 'cohort'), 'pilot or cohort required')
    MODE = mode
    ROOT = PILOT_ROOT if mode == 'pilot' else model.adaptive.ROOT / 'cache-generation-speed-v2'
    KEY = PILOT_KEY if mode == 'pilot' else 'adaptive_cache_generation_speed_v2'
    WORKLOADS = ('varied-prose',) if mode == 'pilot' else ('regular-code', 'varied-prose')
    ARMS = (('reference', 0),) if mode == 'pilot' else (
        ('reference', 0), ('candidate', 8), ('candidate', 8),
        ('reference', 0), ('reference', 0), ('candidate', 8))


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def basis(revision):
    model.configure('adaptive-positive-throughput')
    native, _, providers = model.proof()
    prior = json.loads((model.ROOT / 'record.json').read_text())
    data = build.committed(RESULT)
    require(data[model.KEY] == model.compact(prior)
            and prior['passed'] and prior['complete'] and prior['stage'] == 'closed'
            and prior['ownership_closed'] and not prior['active']
            and prior['binary'] == native['binaries']['strata']
            and prior['native_source_commit'] == native['source_commit']
            and prior['outcome']['native_exit_code'] == 0 and not prior['outcome']['fresh_faults']
            and prior['outcome']['adaptive_work']['deferred_next_long_prefill_boundaries'] > 0,
            'closed fixed-engine positive-swap next-GEN qualification required')
    _, manifest = model.python_manifest()
    recovered = {n: hashlib.sha256(raw).hexdigest()
                 for n, raw in shared.gen.blobs(P, prior['source_commit'], prior['sources'])}
    current = model.pinned_sources(revision)
    require(recovered == prior['sources'] and set(current) == set(recovered)
            and {n for n in current if current[n] != recovered[n]} == CHANGED,
            'only the declared controller,versioned workload and host evidence/test changes are allowed')
    diagnostic = json.loads((DIAGNOSTIC_ROOT / 'record.json').read_text())
    require(data[DIAGNOSTIC_KEY] == compact(diagnostic)
            and diagnostic['passed'] and diagnostic['complete'] and diagnostic['ownership_closed']
            and not diagnostic['active'] and diagnostic['binary'] == prior['binary']
            and len(diagnostic['processes']) == 2
            and all(p['passed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
                    for p in diagnostic['processes']), 'closed same-ELF cache diagnostic required')
    original = {n: hashlib.sha256(raw).hexdigest()
                for n, raw in shared.gen.blobs(P, diagnostic['source_commit'], diagnostic['sources'])}
    require(original == diagnostic['sources'] and set(current) == set(original)
            and {n for n in current if current[n] != original[n]} == CHANGED,
            'undeclared delta from the closed diagnostic')
    failed = json.loads((FAILED_ROOT / 'record.json').read_text())
    require(data[FAILED_KEY] == compact(failed) and failed['stage'] == 'failed'
            and not failed['passed'] and not failed['active'] and failed['ownership_closed']
            and failed['binary'] == prior['binary'] and len(failed['processes']) == 1
            and failed['failure_diagnosis']['observed_native_exit_code'] == 0
            and failed['failure_diagnosis']['observed_benchmark_exit_code'] == 1,
            'preserved closed failed V1 required;no retry or retroactive qualification')
    if MODE == 'cohort':
        pilot = json.loads((PILOT_ROOT / 'record.json').read_text())
        require(data[PILOT_KEY] == compact(pilot) and pilot['passed'] and pilot['complete']
                and pilot['ownership_closed'] and not pilot['active'] and pilot['stage'] == 'closed'
                and pilot['binary'] == prior['binary'] and pilot['sources'] == current
                and pilot['decision'] == pilot_decision(pilot['processes'])
                and pilot['decision']['cohort_permitted'], 'qualified v5 weak-prose pilot required')
    names = ('src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/pool.hpp')
    pool = {n: native['native_sources'][n] for n in names}
    require(all(digest(model.adaptive.ROOT / 'source' / n) == sha for n, sha in pool.items()),
            'actual compiled pool inputs differ')
    text = (model.adaptive.ROOT / 'source' / names[0]).read_text()
    require('std::getenv("STRATA_POOL_SPIN_US")' in text
            and 'spin_before_sleep_ = std::chrono::microseconds' in text,
            'qualified binary source lacks immutable startup knob')
    return prior, providers, manifest, pool


def environment(row, plan):
    require((plan['kernel'], plan['adapt_swaps']) == ARMS[plan['index']] and plan['heads'] == 1,
            'unplanned adaptive-cache arm')
    env = comparison.environment(row, plan)
    for name in OBSERVERS:
        env.pop(name, None)
    env.update(STRATA_SYCL_PREFILL_GDN_BARRIERS3='1', STRATA_SYCL_PREFILL_GDN_FP16='1',
               STRATA_SYCL_PREFILL_GDN_CHUNK64='0', STRATA_MTP_BATCH_F16='1',
               STRATA_SNAPSHOT_VERIFY='1', STRATA_POOL_SPIN_US='20000')
    require(not any(k in env for k in OBSERVERS) and not env.get('LD_PRELOAD')
            and env['CUDA_MODULE_LOADING'] == 'EAGER'
            and env['UR_ENABLE_LAYERS'] == 'UR_LAYER_PARAMETER_VALIDATION',
            'clean timing/private activation environment differs')
    return env


def invariant(row, plan):
    value = dict(config=json.loads((Path(plan['root']) / 'config.json').read_text()),
                 environment=environment(row, plan))
    args = value['config']['args']
    require(args[args.index('--adapt-swaps') + 1] == str(plan['adapt_swaps'])
            and args[args.index('--adapt-every') + 1] == '1', 'cache configuration differs')
    args[args.index('--adapt-swaps') + 1] = '<arm>'
    return hashlib.sha256(json.dumps(value, sort_keys=True).replace(plan['root'], '<process>').encode()).hexdigest()


def current(row):
    require(build.committed(RESULT)[KEY] == compact(row), 'commit exact repeated comparison state')
    prior, files, manifest, pool = basis(row['source_commit'])
    require(row['binary'] == prior['binary'] and row['native_source_commit'] == prior['native_source_commit']
            and row['native_validation_probe'] == prior['native_validation_probe']
            and row['model_assets'] == prior['model_assets'] and row['compiled_pool_sources'] == pool
            and model.pinned_sources(row['source_commit']) == row['sources']
            and row['plan']['mode'] == MODE, 'frozen source/build/mode differs')
    inv = row['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'] and json.loads(Path(inv['path']).read_text()) == files,
            'runtime inventory differs')
    require(digest(row['template']['path']) == row['template']['sha256']
            and build.guards.driver_guard() == row['driver'], 'template/driver differs')
    require(build.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole output budget')
    require(all(invariant(row, p) == row['invariant_signature']
                and comparison.fingerprint(environment(row, p)) == p['environment_sha256']
                and all(digest(Path(p['root']) / n) == sha for n, sha in p['generated'].items())
                for p in row['plans']), 'generated arm inputs differ')
    return files, manifest


def route(text, read):
    """Actual ordinary long route, without requiring disabled trace-only clocks."""
    require(type(read) is int and 32768 <= read <= 65537, 'unplanned ordinary extent')
    n = read - 1
    def fields(prefix):
        lines = [v for v in text.splitlines() if v.startswith(prefix)]
        require(len(lines) == 1, 'missing/duplicate native route: ' + prefix)
        pairs = re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)', lines[0].split(';', 1)[0])
        require(len({k for k, _ in pairs}) == len(pairs), 'duplicate native field')
        return dict(pairs)
    main = fields('strata prefill: layer-major candidate, ')
    window = fields('strata MoE window configuration: ')
    work = fields('strata MoE accumulator window: ')
    split = fields('strata MoE split admission: ')
    bank = fields('strata MoE expert bank: ')
    end = fields('strata MoE expert bank complete: ')
    down = fields('strata MoE Down fusion: ')
    require(main['tokens'] == main['residual_gpu_rows'] == str(n) and main['chunk'] == '8192'
            and main['residual_dtype'] == 'bf16' and main['residual_element_bytes'] == '2'
            and int(main['residual_total']) == int(main['residual_gpu']) == n * 20480
            and main['residual_host'] == main['residual_staging'] == '0'
            and window['requested_mode'] == window['effective_mode'] == '10'
            and window['maximum_rows'] == work['maximum_rows'] == str(n) and work['windows'] == '48'
            and work['operator_max'] == '8192' and work['GPU_resident_windows_only'] == '1'
            and int(work['jobs']) >= 48 and int(work['dequants']) > 0 and int(work['source_H2D_bytes']) >= 0
            and split['active'] == '0' and split['rows'] == str(n) and split['decode_changed'] == '0'
            and split['primary_rows'] == split['tail_payload_bytes'] == split['extra_allocation_bytes'] == '0',
            'ordinary GPU residual/window/split extent differs')
    require(bank['requested_slots'] == bank['effective_slots'] == end['total_slots'] == split['prefill_slots'] == '384'
            and bank['staged'] == '1' and bank['residual_gpu_rows'] == str(n)
            and bank['general_map'] == 'private_scratch_never_published'
            and int(bank['device_bytes']) == int(main['layer_cache']) > 0
            and int(bank['allocation_limit_bytes']) >= int(bank['device_bytes'])
            and 0 <= int(end['permanent_slots']) <= 384 and int(end['uploads']) >= 0
            and int(end['requested_H2D_bytes']) >= 0 and end['final_drain'] == 'checked'
            and int(down['fused_jobs']) > 0 and down['min_rows'] == '2334'
            and int(down['fused_rows']) >= 2334 * int(down['fused_jobs'])
            and int(down['fused_jobs']) + int(down['separate_jobs']) == int(work['jobs']),
            'ordinary bank/Down/checked drain differs')
    generation.gdn.native16_route(text, read)
    generation.gdn.gdn_selection(text, 'candidate')
    target = fields('strata prefill heads: target detached ')
    draft = fields('strata prefill heads: draft detached ')
    require(int(target['bytes']) == int(target['D2H']) == int(target['H2D-on-restore']) == comparison.TARGET_BYTES
            and target['source'] == 'owned-RAM' and int(draft['bytes']) == comparison.DRAFT_BYTES
            and int(draft['retained-map-bytes']) == comparison.MAP_BYTES
            and text.index('strata prefill heads: target detached ') < text.index('strata prefill heads: draft detached ')
            < text.index('strata MoE expert bank: ') < text.index('strata prefill: layer-major candidate, '),
            'head/placement contract differs')
    return dict(read_tokens=read, rows=n, GPU_residual_rows=n, ordinary_staged_slots=384, split=False,
                GDN_barriers=3, windows=48, jobs=int(work['jobs']),
                MTP_prefill=model.mtp_route(text, n),
                dispatch_proof='Observed native16/GDN3 headers and MTP mode2 callbacks with frozen native branch,immutable GDN16/MTP16 selectors. Typed GDN per-launch counts and dense phase logs stay disabled;prior traced same-ELF check supplies that bounded qualification.',
                physical_resource_saturation_qualified=False)


def cache_footer(text):
    hits = re.findall(r'^strata serve: decode expert cache hit rate: ([0-9.]+)% \((\d+) hits / (\d+) lookups\)$', text, re.M)
    require(len(hits) == 1, 'missing/duplicate common cache footer')
    percent, count, lookups = float(hits[0][0]), int(hits[0][1]), int(hits[0][2])
    require(0 <= count <= lookups and lookups > 0 and math.isfinite(percent)
            and abs(percent - 100 * count / lookups) <= .051, 'cache count/percent differs')
    return dict(hits=count, lookups=lookups, hit_percent=percent)


def cache_history(text, expected):
    require(len(re.findall(r'^strata generate: native pack: .+$', text, re.M)) == 1
            and len(re.findall(r'^strata generate: expert arena: .+$', text, re.M)) == 1
            and 'strata generate: experts via mmap ' not in text
            and 'strata serve: resident RAM:' not in text,
            'native ArenaExpertSource startup/history differs')
    lines = [v for v in text.splitlines() if v.startswith('strata serve: decode expert cache hit rate:')]
    require(len(lines) == expected, 'common cache history count differs before continue')
    return [cache_footer(v) for v in lines]


def cache_boundary(root, expected):
    # Native DONE is flushed before its stderr footer. Wait only at the paused
    # frontend gate,outside the reported request clocks,and bind offsets to the
    # same complete byte snapshot used by the parser.
    deadline = time.monotonic() + 5
    while True:
        raw = (root / 'engine.stderr').read_bytes()
        text = raw.decode()
        complete = sum(v.startswith('strata serve: decode expert cache hit rate:')
                       and v.endswith('\n') for v in text.splitlines(keepends=True))
        require(complete <= expected, 'extra common cache footer at request boundary')
        if complete == expected:
            return raw, cache_history(text, expected)
        require(time.monotonic() < deadline, 'common cache footer deadline before continue')
        time.sleep(.01)


def clean_observers(text):
    require(not any(v in text for v in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called',
        'teardown uncertain', 'strata trace:', 'strata MoE progress:', 'strata prefill dense:',
        'strata decode timing:', 'strata prefill cpu ', 'strata serve: ADAPT_', 'strata GDN dispatch:',
        'strata GDN storage:')), 'runtime error or disabled observer appeared')


def prepare():
    require(not ROOT.exists() and KEY not in build.committed(RESULT), 'fresh repeated comparison only')
    revision = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    prior, providers, manifest, pool = basis(revision)
    assets = shared.models.model_pins(prior['configuration'], prior['model_assets'], full=True)
    ROOT.mkdir(mode=0o700)
    write(ROOT / 'runtime-inputs.json', providers)
    row = {k: copy.deepcopy(prior[k]) for k in ('binary', 'native_source_commit', 'native_validation_probe',
        'configuration', 'template', 'boot', 'driver', 'python_contract', 'python_selected')}
    row['environment_template'] = copy.deepcopy(prior['environment'])
    row.update(stage='prepared', source_commit=revision, sources=model.pinned_sources(revision), model_assets=assets,
        compiled_pool_sources=pool, runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'),
            sha256=digest(ROOT / 'runtime-inputs.json'), files=len(providers)),
        plans=[], processes=[], host_passed=False, passed=False, complete=False, active=False, ownership_closed=True,
        controls=selfcheck(), whole_bytes=(96 if MODE == 'pilot' else 448) << 20,
        budgets=dict(wall=2700, cpu=16200, rss_cap=104 << 30, text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20),
        qualification_reuse=dict(model_result_key=model.KEY, model_source_commit=prior['source_commit'],
            original_source_files=len(prior['sources']), host_only_delta=sorted(CHANGED),
            diagnostic_result_key=DIAGNOSTIC_KEY,
            scope='Fixed nine ELFs/source/recipe,positive next-GEN and injected-record failure qualification retained. Original265-file inputs recovered;only declared host source/test changes. Prior v4 workload and failed V1 are not new generation evidence.'),
        plan=dict(mode=MODE, workload_version=5,
            question=('Does the simpler v5 long prose produce thought plus >=512answer characters,>=1Kactual tokens and observed acceptance<=.40 in a fresh reference process?'
                if MODE == 'pilot' else 'Does8swaps/every1 reproducibly improve sustained v5 code and prose generation over0swaps in the same fixed engine?'),
            motivation='V1failed:16Kthinking-only prose,acceptance.59216. Keep the answer and MTP bands;change the task prospectively. Prior diagnostic cache8/0 had15.427/13.530TG,one pair,not clean/repeated evidence or saturation.',
            order=('One fresh reference process,four official warmups,then varied-prose only. This checks task feasibility,not the later after-code history or repeated weak class.'
                if MODE == 'pilot' else 'Six fresh processes R/C,C/R,R/C,three adjacent pairs. Each has four official warmups,then regular-code and varied-prose once each. Same fixed v5 bodies across processes;code first,prose follows. Adaptation stays enabled during all candidate warmups/requests;resident membership may differ.'),
            constants='Same ELF/providers/safety/preload,throughput BF16 HC/GDN16/MTP16,heads1,131072INT8resident,1256cache slots,5workers,spec4,chunk8192,no token graph,caches0,20ms spin,adapt-every1. Only --adapt-swaps0/8 differs;no native build or runtime environment mutation.',
            observers='Parameter validation/private retained activation preload and nominal MTP callback markers only,matching the prior clean117K envelope. Disable STRATA_TRACE,CPU attribution,DECODE_TIMING,ADAPTTRACE and draft timing. Identity/provider/clock checks and a bounded5s stderr-footer wait occur at paused request gates outside reported clocks;no profiler or periodic observer.',
            acceptance=f'All{len(ARMS)}processes/{len(ARMS)*len(WORKLOADS)}long requests must satisfy source/build/assets/provider/body/route/journal/clock/owner gates,>=32768fresh read,>=1024generated including thinking,>=512answer characters,>=256drafts,normal0,quiet validation,no fresh fault. Actual mode2 callbacks/nominal rows,ordinary BF16/GDN3 bank384 and checked drain must agree. Check native-Arena/common hit/lookups after all4warmups and each long request before continue. This source has no resident-RAM exchange footer;adaptive counts,payload/wire bytes and exact cache membership remain unobserved in clean timing. No exact-output or task-quality claim.',
            decision=('Permit the fresh repeated cohort only if this eligible prose has observed acceptance<=.40. Otherwise preserve its failed regime decision and choose a new bounded plan;no retries or relabeling. One pilot does not establish a repeated weak class or after-code behavior.'
                if MODE == 'pilot' else 'Retain all3paired exact TG ratios/geomean,process median/min/max,acceptance and lengths per workload. Reproducible gain requires all3candidate/referenceTG>1;no minimum floor. Code strong only if every sample>=.60,prose weak only if every sample<=.40,per arm. Classification failure does not stop collection. Judge prompt/TTFT separately;dedicated117K prefill,task/payload,agent,other exits,physical262144/RESTORE and adoption remain pending.'),
            stopping='Stop first source/owner/provider/clock/route/validation/fault/budget/sample failure. No retry,exclusion,speed stop or refreshed hashes. Commit each closed process before next;full model hashes before and after cohort.',
            retention='Keep compact individual samples,common cache counts,actual routes and decisions. Raw journals/logs/JIT retire after verification;keep record/runtime and generated plans only while admission consumes them. No new engine generation.'))
    for index, (arm, swaps) in enumerate(ARMS):
        root = ROOT / str(index)
        root.mkdir(mode=0o700)
        for part in ('tmp', 'cache'):
            (root / part).mkdir()
        config = copy.deepcopy(row['configuration'])
        config['log'] = str(root / 'engine.stderr')
        config['args'][config['args'].index('--adapt-swaps') + 1] = str(swaps)
        write(root / 'config.json', config)
        write(root / 'plan.json', dict(binary=row['binary'], native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        plan = dict(index=index, kernel=arm, adapt_swaps=swaps, heads=1, root=str(root),
            generated={n: digest(root / n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--long-generation-only', '--think', '--runs', '1', '--profile-request-gates',
                '--workload-id', WORKLOAD_ID, '--fail-fast-engine-error', '--ready-timeout', '600',
                '--out', str(root / 'bench')])
        if MODE == 'pilot':
            plan['command'] += ['--long-generation-workload', 'varied-prose']
        plan['environment_sha256'] = comparison.fingerprint(environment(row, plan))
        row['plans'].append(plan)
    row['invariant_signature'] = invariant(row, row['plans'][0])
    require(all(invariant(row, p) == row['invariant_signature'] for p in row['plans']), 'undeclared arm difference')
    save(row)
    print(f'Prepared {MODE}: {len(ARMS)}fresh processes/{len(ARMS)*len(WORKLOADS)}long requests;commit before frontend probes.', flush=True)


def generation_contract(suite, rows):
    # Keep the closed native/device controllers' imported v4 helper byte-exact.
    # This prospective v5 experiment checks its own version/order/selection;
    # the unchanged shared collector still checks every sample's eligibility.
    require(suite.get('workload_version') == 5 and suite.get('thinking') is True
            and suite.get('reasoning_budget_tokens') == 0 and suite.get('max_tokens') == 16384,
            'unexpected v5 task or thinking/output budget')
    require(len(rows) == len(WORKLOADS)
            and [(r.get('workload'), r.get('run'), r.get('order')) for r in rows]
                == [(name, 0, i) for i, name in enumerate(WORKLOADS)]
            and all(r.get('controller_eligible') is True for r in rows), 'incomplete/ineligible v5 generation')
    require(suite.get('diagnostic_subset') is (MODE == 'pilot')
            and suite.get('selected_workloads') == list(WORKLOADS), 'v5 mode/workload selection differs')


def official(root, result):
    suite = result['results']['long_generation']
    rows = shared.gen.collect_samples(suite)
    generation_contract(suite, rows)
    require(result['complete'] and not result['results']['failures'] and result['params']['runs'] == 1
            and result['params']['think'] is True and result['params']['long_generation_only'] is True,
            'incomplete official generation comparison')
    journal, journal_rows = shared.gen.journal_evidence(root)
    require(journal['valid'] and journal['complete']
            and [{k: v for k, v in r.items() if k not in ('controller_eligible', 'reasoning_tokens')} for r in rows] == journal_rows,
            'result/journal differ')
    path, = (root / 'bench').glob('*/requests.jsonl')
    events = [json.loads(v) for v in path.read_text().splitlines()]
    warm = shared.gen.warmup_reconciliation(events, WORKLOADS)
    identity, binding = events[0], result['comparison_identity']
    from strata_bench import client_fingerprint
    workload = dict(workload_id=WORKLOAD_ID, nonce=WORKLOAD_ID, nonce_policy='fixed-id')
    require(identity['mode'] == binding['mode'] == 'suite' and identity['requested_runs'] == 1
            and identity['thinking'] is binding['thinking'] is True
            and identity['source'] == binding['client'] == client_fingerprint()
            and identity['context'] == binding['context'] == result['machine']['context'] == 131072
            and identity['workload'] == binding['workload'] == result['params']['workload'] == workload
            and all(identity[k] == binding[k] == result['machine'][k] for k in ('model', 'quant')),
            'client/model/body binding differs')
    return rows, events, dict(warm, valid=True, journal_sha256=digest(path))


def pilot_decision(processes):
    require(len(processes) == 1 and processes[0]['passed'] and processes[0]['complete']
            and processes[0]['ownership_closed'] and processes[0]['kernel'] == 'reference'
            and processes[0]['adapt_swaps'] == 0 and len(processes[0]['samples']) == 1,
            'one closed eligible prose reference required')
    sample, = processes[0]['samples']
    require(sample['workload'] == 'varied-prose' and sample['controller_eligible'], 'eligible prose required')
    weak = sample['acceptance_rate'] <= .40
    return dict(task_sample_eligible=True, observed_acceptance=sample['acceptance_rate'],
        observed_weak_MTP=weak, cohort_permitted=weak, repeated_MTP_class_qualified=False,
        inference_adopted=False, physical_resource_saturation_qualified=False,
        limits='One fresh first-long-request pilot;no after-code history or repeated gain. Keep thinking/answer/length/MTP gates unchanged.')


def decision(processes):
    if MODE == 'pilot':
        return pilot_decision(processes)
    require(len(processes) == len(ARMS) and all(p['passed'] and p['complete'] and p['ownership_closed']
        and (p['kernel'], p['adapt_swaps']) == ARMS[i] for i, p in enumerate(processes)), 'complete frozen cohort required')
    require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ARMS),
            'native process reused')
    workloads = {}
    for i, name in enumerate(WORKLOADS):
        ratios, prompt, ttft = [], [], []
        arms = {arm: [] for arm in ('candidate', 'reference')}
        for first in range(0, len(ARMS), 2):
            pair = {p['kernel']: p['samples'][i] for p in processes[first:first+2]}
            a, b = pair['candidate'], pair['reference']
            exact = lambda s: s['generated_tokens'] * 1000 / s['decode_ms']
            ratios.append(exact(a) / exact(b))
            prompt.append(a['prompt_ms'] / b['prompt_ms'])
            ttft.append(a['ttft_s'] / b['ttft_s'])
            for arm, sample in pair.items():
                arms[arm].append(dict(exact_tg=exact(sample), acceptance=sample['acceptance_rate'],
                                     generated_tokens=sample['generated_tokens']))
        for arm, samples in arms.items():
            values = [s['exact_tg'] for s in samples]
            arms[arm] = dict(samples=samples, median_tg=statistics.median(values), min_tg=min(values), max_tg=max(values),
                observed_MTP_class=('strong' if all(s['acceptance'] >= .60 for s in samples) else 'unconfirmed')
                    if name == 'regular-code' else ('weak' if all(s['acceptance'] <= .40 for s in samples) else 'unconfirmed'))
        workloads[name] = dict(arms=arms, paired_candidate_over_reference_TG=ratios,
            geometric_mean_TG_ratio=math.exp(statistics.mean(math.log(v) for v in ratios)),
            reproducible_TG_gain=all(v > 1 for v in ratios),
            paired_prompt_time_ratios=prompt, paired_TTFT_ratios=ttft)
    return dict(repeated_comparison_complete=True, workloads=workloads, inference_adopted=False,
        dedicated_prefill_nonregression_qualified=False, physical_resource_saturation_qualified=False,
        next='Use the actual repeated strong/weak results to retain or reject the cache-policy candidate. Dedicated prefill/task/payload/agent/full256K and bounded GPU-service attribution remain separate.')


def run(index):
    row = json.loads((ROOT / 'record.json').read_text())
    files, manifest = current(row)
    require(row['host_passed'] and row['stage'] in ('ready', 'running') and len(row['processes']) == index < len(ARMS)
            and all(p['passed'] for p in row['processes']), 'fresh ordered process only;no retry')
    plan = row['plans'][index]
    root = Path(plan['root'])
    env = environment(row, plan)
    require(not (root / 'execution.json').exists() and not (root / 'launch.json').exists(), 'existing native attempt')
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
    outcome, owner = dict(index=index, kernel=plan['kernel'], adapt_swaps=plan['adapt_swaps'], complete=False,
                          passed=False, host_probe_accepted=False, model_executed=False), None
    windows, live, identity = [], [], None
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def persist():
        write(root / 'execution.json', dict(outcome, active=bool(owner and owner.active), commands=owner.commands if owner else []))
    def interact(io, entry):
        nonlocal identity
        for i, name in enumerate(WORKLOADS):
            offsets = {}
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(2400)
                    if line.startswith('STRATA_PROFILE_GATE '):
                        break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                        dict(stage=stage, phase='long-generation/' + name, run=0), 'request gate differs')
                delta = [a-b for a, b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
                require(min(delta) >= 0 and abs(delta[0] - delta[1]) <= 50000000, 'clock/suspend discontinuity')
                claim = json.loads((root / 'engine-start.json').read_text())['native']
                actual = build.supervision.identity(claim['pid'])
                require(actual and actual['start_ticks'] == claim['start_ticks'] and actual['sid'] == entry['pid']
                        and str(Path('/proc', str(claim['pid']), 'exe').resolve()) == row['binary']['path']
                        and (identity is None or identity == claim), 'native identity differs')
                identity = claim
                expected = model.engine_once.native_environment(row, env)
                actual_env = dict(v.split(b'=', 1) for v in Path('/proc', str(claim['pid']), 'environ').read_bytes().split(b'\0') if b'=' in v)
                require(all(actual_env.get(k.encode()) == v.encode() for k, v in expected.items())
                        and set(actual_env) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
                        'native environment differs')
                mapped = shared.runtime.executable_mappings(Path('/proc', str(claim['pid']), 'maps').read_text())
                require(mapped <= set(files) | {row['binary']['path']} and
                        {row['native_validation_probe']['loader']['path'], row['native_validation_probe']['library']['path'], env['ZE_ENABLE_ALT_DRIVERS']} <= mapped,
                        'native providers differ')
                shared.python_runtime.check_running(entry['pid'], P / 'tools/strata_bench.py', manifest['selected'], env,
                    {v['realpath'] for v in manifest['inventory']['files'].values()}, plan['command'][2:])
                comparison.require_quiet_validation(root)
                raw, history = cache_boundary(root, 4 + i + (stage == 'end'))
                offsets[stage] = len(raw)
                if stage == 'begin':
                    path, = (root / 'bench').glob('*/requests.jsonl')
                    events = [json.loads(v) for v in path.read_text().splitlines()]
                    require(len(events) == 10 + 2 * i and events[-1]['kind'] == 'start'
                            and events[-1]['phase'] == 'long-generation/' + name, 'official warmup/prior count differs')
                    shared.gen.warmup_reconciliation(events[:-1], WORKLOADS[:i])
                    prior = [e['sample'] for e in events[:-1] if e['kind'] == 'result']
                    require(all(shared.gen.sample_eligible(v) for v in prior[4:]), 'prior long sample ineligible')
                    shared.gen.request_evidence(raw.decode(), prior)
                else:
                    text = raw[offsets['begin']:offsets['end']].decode()
                    headers = re.findall(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$', text, re.M)
                    require(len(headers) == 1 and headers[0][1] == '0' and headers[0][0] == headers[0][2],
                            'native long footer differs')
                    clean_observers(text)
                    live.append(route(text, int(headers[0][2])))
                outcome.update(model_executed=True, native_identity=claim,
                    request_routes_checked_before_continue=len(live),
                    cache_footers_checked_before_continue=len(history))
                persist()
                print(f'Cache speed process{index} swaps={plan["adapt_swaps"]} {name} {stage}', flush=True)
                io.send(b'continue\n')
            windows.append(offsets)
        io.proc.stdin.close()
        deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            require(time.monotonic() < deadline, 'teardown deadline')
            if select.select([io.proc.stdout], [], [], .02)[0]:
                io.read()
                io.pending.clear()
        io.exit(1)
    try:
        owner = build.supervision.Owner(root / 'owned', persist)
        entry, _, _ = owner.run('bench', plan['command'], env, P, interaction=interact, **row['budgets'])
        outputs = list((root / 'bench').glob('*/results.json'))
        if outputs:
            outcome['samples'] = shared.gen.collect_samples(json.loads(outputs[0].read_text())['results'].get('long_generation', {}))
        require(build.supervision.closed(entry) and entry['exit_code'] == 0 and len(outputs) == 1,
                'owned benchmark failed: ' + entry.get('error', 'exit/result/closure gate'))
        samples, events, journal = official(root, json.loads(outputs[0].read_text()))
        for sample in samples:
            comparison.require_valid_client_ttft(sample)
        require(len(windows) == len(samples) == len(live) == len(WORKLOADS), 'long-generation request count')
        if row['processes']:
            require(journal['request_sha256'] == row['processes'][0]['journal']['request_sha256'], 'paired bodies differ')
        comparison.require_quiet_validation(root)
        end = shared.gen.native_end(root, row, dict(config=json.loads((root / 'config.json').read_text())), row['boot'], entry['pid'])
        raw = (root / 'engine.stderr').read_bytes()
        text = raw.decode()
        clean_observers(text)
        generation.gdn.validate_model_text(text, plan)
        binding = shared.gen.request_evidence(text, [e['sample'] for e in events if e['kind'] == 'result'])
        validation = shared.runtime.validation.preload_evidence(text, end['native']['pid'])
        require(validation['loader'] == row['native_validation_probe']['loader']['path'], 'validation loader differs')
        routes = [route(raw[w['begin']:w['end']].decode(), s['read_tokens']) for w, s in zip(windows, samples)]
        require(routes == live, 'live/final route differs')
        cache = [cache_footer(raw[w['begin']:w['end']].decode()) for w in windows]
        history = cache_history(text, 4 + len(samples))
        require(history[4:] == cache, 'long-window/common footer history differs')
        current(row)
        if index == len(ARMS) - 1:
            shared.models.model_pins(row['configuration'], row['model_assets'], full=True)
        fields = ('workload', 'read_tokens', 'reused_tokens', 'generated_tokens', 'output_chars', 'reasoning_chars',
            'prompt_ms', 'prompt_tps', 'decode_ms', 'decode_tps', 'ttft_s', 'total_s', 'drafts_offered', 'drafts_accepted',
            'acceptance_rate', 'request_sha256', 'output_sha256', 'reasoning_sha256', 'finish_reason', 'controller_eligible')
        outcome.update(complete=True, host_probe_accepted=True, native_exit_code=0, validation=validation,
            routes=routes, cache=cache, common_cache_footers=len(history), journal=journal,
            native_request_binding=dict(requests=len(binding), draft_cells=sum(v['draft_cells'] for v in binding)),
            samples=[{k: s[k] for k in fields if k in s} for s in samples])
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        if owner:
            shared.runtime.validation.finish(outcome, owner, cursor, row['boot'])
        if outcome.get('driver_after') != row['driver']:
            outcome.update(passed=False, final_driver_mismatch=True)
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'])
        row['processes'].append(outcome)
        done = index == len(ARMS) - 1
        row.update(stage='closed' if outcome['passed'] and done else 'running' if outcome['passed'] else 'failed',
            passed=bool(outcome['passed'] and done), complete=bool(outcome['passed'] and done),
            active=bool(owner and owner.active), ownership_closed=outcome.get('ownership_closed', False))
        if row['passed']:
            row['decision'] = decision(row['processes'])
        save(row)
    require(outcome['passed'], outcome.get('error', 'cache speed comparison failed'))
    print(json.dumps({k: outcome.get(k) for k in ('index', 'kernel', 'passed', 'complete', 'error')}), flush=True)


def selfcheck():
    hit = 'strata serve: decode expert cache hit rate: 50.0% (5 hits / 10 lookups)\n'
    prefix = 'strata generate: native pack: /qualified experts\nstrata generate: expert arena: SYCL ordinary host arena\n'
    require(cache_footer(hit) == dict(hits=5, lookups=10, hit_percent=50.)
            and len(cache_history(prefix + hit * 4, 4)) == 4, 'common footer and pre-generation history control')
    invalid = ['', hit + hit, hit.replace('50.0', '49.0'),
               hit.replace('5 hits', '11 hits'), hit.replace('10 lookups', '0 lookups')]
    for value in invalid:
        try:
            cache_footer(value)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('invalid common cache footer accepted')
    bad_history = [hit * 4, prefix + hit * 3, prefix + hit * 5,
        prefix + hit * 4 + 'strata serve: resident RAM: unexpected\n',
        prefix + hit * 4 + 'strata generate: experts via mmap (--mmap-experts)\n']
    for value in bad_history:
        try:
            cache_history(value, 4)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('wrong source/missing warmup statistics accepted')
    clean_observers(hit)
    suite = dict(workload_version=5, thinking=True, reasoning_budget_tokens=0, max_tokens=16384,
                 diagnostic_subset=MODE == 'pilot', selected_workloads=list(WORKLOADS))
    rows = [dict(workload=name, run=0, order=i, controller_eligible=True) for i, name in enumerate(WORKLOADS)]
    generation_contract(suite, rows)
    versions = [dict(suite, **patch) for patch in (
        {'workload_version': 4}, {'thinking': False}, {'reasoning_budget_tokens': 1024},
        {'max_tokens': 32768}, {'diagnostic_subset': MODE != 'pilot'}, {'selected_workloads': []})]
    for value in versions:
        try:
            generation_contract(value, rows)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('wrong v5 generation contract accepted')
    wrong_order, ineligible = copy.deepcopy(rows), copy.deepcopy(rows)
    wrong_order[-1]['order'] += 1
    ineligible[-1]['controller_eligible'] = False
    bad_rows = [rows[:-1], wrong_order, ineligible]
    for value in bad_rows:
        try:
            generation_contract(suite, value)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('wrong/ineligible v5 request rows accepted')
    observers = ['strata trace:', 'strata GDN storage:', 'strata decode timing:', 'strata serve: ADAPT_']
    for value in observers:
        try:
            clean_observers(value)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('diagnostic output accepted as clean timing')
    return dict(passed=True, cache_footer_controls=1, invalid_cache_footers_rejected=len(invalid),
                invalid_pre_generation_histories_rejected=len(bad_history),
                invalid_workload_contracts_rejected=len(versions) + len(bad_rows),
                disabled_observers_rejected=len(observers), GPU_executed=False)


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.guards.no_model_owners([])
        args = sys.argv[1:]
        if args == ['selfcheck']:
            print(json.dumps(selfcheck()))
        else:
            require(len(args) == 2, 'pool_spin.py selfcheck | pilot|cohort prepare|host|INDEX')
            select_mode(args[0])
            if args[1] == 'prepare':
                prepare()
            elif args[1] == 'host':
                comparison.host(sys.modules[__name__])
            elif args[1].isdigit() and int(args[1]) < len(ARMS):
                run(int(args[1]))
            else:
                raise RuntimeError('pool_spin.py pilot|cohort prepare|host|INDEX')
