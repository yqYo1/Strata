#!/usr/bin/env python3
"""Owned six-process observer-cost comparison; no inference adoption.

Main-authored, repository LICENSE. Reuses our e53649fb4 model-smoke controller
and existing source/owner/runtime readers. No third-party application source.
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
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_profile_engine as native
import command_service
import cpu_capacity as models
import generation_engine as gen
import measurement_owner as owners

r, P, B, RESULT = native.r, native.P, native.B, native.RESULT
ROOT = native.ROOT / 'observer-cost-v1'
KEY = 'observer_cost'
TEMPLATE = gen.A / 'phase-prefill-speed-current.json'
HISTORY = gen.A / 'prefill-runtime-comparison-current.json'
READER = P / 'bench/sycl/client_resources.py'
ORDER = ('off', 'on', 'on', 'off', 'off', 'on')
OBSERVERS = ('STRATA_PREFILL_LAYER_MAJOR_PROFILE', 'STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE',
    'STRATA_PREFILL_LAYER_MAJOR_GDN_PROFILE', 'STRATA_PREFILL_CPU_WAIT_PROFILE',
    'STRATA_SYCL_PREFILL_COMMAND_PROFILE', 'STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE')
EXTRA = ('bench/sycl/client_resources.py', 'bench/sycl/hardware_conditions.py')
require, digest, write = r.require, r.digest, r.write


def scope(container):
    return container['event_reconciliation']['fused_command_profile']


def summarize(processes, expected_hashes):
    require(len(processes) == 6 and len(expected_hashes) == 2, 'six processes/two request identities required')
    times, first = [], []
    for index, (process, arm) in enumerate(zip(processes, ORDER)):
        require(process['index'] == index and process['arm'] == arm and process['passed'] is True
            and process['ownership_closed'] is True and process['complete'] is True, 'incomplete or reordered process')
        samples = process['requests']
        require(len(samples) == 2, 'two fresh requests required')
        for run, sample in enumerate(samples):
            require(sample['run'] == run and sample['request_sha256'] == expected_hashes[run] and
                sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == 117489 and
                sample['actual_reused_tokens'] == 0 and sample['generated_tokens'] == 24 and
                sample['thinking_requested'] is True, 'matched request contract differs')
            value = sample['prompt_ms']
            require(type(value) in (int, float) and math.isfinite(value) and value >= 10000,
                    'invalid prefill clock')
        first.append(samples[0]['prompt_ms']); times.append(samples[1]['prompt_ms'])
    ratios = []
    for start in (0, 2, 4):
        pair = {processes[i]['arm']: times[i] for i in (start, start + 1)}
        ratios.append(pair['on'] / pair['off'])
    gm = math.exp(math.fsum(math.log(v) for v in ratios) / 3)
    return dict(first_use_ms=first, subsequent_ms=times, time_ratios_on_over_off=ratios,
        geometric_mean=gm, geometric_mean_gate=gm <= 1.02, every_pair_gate=max(ratios) <= 1.05,
        observer_cost_qualified=gm <= 1.02 and max(ratios) <= 1.05,
        first_use_pooled=False, production_adopted=False,
        limits='Total observer cost on this diagnostic binary/workload only. Not restoration,clean-speed '
            'qualification of on samples,hardware saturation,submit queue-delay validity or production adoption.')


def pins(revision):
    result = gen.benchmark_pins(revision)
    for name in EXTRA:
        raw = gen.git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'commit exact collector input: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def engine():
    container = gen.committed(RESULT); built = json.loads(native.RECORD.read_text())
    require(scope(container)['engine_build'] == native.compact(built) and built['passed']
        and built['ownership_closed'] and not built['active'], 'closed whole-engine build required')
    native.admit(built)
    compiled = built['compiled_inputs']
    require(digest(compiled['manifest']['path']) == compiled['manifest']['sha256'], 'engine manifest changed')
    values = json.loads(Path(compiled['manifest']['path']).read_text())
    require(len(values) == 1403 and all(digest(p) == h for p, h in values.items()), 'compiled input differs')
    require(digest(built['binary']['path']) == built['binary']['sha256'], 'diagnostic engine changed')
    return built


def compact(row):
    result = {k: v for k, v in row.items() if k not in ('source_pins', 'models', 'cases')}
    result['cases'] = [dict(index=c['index'], arm=c['arm'], root=c['root'], command=c['command'],
        input_sha256={k: hashlib.sha256(json.dumps(c[k], sort_keys=True).encode()).hexdigest()
            for k in ('configuration', 'environment', 'native_plan')}) for c in row['cases']]
    return result


def save(row):
    container = json.loads(RESULT.read_text()); scope(container)[KEY] = compact(row); write(RESULT, container)


def current(row):
    require(scope(gen.committed(RESULT))[KEY] == compact(row), 'commit exact observer-cost plan')
    require(pins(row['source_commit']) == row['source_pins'], 'observer-cost source changed')
    built = engine()
    require(built['binary'] == row['binary'] and built['source_commit'] == row['build_source_commit'],
            'whole-engine source/binary differs')
    require(digest(TEMPLATE) == row['template_sha256'] and digest(HISTORY) == row['history_sha256'] and
        digest(r.RESULT) == row['runtime_sha256'], 'qualified references changed')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot changed')
    for path, sha in row['generated_inputs'].items():
        require(digest(path) == sha, 'generated input changed: ' + path)
    models.model_pins(row['configuration'], row['models'])
    return gen.committed(r.RESULT)


def prepare():
    container = gen.committed(RESULT); shared = scope(container)
    require(KEY not in shared and not ROOT.exists(), 'fresh cost comparison only; no retry')
    smoke = shared['model_smoke_v2']; plan = shared['observer_cost_plan']
    require(smoke['passed'] and smoke['ownership_closed'] and smoke['retirement']['complete'] and
        smoke['independent_review']['bytes'] == 4784 and plan['order'] == list(ORDER) and
        plan['prospective_cost_gate']['geometric_mean_maximum'] == 1.02 and
        plan['prospective_cost_gate']['each_pair_maximum'] == 1.05, 'reviewed smoke/prospective cost criteria required')
    built = engine(); rt = gen.committed(r.RESULT)
    history = gen.committed(HISTORY)['historical_regression_audit']['userland_crossover']
    config = copy.deepcopy(history['configuration'])
    original = json.loads(gen.git('show', '1e676d554:' + str(TEMPLATE.relative_to(P))))
    model_pins = models.model_pins(config, original['model_assets'])
    env = gen.committed(TEMPLATE)['environment_template']; old = env['STRATA_MEASUREMENT_ROOT']
    require(not {'LD_PRELOAD', 'STRATA_TRACE', 'STRATA_RESOURCE_CONDITIONS'} & set(env),
            'unobserved base/private native preload required')
    preload = dict(library=rt['preload']['library'], loader=dict(path=rt['preload']['loader'],
        sha256=rt['runtime_files'][rt['preload']['loader']]))
    driver = next(p for p in rt['runtime_files'] if '/libze_intel_gpu.so.' in p)
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    row = dict(stage='prepared', source_commit=revision, source_pins=pins(revision),
        binary=built['binary'], build_source_commit=built['source_commit'], boot=rt['boot'], driver=rt['driver'],
        configuration=config, models=model_pins, preload=preload,
        template_sha256=digest(TEMPLATE), history_sha256=digest(HISTORY), runtime_sha256=digest(r.RESULT),
        requests_expected=history['request_hashes'][:2], generated_inputs={}, cases=[], processes=[],
        active=False, ownership_closed=True, passed=False, complete=False, observer_cost_qualified=False,
        production_selected=False, historical_regression_attributed=False, plan=plan,
        host_controls=dict(command=['/usr/bin/python3', '-B', str(P / 'tools/sycl/test_profile_observer_cost.py')],
            python=dict(path='/usr/bin/python3', sha256=digest('/usr/bin/python3'))),
        budgets=dict(wall=1800, cpu=72000, rss_cap=104 << 30, text_cap=16 << 20,
                     file_cap=32 << 20, total_cap=64 << 20),
        whole_wall_seconds=7200, whole_artifact_bytes=448 << 20, whole_text_bytes=64 << 20,
        budget_reason='Six serial fresh cases retain private JIT/1Hz captures only through replay;64MiB each plus '
            'bounded common metadata. Total text still64MiB. No fullAPI trace,tensors or extra native build.',
        retention=plan['retention'])
    ROOT.mkdir(mode=0o700)
    for index, arm in enumerate(ORDER):
        root = ROOT / (str(index) + '-' + arm); root.mkdir(mode=0o700)
        for name in ('cache', 'tmp'):
            (root / name).mkdir()
        environment = {k: v.replace(old, str(root)) for k, v in env.items()}
        environment.update({k: '1' if arm == 'on' else '0' for k in OBSERVERS})
        environment.update(ZE_ENABLE_ALT_DRIVERS=driver,
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
        if arm == 'on':
            environment.update(STRATA_TRACE='1', STRATA_RESOURCE_CONDITIONS='1')
        cfg = copy.deepcopy(config); cfg['log'] = str(root / 'engine.stderr'); write(root / 'config.json', cfg)
        native_plan = dict(binary=row['binary'], native_validation_probe=preload)
        if arm == 'on':
            native_plan['resource_collection'] = dict(source=dict(path=str(READER), sha256=digest(READER)),
                seconds=900, interval=1)
        write(root / 'plan.json', native_plan)
        row['generated_inputs'].update({str(root / name): digest(root / name) for name in ('config.json', 'plan.json')})
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        command = ['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
            '--prefill-only', '--prefill-tokens', '98304', '--think', '--runs', '2', '--profile-request-gates',
            '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600',
            '--out', str(root / 'bench')]
        row['cases'].append(dict(index=index, arm=arm, root=str(root), configuration=cfg,
            environment=environment, native_plan=native_plan, command=command))
    write(ROOT / 'record.json', row); save(row)
    print('Prepared six-process117K observer-cost plan;commit generated inputs before controls/model execution.', flush=True)


def interaction(case, row, rt, outcome, deadline, io, leader):
    root = Path(case['root'])
    original_poll = io.poll; next_size = 0
    def poll():
        nonlocal next_size
        original_poll(); require(time.monotonic() < deadline, 'whole comparison deadline')
        if time.monotonic() >= next_size:
            require(owners.artifact_bytes(ROOT) <= row['whole_artifact_bytes'], 'whole comparison artifact cap')
            size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and
                       (p.suffix in ('.stdout', '.stderr', '.jsonl') or p.name == 'engine.stderr'))
            require(size <= row['whole_text_bytes'], 'whole comparison text cap')
            next_size = time.monotonic() + 1
    io.poll = poll
    for request in range(2):
        for stage in ('begin', 'end'):
            while True:
                line = io.line(1500)
                if line.startswith('STRATA_PROFILE_GATE '):
                    break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                dict(stage=stage, phase='prefill-only', run=request), 'request gate differs')
            require(not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size,
                    'parameter validation warning')
            if stage == 'begin':
                start = json.loads((root / 'engine-start.json').read_text()); pid = start['native']['pid']
                identity = owners.identity(pid)
                require(identity and identity['start_ticks'] == start['native']['start_ticks'] and
                    identity['sid'] == leader['pid'] and str(Path('/proc', str(pid), 'exe').resolve()) ==
                    str(Path(row['binary']['path']).resolve()), 'native identity differs')
                actual = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
                expected = dict(case['environment'], LD_PRELOAD=row['preload']['library']['path'],
                    STRATA_UR_PROBE_EXPECTED_LOADER=row['preload']['loader']['path'])
                require(all(actual.get(k.encode()) == v.encode() for k, v in expected.items()) and
                    set(actual) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
                    'native immutable environment differs')
                paths = r.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
                require(paths <= set(rt['runtime_files']) | {str(Path(row['binary']['path']).resolve())} and
                    case['environment']['ZE_ENABLE_ALT_DRIVERS'] in paths, 'actual native provider differs')
                outcome['verified_request_boundaries'] = request + 1
            io.send(b'continue\n')
    io.proc.stdin.close(); end = time.monotonic() + 180
    while not io.exit_watch.poll(0) or not io.eof:
        io.poll(); require(time.monotonic() < end, 'case teardown deadline')
        if select.select([io.proc.stdout], [], [], .05)[0]:
            io.read(); io.pending.clear()
    io.exit(1)


def analyze(case, row, entry, outcome):
    root = Path(case['root']); rf, = (root / 'bench').glob('*/results.json')
    result = json.loads(rf.read_text()); samples = result['results']['prefill_only']['requests']
    require(result['complete'] and not result['results']['failures'] and len(samples) == 2 and
        outcome.get('verified_request_boundaries') == 2, 'official case incomplete')
    for sample, sha in zip(samples, row['requests_expected']):
        require(sample['ok'] and sample['eligible'] and sample['stream_complete'] and sample['thinking_requested'] and
            sample['reasoning_observed'] and sample['request_sha256'] == sha and sample['actual_prompt_tokens'] ==
            sample['actual_read_tokens'] == 117489 and sample['actual_reused_tokens'] == 0 and
            sample['generated_tokens'] == 24 and sample['prompt_tps_source'] == 'engine', 'case workload differs')
    sys.path.insert(0, str(P / 'tools'))
    from strata_bench import verify_request_journal
    jf, = (root / 'bench').glob('*/requests.jsonl'); journal = verify_request_journal(jf.parent)
    events = [json.loads(v) for v in jf.read_text().splitlines()]
    require(journal['valid'] and journal['complete'] and [dict(v['sample'], run=v['run']) for v in events
        if v['kind'] == 'result' and v['phase'] == 'prefill-only'] == samples, 'journal binding differs')
    native_end = gen.native_end(root, case['native_plan'], dict(config=case['configuration']), row['boot'], entry['pid'])
    log = (root / 'engine.stderr').read_text(); r.validation.preload_evidence(log, native_end['native']['pid'])
    require(log.count('strata prefill: layer-major candidate, tokens=117488 chunk=8192 residual_dtype=bf16') == 2 and
        log.count('strata MoE window configuration: requested_mode=10 effective_mode=10') == 2 and
        not any(x in log for x in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')),
        'case actual route/runtime differs')
    profiles = []
    if case['arm'] == 'on':
        parsed = command_service.parse_profiles(log, 4); require(len(parsed) == 2, 'two service profiles required')
        for profile in parsed:
            footer = profile['summaries'][0]; require(profile['version'] == 3, 'profile version differs')
            profiles.append(dict(records=int(footer['records']), windows=4, peak_events=int(footer['peak_events']),
                groups=len(profile['groups']), service_complete=True, invalid=0,
                submit_clock_inversions=int(footer['submit_clock_inversions']), queue_delay_qualified=False))
        sys.path.insert(0, str(READER.parent)); import client_resources, hardware_conditions
        captures = [json.loads(v) for v in (root / 'client-resources.jsonl').read_text().splitlines()]
        coverage = []
        for sample in samples:
            span = sample['client_interval']
            client_resources.summarize_capture(captures, native_end['native'], row['boot'], span, before_only=True)
            found = hardware_conditions.summarize(captures, span['request_begin_monotonic_ns'], span['first_output_monotonic_ns'])
            coverage.append(dict(samples=found['contained_samples'], clock_coverage_verified=True))
        outcome['collector_coverage'] = coverage
    else:
        require('strata command ' not in log and not (root / 'client-resources.jsonl').exists(), 'disabled observer ran')
    fields = ('run', 'request_sha256', 'actual_prompt_tokens', 'actual_read_tokens', 'actual_reused_tokens',
              'generated_tokens', 'prompt_ms', 'thinking_requested')
    outcome.update(complete=True, host_probe_accepted=True, profiles=profiles,
        requests=[{k: s[k] for k in fields} for s in samples], native_identity=native_end['native'],
        journal_sha256=digest(jf), engine_log_sha256=digest(root / 'engine.stderr'))


def run(row):
    rt = current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'host-owned').exists(), 'single-use frozen comparison')
    initial = copy.deepcopy(row); deadline = time.monotonic() + row['whole_wall_seconds']
    owner = None; outcomes = []; host_passed = False
    def persist():
        write(ROOT / 'live.json', dict(active=bool(owner and owner.active), processes=outcomes,
            commands=owner.commands if owner else []))
    try:
        host = row['host_controls']; require(digest(host['python']['path']) == host['python']['sha256'], 'host Python changed')
        owner = owners.Owner(ROOT / 'host-owned', persist)
        entry, _, _ = owner.run('controls', host['command'], dict(PATH='/usr/bin:/bin', LANG='C.UTF-8',
            PYTHONDONTWRITEBYTECODE='1'), P, wall=60, cpu=60, rss_cap=1 << 30,
            text_cap=1 << 20, file_cap=1 << 20, total_cap=64 << 20)
        require(owners.closed(entry) and entry['exit_code'] == 0, 'host cost controls failed'); host_passed = True
        for case in row['cases']:
            current(initial); r.guards.no_model_owners([])
            outcome = dict(index=case['index'], arm=case['arm'], complete=False, passed=False)
            outcomes.append(outcome)
            cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
                if v.startswith('-- cursor: '))
            owner = owners.Owner(Path(case['root']) / 'owned', persist)
            try:
                callback = lambda io, leader: interaction(case, row, rt, outcome, deadline, io, leader)
                entry, _, _ = owner.run('bench', case['command'], case['environment'], P,
                    interaction=callback, **row['budgets'])
                require(owners.closed(entry) and entry['exit_code'] == 0, 'owned official case failed')
                analyze(case, row, entry, outcome)
            except BaseException as error:
                outcome['error'] = repr(error)
                raise
            finally:
                r.validation.finish(outcome, owner, cursor, row['boot']); persist()
            require(outcome['passed'] and outcome['complete'], 'case gate failed; stop without retry')
            print(json.dumps(dict(process=case['index'], arm=case['arm'], passed=True,
                prefill_ms=[s['prompt_ms'] for s in outcome['requests']])), flush=True)
        current(initial); models.model_pins(row['configuration'], row['models'], full=True)
        require(time.monotonic() < deadline, 'whole comparison/post-check deadline')
        row.update(complete=True, summary=summarize(outcomes, row['requests_expected']), full_model_hashes=15)
        row['observer_cost_qualified'] = row['summary']['observer_cost_qualified']
        row['passed'] = True
    except BaseException as error:
        row['error'] = repr(error)
    finally:
        row.update(stage='closed', active=False, host_controls_passed=host_passed, processes=outcomes,
            ownership_closed=(owner is None or owner.active is None) and all(p.get('ownership_closed') for p in outcomes))
        row['passed'] = bool(row['passed'] and row['complete'] and row['ownership_closed'])
        write(ROOT / 'record.json', row); save(row)
    print(json.dumps({k: row.get(k) for k in ('passed', 'complete', 'ownership_closed',
        'observer_cost_qualified', 'summary', 'error')}), flush=True)
    return int(not (row['passed'] and row['observer_cost_qualified']))


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); r.guards.no_model_owners([])
        require(sys.argv[1:] in (['prepare'], ['run']), 'profile_observer_cost.py prepare|run')
        sys.exit(prepare() if sys.argv[1] == 'prepare' else run(json.loads((ROOT / 'record.json').read_text())))
