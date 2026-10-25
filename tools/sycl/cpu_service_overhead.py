#!/usr/bin/env python3
"""Owned original/observer-off/observer-on long-generation comparison.

Repository LICENSE; independently authored using maintained internal evidence
APIs. No native rebuild, inference adoption or hardware-saturation claim.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import select
import shutil
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import cpu_service_model as base
import generation_compare as official

gen, runtime, service = base.gen, base.runtime, base.service
require, digest, write = base.require, base.digest, base.write
P, B, RESULT = base.P, base.B, base.RESULT
ROOT = B / 'cpu-service-overhead-v1'
KEY, HOST = 'observer_overhead', 'observer_overhead_host'
ORDER = ('original', 'off', 'on', 'on', 'original', 'off', 'off', 'on', 'original')
TEST = 'tests/core/test_cpu_service_overhead.py'


def source_pins(revision):
    pins = gen.benchmark_pins(revision)
    raw = gen.git('show', revision + ':' + TEST)
    require(raw == (P / TEST).read_bytes(), 'commit exact decision tests first')
    pins[TEST] = hashlib.sha256(raw).hexdigest()
    return pins


def host_check():
    container = gen.committed(RESULT)
    require(HOST not in container and not ROOT.exists(), 'fresh single-use host check')
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = source_pins(revision)
    require('tools/sycl/cpu_service_overhead.py' in pins, 'commit controller before host test')
    ROOT.mkdir(mode=0o700)
    host = dict(passed=False, ownership_closed=True, exit_code=None,
        source_commit=revision, source_files=len(pins), source_fingerprint=base.fingerprint(pins),
        tests=6,
        limits='Decision arithmetic and rejection controls only;no GPU/model or overhead qualification.')
    owner = None
    def persist():
        write(ROOT / 'host-record.json', dict(host, active=bool(owner and owner.active is not None),
            commands=owner.commands if owner else []))
    try:
        owner = base.supervision.Owner(ROOT / 'host-owned', persist)
        entry, out, err = owner.run('decision-tests', ['/usr/bin/python3', '-B', TEST],
            dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8'), P,
            wall=30, cpu=30, rss_cap=1 << 30, text_cap=1 << 20, file_cap=1 << 20, total_cap=2 << 20)
        host.update(exit_code=entry['exit_code'], stdout_sha256=digest(out), stderr_sha256=digest(err))
        host['passed'] = bool(base.supervision.closed(entry) and entry['exit_code'] == 0 and
            'Ran 6 tests' in err.read_text() and err.read_text().rstrip().endswith('OK'))
    except BaseException as error:
        host['error'] = repr(error)
    finally:
        host['ownership_closed'] = owner is None or owner.active is None and all(base.supervision.closed(e) for e in owner.commands)
        host['passed'] = bool(host['passed'] and host['ownership_closed'])
        persist()
        container[HOST] = host
    write(RESULT, container)
    print(json.dumps(host), flush=True)
    return int(not host['passed'])


def config(setup, root):
    value = copy.deepcopy(setup['config'])
    args = value['args']
    args[args.index('--expert-profile') + 1] = str(root / 'expert-profile.bin')
    value.update(log=str(root / 'engine.stderr'), env={})
    return value


def environment(setup, root, arm):
    value = {k: v.replace(str(base.ROOT), str(root)) for k, v in setup['environment'].items()}
    value['STRATA_CPU_SERVICE'] = '1' if arm == 'on' else '0'
    return value


def command(root, port):
    return ['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
        '--long-generation-only', '--long-generation-workload', 'regular-code', '--think', '--runs', '1',
        '--profile-request-gates', '--workload-id', official.WORKLOAD_ID,
        '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')]


def qualification():
    container = gen.committed(RESULT)
    setup = container['native_model']
    require(setup['passed'] and setup['complete'] and setup['ownership_closed'] and not setup['active'],
        'closed observer model/census gate required')
    qualified, models = base.current(setup)
    # base.current verified the original receipt and every original asset.
    original = json.loads(runtime.selected.REFERENCE.read_text())['binary']
    return setup, qualified, models, dict(original=original, observer=setup['binary'])


def prepare():
    container = gen.committed(RESULT)
    require(KEY not in container and (ROOT / 'host-owned').is_dir(), 'fresh cohort after host check')
    host = container[HOST]
    require(host['passed'] and host['ownership_closed'] and host['exit_code'] == 0 and
        base.fingerprint(source_pins(host['source_commit'])) == host['source_fingerprint'], 'host source gate changed')
    setup, qualified, models, binaries = qualification()
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = source_pins(revision)
    base.model.model_pins(models['config'], models['models'], full=True)
    plans, ports = [], set()
    for index, arm in enumerate(ORDER):
        root = ROOT / str(index)
        root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'):
            (root / name).mkdir()
        shutil.copyfile(P / 'data/expert-profile.bin', root / 'expert-profile.bin')
        write(root / 'config.json', config(setup, root))
        binary = binaries['original' if arm == 'original' else 'observer']
        write(root / 'plan.json', dict(binary=binary, native_validation_probe=setup['native_validation_probe']))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        require(port not in ports, 'duplicate frozen port; no root retry')
        ports.add(port)
        plans.append(dict(index=index, arm=arm, port=port,
            generated_inputs={name: digest(root / name) for name in ('config.json', 'plan.json', 'expert-profile.bin')}))
    row = dict(stage='prepared', source_commit=revision, source_files=len(pins),
        source_fingerprint=base.fingerprint(pins), qualified_model_fingerprint=base.fingerprint(setup),
        host_fingerprint=base.fingerprint(host), model_assets_fingerprint=base.fingerprint(models['models']),
        binaries=binaries, boot=qualified['boot'], driver=qualified['driver'], plans=plans, processes=[],
        plan=dict(order=list(ORDER), repetitions_per_arm=3, workload='regular-code',
            input_minimum=32768, output_minimum=1024, output_maximum=16384,
            question='Bound compile-enabled-but-off overhead against original,then CPU-service-on overhead against the same off binary;collect conserved long-request service by format/NT.',
            constants='Original6e40 versus its four scoped observer/test patches only;same selected math/configuration,1256slots,131072residentINT8,8192chunks,spec4,five workers+host. Common parameter validation,retained UR reference,stage trace,CPU census and nested decode clocks;only CPU_SERVICE0/1 and declared binary differ. Environment fixed before exec.',
            decision='Complete all nine independent processes in three Latin-order triples. For each triple compare off/original and on/off actual token throughput for prefill/decode separately. Off/original needs geomean>=0.98 and every ratio>=0.95 in both phases. On/off decode needs geomean>=0.95 and every ratio>=0.90;prefill needs0.98/0.95. Each pairing additionally needs CPU entries per emitted token within5%,each format/NT job count normalized to equal emitted tokens within5% or two jobs for small bins,and absolute MTP acceptance difference<=0.02,otherwise overhead is confounded. Code acceptance>=0.60 is recorded separately;no weak-pair claim. No speed-based early stopping.',
            stopping='Stop first source/runtime/owner/fault/budget/native/request/census/observer error;no retries,replacements or order changes. Commit every process outcome before next launch. Full model hashes before cohort and after final process;inventory/stat/ctime checked each launch and exit.',
            retention='Keep individual samples,paired decisions and conserved on-arm format/NT service needed to choose the next CPU leaf calibration. Retire successful raw traces/logs/cache after each committed process. Keep only three generated inputs per planned process through cohort closure;retire unused extra build/controllers after source recovery when comparison closes.',
            limits='Observed workload-specific overhead and attribution only;not clean speed,physical RAM/PCIe traffic,CPU arithmetic saturation,inference adoption,task quality,weak-MTP,agent-session,262144-context,RESTORE or ordinary runtime destruction qualification.'),
        budgets=dict(wall=1800, cpu=21600, rss_cap=96 << 30, text_cap=32 << 20,
            file_cap=32 << 20, total_cap=64 << 20),
        active=False, ownership_closed=True, passed=False, complete=False, inference_adopted=False)
    write(ROOT / 'frozen-plan.json', row)
    container[KEY] = row
    write(RESULT, container)
    print('Prepared nine ordered independent long-generation processes;commit plan before run.', flush=True)


def current(row):
    setup, qualified, models, binaries = qualification()
    require(base.fingerprint(setup) == row['qualified_model_fingerprint'] and binaries == row['binaries'] and
        qualified['boot'] == row['boot'] and qualified['driver'] == row['driver'] and
        base.fingerprint(models['models']) == row['model_assets_fingerprint'], 'qualification drift')
    container = gen.committed(RESULT)
    pins = source_pins(row['source_commit'])
    require(base.fingerprint(container[HOST]) == row['host_fingerprint'] and len(pins) == row['source_files'] and
        base.fingerprint(pins) == row['source_fingerprint'], 'source/host gate drift')
    frozen = json.loads((ROOT / 'frozen-plan.json').read_text())
    mutable = {'stage', 'processes', 'active', 'ownership_closed', 'passed', 'complete', 'decision', 'error'}
    require({k: v for k, v in row.items() if k not in mutable} ==
        {k: v for k, v in frozen.items() if k not in mutable}, 'frozen cohort changed')
    require(len(row['plans']) == len(ORDER), 'nine frozen processes required')
    for index, plan in enumerate(row['plans']):
        root = ROOT / str(index)
        require(plan['index'] == index and plan['arm'] == ORDER[index], 'arm order changed')
        for name, sha in plan['generated_inputs'].items():
            require(digest(root / name) == sha, 'generated cohort input changed')
        require(json.loads((root / 'config.json').read_text()) == config(setup, root), 'configuration changed')
    base.model.model_pins(models['config'], models['models'])
    return setup, qualified, models


def metric(ratios, geo_floor, individual_floor):
    require(len(ratios) == 3 and all(math.isfinite(x) and x > 0 for x in ratios), 'three finite ratios required')
    geo = math.exp(statistics.mean(math.log(x) for x in ratios))
    return dict(ratios=ratios, geomean=geo, minimum=min(ratios), maximum=max(ratios),
        accepted=geo >= geo_floor and min(ratios) >= individual_floor)


def comparable_work(x, y):
    nx, ny = x['sample']['generated_tokens'], y['sample']['generated_tokens']
    a = {(b['gu'], b['down']): b['jobs_NT'] for b in x['work']['format_nt_jobs']}
    b = {(b['gu'], b['down']): b['jobs_NT'] for b in y['work']['format_nt_jobs']}
    return bool(a.keys() == b.keys() and
        abs(y['sample']['acceptance_rate'] - x['sample']['acceptance_rate']) <= .02 and
        .95 <= (y['work']['cpu_entries'] / ny) / (x['work']['cpu_entries'] / nx) <= 1.05 and
        all(len(a[key]) == len(b[key]) == 4 and
            all(abs(v * nx / ny - u) <= max(2, .05 * u) for u, v in zip(a[key], b[key])) for key in a))


def summarize(processes):
    require(len(processes) == 9 and [p['index'] for p in processes] == list(range(9)) and
        [p['arm'] for p in processes] == list(ORDER) and
        len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == 9,
        'complete ordered independent cohort required')
    require(all(p['passed'] and p['ownership_closed'] and p['native_exit_code'] == 0 and
        p['fresh_faults'] == [] and p['boot_unchanged'] and
        p['effective_environment_verified'] and gen.sample_eligible(p['sample']) and
        p['source_commit'] == processes[0]['source_commit'] and
        math.isfinite(p['sample']['prompt_ms']) and p['sample']['prompt_ms'] > 0 and
        p['sample']['request_sha256'] == processes[0]['sample']['request_sha256'] and
        (p['arm'] != 'on' or p['service']['same_request_census']) for p in processes), 'eligible source-bound samples required')
    original = {p['binary_sha256'] for p in processes if p['arm'] == 'original'}
    observed = {p['binary_sha256'] for p in processes if p['arm'] != 'original'}
    require(len(original) == len(observed) == 1 and original.isdisjoint(observed), 'two declared binaries required')
    decisions = {}
    for name, a, b, decode_floor, decode_each in (
            ('off_vs_original', 'original', 'off', .98, .95), ('on_vs_off', 'off', 'on', .95, .90)):
        pairs = []
        for offset in (0, 3, 6):
            group = {p['arm']: p for p in processes[offset:offset + 3]}
            pairs.append((group[a], group[b]))
        comparable = all(comparable_work(x, y) for x, y in pairs)
        phases = {}
        for phase, token, clock in (('prefill', 'read_tokens', 'prompt_ms'), ('decode', 'generated_tokens', 'decode_ms')):
            ratios = [(y['sample'][token] / y['sample'][clock]) / (x['sample'][token] / x['sample'][clock]) for x, y in pairs]
            phases[phase] = metric(ratios, decode_floor if phase == 'decode' else .98,
                decode_each if phase == 'decode' else .95)
        decisions[name] = dict(work_comparable=comparable, **phases,
            accepted=comparable and all(p['accepted'] for p in phases.values()))
    return dict(comparisons=decisions, overhead_qualified=all(v['accepted'] for v in decisions.values()),
        observed_strong_MTP=all(p['sample']['acceptance_rate'] >= .60 for p in processes),
        inference_adopted=False, clean_speed_qualified=False, hardware_saturation_qualified=False)


def pool_evidence(text, samples, cfg, root, enabled):
    timing = gen.decode_timing_evidence(text, samples)
    bindings = [dict(prompt_tokens=s['prompt_tokens'], windows=t['windows'], rows=t['work']['target_rows'],
        cpu_jobs=t['work']['cpu_unique_jobs'], cpu_entries=t['work']['cpu_entries'],
        pool_ms=t['ms_per_window']['CPU_pool_run'] * t['windows']) for s, t in zip(samples, timing)]
    args = cfg['args']
    formats = service.census.manifest_formats(Path(args[args.index('--pack') + 1]) / 'native_experts.txt')
    membership = service.census.expected_membership(root / 'expert-profile.bin')
    work = service.census.extract(text, bindings, formats, membership)
    if enabled:
        leaf = service.extract(text, bindings, formats, membership)
    else:
        require(not service.records(text), 'off/original unexpectedly published CPU service')
        leaf = []
    return work, timing, leaf


def run():
    container = gen.committed(RESULT)
    row = container[KEY]
    require(row['stage'] in ('prepared', 'collecting') and not row['active'] and row['ownership_closed'] and
        len(row['processes']) < len(ORDER) and all(p['passed'] for p in row['processes']), 'closed/active/failed cohort')
    try:
        setup, qualified, models = current(row)
    except BaseException as error:
        row.update(stage='admission-failed', error=repr(error), passed=False, complete=False,
            active=False, ownership_closed=True)
        write(RESULT, container)
        raise
    index = len(row['processes'])
    plan, arm, root = row['plans'][index], ORDER[index], ROOT / str(index)
    require(not (root / 'record.json').exists(), 'single-use process attempt;never retry')
    binary = row['binaries']['original' if arm == 'original' else 'observer']
    env, cfg = environment(setup, root, arm), config(setup, root)
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(s.split(': ', 1)[1] for s in cursor_text.splitlines() if s.startswith('-- cursor: '))
    observation = dict(index=index, arm=arm, active=False, ownership_closed=True, passed=False,
        binary_sha256=binary['sha256'], source_commit=row['source_commit'])
    owner = None

    def persist():
        observation['active'] = bool(owner and owner.active is not None)
        write(root / 'record.json', dict(observation, commands=owner.commands if owner else []))

    def interact(io, entry):
        for stage in ('begin', 'end'):
            while True:
                line = io.line(1650)
                if line.startswith('STRATA_PROFILE_GATE '):
                    break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                dict(stage=stage, phase='long-generation/regular-code', run=0), 'official request gate changed')
            claim = json.loads((root / 'engine-start.json').read_text())['native']
            actual = base.supervision.identity(claim['pid'])
            require(actual and actual['sid'] == entry['pid'] and actual['start_ticks'] == claim['start_ticks'] and
                Path('/proc', str(claim['pid']), 'exe').resolve() == Path(binary['path']).resolve(), 'live native identity changed')
            require('native_identity' not in observation or observation['native_identity'] == claim,
                'native process changed between request gates')
            actual_env = dict(s.split(b'=', 1) for s in Path('/proc', str(claim['pid']), 'environ').read_bytes().split(b'\0') if b'=' in s)
            expected = dict(env, LD_PRELOAD=setup['native_validation_probe']['library']['path'],
                STRATA_UR_PROBE_EXPECTED_LOADER=setup['native_validation_probe']['loader']['path'])
            require(all(actual_env.get(k.encode()) == v.encode() for k, v in expected.items()), 'native environment differs')
            mapped = runtime.executable_mappings(Path('/proc', str(claim['pid']), 'maps').read_text())
            require(all(p in qualified['runtime_files'] or p == str(Path(binary['path']).resolve()) for p in mapped) and
                '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39758' in mapped, 'native executable providers differ')
            if stage == 'begin':
                path, = (root / 'bench').glob('*/requests.jsonl')
                events = [json.loads(s) for s in path.read_text().splitlines()]
                require(len(events) == 10 and events[-1]['kind'] == 'start' and
                    events[-1]['phase'] == 'long-generation/regular-code', 'warmup journal extent differs')
                gen.warmup_reconciliation(events[:-1], workloads=())
                text = (root / 'engine.stderr').read_text()
                require(not re.search(official.FAILURE, text), 'native warmup failed')
                base.validation.preload_evidence(text, claim['pid'])
                warning = root / 'validation.log'
                require(not warning.exists() or warning.stat().st_size == 0, 'warmup validation warning')
                pool_evidence(text, [e['sample'] for e in events[:-1] if e['kind'] == 'result'],
                    cfg, root, arm == 'on')
            observation.update(native_identity=claim, effective_environment_verified=True)
            persist()
            io.send(b'continue\n')
        io.proc.stdin.close()
        deadline = time.monotonic() + 120
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            require(time.monotonic() < deadline, 'benchmark teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]:
                io.read()
                io.pending.clear()
        io.exit(1)

    try:
        owner = base.supervision.Owner(root / 'owned', persist)
        entry, _, _ = owner.run('bench', command(root, plan['port']), env, P, interaction=interact, **row['budgets'])
        require(base.supervision.closed(entry) and entry['exit_code'] == 0, 'owned benchmark failed')
        result_path, = (root / 'bench').glob('*/results.json')
        samples, journal, events = official.official(root, json.loads(result_path.read_text()), workloads=('regular-code',))
        sample, = samples
        exit_ = gen.native_end(root, dict(binary=binary, native_validation_probe=setup['native_validation_probe']),
            dict(config=cfg), row['boot'], entry['pid'])
        text = (root / 'engine.stderr').read_text()
        base.validation.preload_evidence(text, exit_['native']['pid'])
        warnings = root / 'validation.log'
        require((not warnings.exists() or warnings.stat().st_size == 0) and not re.search(official.FAILURE, text), 'native validation/runtime failure')
        values = [event['sample'] for event in events if event['kind'] == 'result']
        work, timing, service_rows = pool_evidence(text, values, cfg, root, arm == 'on')
        if arm == 'on':
            observation['service'] = service_rows[-1]
        require(len(values) == len(work) == 5 and len(samples) == 1, 'warmup/long-request count differs')
        headers = list(re.finditer(r'^strata serve: prompt .*$', text, re.M))
        require(len(headers) == 5, 'native request footer count differs')
        segment = text[headers[-2].end():headers[-1].start()]
        cells = sample['read_tokens'] - 1
        route = re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 '
            r'residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) '
            r'residual_gpu=(\d+) residual_host=0 residual_staging=0 ', segment, re.M)
        require(len(route) == 1 and tuple(map(int, route[0])) == (cells, cells * 20480, cells, cells * 20480) and
            len(re.findall(r'^strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows='
                + str(min(cells, 81920)) + r' mixed_fp16_bytes=0$', segment, re.M)) == 1 and
            'strata prefill dense: phase=restored residency=Full activations=48 ' in segment,
            'selected long BF16/mode10 route or dense restoration differs')
        sample.pop('client_interval', None)
        observation.update(sample=sample, work=dict({key: work[-1][key] for key in ('cpu_jobs', 'cpu_entries')},
            format_nt_jobs=[dict(gu=b['gu'], down=b['down'], jobs_NT=b['jobs_NT'][:4])
                for b in work[-1]['format_bins']]),
            actual_route=dict(BF16_GPU_rows=cells, MoE_mode=10, restored_dense_layers=48,
                restored_cache_membership_verified=True, cache_payload_hash_verified=False),
            timing=timing[-1], journal=dict(valid=journal['valid'], successful_warmups=journal['successful_warmups'],
                successful_long_requests=journal['successful_long_requests'], journal_sha256=journal['journal_sha256']),
            native_exit_code=0, benchmark_exit_code=0, evidence_sha256=digest(root / 'engine.stderr'))
        current(row)
        base.model.model_pins(models['config'], models['models'], full=index == 8)
        observation['host_probe_accepted'] = True
    except BaseException as error:
        observation['error'] = repr(error)
    finally:
        if owner is not None:
            base.validation.finish(observation, owner, cursor, row['boot'])
        observation['active'] = False
        observation['passed'] = bool(observation.get('passed') and observation.get('driver_after') == row['driver'])
        persist()
        row['processes'].append(observation)
        row.update(stage='collecting' if observation['passed'] else 'failed', active=False,
            ownership_closed=observation['ownership_closed'])
        if len(row['processes']) == 9 and observation['passed']:
            try:
                row.update(decision=summarize(row['processes']), stage='closed', complete=True, passed=True)
            except BaseException as error:
                row.update(stage='failed', error=repr(error))
        container[KEY] = row
        write(RESULT, container)
    print(json.dumps({key: observation.get(key) for key in ('index', 'arm', 'passed', 'ownership_closed', 'error')}), flush=True)
    return int(not observation['passed'] or row['stage'] == 'failed')


if __name__ == '__main__':
    require(sys.argv[1:] in (['host-check'], ['prepare'], ['run']), 'host-check|prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if sys.argv[1] == 'host-check':
            sys.exit(host_check())
        elif sys.argv[1] == 'prepare':
            prepare()
        else:
            sys.exit(run())
