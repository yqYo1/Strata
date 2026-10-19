#!/usr/bin/env python3
"""Main-owned bounded qualification of the decode CPU observer.

CPU fixture modes build no engine and perform no GPU operation. Explicit runtime
modes check a small device fixture or native protocol; they are not speed tests.
Use committed inputs and never retry an existing output directory.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import math
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
INDEX = P / 'bench/results/2026-10-15-pr6-recovery-audit/research-index.json'
BASELINE = INDEX.with_name('decode-long-generation-current.json')


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def committed(path):
    raw = path.read_bytes()
    guards.require(raw == git(P, 'show', 'HEAD:' + str(path.relative_to(P))),
                   'commit exact qualification inputs first: ' + str(path))
    return json.loads(raw)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def runtime_smoke(old_engine=False, bank=False, full_bank=False):
    """Small native-protocol diagnostic; throughput evidence still uses bench.sh."""
    committed_source = git(P, 'show', 'HEAD:' + str(Path(__file__).relative_to(P)))
    guards.require(Path(__file__).read_bytes() == committed_source, 'commit smoke controller first')
    index = committed(INDEX)
    scope = index['current_decode_mtp_verifier_reuse_source_audit']['cpu_observer_source_audit']
    bank = bank or full_bank
    result_key = ('runtime_bank_validation' if full_bank else 'runtime_device_probe' if bank else
                  'runtime_reference' if old_engine else 'runtime_smoke')
    guards.require(result_key not in scope,
                   'stage already has an outcome; preserve it in Git and commit a fresh plan before another run')
    cpu = scope['native_cpu_qualification']
    plan = scope['runtime_smoke_plan']
    base = plan['configuration_reference']
    reference = json.loads(git(P, 'show', base['commit'] + ':' + base['path']))
    guards.require(reference['complete'] and reference['passed'] and reference['ownership_closed'],
                   'closed configuration reference required')
    run_root = (plan['bank_validation_output_root'] if full_bank else plan['bank_output_root'] if bank else
                plan['reference_output_root'] if old_engine else plan['output_root'])
    environment = {k: v.replace(reference['environment']['STRATA_MEASUREMENT_ROOT'], run_root)
                   for k, v in reference['environment'].items()}
    environment.update(plan['environment_overrides'])
    if old_engine:
        environment.pop('STRATA_DECODE_CPU_PROFILE', None)
    build = committed(INDEX.with_name('native-gdn-engine-build.json'))
    full = json.loads((B / 'main-native-gdn-engine/record.json').read_text())
    from bench_engine import compact_build_record
    guards.require(cpu['passed'] and cpu['complete'] and cpu['ownership_closed'] and
                   cpu['source_commit'] == build['source_commit'] == plan['build_source_commit'] and
                   build['passed'] and build['ownership_closed'] and compact_build_record(full) == build,
                   'passed same-build CPU gate and exact native receipt required')
    for group in [full['project'], *full['dependencies']]:
        for name, expected in group['source_sha256'].items():
            guards.require(digest(Path(group['root']) / name) == expected, 'native smoke input changed: ' + name)
    binary = build['fixtures']['sycl_expert_bank_test'] if bank else build['retained_validation_reference']['binary'] if old_engine else build['binary']
    if bank:
        guards.require(binary['sha256'] == build['retained_validation_reference']['fixtures']['sycl_expert_bank_test']['sha256'],
                       'runtime device probe must use byte-identical known bank fixture')
    elif old_engine:
        guards.require(binary['sha256'] == reference['binary']['sha256'], 'fallback differs from measured reference')
    else:
        guards.require(binary['sha256'] == plan['binary_sha256'], 'native engine changed')
    guards.require(digest(binary['path']) == binary['sha256'], 'selected executable changed')
    settings = committed(P / 'tools/sycl/measurement_environment.json')
    if bank:
        environment = {**settings['environment'], **plan['bank_environment_overrides'],
                       'TMPDIR': run_root + '/tmp', 'XDG_CACHE_HOME': run_root + '/cache'}
        if full_bank:
            environment.update(plan['bank_validation_environment_overrides'])
    delta = plan.get('runtime_asset_delta', {})
    guards.require(set(delta) <= set(settings['runtime_elfs']), 'unknown proposed runtime asset')

    def runtime_assets_unchanged():
        for name, item in settings['runtime_elfs'].items():
            proposed = delta.get(name)
            if proposed:
                guards.require(proposed['qualified_sha256'] == item['sha256'], 'old runtime pin changed: ' + name)
            expected = proposed['candidate_sha256'] if proposed else item['sha256']
            guards.require(digest(name) == expected, 'runtime asset changed: ' + name)

    runtime_assets_unchanged()
    if full_bank:
        qualified = scope['runtime_device_probe']
        guards.require(qualified['complete'] and qualified['passed'] and qualified['ownership_closed'] and
                       qualified['runtime_asset_delta'] == delta, 'passed small bank probe required first')
        provider = build['fixtures']['file_expert_source_test']
        guards.require(provider['sha256'] == plan['provider_fixture_sha256'] and
                       digest(provider['path']) == provider['sha256'], 'provider fixture changed')
    if delta and old_engine:
        qualified = scope['runtime_device_probe']
        guards.require(qualified['passed'] and qualified['ownership_closed'] and
                       qualified['runtime_asset_delta'] == delta, 'bounded device probe required before model reference')
    if delta and not old_engine and not bank:
        qualified = scope['runtime_reference']
        guards.require(qualified['passed'] and qualified['ownership_closed'] and
                       qualified['runtime_asset_delta'] == delta,
                       'new runtime needs a successful bounded old-engine reference first')
    profile = plan['expert_profile']
    guards.require(digest(profile['path']) == profile['sha256'], 'expert profile changed')
    args = reference['config']['args'].copy()
    # The Python server prepends --serve; config args alone are CLI generate mode.
    if '--serve' not in args:
        args.insert(0, '--serve')
    args[args.index('--expert-profile') + 1] = profile['path']
    command = [binary['path'], '--validate' if full_bank else '--smoke'] if bank else [binary['path'], *args]
    guards.no_model_owners([])
    guards.driver_guard()
    root = Path(run_root)
    guards.require(root.parent == B and not root.exists(), 'never overwrite/retry a runtime smoke root')
    root.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (root / name).mkdir(mode=0o700)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
    cursor = next(s.split(': ', 1)[1] for s in before.splitlines() if s.startswith('-- cursor: '))
    record = dict(source_commit=git(P, 'rev-parse', 'HEAD').decode().strip(),
                  build_source_commit=build['retained_validation_reference']['source_commit'] if old_engine else build['source_commit'],
                  binary=binary, plan=plan, reference_engine=old_engine, device_probe=bank,
                  full_bank_validation=full_bank, runtime_asset_delta=delta,
                  complete=False, passed=False, active=False, requests=[])
    owner = None

    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (root / 'record.json').write_text(json.dumps(dict(record, commands=owner.commands if owner else []), indent=2) + '\n')

    def interact(io, entry):
        for _ in range(64):
            line = io.line(600)
            guards.require(not line.startswith('ERR'), line)
            if line.startswith('READY '):
                record['ready'] = line
                break
        else:
            raise RuntimeError('bounded startup protocol did not reach READY')
        for ids in plan['input_ids']:
            io.send(('GEN ' + str(plan['max_new']) + ' ' + ','.join(map(str, ids)) + '\n').encode())
            tokens = []
            marks = {}
            for _ in range(plan['max_new'] + 64):
                line = io.line(180)
                guards.require(not line.startswith('ERR'), line)
                if line.startswith(('RESUME ', 'REUSED ')):
                    key, value = line.split()
                    guards.require(key not in marks and int(value) == 0, 'unexpected reuse marker')
                    marks[key] = int(value)
                elif line.startswith('PP '):
                    fields = line.split()
                    guards.require(len(fields) == 5 and 0 <= int(fields[1]) <= int(fields[2]) <= len(ids),
                                   'invalid native prefill progress')
                elif line.startswith('T '):
                    tokens.append(int(line.split()[1]))
                elif line.startswith('DONE '):
                    fields = line.split()
                    guards.require(len(fields) == 15 and int(fields[1]) == len(tokens) and
                        0 < len(tokens) <= plan['max_new'] and int(fields[2]) == len(ids) and
                        fields[5] in ('stop', 'length') and int(fields[8]) == 0 and int(fields[14]) == len(ids) and
                        marks == {'RESUME': 0, 'REUSED': 0},
                        'smoke DONE/request counts or completion differ')
                    record['requests'].append(dict(input_tokens=len(ids), generated_tokens=len(tokens),
                        finish_reason=fields[5], decode_ms=float(fields[4]),
                        output_ids_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()))
                    break
                else:
                    # Drain through DONE and QUIT before rejecting an unknown
                    # status: a harness omission must not kill an active JIT.
                    record.setdefault('unexpected_protocol', []).append(line[:512])
            else:
                raise RuntimeError('bounded request protocol lacks DONE')
        io.send(b'QUIT\n')
        io.proc.stdin.close()
        io.exit(60)

    owner = supervision.Owner(root / 'owned', persist)
    try:
        if full_bank:
            record['provider_fixture'] = provider
            entry, out, err = owner.run('arena-provider', [provider['path']], environment, P,
                                       **plan['bank_validation_budgets'])
            guards.require(supervision.closed(entry) and entry.get('exit_code') == 0 and
                'arena_resident_layer_test private=PASS file_fallback=PASS shared_exclusion=PASS blob_identity=PASS bounds=PASS close_reopen=PASS'
                in out.read_text() and 'file_expert_source_test: PASS' in out.read_text() and not re.search(
                r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed', err.read_text()),
                'real Arena provider check failed/incomplete')
            record['provider_checks_passed'] = True
        entry, out, err = owner.run('native-smoke', command, environment, P,
            interaction=None if bank else interact,
            **(plan['bank_validation_budgets'] if full_bank else plan['bank_budgets'] if bank else plan['budgets']))
        record['exit_code'] = entry.get('exit_code')
        if entry.get('direct_identity'):
            record['native_identity'] = {k: entry['direct_identity'][k] for k in ('pid', 'start_ticks', 'sid')}
        record['protocol_error'] = entry.get('error')
        guards.require(supervision.closed(entry) and entry.get('exit_code') == 0,
                       'native smoke protocol/exit failed: ' + str(entry.get('error')) +
                       '; exit=' + str(entry.get('exit_code')))
        text = err.read_text(errors='replace')
        guards.require(not record.get('unexpected_protocol'), 'unknown protocol status after graceful completion')
        guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', text),
                       'native runtime failure; inspect the original bounded diagnostic')
        observations = [json.loads(s.partition(': ')[2]) for s in text.splitlines() if s.startswith('strata decode cpu: ')]
        if bank:
            cases = [dict(part.split('=', 1) for part in line.split()[1:])
                     for line in out.read_text().splitlines() if line.startswith('expert_bank_case ')]
            # Same original project fixture contract as check_expert_bank.py;
            # full mode additionally exercises 383/384/512 slots and real blob strides.
            expected = ([(383,256,512,1,642,382,2048,0), (384,256,512,2,1282,766,4096,0),
                         (512,256,512,1,513,511,2048,0), (384,2662400,512,1,641,383,2048,0),
                         (384,512,512,2,1282,766,4096,0), (384,2662400,512,1,641,383,2048,0)]
                        if full_bank else [(4,256,6,2,18,6,48,0), (1,256,6,1,12,0,24,0),
                         (4,256,6,1,5,0,10,1), (4,512,6,2,18,6,48,0),
                         (1,512,6,1,12,0,24,0), (4,512,6,1,5,0,10,1)])
            strides = [256,256,256,2662400,256,1382400] if full_bank else [256]*6
            keys = ('slots','stride','experts','layers','uploads','hits','consumers','unwind')
            guards.require(len(cases) == 6 and 'expert_bank_fixture PASS' in out.read_text(), 'missing bank cases')
            for c, values, stride in zip(cases, expected, strides):
                guards.require(tuple(int(c[k]) for k in keys) == values and int(c['source_stride']) == stride and
                    int(c['admission_rejections']) == 7 + 2*(stride != int(c['stride'])) and c['status'] == '0' and
                    all(c[k] == 'PASS' for k in ('immutable_slab', 'guards', 'private_map', 'checked_drain')),
                    'device probe case failed/incomplete')
            record['device_cases_passed'] = len(cases)
        else:
            guards.require(len(record['requests']) == len(plan['input_ids']) and
                           len(observations) == (0 if old_engine else len(record['requests'])),
                           'attempted/completed/CPU-record counts do not reconcile')
        for request, observation in zip(record['requests'], observations):
            guards.require(observation['schema'] == 1 and all(observation.get(k) is True for k in
                ('valid', 'single_stage', 'clock_deltas_valid', 'quiescent', 'worker_jobs_fit', 'main_subsets_fit')) and
                all(type(observation.get(k)) is int and observation[k] == 0 for k in
                    ('counter_errors', 'unwound', 'snapshot_read_errors', 'unexpected_mode_tasks')) and
                observation['input_tokens'] == request['input_tokens'] and
                observation['generated_tokens'] == request['generated_tokens'] and observation['windows'] > 0 and
                all(type(observation.get(k)) is int and observation[k] > 0 for k in
                    ('worker_job_calls', 'native_quant_calls', 'sync_calls', 'main_thread_ns', 'native_process_ns')) and
                math.isfinite(observation['wall_through_commit_ms']) and observation['wall_through_commit_ms'] > 0,
                'invalid or mismatched CPU observation')
        record['observations'] = observations
        runtime_assets_unchanged()
        record['passed'] = True
    except BaseException as error:
        record['error'] = str(error)
    finally:
        record['ownership_closed'] = owner.active is None and all(c.get('direct_child_reaped') and
            c.get('session_empty') and not c.get('survivors') and c.get('ownership_status') == 'observed'
            for c in owner.commands)
        record['complete'] = bool((record.get('device_cases_passed') == 6 if bank else
                                  len(record['requests']) == len(plan['input_ids'])) and
                                  all(supervision.closed(c) for c in owner.commands))
        record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True)
        record['fresh_faults'] = [s for s in after.splitlines() if re.search(
            r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', s)]
        try:
            record['driver_after'] = guards.driver_guard()
        except Exception as error:
            record['driver_error'] = str(error)
            record['passed'] = False
        record['passed'] = bool(record['passed'] and record['ownership_closed'] and
                                record['boot_unchanged'] and not record['fresh_faults'])
        persist()
        compact = {k: v for k, v in record.items() if k not in ('plan', 'driver_after')}
        compact['native_process_started'] = bool(owner.commands and owner.commands[0].get('pid'))
        compact['GPU_initialization_observed'] = bool(record.get('ready') or record.get('device_cases_passed'))
        compact['GPU_executed'] = True if record['requests'] or record.get('device_cases_passed') else None
        compact['limits'] = 'Two tiny native-protocol requests through final commit. Explicit runtime-delta diagnostic only; shared qualified runtime pins remain unchanged. No performance, overhead, quality, IQ numerical, long-context or RESTORE qualification.'
        if bank:
            compact['limits'] = 'Six small production upload/cache byte-reader cases with full API trace and parameter validation on proposed runtime assets. No model/throughput/quality/full-context qualification; shared runtime baseline remains unchanged.'
        if full_bank:
            compact['limits'] = 'Real private Arena provider checks plus six full upload/cache byte-reader cases, including 384 slots and real fixed/compact blob strides, on proposed runtime assets with parameter validation. No throughput/quality/full-context qualification; shared runtime baseline remains unchanged.'
        scope[result_key] = compact
        if not old_engine and not bank:
            scope['revised_candidate']['native_runtime_tested'] = compact['GPU_executed']
        INDEX.write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({k: compact.get(k) for k in ('complete', 'passed', 'ownership_closed', 'error')}))
    return 0 if record['passed'] else 1


def run(native=False):
    committed_source = git(P, 'show', 'HEAD:' + str(Path(__file__).relative_to(P)))
    guards.require(Path(__file__).read_bytes() == committed_source, 'commit controller first')
    index = committed(INDEX)
    baseline = committed(BASELINE)
    summary = baseline.get('independent_process_summary', {})
    guards.require(baseline['complete'] and baseline['passed'] and baseline['ownership_closed'] and
                   not baseline['active'] and summary.get('original_benchmark_inputs_recovered') == 114,
                   'close the frozen generation cohort and reducer first')
    scope = index['current_decode_mtp_verifier_reuse_source_audit']['cpu_observer_source_audit']
    guards.require(('native_cpu_qualification' if native else 'cpu_only_qualification') not in scope,
                   'CPU qualification already has an outcome; commit a fresh plan before another run')
    plan = dict(scope['cpu_only_qualification_plan'])
    native_build = None
    if native:
        native_plan = scope['native_cpu_qualification_plan']
        native_build = committed(INDEX.with_name('native-gdn-engine-build.json'))
        full = json.loads((B / 'main-native-gdn-engine/record.json').read_text())
        from bench_engine import compact_build_record
        guards.require(compact_build_record(full) == native_build and native_build['passed'] and
                       native_build['complete'] and native_build['ownership_closed'] and not native_build['active'] and
                       native_build['source_commit'] == native_plan['build_source_commit'],
                       'commit the exact closed native build first')
        for group in [full['project'], *full['dependencies']]:
            for name, expected in group['source_sha256'].items():
                guards.require(digest(Path(group['root']) / name) == expected, 'native input changed: ' + name)
        for name in ('cpu_clock_test', 'pool_cpu_clock_test'):
            item = native_build['fixtures'][name]
            guards.require(digest(item['path']) == item['sha256'], 'native CPU fixture changed: ' + name)
        plan.update(source_commit=native_build['source_commit'], test=native_plan['test'])
        root = Path(native_plan['output_root'])
        test_root = Path(native_plan['test'][native_plan['test'].index('--test-dir') + 1])
    else:
        child = Path(scope['revised_candidate']['source_root'])
        guards.require(git(child, 'rev-parse', 'HEAD').decode().strip() == plan['source_commit'] and
                       not git(child, 'status', '--porcelain'), 'qualification child differs from pinned source')
        root = Path(plan['configure'][plan['configure'].index('-B') + 1])
        test_root = root
    for arg in plan['configure']:
        if arg.startswith(('-DCMAKE_C_COMPILER=', '-DCMAKE_CXX_COMPILER=')):
            guards.require(digest(arg.split('=', 1)[1]) == plan['compiler_sha256'], 'compiler changed')
    ggml = Path(next(a.split('=', 1)[1] for a in plan['configure'] if a.startswith('-DSTRATA_GGML_DIR=')))
    guards.require(git(ggml, 'rev-parse', 'HEAD').decode().strip() == plan['ggml_commit'] and
                   not git(ggml, 'status', '--porcelain', '--untracked-files=no'), 'ggml inputs changed')
    for name in ('cmake', 'ninja'):
        guards.require(subprocess.check_output([plan[name + '_path'], '--version'], text=True).splitlines()[0]
                       == plan[name + '_version'], name + ' version changed')
    guards.require(root.parent == B and not root.exists(), 'never overwrite or retry a qualification root')
    guards.no_model_owners([])
    root.mkdir(mode=0o700)
    (root / 'tmp').mkdir(mode=0o700)
    compiler = Path(next(a.split('=', 1)[1] for a in plan['configure'] if a.startswith('-DCMAKE_CXX_COMPILER=')))
    environment = dict(PATH=str(compiler.parent) + ':/usr/bin:/bin', HOME=os.environ['HOME'],
                       TMPDIR=str(root / 'tmp'), LC_ALL='C.UTF-8', LANG='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
    for key in plan['environment_unset']:
        guards.require(key not in environment, 'observer arm must be selected by CTest before exec')
    record = dict(source_commit=plan['source_commit'], controller_source_commit=git(P, 'rev-parse', 'HEAD').decode().strip(),
                  plan_commit=git(P, 'log', '-1', '--format=%H', '--', str(INDEX.relative_to(P))).decode().strip(),
                  environment=environment, plan=plan, complete=False, passed=False, active=False)
    owner = None

    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (root / 'record.json').write_text(json.dumps(dict(record, commands=owner.commands if owner else []), indent=2) + '\n')

    owner = supervision.Owner(root / 'owned', persist)
    try:
        budget = plan['budgets']
        for phase in (('test',) if native else ('configure', 'build', 'test')):
            guards.no_model_owners([])
            entry, _, _ = owner.run(phase, plan[phase], environment, P,
                wall=budget[phase + '_wall_seconds'], cpu=budget['cpu_seconds'],
                text_cap=budget['raw_log_bytes'], rss_cap=budget['rss_bytes'],
                file_cap=budget['file_bytes'], total_cap=budget['total_build_bytes'])
            guards.require(supervision.closed(entry) and entry.get('exit_code') == 0, phase + ' failed; inspect original owned logs')
        log = test_root / 'Testing/Temporary/LastTest.log'
        text = log.read_text()
        names = re.findall(r'^\d+/\d+ Testing: (\S+)\s*$', text, re.M)
        expected = {'cpu_clock_off', 'cpu_clock_on', 'pool_cpu_clock_off', 'pool_cpu_clock_on'}
        guards.require(len(names) == 4 and set(names) == expected and text.count('Test Passed.') == 4,
                       'four exact successful CPU tests required')
        record['tests'] = names
        record['test_evidence'] = [s for s in text.splitlines() if s.startswith(('cpu_clock:', 'pool_cpu_clock:'))]
        record['last_test_log_sha256'] = digest(log)
        record['binaries'] = {n: dict(sha256=digest(test_root / n), bytes=(test_root / n).stat().st_size)
                              for n in ('cpu_clock_test', 'pool_cpu_clock_test')}
        record['passed'] = True
    except BaseException as error:
        record['error'] = str(error)
    finally:
        record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        record['complete'] = record['ownership_closed']
        record['passed'] = bool(record['passed'] and record['ownership_closed'])
        persist()
        compact = {k: record[k] for k in ('source_commit', 'controller_source_commit', 'plan_commit',
                   'complete', 'passed', 'ownership_closed')}
        compact['phases'] = [{k: entry.get(k) for k in ('label', 'exit_code', 'elapsed_seconds', 'error')}
                             for entry in owner.commands]
        for key in ('tests', 'test_evidence', 'last_test_log_sha256', 'binaries', 'error'):
            if key in record:
                compact[key] = record[key]
        compact['GPU_executed'] = False
        compact['limits'] = 'CPU observer accounting/output checks only; no IQ-kernel, GPU/runtime, overhead, model throughput, quality or physical-context qualification.'
        scope['native_cpu_qualification' if native else 'cpu_only_qualification'] = compact
        if native:
            scope['revised_candidate']['native_engine_built'] = True
        else:
            scope['revised_candidate']['cpu_fixtures_built'] = any(c['label'] == 'build' and c.get('exit_code') == 0 for c in owner.commands)
            scope['revised_candidate']['cpu_fixtures_tested'] = any(c['label'] == 'test' for c in owner.commands)
        INDEX.write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(compact))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    guards.require(sys.argv[1:] in ([], ['native'], ['smoke'], ['runtime-reference'], ['runtime-device'], ['runtime-bank']),
                   'use check_decode_cpu.py [native|smoke|runtime-reference|runtime-device|runtime-bank]')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.exit(runtime_smoke(old_engine=sys.argv[1:] == ['runtime-reference'], bank=sys.argv[1:] == ['runtime-device'],
                               full_bank=sys.argv[1:] == ['runtime-bank'])
                 if sys.argv[1:] in (['smoke'], ['runtime-reference'], ['runtime-device'], ['runtime-bank'])
                 else run(native=sys.argv[1:] == ['native']))
