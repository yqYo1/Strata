#!/usr/bin/env python3
"""Qualify the retained PTI collector on the current official native generation.

Repository LICENSE; Main-authored from this fork's source/owner/trace helpers.
No native rebuild, inference change, counter request or XeStrata input.
"""
import collections
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import select
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import prefill_resources as resources
import profile_split as trace
import production_profile as known
import matrix_shapes
from build_engine import blobs

P, B, RESULT = resources.P, resources.B, resources.RESULT
ROOT, KEY = B / 'official-kernel-coverage-v1', 'official_kernel_coverage_v1'
require, digest, write = resources.require, resources.digest, resources.write
validation = resources.shared.runtime.validation
supervision = resources.build.supervision
COLLECTOR_CONTROLS = {'tools/sycl/bench_engine.py', 'tools/sycl/generation_engine.py'}


def compact(row):
    return resources.entry.compact(row)


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def collector_assets(data, revision):
    """A new explicit host-controller audit; preserve the old audit/refusal."""
    import build_indexed_collector as build
    previous = data['collector_reuse']
    original = validation.committed(build.RESULT)[build.KEY]
    recovered = {name: build.tool.sha(raw) for name, raw in
                 blobs(P, original['source_commit'], original['input_sha256'])}
    require(recovered == original['input_sha256'], 'original collector inputs unavailable')
    changes = [dict(path=name, original_sha256=sha, current_sha256=digest(P / name))
               for name, sha in recovered.items() if digest(P / name) != sha]
    require({r['path'] for r in changes} == COLLECTOR_CONTROLS,
            'undeclared collector controller difference')
    # The reused checker independently verifies all generated native inputs,
    # compile recipe,460actual compiler inputs and both executable hashes.
    # Only this NEW scoped audit supplies the explicitly allowed current Python
    # identities. Do not modify previous measured receipts or their outcomes.
    reuse = copy.deepcopy(previous)
    reuse['current_source_audit']['host_only_changes'] = changes
    assets = trace.collector_assets(dict(collector_reuse=reuse))
    return assets, dict(source_commit=revision, native_source=original['source_commit'],
        original_input_files=len(recovered), declared_controller_paths=sorted(COLLECTOR_CONTROLS),
        host_only_changes=changes, original_audit_unchanged=True,
        all_generated_native_and_460_compiler_inputs_and_two_ELFs_verified=True,
        dependencies=previous['dependencies'], new_collector_build=False,
        new_runtime_or_target_qualification=False)


def basis(revision):
    built, providers = resources.native_reuse(revision)
    data = resources.build.committed(RESULT)
    pair = data[resources.KEY]
    require(pair['passed'] and pair['complete'] and pair['ownership_closed']
            and not pair['active'] and pair['decision']['attribution_eligible']
            and pair['binary'] == built['binaries']['strata']
            and pair['native_source_commit'] == built['source_commit'],
            'closed same-generation resource pair required')
    assets, audit = collector_assets(data, revision)
    return built, providers, pair, assets, audit


def prepare():
    revision = resources.shared.gen.git('rev-parse', 'HEAD').decode().strip()
    sources = matrix_shapes.sources(revision, Path(__file__).resolve())
    built, providers, pair, assets, audit = basis(revision)
    data = resources.build.committed(RESULT)
    require(KEY not in data and not ROOT.exists(), 'fresh coverage only; inspect an existing run')
    gate = data[resources.entry.component.KEY]
    ROOT.mkdir(mode=0o700)
    write(ROOT / 'runtime-inputs.json', providers)
    cells = []
    for index, (name, args, expected) in enumerate((
            ('prefix', [], 0), ('submission', ['--submission'], 606),
            ('native-graphs', ['--native'], 62))):
        root = ROOT / name
        root.mkdir(mode=0o700)
        for folder in ('tmp', 'cache', 'trace'):
            (root / folder).mkdir()
        binary = built['runtime_basis']['canary'] if index == 0 else built['binaries']['sycl_submission_test']
        prof = dict(**assets, session='StrataOfficial' + secrets.token_hex(12),
            environment=dict(UNITRACE_SystemTime='1', UNITRACE_FollowChildProcess='1',
                UNITRACE_TraceOutputDir=str(root / 'trace'), LD_DEBUG='libs'))
        env = dict(gate['environment'], TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
            CUDA_MODULE_LOADING='EAGER',
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.log'),
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
        env.pop('LD_PRELOAD', None)
        plan = dict(native_validation_probe=pair['native_validation_probe'])
        target = trace.profile_target_environment(plan, env, prof)
        expected_env = trace.expected_target_environment(plan, env, prof)
        expected_env.pop('UNITRACE_StartPaused')  # This fixture records from exec.
        require(target['ZES_ENABLE_SYSMAN'] == '0' and
            target['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and
            target['EnableDirectSubmission'] == '0' and
            target['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1' and 'LD_PRELOAD' not in target,
            'qualified target-only startup safety required')
        command = [assets['launcher']['path'], '--session', prof['session'],
            '--host-timing', '--device-timing', '--chrome-kernel-logging',
            '--chrome-call-logging', '--chrome-event-buffer-size', '128', binary['path'], *args]
        cells.append(dict(index=index, name=name, root=str(root), binary=binary,
            command=command, profile=prof, environment=target,
            expected_environment=expected_env, expected_kernels=expected))
    row = dict(stage='prepared', source_commit=revision, sources=sources,
        native_source_commit=built['source_commit'], binary=built['binaries']['strata'],
        binaries=built['binaries'], native_validation_probe=pair['native_validation_probe'],
        native_qualification_reuse=dict(pair['native_qualification_reuse'], current_controller_source=revision),
        collector_assets=assets, collector_reuse=audit,
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'), files=len(providers),
            sha256=digest(ROOT / 'runtime-inputs.json')), boot=pair['boot'], driver=pair['driver'],
        cells=cells, processes=[], complete=False, passed=False, active=False, ownership_closed=True,
        model_qualified=False, request_window_qualified=False, hardware_saturation_qualified=False,
        budgets=dict(wall=90, cpu=180, rss_cap=4 << 30, text_cap=8 << 20,
            file_cap=16 << 20, total_cap=64 << 20),
        plan=dict(question='Does the retained indexed PTI conserve service/copy/graph work with official2026.2 and these exact seven ELFs?',
            order='CPU canary,submission606kernels,native graph62kernels;fresh processes,commit each closure before next.',
            acceptance='Normal native exit,pre-main single-thread validation before collector init,collector retained '
                'through target body,exact immutable environment/providers,B570PCI05:00.0 trace lane,quiet validation '
                'and no fresh fault. All known program kernels/copies/fills/barriers and requested bytes must '
                'reconcile with the independently expanded UR inventory. Positive kernel service,unique IDs,'
                'one native host source per device record,and CLOCK_MONOTONIC_RAW containment+/-1ms.',
            graph_sinks='Graph-expanded records may lack per-node host sinks. Retain that count;complete numerical '
                'fixture and UR graph expansion qualify aggregate coverage only,not an individual dependency path.',
            source_work='submission=1+(1+119+120+121+241)+3=606;native=1+12+24+24+1=62. '
                'Capture/upload is not execution. Source tests/sycl/compat.cpp is in the original compiled inventory.',
            stopping='First source,owner,environment,warning,fault,budget,count,byte or clock failure stops;no retry '
                'or retrospective threshold/hash change. Session stop only after actual target closure.',
            limitations='No model attribution,geometry-tail reconstruction,clean speed,hardware ceiling/saturation '
                'or inference adoption follows. Copy counts are API-requested bytes,not wire traffic.',
            retention='Keep compact identities,individual pass/failure/count/service evidence and current decision. '
                'Raw timeline/UR/loader/JIT output is temporary for first collector coverage decision;retire after '
                'verification. Current64MiB-per-cell budget;Main owns cleanup.'))
    save(row)
    print('Prepared3same-generation collector checks;commit before execution.', flush=True)


def current():
    row = json.loads((ROOT / 'record.json').read_text())
    require(compact(row) == resources.build.committed(RESULT)[KEY], 'commit coverage plan/result first')
    require(matrix_shapes.sources(row['source_commit'], Path(__file__).resolve()) == row['sources'],
            'coverage source changed')
    built, providers, pair, assets, audit = basis(row['source_commit'])
    require(row['binaries'] == built['binaries'] and row['collector_assets'] == assets
            and row['collector_reuse'] == audit and row['boot'] == pair['boot']
            and row['native_validation_probe'] == pair['native_validation_probe'], 'coverage assets changed')
    manifest = row['runtime_manifest']
    require(digest(manifest['path']) == manifest['sha256'] and
            json.loads(Path(manifest['path']).read_text()) == providers, 'coverage provider basis changed')
    require(resources.build.guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'driver/boot changed')
    return row, providers


def inventory(events):
    out = collections.Counter()
    for event in events:
        if event.get('cat') != 'gpu_op' or event.get('ph') != 'X':
            continue
        name = event['name']
        kind = ('copy' if name.startswith('zeCommandListAppendMemoryCopy') else
            'fill' if name.startswith('zeCommandListAppendMemoryFill') else
            'barrier' if name == 'zeCommandListAppendBarrier' else
            'kernel' if 'main' in name else None)
        require(kind is not None, 'unclassified fixture device operation')
        out[kind] += 1
        if kind in ('copy', 'fill'):
            size = re.search(r'\[(\d+)\]$', name)
            require(size is not None, 'copy/fill size missing')
            out[kind + '_bytes'] += int(size[1])
    return dict(out)


def run(index):
    row, providers = current()
    require(row['stage'] in ('prepared', 'running') and len(row['processes']) == index
            and all(p['passed'] for p in row['processes']), 'fresh next ordered cell only')
    cell = row['cells'][index]
    root, prof = Path(cell['root']), cell['profile']
    session = trace.session_path(prof['session'])
    require(not session.exists() and not (root / 'owned').exists(), 'existing target/session;no retry')
    cursor = trace.cursor()
    result = dict(index=index, name=cell['name'], complete=False, host_probe_accepted=False,
        passed=False, expected_kernels=cell['expected_kernels'], native_executed=False)
    owner = controls = None
    def persist():
        write(root / 'execution.json', dict(result, commands=owner.commands if owner else [],
            controls=controls.commands if controls else [], active=bool((owner and owner.active) or (controls and controls.active))))
    seen = set()
    def interact(io, command):
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            try:
                pid = command['pid']
                if str(Path('/proc', str(pid), 'exe').resolve()) == str(Path(cell['binary']['path']).resolve()):
                    env = dict(v.split('=', 1) for v in Path('/proc', str(pid), 'environ').read_text().split('\0') if '=' in v)
                    trace.same_environment(env, cell['expected_environment'])
                    mapped = resources.entry.component.runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
                    require(mapped <= set(providers) | {str(Path(cell['binary']['path']).resolve()),
                            str(Path(prof['collector']['path']).resolve())}, 'unplanned target provider')
                    seen.update(mapped)
                    result['startup_environment_observed'] = True
            except (FileNotFoundError, ProcessLookupError):
                pass
            except PermissionError:
                require(io.exit_watch.poll(20), 'live target /proc access denied')
            if select.select([io.proc.stdout], [], [], .01)[0]:
                io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner = supervision.Owner(root / 'owned', persist)
        controls = supervision.Owner(root / 'controls', persist)
        begin = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        command, stdout, stderr = owner.run('target', cell['command'], cell['environment'], root,
            interaction=interact, **row['budgets'])
        end = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        require(supervision.closed(command) and command['exit_code'] == 0, 'collector target failed')
        text = stderr.read_text()
        proof = validation.preload_evidence(text, command['pid'])
        pinned = dict(preload=row['native_validation_probe'], collector=prof['collector'])
        if index == 0:
            resources.entry.component.analyze(dict(name='host-canary'), stdout.read_text(), command['pid'])
            for label, phase, path in (('validation', 'init', pinned['preload']['library']['path']),
                                      ('collector', 'init', prof['collector']['path']),
                                      ('fini', 'fini', prof['collector']['path'])):
                positions = [m.start() for m in re.finditer(r'calling '+phase+r':\s+([^\s]+)', text)
                             if Path(m[1]).resolve() == Path(path).resolve()]
                require(len(positions) == 1, 'ambiguous prefix loader marker:' + label)
                result[label + '_position'] = positions[0]
            require(result['validation_position'] < text.index('phase=before-main') <
                result['collector_position'] < text.index('strata UR canary begin') <
                text.index('strata UR canary end') < result['fini_position'], 'prefix constructor order differs')
            events = [event for file in (root / 'trace').rglob('*.json')
                      for event in json.loads(file.read_text())['traceEvents']]
            require(not any(e.get('cat') == 'gpu_op' and e.get('ph') == 'X' for e in events),
                    'CPU canary unexpectedly recorded GPU commands')
            result['GPU_commands'] = 0
        else:
            known.loader_order(text, command['pid'], pinned)
            footer = ('fixture_status=PASS mode=submission' if index == 1 else
                'SYCL host adapter: events, profiling, graph replay, copies and lifecycle passed')
            require(footer in stdout.read_text(), 'fixture correctness footer missing')
            files = list((root / 'trace').rglob('*.json'))
            require(len(files) == 1, 'one complete fixture trace required')
            events = json.loads(files[0].read_text())['traceEvents']
            capture = trace.trace_summary(events, command['pid'], begin, end)
            ur = known.ur_inventory((root / 'ur.log').read_text())
            device = inventory(events)
            require(device == ur['expanded'] and device['kernel'] == cell['expected_kernels'],
                'UR expansion/device operation/byte/kernel coverage differs')
            sinks = collections.Counter(str(e.get('id', '')) for e in events
                if e.get('pid') == command['pid'] and e.get('ph') == 't' and
                e.get('cat', '').startswith('Flow_D2H_'))
            unsunk = collections.Counter('barrier' if e['name'] == 'zeCommandListAppendBarrier' else 'nonbarrier'
                for e in events if e.get('cat') == 'gpu_op' and e.get('ph') == 'X'
                and sinks[str(e['args']['id'])] != 1)
            require(index != 1 or not unsunk['nonbarrier'], 'ordinary work lacks a host completion sink')
            result.update(capture=capture, UR_inventory=ur, device_inventory=device,
                missing_unique_host_sinks=dict(unsunk),
                timeline=dict(sha256=digest(files[0]), bytes=files[0].stat().st_size),
                native_raw_wall_ns=end-begin)
        require(result.get('startup_environment_observed') and
            {str(Path(a['path']).resolve()) for a in
             (pinned['preload']['loader'], pinned['preload']['library'], prof['collector'])} <= seen,
            'actual target environment/provider observation missing')
        require(not any(t in text for t in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_',
            '[WARNING]', '[ERROR]', 'terminate called', 'teardown uncertain')) and
            (not (root / 'validation.log').exists() or (root / 'validation.log').stat().st_size == 0),
            'collector/runtime/validation error or warning')
        trace.session_state(prof['session'], 0)
        require(all(digest(P / name) == sha for name, sha in row['sources'].items()) and
            all(digest(path) == sha for path, sha in providers.items()) and
            all(digest(item['path']) == item['sha256'] for item in
                [*row['binaries'].values(), *row['collector_assets'].values(), cell['binary']]),
            'post-run source/executable/provider changed')
        result.update(complete=True, host_probe_accepted=True, native_exit_code=0,
            native_executed=True, validation=proof, session_recording_from_exec=True)
    except BaseException as error:
        result.update(error=repr(error), complete=False, host_probe_accepted=False)
    finally:
        if owner and owner.active is None and all(supervision.closed(c) for c in owner.commands):
            try:
                if session.exists():
                    stopped, _, _ = controls.run('stop', [prof['launcher']['path'], '--stop', prof['session']],
                        cell['environment'], root, wall=15, cpu=15, rss_cap=2 << 30, text_cap=1 << 20,
                        file_cap=16 << 20, total_cap=64 << 20)
                    require(supervision.closed(stopped) and stopped['exit_code'] == 0, 'session stop failed')
                result['session_removed'] = not session.exists()
            except BaseException as error:
                result.update(cleanup_error=repr(error), session_removed=False)
        if owner:
            validation.finish(result, owner, cursor, row['boot'])
        controls_closed = bool(controls and controls.active is None and all(supervision.closed(c) for c in controls.commands))
        result['passed'] = bool(result.get('passed') and result['complete'] and result.get('session_removed') and controls_closed)
        result['ownership_closed'] = bool(result.get('ownership_closed') and controls_closed)
        persist()
        row['processes'].append(result)
        row.update(stage='running' if result['passed'] and index < 2 else 'closed' if result['passed'] else 'failed',
            complete=bool(result['passed'] and index == 2), passed=bool(result['passed'] and index == 2),
            active=bool((owner and owner.active) or (controls and controls.active)), ownership_closed=result['ownership_closed'])
        save(row)
    print(json.dumps({k:result.get(k) for k in ('name', 'passed', 'complete', 'ownership_closed', 'error', 'device_inventory')}), flush=True)
    require(result['passed'], 'same-generation collector coverage failed;do not continue')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resources.build.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']:
            prepare()
        elif len(sys.argv) == 3 and sys.argv[1] == 'run' and sys.argv[2] in ('0', '1', '2'):
            run(int(sys.argv[2]))
        else:
            raise RuntimeError('official_kernel_profile.py prepare|run0/1/2(separate arguments)')
