#!/usr/bin/env python3
"""Bounded current-source attribution of the original split prefill policy.

Repository LICENSE. Main-authored from this fork's owned comparison and
archived f9f3d12 profile controller; no XeStrata implementation input.
This observes service and host partitions, not physical resource saturation.
"""
import collections
import fcntl
import json
import math
import os
from pathlib import Path
import re
import secrets
import select
import socket
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import compare_selected_split as comparison
import engine_once

d, down = comparison.d, comparison.down
P, B, RESULT = comparison.P, comparison.B, comparison.RESULT
require, digest, write = comparison.require, comparison.digest, comparison.write
ROOT = B/'split-resource-profile-v2'
KEY, FRONTEND = 'split_resource_profile_v2', 'split_resource_frontend_v2'
HOST_KEY = 'split_resource_host_checks_v2'
# Exact launcher ELF/version, not a floating installed-tool preference.
UNITRACE_VERSION = '2.4.1 (4b6d74294cdc72766c00a9211fe2e4261ee94244)'


def session_path(session):
    require(re.fullmatch(r'[A-Za-z0-9]+', session) is not None, 'invalid owned session')
    return Path('/dev/shm')/('uctrl'+str(os.getuid())+'_'+session)


def session_state(session, expected):
    raw = session_path(session).read_bytes()
    require(len(raw) == 1024 and struct.unpack_from('=i', raw)[0] == expected,
            'collector recording state differs')


def clock_pair():
    before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    mono = time.monotonic_ns()
    after = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    return dict(raw_before_ns=before, monotonic_ns=mono, raw_after_ns=after)


def union_us(events):
    end = None
    total = 0.
    for start, finish in sorted((e['ts'], e['ts']+e['dur']) for e in events):
        total += finish-start if end is None or start > end else max(0., finish-end)
        end = finish if end is None else max(end, finish)
    return total


def trace_summary(events, native, begin_ns, end_ns):
    """Validate identity/time joins before publishing nonexclusive service."""
    require(type(events) is list and type(native) is int and native > 0 and
            type(begin_ns) is int and type(end_ns) is int and 0 < begin_ns < end_ns,
            'invalid trace/window identity')
    gpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'gpu_op']
    host = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'cpu_op']
    require(gpu and host, 'GPU/host events missing')
    require(all(type(e.get(k)) in (int, float) and math.isfinite(e[k]) for e in gpu+host
                for k in ('ts', 'dur')) and all(e['dur'] >= 0 for e in gpu+host),
            'invalid trace service/time')
    sources = collections.Counter(str(e.get('id', '')) for e in events
        if e.get('pid') == native and e.get('ph') == 's' and
        e.get('cat', '').startswith('Flow_H2D_'))
    labels = {e['pid']: e.get('args', {}).get('name', '') for e in events
              if e.get('ph') == 'M' and e.get('name') == 'process_name'}
    ids, names = [], {}
    counts, payload = collections.Counter(), collections.Counter()
    begin, end = begin_ns/1000, end_ns/1000
    for e in gpu:
        require(begin-1000 <= e['ts'] <= e['ts']+e['dur'] <= end+1000,
                'GPU event outside controlled raw-clock window')
        identifier = e.get('args', {}).get('id')
        require(type(identifier) in (int, str) and str(identifier) and
                sources[str(identifier)] == 1, 'missing/ambiguous native host source')
        ids.append(str(identifier))
        label = labels.get(e.get('pid'), '')
        require(label.startswith('DEVICE<') and '] 0:5:0:0 #' in label,
                'unexpected GPU PCI lane')
        name = e['name']
        kind = ('copy' if name.startswith('zeCommandListAppendMemoryCopy') else
                'fill' if name.startswith('zeCommandListAppendMemoryFill') else
                'barrier' if name.startswith('zeCommandListAppendBarrier') else
                'other_L0' if name.startswith('zeCommandList') else 'kernel')
        require(kind != 'kernel' or e['dur'] > 0, 'nonpositive kernel service')
        counts[kind] += 1
        if kind in ('copy', 'fill'):
            size = re.search(r'\[(\d+)\]$', name)
            require(size is not None, 'copy/fill payload unavailable')
            payload[name.rsplit('[', 1)[0]] += int(size[1])
        row = names.setdefault(name, dict(kind=kind, count=0, sum_us=0.))
        row['count'] += 1
        row['sum_us'] += e['dur']
    require(len(set(ids)) == len(ids), 'duplicate GPU work identity')
    require(all(e.get('pid') == native for e in host), 'foreign host API process')
    clipped = [dict(ts=max(begin, e['ts']), dur=min(end, e['ts']+e['dur'])-max(begin, e['ts']))
               for e in host if e['ts'] < end and e['ts']+e['dur'] > begin]
    return dict(gpu_events=len(gpu), host_events=len(host), operation_counts=dict(counts),
        observed_copy_fill_payload_bytes=dict(payload), gpu_union_us=union_us(gpu),
        host_api_union_in_window_us=union_us(clipped), controlled_request_wall_us=end-begin,
        largest_service_names=[dict(name=name, **row) for name, row in
            sorted(names.items(), key=lambda pair: pair[1]['sum_us'], reverse=True)[:12]],
        raw_window_containment=True, unique_gpu_ids=True, all_native_host_sources=True,
        limits='Nonexclusive service/copy payloads for the whole controlled request including<=24decode tokens. '
               'No exact native prefill endpoint,complete graph dependency,wire/DRAM byte or saturation claim.')


def profile_environment(root, rt):
    env = comparison.environment(root, rt, 'split')
    env.update(STRATA_TRACE='1', STRATA_PREFILL_CPU_WAIT_PROFILE='1')
    return env


def profile_target_environment(row, env, profiling):
    target = engine_once.native_environment(row, env)
    target['STRATA_UNITRACE_TARGET_PRELOAD'] = target.pop('LD_PRELOAD')
    target.update(profiling['environment'])
    return target


def expected_target_environment(row, env, profiling):
    target = profile_target_environment(row, env, profiling)
    target.update(LD_PRELOAD=profiling['collector']['path']+':'+row['native_validation_probe']['library']['path'],
        ZE_ENABLE_TRACING_LAYER='1', INTEL_LIBITTNOTIFY64='libunitrace_tool.so',
        UNITRACE_VERSION=UNITRACE_VERSION,
        UNITRACE_LD_PRELOAD_OLD=row['native_validation_probe']['library']['path'],
        UNITRACE_StartPaused='1', UNITRACE_HostTiming='1', UNITRACE_DeviceTiming='1',
        UNITRACE_ChromeKernelLogging='1', UNITRACE_ChromeCallLogging='1',
        UNITRACE_ChromeEventBufferSize='128', UNITRACE_Session=profiling['session'])
    return target


def same_environment(actual, expected):
    require(all(actual.get(k) == v for k, v in expected.items()) and
            set(actual)-set(expected) <= {'PWD', 'OLDPWD', 'SHLVL', '_'},
            'profile native startup environment differs')


def collector_assets(record):
    """Recheck actual native inputs; keep old controller refusal unchanged."""
    import build_indexed_collector as build
    from build_engine import blobs
    container = d.runtime.validation.committed(build.RESULT)
    built = container['unitrace_indexed_collector']
    graph = container['unitrace_indexed_graph']
    reuse = record['collector_reuse']
    for row in (built, graph):
        require(row['passed'] and row['complete'] and row['ownership_closed'] and not row['active'],
                'closed indexed collector/graph qualification required')
    require(graph['fresh_faults'] == [] and graph['service_coverage_qualified'] and
            graph['source_commit'] == reuse['graph_source'], 'collector graph identity differs')
    recovered = {n: build.tool.sha(raw) for n, raw in blobs(P, built['source_commit'], built['input_sha256'])}
    require(recovered == built['input_sha256'], 'original collector source unavailable')
    audited = reuse['current_source_audit']
    changes = []
    for name, original in recovered.items():
        current = digest(P/name)
        if current != original:
            changes.append(dict(path=name, original_sha256=original, current_sha256=current))
    require(changes == audited['host_only_changes'], 'unreviewed collector source difference')
    require(build.command(container) == built['command'], 'collector compile recipe changed')
    for name, raw in build.generated().items():
        require(build.tool.sha(raw) == built['generated_sha256'][name] and
                (build.ROOT/name).read_bytes() == raw, 'generated native collector input changed')
    manifest = built['compiler_inputs']
    require(digest(manifest['path']) == manifest['sha256'] == reuse['dependencies']['sha256'],
            'collector dependency manifest differs')
    inputs = json.loads(Path(manifest['path']).read_text())
    require(len(inputs) == manifest['files'] == 460 and
            all(digest(x['path']) == x['sha256'] for x in inputs), 'collector compiler input changed')
    compiled = {str(Path(x['path']).resolve()) for x in inputs}
    require(all(x['path'].endswith('.py') and str((P/x['path']).resolve()) not in compiled
                for x in changes), 'changed input belongs to native collector')
    assets = reuse['assets']
    require(assets['launcher'] == built['artifacts']['unitrace'] == graph['assets']['launcher'] and
            assets['collector'] == built['artifacts']['libunitrace_tool.so'] == graph['assets']['collector'],
            'collector/launcher selection differs')
    for item in assets.values():
        require(digest(item['path']) == item['sha256'] and Path(item['path']).stat().st_size == item['bytes'],
                'collector/launcher bytes changed')
    return assets


def profiling(root, assets):
    return dict(mode='trace', **assets, session='StrataSplit'+secrets.token_hex(12),
        environment=dict(UNITRACE_SystemTime='1', UNITRACE_FollowChildProcess='1',
                         UNITRACE_TraceOutputDir=str(root/'trace'), LD_DEBUG='libs'))


def prepare():
    record, scope, built, _, rt = comparison.native()
    continuation = scope[comparison.CONT_KEY]
    require(continuation['passed'] and continuation['complete'] and continuation['ownership_closed'] and
            not continuation['active'], 'candidate continuation must close first')
    host, _ = down.frontend(root=ROOT, key=FRONTEND)
    checks = scope[HOST_KEY]
    require(checks['passed'] and all(digest(P/n) == v for n, v in checks['sources'].items()),
            'current profile host controls required')
    require(KEY not in scope and not (ROOT/'record.json').exists(), 'fresh profile only')
    assets = collector_assets(record)
    roots = [ROOT/'prefix', ROOT/'model']
    for root in roots:
        require(not root.exists(), 'fresh private profile roots required')
        root.mkdir(mode=0o700)
        for part in ('tmp', 'cache', 'trace'):
            (root/part).mkdir()
    prefix, model = roots
    probe = down.native_probe(rt)
    config = down.configuration(model)
    source = json.loads((comparison.ROOT/'record.json').read_text())
    model_assets = d.models.model_pins(config, source['model_assets'])
    prof = profiling(model, assets)
    native_plan = dict(binary=built['binaries']['strata'], native_validation_probe=probe, native_profile=prof)
    write(model/'config.json', config)
    write(model/'plan.json', native_plan)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    command = ['/bin/sh', str(P/'bench.sh'), '--config', str(model/'config.json'), '--port', str(port),
        '--prefill-exact-tokens', str(comparison.EXACT), '--think', '--runs', '1', '--profile-request-gates',
        '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600',
        '--out', str(model/'bench')]
    clean = [p['samples'][0]['prompt_ms'] for p in scope[comparison.KEY]['processes'] if p['arm'] == 'split']
    require(len(clean) == 3 and all(math.isfinite(v) and v > 0 for v in clean), 'clean first-use basis missing')
    env_binary = Path('/usr/bin/env')
    row = dict(stage='prepared', source_commit=host['source_commit'], sources=host['sources'],
        model_assets=model_assets, asset_basis=dict(path=str(comparison.ROOT/'record.json'),
            sha256=digest(comparison.ROOT/'record.json')), runtime_sha256=digest(d.runtime.RESULT), boot=rt['boot'],
        binary=built['binaries']['strata'], native_source_commit=built['source_commit'], native_validation_probe=probe,
        native_profile=prof, prefix_profile=profiling(prefix, assets), prefix_binary=dict(path=str(env_binary),
            sha256=digest(env_binary), bytes=env_binary.stat().st_size), command=command,
        profile=dict(path=str(P/'data/expert-profile.bin'), sha256=digest(P/'data/expert-profile.bin')),
        generated={n:digest(model/n) for n in ('config.json', 'plan.json')},
        environment_sha256=down.fingerprint(profile_environment(model, rt)),
        request_sha256=scope[comparison.KEY]['request_sha256'][0],
        clean_first_use_ms=clean, maximum_observer_ratio=1.15,
        budgets=dict(wall=1800, cpu=7200, rss_cap=104<<30, text_cap=64<<20,
                     file_cap=1024<<20, total_cap=1536<<20),
        plan=dict(question='Which GPU kernels/copies and disjoint main-thread CPU/wait regions dominate the current117K split request?',
            constants='One117489-read/zero-reuse request,117488BF16GPU rows,128prefill slots/48windows,'
                '8192chunks,1256decode slots,5workers,spec4,INT8 resident131072context and original Down. '
                'Same body as the clean cohort request0;thinking and<=24outputs.',
            observers='Exact indexed UnitTrace assets,non-counter trace;quiet parameter validation,STRATA_TRACE1 '
                'and existing CPU_WAIT_PROFILE1. Command-service profiling stays off. CPU-only prefix/environment '
                'probe must pass and be committed before the single model process.',
            acceptance='Exact source/runtime/assets/owner/device/env/route/body/count/clock checks,valid CPUv4 '
                'partition,unique GPU/host-source IDs and contained raw-clock services,normal closure,quiet '
                'validation and no fresh fault. Report first-use prompt-time ratio against median of3clean '
                'first-use samples;ratio<=1.15 is required to use this observation for tuning selection.',
            raw_budget_reason='Whole-request host/device identity and service joins require the complete temporary '
                'timeline. Prior32K trace used266967750bytes;117K may exceed512MiB. Bound timeline at1GiB '
                'and job total1.5GiB;Main extracts aggregates and retires raw output at attribution closure.',
            limits='Attribution only. No exact native prefill endpoint,complete graph dependency,physical '
                'PCIe/DRAM bytes,hardware saturation,clean speed,decode/task/agent/full262144 or adoption claim.',
            stopping='First admission/owner/source/validation/fault/budget/body/route/count/clock/overhead failure '
                'stops;no retry,hash refresh,replacement or changed threshold. Discard a failed session.'),
        complete=False, passed=False, active=False, ownership_closed=True)
    write(ROOT/'record.json', row)
    scope[KEY] = comparison.compact(row)
    write(RESULT, record)
    print('Prepared prefix-first current-source profile;commit before CPU probe.', flush=True)


def basis():
    record, scope, built, _, rt = comparison.native()
    host, manifest = down.frontend(root=ROOT, key=FRONTEND)
    row = json.loads((ROOT/'record.json').read_text())
    require(comparison.compact(row) == scope[KEY] and row['sources'] == host['sources'] and
            d.pins(row['source_commit']) == row['sources'], 'profile source/record differs')
    assets = collector_assets(record)
    require(row['binary'] == built['binaries']['strata'] and row['native_source_commit'] == built['source_commit'] and
            row['runtime_sha256'] == digest(d.runtime.RESULT) and row['boot'] == rt['boot'] and
            row['native_validation_probe'] == down.native_probe(rt), 'profile native/runtime changed')
    require(all(row['native_profile'][k] == v == row['prefix_profile'][k] for k, v in assets.items()),
            'profile asset selection differs')
    require(digest(row['prefix_binary']['path']) == row['prefix_binary']['sha256'] and
            digest(row['asset_basis']['path']) == row['asset_basis']['sha256'] and
            digest(row['profile']['path']) == row['profile']['sha256'], 'profile asset/basis changed')
    require(all(digest(ROOT/'model'/n) == v for n, v in row['generated'].items()) and
            down.fingerprint(profile_environment(ROOT/'model', rt)) == row['environment_sha256'],
            'profile generated input/environment changed')
    d.models.model_pins(json.loads((ROOT/'model/config.json').read_text()), row['model_assets'])
    return record, scope, row, rt, manifest


def cursor():
    values = re.findall(r'^-- cursor: (.+)$', subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15), re.M)
    require(len(values) == 1, 'fresh fault cursor missing')
    return values[0]


def prefix():
    record, scope, row, rt, _ = basis()
    require(row['stage'] == 'prepared' and 'prefix' not in row, 'fresh committed prefix only')
    root = ROOT/'prefix'
    prof = row['prefix_profile']
    require(not session_path(prof['session']).exists(), 'preexisting prefix session')
    env = profile_environment(root, rt)
    target = profile_target_environment(row, env, prof)
    start_cursor = cursor()
    outcome = dict(complete=False, passed=False, host_probe_accepted=False, native_engine_executed=False)
    owner = d.runtime.supervision.Owner(root/'owned')
    controls = d.runtime.supervision.Owner(root/'controls')
    try:
        argv = engine_once.profile_command(prof, [row['prefix_binary']['path']])
        entry, out, err = owner.run('prefix', argv, target, P, wall=60, cpu=60, rss_cap=2<<30,
            text_cap=2<<20, file_cap=4<<20, total_cap=16<<20)
        require(d.runtime.supervision.closed(entry) and entry['exit_code'] == 0, 'prefix process failed')
        actual = dict(line.split('=', 1) for line in out.read_text().splitlines())
        same_environment(actual, expected_target_environment(row, env, prof))
        text = err.read_text()
        proof = d.runtime.validation.preload_evidence(text, entry['pid'])
        marker = [m.start() for m in re.finditer(r'calling init:\s+'+re.escape(prof['collector']['path'])+r'\s', text)]
        require(len(marker) == 1 and text.index('phase=before-main') < marker[0] and
                proof['loader'] == row['native_validation_probe']['loader']['path'] and
                not any(t in text for t in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called')),
                'prefix constructor/validation evidence differs')
        session_state(prof['session'], 1)
        outcome.update(complete=True, host_probe_accepted=True, environment_checked=True,
                       validation=proof, native_exit_code=0)
    except BaseException as error:
        outcome.update(error=repr(error), complete=False, host_probe_accepted=False)
    finally:
        if owner.active is None and all(d.runtime.supervision.closed(c) for c in owner.commands):
            try:
                if session_path(prof['session']).exists():
                    stop, _, _ = controls.run('stop', [prof['launcher']['path'], '--stop', prof['session']],
                        env, root, wall=15, cpu=15, rss_cap=2<<30, text_cap=1<<20,
                        file_cap=4<<20, total_cap=16<<20)
                    require(d.runtime.supervision.closed(stop) and stop['exit_code'] == 0, 'prefix stop failed')
                outcome['session_removed'] = not session_path(prof['session']).exists()
            except BaseException as error:
                outcome.update(cleanup_error=repr(error), session_removed=False)
        d.runtime.validation.finish(outcome, owner, start_cursor, row['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'] and outcome.get('session_removed') and
            controls.active is None and all(d.runtime.supervision.closed(c) for c in controls.commands))
        row.update(prefix=outcome, stage='prefix-passed' if outcome['passed'] else 'failed',
            active=bool(owner.active or controls.active), ownership_closed=bool(outcome.get('ownership_closed') and
                controls.active is None and all(d.runtime.supervision.closed(c) for c in controls.commands)))
        write(ROOT/'record.json', row)
        scope[KEY] = comparison.compact(row)
        write(RESULT, record)
    print(json.dumps(outcome), flush=True)
    require(outcome['passed'], 'prefix did not qualify;no model launch')


def run():
    record, scope, row, rt, manifest = basis()
    require(row['stage'] == 'prefix-passed' and row['prefix']['passed'] and
            'observation' not in row, 'committed passing prefix and fresh model required')
    root = ROOT/'model'
    prof = row['native_profile']
    require(not session_path(prof['session']).exists() and not (root/'execution.json').exists(),
            'preexisting model/session;no retry')
    env = profile_environment(root, rt)
    outcome = dict(complete=False, passed=False, host_probe_accepted=False,
        clean_speed=False, adoption_qualified=False, hardware_saturation_known=False)
    start_cursor = cursor()
    owner = controls = None
    extent = {}
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def persist():
        write(root/'execution.json', dict(outcome,
            active=bool((owner and owner.active) or (controls and controls.active)),
            commands=owner.commands if owner else [], controls=controls.commands if controls else []))
    def control(action, expected=None):
        entry, _, _ = controls.run(action, [prof['launcher']['path'], '--'+action, prof['session']],
            env, root, wall=15, cpu=15, rss_cap=2<<30, text_cap=1<<20,
            file_cap=row['budgets']['file_cap'], total_cap=row['budgets']['total_cap'])
        require(d.runtime.supervision.closed(entry) and entry['exit_code'] == 0, 'profile session control failed')
        if expected is not None:
            session_state(prof['session'], expected)
    def interact(io, entry):
        identity = None
        for stage in ('begin', 'end'):
            while True:
                line = io.line(1200)
                if line.startswith('STRATA_PROFILE_GATE '):
                    break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                    dict(stage=stage, phase='prefill-only', run=0), 'unexpected profile request gate')
            if stage == 'end':
                outcome.update(request_raw_end_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                               clock_end=clock_pair())
                control('pause', 1)
            delta = [a-b for a, b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
            require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
            actual = json.loads((root/'engine-start.json').read_text())['native']
            pid = actual['pid']
            live = d.runtime.supervision.identity(pid)
            require(live and live['start_ticks'] == actual['start_ticks'] and live['sid'] == entry['pid'] and
                    str(Path('/proc', str(pid), 'exe').resolve()) == str(Path(row['binary']['path']).resolve()) and
                    (identity is None or actual == identity), 'profile owned native differs')
            identity = actual
            namespace = os.readlink('/proc/'+str(pid)+'/ns/time')
            require(namespace == os.readlink('/proc/self/ns/time'), 'profile time namespace differs')
            actual_env = dict(v.split('=', 1) for v in Path('/proc', str(pid), 'environ').read_text().split('\0') if '=' in v)
            same_environment(actual_env, expected_target_environment(row, env, prof))
            mapped = d.runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
            require(mapped <= set(rt['runtime_files']) | {str(Path(row['binary']['path']).resolve()),
                    str(Path(prof['collector']['path']).resolve())} and
                    {row['native_validation_probe']['loader']['path'], row['native_validation_probe']['library']['path'],
                     prof['collector']['path'], env['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'profile provider differs')
            outcome['physical_device'] = down.device.device.live_device(pid, 'Intel(R) Arc(TM) B570 Graphics',
                down.device.device.device_inventory(rt), rt)
            allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
            d.python_runtime.check_running(entry['pid'], P/'tools/strata_bench.py', manifest['selected'],
                env, allowed, row['command'][2:])
            require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size == 0,
                    'profile parameter validation warning/error')
            outcome.update(native_identity=actual, time_namespace=namespace, model_executed=True)
            extent[stage] = (root/'engine.stderr').stat().st_size
            if stage == 'begin':
                session_state(prof['session'], 1)
                control('resume', 0)
                outcome.update(clock_begin=clock_pair(),
                               request_raw_begin_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
            print('Split resource request0 '+stage, flush=True)
            io.send(b'continue\n')
        io.proc.stdin.close()
        deadline = time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'profile benchmark teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]:
                io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner = d.runtime.supervision.Owner(root/'owned', persist)
        controls = d.runtime.supervision.Owner(root/'controls', persist)
        entry, _, _ = owner.run('bench', row['command'], env, P, interaction=interact, **row['budgets'])
        require(d.runtime.supervision.closed(entry) and entry['exit_code'] == 0, 'profile benchmark failed')
        files = list((root/'bench').glob('*/results.json'))
        require(len(files) == 1, 'missing/duplicate profile official result')
        result = json.loads(files[0].read_text())
        cohort = result['results']['prefill_only']
        require(result['complete'] and not result['results']['failures'] and
                len(cohort['requests']) == len(cohort['count_preflight']) == 1, 'incomplete profile request')
        sample, preflight = cohort['requests'][0], cohort['count_preflight'][0]
        require(all(sample.get(k) is True for k in ('ok', 'eligible', 'stream_complete', 'thinking_requested', 'reasoning_observed')) and
                sample['request_sha256'] == preflight['body_sha256'] == row['request_sha256'] and
                preflight['counted_input_tokens'] == sample['actual_prompt_tokens'] ==
                sample['actual_read_tokens'] == comparison.EXACT and sample['actual_reused_tokens'] == 0 and
                1 <= sample['generated_tokens'] <= 24 and sample['prompt_tps_source'] == 'engine' and
                type(sample['prompt_ms']) in (int, float) and math.isfinite(sample['prompt_ms']) and
                sample['prompt_ms'] > 10000, 'profile body/count/clock differs')
        journal = d.client.verify_request_journal(files[0].parent)
        require(journal['valid'] and journal['complete'] and not journal['pending'], 'profile request journal incomplete')
        basis()
        end = d.gen.native_end(root, row, dict(config=json.loads((root/'config.json').read_text())), row['boot'], entry['pid'])
        raw = (root/'engine.stderr').read_bytes()
        text = raw.decode()
        require([x for x in text.splitlines() if x.startswith('[native16-down] ')] ==
                ['[native16-down] fusion=large min_rows=2334 split_kernel=original'] and
                not any(t in text for t in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called',
                    'teardown uncertain', '[WARNING]', '[ERROR]')), 'profile policy/runtime/collector warning')
        proof = d.runtime.validation.preload_evidence(text, end['native']['pid'])
        part = raw[extent['begin']:extent['end']].decode()
        route = down.route_samples(part, 1)[0]
        cpu = d.gen.common.extract_cpu_observers(part, 4)
        traces = list((root/'trace').rglob('*.json'))
        require(len(traces) == 1, 'one complete profile timeline required')
        value = json.loads(traces[0].read_text())
        capture = trace_summary(value['traceEvents'], end['native']['pid'],
            outcome['request_raw_begin_ns'], outcome['request_raw_end_ns'])
        capture.update(timeline_sha256=digest(traces[0]), timeline_bytes=traces[0].stat().st_size)
        median = comparison.statistics.median(row['clean_first_use_ms'])
        ratio = sample['prompt_ms']/median
        outcome.update(sample=sample, route=route, cpu_partition=cpu, capture=capture,
            observer_time_ratio=ratio, observer_cost_qualified=ratio <= row['maximum_observer_ratio'],
            native_exit_code=0, validation=proof)
        require(outcome['observer_cost_qualified'], 'profile observer exceeded frozen time ratio')
        outcome.update(complete=True, host_probe_accepted=True)
    except BaseException as error:
        outcome.update(error=repr(error), complete=False, host_probe_accepted=False)
    finally:
        if owner and owner.active is None and all(d.runtime.supervision.closed(c) for c in owner.commands):
            try:
                if session_path(prof['session']).exists():
                    control('stop')
                outcome['session_removed'] = not session_path(prof['session']).exists()
            except BaseException as error:
                outcome.update(cleanup_error=repr(error), session_removed=False)
        if owner:
            d.runtime.validation.finish(outcome, owner, start_cursor, row['boot'])
        controls_closed = bool(controls and controls.active is None and
            all(d.runtime.supervision.closed(c) for c in controls.commands))
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'] and
            outcome.get('session_removed') and controls_closed)
        outcome['ownership_closed'] = bool(outcome.get('ownership_closed') and controls_closed)
        persist()
        row.update(observation=outcome, stage='closed' if outcome['passed'] else 'failed',
            passed=outcome['passed'], complete=outcome['complete'],
            active=bool((owner and owner.active) or (controls and controls.active)),
            ownership_closed=outcome['ownership_closed'])
        write(ROOT/'record.json', row)
        scope[KEY] = comparison.compact(row)
        write(RESULT, record)
    print(json.dumps({k:outcome.get(k) for k in ('passed', 'complete', 'ownership_closed',
        'fresh_faults', 'observer_time_ratio', 'error')}), flush=True)
    require(outcome['passed'], 'current split resource attribution did not qualify')


if __name__ == '__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args = sys.argv[1:]
        if args == ['prepare-frontend']:
            comparison.native(); down.prepare_frontend(ROOT, FRONTEND)
        elif args == ['frontend']:
            comparison.native(); down.run_frontend(ROOT, FRONTEND)
        elif args == ['prepare']:
            prepare()
        elif args == ['prefix']:
            prefix()
        elif args == ['run']:
            run()
        else:
            raise RuntimeError('prepare-frontend|frontend|prepare|prefix|run')
