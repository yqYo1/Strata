#!/usr/bin/env python3
"""Qualify service coverage of graphs captured/warmed while Unitrace is paused."""
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import production_profile as production

probe = production.probe
tool = probe.tool
P, B, RESULT = tool.P, tool.B, tool.RESULT
ROOT = B / 'unitrace-warmed-graph-window-v1'
KEY = 'unitrace_warmed_graph_window'
SOURCE = 'bench/sycl/graph_record_window.cpp'
NATIVE = ('src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inputs(revision):
    pins = production.inputs(revision)
    names = [SOURCE, 'tools/sycl/graph_record_window.py', 'tests/sycl/graph_envelope.cpp', *NATIVE]
    names += tool.git(P, 'ls-files', 'include').decode().splitlines()
    for name in names:
        original = tool.git(P, 'show', revision + ':' + name)
        tool.guards.require(original == (P/name).read_bytes(), 'commit exact input: ' + name)
        pins[name] = tool.sha(original)
    return pins


def compiler_inputs(compiler):
    install = Path(compiler).parents[1]
    paths = set()
    for root in (install/'bin', install/'include', install/'lib', Path('/usr/include'),
                 Path('/usr/lib/gcc/x86_64-linux-gnu/13/include')):
        paths.update(p.resolve() for p in root.rglob('*') if p.is_file())
    paths.update(Path(p).resolve() for p in ('/usr/bin/ld', '/usr/bin/as',
        '/usr/lib/x86_64-linux-gnu/libc.so.6', '/usr/lib/x86_64-linux-gnu/libm.so.6',
        '/usr/lib/x86_64-linux-gnu/libstdc++.so.6', '/usr/lib/x86_64-linux-gnu/libgcc_s.so.1',
        '/usr/lib/x86_64-linux-gnu/crt1.o', '/usr/lib/x86_64-linux-gnu/crti.o',
        '/usr/lib/x86_64-linux-gnu/crtn.o'))
    return [dict(path=str(p), sha256=digest(p)) for p in sorted(paths)]


def assets(container):
    return probe.pin_assets(container)


def admission(record, container, binary=False):
    tool.guards.require(inputs(record['source_commit']) == record['input_sha256'], 'source changed')
    tool.guards.require(assets(container) == record['assets'], 'runtime/native/collector assets changed')
    manifest = ROOT/'compiler-inputs.json'
    tool.guards.require(digest(manifest) == record['compiler_inputs']['sha256'], 'compiler manifest changed')
    rows = json.loads(manifest.read_text())
    tool.guards.require(len(rows) == record['compiler_inputs']['files'], 'compiler manifest count changed')
    for row in rows:
        tool.guards.require(digest(row['path']) == row['sha256'], 'compiler/header changed: ' + row['path'])
    if binary:
        tool.guards.require(digest(record['binary']['path']) == record['binary']['sha256'], 'binary changed')


def session_state(path, expected):
    raw = path.read_bytes()
    tool.guards.require(len(raw) == 1024 and struct.unpack_from('=i', raw)[0] == expected,
                        'collector session state differs')


def trace_evidence(root, expected, begin, end, pid, driver):
    files = list((root/'trace').rglob('*.json'))
    tool.guards.require(len(files) <= 1, 'multiple collector timelines')
    events = []
    raw_identity = None
    if files:
        raw = files[0].read_bytes()
        raw_identity = dict(sha256=tool.sha(raw), bytes=len(raw))
        parsed = json.loads(raw)
        events = parsed['traceEvents'] if isinstance(parsed, dict) else parsed
    gpu = [e for e in events if e.get('cat') == 'gpu_op' and e.get('ph') == 'X']
    kernels = [e for e in gpu if 'RecordWindow' in e.get('name', '')]
    tool.guards.require(len(kernels) == expected, 'warmed graph kernel loss or recording leakage')
    tool.guards.require(len(gpu) == len(kernels), 'unclassified device operation in recorded window')
    names = collections.Counter(e['name'] for e in kernels)
    if expected:
        tool.guards.require(len(names) == 2 and sorted(names.values()) == [3, 3]
            and any('RecordWindowAdd' in n for n in names)
            and any('RecordWindowMultiply' in n for n in names), 'per-kernel replay counts differ')
    sources = collections.Counter(str(e.get('id', '')) for e in events
        if e.get('pid') == pid and e.get('ph') == 's' and e.get('cat', '').startswith('Flow_H2D_'))
    sinks = collections.Counter(str(e.get('id', '')) for e in events
        if e.get('pid') == pid and e.get('ph') == 't' and e.get('cat', '').startswith('Flow_D2H_'))
    ids = []
    for e in kernels:
        tool.guards.require(all(isinstance(e.get(k), (float, int)) and math.isfinite(e[k])
            for k in ('ts', 'dur')) and e['dur'] > 0, 'invalid kernel service')
        tool.guards.require(begin/1000-1000 <= e['ts'] <= e['ts']+e['dur'] <= end/1000+1000,
                            'kernel outside controlled raw-clock window +/-1ms')
        ident = str(e.get('args', {}).get('id', ''))
        tool.guards.require(ident and sources[ident] == 1, 'kernel lost its unique host source')
        ids.append(ident)
    tool.guards.require(len(set(ids)) == len(ids), 'duplicate device identity')
    pci = re.fullmatch(r'([0-9a-f]+):([0-9a-f]+):([0-9a-f]+)\.([0-9a-f]+)', Path(driver['pci']).name)
    tool.guards.require(pci is not None, 'invalid pinned PCI address')
    address = tuple(int(v, 16) for v in pci.groups())
    labels = {e['pid']: e.get('args', {}).get('name', '') for e in events
              if e.get('ph') == 'M' and e.get('name') == 'process_name'}
    for lane in {e['pid'] for e in kernels}:
        label = labels.get(lane, '')
        match = re.search(r'\] ([0-9a-f]+):([0-9a-f]+):([0-9a-f]+):([0-9a-f]+) #\d+$', label)
        tool.guards.require(label.startswith('DEVICE<') and match
            and tuple(int(v, 16) for v in match.groups()) == address, 'wrong device lane')
    return dict(timeline=raw_identity, program_kernels=len(kernels), names=dict(names),
        kernel_service_us=sum(e['dur'] for e in kernels), kernel_union_us=production.union_us(kernels),
        unique_host_sources=True, unsunk_kernels=sum(sinks[i] != 1 for i in ids),
        raw_window_contained=True,
        limits='Source/replay counts and service only. Does not clear the earlier failed graph host-sink gate, qualify critical path, observer overhead, model-stage mapping or hardware saturation.')


def check(record, container):
    tool.guards.require(record['stage'] == 'check-prepared' and not record.get('observations'),
                        'fresh committed check plan required')
    admission(record, container, True)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
                                     text=True, timeout=15)
    cursor = next(line.split(': ', 1)[1] for line in journal.splitlines() if line.startswith('-- cursor: '))
    record['driver_before'] = tool.guards.driver_guard()
    target = control = None

    def persist():
        record['active'] = any(owner and owner.active is not None for owner in (target, control))
        (ROOT/'check-record.json').write_text(json.dumps(dict(record,
            target_commands=target.commands if target else [], control_commands=control.commands if control else []),
            indent=2) + '\n')

    target = tool.supervision.Owner(ROOT/'target-owned', persist)
    control = tool.supervision.Owner(ROOT/'control-owned', persist)
    record.update(stage='checking', observations=[], passed=False)
    persist()
    try:
        for cell in record['cells']:
            root = ROOT/cell['name']
            session = Path('/dev/shm/uctrl'+str(os.getuid())+'_'+cell['session']) if cell['session'] else None
            tool.guards.require(session is None or not session.exists(), 'preexisting session; inspect, no retry')
            row = dict(name=cell['name'], passed=False, controls=[])
            record['observations'].append(row)
            entry = None

            def change(action, expected=None):
                result, _, _ = control.run(cell['name']+'-'+action,
                    [record['assets']['launcher']['path'], '--'+action, cell['session']],
                    cell['environment'], root, wall=15, cpu=15, rss_cap=2 << 30,
                    text_cap=1 << 20, file_cap=16 << 20, total_cap=128 << 20)
                tool.guards.require(tool.supervision.closed(result) and result['exit_code'] == 0, 'control failed')
                if expected is not None:
                    session_state(session, expected)
                row['controls'].append(action)

            def interact(io, child):
                ready = re.fullmatch(r'graph_window phase=ready pid=(\d+) raw_ns=(\d+)', io.line(30))
                tool.guards.require(ready and int(ready[1]) == child['pid'], 'ready owner/protocol mismatch')
                tool.guards.require(os.readlink('/proc/self/ns/time') == os.readlink('/proc/'+str(child['pid'])+'/ns/time'),
                                    'native/caller time namespaces differ')
                row.update(pid=child['pid'], ready_raw_ns=int(ready[2]))
                if session is not None:
                    session_state(session, 1)
                    if cell['name'] == 'window':
                        change('resume', 0)
                row['release_raw_ns'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                io.send(b'continue\n')
                drained = re.fullmatch(r'graph_window phase=drained pid=(\d+) raw_ns=(\d+)', io.line(30))
                tool.guards.require(drained and int(drained[1]) == child['pid'], 'drain owner/protocol mismatch')
                row['drained_raw_ns'] = int(drained[2])
                if session is not None:
                    if cell['name'] == 'window':
                        change('pause', 1)
                    else:
                        session_state(session, 1)
                io.send(b'continue\n')
                io.proc.stdin.close()
                io.exit(20)

            try:
                entry, out, err = target.run(cell['name'], cell['command'], cell['environment'], root,
                    interaction=interact, wall=90, cpu=90, rss_cap=4 << 30, text_cap=8 << 20,
                    file_cap=32 << 20, total_cap=128 << 20)
                tool.guards.require(tool.supervision.closed(entry) and entry['exit_code'] == 0, 'fixture exit/closure failed')
                text = out.read_text()
                tool.guards.require('graph_window PASS warm_replays=2 window_replays=3 kernels_per_replay=2'
                    ' numerical=PASS guards=PASS checked_teardown=PASS' in text, 'missing numerical/owner footer')
                stderr = err.read_text()
                row['validation'] = probe.validation.preload_evidence(stderr, entry['pid'])
                # A dlsym error-looking line can be nonfatal; retain it and fail
                # this new qualification until its actual origin is resolved.
                lookup = [line for line in stderr.splitlines() if 'symbol lookup error' in line]
                row['loader_lookup_errors'] = lookup
                tool.guards.require(not lookup, 'unresolved loader symbol diagnostic')
                inventory = production.ur_inventory((root/'ur.log').read_text())
                tool.guards.require(inventory['graph_replays'] == 5 and inventory['capture_regions'] == 1
                    and inventory['replayed'].get('kernel') == 10 and inventory['direct'].get('kernel', 0) == 0,
                    'actual capture/replay inventory differs')
                row['inventory'] = inventory
                if session is not None:
                    row['coverage'] = trace_evidence(root, cell['expected_kernels'], row['release_raw_ns'],
                        row['drained_raw_ns'], entry['pid'], record['driver_before'])
                row['execution_passed'] = row['passed'] = True
            finally:
                row['target_closed'] = bool(entry and tool.supervision.closed(entry))
                if session is not None and row['target_closed'] and target.active is None and control.active is None:
                    if session.exists():
                        change('stop')
                    row['session_removed'] = not session.exists()
                else:
                    row['session_removed'] = session is None
                persist()
            tool.guards.require(row['passed'] and row['session_removed'], 'cell failed; no next cell')
        record.update(stage='closed', passed=True, complete=True, service_coverage_qualified=True)
    except BaseException as error:
        record.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        record['ownership_closed'] = all(owner.active is None and all(tool.supervision.closed(c) for c in owner.commands)
                                         for owner in (target, control))
        try:
            record.update(production.fault_evidence(cursor, boot))
        except Exception as error:
            record['fault_probe_error'] = repr(error)
        record['passed'] = bool(record['passed'] and record['ownership_closed'] and record.get('fresh_faults') == []
                                and record.get('boot_unchanged') and 'fault_probe_error' not in record)
        record['service_coverage_qualified'] = record['passed']
        persist()


def main(action):
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tool.guards.no_model_owners([])
        container = json.loads(RESULT.read_text())
        record = container.get(KEY)
        if action == 'prepare':
            tool.guards.require(record is None and not ROOT.exists(), 'inspect existing attempt; no overwrite')
            revision = tool.git(P, 'rev-parse', 'HEAD').decode().strip()
            pins = inputs(revision)
            pinned = assets(container)
            current = json.loads((B/'main-native-gdn-engine/record.json').read_text())
            compiler = current['compiler']
            tool.guards.require(digest(compiler['path']) == compiler['sha256'], 'compiler changed')
            ROOT.mkdir(mode=0o700)
            (ROOT/'tmp').mkdir()
            manifest = ROOT/'compiler-inputs.json'
            rows = compiler_inputs(compiler['path'])
            manifest.write_text(json.dumps(rows, separators=(',', ':')) + '\n')
            install = Path(compiler['path']).parents[1]
            command = [compiler['path'], '-std=c++20', '-O2', '-g0', '-fsycl', '-fsycl-id-queries-range=size_t',
                '-fno-fast-math', '-ffp-contract=off', '-DSTRATA_ENABLE_SYCL=1', '-I'+str(P/'include'),
                '-I'+str(P/'include/strata/sycl/compat'), *[str(P/n) for n in NATIVE], str(P/SOURCE),
                '-pthread', '-ldl', '-o', str(ROOT/'graph_record_window')]
            env = dict(PATH=str(install/'bin')+':/usr/bin:/bin', LANG='C', LC_ALL='C', TMPDIR=str(ROOT/'tmp'),
                       LD_LIBRARY_PATH=str(install/'lib'), PYTHONDONTWRITEBYTECODE='1')
            legacy = json.loads((P/'bench/results/2026-10-15-pr6-recovery-audit/native-gdn-engine-build.json').read_text())
            hold = legacy['legacy_provenance_exception']
            record = dict(schema='strata-unitrace-warmed-graph-window-v1', stage='build-prepared',
                source_commit=revision, input_sha256=pins, assets=pinned, compiler=compiler,
                compiler_inputs=dict(path=str(manifest), sha256=digest(manifest), files=len(rows)),
                command=command, build_environment=env, active=False, ownership_closed=True,
                passed=False, complete=False, service_coverage_qualified=False, model_qualified=False,
                current_engine_untouched=True,
                consumer='Imminent MTP stage-service attribution, requiring paused capture and warmed replay coverage.',
                retention='One standalone profiler-qualification fixture, no new engine generation. Retire build output/raw traces after coverage closure; retain fixture/manifest only until its immediate production-profile consumer closes.',
                gates='Collector-off traced/validated smoke first, then all-paused zero kernels, then resume only after capture+upload+two drained warm replays. Exactly3 replays x2 named kernels with unique host sources,positive raw-contained timestamps,correct PCI lane; final numerical/guards/checked teardown. Original host-sink failure remains failed. Any failure stops later cells; no automatic retry. No speed/critical-path/overhead/saturation claim.',
                budgets=dict(build_wall_s=180, native_wall_s=90, text_bytes_per_job=8 << 20, total_bytes=128 << 20),
                legacy_provenance_holds=dict(result='native-gdn-engine-build.json',
                    retained=True, detail_keys=list(hold),
                    next_action='main-primary-native-build-root-v1: close final dependency/quoted-header and ownership proof using matched cpu-gu-reuse-v3 receipt and exact bab7bf063 overlay. main-pr6-harness-configure-v6: recover9709a316 dependency receipt from Git and close generated-command/dependency/ownership checks. Review at this tooling boundary; do not clear either hold. Preserve current engine and13artifacts.'),
                provenance='Main-authored project fixture/controller; existing project graph owner and MIT Intel PTI collector only. Installed official Intel LLVM toolchain, no XeStrata application source.')
        else:
            tool.guards.require(record is not None and RESULT.read_bytes() == tool.git(P, 'show', 'HEAD:'+str(RESULT.relative_to(P))),
                                'commit prepared state before action')
            if action == 'build':
                tool.guards.require(record['stage'] == 'build-prepared', 'fresh build plan required')
                admission(record, container)
                owner = None

                def persist_build():
                    record['active'] = bool(owner and owner.active is not None)
                    (ROOT/'build-record.json').write_text(json.dumps(dict(record,
                        commands=owner.commands if owner else []), indent=2)+'\n')

                owner = tool.supervision.Owner(ROOT/'build-owned', persist_build)
                record['stage'] = 'building'
                persist_build()
                try:
                    entry, _, _ = owner.run('compile', record['command'], record['build_environment'], P,
                        wall=180, cpu=180, rss_cap=4 << 30, text_cap=8 << 20, file_cap=64 << 20, total_cap=128 << 20)
                    tool.guards.require(tool.supervision.closed(entry) and entry['exit_code'] == 0, 'compile/closure failed')
                    admission(record, container)
                    path = ROOT/'graph_record_window'
                    record.update(stage='built', binary=dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size))
                except BaseException as error:
                    record.update(stage='failed', error=repr(error))
                finally:
                    record['ownership_closed'] = owner.active is None and all(tool.supervision.closed(c) for c in owner.commands)
                    persist_build()
            elif action == 'prepare-check':
                tool.guards.require(record['stage'] == 'built' and record['ownership_closed'], 'closed successful build required')
                admission(record, container, True)
                record['cells'] = []
                base = json.loads((P/'tools/sycl/measurement_environment.json').read_text())['environment']
                for name, expected in (('smoke', None), ('paused', 0), ('window', 6)):
                    root = ROOT/name
                    root.mkdir()
                    for folder in ('tmp', 'cache', 'trace'):
                        (root/folder).mkdir()
                    env = dict(base, TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'), LD_DEBUG='libs',
                        STRATA_UR_PROBE_EXPECTED_LOADER=record['assets']['preload']['loader']['path'],
                        UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'ur.log'))
                    session = 'StrataWarmGraph'+secrets.token_hex(12) if expected is not None else None
                    command = [record['binary']['path']]
                    if session:
                        env.update(STRATA_UNITRACE_TARGET_PRELOAD=record['assets']['preload']['library']['path'],
                            UNITRACE_FollowChildProcess='1', UNITRACE_TraceOutputDir=str(root/'trace'))
                        command = [record['assets']['launcher']['path'], '--session', session, '--start-paused',
                            '--host-timing', '--device-timing', '--chrome-device-logging', '--chrome-event-buffer-size',
                            '128', *command]
                    else:
                        env['LD_PRELOAD'] = record['assets']['preload']['library']['path']
                    record['cells'].append(dict(name=name, expected_kernels=expected, session=session,
                                               command=command, environment=env))
                record['stage'] = 'check-prepared'
            elif action == 'check':
                check(record, container)
            else:
                raise RuntimeError('unknown action')
        container[KEY] = record
        RESULT.write_text(json.dumps(container, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps({k: record.get(k) for k in ('stage', 'passed', 'ownership_closed', 'service_coverage_qualified', 'error')}))
        return 1 if record['stage'] == 'failed' else 0


if __name__ == '__main__':
    tool.guards.require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'prepare-check', 'check'),
                        'use graph_record_window.py prepare|build|prepare-check|check')
    raise SystemExit(main(sys.argv[1]))
