#!/usr/bin/env python3
"""Qualify later graph replay service after recording initial graph construction."""
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
ROOT = B / 'unitrace-warmed-graph-window-v4'
KEY = 'unitrace_warmed_graph_window_v4'
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


def assets(container):
    return probe.pin_assets(container)


def admission(record, container, binary=False):
    tool.guards.require(inputs(record['source_commit']) == record['input_sha256'], 'source changed')
    tool.guards.require(assets(container) == record['assets'], 'runtime/native/collector assets changed')
    manifest = Path(record['compiler_inputs']['path'])
    tool.guards.require(digest(manifest) == record['compiler_inputs']['sha256'], 'compiler manifest changed')
    rows = json.loads(manifest.read_text())
    tool.guards.require(len(rows) == record['compiler_inputs']['files'], 'compiler manifest count changed')
    for row in rows:
        tool.guards.require(digest(row['path']) == row['sha256'], 'compiler/header changed: ' + row['path'])
    for row in record.get('loader_diagnostic_evidence', {}).get('ELFs', []):
        tool.guards.require(digest(row['path']) == row['sha256'], 'inspected loader/driver ELF changed')
    if binary:
        tool.guards.require(digest(record['binary']['path']) == record['binary']['sha256'], 'binary changed')


def session_state(path, expected):
    raw = path.read_bytes()
    tool.guards.require(len(raw) == 1024 and struct.unpack_from('=i', raw)[0] == expected,
                        'collector session state differs')


def command_buffer_inventory(text):
    """Reconcile this runtime's command-buffer API, including paused captures."""
    buffers, finalized, kernels = {}, set(), {}
    replays = direct = 0
    for line in text.splitlines():
        match = re.search(r'<--- (ur\w+)\((.*)\) -> UR_RESULT_SUCCESS;', line)
        if not match:
            continue
        name, args = match.groups()
        def handle(field, output=False):
            pattern = r'\.'+field+r' = 0x[0-9a-f]+ \((0x[0-9a-f]+)\)' if output else r'\.'+field+r' = (0x[0-9a-f]+)'
            found = re.search(pattern, args)
            tool.guards.require(found is not None, 'missing UR handle: '+field)
            return found[1]
        if name == 'urKernelCreate':
            symbol = re.search(r'\.pKernelName = 0x[0-9a-f]+ \(([^)]+)\)', args)
            tool.guards.require(symbol is not None, 'missing kernel name')
            kernels[handle('phKernel', True)] = symbol[1]
        elif name == 'urCommandBufferCreateExp':
            ident = handle('phCommandBuffer', True)
            tool.guards.require(ident not in buffers, 'duplicate command buffer')
            buffers[ident] = []
        elif name == 'urCommandBufferAppendKernelLaunchExp':
            ident = handle('hCommandBuffer')
            tool.guards.require(ident in buffers and ident not in finalized, 'append outside capture')
            buffers[ident].append(kernels[handle('hKernel')])
        elif name == 'urCommandBufferFinalizeExp':
            ident = handle('hCommandBuffer')
            tool.guards.require(ident in buffers and ident not in finalized, 'invalid finalize')
            finalized.add(ident)
        elif name == 'urEnqueueCommandBufferExp':
            tool.guards.require(handle('hCommandBuffer') in finalized, 'unfinalized replay')
            replays += 1
        elif name.startswith('urEnqueueKernelLaunch'):
            direct += 1
        elif name in ('urQueueBeginCaptureIntoGraphExp', 'urEnqueueGraphExp'):
            raise RuntimeError('different UR graph API needs its own inventory contract')
    tool.guards.require(len(buffers) == len(finalized) == 1 and replays == 5 and direct == 0,
                        'capture/replay inventory differs')
    names = next(iter(buffers.values()))
    tool.guards.require(len(names) == 2 and 'RecordWindowAdd' in names[0]
                        and 'RecordWindowMultiply' in names[1], 'captured program differs')
    return dict(capture_regions=1, graph_replays=replays, direct_kernels=direct,
                template=names, executed_program_kernels=replays*len(names))


def trace_evidence(root, expected, launch, ready, begin, end, pid, driver):
    files = list((root/'trace').rglob('*.json'))
    tool.guards.require(len(files) <= 1, 'multiple collector timelines')
    events, raw_identity = [], None
    if files:
        raw = files[0].read_bytes()
        raw_identity = dict(sha256=tool.sha(raw), bytes=len(raw))
        parsed = json.loads(raw)
        events = parsed['traceEvents'] if isinstance(parsed, dict) else parsed
    gpu = [e for e in events if e.get('cat') == 'gpu_op' and e.get('ph') == 'X']
    kernels = [e for e in gpu if 'RecordWindow' in e.get('name', '')]
    result = dict(timeline=raw_identity, total_program_kernels=len(kernels),
                  expected_program_kernels=4+expected,
                  device_names=dict(collections.Counter(e.get('name') for e in gpu)))
    if len(kernels) != 4+expected:
        return dict(result, qualified=False, failure='initial or resumed graph kernels lost/leaked')
    tool.guards.require(all(isinstance(e.get(k), (float, int)) and math.isfinite(e[k])
        for e in kernels for k in ('ts', 'dur')) and all(e['dur'] > 0 for e in kernels),
        'invalid kernel service')
    kernels.sort(key=lambda e:e['ts'])
    prefix, window = kernels[:4], kernels[4:]
    for group, count, lo, hi in ((prefix, 2, launch, ready), (window, expected//2, begin, end)):
        names = collections.Counter(e['name'] for e in group)
        tool.guards.require((not count and not names) or (len(names) == 2 and sorted(names.values()) == [count,count]
            and any('RecordWindowAdd' in n for n in names) and any('RecordWindowMultiply' in n for n in names)),
            'per-kernel initial/resumed replay counts differ')
        tool.guards.require(all(lo/1000-1000 <= e['ts'] <= e['ts']+e['dur'] <= hi/1000+1000 for e in group),
                            'program kernel outside its controlled raw-clock window +/-1ms')
    copies = [e for e in gpu if e.get('name', '').startswith('zeCommandListAppendMemoryCopy')]
    other = [e for e in gpu if e not in kernels and e not in copies and e.get('name') != 'zeCommandListAppendBarrier']
    tool.guards.require(not other and len(copies) == 1 and copies[0]['name'].endswith('[1164]'),
                        'unexpected device work or post-pause readback leakage')
    tool.guards.require(copies[0]['ts']+copies[0]['dur'] <= ready/1000+1000,
                        'input copy outside initial recorded prefix')
    sources = collections.Counter(str(e.get('id', '')) for e in events
        if e.get('pid') == pid and e.get('ph') == 's' and e.get('cat', '').startswith('Flow_H2D_'))
    sinks = collections.Counter(str(e.get('id', '')) for e in events
        if e.get('pid') == pid and e.get('ph') == 't' and e.get('cat', '').startswith('Flow_D2H_'))
    ids = [str(e.get('args', {}).get('id', '')) for e in kernels]
    tool.guards.require(all(ids) and len(set(ids)) == len(ids) and all(sources[i] == 1 for i in ids),
                        'program kernel missing unique host source/identity')
    pci = re.fullmatch(r'([0-9a-f]+):([0-9a-f]+):([0-9a-f]+)\.([0-9a-f]+)', Path(driver['pci']).name)
    tool.guards.require(pci is not None, 'invalid pinned PCI address')
    address = tuple(int(v,16) for v in pci.groups())
    labels = {e['pid']:e.get('args',{}).get('name','') for e in events
              if e.get('ph') == 'M' and e.get('name') == 'process_name'}
    for lane in {e['pid'] for e in kernels}:
        label = labels.get(lane,'')
        match = re.search(r'\] ([0-9a-f]+):([0-9a-f]+):([0-9a-f]+):([0-9a-f]+) #\d+$', label)
        tool.guards.require(label.startswith('DEVICE<') and match
            and tuple(int(v,16) for v in match.groups()) == address, 'wrong device lane')
    return dict(result, qualified=True, initial_kernels=4, recorded_window_kernels=len(window),
        window_kernel_service_us=sum(e['dur'] for e in window),
        window_kernel_union_us=production.union_us(window),
        unique_host_sources=True, unsunk_kernels=sum(sinks[i] != 1 for i in ids), raw_window_contained=True,
        limits='Initial capture was recorded; paused-capture method remains failed. Service/source/replay coverage only, not critical path, observer overhead, full-model stage mapping or hardware saturation. Older graph host-sink failure remains failed.')


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
                result, _, _ = control.run(cell['name']+'-'+str(len(row['controls']))+'-'+action,
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
                    session_state(session, 0)
                    change('pause', 1)
                    if cell['name'] == 'prefix-window':
                        change('resume', 0)
                row['release_raw_ns'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                io.send(b'continue\n')
                drained = re.fullmatch(r'graph_window phase=drained pid=(\d+) raw_ns=(\d+)', io.line(30))
                tool.guards.require(drained and int(drained[1]) == child['pid'], 'drain owner/protocol mismatch')
                row['drained_raw_ns'] = int(drained[2])
                if session is not None:
                    if cell['name'] == 'prefix-window':
                        change('pause', 1)
                    else:
                        session_state(session, 1)
                io.send(b'continue\n')
                io.proc.stdin.close()
                footer = io.line(20)
                tool.guards.require(footer == 'graph_window PASS warm_replays=2 window_replays=3 kernels_per_replay=2'
                    ' numerical=PASS guards=PASS checked_teardown=PASS', 'numerical/owner footer differs')
                io.exit(20)

            try:
                row['launch_raw_ns'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                entry, out, err = target.run(cell['name'], cell['command'], cell['environment'], root,
                    interaction=interact, wall=90, cpu=90, rss_cap=4 << 30, text_cap=8 << 20,
                    file_cap=32 << 20, total_cap=128 << 20)
                tool.guards.require(tool.supervision.closed(entry) and entry['exit_code'] == 0, 'fixture exit/closure failed')
                text = out.read_text()
                tool.guards.require('graph_window PASS warm_replays=2 window_replays=3 kernels_per_replay=2'
                    ' numerical=PASS guards=PASS checked_teardown=PASS' in text, 'missing numerical/owner footer')
                stderr = err.read_text()
                row['validation'] = probe.validation.preload_evidence(stderr, entry['pid'])
                lookup = [line for line in stderr.splitlines() if 'symbol lookup error' in line]
                row['loader_lookup_errors'] = lookup
                # Exact optional driver probes, inspected in the actual ELF;
                # still require normal exit, validation, numerical and coverage
                # gates. Any other lookup diagnostic stops this new attempt.
                for line in lookup:
                    tool.guards.require(re.search(r'undefined symbol: (zelLoaderTranslateHandle|zelSetDriverTeardown) \(fatal\)$', line),
                                        'new loader symbol diagnostic')
                inventory = command_buffer_inventory((root/'ur.log').read_text())
                row['inventory'] = inventory
                if session is not None:
                    row['coverage'] = trace_evidence(root, cell['expected_kernels'], row['launch_raw_ns'],
                        row['ready_raw_ns'], row['release_raw_ns'], row['drained_raw_ns'], entry['pid'], record['driver_before'])
                    tool.guards.require(row['coverage']['qualified'], row['coverage'].get('failure','coverage failed'))
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
            prior = container['unitrace_warmed_graph_window_v3']
            tool.guards.require(prior['stage'] == 'failed' and prior['ownership_closed'] and not prior['active']
                and prior['fresh_faults'] == [] and prior['boot_unchanged']
                and prior['error'] == "RuntimeError('program kernel outside its controlled raw-clock window +/-1ms')",
                'closed clock-domain failure required')
            archive = prior['original_build_archive']
            raw = tool.git(P,'show',archive['commit']+':'+archive['path'])
            tool.guards.require(tool.sha(raw) == archive['sha256'], 'original archive changed')
            original = json.loads(raw)
            build = original['unitrace_warmed_graph_window']
            previous = original['unitrace_warmed_graph_window_v2']
            revision = tool.git(P,'rev-parse','HEAD').decode().strip()
            pins = inputs(revision)
            for name, expected in build['input_sha256'].items():
                tool.guards.require(tool.sha(tool.git(P,'show',build['source_commit']+':'+name)) == expected,
                                    'original source cannot be recovered: '+name)
            tool.guards.require(set(pins) == set(build['input_sha256']) and
                [n for n in pins if pins[n] != build['input_sha256'][n]] == ['tools/sycl/graph_record_window.py'],
                'reuse permits only the explicitly changed controller')
            pinned = assets(container)
            tool.guards.require(pinned == previous['assets'], 'original runtime/native/tool assets changed')
            ROOT.mkdir(mode=0o700)
            record = dict(schema='strata-unitrace-warmed-graph-window-v4',stage='check-prepared',
                source_commit=revision,input_sha256=pins,assets=pinned,compiler=build['compiler'],
                binary_source_commit=build['source_commit'],binary=build['binary'],
                compiler_inputs=build['compiler_inputs'],original_build_archive=archive,
                loader_diagnostic_evidence=previous['loader_diagnostic_evidence'],
                previous_attempt=dict(key='unitrace_warmed_graph_window_v3',commit='99ee897a1',
                    outcome='Failed clock-domain check: UNITRACE_SystemTime=1 and host-call/source logging were omitted. Native processes closed normally; original paused-capture and host-sink failures also remain failed.'),
                qualified_collector_configuration_reference=dict(source_commit='6269902082bfb811ca237bf45faf07e3152898ea',
                    result_key='unitrace_recording_window',SystemTime='1',chrome_logging=['kernel','call'],
                    changes='Record initial capture/warm prefix, then pause/resume; fixture differs and requires this new coverage qualification.'),
                reuse='No recompilation. Original187 non-controller inputs,recipe,compiler manifest and fixture binary unchanged.',
                passed=False,complete=False,active=False,ownership_closed=True,
                service_coverage_qualified=False,model_qualified=False,current_engine_untouched=True,
                consumer='MTP stage-service attribution using recorded graph construction before later pause/resume.',
                retention='Same227680-byte fixture and8697-input manifest for imminent profile consumer; retire raw output after closure. No new model/tooling generation.',
                gates='Traced/validated smoke,then two independent collector processes initially recording graph construction and two drained warm replays. Pause at ready; either stay paused or resume for three replays. Expect4 initial kernels in each,0 versus6 later kernels,one1164-byte input copy,no final readback leakage,unique host sources,positive raw-contained clocks,correct PCI,numerical/guards/checked teardown,normal ownership and no fresh fault. Any failure stops. Original paused-capture and host-sink failures stay failed. No speed/critical-path/overhead/saturation claim.',
                budgets=dict(native_wall_s=90,text_bytes_per_job=8<<20,total_bytes=128<<20),
                legacy_provenance_holds='Unchanged in native-gdn-engine-build.json. PR6 receipt9709a316 recovered atb1c0e1449;final generated-command/dependency/ownership join and primary dependency/quoted-header/ownership proof remain open. No build/retirement here.',
                cells=[])
            admission(record,container,True)
            base = json.loads((P/'tools/sycl/measurement_environment.json').read_text())['environment']
            for name,expected in (('smoke',None),('prefix-only',0),('prefix-window',6)):
                root = ROOT/name
                root.mkdir()
                for folder in ('tmp','cache','trace'):
                    (root/folder).mkdir()
                env = dict(base,TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),LD_DEBUG='libs',UNITRACE_SystemTime='1',
                    STRATA_UR_PROBE_EXPECTED_LOADER=pinned['preload']['loader']['path'],
                    UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'ur.log'))
                session = 'StrataGraphPrefix'+secrets.token_hex(12) if expected is not None else None
                command = [record['binary']['path']]
                if session:
                    env.update(STRATA_UNITRACE_TARGET_PRELOAD=pinned['preload']['library']['path'],
                        UNITRACE_FollowChildProcess='1',UNITRACE_TraceOutputDir=str(root/'trace'))
                    command = [pinned['launcher']['path'],'--session',session,
                        '--host-timing','--device-timing','--chrome-kernel-logging','--chrome-call-logging','--chrome-event-buffer-size',
                        '128',*command]
                else:
                    env['LD_PRELOAD'] = pinned['preload']['library']['path']
                record['cells'].append(dict(name=name,expected_kernels=expected,session=session,
                                           command=command,environment=env))
        else:
            tool.guards.require(action == 'check' and record is not None and RESULT.read_bytes() ==
                tool.git(P,'show','HEAD:'+str(RESULT.relative_to(P))), 'commit prepared state before check')
            check(record,container)
        container[KEY] = record
        RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({k:record.get(k) for k in ('stage','passed','ownership_closed','service_coverage_qualified','error')}))
        return 1 if record['stage'] == 'failed' else 0


if __name__ == '__main__':
    tool.guards.require(len(sys.argv) == 2 and sys.argv[1] in ('prepare','check'),
                        'use graph_record_window.py prepare|check')
    raise SystemExit(main(sys.argv[1]))
