#!/usr/bin/env python3
"""Main-owned frozen activation/extraction for the current QSA native observer.

Repository LICENSE. Never qualifies the rejected staging candidate for a model.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_qsa_k_staging as c
b = c.b
P, RESULT = b.P, b.RESULT
ROOT = b.ROOT / 'native-image-v1'
KEY = 'native_image_diagnostic'
SELF = str(Path(__file__).resolve().relative_to(P))
require = b.guards.require


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def original_host_plan(data):
    ref = data['host_plan_ref']
    plan = json.loads(b.gen.git('show', ref['commit'] + ':' + ref['path']))
    for key in ref['key']:
        plan = plan[key]
    require(fingerprint(plan) == ref['canonical_json_sha256'], 'original host plan changed')
    return plan


def observer(record):
    data = record[KEY]
    host = data['host_check']
    require(host['passed'] and host['complete'] and host['ownership_closed'] and
        not host['active'] and len(host['samples']) == 21 and not host['GPU_executed'],
        'closed 21-case host observer gate required')
    require(not record['decision_summary']['performance_gate_passed'] and
        not record['decision_summary']['advance_to_model'], 'preserve packed performance rejection')
    plan = original_host_plan(data)
    path = data['source']
    raw = b.gen.git('show', plan['source_commit'] + ':' + path)
    require(hashlib.sha256(raw).hexdigest() == data['source_sha256'] == b.digest(P / path),
        'original compiled observer source changed')
    artifact = host['artifact_inventory']['libstrata_ur_kernel_binary.so']
    require(b.digest(artifact['path']) == artifact['sha256'] and
        Path(artifact['path']).stat().st_size == artifact['bytes'], 'observer binary changed')
    deps = shlex.split((ROOT / 'observer.d').read_text().replace('\\\n', ' ').split(':', 1)[1])
    actual = {str(Path(p).resolve()): b.digest(p) for p in deps}
    require(all(plan['dependencies'].get(p) == sha for p, sha in actual.items()),
        'original observer actual compiler input changed')
    census = host['actual_dependency_census']['observer']
    require(len(actual) == census['files'] and fingerprint(actual) == census['sha256'],
        'observer actual compiler closure changed')
    for path, sha in plan['compiler_tools'].items():
        require(b.digest(path) == sha, 'observer compiler tool changed: ' + path)
    framework = plan['framework']
    require(b.digest(framework['path']) == framework['sha256'], 'XPTI dispatcher changed')
    return dict(source_commit=plan['source_commit'], source_sha256=data['source_sha256'],
        library=artifact, framework=framework, compiler_dependencies=len(actual),
        actual_dependency_sha256=fingerprint(actual), host_plan=data['host_plan_ref'])


def admission(record):
    rt = c.run_admission(record, record['actual_id_plan'])
    b.qualified_runtime.current(rt)
    info = observer(record)
    original = original_host_plan(record[KEY])
    require(original['runtime_files'] == rt['runtime_files'], 'host observer runtime qualification changed')
    return rt, info


def controller(revision, record):
    shared = {**b.inputs(revision), **c.inputs(revision)}
    expected = {**record['input_sha256'], **record['actual_id_plan']['controller_inputs']}
    require(shared == expected, 'shared native/controller source changed')
    raw = b.gen.git('show', revision + ':' + SELF)
    require(raw == (P / SELF).read_bytes(), 'commit exact native-image controller')
    return dict(source_commit=revision, path=SELF, sha256=hashlib.sha256(raw).hexdigest(),
        shared_source_files=len(shared), shared_source_sha256=fingerprint(shared))


def prepare_activation():
    record = b.gen.committed(RESULT);data = record[KEY]
    folder = ROOT / 'activation'
    require('activation_plan' not in data and not folder.exists(), 'fresh activation only;no retry')
    rt, info = admission(record)
    revision = b.gen.git('rev-parse', 'HEAD').decode().strip()
    code = controller(revision, record)
    canary = rt['preload']['canary']
    require(b.digest(canary['path']) == canary['sha256'], 'qualified no-device canary changed')
    folder.mkdir(mode=0o700)
    for name in ['captures', 'tmp', 'cache']:
        (folder / name).mkdir(mode=0o700)
    library = info['library']['path']
    env = dict(rt['environment'], TMPDIR=str(folder / 'tmp'), XDG_CACHE_HOME=str(folder / 'cache'),
        XPTI_TRACE_ENABLE='1', XPTI_FRAMEWORK_DISPATCHER=info['framework']['path'],
        XPTI_SUBSCRIBERS=library, STRATA_NATIVE_CAPTURE_SUBSCRIBER=library,
        STRATA_NATIVE_CAPTURE_ROOT=str(folder / 'captures'), LD_DEBUG='files',
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'api.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(folder / 'validation.log'))
    env.update({'STRATA_NATIVE_KERNEL_' + str(i): name for i, name in enumerate(c.kernel_ids(record))})
    plan = dict(stage='prepared', controller=code, observer=info, canary=canary,
        runtime_qualification_sha256=b.digest(b.qualified_runtime.RESULT),
        command=[canary['path']], environment=env, output_root=str(folder),
        budgets=dict(wall=30, cpu=30, rss_cap=1 << 30, text_cap=2 << 20,
            file_cap=2 << 20, total_cap=8 << 20),
        criteria='One owned no-device canary closes0. Exact provider/pre-main validation proof,expected46/unchanged version,one same-PID observer load/subscribe,expected namespace0 library/dispatcher init and no other subscriber. Trace has null-platform API version check and no device/context/queue/enqueue/module/kernel work;zero captures,quiet validation,no freshfault/bootchange. This qualifies only subscriber startup,not native getters/extraction/teardown/performance/model.',
        retirement='Keep compact activation/provenance/fault/closure result;delete successful loader/API trace and duplicate ownership output after extraction. Same observer generation remains for immediate three-query extraction only.')
    data['activation_plan'] = plan;b.write(folder / 'plan.json', plan);b.write(RESULT, record)
    print('Prepared no-device actual subscriber activation;commit exact plan before run.')


def run_activation():
    record = b.gen.committed(RESULT);data = record[KEY];plan = data['activation_plan']
    folder = Path(plan['output_root'])
    require(plan['stage'] == 'prepared' and folder == ROOT / 'activation' and
        plan == json.loads((folder / 'plan.json').read_text()) and 'activation_check' not in data,
        'fresh committed activation plan required')
    rt, info = admission(record)
    require(controller(plan['controller']['source_commit'], record) == plan['controller'] and
        info == plan['observer'] and b.digest(b.qualified_runtime.RESULT) == plan['runtime_qualification_sha256']
        and b.digest(plan['canary']['path']) == plan['canary']['sha256'], 'activation inputs changed')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursors = re.findall(r'^-- cursor: (.+)$', subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor'], text=True, timeout=15), re.M)
    require(len(cursors) == 1, 'fresh journal cursor required')
    result = dict(passed=False, complete=False, active=False, ownership_closed=False,
        controller=plan['controller'], GPU_executed=False, native_getters_qualified=False)
    owner = None
    def persist():
        result['active'] = bool(owner and owner.active is not None)
        b.write(folder / 'owned.json', dict(result, commands=owner.commands if owner else []))
    owner = b.supervision.Owner(folder / 'owned', persist)
    try:
        entry, out, err = owner.run('activation', plan['command'], plan['environment'], P, **plan['budgets'])
        require(b.supervision.closed(entry) and entry['exit_code'] == 0, 'actual subscriber canary failed')
        body = json.loads(out.read_text());text = err.read_text()
        require(body == dict(pid=entry['pid'], initialized=0, checked=46, expected=46,
            version=4294967295, closed=0), 'actual canary result differs')
        preload = c.validation.preload_evidence(text, entry['pid'])
        require(preload['loader'] == rt['preload']['loader'], 'actual UR provider differs')
        rows = [json.loads(x) for x in (folder / 'captures/captures.jsonl').read_text().splitlines()]
        require(rows == [dict(kind='load', pid=entry['pid']), dict(kind='subscribe', pid=entry['pid'])],
            'exact no-device observer activation receipt required')
        initialized = re.findall(r'calling init:\s*(\S+)', text)
        for expected in [info['library']['path'], info['framework']['path']]:
            require(sum(Path(x).resolve() == Path(expected).resolve() for x in initialized) == 1,
                'expected single observer/framework initialization: ' + expected)
            family = 'libxptifw.so' if 'libxptifw.so' in expected else Path(expected).name
            objects = [(name, namespace) for name, namespace in
                re.findall(r'file=(\S+)\s+\[(\d+)\]', text) if Path(name).name.startswith(family)]
            require(objects and all(namespace == '0' and
                (not name.startswith('/') or Path(name).resolve() == Path(expected).resolve())
                for name, namespace in objects), 'observer/framework loader namespace differs')
        require(not any('libstrata_ur_' in x and str(Path(x).resolve()) not in
            [rt['preload']['library']['path'], info['library']['path']] for x in initialized),
            'unqualified co-loaded subscriber')
        trace = (folder / 'api.trace').read_text()
        require('urPlatformGetApiVersion' in trace and 'UR_RESULT_ERROR_INVALID_NULL_HANDLE' in trace and
            not re.search(r'ur(?:Enqueue|Kernel|Queue|Device|Context|Program|USM|Mem|PlatformGet\()', trace),
            'activation trace missing or unexpected native device work')
        require(not (folder / 'validation.log').exists() or not (folder / 'validation.log').stat().st_size,
            'activation validation diagnostic')
        require({p.name for p in (folder / 'captures').iterdir()} == {'captures.jsonl'},
            'no-device activation unexpectedly captured binary')
        result.update(host_probe_accepted=True, canary=body, preload=preload,
            observer_load_and_subscription=True, sole_subscriber=True,
            observed_loader_namespace=0,
            actual_observer=info['library'], dispatcher=info['framework'],
            API_trace_bytes=(folder / 'api.trace').stat().st_size)
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        c.validation.finish(result, owner, cursors[0], boot)
        result['complete'] = result['ownership_closed'];persist()
        data['activation_check'] = result;data['activation_plan']['stage'] = 'closed'
        if result['passed']:data['stage'] = 'actual_subscriber_startup_passed_native_extraction_pending'
        b.write(RESULT, record)
    require(result['passed'], 'no-device activation failed final gates')
    print('PASS actual no-device subscriber startup;native getters/GPU extraction still unqualified.')


def prepare_extraction():
    record = b.gen.committed(RESULT);data = record[KEY]
    activation = data['activation_check'];folder = ROOT / 'extraction'
    require(activation['passed'] and activation['complete'] and activation['ownership_closed'] and
        not activation['active'] and activation['fresh_faults'] == [], 'actual no-device startup must pass')
    require('extraction_plan' not in data and not folder.exists(), 'fresh extraction only;no retry')
    rt, info = admission(record)
    require(activation['actual_observer'] == info['library'] and activation['dispatcher'] == info['framework'],
        'activation used another observer/framework')
    revision = b.gen.git('rev-parse', 'HEAD').decode().strip();code = controller(revision, record)
    folder.mkdir(mode=0o700)
    for name in ['captures', 'tmp', 'cache']:(folder / name).mkdir(mode=0o700)
    library = info['library']['path']
    env = dict(rt['environment'], TMPDIR=str(folder / 'tmp'), XDG_CACHE_HOME=str(folder / 'cache'),
        XPTI_TRACE_ENABLE='1', XPTI_FRAMEWORK_DISPATCHER=info['framework']['path'],
        XPTI_SUBSCRIBERS=library, STRATA_NATIVE_CAPTURE_SUBSCRIBER=library,
        STRATA_NATIVE_CAPTURE_ROOT=str(folder / 'captures'), STRATA_SYCL_PREFILL_COMMAND_PROFILE='0',
        ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'api.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(folder / 'validation.log'))
    env.pop('LD_DEBUG', None)
    names = c.kernel_ids(record)
    env.update({'STRATA_NATIVE_KERNEL_' + str(i): name for i, name in enumerate(names)})
    plan = dict(stage='prepared', controller=code, observer=info, kernel_names=names,
        runtime_qualification_sha256=b.digest(b.qualified_runtime.RESULT),
        physical_device=c.device_inventory(rt), fixture=record['binaries']['fixture'],
        selected_ids=record['selected_fixture'], command=[record['binaries']['fixture']['path'],
            '--smoke', record['selected_fixture']['path']], environment=env, output_root=str(folder),
        budgets=dict(wall=180, cpu=180, rss_cap=4 << 30, text_cap=8 << 20,
            file_cap=16 << 20, total_cap=64 << 20),
        criteria='One existing owned actual-ID3query smoke,unchanged binary/safety environment,live original/observer/framework/runtime-map admission and B570 DRM BDF proof before kernels. Native exact-output/guard/input checks and all3attention identities with2launches each. Require3same-PID/device/instance-bound GEN ISA and3module captures,finite bounds/hash/census,quiet validation,normal closure,no freshfault/bootchange. This proves fresh extraction only;ELF-text association and offline disassembly remain later gates. No new timing/model trial or historical timed-image claim.',
        retirement='Keep exact raw GEN/module only through imminent ELF/disassembly and source-bound comparison. Retire successful traces/duplicate dumps after compact extraction evidence;review entire rejected component generation at diagnostic closure.')
    data['extraction_plan'] = plan;b.write(folder / 'plan.json', plan);b.write(RESULT, record)
    print('Prepared one bounded3-query native extraction;commit exact plan before run.')


def run_extraction():
    record = b.gen.committed(RESULT);data = record[KEY];plan = data['extraction_plan']
    folder = Path(plan['output_root'])
    require(plan['stage'] == 'prepared' and folder == ROOT / 'extraction' and
        plan == json.loads((folder / 'plan.json').read_text()) and 'extraction_check' not in data,
        'fresh committed extraction required')
    rt, info = admission(record)
    require(controller(plan['controller']['source_commit'], record) == plan['controller'] and
        info == plan['observer'] and b.digest(b.qualified_runtime.RESULT) == plan['runtime_qualification_sha256'] and
        c.device_inventory(rt) == plan['physical_device'], 'native extraction inputs changed')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursors = re.findall(r'^-- cursor: (.+)$', subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor'], text=True, timeout=15), re.M)
    require(len(cursors) == 1, 'fresh fault cursor required')
    result = dict(passed=False, complete=False, active=False, ownership_closed=False,
        controller=plan['controller'], GPU_executed=False, model_qualified=False,
        historical_timed_image_proven=False, ELF_text_identity_qualified=False)
    owner = None
    def persist():
        result['active'] = bool(owner and owner.active is not None)
        b.write(folder / 'owned.json', dict(result, commands=owner.commands if owner else []))
    owner = b.supervision.Owner(folder / 'owned', persist)
    try:
        b.guards.no_model_owners([]);b.guards.driver_guard()
        start = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
        def interact(io, entry):
            device_name = None
            for stage in ['before-runtime', 'runtime-ready', 'after-cleanup']:
                while True:
                    line = io.line(120)
                    if line.startswith('device='):
                        require(device_name is None, 'duplicate device name');device_name = line.removeprefix('device=')
                    if line.startswith('QSA_K_STAGING_GATE '):break
                require(line == 'QSA_K_STAGING_GATE ' + stage, 'native capture gate differs')
                delta = [a - z for a, z in zip(
                    (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), start)]
                require(min(delta) >= 0 and abs(delta[0] - delta[1]) <= 50000000, 'suspend/clock discontinuity')
                proc = Path('/proc', str(io.proc.pid))
                require(str((proc / 'exe').resolve()) == str(Path(plan['command'][0]).resolve()), 'wrong native executable')
                env = dict(x.split(b'=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if b'=' in x)
                require(all(env.get(k.encode()) == v.encode() for k, v in plan['environment'].items()), 'native environment changed')
                mapped = b.qualified_runtime.executable_mappings((proc / 'maps').read_text())
                additions = {str(Path(x).resolve()) for x in
                    [plan['command'][0], info['library']['path'], info['framework']['path']]}
                require(mapped <= set(rt['runtime_files']) | additions,
                    'unqualified native code map: ' + str(sorted(mapped - set(rt['runtime_files']) - additions)))
                require({info['library']['path'], info['framework']['path']} <= mapped,
                    'actual observer/framework not mapped')
                if stage == 'runtime-ready':
                    require({rt['preload']['loader'], rt['preload']['library']['path'],
                        plan['environment']['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'qualified native provider missing')
                    entry['physical_device'] = c.live_device(io.proc.pid, device_name, plan['physical_device'], rt)
                io.send(b'X')
            require(json.loads(io.line(20)) == {'kind': 'PASS'}, 'native smoke footer missing')
            io.proc.stdin.close();io.exit(20)
        entry, out, err = owner.run('capture', plan['command'], plan['environment'], P,
            interaction=interact, **plan['budgets'])
        require(b.supervision.closed(entry) and entry['exit_code'] == 0, 'native extraction process failed')
        proof = c.validation.preload_evidence(err.read_text(), entry['pid'])
        require(proof['loader'] == rt['preload']['loader'], 'native UR provider changed')
        require(not (folder / 'validation.log').exists() or not (folder / 'validation.log').stat().st_size,
            'native capture validation diagnostic')
        numerical = c.parsed_output(out.read_text(), 'smoke', 0)
        trace = (folder / 'api.trace').read_text()
        launches = c.local_launches(trace, plan['kernel_names'], [0, 1, 2])
        require(all(v['calls'] == 2 for v in launches.values()), 'actual attention dispatch count differs')
        rows = [json.loads(x) for x in (folder / 'captures/captures.jsonl').read_text().splitlines()]
        require(all(x['pid'] == entry['pid'] for x in rows), 'capture PID differs')
        groups = {kind: [x for x in rows if x['kind'] == kind]
            for kind in ['load', 'subscribe', 'device', 'capture', 'module', 'launch']}
        require({k: len(v) for k, v in groups.items()} ==
            {'load': 1, 'subscribe': 1, 'device': 3, 'capture': 3, 'module': 3, 'launch': 6}
            and sum(map(len, groups.values())) == len(rows), 'capture event census differs')
        created = {}
        for line in trace.splitlines():
            if '<--- urKernelCreate(' in line and 'UR_RESULT_SUCCESS' in line:
                match = re.search(r'\((_ZTS\S+)\).*?\.phKernel = .*?\((0x[0-9a-f]+)\)', line)
                if match:created[match[2]] = match[1]
        captures = []
        for arm in range(3):
            selected = {kind: [x for x in groups[kind] if x['arm'] == arm]
                for kind in ['device', 'capture', 'module', 'launch']}
            require(all(len(selected[k]) == n for k, n in
                [('device', 1), ('capture', 1), ('module', 1), ('launch', 2)]), 'per-arm event census differs')
            cap, module, device = (selected[k][0] for k in ['capture', 'module', 'device'])
            require(cap['name'] == plan['kernel_names'][arm] == created.get(cap['ur_kernel']) and
                cap['instance'] == module['instance'] == selected['launch'][0]['instance'] and
                [x['ordinal'] for x in selected['launch']] == [1, 2] and
                all(x['api'] == 'urEnqueueKernelLaunchWithArgsExp' for x in selected['launch']) and
                device['program_devices'] == 1 and device['same_queue_device'] and
                device['vendor'] == 0x8086 and device['device'] == 0xe20c and
                device['binary_extension_version'] >= 65536, 'capture launch/device/native identity differs')
            for row, suffix, maximum in [(cap, '.gen', 4 << 20), (module, '.zebin', 8 << 20)]:
                require(row['file'] == 'kernel%d%s' % (arm, suffix), 'unexpected captured filename')
                path = folder / 'captures' / row['file']
                require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes'] and
                    0 < row['bytes'] <= maximum, 'captured binary size differs')
                row['path'] = str(path);row['sha256'] = b.digest(path)
            captures.append(dict(arm=arm, kernel=cap, module=module, device=device,
                accepted_launches=selected['launch']))
        require({x.name for x in (folder / 'captures').iterdir()} == {'captures.jsonl'} |
            {'kernel%d%s' % (i, suffix) for i in range(3) for suffix in ['.gen', '.zebin']},
            'unexpected capture payload')
        admission(record)
        result.update(host_probe_accepted=True, physical_device=entry['physical_device'], numerical=numerical,
            local_launches=launches, captures=captures, GPU_executed=True, fresh_native_getters_passed=True)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        trace_path = folder / 'api.trace'
        observed_trace = trace_path.read_text(errors='replace') if trace_path.exists() else ''
        result['GPU_submission_attempted'] = bool(re.search(r'urEnqueue\w*Kernel', observed_trace))
        result['GPU_executed'] = result['GPU_executed'] or result['GPU_submission_attempted']
        result['GPU_completion_proven'] = 'numerical' in result
        stderr_path = folder / 'owned/capture.stderr'
        if not result.get('host_probe_accepted') and stderr_path.exists():
            lines = stderr_path.read_text(errors='replace').splitlines()
            result['native_error_excerpt'] = '\n'.join(x for x in lines if
                'native kernel capture:' in x or 'UR_RESULT_ERROR_' in x or 'ZE_RESULT_ERROR_' in x)[-4096:]
        c.validation.finish(result, owner, cursors[0], boot)
        result['complete'] = result['passed'];persist()
        data['extraction_check'] = result;data['extraction_plan']['stage'] = 'closed'
        data['GPU_executed'] = bool(result['GPU_executed'])
        if result['passed']:data['stage'] = 'fresh_native_capture_passed_ELF_and_disassembly_pending'
        b.write(RESULT, record)
    require(result['passed'], 'native extraction failed final gates')
    print('PASS fresh3-kernel GEN/module capture;ELF-text association/disassembly/performance remain unqualified.')


if __name__ == '__main__':
    with (b.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare-activation': prepare_activation, 'run-activation': run_activation,
            'prepare-extraction': prepare_extraction, 'run-extraction': run_extraction}[sys.argv[1]]()
