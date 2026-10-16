#!/usr/bin/env python3
"""Main-only, source-delivered resource timeline diagnostic. Repository LICENSE.

No unitrace executable, build, automatic arm/retry, runtime environment mutation,
new GPU runtime/context/queue, tensor capture, service, reset or cleanup.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import select
import selectors
import shutil
import subprocess
import sys
import time
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import timeline

D = Path(__file__).resolve().parent
A = D.parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
OWNER_SHA = 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
PY = '/home/yayoi/.local/share/strata-sycl/venv/bin/python'
RUNNER = D / 'run_server_job.py'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
SOURCE = '63326f98c36656cf3a3e1adcc54c3c25a02aa6ee'
BUILD_SHA = '3fca64de3664a74ac5a376e4e40cb8a3e2910113f29a4597a564c2f903381327'
ENGINE_SHA = '6334491b623ead7b83c7ff5aaee84583294b3ecd4b12c6d06b77d1fe7e7a691d'
LIB = B / 'main-primary-unitrace-build-root-v1/build/libunitrace_tool.so'
LIB_SHA = '8a8fe37441606bf66ab42fc39cc447b5257f48d09f04fbcc60703e30826671eb'
RAW_CAP, DISK_CAP, RESERVE = 62 << 20, 1 << 30, 150
HOST_ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
ERROR = re.compile(r'\[ERROR\]|(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|retirement failed|Strata SYCL cleanup:|Unable to (?:create|trace|initialize)|not enabled for tracing|(?:collection|profile|trace).*truncat', re.I)
MODEL = re.compile(r'(^|/)(strata(?:[-_.][^/]*)?|sycl_[^/]*|paired_gemm|strata_vllm_xe2_numerics|bench\.sh|engine[-_]wrapper\.py|run_server_job\.py|ninja|cmake|ctest|unitrace|clang(?:\+\+)?(?:-[0-9]+)?)$')
COLLECTOR = dict(UNITRACE_HostTiming='1', UNITRACE_DeviceTiming='1',
    UNITRACE_DeviceTimeline='1', UNITRACE_Verbose='1',
    UNITRACE_FollowChildProcess='1', UNITRACE_LogToFile='1', ZE_ENABLE_TRACING_LAYER='1',
    CUDA_MODULE_LOADING='EAGER')
WARNINGS = dict(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
    UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
    UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
    UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
QUALIFICATION = ('quality_qualified', 'model_quality_qualified', 'performance_qualified',
    'performance_adopted', 'default_adopted', 'physical_262144_qualified', 'restore_qualified')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def artifact(path):
    path = Path(path)
    st = path.stat()
    return dict(path=str(path), realpath=str(path.resolve(strict=True)), bytes=st.st_size, sha256=sha(path))


def safe_path(path):
    path = Path(path).resolve(strict=True)
    require(not path.is_relative_to(B / 'prefill-gemm-only-contract-root-v1'), 'protected input forbidden')
    llvm = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
    require('XeStrata' not in str(path) or path.is_relative_to(llvm), 'excluded engine source forbidden')
    return path


def check_pin(path, pin):
    p = safe_path(path)
    require(str(p) == pin['realpath'] and p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256'], 'file pin: ' + str(path))


def identity(pid):
    try:
        v = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=v[0], ppid=int(v[1]), pgid=int(v[2]), sid=int(v[3]), start_ticks=int(v[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def absence(commands):
    rows = []
    for command in commands:
        require(command.get('direct_identity') and command.get('owners'), 'required owned PID/start evidence')
        for old in command['owners'] + [command['direct_identity']]:
            now = identity(old['pid'])  # PermissionError for an owned identity is never waived.
            absent = now is None or now['start_ticks'] != old['start_ticks']
            rows.append(dict(original=old, current=now, original_absent=absent))
            require(absent, 'owned PID/start remains: ' + str(old))
    return rows


def health_gate(path, digest, module, current_boot, manifest, tiny=False):
    require(re.fullmatch('[0-9a-f]{64}', digest or ''), 'explicit receipt SHA256')
    require(sha(path) == digest, 'selected receipt hash')
    value = json.loads(Path(path).read_text())
    require(value.get('passed') is True and value.get('complete') is True and value.get('active') is False and
            value.get('ownership_closed') is True and value.get('boot_id') == current_boot and
            value.get('boot_unchanged') is True and value.get('driver_dump_exists') is False and
            value.get('fresh_faults') == [], 'explicit latest closed PASS, sameboot, fault/dump gate')
    require(value.get('source_commit') == SOURCE and value.get('source_pins') == manifest['source_pins'], 'same frozen product sources')
    commands = value.get('commands', [])
    require(commands and all(module.closed(c) and c.get('exit_code') == 0 for c in commands), 'all selected receipt commands normally closed')
    absent = absence(commands)
    if tiny:
        require(value.get('schema') == 'strata-unitrace-direct-v1' and value.get('mode') == 'tiny' and
                value.get('collector_compatible') is True and value.get('manifest_sha256') == manifest['_sha256'] and
                value.get('collector_library_sha256') == LIB_SHA and value.get('binary_sha256') == manifest['tiny']['sha256'] and
                value.get('collection', {}).get('proved') is True and
                value.get('collection', {}).get('resource_timeline', {}).get('proved') is True, 'same-package tiny collector compatibility PASS required')
    else:
        require(value.get('schema') == 'strata-prompt-softmax-long-v1' and
                value.get('binary_sha256') == ENGINE_SHA and
                value.get('build_receipt_sha256') == BUILD_SHA and
                value.get('arm') == '0' and value.get('phase') in ('diagnostic', 'clean'),
                'explicit current qualified softmax0 long health gate')
        require(value['environment']['STRATA_SYCL_PROMPT_SOFTMAX_SUBGROUP'] == '0' and
                value['environment']['STRATA_PREFILL_LAYER_MAJOR_R_GPU'] == 'auto' and
                value['environment']['STRATA_SYCL_PREFILL_NATIVE16_BACKEND'] == 'vllm',
                'same runtime route/softmax/placement')
        check_pin(manifest['engine']['path'], manifest['engine'])
    return dict(selection='Main explicitly selected latest closed PASS; no discovery/fallback/waiver',
                receipt=artifact(path), original_status_preserved=True, owner_absence=absent, schema=value.get('schema'))


def owner_scan(model_files):
    """Read-only same-UID scan. Protected services are visible limits, never a clearance claim."""
    limits, suspects = [], []
    population = list(Path('/proc').iterdir())
    require(len(population) < 131072, 'bounded proc population')
    for p in population:
        if not p.name.isdecimal() or int(p.name) == os.getpid():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            try:
                args = [a.decode(errors='replace') for a in (p / 'cmdline').read_bytes().split(b'\0') if a]
            except PermissionError as error:
                comm = (p / 'comm').read_text().strip()
                require(comm in ('systemd', '(sd-pam)', 'sd-pam', 'gpg-agent', 'ssh-agent'), 'unknown same-UID cmdline visibility denial')
                limits.append(dict(pid=int(p.name), comm=comm, cmdline_visibility='denied', error=repr(error)))
                continue
            matches = [a for a in args if a in model_files or a.endswith('.gguf') or
                       (not a.startswith('-') and MODEL.search(a)) or a.startswith('serve.') or
                       (str(P / 'serve') in a and a.endswith('.py'))]
            exe = None
            try:
                exe = str((p / 'exe').resolve(strict=True))
            except PermissionError as error:
                limits.append(dict(pid=int(p.name), cmdline=args, exe_visibility='denied', error=repr(error)))
            if exe and MODEL.search(exe):
                matches.append(exe)
            if matches:
                suspects.append(dict(pid=int(p.name), cmdline=args, exe=exe, matches=matches))
        except (FileNotFoundError, ProcessLookupError):
            continue
    require(not suspects, 'foreign model/fixture/build/test owner: ' + json.dumps(suspects))
    return dict(same_uid_suspects=suspects, executable_visibility_limits=limits,
        scope='Same-UID cmdline plus accessible exe scan; cannot exclude hidden/foreign GPU owners. Main serializes all GPU/build/test/cleanup work.')


def native_snapshot(pid, expected, env, pins, expected_sid):
    """Only our already owned child. Bind actual maps dev/inode to pre-pinned ELF bytes."""
    before = identity(pid)
    require(before and before['sid'] == expected_sid, 'actual native session')
    proc = Path('/proc', str(pid))
    raw = (proc / 'maps').read_text()
    needed = ('libunitrace_tool.so', 'libxptifw.so', 'libsycl.so', 'libze_loader.so',
              'libze_tracing_layer.so', 'libze_intel_gpu.so', 'libur_adapter_level_zero')
    if not all(n in raw for n in needed):
        return None
    args = [a.decode() for a in (proc / 'cmdline').read_bytes().split(b'\0') if a]
    effective = dict(a.decode().split('=', 1) for a in (proc / 'environ').read_bytes().split(b'\0') if a)
    require(effective == env and args and str(Path(args[0]).resolve()) == expected['realpath'], 'actual native exec argv/environment')
    require(str((proc / 'exe').resolve(strict=True)) == expected['realpath'], 'actual /proc native executable')
    with (proc / 'exe').open('rb') as stream:
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == expected['sha256'], 'actual executable open-file SHA')
    loaded = []
    for line in raw.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith('/') and 'x' in fields[1]:
            require(not fields[5].endswith(' (deleted)'), 'mapped ELF deleted')
            loaded.append(str(safe_path(fields[5])))
    missing = sorted(set(loaded) - set(pins))
    require(not missing, 'unsealed actual loaded dependencies (all at snapshot): ' + json.dumps(missing))
    mappings = []
    seen = set()
    for line in raw.splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith('/') or 'x' not in fields[1]:
            continue
        name = fields[5]
        require(not name.endswith(' (deleted)'), 'mapped ELF deleted')
        path = safe_path(name)
        if str(path) in seen:
            continue
        seen.add(str(path))
        require(str(path) in pins, 'unsealed actual loaded dependency: ' + str(path))
        st = path.stat()
        major, minor = [int(v, 16) for v in fields[3].split(':')]
        require(st.st_ino == int(fields[4]) and os.major(st.st_dev) == major and os.minor(st.st_dev) == minor, 'mapped ELF dev/inode')
        check_pin(path, pins[str(path)])
        mappings.append(dict(mapping=line, **artifact(path)))
    require(str(LIB.resolve()) in seen, 'actual loaded collector mapping required')
    after = identity(pid)
    require(after and after['start_ticks'] == before['start_ticks'], 'native PID/start stable during snapshot')
    signature = '\n'.join(l for l in raw.splitlines() if len(l.split(None, 5)) == 6 and
        'x' in l.split(None, 5)[1] and l.split(None, 5)[5].startswith('/'))
    return dict(native=before, argv=args, executable=expected, effective_environment=effective,
        executable_mappings=mappings, maps_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        executable_maps_sha256=hashlib.sha256(signature.encode()).hexdigest(),
        captured_utc=utc(), anonymous_executable_mapping_scope='File-backed executable maps at capture only; no JIT ISA/occupancy claim')


def native_wrapper(spec_path, argv):
    """Private model wrapper entry. Python is never preloaded; only native Popen env is."""
    spec = json.loads(Path(spec_path).read_text())
    require(not any(k.startswith(('UNITRACE_', 'PTI_', 'XPTI_', 'ZE_ENABLE_')) or k == 'LD_PRELOAD' for k in os.environ), 'wrapper must never inherit collector environment')
    check_pin(spec['engine']['path'], spec['engine'])
    check_pin(D / 'timeline.py', spec['timeline_helper'])
    env = spec['native_environment']
    start = dict(engine=spec['engine']['path'], binary_sha256=spec['engine']['sha256'],
        argv=[spec['engine']['path'], *argv], wrapper=identity(os.getpid()), started_utc=utc(),
        experiment_environment=env, collector_snapshot=None, mapping_snapshots=[], errors=[])
    proc = subprocess.Popen(start['argv'], env=env, cwd=spec['native_cwd'])
    fd = os.pidfd_open(proc.pid)
    watch = select.poll(); watch.register(fd, select.POLLIN)
    try:
        native = identity(proc.pid)
        require(native and native['ppid'] == os.getpid() and native['sid'] == os.getsid(0), 'native wrapper child binding')
        start['native'] = native
        pipe = os.stat('/proc/' + str(proc.pid) + '/fd/0')
        start['stdin'] = dict(device=pipe.st_dev, inode=pipe.st_ino)
        Path(spec['start_path']).write_text(json.dumps(start, indent=2) + '\n')
        deadline = time.monotonic() + 890
        while not watch.poll(0):
            require(time.monotonic() < deadline, 'native wrapper deadline; outer Owner performs pidfd closure')
            maps = Path('/proc', str(proc.pid), 'maps').read_text()
            signature = '\n'.join(l for l in maps.splitlines() if len(l.split(None, 5)) == 6 and
                'x' in l.split(None, 5)[1] and l.split(None, 5)[5].startswith('/'))
            digest = hashlib.sha256(signature.encode()).hexdigest()
            if start['collector_snapshot'] is None or digest != start['collector_snapshot']['executable_maps_sha256']:
                snapshot = native_snapshot(proc.pid, spec['engine'], env, spec['elf_pins'], os.getsid(0))
                if snapshot:
                    require(len(start['mapping_snapshots']) < 64, 'bounded executable-map changes')
                    start['mapping_snapshots'].append(snapshot)
                    start['collector_snapshot'] = snapshot
                    Path(spec['start_path']).write_text(json.dumps(start, indent=2) + '\n')
            watch.poll(2 if start['collector_snapshot'] is None else 50)
        start.update(pid=proc.pid, exit_code=proc.wait(), finished_utc=utc())
        require(start['collector_snapshot'] is not None, 'actual native library mapping absent')
    except BaseException as error:
        start['errors'].append(repr(error))
        # Leave any live native in the anchored outer Owner session; do not adopt
        # or signal numeric PIDs. An error can never become a normal collection.
        raise
    finally:
        os.close(fd)
        Path(spec['exit_path']).write_text(json.dumps(start, indent=2) + '\n')
    return 0 if start['exit_code'] == 0 else 1


def parse_collection(directory, pid, suite=None):
    """Official timing CSV plus compute-only Kernel Properties; names may contain commas."""
    files = sorted(p for p in directory.rglob('*') if p.is_file())
    expected = directory / ('timing.' + str(pid) + '.log')
    require(files == [expected], 'one actual native PID-named summary; unexpected/partial collector sides fail')
    require(expected.stat().st_size <= timeline.MAX_RAW, 'finite offline timeline input budget')
    raw = expected.read_text(encoding='utf-8', errors='strict')
    require(raw.endswith('\n') and not ERROR.search(raw), 'complete collector log without errors/truncation')
    events, raw = timeline.split(raw)
    sections = {'API': [], 'Device': []}
    totals, clocks, headers, properties = {}, {}, {}, []
    section = backend = table = None
    property_header = False
    def name_of(text):
        return text[1:-1] if len(text) >= 2 and text[0] == text[-1] == '"' else text
    for line in raw.splitlines():
        match = re.fullmatch(r'=== (API|Device) Timing Summary ===', line.strip())
        if match:
            section, backend, table = match[1], None, None
            require(section not in totals, 'duplicate collector section')
            totals[section] = None
            continue
        if line.strip() == '=== Kernel Properties ===':
            require(section == 'Device' and not property_header, 'single kernel-properties section after device table')
            section, table = 'Properties', None
            continue
        if not line.strip():
            table = None
            continue
        if section == 'Properties':
            fields = [v.strip() for v in line.rsplit(',', 7)]
            head = ['Kernel', 'Compiled', 'SIMD', 'Number of Arguments', 'SLM Per Work Group',
                    'Private Memory Per Thread', 'Spill Memory Per Thread', 'Register File Size Per Thread']
            if fields == head:
                require(not property_header, 'one official kernel-properties header')
                property_header = True
                continue
            require(property_header and len(fields) == 8, 'complete properties row: ' + line)
            name, compiled, simd, nargs, slm, private, spill, grf = fields
            require(compiled in ('AOT', 'JIT') and (simd == 'ANY' or re.fullmatch(r'[1-9][0-9]*', simd)) and
                    all(re.fullmatch(r'[0-9]+', v) for v in (nargs, slm, private, spill)) and
                    (grf == 'unknown' or re.fullmatch(r'[1-9][0-9]*', grf)), 'official kernel property values')
            properties.append(dict(name=name_of(name), compiled=compiled, reported_SIMD=simd,
                arguments=int(nargs), SLM_bytes_per_workgroup=int(slm), private_bytes_per_thread=int(private),
                spill_bytes_per_thread=int(spill), register_file_registers_per_thread=None if grf == 'unknown' else int(grf),
                raw_row=line, scope='Reported compiled-kernel properties; no occupancy or all-input spill guarantee'))
            continue
        if section is None:
            require(not line.strip(), 'unexpected collector preamble')
            continue
        match = re.fullmatch(r'\s*Total Execution Time \(ns\):\s*(\d+)\s*', line)
        if match:
            clocks[section] = int(match[1]); continue
        match = re.fullmatch(r'\s*Total (API|Device) Time for L0 backend \(ns\):\s*(\d+)\s*', line)
        if match:
            require(match[1] == section and totals[section] is None, 'summary type/uniqueness')
            totals[section] = int(match[2]); continue
        if line.strip() == '== L0 Backend ==':
            backend = 'L0'; continue
        fields = [v.strip() for v in line.rsplit(',', 6)]
        head = ['Function' if section == 'API' else 'Kernel', 'Calls', 'Time (ns)', 'Time (%)', 'Average (ns)', 'Min (ns)', 'Max (ns)']
        if fields == head:
            require(backend == 'L0' and section not in headers, 'one L0 table header')
            headers[section] = line; table = section; continue
        require(table == section and backend == 'L0' and len(fields) == 7, 'unparsed/partial timing row: ' + line)
        name, calls, total, percent, average, minimum, maximum = fields
        require(all(re.fullmatch(r'\d+', v) for v in (calls, total, average, minimum, maximum)), 'integer aggregate units')
        calls, total, average, minimum, maximum = map(int, (calls, total, average, minimum, maximum))
        percentage = float(percent)
        require(name and calls > 0 and total >= 0 and math.isfinite(percentage) and 0 <= percentage <= 100.01 and
                0 <= minimum <= maximum and average == total // calls and minimum <= average <= maximum, 'timing row values')
        sections[section].append(dict(name=name_of(name), calls=calls, time_ns=total, percent=percentage,
            average_ns=average, min_ns=minimum, max_ns=maximum, raw_row=line))
    require(set(totals) == set(headers) == set(clocks) == {'API', 'Device'}, 'both emitted summaries required')
    for section, rows in sections.items():
        require(rows and totals[section] and sum(v['time_ns'] for v in rows) == totals[section], 'all rows and exact aggregate sum: ' + section)
    require(any(r['name'].startswith('ze') for r in sections['API']), 'actual Level Zero host API rows')
    require(property_header and properties and all(any(timeline.base_name(r['name']) == prop['name'] for r in sections['Device']) for prop in properties),
            'compute-kernel properties must match actual device timing rows; copies/barriers alone are not kernel collection')
    require({prop['name'] for prop in properties} ==
            {timeline.base_name(r['name']) for r in sections['Device'] if timeline.SHAPE.search(r['name'])},
            'complete two-way compute-property/name coverage; truncated property tail fails')
    result = dict(proved=True, emitted_collection_pid=pid, emitted_pid_evidence=expected.name,
        files=[artifact(expected)], raw_summary=raw, headers=headers, totals_ns=totals,
        collector_execution_time_ns=clocks, host_API_rows=sections['API'], device_rows=sections['Device'],
        kernel_properties=properties, command_counts={k: sum(r['calls'] for r in v) for k, v in sections.items()},
        interpretation='Nonexclusive whole-process kernel/command sums and host API elapsed times. Includes startup/teardown and may overlap. Compiled properties are not measured occupancy. No busy union, critical path, prefill-only attribution or clean speed claim.')
    result['resource_timeline'] = timeline.extract(events, result, expected, directory.parent / 'timeline-samples.json', suite)
    return result


def validate_model(root, result):
    require(result.get('passed') is True and result.get('bench_exit_code') == result.get('server_exit_code') == 0, 'normal benchmark/server closure')
    entry = result['real_engine_exit']
    require(entry['exit_code'] == 0 and not entry['errors'] and entry['binary_sha256'] == ENGINE_SHA, 'normal actual native exit')
    suites = list((root / 'stock-results').glob('*/results.json'))
    require(len(suites) == 1, 'exactly one fresh official benchmark suite')
    suite = json.loads(suites[0].read_text())
    cohort = suite['results']['prefill_only']
    require(suite['complete'] and not suite['results']['failures'] and cohort['complete'] and
            cohort['eligible_runs'] == len(cohort['requests']) == 1 and suite['params']['think'] is True and
            suite['params']['mode'] == 'prefill-only', 'complete thinkingON official prefill-only request')
    request = cohort['requests'][0]
    require(request['ok'] and request['eligible'] and request['stream_complete'] and
            request['actual_prompt_tokens'] == request['actual_read_tokens'] == 39262 and
            request['actual_reused_tokens'] == 0 and request['generated_tokens'] == 24 and request['thinking_requested'], 'exact actual39262 input/24 output/no reuse')
    native = (root / 'owned/engine.stderr').read_text(errors='strict')
    require(not ERROR.search(native), 'native/validation/collector error')
    placement = [l for l in native.splitlines() if l.startswith('strata prefill: layer-major candidate,')]
    require(len(placement) == 1 and all(v in placement[0] for v in
        ('tokens=39261 ', 'chunk=8192 ', 'residual_dtype=bf16 ', 'residual_gpu_rows=39261 ', 'residual_host=0 ')), 'actual batched count, chunk and placement; no backoff')
    require('strata prefill: native16 BF16 HC residual/projection IO' in native and
            '[native16-route] engine=bf16 backend=vllm' in native, 'actual BF16/vLLM route')
    completed = re.findall(r'^strata trace: layer-major layer (\d+) of (\d+): completed (\d+) chunks$', native, re.M)
    require(completed == [(str(i), '48', '5') for i in range(48)], 'all48 layers/five chunks normally completed')
    wait = [l for l in native.splitlines() if l.startswith('strata prefill cpu wait:')]
    host = [l for l in native.splitlines() if l.startswith('strata prefill cpu host:')]
    coarse = [l for l in native.splitlines() if l.startswith('strata lm profile:')]
    qsa = [l for l in native.splitlines() if 'qsa profile:' in l]
    require(coarse and qsa and len(wait) == len(host) == 1 and 'available=1 complete=1' in wait[0] and
            'available=1 open_regions=0' in host[0], 'existing closed coarse/QSA/main-thread CPU-wait evidence')
    return dict(suite=suite, suite_file=artifact(suites[0]), placement=placement,
        coarse_profile_lines=coarse, qsa_profile_lines=qsa, cpu_wait_lines=wait, cpu_host_lines=host,
        CPU_interpretation='Issuing thread CLOCK_THREAD_CPUTIME_ID; explicit waits separately recorded. Host API elapsed includes waits, not CPU computation. Other threads/unclassified CPU remain separate.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('tiny', 'model'), required=True)
    parser.add_argument('--run-id', required=True, help='fresh vN; never retry/resume/overwrite')
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--health-receipt', type=Path, required=True, help='Main explicitly chooses latest closed sameboot PASS')
    parser.add_argument('--health-sha256', required=True)
    parser.add_argument('--tiny-receipt', type=Path)
    parser.add_argument('--tiny-sha256')
    parser.add_argument('--raw-budget-mib', type=int, choices=(64, 128, 256), default=64)
    parser.add_argument('--retention-review-utc', required=True, help='explicit UTC review deadline within48h')
    args = parser.parse_args()
    global RAW_CAP
    RAW_CAP = (args.raw_budget_mib << 20) - (2 << 20)
    require(re.fullmatch(r'v[1-9][0-9]{0,3}', args.run_id), 'finite fresh run ID')
    require((args.mode == 'model' and args.tiny_receipt is not None and args.tiny_sha256 is not None) or
            (args.mode == 'tiny' and args.tiny_receipt is None and args.tiny_sha256 is None), 'model requires explicit latest tiny PASS; tiny runs no model')
    require(not any(k.startswith(('UNITRACE_', 'PTI_', 'XPTI_', 'ZE_ENABLE_', 'ZEL_')) or k == 'LD_PRELOAD' for k in os.environ), 'controller cannot be preloaded/inherit collector controls')
    manifest_path = D / 'manifest.json'
    require(sha(manifest_path) == args.manifest_sha256, 'externally selected sealed manifest')
    manifest = json.loads(manifest_path.read_text()); manifest['_sha256'] = args.manifest_sha256
    for name, pin in manifest['package_files'].items():
        check_pin(D / name, pin)
    spec = importlib.util.spec_from_file_location('direct_unitrace_owner', OWNER)
    require(sha(OWNER) == OWNER_SHA, 'qualified Owner source')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    review = datetime.datetime.fromisoformat(args.retention_review_utc.replace('Z', '+00:00'))
    now = datetime.datetime.now(datetime.timezone.utc)
    require(review.tzinfo and review.utcoffset() == datetime.timedelta(0) and
            datetime.timedelta(0) < review - now <= datetime.timedelta(hours=48), 'finite UTC retention deadline')
    require(D == P / 'bench/results/2026-10-15-pr6-recovery-audit/own-vllm-resource-timeline-v1' and B.resolve() == B, 'canonical package/state roots')
    root = B / ('main-vllm-resource-timeline-' + args.mode + '-' + args.run_id)
    target = A / ('own-vllm-resource-timeline-' + args.mode + '-' + args.run_id + '.json')
    whole, native_wall = (300, 90) if args.mode == 'tiny' else (1200, 900)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not root.exists() and not root.is_symlink() and not target.exists() and not target.is_symlink(), 'fresh private root/receipt')
        root.mkdir(mode=0o700); (root / 'collector').mkdir(mode=0o700)
        started = time.monotonic(); deadline = started + whole; work_deadline = deadline - RESERVE
        record = dict(schema='strata-unitrace-direct-v1', mode=args.mode, run_id=args.run_id,
            status='RUNNING', passed=False, complete=False, active=False, GPU_executed=False,
            source_commit=SOURCE, source_pins=manifest['source_pins'],
            binary_sha256=manifest['tiny' if args.mode == 'tiny' else 'engine']['sha256'], engine_sha256=ENGINE_SHA,
            collector_library_sha256=LIB_SHA, collector_compatible=False, manifest_sha256=args.manifest_sha256,
            boot_id=boot_id(), boot_unchanged=False, driver_dump_exists=None, fresh_faults=[],
            commands=[], fault_windows=[], pin_checkpoints=[], owner_scans=[], started_utc=utc(),
            controller=artifact(__file__), controller_identity=identity(os.getpid()),
            python=artifact(sys.executable), invocation=list(sys.orig_argv), private_root=str(root),
            original_failure_status_preserved=True, ownership_closed=False,
            budget=dict(whole_seconds=whole, native_seconds=native_wall, closure_reserve_seconds=RESERVE,
                native_and_side_raw_bytes=RAW_CAP, closure_raw_bytes=2 << 20, Owner_aggregate_bytes=64 << 20, total_raw_bytes=args.raw_budget_mib << 20,
                private_bytes=DISK_CAP, tensors=False),
            retention=dict(owner='Main', consumer='Main direct collector compatibility and BF16 long-prefill review',
                byte_budget=args.raw_budget_mib << 20,
                exception_consumer='Exact whole-command/QSA-envelope copy-compute overlap, per-copy requested bytes, all interval-to-summary counts; text device timeline only, no full API trace',
                review_deadline_utc=review.isoformat(), cleanup_performed=False,
                next_review='After aggregate/oracle verification, retain compact evidence; Main handles serialized raw-log retirement'),
            **{k: False for k in QUALIFICATION})
        owner = None
        closing = settling = False

        def budget():
            files = [p for p in root.rglob('*') if p.is_file()]
            private = sum(p.stat().st_size for p in files)
            raw = sum(p.stat().st_size for p in files if p.is_relative_to(root / 'collector') or
                p.suffix in ('.stdout', '.stderr', '.log', '.jsonl', '.partial'))
            record.update(private_bytes=private, raw_text_bytes=raw)
            require(private <= DISK_CAP and raw <= ((args.raw_budget_mib << 20) if closing else RAW_CAP), 'whole private/log/collector-side budget')

        def persist():
            record['active'] = owner is not None and owner.active is not None
            problem = None
            if owner:
                try:
                    budget()  # Every Owner scan, not only stdout/stderr; includes ALL collector sides.
                except BaseException as error:
                    record['budget_error'] = repr(error)
                    record['passed'] = record['complete'] = record['collector_compatible'] = False
                    problem = error
            (root / 'record.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
            if problem and record['active'] and not settling:
                raise problem  # Owner catches, pins and closes its own session.

        owner = module.Owner(root / 'owned', persist); record['commands'] = owner.commands
        persist()
        cursor = None

        def remaining(cap):
            end = deadline if closing else work_deadline
            require(time.monotonic() < end, 'controller parent deadline')
            return min(cap, end - time.monotonic())

        def utility(label, argv):
            c, out, err = owner.run(label, argv, HOST_ENV, P, wall=remaining(12), cpu=60,
                rss_cap=2 << 30, text_cap=2 << 20, total_cap=DISK_CAP, file_cap=2 << 20)
            require(module.closed(c) and c.get('exit_code') == 0, 'normal utility closure: ' + label)
            absence([c]); budget()
            return out.read_text()

        def frozen(label):
            remaining(1)
            require(boot_id() == record['boot_id'] and sha(manifest_path) == args.manifest_sha256, 'frozen boot/manifest')
            for name, pin in manifest['package_files'].items():
                check_pin(D / name, pin)
            for path, pin in manifest['file_pins'].items():
                check_pin(path, pin)
            for name, digest in manifest['source_pins'].items():
                require(sha(P / name) == digest, 'qualified source freeze: ' + name)
            for path, pin in manifest['model_stat_pins'].items():
                st = safe_path(path).stat()
                require(st.st_size == pin['bytes'] and st.st_mtime_ns == pin['mtime_ns'], 'model stat identity (not full shard SHA)')
            record['pin_checkpoints'].append(dict(label=label, checked_utc=utc(), source_files=len(manifest['source_pins']),
                build_receipt_sha256=BUILD_SHA, products=len(manifest['products']), file_pins=len(manifest['file_pins'])))
            remaining(1)

        def driver(label):
            require(boot_id() == record['boot_id'] and (DEVICE / 'driver').resolve().name == 'xe' and
                    (DEVICE / 'vendor').read_text().strip() == '0x8086', 'sameboot Intel xe device')
            dumps = list(Path('/sys/class/devcoredump').glob('devcd*'))
            record['driver_dump_exists'] = (DEVICE / 'devcoredump').exists() or bool(dumps)
            record['boot_unchanged'] = True
            snap = dict(label=label, kernel=os.uname().release, driver='xe',
                pci={n: (DEVICE / n).read_text().strip() for n in ('vendor','device','revision','subsystem_vendor','subsystem_device')},
                dumps=list(map(str, dumps)))
            previous = record.setdefault('driver_snapshots', [])
            if previous:
                require(snap['pci'] == previous[0]['pci'] and snap['kernel'] == previous[0]['kernel'], 'driver identity unchanged')
            previous.append(snap)
            require(not record['driver_dump_exists'], 'driver dump present; no reset/deletion')

        try:
            frozen('preflight')
            for component in manifest['component_receipts']:
                check_pin(component['path'], component)
                gate = json.loads(Path(component['path']).read_text())
                require(gate['passed'] and gate['complete'] and not gate['active'] and gate['ownership_closed'] and
                        gate['boot_id'] == record['boot_id'] and gate['source_pins'] == manifest['source_pins'] and
                        gate['source_commit'] == SOURCE and gate['fresh_faults'] == [] and gate['driver_dump_exists'] is False and
                        gate['products'] == manifest['products'] and all(module.closed(c) and c['exit_code'] == 0 for c in gate['commands']),
                        'exact sameboot current components and tiny/smoke oracle prerequisite')
                absence(gate['commands'])
            record['health_selection'] = health_gate(args.health_receipt, args.health_sha256, module, record['boot_id'], manifest)
            if args.mode == 'model':
                record['tiny_selection'] = health_gate(args.tiny_receipt, args.tiny_sha256, module, record['boot_id'], manifest, tiny=True)
            record['owner_scans'].append(owner_scan(manifest['model_stat_pins']))
            driver('before')
            profile = manifest['profile']
            base_env = dict(profile['environment'])
            for name in ('STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT'):
                base_env.pop(name, None)
            base_env.update(XDG_CACHE_HOME=str(root / 'cache'), **WARNINGS)
            native_env = dict(base_env, **COLLECTOR, LD_PRELOAD=str(LIB), UNITRACE_LogFilename=str(root / 'collector/timing.log'))
            require(not any(k.startswith(('PTI_', 'XPTI_', 'ZEL_', 'IGC_')) for k in native_env) and
                    native_env['UR_ENABLE_LAYERS'] == 'UR_LAYER_PARAMETER_VALIDATION', 'no conditional control/XPTI/full API trace environment')
            require(set(k for k in native_env if k.startswith('UNITRACE_')) == set(k for k in COLLECTOR if k.startswith('UNITRACE_')) | {'UNITRACE_LogFilename'}, 'only exact timing/text-timeline/verbose native controls; no metrics/Chrome/tiles/filter/API logs')
            if args.mode == 'tiny':
                native_env['STRATA_SYCL_PREFILL_NATIVE16_BACKEND'] = 'own'
            record['native_environment'] = native_env
            record['host_environment'] = base_env
            match = re.search(r'-- cursor: (\S+)', utility('journal-before', ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))
            require(match is not None, 'fresh kernel journal cursor')
            cursor = match[1]
            record['fault_windows'].append(dict(command='native-smoke' if args.mode == 'tiny' else 'official-bench', before_cursor=cursor, boot_id=record['boot_id']))
            if args.mode == 'tiny':
                exe = manifest['tiny']
                proof = {}

                def smoke_io(io, entry):
                    record['GPU_executed'] = True
                    io.proc.stdin.close()
                    with selectors.DefaultSelector() as ready:
                        ready.register(io.proc.stdout, selectors.EVENT_READ)
                        while not io.exit_watch.poll(0) or not io.eof:
                            io.poll(); remaining(1)
                            if not proof and not io.exit_watch.poll(0):
                                snap = native_snapshot(io.proc.pid, exe, native_env, manifest['elf_pins'], io.proc.pid)
                                if snap:
                                    proof.update(snap)
                            if ready.select(.002):
                                io.read(); io.pending.clear()

                c, out, err = owner.run('native-smoke', [exe['path'], '--smoke'], native_env, root / 'collector', interaction=smoke_io,
                    wall=remaining(native_wall), cpu=180, rss_cap=8 << 30, text_cap=RAW_CAP, total_cap=DISK_CAP, file_cap=RAW_CAP)
                require(module.closed(c) and c.get('exit_code') == 0, 'normal tiny owned native closure')
                absence([c]); record['actual_native'] = proof
                require(proof and proof['native']['pid'] == c['pid'] and proof['native']['start_ticks'] == c['direct_identity']['start_ticks'], 'actual tiny PID/start/library capture')
                stdout, stderr = out.read_text(), err.read_text()
                record['oracle_stdout'] = stdout
                rows = [l for l in stdout.splitlines() if l.startswith('case dtype=bf16 ')]
                require(len(rows) == 2 and all('oracle=FP64_full ' in l and
                    'finite=PASS guards=PASS inputs=PASS qualification=COMPONENT_ONLY_UNQUALIFIED' in l for l in rows) and
                    stdout.count('fixture=PASS model_quality=UNRUN whole_prefill=UNRUN physical262144=UNRUN faults=EXTERNAL_REQUIRED') == 1 and
                    not ERROR.search(stderr), 'unchanged tiny FP64 oracle/finite/guards/input checks')
                record['collection'] = parse_collection(root / 'collector', c['pid'])
                record['wrapper_wall_seconds'] = c['elapsed_seconds']
            else:
                require(profile['config']['host'] == '127.0.0.1' and profile['config']['parallel'] == 1, 'loopback/single request config')
                args_native = list(profile['config']['args'])
                external = Path(args_native[args_native.index('--expert-profile') + 1])
                local = root / 'expert-profile.bin'; shutil.copyfile(external, local)
                record['expert_profile'] = dict(external=artifact(external), local_initial=artifact(local))
                args_native[args_native.index('--expert-profile') + 1] = str(local)
                wrapper_spec = dict(engine=manifest['engine'], native_environment=native_env, elf_pins=manifest['elf_pins'],
                    timeline_helper=manifest['package_files']['timeline.py'],
                    native_cwd=str(root / 'collector'), start_path=str(root / 'engine-start.json'), exit_path=str(root / 'engine-exit.json'))
                spec_path = root / 'native-spec.json'; spec_path.write_text(json.dumps(wrapper_spec, indent=2) + '\n')
                wrapper = root / 'engine-wrapper.py'
                wrapper.write_text('#!' + PY + '\nimport hashlib,runpy,sys\n'
                    + 'p=' + repr(str(Path(__file__).resolve())) + '\n'
                    + 'with open(p,"rb") as f: assert hashlib.file_digest(f,"sha256").hexdigest()==' + repr(sha(__file__)) + '\n'
                    + 'sys.exit(runpy.run_path(p)["native_wrapper"](' + repr(str(spec_path)) + ',sys.argv[1:]))\n')
                wrapper.chmod(0o700)
                config = dict(profile['config'], exe=str(wrapper), args=args_native, env={}, log=str(root / 'owned/engine.stderr'))
                plan = dict(port=18489, url='http://127.0.0.1:18489', runs=1,
                    benchmark_options=profile['benchmark_options'], bench_timeout_seconds=720)
                (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
                (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
                record.update(config=config, plan=plan, wrapper=artifact(wrapper), native_spec=artifact(spec_path),
                    generated_inputs=[artifact(root / n) for n in ('engine-wrapper.py','native-spec.json','config.json','plan.json')])
                c, out, err = owner.run('official-bench', [PY, str(RUNNER), str(root), 'extended-bench'], base_env, P,
                    wall=remaining(native_wall), cpu=21600, rss_cap=88 << 30, text_cap=RAW_CAP, total_cap=DISK_CAP, file_cap=RAW_CAP)
                record['GPU_executed'] = (root / 'engine-start.json').exists()
                require(module.closed(c) and c.get('exit_code') == 0, 'normal official bench owned session')
                absence([c])
                result = json.loads((root / 'result.json').read_text()); record['result'] = result
                for pin in record['generated_inputs']:
                    check_pin(pin['path'], pin)
                record['model_evidence'] = validate_model(root, result)
                entry = result['real_engine_exit']
                require(entry['experiment_environment'] == native_env and entry['native']['sid'] == c['pid'] and
                        entry['native']['ppid'] == entry['wrapper']['pid'], 'actual model native immutable environment/session')
                record['actual_native'] = entry['collector_snapshot']
                require(record['actual_native']['native']['start_ticks'] == entry['native']['start_ticks'] and
                        record['actual_native']['native']['pid'] == entry['native']['pid'], 'actual model collector mapping bound to exec')
                record['collection'] = parse_collection(root / 'collector', entry['native']['pid'], record['model_evidence']['suite'])
                verify = utility('verify-request-journal', ['/usr/bin/python3','-B','-c',
                    'import sys,json;sys.path.insert(0,"tools");from strata_bench import verify_request_journal;print(json.dumps(verify_request_journal(sys.argv[1])))',
                    str(Path(record['model_evidence']['suite_file']['path']).parent)])
                record['request_journal_verification'] = json.loads(verify)
                require(record['request_journal_verification']['valid'] and record['request_journal_verification']['complete'] and
                        not record['request_journal_verification']['pending'], 'official request identity journal')
                require(sha(external) == record['expert_profile']['external']['sha256'], 'external profile untouched')
                record['wrapper_wall_seconds'] = c['elapsed_seconds']
            budget()
            frozen('post-native')
            record['candidate_collection_complete'] = True
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            closing = True
            try:
                if owner.active is None and cursor:
                    boundary = utility('journal-end-cursor', ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'])
                    cursor_match = re.search(r'-- cursor: (\S+)', boundary)
                    require(cursor_match is not None, 'final kernel journal cursor')
                    # End boundary is obtained BEFORE the after-cursor query:
                    # the query covers that boundary and any later rows it sees.
                    # JSON mode has no artificial "no entries" row to parse.
                    text = utility('journal-after', ['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json'])
                    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
                    require(all(v.get('_BOOT_ID') == record['boot_id'].replace('-', '') for v in rows), 'journal sameboot identity')
                    fresh = [v for v in rows if FAULT.search(str(v.get('MESSAGE', '')))]
                    record['fresh_faults'] = fresh
                    record['fault_windows'][-1].update(after_cursor=cursor_match[1],
                        last_returned_cursor=rows[-1].get('__CURSOR') if rows else None, checked_utc=utc(),
                        fresh_faults=fresh, fault_filter=FAULT.pattern, row_count=len(rows), complete=True)
                    require(not fresh, 'fresh GPU faults')
                require(cursor and record['fault_windows'][-1].get('complete'), 'per-command fault window complete')
                driver('after')
                frozen('closure')
                record['final_PID_start_absence'] = absence(owner.commands)
                if (root / 'engine-start.json').exists():
                    entry = json.loads((root / 'engine-start.json').read_text())
                    record['wrapper_native_PID_start_absence'] = absence([dict(
                        direct_identity=entry['wrapper'], owners=[entry['wrapper'], entry['native']])])
                record['owner_scans'].append(owner_scan(manifest['model_stat_pins']))
                for selection in ('health_selection', 'tiny_selection'):
                    if selection in record:
                        pin = record[selection]['receipt']; check_pin(pin['path'], pin)
                budget(); remaining(1)
                record['ownership_closed'] = owner.active is None and all(module.closed(c) and c.get('exit_code') == 0 for c in owner.commands)
                require(record['ownership_closed'], 'every command normally reaped/session empty')
                record['passed'] = bool(record.get('candidate_collection_complete') and not record.get('error') and not record.get('budget_error'))
                record['complete'] = record['passed']
                record['collector_compatible'] = record['passed']
            except BaseException as error:
                record['closure_error'] = repr(error)
                record['passed'] = record['complete'] = record['collector_compatible'] = False
            record['active'] = owner.active is not None
            # Inventory failure/partial sides too; never silently erase or count
            # an unexpected side as successful collection. No cleanup here.
            record['collector_sides'] = [(artifact(p) if owner.active is None else
                dict(path=str(p), bytes=p.stat().st_size, status='ACTIVE_UNSEALED'))
                for p in sorted((root / 'collector').rglob('*')) if p.is_file()]
            record['status'] = 'PASS' if record['passed'] else 'FAIL'
            record.update(finished_utc=utc(), elapsed_seconds=time.monotonic() - started)
            # Failure remains a failure even when a valid partial table survived.
            settling = True
            persist()
            record['status'] = 'PASS' if record['passed'] else 'FAIL'
            with target.open('x') as stream:
                json.dump(record, stream, indent=2, allow_nan=False); stream.write('\n')
        print(json.dumps(dict(status=record['status'], receipt=str(target), sha256=sha(target),
            clean_speed=False, adoption=False)), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
