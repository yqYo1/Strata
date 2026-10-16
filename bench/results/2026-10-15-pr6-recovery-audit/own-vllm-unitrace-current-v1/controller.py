#!/usr/bin/env python3
"""Main-only bounded current6354 unitrace diagnostics. Repository LICENSE.

SOURCE-ONLY / UNRUN delivery. No builds, retries, reset, service management,
artifact cleanup, global environment changes, or performance/quality adoption.
Only main() launches work; importing this module does not inspect processes.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import sys
import time

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
A = D.parent
P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
PY = '/home/yayoi/.local/share/strata-sycl/venv/bin/python'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
RUNNER = P / 'bench/results/2026-10-14-long-generation-benchmark/run_server_job.py'
TEMPLATE = A / 'qsa-device-batch-model-wrapper-v1/engine_wrapper.py'
BUILD = A / 'own-vllm-engine-build-v2.json'
BUILD_SHA = '990635731be1198706abeb682817fdab247f8f51d31bab971d3d4f4db53cb639'
BINARY_SHA = '85643f3926bb9d212945659cb7ddb264e101ca9b2d5fa8f19430e945140ad191'
COMMIT = '6354f0b4ff3731466795410e11ca9ef785c9d326'
SCHEMA = 'strata-vllm-unitrace-current-v1'
SOURCE_BLOCKER = (
    'Official pinned pti-gpu6c0d6b0 --result-dir calls CreateConfigLog/StringifyDevice/zeInit, '
    'then main changes UNITRACE_LD_PRELOAD_OLD, ZE_ENABLE_TRACING_LAYER, ZES_ENABLE_SYSMAN '
    'and LD_PRELOAD before execvp. This violates the required process-environment immutability '
    'after provider initialization. The coordinator confirmed no pre-exec launcher exception '
    'is authorized. Current tool qualification and every launch remain UNRUN; Main must '
    'choose another method separately. No bypass flag is provided.')
HOST = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
ERROR = re.compile(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed|CHECKED_DRAIN_FAILED|FIXTURE_FAILED|\[ERROR\]|\b(?:NaN|Inf)\b', re.I)
SAFE = dict(ONEAPI_DEVICE_SELECTOR='level_zero:gpu', OMP_NUM_THREADS='1', NEOReadDebugKeys='1',
    EnableDirectSubmission='0', UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', SYCL_CACHE_PERSISTENT='0',
    STRATA_SYCL_PREFILL_SAME_QUEUE='1', STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0', STRATA_ROPE_TABLE='1',
    STRATA_SYCL_PROMPT_DIRECT='1', STRATA_PREFILL_LAYER_MAJOR='1', STRATA_PREFILL_RING='8',
    STRATA_PREFILL_NATIVE_DENSE_BANK='1', STRATA_PREFILL_CANONICAL_DENSE_BANK='1',
    STRATA_SYCL_PREFILL_SCRATCH_ALIAS='1')
VALIDATION = dict(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
    UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
    UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
    UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
LIMITATION = ('Diagnostic whole-native-process Level Zero device execution/host API aggregates, including '
    'startup, initial prefill, 24-token decode and teardown. Per-name sums/count/min/max are not a '
    'busy-time union, exclusive Amdahl fractions, critical path, clean speed or quality evidence. '
    'Coarse prefill windows/CPU counters can inform source attribution but cannot isolate these '
    'whole-process kernel rows to prefill. No timeline, full API trace or phase subtraction is enabled.')


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def utc():
    return dt.datetime.now(dt.timezone.utc)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def allowed(path):
    path = Path(path).resolve()
    llvm = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
    require('XeStrata' not in str(path) or path.is_relative_to(llvm), 'excluded implementation path')
    require(not path.is_relative_to(B / 'prefill-gemm-only-contract-root-v1'), 'protected path forbidden')
    return path


def artifact(path):
    path = allowed(path)
    require(path.is_file(), 'regular artifact required: ' + str(path))
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def pin(path, digest, size=None):
    path = allowed(path)
    require(path.is_file() and (size is None or path.stat().st_size == size) and sha(path) == digest,
            'file pin mismatch: ' + str(path))
    return path


def read_json(path, digest=None, cap=16 << 20):
    path = allowed(path)
    require(path.is_file() and path.stat().st_size <= cap, 'bounded regular JSON: ' + str(path))
    if digest is not None:
        pin(path, digest)
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    def bad(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=bad)


def bound_artifacts(value, result):
    """Pin consumed current receipt artifacts through final closure, including captures."""
    if isinstance(value, dict):
        if all(k in value for k in ('path', 'bytes', 'sha256')) and isinstance(value['path'], str):
            require(Path(value['path']).is_absolute() and type(value['bytes']) is int and
                re.fullmatch('[0-9a-f]{64}', value['sha256'] or ''), 'bound artifact identity')
            item = {k: value[k] for k in ('path', 'bytes', 'sha256')}
            path = str(allowed(item['path']))
            if path in result:
                require(result[path]['sha256'] == item['sha256'] and result[path]['bytes'] == item['bytes'], 'conflicting frozen artifact')
            result[path] = item
            require(len(result) <= 4096, 'finite external artifact bindings')
        for child in value.values():
            bound_artifacts(child, result)
    elif isinstance(value, list):
        for child in value:
            bound_artifacts(child, result)


def module(name, path, manifest):
    expected = manifest['external_files'][str(Path(path).resolve())]
    pin(path, expected['sha256'], expected['bytes'])
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def absence(owner_module, commands):
    rows = []
    for c in commands:
        require(c.get('direct_identity') and c.get('owners'), 'original PID/start/session evidence missing')
        for old in c['owners'] + [c['direct_identity']]:
            require(type(old.get('pid')) is int and type(old.get('start_ticks')) is int, 'PID/start types')
            now = owner_module.identity(old['pid'])
            absent = now is None or now['start_ticks'] != old['start_ticks']
            rows.append(dict(original=old, current=now, original_absent=absent))
            require(absent, 'original PID/start still present; no adoption/signals authorized')
    return rows


def closed_gate(value, owner_module, boot):
    require(value.get('passed') is True and value.get('complete') is True and value.get('active') is False and
        value.get('ownership_closed') is True and value.get('commands') and not value.get('error') and
        not value.get('closure_errors') and not value.get('final_fault_check_error'), 'original normal PASS required')
    require(value.get('boot_id') == boot and value.get('boot_unchanged') is True and
        value.get('fresh_faults') == [] and value.get('driver_dump_exists') is False, 'sameboot clean final fault/dump evidence')
    require(all(owner_module.closed(c) and c.get('exit_code') == 0 for c in value['commands']), 'all original commands normally closed')
    return absence(owner_module, value['commands'])


def prior_gate(path, digest, build, M, S, deadline):
    require(re.fullmatch('[0-9a-f]{64}', digest or ''), 'Main must supply latest closed current-engine receipt+SHA')
    require(digest != '4f88862459b71fc9c64d7831ad043ded677578d33047ef22896c8671bd51968f',
            'historical selected-smoke-v2 is provenance only; Main must select a newer closed current-engine receipt')
    value = read_json(path, digest)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    absent = closed_gate(value, M, boot)
    require(value.get('GPU_executed') is True and value.get('source_commit') == COMMIT and
        value.get('source_pins') == build['source_pins'] and value.get('binary_sha256') == BINARY_SHA,
        'same6354 engine/source binary, not historical collector health')
    if value.get('schema') == S.SCHEMA:
        # Exact pinned v2 parser validates the schema, binding, individual command,
        # recipe, request, capture and math-oracle evidence; false quality flags
        # are legitimate and required. This is bounded read-only CPU validation.
        S.closed_receipt(value, boot=boot, gpu=True)
        verified = S.compare_record(value, deadline=deadline)
        require(verified.get('valid') is True and value.get('comparison', {}).get('valid') is True and
            verified['binding_sha256'] == value['binding_sha256'] == value['comparison']['binding_sha256'],
            'exact selected-v2 binding/comparison')
        proof = dict(schema=value['schema'], mode=value['mode'], binding_sha256=value['binding_sha256'],
                     exact_selected_v2_replay_valid=True, quality_accepted=False)
    else:
        require(value.get('schema') == SCHEMA and value.get('mode') == 'model' and value.get('collected') is True and
            value.get('collector_contract_sha256') == read_json(D / 'manifest.json')['collector_contract_sha256'] and
            value.get('performance_qualified') is False and value.get('quality_qualified') is False,
            'only exact selected-v2 or this normally closed current-engine profile schema')
        proof = dict(schema=SCHEMA, arm=value['arm'], diagnostic_only=True)
    require(dt.datetime.fromisoformat(value['finished_utc']).tzinfo is not None, 'prior closure timestamp')
    return value, dict(artifact=artifact(path), original_PID_start_absence=absent, binding=proof,
        selection='Main explicitly supplies newest normally closed current-engine PASS; no autodiscovery, fallback or old PASS substitution')


def parse_table(path, kind):
    """Parse official comma-delimited numeric suffix; C++ names may contain commas."""
    text = Path(path).read_text()
    first = 'Kernel' if kind == 'device' else 'Function'
    header = [first, 'Calls', 'Time (ns)', 'Time (%)', 'Average (ns)', 'Min (ns)', 'Max (ns)']
    active = False
    rows = []
    for line in text.splitlines():
        if [v.strip() for v in line.split(',')] == header:
            require(not active and not rows, 'one official L0 timing table')
            active = True
            continue
        if active:
            if not line.strip():
                active = False
                continue
            fields = [v.strip() for v in line.rsplit(',', 6)]
            require(len(fields) == 7 and fields[0], 'timing row numeric suffix')
            require(all(re.fullmatch('[0-9]+', fields[i]) for i in (1, 2, 4, 5, 6)), 'integer ns/count columns')
            calls, total, avg, lo, hi = (int(fields[i]) for i in (1, 2, 4, 5, 6))
            percent = float(fields[3])
            require(calls > 0 and math.isfinite(percent) and 0 <= percent <= 100.000001 and
                lo <= avg <= hi and total // calls == avg and lo * calls <= total <= hi * calls, 'valid kernel/API statistics')
            rows.append(dict(name=fields[0], calls=calls, time_ns=total, time_percent=percent,
                average_ns=avg, min_ns=lo, max_ns=hi))
            require(len(rows) <= 8192, 'finite timing row count')
    require('== L0 Backend ==' in text and rows and sum(r['time_ns'] for r in rows) > 0,
            'actual nonempty positive L0 kernel/API timing records required')
    return dict(artifact=artifact(path), rows=rows, calls=sum(r['calls'] for r in rows),
        summed_time_ns=sum(r['time_ns'] for r in rows),
        interpretation='Sum of measured executions/API durations, not busy union/critical path/exclusive prefill time')


def provider_records(directory, native_pid, argv, native_argv):
    files = [p for p in directory.rglob('*') if p.is_file()]
    require(1 <= len(files) <= 16 and all(not p.is_symlink() for p in directory.rglob('*')), 'bounded private provider files without aliases')
    require(all(p.name in ('host_timing.txt', 'device_timing.txt', 'run_config.json') for p in files),
            'unexpected provider output; timeline/full API/metrics forbidden')
    configs = [p for p in files if p.name == 'run_config.json']
    device = [p for p in files if p.name == 'device_timing.txt']
    host = [p for p in files if p.name == 'host_timing.txt']
    require(len(configs) == len(device) == len(host) == 1, 'one native config plus both timing tables')
    config = read_json(configs[0], cap=64 << 10)
    require(int(config['pid']) == native_pid and config['app'] == native_argv[0] and
        config['app_args'] == native_argv[1:] and config['unitrace_args'] == argv[1:5], 'provider native PID/argv/CLI proof')
    # Config is emitted before exec under unitrace.PID. Native timing uses its
    # own (Linux comm, possibly15-character) name.PID, preserving that same PID.
    require(device[0].parent == host[0].parent and
        all(p.parent.name.endswith('.' + str(native_pid)) for p in (device[0], host[0], configs[0])) and
        configs[0].parent.name == 'unitrace.' + str(native_pid), 'provider outputs tied to actual native PID across exec')
    kernels, api = parse_table(device[0], 'device'), parse_table(host[0], 'host')
    actual = [r for r in kernels['rows'] if not r['name'].startswith('zeCommandListAppend')]
    require(actual and any(r['name'].startswith('ze') for r in api['rows']), 'real compute kernel names plus Level Zero host calls')
    return dict(files=[artifact(p) for p in sorted(files)], config=config,
        device=kernels, host=api, actual_compute_name_count=len(actual), exclusive_prefill=False)


def cli(tool, reports, exe, args):
    return [str(tool), '--device-timing', '--host-timing', '--result-dir', str(reports), str(exe), *args]


def make_wrapper(build, tool, provider, root, env):
    """Frozen project wrapper adaptation; unitrace exec replaces its own image."""
    template = TEMPLATE.read_text()
    require(template.count('import subprocess\n') == 1, 'one wrapper subprocess import seam')
    template = template.replace('import subprocess\n', 'import subprocess\nimport time\n')
    template = template.replace("assert env['STRATA_SYCL_PROMPT_DIRECT'] == '0'", "assert env['STRATA_SYCL_PROMPT_DIRECT'] == '1'")
    template, count = re.subn(r'^ENGINE = .*?$', 'ENGINE = ' + repr(build['binary']), template, flags=re.M)
    require(count == 1, 'one original engine seam')
    template, count = re.subn(r'^ENGINE_SHA256 = .*?$', 'ENGINE_SHA256 = ' + repr(BINARY_SHA), template, flags=re.M)
    require(count == 1, 'one original engine hash seam')
    old = '    child = subprocess.Popen([ENGINE, *sys.argv[1:]], env=env)'
    require(template.count(old) == 1, 'one original launcher seam')
    prefix = cli(tool, provider, build['binary'], [])[:-1]
    replacement = ('    launch_argv = ' + repr(prefix) + ' + [ENGINE, *sys.argv[1:]]\n'
        '    native_launch_begin_monotonic = time.monotonic()\n'
        '    child = subprocess.Popen(launch_argv, env=env)')
    template = template.replace(old, replacement)
    anchor = "    pipe = os.stat('/proc/' + str(child.pid) + '/fd/0')"
    require(template.count(anchor) == 1, 'one original identity seam')
    template = template.replace(anchor,
        "    entry['launcher_argv'] = launch_argv\n"
        "    entry['native_launch_begin_monotonic'] = native_launch_begin_monotonic\n"
        "    entry['launcher_environment'] = env\n"
        "    entry['experiment_environment'] = {k: env[k] for k in " + repr(tuple(env)) + "}\n"
        "    entry['launcher_semantics'] = 'official6c non-metrics execvp; native identity requires external exe/env/maps observation'\n" + anchor)
    # Publish each identity receipt once atomically: the external observer must
    # not parse a write-in-progress start file. Native streams remain inherited.
    anchor = 'def main():\n'
    require(template.count(anchor) == 1, 'one wrapper main seam')
    template = template.replace(anchor,
        "def publish(path, value):\n"
        "    path = Path(path)\n"
        "    temporary = Path(str(path) + '.partial')\n"
        "    assert not path.exists() and not temporary.exists()\n"
        "    with temporary.open('x') as stream:\n"
        "        stream.write(value)\n"
        "    os.replace(temporary, path)\n\n" + anchor)
    for key in ('STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT'):
        old = "Path(os.environ['" + key + "']).write_text(json.dumps(entry, indent=2) + '\\n')"
        require(template.count(old) == 1, 'one wrapper identity publication: ' + key)
        template = template.replace(old, "publish(os.environ['" + key + "'], json.dumps(entry, indent=2) + '\\n')")
    wrapper = root / 'engine-wrapper.py'
    compile(template, str(wrapper), 'exec')
    with wrapper.open('x') as stream:
        stream.write(template)
    wrapper.chmod(0o700)
    return wrapper


class NativeObserver:
    """Read only known PID/start identities; Owner alone signals/reaps sessions."""
    def __init__(self, M, record, root, expected_exe, expected_env, expected_argv, tool_lib, wall):
        self.M, self.record, self.root = M, record, root
        self.exe, self.env, self.argv = Path(expected_exe).resolve(), expected_env, expected_argv
        self.tool_lib, self.wall = str(Path(tool_lib).resolve()), wall
        self.first = None
        self.key = None
        self.hashes = {}
        self.next_maps = 0
        self.evidence = dict(observed=False, environment_observations=0, libraries={}, transitions=[])

    def poll(self, entry, *, model):
        if model:
            start = self.root / 'engine-start.json'
            if not start.exists():
                return
            start = read_json(start, cap=2 << 20)
            old = start['native']
            require(start['engine'] == str(self.exe) and start['binary_sha256'] == BINARY_SHA and
                start['argv'] == self.argv and start['launcher_environment'] == self.env and
                start['experiment_environment'] == self.env and start['native']['ppid'] == start['wrapper']['pid'],
                'actual frozen-wrapper launcher/native argv/environment relationship')
        else:
            old = entry['direct_identity']
        key = (old['pid'], old['start_ticks'])
        if self.key is None:
            self.key = key
            self.first = start['native_launch_begin_monotonic'] if model else entry['spawn_begin_monotonic']
            require(math.isfinite(self.first) and 0 < self.first <= time.monotonic(), 'actual launch monotonic boundary')
        require(key == self.key, 'one independent native PID/start')
        now = self.M.identity(old['pid'])
        if now is None or now['start_ticks'] != old['start_ticks'] or now['state'] == 'Z':
            self.evidence.setdefault('native_absence_or_zombie_observed_monotonic', time.monotonic())
            return
        require(time.monotonic() - self.first <= self.wall, 'native900s/component90s deadline')
        require(now['sid'] == entry['direct_identity']['sid'] and
            (model or now['ppid'] == os.getpid()), 'native stays in actual anchored session/direct relationship')
        proc = Path('/proc', str(old['pid']))
        try:
            exe = (proc / 'exe').resolve(strict=True)
            if not self.evidence['transitions'] or self.evidence['transitions'][-1] != str(exe):
                self.evidence['transitions'].append(str(exe))
            if exe != self.exe:
                return  # CLI initialization, not evidence of native execution.
            raw = (proc / 'environ').read_bytes()
            cmdline = [v.decode() for v in (proc / 'cmdline').read_bytes().split(b'\0') if v]
            observed = dict(v.decode().split('=', 1) for v in raw.split(b'\0') if v)
            require(cmdline == self.argv and all(observed.get(k) == v for k, v in self.env.items()),
                'actual native argv/env differ from launcher')
            require(observed.get('LD_PRELOAD') == self.tool_lib and observed.get('ZE_ENABLE_TRACING_LAYER') == '1' and
                observed.get('UNITRACE_DeviceTiming') == observed.get('UNITRACE_HostTiming') == '1' and
                observed.get('UNITRACE_UseResultDirectory') == '1', 'actual CLI-only child injection')
            expected_directory = self.root / 'provider' if model else self.root / 'provider' / entry['label']
            require(observed.get('UNITRACE_ResultDirectory') == str(expected_directory), 'actual private result directory')
            if model:
                wrapper_now = self.M.identity(start['wrapper']['pid'])
                require(wrapper_now and wrapper_now['start_ticks'] == start['wrapper']['start_ticks'] and
                    now['ppid'] == wrapper_now['pid'] and now['sid'] == wrapper_now['sid'] and
                    start['launcher_argv'] == cli(Path(self.tool_lib).parent / 'unitrace', expected_directory, self.exe, self.argv[1:]),
                    'actual wrapper/native parent/session and prefixed launcher argv')
            require(not any(('Chrome' in k or 'Metric' in k or 'CallLogging' in k or 'DeviceTimeline' in k) and v not in ('', '0')
                for k, v in observed.items() if k.startswith('UNITRACE_') and k != 'UNITRACE_ChromeEventBufferSize'),
                'no unrequested timeline/API/metrics')
            require(observed.get('UNITRACE_ChromeEventBufferSize') == '-1', 'official CLI default buffer size; no Chrome collector enabled')
            if self.evidence['observed']:
                require(observed == self.evidence['environment'], 'native environment must stay immutable after exec')
            self.evidence.update(observed=True, identity=now, argv=cmdline, executable=str(exe),
                environment=observed, environment_sha256=hashlib.sha256(raw).hexdigest(),
                environment_observations=self.evidence['environment_observations'] + 1)
            if 'stdio' not in self.evidence:
                stdio = {}
                for fd in (0, 1, 2):
                    st = (proc / 'fd' / str(fd)).stat()
                    stdio[str(fd)] = dict(device=st.st_dev, inode=st.st_ino, mode=st.st_mode)
                if model:
                    require(stdio['0']['device'] == start['stdin']['device'] and
                        stdio['0']['inode'] == start['stdin']['inode'], 'actual native stdin remains original server pipe')
                self.evidence['stdio'] = stdio
            require(now['rss_bytes'] <= (88 << 30 if model else 4 << 30), 'native RSS budget')
            if time.monotonic() >= self.next_maps:
                maps = (proc / 'maps').read_text()
                require(len(maps) <= 8 << 20, 'finite native maps')
                for row in maps.splitlines():
                    fields = row.split(None, 5)
                    if len(fields) != 6 or not fields[5].startswith('/'):
                        continue
                    mapped = fields[5]
                    require(not mapped.endswith(' (deleted)'), 'mapped file deleted during measurement')
                    if '.so' not in Path(mapped).name:
                        continue
                    mapped = str(allowed(mapped))
                    if mapped not in self.hashes:
                        self.hashes[mapped] = artifact(mapped)
                        require(len(self.hashes) <= 128, 'bounded mapped DSO identities')
                self.evidence['libraries'] = dict(self.hashes)
                self.next_maps = time.monotonic() + .5
        except (FileNotFoundError, ProcessLookupError):
            now = self.M.identity(old['pid'])
            require(now is None or now['start_ticks'] != old['start_ticks'] or now['state'] == 'Z', 'live native observation became unavailable')

    def finish(self, entry, *, model):
        require(self.evidence['observed'] and self.key is not None, 'actual native image/env observation unavailable; never infer from CLI0')
        require(self.tool_lib in self.hashes and
            any(Path(p).name.startswith('libur_adapter_level_zero_v2.so') for p in self.hashes),
            'actual collector and forced URv2 library mappings required')
        require(not any(Path(p).name.startswith('libur_adapter_level_zero.so') for p in self.hashes), 'unexpected legacy adapter')
        require(self.key in {(v['pid'], v['start_ticks']) for v in entry['owners']}, 'native identity actually supervised by Owner')
        if model:
            end = read_json(self.root / 'engine-exit.json', cap=2 << 20)
            require(end['exit_code'] == 0 and (end['native']['pid'], end['native']['start_ticks']) == self.key and
                end['native']['ppid'] == end['wrapper']['pid'] and end['launcher_environment'] == self.env,
                'wrapper actually waited/reaped its native exec replacement normally')
            self.evidence['wrapper_exit'] = end
        self.evidence['native_binary'] = artifact(self.exe)
        return self.evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('component', 'model'))
    ap.add_argument('--arm', choices=('fp32', 'bf16', 'qsa'))
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--manifest-sha256', required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--prior-sha256', required=True)
    ap.add_argument('--component', type=Path)
    ap.add_argument('--component-sha256')
    ap.add_argument('--retention-review-utc', required=True)
    opts = ap.parse_args()
    if SOURCE_BLOCKER:
        # Source-audit stop precedes imports, process inspection, lock/root/
        # receipt creation, health probing and every native/tool invocation.
        # Preserved implementation below is review material, not authorization.
        raise SystemExit('SOURCE_BLOCKED / UNRUN: ' + SOURCE_BLOCKER)
    require((opts.mode == 'model') == (opts.arm is not None) and
        (opts.mode == 'model') == (opts.component is not None and opts.component_sha256 is not None), 'component first; model arm plus exact component receipt/SHA')
    require(re.fullmatch('v[1-9][0-9]{0,3}', opts.run_id), 'finite fresh vN ID')
    require(D == A / 'own-vllm-unitrace-current-v1' and A == P / 'bench/results/2026-10-15-pr6-recovery-audit' and B.resolve() == B,
            'canonical source/state roots')
    meta = read_json(D / 'manifest.json', opts.manifest_sha256)
    require(meta['status'] == 'UNRUN' and meta['source_commit'] == COMMIT and
        canonical(meta['collector_contract']) == meta['collector_contract_sha256'], 'sealed UNRUN source contract')
    require(canonical(meta['external_files']) == meta['external_bindings_sha256'] and
        canonical(meta['frozen_bindings']) == meta['frozen_bindings_sha256'], 'all consumed external binding hashes')
    require(set(p.name for p in D.iterdir()) == {'controller.py', 'manifest.json', 'Main-review.ja.md'}, 'exact three-file source package')
    for name, item in meta['package_files'].items():
        pin(D / name, item['sha256'], item['bytes'])
    require(canonical(meta['package_files']) == meta['source_package_content_sha256'], 'entire non-self package hash')
    review = dt.datetime.fromisoformat(opts.retention_review_utc.replace('Z', '+00:00'))
    require(review.tzinfo is not None and review.utcoffset() == dt.timedelta(0) and
        dt.timedelta(0) < review - utc() <= dt.timedelta(hours=48), 'explicit UTC retention review within48h')
    for path, item in meta['external_files'].items():
        pin(path, item['sha256'], item['bytes'])
    M = module('unitrace_pinned_owner', OWNER, meta)
    S = module('unitrace_selected_v2', A / 'own-vllm-selected-quality-v2/compare_screen.py', meta)
    R = module('unitrace_selected_source_checks', A / 'own-vllm-selected-quality-v2/run_screen.py', meta)
    R.require, R.sha = require, sha
    build = read_json(BUILD, BUILD_SHA)
    historical = read_json(A / 'own-vllm-model-smoke-v1.json')
    old = read_json(A / 'own-selector-clean-1-r2-v1.json')
    tool_receipt = read_json(meta['tool_build_receipt']['path'], meta['tool_build_receipt']['sha256'])
    tool = Path(tool_receipt['build_path']) / 'unitrace'
    tool_lib = Path(tool_receipt['build_path']) / 'libunitrace_tool.so'
    model = opts.mode == 'model'
    suffix = ('model-' + opts.arm if model else 'component') + '-' + opts.run_id
    root, target = B / ('main-vllm-unitrace-' + suffix), A / ('own-vllm-unitrace-' + suffix + '.json')
    whole, reserve = (1800 if model else 135), 45
    disk, raw_work = (512 << 20 if model else 64 << 20), (62 << 20 if model else 8 << 20)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not root.exists() and not root.is_symlink() and not target.exists() and not target.is_symlink(), 'fresh private root/A receipt; no overwrite/resume')
        root.mkdir(mode=0o700)
        start = time.monotonic()
        deadline, work_deadline = start + whole, start + whole - reserve
        phase_deadline = work_deadline
        record = dict(schema=SCHEMA, mode=opts.mode, arm=opts.arm, run_id=opts.run_id, status='FAIL',
            passed=False, complete=False, active=False, collected=False, GPU_executed=False, native_launch_attempted=False,
            ownership_closed=False, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            boot_unchanged=False, driver_dump_exists=None, fresh_faults=[], journal_windows=[], commands=[],
            source_commit=COMMIT, source_pins=build['source_pins'], binary_sha256=BINARY_SHA,
            manifest=artifact(D / 'manifest.json'), source_package_content_sha256=meta['source_package_content_sha256'],
            collector_contract_sha256=meta['collector_contract_sha256'], collector_contract=meta['collector_contract'],
            controller=dict(identity=M.identity(os.getpid()), argv=list(sys.orig_argv), source=artifact(__file__),
                            python=artifact(Path(sys.executable).resolve())),
            references=dict(build=artifact(BUILD), tool_build=artifact(meta['tool_build_receipt']['path'])),
            external_bindings=meta['external_files'], cases=[], native_observations=[], pin_checkpoints=[],
            external_bindings_sha256=meta['external_bindings_sha256'], frozen_bindings_sha256=meta['frozen_bindings_sha256'],
            started_utc=M.utc(), retention=dict(owner='Main', consumer='current6354 kernel attribution',
                byte_budget=disk, raw_text_provider_budget=64 << 20 if model else 8 << 20,
                review_deadline_utc=review.isoformat(), cleanup_performed=False,
                next_review='Verify compact individual kernel/API summaries then review raw retirement; no later than explicit UTC deadline'),
            limits=dict(native_wall_seconds=900 if model else 90, native_RSS_bytes=88 << 30 if model else 4 << 30,
                whole_wall_seconds=whole, closure_reserve_seconds=reserve, private_bytes=disk,
                native_text_bytes=raw_work, native_file_bytes=62 << 20 if model else 8 << 20,
                owner_text_bytes=64 << 20, raw_work_text_plus_provider_bytes=raw_work,
                final_text_plus_provider_bytes=64 << 20 if model else 8 << 20,
                closure_owner_separate=True, CPU_seconds=21600 if model else 180),
            interpretation=LIMITATION, performance_qualified=False, quality_qualified=False, quality_accepted=False,
            model_quality_qualified=False, physical_262144_qualified=False, restore_qualified=False,
            dense_NLL_quarantine_preserved=True, original_failed_receipts_unchanged=True)
        owners = []
        def persist():
            record['commands'] = [c for o in owners for c in o.commands]
            record['active'] = any(o.active is not None for o in owners)
            if any(c.get('direct_identity') and c['label'].startswith(('component-', 'model-profile')) for c in record['commands']):
                record['native_launch_attempted'] = True
            (root / 'record.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        owner = M.Owner(root / 'owned', persist)
        closure_owner = M.Owner(root / 'closure', persist)
        owners.extend([owner, closure_owner])
        cursor, final_error, active_observer = None, None, None
        runtime_dynamic = {}
        prior = None
        persist()

        def remaining(cap):
            require(time.monotonic() < phase_deadline, 'finite work/closure deadline')
            return min(cap, phase_deadline - time.monotonic())

        def frozen(label):
            remaining(1)
            require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id'], 'boot changed')
            pin(D / 'manifest.json', opts.manifest_sha256)
            for name, item in meta['package_files'].items():
                pin(D / name, item['sha256'], item['bytes'])
            for path, item in meta['external_files'].items():
                pin(path, item['sha256'], item['bytes'])
            R.source_pin_check(build)
            require(build['source_commit'] == COMMIT and build['binary_sha256'] == BINARY_SHA and
                len(build['products']) == 13 and len(historical['runtime_files']) == 37 and
                historical['source_pins'] == build['source_pins'], 'frozen6354 source/build/product/runtime census')
            bindings = meta['frozen_bindings']
            require(all(bindings[k] == v for k, v in dict(source_pins=build['source_pins'], build_file_pins=build['pins'],
                products=build['products'], runtime_files=historical['runtime_files'], model_identity=historical['model_identity'],
                pack_metadata_sha256=historical['pack_metadata_sha256'], MTP_metadata_sha256=historical['MTP_metadata_sha256']).items()),
                'entire frozen external source/tool/build/DSO/model binding snapshot')
            for item in historical['runtime_files']:
                pin(item['path'], item['sha256'])
            for group in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
                for path, digest in historical[group].items():
                    pin(path, digest)
            for path, item in historical['model_identity'].items():
                path = allowed(path)
                require(path.stat().st_size == item['bytes'] and path.stat().st_mtime_ns == item['mtime_ns'], 'model stat identity, not full shard hash')
            pin(P / 'data/expert-profile.bin', historical['external_expert_profile_sha256'])
            require(tool_receipt['source_revision'] == meta['tool_source_commit'] and tool_receipt['passed'] is True and
                tool_receipt['complete'] is True and tool_receipt['active'] is False, 'official qualified tool build provenance only')
            for name, digest in tool_receipt['binaries'].items():
                pin(Path(tool_receipt['build_path']) / name, digest)
            for path, digest in tool_receipt['XPTI_provider_pins'].items():
                pin(path, digest)
            for path, item in runtime_dynamic.items():
                pin(path, item['sha256'], item['bytes'])
            for value in record['references'].values():
                pin(value['path'], value['sha256'], value['bytes'])
            record['pin_checkpoints'].append(dict(label=label, utc=M.utc(), source=432, build=314, products=13, runtime=37))

        def budget(final=False):
            files = list(root.rglob('*'))
            require(len(files) <= 256 and all(not p.is_symlink() for p in files), 'finite private files without aliases')
            files = [p for p in files if p.is_file()]
            size = sum(p.stat().st_size for p in files)
            raw = sum(p.stat().st_size for p in files if p.suffix in ('.stdout', '.stderr') or (root / 'provider') in p.parents)
            record.update(private_bytes=size, raw_text_provider_bytes=raw)
            require(size <= disk and raw <= ((64 << 20 if model else 8 << 20) if final else raw_work), 'aggregate private/provider/text budget')
            require(all(p.stat().st_size <= (62 << 20 if model else 8 << 20) for p in files), 'per-file budget')

        def utility(label, argv):
            command, out, err = closure_owner.run(label, argv, HOST, P, wall=remaining(12), cpu=60,
                rss_cap=2 << 30, text_cap=256 << 10, total_cap=disk, file_cap=256 << 10)
            require(M.closed(command) and command['exit_code'] == 0, 'normal closure utility: ' + label)
            absence(M, [command])
            return out

        def driver(label):
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            dump = (DEVICE / 'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*')))
            snapshot = dict(label=label, pci_bdf=DEVICE.name, kernel=os.uname().release,
                pci={n: (DEVICE / n).read_text().strip() for n in ('vendor', 'device', 'revision', 'subsystem_vendor', 'subsystem_device')},
                driver=(DEVICE / 'driver').resolve().name, dump_exists=dump)
            record.setdefault('driver_snapshots', []).append(snapshot)
            record.update(boot_unchanged=boot == record['boot_id'], driver_dump_exists=dump)
            require(record['boot_unchanged'] and not dump and snapshot['driver'] == 'xe' and snapshot['pci']['vendor'] == '0x8086', 'sameboot Intel xe/no dump')
            if prior is not None:
                old_driver = prior['driver_snapshots'][0]
                require(snapshot['pci'] == old_driver['pci'] and snapshot['kernel'] == old_driver['kernel'] and
                    snapshot['pci_bdf'] == old_driver['pci_bdf'], 'same GPU/kernel as explicit prior')

        def journal(label, initial=False):
            nonlocal cursor
            driver(label)
            args = ['/usr/bin/journalctl', '-k', '-b', '--no-pager', '-o', 'json', '--show-cursor']
            args += ['-n', '0'] if initial else ['--after-cursor', cursor]
            out = utility(label, args)
            lines = out.read_text().splitlines()
            ends = [s.split(': ', 1)[1] for s in lines if s.startswith('-- cursor: ')]
            rows = [json.loads(s) for s in lines if s and not s.startswith('-- cursor: ') and s != '-- No entries --']
            require(len(ends) == 1 or (not initial and not rows and not ends and cursor), 'explicit fresh kernel cursor')
            require(all(v.get('_BOOT_ID') == record['boot_id'].replace('-', '') for v in rows), 'journal boot')
            faults = [v for v in rows if FAULT.search(str(v.get('MESSAGE', '')))]
            end = ends[0] if ends else cursor
            record['journal_windows'].append(dict(label=label, after_cursor=cursor, end_cursor=end,
                entries=len(rows), faults=faults, artifact=artifact(out), fault_filter=FAULT.pattern))
            cursor = end
            record['fresh_faults'].extend(faults)
            persist()
            require(not record['fresh_faults'], 'faults accumulate permanently; stop at first failure')

        def run_observed(label, argv, env, exe, native_args, provider, wall, is_model):
            nonlocal active_observer
            observer = NativeObserver(M, record, root, exe, env, [str(exe), *native_args], tool_lib, wall)
            active_observer = observer
            record['native_observations'].append(observer.evidence)
            def interaction(io, entry):
                original_poll = io.poll
                def poll():
                    original_poll()
                    remaining(1)
                    budget()
                    observer.poll(entry, model=is_model)
                    record['GPU_executed'] = bool(record['GPU_executed'] or observer.evidence['observed'])
                io.poll = poll
                io.proc.stdin.close()
                with selectors.DefaultSelector() as ready:
                    ready.register(io.proc.stdout, selectors.EVENT_READ)
                    while not io.exit_watch.poll(0) or not io.eof:
                        poll()
                        if ready.select(.02):
                            io.read()
                            io.pending.clear()
                poll()
            command, out, err = owner.run(label, argv, env, P, interaction=interaction, wall=remaining(1800 if is_model else 90),
                cpu=21600 if is_model else 180, rss_cap=88 << 30 if is_model else 4 << 30,
                text_cap=raw_work, total_cap=disk, file_cap=62 << 20 if is_model else 8 << 20)
            case = dict(label=label, command_index=owner.commands.index(command), completed=False,
                stdout=artifact(out), stderr=artifact(err), provider_dir=str(provider))
            record['cases'].append(case)
            case['post_exit_PID_start_absence'] = absence(M, [command])
            require(M.closed(command) and command['exit_code'] == 0, 'normal independent fixture/runner closure: ' + label)
            proof = observer.finish(command, model=is_model)
            runtime_dynamic.update(proof['libraries'])
            pid = proof['identity']['pid']
            native_argv = [str(exe), *native_args]
            case['collector'] = provider_records(provider, pid, cli(tool, provider, exe, native_args), native_argv)
            case['native'] = proof
            require(not ERROR.search(err.read_text()), 'native/runner/provider failure text')
            case['completed'] = True
            active_observer = None
            return case, out, err

        def expire(signum, frame):
            raise TimeoutError('whole diagnostic deadline including45s closure reserve')
        previous_alarm = signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, whole - reserve)
        try:
            frozen('preflight')
            prior, proof = prior_gate(opts.prior, opts.prior_sha256, build, M, S, work_deadline)
            bound_artifacts(prior, runtime_dynamic)
            record['prior_gate'] = proof
            record['references']['Main_latest_prior'] = artifact(opts.prior)
            require(prior['driver_snapshots'], 'prior exact GPU/driver snapshot required')
            if model:
                component = read_json(opts.component, opts.component_sha256)
                record['references']['component'] = artifact(opts.component)
                record['component_original_absence'] = closed_gate(component, M, record['boot_id'])
                require(component.get('schema') == SCHEMA and component.get('mode') == 'component' and
                    component.get('collected') is True and component.get('GPU_executed') is True and
                    component.get('source_pins') == build['source_pins'] and component.get('binary_sha256') == BINARY_SHA and
                    component.get('collector_contract_sha256') == meta['collector_contract_sha256'] and
                    component.get('manifest', {}).get('sha256') == opts.manifest_sha256 and
                    [c['label'] for c in component['cases']] == ['component-prompt', 'component-vllm'] and
                    all(c['completed'] and c['oracle_passed'] and c['collector']['actual_compute_name_count'] > 0 and
                        c['collector']['host']['calls'] > 0 for c in component['cases']), 'exact current source/tool two-component PASS required')
                require(component['driver_snapshots'][0]['pci'] == prior['driver_snapshots'][0]['pci'], 'component same GPU')
                bound_artifacts(component, runtime_dynamic)
            R.no_model_owners(list(historical['model_identity']))
            record['initial_owner_scan'] = dict(passed=True, scope='bounded same-UID scan; Main excludes other workloads/build/cleanup')
            journal('journal-start', initial=True)
            # Frozen current clean coarse recipe, explicit allowlist. No inherited
            # host/Ruby/lint/PYTHONPATH, observer or profiler environment survives.
            allowed_host = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'MKLROOT', 'OMP_NUM_THREADS',
                'CPLUS_INCLUDE_PATH', 'ONEAPI_DEVICE_SELECTOR', 'UR_ADAPTERS_FORCE_LOAD', 'SYCL_CACHE_PERSISTENT',
                'NEOReadDebugKeys', 'EnableDirectSubmission', 'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD',
                'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED')
            env = {k: v for k, v in old['environment'].items() if (k.startswith('STRATA_') and not k.startswith('STRATA_MEASUREMENT_')) or k in allowed_host}
            require(not any(k.startswith(('STRATA_PREFILL_NLL', 'STRATA_PREFILL_LOGITS', 'UNITRACE_', 'PTI_')) for k in env), 'no inherited capture/profiler')
            require(all(env.get(k) == v for k, v in SAFE.items() if k != 'STRATA_ROPE_TABLE'), 'safe current frozen runtime/bank recipe')
            env.update(SAFE, **VALIDATION)
            # This exact SYCL executable compiles src/program/generate.cpp.
            # Its unconditional main-entry default would otherwise call setenv
            # after the preloaded collector constructor initialized a provider.
            # Supply the same value before exec; do not mutate product source.
            env.update(XDG_CACHE_HOME=str(root / 'cache'), STRATA_TRACE='1', CUDA_MODULE_LOADING='EAGER')
            record['explicit_launcher_environment_additions'] = dict(CUDA_MODULE_LOADING='EAGER',
                reason='Pinned generate.cpp1451 default is supplied before exec to avoid native setenv after collector initialization; no CUDA execution/implementation is involved')
            if not model:
                for name, args, label in [('sycl_prompt_native16_test', ['--smoke'], 'component-prompt'),
                    ('sycl_vllm_native16_test', ['--device', '0', '--suite', 'smoke'], 'component-vllm')]:
                    journal('journal-before-' + label)
                    frozen('before-' + label)
                    exe = Path(build['products'][name]['path'])
                    reports = root / 'provider' / label
                    reports.mkdir(parents=True, mode=0o700)
                    case, out, err = run_observed(label, cli(tool, reports, exe, args), env, exe, args, reports, 90, False)
                    text = out.read_text()
                    if name == 'sycl_prompt_native16_test':
                        rows = [s for s in text.splitlines() if s.startswith('case dtype=bf16 ')]
                        require(len(rows) == 2 and all('oracle=FP64_full ' in s and
                            'finite=PASS guards=PASS inputs=PASS' in s for s in rows) and
                            text.count('fixture=PASS model_quality=UNRUN whole_prefill=UNRUN physical262144=UNRUN faults=EXTERNAL_REQUIRED') == 1,
                            'unchanged independent FP64/full smoke oracle + guard/input completion')
                        case['oracle_lines'] = rows
                    else:
                        rows = [s for s in text.splitlines() if s.startswith('dyadic PASS ')]
                        require(len(rows) == 32 and all(' n=128 k=64 ld=136 ' in s for s in rows) and
                            text.count('COMPONENT_SUITE_PASS only; DPAS_BINARY_PROOF/model_quality/full262144/performance UNRUN') == 1,
                            'all32 unchanged dyadic exact-RNE compact/wide AOT smoke cases')
                        case['oracle_lines'] = rows
                    case['oracle_passed'] = True
                    journal('journal-after-' + label)
                    frozen('after-' + label)
                    budget()
            else:
                env.update(STRATA_SYCL_PREFILL_NATIVE16='0' if opts.arm == 'fp32' else 'bf16',
                    STRATA_PREFILL_LAYER_MAJOR_R_DTYPE='fp32' if opts.arm == 'fp32' else 'bf16',
                    STRATA_SYCL_PREFILL_NATIVE16_BACKEND='own' if opts.arm == 'fp32' else 'vllm',
                    STRATA_SYCL_PREFILL_NATIVE16_QSA='1' if opts.arm == 'qsa' else '0',
                    STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto', STRATA_PREFILL_LAYER_MAJOR_R_GPU='32768',
                    STRATA_PREFILL_LAYER_MAJOR_PROFILE='1', STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='1',
                    STRATA_PREFILL_CPU_WAIT_PROFILE='1', STRATA_MEASUREMENT_ENGINE_START=str(root / 'engine-start.json'),
                    STRATA_MEASUREMENT_ENGINE_EXIT=str(root / 'engine-exit.json'))
                reports = root / 'provider'
                reports.mkdir(mode=0o700)
                wrapper = make_wrapper(build, tool, reports, root, env)
                args = list(old['config']['args'])
                for flag, value in [('--max-context', '65536'), ('--kv-resident', '65536')]:
                    require(args.count(flag) == 1, 'one model CLI flag')
                    args[args.index(flag) + 1] = value
                external_profile = Path(args[args.index('--expert-profile') + 1])
                local = root / 'expert-profile.bin'
                shutil.copyfile(external_profile, local)
                args[args.index('--expert-profile') + 1] = str(local)
                for flag, expected in [('--prefill', '8192'), ('--expert-cache', '128'), ('--pool-workers', '5'), ('--spec', '4')]:
                    require(args.count(flag) == 1 and args[args.index(flag) + 1] == expected, 'exact model recipe ' + flag)
                require('--expert-cache-per-layer' in args and '--no-token-graph' in args and '--no-coupled-draft' in args, 'per-layer cache/MTP/current token path')
                config = dict(old['config'], exe=str(wrapper), args=args, env={}, log=str(root / 'owned/engine.stderr'))
                options = ['--prefill-only', '--prefill-tokens', '32768', '--workload-id', 'prefill_v1_20261015']
                port = 18540 + ('fp32', 'bf16', 'qsa').index(opts.arm)
                plan = dict(port=port, url=f'http://127.0.0.1:{port}', runs=1, benchmark_options=options, bench_timeout_seconds=900)
                for name, value in [('config.json', config), ('plan.json', plan)]:
                    with (root / name).open('x') as stream:
                        json.dump(value, stream, indent=2); stream.write('\n')
                record.update(environment=env, config=config, benchmark_options=options, wrapper=artifact(wrapper),
                    config_artifact=artifact(root / 'config.json'), plan_artifact=artifact(root / 'plan.json'),
                    external_profile=artifact(external_profile), local_profile_initial=artifact(local))
                for item in (record['wrapper'], record['config_artifact'], record['plan_artifact']):
                    runtime_dynamic[item['path']] = item
                journal('journal-before-model')
                frozen('before-model')
                case, out, err = run_observed('model-profile', [PY, str(RUNNER), str(root), 'extended-bench'],
                    env, build['binary'], ['--serve', *args], reports, 900, True)
                result = read_json(root / 'result.json')
                require(result['passed'] is True and result['server_exit_code'] == result['real_engine_exit']['exit_code'] == 0 and
                    result.get('bench_exit_code') == 0 and not result.get('shutdown_error'), 'actual client/server/wrapper/native normal QUIT path')
                require(result['real_engine_exit'] == case['native']['wrapper_exit'], 'runner/wrapper native receipt binding')
                suites = list((root / 'stock-results').glob('*/results.json'))
                require(len(suites) == 1, 'one fresh official bench suite')
                suite = read_json(suites[0])
                cohort = suite['results']['prefill_only']
                require(suite['complete'] is True and not suite['results']['failures'] and cohort['complete'] is True and
                    cohort['eligible_runs'] == len(cohort['requests']) == 1 and suite['params']['think'] is True and
                    suite['params']['mode'] == 'prefill-only' and suite['params']['prefill_tokens'] == 32768 and
                    suite['params']['workload']['workload_id'] == 'prefill_v1_20261015', 'one exact thinkingON officialbench request')
                require(all(r['ok'] and r['eligible'] and r['stream_complete'] and r['actual_prompt_tokens'] == r['actual_read_tokens'] == 39262 and
                    r['actual_reused_tokens'] == 0 and r['generated_tokens'] == 24 and r['thinking_requested'] and
                    not r.get('eligibility_errors') for r in cohort['requests']), 'actual39262 read/zero reuse/24 generated, no backoff')
                journal_check = json.loads(utility('verify-request-journal', ['/usr/bin/python3', '-B', '-c',
                    'import sys,json;sys.path.insert(0,"tools");from strata_bench import verify_request_journal;print(json.dumps(verify_request_journal(sys.argv[1])))',
                    str(suites[0].parent)]).read_text())
                require(journal_check['valid'] is True and journal_check['complete'] is True and not journal_check['pending'], 'strict request journal identity/completion')
                native_path = root / 'owned/engine.stderr'
                native = native_path.read_text()
                require(not ERROR.search(native), 'native error or nonfinite diagnostic')
                dtype = 'fp32' if opts.arm == 'fp32' else 'bf16'
                placement = [s for s in native.splitlines() if s.startswith('strata prefill: layer-major candidate,')]
                require(len(placement) == 1 and f'tokens=39261 chunk=8192 residual_dtype={dtype} ' in placement[0] and
                    'residual_gpu_rows=32768 ' in placement[0], 'actual39261 batched/chunk8192/GPU32K placement')
                require(('strata prefill: native16 BF16 HC residual/projection IO' in native) == (opts.arm != 'fp32') and
                    '[native16-route] engine=' + ('0 backend=own' if opts.arm == 'fp32' else 'bf16 backend=vllm') in native,
                    'actual native16/vLLM arm dispatch')
                completed = [tuple(map(int, row)) for row in re.findall(r'layer-major layer (\d+) of (\d+): completed (\d+) chunks', native)]
                require(completed == [(i, 48, 5) for i in range(48)], 'all48 layers/all5 chunks, no backoff')
                record.update(result=result, benchmark_result=suite, request_journal_verification=journal_check,
                    placement_lines=placement, native_log=artifact(native_path),
                    coarse_profile_lines=[s for s in native.splitlines() if s.startswith('strata lm profile:')],
                    qsa_profile_lines=[s for s in native.splitlines() if 'qsa profile:' in s],
                    CPU_wait_lines=[s for s in native.splitlines() if s.startswith('strata prefill cpu wait:')],
                    CPU_host_lines=[s for s in native.splitlines() if s.startswith('strata prefill cpu host:')])
                require(record['coarse_profile_lines'] and len(record['CPU_wait_lines']) == len(record['CPU_host_lines']) == 1 and
                    'available=1 complete=1' in record['CPU_wait_lines'][0] and 'available=1 open_regions=0' in record['CPU_host_lines'][0],
                    'complete coarse and issuing-thread host/wait diagnostics')
                record['CPU_interpretation'] = 'Issuing-thread CLOCK_THREAD_CPUTIME_ID; explicit waits separate from host work. Polling CPU does not prove a host-computation bottleneck; other threads remain unclassified.'
                journal('journal-after-model')
                frozen('after-model')
                for item in (record['wrapper'], record['config_artifact'], record['plan_artifact'], record['external_profile']):
                    pin(item['path'], item['sha256'], item['bytes'])
                record['local_profile_final'] = artifact(local)
            record.update(collected=True, passed=True, complete=True, status='PASS')
        except BaseException as error:
            final_error = error
            record['error'] = repr(error)
        finally:
            phase_deadline = deadline
            signal.setitimer(signal.ITIMER_REAL, max(.01, deadline - time.monotonic()))
            def step(label, operation):
                try:
                    remaining(1)
                    operation()
                except BaseException as error:
                    record.setdefault('closure_errors', []).append(dict(step=label, error=repr(error)))
                    record.update(passed=False, complete=False, status='FAIL')
            def final_journal():
                require(cursor is not None and all(o.active is None for o in owners), 'final journal cursor/ownership unavailable')
                journal('journal-final')
            def final_absence():
                record['final_PID_start_absence'] = absence(M, [c for o in owners for c in o.commands])
            def partial_provider():
                if active_observer is not None:
                    runtime_dynamic.update(active_observer.hashes)
                files = [p for p in (root / 'provider').rglob('*') if p.is_file()]
                require(len(files) <= 32, 'finite provider partial records')
                record['provider_file_inventory_final'] = [artifact(p) for p in sorted(files)]
            def normal():
                require(all(o.active is None for o in owners) and bool(record['commands']) and
                    all(M.closed(c) and c['exit_code'] == 0 for c in record['commands']) and
                    record['fresh_faults'] == [] and record['boot_unchanged'] is True and record['driver_dump_exists'] is False,
                    'all commands normally closed plus fresh final GPU/boot/dump proof')
            try:
                for label, action in [('fresh-final-journal', final_journal), ('driver-final', lambda: driver('final')),
                    ('partial-provider-inventory', partial_provider), ('frozen-pins-final', lambda: frozen('final')),
                    ('all-PID-start-absence', final_absence),
                    ('no-model-owners', lambda: R.no_model_owners(list(historical['model_identity']))),
                    ('aggregate-budgets', lambda: budget(True)), ('normal-closure', normal)]:
                    step(label, action)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
                record.update(active=any(o.active is not None for o in owners), finished_utc=M.utc(),
                    elapsed_seconds=time.monotonic() - start,
                    ownership_closed=bool(record['commands']) and all(o.active is None for o in owners) and all(M.closed(c) for c in record['commands']))
                if final_error is not None or record.get('closure_errors') or record['active'] or not record['ownership_closed'] or record['elapsed_seconds'] >= whole:
                    record.update(passed=False, complete=False, status='FAIL')
                record['closure_validated'] = not record.get('closure_errors') and record['ownership_closed']
                record['consumed_dynamic_artifact_pins'] = runtime_dynamic
                record['consumed_dynamic_artifact_pins_sha256'] = canonical(runtime_dynamic)
                persist()
                with target.open('x') as stream:
                    json.dump(record, stream, indent=2, allow_nan=False); stream.write('\n')
        print(json.dumps(dict(passed=record['passed'], complete=record['complete'], receipt=str(target),
            sha256=sha(target), diagnostic_only=True)), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
