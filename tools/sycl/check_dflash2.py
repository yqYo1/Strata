#!/usr/bin/env python3
"""One bounded traced DFlash2 kernel gate; Main only, no model or reset.

Verify committed source, executable and qualified runtime before launch. Inspect
the actual child's executable mappings at its post-compute retirement gate.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import build_upstream_features as build
import build_engine as source
import measurement_guards as guards
import measurement_owner as ownership

P, B, STATUS = build.P, build.B, build.STATUS
ROOT = B / 'dflash2-kernel-contract-v2'
PREVIOUS = B / 'dflash2-kernel-contract-v1'
REFERENCE = B / 'decode-iq3-audit-speed-v2'
INPUTS = ['tools/sycl/check_dflash2.py', 'tools/sycl/build_upstream_features.py',
          'tools/sycl/build_engine.py', 'tools/sycl/measurement_guards.py', 'tools/sycl/measurement_owner.py']
ENV_KEYS = ['PATH', 'HOME', 'LANG', 'LC_ALL', 'PYTHONDONTWRITEBYTECODE', 'LD_LIBRARY_PATH', 'MKLROOT',
            'ONEAPI_DEVICE_SELECTOR', 'UR_ADAPTERS_FORCE_LOAD', 'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD',
            'NEOReadDebugKeys', 'EnableDirectSubmission', 'STRATA_SYCL_VERIFY_NATIVE_CAPTURE',
            'STRATA_SYCL_PREFILL_SAME_QUEUE', 'SYCL_CACHE_PERSISTENT', 'ZES_ENABLE_SYSMAN', 'ZE_ENABLE_ALT_DRIVERS']
require, write = guards.require, build.write


def public():
    return json.loads(source.git(P, 'show', 'HEAD:' + str(STATUS.relative_to(P))))['current_sync']['dflash2_device_check']


def compact(row):
    out = {k: v for k, v in row.items() if k not in ['commands', 'runtime_files', 'controller', 'native_build']}
    out['controller_source'] = dict(commit=row['controller']['revision'],
        aggregate_sha256=row['controller']['aggregate_sha256'], files=row['controller']['count'])
    native = row['native_build']
    out['native_build'] = {k: native[k] for k in ['source_commit', 'project_source_sha256', 'record', 'passed', 'ownership_closed']}
    out['commands'] = [dict(label=c['label'], exit_code=c.get('exit_code'), error=c.get('error'), ownership_closed=ownership.closed(c),
                            errors=c.get('errors', []), cleanup=c.get('cleanup', [])) for c in row.get('commands', [])]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    status = json.loads(STATUS.read_text())
    status['current_sync']['dflash2_device_check'] = compact(row)
    write(STATUS, status)


def verify(row):
    require(public() == compact(json.loads((ROOT / 'record.json').read_text())), 'commit exact device plan/result')
    require(source.digest(row['previous_gate']['path']) == row['previous_gate']['sha256'], 'original failed gate changed')
    for pin in [row['controller']]:
        require(build.signature(pin['source_sha256']) == pin['aggregate_sha256'], 'controller aggregate changed')
        require({n: source.digest(P / n) for n in pin['source_sha256']} == pin['source_sha256'], 'controller changed')
        require({n: hashlib.sha256(v).hexdigest() for n, v in
                 source.blobs(P, pin['revision'], pin['source_sha256'])} == pin['source_sha256'], 'controller not recoverable')
    native = json.loads((build.ROOT / 'record.json').read_text())
    require(build.compact(native) == row['native_build'] and native['passed'] and native['ownership_closed'], 'native build changed')
    build.check_inputs(native)
    for value in native['binaries'].values():
        require(source.digest(value['path']) == value['sha256'], 'native binary changed')
    require(source.digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
            json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'], 'runtime map changed')
    for path, value in row['runtime_files'].items():
        require(source.digest(path) == value, 'runtime ELF changed: ' + path)
    require(guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')
    guards.no_model_owners([])


def prepare():
    require(not ROOT.exists(), 'one fresh finite device gate; no automatic retry')
    previous = json.loads((PREVIOUS / 'record.json').read_text())
    prior_public = public()
    require(previous['stage'] == prior_public['stage'] == 'failed' and not previous['passed'] and
            prior_public['process_retirement']['actual_owned_processes_gone'] and
            all(c['direct_child_reaped'] and c['session_empty'] and not c.get('survivors')
                for c in previous['commands']), 'commit failed gate and establish actual process retirement first')
    require(not json.loads((PREVIOUS / 'active.json').read_text())['active'], 'previous owner still active')
    native = json.loads((build.ROOT / 'record.json').read_text())
    require(native['passed'] and native['stage'] == 'built' and native['ownership_closed'] and
            build.committed_plan() == build.compact(native), 'commit completed feature build first')
    build.check_inputs(native)
    reference = json.loads((REFERENCE / 'record.json').read_text())
    environment = {k: reference['environment_template'][k] for k in ENV_KEYS}
    environment.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
        UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
        UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
    runtime_manifest = dict(path=str(REFERENCE / 'runtime-inputs.json'), sha256=source.digest(REFERENCE / 'runtime-inputs.json'))
    row = dict(stage='prepared', controller=source.pin(P, INPUTS), native_build=build.compact(native),
        runtime_manifest=runtime_manifest, runtime_files=json.loads(Path(runtime_manifest['path']).read_text()),
        environment=environment, driver=guards.driver_guard(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        environment_origin=dict(path=str(REFERENCE / 'record.json'), sha256=source.digest(REFERENCE / 'record.json')),
        budgets=dict(wall=180, cpu=180, text_cap=64 << 20, file_cap=64 << 20, total_cap=96 << 20, rss_cap=8 << 30),
        cases=9, fixture=native['binaries']['sycl_dflash2_test'], commands=[], passed=False, ownership_closed=True,
        meaning='Eight active-column GEMMs with3matrices,exact signed integer oracle and output guards;FP32 mean/pending-GR taps. No model,BF16 tap-route,ring recovery,task quality,performance or physical full-context qualification.')
    # libsycl may statically include UR. Requiring a separate libur_loader mapping
    # rejects that supported linkage. Check all actual executable files plus the
    # exact SYCL, forced adapter and GPU driver providers instead. Keep the UR
    # tracing requirement; do not load another loader just to satisfy an observer.
    row['required_executable_mappings'] = [str(Path(p).resolve()) for p in [
        '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0',
        environment['UR_ADAPTERS_FORCE_LOAD'], environment['ZE_ENABLE_ALT_DRIVERS']]]
    require(set(row['required_executable_mappings']) <= {str(Path(p).resolve()) for p in row['runtime_files']},
            'required provider is not in the qualified runtime pins')
    row['previous_gate'] = dict(path=str(PREVIOUS / 'record.json'), sha256=source.digest(PREVIOUS / 'record.json'),
        evidence_commit=source.git(P, 'rev-parse', 'HEAD').decode().strip(), stage='failed',
        reason='Prior separate-UR-loader presence predicate failed after9numerical markers. Preserve failure;new prospective provider gate admits pinned static UR within libsycl. Original map count/set was not recorded,so the exact absent predicate is unknown.',
        source='https://github.com/intel/llvm/blob/sycl/sycl/source/CMakeLists.txt')
    require(row['driver'] == reference['driver'] and row['boot'] == reference['boot'], 'requalify changed boot/driver before reuse')
    require(environment['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and environment['EnableDirectSubmission'] == '0' and
            environment['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0' and environment['ZES_ENABLE_SYSMAN'] == '0', 'qualified safety settings')
    ROOT.mkdir(mode=0o700)
    for name in ['tmp', 'cache']: (ROOT / name).mkdir(mode=0o700)
    save(row)
    print('Prepared nine-case traced device check; commit before run.', flush=True)


def kernel_log(*args):
    proc = subprocess.run(['/usr/bin/journalctl', '-k', '-b', *args, '--no-pager'], capture_output=True, text=True, timeout=15)
    require(proc.returncode == 0 and len(proc.stdout) < 1 << 20 and
            'permission' not in proc.stderr.lower(), 'kernel journal unavailable or oversized')
    return proc.stdout


def run():
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared', 'no duplicate or automatic retry')
    verify(row)
    before = kernel_log('-n', '0', '--show-cursor')
    cursor = next(line.split(': ', 1)[1] for line in before.splitlines() if line.startswith('-- cursor: '))
    monitor = None
    def persist():
        write(ROOT / 'active.json', dict(active=bool(monitor and monitor.active), commands=monitor.commands if monitor else []))
    try:
        monitor = ownership.Owner(ROOT / 'owned', persist)
        def inspect(io, entry):
            lines = []
            while True:
                line = io.line(seconds=170); lines.append(line)
                require(len(lines) <= 24, 'unexpected fixture stdout')
                if line == 'DF2_WAIT': break
            require('dflash2 fixture cases=9 PASS' in lines and
                    all(f'dflash2 gemm columns={n} matrices=3 PASS' in lines for n in range(1,9)) and
                    'dflash2 taps rows=3 fp32,pending_GR,interleaved_guards PASS' in lines, 'fixture result incomplete')
            actual = set()
            for line in Path('/proc', str(entry['pid']), 'maps').read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and 'x' in fields[1] and fields[5].startswith('/'):
                    require(not fields[5].endswith(' (deleted)'), 'mapped executable removed')
                    actual.add(str(Path(fields[5]).resolve()))
            allowed = {str(Path(p).resolve()): h for p,h in row['runtime_files'].items()}
            allowed[str(Path(row['fixture']['path']).resolve())] = row['fixture']['sha256']
            unknown = sorted(actual - allowed.keys())
            row['unqualified_executable_mappings'] = unknown
            row['actual_executable_mappings'] = {p: source.digest(p) for p in sorted(actual)}
            row['missing_executable_providers'] = sorted(set(row['required_executable_mappings']) - actual)
            require(not unknown and len(actual) >= 8 and not row['missing_executable_providers'],
                    'unexpected/incomplete executable mappings: ' + repr(dict(
                        unknown=unknown, count=len(actual), missing=row['missing_executable_providers'])))
            require(all(source.digest(p) == allowed[p] for p in actual), 'actual mapped ELF changed')
            row['verified_executable_mappings'] = len(actual)
            io.send(b'QUIT\n'); io.exit(20)
        entry, stdout, stderr = monitor.run('dflash2', [row['fixture']['path'], '--supervised'],
            row['environment'], P, interaction=inspect, **row['budgets'])
        row['commands'] = monitor.commands
        text = stderr.read_text(errors='replace')
        require(ownership.closed(entry) and entry['exit_code'] == 0,
                'device process did not close normally: ' + str(entry.get('error')))
        require('urEnqueueKernelLaunch' in text and not re.search(r'UR_RESULT_ERROR_[A-Z_]+', text),
                'missing tracing or UR error')
        verify(row)
        row.update(stage='passed', passed=True, numerical_cases_passed=9)
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
    finally:
        row['commands'] = monitor.commands if monitor else []
        row['ownership_closed'] = not monitor or monitor.active is None and all(ownership.closed(c) for c in monitor.commands)
        try:
            after = kernel_log('--after-cursor', cursor, '-n', '2048')
            require(len(after.splitlines()) < 2048, 'incomplete kernel window')
            row['fresh_faults'] = [line for line in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', line)]
            row['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot']
            row['fault_probe_successful'] = True
            require(guards.driver_guard() == row['driver'] and not row['fresh_faults'] and row['boot_unchanged'], 'fresh device fault or boot change')
        except BaseException as error:
            row.update(passed=False, stage='failed', fault_error=repr(error))
        if not row['ownership_closed']: row.update(passed=False, stage='failed')
        save(row)
    require(row['passed'] and row['ownership_closed'], row.get('error', row.get('fault_error', 'device gate failed')))
    print('DFlash2 small kernel gate passed9cases with normal exit,qualified executable maps and no fresh fault.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'run'], 'use prepare or run')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (prepare if sys.argv[1] == 'prepare' else run)()
