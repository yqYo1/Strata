#!/usr/bin/env python3
"""Refresh production-compatible copy service under the qualified SYCL runtime.

Repository LICENSE. One tiny standalone tool, no new engine generation. Own all
builds and copies; prepare -> commit -> build -> commit -> run. No automatic retry.
"""
import ast
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import build_fused_profile as closure
import measurement_owner as supervision
import measurement_guards as guards

r, P, B = closure.r, closure.P, closure.B
ROOT = B / 'transfer-refresh-39758-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'transfer_refresh'
NATIVE = ['src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
          'tools/sycl/host_transfer_probe.cpp']
FLAGS = ['-O3', '-DNDEBUG', '-std=c++17', '-fsycl', '-fsycl-id-queries-range=size_t',
         '-DSTRATA_ENABLE_SYCL=1', '-pthread', '-I', str(P / 'include'),
         '-I', str(P / 'include/strata/sycl/compat')]
BUILD_LIMITS = dict(wall=240, cpu=600, rss_cap=4 << 30, text_cap=8 << 20,
                    file_cap=128 << 20, total_cap=256 << 20)
RUN_LIMITS = dict(wall=90, cpu=90, rss_cap=3 << 30, text_cap=8 << 20,
                  file_cap=16 << 20, total_cap=64 << 20)
require, digest, write = guards.require, supervision.sha, closure.write


def git(*args):
    return r.build.git(P, *args)


def pins(revision):
    names = set(NATIVE) | set(git('ls-files', 'include').decode().splitlines())
    pending, seen = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path); names.add(str(path.relative_to(P)))
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [n.name for n in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                candidate = path.parent / ((module or '').split('.')[0] + '.py')
                if candidate.is_file():
                    pending.append(candidate)
    names.add('tools/sycl/measurement_environment.json')
    result = {}
    for name, raw in r.build.blobs(P, revision, sorted(names)):
        require(raw == (P / name).read_bytes(), 'commit transfer build input: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def public():
    data = json.loads(RESULT.read_text())
    committed = json.loads(git('show', 'HEAD:' + str(RESULT.relative_to(P))))
    require(data.get(KEY) == committed.get(KEY), 'commit exact transfer plan/result')
    return data


def commands(compiler):
    rows = [[compiler, *FLAGS, '-MD', '-MF', str(ROOT / (str(i) + '.d')),
             '-c', str(P / source), '-o', str(ROOT / (str(i) + '.o'))] for i, source in enumerate(NATIVE)]
    rows.append([compiler, '-fsycl', '-fsycl-id-queries-range=size_t', '-pthread',
                 *[str(ROOT / (str(i) + '.o')) for i in range(len(NATIVE))], '-o', str(ROOT / 'host_transfer_probe')])
    return rows


def runtime():
    rt = r.validation.committed(r.RESULT)
    require(rt['passed'] and rt['ownership_closed'] and not rt['active'] and not rt['fresh_faults'], 'qualified runtime required')
    full = r.current(rt)
    return rt, full


def verify(row):
    rt, full = runtime()
    require(digest(r.RESULT) == row['runtime_sha256'] and full['binary'] == row['selected_binary'], 'runtime/selected engine changed')
    require(guards.driver_guard() == row['driver'] and rt['boot'] == row['boot'], 'transfer boot/driver')
    require(closure.sdk_inputs() == row['sdk'], 'transfer compiler SDK changed')
    require(digest(row['inputs']['path']) == row['inputs']['sha256'], 'transfer input manifest changed')
    inputs = json.loads(Path(row['inputs']['path']).read_text())
    require(inputs['project'] == pins(row['source_commit']), 'transfer source closure changed')
    for name, sha in inputs['external'].items():
        require(digest(name) == sha, 'transfer external input changed: ' + name)
    require(commands(full['compiler']['path']) == row['commands'], 'transfer build command changed')
    return inputs, rt


def prepare(data):
    require(KEY not in data and not ROOT.exists(), 'fresh transfer plan only')
    rt, full = runtime()
    revision = git('rev-parse', 'HEAD').decode().strip(); project = pins(revision)
    sdk = closure.sdk_inputs(); external = closure.external_inputs(rt)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    write(ROOT / 'inputs.json', dict(project=project, external=external))
    build_env = dict(full['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    native_env = dict(rt['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
                      STRATA_TRANSFER_PROBE_GATES='1')
    native_env.pop('UR_LOG_TRACING', None)
    native_env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
    cases = [dict(host=host, size=2 << 20, repeats=1, diagnostic=True) for host in ('ordinary', 'usm')]
    cases += [dict(host=host, size=size, repeats=3, diagnostic=False) for size in (2 << 20, 320 << 20)
              for host in ('ordinary', 'usm', 'usm', 'ordinary', 'ordinary', 'usm')]
    row = dict(stage='prepared', source_commit=revision, inputs=dict(path=str(ROOT / 'inputs.json'),
               sha256=digest(ROOT / 'inputs.json'), project_files=len(project), external_files=len(external)),
               sdk=sdk, runtime_sha256=digest(r.RESULT), selected_binary=full['binary'],
               commands=commands(full['compiler']['path']), build_environment=build_env,
               native_environment=native_env, runtime_files=rt['runtime_files'],
               driver=guards.driver_guard(), boot=rt['boot'], cases=cases,
               build_limits=BUILD_LIMITS, run_limits=RUN_LIMITS,
               criteria='Two2MiB traced/validated cases before12 clean cases;each case fresh owned process. Exact B570,preinit/provider/executable/env gates,full payload/source/guards all repetitions,both directions,normal allocation retirement/exit,no fresh fault. Separate first copy and3 measured copies. Stop at first failed case.',
               scope='Current-runtime production Runtime/compat copy service and caller CPU/wall split. TwoMiB expert-sized and320MiB bulk transfers;3processes per allocation/size. No model speed,physicalPCIe byte counter,global hardware saturation,overlap or arithmetic ceiling claim.',
               provenance='Existing independent Main transfer tool with owned gates;unchanged production runtime/compat/profile implementation. SDK/compiler only,not XeStrata source.',
               retention='One tiny helper for imminent resource budgeting. Retire objects/normal logs after source recovery and compact individual samples;engine fallback/candidate untouched.',
               active=False, ownership_closed=True, passed=False)
    data[KEY] = row; write(RESULT, data)
    print('Transfer source/recipe prepared;commit before build.', flush=True)


def build(data):
    row = data[KEY]; require(row['stage'] == 'prepared' and not (ROOT / 'build').exists(), 'one committed transfer build')
    inputs, _ = verify(row)
    supervisor = None
    supervisor = supervision.Owner(ROOT / 'build', lambda: write(ROOT / 'build-owner.json', supervisor.commands))
    try:
        for i, command in enumerate(row['commands']):
            entry, out, err = supervisor.run('compile-' + str(i), command, row['build_environment'], P, **BUILD_LIMITS)
            require(supervision.closed(entry) and entry['exit_code'] == 0, 'transfer compiler/link failed')
        allowed = {str((P / name).resolve()): sha for name, sha in inputs['project'].items()}
        allowed.update(inputs['external'])
        allowed.update({str(Path(x['path']).resolve()): closure.SYSTEM_HEADER_UPDATES.get(x['path'], (None, x['sha256']))[1]
                        for x in json.loads(closure.lineage.SDK.read_text())})
        observed = {}
        for i in range(len(NATIVE)):
            body = (ROOT / (str(i) + '.d')).read_text().replace('\\\n', ' ').split(':', 1)[1]
            require('\\' not in body, 'unexpected transfer dependency escaping')
            for path in body.split():
                path = str(Path(path).resolve()); sha = digest(path)
                require(path in allowed and sha == allowed[path], 'unfrozen transfer compiler input: ' + path)
                observed[path] = sha
        require(all(str(P / name) in observed for name in NATIVE), 'incomplete transfer compiled source coverage')
        write(ROOT / 'compiled-inputs.json', observed)
        verify(row)
        binary = ROOT / 'host_transfer_probe'
        row.update(stage='built', build_passed=True, binary=dict(path=str(binary), sha256=digest(binary), bytes=binary.stat().st_size),
                   compiled_inputs=dict(path=str(ROOT / 'compiled-inputs.json'), sha256=digest(ROOT / 'compiled-inputs.json'),
                                        files=len(observed), uncovered=0))
    except BaseException as error:
        row.update(stage='build-failed', build_passed=False, error=repr(error))
    finally:
        row.update(active=supervisor.active is not None,
                   ownership_closed=supervisor.active is None and all(supervision.closed(c) for c in supervisor.commands))
        write(RESULT, data)
    print(json.dumps({k: row.get(k) for k in ('stage', 'build_passed', 'ownership_closed', 'error')}), flush=True)


def run(data):
    row = data[KEY]
    require(row['stage'] == 'built' and row['build_passed'] and row['ownership_closed'], 'commit successful transfer build')
    _, rt = verify(row)
    require(digest(row['binary']['path']) == row['binary']['sha256'] and not (ROOT / 'native').exists(), 'fresh verified transfer run')
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(line.split(': ', 1)[1] for line in cursor_text.splitlines() if line.startswith('-- cursor: '))
    supervisor = None
    supervisor = supervision.Owner(ROOT / 'native', lambda: write(ROOT / 'native-owner.json', supervisor.commands))
    cases = row['measurements'] = []
    try:
        for i, case in enumerate(row['cases']):
            env = dict(row['native_environment'])
            warnings = ROOT / 'native' / ('case-' + str(i) + '-validation.log')
            trace = ROOT / 'native' / ('case-' + str(i) + '-trace.log')
            env['UR_LOG_VALIDATION'] = 'level:warning;flush:warning;output:file,' + str(warnings)
            if case['diagnostic']:
                env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                           UR_LOG_TRACING='level:info;flush:info;output:file,' + str(trace))
            argv = [row['binary']['path'], '--host', case['host'], '--bytes', str(case['size']), '--repeats', str(case['repeats'])]
            gates, lines, code_maps = [], [], []

            def observe(io, entry):
                while True:
                    line = io.line(seconds=25); lines.append(line)
                    if not line.startswith('GATE stage='):
                        continue
                    stage = line.split('=', 1)[1]
                    require(len(gates) < 3 and stage == ['before-runtime', 'runtime-ready', 'before-exit'][len(gates)], 'transfer gate order')
                    proc = Path('/proc', str(entry['pid']))
                    require(str((proc / 'exe').resolve()) == row['binary']['path'] and digest(proc / 'exe') == row['binary']['sha256'], 'transfer actual executable')
                    actual_env = dict(v.split(b'=', 1) for v in (proc / 'environ').read_bytes().split(b'\0') if v)
                    require(actual_env == {k.encode(): v.encode() for k, v in env.items()}, 'transfer immutable child env')
                    mapped = r.executable_mappings((proc / 'maps').read_text())
                    allowed = set(rt['runtime_files']) | {row['binary']['path']}
                    require(mapped <= allowed, 'unfrozen transfer provider: ' + repr(mapped - allowed))
                    for name in mapped:
                        require(digest(name) == (row['binary']['sha256'] if name == row['binary']['path'] else rt['runtime_files'][name]),
                                'transfer provider changed: ' + name)
                    if stage != 'before-runtime':
                        require(any('libze_intel_gpu.so.1.17.39758' in name for name in mapped) and
                                any('libur_adapter_level_zero_v2.so' in name for name in mapped), 'expected transfer providers absent')
                    gates.append(stage); code_maps.append(dict(stage=stage, maps=len(mapped)))
                    io.send(b'continue\n')
                    if stage == 'before-exit':
                        io.exit(10); break

            entry, out, err = supervisor.run('case-' + str(i), argv, env, P, interaction=observe, **RUN_LIMITS)
            rows = [{part.split('=', 1)[0]: part.split('=', 1)[1] for part in line.split()[1:]} for line in lines if line.startswith('COPY ')]
            checks = [line for line in lines if line.startswith('CHECK ')]
            passed = supervision.closed(entry) and entry['exit_code'] == 0 and gates == ['before-runtime', 'runtime-ready', 'before-exit']
            passed = passed and len(rows) == len(checks) == 2 * (case['repeats'] + 1) and all(
                'full=PASS guards=PASS source=PASS' in line for line in checks)
            require(any(line.startswith('DEVICE name=') and 'B570' in line for line in lines), 'actual transfer GPU differs')
            require(any(line.startswith('DONE ') for line in lines) and not any('FAIL' in line for line in lines), 'transfer completion failed')
            preload = r.validation.preload_evidence(err.read_text(), entry['pid'])
            require(preload['loader'] == rt['preload']['loader'], 'transfer actual UR loader differs')
            require(not warnings.exists() or not warnings.stat().st_size, 'transfer parameter validation warning')
            if case['diagnostic']:
                require(trace.exists() and 'urEnqueue' in trace.read_text(), 'transfer enqueue trace absent')
            for sample in rows:
                require(sample['host'] == case['host'] and int(sample['bytes']) == case['size'] and
                        all(math.isfinite(float(sample[k])) and float(sample[k]) > 0 for k in ('total_GBps', 'total_wall_s')),
                        'transfer sample configuration/clock mismatch')
            cases.append(dict(case=case, passed=bool(passed), samples=rows, gates=code_maps,
                              process={k: entry.get(k) for k in ('pid', 'exit_code', 'elapsed_seconds', 'peak_session_rss_bytes',
                              'observation_complete', 'direct_child_reaped', 'session_empty', 'error')}))
            write(RESULT, data)
            require(passed, 'transfer case failed')
            guards.driver_guard()
        verify(row)
        require(digest(row['binary']['path']) == row['binary']['sha256'], 'transfer binary changed')
        row.update(stage='closed', passed=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        try:
            raw = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'], timeout=15)
            require(len(raw) < 2 << 20, 'transfer fault evidence budget')
            row['fresh_faults'] = [s for s in [json.loads(line).get('MESSAGE', '') for line in raw.decode().splitlines() if line]
                                   if re.search(r'(?:\bxe\b|drm|GPU).*(?:fault|hang|wedg|reset|timed out)', s, re.I)]
            row['driver_after'] = guards.driver_guard()
        except BaseException as error:
            row.update(passed=False, fault_evidence_error=repr(error))
        row.update(active=supervisor.active is not None,
                   ownership_closed=supervisor.active is None and all(supervision.closed(c) for c in supervisor.commands))
        row['passed'] = bool(row.get('passed') and row['ownership_closed'] and not row.get('fresh_faults'))
        write(RESULT, data)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'error')}), flush=True)


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'run'), 'choose prepare,build,run')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        data = public(); {'prepare': prepare, 'build': build, 'run': run}[sys.argv[1]](data)


if __name__ == '__main__':
    main()
