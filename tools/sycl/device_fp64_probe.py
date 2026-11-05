#!/usr/bin/env python3
"""Own the exact-B570 FP64 diagnostic; Main-authored, repository LICENSE.

This is a capability/numerical check for the next sampler decision, not a
sampler benchmark or proof of native FP64 throughput. Do not run it during
the frozen IQ3 audit comparison. Every transition requires committed evidence.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import mapped_row_copy_component as component

P, B, RESULT = component.P, component.B, component.RESULT
ROOT = B / 'device-fp64-probe-v1'
KEY = 'device_fp64_probe_v1'
COHORT = 'decode_iq3_audit_speed_v2'
SOURCE = 'tools/sycl/device_fp64_probe.cpp'
HELPERS = (SOURCE, 'tools/sycl/device_fp64_probe.py',
    *component.HELPERS, 'tools/sycl/measurement_environment.json')
OPERATIONS = ('query', 'arithmetic', 'exp')
COMPILER_ROOT = Path('/opt/intel/oneapi/compiler/2026.2')
owner, guards, validation = component.owner, component.guards, component.validation
require, sha, git = component.require, component.sha, component.git


def compact(row):
    out = {k: v for k, v in row.items()
        if k not in ('sources', 'runtime_pins', 'toolchain', 'commands')}
    for name in ('sources', 'runtime_pins'):
        if name in row:
            out[name + '_signature'] = component.signature(row[name])
    if 'toolchain' in row:
        out['toolchain_signature'] = component.signature(row['toolchain']['files'])
        out['toolchain_bytes'] = row['toolchain']['bytes']
    return out


def save(row):
    (ROOT / 'record.json').write_text(json.dumps(row, indent=2) + '\n')
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    RESULT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def closed_cohort():
    cohort = validation.committed(RESULT, (COHORT,))
    require(cohort['stage'] == 'closed' and cohort['passed'] and cohort['complete']
        and not cohort['active'] and cohort['ownership_closed']
        and len(cohort['processes']) == 6
        and all(p['passed'] and p['complete'] and p['ownership_closed']
            and p['native_exit_code'] == 0 and p['fresh_faults'] == []
            for p in cohort['processes']), 'close all six IQ3 audit processes first')
    return cohort


def file_image(roots, paths):
    """Hash full named trees and package members, including symlink/membership changes."""
    deadline = time.monotonic() + 180
    names, pending, visited = set(map(str, paths)), list(map(Path, roots)), set()
    while pending:
        directory = pending.pop()
        real = directory.resolve(strict=True)
        names.add(str(directory))
        if real in visited:
            continue
        visited.add(real)
        for path in sorted(directory.iterdir()):
            names.add(str(path))
            require(len(names) <= 131072 and time.monotonic() < deadline,
                'finite toolchain tree traversal')
            if path.is_dir():
                pending.append(path)
    files, content, total = {}, {}, 0
    for name in sorted(names):
        require(time.monotonic() < deadline, 'toolchain hashing deadline')
        path = Path(name)
        require(path.is_absolute(), 'absolute toolchain member required')
        item = dict(realpath=str(path.resolve()))
        if path.is_symlink():
            item['symlink'] = os.readlink(path)
        if path.is_file():
            real = item['realpath']
            if real not in content:
                pin = component.python_runtime.file_pin(path)
                total += pin['bytes']
                require(total <= 16 << 30, 'toolchain payload budget')
                content[real] = pin
            item.update(kind='file', **content[real])
        elif path.is_dir():
            item['kind'] = 'directory'
        else:
            require(not path.exists(), 'special toolchain member')
            item['kind'] = 'dangling_symlink' if path.is_symlink() else 'missing'
        files[name] = item
    require(len(files) <= 131072, 'toolchain member count budget')
    return dict(files=files, bytes=total)


def toolchain():
    package_files = [p for p in subprocess.check_output(['/usr/bin/dpkg-query', '-L',
        *component.PACKAGES], text=True, timeout=15).splitlines() if p]
    require(package_files and all(p.startswith('/') for p in package_files),
        'complete named toolchain package file lists required')
    image = dict(file_image([COMPILER_ROOT], package_files),
        packages=component.packages(), compiler_root=str(COMPILER_ROOT))
    require(len(json.dumps(image).encode()) <= 8 << 20, 'toolchain manifest byte budget')
    return image


def build_inputs(row, text):
    """Check actual compiler include and host-link inputs against the prebuild image."""
    allowed = {v['realpath']: v['sha256'] for v in row['toolchain']['files'].values()
        if v['kind'] == 'file'}
    allowed.update({str(Path(p).resolve()): h for p, h in row['runtime_pins'].items()})
    allowed[str((ROOT / 'probe.cpp').resolve())] = row['sources'][SOURCE]
    tokens = shlex.split((ROOT / 'headers.d').read_text().replace('\\\n', ''))
    require(tokens and tokens[0] == row['executable'] + ':', 'dependency target differs')
    headers = tokens[1:]
    links = [s.strip() for s in text.splitlines() if s.startswith('/')]
    require(headers and links, 'actual include/link input evidence missing')
    actual, generated = {}, []
    for name in dict.fromkeys(headers + links):
        path = Path(name)
        require(path.is_absolute(), 'relative compiler input')
        if path.is_relative_to(ROOT / 'tmp'):
            generated.append(name)
            continue
        real = str(path.resolve(strict=True))
        require(real in allowed and sha(path) == allowed[real],
            'unqualified compiler include/link input: ' + name)
        actual[real] = allowed[real]
    require(any(Path(n).name.startswith('libc.') for n in links)
        and any(Path(n).name.startswith('libstdc++.') for n in links),
        'host runtime link trace missing')
    return dict(verified_input_signature=component.signature(actual),
        host_link_inputs={str(Path(n).resolve()): actual[str(Path(n).resolve())]
            for n in links if not Path(n).is_relative_to(ROOT / 'tmp')},
        derived_input_count=len(generated),
        header_count=len(headers), host_link_count=len(links),
        limits='Compiler-generated private temporaries are derived inputs,not separately retained objects;the full installed compiler image and named host toolchain packages are pinned.')


def prepare():
    cohort = closed_cohort()
    require(not ROOT.exists() and KEY not in validation.committed(RESULT),
        'fresh diagnostic only;no retry or overwritten outcome')
    revision = git('rev-parse', 'HEAD').decode().strip()
    sources = {n: sha(P / n) for n in set(HELPERS)}
    require(all(hashlib.sha256(git('show', revision + ':' + n)).hexdigest() == h
        for n, h in sources.items()), 'commit all diagnostic inputs first')
    ROOT.mkdir(mode=0o700)
    row = dict(stage='preparing', source_commit=revision, sources=sources,
        active=False, ownership_closed=True, passed=False, complete=False,
        model_executed=False, kernels_submitted=False, results=[], commands=[])
    try:
        save(row)
        env = {k: v for k, v in cohort['environment_template'].items()
            if not k.startswith('STRATA_') and k not in ('LD_PRELOAD',
                'PYTHONPYCACHEPREFIX', 'UR_ENABLE_LAYERS', 'UR_LOG_VALIDATION',
                'UR_LOG_TRACING', 'UR_LOG_LOADER', 'UR_LOG_LEVEL_ZERO')}
        env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
            SYCL_CACHE_PERSISTENT='0', ZES_ENABLE_SYSMAN='0',
            PATH=str(COMPILER_ROOT / 'bin') + ':/usr/bin:/bin')
        runtime = cohort['runtime_manifest']
        require(sha(runtime['path']) == runtime['sha256'], 'provider inventory changed')
        pins = json.loads(Path(runtime['path']).read_text())
        require(len(pins) == runtime['files'] and all(sha(n) == h for n, h in pins.items()),
            'qualified runtime files changed')
        probe = cohort['native_validation_probe']
        for asset in probe.values():
            require(pins.get(str(Path(asset['path']).resolve())) == asset['sha256'],
                'validation library/loader changed')
        compiler = str(COMPILER_ROOT / 'bin/icpx')
        tools = {n: dict(path=s, realpath=str(Path(s).resolve()), sha256=sha(s))
            for n, s in dict(compiler=compiler, assembler='/usr/bin/as',
                linker='/usr/bin/ld', package_query='/usr/bin/dpkg-query',
                python=sys.executable).items()}
        row.update(environment=env, runtime_manifest=runtime, runtime_pins=pins,
            native_validation_probe=probe, tools=tools,
            compiler_version=subprocess.check_output([compiler, '--version'],
                env=env, text=True, timeout=15).splitlines()[0],
            toolchain=toolchain(),
            boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            driver=guards.driver_guard(), executable=str(ROOT / 'probe'),
            command=[compiler, '-fsycl', '-std=c++17', '-O2', '-Wall', '-Wextra',
                '-Werror', '-Wno-error=deprecated-declarations',
                '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
                '-MD', '-MT', str(ROOT / 'probe'), '-MF', str(ROOT / 'headers.d'), '-Wl,-t',
                str(ROOT / 'probe.cpp'), '-o', str(ROOT / 'probe')],
            plan=dict(order='Compile,query,arithmetic,exp;one fresh process each. Commit each transition before next. No retries.',
                prerequisite='Completed six-process same-binary IQ3 audit comparison.',
                acceptance='Exact B570 device/vendor/PCI identity,qualified safety and validation activation,normal exit0,quiet validation,closed ownership,unchanged inputs/runtime/boot/driver and no fresh fault. Query submits no kernel. If fp64 is absent,stop before arithmetic. Arithmetic preserves2^-40;64exp values over[-63,0] have relative error<=1e-12. Source and allocation guards stay unchanged.',
                limits='Capability and bounded numerical checks only;not sampler equivalence,physical/native FP64 throughput,absence of emulation,ordinary provider teardown,model speed or adoption.'),
            observer_limits='Trace substrings establish activation/expected symbols,not parsed call results/counts/handles. Query submits no kernel by its pinned C++ branch. CPU and file caps are per-process RLIMIT_CPU/RLIMIT_FSIZE;RSS and present artifact bytes are sampled session bounds plus exit checks,not cumulative writes or strict cgroup quotas.',
            retention='Source/recipe and compact operation outcomes. Retire successful traces/logs/private source and diagnostic executable when the sampler capability decision consumes them;no inference engine generation.',
            budgets=dict(build=dict(wall=180, cpu=180, rss_cap=4 << 30,
                    text_cap=1 << 20, file_cap=4 << 20, total_cap=16 << 20),
                native=dict(wall=60, cpu=60, rss_cap=1 << 30,
                    text_cap=1 << 20, file_cap=1 << 20, total_cap=16 << 20)),
            whole_bytes=16 << 20, stage='prepared')
        for name in ('tmp', 'cache'): (ROOT / name).mkdir()
        (ROOT / 'probe.cpp').write_bytes(P.joinpath(SOURCE).read_bytes())
        save(row)
    except BaseException as error:
        row.update(stage='failed', preparation_error=repr(error))
        save(row)
        raise


def current():
    closed_cohort()
    row = json.loads((ROOT / 'record.json').read_text())
    require(validation.committed(RESULT, (KEY,)) == compact(row),
        'commit this diagnostic transition before continuing')
    require(not row['active'] and row['ownership_closed'], 'previous ownership unresolved')
    require(row['stage'] not in ('failed', 'unsupported', 'closed'), 'terminal diagnostic')
    for name, expected in row['sources'].items():
        require(sha(P / name) == expected and hashlib.sha256(
            git('show', row['source_commit'] + ':' + name)).hexdigest() == expected,
            'diagnostic source changed: ' + name)
    require(sha(ROOT / 'probe.cpp') == row['sources'][SOURCE], 'private source changed')
    for asset in row['tools'].values():
        require(str(Path(asset['path']).resolve()) == asset['realpath']
            and sha(asset['path']) == asset['sha256'], 'tool changed')
    require(all(sha(n) == h for n, h in row['runtime_pins'].items()), 'runtime changed')
    require(toolchain() == row['toolchain'], 'compiler/package membership or bytes changed')
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'],
        'boot/driver changed')
    require(owner.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole artifact budget')
    return row


def run(operation):
    row = current()
    building = operation == 'build'
    require(row['stage'] == ('prepared' if building else 'built'), 'wrong build stage')
    execution = ROOT / (operation + '-execution.json')
    require(not execution.exists(), 'existing operation receipt;inspect original ownership')
    if building:
        require(not Path(row['executable']).exists(), 'existing executable')
        command, env = row['command'], row['environment']
    else:
        require(operation == OPERATIONS[len(row['results'])], 'strict operation order')
        require(sha(row['executable']) == row['executable_sha256'], 'executable changed')
        folder = ROOT / operation
        require(not folder.exists(), 'existing operation attempt')
        env = dict(row['environment'], LD_PRELOAD=row['native_validation_probe']['library']['path'],
            STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'trace.log'),
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(folder / 'validation.log'))
        command = [row['executable'], '--operation', operation]
    outcome = dict(operation=operation, complete=False, host_probe_accepted=False,
        native_started=False)
    job, cursor = None, None
    def persist():
        row.update(active=bool(job and job.active is not None),
            ownership_closed=job is None or (job.active is None
                and all(owner.closed(c) for c in job.commands)))
        row['commands'] = job.commands if job else []
        execution.write_text(json.dumps(dict(outcome,
            active=row['active'], commands=row['commands']), indent=2) + '\n')
    try:
        # An attempt exists before cursor/Owner/directory setup. A failure here
        # is terminal and retained, even though no native process was started.
        with execution.open('x') as stream:
            stream.write(json.dumps(dict(outcome, active=False, commands=[]), indent=2) + '\n')
        cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
            ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
            text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
        job = owner.Owner(ROOT / 'owned', persist)
        if not building:
            folder.mkdir()
        entry, stdout, stderr = job.run(operation, command, env, P,
            **row['budgets']['build' if building else 'native'])
        outcome['native_started'] = 'pid' in entry
        require(owner.closed(entry) and entry['exit_code'] == 0, 'owned operation failed')
        if building:
            row['compiler_input_evidence'] = build_inputs(row, stdout.read_text())
            row['executable_sha256'] = sha(row['executable'])
        else:
            text, errors = stdout.read_text(), stderr.read_text()
            proof = validation.preload_evidence(errors, entry['pid'])
            require(proof['loader'] == row['native_validation_probe']['loader']['path'],
                'actual validation loader differs')
            require('ERROR ' not in text + errors, 'device error output')
            log = ROOT / operation / 'validation.log'
            require(not log.exists() or log.stat().st_size == 0, 'validation output')
            trace = (ROOT / operation / 'trace.log').read_text()
            require('urDeviceGetInfo' in trace, 'device query API trace missing')
            submitted = any(v in trace for v in ('urEnqueueKernelLaunch',
                'urEnqueueCooperativeKernelLaunch', 'urEnqueueCommandBufferExp'))
            require(submitted is (operation != 'query'), 'kernel trace differs from operation')
            devices = [v for v in text.splitlines() if v.startswith('DEVICE ')]
            passes = [v for v in text.splitlines() if v.startswith('PASS ')]
            require(len(devices) == len(passes) == 1, 'missing/duplicate device/pass receipt')
            device = dict(v.split('=', 1) for v in shlex.split(devices[0])[1:])
            require(device['backend'] == 'level_zero' and device['vendor'] == '32902'
                and device['device_id'] == '57868' and device['pci'] == '0000:05:00.0'
                and device['name'] == 'Intel Arc B570 Graphics'
                and device['fp64'] in ('0', '1'), 'exact-device receipt differs')
            require(passes[0].startswith('PASS operation=' + operation + ' '), 'wrong operation receipt')
            if operation == 'query':
                require(passes[0] == 'PASS operation=query kernels_submitted=0', 'query submitted work')
            else:
                require(device['fp64'] == '1' and 'checked=64 guards=pass source=unchanged ' in passes[0],
                    'FP64 arithmetic receipt differs')
            outcome.update(device=device, receipt=passes[0], validation=proof,
                kernels_submitted=operation != 'query')
        current()  # Recheck the original frozen inputs after the owned child closes.
        outcome.update(complete=True, host_probe_accepted=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        if job:
            outcome['native_started'] = any('pid' in c for c in job.commands)
            validation.finish(outcome, job, cursor, row['boot'])
        else:
            outcome.update(passed=False, ownership_closed=True,
                fresh_faults=None, boot_unchanged=None,
                prelaunch_failure=True, limits='No native child was launched;post-run fault history is unqualified.')
        outcome['prelaunch_failure'] = not outcome['native_started']
        outcome['passed'] = bool(outcome['passed'] and outcome['complete']
            and outcome.get('driver_after') == row['driver'])
        row.update(active=bool(job and job.active is not None),
            ownership_closed=outcome['ownership_closed'])
        if building:
            row['build_outcome'] = outcome
        else:
            row['results'].append(outcome)
        row['stage'] = 'built' if outcome['passed'] else 'failed'
        if outcome['passed'] and operation == 'query' and outcome['device']['fp64'] == '0':
            row['stage'] = 'unsupported'
        if outcome['passed'] and operation == 'exp':
            row.update(stage='closed', complete=True, passed=True)
        row['kernels_submitted'] = any(x.get('kernels_submitted') for x in row['results'])
        save(row)
        persist()
    require(outcome['passed'], outcome.get('error', 'diagnostic operation failed'))
    print(json.dumps(dict(operation=operation, stage=row['stage'], passed=outcome['passed'])), flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', *OPERATIONS),
        'use prepare|build|query|arithmetic|exp')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else run(sys.argv[1])
