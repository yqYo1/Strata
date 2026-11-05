#!/usr/bin/env python3
"""Own an offline Q6 expression compiler check; repository LICENSE.

Prepare and commit the installed inputs before compiling once. This does not
link an executable, load a SYCL provider, query a device or run either kernel.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig

sys.dont_write_bytecode = True
import device_fp64_probe as probe

P, B, RESULT = probe.P, probe.B, probe.RESULT
ROOT = B / 'q6-dot-codegen-v1'
KEY = 'q6_dot_codegen_v1'
SOURCE = 'bench/sycl/q6_dot_codegen.cpp'
NATIVE = B / 'decode-iq3-pair-model-v1'
owner, guards, require, sha, git = probe.owner, probe.guards, probe.require, probe.sha, probe.git


def compact(row):
    out = {k: v for k, v in row.items()
        if k not in ('sources', 'toolchain', 'python', 'runtime_pins', 'commands')}
    for name in ('sources', 'toolchain', 'python', 'runtime_pins'):
        if name in row:
            out[name + '_signature'] = probe.component.signature(row[name])
    return out


def save(row):
    raw = json.dumps(row, indent=2) + '\n'
    require(len(raw.encode()) <= 12 << 20, 'finite compiler receipt')
    (ROOT / 'record.json').write_text(raw)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row)
    RESULT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def sources():
    names = {SOURCE, 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md'}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and Path(name).resolve().is_relative_to(P):
            path = Path(name).resolve()
            require(path.suffix == '.py', 'unexpected project module type')
            names.add(str(path.relative_to(P)))
    return {n: sha(P / n) for n in sorted(names)}


def python_inputs():
    runtime = probe.component.python_runtime
    explicit = [sys.executable, *runtime.code_maps(os.getpid())]
    return runtime.inventory([sysconfig.get_path('stdlib')], explicit)


def production_command():
    database = NATIVE / 'build/compile_commands.json'
    commands = [v for v in json.loads(database.read_text())
        if v['file'].endswith('/src/kernels/sycl/native_mmvq.cpp')]
    require(len(commands) == 1, 'one actual native Q6 compilation required')
    source = Path(commands[0]['file'])
    native = json.loads((NATIVE / 'record.json').read_text())
    expected = native['native_sources']['src/kernels/sycl/native_mmvq.cpp']
    require(sha(source) == expected and sha(P / 'src/kernels/sycl/native_mmvq.cpp') == expected,
        'scalar reference differs from the compiled Q6 input')
    # The private discriminator has no production include or define dependency.
    # Pin the actual command; reproduce its optimization/precision settings.
    flags = ['-O3', '-DNDEBUG', '-std=c++20', '-ffp-contract=off', '-fsycl',
        '-fsycl-id-queries-range=size_t', '-Wall', '-Wextra', '-Werror',
        '-Wno-error=deprecated-declarations']
    require(all(f in commands[0]['command'].split() for f in flags),
        'native optimization settings differ')
    return dict(database=dict(path=str(database), sha256=sha(database)),
        native_source_commit=native['source_commit'], native_q6_sha256=expected,
        entry=commands[0], reproduced_flags=flags)


def prepare():
    cohort = probe.closed_cohort()  # Refuse before creating an attempt directory.
    require(not ROOT.exists() and KEY not in probe.validation.committed(RESULT),
        'fresh compiler attempt only')
    revision = git('rev-parse', 'HEAD').decode().strip()
    pins = sources()
    require(all(hashlib.sha256(git('show', revision + ':' + n)).hexdigest() == h
        for n, h in pins.items()), 'commit exact compiler controls first')
    native = production_command()
    row = dict(stage='preparing', source_commit=revision, sources=pins,
        passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, native=native, whole_bytes=32 << 20,
        plan=dict(question='Does the public signed-byte vector dot expression differ from the current scalar lowering?',
            prerequisite='All six IQ3 audit processes close normally,without fresh faults or changed inputs.',
            acceptance='One owned device-only LLVM compilation,normal exit0,closed commands,unchanged source/toolchain/Python inputs,actual header dependency admission and both named kernel definitions. No automatic retry.',
            decision='Main inspects both optimized instruction bodies. Equal lowering ends this expression proposal. A difference only motivates a separately committed whole-Q6 experiment;no instruction throughput,device support,model speed or adoption follows.',
            limits='Isolated expressions with unknown signed packed bytes,zero accumulator,128-item groups and subgroup32. Production reconstruction,scales,reductions and surrounding optimizations are absent. LLVM IR is not native B570 ISA or proof of hardware DP4A.'),
        retention='Keep source,recipe and compact outcome. Keep generated IR only for Main instruction comparison;retire it and the fixture when that finite consumer ends. No inference engine generation.',
        budgets=dict(wall=60, cpu=60, rss_cap=2 << 30, text_cap=256 << 10,
            file_cap=8 << 20, total_cap=32 << 20))
    ROOT.mkdir(mode=0o700)
    try:
        runtime = cohort['runtime_manifest']
        require(sha(runtime['path']) == runtime['sha256'], 'qualified library inventory changed')
        row['runtime_manifest'] = runtime
        row['runtime_pins'] = json.loads(Path(runtime['path']).read_text())
        require(len(row['runtime_pins']) == runtime['files'] and all(
            sha(n) == h for n, h in row['runtime_pins'].items()), 'qualified libraries changed')
        row['toolchain'] = probe.toolchain()
        row['python'] = python_inputs()
        tools = [shutil.which('git'), '/usr/bin/dpkg-query']
        require(all(tools), 'compiler selector/provenance tools missing')
        row['selector_tools'] = {n: probe.component.python_runtime.file_pin(n) for n in tools}
        env = dict(cohort['environment_template'])
        for name in tuple(env):
            if name.startswith(('STRATA_', 'UNITRACE_', 'UR_LOG_')) or name in (
                'LD_PRELOAD', 'UR_ENABLE_LAYERS', 'PYTHONPATH', 'PYTHONHOME'):
                env.pop(name)
        env.update(PATH=str(probe.COMPILER_ROOT / 'bin') + ':/usr/bin:/bin',
            TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
            SYCL_CACHE_PERSISTENT='0')
        row['environment'] = env
        compiler = str(probe.COMPILER_ROOT / 'bin/icpx')
        row['command'] = [compiler, *native['reproduced_flags'],
            '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
            '-fsycl-device-only', '-S', '-emit-llvm', '-v',
            '-MD', '-MT', str(ROOT / 'dot.ll'), '-MF', str(ROOT / 'headers.d'),
            str(ROOT / 'source.cpp'), '-o', str(ROOT / 'dot.ll')]
        for name in ('tmp', 'cache'): (ROOT / name).mkdir()
        (ROOT / 'source.cpp').write_bytes((P / SOURCE).read_bytes())
        row.update(stage='prepared', compiler_version=subprocess.check_output(
            [compiler, '--version'], env=env, text=True, timeout=15).splitlines()[0])
    except BaseException as error:
        row.update(stage='failed', preparation_error=repr(error))
        raise
    finally:
        save(row)


def current():
    probe.closed_cohort()
    row = json.loads((ROOT / 'record.json').read_text())
    require(probe.validation.committed(RESULT, (KEY,)) == compact(row),
        'commit compiler preparation before execution')
    require(row['stage'] == 'prepared' and not row['active'] and row['ownership_closed'],
        'terminal or owned compiler attempt')
    require(sources() == row['sources'] and all(hashlib.sha256(
        git('show', row['source_commit'] + ':' + n)).hexdigest() == h
        for n, h in row['sources'].items()), 'compiler source/control changed')
    require(sha(ROOT / 'source.cpp') == row['sources'][SOURCE], 'private source changed')
    require(production_command() == row['native'], 'production recipe/source changed')
    require(all(probe.component.python_runtime.file_pin(n) == p
        for n, p in row['selector_tools'].items()), 'selector/provenance tool changed')
    require(probe.toolchain() == row['toolchain'], 'compiler image changed')
    require(python_inputs() == row['python'], 'Python/runtime inputs changed')
    require(sha(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256']
        and all(sha(n) == h for n, h in row['runtime_pins'].items()), 'qualified libraries changed')
    require(owner.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole receipt/output budget')
    return row


def header_evidence(row):
    import shlex
    tokens = shlex.split((ROOT / 'headers.d').read_text().replace('\\\n', ''))
    require(tokens and tokens[0] == str(ROOT / 'dot.ll') + ':', 'dependency target differs')
    allowed = {v['realpath']: v['sha256'] for v in row['toolchain']['files'].values()
        if v['kind'] == 'file'}
    allowed[str((ROOT / 'source.cpp').resolve())] = row['sources'][SOURCE]
    verified = {}
    for name in tokens[1:]:
        path = Path(name)
        require(path.is_absolute(), 'relative compiler include')
        real = str(path.resolve(strict=True))
        require(real in allowed and sha(path) == allowed[real], 'unqualified compiler include: ' + name)
        verified[real] = allowed[real]
    require(len(verified) > 1, 'actual compiler includes absent')
    return probe.component.signature(verified)


def compile_once():
    row = current()
    receipt = ROOT / 'execution.json'
    require(not receipt.exists() and not (ROOT / 'dot.ll').exists(), 'existing compiler operation')
    outcome, job = dict(complete=False, passed=False, native_started=False), None
    def persist():
        receipt.write_text(json.dumps(dict(outcome, active=bool(job and job.active),
            commands=job.commands if job else []), indent=2) + '\n')
    with receipt.open('x') as stream: stream.write(json.dumps(outcome) + '\n')
    try:
        job = owner.Owner(ROOT / 'owned', persist)
        entry, stdout, stderr = job.run('compile', row['command'], row['environment'], P,
            **row['budgets'])
        require(owner.closed(entry) and entry['exit_code'] == 0, 'owned compiler failed')
        text = (ROOT / 'dot.ll').read_text()
        names = [n for n in re.findall(r'^define\b.*?@([^\s(]+)', text, re.MULTILINE)
            if 'Q6DotExpression' in n]
        require(len(names) == 2 and any('ILb0E' in n for n in names)
            and any('ILb1E' in n for n in names), 'two distinct kernel definitions required')
        require('spir64' in stderr.read_text(), 'actual compiler device triple absent')
        outcome.update(IR=dict(sha256=sha(ROOT / 'dot.ll'), bytes=(ROOT / 'dot.ll').stat().st_size,
            kernel_definitions=names), headers=header_evidence(row))
        current()
        outcome.update(complete=True, passed=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        outcome.update(native_started=bool(job and any('pid' in c for c in job.commands)),
            ownership_closed=job is None or (job.active is None and all(owner.closed(c) for c in job.commands)))
        outcome['passed'] = bool(outcome['passed'] and outcome['complete'] and outcome['ownership_closed'])
        row.update(stage='closed' if outcome['passed'] else 'failed', outcome=outcome,
            complete=outcome['complete'], passed=outcome['passed'],
            ownership_closed=outcome['ownership_closed'], active=bool(job and job.active),
            commands=job.commands if job else [])
        save(row); persist()
    require(outcome['passed'], outcome.get('error', 'compiler check failed'))
    print(json.dumps(dict(stage=row['stage'], passed=row['passed'], GPU_executed=False)))


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'compile'), 'use prepare|compile')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else compile_once()
