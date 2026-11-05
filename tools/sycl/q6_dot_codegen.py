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

P, B = probe.P, probe.B
RESULT = probe.RESULT.with_name('q6-codegen-current.json')
ROOT = B / 'q6-dot-codegen-v3'
KEY = 'q6_dot_codegen_v3'
SOURCE = 'bench/sycl/q6_dot_codegen.cpp'
HEADER = 'include/strata/kernels/q6_packed_center.hpp'
NATIVE = B / 'decode-iq3-pair-model-v1'
owner, guards, require, sha = probe.owner, probe.guards, probe.require, probe.sha
GIT = shutil.which('git')
require(GIT is not None, 'Git executable missing')
GIT = str(Path(GIT).resolve(strict=True))


def git(*args):
    return subprocess.check_output([GIT, *args], cwd=P, timeout=30)


def committed(path):
    raw = path.read_bytes()
    require(raw == git('show', 'HEAD:' + str(path.relative_to(P))),
        'commit exact compiler/cohort evidence first: ' + str(path))
    return json.loads(raw)


def check_boundary(cohort):
    processes = cohort['processes']
    count = len(processes)
    require(count in (2, 4, 6) and not cohort['active'] and cohort['ownership_closed']
        and cohort['stage'] == ('closed' if count == 6 else 'running')
        and (count != 6 or cohort['passed'] and cohort['complete'])
        and all(p['index'] == i and p['passed'] and p['complete']
            and p['ownership_closed'] and p['native_exit_code'] == 0
            and p['fresh_faults'] == [] for i, p in enumerate(processes)),
        'Q6 offline compile requires an acceptable closed IQ3 pair boundary')
    return dict(completed_processes=count, signature=probe.component.signature(cohort))


def closed_boundary():
    guards.no_model_owners([])
    cohort = committed(probe.RESULT)[probe.COHORT]
    boundary = check_boundary(cohort)
    local = json.loads((B / 'decode-iq3-audit-speed-v2/record.json').read_text())
    expected = {k: v for k, v in local.items() if k not in ('sources', 'model_assets')}
    expected['source_inputs'] = probe.component.signature(local['sources'])
    require(expected == cohort
        and all(sha(P / n) == h and hashlib.sha256(
            git('show', local['source_commit'] + ':' + n)).hexdigest() == h
            for n, h in local['sources'].items()), 'IQ3 record or frozen source changed')
    require(sha(cohort['binary']['path']) == cohort['binary']['sha256'],
        'qualified native executable changed')
    return cohort, boundary


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
    data = json.loads(RESULT.read_text()) if RESULT.exists() else dict(schema=1)
    data[KEY] = compact(row)
    RESULT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def sources():
    names = {SOURCE, HEADER, 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md'}
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
    # The discriminator adds only its standalone candidate header,no production defines.
    # Pin the actual command; reproduce its optimization/precision settings.
    flags = ['-O3', '-DNDEBUG', '-std=c++20', '-ffp-contract=off', '-fsycl',
        '-fsycl-id-queries-range=size_t', '-Wall', '-Wextra', '-Werror',
        '-Wno-error=deprecated-declarations']
    require(all(f in commands[0]['command'].split() for f in flags),
        'native optimization settings differ')
    return dict(database=dict(path=str(database), sha256=sha(database)),
        native_source_commit=native['source_commit'], native_q6_sha256=expected,
        entry=commands[0], reproduced_flags=flags)


def committed_sources():
    revision = git('rev-parse', 'HEAD').decode().strip()
    pins = sources()
    require(all(hashlib.sha256(git('show', revision + ':' + n)).hexdigest() == h
        for n, h in pins.items()), 'commit exact compiler controls first')
    return revision, pins


def prepare():
    cohort, boundary = closed_boundary()  # Refuse before creating an attempt directory.
    previous = committed(RESULT)['q6_dot_codegen_v1']
    require(previous['stage'] == 'failed' and not previous['passed']
        and not previous['GPU_executed'] and previous['ownership_closed']
        and previous['outcome']['compiler_diagnostic']['message'] == 'IR output is not supported.',
        'preserve V1 textual-IR rejection before a different output recipe')
    second = committed(RESULT)['q6_dot_codegen_v2']
    require(second['stage'] == 'failed' and not second['passed']
        and not second['GPU_executed'] and second['ownership_closed']
        and second['outcome']['compiler_stage']['exit_code'] == 0
        and second['outcome']['translation_diagnostic']['message'] ==
            'llvm-spirv was built without --spirv-tools-dis support',
        'preserve V2 unavailable text backend before selecting internal SPIR-V text')
    require(not ROOT.exists() and KEY not in committed(RESULT),
        'fresh compiler attempt only')
    revision, pins = committed_sources()
    require(all(pins[n] == hashlib.sha256(git('show', previous['source_commit'] + ':' + n)).hexdigest()
        for n in (SOURCE, HEADER)), 'V3 changes output recipe,not either expression treatment')
    native = production_command()
    row = dict(stage='preparing', source_commit=revision, sources=pins,
        passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, native=native, boundary=boundary,
        git_executable=GIT, whole_bytes=32 << 20,
        plan=dict(question='Independently compare scalar/vector signed-byte dot and scalar/packed Q6 byte centering lowerings.',
            prerequisite='At a closed adjacent IQ3 pair boundary (2,4,6 completed processes),all prior processes pass normally without fresh faults. Hold the shared exclusive lock throughout,exclude live model owners and preserve the completed-cohort signature before/after compilation. Existing A/B and FP64 plans stay unchanged.',
            acceptance='Two owned CPU operations:device-only LLVM bitcode compilation,then installed llvm-spirv internal-text conversion;both normal exit0,closed commands,unchanged source/toolchain/Python/cohort inputs,actual header dependency admission,bitcode and SPIR-V magic,emitted Physical64/OpenCL (2,2) memory model and two kernel entrypoints/bodies in each dot/centering family. Compile-time centering byte relation and mixed-lane example must pass. No automatic retry.',
            decision='Main compares each optimized SPIR-V instruction pair independently. Equal lowering ends that expression proposal. A difference only motivates a separately committed whole-Q6 experiment;no instruction throughput,device support,model speed or adoption follows.',
            limits='Isolated dot with unknown signed packed bytes and zero accumulator;independent centering contrast masks unknown words to six bits per byte. Both use128-item groups and subgroup32. Full production reconstruction,scales,reductions and surrounding optimizations are absent. Portable SPIR-V is not native B570 ISA or proof of hardware DP4A.',
            output_mode='V1 textual LLVM emission failed;V2 binary compilation passed but SPIRV-Tools text output was not built into the translator. V3 retains the same installed compiler/optimization settings and both expression treatments,then requests the translator internal text only. Target proof comes from emitted module memory-model operands,not verbose driver text.'),
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
        require(probe.component.signature(row['toolchain']) == previous['toolchain_signature'],
            'V3 must use the same installed compiler image as V1/V2')
        row['python'] = python_inputs()
        tools = [GIT, '/usr/bin/dpkg-query']
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
            '-fsycl-device-only', '-c', '-emit-llvm', '-v',
            '-I', str(P / 'include'),
            '-MD', '-MT', str(ROOT / 'dot.bc'), '-MF', str(ROOT / 'headers.d'),
            str(ROOT / 'source.cpp'), '-o', str(ROOT / 'dot.bc')]
        row['translate_command'] = [str(probe.COMPILER_ROOT / 'bin/compiler/llvm-spirv'),
            '--spirv-text', str(ROOT / 'dot.bc'), '-o', str(ROOT / 'dot.spt')]
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
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['git_executable'] == GIT and all(probe.component.python_runtime.file_pin(n) == p
        for n, p in row['selector_tools'].items()), 'selector/provenance tool changed')
    _, boundary = closed_boundary()
    require(boundary == row['boundary'], 'completed IQ3 cohort changed since compiler preparation')
    require(committed(RESULT)[KEY] == compact(row),
        'commit compiler preparation before execution')
    require(row['stage'] == 'prepared' and not row['active'] and row['ownership_closed'],
        'terminal or owned compiler attempt')
    require(sources() == row['sources'] and all(hashlib.sha256(
        git('show', row['source_commit'] + ':' + n)).hexdigest() == h
        for n, h in row['sources'].items()), 'compiler source/control changed')
    require(sha(ROOT / 'source.cpp') == row['sources'][SOURCE], 'private source changed')
    require(production_command() == row['native'], 'production recipe/source changed')
    require(probe.toolchain() == row['toolchain'], 'compiler image changed')
    require(python_inputs() == row['python'], 'Python/runtime inputs changed')
    require(sha(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256']
        and all(sha(n) == h for n, h in row['runtime_pins'].items()), 'qualified libraries changed')
    require(owner.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole receipt/output budget')
    return row


def header_evidence(row):
    import shlex
    tokens = shlex.split((ROOT / 'headers.d').read_text().replace('\\\n', ''))
    require(tokens and tokens[0] == str(ROOT / 'dot.bc') + ':', 'dependency target differs')
    allowed = {v['realpath']: v['sha256'] for v in row['toolchain']['files'].values()
        if v['kind'] == 'file'}
    allowed.update({str((P / name).resolve()): h for name, h in row['sources'].items()})
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


def ir_evidence(text):
    headers = re.findall(r'^\s*119734787\s+(\d+)\s+(\d+)\s+(\d+)\s+0\s*$', text, re.MULTILINE)
    require(len(headers) == 1, 'SPIR-V internal text header missing')
    models = re.findall(r'^\s*3\s+MemoryModel\s+(\d+)\s+(\d+)\s*$', text, re.MULTILINE)
    require(models == [('2', '2')], 'emitted Physical64/OpenCL memory model differs')
    entries = re.findall(r'^\s*\d+\s+EntryPoint\s+6\s+(\d+)\s+"([^"]+)"', text, re.MULTILINE)
    bodies = re.findall(r'^\s*5\s+Function\s+\d+\s+(\d+)\s', text, re.MULTILINE)
    require(len(entries) == 4 and len({symbol for symbol, _ in entries}) == 4
        and all(bodies.count(symbol) == 1 for symbol, _ in entries),
        'kernel entrypoints must bind distinct SPIR-V function bodies')
    families = {kind: [name for _, name in entries if kind in name]
        for kind in ('Q6DotExpression', 'Q6CenterExpression')}
    require(all(len(v) == 2 and any('ILb0E' in n for n in v)
        and any('ILb1E' in n for n in v) for v in families.values()),
        'two distinct SPIR-V kernel definitions per family required')
    return dict(format='LLVM-SPIRV-Translator internal text', memory_model=['Physical64', 'OpenCL'],
        memory_model_operands=[2, 2], kernel_definitions=families)


def host():
    revision, pins = committed_sources()
    cohort, boundary = closed_boundary()
    native = production_command()
    rejected = 0

    def reject(function, value):
        nonlocal rejected
        try:
            function(value)
        except RuntimeError:
            rejected += 1
        else:
            raise RuntimeError('host control admitted invalid evidence')

    # Corrupt the real closed-boundary receipt without changing any disk input.
    for key, value in (('active', True), ('ownership_closed', False), ('stage', 'failed')):
        bad = json.loads(json.dumps(cohort)); bad[key] = value
        reject(check_boundary, bad)
    for key, value in (('passed', False), ('complete', False), ('ownership_closed', False),
                       ('native_exit_code', 9), ('fresh_faults', ['synthetic fault']), ('index', 9)):
        bad = json.loads(json.dumps(cohort)); bad['processes'][0][key] = value
        reject(check_boundary, bad)
    bad = json.loads(json.dumps(cohort)); bad['processes'].pop()
    reject(check_boundary, bad)
    changed = json.loads(json.dumps(cohort)); changed['processes'][0]['synthetic_extra'] = True
    require(check_boundary(changed) != boundary, 'cohort signature missed changed evidence')
    names = [f'{kind}ILb{arm}E' for kind in ('Q6DotExpression', 'Q6CenterExpression') for arm in (0, 1)]
    text = '119734787 65536 393230 100 0\n3 MemoryModel 2 2\n' + '\n'.join(
        f'9 EntryPoint 6 {index} "{name}"\n5 Function 9 {index} 0 10\n1 FunctionEnd'
        for index, name in enumerate(names, 1))
    ir_evidence(text)
    for bad in (text.replace('MemoryModel 2 2', 'MemoryModel 1 2'),
                text.replace('3 MemoryModel 2 2\n', ''), '3 MemoryModel 2 2\n' + text,
                text.replace('EntryPoint 6', 'EntryPoint 5'),
                text.replace('5 Function 9 1 0', '5 Function 9 99 0'),
                text.replace('119734787', '0'),
                text.replace('ILb1E', 'ILb0E')):
        reject(ir_evidence, bad)
    require(not ROOT.exists() and KEY not in committed(RESULT), 'host checks created compiler artifacts')
    print(json.dumps(dict(source_commit=revision, passed=True, sources=len(pins),
        boundary=boundary, rejected_controls=rejected, changed_cohort_detected=True,
        production_flags=len(native['reproduced_flags']), compiler_started=False,
        GPU_executed=False, root_created=False)))


def compile_once():
    row = current()
    receipt = ROOT / 'execution.json'
    require(not receipt.exists() and not (ROOT / 'dot.bc').exists()
        and not (ROOT / 'dot.spt').exists(), 'existing compiler operation')
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
        with (ROOT / 'dot.bc').open('rb') as stream:
            require(stream.read(4) == b'BC\xc0\xde', 'LLVM bitcode output missing')
        entry, stdout, stderr = job.run('translate', row['translate_command'], row['environment'], P,
            **row['budgets'])
        require(owner.closed(entry) and entry['exit_code'] == 0, 'owned SPIR-V translation failed')
        text = (ROOT / 'dot.spt').read_text()
        outcome.update(IR=dict(sha256=sha(ROOT / 'dot.spt'), bytes=(ROOT / 'dot.spt').stat().st_size,
            bitcode_sha256=sha(ROOT / 'dot.bc'),
            **ir_evidence(text)), headers=header_evidence(row))
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
    require(len(sys.argv) == 2 and sys.argv[1] in ('host', 'prepare', 'compile'), 'use host|prepare|compile')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        {'host': host, 'prepare': prepare, 'compile': compile_once}[sys.argv[1]]()
