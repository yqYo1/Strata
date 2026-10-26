#!/usr/bin/env python3
"""Build our matched UR observer control; Main only, repository LICENSE.

Official UR headers/link provider, system C compiler, no historical SDK build.
No model/GPU execution or change to the selected engine and runtime settings.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True
import qualify_native16_control as q

r, P, B, RESULT = q.r, q.P, q.B, q.RESULT
ROOT = B / 'ur-hold-control-v1'
KEY = 'ur_hold_control_build_v1'
SOURCE = P / 'bench/sycl/ur_hold_control.c'
OFFICIAL = Path('/opt/intel/oneapi/compiler/2026.2')
require, digest, write = q.require, q.digest, q.write


def compact(row):
    return {k: v for k, v in row.items() if k not in
            ('controller_inputs', 'native_inputs', 'commands', 'environment',
             'expected_header_paths', 'expected_link_paths')}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(row):
    raw = RESULT.read_text(); desired = json.loads(raw)
    q.build.scope(desired)[KEY] = compact(row)
    literal = json.dumps(KEY) + ':'
    value = json.dumps(compact(row), indent=2).replace('\n', '\n      ')
    if KEY in q.build.scope(json.loads(raw)):
        require(raw.count(literal) == 1, 'ambiguous hold-control scope')
        start = raw.index(literal) + len(literal)
        while raw[start].isspace(): start += 1
        _, size = json.JSONDecoder().raw_decode(raw[start:])
        raw = raw[:start] + value + raw[start + size:]
    else:
        anchor = '"fused_command_profile": {'
        require(raw.count(anchor) == 1, 'ambiguous result container')
        raw = raw.replace(anchor, anchor + '\n      ' + literal + ' ' + value + ',', 1)
    require(json.loads(raw) == desired, 'scoped update changed unrelated evidence')
    write(ROOT / 'record.json', row); RESULT.write_text(raw)


def dependencies(text):
    names = shlex.split(text.replace('\\\n', ' '))[1:]
    return {str(Path(name).resolve()) for name in names}


def controller(revision, worktree=True):
    return q.gen.benchmark_pins(revision, match_worktree=worktree)


def prepare():
    require(not ROOT.exists() and KEY not in q.build.scope(r.validation.committed(RESULT)),
            'single-use helper build;inspect existing attempt')
    require(not (B / 'native16-legacy-gemm-v1').exists() and
        not (B / 'legacy-gemm-abi-v1').exists(), 'retire closed legacy generation first')
    q.current(json.loads((q.ROOT / 'record.json').read_text()))
    rt = r.validation.committed(r.RESULT); selected = r.current(rt)
    revision = r.validation.git('rev-parse', 'HEAD').decode().strip()
    pins = controller(revision)
    require(SOURCE.read_bytes() == r.validation.git('show', revision + ':' + str(SOURCE.relative_to(P))),
        'commit exact native source')
    compiler = str(Path('/usr/bin/cc').resolve())
    tools = dict(cc=compiler, readelf=str(Path('/usr/bin/readelf').resolve()))
    for name in ('cc1', 'as', 'ld', 'collect2'):
        value = subprocess.check_output([compiler, '-print-prog-name=' + name], text=True, timeout=10).strip()
        path = Path(shutil.which(value) or value).resolve()
        require(path.is_file(), 'missing build tool: ' + name); tools[name] = str(path)
    flags = ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-fPIC', '-I' + str(OFFICIAL / 'include')]
    headers = dependencies(subprocess.check_output([compiler, *flags, '-M', str(SOURCE)], text=True, timeout=15))
    require(str(OFFICIAL / 'include/unified-runtime/ur_api.h') in headers and
        str(SOURCE) in headers, 'actual official header closure missing')
    link = {str((OFFICIAL / 'lib/libur_loader.so').resolve())}
    for name in ('crti.o', 'crtn.o', 'crtbeginS.o', 'crtendS.o', 'libgcc.a', 'libgcc_s.so',
                 'libgcc_s.so.1', 'libc.so', 'libc.so.6', 'libc_nonshared.a', 'libdl.a', 'ld-linux-x86-64.so.2'):
        value = subprocess.check_output([compiler, '-print-file-name=' + name], text=True, timeout=10).strip()
        require(Path(value).is_file(), 'unresolved planned linker input: ' + name)
        link.add(str(Path(value).resolve()))
    inputs = {path: digest(path) for path in sorted(headers | link | set(tools.values()))}
    require(not any('/MistVVK/' in path for path in inputs), 'historical SDK input in new helper')
    env = dict(r.validation.committed(r.SETTINGS)['environment'],
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    require('LD_PRELOAD' not in env, 'preload belongs only to future native child')
    row = dict(stage='prepared', source_commit=revision, controller_inputs=pins,
        native_source=dict(path=str(SOURCE), sha256=digest(SOURCE)), native_inputs=inputs,
        official_header=str(OFFICIAL / 'include/unified-runtime/ur_api.h'),
        compiler=dict(path=compiler, sha256=digest(compiler),
            version=subprocess.check_output([compiler, '--version'], text=True, timeout=10).splitlines()[0]),
        tools={k: dict(path=v, sha256=digest(v)) for k, v in tools.items()},
        expected_header_paths=sorted(headers), expected_link_paths=sorted(link),
        compile_argv=[compiler, *flags, '-MD', '-MF', str(ROOT / 'actual-deps.d'),
            '-c', str(SOURCE), '-o', str(ROOT / 'control.o')],
        link_argv=[compiler, '-shared', str(ROOT / 'control.o'), '-L' + str(OFFICIAL / 'lib'),
            '-Wl,-rpath,' + str(OFFICIAL / 'lib'), '-Wl,-t', '-lur_loader', '-ldl',
            '-o', str(ROOT / 'libstrata_ur_hold_control.so')],
        environment=env, runtime_sha256=digest(r.RESULT), selected_binary=selected['binary'],
        fallback_component='native16-public-control-v4', boot=rt['boot'], driver=rt['driver'],
        budgets=dict(wall=30, cpu=30, rss_cap=1 << 30, text_cap=2 << 20,
            file_cap=8 << 20, total_cap=64 << 20),
        complete=False, passed=False, active=False, ownership_closed=True,
        GPU_executed=False, engine_built=False, model_adoption_qualified=False,
        question='Build one matched on/off parameter-validation preload retaining the loader reference in both arms.',
        criteria='Committed source/recipe/compiler/header and imported-link closure;finite normal owned '
            'compile/link exits;exact helper ELF. No old SDK compiler/header input. No GPU/model or '
            'ordinary-teardown qualification. Audit/off-mode host gates precede any GPU comparison.',
        retention='Keep this one tiny helper candidate only for imminent host/provider/submission and '
            'same-engine validation-cost comparison. Official v4fallback/original13model assets remain '
            'intact. Retire helper/raw output after that decision and original-source recovery.')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    write(ROOT / 'native-inputs.json', inputs)
    row['native_manifest'] = dict(path=str(ROOT / 'native-inputs.json'),
        sha256=digest(ROOT / 'native-inputs.json'), files=len(inputs))
    row.update(expected_header_paths_sha256=identity(row['expected_header_paths']),
        expected_link_paths_sha256=identity(row['expected_link_paths']), environment_sha256=identity(env))
    save(row); print('Prepared;commit exact helper build plan first.', flush=True)


def current(row):
    require(q.build.scope(r.validation.committed(RESULT))[KEY] == compact(row), 'commit exact helper plan')
    require(controller(row['source_commit']) == row['controller_inputs'] and
        all(digest(p) == sha for p, sha in row['native_inputs'].items()), 'build inputs changed')
    require(all(identity(row[name]) == row[name + '_sha256'] for name in
        ('expected_header_paths', 'expected_link_paths', 'environment')), 'frozen recipe component changed')
    require(digest(row['native_manifest']['path']) == row['native_manifest']['sha256'] and
        json.loads(Path(row['native_manifest']['path']).read_text()) == row['native_inputs'], 'native manifest changed')
    require(digest(r.RESULT) == row['runtime_sha256'] and
        r.current(r.validation.committed(r.RESULT))['binary'] == row['selected_binary'], 'selected runtime changed')


def build(row):
    current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use helper build')
    initial = copy.deepcopy(row); owner = None
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True,
        timeout=15).splitlines() if v.startswith('-- cursor: '))
    def persist():
        write(ROOT / 'execution.json', dict(commands=owner.commands if owner else [],
            active=bool(owner and owner.active)))
    try:
        owner = r.supervision.Owner(ROOT / 'owned', persist)
        entry, _, _ = owner.run('compile', row['compile_argv'], row['environment'], P, **row['budgets'])
        require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'helper compilation failed')
        actual = dependencies((ROOT / 'actual-deps.d').read_text())
        require(actual == set(row['expected_header_paths']), 'actual compile closure differs')
        row['object'] = dict(path=str(ROOT / 'control.o'), sha256=digest(ROOT / 'control.o'))
        entry, out, _ = owner.run('link', row['link_argv'], row['environment'], P, **row['budgets'])
        require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'helper linking failed')
        require(digest(row['object']['path']) == row['object']['sha256'], 'compiled object changed during link')
        linked = {str(Path(v).resolve()) for v in out.read_text().splitlines() if v.startswith('/')}
        require(linked <= set(row['expected_link_paths']) | {row['object']['path']} and
            {row['object']['path'], str((OFFICIAL / 'lib/libur_loader.so').resolve())} <= linked,
            'uncovered/missing actual linker inputs: ' + repr(linked - set(row['expected_link_paths']) - {row['object']['path']}))
        write(ROOT / 'actual-link-inputs.json', {path: digest(path) for path in sorted(linked)})
        row['actual_compiled_headers'] = len(actual)
        row['actual_link_manifest'] = dict(path=str(ROOT / 'actual-link-inputs.json'),
            sha256=digest(ROOT / 'actual-link-inputs.json'), files=len(linked), uncovered=0)
        artifact = ROOT / 'libstrata_ur_hold_control.so'
        row['library'] = dict(path=str(artifact), sha256=digest(artifact), bytes=artifact.stat().st_size)
        current(initial); row.update(complete=True, host_probe_accepted=True)
    except BaseException as error: row['error'] = repr(error)
    finally:
        if owner: r.validation.finish(row, owner, cursor, row['boot'])
        row.update(stage='built' if row.get('passed') else 'failed', active=False)
        row['commands'] = owner.commands if owner else []; persist(); save(row)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'complete', 'ownership_closed', 'error')}), flush=True)
    return int(not row['passed'])


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); r.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare(); return 0
        if sys.argv[1:] == ['build']: return build(json.loads((ROOT / 'record.json').read_text()))
        raise RuntimeError('use prepare or build')


if __name__ == '__main__': sys.exit(main())
