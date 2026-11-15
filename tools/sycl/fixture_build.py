#!/usr/bin/env python3
"""Main-owned standalone SYCL fixture build/admission helpers; repository LICENSE.

Callers set their private module context's ROOT, KEY, TARGET and TUS before use.
The original mapped-row campaign is recoverable from Git, not a live consumer.
"""
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import build_engine as source
import build_upstream_features as build
import check_cache_loan_cli as probe
import engine_once
import measurement_guards as guards
import measurement_owner as ownership
import ur_validation as validation

P, B = source.P, source.B
ROOT, KEY, TARGET, TUS = None, None, None, set()
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
PREFIXES = ['CMakeLists.txt', 'cmake', 'src', 'include', 'third_party', 'tests',
            'bench/sycl', 'tools/sycl', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md']
require, sha, signature = guards.require, source.digest, build.signature


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def compact(row):
    value = {k: v for k, v in row.items()
             if k not in ('project', 'dependencies', 'runtime_files', 'commands', 'actual_inputs')}
    value['project'] = {k: v for k, v in row['project'].items() if k != 'source_sha256'}
    value['dependencies'] = [{k: v for k, v in p.items() if k != 'source_sha256'}
                             for p in row['dependencies']]
    value['runtime_files_sha256'] = signature(row['runtime_files'])
    if 'actual_inputs' in row:
        value['actual_inputs'] = {k: dict(files=len(v), sha256=signature(v)) if isinstance(v, dict) else v
                                 for k, v in row['actual_inputs'].items()}
    return value

def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)

def check(row, *, committed=True):
    if committed:
        require(validation.committed(RESULT, (KEY,)) == compact(row), 'commit exact experiment plan/result')
    build.check_inputs(row)
    runtime = row['runtime_manifest']
    require(sha(runtime['path']) == runtime['sha256'] and
            json.loads(Path(runtime['path']).read_text()) == row['runtime_files'] and
            all(sha(p) == h for p, h in row['runtime_files'].items()), 'runtime bytes changed')
    actual_probe, reuse = probe.current_probe()
    require(actual_probe == row['native_validation_probe'] and reuse == row['runtime_reuse'], 'probe changed')
    require(guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')
    if row.get('binary'):
        require(sha(row['binary']['path']) == row['binary']['sha256'], 'fixture changed')
        for group in ('compiled', 'generated', 'links', 'objects'):
            require(all(sha(p) == h for p, h in row['actual_inputs'][group].items()), 'actual input changed: ' + group)

def environment(row, label, traced=False):
    root = ROOT / label
    env = {k: v for k, v in row['recipe']['environment'].items()
           if k in probe.ENV_KEYS}
    env.update(TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
        CUDA_MODULE_LOADING='EAGER', STRATA_CPU_SERVICE='0',
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'),
        UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
        UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
    env['ZE_ENABLE_ALT_DRIVERS'] = row['driver_library']
    if traced:
        env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                   UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.trace'))
    return engine_once.native_environment(row, env)

def host_controls(row, owner):
    entry, out, err = owner.run('host', row['host_command'], row['recipe']['environment'], P,
        wall=60, cpu=60, rss_cap=1 << 30, text_cap=1 << 20, file_cap=2 << 20, total_cap=8 << 20)
    text = out.read_text() + err.read_text()
    require(ownership.closed(entry) and entry['exit_code'] == 0 and
            'Ran ' + str(row['host_tests']) + ' tests' in text and '\nOK\n' in text and
            'skipped' not in text, 'all host controls required')
    row['host_controls'] = dict(tests=row['host_tests'], skipped=0, passed=True, exit_code=0, ownership_closed=True)

def census(row, compile_output):
    root = ROOT / 'build'
    commands = json.loads((root / 'compile_commands.json').read_text())
    selected = [x for x in commands if TARGET + '.dir/' in x['output'] or
                'strata_sycl_runtime.dir/' in x['output']]
    require(len(selected) == 5 and {str(Path(x['file']).relative_to(P)) for x in selected} == TUS,
            'exact five freshly compiled translation units required')
    allowed = {}
    for pin in [row['project'], *row['dependencies']]:
        allowed.update({str((Path(pin['root']) / p).resolve()): h for p, h in pin['source_sha256'].items()})
    for meta in (row['external_inputs'],):
        allowed.update({str(Path(p).resolve()): h for p, h in json.loads(Path(meta['path']).read_text()).items()})
    compiled, generated, objects = {}, {}, {}
    for item in selected:
        obj = root / item['output']; objects[str(obj)] = sha(obj)
        args = shlex.split(item['command'])
        require('-DSTRATA_ENABLE_SYCL=1' in args and '-DSTRATA_CPU_SERVICE_PROFILE=1' in args,
                'consistent runtime class conditions required')
        text = subprocess.check_output(['/usr/bin/ninja', '-t', 'deps', item['output']], cwd=root, text=True, timeout=30)
        lines = text.splitlines()
        require(len(lines) > 8 and 'VALID' in lines[0], 'missing actual compiler dependency closure')
        for line in lines[1:]:
            if not line.startswith('    '): continue
            dep = Path(line.strip()); dep = dep if dep.is_absolute() else root / dep
            name = str(dep.resolve()); digest = sha(dep)
            if name in allowed:
                require(allowed[name] == digest, 'unadmitted compiler input: ' + name); compiled[name] = digest
            else:
                require(dep.resolve().is_relative_to(root) and dep.suffix in ('.h', '.hpp'), 'unknown external compiler input: ' + name)
                generated[name] = digest
    links = {}
    produced = {str(Path(p).resolve()): h for p, h in objects.items()}
    archive = root / 'libstrata_sycl_runtime.a'; produced[str(archive.resolve())] = sha(archive)
    for line in Path(compile_output).read_text().splitlines():
        p = Path(line.strip()); p = p if p.is_absolute() else root / p
        if not p.is_file(): continue
        name = str(p.resolve())
        if name in produced: require(sha(p) == produced[name], 'link object changed')
        elif name in allowed:
            require(sha(p) == allowed[name], 'external link input changed'); links[name] = allowed[name]
        elif line.startswith('/'):
            raise RuntimeError('unknown link input: ' + name)
    require(len(compiled) > 100 and len(links) > 8, 'incomplete compiler/link evidence')
    return dict(compiled=compiled, generated=generated, links=links, objects=objects,
                translation_units=5, reused_objects=0, uncovered=0)

def compile_fixture():
    row = json.loads((ROOT / 'record.json').read_text()); check(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'build-attempt.json').exists(), 'one build attempt')
    write(ROOT / 'build-attempt.json', dict(source_commit=row['project']['revision']))
    owner = ownership.Owner(ROOT / 'build-owned')
    try:
        host_controls(row, owner)
        for label, args in [('configure', row['recipe']['configure']),
                ('compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', TARGET, '-j', '4'])]:
            check(row, committed=False)
            entry, output, _ = owner.run(label, args, row['recipe']['environment'], P, **row['budget'])
            require(ownership.closed(entry) and entry['exit_code'] == 0, label + ' failed')
        row['actual_inputs'] = census(row, output)
        path = ROOT / 'build' / TARGET
        row['binary'] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
        row['native_source_commit'] = row['project']['revision']
        check(row, committed=False)
        row.update(stage='built', passed=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error)); raise
    finally:
        row['commands'] = owner.commands; row['active'] = owner.active is not None
        row['ownership_closed'] = owner.active is None and all(ownership.closed(c) for c in owner.commands)
        save(row)
    print('Fresh five-TU fixture built;no engine rebuilt or GPU executed.', flush=True)
