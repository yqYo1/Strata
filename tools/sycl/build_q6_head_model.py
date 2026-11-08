#!/usr/bin/env python3
"""Build the opt-in Q6 head with production objects; Main owns every command.

Reuse the selected feature recipe and current committed primary sources. Keep
the independent prefill/decode references. This builds, never runs, GPU code.
Main-authored; repository LICENSE applies.
"""
import ast
import copy
import fcntl
import json
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode = True
import build_engine as source
import build_upstream_features as build
import build_compat_profile as selected
import measurement_guards as guards
import measurement_owner as ownership
import q6_dpas_component as codegen

P, B = source.P, source.B
ROOT = B / 'q6-head-model-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'q6_head_model_build_v1'
NATIVE = 'afbd954c0f5abad0a1488edbb791ed4c62833d78'
READER = '484d403e60c49361422f00a16dfdd774e4fe90d5'
TARGETS = ['strata', 'sycl_q6_head_owner_test', 'sycl_submission_test',
           'sycl_verifier_retirement_test']
PREFIXES = ['CMakeLists.txt', 'cmake', 'src', 'include', 'third_party',
            'tests', 'bench/sycl', 'tools/sycl']
HOST_PYTHON = '/home/yayoi/.local/state/strata-sycl/upstream-host-tests-146df3c4/bin/python'
BUDGET = dict(wall=2400, cpu=7200, rss_cap=12 << 30, text_cap=48 << 20,
              file_cap=512 << 20, total_cap=4 << 30)
require, digest, write, signature = guards.require, source.digest, build.write, build.signature


def public():
    return json.loads(source.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))))


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('project', 'dependencies', 'commands')}
    out['project'] = {k: v for k, v in row['project'].items() if k != 'source_sha256'}
    out['dependencies'] = [{k: v for k, v in d.items() if k != 'source_sha256'}
                           for d in row['dependencies']]
    out['commands'] = [dict(label=c['label'], exit_code=c.get('exit_code'),
        ownership_closed=ownership.closed(c), error=c.get('error'), errors=c.get('errors', []))
        for c in row['commands']]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def prerequisites():
    value = public()['q6_head_owner_v2']
    require(value['passed'] and value['complete'] and value['ownership_closed']
        and value['stage'] == 'closed' and value['source_commit'] == NATIVE,
        'real-head owner qualification is required')
    for name in ('src/core/native_head.cpp', 'include/strata/core/native_head.hpp',
                 'include/strata/core/verify_window_policy.hpp',
                 'src/kernels/sycl/q6_dpas_component.cpp',
                 'src/kernels/sycl/q6_dpas_component.hpp', 'tests/sycl/native_head_dpas.cpp'):
        require((P / name).read_bytes() == source.git(P, 'show', NATIVE + ':' + name),
                'qualified native input changed: ' + name)
    name = 'tools/sycl/q6_head_owner.py'
    before = ast.parse(source.git(P, 'show', READER + ':' + name))
    after = ast.parse((P / name).read_text())
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False)
                             for n in tree.body if isinstance(n, ast.FunctionDef)}
    a, b = functions(before), functions(after)
    require(set(b) == {'parse_native', 'finish_gpu_probe'} and
        all(a[n] == b[n] for n in b), 'retained reader bodies changed')
    return dict(native_source=NATIVE, reader_source=READER,
        evidence='0840eccd2398fdcc43cb445edaf49086d47f16dd:q6_head_owner_v2',
        reader_bodies_identical=True, production_Verifier_qualified=False,
        model_qualified=False)


def current(row):
    require(public()[KEY] == compact(row) and
        json.loads(RESULT.read_text())[KEY] == compact(row), 'commit the exact build stage first')
    require(prerequisites() == row['prerequisites'], 'head prerequisite changed')
    _, identity = selected.basis()
    require(signature(identity) == row['selected_native_sha256'], 'selected feature basis changed')
    build.check_inputs(row)
    require(digest(HOST_PYTHON) == row['host_python_sha256'], 'host interpreter changed')
    for path, sha in (row.get('generated_inputs', {}) | row.get('configured_files', {})).items():
        require(digest(path) == sha, 'configured generated input changed: ' + path)


def prepare():
    require(not ROOT.exists() and KEY not in public(), 'fresh build only; never retry a failed plan')
    guards.no_model_owners([])
    native, identity = selected.basis()
    recipe = copy.deepcopy(native['recipe'])
    recipe['configure'] = [v.replace(native['project']['root'], str(P)).replace(
        str(selected.feature.NATIVE), str(ROOT)) for v in recipe['configure']]
    recipe['environment'] = {k: v.replace(str(selected.feature.NATIVE), str(ROOT))
                             for k, v in recipe['environment'].items()}
    recipe['targets'] = recipe['build_targets'] = TARGETS
    recipe.pop('host_modes', None)
    recipe['environment']['STRATA_SYCL_DECODE_Q6_DPAS_HEAD'] = '0'
    require(recipe['configure'][recipe['configure'].index('-S') + 1] == str(P) and
        recipe['configure'][recipe['configure'].index('-B') + 1] == str(ROOT / 'build'),
        'primary source and private build directory required')
    row = dict(stage='prepared', project=source.pin(P, PREFIXES),
        dependencies=copy.deepcopy(native['dependencies']), recipe=recipe,
        toolchain=native['toolchain'], external_inputs=native['external_inputs'],
        fallbacks={n: native['fallbacks'][n] for n in ('prefill', 'decode')} |
                  dict(selected_feature=identity['engine']),
        selected_native_sha256=signature(identity), prerequisites=prerequisites(),
        host_python=HOST_PYTHON, host_python_sha256=digest(HOST_PYTHON),
        budgets=BUDGET, commands=[], passed=False, complete=False, ownership_closed=True,
        GPU_executed=False, model_qualified=False, production_Verifier_qualified=False,
        plan=dict(host='14 existing linked-image, native protocol and actual finalizer controls;'
                       'retained reader ASTs must match484d before running',
            configure='Commit configure recipe first; record generated inputs and selected Ninja commands;'
                      'commit that configured stage before compiling.',
            compile='Build the engine plus production-linked head, submission and verifier-retirement fixtures.'
                    'Verify actual compiler/link inputs and packaged C4 code before any GPU execution.',
            subsequent='Separate committed small traced/validated qualification, then matched head0/1 model'
                       'routes and long/agent/thinking performance. No performance adoption from this build.'),
        retention='One temporary fourth engine for imminent production-head admission and same-binary0/1'
                  ' comparison. Preserve90f feature and552b/e1b independent phase guards. Select and retire'
                  ' the superseded feature generation after applicable model/phase validation; retire this'
                  ' candidate on rejection after exact source/recipe recovery. Standalone owner consumer ended.'
                  ' Legacy56ELF provenance holds remain unresolved and are not qualified references.')
    build.check_inputs(row)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir(mode=0o700)
    save(row)
    print('Prepared production-head model build; commit before host/configure.', flush=True)


def census(row, dependency_output, link_output):
    allowed = {}
    for pin in (row['project'], *row['dependencies']):
        allowed.update({str((Path(pin['root']) / p).resolve()): h
                        for p, h in pin['source_sha256'].items()})
    allowed.update({str(Path(p).resolve()): h for p, h in
        json.loads(Path(row['external_inputs']['path']).read_text()).items()})
    allowed.update({str(Path(p).resolve()): h for p, h in row['toolchain'].items()})
    allowed.update(row['generated_inputs'])
    compiled, generated = {}, {}
    for line in dependency_output.read_text().splitlines():
        if not line.startswith('    ') or not line.strip(): continue
        path = Path(line.strip())
        path = (path if path.is_absolute() else ROOT / 'build' / path).resolve()
        require(path.is_file() and allowed.get(str(path)) == digest(path),
                'unfrozen actual compiler input: ' + str(path))
        (generated if path.is_relative_to(ROOT / 'build') else compiled)[str(path)] = digest(path)
    expected = set(row['planned_translation_units'])
    observed = {p for p in compiled if Path(p).suffix in ('.cpp', '.c', '.cc', '.cxx', '.cu')}
    require(expected == observed and len(expected) > 20, 'actual selected translation-unit census differs')
    database = json.loads((ROOT / 'build/compile_commands.json').read_text())
    commands = [c for c in database if str(Path(c['file']).resolve()) in expected]
    require({str(Path(c['file']).resolve()) for c in commands} == expected,
            'compile database is missing an actual selected translation unit')
    links, owned = {}, set()
    for line in link_output.read_text(errors='replace').splitlines():
        name = line.strip(); path = Path(name)
        if not name or any(c.isspace() for c in name) or not (
            path.is_absolute() or path.suffix in ('.o', '.a') or '.so' in path.name): continue
        path = (path if path.is_absolute() else ROOT / 'build' / path).resolve()
        if any(path.is_relative_to(ROOT / n) for n in ('build', 'tmp')):
            owned.add(str(path)); continue
        require(path.is_file() and allowed.get(str(path)) == digest(path),
                'unfrozen actual linker input: ' + str(path))
        links[str(path)] = digest(path)
    require(len(links) > 15, 'external link census incomplete')
    record = dict(compiled=compiled, generated=generated, links=links,
                  owned_link_products=sorted(owned), compile_commands=commands)
    path = ROOT / 'actual-inputs.json'; write(path, record)
    return dict(path=str(path), sha256=digest(path), compiler_files=len(compiled),
        generated_files=len(generated), external_link_files=len(links), uncovered=0,
        translation_units=len(expected), source_commit=row['project']['revision'])


def execute(stage):
    row = json.loads((ROOT / 'record.json').read_text()); current(row)
    require(row['stage'] == ('prepared' if stage == 'configure' else 'configured'), 'wrong build stage')
    guards.no_model_owners([])
    frozen = copy.deepcopy(row); owner = None; outputs = {}
    def persist():
        write(ROOT / 'execution.json', dict(active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    try:
        owner = ownership.Owner(ROOT / ('owned-' + stage), persist)
        commands = ([('host-controls', [HOST_PYTHON, '-B', '-m', 'unittest', '-v',
                                       'tools.sycl.test_q6_dpas_component']),
                     ('configure', row['recipe']['configure']),
                     ('planned-commands', ['/usr/bin/ninja', '-C', str(ROOT / 'build'),
                                          '-t', 'commands', *TARGETS])]
            if stage == 'configure' else
            [('compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target',
                          *TARGETS, '-j', '4']),
             ('dependencies', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'])])
        for label, command in commands:
            current(frozen)
            entry, out, err = owner.run(label, command, row['recipe']['environment'], P, **BUDGET)
            outputs[label] = out
            require(ownership.closed(entry) and entry['exit_code'] == 0,
                    'build command failed: ' + label + '; inspect the owned bounded diagnostic')
            if label == 'host-controls':
                text = out.read_text() + err.read_text()
                require('Ran 14 tests' in text and '\nOK\n' in text, 'all14existing controls required')
                row['host_controls'] = dict(passed=True, cases=14, reader_ASTs_identical=True,
                    GPU_executed=False, source_commit=row['project']['revision'])
            print(json.dumps(dict(label=label, exit_code=entry['exit_code'], closed=True)), flush=True)
        if stage == 'configure':
            planned = set()
            for line in outputs['planned-commands'].read_text().splitlines():
                args = shlex.split(line)
                if '-c' in args:
                    p = Path(args[args.index('-c') + 1])
                    planned.add(str((p if p.is_absolute() else ROOT / 'build' / p).resolve()))
            require(len(planned) > 20, 'selected target compiler commands absent')
            row['planned_translation_units'] = sorted(planned)
            row['generated_inputs'] = {str(p.resolve()): digest(p)
                for p in (ROOT / 'build').rglob('*') if p.is_file() and
                p.suffix in ('.h', '.hpp', '.c', '.cpp', '.cxx', '.inc')}
            row['configured_commands_sha256'] = digest(outputs['planned-commands'])
            row['configured_files'] = {str(p.resolve()): digest(p)
                for p in [ROOT / 'build/build.ninja', ROOT / 'build/CMakeCache.txt',
                          ROOT / 'build/compile_commands.json', outputs['planned-commands'],
                          *sorted((ROOT / 'build/CMakeFiles').glob('*.ninja'))]}
            row['stage'] = 'configured'
        else:
            row['actual_inputs'] = census(row, outputs['dependencies'], outputs['compile'])
            row['binaries'] = {n: dict(path=str(ROOT / 'build' / n),
                sha256=digest(ROOT / 'build' / n), bytes=(ROOT / 'build' / n).stat().st_size,
                origin_source_commit=row['project']['revision']) for n in TARGETS}
            row['codegen'] = {n: codegen.embedded_dpas((ROOT / 'build' / n).read_bytes())
                for n in ('strata', 'sycl_q6_head_owner_test')}
            require(all(v['image_sha256'] ==
                'e9f6543e839747035fab433ec3cc6f773ddd69cb9eab36f89b6493b54cb46d8d'
                for v in row['codegen'].values()), 'production-linked C4 image changed')
            row.update(stage='built', passed=True, complete=True)
        current(frozen)
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        row['commands'] += owner.commands if owner else []
        row['active'] = bool(owner and owner.active)
        row['ownership_closed'] = not row['active'] and all(ownership.closed(c) for c in row['commands'])
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        save(row)
    require(row['stage'] != 'failed' and row['ownership_closed'], row.get('error', 'build did not close'))
    print('Closed ' + stage + '; commit its exact result before the next stage.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['configure'], ['compile']),
            'build_q6_head_model.py prepare|configure|compile')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
