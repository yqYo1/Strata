#!/usr/bin/env python3
"""Build the opt-in Q6 head with production objects; Main owns every command.

Reuse the selected feature recipe and current committed primary sources. Keep
the independent prefill/decode references. This builds, never runs, GPU code.
Main-authored; repository LICENSE applies.
"""
import ast
import copy
import fcntl
import hashlib
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
import q6_dpas_component as spirv
from q6_canonical_component import image as canonical_image

P, B = source.P, source.B
ROOT = B / 'q6-head-model-v2'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'q6_head_model_build_v4'
NATIVE = 'cdbd687965b9b1dd96e7a76002381cc257019da5'
COMPONENT_EVIDENCE = '0ccc4ed45cbf219e835a9c6b03b08951fa321ab8'
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
    out = {k: v for k, v in row.items() if k not in ('project', 'admission_project', 'dependencies', 'commands')}
    out['project'] = {k: v for k, v in row['project'].items() if k != 'source_sha256'}
    if 'admission_project' in row:
        out['admission_project'] = {k: v for k, v in row['admission_project'].items() if k != 'source_sha256'}
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
    value = public()['q6_canonical_component_v4']
    require(value['passed'] and value['complete'] and value['ownership_closed']
        and value['stage'] == 'closed' and value['native_source_commit'] == NATIVE
        and value['decision']['head_component_gain']
        and not value['decision']['cells']['640']['component_gain_qualified']
        and not value['decision']['inference_adopted'], 'closed canonical-only full-head component required')
    evidence = json.loads(source.git(P, 'show', COMPONENT_EVIDENCE + ':' + str(RESULT.relative_to(P))))[
        'q6_canonical_component_v4']
    require(all(value[k] == evidence[k] for k in ('decision', 'binary', 'codegen', 'summary', 'actual_inputs')),
            'canonical component evidence changed')
    for name in ('src/kernels/sycl/q6_dpas_component.cpp',
                 'src/kernels/sycl/q6_dpas_component.hpp', 'bench/sycl/q6_canonical.cpp',
                 'cmake/sycl_backend.cmake'):
        require((P / name).read_bytes() == source.git(P, 'show', NATIVE + ':' + name),
                'qualified component input changed: ' + name)
    require(digest(value['binary']['path']) == value['binary']['sha256'] and
        canonical_image(Path(value['binary']['path']).read_bytes()) == value['codegen'],
        'qualified canonical component executable changed')
    inputs = value['actual_inputs']
    require(digest(inputs['path']) == inputs['sha256'], 'canonical compiler census changed')
    census = json.loads(Path(inputs['path']).read_text())
    require(all(digest(p) == sha for field in ('compiled', 'generated', 'links')
        for p, sha in census[field].items()), 'qualified component compiler/link input changed')
    name = 'tools/sycl/q6_head_owner.py'
    before = ast.parse(source.git(P, 'show', READER + ':' + name))
    after = ast.parse((P / name).read_text())
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False)
                             for n in tree.body if isinstance(n, ast.FunctionDef)}
    a, b = functions(before), functions(after)
    require(set(b) == {'parse_native', 'finish_gpu_probe'} and
        a['finish_gpu_probe'] == b['finish_gpu_probe'], 'retained fault finalizer changed')
    previous = public()['q6_head_model_build_v1']
    require(previous['passed'] and previous['complete'] and previous['ownership_closed'] and
        all(digest(v['path']) == v['sha256'] for v in previous['binaries'].values()),
        'preserve original0b engine and same-build fixtures before building replacement')
    return dict(component_native_source=NATIVE, evidence=COMPONENT_EVIDENCE + ':q6_canonical_component_v4',
        canonical_image_sha256=value['codegen']['image_sha256'],
        component_binary=value['binary'], previous_model_binaries=previous['binaries'],
        reader_source=READER, fault_finalizer_AST_identical=True,
        production_Verifier_qualified=False, model_qualified=False)


def current(row):
    require(public()[KEY] == compact(row) and
        json.loads(RESULT.read_text())[KEY] == compact(row), 'commit the exact build stage first')
    require(prerequisites() == row['prerequisites'], 'head prerequisite changed')
    _, identity = selected.basis()
    require(signature(identity) == row['selected_native_sha256'], 'selected feature basis changed')
    check = dict(row)
    if 'admission_project' in row:
        require(row['native_source_commit'] == row['project']['revision'] and
            row['admission_source_commit'] == row['admission_project']['revision'], 'native/admission identity changed')
        check['project'] = row['admission_project']
        inputs = row['actual_inputs']
        require(digest(inputs['path']) == inputs['sha256'], 'original actual-input inventory changed')
        census = json.loads(Path(inputs['path']).read_text())
        require(all(digest(p) == sha for field in ('compiled', 'generated', 'links')
            for p, sha in census[field].items()), 'original actual compiler/link input changed')
    build.check_inputs(check)
    require(digest(HOST_PYTHON) == row['host_python_sha256'], 'host interpreter changed')
    for path, sha in (row.get('generated_inputs', {}) | row.get('configured_files', {})).items():
        require(digest(path) == sha, 'configured generated input changed: ' + path)


def structure(binary, image):
    blob = Path(binary['path']).read_bytes()
    require(hashlib.sha256(blob).hexdigest() == binary['sha256'] and
        canonical_image(blob) == image, 'actual linked image/executable changed')
    raw = blob[image['image_offset']:image['image_offset'] + image['image_bytes']]
    require(hashlib.sha256(raw).hexdigest() == image['image_sha256'], 'actual embedded SPIR-V changed')
    return spirv.spirv_structure(raw)


def admit():
    """New controller-only plan; preserve the failed exact-byte V3 outcome."""
    require(KEY not in public(), 'fresh prospective admission only')
    original = json.loads((ROOT / 'record.json').read_text())
    prior = public()['q6_head_model_build_v3']
    require({k: v for k, v in prior.items() if k != 'image_identity_refusal'} == compact(original) and
        original['stage'] == 'failed' and original['ownership_closed'] and not original['GPU_executed'] and
        original['error'] == "RuntimeError('production-linked canonical C4 image changed')" and
        len(original['commands']) == 5 and all(ownership.closed(c) and c['exit_code'] == 0
        for c in original['commands']), 'original compiled/native-refused V3 must stay failed')
    require(all(digest(v['path']) == v['sha256'] for v in original['binaries'].values()),
            'preserve exact original four ELFs;no rebuild')
    project = source.pin(P, PREFIXES); before = original['project']['source_sha256']; after = project['source_sha256']
    require(before == {n: hashlib.sha256(raw).hexdigest() for n, raw in
        source.blobs(P, original['project']['revision'], before)}, 'original native source not recoverable')
    changed = {n for n in before if after.get(n) != before[n]}
    require(set(before) == set(after) and changed == {'tools/sycl/build_q6_head_model.py',
        'tools/sycl/q6_dpas_component.py', 'tools/sycl/test_compile_census.py'},
        'only declared reader/controller/tests may change;all native/recipe inputs must match')
    inputs = original['actual_inputs']; require(digest(inputs['path']) == inputs['sha256'], 'original census changed')
    census = json.loads(Path(inputs['path']).read_text())
    require(all(digest(p) == sha for field in ('compiled', 'generated', 'links')
        for p, sha in census[field].items()), 'original actual compiler/link inputs changed')
    row = copy.deepcopy(original); row.pop('error'); row.pop('host_controls', None)
    row.update(stage='admitted', commands=[], passed=False, complete=False, active=False, ownership_closed=True,
        admission_project=project, admission_source_commit=project['revision'],
        native_source_commit=original['project']['revision'], compilation_passed=True,
        qualification_reuse=dict(original_failed_key='q6_head_model_build_v3',
            original_evidence_commit=source.git(P, 'rev-parse', 'HEAD').decode().strip(),
            original_private_record_sha256=digest(ROOT / 'record.json'),
            native_source_commit=original['project']['revision'], changed_controller_files=sorted(changed),
            native_recompiled=False, GPU_qualification_inherited=False,
            scope='Same127TUs/1461compiler inputs/generated1/29external links and four exact ELFs.'
                  'New36CPU controls must prove exact selected-module instructions/literals/annotations'
                  ' and function-body order after only ID/global/function declaration reordering.'
                  'Unknown instructions/decorations/modes and ambiguous identities refuse admission.'
                  'Fresh production GPU/real-model checks and matched speed still required.'))
    row['plan'] = dict(host='36CPU controls including ID/declaration reorder positives and literal/annotation/'
        'operand-edge/body-order/unknown-op/ambiguity/truncation negatives;then compare all1187instructions'
        ' in the actual standalone,production head fixture and engine C4 images.',
        subsequent='Only structural identity and all controls permit fresh traced production-linked GPU'
        ' owner checks,then bounded128K model handoffs. No GPU result or model speed inherited.')
    save(row); print('Admitted exact native artifacts for new structural reader controls;commit first.', flush=True)


def planned_compilation(text):
    """Read actual Ninja outputs, including dependencies in nested directories."""
    units = []
    for line in text.splitlines():
        args = shlex.split(line)
        if '-c' not in args: continue
        require(args.count('-c') == args.count('-o') == 1, 'one compile source/object required')
        p = Path(args[args.index('-c') + 1])
        name = str((p if p.is_absolute() else ROOT / 'build' / p).resolve())
        output = Path(args[args.index('-o') + 1])
        absolute = (output if output.is_absolute() else ROOT / 'build' / output).resolve()
        require(absolute.is_relative_to(ROOT / 'build'), 'compile output escaped owned build')
        relative = absolute.relative_to(ROOT / 'build')
        parts = relative.parts
        require(parts.count('CMakeFiles') == 1, 'selected target compile output required')
        at = parts.index('CMakeFiles') + 1
        require(at + 1 < len(parts) and parts[at].endswith('.dir') and len(parts[at]) > 4,
                'selected target compile output required')
        units.append(dict(file=name, output=str(relative), target=parts[at][:-4]))
    require(len(units) == len({v['file'] for v in units}),
            'duplicate selected source compilation requires explicit admission')
    return units


def prepare():
    require(KEY not in public(), 'fresh prospective build plan only')
    guards.no_model_owners([])
    project = source.pin(P, PREFIXES)
    previous = None
    if ROOT.exists():
        original = json.loads((ROOT / 'record.json').read_text())
        require(public()['q6_head_model_build_v2'] == compact(original) and
            original['stage'] == 'failed' and original['ownership_closed'] and
            not original['GPU_executed'] and original['error'] ==
            "RuntimeError('selected target compile output required')" and
            len(original['commands']) == 3 and all(ownership.closed(c) and c['exit_code'] == 0
            for c in original['commands']) and not any((ROOT / 'build' / n).exists() for n in TARGETS),
            'only the closed failed V2 configure tree may be reused;never retry its plan')
        pin = original['project']; before, after = pin['source_sha256'], project['source_sha256']
        require(before ==
            {n: hashlib.sha256(raw).hexdigest() for n, raw in
                source.blobs(P, pin['revision'], before)}, 'original failed configure source not recoverable')
        allowed = {'tools/sycl/build_q6_head_model.py', 'tools/sycl/test_compile_census.py'}
        added = {'tools/sycl/q6_head_generation_speed.py', 'tools/sycl/test_q6_head_generation_speed.py'}
        require({n for n in before if after.get(n) != before[n]} == allowed and
            set(after) - set(before) == added, 'only named controller/test updates;native inputs unchanged')
        previous = dict(key='q6_head_model_build_v2', evidence_commit=source.git(P, 'rev-parse', 'HEAD').decode().strip(),
            source_commit=pin['revision'], private_record_sha256=digest(ROOT / 'record.json'),
            source_files=pin['count'], source_sha256=pin['aggregate_sha256'],
            native_source_bytes_unchanged=True, reason='Prospective V3 fixes nested Ninja object parsing;'
            'reuses only configured build directory,not a native executable or failed result.')
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
    row = dict(stage='prepared', project=project,
        dependencies=copy.deepcopy(native['dependencies']), recipe=recipe,
        toolchain=native['toolchain'], external_inputs=native['external_inputs'],
        fallbacks={n: native['fallbacks'][n] for n in ('prefill', 'decode')} |
                  dict(selected_feature=identity['engine']),
        selected_native_sha256=signature(identity), prerequisites=prerequisites(),
        host_python=HOST_PYTHON, host_python_sha256=digest(HOST_PYTHON),
        budgets=BUDGET, commands=[], passed=False, complete=False, ownership_closed=True,
        GPU_executed=False, model_qualified=False, production_Verifier_qualified=False,
        plan=dict(host='28 linked-image,census,current canonical owner/long-cohort protocol and actual finalizer controls;'
                       'retained fault-finalizer AST must match484d before running',
            configure='Commit configure recipe first; record generated inputs and selected Ninja commands;'
                      'commit that configured stage before compiling.',
            compile='Build the engine plus production-linked head, submission and verifier-retirement fixtures.'
                    'Verify selected actual compiler/link inputs and exact qualified canonical C4 image'
                    ' before any GPU execution;source-only integration inherits no production GPU result.',
            subsequent='Separate committed small traced/validated qualification, then matched head0/1 model'
                       'routes and long/agent/thinking performance. No performance adoption from this build.'),
        retention='One temporary fifth engine for imminent canonical production-head admission and same-binary0/1'
                  ' comparison. Preserve0b model fallback,90f feature and552b/e1b independent phase guards.'
                  ' Select and retire0b after replacement owner/model/128K placement qualification; retire this'
                  ' candidate on rejection after exact source/recipe recovery. Standalone owner consumer ended.'
                  ' Legacy56ELF provenance holds remain unresolved and are not qualified references.')
    build.check_inputs(row)
    if previous: row['previous_failed_plan'] = previous
    ROOT.mkdir(mode=0o700, exist_ok=bool(previous))
    for name in ('tmp', 'cache'): (ROOT / name).mkdir(mode=0o700, exist_ok=bool(previous))
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
    commands = []
    groups = {}
    for item in row['planned_compilation']:
        groups.setdefault(item['target'], set()).add(item['file'])
    for target, files in groups.items():
        commands += selected.selected_compile_commands(database, files, target)
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
    require(row['stage'] == {'configure': 'prepared', 'compile': 'configured', 'controls': 'admitted'}[stage],
            'wrong build stage')
    guards.no_model_owners([])
    frozen = copy.deepcopy(row); owner = None; outputs = {}
    def persist():
        write(ROOT / 'execution.json', dict(active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    try:
        owner = ownership.Owner(ROOT / (('v4-owned-' if stage == 'controls' else 'v3-owned-') + stage), persist)
        host = ('host-controls', [HOST_PYTHON, '-B', '-m', 'unittest', '-v',
                                       'tools.sycl.test_q6_dpas_component', 'tools.sycl.test_compile_census',
                                       'tools.sycl.test_q6_head_generation_speed'])
        commands = ([host] if stage == 'controls' else [host,
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
                count = 36 if stage == 'controls' else 28
                require('Ran ' + str(count) + ' tests' in text and '\nOK\n' in text, 'all CPU controls required')
                row['host_controls'] = dict(passed=True, cases=count, fault_finalizer_AST_identical=True,
                    GPU_executed=False, source_commit=row.get('admission_source_commit', row['project']['revision']))
            print(json.dumps(dict(label=label, exit_code=entry['exit_code'], closed=True)), flush=True)
        if stage == 'controls':
            component = public()['q6_canonical_component_v4']
            reference = structure(component['binary'], component['codegen'])
            actual = {n: structure(row['binaries'][n], row['codegen'][n]) for n in row['codegen']}
            require(all(v == reference for v in actual.values()), 'linked C4 instructions/literals/annotations differ')
            row['linked_structure'] = dict(component=reference, production=actual,
                raw_image_sha256={n: v['image_sha256'] for n, v in row['codegen'].items()},
                scope='Exact selected module under ID/declaration reordering,not final GPU ISA or speed.')
            row.update(stage='built', passed=True, complete=True)
        elif stage == 'configure':
            compilation = planned_compilation(outputs['planned-commands'].read_text())
            planned = {v['file'] for v in compilation}
            require(len(planned) > 20, 'selected target compiler commands absent')
            row['planned_translation_units'] = sorted(planned)
            row['planned_compilation'] = compilation
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
            row['codegen'] = {n: canonical_image((ROOT / 'build' / n).read_bytes())
                for n in ('strata', 'sycl_q6_head_owner_test')}
            require(all(v['image_sha256'] ==
                row['prerequisites']['canonical_image_sha256']
                for v in row['codegen'].values()), 'production-linked canonical C4 image changed')
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
    require(sys.argv[1:] in (['prepare'], ['configure'], ['compile'], ['admit'], ['controls']),
            'build_q6_head_model.py prepare|configure|compile|admit|controls')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] == 'admit': admit()
        else: execute(sys.argv[1])
