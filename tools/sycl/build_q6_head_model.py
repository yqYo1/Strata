#!/usr/bin/env python3
"""Build the sole-packed Q6 head with production objects; Main owns every command.

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
ROOT = B / 'q6-head-packed-primary-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'q6_head_packed_primary_build_v2'
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
    data=public();value=data['q6_dpas_component_v7']
    require(value['passed'] and value['complete'] and value['ownership_closed']
        and value['stage'] == 'closed'
        and value['decision']['head_component_gain']
        and not value['decision']['cells']['640']['component_gain_qualified']
        and not value['decision']['inference_adopted'], 'owned packed C4 full-head component gain required')
    model=data['q6_head_model_entry_v4'];previous=data['q6_head_model_build_v4']
    require(model['passed'] and model['complete'] and model['ownership_closed'] and
        model['fresh_faults']==[] and previous['passed'] and previous['complete'] and
        previous['ownership_closed'] and previous['binaries']==model['native_build']['binaries'] and
        all(digest(v['path'])==v['sha256'] for v in previous['binaries'].values()) and
        digest(model['native_record']['path'])==model['native_record']['sha256'],
        'preserve the qualified5a6 engine and all four fixture/recipe identities')
    original=json.loads(Path(model['native_record']['path']).read_text())['project']
    require(original['source_sha256']=={n:hashlib.sha256(raw).hexdigest() for n,raw in
        source.blobs(P,original['revision'],original['source_sha256'])},'original fallback source not recoverable')
    name = 'tools/sycl/q6_head_owner.py'
    before = ast.parse(source.git(P, 'show', READER + ':' + name))
    after = ast.parse((P / name).read_text())
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False)
                             for n in tree.body if isinstance(n, ast.FunctionDef)}
    a, b = functions(before), functions(after)
    require(set(b) == {'parse_native', 'finish_gpu_probe'} and
        a['finish_gpu_probe'] == b['finish_gpu_probe'], 'retained fault finalizer changed')
    return dict(component_evidence='q6_dpas_component_v7',component_decision=value['decision'],
        previous_native_source=original['revision'],previous_model_binaries=previous['binaries'],
        previous_model_key='q6_head_model_entry_v4',fallback_source_recoverable=True,
        previous_native_record=model['native_record'],
        reader_source=READER, fault_finalizer_AST_identical=True,
        native_kernel_or_GPU_result_inherited=False,production_Verifier_qualified=False, model_qualified=False)


def current(row):
    require(public()[KEY] == compact(row) and
        json.loads(RESULT.read_text())[KEY] == compact(row), 'commit the exact build stage first')
    require(prerequisites() == row['prerequisites'], 'head prerequisite changed')
    _, identity = selected.basis()
    require(signature(identity) == row['selected_native_sha256'], 'selected feature basis changed')
    require(row['native_source_commit']==row['project']['revision'],'new native source identity changed')
    check=dict(row)
    if 'admission_project' in row:
        require(row['admission_source_commit']==row['admission_project']['revision'],'controller admission changed')
        check['project']=row['admission_project']
        actual=row['actual_inputs'];require(digest(actual['path'])==actual['sha256'],'original census changed')
        census=json.loads(Path(actual['path']).read_text())
        require(all(digest(p)==h for field in ('compiled','generated','links') for p,h in census[field].items()),
                'original actual compiler/fixture/link inputs changed')
        require(all(digest(v['path'])==v['sha256'] for v in row['binaries'].values()),'original four ELFs changed')
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
    """Prospective opcode70 reader controls;keep the compiled V1 failure intact."""
    require(KEY not in public(),'fresh reader admission only')
    original=json.loads((ROOT/'record.json').read_text())
    require(public()['q6_head_packed_primary_build_v1']==compact(original) and
        original['stage']=='failed' and original['ownership_closed'] and not original['GPU_executed'] and
        original['error']=="RuntimeError('unreviewed SPIR-V opcode: 70')" and
        len(original['commands'])==5 and all(ownership.closed(c) and c['exit_code']==0 for c in original['commands']),
        'preserve the original compiled/native-unexecuted failure')
    project=source.pin(P,PREFIXES);before=original['project']['source_sha256'];after=project['source_sha256']
    require(before=={n:hashlib.sha256(raw).hexdigest() for n,raw in
        source.blobs(P,original['project']['revision'],before)},'original native source not recoverable')
    require(set(before)==set(after) and {n for n in before if before[n]!=after[n]}=={
        'tools/sycl/build_q6_head_model.py','tools/sycl/q6_dpas_component.py','tools/sycl/test_compile_census.py'},
        'only the declared reader/controller/CPU controls may change;native and fixtures stay exact')
    actual=original['actual_inputs'];require(digest(actual['path'])==actual['sha256'],'original actual-input census changed')
    census=json.loads(Path(actual['path']).read_text())
    require(all(digest(p)==h for field in ('compiled','generated','links') for p,h in census[field].items()) and
        all(digest(v['path'])==v['sha256'] for v in original['binaries'].values()),'original native/fixture bytes changed')
    row=copy.deepcopy(original);row.pop('error');row.pop('host_controls',None);row.pop('image_reader_refusal')
    row.update(stage='admitted',commands=[],passed=False,complete=False,active=False,ownership_closed=True,
        admission_project=project,admission_source_commit=project['revision'],compilation_passed=True,
        qualification_reuse=dict(original_failed_key='q6_head_packed_primary_build_v1',
            original_evidence_commit=source.git(P,'rev-parse','HEAD').decode().strip(),
            original_private_record_sha256=digest(ROOT/'record.json'),native_source_commit=original['project']['revision'],
            native_recompiled=False,GPU_qualification_inherited=False,
            scope='Same four ELFs,127TUs/1461compiler inputs/generated1/29external links and recipe.'
                  'Add only reviewed OpInBoundsPtrAccessChain70;preserve its opcode and every operand ID.'
                  ' Require40CPU controls and complete2443-instruction engine/fixture structural identity.'
                  ' Other unknown instructions and every prior bounds/ambiguity rejection stay refused.'))
    row['plan']['host']='40CPU controls,including inbounds-pointer ID/order positives and opcode/element-edge/arity/undefined-ID negatives;then exact selected-module identity. No GPU result inherited.'
    save(row);print('Admitted exact compiled artifacts for new40CPU reader controls;commit first.',flush=True)


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
    row = dict(stage='prepared', project=project,native_source_commit=project['revision'],
        dependencies=copy.deepcopy(native['dependencies']), recipe=recipe,
        toolchain=native['toolchain'], external_inputs=native['external_inputs'],
        fallbacks={n: native['fallbacks'][n] for n in ('prefill', 'decode')} |
                  dict(selected_feature=identity['engine']),
        selected_native_sha256=signature(identity), prerequisites=prerequisites(),
        host_python=HOST_PYTHON, host_python_sha256=digest(HOST_PYTHON),
        budgets=BUDGET, commands=[], passed=False, complete=False, ownership_closed=True,
        GPU_executed=False, model_qualified=False, production_Verifier_qualified=False,
        plan=dict(host='36 linked-image,census,owner/protocol and actual finalizer controls;'
                       'retained fault-finalizer AST must match484d before running',
            configure='Commit configure recipe first; record generated inputs and selected Ninja commands;'
                      'commit that configured stage before compiling.',
            compile='Build the engine plus production-linked head, submission and verifier-retirement fixtures.'
                    'Verify all actual compiler/link inputs and fresh engine/fixture selected-module identity'
                    ' before any GPU execution;C1..3 and the shared-MTP path inherit no GPU result.',
            subsequent='Fresh CPU guards and bounded traced submission/phase/real-GGUF0/1/2owner checks,'
                       'then three timing processes R/C,C/R,R/C with5samples/arm and256replays/sample.'
                       'Native fixtures enforce max(10ms,300timer-units). Only a repeated C4 gain plus'
                       ' all numerical/lifetime gates permits same-engine0/2 MTP4 model handoffs and'
                       ' matched bench.sh long/agent/strong-and-weak thinking speed. No adoption from build.'),
        retention='One temporary candidate for sole-packed0/2 qualification. Preserve qualified5a6 engine'
                  ' and its four fixture/recipe identities,90f feature and552b/e1b independent phase guards.'
                  ' The prior0b comparison consumer ended at V4 model closure;check and retire its binaries'
                  ' from exact original Git inputs before the new compile. Select the retained generation'
                  ' after fresh qualification;retire a rejected candidate after original source recovery.'
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
    require(row['stage'] == {'configure': 'prepared', 'compile': 'configured','controls':'admitted'}[stage],
            'wrong build stage')
    guards.no_model_owners([])
    frozen = copy.deepcopy(row); owner = None; outputs = {}
    def persist():
        write(ROOT / 'execution.json', dict(active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    try:
        owner = ownership.Owner(ROOT / ('owned-' + stage), persist)
        host = ('host-controls', [HOST_PYTHON, '-B', '-m', 'unittest', '-v',
                                       'tools.sycl.test_q6_dpas_component', 'tools.sycl.test_compile_census',
                                       'tools.sycl.test_q6_head_generation_speed'])
        commands = ([host] if stage=='controls' else [host,
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
                count = 40 if stage=='controls' else 36
                require('Ran ' + str(count) + ' tests' in text and '\nOK\n' in text, 'all CPU controls required')
                row['host_controls'] = dict(passed=True, cases=count, fault_finalizer_AST_identical=True,
                    GPU_executed=False, source_commit=row.get('admission_source_commit', row['project']['revision']))
            print(json.dumps(dict(label=label, exit_code=entry['exit_code'], closed=True)), flush=True)
        if stage=='controls':
            actual={n:structure(row['binaries'][n],row['codegen'][n]) for n in row['codegen']}
            require(actual['strata']==actual['sycl_q6_head_owner_test'],'fresh selected-module instructions differ')
            row['linked_structure']=dict(production=actual,scope='Exact selected-module identity under only ID/declaration order;not final ISA,GPU behavior or speed.')
            row.update(stage='built',passed=True,complete=True)
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
            actual={n:structure(row['binaries'][n],row['codegen'][n]) for n in row['codegen']}
            require(actual['strata']==actual['sycl_q6_head_owner_test'],
                    'fresh production/fixture selected module differs')
            row['linked_structure']=dict(production=actual,
                scope='Fresh selected-module identity between the engine and fixture,not old kernel/GPU qualification or final ISA.')
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
    require(sys.argv[1:] in (['prepare'], ['configure'], ['compile'],['admit'],['controls']),
            'build_q6_head_model.py prepare|configure|compile|admit|controls')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1]=='admit':admit()
        else: execute(sys.argv[1])
