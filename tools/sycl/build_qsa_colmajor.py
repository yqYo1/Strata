#!/usr/bin/env python3
"""Build the isolated column-major comparison without rebuilding the engine."""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import qsa_colmajor_sources as proposal
import resource_engine as resource

P, B = gen.P, gen.B
ROOT = B / 'qsa-colmajor-v1'
RESULT = gen.A / 'qsa-colmajor-current.json'
FIXTURE = 'bench/sycl/qsa_colmajor_numerical.cpp'
NATIVE = ('src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
          'src/kernels/sycl/prompt_attention.cpp', 'src/kernels/sycl/qsa_bf16_prefix.cpp')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def inputs(revision):
    names = set((*NATIVE, FIXTURE, 'tests/sycl/qsa_bf16_prefix.cpp',
        'tests/sycl/qsa_bf16_prefix_layout_bench.cpp', 'tools/sycl/measurement_environment.json'))
    names.update(gen.git('ls-files', 'include').decode().splitlines())
    pending, seen = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        names.add(str(path.relative_to(P)))
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([v.name for v in node.names] if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) else [])
            for module in modules:
                local = path.parent / ((module or '').split('.')[0] + '.py')
                if local.is_file():
                    pending.append(local)
    pins = {}
    for name in sorted(names):
        raw = gen.git('show', revision + ':' + name)
        guards.require(raw == (P / name).read_bytes(), 'commit exact comparison input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    for name in NATIVE:
        guards.require(gen.git('show', proposal.BASE + ':' + name) == (P / name).read_bytes(),
            'selected native input changed: ' + name)
    return pins


def admission(record, binaries=False):
    guards.require(inputs(record['source_commit']) == record['inputs'], 'source/recipe input changed')
    full, proof, _ = resource.qualification()
    guards.require(full['binary'] == record['retained_engine'] and proof == record['native_qualification'],
        'selected engine/fixture qualification changed')
    for name, body in proposal.generated(P).items():
        path = ROOT / name
        guards.require(path.read_bytes() == body and sha(path) == record['generated'][str(path)]['sha256'],
            'generated input changed: ' + name)
    manifest = record['compiler_inputs']
    guards.require(sha(manifest['path']) == manifest['sha256'], 'SDK manifest changed')
    rows = json.loads(Path(manifest['path']).read_text())
    guards.require(len(rows) == manifest['files'], 'SDK inventory count changed')
    for row in rows:
        guards.require(sha(row['path']) == row['sha256'], 'SDK/link input changed: ' + row['path'])
    if binaries:
        for row in record['binaries'].values():
            guards.require(sha(row['path']) == row['sha256'], 'comparison executable changed')


def prepare():
    prior = gen.committed(RESULT)
    guards.require(prior['stage'] == 'source_prepared' and not prior['active'] and prior['ownership_closed'],
        'inspect existing column-major owner; no retry')
    guards.require(not (ROOT / 'build-plan.json').exists(), 'existing build plan; no duplicate')
    audit = prior['source_audit']
    guards.require(audit['review_complete'] and audit['prebuild_findings_resolved'],
        'record the independently reviewed source/coverage gates before preparing build')
    closed = gen.committed(gen.A / 'qsa-fixed-heads-current.json')
    guards.require(closed['retirement']['complete'] and not (B / 'qsa-fixed-heads-v1').exists(),
        'retire the consumed auxiliary generation before another comparison build')
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = inputs(revision)
    for name, body in proposal.generated(P).items():
        path = ROOT / name
        guards.require(path.read_bytes() == body and sha(path) == prior['generated'][str(path)]['sha256'],
            'prepared source changed: ' + name)
    full, proof, _ = resource.qualification()
    parent = gen.committed(gen.A / 'cpu-profiler-current.json')['unitrace_indexed_graph']['compiler_inputs']
    guards.require(sha(parent['path']) == parent['sha256'], 'parent SDK manifest changed')
    rows = json.loads(Path(parent['path']).read_text())
    guards.require(len(rows) == parent['files'], 'parent SDK count changed')
    paths = {Path(row['path']) for row in rows}
    compiler = full['compiler']
    guards.require(sha(compiler['path']) == compiler['sha256'], 'compiler changed')
    for row in rows:
        guards.require(sha(row['path']) == row['sha256'], 'parent SDK input changed')
    for name in ('crtbeginS.o', 'crtendS.o', 'libstdc++.so', 'libgcc_s.so'):
        path = Path(subprocess.check_output([compiler['path'], '-print-file-name=' + name], text=True, timeout=15).strip()).resolve()
        guards.require(path.is_file(), 'link input absent')
        paths.add(path)
    manifest = ROOT / 'compiler-inputs.json'
    write(manifest, [dict(path=str(path), sha256=sha(path)) for path in sorted(paths)])
    (ROOT / 'tmp').mkdir()
    flags = ['-std=c++20', '-O3', '-DNDEBUG', '-g0', '-fsycl', '-fsycl-id-queries-range=size_t',
        '-fno-fast-math', '-ffp-contract=off', '-DSTRATA_ENABLE_SYCL=1',
        '-I' + str(P / 'include'), '-I' + str(P / 'include/strata/sycl/compat'), '-I' + str(ROOT)]
    sources = [P / name for name in NATIVE] + [ROOT / 'control.cpp', ROOT / 'colmajor.cpp',
        P / FIXTURE, ROOT / 'timing.cpp']
    objects = [ROOT / f'input-{i}.o' for i in range(len(sources))]
    commands = [[compiler['path'], *flags, '-c', str(src), '-o', str(obj)] for src, obj in zip(sources, objects)]
    for index, name in ((7, 'numerical'), (8, 'timing')):
        commands.append([compiler['path'], *flags, *map(str, objects[:7]), str(objects[index]),
            '-pthread', '-ldl', '-o', str(ROOT / name)])
    for name in ('control', 'colmajor'):
        commands.append([compiler['path'], *flags, '-fsycl-device-only', '-S', '-emit-llvm',
            str(ROOT / (name + '.cpp')), '-o', str(ROOT / (name + '.ll'))])
    install = Path(compiler['path']).parents[1]
    legacy = gen.committed(gen.A / 'native-gdn-engine-build.json')['legacy_provenance_exception']['additional_old_product_review']
    holds = []
    for row in legacy['bounded_review_roots']:
        path = B / row['root']
        if row.get('retirement', {}).get('complete'):
            guards.require(not path.exists(), 'retired legacy root reappeared')
            continue
        products = [f for f in path.rglob('*') if f.is_file()
            and (f.suffix in ('.o', '.a') or f.name in ('strata', 'pr6_prefill_api'))]
        holds.append(dict(path=str(path), exists=path.exists(), selected_files=len(products),
            selected_bytes=sum(f.stat().st_size for f in products), next_action=row['next_action'],
            full_final_source_proof=False))
    record = dict(prior, stage='prepared', source_commit=revision, inputs=pins,
        retained_engine=full['binary'], native_qualification=proof, compiler=compiler,
        compiler_inputs=dict(path=str(manifest), sha256=sha(manifest), files=len(paths)), commands=commands,
        build_environment=dict(PATH=str(install / 'bin') + ':/usr/bin:/bin', LANG='C', LC_ALL='C',
            TMPDIR=str(ROOT / 'tmp'), LD_LIBRARY_PATH=str(install / 'lib'), PYTHONDONTWRITEBYTECODE='1'),
        build_budgets=dict(wall_s=1800, per_command_wall_s=420, cpu_s=900, rss_bytes=12 << 30,
            text_bytes=8 << 20, file_bytes=512 << 20, total_bytes=1 << 30),
        build_status='Prepared13-command recipe; commit exact inputs/plan before build.', passed=False, complete=False)
    record['retained_generations_review'] = dict(selected_engine_and_same_build_fixtures=13,
        previous_auxiliary_root_absent=True, new_auxiliary_consumer='Immediate column-major numerical/codegen/clean comparison only; retire at closure.',
        separate_legacy_holds=holds, hold_inventory_scope='Actual .o/.a, strata and pr6_prefill_api paths in the remaining named legacy roots; not validated generations.',
        separate_ownership_holds='Six older Orca lifecycle holds remain unchanged in research-index.json; no evidence here closes them.')
    admission(record)
    write(ROOT / 'build-plan.json', record)
    write(RESULT, record)
    print('Prepared column-major build; commit the plan before execution.', flush=True)


def build():
    record = gen.committed(RESULT)
    guards.require(record['stage'] == 'prepared' and record == json.loads((ROOT / 'build-plan.json').read_text())
        and not (ROOT / 'build-record.json').exists(), 'fresh committed plan required; inspect existing owner')
    admission(record)
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'build-record.json', dict(record, owned_commands=owner.commands if owner else []))
    owner = supervision.Owner(ROOT / 'build-owned', persist)
    record.update(stage='building', ownership_closed=False)
    persist()
    budget = record['build_budgets']
    deadline = time.monotonic() + budget['wall_s']
    try:
        for index, command in enumerate(record['commands']):
            remaining = deadline - time.monotonic()
            guards.require(remaining > 1, 'whole build deadline')
            print(f'Compiling/linking column-major input {index + 1}/{len(record["commands"])}', flush=True)
            entry, _, _ = owner.run('build-' + str(index), command, record['build_environment'], P,
                wall=min(remaining, budget['per_command_wall_s']), cpu=budget['cpu_s'], rss_cap=budget['rss_bytes'],
                text_cap=budget['text_bytes'], file_cap=budget['file_bytes'], total_cap=budget['total_bytes'])
            guards.require(supervision.closed(entry) and entry['exit_code'] == 0, 'comparison compile/link failed')
        admission(record)
        record.update(stage='built', passed=True, complete=True, build_status='Compiled; GPU qualification not yet performed.',
            binaries={name: dict(path=str(ROOT / name), bytes=(ROOT / name).stat().st_size,
                sha256=sha(ROOT / name)) for name in ('numerical', 'timing')},
            device_IR={name: dict(path=str(ROOT / (name + '.ll')), sha256=sha(ROOT / (name + '.ll')))
                for name in ('control', 'colmajor')})
    except BaseException as error:
        record.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        persist()
        write(RESULT, record)
    print(json.dumps({k: record.get(k) for k in ('stage', 'passed', 'complete', 'ownership_closed', 'error')}), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    guards.require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build'), 'use prepare|build')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.driver_guard()
        sys.exit(prepare() if sys.argv[1] == 'prepare' else build())
