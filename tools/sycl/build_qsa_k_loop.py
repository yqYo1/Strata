#!/usr/bin/env python3
"""Own MIT native component build; never launches a GPU workload.

Uses the previously qualified own b0b863cc9 harness with mechanical naming
changes. The new row-invariant source has its separate source/IR contract.
"""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qsa_k_loop_sources as proposal
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision

P, B, A = gen.P, gen.B, gen.A
ROOT = B / 'qsa-direct-k-native-v1'
RESULT = A / 'qsa-codegen-current.json'
OLD = A / 'qsa-k-staging-current.json'
KEY = 'direct_K_loop_invariant'
SELF = str(Path(__file__).resolve().relative_to(P))
FIXTURE = 'bench/sycl/qsa_k_loop.cpp'
ADMISSION = 'bench/sycl/qsa_k_loop_admission.hpp'
ADAPTER = 'bench/sycl/qsa_k_loop_oracle_adapter.cpp'
ORACLE = 'tests/sycl/qsa_bf16_prefix.cpp'
NATIVE = ['src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
          'src/kernels/sycl/prompt_attention.cpp', proposal.PREFIX]
require = guards.require


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def inputs(revision):
    names = set([*NATIVE, FIXTURE, ADMISSION, ADAPTER, ORACLE])
    names.update(gen.git('ls-tree', '-r', '--name-only', revision, '--', 'include').decode().splitlines())
    pending, seen = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        names.add(str(path.relative_to(P)))
        raw = gen.git('show', revision + ':' + str(path.relative_to(P)))
        require(raw == path.read_bytes(), 'commit controller source first')
        for node in ast.walk(ast.parse(raw)):
            modules = [x.name for x in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                local = path.parent / ((module or '').split('.')[0] + '.py')
                if local.is_file():
                    pending.append(local)
    result = {}
    for name in sorted(names):
        raw = gen.git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'source changed: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def generated():
    result = {}
    header = ('// Own MIT component declarations; no production caller.\n#pragma once\n'
              '#include "strata/kernels/qsa_bf16_prefix.hpp"\n#include <sycl/sycl.hpp>\n')
    for name, original in proposal.generated(P).items():
        text = original.decode()
        edits = [
            ('QsaPrefixStateProbe probe = {}) {',
             'QsaPrefixStateProbe probe = {}, sycl::event *submitted = nullptr) {'),
            ('  sycl_backend::finish(stream, event);\n  return true;',
             '  if (submitted) *submitted = event;\n  sycl_backend::finish(stream, event);\n  return true;'),
            ('void *stream) {', 'void *stream, sycl::event *submitted) {'),
            ('output, queries, bounds, live_cells, scratch, stream);',
             'output, queries, bounds, live_cells, scratch, stream, {}, submitted);')]
        for before, after in edits:
            text = proposal.once(text, before, after)
        recovered = text
        for before, after in reversed(edits):
            recovered = proposal.once(recovered, after, before)
        require(recovered.encode() == original, 'undeclared native wrapper change')
        result[name] = text.encode()
        wrapper = text[text.index('bool ' + proposal.EXPORT + '('):]
        signature = wrapper[:wrapper.index(' {')]
        header += 'namespace strata::kernels::' + Path(name).stem + ' {\n' + signature + ';\n}\n'
    result['qsa-direct-k.hpp'] = header.encode()
    return result


def contract(record):
    source = record[KEY]
    require(source['outcome']['passed'] and source['outcome']['ownership_closed'] and
            not source['outcome']['active'], 'closed compiled-IR evidence required')
    review = source['source_contract']
    require(review['source_reviewed'] and review['review_model'] == 'gpt-6-luna max' and
            review['generator_sha256'] == digest(P / 'tools/sycl/qsa_k_loop_sources.py'),
            'exact independent source contract required')
    require(review['compiler_origin']['installed_build_identity_passed'] and
            review['compiler_origin']['compiler_sha256'] == source['compiler']['sha256'],
            'compiler origin unresolved')
    require(proposal.mapping_proof() == source['mapping_proof'], 'source mapping proof differs')
    return review


def kernel_ids(artifacts):
    """Own b0b863cc9 checker: bind the exact attention signature, not a name hint."""
    names = []
    for arm in ['original', 'control', 'candidate']:
        asset = artifacts[arm]
        require(digest(asset['path']) == asset['sha256'], 'device IR changed')
        text = Path(asset['path']).read_text()
        headers = re.findall(r'^define [^\n]*prefix_batch_implILi33'
            r'ELb0ELb0ELb1ELb1ELb1ELb1ELb0EEE[^\n]*', text, re.M)
        selected = [h for h in headers if '!intel_reqd_sub_group_size ' in h]
        require(len(selected) == 1, 'one exact ordered-exp attention kernel per arm')
        header = selected[0]
        local = {int(index): int(alignment) for alignment, index in re.findall(
            r'ptr addrspace\(3\) noundef align (\d+) %(\d+)', header)}
        require(local == {0: 4, 1: 4, 3: 4, 4: 4, 5: 4, 13: 8, 15: 4, 16: 4} and
            [int(v) for v in re.findall(r'%(\d+)(?=,|\))', header)] == list(range(18)),
            'compiled argument/local-pointer layout differs')
        metadata = re.search(r'!intel_reqd_sub_group_size !(\d+)', header)[1]
        require(re.search(r'^!' + metadata + r' = !\{i32 16\}$', text, re.M),
                'compiled SIMD contract differs')
        name = re.search(r'spir_kernel void @([^ (]+)\(', header)[1]
        require(name.startswith('_ZTS'), 'unexpected kernel identity encoding')
        names.append(name)
    require(len(set(names)) == 3, 'compiled arm identities overlap')
    return names


def compiler_inputs(plan):
    origin = plan['compiler_manifest']
    require(digest(origin['path']) == origin['sha256'], 'shared compiler manifest changed')
    rows = json.loads(Path(origin['path']).read_text())
    require(len(rows) == origin['files'], 'compiler manifest count changed')
    result = {}
    for row in rows:
        expected = plan['declared_header_changes'].get(row['path'], {}).get('new_build_sha256', row['sha256'])
        require(digest(row['path']) == expected, 'compiler dependency changed: ' + row['path'])
        result[str(Path(row['path']).resolve())] = expected
    for path, sha in plan['link_inputs'].items():
        require(digest(path) == sha, 'link input changed')
        result[str(Path(path).resolve())] = sha
    return result


def admit(record, plan):
    require(inputs(plan['source_commit']) == plan['input_sha256'], 'native source changed')
    require(contract(record) == plan['source_contract'], 'source contract changed')
    require(digest(plan['compiler']['path']) == plan['compiler']['sha256'], 'compiler changed')
    allowed = compiler_inputs(plan)
    for name, sha in plan['input_sha256'].items():
        allowed[str((P / name).resolve())] = sha
    for name, data in generated().items():
        path = ROOT / name
        require(path.read_bytes() == data and digest(path) == plan['generated'][name], 'generated input changed')
        allowed[str(path)] = plan['generated'][name]
    return allowed


def prepare():
    record = gen.committed(RESULT)
    require('native_build' not in record[KEY] and not ROOT.exists(), 'fresh native build only;no retry')
    review = contract(record)
    previous = gen.committed(OLD)
    require(previous['generation_retirement']['complete'], 'retire rejected generation before new build')
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = inputs(revision)
    compiler = previous['compiler']
    require(digest(compiler['path']) == compiler['sha256'], 'compiler changed')
    shared = gen.committed(A / 'cpu-profiler-current.json')['unitrace_indexed_graph']['compiler_inputs']
    ROOT.mkdir(mode=0o700)
    (ROOT / 'tmp').mkdir()
    blobs = generated()
    for name, raw in blobs.items():
        (ROOT / name).write_bytes(raw)
    sdk = Path(compiler['path']).parents[1]
    environment = dict(PATH=str(sdk / 'bin') + ':/usr/bin:/bin', LANG='C', LC_ALL='C',
        LD_LIBRARY_PATH=str(sdk / 'lib'), TMPDIR=str(ROOT / 'tmp'), PYTHONDONTWRITEBYTECODE='1')
    owner = supervision.Owner(ROOT / 'discovery')
    links = {}
    for name in ['crtbeginS.o', 'crtendS.o', 'libstdc++.so', 'libgcc_s.so']:
        entry, out, err = owner.run(name, [compiler['path'], '-print-file-name=' + name],
            environment, P, wall=20, cpu=20, rss_cap=1 << 30, text_cap=1 << 20, file_cap=1 << 20, total_cap=4 << 20)
        require(supervision.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size,
                'compiler link-input discovery failed')
        path = Path(out.read_text().strip()).resolve()
        require(path.is_file(), 'link input absent')
        links[str(path)] = digest(path)
    flags = ['-std=c++20', '-O3', '-DNDEBUG', '-g0', '-fsycl', '-fsycl-id-queries-range=size_t',
        '-fno-fast-math', '-ffp-contract=off', '-DSTRATA_ENABLE_SYCL=1', '-DSTRATA_QSA_PACKED_GUARD=17',
        '-I' + str(P / 'include'), '-I' + str(P / 'include/strata/sycl/compat'), '-I' + str(ROOT)]
    sources = [P / x for x in NATIVE] + [ROOT / 'direct_k_control.cpp', ROOT / 'direct_k_invariant.cpp',
        P / FIXTURE, P / ADAPTER, P / ORACLE]
    objects = [ROOT / ('input-' + str(i) + '.o') for i in range(len(sources))]
    commands = []
    for path, obj in zip(sources, objects):
        extra = ['-Dqsa_bf16_prefix_batch=qsa_k_loop_test_batch', '-DSTRATA_QSA_COMPARE_PREFIX33=1'] if path == P / ORACLE else []
        commands.append([compiler['path'], *flags, *extra, '-MD', '-MF', str(obj.with_suffix('.d')),
                         '-c', str(path), '-o', str(obj)])
    for name, selected in [('fixture', objects[:8]), ('numerical', objects[:7] + objects[8:])]:
        commands.append([compiler['path'], *flags, *map(str, selected), '-pthread', '-ldl', '-o', str(ROOT / name)])
    for name, path in [('original', P / proposal.PREFIX), ('control', sources[5]), ('candidate', sources[6])]:
        commands.append([compiler['path'], *flags, '-fsycl-device-only', '-MD', '-MF', str(ROOT / (name + '-ir.d')),
            '-S', '-emit-llvm', str(path), '-o', str(ROOT / (name + '.ll'))])
    plan = dict(stage='prepared', source_commit=revision, input_sha256=pins, source_contract=review,
        compiler=compiler, compiler_manifest=shared, declared_header_changes=previous['declared_compiler_header_delta'],
        link_inputs=links, generated={name: hashlib.sha256(raw).hexdigest() for name, raw in blobs.items()},
        commands=commands, environment=environment,
        budgets=dict(wall=420, cpu=900, rss_cap=12 << 30, text_cap=8 << 20, file_cap=512 << 20, total_cap=1 << 30),
        total_wall_seconds=1800, ownership_closed=True, active=False, GPU_executed=False,
        provenance='Own b0b863cc9 fixture/admission/oracle adapter with namespace/include/diagnostic-name substitutions. New direct-K proposal139030e09;host event-return additions reverse exactly to the CPU-IR constructor. No XeStrata application source.',
        criteria='Compile/link only. Verify exact native/controller/generated/compiler/link inputs,all actual dependencies and3unique18arg SG16 attention definitions before property/numerical plans. No GPU or performance/model qualification.',
        retention='One additional fixture/numerical component generation beside selected6e40/12fixtures for immediate property/smoke/boundary/oracle and frozen matched speed decision. Retire rejected/closed outputs after original source/recipe recovery;legacy holds separate.')
    record[KEY]['native_build'] = plan
    admit(record, plan)
    write(ROOT / 'plan.json', plan)
    write(RESULT, record)
    print('Prepared15source-closed compile/link/IR commands;commit before build.')


def build():
    record = gen.committed(RESULT)
    plan = record[KEY]['native_build']
    require(plan['stage'] == 'prepared' and plan == json.loads((ROOT / 'plan.json').read_text()) and
            not (ROOT / 'owned.json').exists(), 'fresh committed build only')
    allowed = admit(record, plan)
    result = dict(passed=False, complete=False, active=False, ownership_closed=False, GPU_executed=False)
    owner = None
    def persist():
        result['active'] = owner is not None and owner.active is not None
        write(ROOT / 'owned.json', dict(result, commands=owner.commands if owner else []))
    owner = supervision.Owner(ROOT / 'owned', persist)
    deadline = time.monotonic() + plan['total_wall_seconds']
    try:
        for index, command in enumerate(plan['commands']):
            remaining = deadline - time.monotonic()
            require(remaining > 1, 'whole build deadline')
            print('CPU compile/link', index + 1, '/', len(plan['commands']), flush=True)
            budgets = dict(plan['budgets'], wall=min(remaining, plan['budgets']['wall']))
            entry, out, err = owner.run(str(index), command, plan['environment'], P, **budgets)
            require(supervision.closed(entry) and entry['exit_code'] == 0, 'component compile/link failed')
        admit(record, plan)
        actual = {}
        for dep in ROOT.glob('*.d'):
            text = dep.read_text().replace('\\\n', ' ')
            for name in shlex.split(text.split(':', 1)[1]):
                path = str(Path(name).resolve());sha = digest(path)
                require(allowed.get(path) == sha, 'unfrozen actual compiler input: ' + path)
                actual[path] = sha
        require(len(actual) > 600, 'incomplete actual dependency closure')
        def artifact(path):
            return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))
        images = {name: artifact(ROOT / (name + '.ll')) for name in ['original', 'control', 'candidate']}
        names = kernel_ids(images)
        result.update(passed=True, complete=True, actual_compiler_inputs=actual,
            binaries={name: artifact(ROOT / name) for name in ['fixture', 'numerical']},
            device_IR=images, attention_kernel_ids=names)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['ownership_closed'] = owner.active is None and all(x.get('direct_child_reaped') and
            x.get('session_empty') and not x.get('survivors') for x in owner.commands)
        persist();plan['stage'] = 'closed';plan['outcome'] = result;write(RESULT, record)
    print('PASSnative component build;GPU/property/numerical/speed remain UNRUN.')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare': prepare, 'build': build}[sys.argv[1]]()
