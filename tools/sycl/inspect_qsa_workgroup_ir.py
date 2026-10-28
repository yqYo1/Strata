#!/usr/bin/env python3
"""Source-bound exact-WG discriminator; compile only, never load a GPU image.

Adding this independent controller does not modify the frozen model cohort.
Prepare/run only between owned jobs. A metadata change is not a speed result.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct
import sys

sys.dont_write_bytecode = True
import measure_qsa_k_model as d

m, P, B, RESULT = d.m, d.P, d.B, d.RESULT
ROOT = B / 'qsa-exact-workgroup-ir-v1'
KEY = 'exact_workgroup_IR'
SELF = str(Path(__file__).resolve().relative_to(P))
PREFIX = m.ROOT / 'source/src/kernels/sycl/qsa_bf16_prefix.cpp'
BEFORE = '[=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {'
AFTER = BEFORE[:-1] + '[[sycl::reqd_work_group_size(128)]] {'
require, digest, write = d.require, d.digest, d.write


def inputs(revision):
    raw = m.component.gen.git('show', revision + ':' + SELF)
    require(raw == (P / SELF).read_bytes(), 'commit exact IR controller first')
    return hashlib.sha256(raw).hexdigest()


def source(arm):
    raw = PREFIX.read_text()
    require(raw.count(BEFORE) == 1 and 'Cells = 32, Workgroup = 128;' in raw,
            'exact selected WG/lambda required')
    if arm == 'exact128':
        changed = raw.replace(BEFORE, AFTER)
        require(changed.replace(AFTER, BEFORE) == raw, 'undeclared source edit')
        return changed.encode()
    require(arm == 'control', 'unknown source arm')
    return raw.encode()


def compact(row):
    return {k:v for k,v in row.items() if k != 'commands_executed'}


def admission(row=None):
    record, built, rt = d.native()
    database = m.ROOT / 'build/compile_commands.json'
    if row:
        require(inputs(row['source_commit']) == row['controller_sha256'] and
                digest(database) == row['compile_database_sha256'] and
                digest(PREFIX) == row['prefix_sha256'], 'frozen IR inputs changed')
        for arm in row['arms']:
            require(hashlib.sha256(source(arm['name'])).hexdigest() == arm['source_sha256'],
                    'generated arm changed')
        for path, sha in row['tools'].items():
            require(digest(path) == sha, 'compiler/translator changed')
    return record, built, rt


def prepare():
    record, built, rt = admission()
    require(not ROOT.exists() and KEY not in record[m.KEY], 'fresh IR comparison only')
    revision = m.component.gen.git('rev-parse', 'HEAD').decode().strip()
    controller_sha = inputs(revision)
    database = m.ROOT / 'build/compile_commands.json'
    commands = [v for v in json.loads(database.read_text()) if v['file'] == str(PREFIX)]
    require(len(commands) == 1, 'one production compile recipe required')
    command = shlex.split(commands[0]['command'])
    require(command[-2:] == ['-c', str(PREFIX)] and command.count('-o') == 1,
            'unexpected production recipe')
    at = command.index('-o')
    base = command[:at] + command[at+2:-2]
    compiler = Path(base[0]); translator = compiler.parent / 'llvm-spirv'
    require(compiler.is_file() and translator.is_file(), 'qualified compiler tools required')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir()
    arms = []
    for name in ('control', 'exact128'):
        arms.append(dict(name=name, source_sha256=hashlib.sha256(source(name)).hexdigest(),
            compile=base + ['-fsycl-device-only', '-S', '-emit-llvm', '-MD', '-MF',
                str(ROOT / (name + '.d')), '-o', str(ROOT / (name + '.ll')), str(ROOT / 'source.cpp')],
            spirv=[str(translator), '--spirv-ext=+all', str(ROOT / (name + '.ll')),
                '-o', str(ROOT / (name + '.spv'))]))
    environment = dict(PATH=str(compiler.parent) + ':/usr/bin:/bin', HOME='/home/yayoi',
        LANG='C.UTF-8', LC_ALL='C.UTF-8', LD_LIBRARY_PATH=str(compiler.parent.parent / 'lib'),
        CPLUS_INCLUDE_PATH=str(compiler.parent.parent / 'include'),
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    row = dict(stage='prepared', complete=False, passed=False, active=False, GPU_executed=False,
        source_commit=revision, controller_sha256=controller_sha,
        compile_database_sha256=digest(database), prefix_sha256=digest(PREFIX),
        native_build_source=built['source_commit'], runtime_sha256=digest(m.runtime.RESULT),
        tools={str(p):digest(p) for p in (compiler, translator)}, arms=arms, environment=environment,
        budgets=dict(wall=120, cpu=180, rss_cap=4 << 30, text_cap=2 << 20,
                     file_cap=16 << 20, total_cap=64 << 20),
        plan=dict(question='Does exact WG128 change optimized production attention instructions beyond its required-size metadata?',
            order=['control', 'exact128'], actual_source='Isolated current direct-K production prefix;one exact-size attribute only.',
            guards='Recover source/build/toolchain through existing native admission;use the same source.cpp path and immutable compiler environment in separate compiler processes. Check every actual dependency against the native compiler manifest. Require matching kernel names and exact LLVM/SPIR-V workgroup metadata.',
            stop='Metadata-only or equivalent instruction bodies closes this candidate without device execution. A changed body remains manual-review-only;no native image,resource,correctness or performance advancement without a new committed gate.',
            limits='CPU-only codegen discrimination,not device validation,physical traffic,hardware saturation,model adoption or throughput.',
            retention='Keep hashes,counts,metadata and compact relevant differences. Retire successful raw IR/SPIR-V/dependencies/compiler output after review;no third engine generation.'))
    write(ROOT / 'record.json', row); record[m.KEY][KEY] = compact(row); write(RESULT, record)
    print('Prepared CPU-only exact-WG comparison;commit the plan before run.', flush=True)


def module(raw):
    metadata = dict(re.findall(r'^!(\d+) = (.+)$', raw, re.M))
    attributes = dict(re.findall(r'^attributes #(\d+) = (.+)$', raw, re.M))
    functions = {}; kernels = {}; geometry = {}
    for match in re.finditer(r'^define (.+?)^}', raw, re.M | re.S):
        body = match.group(0); header = body.split('\n', 1)[0]
        name = re.search(r'@([^ (]+)\(', header).group(1)
        if 'spir_kernel' in header:
            kernels[name] = body
            node = re.search(r'!reqd_work_group_size !(\d+)', header)
            geometry[name] = metadata[node[1]] if node else None
        # Preserve instruction/SSA text. Resolve numbered metadata by a stable
        # root traversal,including cycles;only the deliberate WG attachment is
        # excluded. Other metadata and function attributes stay in the digest.
        body = re.sub(r' !reqd_work_group_size !\d+', '', body)
        seen, nodes = {}, []
        def resolve(node):
            if node not in seen:
                require(node in metadata, 'unresolved LLVM metadata')
                seen[node] = len(nodes); nodes.append(None)
                nodes[seen[node]] = re.sub(r'!(\d+)', lambda v:resolve(v[1]), metadata[node])
            return '!node' + str(seen[node])
        body = re.sub(r'!(\d+)', lambda v:resolve(v[1]), body)
        body = re.sub(r'#(\d+)', lambda v:attributes[v[1]], body)
        functions[name] = hashlib.sha256(json.dumps([body, nodes]).encode()).hexdigest()
    require(kernels and functions, 'missing LLVM definitions')
    return dict(functions=functions, kernels=sorted(kernels), required_workgroup=geometry)


def spirv_geometry(path):
    # SPIR-V core opcodes: OpEntryPoint15,OpExecutionMode16,LocalSize17.
    # https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
    raw = path.read_bytes()
    require(len(raw) >= 20 and len(raw) % 4 == 0, 'malformed SPIR-V extent')
    words = struct.unpack('<' + 'I' * (len(raw) // 4), raw)
    require(words[0] == 0x07230203, 'SPIR-V magic/endianness differs')
    entries, sizes = {}, {}
    at = 5
    while at < len(words):
        count, opcode = words[at] >> 16, words[at] & 65535
        require(count > 0 and count <= len(words) - at, 'SPIR-V instruction extent')
        operands = words[at+1:at+count]
        if opcode == 15:
            require(count >= 4, 'short SPIR-V entry')
            name = struct.pack('<' + 'I' * len(operands[2:]), *operands[2:]).split(b'\0', 1)[0].decode()
            require(operands[1] not in entries, 'duplicate entry')
            entries[operands[1]] = name
        if opcode == 16 and len(operands) >= 2 and operands[1] == 17:
            require(count == 6 and operands[0] not in sizes, 'LocalSize contract differs')
            sizes[operands[0]] = list(operands[2:])
        at += count
    require(at == len(words), 'SPIR-V trailing words')
    return {name:sizes.get(index) for index,name in entries.items()}


def inspect(root, arm, allowed):
    name = arm['name']; dep = (root / (name + '.d')).read_text().replace('\\\n', ' ')
    require(': ' in dep, 'malformed dependency output')
    dependencies = shlex.split(dep.split(': ', 1)[1])
    for value in dependencies:
        path = Path(value).resolve()
        expected = arm['source_sha256'] if path == (root / 'source.cpp') else allowed.get(str(path))
        require(expected and digest(path) == expected, 'unfrozen actual compiler input: ' + str(path))
    raw = (root / (name + '.ll')).read_text(); parsed = module(raw)
    parsed.update(IR_sha256=digest(root / (name + '.ll')), SPIRV_sha256=digest(root / (name + '.spv')),
        actual_dependencies=len(dependencies), SPIRV_local_sizes=spirv_geometry(root / (name + '.spv')))
    return parsed


def run():
    record, _, _ = admission(); row = json.loads((ROOT / 'record.json').read_text())
    require(record[m.KEY][KEY] == compact(row) and row['stage'] == 'prepared', 'committed fresh plan required')
    admission(row); m.runtime.guards.no_model_owners([])
    require(digest(m.runtime.RESULT) == row['runtime_sha256'], 'runtime identity changed')
    allowed = {str(Path(p).resolve()):sha for p,sha in json.loads(
        (m.ROOT / 'compiler-inputs.json').read_text()).items()}
    outcomes = []
    def persist():
        write(ROOT / 'execution.json', dict(active=owner.active is not None, commands=owner.commands))
    owner = m.runtime.supervision.Owner(ROOT / 'owned', persist)
    try:
        for arm in row['arms']:
            (ROOT / 'source.cpp').write_bytes(source(arm['name']))
            for step in ('compile', 'spirv'):
                entry, _, _ = owner.run(arm['name'] + '-' + step, arm[step], row['environment'],
                    m.ROOT / 'build', **row['budgets'])
                require(m.runtime.supervision.closed(entry) and entry['exit_code'] == 0,
                        'CPU-only ' + step + ' failed')
            outcomes.append(inspect(ROOT, arm, allowed))
        a, b = outcomes
        require(a['kernels'] == b['kernels'], 'kernel identity changed')
        attention = [name for name in a['kernels'] if 'prefix_batch_impl' in name]
        require(len(attention) == 3 and all(a['required_workgroup'][n] is None and
            b['required_workgroup'][n] == '!{i32 128, i32 1, i32 1}' for n in attention),
            'expected exact LLVM workgroup metadata missing')
        require(all(b['SPIRV_local_sizes'].get(n.strip('"')) == [128, 1, 1] for n in attention),
                'exact SPIR-V LocalSize contract missing')
        changed = sorted(n for n in set(a['functions']) | set(b['functions'])
                         if a['functions'].get(n) != b['functions'].get(n))
        admission(row)
        observed = [dict(IR_sha256=o['IR_sha256'], SPIRV_sha256=o['SPIRV_sha256'],
            actual_dependencies=o['actual_dependencies'], functions=len(o['functions']),
            attention_kernels=len(attention), kernel_set_sha256=hashlib.sha256(
                json.dumps(o['kernels']).encode()).hexdigest(),
            required_attention_workgroups=[o['required_workgroup'][n] for n in attention],
            SPIRV_attention_local_sizes=[o['SPIRV_local_sizes'].get(n.strip('"')) for n in attention])
                    for o in outcomes]
        row.update(stage='closed', complete=True, passed=True, arms_observed=observed,
            decision=dict(changed_function_names=changed, relevant_IR_delta_unproven=bool(changed),
                candidate_closed_without_GPU=not changed, native_or_model_advance_permitted=False))
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row.update(active=owner.active is not None, ownership_closed=owner.active is None and
            all(m.runtime.supervision.closed(c) for c in owner.commands))
        row['passed'] = bool(row['passed'] and row['ownership_closed'] and not row.get('error'))
        row['commands_executed'] = owner.commands; persist()
        write(ROOT / 'record.json', row); record[m.KEY][KEY] = compact(row); write(RESULT, record)
    print(json.dumps({k:row.get(k) for k in ('stage', 'complete', 'passed', 'decision', 'error')}), flush=True)
    require(row['passed'], 'CPU-only exact-WG comparison failed')


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['run']), 'prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare() if sys.argv[1] == 'prepare' else run()
