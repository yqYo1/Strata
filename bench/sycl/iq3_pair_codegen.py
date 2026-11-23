"""Compare emitted code for the two decode IQ3_XXS pair schedules."""

import fcntl
import hashlib
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_mixed_gather_measure as reference
from prefill_attention_bank_component import dependency_paths

bank, require = reference.bank, reference.require
owner = reference.owner
ROOT = bank.B / 'cpu-iq3-pair-codegen-v1'
KEY = 'cpu_iq3_pair_codegen_v1'
BASE = '63392f03f8536fe83a7f04b9d14ccd1c79da0f5d'
CANDIDATE = 'fb7af68e907ed4681e9650691d84b8c08dc192c5'
SOURCE = 'src/kernels/cpu/iq_single_avx2.cpp'
BASE_OBJECT_SHA = '3a8460290231cec4fb82845564a3d2522e67229634dbc01fcbe21b08372beefd'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
OUTPUT = ROOT / 'iq_single_avx2.cpp.o'
DEPS = ROOT / 'iq_single_avx2.cpp.d'
PRIVATE_SOURCE = ROOT / Path(SOURCE).name
LIMITS = dict(wall=90, cpu=180, rss_cap=4 << 30, text_cap=2 << 20,
              file_cap=16 << 20, total_cap=32 << 20)


def emitted(object_path):
    symbols = subprocess.check_output(['/usr/bin/nm', '--defined-only', str(object_path)],
                                      text=True, timeout=15)
    symbol, = (line.split()[-1] for line in symbols.splitlines() if 'iq3xxs_pair_dot' in line)
    raw = subprocess.check_output(['/usr/bin/objdump', '-d', '--no-show-raw-insn',
        '--disassemble=' + symbol, str(object_path)], text=True, timeout=20)
    instructions = re.findall(r'^[ \t]*([0-9a-f]+):[ \t]+(\S+)[ \t]*(.*)$', raw, re.M)
    require(instructions, 'missing actual pair dot instructions')
    first = int(instructions[0][0], 16)
    body = [(int(a, 16) - first, op, operands.partition('#')[0].strip())
            for a, op, operands in instructions]
    return body, raw


def reduction(body):
    def location(token):
        return re.sub(r'^%xmm(\d+)$', r'%ymm\1', token.strip())

    definitions, outputs, adds = {}, [], 0
    for address, op, operands in body:
        args = re.split(r',(?=[^()]*(?:\(|$))', operands)
        dst = location(args[-1])
        if op == 'vpmaddwd':
            definitions[dst] = (0, frozenset({address}))
        elif op == 'vpaddd':
            a, b = (definitions[location(arg)] for arg in args[:2])
            require(a[1].isdisjoint(b[1]), 'overlapping integer partial sums')
            definitions[dst] = (1 + max(a[0], b[0]), a[1] | b[1])
            adds += 1
        elif op in ('vmovdqa', 'vmovdqu', 'vmovaps', 'vmovups', 'vmovdqa64') and location(args[0]) in definitions:
            definitions[dst] = definitions[location(args[0])]
        elif op == 'vcvtdq2ps':
            outputs.append(definitions[location(args[0])])
            definitions.pop(dst, None)
            if len(outputs) == 2:
                break
        else:
            definitions.pop(dst, None)
    require(adds == 14 and [len(a[1]) for a in outputs] == [8, 8] and
            outputs[0][1].isdisjoint(outputs[1][1]), 'different compiled reduction shape')
    return dict(depths=[a[0] for a in outputs], integer_adds=adds)


def summary(body):
    return dict(instructions_sha256=bank.signature(body),
                stack_vector_operands=sum(op.startswith('v') and '(%rsp)' in args for _,op,args in body))


def compiler_inputs(native, parent):
    source, build = parent / 'source' / SOURCE, parent / 'build'
    db = bank.previous.read(build / 'compile_commands.json')
    entry, = (e for e in db if e['file'] == str(source))
    args = shlex.split(entry['command'])
    original = build / entry['output']
    require(entry['directory'] == str(build) and args[-4:] == ['-o', entry['output'], '-c', str(source)]
            and all(f in args for f in ('-O3', '-march=znver3', '-mavx2')) and '-fsycl' not in args
            and native['actual_inputs']['objects'][str(original)] == bank.sha(original) == BASE_OBJECT_SHA,
            'expected original Zen 3 compilation and object')
    lines = subprocess.check_output(['/usr/bin/ninja', '-t', 'deps', entry['output']], cwd=build,
                                    text=True, timeout=20).splitlines()
    match = re.fullmatch(re.escape(entry['output']) + r': #deps (\d+), deps mtime \d+ \(VALID\)', lines[0])
    deps = {str((build / line.strip()).resolve()) for line in lines[1:] if line.startswith('    ')}
    require(match and len(deps) == int(match[1]) and str(source) in deps, 'missing original compiler dependencies')
    pinned = native['actual_inputs']['compiled']
    require(all(pinned.get(p) == bank.sha(p) for p in deps), 'original compiler inputs changed')
    new_args = [*args[:-4], '-MD', '-MF', str(DEPS), '-o', str(OUTPUT), '-c', str(PRIVATE_SOURCE)]
    deps.remove(str(source))
    return original, new_args, {p: pinned[p] for p in sorted(deps)}


def inputs():
    native, _, parent, build = reference.coherent_inputs()
    original, command, headers = compiler_inputs(native, parent)
    require(reference.fixture.git_file(BASE, SOURCE) == (parent / 'source' / SOURCE).read_bytes(),
            'comparison parent source differs from the original object input')
    changed = subprocess.check_output(['git', 'diff', '--name-only', BASE, CANDIDATE, '--'],
                                      cwd=bank.P, text=True, timeout=20).splitlines()
    require(changed == [SOURCE], 'expected one isolated CPU translation unit change')
    candidate = reference.fixture.git_file(CANDIDATE, SOURCE)
    externals = native['external_inputs']
    require(bank.sha(externals['path']) == externals['sha256'], 'changed build SDK manifest')
    compiler = str(Path(command[0]).resolve())
    require(bank.previous.read(Path(externals['path'])).get(compiler) == bank.sha(compiler),
            'compiler differs from the original build')
    bench_control = [str(Path(m.__file__).relative_to(bank.P)) for m in tuple(sys.modules.values())
        if getattr(m, '__file__', None) and Path(m.__file__).is_relative_to(bank.P / 'bench' / 'sycl')]
    pins = bank.stage.source.pin(bank.P, sorted({THIS, 'tools/sycl', *bench_control}))
    require(THIS in pins['source_sha256'], 'commit this controller first')
    env = dict(native['recipe']['environment'], TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache'))
    proof = dict(base=BASE, candidate=CANDIDATE, source_sha256=hashlib.sha256(candidate).hexdigest(),
        parent_build=build['build_sha256'], baseline_object=BASE_OBJECT_SHA, command=command, cwd=str(parent/'build'),
        dependency_count=len(headers)+1, dependency_sha256=bank.signature(headers),
        tool_sha256={p: bank.sha(p) for p in (compiler, '/usr/bin/ninja', '/usr/bin/nm', '/usr/bin/objdump')},
        environment_sha256=bank.signature(env), controller_sha256=pins['aggregate_sha256'], limits=LIMITS)
    return original, command, headers, candidate, env, proof, pins['revision']


def save(row):
    bank.stage.write(ROOT/'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def execute(action):
    original, command, headers, candidate, env, proof, controller_commit = inputs()
    if action == 'prepare':
        require(not ROOT.exists() and KEY not in reference.validation.committed(bank.RESULT),
                'inspect existing experiment before preparing')
        body, _ = emitted(original)
        base = dict(**summary(body), **reduction(body))
        require(base['depths'] == [4, 4], 'original reduction no longer matches the plan')
        ROOT.mkdir(mode=0o700)
        for name in ('tmp', 'cache'):
            (ROOT/name).mkdir(mode=0o700)
        PRIVATE_SOURCE.write_bytes(candidate)
        save(dict(stage='prepared', active=False, ownership_closed=True, proof=proof, original=base,
                  controller_commit=controller_commit,
                  comparison='Check emitted code first. A differing pair dot may advance to the three-arm CPU fixture.'))
        return
    row = bank.previous.read(ROOT/'record.json')
    require(row == reference.validation.committed(bank.RESULT, (KEY,)) and row['stage'] == 'prepared'
            and row['proof'] == proof and PRIVATE_SOURCE.read_bytes() == candidate and not OUTPUT.exists(),
            'exact committed build plan and source required')
    job = owner.Owner(ROOT/'owned', lambda: bank.stage.write(ROOT/'execution.json', job.commands))
    try:
        entry, _, err = job.run('compile', command, env, proof['cwd'], **LIMITS)
        require(owner.closed(entry) and entry['exit_code'] == 0, 'candidate compilation failed: ' + err.read_text()[-900:])
        expected = dict(headers, **{str(PRIVATE_SOURCE): proof['source_sha256']})
        paths = dependency_paths(DEPS.read_text())
        require(paths == expected.keys() and all(bank.sha(p) == h for p,h in expected.items()),
                'actual comparison dependency set differs')
        body, raw = emitted(OUTPUT)
        after = summary(body)
        try:
            after.update(reduction(body))
        except (KeyError, RuntimeError) as error:
            (ROOT/'candidate.asm').write_text(raw)
            after['reduction_unresolved'] = repr(error)
        row.update(stage='closed', result=after, candidate_object=bank.sha(OUTPUT),
                   different_code=after['instructions_sha256'] != row['original']['instructions_sha256'])
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        raise
    finally:
        row.update(active=bool(job.active), ownership_closed=not job.active and
                   all(owner.closed(c) for c in job.commands))
        save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', 'build') and sys.executable == bank.basis.reference.model.PYTHON,
            'use the qualified interpreter and a single code-generation stage')
    with (bank.B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        execute(action)
