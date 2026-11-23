"""Check the bounded MTP candidate with the measured engine's SYCL recipes."""

import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'tools/sycl'))
from measurement_guards import no_model_owners, require

STATE = Path.home() / '.local/state/strata-sycl/post-reboot-tuning-20261007'
PARENT = STATE / 'prefill-large-gather-kv-model-v1'
RECORD = 'f8ee14198f918d3804dcf2878d8955a958b369f0a8880ef88944d6e37524a7cb'
CHANGE = '4d2efd356739a932b9839bb305e3b9e43087545a'
BOUNDARY = '178d53212581cc5d11a0e80d8991e9fda3aaa339'
UNIT = 'src/core/mtp.cpp'
GENERATOR = 'src/program/generate.cpp'
TARGETS = ((UNIT, 'strata_engine'), (GENERATOR, 'strata'),
           (GENERATOR, 'strata_full_context_diagnostic'))
PROJECTED = {UNIT: '33f2e63708a5b83a86cd5cd87d490abba380963508bb9849f3a8a0d70bce2fc5',
             GENERATOR: '81fe4b074aa94c1dc86730ec930b004d246dc8328e59d33a13d2640f91344dc9'}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def blob(revision, name):
    return subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=PROJECT, timeout=30)


def recipe(directory):
    record = PARENT / 'record.json'
    require(digest(record) == RECORD, 'original coherent build record required')
    row = json.loads(record.read_bytes())
    exe = row['binaries']['strata']
    require(row['complete'] and row['passed'] and row['ownership_closed']
            and digest(exe['path']) == exe['sha256'], 'parent executable differs or is unqualified')
    build = PARENT / 'build'
    table = build / 'compile_commands.json'
    require(digest(table) == row['configured_files'][str(table)], 'compiler table changed')
    commands = json.loads(table.read_bytes())
    ext = row['external_inputs']
    require(digest(ext['path']) == ext['sha256'], 'external toolchain inventory changed')
    tools = json.loads(Path(ext['path']).read_bytes())
    for name in PROJECTED:
        original = PARENT / 'source' / name
        source = original.read_bytes()
        require(hashlib.sha256(source).hexdigest() == row['actual_inputs']['compiled'][str(original)]
                and (PROJECT / name).read_bytes() == blob(BOUNDARY, name), 'source or projection base changed')
        if name == UNIT:
            require(source == blob(CHANGE + '^', UNIT), 'MTP projection base differs')
            source = blob(BOUNDARY, UNIT)
        candidate = directory / name
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(source)
    patch = subprocess.check_output(['git', 'diff', '--unified=1', BOUNDARY + '^', BOUNDARY, '--', GENERATOR],
                                    cwd=PROJECT, timeout=30)
    subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', '-'], cwd=directory,
                   input=patch, check=True, timeout=30)
    require({name: digest(directory / name) for name in PROJECTED} == PROJECTED, 'projected source differs')

    recipes = []
    for name, target in TARGETS:
        original = PARENT / 'source' / name
        output = f'CMakeFiles/{target}.dir/{name}.o'
        unit, = [u for u in commands if u['file'] == str(original) and u['output'] == output]
        args = shlex.split(unit['command'])
        diagnostic = any(flag.startswith('-DSTRATA_SYCL_FULL_CONTEXT_DIAGNOSTIC') for flag in args)
        require(unit['directory'] == str(build) and args[-4:] == ['-o', output, '-c', str(original)]
                and '-fsycl' in args and '-Werror' in args and digest(args[0]) == tools[args[0]]
                and diagnostic == (target == 'strata_full_context_diagnostic'),
                'production SYCL compile command required for ' + target)
        recipes.append((name, target, original, args[:-4]))
    return row, recipes


def check(directory, action):
    row, recipes = recipe(directory)
    build = PARENT / 'build'
    env = dict(row['recipe']['environment'], TMPDIR=str(directory), XDG_CACHE_HOME=str(directory))
    known = row['actual_inputs']['compiled']
    for name, target, original, args in recipes:
        if action == 'recipe':
            print(f'{target}: {name} {PROJECTED[name]}, compiler {args[0]}', flush=True)
            continue
        candidate = directory / name
        depfile = directory / (target + '.d')
        cmd = [*args, '-iquote', str(original.parent), '-fsyntax-only', '-MD', '-MF', str(depfile),
               '-MT', 'checked', str(candidate)]
        subprocess.run(cmd, cwd=build, env=env, check=True, timeout=240)
        deps = shlex.split(depfile.read_text().replace('\\\n', ' ').split(':', 1)[1])
        paths = {(build / p).resolve() for p in deps}
        require(candidate in paths and original not in paths
                and PARENT / 'source/include/strata/core/mtp.hpp' in paths,
                'candidate or original production headers missing')
        paths.remove(candidate)
        require(all(known.get(str(p)) == digest(p) for p in paths), 'unqualified compiler dependency')
        print(f'{target}: SYCL syntax passed with {len(paths)} matched includes', flush=True)


def main(action):
    for name in ('bench/sycl/decode_mtp_prefix_source.py', 'tools/sycl/measurement_guards.py'):
        require(blob('HEAD', name) == (PROJECT / name).read_bytes(), 'uncommitted check input: ' + name)
    if action == 'recipe':
        with tempfile.TemporaryDirectory(prefix='decode-mtp-recipe-', dir=STATE) as temp:
            check(Path(temp), action)
        return
    with (STATE / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        no_model_owners([])
        with tempfile.TemporaryDirectory(prefix='decode-mtp-prefix-', dir=STATE) as temp:
            check(Path(temp), action)


if __name__ == '__main__':
    require(sys.argv[1:] in (['recipe'], ['check']), 'select recipe or check')
    main(sys.argv[1])
