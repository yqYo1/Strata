"""Check the isolated MTP translation unit with the measured engine's SYCL recipe."""

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
UNIT = 'src/core/mtp.cpp'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def blob(revision, name):
    return subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=PROJECT, timeout=30)


def recipe():
    record = PARENT / 'record.json'
    require(digest(record) == RECORD, 'original coherent build record required')
    row = json.loads(record.read_bytes())
    exe = row['binaries']['strata']
    require(row['complete'] and row['passed'] and row['ownership_closed']
            and digest(exe['path']) == exe['sha256'], 'parent executable differs or is unqualified')
    original = PARENT / 'source' / UNIT
    source = blob(CHANGE, UNIT)
    require(original.read_bytes() == blob(CHANGE + '^', UNIT)
            and digest(original) == row['actual_inputs']['compiled'][str(original)]
            and source == (PROJECT / UNIT).read_bytes(), 'MTP source or projection base changed')
    build = PARENT / 'build'
    table = build / 'compile_commands.json'
    require(digest(table) == row['configured_files'][str(table)], 'compiler table changed')
    unit, = [u for u in json.loads(table.read_bytes()) if u['file'] == str(original)]
    args = shlex.split(unit['command'])
    require(unit['directory'] == str(build) and args[-4:] == ['-o', unit['output'], '-c', str(original)]
            and '-fsycl' in args and '-Werror' in args, 'production SYCL compile command required')
    ext = row['external_inputs']
    require(digest(ext['path']) == ext['sha256'], 'external toolchain inventory changed')
    tools = json.loads(Path(ext['path']).read_bytes())
    require(digest(args[0]) == tools[args[0]], 'compiler differs')
    return row, original, source, args[:-4]


def main(action):
    for name in ('bench/sycl/decode_mtp_prefix_source.py', 'tools/sycl/measurement_guards.py'):
        require(blob('HEAD', name) == (PROJECT / name).read_bytes(), 'uncommitted check input: ' + name)
    row, original, source, args = recipe()
    if action == 'recipe':
        print(f'{UNIT} {hashlib.sha256(source).hexdigest()}, compiler {args[0]}', flush=True)
        return
    with (STATE / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        no_model_owners([])
        with tempfile.TemporaryDirectory(prefix='decode-mtp-prefix-', dir=STATE) as temp:
            directory = Path(temp)
            candidate = directory / 'mtp.cpp'
            candidate.write_bytes(source)
            depfile = directory / 'headers.d'
            cmd = [*args, '-iquote', str(original.parent), '-fsyntax-only', '-MD', '-MF', str(depfile),
                   '-MT', 'checked', str(candidate)]
            env = dict(row['recipe']['environment'], TMPDIR=temp, XDG_CACHE_HOME=temp)
            build = PARENT / 'build'
            subprocess.run(cmd, cwd=build, env=env, check=True, timeout=240)
            deps = shlex.split(depfile.read_text().replace('\\\n', ' ').split(':', 1)[1])
            paths = {(build / p).resolve() for p in deps}
            require(candidate in paths and original not in paths
                    and PARENT / 'source/include/strata/core/mtp.hpp' in paths,
                    'candidate or original production headers missing')
            paths.remove(candidate)
            known = row['actual_inputs']['compiled']
            require(all(known.get(str(p)) == digest(p) for p in paths), 'unqualified compiler dependency')
            print(f'MTP candidate SYCL syntax passed with {len(paths)} matched includes', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['recipe'], ['check']), 'select recipe or check')
    main(sys.argv[1])
