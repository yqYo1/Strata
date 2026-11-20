"""Project the KV reload change alone onto the measured 117K engine and check its real SYCL compile unit."""
import fcntl
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import build_prefill_mixed_gather as fixture

bank, require = fixture.bank, fixture.require
FILE = 'src/prefill/prefill.cpp'
CHANGE = 'fea0f3c79f71297e03090c7d1c690b9e79587d85'
EXPECTED = '5b9824473c8e4e793d072bbacbebfb7189080187371df6969a7f30ba348fc610'


def isolated_source(original, directory):
    require(original == fixture.git_file(bank.INPUT_SOURCE, FILE), 'expected the measured prefill translation unit')
    for name, payload in [('original', original), ('before', fixture.git_file(CHANGE + '^', FILE)),
                          ('after', fixture.git_file(CHANGE, FILE))]:
        (directory / name).write_bytes(payload)
    source = subprocess.check_output(['git', 'merge-file', '-p', 'original', 'before', 'after'],
                                     cwd=directory, timeout=30)
    require(hashlib.sha256(source).hexdigest() == EXPECTED, 'the isolated KV stage projection changed')
    return source


def check():
    parent = bank.previous.read(bank.ROOT / 'record.json')
    bank.check(parent)
    here = Path(__file__).resolve()
    require(here.read_bytes() == fixture.git_file('HEAD', str(here.relative_to(bank.P))),
            'commit the projection and compiler check first')
    original = fixture.SOURCE / FILE
    unit, = [u for u in bank.previous.read(fixture.BUILD / 'compile_commands.json')
             if u['file'] == str(original) and u['output'].startswith('CMakeFiles/strata_prefill.dir/')]
    args = shlex.split(unit['command'])
    require(unit['directory'] == str(fixture.BUILD) and args[-4:] == ['-o', unit['output'], '-c', str(original)]
            and '-fsycl' in args and '-Werror' in args, 'expected the measured SYCL prefill compiler command')
    with tempfile.TemporaryDirectory(prefix='prefill-kv-stage-check-', dir=bank.B) as raw:
        directory = Path(raw)
        source = directory / 'prefill.cpp'
        source.write_bytes(isolated_source(original.read_bytes(), directory))
        depfile = directory / 'headers.d'
        cmd = [*args[:-4], '-iquote', str(original.parent), '-fsyntax-only', '-MD', '-MF', str(depfile),
               '-MT', 'checked', str(source)]
        env = dict(parent['recipe']['environment'], TMPDIR=raw, XDG_CACHE_HOME=raw)
        subprocess.run(cmd, cwd=fixture.BUILD, env=env, check=True, timeout=240)
        deps = shlex.split(depfile.read_text().replace('\\\n', ' ').split(':', 1)[1])
        dependencies = {str((fixture.BUILD / path).resolve()) for path in deps}
        require(str(source) in dependencies and str(original.parent / 'layer_major.inc') in dependencies,
                'compiler omitted the projected source or layer-major implementation')
        dependencies.remove(str(source))
        known = parent['actual_inputs']['compiled']
        require(all(path in known and fixture.sha(path) == known[path] for path in dependencies),
                'a consumed include differs from the measured compile')
        print(f'Isolated SYCL syntax check passed: {EXPECTED}, {len(dependencies)} matched includes')


if __name__ == '__main__':
    require(sys.argv[1:] == ['check'], 'use check')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        check()
