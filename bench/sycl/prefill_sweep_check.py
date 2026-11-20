"""Check the host-residual sweep alone against the measured ordinary prefill source."""
import fcntl
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import build_prefill_mixed_gather as fixture

bank, require = fixture.bank, fixture.require
CHANGE = '0e7cf81078c69046199af125deef0b8c071a2a2a'
UNIT = 'src/prefill/prefill.cpp'
LAYER = 'src/prefill/layer_major.inc'
PLAN = 'include/strata/prefill/layer_major_plan.hpp'
TEST = 'src/prefill/layer_major_plan_test.cpp'
EXPECTED = {
    LAYER: 'e998c59adcadeb06bf1a1f6435500682a858c877155a1f4c004039b36da476ae',
    PLAN: 'c2c5dd4318a1cdc45546478bfe57a616cf866e56380ccf6edeff6aa9b2d00a33',
    TEST: '70f8ba888d0f740aa3a08525958e66347eab2708d424efe8d2f55e32c4da3f7f',
}


def project(directory):
    overlay = {}
    for name in (UNIT, *EXPECTED):
        payload = fixture.git_file(bank.INPUT_SOURCE, name)
        require(payload == (fixture.SOURCE / name).read_bytes(), 'measured parent differs: ' + name)
        if name in EXPECTED:
            for key, data in [('original', payload), ('before', fixture.git_file(CHANGE + '^', name)),
                              ('after', fixture.git_file(CHANGE, name))]:
                (directory / key).write_bytes(data)
            payload = subprocess.check_output(['git', 'merge-file', '-p', 'original', 'before', 'after'],
                                              cwd=directory, timeout=30)
            require(hashlib.sha256(payload).hexdigest() == EXPECTED[name], 'unexpected sweep image: ' + name)
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        overlay[str(path)] = hashlib.sha256(payload).hexdigest()
    return overlay


def check(directory):
    parent = bank.previous.read(bank.ROOT / 'record.json')
    bank.check(parent)
    for script in (Path(__file__).resolve(), Path(fixture.__file__).resolve()):
        require(script.read_bytes() == fixture.git_file('HEAD', str(script.relative_to(bank.P))),
                'commit sweep qualification inputs first')
    external = parent['external_inputs']
    require(fixture.sha(external['path']) == external['sha256'], 'external input list differs')
    known = bank.previous.read(Path(external['path']))
    known.update(parent['actual_inputs']['compiled'])
    known.update(parent['actual_inputs']['links'])
    known.update({str(fixture.SOURCE / f): h for f, h in parent['source_files'].items()})
    overlay = project(directory)
    known.update(overlay)
    env = dict(parent['recipe']['environment'], TMPDIR=str(directory), XDG_CACHE_HOME=str(directory))
    units = bank.previous.read(fixture.BUILD / 'compile_commands.json')
    obj = directory / 'planner.o'
    for file, target in [(UNIT, 'strata_prefill'), (TEST, 'prefill_layer_major_plan_test')]:
        source = fixture.SOURCE / file
        unit, = [u for u in units if u['file'] == str(source) and u['output'].startswith('CMakeFiles/' + target + '.dir/')]
        args = shlex.split(unit['command'])
        compiler = args[0]
        require(unit['directory'] == str(fixture.BUILD) and args[-4:] == ['-o', unit['output'], '-c', str(source)]
                and fixture.sha(compiler) == known[compiler], 'compiler or original arguments differ')
        flags = args[1:-4]
        require(('-fsycl' in flags) == (file == UNIT) and '-MF' not in flags, 'unexpected compilation mode')
        dep = directory / 'deps.d'
        mode = ['-fsyntax-only'] if file == UNIT else ['-c', '-o', str(obj)]
        cmd = [compiler, '-I' + str(directory / 'include'), '-iquote', str(source.parent), *flags,
               '-MD', '-MF', str(dep), '-MT', 'verified', *mode, str(directory / file)]
        subprocess.run(cmd, cwd=fixture.BUILD, env=env, check=True, timeout=240)
        paths = {str((fixture.BUILD / p).resolve()) for p in
                 shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1])}
        expected = {str(directory / n) for n in ((UNIT, PLAN, LAYER) if file == UNIT else (TEST, PLAN))}
        require(paths & overlay.keys() == expected and all(str(fixture.SOURCE / n) not in paths for n in EXPECTED)
                and all(known.get(p) == fixture.sha(p) for p in paths), 'unexpected compiler dependencies')
        print(f'{target}: {len(paths)} verified compiler inputs', flush=True)
    # Use the original host link command and replace only this test object and destination.
    link = shlex.split(subprocess.check_output(['/usr/bin/ninja', '-t', 'commands', target],
                                               cwd=fixture.BUILD, text=True, timeout=20).splitlines()[-1])
    require(link[:3] == [':', '&&', compiler] and link[-5:] == [unit['output'], '-o', target, '&&', ':']
            and '-Wl,-t' in link, 'unexpected host planner link')
    exe = directory / target
    linked = subprocess.check_output([*link[2:-5], str(obj), '-o', str(exe)], cwd=fixture.BUILD,
                                     env=env, timeout=90, text=True)
    inputs = [str((fixture.BUILD / line.strip()).resolve()) for line in linked.splitlines() if line.strip()]
    require(str(obj) in inputs and all(p == str(obj) or known.get(p) == fixture.sha(p) for p in inputs),
            'host link used an unknown input')
    digest = fixture.sha(exe)
    require(os.access(exe, os.X_OK) and len(digest) == 64, 'planner executable missing')
    subprocess.run([str(exe)], cwd=directory, env=env, check=True, timeout=30)
    require(fixture.sha(exe) == digest, 'planner executable changed')
    print('Planner passed from 98305 through 262144 rows; executable SHA-256 ' + digest)


if __name__ == '__main__':
    require(sys.argv[1:] == ['check'], 'use check')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        with tempfile.TemporaryDirectory(prefix='prefill-sweep-', dir=bank.B) as root:
            check(Path(root))
