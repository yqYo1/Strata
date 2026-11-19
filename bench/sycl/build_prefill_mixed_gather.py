"""Build the scalar/vec8 prefill gather fixture against the measured B570 runtime."""

import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'tools/sycl'))
import prefill_resident_input_build as bank

ROOT = bank.B / 'prefill-mixed-gather-dispatch-v1'
BUILD = bank.ROOT / 'build'
SOURCE = bank.ROOT / 'source'
BASE = '0ed10ef1d81428ce94e463494d8eb43a1c5fa33b'
CANDIDATE = 'c386a8051714b374013a8a2f55e3834591fa8356'
HEADER = 'include/strata/prefill/native16.hpp'
KERNEL = 'src/prefill/sycl/native16.cpp'
FIXTURE = 'tests/sycl/moe_accum.cpp'
TARGET = 'sycl_moe_accum_test'
ARCHIVE = 'libstrata_prefill.a'
require, sha, write = bank.require, bank.sha, bank.stage.write


def git_file(revision, name):
    return subprocess.check_output(['git', 'show', revision + ':' + name], cwd=PROJECT, timeout=30)


def recipe(parent):
    overlay = {}
    for name in (HEADER, KERNEL, FIXTURE):
        before = git_file(BASE, name)
        after = git_file(CANDIDATE, name)
        require(before == (SOURCE / name).read_bytes() and after == (PROJECT / name).read_bytes(),
                'gather source differs from the declared parent or candidate: ' + name)
        overlay[name] = after
    # Only the vec8 declaration and namespace-level selection policy were added.
    # Keep all class definitions identical across the reused translation units.
    header = overlay[HEADER]
    start = header.index(b'// Eight raw FP16 values per work item.\n'
                         b'void prefill_moe_gather_mixed_vec8(')
    end = header.index(b'void moe_accumulate(', start)
    require(hashlib.sha256(header[start:end]).hexdigest() ==
            '48a8668d7d54f6d019689abfdab5bdbf82263b7e8a4b5d3a411714651663a1b9'
            and header[:start] + header[end:] == git_file(BASE, HEADER), 'shared header definition changed')
    units = json.loads((BUILD / 'compile_commands.json').read_bytes())
    objects, commands = {}, []
    for name in (KERNEL, FIXTURE):
        unit, = [u for u in units if u['file'] == str(SOURCE / name)]
        argv = shlex.split(unit['command'])
        require(Path(unit['directory']) == BUILD and argv[-4:] == ['-o', unit['output'], '-c', unit['file']],
                'unexpected production compile recipe')
        obj = ROOT / Path(unit['output']).name
        objects[name] = unit['output']
        commands.append([argv[0], '-I' + str(ROOT / 'include'), *argv[1:-4], '-MD', '-MF',
                         str(obj) + '.d', '-o', str(obj), '-c', str(ROOT / name)])

    members = subprocess.check_output(['/usr/bin/ar', 't', ARCHIVE], cwd=BUILD, text=True, timeout=20).splitlines()
    original = {Path(p).name: Path(p) for p in parent['actual_inputs']['objects']
                if Path(p).is_relative_to(BUILD / 'CMakeFiles/strata_prefill.dir')}
    require(len(members) == len(set(members)) == len(original) and set(members) == original.keys()
            and original['native16.cpp.o'] == BUILD / objects[KERNEL], 'prefill archive member identity differs')
    original['native16.cpp.o'] = ROOT / 'native16.cpp.o'
    commands.append(['/usr/bin/ar', 'rcsD', str(ROOT / ARCHIVE), *(str(original[m]) for m in members)])
    link = subprocess.check_output(['/usr/bin/ninja', '-t', 'commands', TARGET],
                                   cwd=BUILD, text=True, timeout=20).splitlines()[-1]
    words = shlex.split(link)
    require(words[:2] == [':', '&&'] and words[-2:] == ['&&', ':'] and words.count(ARCHIVE) == 1
            and words.count(objects[FIXTURE]) == 1 and words[words.index('-o') + 1] == TARGET,
            'unexpected CMake fixture link inputs')
    replacements = {objects[FIXTURE]: str(ROOT / Path(objects[FIXTURE]).name), TARGET: str(ROOT / TARGET),
                    ARCHIVE: str(ROOT / ARCHIVE)}
    commands.append([words[2], '-v', *(replacements.get(w, w) for w in words[3:-2])])
    return overlay, commands


def fixture_host_slices(log, fixture_object):
    """Extract the host output of each actual icpx unbundle for this fat object."""
    slices = set()
    for line in log.splitlines():
        if 'clang-offload-bundler' not in line or '-unbundle' not in line:
            continue
        words = shlex.split(line)
        if '-input=' + str(fixture_object) not in words or '-unbundle' not in words:
            continue
        targets, = [w.split('=', 1)[1].split(',') for w in words if w.startswith('-targets=')]
        outputs = [w.split('=', 1)[1] for w in words if w.startswith('-output=')]
        require(len(outputs) == len(targets), 'unbundle target/output count differs')
        host, = [Path(p) for target, p in zip(targets, outputs) if target.startswith('host-')]
        require(host.is_relative_to(ROOT / 'tmp') and host.suffix == '.o', 'host slice escaped compiler scratch')
        slices.add(str(host))
    require(slices, 'missing icpx fixture host unbundle')
    return slices


def main(action):
    screen = bank.basis.validation.committed(bank.RESULT, ('prefill_mixed_gather_component_v1',))
    require(screen['faster_each_process'] == [False, False, True, True]
            and [s['stage'] for s in screen['stages']] == ['contiguous', 'ragged', '0', '1', '2']
            and all(s['passed'] for s in screen['stages'])
            and all([(p['rows'], p['cluster']) for p in s['samples']] ==
                    [(211, 1), (211, 16), (6640, 1), (6640, 16)]
                    for s in screen['stages'][2:]), 'complete the raw gather comparison first')
    parent = bank.previous.read(bank.ROOT / 'record.json')
    bank.check(parent)
    require(parent['stage'] == 'built' and parent['passed'] and parent['ownership_closed']
            and parent['binaries']['strata']['sha256'] ==
            'b35d66a4f29737a47abcdd7180c80eaf838b8658ec49ca2586948198d6a7613f', 'qualified parent required')
    name = str(Path(__file__).resolve().relative_to(PROJECT))
    require(git_file('HEAD', name) == Path(__file__).read_bytes(), 'commit the exact build driver first')
    overlay, commands = recipe(parent)
    actual = parent['actual_inputs']
    allowed = {p: h for group in ('compiled', 'generated', 'links', 'objects', 'produced')
               for p, h in actual[group].items()}
    external = parent['external_inputs']
    require(sha(external['path']) == external['sha256'], 'external input record changed')
    allowed.update(json.loads(Path(external['path']).read_bytes()))
    allowed.update({str(SOURCE / p): h for p, h in parent['source_files'].items()})
    allowed.update({str(ROOT / p): hashlib.sha256(raw).hexdigest() for p, raw in overlay.items()})
    for argv in commands:
        tool = Path(argv[0]).resolve()
        require(allowed.get(str(tool), allowed.get(argv[0])) == sha(tool), 'compiler or archiver differs')
    if action == 'check':
        print(f'{len(overlay)} source files, two production compile recipes, one archive and one fixture link')
        return

    ROOT.mkdir(mode=0o700)
    for folder in ('tmp', 'cache'):
        (ROOT / folder).mkdir(mode=0o700)
    for path, raw in overlay.items():
        dest = ROOT / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    env = dict(parent['recipe']['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    receipt = dict(driver=sha(__file__), candidate=CANDIDATE, base=BASE,
                   driver_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
                   parent=bank.signature(bank.builder.compact(parent)), commands=commands, environment=env,
                   complete=False, ownership_closed=False)
    child = bank.builder.ownership.Owner(ROOT / 'owned', lambda: write(ROOT / 'execution.json', child.commands))
    write(ROOT / 'record.json', receipt)
    try:
        for index, argv in enumerate(commands):
            if index == len(commands) - 1:
                driver_inputs = {name: sha(ROOT / name) for name in (Path(FIXTURE).name + '.o', ARCHIVE)}
            entry, out, err = child.run(f'command-{index}', argv, env, BUILD, **bank.builder.LIMITS)
            require(bank.builder.ownership.closed(entry) and entry['exit_code'] == 0,
                    'fixture build failed: ' + err.read_text()[-1200:])
        compiled = {}
        for unit in (KERNEL, FIXTURE):
            depfile = ROOT / (Path(unit).name + '.o.d')
            deps = shlex.split(depfile.read_text().replace('\\\n', ' ').split(':', 1)[1])
            for dep in deps:
                path = (BUILD / dep).resolve()
                require(allowed.get(str(path)) == sha(path), 'unverified compiler input: ' + str(path))
                compiled[str(path)] = allowed[str(path)]
        require(all(str(ROOT / p) in compiled for p in overlay) and len(compiled) > 100,
                'incomplete gather compiler evidence')
        linked = {}
        require(all(sha(ROOT / name) == digest for name, digest in driver_inputs.items()),
                'fixture compiler input changed during link')
        host_slices = fixture_host_slices(err.read_text(), ROOT / (Path(FIXTURE).name + '.o'))
        traced_slices = set()
        for line in out.read_text().splitlines():
            token = line.strip()
            if not token or any(c.isspace() for c in token) or len(token) > 4096:
                continue
            path = (BUILD / token).resolve()
            if not path.is_file():
                if Path(token).is_absolute():
                    require(path.is_relative_to(ROOT / 'tmp') and path.suffix == '.o',
                            'unidentified linker input: ' + token)
                    if str(path) in host_slices:
                        traced_slices.add(str(path))
                continue
            digest = sha(path)
            require(path == ROOT / ARCHIVE or allowed.get(str(path)) == digest, 'unverified link input: ' + str(path))
            linked[str(path)] = digest
        require(traced_slices == host_slices and linked.get(str(ROOT / ARCHIVE)) == driver_inputs[ARCHIVE]
                and len(linked) > 10, 'incomplete fixture link trace')
        receipt.update(compiled=compiled, linked=linked, driver_inputs=driver_inputs,
                       fixture_host_slices=len(traced_slices),
                       binary=dict(path=str(ROOT / TARGET), sha256=sha(ROOT / TARGET)),
                       complete=True)
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        receipt['ownership_closed'] = child.active is None and all(bank.builder.ownership.closed(c) for c in child.commands)
        write(ROOT / 'record.json', receipt)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('check', 'build'), 'use check or build')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        main(action)
