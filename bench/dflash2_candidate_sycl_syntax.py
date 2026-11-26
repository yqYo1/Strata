"""Recheck isolated DFlash2 source candidates with the recorded SYCL compiler."""

import fcntl
import json
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory

sys.dont_write_bytecode = True
import dflash2_decode_ranked_refill_build as refill
import dflash2_decode_small_kv_build as small_kv

bank, require, validation = refill.bank, refill.require, refill.validation
THIS = str(Path(__file__).resolve().relative_to(bank.P))
GENERATOR = 'src/program/generate.cpp'
SERVING_UNITS = ((GENERATOR, 'strata'), (GENERATOR, 'strata_full_context_diagnostic'))
CANDIDATES = (
    ('ranked-refill', refill, SERVING_UNITS),
    ('small-kv', small_kv, (*SERVING_UNITS, ('src/core/layer.cpp', 'strata_engine'), ('tests/sycl/kv.cpp', 'sycl_kv_test'))),
)
DRIVER_WARNING = ("icpx: warning: argument unused during compilation: '-fsycl-id-queries-range=size_t' "
                  '[-Wunused-command-line-argument]')
LIMITS = dict(wall=70, cpu=70, rss_cap=5 << 30, text_cap=128 << 10, file_cap=2 << 20, total_cap=512 << 20)


def command(database, name, target, old_tree, tree, build, depfile, toolkit):
    original, = [row for row in database if row['file'] == str(old_tree / name)
                 and row['output'] == f'CMakeFiles/{target}.dir/{name}.o']
    args = shlex.split(original['command'])
    require(original['directory'] == str(build) and args[-4:] == ['-o', original['output'], '-c', original['file']]
            and args[0] in toolkit and bank.sha(args[0]) == toolkit[args[0]] and '-fsycl' in args,
            'the recorded compiler or translation-unit recipe changed')
    return [arg.replace(str(old_tree), str(tree)) for arg in args[:-4]] + [
        '-fsyntax-only', '-MD', '-MF', str(depfile), str(tree / name)]


def dependencies(depfile, build, tree, candidate, unchanged):
    raw = depfile.read_text().replace('\\\n', '')
    _, sep, body = raw.partition(':')
    require(sep and body.strip(), 'compiler dependency manifest missing or malformed')
    files = {}
    for word in shlex.split(body):
        path = (build / word).resolve()
        if path.is_relative_to(tree):
            name = path.relative_to(tree).as_posix()
            expected = candidate.get(name)
        else:
            name = str(path)
            expected = unchanged.get(name)
        require(expected is not None and bank.sha(path) == expected, 'unverified compiler dependency: ' + str(path))
        files[name] = expected
    return files


def run():
    revision = validation.git('rev-parse', 'HEAD').decode().strip()
    _, base, original, payload, _ = refill.retained_build_inputs(revision, (THIS, *small_kv.CONTROLLERS))
    old_root = refill.primary.ROOT
    old_tree, build = old_root / 'source', old_root / 'build'
    db = build / 'compile_commands.json'
    require(bank.sha(db) == base['configured_files'][str(db)], 'the compile database changed')
    database = bank.previous.read(db)
    external = base['external_inputs']
    require(bank.sha(external['path']) == external['sha256'], 'the compiler SDK dependency list changed')
    unchanged = (bank.previous.read(Path(external['path'])) | base['actual_inputs']['compiled'] |
                 base['actual_inputs']['generated'])
    unchanged = {str(Path(p).resolve()): h for p, h in unchanged.items()}
    owner_module = refill.builder.ownership
    for label, spec, units in CANDIDATES:
        projection = payload | spec.overlay(payload)
        files = bank.stage.hashed(projection)
        with TemporaryDirectory(prefix='strata-' + label + '-syntax-') as directory:
            root = Path(directory)
            tree = root / 'source'
            for name, raw in projection.items():
                dest = tree / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(raw)
            for folder in ('tmp', 'cache'):
                (root / folder).mkdir(mode=0o700)
            env = dict(base['recipe']['environment'], TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
            owner = owner_module.Owner(root / 'owned')
            for index, (name, target) in enumerate(units):
                depfile = root / f'{index}.d'
                args = command(database, name, target, old_tree, tree, build, depfile, original['toolchain'])
                entry, out, err = owner.run(str(index), args, env, build, **LIMITS)
                stderr = err.read_text().splitlines()
                print(json.dumps(dict(candidate=label, source=spec.PATCH, unit=name, target=target,
                                      status=entry.get('exit_code'), diagnostics=stderr, controller=revision)), flush=True)
                require(owner_module.closed(entry) and entry['exit_code'] == 0 and not out.read_bytes()
                        and stderr in ([], [DRIVER_WARNING]), 'unexpected compiler failure or diagnostic')
                actual = dependencies(depfile, build, tree, files, unchanged)
                require(actual.get(name) == files[name], 'expected translation unit absent from compiler inputs')
                print(json.dumps(dict(candidate=label, target=target, dependencies=len(actual),
                                      dependencies_sha256=bank.signature(actual), verified=True)), flush=True)


if __name__ == '__main__':
    require(sys.executable == bank.basis.reference.model.PYTHON and len(sys.argv) == 1, 'use the pinned Python')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refill.builder.guards.no_model_owners([])
        run()
