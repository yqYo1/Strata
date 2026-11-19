"""Check the partial-bank host fixture with its committed sources and pinned Nix toolchain."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


PROJECT = Path(__file__).resolve().parents[2]
STATE = Path.home() / '.local/state/strata-sycl/post-reboot-tuning-20261007'
TOOLCHAIN = Path('/nix/store/i52d7j6ma8bh4ggg4r2hl4af7bychdpa-clang-wrapper-21.1.8')
CLOSURE = 'b6ea6ec00d9d0023b9397f0024278bdac1f42ac09632b20bd7e4d5cde19ac0c1'
FIXTURE = 'tests/prefill/fullfit_upload_plan_test.cpp'
HEADERS = ['include/strata/prefill/' + name + '.hpp' for name in
           ('fullfit_upload_plan', 'expert_bank_plan', 'layer_major_plan')]


def run(*args, env=None):
    return subprocess.check_output(args, cwd=PROJECT, timeout=60, env=env)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    revision = run('git', 'rev-parse', 'HEAD').decode().strip()
    files = [FIXTURE, *HEADERS, str(Path(__file__).resolve().relative_to(PROJECT))]
    pins = {f: digest(run('git', 'show', revision + ':' + f)) for f in files}
    closure = sorted(run('nix-store', '-qR', TOOLCHAIN).decode().splitlines())
    check(digest(('\n'.join(closure) + '\n').encode()) == CLOSURE, 'compiler closure changed')
    for name, expected in pins.items():
        check(digest((PROJECT / name).read_bytes()) == expected, 'source changed: ' + name)
    env = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}
    with tempfile.TemporaryDirectory(prefix='prefill-demand-fit-host-', dir=STATE) as directory:
        root = Path(directory)
        env['TMPDIR'] = str(root)
        target = root / 'host'
        deps = root / 'host.d'
        run(TOOLCHAIN / 'bin/clang++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
            '-Iinclude', '-MD', '-MF', deps, FIXTURE, '-o', target, env=env)
        includes = deps.read_text().replace('\\\n', '').split(':', 1)[1].split()
        local = set()
        for name in includes:
            p = (PROJECT / name).resolve()
            if p.is_relative_to(PROJECT):
                local.add(str(p.relative_to(PROJECT)))
            else:
                check(any(str(parent) in closure for parent in p.parents), 'untracked include: ' + name)
        check(local == {FIXTURE, *HEADERS}, 'project dependency closure changed')
        for name, expected in pins.items():
            check(digest((PROJECT / name).read_bytes()) == expected, 'compiled source changed: ' + name)
        record = dict(source=revision, project=pins, toolchain_closure=CLOSURE,
                      dependencies=len(set(includes)), executable=digest(target.read_bytes()))
        receipt = root / 'build.json'
        receipt.write_text(json.dumps(record))
        check(digest(target.read_bytes()) == json.loads(receipt.read_text())['executable'],
              'host executable differs from build')
        output = run(target, env=env).decode().splitlines()
        check('prefill demand-fit: exact fit, overflow, scratch and failed publication passed' in output,
              'missing partial-bank completion')
        print(json.dumps(dict(record, output=output), sort_keys=True))


if __name__ == '__main__':
    with (STATE / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
