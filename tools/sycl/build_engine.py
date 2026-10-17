#!/usr/bin/env python3
"""Build the current comparison engine from verified Git inputs; Main only.

Uses the known-good engine's committed configure recipe, changing only the
output directory. No engine/GPU execution or overwriting an existing build.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'main-native-gdn-engine'
BASELINE = P / 'bench/results/2026-10-15-pr6-recovery-audit/qsa-score-gemm-entry-engine-build-v2.json'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/native-gdn-engine-build.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(root, prefixes):
    revision = git(root, 'rev-parse', 'HEAD').decode().strip()
    paths = git(root, 'ls-files', '-z', '--', *prefixes).decode().strip('\0').split('\0')
    guards.require(paths and all(paths), 'empty source selection')
    pins = {}
    for name in sorted(paths):
        path = root / name
        data = git(root, 'show', revision + ':' + name)
        guards.require(path.read_bytes() == data, 'uncommitted build input: ' + str(path))
        pins[name] = hashlib.sha256(data).hexdigest()
    extras = git(root, 'ls-files', '--others', '--exclude-standard', '-z', '--', *prefixes).decode().split('\0')
    guards.require(not any(x and Path(x).suffix in ('.h', '.hpp', '.cpp', '.c', '.cmake') for x in extras),
                   'untracked source in build tree: ' + str(root))
    aggregate = hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(root=str(root), revision=revision, count=len(pins), source_sha256=pins, aggregate_sha256=aggregate)


def main():
    guards.require(len(sys.argv) == 1, 'build_engine.py takes no options')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        device = guards.driver_guard()
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        guards.require(not ROOT.exists(), 'retire the previous comparison build before rebuilding')
        project = pin(P, ['CMakeLists.txt', 'cmake', 'include', 'src', 'tests',
                          'tools/sycl/build_engine.py', 'tools/sycl/measurement_owner.py',
                          'tools/sycl/measurement_guards.py', str(SETTINGS.relative_to(P)),
                          str(BASELINE.relative_to(P))])
        baseline = json.loads(BASELINE.read_text())
        argv = list(baseline['configure_argv'])
        argv[argv.index('-B') + 1] = str(ROOT / 'build')
        dependencies = []
        for prefix in ('-DSTRATA_GGML_DIR:PATH=', '-DSYCL_TLA_SOURCE_DIR='):
            root = Path(next(a[len(prefix):] for a in argv if a.startswith(prefix)))
            dependencies.append(pin(root, ['.']))
        settings = json.loads(SETTINGS.read_text())
        compiler = Path(next(a.split('=', 1)[1] for a in argv if a.startswith('-DCMAKE_CXX_COMPILER=')))
        guards.require(digest(compiler) == settings['compiler_sha256'], 'compiler changed')
        ROOT.mkdir(mode=0o700)
        for name in ('tmp', 'cache'):
            (ROOT / name).mkdir()
        env = dict(baseline['environment'])
        env.update(XDG_CACHE_HOME=str(ROOT / 'cache'), TMPDIR=str(ROOT / 'tmp'))
        # Configuration/build never executes the model. The future first model
        # invocation must apply the qualified tracing/validation environment.
        record = dict(source_commit=project['revision'], project=project, dependencies=dependencies,
                      compiler=dict(path=str(compiler), sha256=digest(compiler)),
                      configure_argv=argv, environment=env, device=device,
                      recipe_reference=dict(path=str(BASELINE.relative_to(P)), sha256=digest(BASELINE)),
                      complete=False, passed=False, active=False, GPU_executed=False,
                      production_selected=False, model_quality_qualified=False,
                      retention_consumer='Imminent native GDN opt-in model correctness/performance A/B; retire after comparison.')
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            if owner:
                record['commands'] = owner.commands
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = supervision.Owner(ROOT / 'owned', persist)
        persist()
        try:
            for label, command in [('configure', argv), ('compile-link',
                    ['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', 'strata', '-j', '4'])]:
                entry, out, err = owner.run(label, command, env, P, wall=1800, cpu=7200,
                    rss_cap=12 << 30, file_cap=256 << 20, text_cap=64 << 20, total_cap=4 << 30)
                guards.require(supervision.closed(entry) and entry.get('exit_code') == 0,
                               label + ' failed: ' + err.read_text(errors='replace')[-8192:])
            # Detect input changes made while compilation was in progress.
            for source in [project, *dependencies]:
                for name, expected in source['source_sha256'].items():
                    guards.require(digest(Path(source['root']) / name) == expected, 'build input changed: ' + name)
            database = ROOT / 'build/compile_commands.json'
            entries = json.loads(database.read_text())
            roots = [P, ROOT, *(Path(d['root']) for d in dependencies)]
            guards.require(all(any(Path(e['file']).resolve().is_relative_to(root) for root in roots)
                               for e in entries), 'unqualified configured source root')
            for name in ('src/prefill/prefill.cpp', 'src/prefill/sycl/gdn_chunk64.cpp'):
                guards.require(any(Path(e['file']).resolve() == P / name for e in entries), 'missing configured source: ' + name)
            binary = ROOT / 'build/strata'
            guards.require(binary.is_file(), 'no engine executable')
            guards.driver_guard()
            guards.require(boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'boot changed')
            record.update(binary=dict(path=str(binary), bytes=binary.stat().st_size, sha256=digest(binary)),
                          compile_database_sha256=digest(database), configured_compile_entries=len(entries),
                          complete=True, passed=True)
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            if not record['ownership_closed']:
                record['passed'] = False
            persist()
            compact = {k: v for k, v in record.items() if k not in ('project', 'dependencies', 'commands')}
            compact['inputs'] = [{k: v for k, v in source.items() if k != 'source_sha256'}
                                 for source in [project, *dependencies]]
            RESULT.write_text(json.dumps(compact, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('source_commit', 'complete', 'passed', 'ownership_closed', 'error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
