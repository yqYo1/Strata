#!/usr/bin/env python3
"""Build the current comparison engine from verified Git inputs; Main only.

Uses the known-good engine's committed configure recipe, changing only the
output directory. No engine/GPU execution. An explicit rebuild reuses a closed
comparison's cache only after verifying its original inputs from Git.
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


def blobs(root, revision, names):
    """Recover exact Git blobs in bounded batches, without one process per file."""
    names = list(names)
    for start in range(0, len(names), 128):
        batch = names[start:start + 128]
        guards.require(all('\n' not in n and '\r' not in n for n in batch), 'unsupported source pathname')
        request = ''.join(revision + ':' + n + '\n' for n in batch).encode()
        data = subprocess.check_output(['git', '-C', str(root), 'cat-file', '--batch'], input=request)
        offset = 0
        for name in batch:
            end = data.index(b'\n', offset)
            fields = data[offset:end].split()
            guards.require(len(fields) == 3 and fields[1] == b'blob', 'unrecoverable source: ' + name)
            length = int(fields[2]); begin = end + 1; end = begin + length
            guards.require(length <= 256 << 20 and data[end:end + 1] == b'\n', 'Git blob extent')
            yield name, data[begin:end]
            offset = end + 1
        guards.require(offset == len(data), 'unexpected Git batch output')


def pin(root, prefixes):
    revision = git(root, 'rev-parse', 'HEAD').decode().strip()
    paths = git(root, 'ls-files', '-z', '--', *prefixes).decode().strip('\0').split('\0')
    guards.require(paths and all(paths), 'empty source selection')
    pins = {}
    for name, data in blobs(root, revision, sorted(paths)):
        path = root / name
        guards.require(path.read_bytes() == data, 'uncommitted build input: ' + str(path))
        pins[name] = hashlib.sha256(data).hexdigest()
    extras = git(root, 'ls-files', '--others', '--exclude-standard', '-z', '--', *prefixes).decode().split('\0')
    guards.require(not any(x and Path(x).suffix in ('.h', '.hpp', '.cpp', '.c', '.cmake') for x in extras),
                   'untracked source in build tree: ' + str(root))
    aggregate = hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(root=str(root), revision=revision, count=len(pins), source_sha256=pins, aggregate_sha256=aggregate)


def main():
    guards.require(len(sys.argv) == 1 or sys.argv[1:] in (['rebuild'], ['rebuild-attention']),
                   'use build_engine.py [rebuild|rebuild-attention]')
    rebuild = len(sys.argv) == 2
    attention = sys.argv[1:] == ['rebuild-attention']
    targets = ['strata']
    if attention:
        targets += ['sycl_qsa_bf16_late_flags_test', 'sycl_qsa_bf16_prefix_optimized_test',
                    'sycl_qsa_bf16_prefix_rows_test', 'sycl_qsa_bf16_prefix_rows_bench']
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        device = guards.driver_guard()
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        guards.require(ROOT.exists() == rebuild, 'use rebuild only for the closed comparison cache')
        previous = None
        if rebuild:
            committed = git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P)))
            guards.require(committed == RESULT.read_bytes(), 'commit the previous build identity first')
            old = json.loads((ROOT / 'record.json').read_text())
            guards.require(old['passed'] and old['ownership_closed'] and not old['active'], 'previous build not closed')
            guards.require(digest(old['binary']['path']) == old['binary']['sha256'], 'previous executable changed')
            for source in [old['project'], *old['dependencies']]:
                for name, data in blobs(Path(source['root']), source['revision'], source['source_sha256']):
                    guards.require(hashlib.sha256(data).hexdigest() == source['source_sha256'][name],
                                   'previous inputs not recoverable')
            previous = dict(source_commit=old['source_commit'], binary=old['binary'],
                            reason='Closed unused comparison engine; original source/recipe recovered from Git. '
                                   'Reuse its cache for the immediate corrected build, not a retained old generation.')
        project = pin(P, ['CMakeLists.txt', 'cmake', 'include', 'src', 'tests',
                          'tools/sycl/build_engine.py', 'tools/sycl/measurement_owner.py',
                          'tools/sycl/check_attention.py',
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
        ROOT.mkdir(mode=0o700, exist_ok=rebuild)
        for name in ('tmp', 'cache'):
            (ROOT / name).mkdir(exist_ok=rebuild)
        env = dict(baseline['environment'])
        env.update(XDG_CACHE_HOME=str(ROOT / 'cache'), TMPDIR=str(ROOT / 'tmp'))
        # Configuration/build never executes the model. The future first model
        # invocation must apply the qualified tracing/validation environment.
        record = dict(source_commit=project['revision'], project=project, dependencies=dependencies,
                      compiler=dict(path=str(compiler), sha256=digest(compiler)),
                      configure_argv=argv, environment=env, device=device,
                      recipe_reference=dict(path=str(BASELINE.relative_to(P)), sha256=digest(BASELINE)),
                      complete=False, passed=False, active=False, GPU_executed=False,
                      production_selected=False, model_quality_qualified=False, replaces=previous,
                      targets=targets,
                      retention_consumer='Current long-prefill comparison and its attention fixtures; retire when comparison closes.')
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            if owner:
                record['commands'] = owner.commands
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = supervision.Owner(ROOT / ('owned-' + project['revision'][:12]), persist)
        persist()
        try:
            for label, command in [('configure', argv), ('compile-link',
                    ['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', *targets, '-j', '4'])]:
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
            record['fixtures'] = {
                name: dict(path=str(ROOT / 'build' / name), sha256=digest(ROOT / 'build' / name))
                for name in targets if name != 'strata'}
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
