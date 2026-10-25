#!/usr/bin/env python3
"""Build the actual Strata oneMKL wrapper probe; no engine/GPU execution.

Main-authored, repository LICENSE. Reuse the selected engine's recipe and
independent SDK/dependencies; keep its13 validated native artifacts untouched.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import build_full_context as lineage
import runtime_probe as r

build = r.build
P, B = r.P, r.B
ROOT = B / 'onemkl-wrapper-work-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/prefill-resources-current.json'
KEY = 'oneMKL_wrapper_build'
TARGET = 'sycl_gemm_device_probe'
PREFIXES = ['CMakeLists.txt', 'cmake', 'include', 'src', 'tests', 'third_party', 'bench/sycl',
    'tools/sycl/gemm_device_probe.cpp', 'tools/sycl/onemkl_work.py', 'tools/sycl/build_full_context.py',
    'tools/sycl/runtime_probe.py', 'tools/sycl/build_engine.py', 'tools/sycl/check_full_context_source.py',
    'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_guards.py', 'tools/sycl/ur_validation.py',
    'tools/sycl/measurement_environment.json']
require, digest, write = r.require, r.digest, r.write


def summary(row):
    out = {k: v for k, v in row.items() if k not in ('project', 'dependencies', 'environment', 'commands', 'external_inputs')}
    out['source_groups'] = [{k: s[k] for k in ('root', 'revision', 'count', 'aggregate_sha256')}
                            for s in [row['project'], *row['dependencies']]]
    return out


def external_inputs(rt):
    mkl = Path('/opt/intel/oneapi/mkl/2026.1')
    paths = {p.resolve() for d in ('include', 'lib/cmake/mkl') for p in (mkl / d).rglob('*') if p.is_file()}
    paths.update(Path(p) for p in rt['runtime_files'])
    # Imported MKL link names can resolve to their versioned runtime libraries.
    paths.update(p.resolve() for p in (mkl / 'lib').glob('libmkl*.so') if p.is_file())
    return {str(p): digest(p) for p in sorted(paths)}


def current(row):
    require(build.pin(P, PREFIXES)['source_sha256'] == row['project']['source_sha256'], 'build/controller source changed')
    for dep in row['dependencies']:
        require(build.pin(Path(dep['root']), ['.']) == dep, 'independent dependency changed')
    for p, h in row['external_inputs'].items(): require(digest(p) == h, 'external build input changed: ' + p)
    for item in row['tools'].values(): require(digest(item['path']) == item['sha256'], 'build tool changed')
    require(lineage.sdk() == row['sdk'], 'SDK changed')
    rt = r.validation.committed(r.RESULT)
    full = r.current(rt)
    require(digest(r.RESULT) == row['runtime_sha256'] and full['binary'] == row['selected_binary'], 'selected runtime/engine changed')


def dependencies(row, text):
    expected = {}
    for s in [row['project'], *row['dependencies']]:
        expected.update({str((Path(s['root']) / n).resolve()): h for n, h in s['source_sha256'].items()})
    expected.update({str(Path(v['path']).resolve()): v['sha256'] for v in json.loads(lineage.SDK.read_text())})
    expected.update(row['external_inputs'])
    paths = {Path(v.strip()) for v in text.splitlines() if v.startswith('    ') and v.strip()}
    paths = {(p if p.is_absolute() else ROOT / 'build' / p).resolve() for p in paths}
    require(P / 'src/prefill/sycl/gemm.cpp' in paths and P / 'tools/sycl/gemm_device_probe.cpp' in paths and
        len(paths) >= 100, 'incomplete actual compiler-input closure')
    observed, generated = {}, {}
    for p in sorted(paths):
        sha = digest(p)
        if str(p) in expected:
            require(sha == expected[str(p)], 'compiled input changed: ' + str(p))
        else:
            require(p.is_relative_to(ROOT), 'compiled input absent from prebuild pins: ' + str(p))
            generated[str(p)] = sha
        observed[str(p)] = sha
    manifest = ROOT / 'compiler-inputs.json'; write(manifest, observed)
    return dict(files=len(observed), manifest=dict(path=str(manifest), sha256=digest(manifest)),
        generated_sha256=generated, uncovered_files=0)


def main():
    container = r.validation.committed(RESULT)
    require(not ROOT.exists() and KEY not in container['event_reconciliation'], 'fresh single-use build only;inspect existing owner')
    route = container['event_reconciliation']['route_discovery']
    require(route['passed'] and route['ownership_closed'] and not route['active'] and route['fresh_faults'] == [],
            'closed selected117K route required')
    rt = r.validation.committed(r.RESULT); full = r.current(rt)
    project = build.pin(P, PREFIXES)
    deps = [build.pin(Path(d['root']), ['.']) for d in full['dependencies']]
    require(all(a['source_sha256'] == b['source_sha256'] for a, b in zip(deps, full['dependencies'])), 'selected dependencies differ')
    for name in ('src/prefill/sycl/gemm.cpp', 'src/prefill/sycl/gemm16.cpp', 'src/sycl/runtime.cpp', 'src/sycl/compat.cpp'):
        require(build.git(P, 'show', '6e40afc1beddd059ee518617f6c1efce1da14aca:' + name) == (P / name).read_bytes(),
                'called production wrapper/runtime changed: ' + name)
    sdk = lineage.sdk(); external = external_inputs(rt)
    tool_paths = {n: str(Path('/usr/bin/' + n).resolve()) for n in ('cmake', 'ninja', 'cc', 'ar', 'ranlib', 'as', 'ld')}
    tool_paths['cc1'] = subprocess.check_output(['/usr/bin/cc', '-print-prog-name=cc1'], text=True, timeout=10).strip()
    tool_paths['sycl'] = full['compiler']['path']
    tools = {n: dict(path=p, sha256=digest(p)) for n, p in tool_paths.items()}
    argv = list(full['configure_argv']); argv[argv.index('-B') + 1] = str(ROOT / 'build')
    require(argv[argv.index('-S') + 1] == str(P), 'unexpected source resolver')
    env = dict(full['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    row = dict(stage='building', source_commit=project['revision'], project=project, dependencies=deps,
        sdk=sdk, external_inputs=external, tools=tools, configure_argv=argv, environment=env,
        target=TARGET, selected_binary=full['binary'], runtime_sha256=digest(r.RESULT),
        boot=rt['boot'], driver=rt['driver'], active=False, ownership_closed=True, passed=False,
        GPU_executed=False, inference_changed=False, engine_built=False,
        budgets=dict(wall=1800, cpu=7200, rss_cap=12 << 30, text_cap=32 << 20,
                     file_cap=256 << 20, total_cap=4 << 30),
        question='Build current production Gemm::bf16 wrapper with one64x320x10240 BF16 HC-down call, '
            'FP32 strided output,exact full-output oracle,input/output guards and checked drain. '
            'The existing probe default/smoke modes remain;no GPU execution in this build step.',
        retention='Selected original engine+12fixtures untouched. One extra wrapper fixture only for imminent '
            'oneMKL all-thread UR/L0/PTI coverage. Keep build cache only through build/first-observer admission; '
            'retire normal logs/objects/JIT after source/recipe recovery and fixture selection. '
            'No model speed,observer cost,hardware bottleneck,decode preservation or adoption claim.')
    owner = None
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(row, commands=owner.commands if owner else []))
    try:
        owner = r.supervision.Owner(ROOT / 'owned', persist); persist()
        for label, command in (
            ('configure', argv),
            ('compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', TARGET, '-j', '4']),
            ('dependencies', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'])):
            entry, out, err = owner.run(label, command, env, P, **row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code'] == 0,
                label + ' failed: ' + (out.read_text(errors='replace') + err.read_text(errors='replace'))[-4096:])
            if label == 'dependencies': row['compiled_inputs'] = dependencies(row, out.read_text())
        binary = ROOT / 'build' / TARGET
        require(binary.is_file() and not (ROOT / 'build/strata').exists(), 'only wrapper fixture must be built')
        ninja = (ROOT / 'build/build.ninja').read_text()
        stanza = ninja[ninja.index('build ' + TARGET + ':'):].split('\n\n', 1)[0]
        links = shlex.split(next(v.split(' = ', 1)[1] for v in stanza.splitlines() if v.startswith('  LINK_LIBRARIES = ')))
        link_inputs = {**{str(Path(v['path']).resolve()): v['sha256'] for v in json.loads(lineage.SDK.read_text())}, **external}
        for path in (Path(p).resolve() for p in links if p.startswith('/')):
            require(str(path) in link_inputs and digest(path) == link_inputs[str(path)], 'imported library not pinned: ' + str(path))
        current(row)
        row.update(stage='built-not-executed', passed=True, binary=dict(path=str(binary), sha256=digest(binary), bytes=binary.stat().st_size))
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row['ownership_closed'] = owner is None or owner.active is None and all(r.supervision.closed(c) for c in owner.commands)
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        persist(); container['event_reconciliation'][KEY] = summary(row); write(RESULT, container)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'binary', 'error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    require(sys.argv[1:] == ['build'], 'onemkl_work.py build')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r.guards.no_model_owners([]); r.guards.driver_guard()
        sys.exit(main())
