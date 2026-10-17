#!/usr/bin/env python3
"""Main-owned private score GEMM fixture; reuse the qualified engine archives."""
import datetime, fcntl, hashlib, importlib.util, json, shlex, sys
from pathlib import Path
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')

def pin(path):
    p = Path(path)
    with p.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(p), realpath=str(p.resolve()), bytes=p.stat().st_size, sha256=digest)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def main():
    root = B / 'main-qsa-score-gemm-probe-build-v1'
    out = A / 'qsa-score-gemm-probe-build-v1.json'
    assert not root.exists() and not out.exists()
    gates = [A / f'qsa-score-headparallel-model-headparallel{i}-r0-tokens65536-warm4-v1.json' for i in (0, 1)]
    for path in gates:
        r = json.loads(path.read_text())
        assert r['passed'] and r['complete'] and r['ownership_closed'] and not r['active'] and not r['fresh_faults'] and not r['driver_dump_exists']
    engine_path = A / 'qsa-score-headparallel-engine-build-v1.json'
    engine = json.loads(engine_path.read_text())
    assert engine['passed'] and engine['complete'] and engine['ownership_closed']
    build = Path(engine['binary']['path']).parent
    compiler = pin(L / 'bin/clang++')
    assert compiler['sha256'] == '1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
    owner_path = A / 'direct_owner_transient_artifacts_v1.py'
    assert pin(owner_path)['sha256'] == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owned = module('score_gemm_build_owner', owner_path)
    package = A / 'own-vllm-xe2-qualification-v1'
    meta = json.loads((package / 'manifest.json').read_text())
    helper = package / 'main_controller.py'
    assert pin(helper)['sha256'] == meta['package_files']['main_controller.py']['sha256']
    guards = module('score_gemm_build_guards', helper)
    candidate = A / 'qsa-score-gemm-prototype-v1'
    proposal = json.loads((candidate / 'proposal.json').read_text())
    for name, digest in proposal['files'].items():
        assert pin(candidate / name)['sha256'] == digest
    fixture = json.loads((candidate / 'fixture-proposal.json').read_text())
    assert pin(candidate / 'fixture.cpp')['sha256'] == fixture['fixture']['sha256']
    ninja = (build / 'build.ninja').read_text()
    start = ninja.index('build strata: ')
    block = ninja[start:ninja.index('\n\n', start)]
    libraries = shlex.split(next(line.split(' = ', 1)[1] for line in block.splitlines() if line.startswith('  LINK_LIBRARIES = ')))
    # Freeze every archive, isolated device object and explicit shared-library input.
    link_inputs = [build / item if not Path(item).is_absolute() else Path(item)
                   for item in libraries if item.endswith(('.a', '.o', '.so'))]
    assert all(p.is_file() for p in link_inputs)
    assert all(p.stat().st_mtime_ns <= Path(engine['binary']['path']).stat().st_mtime_ns for p in link_inputs if p.suffix in ('.a', '.o'))
    inputs = [Path(__file__), owner_path, helper, engine_path, *gates,
              build / 'build.ninja', build / 'compile_commands.json',
              *link_inputs, L / 'bin/clang++', *candidate.glob('*'),
              *list((P / 'include').rglob('*.hpp'))]
    frozen = {str(p): pin(p) for p in inputs if p.is_file()}
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners(meta['known_model_files'])
        guards.driver_guard()
        root.mkdir(mode=0o700)
        (root / 'tmp').mkdir()
        env = dict(PATH=str(L / 'bin') + ':/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=str(L / 'lib') + ':/opt/intel/oneapi/mkl/2026.1/lib', TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
        rec = dict(schema=1, passed=False, complete=False, active=False, GPU_executed=False,
                   frozen_files=frozen, compiler=compiler, environment=env, commands=[],
                   engine_receipt=pin(engine_path), link_inputs=[pin(p) for p in link_inputs],
                   provenance=fixture['provenance'], production_selected=False,
                   model_quality_qualified=False, hardware_XMX_qualified=False,
                   scope='Private FP16-input, FP32-output score GEMM fixture build only. Existing current engine archives and link recipe; toolchain assets only from the excluded-fork directory.')
        owner = None
        def save():
            rec['active'] = bool(owner and owner.active is not None)
            out.write_text(json.dumps(rec, indent=2) + '\n')
        owner = owned.Owner(root / 'owned', save)
        rec['commands'] = owner.commands
        def run(label, argv):
            c, so, se = owner.run(label, argv, env, build, wall=300, cpu=900, rss_cap=12 << 30, text_cap=16 << 20, file_cap=256 << 20, total_cap=2 << 30)
            assert owned.closed(c) and c['exit_code'] == 0, (label, se.read_text()[-8000:])
            return so.read_text()
        try:
            assert all(pin(p) == value for p, value in frozen.items())
            flags = [compiler['path'], '-std=c++20', '-O3', '-DNDEBUG', '-DSTRATA_ENABLE_SYCL=1',
                     '-I' + str(P / 'include'), '-I' + str(P / 'include/strata/sycl/compat'),
                     '-fno-fast-math', '-ffp-contract=off', '-fsycl', '-fsycl-id-queries-range=size_t',
                     '-Wall', '-Wextra', '-Werror']
            exe = root / 'sycl_qsa_score_gemm_probe'
            run('compile-link', flags + [str(candidate / 'fixture.cpp'), str(candidate / 'qsa_score_gemm.cpp'), *libraries, '-o', str(exe)])
            rec['binary'] = pin(exe)
            rec['dynamic'] = run('readelf', ['/usr/bin/readelf', '-d', str(exe)])
            rec['ldd'] = run('ldd', ['/usr/bin/ldd', str(exe)])
            assert all(pin(p) == value for p, value in frozen.items())
            rec.update(passed=True, complete=True)
        except BaseException as error:
            import traceback
            rec.update(error=repr(error), traceback=traceback.format_exc())
        finally:
            rec['ownership_closed'] = owner.active is None and all(owned.closed(c) for c in owner.commands)
            rec['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save()
        print(json.dumps(dict(receipt=str(out), passed=rec['passed'], error=rec.get('error'))), flush=True)
        return 0 if rec['passed'] else 1

if __name__ == '__main__':
    sys.exit(main())
