"""Root-only build and CPU CLI contracts for pristine-library grouped qualifier."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, subprocess, types
B = Path(__file__).parent
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/eval-b570-20261010')
OUT = B / 'xestrata-grouped-gemm-build-root-v2'
SOURCE = B / 'xestrata-grouped-gemm-qualifier-v1/qualify_grouped.cpp'
OWNER = B / 'xestrata-pristine-icpx-build-root-v1/build_owner.py'
LIB = W / 'build/eval-b570-icpx-v1'
def ident(p):
    with p.open('rb') as f: h = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h)
with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not OUT.exists()
    assert ident(SOURCE)['sha256'] == '61184b7acc758ef84ece9e341ab7d48c38dc17ce80f13d41d2d662af57ec2b10'
    assert ident(OWNER)['sha256'] == 'da37d448cc14450ea4d151a90ada780cae5139cafbd21df3727a0bfd5c2058c8'
    assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True) == ''
    libs = [LIB / ('libstrata_' + name + '.a') for name in ['kernels', 'core', 'ngram', 'platform']]
    lib_ids = [ident(p) for p in libs]
    OUT.mkdir(mode=0o700)
    binary = OUT / 'qualify_grouped'
    m = types.ModuleType('owned_cpu_build'); exec(compile(OWNER.read_text(), str(OWNER), 'exec'), m.__dict__); m.W = W
    owner = m.Owner(OUT)
    env = {k: v for k, v in os.environ.items() if k in {'PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_CTYPE'}}
    env.update(LC_ALL='C', LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/mkl/2026.1/lib')
    argv = ['/opt/intel/oneapi/compiler/2026.1/bin/icpx', '-std=c++20', '-O3', '-fno-fast-math', '-ffp-contract=off',
            '-Wall', '-Wextra', '-Werror', '-fsycl', '-foffload-fp32-prec-div', '-foffload-fp32-prec-sqrt',
            '-I' + str(W / 'include'), str(SOURCE), '-Wl,--start-group'] + list(map(str, libs)) + [
            '-Wl,--end-group', '-pthread', '-Xspirv-translator', '--spirv-ext=+SPV_KHR_integer_dot_product', '-o', str(binary)]
    r = dict(active=True, passed=False, complete=False, gpu_executed=False, model_executed=False,
             started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), controller=ident(Path(__file__)),
             source=ident(SOURCE), owner=ident(OWNER), libraries=lib_ids, commands=owner.commands, tests=[])
    def save():
        p = OUT / 'record.json.tmp'; p.write_text(json.dumps(r, indent=2) + '\n'); p.replace(OUT / 'record.json')
    owner.persist = save
    try:
        e, so, se = owner.run('build', argv, env, wall=600, text_cap=64 << 20, file_cap=512 << 20)
        assert m.completed(e) and e['exit_code'] == 0, 'build failed'
        r['binary'] = ident(binary)
        for role in ['GU', 'Down']:
            for case in ['1', '127', '128', '129', '257', 'sparse', 'empty']:
                e, so, se = owner.run('cpu-' + role + '-' + case, [str(binary), '--role', role, '--case', case, '--cpu-preflight'], env, wall=10)
                assert m.completed(e) and e['exit_code'] == 0
                v = json.loads(so.read_text()); assert v['passed'] and not v['gpu_executed']
                r['tests'].append(v)
        negatives = [[], ['--cpu-preflight'], ['--role', 'bad', '--case', '1'], ['--role', 'GU', '--case', 'bad'],
                     ['--role', 'GU', '--case', '1', '--role', 'GU'], ['--role', 'GU', '--case', '1', '--cpu-preflight'],
                     ['--unknown'], ['--role'], ['--role', 'GU', '--case'], ['--role', 'X' * 17, '--case', '1']]
        for i, values in enumerate(negatives):
            e, so, se = owner.run('negative-' + str(i), [str(binary)] + values + ['--cpu-preflight'], env, wall=10)
            assert m.completed(e) and e['exit_code'] == 2
            v = json.loads(se.read_text()); assert not v['passed'] and not v['gpu_executed']
            r['tests'].append(dict(case='negative-' + str(i), rejected=True, output=v))
        assert [ident(p) for p in libs] == lib_ids
        assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True) == ''
        r.update(complete=True, passed=True)
    except BaseException as e: r['error'] = type(e).__name__ + ': ' + str(e)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = r['passed'] and not r['active']; save()
        print(json.dumps({k: r.get(k) for k in ['active', 'passed', 'complete', 'error', 'binary']}))
    if not r['passed']: raise SystemExit(1)
