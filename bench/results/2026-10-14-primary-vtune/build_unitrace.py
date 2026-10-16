#!/usr/bin/env python3
"""Main-owned ghq/worktree profiler build, without native engine changes."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
O = B/'main-primary-unitrace-build-root-v1'
SEED = P/'bench/results/2026-10-14-primary-transfer-ceiling/receipts/main-primary-transfer-ceiling-root-v1/record.json.gz'
OWNER = P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
T = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
G = Path('/home/yayoi/ghq/github.com/intel/pti-gpu')
GC = Path('/home/yayoi/ghq/github.com/intel/compute-runtime')
GL = Path('/home/yayoi/ghq/github.com/oneapi-src/level-zero')
WT = G/'.worktree/tools-strata-unitrace-20261014'
WTC = GC/'.worktree/tools-strata-unitrace-headers-20261014'
WTL = GL/'.worktree/tools-strata-unitrace-headers-20261014'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    seed = json.loads(gzip.decompress(SEED.read_bytes()))
    assert seed['passed'] and not seed['active']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists()
        O.mkdir()
        env = dict(seed['environment'])
        record = dict(active=True, passed=False, complete=False, commands=[], environment=env,
            recipe_sha256=sha(Path(__file__)), GPU_executed=False, model_executed=False,
            source_revision='6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033',
            compute_runtime_headers_revision='050536ff3b6f830199ec99cff8ffcac3d804c40a',
            level_zero_headers_tag='v1.32.0',
            limits=dict(build_wall_seconds=900, job_bytes=4<<30, text_bytes=64<<20, RSS_bytes=8<<30),
            build_consumer='bounded runtime unitrace API/device trace smoke, then primary marked model attribution',
            review_point='retire temporary header/tool worktrees after successful build and source pinning',
            primary_engine_unchanged=True, default_roots_unchanged=True)
        def persist():
            (O/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        owner = module.Owner(O/'owned', persist)
        record['commands'] = owner.commands
        persist()
        def run(label, argv, wall=30):
            entry, out, err = owner.run(label, argv, env, P, wall=wall, cpu=wall,
                text_cap=64<<20, rss_cap=8<<30, total_cap=4<<30, file_cap=256<<20)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), entry.get('exit_code'), err.read_text()[-3500:])
            return out.read_text(), err.read_text()
        def git(label, root, args, wall=30):
            return run(label, ['git', '-C', str(root), *args], wall)
        try:
            for root in (G, GC):
                out, _ = git('clean-'+root.name, root, ['status', '--porcelain'])
                assert not out
                branch, _ = git('branch-'+root.name, root, ['branch', '--show-current'])
                assert branch.strip() == 'master'
            run('ghq-level-zero', ['/home/yayoi/.nix-profile/bin/ghq', 'get', '--shallow', '--no-recursive', 'github.com/oneapi-src/level-zero'], 240)
            out, _ = git('clean-level-zero', GL, ['status', '--porcelain'])
            assert not out
            git('fetch-level-zero-header-tag', GL, ['fetch', '--depth', '1', 'origin', 'tag', 'v1.32.0'], 180)
            for label, root, worktree, ref in [('tool', G, WT, record['source_revision']),
                    ('compute-headers', GC, WTC, record['compute_runtime_headers_revision']),
                    ('l0-headers', GL, WTL, 'v1.32.0')]:
                assert not worktree.exists()
                branch = 'tools/strata-unitrace-20261014' if label == 'tool' else 'tools/strata-unitrace-headers-20261014'
                git('worktree-'+label, root, ['worktree', 'add', '-b', branch, str(worktree), ref])
                tip, _ = git('revision-'+label, worktree, ['rev-parse', 'HEAD'])
                record.setdefault('worktrees', []).append(dict(path=str(worktree), root=str(root), branch=branch, source_revision=tip.strip()))
                persist()
            build = O/'build'
            build.mkdir()
            # Vendor helpers skip clone/checkout when these paths already exist.
            # Pinned ghq worktrees keep the default mirrors untouched.
            (build/'compute-runtime').symlink_to(WTC, target_is_directory=True)
            (build/'level-zero').symlink_to(WTL, target_is_directory=True)
            record['build_path'] = str(build)
            config = ['cmake', '-S', str(WT/'tools/unitrace'), '-B', str(build), '-G', 'Ninja',
                '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER='+str(T/'bin/clang++'),
                '-DCMAKE_C_COMPILER='+str(T/'bin/clang'), '-DONEAPI_COMPILER_HOME='+str(T),
                '-DBUILD_WITH_MPI=0', '-DBUILD_WITH_ITT=0', '-DBUILD_WITH_OMP=0',
                '-DBUILD_WITH_OPENCL=0', '-DBUILD_WITH_L0=1', '-DBUILD_WITH_PERFETTO=OFF',
                '-DBUILD_WITH_XPTI=1', '-DXptifw_LIBRARY='+str(T/'lib/libxptifw.so.1.0.1'),
                '-DXptifw_INCLUDE_DIR='+str(T/'include')]
            run('configure-unitrace', config, 120)
            run('build-unitrace', ['cmake', '--build', str(build), '--parallel', '2'], 900)
            record['binaries'] = {name:sha(build/name) for name in ('unitrace', 'libunitrace_tool.so')}
            record['generated_header_pins'] = {str(f.relative_to(build)):sha(f)
                for sub in ('level_zero', 'CL') for f in (build/sub).rglob('*') if f.is_file()}
            record['compile_commands_sha256'] = sha(build/'compile_commands.json')
            record['XPTI_provider_pins'] = {str(f):sha(f) for f in (T/'lib/libxptifw.so.1.0.1', T/'include/xpti/xpti_trace_framework.h')}
            for entry in record['worktrees']:
                out, _ = git('post-build-clean-'+Path(entry['root']).name, Path(entry['path']), ['status', '--porcelain'])
                assert not out
            version, err = run('unitrace-version', [str(build/'unitrace'), '--version'])
            record['version_stdout'] = version
            record['version_stderr'] = err
            record['build_logical_bytes'] = sum(f.stat().st_size for f in build.rglob('*') if f.is_file())
            assert record['build_logical_bytes'] < 4<<30
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = False
            persist()
        print(json.dumps(dict(passed=True, GPU_executed=False, build_path=record['build_path'],
            profiler=record['binaries'], capability_qualified=False)), flush=True)


if __name__ == '__main__':
    main()
