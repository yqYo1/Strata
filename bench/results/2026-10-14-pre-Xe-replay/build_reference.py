#!/usr/bin/env python3
"""Build the qualified pre-Xe source in this worktree; no model/GPU invocation."""
import fcntl
import hashlib
import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
O = B/'main-pre-Xe-1eb-build-root-v1'
S = P/'.tools/pre-xe-1eb-source-v1'
REV = '1eb89482a4afd20277ae0405780ed4f8eb98eb20'
GGML = Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
C = Path('/opt/intel/oneapi/compiler/2026.1')
OWNER = P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
PREFIXES = ['CMakeLists.txt', 'sycl/CMakeLists.txt', 'sycl/include', 'sycl/src',
            'src', 'include', 'third_party', 'data']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    assert subprocess.check_output(['git', '-C', str(GGML), 'rev-parse', 'HEAD']).decode().strip() == '3cf03257f219afbe7334045ff7c6a06ac68c627d'
    assert not subprocess.check_output(['git', '-C', str(GGML), 'status', '--porcelain'])
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists() and not S.exists()
        O.mkdir()
        S.mkdir(parents=True)
        env = dict(PATH=str(C/'bin')+':/usr/bin:/bin',
            LD_LIBRARY_PATH=':'.join(map(str, [C/'lib', C/'opt/compiler/lib',
                Path('/opt/intel/oneapi/mkl/2026.1/lib'), Path('/opt/intel/oneapi/umf/1.1/lib'),
                Path('/usr/lib/x86_64-linux-gnu')])),
            MKLROOT='/opt/intel/oneapi/mkl/2026.1', CMPLR_ROOT=str(C), ONEAPI_ROOT='/opt/intel/oneapi')
        record = dict(active=True, passed=False, complete=False, commands=[], environment=env,
            GPU_executed=False, model_executed=False, source_revision=REV, source_path=str(S),
            primary_revision=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD']).decode().strip(),
            primary_engine_unchanged=True, source_patches=[], recipe_sha256=sha(Path(__file__)),
            source_prefixes=PREFIXES, ggml_revision='3cf03257f219afbe7334045ff7c6a06ac68c627d',
            ggml_path=str(GGML),
            limits=dict(build_wall_seconds=1800, build_bytes=4<<30, source_bytes=64<<20,
                log_bytes=64<<20, process_RSS_bytes=48<<30),
            consumer='Isolated own-fork Niko0.1.41 parity and matched stock32K comparison; not adopted product source.',
            next_review='Retire source snapshot/build after matched source-component decisions and archived evidence.',
            historical_math_and_SG32_contract_preserved=True,
            size_t_flag_note='No blanket flag mutation: qualified old migrated kernels have bounded launch geometry; current root size_t requirement is retained in current source. Runtime parameter checks and max-input gate remain separate.')
        def persist():
            (O/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        owner = module.Owner(O/'owned', persist)
        record['commands'] = owner.commands
        persist()
        def run(label, argv, wall=30):
            entry, out, err = owner.run(label, argv, env, P, wall=wall, cpu=wall,
                text_cap=64<<20, rss_cap=48<<30, total_cap=4<<30, file_cap=256<<20)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), err.read_text()[-3500:])
            print(json.dumps(dict(passed=label)), flush=True)
            return out.read_text(), err.read_text()
        try:
            archive = O/'source.tar'
            run('export-source', ['git', '-C', str(P), 'archive', '--format=tar', '--output='+str(archive), REV, '--', *PREFIXES])
            record['source_archive_sha256'] = sha(archive)
            with tarfile.open(archive) as source:
                source.extractall(S, filter='data')
            lines = subprocess.check_output(['git', '-C', str(P), 'ls-tree', '-rl', REV, '--', *PREFIXES]).decode().splitlines()
            manifest = []
            for line in lines:
                meta, name = line.split('\t', 1)
                mode, kind, blob, size = meta.split()
                if kind == 'commit':
                    continue
                assert kind == 'blob' and mode != '120000'
                path = S/name
                assert path.is_file() and path.stat().st_size == int(size), name
                digest = sha(path)
                assert subprocess.check_output(['git', '-C', str(P), 'hash-object', '--no-filters', str(path)]).decode().strip() == blob
                manifest.append(dict(path=name, bytes=int(size), git_blob=blob, sha256=digest))
            record['source_manifest'] = manifest
            record['source_bytes'] = sum(x['bytes'] for x in manifest)
            assert record['source_bytes'] < record['limits']['source_bytes']
            persist()
            run('compiler-version', [str(C/'bin/icpx'), '--version'])
            run('configure-reference', ['cmake', '-S', str(S/'sycl'), '-B', str(O/'build'), '-G', 'Ninja',
                '-DCMAKE_CXX_COMPILER='+str(C/'bin/icpx'), '-DCMAKE_C_COMPILER='+str(C/'bin/icx'),
                '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                '-DSTRATA_GGML_DIR='+str(GGML)], 180)
            run('build-reference', ['cmake', '--build', str(O/'build'), '--parallel', '2', '--target',
                'strata', 'qsa_prompt_attn_parity', 'kv_stream_parity', 'quantize_act_parity', 'verify_parity'], 1800)
            record['build_path'] = str(O/'build')
            names = ['strata', 'qsa_prompt_attn_parity', 'kv_stream_parity', 'quantize_act_parity', 'verify_parity']
            record['binaries'] = {n:sha(O/'build'/n) for n in names}
            compile_db = json.loads((O/'build/compile_commands.json').read_text())
            actual = [x for x in compile_db if Path(x['file']).is_relative_to(S)]
            record['compiled_sources'] = actual
            record['compile_database_sha256'] = sha(O/'build/compile_commands.json')
            assert actual and all('-fp-model=precise' in x['command'] for x in actual)
            assert all('-fsycl-default-sub-group-size=32' in x['command'] for x in actual)
            assert any(x['file'].endswith('/sycl/src/prefill/gemm.dp.cpp') for x in actual)
            assert not any('moe_mmq.dp.cpp' in x['file'] or 'moe_fused_iq.dp.cpp' in x['file'] for x in actual)
            record['build_bytes'] = sum(f.stat().st_size for f in (O/'build').rglob('*') if f.is_file())
            assert record['build_bytes'] < record['limits']['build_bytes']
            run('linked-library-inventory', ['ldd', str(O/'build/strata')])
            for item in manifest:
                assert sha(S/item['path']) == item['sha256']
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = False
            persist()
        print(json.dumps(dict(passed=True, model_executed=False, GPU_executed=False,
            binary_sha256=record['binaries']['strata'], source_bytes=record['source_bytes'])), flush=True)


if __name__ == '__main__':
    main()
