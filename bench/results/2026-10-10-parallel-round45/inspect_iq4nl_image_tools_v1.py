"""Root-owned CPU-only installed extractor admission; never loads a GPU runtime."""
from pathlib import Path
import fcntl, hashlib, json, os, sys, types

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-prefill-iq4nl-direct-20261010')
OUT = B / 'iq4nl-image-tools-admission-v1'
PARENT = B / 'run_gdn_gate_factor_probe_v2.py'
PARENT_HASH = '7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
TOOLS = Path('/opt/intel/oneapi/compiler/2026.1/bin/compiler')

def pin(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)

def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert pin(PARENT)['sha256'] == PARENT_HASH
        module = types.ModuleType('image_tool_owner')
        exec(compile(PARENT.read_text(), str(PARENT), 'exec'), module.__dict__)
        module.W = W
        build_path = B / 'prefill-iq4nl-parity-cpu-build-v1/record.json'
        build = json.loads(build_path.read_text())
        assert build['passed'] and build['complete'] and not build['active']
        binary = Path(build['binary']['path'])
        assert pin(binary) == build['binary']
        assert not OUT.exists()
        OUT.mkdir(mode=0o700)
        owner = module.Owner(OUT)
        env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                   LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib')
        tools = {name: pin(TOOLS / name) for name in ['clang-offload-extract', 'clang-offload-bundler', 'llvm-spirv']}
        record = dict(active=True, complete=False, passed=False, started_utc=module.utc(),
                      controller=pin(Path(__file__)), owner_sha256=PARENT_HASH,
                      environment=env, tools=tools, binary=build['binary'],
                      build_receipt=pin(build_path), commands=owner.commands,
                      GPU_executed=False, model_opened=False, adopted=False,
                      scope='CPU-only installed CLI syntax; no image absence, lowering or runtime claim')

        def save():
            temp = OUT / 'record.json.tmp'
            with temp.open('w') as stream:
                json.dump(record, stream, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            temp.replace(OUT / 'record.json')

        owner.persist = save
        try:
            for name in tools:
                entry, stdout, stderr = owner.run(name + '-help', [str(TOOLS / name), '--help'],
                                                 env, wall=10, text_cap=2 << 20, file_cap=2 << 20)
                assert module.completed(entry) and entry['exit_code'] == 0, (name, entry)
                assert stdout.stat().st_size or stderr.stat().st_size
            for name, identity in tools.items():
                assert pin(TOOLS / name) == identity
            assert pin(binary) == build['binary']
            record.update(complete=True, passed=True)
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            record['passed'] = record['passed'] and not record['active'] and all(module.completed(c) for c in owner.commands)
            save()
            print(json.dumps(dict(receipt=str(OUT / 'record.json'), passed=record['passed'], error=record.get('error'))))
        return 0 if record['passed'] else 1

if __name__ == '__main__':
    sys.exit(main())
