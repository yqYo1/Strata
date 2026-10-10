"""CPU-only extraction of the already-qualified ON fixture; not an OFF/ON proof."""
from pathlib import Path
import fcntl, hashlib, json, os, sys, types

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-prefill-iq4nl-direct-20261010')
OUT = B / 'iq4nl-on-device-image-inventory-v1'
PARENT = B / 'run_gdn_gate_factor_probe_v2.py'
PARENT_HASH = '7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'

def pin(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)

def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert pin(PARENT)['sha256'] == PARENT_HASH
        module = types.ModuleType('image_extract_owner')
        exec(compile(PARENT.read_text(), str(PARENT), 'exec'), module.__dict__)
        module.W = W
        admission_path = B / 'iq4nl-image-tools-admission-v1/record.json'
        admission = json.loads(admission_path.read_text())
        assert admission['complete'] and admission['passed'] and not admission['active']
        binary = Path(admission['binary']['path'])
        assert pin(binary) == admission['binary']
        tool = Path(admission['tools']['clang-offload-extract']['path'])
        assert pin(tool) == admission['tools']['clang-offload-extract']
        assert not OUT.exists()
        OUT.mkdir(mode=0o700)
        images = OUT / 'images'
        images.mkdir()
        owner = module.Owner(OUT)
        record = dict(active=True, complete=False, passed=False, started_utc=module.utc(),
                      controller=pin(Path(__file__)), owner_sha256=PARENT_HASH,
                      tool=pin(tool), tool_admission=pin(admission_path),
                      binary=admission['binary'], commands=owner.commands,
                      GPU_executed=False, model_opened=False, adopted=False,
                      OFF_absence_qualified=False, native_lowering_qualified=False,
                      scope='ON linked fixture embedded-image inventory; no matched OFF build or native ISA')

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
            entry, stdout, stderr = owner.run('extract-linked-ON-fixture',
                                             [str(tool), '--stem=' + str(images / 'target.bin'), str(binary)],
                                             admission['environment'], wall=30,
                                             text_cap=2 << 20, file_cap=16 << 20)
            assert module.completed(entry) and entry['exit_code'] == 0, entry
            assert stderr.stat().st_size == 0, stderr.read_text()
            files = sorted(images.iterdir())
            assert 0 < len(files) <= 1024
            assert all(path.is_file() and not path.is_symlink() for path in files)
            assert sum(path.stat().st_size for path in files) <= 64 << 20
            inventory = []
            for path in files:
                identity = pin(path)
                with path.open('rb') as stream:
                    identity['header_hex'] = stream.read(32).hex()
                inventory.append(identity)
            record['images'] = inventory
            record['image_count'] = len(inventory)
            record['image_total_bytes'] = sum(item['bytes'] for item in inventory)
            assert pin(binary) == admission['binary']
            assert pin(tool) == admission['tools']['clang-offload-extract']
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            record['passed'] = record['passed'] and not record['active'] and all(module.completed(c) for c in owner.commands)
            save()
            print(json.dumps(dict(receipt=str(OUT / 'record.json'), passed=record['passed'],
                                  image_count=record.get('image_count'),
                                  image_total_bytes=record.get('image_total_bytes'), error=record.get('error'))))
        return 0 if record['passed'] else 1

if __name__ == '__main__':
    sys.exit(main())
