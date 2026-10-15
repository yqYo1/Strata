#!/usr/bin/env python3
"""Offline exact prompt SPIR-V entrypoint extraction; no GPU/JIT claim."""
import fcntl
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bp = A / 'own-prefill-select-scope-build-v1.json'
        build = json.loads(bp.read_text())
        carrier = build['retained_prompt_component']
        exe = Path(carrier['path'])
        require(sha(exe) == carrier['sha256'], 'retained exact prompt component')
        install = Path(build['compiler']).parent.parent
        extract = install / 'bin/clang-offload-extract'
        translate = install / 'bin/llvm-spirv'
        root = B / 'main-prompt-jit-identity-root-v1'
        target = A / 'prompt-jit-identity-v1.json'
        require(not root.exists() and not target.exists(), 'fresh offline job')
        root.mkdir(mode=0o700)
        record = dict(passed=False, complete=False, active=True, GPU_executed=False,
            commands=[], recipe_sha256=sha(Path(__file__)), reference_build_sha256=sha(bp),
            binary=str(exe), binary_sha256=sha(exe),
            tools={str(p): sha(p) for p in (extract, translate)},
            limits=dict(wall_per_tool_seconds=60, RSS_bytes=4 << 30, text_bytes=64 << 20,
                        artifact_bytes=128 << 20, job_bytes=256 << 20),
            scope='Offline image/entrypoint identity for old exact48eeb prompt carrier only. '
                  'No GPU execution, runtimeJIT, register/spill/SLM or performance measurement.')
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        env = {k: v for k, v in build['environment'].items()
               if k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH')}
        def run(label, argv):
            c, out, err = owner.run(label, argv, env, root, wall=60, cpu=120,
                rss_cap=4 << 30, text_cap=64 << 20, total_cap=256 << 20, file_cap=64 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label + ': ' + err.read_text()[-3000:])
            return out.read_text()
        try:
            run('extract', [str(extract), '--stem=' + str(root / 'device'), str(exe)])
            images = sorted(root.glob('device.*'))
            require(0 < len(images) <= 32 and sum(p.stat().st_size for p in images) <= 64 << 20, 'finite image inventory')
            found = {}
            inventory = []
            for index, path in enumerate(images):
                require(not path.is_symlink(), 'private regular image')
                with path.open('rb') as f:
                    magic = f.read(4)
                item = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path), magic_hex=magic.hex())
                if magic == b'\x03\x02\x23\x07':
                    output = root / f'spirv-{index}.txt'
                    run('translate-' + str(index), [str(translate), '-to-text', str(path), '-o', str(output)])
                    entries = []
                    with output.open() as stream:
                        for line in stream:
                            if 'EntryPoint' not in line:
                                continue
                            entries.append(line.strip())
                            for name in re.findall(r'"([^"\n]+)"', line):
                                m = re.search(r'launchILi1ELb1ELb0ELi([012])EE', name)
                                if m:
                                    require(m[1] not in found, 'one exact specialization per mapping')
                                    found[m[1]] = dict(name=name, image_sha256=item['sha256'], entrypoint_line=line.strip())
                    require(len(entries) <= 256, 'bounded entrypoint evidence')
                    item.update(text_path=str(output), text_bytes=output.stat().st_size,
                                text_sha256=sha(output), entrypoints=entries)
                inventory.append(item)
            require(set(found) == {'0', '1', '2'}, 'all three actual Mode1/Direct1 mapping symbols identified')
            record.update(images=inventory, prompt_specializations=found,
                demangled=run('demangle', ['/usr/bin/c++filt', *[found[str(i)]['name'] for i in range(3)]]),
                passed=True, complete=True,
                retention_consumer='Main R739 exact regex/liveJIT correspondence; max128MiB images+text, review after bounded liveJIT capture or replacement')
            require(sha(exe) == record['binary_sha256'], 'carrier unchanged')
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):
                record.update(passed=False, complete=False)
            persist()
            with target.open('x') as f:
                json.dump(record, f, indent=2)
                f.write('\n')
        print(json.dumps(dict(passed=record['passed'], images=len(record['images']),
                             mappings=list(record['prompt_specializations']))), flush=True)


if __name__ == '__main__':
    main()
