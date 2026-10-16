#!/usr/bin/env python3
"""Compile the reviewed harness-only change; reuse byte-pinned engine archives."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
C = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-pr6-prefill-restore')


def pin(path):
    p = Path(path)
    with p.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(p), realpath=str(p.resolve()), bytes=p.stat().st_size, sha256=digest)


def check(v):
    assert pin(v['path']) == v, v['path']


def main():
    root = B / 'main-pr6-gated-harness-build-v1'
    target = A / 'own-pr6-gated-harness-build-v1.json'
    assert not root.exists() and not target.exists()
    config_path = A / 'own-pr6-harness-configure-v6.json'
    prior_path = A / 'own-pr6-harness-build-v4.json'
    assert pin(config_path)['sha256'] == 'aa3e36da8b2f4f26718393271ba595db227835b89738577a54b1b724289e6f27'
    assert pin(prior_path)['sha256'] == 'b842d48054338673ca404d99ba0d39719bd6442e1ff3f7169682cf7079632d51'
    config = json.loads(config_path.read_text())
    prior = json.loads(prior_path.read_text())
    assert config['passed'] and prior['passed'] and prior['complete'] and prior['ownership_closed']
    build = Path(config['build_path'])
    source = C / 'validation/pr6-restore/harness/prefill_api.cpp'
    assert pin(source)['sha256'] == '2641d731d0ba81b6e467f35b81bdb532689afa231708c053d7c93d86498bee6f'
    head = subprocess.check_output(['git', '-C', str(C), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == '3602e9c4e1a33ab7a4148d63c5f340a3d25682d4'
    changed = subprocess.check_output(['git', '-C', str(C), 'diff', '--name-only',
        config['source_commit'], head, '--', 'validation/pr6-restore'], text=True).splitlines()
    assert changed == ['validation/pr6-restore/harness/prefill_api.cpp']
    frozen = [*config['generated_inputs'], *config['compilers'], config['oneMKL_config'],
        prior['binary'], pin(config_path), pin(prior_path), pin(source), pin(__file__)]
    source_changes = []
    for v in config['source_pins'].values():
        if Path(v['path']) == source:
            source_changes.append(dict(before=v, after=pin(source)))
        else:
            frozen.append(v)
    assert len(source_changes) == 1
    rows = json.loads((build / 'compile_commands.json').read_text())
    row, = [r for r in rows if r['file'] == str(source)]
    compile_before = shlex.split(row['command'])
    compile_after = list(compile_before)
    oi = compile_after.index('-o') + 1
    old_object = compile_after[oi]
    compile_after[oi] = str(root / 'prefill_api.cpp.o')
    for flag in ['-MT', '-MF']:
        if flag in compile_after:
            i = compile_after.index(flag) + 1
            compile_after[i] = str(root / ('prefill_api.cpp.o.d' if flag == '-MF' else 'prefill_api.cpp.o'))
    assert '-fsycl' in compile_after and '-std=c++20' in compile_after and '-ffp-model=precise' in compile_after
    assert '-fno-sycl' not in compile_after
    p = subprocess.run(['/usr/bin/ninja', '-C', str(build), '-t', 'commands', 'pr6_prefill_api'],
        capture_output=True, text=True, timeout=15, env=config['commands'][0]['environment'])
    assert p.returncode == 0 and len(p.stdout.encode()) < 8 << 20
    link_line = p.stdout.splitlines()[-1]
    tokens = shlex.split(link_line)
    assert tokens[:2] == [':', '&&'] and tokens[-2:] == ['&&', ':']
    link_before = tokens[2:-2]
    assert not any(v in ('&&', '|', ';', ':') for v in link_before)
    assert link_before.count(old_object) == 1
    link_after = [str(root / 'prefill_api.cpp.o') if v == old_object else v for v in link_before]
    link_after[link_after.index('-o') + 1] = str(root / 'pr6_prefill_api')
    libraries = []
    for arg in link_before:
        if arg.endswith(('.a', '.so')):
            libraries.append(pin(Path(arg) if Path(arg).is_absolute() else build / arg))
    assert len([v for v in libraries if v['path'].endswith('.a')]) == 7
    assert len([v for v in libraries if v['path'].endswith('.so')]) == 4
    frozen.extend(libraries)
    owner_path = A.parents[2] / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert pin(owner_path)['sha256'] == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('pr6_gated_build_owner', owner_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    def freeze():
        for v in frozen:
            check(v)

    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze()
        root.mkdir(mode=0o700)
        rec = dict(schema='strata-pr6-gated-harness-build-v1', passed=False, complete=False,
            active=False, commands=[], GPU_executed=False, legacy_binary_executed=False,
            source_commit=head, source_changes=source_changes, immutable_inputs=frozen,
            compile_before=compile_before, compile_after=compile_after,
            link_before=link_before, link_after=link_after, libraries=libraries,
            archive_reuse='Seven byte-pinned complete engine/ggml archives from passed build-v4; all their source/header pins unchanged. Only reviewed harness TU changes.',
            scope='Compile/link only. No native or model qualification; preserve original ungated binary untouched.',
            retention=dict(owner='Main', consumer='Next ELF/environment/maps/model-small safety gate',
                next_review='After compact compile/link proof and first native outcome'))
        owner = None

        def persist():
            rec['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(rec, indent=2) + '\n')

        owner = m.Owner(root / 'owned', persist)
        rec['commands'] = owner.commands
        try:
            for label, argv in [('compile-gated-harness', compile_after), ('link-gated-harness', link_after)]:
                freeze()
                command, out, err = owner.run(label, argv, config['commands'][0]['environment'], build,
                    wall=300, cpu=600, rss_cap=12 << 30, text_cap=16 << 20,
                    file_cap=128 << 20, total_cap=512 << 20)
                rec[label] = dict(exit_code=command['exit_code'], stdout=pin(out), stderr=pin(err))
                assert m.closed(command) and command['exit_code'] == 0, (label, out.read_text()[-4096:], err.read_text()[-4096:])
            freeze()
            rec.update(passed=True, complete=True, binary=pin(root / 'pr6_prefill_api'))
        except BaseException as error:
            import traceback
            rec.update(error=repr(error), traceback=traceback.format_exc())
        finally:
            rec['ownership_closed'] = owner.active is None and all(m.closed(c) for c in owner.commands)
            for command in owner.commands:
                for old in command['owners'] + [command['direct_identity']]:
                    now = m.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
            persist()
            target.write_bytes((root / 'record.json').read_bytes())
        print(json.dumps(dict(receipt=str(target), passed=rec['passed'], complete=rec['complete'],
            owned_closed=rec['ownership_closed'], GPU_executed=False, error=rec.get('error', '')[:300])))
        return 0 if rec['passed'] and rec['complete'] and rec['ownership_closed'] else 1


if __name__ == '__main__':
    sys.exit(main())
