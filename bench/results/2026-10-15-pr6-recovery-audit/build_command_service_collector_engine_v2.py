#!/usr/bin/env python3
"""Fresh default-off observer build; calibration gates closed before execution."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-moe-window')
C = W / 'bench/results/2026-10-15-pr6-recovery-audit'
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEV = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), realpath=str(path.resolve()), bytes=path.stat().st_size, sha256=digest)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = B / 'main-command-service-collector-engine-build-v2'
    out = A / 'command-service-collector-engine-build-v2.json'
    assert not root.exists() and not out.exists()
    owner_path = A / 'direct_owner_transient_artifacts_v1.py'
    assert pin(owner_path)['sha256'] == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owned = load('subgroup_engine_owner', owner_path)
    gate_path = A / 'event-service-calibration-v1.json'
    gate = json.loads(gate_path.read_text())
    assert gate['passed'] and gate['complete'] and gate['ownership_closed'] and not gate['active']
    assert gate['fresh_faults'] == [] and not gate['driver_dump_entries']
    for command in gate['commands']:
        assert owned.closed(command) and command['exit_code'] == 0
        for old in command['owners'] + [command['direct_identity']]:
            now = owned.identity(old['pid'])
            assert now is None or now['start_ticks'] != old['start_ticks']
    package = A / 'own-vllm-xe2-qualification-v1'
    metadata_path = package / 'manifest.json'
    assert pin(metadata_path)['sha256'] == '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    metadata = json.loads(metadata_path.read_text())
    helper = package / 'main_controller.py'
    assert pin(helper)['sha256'] == metadata['package_files']['main_controller.py']['sha256']
    health = load('subgroup_engine_health', helper)
    previous_path = A / 'qsa-rowmajor-subgroup-engine-build-v1.json'
    previous = json.loads(previous_path.read_text())
    assert previous['passed'] and previous['complete'] and previous['ownership_closed']
    pair_path = A/'command-service-model-diagnostic-cohort-v1.json'
    pair = json.loads(pair_path.read_text())
    assert pair['active_sample'] is None and len(pair['completed']) == 2
    for item in pair['completed']:
        assert pin(item['path'])['sha256'] == item['sha256']
        value = json.loads(Path(item['path']).read_text())
        assert value['ownership_closed'] and not value['active'] and not value['fresh_faults'] and not value['driver_dump_exists']
        assert value['result']['real_engine_exit']['exit_code'] == 0
    stress_path = A/'event-service-stress-v1.json'
    stress = json.loads(stress_path.read_text())
    assert stress['passed'] and stress['complete'] and stress['ownership_closed'] and not stress['active']
    source_path = C / 'command-service-collector-source-v2.json'
    source = json.loads(source_path.read_text())
    assert source['contract']['event_handle_cap'] == 160 and source['contract']['numeric_record_cap'] == 640
    for name, digest in source['source_pins'].items():
        assert pin(W / name)['sha256'] == digest
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'src', 'include', 'tests', 'cmake', 'CMakeLists.txt'], cwd=W, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip()
    assert commit.startswith('af99f294')
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--', 'src', 'include', 'tests', 'cmake', 'CMakeLists.txt'], cwd=W, text=True).splitlines()
    frozen = {str(W / name): pin(W / name) for name in names}
    for name, digest in previous['source_pins'].items():
        path = Path(name)
        if path.is_relative_to(W) or path.is_relative_to(P) or path.is_relative_to(B):
            continue
        if '/MistVVK/XeStrata/' in name:
            assert '/.tools/intel-llvm-v711/install/' in name
        value = pin(path)
        assert value['sha256'] == digest
        frozen[name] = value
    for path in [Path(__file__), owner_path, gate_path, source_path, previous_path, metadata_path, helper, pair_path, stress_path]:
        frozen[str(path)] = pin(path)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert boot == previous['boot_id']
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        health.no_model_owners(metadata['known_model_files'])
        health.driver_guard()
        root.mkdir(mode=0o700)
        (root / 'tmp').mkdir()
        env = dict(previous['environment'])
        env.update(TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
        rec = dict(schema='strata-command-service-collector-engine-build-v2', passed=False, complete=False,
                   active=False, GPU_executed=False, source_commit=commit, boot_id=boot,
                   environment=env, commands=[], frozen_files=frozen,
                   source_pins={name:value['sha256'] for name,value in frozen.items()},
                   provenance=source['provenance'], production_selected=False, performance_adopted=False,
                   model_qualified=False, physical262144_qualified=False, RESTORE_qualified=False,
                   scope='Fresh Release engine and bounded command-profiler synthetic target. No reuse or overwrite of the older build; inference and fixture execution are separate.')
        owner = None

        def save():
            rec['active'] = bool(owner and owner.active is not None)
            rec['ownership_closed'] = bool(owner and owner.active is None and all(owned.closed(c) for c in owner.commands))
            out.write_text(json.dumps(rec, indent=2) + '\n')

        owner = owned.Owner(root / 'owned', save)
        rec['commands'] = owner.commands

        def run(label, argv, wall=60, cpu=120):
            command, stdout, stderr = owner.run(label, argv, env, W, wall=wall, cpu=cpu,
                 rss_cap=16 << 30, text_cap=64 << 20, file_cap=1 << 30, total_cap=4 << 30)
            assert owned.closed(command) and command['exit_code'] == 0, (label, command.get('exit_code'), str(stderr))
            return stdout.read_text()

        cursor = None
        save()
        try:
            raw = run('journal-before', ['/usr/bin/journalctl','-k','-b','-n','0','--show-cursor','--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', raw)[1]
            rec['journal_before_cursor'] = cursor
            configure = list(previous['configure_argv'])
            configure[configure.index('-B') + 1] = str(root / 'build')
            rec['configure_argv'] = configure
            run('configure', configure, wall=300, cpu=600)
            targets = ['strata', 'sycl_command_profile_test']
            run('compile-link', ['/usr/bin/cmake','--build',str(root / 'build'),'--target',*targets,'--parallel','2','--verbose'], wall=2400, cpu=18000)
            rec['products'] = {name:pin(root / 'build' / name) for name in targets}
            rec['binary'] = rec['products']['sycl_command_profile_test']
            entries = json.loads((root / 'build/compile_commands.json').read_text())
            selected = ['src/sycl/command_profile.cpp','src/sycl/compat.cpp','src/prefill/prefill.cpp','tests/sycl/command_profile.cpp']
            rec['actual_compile_entries'] = [entry for entry in entries if entry['file'] in [str(W / name) for name in selected]]
            assert {entry['file'] for entry in rec['actual_compile_entries']} == {str(W / name) for name in selected}
            rec['compile_database'] = pin(root / 'build/compile_commands.json')
            assert all(pin(name) == value for name,value in frozen.items())
            rec.update(passed=True, complete=True)
        except BaseException as error:
            rec['error'] = repr(error)
        finally:
            try:
                assert owner.active is None
                if cursor:
                    raw = run('journal-final', ['/usr/bin/journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json'])
                    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    rec['fresh_faults'] = [row for row in rows if FAULT.search(str(row.get('MESSAGE','')))]
                rec['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot
                rec['driver_dump_entries'] = [str(x) for x in DEV.glob('devcoredump*')]
                assert cursor and rec['fresh_faults'] == [] and rec['boot_unchanged'] and not rec['driver_dump_entries']
                health.driver_guard()
            except BaseException as error:
                rec.update(closure_error=repr(error),passed=False,complete=False)
            rec['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save()
        print(json.dumps(dict(passed=rec['passed'],complete=rec['complete'],receipt=str(out),error=rec.get('error'))),flush=True)
        return 0 if rec['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
