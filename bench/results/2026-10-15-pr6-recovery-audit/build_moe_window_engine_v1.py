#!/usr/bin/env python3
"""Fresh candidate build. Refuse live foreign model work; never wait/retry it."""
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
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEV = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = B / 'main-moe-window-engine-build-v1'
    out = A / 'moe-window-engine-build-v1.json'
    assert not root.exists() and not out.exists(), 'closed receipts are immutable; no implicit retry'
    owner_path = A / 'direct_owner_transient_artifacts_v1.py'
    assert sha(owner_path) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owned = load('moe_build_owner', owner_path)
    # This exact foreign CPU benchmark is also excluded. A reused PID with a
    # different start time is not the same job, and is never signalled here.
    foreign = owned.identity(3431768)
    assert foreign is None or foreign['start_ticks'] != 52544839 or foreign['state'] == 'Z', 'foreign CPU llama-bench still live'
    package = A / 'own-vllm-xe2-qualification-v1'
    metadata = json.loads((package / 'manifest.json').read_text())
    assert sha(package / 'manifest.json') == '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    helper = package / 'main_controller.py'
    assert sha(helper) == metadata['package_files']['main_controller.py']['sha256']
    health = load('moe_build_health', helper)
    previous_path = A / 'own-qsa-bf16-prefix-tilepack-engine-build-v2.json'
    assert sha(previous_path) == 'a1b8916840e5482fcf7dca9bd1fead4a68882e7dcf5af676432980afe4771fc8', 'qualified build recipe changed'
    previous = json.loads(previous_path.read_text())
    assert previous['passed'] and previous['complete'] and previous['ownership_closed']
    assert all(owned.closed(c) and c['exit_code'] == 0 for c in previous['commands'])
    engine = json.loads((A / 'moe-window-engine-source-v1.json').read_text())
    assert sha(A / 'moe-window-fixture-syntax-v1.json') == '8025294a62c04fda5fd6366dfca31509fe675c1e8006338e773af1c90cb2ffa2', 'fixture syntax receipt changed'
    syntax = json.loads((A / 'moe-window-fixture-syntax-v1.json').read_text())
    assert syntax['passed'] and syntax['complete'] and syntax['ownership_closed']
    # The engine manifest and fixture syntax receipt are separate pinned source
    # boundaries. Later reports/commits do not authorize changing their code.
    assert sha(A / 'moe-window-engine-source-v1.json') == '3a3c145d1b6b9dc233f032141d21380e5e501dbb716729fb45133ab986254cc2'
    for name, value in syntax['sources'].items():
        assert sha(P / name) == value, name
    # engine source manifest uses its own explicitly named six-file hash map.
    engine_pins = engine['source_pins']
    for name, value in engine_pins.items():
        assert sha(P / name) == value, name
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'src', 'include', 'tests', 'cmake', 'CMakeLists.txt'], cwd=P, check=True)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--', 'src', 'include', 'tests', 'cmake', 'CMakeLists.txt'], cwd=P, text=True).splitlines()
    pins = {str(P / name): sha(P / name) for name in names}
    old_primary = Path('/home/yayoi/orca/workspaces/Strata/sycl')
    for name, value in previous['frozen_files'].items():
        # Old build products/configurations are historical evidence, not inputs
        # to this fresh build. Their retirement must not invalidate this recipe.
        if not Path(name).is_relative_to(old_primary) and not Path(name).is_relative_to(B):
            if '/MistVVK/XeStrata/' in name:
                llvm = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
                assert Path(name).is_relative_to(llvm), 'XeStrata project source is excluded'
            assert sha(name) == value['sha256'], name
            pins[name] = value['sha256']
    for path in (Path(__file__), owner_path, helper, previous_path):
        pins[str(path)] = sha(path)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert boot == previous['boot_id'], 'prior boot changed; qualify current runtime first'
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        health.no_model_owners(metadata['known_model_files'])
        health.driver_guard()
        foreign = owned.identity(3431768)
        assert foreign is None or foreign['start_ticks'] != 52544839 or foreign['state'] == 'Z'
        root.mkdir(mode=0o700)
        (root / 'tmp').mkdir()
        env = dict(previous['environment'])
        env.update(TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
        env.pop('PYTHONPATH', None)
        record = dict(schema='strata-moe-window-engine-build-v1', passed=False, complete=False,
                      active=False, GPU_executed=False, offline_AOT_may_query_device=True,
                      source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
                      source_pins=pins, boot_id=boot, environment=env, commands=[],
                      fresh_faults=None, performance_adopted=False, model_qualified=False,
                      physical262144_qualified=False, RESTORE_qualified=False,
                      scope='Fresh Release engine and manual GPU fixture build only. No inference or fixture execution.',
                      retention=dict(root=str(root), raw_budget_bytes=4 << 30,
                                     consumer='Main MoE window codegen/link and subsequent GPU validation',
                                     review_point='After candidate GPU/long-model qualification or distinct build failure correction.'))
        owner = None

        def save():
            record['active'] = bool(owner and owner.active is not None)
            out.write_text(json.dumps(record, indent=2) + '\n')

        owner = owned.Owner(root / 'owned', save)
        record['commands'] = owner.commands

        def run(label, argv, wall=60, cpu=120):
            result, stdout, stderr = owner.run(label, argv, env, P, wall=wall, cpu=cpu,
                                               rss_cap=16 << 30, text_cap=64 << 20,
                                               file_cap=1 << 30, total_cap=4 << 30)
            assert owned.closed(result) and result['exit_code'] == 0, (label, result.get('exit_code'), str(stderr))
            return stdout.read_text()

        cursor = None
        save()
        try:
            cursor = re.search(r'-- cursor: (\S+)', run('journal-before', ['/usr/bin/journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager']))[1]
            record['journal_before_cursor'] = cursor
            configure = list(previous['configure_argv'])
            configure[configure.index('-S') + 1] = str(P)
            configure[configure.index('-B') + 1] = str(root / 'build')
            record['configure_argv'] = configure
            run('configure', configure, wall=300, cpu=600)
            targets = ['strata', 'sycl_moe_window_test']
            run('compile-link', ['/usr/bin/cmake', '--build', str(root / 'build'), '--target', *targets, '--parallel', '2', '--verbose'], wall=2400, cpu=18000)
            record['products'] = {name: dict(path=str(root / 'build' / name), sha256=sha(root / 'build' / name), bytes=(root / 'build' / name).stat().st_size) for name in targets}
            database = root / 'build/compile_commands.json'
            entries = json.loads(database.read_text())
            selected = ['src/prefill/prefill.cpp', 'src/program/generate.cpp', 'tests/sycl/moe_window.cpp']
            record['actual_compile_entries'] = [next(e for e in entries if e['file'] == str(P / name)) for name in selected]
            record['compile_database_sha256'] = sha(database)
            assert all(sha(name) == value for name, value in pins.items()), 'build inputs changed'
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            try:
                assert owner.active is None
                # Collect fault evidence even when the device guard now fails.
                if cursor:
                    raw = run('journal-final', ['/usr/bin/journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    record['fresh_faults'] = [row for row in rows if FAULT.search(str(row.get('MESSAGE', '')))]
                record['ownership_closed'] = all(owned.closed(c) for c in owner.commands)
                record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot
                record['driver_dump_exists'] = (DEV / 'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*')))
                assert cursor and record['fresh_faults'] == [] and record['ownership_closed'] and record['boot_unchanged'] and not record['driver_dump_exists']
                health.driver_guard()
            except BaseException as error:
                record.update(closure_error=repr(error), passed=False, complete=False)
            record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save()
        print(json.dumps(dict(passed=record['passed'], complete=record['complete'], receipt=str(out))), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
