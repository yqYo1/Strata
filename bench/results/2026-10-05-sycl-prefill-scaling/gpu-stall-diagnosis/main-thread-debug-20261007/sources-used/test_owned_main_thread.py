"""Real CPU/GDB checks for watchdog/main register attribution and partial reads."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import time

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = base / 'owned-main-thread-cpu'
out.mkdir(mode=0o700)
source = out / 'thread-fixture.cpp'
source.write_text('''#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>
std::atomic<bool> ready{false};
[[gnu::noinline]] void main_wait() { for (;;) ::pause(); }
int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "bad-pc") == 0) {
        reinterpret_cast<void(*)()>(1)(); // Intentional CPU crash fixture only.
        return 9;
    }
    const bool crash = argc == 2 && std::strcmp(argv[1], "worker-abort") == 0;
    std::thread worker([crash] {
        while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
        if (crash) {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            std::abort();
        }
        for (;;) ::pause();
    });
    ready.store(true, std::memory_order_release);
    main_wait();
}
''')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

env = dict(os.environ)
env.pop('LD_LIBRARY_PATH', None)
env.pop('LD_PRELOAD', None)
env['PATH'] = '/usr/bin:/bin'
binary = out / 'thread-fixture'
record = {'scope': 'CPU-only owned-GDB main-thread register attribution, crash preservation, partial unreadable memory and SIGINT resume; no GPU loading/submission',
          'passed': False, 'cases': [], 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'source_sha256': digest(Path(__file__))}
compiler = ['/usr/bin/g++', '-std=c++17', '-O0', '-g', '-pthread', str(source), '-o', str(binary)]
record['compiler_argv'] = compiler
with (out/'build.stdout').open('wb') as stdout, (out/'build.stderr').open('wb') as stderr:
    result = subprocess.run(compiler, env=env, stdout=stdout, stderr=stderr, timeout=20)
assert result.returncode == 0
record['fixture_sha256'] = digest(source)
record['binary_sha256'] = digest(binary)

def run_case(name, helper_path, mode, expected):
    spec = importlib.util.spec_from_file_location(name, helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    item = {'name': name, 'helper_path': str(helper_path), 'helper_sha256': digest(helper_path),
            'mode': mode, 'expected': expected, 'passed': False}
    record['cases'].append(item)
    g = module.OwnedGdb([str(binary), mode], out/name, env)
    try:
        g.run()
        deadline = time.monotonic() + 15
        if mode == 'sleep':
            while time.monotonic() < deadline:
                g.poll(.05)
                if g.inferior and 'thread-created,id="2"' in (out/name/'gdb-mi.stdout').read_text():
                    break
            else:
                raise TimeoutError('CPU worker did not start')
        else:
            while not g.stops and time.monotonic() < deadline:
                g.poll(.05)
            assert g.stops, 'CPU fixture did not stop'
        snap = g.snapshot('snapshot', resume=True)
        item['snapshot'] = snap
        item['inferior'] = g.inferior
        item['debugger'] = g.debugger_identity
        current = re.search(r'current-thread-id="(\d+)"', g.command('-thread-info'))[1]
        item['selected_thread_after_snapshot'] = current
        decoded = ''
        for line in Path(snap['path']).read_text().splitlines():
            if line.startswith('~'):
                try:
                    decoded += json.loads(line[1:])
                except ValueError:
                    pass
        item['register_dump_count'] = len(re.findall(r'^rax\s', decoded, re.M))
        if mode == 'worker-abort':
            assert 'signal-name="SIGABRT"' in snap['stop'] and not snap['resumed']
            assert current == '2', 'First crash thread selection was not restored'
        elif mode == 'sleep':
            assert 'signal-name="SIGINT"' in snap['stop'] and snap['resumed']
            assert module.process_identity(g.inferior['pid'])
        else:
            assert 'signal-name="SIGSEGV"' in snap['stop'] and not snap['resumed']
        if expected == 'missing-main-registers':
            assert snap.get('main_thread_id') is None and item['register_dump_count'] == 1
        else:
            assert snap['main_thread_id'] == '1' and snap['main_registers_captured']
            assert snap['main_stack_words_captured'] and item['register_dump_count'] == 2
            assert 'strata diagnostic: main thread 1' in decoded
            if mode == 'bad-pc':
                assert not snap['main_instructions_captured'] and snap['capture_errors']
            else:
                assert snap['main_instructions_captured'] and not snap['capture_errors']
            if mode == 'worker-abort':
                assert snap['register_thread_id'] == '2'
        item['passed'] = True
    except BaseException as error:
        item['error'] = repr(error)
        raise
    finally:
        item['cleanup'] = g.close()
        assert not item['cleanup']['inferior_survived'] and not item['cleanup']['gdb_survived']
        (out/'record.json').write_text(json.dumps(record, indent=2)+'\n')

try:
    run_case('previous-worker-abort', base/'owned_gdb-before-main-registers.py', 'worker-abort', 'missing-main-registers')
    run_case('current-worker-abort', root/'sycl/tools/owned_gdb.py', 'worker-abort', 'captures-main-registers')
    run_case('current-interrupt-resume', root/'sycl/tools/owned_gdb.py', 'sleep', 'captures-main-registers')
    run_case('current-unreadable-pc', root/'sycl/tools/owned_gdb.py', 'bad-pc', 'captures-main-registers')
    record['passed'] = True
except BaseException as error:
    record['error'] = repr(error)
    raise
finally:
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'record.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
