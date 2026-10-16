#!/usr/bin/env python3
"""Bounded real-adapter static-owner teardown diagnostic on the failed build."""
import fcntl
import json
from pathlib import Path
import re
import shlex
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent; P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failure = json.loads((A / 'own-native16-model-smoke-v1.json').read_text())
        require(not failure['passed'] and not failure['active'] and not failure['fresh_faults'] and failure['boot_unchanged'] and not failure['driver_dump_exists'], 'host failure closed without GPU fault')
        for c in failure['commands']:
            require(module.closed(c), 'failed process closed')
            for old in c['owners']:
                now = module.identity(old['pid']); require(now is None or now['start_ticks'] != old['start_ticks'], 'old owned process alive')
        build = json.loads((A / 'own-native16-prefill-build-v1.json').read_text())
        require(all(sha(P / n) == h for n, h in build['source_pins'].items()), 'exact failed source')
        bd = Path(build['build_path']); root = B / 'main-own-registry-teardown-repro-root-v1'
        target = A / 'own-registry-teardown-repro-v1.json'
        require(not root.exists() and not target.exists(), 'fresh diagnostic')
        root.mkdir(mode=0o700); exe = root / 'allocation_teardown'
        env = json.loads((A / 'own-native16-ops-correctness-v1.json').read_text())['environment']
        env = dict(env, XDG_CACHE_HOME=str(root / 'cache'), UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION')
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[], boot_id=boot_id(),
            hypothesis='Function-static DeviceRegistry destroyed before the pre-main compatibility allocations map; late Allocation::release erases a dead map.',
            source_commit=build['source_commit'], registry_source_sha256=sha(P / 'src/sycl/runtime.cpp'),
            adapter_source_sha256=sha(P / 'src/sycl/compat.cpp'), fixture_sha256=sha(P / 'tests/sycl/allocation_teardown.cpp'),
            environment=env, recipe_sha256=sha(Path(__file__)), scope='Host static-owner lifetime reproducer; no model/kernel speed/quality claim',
            diagnostic_budget_bytes=512 << 20)
        def persist(): (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist); record['commands'] = owner.commands; persist()
        hostenv = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        def run(label, argv, current, wall=180):
            c, out, err = owner.run(label, argv, current, bd, wall=wall, cpu=1800, rss_cap=8 << 30,
                text_cap=64 << 20, total_cap=1 << 30, file_cap=512 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label + ': ' + err.read_text()[-4000:])
            return out, err
        cursor = None
        try:
            cc = json.loads((bd / 'compile_commands.json').read_text())
            args = shlex.split(next(v['command'] for v in cc if v['file'] == str(P / 'tests/sycl/runtime.cpp')))
            obj = root / 'fixture.o'; args[args.index('-o') + 1] = str(obj); args[args.index('-c') + 1] = str(P / 'tests/sycl/allocation_teardown.cpp')
            record['actual_compile_command'] = args; persist(); run('compile', args, build['environment'])
            out, _ = run('link-recipe', ['ninja', '-C', str(bd), '-t', 'commands', 'sycl_compat_test'], build['environment'])
            toks = shlex.split(out.read_text().splitlines()[-1]); require(toks[:2] == [':', '&&'] and toks[-2:] == ['&&', ':'], 'link envelope')
            args = toks[2:-2]; old = 'CMakeFiles/sycl_compat_test.dir/tests/sycl/compat.cpp.o'
            args[args.index(old)] = str(obj); args[args.index('-o') + 1] = str(exe)
            record['actual_link_command'] = args; persist(); run('link', args, build['environment'])
            require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'GPU preflight')
            out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], hostenv)
            cursor = re.search(r'-- cursor: (\S+)', out.read_text())[1]
            record['GPU_executed'] = True
            current = dict(env, UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION', UR_LOG_TRACING='level:info;flush:info;output:stderr')
            out, err = run('early', [str(exe), '--early'], current)
            record['early_stdout'] = out.read_text(); require('early main complete' in out.read_text(), 'early control')
            out, err = run('late-gdb', ['gdb', '--batch', '--quiet', '-ex', 'set confirm off', '-ex', 'run',
                '-ex', 'thread apply all bt 12', '-ex', 'quit', '--args', str(exe)], env)
            text = out.read_text() + err.read_text()
            record['gdb_stdout'] = out.read_text(); record['gdb_stderr'] = err.read_text()
            record['hypothesis_reproduced'] = all(s in text for s in ('SIGABRT', 'Allocation::release', 'fasttop'))
            require(record['hypothesis_reproduced'], 'not the predicted failure; inspect without assuming cause')
            record.update(passed=True, complete=True, binary=str(exe), binary_sha256=sha(exe))
        except BaseException as error: record['error'] = repr(error); raise
        finally:
            try:
                if cursor and owner.active is None:
                    out, _ = run('journal-final', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'], hostenv)
                    rows = [json.loads(s) for s in out.read_text().splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in rows if FAULT.search(str(v.get('MESSAGE', '')))], boot_unchanged=boot_id() == record['boot_id'], driver_dump_exists=(DEVICE / 'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None, finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'): record.update(passed=False, complete=False)
                persist()
                with target.open('x') as f: json.dump(record, f, indent=2); f.write('\n')
        print(json.dumps(dict(passed=record['passed'], reproduced=record['hypothesis_reproduced'], active=record['active'])), flush=True)


if __name__ == '__main__': main()
