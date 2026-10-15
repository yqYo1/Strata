#!/usr/bin/env python3
"""Bounded actual runtime IGC capture on an exact previously qualified carrier."""
import fcntl
import json
from pathlib import Path
import re
import selectors
import sys
import time
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id


def main(arm):
    require(arm in (0, 1, 2), 'declared prompt mapping')
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ip = A / 'prompt-jit-identity-v1.json'
        identity = json.loads(ip.read_text())
        require(identity['passed'] and identity['complete'] and not identity['active'], 'actual image/name extraction closed')
        exe = Path(identity['binary'])
        require(sha(exe) == identity['binary_sha256'], 'exact carrier')
        lp = A / 'own-qsa-select-model-a1-library-hashes-v1.json'
        libraries = json.loads(lp.read_text())
        require(all(sha(v['path']) == v['sha256'] for v in libraries['files']), 'observed actual library files unchanged')
        previous = json.loads((A / 'own-prompt-k-padded-bench-d1-a0-r0-v1.json').read_text())
        require(previous['passed'] and previous['complete'] and not previous['active'] and
                previous['binary_sha256'] == sha(exe), 'qualified actual prompt carrier')
        require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'GPU preflight')
        root = B / f'main-prompt-jit-resources-a{arm}-root-v1'
        target = A / f'prompt-jit-resources-a{arm}-v1.json'
        require(not root.exists() and not target.exists(), 'fresh diagnostic')
        root.mkdir(mode=0o700)
        (root / 'igc').mkdir(mode=0o700)
        env = dict(previous['environment'])
        name = identity['prompt_specializations'][str(arm)]['name']
        env.update(XDG_CACHE_HOME=str(root / 'cache'), STRATA_SYCL_PROMPT_K_CELL_MAJOR=str(arm),
            NEO_CACHE_PERSISTENT='0', IGC_ShaderDumpEnable='1', IGC_ShaderDumpEnableRAMetadata='1',
            IGC_DumpToCustomDir=str(root / 'igc'), IGC_ShaderDumpRegexFilter='^' + re.escape(name) + '$',
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:stderr',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
            UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
            UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[], arm=arm,
            recipe_sha256=sha(Path(__file__)), binary_sha256=sha(exe), environment=env, boot_id=boot_id(),
            identity_receipt_sha256=sha(ip), library_receipt_sha256=sha(lp), exact_kernel_name=name,
            input_image_sha256=identity['prompt_specializations'][str(arm)]['image_sha256'],
            limits=dict(wall_seconds=180, RSS_bytes=8 << 30, text_bytes=64 << 20,
                        file_bytes=64 << 20, total_bytes=256 << 20, IGC_artifacts_bytes=64 << 20),
            performance_qualified=False, dynamic_resource_qualified=False,
            scope='Actual IGC runtime dump diagnostic from immutable known prompt component, nq256/repeat1. '
                  'Not clean timing/model PP, static resources are not occupancy/spill traffic.')
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        hostenv = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        def utility(label, argv):
            c, out, err = owner.run(label, argv, hostenv, root, wall=30, cpu=30,
                rss_cap=1 << 30, text_cap=4 << 20, total_cap=256 << 20, file_cap=64 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label + ': ' + err.read_text()[-2000:])
            return out.read_text()
        cursor = None
        try:
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before',
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager']))[1]
            def collect(io, entry):
                io.proc.stdin.close()
                last = 0.0
                with selectors.DefaultSelector() as ready:
                    ready.register(io.proc.stdout, selectors.EVENT_READ)
                    while not io.exit_watch.poll(0) or not io.eof:
                        io.poll()
                        if 'loaded_libraries' not in record and time.monotonic() - last > .1:
                            last = time.monotonic()
                            now = module.identity(io.proc.pid)
                            require(now is not None and now['start_ticks'] == entry['direct_identity']['start_ticks'], 'owned native mapping identity')
                            try:
                                lines = Path('/proc', str(io.proc.pid), 'maps').read_text().splitlines()
                            except FileNotFoundError:
                                lines = []
                            expected = {v['path'] for v in libraries['files']}
                            observed = {line.rsplit(None, 1)[-1] for line in lines}
                            if expected <= observed:
                                record['loaded_libraries'] = dict(identity=now, paths=sorted(expected),
                                    maps_lines=[line for line in lines if line.rsplit(None, 1)[-1] in expected])
                        if ready.select(.02):
                            io.read()
                            io.pending.clear()
            dump = root / 'output.f32'
            argv = [str(exe), '--bench', '256', '1', '--dump', str(dump)]
            c, out, err = owner.run('component-jit-capture', argv, env, root,
                interaction=collect, wall=180, cpu=600, rss_cap=8 << 30,
                text_cap=64 << 20, total_cap=256 << 20, file_cap=64 << 20)
            record.update(GPU_executed=c.get('direct_identity') is not None, stdout=out.read_text(),
                stderr_bytes=err.stat().st_size, stderr_sha256=sha(err))
            require(module.closed(c) and c['exit_code'] == 0, 'native capture: ' + err.read_text()[-3000:])
            require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:', err.read_text()), 'runtime error')
            require(dump.stat().st_size == 6291520 and sha(dump) == previous['dump']['sha256'], 'all output and guards equal qualified clean fingerprint')
            require('loaded_libraries' in record, 'actual same loaded libraries observed')
            files = sorted(f for f in (root / 'igc').rglob('*') if f.is_file())
            require(0 < len(files) <= 256 and sum(f.stat().st_size for f in files) <= 64 << 20, 'bounded actual IGC artifacts')
            record['artifacts'] = [dict(path=str(f), bytes=f.stat().st_size, sha256=sha(f)) for f in files]
            record.update(passed=True, complete=True, numerical_fingerprint_equal=True,
                retention_consumer='Main actual JIT static-resource analysis, 64MiB artifact budget, review after compact ISA/RA/resource extraction')
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor and owner.active is None:
                    entries = [json.loads(s) for s in utility('journal-final',
                        ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json']).splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in entries if FAULT.search(str(v.get('MESSAGE', '')))],
                        boot_unchanged=boot_id() == record['boot_id'], driver_dump_exists=(DEVICE / 'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None, finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False, complete=False)
                persist()
                with target.open('x') as f:
                    json.dump(record, f, indent=2)
                    f.write('\n')
        print(json.dumps(dict(passed=record['passed'], arm=arm, files=len(record['artifacts']),
                             bytes=sum(v['bytes'] for v in record['artifacts']))), flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2, 'arm0|1|2')
    main(int(sys.argv[1]))
