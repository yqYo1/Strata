#!/usr/bin/env python3
"""Main's first native-only Chrome device-timeline schema check; no speed claim."""
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import re
import selectors
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'main-resource-timeline-schema-smoke-v1'
TARGET = A / 'own-resource-timeline-schema-smoke-v1.json'


def main():
    source = A / 'own-vllm-unitrace-direct-v6/controller.py'
    spec = importlib.util.spec_from_file_location('qualified_direct_collector', source)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    manifest = json.loads((source.parent / 'manifest.json').read_text())
    h.require(h.sha(source.parent / 'manifest.json') == '1e9fa3cb57e76405deda315dbdc91b8d43cbf0a98c1d62172661fa5d31da47c0', 'original collector manifest')
    h.check_pin(source, manifest['package_files']['controller.py'])
    h.require(h.sha(h.OWNER) == h.OWNER_SHA, 'qualified owner')
    spec = importlib.util.spec_from_file_location('owned_schema_probe', h.OWNER)
    owner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner_module)
    build_path = A / 'own-prompt-softmax-build-v3.json'
    h.require(h.sha(build_path) == '3fca64de3664a74ac5a376e4e40cb8a3e2910113f29a4597a564c2f903381327', 'qualified current build')
    build = json.loads(build_path.read_text())
    prior_path = A / 'own-prompt-softmax-components-smoke-v1.json'
    h.require(h.sha(prior_path) == 'f59bc11c027046d36defe2fba2ab496b1165671cca9f2eb79ad84cdab5093e4e', 'qualified original numeric smoke')
    prior = json.loads(prior_path.read_text())
    for v in (build, prior):
        h.require(v['passed'] and v['complete'] and not v['active'] and v['ownership_closed'] and not v['fresh_faults'] and v['driver_dump_exists'] is False and v['boot_id'] == h.boot_id(), 'closed current build/component')
        for c in v['commands']:
            h.require(owner_module.closed(c) and c['exit_code'] == 0, 'original normal commands')
        h.absence(v['commands'])
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        h.require(not ROOT.exists() and not TARGET.exists(), 'fresh schema probe')
        for name, digest in build['source_pins'].items():
            h.require(h.sha(P / name) == digest, 'current compiled source')
        for name, pin in build['pins'].items():
            h.check_pin(name, pin)
        for pin in manifest['elf_pins'].values():
            h.check_pin(pin['path'], pin)
        for name in ('tools/unitrace/src/chromelogger.h', 'tools/unitrace/src/tracer.cc',
                     'tools/unitrace/src/levelzero/ze_collector.h'):
            path = '/home/yayoi/ghq/github.com/intel/pti-gpu/' + name
            h.check_pin(path, manifest['file_pins'][path])
        h.require(not (h.DEVICE / 'devcoredump').exists() and (h.DEVICE / 'driver').resolve().name == 'xe', 'driver/dump gate')
        ROOT.mkdir(mode=0o700)
        collector = ROOT / 'collector'
        collector.mkdir(mode=0o700)
        for name in ('cache', 'tmp'):
            (ROOT / name).mkdir(mode=0o700)
        exe = build['products']['sycl_prompt_native16_test']
        h.check_pin(exe['path'], exe)
        pins = dict(manifest['elf_pins'])
        h.require(exe['realpath'] not in pins, 'new executable pin is explicit, no dependency overwrite')
        pins[exe['realpath']] = exe
        env = dict(next(c['environment'] for c in prior['commands'] if c['label'] == 'bf16-smoke-0'))
        env.update(h.COLLECTOR, UNITRACE_ChromeDeviceLogging='1', UNITRACE_LogFilename=str(collector / 'timing.log'), LD_PRELOAD=str(h.LIB), XDG_CACHE_HOME=str(ROOT / 'cache'), TMPDIR=str(ROOT / 'tmp'))
        h.require(h.sha(h.LIB) == h.LIB_SHA, 'qualified official collector binary')
        record = dict(schema='strata-resource-timeline-schema-smoke-v1', passed=False, complete=False, active=False, GPU_executed=False,
            boot_id=h.boot_id(), commands=[], source_commit=build['source_commit'], source_pins=build['source_pins'],
            controller=h.artifact(__file__), build_receipt=h.artifact(build_path), prior_component=h.artifact(prior_path),
            executable=exe, collector_library=h.artifact(h.LIB), reused_controller=h.artifact(source), environment=env,
            budget=dict(native_seconds=90, raw_bytes=64 << 20, private_bytes=1 << 30),
            retention=dict(owner='Main', consumer='First exact-runtime device-timeline schema and copy-byte evidence; raw consumed by resource timeline parser qualification', review='at parser qualification or2026-10-17T03:00:00Z, whichever comes first'),
            scope='Two existing FP64 component cases with newly enabled native-only device timeline. API tracing/validation enabled before exec. No model, clean speed, PMU, physical bus bytes, occupancy, overlap, critical path or model-quality claim.',
            hardware_bottleneck_proven=False, performance_qualified=False, model_quality_qualified=False, physical_262144_qualified=False, restore_qualified=False)
        owner = None

        def persist():
            record['active'] = owner is not None and owner.active is not None
            total = sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
            raw = sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file() and (f.is_relative_to(collector) or f.suffix in ('.stdout', '.stderr', '.log')))
            record.update(private_bytes=total, raw_bytes=raw)
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
            h.require(total <= 1 << 30 and raw <= 64 << 20, 'bounded native and side files')

        owner = owner_module.Owner(ROOT / 'owned', persist)
        record['commands'] = owner.commands
        persist()

        def utility(label, args):
            c, out, err = owner.run(label, args, h.HOST_ENV, P, wall=30, cpu=60, rss_cap=2 << 30, text_cap=2 << 20, total_cap=1 << 30, file_cap=2 << 20)
            h.require(owner_module.closed(c) and c['exit_code'] == 0, label)
            return out.read_text()

        cursor = None
        proof = {}
        try:
            record['owner_scan'] = h.owner_scan(manifest['model_stat_pins'])
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager']))[1]

            def interaction(io, entry):
                record['GPU_executed'] = True
                io.proc.stdin.close()
                with selectors.DefaultSelector() as ready:
                    ready.register(io.proc.stdout, selectors.EVENT_READ)
                    while not io.exit_watch.poll(0) or not io.eof:
                        io.poll()
                        if not proof and not io.exit_watch.poll(0):
                            snapshot = h.native_snapshot(io.proc.pid, exe, env, pins, io.proc.pid)
                            if snapshot:
                                proof.update(snapshot)
                        if ready.select(.002):
                            io.read()
                            io.pending.clear()

            c, out, err = owner.run('timeline-smoke', [exe['path'], '--smoke'], env, collector, interaction=interaction,
                wall=90, cpu=180, rss_cap=8 << 30, text_cap=64 << 20, total_cap=1 << 30, file_cap=64 << 20)
            h.require(owner_module.closed(c) and c['exit_code'] == 0, 'native normal closure')
            record['actual_native'] = proof
            h.require(proof and proof['native']['pid'] == c['pid'] and proof['native']['start_ticks'] == c['direct_identity']['start_ticks'], 'actual owned native mapping/environment')
            stdout, stderr = out.read_text(), err.read_text()
            record['oracle_stdout'] = stdout
            cases = [s for s in stdout.splitlines() if s.startswith('case dtype=bf16 ')]
            h.require(len(cases) == 2 and all('oracle=FP64_full ' in s and 'finite=PASS guards=PASS inputs=PASS qualification=COMPONENT_ONLY_UNQUALIFIED' in s for s in cases) and 'fixture=PASS model_quality=UNRUN whole_prefill=UNRUN physical262144=UNRUN faults=EXTERNAL_REQUIRED' in stdout and not h.ERROR.search(stderr), 'unchanged independent oracle and native success')
            files = [f for f in collector.iterdir() if f.is_file()]
            timelines = [f for f in files if f.suffix == '.json']
            timing = collector / ('timing.' + str(c['pid']) + '.log')
            h.require(len(timelines) == 1 and set(files) == {timing, timelines[0]} and str(c['pid']) in timelines[0].name, 'exact native PID-named summary and complete timeline, no unexpected side')
            data = json.loads(timelines[0].read_text())
            h.require(isinstance(data.get('traceEvents'), list), 'Chrome trace container')
            gpu = [v for v in data['traceEvents'] if v.get('cat') == 'gpu_op' and v.get('ph') == 'X']
            h.require(gpu and all(isinstance(v.get('ts'), (int, float)) and isinstance(v.get('dur'), (int, float)) and math.isfinite(v['ts']) and math.isfinite(v['dur']) and v['dur'] >= 0 and isinstance(v.get('args'), dict) for v in gpu), 'finite complete device intervals')
            copies = [v for v in gpu if 'MemoryCopy' in v['name']]
            record['timeline_schema'] = dict(file=h.artifact(timelines[0]), summary_file=h.artifact(timing), top_level_keys=list(data), event_count=len(data['traceEvents']), GPU_rows=len(gpu), copy_rows=len(copies), GPU_example=gpu[0], copy_examples=copies[:2], no_interval_deduplication_or_utilization_claim=True)
            h.require(copies and all('bytes' in v.get('args', {}) or re.search(r'\[\d+\]', v['name']) for v in copies), 'explicit copy size evidence')
            for name, digest in build['source_pins'].items():
                h.require(h.sha(P / name) == digest, 'same source after')
            for name, pin in build['pins'].items():
                h.check_pin(name, pin)
            h.check_pin(exe['path'], exe)
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            if cursor and owner.active is None:
                try:
                    rows = [json.loads(s) for s in utility('journal-final', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json']).splitlines() if s.strip()]
                    record['fresh_faults'] = [v for v in rows if h.FAULT.search(str(v.get('MESSAGE', '')))]
                except BaseException as error:
                    record['fault_check_error'] = repr(error)
            record.update(boot_unchanged=h.boot_id() == record['boot_id'], driver_dump_exists=(h.DEVICE / 'devcoredump').exists(), ownership_closed=owner.active is None and all(owner_module.closed(c) for c in owner.commands), finished_utc=h.utc())
            if not record['ownership_closed'] or record.get('fresh_faults') != [] or not record['boot_unchanged'] or record['driver_dump_exists']:
                record.update(passed=False, complete=False)
            h.absence(owner.commands)
            persist()
            with TARGET.open('x') as f:
                json.dump(record, f, indent=2)
                f.write('\n')
        print(json.dumps(dict(passed=record['passed'], error=record.get('error'), receipt=str(TARGET), sha256=h.sha(TARGET))), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
