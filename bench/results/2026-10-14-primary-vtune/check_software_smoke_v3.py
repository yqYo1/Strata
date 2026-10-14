#!/usr/bin/env python3
"""Collect one bounded counter-free XPU smoke; capability remains unproven until extraction."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import re
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
O = B/'main-primary-vtune-software-runtime-smoke-root-v3'
TRANSFER = B/'main-primary-transfer-ceiling-root-v1/record.json'
SEED = P/'bench/results/2026-10-14-primary-model-baseline/receipts/main-primary-watchdog-lifecycle-build-root-v4/record.json.gz'
BUILD = B/'main-primary-native-build-root-v1/build'
VT = Path('/opt/intel/oneapi/vtune/2026.4/bin64/vtune')
OWNER = P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    transfer = json.loads(TRANSFER.read_text())
    assert transfer['passed'] and transfer['complete'] and not transfer['active']
    seed = json.loads(gzip.decompress(SEED.read_bytes()))
    executable = BUILD/'sycl_runtime_test'
    assert sha(executable) == seed['binaries']['sycl_runtime_test']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists()
        O.mkdir()
        env = dict(transfer['environment'], XDG_CACHE_HOME=str(O/'cache'))
        assert not any(k in env for k in ('UR_ENABLE_LAYERS', 'ZE_ENABLE_VALIDATION_LAYER', 'STRATA_TRACE', 'LD_PRELOAD'))
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        record = dict(active=True, passed=False, complete=False, collected=False, capability_qualified=False,
            model_profile_executed=False, commands=[], environment=env, boot_id=boot, fault_filter=FAULT.pattern,
            runtime_binary_sha256=sha(executable), profiler_binary_sha256=sha(VT),
            recipe_pins={str(x.relative_to(P)):sha(x) for x in (D/'runtime_child.py', Path(__file__))},
            scope='one already-qualified64-cross-queue runtime test under VTune hotspots software sampling; no CPU PMU/GPU counters or GPU timeline claim',
            limits=dict(collector_wall_seconds=180, data_limit_MB=100, text_bytes=64<<20, job_bytes=256<<20, RSS_bytes=4<<30),
            raw_result_consumer='Main API/GPU task timeline extraction and collector capability review',
            raw_result_review_point='after verifying compact timeline evidence or recording unsupported collector')
        def persist():
            (O/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        owner = module.Owner(O/'owned', persist)
        record['commands'] = owner.commands
        persist()
        cursor = None
        def run(label, argv, wall=30):
            entry, out, err = owner.run(label, argv, env, P, wall=wall, cpu=wall,
                text_cap=64<<20, rss_cap=4<<30, total_cap=256<<20)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), entry.get('exit_code'), err.read_text()[-3500:])
            return out.read_text(), err.read_text()
        try:
            assert (DEVICE/'driver').resolve().name == 'xe' and not (DEVICE/'devcoredump').exists()
            out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', out).group(1)
            record['journal_before_cursor'] = cursor
            help_out, help_err = run('hotspots-analysis-help', [str(VT), '-help', 'collect', 'hotspots'])
            assert 'sampling-mode' in help_out + help_err
            assert 'sw' in help_out + help_err
            argv = [str(VT), '-collect', 'hotspots', '-follow-child', '-data-limit=100', '-r', str(O/'result'),
                '-knob', 'sampling-mode=sw', '-knob', 'enable-stack-collection=true', '--',
                'python3', str(D/'runtime_child.py'), str(O), str(executable)]
            record['previous_gate_correction'] = 'XPU analysis rejects this Ryzen before target launch; use separately documented hotspots software sampling rather than PMU/XPU analysis'
            record['collector_argv'] = argv
            persist()
            out, err = run('collect-runtime', argv, 180)
            assert '64 changing cross-queue handoffs' in out and 'sticky asynchronous errors: PASS' in out
            real_exit = json.loads((O/'runtime-exit.json').read_text())
            assert real_exit['exit_code'] == 0 and real_exit['binary_sha256'] == record['runtime_binary_sha256']
            record.update(real_runtime_exit=real_exit, collected=True)
            summary, summary_err = run('summary', [str(VT), '-report', 'summary', '-r', str(O/'result'), '-format', 'csv'], 60)
            assert summary.strip()
            record['summary'] = summary
            record['summary_stderr'] = summary_err
            run('timeline-help', [str(VT), '-help', 'report', 'timeline'])
            run('hotspots-help', [str(VT), '-help', 'report', 'hotspots'])
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            if cursor:
                try:
                    out, err = run('journal-after', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    entries = [json.loads(x) for x in out.splitlines() if x.strip()]
                    record.update(fresh_faults=[x for x in entries if FAULT.search(str(x.get('MESSAGE', '')))],
                        driver_dump_exists=(DEVICE/'devcoredump').exists(), boot_unchanged=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot)
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False, complete=False)
                except BaseException as error:
                    record.update(passed=False, complete=False, fault_check_error=repr(error))
            record['active'] = False
            persist()
        assert record['passed'] and record['complete']
        print(json.dumps(dict(collected=record['collected'], capability_qualified=False,
            required_next='prove nonempty CPU software samples and stacks; GPU/API timeline remains unavailable')), flush=True)


if __name__ == '__main__':
    main()
