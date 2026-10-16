#!/usr/bin/env python3
"""Main serial CPU/RAM copy capacity proxy, three independent processes.
Repository LICENSE. No SYCL/Level Zero initialization or GPU access.
"""
import collections
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def pin(p):
    p = Path(p)
    with p.open('rb') as stream:
        h = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(p), realpath=str(p.resolve()), bytes=p.stat().st_size, sha256=h)


def main():
    root = B / 'main-host-expert-copy-capacity-v1'
    target = A / 'own-host-expert-copy-capacity-v1.json'
    assert not root.exists() and not target.exists()
    source = A / 'host_expert_copy_capacity_v1.cpp'
    metadata = A / 'prefill-capacity-metadata-v2.json'
    capacity = json.loads(metadata.read_text())
    frequencies = collections.Counter(v['expert_bytes'] for v in capacity['layers'])
    assert sorted(frequencies) == [1510400,1715200,1868800,1971200,2176000,2329600,2662400]
    assert sum(frequencies.values()) == 48
    payload = sum(blob * 512 * count for blob, count in frequencies.items())
    assert payload == 50292326400
    physical = {}
    topology = []
    for cpu in sorted(os.sched_getaffinity(0)):
        t = Path('/sys/devices/system/cpu') / ('cpu' + str(cpu)) / 'topology'
        package = int((t / 'physical_package_id').read_text())
        core = int((t / 'core_id').read_text())
        topology.append(dict(cpu=cpu, package=package, core=core,
            siblings=(t / 'thread_siblings_list').read_text().strip()))
        physical.setdefault((package, core), cpu)
    assert len(physical) >= 6
    cpus = list(physical.values())[:6]
    owner_source = A.parents[2] / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert pin(owner_source)['sha256'] == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('host_copy_capacity_owner', owner_source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
        record = dict(schema='strata-host-expert-copy-capacity-v1', passed=False, complete=False,
            active=False, commands=[], GPU_executed=False, source=pin(source), metadata=pin(metadata),
            actual_product_copy_source=pin(A.parents[2] / 'src/prefill/layer_major.inc'),
            actual_expert_source=pin(A.parents[2] / 'src/core/expert_source.cpp'),
            compiler=pin('/usr/bin/g++'), libc=pin('/usr/lib/x86_64-linux-gnu/libc.so.6'),
            standard_library=pin('/usr/lib/x86_64-linux-gnu/libstdc++.so.6'),
            environment=env, topology=topology, selected_physical_cpus=cpus,
            expert_size_layer_frequencies=dict(frequencies), expected_layer_count=48,
            full_layer_payload_bytes=payload, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            kernel=os.uname().release, samples=[], scope='Ordinary aligned anonymous RAM, 512 independent std::memcpy calls per layer, 7 actual blob sizes, 1/3/6 pinned physical cores. Not live USM/pinned/mapped-model input or physical memory-controller traffic.',
            limits=dict(process_seconds=90, process_count=3, raw_text_bytes=64 << 20, process_rss_bytes=6 << 30),
            retention=dict(owner='Main', consumer='Resource capacity comparison with source-copy phase',
                next_review_utc='2026-10-17T03:00:00Z'))
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = m.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        binary = root / 'host-expert-copy-capacity'
        try:
            argv = ['/usr/bin/g++', '-O3', '-std=c++20', '-pthread', '-Wall', '-Wextra', '-Werror',
                    '-fno-builtin-memcpy', str(source), '-o', str(binary)]
            c, _, _ = owner.run('host-build', argv, env, A.parents[2], wall=45, cpu=60,
                rss_cap=2 << 30, text_cap=1 << 20, file_cap=1 << 20, total_cap=64 << 20)
            assert m.closed(c) and c['exit_code'] == 0
            record['binary'] = pin(binary)
            for process in range(3):
                assert pin(source) == record['source'] and pin(metadata) == record['metadata']
                c, out, err = owner.run('host-copy-' + str(process), [str(binary), *map(str, cpus)], env,
                    A.parents[2], wall=90, cpu=180, rss_cap=6 << 30, text_cap=1 << 20,
                    file_cap=1 << 20, total_cap=64 << 20)
                assert m.closed(c) and c['exit_code'] == 0 and not err.read_text()
                rows = [json.loads(line) for line in out.read_text().splitlines()]
                assert len(rows) == 84 and all(v['passed'] and v['full_compare'] and v['affinity'] for v in rows)
                assert {(v['blob_bytes'],v['threads'],v['rep']) for v in rows} == {
                    (b,t,r) for b in frequencies for t in (1,3,6) for r in range(4)}
                for row in rows:
                    row['process'] = process
                    assert row['payload_bytes'] == 512 * row['blob_bytes'] and row['warmup'] == (row['rep'] == 0)
                record['samples'].extend(rows)
            summaries = []
            for blob in sorted(frequencies):
                for threads in (1,3,6):
                    rows = [v for v in record['samples'] if v['blob_bytes'] == blob and v['threads'] == threads and not v['warmup']]
                    process_medians = [statistics.median(v['payload_GBps'] for v in rows if v['process'] == p) for p in range(3)]
                    summaries.append(dict(blob_bytes=blob,threads=threads,samples=len(rows),
                        process_medians_GBps=process_medians,median_of_process_medians_GBps=statistics.median(process_medians),
                        sample_min_GBps=min(v['payload_GBps'] for v in rows),sample_max_GBps=max(v['payload_GBps'] for v in rows)))
            record['summary'] = summaries
            record['synthetic_48_layer_copy_seconds'] = {str(t): sum(
                blob * 512 * count / next(v['median_of_process_medians_GBps'] for v in summaries if v['threads'] == t and v['blob_bytes'] == blob) / 1e9
                for blob,count in frequencies.items()) for t in (1,3,6)}
            record.update(passed=True, complete=True)
        except BaseException as error:
            import traceback
            record.update(error=repr(error), traceback=traceback.format_exc())
        finally:
            record['ownership_closed'] = owner.active is None and all(m.closed(c) for c in owner.commands)
            for c in owner.commands:
                for old in c['owners'] + [c['direct_identity']]:
                    current = m.identity(old['pid'])
                    assert current is None or current['start_ticks'] != old['start_ticks']
            persist()
            target.write_bytes((root / 'record.json').read_bytes())
        print(json.dumps(dict(receipt=pin(target), passed=record['passed'], error=record.get('error'),
            all_48_layers_proxy_seconds=record.get('synthetic_48_layer_copy_seconds'),
            sample_count=len(record['samples']), GPU_executed=False)))
        return 0 if record['passed'] and record['ownership_closed'] else 1


if __name__ == '__main__':
    sys.exit(main())
