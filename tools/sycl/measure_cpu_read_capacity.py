#!/usr/bin/env python3
"""Bounded Zen3 read-payload calibration; independently authored, repository LICENSE.

Reuse the owned host compiler/test supervisor. This is not physical bus traffic,
an absolute hardware ceiling,CPU arithmetic saturation or model throughput.
"""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import decode_service_contract as original

SELF = Path(__file__).resolve()
SOURCE = 'bench/sycl/cpu_read_capacity.cpp'
require = original.require


def hardware():
    cpus = sorted(os.sched_getaffinity(0))
    require(len(cpus) == 12 and os.sysconf('SC_PAGE_SIZE') == 4096,
            'this12-thread/4KiB-page host required')
    cores = [int(Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/core_id').read_text()) for cpu in cpus]
    require(len(set(cores[:6])) == 6, 'first six IDs must be distinct physical cores')
    cpuinfo = Path('/proc/cpuinfo').read_text()
    require('AMD Ryzen 5 5600X' in cpuinfo and 'avx2' in cpuinfo, 'qualified Zen3/AVX2 machine required')
    return dict(cpus=cpus, cores=cores, page_bytes=4096,
        groups=[cpus[:n] for n in (1,2,4,6,12)],
        THP_policy=Path('/sys/kernel/mm/transparent_hugepage/enabled').read_text().strip(),
        L3=Path(f'/sys/devices/system/cpu/cpu{cpus[0]}/cache/index3/size').read_text().strip())


def output(fixture, path, row):
    value = json.loads(path.read_text())
    require(value['working_set_bytes'] == 1 << 30 and value['passes'] == 7 and value['samples'] == 5 and
        value['first_touch_cpu'] == row['hardware']['cpus'][0] and
        value['checksums_and_affinity_passed'] is True and value['GPU_executed'] is False and
        len(value['conditions']) == 5, 'complete payload/affinity controls')
    n = (1 << 30)//8
    expected = ((n*(n-1)//2*0x9e3779b97f4a7c15+n*0x123456789abcdef0)*7) % (1 << 64)
    for condition, cpus in zip(value['conditions'], row['hardware']['groups']):
        require(condition['threads'] == len(cpus) and condition['cpus'] == cpus and
            condition['checksum'] == f'{expected:016x}' and len(condition['elapsed_ns']) == 5 and
            len(condition['minor_faults']) == len(condition['major_faults']) == 5 and
            all(type(t) is int and 100000000 <= t <= 10000000000 for t in condition['elapsed_ns']) and
            all(t == 0 for t in condition['minor_faults'] + condition['major_faults']),
            'work/checksum/duration/fault eligibility')
        rates = [(1 << 30)*7 / elapsed for elapsed in condition['elapsed_ns']]
        condition.update(payload_GB_per_s=rates, median_payload_GB_per_s=statistics.median(rates),
            minimum_payload_GB_per_s=min(rates), maximum_payload_GB_per_s=max(rates))
    row['read_capacity'] = dict(value, physical_bus_traffic_qualified=False,
        absolute_hardware_ceiling_qualified=False, model_bottleneck_qualified=False,
        assembly_review_pending=True,
        scope='1GiB ordinary anonymous4KiB-page mapping,first-touch CPU0,one untimed pass then five7-pass batches per CPU group. Parent start/finish barriers and thread scheduling remain in wall time. Payload rates are decimalGB/s,not measured DRAM transactions or inference speed.')


def context():
    spec = importlib.util.spec_from_file_location('strata_cpu_read_capacity_host', original.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY = ctx.B / 'cpu-read-capacity-v1', 'cpu_read_capacity_v1'
    ctx.SOURCES = (SOURCE, str(SELF.relative_to(ctx.P)), 'tools/sycl/decode_service_contract.py',
        'tools/sycl/measurement_guards.py', 'tools/sycl/measurement_owner.py',
        'AGENTS.md', 'docs/ARTIFACT_RETENTION.md')
    ctx.FIXTURES = (('read', SOURCE, 5, 'validated JSON capacity conditions'),)
    ctx.check_output = output
    return ctx


def prepare(ctx):
    host = hardware()
    ctx.prepare()
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    row.pop('profile_schema_version')
    row.update(capacity_schema_version=1, hardware=host,
        plan='Compile one disposable CPU-only fixture,then run its small checksum control and five ordered1/2/4/6/12-thread payload conditions. One1GiB4KiB-page mapping and first-touch CPU0;each condition has one untimed full pass and five timed7-pass samples. Check every worker checksum and actual affinity,zero timed minor/major faults,100ms minimum batch,normal0/quiet stderr/closed owners and exact committed source/compiler/dependencies. Barriers/scheduling stay charged. Do not retry,refresh hashes,discard samples or change floors. Static assembly review precedes interpreting rates. No GPU,engine build,model execution,physical bus counter or saturation claim.',
        retention='One short-lived auxiliary CPU-capacity fixture,not a third inference engine. Keep source,recipe,individual times/rates/checksums/affinity/faults and interpretation. Preserve both retained engine generations. Retire this auxiliary ELF,dependency output and successful raw streams after useful evidence is pushed and exact source/recipe is recovered;consumer ends at this capacity result.')
    command = row['fixtures'][0]['compile']
    command[command.index('-O2')] = '-O3'
    command[command.index('-o'):command.index('-o')] = ['-march=znver3','-pthread','-fno-lto']
    row['test_budget'] = dict(wall=60,cpu=120,rss_cap=2 << 30,text_cap=64 << 10,
        file_cap=1 << 20,total_cap=16 << 20)
    ctx.save(row)
    print('Prepared five CPU read-payload conditions;commit before compile/run.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'],['run']), 'use prepare|run')
    ctx = context()
    with (ctx.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        ctx.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare(ctx)
        else:
            row = json.loads((ctx.ROOT / 'record.json').read_text())
            require(hardware() == row['hardware'], 'host topology/placement changed')
            ctx.run()
