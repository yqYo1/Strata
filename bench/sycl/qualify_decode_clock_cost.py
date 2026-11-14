#!/usr/bin/env python3
"""Own the production-clock read calibration; repository LICENSE.

Independent Main implementation. Pin the actual diagnostic header and compiler;
measure five bands on host core0. This does not qualify whole-observer overhead.
"""
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'tools/sycl'))
import decode_service_contract as template

ROOT, KEY = template.B / 'decode-clock-cost-v1', 'decode_clock_cost_v1'
require = template.require


def check_output(fixture, output, row):
    lines = output.read_text().splitlines()
    require(len(lines) == 6, 'one resolution header and five complete clock bands required')
    match = re.fullmatch(r'host_clock_cost version=1 cpu=0 bands=5 reads=50000 '
        r'wall_resolution_ns=([1-9][0-9]*) cpu_resolution_ns=([1-9][0-9]*)', lines[0])
    require(match is not None, 'malformed production-clock header')
    wall_resolution, cpu_resolution = map(int, match.groups())
    require(max(wall_resolution, cpu_resolution) <= 1000, 'clock resolution exceeds declared1us limit')
    samples = []
    for index, line in enumerate(lines[1:]):
        match = re.fullmatch(r'host_clock_band index=' + str(index) +
            r' reads=50000 wall_ns=([1-9][0-9]*) cpu_ns=([1-9][0-9]*) checksum=([0-9]+)', line)
        require(match is not None, 'malformed,missing or reordered clock band')
        wall, cpu, checksum = map(int, match.groups())
        require(2000000 <= wall <= 500000000 and 2000000 <= cpu <= 500000000
            and checksum < 2**64, 'ineligible clock band:2ms floor/500ms ceiling')
        samples.append(dict(reads=50000, wall_ns=wall, cpu_ns=cpu))
    row['clock_cost'] = dict(passed=True, host_cpu=0,
        monotonic_resolution_ns=wall_resolution, thread_cpu_resolution_ns=cpu_resolution,
        samples=samples, wall_ns_per_read=[s['wall_ns'] / s['reads'] for s in samples],
        cpu_ns_per_read=[s['cpu_ns'] / s['reads'] for s in samples],
        loop_and_checks_included=True, whole_observer_overhead_qualified=False,
        limits='Observed production clock_tick backend plus loop/check cost under GCC13/O2,on core0;not production inlining cost,a worst-case bound,model overhead or hardware saturation.')


def context():
    spec = importlib.util.spec_from_file_location('strata_decode_clock_cost', template.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY = ROOT, KEY
    ctx.SOURCES = ('include/strata/sycl/decode_service_profile.hpp',
        'bench/sycl/decode_clock_cost.cpp', 'bench/sycl/qualify_decode_clock_cost.py',
        'tools/sycl/decode_service_contract.py', 'tools/sycl/measurement_guards.py',
        'tools/sycl/measurement_owner.py', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md')
    ctx.FIXTURES = (('clock-cost', 'bench/sycl/decode_clock_cost.cpp', 5, ''),)
    ctx.check_output = check_output
    return ctx


def prepare(ctx):
    native = json.loads((template.B / 'decode-service-model-v1/record.json').read_text())
    header = 'include/strata/sycl/decode_service_profile.hpp'
    require(template.sha(P / header) == native['native_sources'][header]
        and template.sha(native['binaries']['strata']['path']) == native['binaries']['strata']['sha256'],
        'clock backend must match the current real-model diagnostic engine')
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['native_header_binding'] = dict(source_commit=native['source_commit'],
        engine_sha256=native['binaries']['strata']['sha256'], header_sha256=native['native_sources'][header])
    row['plan'] = ('One owned GCC13/O2 compile,then one pure-host child pinned to core0.'
        'Five50000-read production clock_tick bands,including loop/check overhead.'
        'Require exact output,monotonic valid reads,clock resolution<=1us,and each band2..500ms.'
        'Pin source/compiler/actual dependencies/ELF and require normal0,quiet stderr,closed owners.'
        'Stop first failure;no retry or changing duration criteria. No GPU/model or whole-observer overhead qualification.')
    row['retention'] = ('Keep five individual band totals,resolutions,source/recipe and limits for the imminent profile0/1 decision. Retire one-use ELF,dependency list and normal output after committed result and source recovery;no engine generation.')
    ctx.save(row)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'use prepare|run')
    with (template.B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        template.guards.no_model_owners([])
        ctx = context()
        {'prepare': lambda: prepare(ctx), 'run': ctx.run}[sys.argv[1]]()
