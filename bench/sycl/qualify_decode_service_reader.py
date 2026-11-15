#!/usr/bin/env python3
"""Own one real C++ producer and the model-ledger reader controls.

Main-authored,repository LICENSE. This is schema/accounting qualification only;
no device/model execution,clock-overhead or inference-performance claim.
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
import decode_service_evidence as reader

SELF = Path(__file__).resolve()
ROOT = template.B / 'decode-service-reader-contract-v1'
KEY = 'decode_service_reader_contract_v1'
require = template.require


def check_output(fixture, output, row):
    text = output.read_text()
    lines = text.splitlines()
    require(len(lines) == 2, 'real producer must emit only requests1 and3')
    records = reader.extract(text, [1, 3])
    require(records[0]['wall_ns'] == 39000 and records[0]['cpu_ns'] == 3900
        and records[0]['unclassified_wall_ns'] == 12000
        and records[1]['wall_ns'] == 1000 and records[1]['cpu_ns'] == 100,
        'independent fake-clock arithmetic differs from the real C++ producer')
    first = reader.compact(records[0])
    require(first['native_wait_wall_ns'] == 10000 and first['native_wait_caller_cpu_ns'] == 1000
        and all(v['calls'] == v['native_wait_calls'] == 1 for v in first['regions'].values()),
        'nested suppression or compact caller/native-wait totals changed')
    require(reader.extract('', []) == [], 'disabled observation must accept no records')

    def changed(key, value):
        result, count = re.subn(r'(?<= )' + key + r'=[^ ]+', key + '=' + str(value), lines[0])
        require(count == 1, 'one actual producer field to corrupt: ' + key)
        return result + '\n' + lines[1] + '\n'

    bad = [(changed(k, v), [1, 3]) for k, v in (
        ('version', '02'), ('version', 3), ('available', 'true'), ('available', 0),
        ('complete', 0), ('open_parts', 1), ('open_waits', 1), ('request', 0),
        ('wall_ns', 1 << 64), ('cpu_ns', 3899), ('pool_calls', 0), ('pool_throws', 1),
        ('pool_wait_throws', 1), ('pool_wall_ns', 3001), ('pool_wait_wall_ns', 3001),
        ('unclassified_wait_wall_ns', 12001), ('pool_cpu_ns', -1))]
    bad += [
        (text.replace(' pool_cpu_ns=300', '', 1), [1, 3]),
        (lines[0] + ' pool_cpu_ns=300\n' + lines[1], [1, 3]),
        (lines[0] + ' unknown_ns=1\n' + lines[1], [1, 3]),
        (text + lines[0] + '\n', [1, 3]),
        (lines[1] + '\n' + lines[0], [1, 3]),
        (lines[0], [1, 3]), (text, []), (text, [True, 3]),
        (text + reader.PREFIX[:-2], [1, 3]),
        ('garbage ' + lines[0] + '\n' + lines[1], [1, 3]),
        (lines[0] + ' ' * 16385 + '\n' + lines[1], [1, 3])]
    for candidate, expected in bad:
        try:
            reader.extract(candidate, expected)
        except RuntimeError:
            continue
        raise RuntimeError('corrupt model ledger accepted')
    row['reader_controls'] = dict(passed=True, actual_CXX_records=2,
        disabled_absence_passed=True, corrupt_cases_rejected=len(bad),
        all_nine_regions=True, compact_nested_wait_suppression=True,
        exact_integer_clock_conservation=True, GPU_executed=False)


def context():
    spec = importlib.util.spec_from_file_location('strata_decode_reader_contract', template.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY = ROOT, KEY
    ctx.SOURCES = ('include/strata/sycl/decode_service_profile.hpp',
        'tests/core/decode_service_evidence_fixture.cpp',
        'tools/sycl/decode_service_evidence.py', 'tools/sycl/decode_service_contract.py',
        'bench/sycl/qualify_decode_service_reader.py', 'tools/sycl/measurement_guards.py',
        'tools/sycl/measurement_owner.py', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md')
    ctx.FIXTURES = (('producer', 'tests/core/decode_service_evidence_fixture.cpp', 2, ''),)
    ctx.check_output = check_output
    return ctx


def prepare(ctx):
    native_path = template.B / 'decode-service-model-v1/record.json'
    native = json.loads(native_path.read_text())
    header = 'include/strata/sycl/decode_service_profile.hpp'
    require(native['diagnostic_native_recipe_checks']['passed'] is True
        and template.sha(P / header) == native['native_sources'][header]
        and template.sha(native['binaries']['strata']['path']) ==
            native['binaries']['strata']['sha256'], 'producer header must match the actual diagnostic engine')
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['native_header_binding'] = dict(native_source_commit=native['source_commit'],
        native_engine_sha256=native['binaries']['strata']['sha256'],
        header_sha256=native['native_sources'][header])
    row['plan'] = ('One owned GCC13/O2 compile and one owned real C++ schema2 producer.'
        'Emit active request1 covering all9regions/nested waits,disabled request2,'
        'then active empty request3. Validate Python reader against the actual wire'
        'records and independent fake-clock arithmetic,then all committed corrupt'
        'record variants;require complete source/compiler/dependency/ELF/owner gates.'
        'Stop first failure,no retry. No GPU/model,metrology/overhead or speed claim.')
    row['retention'] = ('Keep source/recipe and compact reader controls for the first'
        ' disjoint-ledger model admission. Retire the one-use producer ELF,record'
        ' and normal output after committed results and source recovery.')
    ctx.save(row)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'use prepare|run')
    ctx = context()
    with (template.B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        template.guards.no_model_owners([])
        {'prepare': lambda: prepare(ctx), 'run': ctx.run}[sys.argv[1]]()
