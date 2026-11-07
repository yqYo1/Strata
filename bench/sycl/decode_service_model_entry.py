#!/usr/bin/env python3
"""Join the schema2 caller ledger to the real model's leaf service and census.

Main-authored; repository LICENSE. Reuse the owned bench.sh supervisor in a
private context, with this diagnostic generation's actual host/device gates.
This short entry checks records and lifecycle, not observer overhead or speed.
"""
import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'tools/sycl'))
import decode_service_checks as checks
import cpu_service_model_entry as leaf_template
import decode_service_evidence as reader
import verify_window_model_entry as entry
from iq3_model_entry import dispatches
from q6_centering_model_entry import captures as centering_captures

B, RESULT, base = checks.B, checks.RESULT, checks.base
ROOT, KEY = B / 'decode-service-model-entry-v1', 'decode_service_model_entry_v1'
SELF = Path(__file__).resolve()
require, digest = checks.require, checks.model.digest


def reader_proof(native):
    row = base.committed(RESULT)['decode_service_reader_contract_v1']
    controls = row['reader_controls']
    require(row['passed'] is True and row['complete'] is True
        and row['ownership_closed'] is True and row['active'] is False
        and controls['passed'] is True and controls['actual_CXX_records'] == 2
        and controls['corrupt_cases_rejected'] == 28
        and all(controls[k] is True for k in ('disabled_absence_passed',
            'all_nine_regions', 'compact_nested_wait_suppression',
            'exact_integer_clock_conservation'))
        and all(digest(P / name) == sha for name, sha in row['sources'].items())
        and row['native_header_binding'] == dict(native_source_commit=native['source_commit'],
            native_engine_sha256=native['binaries']['strata']['sha256'],
            header_sha256=native['native_sources']['include/strata/sycl/decode_service_profile.hpp']),
        'unchanged closed real-producer reader and exact diagnostic header required')
    leaf_template.reader_proof()
    return row


def proof():
    gpu = checks.device_context()
    row = json.loads((checks.DEVICE_ROOT / 'record.json').read_text())
    gpu.unchanged(row)
    native = gpu.built()
    require(row['stage'] == 'closed' and row['passed'] is True and row['complete'] is True
        and row['ownership_closed'] is True and row['active'] is False
        and row['GPU_executed'] is True and row['fresh_faults'] == []
        and row['boot_unchanged'] is True
        and [(r['name'], r['native_exit'], r['checks']) for r in row['results']] ==
            [('submission', 0, 10), ('phase-cache', 0, 16)],
        'fresh diagnostic same-build host and small-device qualification required')
    reader_proof(native)
    files = json.loads(Path(row['runtime_manifest']['path']).read_text())
    prior = base.committed(RESULT)['cache_map_model_entry_zero_v1']
    require(prior['passed'] and prior['complete'] and prior['ownership_closed']
        and not prior['active'] and prior['binary']['sha256'] ==
            '552b1616d21ad1b9aa6a317cb940d375c003246b896a255a935e09292804c96a',
        'retain the validated prefill fallback and model environment')
    driver = prior['environment']['ZE_ENABLE_ALT_DRIVERS']
    require(files.get(driver) == digest(driver), 'model ALT driver not in qualified inventory')
    native = dict(native, environment=native['recipe']['environment'],
        boot=row['boot'], driver=row['driver'])
    return native, dict(source_commit=row['source_commit'],
        environment=dict(row['environment'], ZE_ENABLE_ALT_DRIVERS=driver)), files


def pinned_sources(revision):
    pins = entry.pinned_sources(revision)
    pins.update(checks.controls(revision))
    for path in (SELF, Path(leaf_template.__file__), Path(reader.__file__),
            P / 'tools/sycl/q6_centering_model_entry.py'):
        pins.update(base.controls(revision, controller=path))
    return pins


def contract(row):
    formats, membership = leaf_template.contract(row)
    env = row['environment']
    require(env['STRATA_SYCL_DECODE_SERVICE_PROFILE'] == '1'
        and env['STRATA_SYCL_DECODE_Q6_CENTERING'] == '0'
        and env['STRATA_IQ3_PAIR'] == env['STRATA_IQ3_PAIR_AUDIT'] == '1'
        and env['STRATA_IQ_SINGLE_EXACT'] == '0',
        'freeze schema2 observer,centering0 and actual IQ3 selector/audit before exec')
    return formats, membership


def context():
    ctx = entry.context('columns')
    ctx.ROOT, ctx.KEY, ctx.proof, ctx.pinned_sources = ROOT, KEY, proof, pinned_sources
    old_current, old_route = ctx.current, ctx.profile_route

    def current(row):
        contract(row)
        return old_current(row)

    def route(text):
        result = old_route(text)
        result['centering_captures'] = centering_captures(text, 0)
        selected = dispatches(text, 1, positive=True)
        full = dispatches((ROOT / 'engine.stderr').read_text(), 1, positive=False)
        require(len(selected) == 1 and [r['request'] for r in full] == [1, 2, 3]
            and selected[0]['request'] in (2, 3), 'join actual IQ3 captures to all requests')
        result['IQ3_dispatch'] = selected[0]
        return result

    ctx.current, ctx.profile_route = current, route
    return ctx


def prepare(ctx):
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['environment'].update(STRATA_SYCL_DECODE_SERVICE_PROFILE='1',
        STRATA_SYCL_DECODE_Q6_CENTERING='0', STRATA_SYCL_DECODE_Q6_COLUMNS='1',
        STRATA_SYCL_ADAPT_SYNC='1', STRATA_IQ3_PAIR='1', STRATA_IQ3_PAIR_AUDIT='1',
        STRATA_IQ_SINGLE_EXACT='0', STRATA_CPU_SERVICE='1',
        STRATA_CPU_WORK_CENSUS='1', STRATA_DECODE_TIMING='1')
    contract(row)
    row['qualification_basis'] = dict(build_key=checks.model.KEY,
        host_key=checks.HOST_KEY, device_key=checks.DEVICE_KEY,
        caller_reader_key='decode_service_reader_contract_v1',
        leaf_reader_key='cache_map_cpu_service_sources_v1.reader_v2_host',
        reuse='Exact566 projected native inputs,five hashed diagnostic outputs,six CPU children,submission10/phase16 and closed C++ producer/28 corrupt controls. No inherited model qualification.')
    row['service_plan'] = dict(
        question='Can this real diagnostic engine conserve complete caller and worker records joined to the same native/API requests?',
        order='One thinking-on short warmup,then two fresh32769-input/1..24-output requests through unchanged maintained bench.sh. Once only,no retry.',
        observers='SERVICE_PROFILE1,CPU_SERVICE1,CPU_WORK_CENSUS1,DECODE_TIMING1,IQ3PAIR1/AUDIT1,centering0. Immutable before exec;parameter validation and progress,without full API trace.',
        acceptance='All source/build/ELF/provider,15asset,frontend/body/token/route/cache/Q6/IQ3/clock/validation/fault/owner gates. Exact request IDs1/2/3,complete leaf+census joins,all schema2 fields,closed clean scopes,integer caller conservation and contained nested waits. Caller scope includes final commit after the reported decode interval;allow0.1ms footer rounding only when checking its lower bound.',
        stopping='Stop on first source,frontend,record,budget,native,owner or fresh driver fault failure. Preserve failure;no hash refresh or automatic continuation.',
        interpretation='Short request accounting/lifecycle only. Native waits are nested inside caller regions;do not add them. Leaf thread CPU may overlap caller CPU and is not whole-process CPU. Short NT mix is not representative. Clock-read cost and same-binary profile0/1 observer overhead must be qualified before long attribution. No clean speed,physical RAM/PCIe/VRAM traffic,saturation,70TG,agent/task,physical262144/RESTORE or adoption claim.',
        next='After closure,qualify clock resolution/read cost and a same-binary repeated profile0/1 long-generation comparison with CPU_SERVICE1/CENSUS1 in both. Preserve thinking-on observed MTP-strong/weak workloads;collect clean speed separately.')
    row['criteria'] += ' Require all three native/API/leaf/census/schema2 joins and the service_plan acceptance.'
    row['retention'] = ('Keep compact per-request identities,caller regions/nested waits,leaf operation totals,conservation and source/fault/owner evidence. Retire successful journals/logs/JIT after result commit and source recovery. Keep the generated configuration and runtime manifest only through the imminent same-binary observer-overhead decision. No new native generation.')
    ctx.save(row)
    print('Prepared diagnostic32K model/ledger admission;commit before frontend probes.', flush=True)


def evidence(row):
    leaf = checks.load('strata_disjoint_leaf_evidence', leaf_template.__file__)
    leaf.ROOT, leaf.contract = ROOT, contract
    result = leaf.evidence(row)
    requests = result['requests']
    ids = [r['work']['request'] for r in requests]
    require(ids == [1, 2, 3], 'exact warmup and two fresh native/API/leaf request joins required')
    records = reader.extract((ROOT / 'engine.stderr').read_text(), ids)
    for request, record in zip(requests, records):
        decode_ns = request['sample']['decode_ms'] * 1000000
        require(record['wall_ns'] + 100000 >= decode_ns,
            'caller interval must enclose the reported decode interval and final commit')
        request['caller'] = reader.compact(record)
        request['caller_scope_minus_reported_decode_ns'] = record['wall_ns'] - decode_ns
    result.update(schema2_request_joins=True, clock_metrology_qualified=False,
        caller_and_leaf_cpu_additive=False, clean_speed_qualified=False)
    return result


def run(ctx):
    ctx.run()
    row = json.loads((ROOT / 'record.json').read_text())
    row['model_route_check_passed'] = row['passed']
    try:
        require(row['passed'] and row['complete'] and row['ownership_closed'] and not row['active'],
            'closed actual model route required before ledger admission')
        row['decode_service'] = evidence(row)
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, decode_service_error=repr(error))
    ctx.save(row)
    require(row['passed'], row.get('decode_service_error', 'disjoint model-ledger entry failed'))
    print('Three real-model requests joined conserved caller/leaf/census records;overhead and long attribution remain pending.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host', 'run'), 'use prepare|host|run')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        ctx = context()
        {'prepare': lambda: prepare(ctx), 'host': ctx.host, 'run': lambda: run(ctx)}[sys.argv[1]]()
