#!/usr/bin/env python3
"""Bind the observer generation to actual short model requests.

Main-authored; repository LICENSE. Reuse this fork's owned bench.sh/model
supervisor in a private context. The small device gate and real-pool parity
admit this entry, not model attribution. Require the independently qualified v2
reader and same-request job census before collecting representative service or
comparing observer overhead. No new native build or clean speed claim.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import cpu_service_device as device
import cpu_service_evidence as service
import cpu_work_census as census
import generation_engine as gen
import verify_window_model_entry as entry

build, base = device.build, device.build.base
P, B, RESULT = build.P, build.B, build.RESULT
ROOT = build.ROOT / 'cpu-service-model-entry-v1'
KEY = 'verify_window_cpu_service_model_entry_v1'
SELF = Path(__file__).resolve()
require, digest = build.require, build.digest
TESTS = ('tests/core/test_cpu_service_evidence.py', 'tests/core/test_cpu_work_census.py')


def reader_proof():
    row = base.committed(RESULT)['cache_map_cpu_service_sources_v1']['reader_v2_host']
    require(row['passed'] and row['complete'] and row['ownership_closed']
        and not row['active'] and row['result']['tests'] == 12
        and row['result']['failed'] == row['result']['skipped'] == 0
        and row['result']['service_prefix_truncations_refused_with_valid_census'] == 15
        and all(digest(P / name) == sha for name, sha in row['source_inputs'].items()),
        'reuse only the unchanged closed twelve-case v2 reader qualification')
    return row


def proof():
    gpu = device.context()
    row = json.loads((device.ROOT / 'record.json').read_text())
    gpu.unchanged(row)
    native = gpu.built()
    require(row['stage'] == 'closed' and row['passed'] and row['complete']
        and row['ownership_closed'] and not row['active'] and row['GPU_executed']
        and row['fresh_faults'] == [] and row['boot_unchanged']
        and [(r['name'], r['native_exit'], r['checks']) for r in row['results']] ==
            [('submission', 0, 10), ('phase-cache', 0, 16)],
        'closed same-build observer submission/phase qualification required')
    reader_proof()
    files = json.loads(Path(row['runtime_manifest']['path']).read_text())
    prior = base.committed(RESULT)['cache_map_model_entry_zero_v1']
    require(prior['passed'] and prior['complete'] and prior['ownership_closed']
        and not prior['active'] and prior['binary']['sha256'] ==
            '552b1616d21ad1b9aa6a317cb940d375c003246b896a255a935e09292804c96a',
        'retain the qualified fallback model environment')
    driver = prior['environment']['ZE_ENABLE_ALT_DRIVERS']
    require(files.get(driver) == digest(driver),
        'model ALT driver must be in the current GPU-qualified provider inventory')
    native = dict(native, environment=native['recipe']['environment'],
        boot=row['boot'], driver=row['driver'])
    gate = dict(source_commit=row['source_commit'],
        environment=dict(row['environment'], ZE_ENABLE_ALT_DRIVERS=driver))
    return native, gate, files


def pinned_sources(revision):
    pins = entry.pinned_sources(revision)
    pins.update(base.controls(revision, controller=SELF))
    for name in TESTS:
        raw = gen.git('show', revision + ':' + name)
        require((P / name).read_bytes() == raw, 'commit exact qualified reader controls')
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def contract(row):
    args, env = row['configuration']['args'], row['environment']
    require(args[args.index('--adapt-swaps') + 1] == '0'
        and args[args.index('--spec') + 1] == '4'
        and args[args.index('--expert-cache') + 1] == '1256'
        and '--no-split-rows' not in args
        and env['STRATA_CPU_SERVICE'] == env['STRATA_CPU_WORK_CENSUS'] == env['STRATA_DECODE_TIMING'] == '1'
        and env['STRATA_SYCL_ADAPT_SYNC'] == env['STRATA_SYCL_DECODE_Q6_COLUMNS'] == '1'
        and row['read_tokens'] == 32769 and row['runs'] == 2,
        'freeze observer1/static-cache/split/spec4/columns and short declared extent')
    formats = census.manifest_formats(Path(args[args.index('--pack') + 1]) / 'native_experts.txt')
    membership = census.expected_membership(args[args.index('--expert-profile') + 1])
    require(len(formats) == len(membership) == 48, 'only the audited48-layer512-expert model')
    return formats, membership


def context():
    ctx = entry.context('columns')
    ctx.ROOT, ctx.KEY, ctx.proof, ctx.pinned_sources = ROOT, KEY, proof, pinned_sources
    original_current = ctx.current

    def current(row):
        contract(row)
        return original_current(row)

    ctx.current = current
    return ctx


def prepare(ctx):
    reader = reader_proof()
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    try:
        row['environment'].update(STRATA_CPU_SERVICE='1', STRATA_CPU_WORK_CENSUS='1',
            STRATA_DECODE_TIMING='1', STRATA_SYCL_ADAPT_SYNC='1', STRATA_SYCL_DECODE_Q6_COLUMNS='1')
        contract(row)
        row['qualification_basis'] = dict(build_result_key=build.KEY,
            device_result_key=device.KEY, reader_source_commit=reader['source_commit'],
            reader_tests=12, reader_prefix_truncations=15,
            reuse='Same563native inputs,four hashed outputs,real-pool28-case off/on checksum parity,small submission10/phase16 lifecycle. Unchanged twelve-case reader and independent census controls;no duplicate reader test run.',
            limits='Observer-only generation;no inherited model qualification,measured CPU service,overhead,speed or ordinary-provider-teardown claim.')
        row['service_plan'] = dict(
            question='Can the actual observer generation emit complete v2 leaf CPU/wall records conserved against all three native/API request censuses?',
            order='One existing short thinking-on warmup,then two fresh32769-input/1..24-output requests through unchanged bench.sh. Once only;no retries or automatic long-generation continuation.',
            observers='CPU_SERVICE1,CPU_WORK_CENSUS1,DECODE_TIMING1,STRATA_TRACE1,parameter validation and qualified retained-reference preload. Immutable before native exec;no full API trace.',
            acceptance='Maintained frontend/source/build/ELF/provider/route/body/token/clock/validation/fault/owner gates. Exact three native/API joins including warmup,normal0,complete v2 headers/ordered bins/thread slots/trailers,no write/flush failure,job/row/entry/format/NT/clock conservation. Final hashes of all15model assets.',
            interpretation='Leaf wall and thread-CPU sums include observer cost and memory stalls;concurrent sums are not an additive critical path. Short NT mix is not representative long decode. No hardware ceiling,physical RAM/PCIe traffic,saturation,clean model speed,task/agent/full262144/RESTORE or adoption qualification.',
            next='Only after closure,freeze a same-binary observer0/1 repeated long-generation comparison with useful leaf summaries and a declared overhead bound. Keep both official thinking-on strong/weak-MTP workloads;do not use diagnostic TG as clean speed.')
        row['criteria'] += ' Require the prospective service_plan acceptance and all three same-request independent censuses.'
        row['retention'] = ('Keep three compact API identities,per-operation CPU/wall totals,NT/job counts,completion/conservation and source/fault/owner evidence. Retire successful logs/journals/JIT after verified result commit;protect the frozen generated config/plan/runtime record for the imminent observer-overhead comparison. No new generation.')
    except BaseException as error:
        row.update(stage='failed-preparation', passed=False, complete=False, error=repr(error))
        ctx.save(row)
        raise
    ctx.save(row)
    print('Prepared observer short-model/census entry;commit before frontend probes.', flush=True)


def evidence(row):
    formats, membership = contract(row)
    journals = list((ROOT / 'bench').glob('*/requests.jsonl'))
    require(len(journals) == 1, 'one exact model request journal required')
    events = [json.loads(line) for line in journals[0].read_text().splitlines()]
    results = [e for e in events if e['kind'] == 'result']
    require([(e['phase'], e['run']) for e in results] ==
        [('warm-up/chat', None), ('prefill-only', 0), ('prefill-only', 1)],
        'warmup and both long-input requests must bind')
    samples = [e['sample'] for e in results]
    text = (ROOT / 'engine.stderr').read_text()
    timing = gen.decode_timing_evidence(text, samples)
    bindings = [dict(prompt_tokens=s['prompt_tokens'], windows=t['windows'],
        rows=t['work']['target_rows'], cpu_jobs=t['work']['cpu_unique_jobs'],
        cpu_entries=t['work']['cpu_entries'],
        pool_ms=t['ms_per_window']['CPU_pool_run'] * t['windows'])
        for s, t in zip(samples, timing)]
    work = census.extract(text, bindings, formats, membership)
    leaves = service.extract(text, bindings, formats, membership)
    require(len(work) == len(leaves) == len(samples) == 3, 'three complete native/API/service joins required')
    fields = ('prompt_tokens', 'reused_tokens', 'read_tokens', 'generated_tokens',
        'decode_ms', 'drafts_offered', 'drafts_accepted', 'request_sha256')
    retained = ('request', 'windows', 'rows', 'cpu_jobs', 'cpu_entries', 'vram_entries',
        'jobs_NT', 'NT1_fraction', 'pool_ms', 'pool_clock_rounding_bound_ms',
        'membership_sha256', 'logical_job_blob_bytes')
    requests = []
    for e, s, w, leaf, clock in zip(results, samples, work, leaves, timing):
        operations = {op: {k: sum(v[k] for v in leaf['format_bins'] if v['op'] == op)
            for k in ('calls', 'rows', 'token_rows', 'wall_ns', 'cpu_ns')} for op in service.OPS}
        requests.append(dict(phase=e['phase'], run=e['run'], sample={k: s[k] for k in fields},
            work={k: w[k] for k in retained}, clock_binding=clock['clock_binding'],
            service={k: leaf[k] for k in ('request', 'record_version', 'completion_checked',
                'same_request_census', 'thread_count', 'calls', 'wall_ns', 'cpu_ns')},
            operations=operations, service_bins_verified=len(leaf['format_bins']),
            census_bins_verified=len(w['format_bins'])))
    return dict(passed=True, completed_requests=3, requests=requests,
        long_generation_mix_qualified=False, observer_overhead_qualified=False,
        physical_traffic_qualified=False, hardware_saturation_qualified=False,
        clean_speed_qualified=False, inference_adopted=False)


def run(ctx):
    row = json.loads((ROOT / 'record.json').read_text())
    contract(row)
    reader_proof()
    ctx.run()
    row = json.loads((ROOT / 'record.json').read_text())
    row['model_route_check_passed'] = row['passed']
    try:
        require(row['passed'] and row['complete'] and row['ownership_closed'] and not row['active'],
            'closed actual-model route required before CPU-service admission')
        row['cpu_service'] = evidence(row)
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, cpu_service_error=repr(error))
    ctx.save(row)
    require(row['passed'], row.get('cpu_service_error', 'CPU-service model entry failed'))
    print('Three actual model requests conserve complete v2 CPU service and independent census;overhead and long mix remain unqualified.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host', 'run'), 'use prepare|host|run')
    ctx = context()
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        {'prepare': lambda: prepare(ctx), 'host': ctx.host, 'run': lambda: run(ctx)}[sys.argv[1]]()
