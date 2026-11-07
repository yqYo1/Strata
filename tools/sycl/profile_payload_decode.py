#!/usr/bin/env python3
"""Current payload-engine GPU intervals and conserved caller regions.

Main-authored composition of this fork's owned benchmark/collector/readers;
repository LICENSE. No native rebuild, XeStrata input or clean-speed claim.
"""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import payload_gpu_coverage as coverage
import profile_iq3_decode as legacy
import compare_adaptive_payload96_generation as speed
import check_q6_native_half_model as half
import decode_service_evidence as caller

P, B, RESULT, base = coverage.P, coverage.B, coverage.RESULT, coverage.base
ROOT = B / 'payload-decode-device-service-v3'
KEY = 'adaptive_payload96_device_service_v3'
SELF = Path(__file__).resolve()
require, digest = coverage.require, coverage.digest
HEADER = 'include/strata/sycl/decode_service_profile.hpp'


def basis():
    native, prior, providers, assets = coverage.selected_basis()
    data = base.committed(RESULT)
    control = data[coverage.KEY]
    require(control['stage'] == 'closed' and control['passed'] and control['complete'] and
        control['ownership_closed'] and not control['active'] and
        control['native_source_commit'] == native['source_commit'] and
        control['engine'] == native['binaries']['strata'] and
        control['binary'] == native['binaries']['sycl_submission_test'] and
        control['assets'] == assets and len(control['cases']) == 1 and
        control['cases'][0]['native_exit_code'] == 0 and control['cases'][0]['fresh_faults'] == [] and
        control['cases'][0]['trace']['observed_kernels'] == 62 and
        control['cases'][0]['trace']['native_host_flow_joins_unique'] and
        control['cases'][0]['trace']['collection_active_from_exec'],
        'closed exact-current62-kernel collector control required')
    proof = data['decode_service_reader_contract_v1']
    require(proof['stage'] == 'closed' and proof['passed'] and proof['complete'] and
        proof['ownership_closed'] and not proof['active'] and
        proof['native_header_binding']['header_sha256'] == native['native_sources'][HEADER] and
        proof['reader_controls']['corrupt_cases_rejected'] == 28 and
        proof['reader_controls']['exact_integer_clock_conservation'] and
        '-DSTRATA_CPU_SERVICE_PROFILE:BOOL=ON' in native['recipe']['configure'],
        'same compiled conserved caller producer/reader required')
    for name, sha in proof['sources'].items():
        if name not in ('AGENTS.md', 'docs/ARTIFACT_RETENTION.md'):
            require(digest(P / name) == sha, 'caller reader changed: ' + name)
    selected = speed.context('regular-code')
    cohort = json.loads((selected.ROOT / 'record.json').read_text())
    require(speed.clean.compact(cohort) == data['adaptive_payload96_generation_regular_code_v1'],
        'original closed cohort inputs must match all committed signatures')
    require(cohort['stage'] == 'closed' and cohort['passed'] and cohort['complete'] and
        cohort['ownership_closed'] and not cohort['active'] and
        cohort['binary'] == native['binaries']['strata'] and
        cohort['decision']['repeated_TG_gain'] and len(cohort['processes']) == 6,
        'closed current32/96code cohort required')
    return native, prior, providers, assets, cohort


def proof():
    native, prior, providers, _, _ = basis()
    probe = prior['native_validation_probe']
    gate = dict(source_commit=prior['source_commit'], environment=dict(prior['environment'],
        LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path']))
    return dict(native, environment=native['recipe']['environment'], boot=prior['boot'],
                driver=prior['driver']), gate, providers


def clean_inputs(cohort):
    ctx = speed.context('regular-code')
    plan = cohort['plans'][1]
    require(plan['kernel'] == 'candidate' and plan['adapt_swaps'] == 96,
        'exact closed96process configuration required')
    config, env = ctx.configuration(cohort, plan), ctx.environment(cohort, plan)
    old = env['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(ROOT)) for k, v in env.items()}
    config['log'] = str(ROOT / 'engine.stderr')
    env.update(STRATA_DECODE_TIMING='1', STRATA_SYCL_DECODE_SERVICE_PROFILE='1',
        STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0',
        UNITRACE_ChromeKernelLogging='1', UNITRACE_ChromeEventBufferSize='128')
    return config, env


def context():
    spec = importlib.util.spec_from_file_location('strata_payload_device_profile', legacy.__file__)
    run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
    run.ROOT, run.KEY, run.SOURCE = ROOT, KEY, SELF
    spec = importlib.util.spec_from_file_location('strata_payload_model_entry', legacy.entry.template.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.configure('adaptive-throughput')
    ctx.ROOT, ctx.KEY, ctx.RUNS, ctx.proof = ROOT, KEY, 1, proof
    inherited_pins, inherited_current, inherited_route, inherited_save = (
        ctx.pinned_sources, ctx.current, ctx.profile_route, ctx.save)

    def pins(revision):
        result = inherited_pins(revision)
        result.update(base.controls(revision, controller=SELF))
        for name in ('AGENTS.md', 'docs/ARTIFACT_RETENTION.md', 'tests/core/test_decode_window_trace.py'):
            raw = base.basis.build.git(P, 'show', revision + ':' + name)
            require(raw == (P / name).read_bytes(), 'commit profile input: ' + name)
            result[name] = digest(P / name)
        return result

    def current(row):
        _, prior, providers, assets, cohort = basis()
        config, env = clean_inputs(cohort)
        require(row['configuration'] == config and row['environment'] == env and
            row['read_tokens'] == 32769 and row['rows'] == 32768 and row['runs'] == 1 and
            row['native_validation_probe'] == prior['native_validation_probe'] and
            row['native_profile']['mode'] == 'aggregate' and
            all(row['native_profile'][k] == v for k, v in assets.items()) and
            row['native_profile']['environment']['UNITRACE_TraceOutputDir'] == str(ROOT / 'trace') and
            row['budgets'] == dict(wall=1200, cpu=7200, rss_cap=104 << 30,
                text_cap=64 << 20, file_cap=512 << 20, total_cap=768 << 20) and
            row['whole_bytes'] == 1024 << 20 and row['maximum_observer_ratio'] == 1.5,
            'frozen current native workload/observer/budget differs')
        actual, manifest = inherited_current(row)
        require(actual == providers, 'current native providers differ')
        return actual, manifest

    def route(text):
        result = inherited_route(text)
        full = (ROOT / 'engine.stderr').read_text()
        result['IQ3_dispatch'] = legacy.dispatch(text, full)
        result['Q6_snapshots'] = half.snapshots(text, 1, 2)
        result['caller'] = caller.compact(caller.extract(full, [1, 2])[-1])
        return result

    def save(row):
        if 'outcome' in row:
            out = row['outcome']
            if 'observer_cost_qualified' in out:
                out['prompt_observer_cost_screen_passed'] = out.pop('observer_cost_qualified')
            out.update(decode_observer_cost_qualified=False, complete_device_coverage_qualified=False,
                physical_traffic_qualified=False, critical_path_qualified=False)
        inherited_save(row)

    ctx.pinned_sources, ctx.current, ctx.profile_route, ctx.save = pins, current, route, save
    return ctx, run


def prepare(ctx):
    _, prior, _, assets, cohort = basis()
    config, env = clean_inputs(cohort)
    old = base.committed(RESULT)['decode_iq3_device_service_v1']
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['configuration'], row['environment'] = config, env
    (ROOT / 'trace').mkdir()
    row['native_profile'] = dict(legacy.profile.profiling(ROOT, assets), mode='aggregate')
    base.write(ROOT / 'config.json', row['configuration'])
    base.write(ROOT / 'plan.json', dict(binary=row['binary'],
        native_validation_probe=row['native_validation_probe'], native_profile=row['native_profile']))
    row['generated'].update({name: digest(ROOT / name) for name in ('config.json', 'plan.json')})
    row.update(request_sha256=old['request_sha256'],
        observer_reference_prompt_ms=cohort['processes'][1]['samples'][0]['prompt_ms'],
        maximum_observer_ratio=1.5, whole_bytes=1024 << 20,
        budgets=dict(wall=1200, cpu=7200, rss_cap=104 << 30,
            text_cap=64 << 20, file_cap=512 << 20, total_cap=768 << 20))
    row['profile_plan'] = dict(
        question='Current e1b011 observed GPU intervals versus exactly conserved main-thread caller/waits;which concrete source path merits the next optimization?',
        order='Commit this source and plan;two current frontend probes and existing guarded-window tests;commit host results;one active-from-exec benchmark process,one short warmup and one thinking-on32769-input/1..24-output diagnostic. No pause,retry,environment mutation or native rebuild.',
        constants='Exact bbd345/e1b011 and six ELFs;closed regular-code96configuration and immutable environment,131072resident INT8,1256slots,5workers,20ms spin,8192chunk,spec4/maxT4,native-half1,throughput BF16/GDN16/MTP16. Only caller/decode clocks and active collector are added;leaf/census remain off.',
        acceptance='Committed native/controller/generated/runtime/model identities;both frontend probes and window controls;request/body/count/reuse/route/IQ3/Q6/native0/journal/validation/driver/fault/owner gates. All observed GPU IDs must join unique native host flows with finite contained RAW clocks;do not infer unobserved work is idle. Caller schema2 must exactly conserve regions and nested waits for requests1/2,with no open scopes or exceptions.',
        observer='Prompt1.5screen only versus the closed96long-code first-use prompt;different body/count makes this an admission sanity check,not a matched observer estimate. Decode observer overhead requires a separate matched on/off comparison and remains unqualified. No clean PP/TG from this process.',
        clocks='Existing four RAW/MONOTONIC pairs,<=1ms brackets,<=50ms offset envelope and BOOTTIME continuity. Interior client decode uses first-output+maxOffset+100ms to end+minOffset-100ms,at least100ms. Sampled clock envelopes do not prove unsampled error bounds.',
        limits='Short diagnostic,not sustained generation,MTP-friendly/weak classification,full device coverage,critical path or compute/RAM/VRAM/PCIe saturation,task/agent,physical262144/RESTORE,ordinary provider teardown or adoption. Native caller CPU includes polling during waits;worker CPU is separate. Never add overlapping services.',
        stopping='First identity,body,count,route,clock,validation,fault,owner,output or prompt-admission failure ends the attempt;preserve its status. No pin refresh or in-place retry.',
        retention='512MiB/file,768MiBjob,1GiBwhole and1200s/104GiBRSS. Full trace temporarily needed for graph IDs and post-flush guarded reduction. Keep compact source/count/interval/caller/request/fault evidence,retire successful raw output after verification.')
    row['criteria'] = row['profile_plan']['acceptance']
    row['retention'] = row['profile_plan']['retention']
    ctx.save(row)
    print('Prepared current payload GPU/caller diagnostic;commit before host checks.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['host'], ['run']), 'use prepare|host|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([])
        ctx, run = context()
        {'prepare': lambda: prepare(ctx), 'host': lambda: run.host(ctx), 'run': lambda: run.run(ctx)}[sys.argv[1]]()
