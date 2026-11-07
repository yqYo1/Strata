#!/usr/bin/env python3
"""Qualify the selected engine's existing decode-job census; repository LICENSE.

No native generation or arithmetic change. Reuse the exact selected engine and
its checked request controller in a private context. First collect one ordinary
warmup and two 32K/24-output requests; only a closed, conserved result can admit
a later, separately committed long-generation census. These are diagnostic
counts and nested host wall clocks, never clean speed or physical bus traffic.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import cache_map_model_entry as entry
import cpu_work_census as census
import generation_engine as gen

base, P, B, RESULT = entry.base, entry.P, entry.B, entry.RESULT
require, digest = base.require, base.digest
ROOT = entry.model.ROOT / 'cpu-census-entry-v1'
KEY = 'cache_map_cpu_census_entry_v1'
SOURCE = Path(__file__).resolve()
NATIVE = ('include/strata/core/cpu_work_census.hpp',
          'src/core/expert_source.cpp', 'src/program/generate.cpp')


def pinned_sources(revision):
    pins = entry.pinned_sources(revision)
    pins.update(base.controls(revision, controller=SOURCE))
    name = 'tests/core/test_cpu_work_census.py'
    raw = gen.git('show', revision + ':' + name)
    require((P / name).read_bytes() == raw, 'commit exact census reader tests')
    pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def context():
    ctx = entry.context('zero')
    ctx.ROOT, ctx.KEY, ctx.pinned_sources = ROOT, KEY, pinned_sources
    return ctx


def contract(row):
    args, env = row['configuration']['args'], row['environment']
    require(args[args.index('--adapt-swaps') + 1] == '0'
            and args[args.index('--spec') + 1] == '4'
            and args[args.index('--expert-cache') + 1] == '1256'
            and '--no-split-rows' not in args
            and env['STRATA_SYCL_ADAPT_SYNC'] == '1'
            and env['STRATA_CPU_WORK_CENSUS'] == env['STRATA_DECODE_TIMING'] == '1'
            and row['read_tokens'] == 32769 and row['runs'] == 2,
            'census requires immutable static cache, spec4, split native rows and declared extent')
    formats = census.manifest_formats(Path(args[args.index('--pack') + 1]) / 'native_experts.txt')
    membership = census.expected_membership(args[args.index('--expert-profile') + 1])
    require(len(formats) == len(membership) == 48, 'only the audited 48-layer/512-expert model')
    return formats, membership


def prepare(ctx):
    native, _, _ = entry.proof()
    entry.prepare(ctx, 'zero')
    row = json.loads((ROOT / 'record.json').read_text())
    row['environment'].update(STRATA_CPU_WORK_CENSUS='1', STRATA_DECODE_TIMING='1')
    require(native['source_commit'] == '479a6fdac8cbaa35a9bf3a41d2b6e23be5260e54'
            and native['binaries']['strata']['sha256'] ==
                '552b1616d21ad1b9aa6a317cb940d375c003246b896a255a935e09292804c96a',
            'use the current selected generation, without a rebuild')
    row['census_plan'] = dict(
        native_sources={n: native['native_sources'][n] for n in NATIVE},
        question='Can the unchanged selected engine report final decode CPU jobs by format and actual NT, conserving native/API clocks and exact restored static cache membership?',
        source_audit='The request owner creates the census after prefill/refill, records completed synchronous native pool jobs on the same host thread, and detaches it on every exit. No worker or GPU command owns the object. Bounds are48layers/512experts/NT<=8/128entries;this model is48/512/spec4/top10. adapt0 leaves usage empty;no peer or split-layer route. Census pool_ns and ms_run cover the identical c3..c4 host interval;both exclude the later census callback. Existing copy footer uses the same post-prefill ds0 deltas.',
        order='Existing one short thinking-on warmup, then two exact32769-input/1..24-output thinking-on requests through unchanged bench.sh. No automatic retry or long-generation continuation.',
        acceptance='Existing source/build/fixture/provider/owner/route/count/body/journal/clock/validation/fault gates plus one valid census per completed request. Independently conserve every layer membership,format,NT/job/entry/window count and rounded pool clock. Three complete API samples must bind to three native request footers,including the warmup.',
        limitations='Qualification of existing counters and short request lifecycle only. No representative long-generation NT mix,clean speed,observer overhead,physical RAM/PCIe traffic,saturation,task/agent,full262144/RESTORE or inference adoption.',
        next='If all gates pass,commit a fresh static-cache long-generation census plan using the same engine and both official thinking-on workloads with>=1K output each. Use its actual NT/format mix to select a CPU leaf experiment;new CPU leaf timing remains separate.')
    contract(row)
    row['criteria'] += ' Also require the prospective census_plan acceptance;model-route success alone does not qualify census evidence.'
    row['retention'] = ('Keep compact API identities,per-format job-NT counts,pool clocks,conservation outcomes and source/fault/owner proof. Successful full journals/logs/JIT retire after binding is verified. No new engine generation. Review this short counter qualification when the separately planned long census closes.')
    ctx.save(row)
    print('Prepared unchanged-engine CPU census entry;commit before host/model execution.', flush=True)


def evidence(row):
    formats, membership = contract(row)
    outputs = list((ROOT / 'bench').glob('*/requests.jsonl'))
    require(len(outputs) == 1, 'one exact request journal required')
    events = [json.loads(line) for line in outputs[0].read_text().splitlines()]
    results = [e for e in events if e['kind'] == 'result']
    require([(e['phase'], e['run']) for e in results] ==
            [('warm-up/chat', None), ('prefill-only', 0), ('prefill-only', 1)],
            'census needs the warmup and both official long-input requests')
    samples = [e['sample'] for e in results]
    text = (ROOT / 'engine.stderr').read_text()
    timing = gen.decode_timing_evidence(text, samples)
    bindings = [dict(prompt_tokens=s['prompt_tokens'], windows=t['windows'],
        rows=t['work']['target_rows'], cpu_jobs=t['work']['cpu_unique_jobs'],
        cpu_entries=t['work']['cpu_entries'],
        pool_ms=t['ms_per_window']['CPU_pool_run'] * t['windows'])
        for s, t in zip(samples, timing)]
    work = census.extract(text, bindings, formats, membership)
    require(len(work) == len(samples) == 3, 'all native requests must bind')
    fields = ('prompt_tokens', 'reused_tokens', 'read_tokens', 'generated_tokens',
              'decode_ms', 'drafts_offered', 'drafts_accepted', 'request_sha256')
    retained = ('request', 'windows', 'rows', 'cpu_jobs', 'cpu_entries', 'vram_entries',
                'jobs_NT', 'NT1_fraction', 'pool_ms', 'pool_clock_rounding_bound_ms',
                'membership_sha256', 'logical_job_blob_bytes')
    return dict(passed=True, requests=[dict(phase=e['phase'], run=e['run'],
        sample={k: s[k] for k in fields}, work={k: w[k] for k in retained},
        format_bins_verified=len(w['format_bins']), clock_binding=t['clock_binding'])
        for e, s, w, t in zip(results, samples, work, timing)],
        clean_speed_qualified=False, physical_traffic_qualified=False,
        hardware_saturation_qualified=False, long_generation_mix_qualified=False,
        inference_adopted=False)


def host(ctx):
    ctx.host()
    row = json.loads((ROOT / 'record.json').read_text())
    owner = base.supervision.Owner(ROOT / 'reader-tests')
    command = [row['python_selected']['path'], str(P / 'tests/core/test_cpu_work_census.py')]
    outcome, out, err = owner.run('reader', command, row['environment'], P,
        wall=30, cpu=20, rss_cap=512 << 20, text_cap=64 << 10,
        file_cap=2 << 20, total_cap=8 << 20)
    good = (base.supervision.closed(outcome) and outcome['exit_code'] == 0
            and out.stat().st_size == 0 and 'Ran 6 tests' in err.read_text())
    row['census_reader_tests'] = dict(passed=good, cases=6,
        native_exit_code=outcome.get('exit_code'), ownership_closed=base.supervision.closed(outcome))
    if not good:
        row.update(stage='failed', host_passed=False,
                   census_error='existing six-case census reader controls failed')
    ctx.save(row)
    require(good, row.get('census_error', 'reader test failure'))
    print('Six existing census reader controls passed;commit before model execution.', flush=True)


def run(ctx):
    row = json.loads((ROOT / 'record.json').read_text())
    contract(row)
    require(row['census_reader_tests']['passed'], 'closed census reader controls required')
    ctx.run()
    row = json.loads((ROOT / 'record.json').read_text())
    row['model_route_check_passed'] = row['passed']
    try:
        require(row['passed'] and row['complete'] and row['ownership_closed']
                and not row['active'], 'closed model route required')
        row['census'] = evidence(row)
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False,
                   census_error=repr(error))
    ctx.save(row)
    require(row['passed'], row.get('census_error', 'census admission failed'))
    print('Three short decode requests conserve exact CPU format/NT/jobs and pool clocks;long-generation mix remains unmeasured.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host', 'run'),
            'use prepare|host|run')
    ctx = context()
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        {'prepare': lambda: prepare(ctx), 'host': lambda: host(ctx),
         'run': lambda: run(ctx)}[sys.argv[1]]()
