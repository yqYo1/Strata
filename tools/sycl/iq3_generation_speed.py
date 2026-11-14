#!/usr/bin/env python3
"""Repeated same-binary IQ3 decode comparison through owned bench.sh requests.

Repository LICENSE. Reuse the maintained v5 workload and finite supervisor.
Both arms retain only the common IQ3 dispatch audit and parameter validation;
the actual selector is immutable before native exec. No native rebuild.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import iq3_model_entry as short
import pool_spin as protocol

P, B, RESULT, base = short.P, short.B, short.RESULT, short.base
ROOT = B / 'decode-iq3-generation-v1'
KEY = 'decode_iq3_generation_speed_v1'
SELF = Path(__file__).resolve()
ARMS = (('reference', 0), ('candidate', 0), ('candidate', 0),
    ('reference', 0), ('reference', 0), ('candidate', 0))
require, digest = short.require, short.entry.digest


def pinned_sources(revision):
    pins = short.pinned_sources(revision)
    pins.update(base.controls(revision, controller=SELF))
    return pins


def normalized(value, root):
    if isinstance(value, str):
        return value.replace(str(root), '<entry>')
    if isinstance(value, dict):
        return {k: normalized(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [normalized(v, root) for v in value]
    return value


def basis(revision):
    first = short.context(0)
    prior = json.loads((first.ROOT / 'record.json').read_text())
    providers, manifest = first.current(prior)
    current = pinned_sources(revision)
    require(all(current.get(n) == h for n, h in prior['sources'].items())
        and set(current) - set(prior['sources']) == {str(SELF.relative_to(P))},
        'only the new comparison controller may extend qualified model inputs')
    common = normalized(prior['environment'], first.ROOT)
    common.pop('STRATA_IQ3_PAIR')
    for arm in (0, 1):
        ctx = short.context(arm)
        row = json.loads((ctx.ROOT / 'record.json').read_text())
        short.contract(row, arm)
        require(base.committed(RESULT)[ctx.KEY] == ctx.compact(row)
            and row['stage'] == 'closed' and row['passed'] and row['complete']
            and row['host_passed'] and row['ownership_closed'] and not row['active']
            and row['model_executed'] and row['outcome']['native_exit_code'] == 0
            and row['outcome']['fresh_faults'] == [] and row['outcome']['boot_unchanged']
            and row['binary'] == prior['binary']
            and row['native_source_commit'] == prior['native_source_commit']
            and row['sources'] == prior['sources'] and row['model_assets'] == prior['model_assets']
            and row['native_validation_probe'] == prior['native_validation_probe'],
            'closed same-build IQ3-off/on short model admissions required')
        env = normalized(row['environment'], ctx.ROOT)
        require(env.pop('STRATA_IQ3_PAIR') == str(arm) and env == common
            and normalized(row['configuration'], ctx.ROOT) ==
                normalized(prior['configuration'], first.ROOT), 'undeclared admitted arm difference')
        require(all(digest(ctx.ROOT / n) == h for n, h in row['generated'].items())
            and digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256']
            and json.loads(Path(row['runtime_manifest']['path']).read_text()) == providers,
            'admitted generated inputs/provider inventory changed')
        require(len(row['outcome']['samples']) == len(row['outcome']['routes']) == 2
            and all(s['actual_read_tokens'] == 32769 and s['actual_reused_tokens'] == 0
                and 1 <= s['generated_tokens'] <= 24 and s['reasoning_observed']
                for s in row['outcome']['samples']), 'short request extent differs')
        for request, route in enumerate(row['outcome']['routes'], 2):
            d = route['IQ3_dispatch']
            require(d['request'] == request and d['arm'] == arm and d['calls'] > 0
                and d['nt1_rows'] > 0 and d['pair_rows'] == arm * d['nt1_rows']
                and route['GDN_FP16_storage_calls'] == 144
                and route['GDN_FP16_storage_tokens'] == 36 * 32768
                and route['MTP_prefill']['callbacks'] == 4
                and route['MTP_prefill']['nominal_cells'] == 32768,
                'actual IQ3/GDN/MTP route admission differs')
    native_root = short.device.host.build.ROOT
    native = json.loads((native_root / 'record.json').read_text())
    names = ('src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/pool.hpp')
    pool = {n: native['native_sources'][n] for n in names}
    require(all(digest(native_root / 'source' / n) == h for n, h in pool.items()),
        'actual compiled pool input changed')
    text = (native_root / 'source' / names[0]).read_text()
    require('std::getenv("STRATA_POOL_SPIN_US")' in text
        and 'spin_before_sleep_ = std::chrono::microseconds' in text,
        'qualified immutable startup pool-spin control required')
    return prior, providers, manifest, pool


def context(index=None):
    spec = importlib.util.spec_from_file_location('strata_iq3_generation_context', protocol.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.select_mode('cohort')
    ctx.ROOT, ctx.KEY, ctx.ARMS = ROOT, KEY, ARMS
    ctx.model = short.context(0)
    ctx.model.pinned_sources = pinned_sources
    ctx.basis = basis
    old_environment, old_route, old_decision = ctx.environment, ctx.route, ctx.decision
    ctx.boundary_checks = []

    def environment(row, plan):
        # The legacy clean-envelope constructor rejects every nonzero audit.
        # Give that constructor a private baseline, then explicitly add the
        # declared common IQ3 audit before exec. Never alter the live process.
        baseline = dict(row, environment_template=dict(row['environment_template'],
            STRATA_IQ3_PAIR_AUDIT='0'))
        env = old_environment(baseline, plan)
        env.update(STRATA_IQ3_PAIR=str(int(plan['kernel'] == 'candidate')),
            STRATA_IQ3_PAIR_AUDIT='1', STRATA_IQ_SINGLE_EXACT='0',
            STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0',
            STRATA_SYCL_DECODE_Q6_COLUMNS='1', STRATA_SYCL_ADAPT_SYNC='1')
        # The measured generator enables decode timing by variable presence,
        # including the string "0". The shared constructor removes it.
        require(env['EnableDirectSubmission'] == '0'
            and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
            and env['STRATA_POOL_SPIN_US'] == '20000' and plan['adapt_swaps'] == 0
            and 'STRATA_DECODE_TIMING' not in env,
            'preserve common runtime,static cache and pool-spin settings')
        return env

    def invariant(row, plan):
        config = json.loads((Path(plan['root']) / 'config.json').read_text())
        require(config['args'] == row['configuration']['args'], 'only IQ3 selector may vary')
        env = environment(row, plan)
        env.pop('STRATA_IQ3_PAIR')
        return hashlib.sha256(json.dumps(dict(config=config, environment=env), sort_keys=True)
            .replace(plan['root'], '<process>').encode()).hexdigest()

    def selected_arm():
        require(type(index) is int and 0 <= index < len(ARMS), 'actual process selector required')
        return int(ARMS[index][0] == 'candidate')

    def route(text, read):
        result = old_route(text, read)
        d = short.dispatches(text, selected_arm(), positive=True)
        require(len(d) == 1 and d[0]['request'] in (5, 6), 'one IQ3 footer per long request')
        result['IQ3_dispatch'] = d[0]
        return result

    def boundary(root, expected):
        raw, history = protocol.cache_boundary(root, expected)
        rows = short.dispatches(raw.decode(), selected_arm(), positive=False)
        require([v['request'] for v in rows] == list(range(1, expected + 1)),
            'bind all completed warmup/long IQ3 requests before continuing')
        require(all(v['calls'] > 0 and v['nt1_rows'] > 0 for v in rows[4:]),
            'long request never exercised eligible CPU IQ3')
        ctx.boundary_checks.append(expected)
        return raw, history

    def decision(processes):
        result = old_decision(processes)
        for name, value in result['workloads'].items():
            wanted = 'strong' if name == 'regular-code' else 'weak'
            value['expected_MTP_class_observed'] = all(
                arm['observed_MTP_class'] == wanted for arm in value['arms'].values())
            value['model_TG_gain_qualified'] = (value['reproducible_TG_gain']
                and value['expected_MTP_class_observed'])
        result['IQ3_model_TG_gain_qualified'] = all(
            v['model_TG_gain_qualified'] for v in result['workloads'].values())
        result['common_IQ3_dispatch_audit_retained'] = True
        result['next'] = ('Assess IQ3 per-workload gain independently of the70TG target;'
            'keep all samples and MTP classes. Dedicated64K/117K PP/TTFT,agent/task,'
            'physical262144/RESTORE and ordinary-provider teardown remain unqualified.'
            'Hardware attribution still needs disjoint host/device/transfer evidence.')
        return result

    ctx.environment, ctx.invariant, ctx.route = environment, invariant, route
    ctx.cache_boundary, ctx.decision = boundary, decision
    return ctx


def prepare(ctx):
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['qualification_reuse'] = dict(build_result_key='decode_iq3_pair_model_build_v1',
        model_result_keys=['decode_iq3_model_entry_arm0_v1', 'decode_iq3_model_entry_arm1_v1'],
        native_source_commit=row['native_source_commit'], engine_sha256=row['binary']['sha256'],
        scope='Exact five-ELF native generation;384/80 arithmetic,28-case pool parity,small traced device and both two-request actual32K IQ3 selectors. Only this new controller extends the identical302 model inputs.')
    row['plan'] = dict(mode='cohort', workload_version=5,
        question='Does the exact IQ3 NT1 GU pair improve sustained model decode?',
        order='Six independent processes0,1,1,0,0,1;each four official warmups then fixed v5 regular-code and varied-prose once. Commit each closed process before next. No retries.',
        constants='Same b3103538 engine/native9922f8a5,providers/safety/private retained preload,throughput BF16 HC/GDN16/MTP16,heads1,131072INT8resident,static1256cache,5workers,spec4,chunk8192,short-read0,caches0,20ms spin,Q6columns1. Only immutable STRATA_IQ3_PAIR0/1 differs.',
        observers='Common IQ3_PAIR_AUDIT1 is retained in both arms to prove actual production dispatch;its incremental overhead is not separately measured. TRACE,CPU_SERVICE,CPU_WORK_CENSUS,DECODE_TIMING,draft/CPU attribution and full API trace are disabled. Parameter validation and existing MTP callbacks remain. No runtime environment mutation.',
        acceptance='Maintained source/build/ELF/model/provider/body/route/journal/clock/validation/fault/owner gates;each long request >=32768fresh input,>=1024generated including thinking,>=512answer characters,>=256drafts,normal0 and quiet validation. At each paused boundary require complete ordered IQ3 IDs1..expected with selected rows=arm*eligible NT1 rows;long IDs5/6 require positive calls/NT1 rows. Require six final IQ3 footers,full logical request binding and all15model hashes before/after cohort.',
        decision='Keep all3paired exact TG ratios,geomean,individual generated lengths/acceptance and PP/TTFT separately. Reproducible gain requires all3candidate/referenceTG>1 per workload,without a minimum percentage floor. Qualification of both MTP regimes additionally requires every code sample>=.60 and prose<=.40. Classification or speed failure preserves all samples and does not stop collection. No exact-text requirement.',
        stopping='Stop first input,owner,provider,budget,route,token,clock,journal,counter,validation or driver fault failure. No retry,replacement,hash refresh,speed-based early stop or criterion changes.',
        retention='Keep compact individual samples,request/selector counters and route/identity/decision evidence. Retire successful journals/logs/JIT after each verified result commit;retain global record/runtime/generated plans only while admission consumes them. No new binary generation.')
    row['budgets'] = dict(wall=3600, cpu=43200, rss_cap=104 << 30,
        text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20)
    ctx.save(row)
    print('Prepared six IQ3-only processes/twelve long requests;commit before host probes.', flush=True)


def run(ctx, index):
    ctx.run(index)
    row = json.loads((ROOT / 'record.json').read_text())
    try:
        require(ctx.boundary_checks == [4, 5, 5, 6], 'all four paused IQ3 boundary gates required')
        proc = row['processes'][index]
        arm = int(ARMS[index][0] == 'candidate')
        root = Path(row['plans'][index]['root'])
        rows = short.dispatches((root / 'engine.stderr').read_text(), arm, positive=False)
        require([v['request'] for v in rows] == list(range(1, 7))
            and [v['IQ3_dispatch'] for v in proc['routes']] == rows[4:],
            'final six-request/long-window IQ3 evidence differs')
        proc['IQ3_dispatch'] = dict(passed=True, arm=arm, boundary_checks=ctx.boundary_checks,
            completed_requests=6, long_requests=rows[4:])
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, IQ3_dispatch_error=repr(error))
    ctx.save(row)
    require(row['stage'] != 'failed', row.get('IQ3_dispatch_error', 'IQ3 generation failed'))


if __name__ == '__main__':
    args = sys.argv[1:]
    require(len(args) == 1 and (args[0] in ('prepare', 'host') or
        args[0].isdigit() and int(args[0]) < len(ARMS)), 'use prepare|host|INDEX0..5')
    index = int(args[0]) if args[0].isdigit() else None
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        ctx = context(index)
        if args[0] == 'prepare':
            prepare(ctx)
        elif args[0] == 'host':
            ctx.comparison.host(ctx)
        else:
            run(ctx, index)
