#!/usr/bin/env python3
"""Matched sustained centering comparison; Main-authored, repository LICENSE.

Reuse the maintained owned bench.sh v5 protocol in a private module context.
The six fresh processes use one qualified native generation. Only the immutable
decode-centering selector changes. Graph-capture counts are not replay counts.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import q6_centering_model_entry as short
import pool_spin as protocol

P, B, RESULT, base = short.P, short.B, short.RESULT, short.base
ROOT = short.checks.model.ROOT / 'generation-centering-v1'
KEY = 'q6_centering_generation_speed_v1'
SELF = Path(__file__).resolve()
ARMS = (('reference', 0), ('candidate', 0), ('candidate', 0),
        ('reference', 0), ('reference', 0), ('candidate', 0))
require, digest = short.require, short.digest


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
    ref = short.context(0)
    prior = json.loads((ref.ROOT / 'record.json').read_text())
    providers, manifest = ref.current(prior)
    data = base.committed(RESULT)
    environments, configurations = [], []
    for arm in (0, 1):
        ctx = short.context(arm)
        row = json.loads((ctx.ROOT / 'record.json').read_text())
        require(data[ctx.KEY] == ctx.compact(row)
            and row['stage'] == 'closed' and row['passed'] and row['complete']
            and row['host_passed'] and row['ownership_closed'] and not row['active']
            and row['model_executed'] and row['outcome']['native_exit_code'] == 0
            and row['outcome']['fresh_faults'] == [] and row['outcome']['boot_unchanged']
            and row['binary'] == prior['binary']
            and row['native_source_commit'] == prior['native_source_commit']
            and row['sources'] == prior['sources'] and row['model_assets'] == prior['model_assets']
            and row['native_validation_probe'] == prior['native_validation_probe']
            and short.pinned_sources(row['source_commit']) == row['sources'],
            'closed exact same-build centering0/1 model entries required')
        require(len(row['outcome']['samples']) == len(row['outcome']['routes']) == 2
            and all(s['actual_read_tokens'] == 32769 and s['actual_reused_tokens'] == 0
                and 1 <= s['generated_tokens'] <= 24 and s['reasoning_observed']
                for s in row['outcome']['samples']), 'short model extent differs')
        for request, route in enumerate(row['outcome']['routes'], 2):
            d = route['IQ3_dispatch']
            captures = route['centering_captures']
            require(d['request'] == request and d['arm'] == 1 and d['calls'] > 0
                and d['nt1_rows'] == d['pair_rows'] > 0 and captures
                and all(c['enabled'] == arm and 0 <= c['centered'] <= c['columns']
                    and (not c['centered'] or (arm == 1 and c['phase'] == 'decode'))
                    for c in captures)
                and any(c['phase'] == 'decode' for c in captures)
                and (arm == 0 or any(c['centered'] > 0 for c in captures)),
                'original selected-graph/IQ3 route qualification differs')
        env = normalized(row['environment'], ctx.ROOT)
        require(env.pop('STRATA_SYCL_DECODE_Q6_CENTERING') == str(arm),
                'original centering selector differs')
        environments.append(env)
        configurations.append(normalized(row['configuration'], ctx.ROOT))
        inv = row['runtime_manifest']
        require(digest(inv['path']) == inv['sha256']
            and json.loads(Path(inv['path']).read_text()) == providers
            and all(digest(ctx.ROOT / n) == sha for n, sha in row['generated'].items())
            and [s['request_sha256'] for s in row['outcome']['samples']] ==
                [s['request_sha256'] for s in prior['outcome']['samples']],
            'admitted runtime/configuration/request evidence changed')
    require(environments[0] == environments[1] and configurations[0] == configurations[1],
            'only the admitted centering selector may differ')
    current = pinned_sources(revision)
    require(set(current) - set(prior['sources']) == {str(SELF.relative_to(P))}
        and all(current.get(n) == h for n, h in prior['sources'].items()),
        'only this comparison controller may extend unchanged model inputs')
    native = json.loads((short.checks.model.ROOT / 'record.json').read_text())
    names = ('src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/pool.hpp')
    pool = {n: native['native_sources'][n] for n in names}
    require(native['binaries']['strata'] == prior['binary']
        and native['source_commit'] == prior['native_source_commit']
        and all(digest(short.checks.model.ROOT / 'source' / n) == h for n, h in pool.items()),
        'qualified native generation/pool changed')
    text = (short.checks.model.ROOT / 'source' / names[0]).read_text()
    require('std::getenv("STRATA_POOL_SPIN_US")' in text
        and 'spin_before_sleep_ = std::chrono::microseconds' in text,
        'qualified immutable startup pool-spin control required')
    return prior, providers, manifest, pool


def context(index=None):
    spec = importlib.util.spec_from_file_location('strata_centering_speed_context', protocol.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.select_mode('cohort')
    ctx.ROOT, ctx.KEY, ctx.ARMS = ROOT, KEY, ARMS
    ctx.model = short.context(0)
    ctx.model.pinned_sources = pinned_sources
    ctx.basis = basis
    old_environment, old_route, old_decision = ctx.environment, ctx.route, ctx.decision
    old_checks = ctx.selfcheck
    ctx.boundary_checks = []

    def selected_arm():
        require(type(index) is int and 0 <= index < len(ARMS), 'actual process index required')
        return int(ARMS[index][0] == 'candidate')

    def environment(row, plan):
        baseline = dict(row, environment_template=dict(row['environment_template'],
            STRATA_IQ3_PAIR_AUDIT='0'))
        env = old_environment(baseline, plan)
        env.update(STRATA_SYCL_DECODE_Q6_CENTERING=str(int(plan['kernel'] == 'candidate')),
            STRATA_SYCL_DECODE_Q6_COLUMNS='1', STRATA_SYCL_ADAPT_SYNC='1',
            STRATA_IQ3_PAIR='1', STRATA_IQ3_PAIR_AUDIT='1', STRATA_IQ_SINGLE_EXACT='0',
            STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0')
        require(env['EnableDirectSubmission'] == '0'
            and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
            and env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == env['ZES_ENABLE_SYSMAN'] == '0'
            and env['STRATA_POOL_SPIN_US'] == '20000' and plan['adapt_swaps'] == 0
            and 'STRATA_DECODE_TIMING' not in env,
            'freeze runtime safety,static cache and clean timing envelope')
        return env

    def invariant(row, plan):
        config = json.loads((Path(plan['root']) / 'config.json').read_text())
        require(config['args'] == row['configuration']['args'], 'native arguments must match')
        env = environment(row, plan)
        env.pop('STRATA_SYCL_DECODE_Q6_CENTERING')
        return hashlib.sha256(json.dumps(dict(config=config, environment=env), sort_keys=True)
            .replace(plan['root'], '<process>').encode()).hexdigest()

    def route(text, read):
        result = old_route(text, read)
        result['Q6_graph_captures'] = short.captures(text, selected_arm())
        rows = short.dispatches(text, 1, positive=True)
        require(len(rows) == 1 and rows[0]['request'] in (5, 6),
                'one actual common IQ3 footer per long request required')
        result['IQ3_dispatch'] = rows[0]
        result['graph_capture_counts_are_replay_counts'] = False
        return result

    def boundary(root, expected):
        raw, history = protocol.cache_boundary(root, expected)
        rows = short.dispatches(raw.decode(), 1, positive=False)
        require([r['request'] for r in rows] == list(range(1, expected + 1))
            and all(r['calls'] > 0 and r['nt1_rows'] > 0 for r in rows[4:]),
            'bind common IQ3 audit to all warmups and timed requests')
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
        result.update(centering_model_TG_gain_qualified=all(
            v['model_TG_gain_qualified'] for v in result['workloads'].values()),
            common_IQ3_dispatch_audit_retained=True,all18_model_shapes_qualified=False,
            next='Judge each actual MTP workload separately;keep all paired samples and unresolved variability. '
                'Dedicated64K/117K prefill and client TTFT,agent/task,physical262144/tails/RESTORE,'
                'ordinary-provider teardown and adoption remain separate. No physical-resource saturation claim.')
        return result

    def checks():
        result = old_checks()
        rows = [dict(kernel=arm, adapt_swaps=swaps, passed=True, complete=True,
            ownership_closed=True, native_identity=dict(pid=i + 1, start_ticks=i + 10),
            samples=[dict(generated_tokens=1024,decode_ms=999.9 if arm == 'candidate' else 1000.,
                prompt_ms=1000.,ttft_s=1.1,acceptance_rate=.8 if name == 'regular-code' else .3)
                for name in ctx.WORKLOADS]) for i, (arm, swaps) in enumerate(ARMS)]
        require(decision(rows)['centering_model_TG_gain_qualified'], 'small paired gains stay eligible')
        slower = copy.deepcopy(rows)
        slower[1]['samples'][0]['decode_ms'] = 1000.1
        require(not decision(slower)['centering_model_TG_gain_qualified'], 'retain one slower pair')
        wrong_class = copy.deepcopy(rows)
        wrong_class[1]['samples'][1]['acceptance_rate'] = .5
        wrong = decision(wrong_class)['workloads']['varied-prose']
        require(wrong['reproducible_TG_gain'] and not wrong['model_TG_gain_qualified'],
                'keep speed and observed MTP class separate')
        duplicate = copy.deepcopy(rows)
        duplicate[1]['native_identity'] = duplicate[0]['native_identity']
        for invalid in (rows[:-1], duplicate):
            try: decision(invalid)
            except RuntimeError: continue
            raise RuntimeError('incomplete/reused-process cohort accepted')
        result.update(centering_capture_parser=short.parser_controls(),paired_TG_controls=True,
            small_gain_retained=True,slower_pair_retained=True,MTP_class_separate=True,
            invalid_cohorts_rejected=2)
        return result

    ctx.environment, ctx.invariant, ctx.route = environment, invariant, route
    ctx.cache_boundary, ctx.decision, ctx.selfcheck = boundary, decision, checks
    return ctx


def prepare(ctx):
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['qualification_reuse'] = dict(build_result_key=short.checks.model.KEY,
        model_result_keys=[short.context(a).KEY for a in (0, 1)],
        native_source_commit=row['native_source_commit'],engine_sha256=row['binary']['sha256'],
        scope='Exact five-output native generation,closed fresh host/device and matched two-request32K '
            'centering0/1 entries at their original revisions. Only this new Python policy extends '
            'the identical313model inputs;no native rebuild or inherited sustained-speed result.')
    row['plan'] = dict(mode='cohort',workload_version=5,
        question='Does the decode-only18-shape Q6 centering selector improve sustained model generation?',
        order='Six independent processes0,1,1,0,0,1;three counterbalanced adjacent pairs. Four official warmups,then fixed v5 regular-code and varied-prose once per process. Commit each closed process before next;no retries.',
        constants='Same92832/a1b0 native source/ELF,providers/safety/preload,throughput BF16 HC/GDN16/MTP16,heads1,131072INT8resident,static1256cache,5workers,spec4,chunk8192,short-read0,caches0,20ms pool spin,Q6columns1,IQ3_PAIR1/AUDIT1. Only immutable STRATA_SYCL_DECODE_Q6_CENTERING0/1 changes.',
        observers='Keep parameter validation,private retained-runtime preload,sparse native/Q6 graph captures,MTP callbacks and common IQ3 dispatch audit. TRACE,CPU_SERVICE,CPU_WORK_CENSUS,DECODE_TIMING,draft/CPU attribution,full API trace and profiler are disabled. IQ3 audit overhead is common and not newly measured.',
        acceptance='Frozen source/build/ELF/model/provider/body/route/journal/clock/fault/owner gates. Each long request>=32768fresh input,>=1024generated including thinking,>=512answer characters and>=256drafts. Join every Q6 centering capture to original phase/T/column receipt;arm0 centered0,arm1 positive decode,prompt centered0. Preserve all6ordered common IQ3 IDs with selected=eligible NT1 rows and positive long IDs5/6. Require all4paused common-cache gates,normal0,quiet validation,full15asset hashes before/after cohort.',
        interpretation='Q6 counts describe captured phase/T graphs,not per-replay invocation counts. Same frozen Verifier::run calls capture(T,input),then executes that keyed graph. Per-request actual capture and footer receipts qualify this bounded immutable route,not execution of every18component-eligible shape or physical resource saturation.',
        decision='Keep all3exact candidate/reference TG ratios,geomean,arm median/min/max and all individual PP/TTFT,length and MTP samples. Reproducible gain requires all3TG ratios>1 per workload,without a minimum percentage. Both MTP regimes additionally require every code acceptance>=.60 and prose<=.40. Speed/class failure does not stop collection or erase samples. No exact-output requirement;prefill and TTFT are independent decisions.',
        stopping='Stop first source,owner,provider,budget,route,body/token,clock,journal,counter,validation or driver-fault failure. No retry,replacement,hash refresh,speed-based early stop or changed criteria.',
        retention='Keep compact individual samples,graph/dispatch/cache conservation,source/configuration and decisions. Retire successful raw logs,journals,JIT after each verified result commit. Keep record/runtime/generated plans only while consumed. Preserve552b fallback;this uses the existing a1b0 generation,not a third engine.')
    row['budgets'] = dict(wall=3600,cpu=43200,rss_cap=104 << 30,
        text_cap=16 << 20,file_cap=32 << 20,total_cap=64 << 20)
    ctx.save(row)
    print('Prepared six matched Q6-centering processes/twelve long requests;commit before frontend probes.',flush=True)


def run(ctx, index):
    ctx.run(index)
    row = json.loads((ROOT / 'record.json').read_text())
    try:
        require(ctx.boundary_checks == [4, 5, 5, 6], 'all four paused request/cache gates required')
        proc = row['processes'][index]
        raw = (Path(row['plans'][index]['root']) / 'engine.stderr').read_text()
        rows = short.dispatches(raw, 1, positive=False)
        captures = short.captures(raw, int(ARMS[index][0] == 'candidate'))
        require([v['request'] for v in rows] == list(range(1, 7))
            and [v['IQ3_dispatch'] for v in proc['routes']] == rows[4:],
            'final request/long-window dispatch evidence differs')
        proc['centering_route'] = dict(passed=True,arm=int(ARMS[index][0] == 'candidate'),
            captured_windows=len(captures),boundary_checks=ctx.boundary_checks,
            graph_capture_counts_are_replay_counts=False,completed_requests=6,
            prompt_graph_observed=any(v['phase'] == 'prompt' for v in captures))
    except BaseException as error:
        row.update(stage='failed',passed=False,complete=False,centering_route_error=repr(error))
    ctx.save(row)
    require(row['stage'] != 'failed', row.get('centering_route_error','centering speed comparison failed'))


if __name__ == '__main__':
    args = sys.argv[1:]
    require(len(args) == 1 and (args[0] in ('prepare', 'host', 'controls') or
        args[0].isdigit() and int(args[0]) < len(ARMS)), 'use prepare|host|controls|INDEX0..5')
    index = int(args[0]) if args[0].isdigit() else None
    ctx = context(index)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if args[0] == 'prepare': prepare(ctx)
        elif args[0] == 'host': ctx.comparison.host(ctx)
        elif args[0] == 'controls': print(json.dumps(ctx.selfcheck(),sort_keys=True))
        else: run(ctx,index)
