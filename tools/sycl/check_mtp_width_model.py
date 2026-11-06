#!/usr/bin/env python3
"""Fixed4/fixed2 model entry and matched generation; Main, repository LICENSE.

Reuse this fork's owned bench.sh supervisor and unchanged qualified native build.
The startup config port is a declared frontend change requiring fresh probes.
These traced short-output checks cannot establish sustained speed or adoption.
The separate generation cohort requires both entries and disables their trace.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
import check_q6_native_half_model as short
import q6_native_half_generation_speed as q6

P, B, RESULT, base = short.P, short.B, short.RESULT, short.base
SELF = Path(__file__).resolve()
FRONTEND = '92f7b07227bae2117087cdf270b9d03c41d5ebd8'
FRONTEND_FILES = ('serve/server.py', 'serve/config_file.py', 'tests/core/test_config_file.py')
require, digest, write = short.require, short.digest, short.write


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def pinned_sources(revision):
    values = dict(short.model_context(0).pinned_sources(revision),
                  **base.controls(revision, controller=SELF))
    for name in FRONTEND_FILES:
        raw = short.build.blob(revision, name)
        require((P / name).read_bytes() == raw, 'uncommitted frontend: ' + name)
        values[name] = hashlib.sha256(raw).hexdigest()
    return values


def source_delta(original, current, additions):
    require(set(current) - set(original) == set(additions) and
            not set(original) - set(current) and
            {n for n in original if current[n] != original[n]} == {'serve/server.py'},
            'only the explicit startup port and new controller/test inputs may differ')


def prospective_basis(revision):
    ctx = q6.context()
    cohort = json.loads((q6.ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[q6.KEY] == ctx.compact(cohort) and
            cohort['stage'] == 'closed' and cohort['passed'] and cohort['complete'] and
            cohort['ownership_closed'] and not cohort['active'] and len(cohort['processes']) == 6 and
            cohort['decision'] == ctx.decision(cohort['processes']),
            'close the original six-process Q6 comparison before a new model entry')
    original_ctx = short.model_context(0)
    original = json.loads((original_ctx.ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[original_ctx.KEY] == original_ctx.compact(original) and
            original['passed'] and original['complete'] and original['ownership_closed'] and
            not original['active'] and original['binary'] == cohort['binary'] and
            original['native_source_commit'] == cohort['native_source_commit'],
            'closed original same-build model entry required')
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                 ctx.shared.gen.blobs(P, original['source_commit'], original['sources'])}
    require(recovered == original['sources'] and len(recovered) == 317,
            'recover the actual original model inputs from Git')
    cohort_sources = {n: hashlib.sha256(raw).hexdigest() for n, raw in
        ctx.shared.gen.blobs(P, cohort['source_commit'], cohort['sources'])}
    require(cohort_sources == cohort['sources'] and len(cohort_sources) == 318 and
            set(cohort_sources) - set(recovered) == {str(q6.SELF.relative_to(P))} and
            all(cohort_sources.get(n) == h for n, h in recovered.items()),
            'recover and preserve all original comparison inputs including its decision controller')
    sources = pinned_sources(revision)
    added = {str(SELF.relative_to(P)), str(q6.SELF.relative_to(P)), FRONTEND_FILES[2]}
    source_delta(recovered, sources, added)
    source_delta(cohort_sources, sources, added - {str(q6.SELF.relative_to(P))})
    require(all(sources[n] == hashlib.sha256(short.build.blob(FRONTEND, n)).hexdigest()
                for n in FRONTEND_FILES), 'use the tested exact startup-config port')
    # An unsuccessful half optimization must not block the independent width test.
    half = int(cohort['decision']['native_half_model_TG_gain_qualified'])
    return dict(cohort_key=q6.KEY, cohort_source_commit=cohort['source_commit'],
        cohort_sha256=signature(ctx.compact(cohort)), half_arm=half,
        original_model_source_commit=original['source_commit'], original_input_count=317,
        current_input_count=len(sources), changed_original_inputs=['serve/server.py'],
        added_inputs=sorted(added), frontend_source_commit=FRONTEND,
        scope='Same six native ELFs/recipe/providers/safety;fresh probes and bounded model checks for the exact startup port and MTP cap. No inherited sustained-speed or full-context qualification.')


def lease_route(text):
    def one(prefix, pattern):
        rows = [v for v in text.splitlines() if v.startswith(prefix)]
        require(len(rows) == 1, 'missing/duplicate lease marker: ' + prefix)
        match = re.fullmatch(pattern, rows[0])
        require(match is not None, 'malformed lease marker: ' + prefix)
        values = tuple(map(int, match.groups()))
        require(all(0 <= v < 2**64 for v in values), 'lease counter outside uint64')
        return values
    layout = one('strata loan audit: phase=layout ',
        r'strata loan audit: phase=layout chunk=(\d+) first=(\d+) slots=(\d+) bytes=(\d+) need=(\d+) lent=(\d+) mask_match=1')
    chunk, first, slots, size, needed, lent = layout
    require(chunk == 8192 and 128 <= first < slots == 1256 and
            0 < lent <= slots - first and size >= needed > 0, 'positive occupied tail/capacity required')
    states = []
    for phase, ready in [('masked', 0), ('ready', 1)]:
        values = one('strata cache loan: phase=' + phase + ' ',
            r'strata cache loan: phase=' + phase +
            r' generation=(\d+) chunk=(\d+) first=(\d+) lent=(\d+) decode_ready=' + str(ready))
        require(values[0] > 0 and values[1:] == (chunk, first, lent), 'lease generation/extent differs')
        states.append(values)
    require(states[0] == states[1], 'a different loan was returned')
    tables = []
    for phase in ('lend', 'refill'):
        entries, free, total = one('strata loan audit: phase=' + phase + ' ',
            r'strata loan audit: phase=' + phase +
            r' table_entries=(\d+) table_match=1 free_bytes=(\d+) total_bytes=(\d+)')
        require(entries > 0 and 0 <= free <= total and total > 0, 'residency audit extent differs')
        tables.append((entries, total))
    require(tables[0] == tables[1], 'residency table/device extent changed')
    rows, payload, returned = one('strata loan audit: phase=payload ',
        r'strata loan audit: phase=payload rows=(\d+) bytes=(\d+) lent=(\d+) same_slots=1 payload_match=1')
    require(returned == lent and lent <= rows <= slots and payload > 0,
            'resident prefix/returned payload audit differs')
    order = ['strata loan audit: phase=layout ', 'strata loan audit: phase=lend ',
        'strata cache loan: phase=masked ', 'strata prefill dense: phase=retired ',
        'strata prefill dense: phase=bank ', 'strata prefill dense: phase=restored ',
        'strata loan audit: phase=payload ', 'strata loan audit: phase=refill ',
        'strata cache loan: phase=ready ', 'strata trace: window ']
    require(all(text.count(v) == 1 for v in order[:-1]) and text.count(order[-1]) > 1,
            'one complete owner transaction and multiple decoded windows required')
    offsets = [text.index(v) for v in order]
    require(offsets == sorted(offsets), 'borrower/graph retirement/refill/decode ordering differs')
    return dict(generation=states[0][0], chunk=chunk, first=first, slots=slots, lent=lent,
        capacity_bytes=size, needed_bytes=needed, resident_rows_payload_checked=rows,
        resident_payload_checked_bytes=payload, residency_table_entries=tables[0][0],
        masked_decode_ready=False, returned_decode_ready=True,
        scope='Native checked table and resident payload/prefix audit;not bus traffic or recurrent-state numerical parity.')


def width_route(text, cap, half, request):
    require(type(cap) is int and cap in (2, 4), 'only allocated4/cap2 or4')
    lines = [v for v in text.splitlines() if v.startswith('strata trace: window ')]
    parsed = [re.fullmatch(r'strata trace: window (\d+) (\d+)', v) for v in lines]
    require(len(parsed) > 1 and all(parsed), 'missing/malformed actual verifier windows')
    rounds = [tuple(map(int, v.groups())) for v in parsed]
    require(rounds[0] == (32768, 1) and all(t == cap for _, t in rounds[1:]) and
            all(0 < b[0] - a[0] <= a[1] for a, b in zip(rounds, rounds[1:])),
            'actual fixed-width/position sequence differs')
    snap = short.snapshots(text, half, request)
    windows = snap['windows']
    require({w['tokens'] for w in windows} == {1, cap} and len(windows) == 2 and
            all(w['phase'] == 'decode' and all(s['shape'][2] == w['tokens'] for s in w['shape_calls'])
                and w['replay_completed'] == sum(t == w['tokens'] for _, t in rounds)
                for w in windows), 'fresh decode captures/replay counts do not match the traced windows')
    return dict(request=request, allocated_width=4, actual_width_cap=cap, first_position=rounds[0][0],
        rounds=len(rounds), captures=[{k: w[k] for k in ('tokens', 'capture_id', 'replay_started', 'replay_completed')}
            for w in windows], completed_head_capture_ids=snap['completed_head_capture_ids'],
        scope='Live cumulative captures matched to all traced rounds after this request retired its prior graph readers.')


def normalized(row, root):
    config, env = copy.deepcopy(row['configuration']), dict(row['environment'])
    args = config['args']
    args[args.index('--mtp-max-t') + 1] = '<cap>'
    return json.dumps(dict(config=config, environment=env), sort_keys=True).replace(str(root), '<entry>')


def frozen_entry(cap, half, root):
    original_root = short.model_context(0).ROOT
    original = json.loads((original_root / 'record.json').read_text())
    def relocate(value):
        if isinstance(value, str): return value.replace(str(original_root), str(root))
        if isinstance(value, list): return [relocate(v) for v in value]
        if isinstance(value, dict): return {k: relocate(v) for k, v in value.items()}
        return value
    config, env = (relocate(original[k]) for k in ('configuration', 'environment'))
    require('--mtp-max-t' not in config['args'] and '--spec-min-p' not in config['args'],
            'unexpected inherited width selectors')
    config['args'].extend(['--mtp-max-t', str(cap), '--spec-min-p', '0'])
    env.update(STRATA_SYCL_DECODE_Q6_NATIVE_HALF=str(half), STRATA_SYCL_CACHE_LOAN_AUDIT='1',
               STRATA_POOL_SPIN_US='20000')
    env.pop('STRATA_DECODE_TIMING', None)
    return config, env


def context(cap):
    require(type(cap) is int and cap in (2, 4), 'use cap4 then cap2')
    spec = importlib.util.spec_from_file_location('strata_mtp_width_model', short.model_template.template.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.configure('adaptive-throughput')
    ctx.ROOT = short.build.ROOT / ('model-entry-mtp' + str(cap) + '-v1')
    ctx.KEY = 'mtp_width_model_entry_cap' + str(cap) + '_v1'
    ctx.proof, ctx.pinned_sources = short.model_proof, pinned_sources
    original_current, original_route = ctx.current, ctx.profile_route

    def current(row):
        basis = prospective_basis(row['source_commit'])
        require(row['qualification_basis'] == basis and row['cap'] == cap,
                'prospective frontend/cohort basis changed')
        env, args = row['environment'], row['configuration']['args']
        config_expected, env_expected = frozen_entry(cap, basis['half_arm'], ctx.ROOT)
        require(row['configuration'] == config_expected and env == env_expected,
                'only cap/common half,private paths,loan audit and fixed pool spin may differ from the original entry')
        for flag, value in [('--spec', '4'), ('--mtp-max-t', str(cap)), ('--spec-min-p', '0'),
                ('--suffix-draft', '0'), ('--short-read', '0'), ('--adapt-swaps', '0'),
                ('--expert-cache', '1256'), ('--prefill', '8192')]:
            require(args.count(flag) == 1 and args[args.index(flag) + 1] == value, 'frozen argument differs: ' + flag)
        require('--no-prefill-borrow' not in args and '--no-token-graph' in args and
                env['STRATA_SYCL_CACHE_SCRATCH_LEASE'] == env['STRATA_PREFILL_NATIVE_DENSE_BANK'] ==
                env['STRATA_SYCL_CACHE_LOAN_AUDIT'] == env['STRATA_SYCL_Q6_ROUTE_AUDIT'] == env['STRATA_TRACE'] == '1' and
                env['STRATA_SYCL_DECODE_Q6_NATIVE_HALF'] == str(basis['half_arm']) and
                row['read_tokens'] == 32769 and row['runs'] == 2, 'freeze placement,loan and width observers')
        if cap == 2:
            reference = json.loads((context(4).ROOT / 'record.json').read_text())
            require(base.committed(RESULT)[context(4).KEY] == ctx.compact(reference) and
                    reference['passed'] and reference['complete'] and reference['ownership_closed'] and
                    not reference['active'] and reference['sources'] == row['sources'] and
                    reference['binary'] == row['binary'] and
                    normalized(reference, context(4).ROOT) == normalized(row, ctx.ROOT),
                    'closed same-source/frontend/ELF/placement cap4 entry required before cap2')
        return original_current(row)

    def route(text):
        result = original_route(text)
        ids = short.dispatches(text, 1, positive=True)
        require(len(ids) == 1 and ids[0]['request'] in (2, 3), 'join actual model request')
        result['cache_lease'] = lease_route(text)
        result['mtp_width'] = width_route(text, cap, prospective_basis(
            json.loads((ctx.ROOT / 'record.json').read_text())['source_commit'])['half_arm'], ids[0]['request'])
        return result

    ctx.current, ctx.profile_route = current, route
    return ctx


def prepare(ctx, cap):
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    basis = prospective_basis(revision)
    if cap == 2:
        reference_ctx = context(4)
        reference = json.loads((reference_ctx.ROOT / 'record.json').read_text())
        reference_ctx.current(reference)
        require(reference['passed'] and reference['complete'], 'cap4 must close first')
    checks = controls()
    ctx.prepare()
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    row['configuration'], row['environment'] = frozen_entry(cap, basis['half_arm'], ctx.ROOT)
    write(ctx.ROOT / 'config.json', row['configuration'])
    row['generated']['config.json'] = digest(ctx.ROOT / 'config.json')
    row.update(cap=cap, qualification_basis=basis, width_parser_controls=checks)
    row['criteria'] = ('Fresh benchmark/server probes for the declared startup port;cap4 then cap2 in fresh processes. '
        'Same six native ELFs,providers,safety,throughput placement,spec4 allocation and common half choice from the closed Q6 cohort. '
        'One thinking warmup and two fresh32769-input/1..24-output requests per process. Require existing body/count/clock/owner/asset/validation/fault gates, '
        'a positive tail loan,checked full resident prefix/returned payload,masked->retired->bank->restored->ready ordering, '
        'actual firstT1 then fixed cap2/4 windows and exact completed fresh-capture replay counts. Require increasing lease generations '
        'and disjoint capture IDs across long requests. No retry,hash refresh or exact-text equality gate. '
        'This diagnostic is not clean/sustained TG,task,agent,head-payload parity,ordinary provider teardown or physical262144/RESTORE qualification.')
    row['retention'] = ('Imminent width4/2 model comparison only. Keep compact actual request/loan/capture/count results and identities;'
        'retire successful trace/journal/JIT after verified result commit. Reuse current candidate and retained552b fallback;no new build generation.')
    ctx.save(row)
    print('Prepared fixed-width route entry;commit before frontend probes.', flush=True)


def run(ctx, cap):
    try:
        ctx.run()
        row = json.loads((ctx.ROOT / 'record.json').read_text())
        routes = row['outcome']['routes']
        widths = [r['mtp_width'] for r in routes]
        leases = [r['cache_lease'] for r in routes]
        require([w['request'] for w in widths] == [2, 3] and leases[1]['generation'] > leases[0]['generation'] and
                set(w['capture_id'] for w in widths[0]['captures']).isdisjoint(
                    w['capture_id'] for w in widths[1]['captures']), 'request loan/capture lifetime was not renewed')
        if cap == 2:
            reference = json.loads((context(4).ROOT / 'record.json').read_text())
            require([s['request_sha256'] for s in row['outcome']['samples']] ==
                    [s['request_sha256'] for s in reference['outcome']['samples']], 'matched request bodies changed')
        row['outcome']['width_lifetime_checks_passed'] = True
        ctx.save(row)
    except BaseException as error:
        if (ctx.ROOT / 'record.json').exists():
            row = json.loads((ctx.ROOT / 'record.json').read_text())
            row.update(stage='failed', passed=False, width_gate_error=repr(error))
            ctx.save(row)
        raise


def controls():
    good = ('strata loan audit: phase=layout chunk=8192 first=256 slots=1256 bytes=4096 need=2048 lent=10 mask_match=1\n'
        'strata loan audit: phase=lend table_entries=24576 table_match=1 free_bytes=100 total_bytes=1000\n'
        'strata cache loan: phase=masked generation=2 chunk=8192 first=256 lent=10 decode_ready=0\n'
        'strata prefill dense: phase=retired residency=Full\nstrata prefill dense: phase=bank residency=Bank\n'
        'strata prefill dense: phase=restored residency=Full\n'
        'strata loan audit: phase=payload rows=1256 bytes=5000 lent=10 same_slots=1 payload_match=1\n'
        'strata loan audit: phase=refill table_entries=24576 table_match=1 free_bytes=100 total_bytes=1000\n'
        'strata cache loan: phase=ready generation=2 chunk=8192 first=256 lent=10 decode_ready=1\n'
        'strata trace: window 32768 1\nstrata trace: window 32769 2\n')
    lease_route(good)
    corrupt = [good.replace('first=256', 'first=0'), good.replace('lent=10', 'lent=0'),
        good.replace('need=2048', 'need=8192'), good.replace('decode_ready=1', 'decode_ready=0'),
        good.replace('phase=ready generation=2', 'phase=ready generation=3'),
        good.replace('payload_match=1', 'payload_match=0'), good.replace('same_slots=1', 'same_slots=0'),
        good.replace('phase=refill table_entries=24576', 'phase=refill table_entries=1'),
        good.replace('phase=restored', 'phase=missing'), good + good,
        good.replace('generation=2', 'generation=' + str(2**64)),
        good.replace('strata trace: window 32769 2\n', '')]
    for value in corrupt:
        try: lease_route(value)
        except RuntimeError: continue
        raise RuntimeError('corrupt lease evidence accepted')
    old = {'serve/server.py': 'a', 'unchanged': 'b'}
    source_delta(old, dict(old, **{'serve/server.py': 'c', 'new': 'd'}), {'new'})
    bad_sources = [dict(old, new='d'), {'serve/server.py': 'c', 'new': 'd'},
        {'serve/server.py': 'c', 'unchanged': 'x', 'new': 'd'}]
    for current in bad_sources:
        try: source_delta(old, current, {'new'})
        except RuntimeError: continue
        raise RuntimeError('undeclared source delta accepted')
    positive = negative = 0
    for cap in (2, 4):
        for half in (0, 1):
            text = good.rsplit('strata trace: window ', 1)[0] + f'strata trace: window 32769 {cap}\n'
            for t, capture in [(1, 7), (cap, 8)]:
                text += (f'strata Q6_ROUTE request=2 device=0 phase=decode tokens={t} capture_id={capture} scope=live_capture_cumulative native_half_enabled={half} complete=1 shapes=1 cache_hits=0 replay_started=1 replay_completed=1\n'
                    f'strata Q6_SHAPE request=2 device=0 phase=decode tokens={t} capture_id={capture} ni=2560 no=248320 nc={t} reference_calls=0 columns_calls={1-half} centered_calls=0 native_half_calls={half}\n')
            width_route(text, cap, half, 2); positive += 1
            bad = [text.replace('window 32769 ' + str(cap), 'window 32769 3'),
                text.replace('window 32768 1', 'window 32768 2'), text.replace('window 32769', 'window 32768'),
                text.replace('replay_started=1 replay_completed=1', 'replay_started=2 replay_completed=2'),
                text.replace('phase=decode', 'phase=prompt'), text.replace('request=2', 'request=3')]
            for value in bad:
                try: width_route(value, cap, half, 2)
                except RuntimeError: negative += 1; continue
                raise RuntimeError('corrupt fixed-width trace/capture evidence accepted')
    return dict(passed=True, lease_positive=1, lease_negative=len(corrupt), source_positive=1,
        source_negative=len(bad_sources), width_positive=positive, width_negative=negative,
        GPU_executed=False, scope='Parser/admission controls only;no native model,lease or speed execution.')


SPEED_ROOT = short.build.ROOT / 'generation-mtp-width-v1'
SPEED_KEY = 'mtp_width_generation_speed_v1'


def generation_basis(revision):
    prior, providers, manifest = None, None, None
    for cap in (4, 2):
        ctx = context(cap)
        row = json.loads((ctx.ROOT / 'record.json').read_text())
        files, python = ctx.current(row)
        outcome = row.get('outcome', {})
        require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
                row['host_passed'] and row['ownership_closed'] and not row['active'] and
                row['model_executed'] and outcome['native_exit_code'] == 0 and
                outcome['width_lifetime_checks_passed'] and outcome['fresh_faults'] == [] and
                outcome['boot_unchanged'] and row['sources'] == pinned_sources(revision),
                'both closed same-source/frontend/ELF width entries are required')
        if prior is None: prior, providers, manifest = row, files, python
        else:
            require(files == providers and python == manifest and
                    all(row[n] == prior[n] for n in ('binary', 'native_source_commit', 'model_assets',
                        'native_validation_probe', 'qualification_basis')) and
                    normalized(row, ctx.ROOT) == normalized(prior, context(4).ROOT),
                    'width entry constants differ')
    native = json.loads((short.build.ROOT / 'record.json').read_text())
    names = ('src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/pool.hpp')
    pool = {n: native['native_sources'][n] for n in names}
    require(native['binaries']['strata'] == prior['binary'] and
            all(digest(short.build.ROOT / 'source' / n) == h for n, h in pool.items()),
            'qualified native pool changed')
    return prior, providers, manifest, pool


def generation_decision(processes):
    result = q6.context().decision(processes)
    result.pop('native_half_model_TG_gain_qualified')
    for name, value in result['workloads'].items():
        wanted = 'strong' if name == 'regular-code' else 'weak'
        reference_class = value['arms']['reference']['observed_MTP_class']
        # Draft depth changes the denominator. Do not impose the fixed4 band
        # on fixed2 or retrospectively discard a faster fixed2 sample.
        value['arms']['candidate']['acceptance_band_at_cap2'] = value['arms']['candidate'].pop('observed_MTP_class')
        value.pop('expected_MTP_class_observed')
        value['reference_MTP_class_observed'] = reference_class == wanted
        value['model_TG_gain_qualified'] = value['reproducible_TG_gain'] and value['reference_MTP_class_observed']
    result.update(mtp_width_model_TG_gain_qualified=all(v['model_TG_gain_qualified'] for v in result['workloads'].values()),
        next='Judge fixed2 versus fixed4 separately per workload;retain slower pairs and unresolved variance. Dedicated64K/117K PP/TTFT,handoff,agent/task,physical262144/tails/RESTORE and ordinary provider teardown remain separate. No physical saturation claim.')
    return result


def generation_controls():
    rows = [dict(kernel=arm, adapt_swaps=swaps, passed=True, complete=True, ownership_closed=True,
        native_identity=dict(pid=i + 1, start_ticks=i + 10), samples=[dict(generated_tokens=1024,
            decode_ms=999.9 if arm == 'candidate' else 1000., prompt_ms=1000., ttft_s=1.1,
            acceptance_rate=(.8 if name == 'regular-code' else .7 if arm == 'candidate' else .3))
            for name in ('regular-code', 'varied-prose')]) for i, (arm, swaps) in enumerate(q6.ARMS)]
    require(generation_decision(rows)['mtp_width_model_TG_gain_qualified'],
            'a different fixed2 acceptance band must not exclude its small speed gain')
    slower = copy.deepcopy(rows); slower[1]['samples'][0]['decode_ms'] = 1000.1
    require(not generation_decision(slower)['mtp_width_model_TG_gain_qualified'], 'retain a slower pair')
    wrong = copy.deepcopy(rows); wrong[0]['samples'][1]['acceptance_rate'] = .5
    decision = generation_decision(wrong)['workloads']['varied-prose']
    require(decision['reproducible_TG_gain'] and not decision['model_TG_gain_qualified'],
            'reference classification and speed must remain separate')
    return dict(passed=True, depth_denominator_change_retained=True, small_gain_retained=True,
        slower_pair_retained=True, reference_MTP_class_separate=True, GPU_executed=False,
        configuration_controls=generation_configuration_controls(),
        unchanged_protocol_controls=q6.context().selfcheck())


def generation_context(index=None):
    spec = importlib.util.spec_from_file_location('strata_mtp_width_generation', q6.protocol.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx); ctx.select_mode('cohort')
    ctx.ROOT, ctx.KEY, ctx.ARMS = SPEED_ROOT, SPEED_KEY, q6.ARMS
    ctx.model, ctx.basis = context(4), generation_basis
    old_environment, old_current, old_route = ctx.environment, ctx.current, ctx.route
    ctx.boundary_checks = []
    ctx.preparing_half = None

    def cap_of(plan):
        require(type(plan['index']) is int and 0 <= plan['index'] < 6 and
                (plan['kernel'], plan['adapt_swaps']) == q6.ARMS[plan['index']], 'unplanned width arm')
        return 2 if plan['kernel'] == 'candidate' else 4

    def environment(row, plan):
        half = row.get('selected_half')
        if half is None:
            require(ctx.preparing_half in (0, 1), 'common half must be explicit outside preparation')
            half = ctx.preparing_half
        baseline = dict(row, environment_template=dict(row['environment_template'],
            STRATA_IQ3_PAIR_AUDIT='0', STRATA_SYCL_Q6_ROUTE_AUDIT='0', STRATA_SYCL_CACHE_LOAN_AUDIT='0'))
        env = old_environment(baseline, plan)
        env.update(STRATA_SYCL_DECODE_Q6_NATIVE_HALF=str(half),
            STRATA_SYCL_Q6_ROUTE_AUDIT='1', STRATA_SYCL_DECODE_Q6_COLUMNS='1',
            STRATA_SYCL_DECODE_Q6_CENTERING='0', STRATA_SYCL_ADAPT_SYNC='1',
            STRATA_IQ3_PAIR='1', STRATA_IQ3_PAIR_AUDIT='1', STRATA_IQ_SINGLE_EXACT='0',
            STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0')
        require(env['STRATA_SYCL_CACHE_LOAN_AUDIT'] == '0' and
                env['STRATA_SYCL_CACHE_SCRATCH_LEASE'] == '1' and
                env['EnableDirectSubmission'] == '0' and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and
                env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == env['ZES_ENABLE_SYSMAN'] == '0' and
                'STRATA_DECODE_TIMING' not in env and 'STRATA_TRACE' not in env,
                'preserve runtime safety and disable expensive entry-only observers')
        return env

    def invariant(row, plan):
        config = json.loads((Path(plan['root']) / 'config.json').read_text())
        config['args'][config['args'].index('--mtp-max-t') + 1] = '<cap>'
        return signature(dict(config=json.loads(json.dumps(config).replace(plan['root'], '<process>')),
            environment=json.loads(json.dumps(environment(row, plan)).replace(plan['root'], '<process>'))))

    def current(row):
        require(row['selected_half'] == prospective_basis(row['source_commit'])['half_arm'], 'common half choice changed')
        prior = json.loads((context(4).ROOT / 'record.json').read_text())
        require(row['configuration'] == prior['configuration'] and row['environment_template'] == prior['environment'],
                'base entry configuration/environment changed')
        for plan in row['plans']:
            expected = copy.deepcopy(row['configuration']); expected['log'] = str(Path(plan['root']) / 'engine.stderr')
            expected['args'][expected['args'].index('--mtp-max-t') + 1] = str(cap_of(plan))
            require(json.loads((Path(plan['root']) / 'config.json').read_text()) == expected,
                    'only the fixed cap/private paths may differ between generation arms')
        return old_current(row)

    def route(text, read):
        require(type(index) is int and 0 <= index < 6, 'actual generation index required')
        row = json.loads((SPEED_ROOT / 'record.json').read_text()); cap = cap_of(row['plans'][index])
        value = old_route(text, read); ids = short.dispatches(text, 1, positive=True)
        require(len(ids) == 1 and ids[0]['request'] in (5, 6), 'join long IQ3/Q6 request')
        snap = short.snapshots(text, row['selected_half'], ids[0]['request'])
        require(len(snap['windows']) == 2 and {w['tokens'] for w in snap['windows']} == {1, cap} and
                all(w['phase'] == 'decode' and all(s['shape'][2] == w['tokens'] for s in w['shape_calls'])
                    for w in snap['windows']), 'actual long decode width differs')
        value.update(IQ3_dispatch=ids[0], Q6_live_snapshot=snap, allocated_width=4, actual_width_cap=cap)
        return value

    def boundary(root, expected):
        raw, history = q6.protocol.cache_boundary(root, expected)
        ids = short.dispatches(raw.decode(), 1, positive=False)
        require([v['request'] for v in ids] == list(range(1, expected + 1)) and
                all(v['calls'] > 0 and v['nt1_rows'] > 0 for v in ids[4:]), 'ordered IQ3 request history differs')
        ctx.boundary_checks.append(expected)
        return raw, history

    ctx.environment, ctx.invariant, ctx.current, ctx.route = environment, invariant, current, route
    ctx.cache_boundary, ctx.decision, ctx.selfcheck = boundary, generation_decision, generation_controls
    # The generic preparer writes cap4 first; the wrapper below finalizes cap2
    # and its hashes before committing or permitting any host/model execution.
    original_prepare = ctx.prepare
    def prepare():
        # environment() is called by the generic preparer before save().
        prior = json.loads((context(4).ROOT / 'record.json').read_text())
        selected = prior['qualification_basis']['half_arm']
        ctx.preparing_half = selected
        try: original_prepare()
        finally: ctx.preparing_half = None
        row = json.loads((SPEED_ROOT / 'record.json').read_text()); row['selected_half'] = selected
        for plan in row['plans']:
            path = Path(plan['root']) / 'config.json'; config = json.loads(path.read_text())
            config['args'][config['args'].index('--mtp-max-t') + 1] = str(cap_of(plan))
            write(path, config); plan['generated']['config.json'] = digest(path)
            plan['environment_sha256'] = ctx.comparison.fingerprint(environment(row, plan))
        row['invariant_signature'] = invariant(row, row['plans'][0])
        require(all(invariant(row, plan) == row['invariant_signature'] for plan in row['plans']), 'unmatched width arm')
        row['qualification_reuse'] = dict(model_result_keys=[context(cap).KEY for cap in (4, 2)],
            native_source_commit=row['native_source_commit'], engine_sha256=row['binary']['sha256'],
            basis=prior['qualification_basis'], scope='Both freshly qualified frontend/width/loan entries;same six ELFs and frozen source. No inherited sustained speed or physical-context qualification.')
        row['plan'] = dict(mode='cohort', workload_version=5,
            question='Does fixed cap2 improve sustained generation over fixed4 without changing allocated width or prefill placement?',
            motivation='The prior long fixed4 workloads have about.824/.389 draft acceptance. Cap2 may reduce unused CPU/GPU target work but adds round/submission overhead;measure whole TG. Those counts do not establish CPU,RAM,PCIe or GPU saturation.',
            order='Six fresh processes4,2,2,4,4,2;three adjacent counterbalanced pairs. Four official warmups then fixed thinking-on v5 code/prose once each. Commit each closed process before next;no retries.',
            constants='Same qualified cb7 engine/ec4 native source,six ELFs,84providers/safety/private preload,common half choice,throughput BF16 HC/GDN16/MTP16,heads1,131072INT8resident,1256static slots,5workers,spec4 allocation,8192chunk,short-read0,min-p0,suffix0,caches0,20ms pool spin,Q6columns1/centering0,IQ3_PAIR1. Only --mtp-max-t4/2 differs.',
            observers='Parameter validation,common IQ3 and Q6 live-capture audits;disable TRACE,loan payload audit,CPU attribution,DECODE_TIMING,API trace and profiler. Paused checks outside reported clocks. No observer-overhead or ordinary-teardown claim.',
            acceptance='Existing source/build/fixture/asset/provider/body/journal/clock/fault/owner gates;each long request>=32768fresh input,>=1024generated including thinking,>=512answer characters,>=256drafts. Require actual T1/T4 or T1/T2 and completed selected Q6 head replays,all four paused gates,six cache/IQ3 footers,normal0 and quiet validation.',
            decision='Retain all3exact candidate/reference TG ratios,geomean and individual PP/TTFT/length/MTP samples. Gain requires all3TG ratios>1;no minimum percentage. Classify code>=.60/prose<=.40 only from every fixed4 reference. Fixed2 acceptance/yield is reported at its changed denominator and never used to retrospectively exclude a speed sample. Speed/classification failures do not stop collection.',
            stopping='Stop first source,owner,provider,budget,route,body/token,clock,journal,validation or fresh-fault failure. No retry,replacement,hash refresh,speed stop or changed criteria.',
            retention='Compact individual samples,actual route/cache conservation,identities and decisions. Retire successful raw output/JIT after verified result commit. No extra build. Dedicated PP/TTFT,handoff,agent/task,full262144/tails/RESTORE and adoption remain separate.')
        row['budgets'] = dict(wall=3600, cpu=43200, rss_cap=104 << 30, text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20)
        ctx.save(row)
        print('Prepared fixed4/fixed2 sustained cohort;commit before fresh host probes.', flush=True)
    ctx.prepare = prepare
    return ctx


def generation_configuration_controls():
    """Exercise the actual environment/signature,including preparation bootstrap."""
    ctx = generation_context()
    with tempfile.TemporaryDirectory(prefix='strata-mtp-width-controls-') as temporary:
        root = Path(temporary)
        config, env = frozen_entry(4, 0, root / 'entry')
        row = dict(configuration=config, environment_template=env, selected_half=0)
        original = signature(row)
        plans = []
        for index, cap in enumerate((4, 2)):
            folder = root / str(index); folder.mkdir()
            plan = dict(index=index, kernel='reference' if cap == 4 else 'candidate',
                adapt_swaps=0, heads=1, root=str(folder))
            actual = copy.deepcopy(config); actual['log'] = str(folder / 'engine.stderr')
            actual['args'][actual['args'].index('--mtp-max-t') + 1] = str(cap)
            write(folder / 'config.json', actual); plans.append(plan)
        baseline = ctx.invariant(row, plans[0])
        require(baseline == ctx.invariant(row, plans[1]), 'cap-only paired plans must share one invariant')
        require(signature(row) == original, 'plan generation must not mutate the entry template')
        selected = ctx.environment(row, plans[0])
        require(selected['STRATA_SYCL_DECODE_Q6_NATIVE_HALF'] == '0' and
                selected['STRATA_SYCL_CACHE_LOAN_AUDIT'] == '0' and
                'STRATA_TRACE' not in selected and 'STRATA_DECODE_TIMING' not in selected,
                'clean observer/half selection differs')
        preparing = dict(row); preparing.pop('selected_half')
        ctx.preparing_half = 1
        try: prepared = ctx.environment(preparing, plans[0])
        finally: ctx.preparing_half = None
        expected = dict(selected, STRATA_SYCL_DECODE_Q6_NATIVE_HALF='1')
        require(prepared == expected, 'explicit preparation half must preserve every other environment value')
        try: ctx.environment(preparing, plans[0])
        except RuntimeError: pass
        else: raise RuntimeError('missing common half accepted outside preparation')
        unsafe = copy.deepcopy(row); unsafe['environment_template']['EnableDirectSubmission'] = '1'
        try: ctx.environment(unsafe, plans[0])
        except RuntimeError: pass
        else: raise RuntimeError('changed runtime safety accepted')
        path = Path(plans[1]['root']) / 'config.json'; altered = json.loads(path.read_text())
        altered['args'][altered['args'].index('--prefill') + 1] = '4096'
        write(path, altered)
        require(ctx.invariant(row, plans[1]) != baseline, 'prefill drift hidden by cap normalization')
    return dict(passed=True, matched_cap_pair=True, immutable_template=True,
        clean_observers=True, preparation_half_bootstrap=True,
        missing_half_rejected=True, unsafe_runtime_rejected=True, prefill_drift_detected=True,
        GPU_executed=False)


def generation_run(ctx, index):
    ctx.run(index)
    row = json.loads((SPEED_ROOT / 'record.json').read_text())
    try:
        require(ctx.boundary_checks == [4, 5, 5, 6], 'all four paused request/cache gates required')
        proc = row['processes'][index]
        require([r['IQ3_dispatch']['request'] for r in proc['routes']] == [5, 6] and
                all(r['Q6_live_snapshot']['completed_head_capture_ids'] for r in proc['routes']),
                'final long width/head request evidence differs')
        captures = [{w['capture_id'] for w in r['Q6_live_snapshot']['windows']} for r in proc['routes']]
        require(max(captures[0]) < min(captures[1]), 'long request reused a retired capture identity')
        proc['width_route'] = dict(passed=True, allocated_width=4,
            cap=2 if proc['kernel'] == 'candidate' else 4, boundary_checks=ctx.boundary_checks)
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, width_route_error=repr(error))
    ctx.save(row); require(row['stage'] != 'failed', row.get('width_route_error', 'width cohort failed'))


if __name__ == '__main__':
    args = sys.argv[1:]
    speed = len(args) == 2 and args[0] == 'speed' and (args[1] in ('prepare', 'host') or
        args[1].isdigit() and int(args[1]) < 6)
    require(args == ['controls'] or speed or len(args) == 2 and args[0] in ('4', '2') and
            args[1] in ('prepare', 'host', 'run'), 'use controls,4|2 prepare|host|run,or speed prepare|host|INDEX0..5')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([])
        if args == ['controls']: print(json.dumps(dict(entry=controls(), generation=generation_controls()), sort_keys=True))
        elif speed:
            index = int(args[1]) if args[1].isdigit() else None; ctx = generation_context(index)
            if args[1] == 'prepare': ctx.prepare()
            elif args[1] == 'host': ctx.comparison.host(ctx)
            else: generation_run(ctx, index)
        else:
            cap = int(args[0]); ctx = context(cap)
            {'prepare': lambda: prepare(ctx, cap), 'host': ctx.host, 'run': lambda: run(ctx, cap)}[args[1]]()
