#!/usr/bin/env python3
"""Matched long-prefill preservation between the two retained native engines.

Main-authored from this fork's owned bench.sh controls; repository LICENSE.
No native rebuild. Keep cold/first use, warmed PP, client TTFT and independent
decode selection separate. Main owns every serialized execution and cleanup.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import socket
import statistics
import sys

sys.dont_write_bytecode = True
import cache_map_model_entry as fallback
import verify_window_model_entry as candidate
import compare_prefill_strategies as routes

base, shared, comparison = candidate.base, candidate.template.shared, routes.comparison
P, B, RESULT = candidate.P, candidate.B, candidate.RESULT
ROOT, KEY = candidate.build.ROOT / 'prefill-preservation-v2', 'native_prefill_preservation_v2'
FAILED_KEY = 'native_prefill_preservation_v1'
FAILED_SOURCE = 'f87a1ff07b2ea763026d7febb19dd383e85f7c6c'
MTP_WINDOW = 32768
SELF = Path(__file__).resolve()
READS, RUNS = (65537, 117489), 4
PAIRS = (('reference', 'candidate'), ('candidate', 'reference'), ('reference', 'candidate'))
CASES = [(read, arm) for pair in PAIRS for read in READS for arm in pair]
ORDER = [(read, int(read == 117489)) for read, _ in CASES]
FRONTEND_CHANGED = {'tools/strata_bench.py', 'tools/long_generation_bench.py', 'tools/test_strata_bench.py'}
PRIOR_KEY = 'verify_window_generation_speed_v1'
PRIOR_ROOT = candidate.build.ROOT / 'generation-speed-v1'
require, digest, write = candidate.require, candidate.digest, candidate.write


def compact(row):
    value = candidate.template.compact(row)
    # Full per-request route records remain temporary controller output. The
    # retained shape plus every request's varying work/capture values suffice
    # for this question; do not repeat the identical byte geometry four times.
    value['processes'] = copy.deepcopy(row['processes'])
    varying = ('jobs', 'fused_down_jobs', 'requested_bank_H2D_bytes',
               'requested_one_routing_map_H2D_bytes', 'Q6_captures')
    for process in value['processes']:
        actual = process.pop('routes', [])
        if actual:
            process['route_shape'] = {k: v for k, v in actual[0].items() if k not in varying}
            process['request_route_checks'] = [{k: v for k, v in r.items() if k in varying} for r in actual]
            require(all({k: v for k, v in r.items() if k not in varying} == process['route_shape']
                        for r in actual), 'request route shape changed within one process')
    return value


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def pinned_sources(revision):
    values = candidate.pinned_sources(revision)
    values.update(base.controls(revision, controller=SELF))
    return values


def failed_basis():
    row = base.committed(RESULT)[FAILED_KEY]
    require(row['stage'] == 'failed' and not row['passed'] and not row['complete']
            and row['ownership_closed'] and not row['active'] and row['source_commit'] == FAILED_SOURCE
            and len(row['processes']) == 1 and row['failure_diagnosis']['preserved_failed_cohort']
            and not row['failure_diagnosis']['speed_eligible'], 'preserve the closed failed V1')
    return row['failure_diagnosis']


def basis(revision):
    old, _, old_providers = fallback.proof()
    new, _, providers = candidate.proof()
    evidence = base.committed(RESULT)
    failed_basis()
    prior = json.loads((PRIOR_ROOT / 'record.json').read_text())
    require(compact_prior(prior) == evidence[PRIOR_KEY] and prior['stage'] == 'closed'
            and all(prior[k] for k in ('passed', 'complete', 'ownership_closed', 'host_passed'))
            and not prior['active'] and len(prior['processes']) == 6
            and all(v['reproducible_TG_gain'] for v in prior['decision']['workloads'].values())
            and prior['binary'] == new['binaries']['strata'], 'closed repeated decode comparison required')
    clean = evidence['cache_map_prefill_speed117489_v1']
    require(clean['stage'] == 'closed' and clean['passed'] and clean['complete']
            and clean['ownership_closed'] and not clean['active']
            and clean['binary'] == old['binaries']['strata']
            and clean['native_source_commit'] == old['source_commit'], 'validated fastest prefill fallback required')
    require(old_providers == providers and old['boot'] == new['boot'] == prior['boot']
            and old['driver'] == new['driver'] == prior['driver']
            and old['native_validation_probe'] == prior['native_validation_probe'], 'native runtime/probe differs')
    phase = candidate.build.phase
    changed = set(phase.PHASE_FILES) | set(phase.Q6_FILES) | {phase.POLICY, phase.FIXTURE}
    actual = {n for n in old['native_sources'].keys() | new['native_sources'].keys()
              if old['native_sources'].get(n) != new['native_sources'].get(n)}
    require(actual == changed and len(changed) == 8, 'undeclared native difference')
    original = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                shared.gen.blobs(P, prior['source_commit'], prior['sources'])}
    current = pinned_sources(revision)
    require(original == prior['sources'] and set(current) - set(original) == {str(SELF.relative_to(P))}
            and not set(original) - set(current)
            and {n for n in original if current[n] != original[n]} == FRONTEND_CHANGED,
            'only the committed frontend sampling extension and new controller may differ')
    contract, manifest = candidate.template.python_manifest()
    require(contract == prior['python_contract'] and manifest['selected'] == prior['python_selected'],
            'qualified Python provider changed')
    return {'reference': old, 'candidate': new}, providers, manifest, prior


def compact_prior(row):
    return candidate.template.compact(row)


def native_context(row, plan):
    require(type(plan['index']) is int and 0 <= plan['index'] < len(CASES)
            and (plan['read'], plan['kernel']) == CASES[plan['index']]
            and (plan['read'], plan['heads']) == ORDER[plan['index']], 'unplanned native selection')
    return dict(row['arms'][plan['kernel']], native_validation_probe=row['native_validation_probe'], boot=row['boot'])


def environment(row, plan):
    native_context(row, plan)
    env = comparison.environment(row, plan)
    env.update(STRATA_SYCL_DECODE_Q6_COLUMNS=str(int(plan['kernel'] == 'candidate')),
               STRATA_POOL_SPIN_US='20000')
    require(all(env.get(k) == v for k, v in dict(STRATA_SYCL_PREFILL_GDN_BARRIERS3='1',
        STRATA_SYCL_PREFILL_GDN_FP16='1', STRATA_SYCL_PREFILL_GDN_CHUNK64='0',
        STRATA_MTP_BATCH_F16='1', STRATA_SYCL_ADAPT_SYNC='1', STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0',
        STRATA_PREFILL_LAYER_MAJOR='1', NEOReadDebugKeys='1', EnableDirectSubmission='0',
        UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1').items()), 'qualified route/safety differs')
    return env


def invariant(row, plan):
    config = json.loads((Path(plan['root']) / 'config.json').read_text())
    require(config['args'] == row['configuration']['args'], 'native arguments changed')
    env = environment(row, plan); env.pop('STRATA_SYCL_DECODE_Q6_COLUMNS')
    return hashlib.sha256(json.dumps(dict(config=config, environment=env), sort_keys=True)
                          .replace(plan['root'], '<process>').encode()).hexdigest()


def configuration(prior):
    config = copy.deepcopy(prior['configuration'])
    require('--mtp-window' not in config['args'], 'unexpected inherited MTP override')
    config['args'].extend(['--mtp-window', str(MTP_WINDOW)])
    return config


def request_route(text, plan):
    result = comparison.route(text, plan['read'], plan['heads'])
    require(result['residual_gpu_rows'] == plan['read'] - 1 and result['residual_host_bytes'] == 0,
            'full GPU residual route required in both engines')
    routes.native16_route(text, plan['read']); routes.gdn_selection(text, 'candidate')
    result['requested_one_routing_map_H2D_bytes'] = result.pop('requested_dequant_H2D_bytes')
    result['MTP_prefill'] = mtp_boundary_route(text, plan['read'])
    if plan['kernel'] == 'candidate':
        result['Q6_captures'] = candidate.captures(text, 'columns')
    result.update(prefill_strategy='throughput', GDN_FP16_storage=True, GDN_barriers=3,
                  physical_resource_saturation_qualified=False)
    return result


validate_model_text = routes.validate_model_text


def mtp_boundary_route(text, read):
    """Freeze the compiled 32768-window/sub64 admission, not all fallbacks.

    generate.cpp sets prompt_len to the full input and calls on_chunk in order.
    mtp.hpp first_needed is prompt_len-window-64. draft_kv skips wholly unneeded
    chunks, but deliberately returns false when 1..63 needed cells remain.
    Logged cells are nominal callback cells, not measured internal work.
    """
    require(type(read) is int and 32768 <= read <= 131072, 'unplanned MTP input')
    first_needed, expected, tails = read - MTP_WINDOW - 64, [], []
    for index, offset in enumerate(range(0, read - 1, 8192)):
        cells = min(8192, read - 1 - offset)
        needed = max(0, cells - max(0, first_needed - offset))
        token = 0 < needed < 64
        expected.append(('token' if token else 'batched', '2', str(cells)))
        if token: tails.append(dict(callback=index, source_derived_needed_cells=needed))
    lines = [v for v in text.splitlines() if v.startswith('strata serve: DRAFT_PREFILL')]
    actual = [re.fullmatch(r'strata serve: DRAFT_PREFILL path=(batched|token) mode=(\d+) cells=(\d+)', v) for v in lines]
    require(all(actual) and [v.groups() for v in actual] == expected,
            'MTP callback differs from the frozen window/sub64 boundary')
    return dict(placement_mode=2, callbacks=len(actual), nominal_cells=read-1, declared_window=MTP_WINDOW,
        source_derived_first_needed=first_needed, source_derived_token_boundaries=tails,
        internal_processed_rows_and_shapes_observed=False)


def decision(processes):
    require(len(processes) == len(CASES) and all(p['passed'] and p['complete']
        and p['ownership_closed'] and not p['fresh_faults'] and len(p['samples']) == RUNS
        and (p['read'], p['kernel']) == CASES[i] for i, p in enumerate(processes)), 'complete ordered cohort required')
    require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes})
            == len(CASES), 'native process reused')
    lengths = {}
    for read in READS:
        chosen = [p for p in processes if p['read'] == read]
        prompt, ttft = [], []
        for start in (0, 2, 4):
            pair = {p['kernel']: p for p in chosen[start:start+2]}
            c, r = pair['candidate'], pair['reference']
            prompt.append(c['subsequent_median_ms'] / r['subsequent_median_ms'])
            ttft.append(statistics.median(s['ttft_s'] for s in c['samples'][1:]) /
                        statistics.median(s['ttft_s'] for s in r['samples'][1:]))
        lengths[str(read)] = dict(candidate_over_reference_prompt_time=prompt,
            candidate_over_reference_TTFT=ttft, warmed_prompt_nonregression_evidence=all(v <= 1 for v in prompt),
            warmed_TTFT_nonregression_evidence=all(v <= 1 for v in ttft),
            consistent_prompt_regression=all(v > 1 for v in prompt), consistent_TTFT_regression=all(v > 1 for v in ttft),
            mixed_directions_require_further_evidence=True, minimum_percentage_gain_required=False,
            arms={arm: dict(first_use_pp=[read*1000/p['samples'][0]['prompt_ms'] for p in chosen if p['kernel'] == arm],
                subsequent_process_pp=[read*1000/p['subsequent_median_ms'] for p in chosen if p['kernel'] == arm],
                subsequent_process_TTFT=[statistics.median(s['ttft_s'] for s in p['samples'][1:]) for p in chosen if p['kernel'] == arm])
                for arm in ('reference', 'candidate')})
    return dict(lengths=lengths, inference_adopted=False, physical262144_qualified=False,
                physical_resource_saturation_qualified=False, TTFT_strategy_qualified=False,
                limits='Matched throughput-route PP/client-TTFT preservation only;not a sustained TG or task/agent/full-context/RESTORE test.')


def selfcheck():
    result = dict(routes=routes.selfcheck(), Q6_capture_parser=candidate.parser_check())
    # Independent explicit boundary cases, then replay the real failed request.
    for read, sizes, tokens in ((32769, [8192]*4, {}), (65536, [8192]*7+[8191], {}),
            (65537, [8192]*8, {3: 63}), (65538, [8192]*8+[1], {3: 62, 8: 1}),
            (117489, [8192]*14+[2800], {})):
        lines = [f"strata serve: DRAFT_PREFILL path={'token' if i in tokens else 'batched'} mode=2 cells={size}"
                 for i, size in enumerate(sizes)]
        good = '\n'.join(lines); parsed = mtp_boundary_route(good, read)
        require(parsed['source_derived_token_boundaries'] == [dict(callback=i, source_derived_needed_cells=n) for i,n in tokens.items()],
                'sub64 boundary calculation differs')
        for bad in ('\n'.join(lines[:-1]), good+'\n'+lines[-1], good.replace('mode=2', 'mode=0'),
                    good.replace('cells=8192', 'cells=8191'), good.replace('path=batched', 'path=token', 1)):
            try: mtp_boundary_route(bad, read)
            except RuntimeError: continue
            raise RuntimeError('unexpected MTP fallback/geometry accepted')
    failure = failed_basis(); fixture = failure['fixture']; raw = Path(fixture['path']).read_bytes()
    require(len(raw) == fixture['bytes'] <= 256 << 10 and hashlib.sha256(raw).hexdigest() == fixture['sha256'],
            'original failed parser fixture changed')
    text = raw.decode()
    headers = list(re.finditer(r'^strata serve: prompt (\d+) tokens = [^\n]+$', text, re.M))
    require([int(v.group(1)) for v in headers] == [74, 65537], 'one warmup then the failed long fixture required')
    # Live request gates exclude the warmup. Match that scope in offline replay.
    actual = request_route(text[headers[0].end():headers[1].end()], dict(read=65537, heads=0, kernel='reference'))
    require(actual['MTP_prefill']['source_derived_token_boundaries'] == [dict(callback=3, source_derived_needed_cells=63)],
            'real64K boundary replay differs')
    result['MTP_boundary'] = dict(explicit_shapes=5, invalid_variants=25, actual_failed_fixture_replayed=True,
        original_failure_preserved=True, fixture_sha256=fixture['sha256'],
        fixture_request_window=[headers[0].end(), headers[1].end()],
        original_replay_failure=dict(source='96e3ce6d5', exit_code=1,
            reason='Whole-log replay included73-cell warmup callback;live route gates are scoped to the long request.'))
    rows = [dict(read=read, kernel=arm, passed=True, complete=True, ownership_closed=True,
        fresh_faults=[], native_identity=dict(pid=i+1, start_ticks=i+10),
        subsequent_median_ms=99999 if arm == 'candidate' else 100000,
        samples=[dict(prompt_ms=110000, ttft_s=101 if arm == 'candidate' else 100) for _ in range(RUNS)])
        for i, (read, arm) in enumerate(CASES)]
    good = decision(rows)
    require(all(v['warmed_prompt_nonregression_evidence'] and not v['warmed_TTFT_nonregression_evidence']
                for v in good['lengths'].values()), 'PP and TTFT decisions coupled')
    bad = copy.deepcopy(rows); bad[1]['subsequent_median_ms'] = 100001
    require(not decision(bad)['lengths'][str(READS[0])]['warmed_prompt_nonregression_evidence'], 'slower pair erased')
    for invalid in (rows[:-1], [rows[0], rows[0], *rows[2:]]):
        try: decision(invalid)
        except RuntimeError: continue
        raise RuntimeError('incomplete or reused cohort accepted')
    result.update(passed=True, PP_TTFT_independent=True, small_changes_retained=True,
                  slower_pair_retained=True, invalid_cohorts_rejected=2, GPU_executed=False)
    return result


def prepare():
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'one fresh prospective cohort only')
    checked = selfcheck(); revision = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    natives, providers, manifest, prior = basis(revision)
    assets = shared.models.model_pins(prior['configuration'], prior['model_assets'], full=True)
    ROOT.mkdir(mode=0o700); write(ROOT / 'runtime-inputs.json', providers)
    env = dict(prior['environment_template']); env.pop('STRATA_TRACE', None)
    row = dict(stage='prepared', source_commit=revision, sources=pinned_sources(revision), model_assets=assets,
        arms={arm: dict(binary=n['binaries']['strata'], native_source_commit=n['source_commit']) for arm, n in natives.items()},
        native_validation_probe=prior['native_validation_probe'], configuration=configuration(prior),
        environment_template=env, python_contract=prior['python_contract'], python_selected=manifest['selected'],
        template=prior['template'], boot=prior['boot'], driver=prior['driver'],
        runtime_manifest=dict(path=str(ROOT/'runtime-inputs.json'), sha256=digest(ROOT/'runtime-inputs.json'), files=len(providers)),
        qualification_reuse=dict(native_rebuilt=False, native_changed_files=sorted(candidate.build.phase.PHASE_FILES + candidate.build.phase.Q6_FILES + (candidate.build.phase.POLICY, candidate.build.phase.FIXTURE)),
            prior_decode_key=PRIOR_KEY, frontend_changed=sorted(FRONTEND_CHANGED),
            preserved_failed_cohort=FAILED_KEY, explicit_MTP_window=MTP_WINDOW,
            scope='Exact two existing builds/native inputs/recipes/fixtures and qualified providers. Closed32K phase/Q6 entries plus repeated generation allow bounded longer prefill;all prefill source files are unchanged. New frontend profile does not select sampling for prefill-only.'),
        controls=checked, plans=[], processes=[], request_hashes={}, host_passed=False,
        passed=False, complete=False, active=False, ownership_closed=True, inference_adopted=False,
        whole_bytes=320 << 20, budgets=dict(wall=1800, cpu=21600, rss_cap=104 << 30,
            text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20),
        plan=dict(question='Does the decode-improved05a9/a99 engine preserve dedicated long-prefill PP and client TTFT against the validated479a/552b engine?',
            prior_failure='V1 remains failed at its first64K end gate. V2 checks ordered MTP callbacks against the unchanged source first_needed=read-window-64 and B<64 boundary. Only1..63needed-cell token callbacks are predicted;all other unexpected fallback/mode/size/count still fail. Native sources/ELFs/runtime safety and workload are unchanged;explicit --mtp-window32768 freezes the existing default in both arms.',
            order='12fresh processes:64K R/C then117K R/C;64K C/R then117K C/R;64K R/C then117K R/C. Commit each result before next. Four exact-count thinking-on fresh inputs per process;first use separate,median of subsequent3 is the process unit. First candidate64K must pass before candidate117K.',
            constants='Both throughput BF16 HC,GDN16/MTP16,originalDown,131072INT8resident context,1256static decode slots,5workers,spec4,8192chunk,short-read0,caches/adapt0,20ms poolspin. Head retirement0at64K and1at117K,selected manually in both arms. Poolspin20ms matches the current decode configuration;historical1047PP used its default,so historical PP is context rather than a matched arm.',
            differences='Only exact native generation and its immutable decode Q6 selector0reference/1candidate differ within a length. No process environment mutation;no third engine. Same current bench.sh/frontend and frozen body vector in both arms.',
            observers='No detailed trace/profiler/CPU or decode observers. Keep parameter validation and retained preload reference,immutable copy-offload/direct-submission avoidance,and actual route/capture headers.',
            acceptance='All48samples require exact actual input,zero reuse,thinking,1..24outputs,valid engine/client clocks,matching body/journal,full GPU residuals,ordinary384-slot64K or checked128-slot split117K,GDN3/native16/source-derived MTP callbacks,closed native/benchmark0,quiet final validation,unchanged source/providers/models/boot and no fresh fault. Candidate per-request Q6 decode captures must select columns;any prompt captures remain reference.',
            decisions='Retain all3paired warmed prompt-time and TTFT ratios separately per length,first-use and warmed process PP/TTFT and all samples. All ratios<=1 support nonregression for that metric;all>1 establish consistent slowdown;mixed directions stay unresolved. No minimum gain floor,no speed exclusion or retry. A short output does not qualify sustained TG or broader adoption.',
            stopping='Stop first source/provider/owner/body/clock/route/validation/fault/budget failure;no retries,replacements,hash refresh or relaxed criteria. Full15model hashes before/after cohort and quick identities throughout.',
            retention='Keep compact individual samples,one route shape per process with varying request work/captures,and source/config/limits. Retire successful raw output after committed evidence/source recovery. Keep original closed generation record/runtime/plans only until this qualification consumer ends. No statistical saturation,task/agent,prompt-verify,physical262144/RESTORE or ordinary-runtime-teardown claim.'))
    for index, (read, arm) in enumerate(CASES):
        root = ROOT/str(index); root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root/name).mkdir()
        config = copy.deepcopy(row['configuration']); config['log'] = str(root/'engine.stderr')
        write(root/'config.json', config); write(root/'plan.json', dict(binary=row['arms'][arm]['binary'], native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        plan = dict(index=index, read=read, heads=ORDER[index][1], kernel=arm, root=str(root),
            generated={n:digest(root/n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P/'bench.sh'), '--config', str(root/'config.json'), '--port', str(port),
                '--prefill-exact-tokens', str(read), '--think', '--runs', str(RUNS), '--profile-request-gates',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root/'bench')])
        plan['environment_sha256'] = comparison.fingerprint(environment(row, plan)); row['plans'].append(plan)
    row['per_length_invariant'] = {str(read): invariant(row, next(p for p in row['plans'] if p['read'] == read)) for read in READS}
    save(row); print('Prepared matched retained-engine64K/117K prefill preservation;commit before host probes.', flush=True)


def current(row):
    require(base.committed(RESULT)[KEY] == compact(row), 'commit exact prefill comparison state')
    natives, providers, manifest, prior = basis(row['source_commit'])
    require(row['arms'] == {arm: dict(binary=n['binaries']['strata'], native_source_commit=n['source_commit']) for arm, n in natives.items()}
            and pinned_sources(row['source_commit']) == row['sources'] and row['model_assets'] == prior['model_assets']
            and row['configuration'] == configuration(prior), 'frozen native/frontend/model selection changed')
    inv = row['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'] and json.loads(Path(inv['path']).read_text()) == providers
            and len(providers) == inv['files'], 'runtime inventory changed')
    require(digest(row['template']['path']) == row['template']['sha256'] and base.guards.driver_guard() == row['driver'], 'tokenizer/platform differs')
    require(all(invariant(row, p) == row['per_length_invariant'][str(p['read'])] for p in row['plans']), 'arm configuration differs')
    require(base.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole output budget exceeded')
    shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    return providers, manifest


if __name__ == '__main__':
    require(len(sys.argv) == 2 and (sys.argv[1] in ('selfcheck', 'prepare', 'host') or
            sys.argv[1].isdigit() and int(sys.argv[1]) < len(CASES)), 'use selfcheck|prepare|host|INDEX0..11')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([])
        if sys.argv[1] == 'selfcheck': print(json.dumps(selfcheck()), flush=True)
        elif sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] == 'host': comparison.host(sys.modules[__name__])
        else: comparison.run(int(sys.argv[1]), sys.modules[__name__])
