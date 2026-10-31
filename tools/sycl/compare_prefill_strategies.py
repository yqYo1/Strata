#!/usr/bin/env python3
"""Matched retained-official versus same-binary-policy/GDN3 long prefill.

Main-authored from this fork's owned benchmark controls; repository LICENSE.
The executable generations differ. This measures the resulting production
candidate, not an isolated attribution of every difference to GDN barriers.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import statistics
import sys

sys.dont_write_bytecode = True
import check_prefill_strategy_model as model
import prefill_resources as original

comparison, shared, build = model.comparison, model.shared, model.build
P, B, RESULT = model.P, model.B, model.RESULT
ROOT, KEY = B / 'official-prefill-strategy-speed-v1', 'official_prefill_strategy_speed_v1'
READS, RUNS = (65537, 117489), 4
ARMS = ('candidate', 'reference', 'reference', 'candidate', 'candidate', 'reference')
CASES = [(read, arm) for pair in range(3) for read in READS for arm in ARMS[2*pair:2*pair+2]]
ORDER = [(read, int(read == 117489)) for read, _ in CASES]
CHANGED, ADDED = {'tools/sycl/compare_official_prefill.py'}, {'tools/sycl/compare_prefill_strategies.py'}
require, digest, write = model.require, model.digest, model.write
compact = model.compact


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row); write(RESULT, data)


def proof(revision):
    candidate, _, providers = model.proof()
    baseline, old_providers = original.native_reuse(revision)
    require(providers == old_providers and candidate['boot'] == baseline['boot'] and
            candidate['driver'] == baseline['driver'], 'matched native runtime/platform required')
    data = build.committed(RESULT)
    require(data['official_row64_model_v2']['passed'] and data['official_row64_model_v2']['complete']
            and data['official_row64_model_v2']['ownership_closed'] and
            not data['official_row64_model_v2']['active'], 'closed original model reference required')
    qualified = {}
    for strategy in ('ttft', 'throughput'):
        key = 'official_prefill_strategy_' + strategy + '_model_v1'
        row = json.loads((model.candidate.ROOT / ('model-' + strategy + '-v1') / 'record.json').read_text())
        require(compact(row) == data[key] and row['passed'] and row['complete'] and row['host_passed']
                and row['ownership_closed'] and not row['active'] and row['binary'] == candidate['binaries']['strata']
                and row['native_source_commit'] == candidate['source_commit']
                and row['outcome']['native_exit_code'] == 0 and not row['outcome']['fresh_faults']
                and len(row['outcome']['routes']) == len(row['outcome']['samples']) == 2,
                'same-generation two-strategy model qualification required')
        recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                     shared.gen.blobs(P, row['source_commit'], row['sources'])}
        require(recovered == row['sources'], 'original model controller recovery failed')
        current = model.pinned_sources(revision)
        require(set(current) - set(recovered) == ADDED and not (set(recovered) - set(current))
                and {n for n in recovered if recovered[n] != current[n]} == CHANGED,
                'undeclared model/controller change')
        qualified[strategy] = row
    return candidate, baseline, providers, qualified


def native_context(row, plan):
    require((plan['read'], plan['kernel']) == CASES[plan['index']] and
            plan['heads'] == int(plan['read'] == 117489), 'unplanned process selection')
    arm = row['arms'][plan['kernel']]
    return dict(binary=arm['binary'], native_source_commit=arm['native_source_commit'],
                native_validation_probe=row['native_validation_probe'], boot=row['boot'])


def environment(row, plan):
    native_context(row, plan)
    env = comparison.environment(row, plan)
    env['STRATA_SYCL_PREFILL_GDN_BARRIERS3'] = str(int(plan['kernel'] == 'candidate'))
    require(env['STRATA_PREFILL_LAYER_MAJOR'] == '1' and
            env['STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL'] == 'original' and
            env['STRATA_SYCL_PREFILL_GDN_COLUMNS'] == '4' and
            env['STRATA_SYCL_PREFILL_GDN_CHUNK64'] == '0', 'prefill selection differs')
    require(not env.get('LD_PRELOAD') and not env.get('STRATA_TRACE') and
            env['CUDA_MODULE_LOADING'] == 'EAGER', 'private native validation/clean observer environment required')
    return env


def gdn_selection(text, arm):
    require(arm in ('candidate', 'reference'), 'unknown executable arm')
    expected = ('strata prefill: GDN columns=4 workgroups=192 lanes=128'
                + (' barriers=3' if arm == 'candidate' else '')
                + ' epilogue=GPU_FP32_same_order scratch=existing_y')
    choices = [v for v in text.splitlines() if v.startswith('strata prefill: GDN columns=')]
    require(choices == [expected], 'actual generation-specific GDN selection differs')


def request_route(text, plan):
    result = comparison.route(text, plan['read'], plan['heads'])
    gdn_selection(text, plan['kernel'])
    require(result['residual_gpu_rows'] == plan['read'] - 1
            and result['residual_host_bytes'] == 0, 'actual GDN/full-residency route differs')
    # The legacy parser's field counts one routing-index map, not weights.
    result['requested_one_routing_map_H2D_bytes'] = result.pop('requested_dequant_H2D_bytes')
    result.update(executable_arm=plan['kernel'], GDN_barriers=3 if plan['kernel'] == 'candidate' else 5,
                  prefill_strategy='throughput' if plan['kernel'] == 'candidate' else 'legacy-layer-major',
                  dispatch_proof='Per-request production header plus recovered native source/recipe and immutable selector;no per-launch count claimed.')
    return result


def validate_model_text(text, plan):
    require(re.findall(r'^\[native16-down\] fusion=large min_rows=2334 split_kernel=(\w+)$', text, re.M)
            == ['original'], 'actual immutable production Down selection differs')


def decision(processes):
    require(len(processes) == len(CASES) and all(p['passed'] and p['complete'] and
            p['ownership_closed'] and not p['fresh_faults'] and
            (p['read'], p['kernel']) == CASES[i] for i, p in enumerate(processes)), 'complete frozen cohort required')
    require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes})
            == len(CASES), 'native process reused')
    lengths = {}
    for read in READS:
        selected = [p for p in processes if p['read'] == read]
        ratios, ttft, speeds = [], [], []
        for first in (0, 2, 4):
            pair = {p['kernel']: p for p in selected[first:first+2]}
            a, b = pair['candidate'], pair['reference']
            require(all(math.isfinite(p['subsequent_median_ms']) and p['subsequent_median_ms'] > 10000
                        for p in pair.values()), 'invalid process prompt clock')
            ratios.append(a['subsequent_median_ms'] / b['subsequent_median_ms'])
            ttft.append(statistics.median(s['ttft_s'] for s in a['samples'][1:]) /
                        statistics.median(s['ttft_s'] for s in b['samples'][1:]))
            speeds.append(read * 1000 / a['subsequent_median_ms'])
        gm = math.exp(statistics.mean(math.log(v) for v in ratios))
        lengths[str(read)] = dict(candidate_over_reference_prompt_time=ratios,
            geometric_mean_time_ratio=gm, reproducible_prompt_gain=all(v < 1 for v in ratios) and gm < 1,
            minimum_percentage_gain_required=False, candidate_over_reference_TTFT=ttft,
            candidate_subsequent_process_pp=speeds,
            reference_subsequent_process_pp=[read * 1000 / p['subsequent_median_ms'] for p in selected if p['kernel'] == 'reference'],
            all_candidate_processes_at_least1000=all(v >= 1000 for v in speeds))
    return dict(lengths=lengths, inference_adopted=False, sustained_decode_qualified=False,
                physical262144_qualified=False, physical_resource_saturation_qualified=False)


def selfcheck():
    result = dict(route=comparison.selfcheck(), finalization=original.finalization_controls())
    base = 'strata prefill: GDN columns=4 workgroups=192 lanes=128'
    tail = ' epilogue=GPU_FP32_same_order scratch=existing_y'
    for arm, header in [('reference', base+tail), ('candidate', base+' barriers=3'+tail)]:
        gdn_selection(header, arm)
        for wrong in ('', header+'\n'+header, header.replace('columns=4', 'columns=2'),
                      base+' barriers=5'+tail, base+(' barriers=3' if arm == 'reference' else '')+tail):
            try: gdn_selection(wrong, arm)
            except RuntimeError: pass
            else: raise RuntimeError('invalid generation-specific GDN route accepted')
    result['GDN_selection'] = dict(generation_specific_headers_accepted=2, invalid_or_wrong_generation_rejected=10)
    def process(i):
        read, arm = CASES[i]
        return dict(read=read, kernel=arm, passed=True, complete=True, ownership_closed=True,
            fresh_faults=[], native_identity=dict(pid=i+1, start_ticks=i+100),
            subsequent_median_ms=119880 if arm == 'candidate' else 120000,
            samples=[dict(ttft_s=121.1 if arm == 'candidate' else 121.0) for _ in range(RUNS)])
    rows = [process(i) for i in range(len(CASES))]; value = decision(rows)
    require(all(v['reproducible_prompt_gain'] and all(t > 1 for t in v['candidate_over_reference_TTFT'])
                for v in value['lengths'].values()), 'small PP gain or separate TTFT rejected')
    rows[0]['subsequent_median_ms'] = 120001
    require(not decision(rows)['lengths'][str(READS[0])]['reproducible_prompt_gain'], 'slower pair accepted')
    rows[0] = process(0); rows[1]['native_identity'] = rows[0]['native_identity']
    try: decision(rows)
    except RuntimeError: result['selection'] = dict(small_gain_kept=True, TTFT_independent=True, slower_pair_rejected=True, duplicate_native_rejected=True)
    else: raise RuntimeError('duplicate native identity accepted')
    return result


def prepare():
    checks = selfcheck(); revision = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    candidate, baseline, providers, qualified = proof(revision)
    require(not ROOT.exists() and KEY not in build.committed(RESULT), 'fresh comparison only')
    prior = qualified['throughput']; contract, manifest = model.python_manifest()
    require(contract == prior['python_contract'] and manifest['selected'] == prior['python_selected'], 'frontend provider differs')
    assets = shared.models.model_pins(prior['configuration'], prior['model_assets'], full=True)
    ROOT.mkdir(mode=0o700); write(ROOT / 'runtime-inputs.json', providers)
    env = dict(prior['environment']); env.pop('STRATA_TRACE', None)
    row = dict(stage='prepared', source_commit=revision, sources=model.pinned_sources(revision), model_assets=assets,
        arms={name:dict(binary=native['binaries']['strata'], native_source_commit=native['source_commit'])
              for name, native in (('candidate', candidate), ('reference', baseline))},
        native_validation_probe=prior['native_validation_probe'], configuration=prior['configuration'],
        environment_template=env, python_contract=contract, python_selected=manifest['selected'],
        template=prior['template'], boot=prior['boot'], driver=prior['driver'],
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'), sha256=digest(ROOT / 'runtime-inputs.json'), files=len(providers)),
        qualification_reuse=dict(component_and_both_model_routes=True, changed_controls=sorted(CHANGED),
            added_controls=sorted(ADDED), native_recompiled=False, current_source_and_recipe_checked=True),
        plans=[], processes=[], request_hashes={}, controls=checks, host_passed=False,
        passed=False, complete=False, active=False, ownership_closed=True, inference_adopted=False,
        whole_bytes=800 << 20, budgets=dict(wall=1800, cpu=5400, rss_cap=104 << 30, text_cap=16 << 20,
            file_cap=16 << 20, total_cap=64 << 20),
        plan=dict(question='Does the qualified same-binary-policy/GDN3 candidate improve long prefill and client TTFT over the retained official engine?',
            order='12fresh processes:3paired comparisons per64K/117K length,interleaved lengths;candidate/reference,reference/candidate,candidate/reference. Four exact-input thinking-on fresh requests each;first use separate,median of subsequent3 is the process unit.',
            placement='Same per-length config in both arms:head retirement0at64K,1at117K,chosen from closed official_joint_speed_v1. Manual benchmark settings,not an automatic threshold. Full GPU BF16 residuals,384staged slots/ordinary64K,128staged slots/checked split117K. Same1256decode slots,5workers,spec4,8192chunk,131072INT8resident context,Top-k0,originalDown,qualified immutable runtime safety.',
            differences='Two exact retained ELF generations,policy selector and3-versus5GDN barriers. Candidate explicit CLIthroughput;reference legacy layer-major startup selection. No clean model speed inherited from component or diagnostic results;not an isolated causal GDN attribution.',
            acceptance='Every request must satisfy all existing sample,body/actual tokens/reuse,thinking,clock,owner,provider,quiet-final-validation,native-exit and fault checks. Actual per-request GDN schedule,full-GPU placement,bank/split drain/guard and head-retirement choice must match. Keep all48individual samples. All3paired warmed prompt-time ratios<1 and geometric mean<1 allow follow-up at that length,no minimum percentage. TTFT judged separately.1000target separately requires all3candidate process medians>=1000at a length>=64K. No sustained TG/adoption from24outputs.',
            stopping='Stop first source,provider,owner,clock,route,validation,fault,budget or sample failure. No speed-based stop,rerun,replacement,exclusion or hash refresh. Commit each process before the next. Full model hashes before/after cohort,identities throughout.',
            limits='Neither strategy globally wins without length/latency measurements. No automatic length selection,sustained decode,agent/task,physical262144/RESTORE qualification or resource saturation claim. Both useful profiles remain available in candidate ELF.',
            retention='Compact individual samples,route conservation,source/config and decisions. Retire verified successful raw logs,journals,JIT and ownership output at closure;keep both retained generations for this named comparison and remaining phase gates.'))
    for index, (read, arm) in enumerate(CASES):
        root = ROOT / str(index); root.mkdir(mode=0o700)
        for part in ('tmp', 'cache'): (root / part).mkdir()
        config = copy.deepcopy(row['configuration']); config['log'] = str(root / 'engine.stderr')
        args = config['args']; require(args[-2:] == ['--prefill-strategy', 'throughput'], 'qualified explicit strategy missing')
        if arm == 'reference': config['args'] = args[:-2]
        native = row['arms'][arm]
        write(root / 'config.json', config); write(root / 'plan.json', dict(binary=native['binary'], native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        plan = dict(index=index, kernel=arm, read=read, heads=ORDER[index][1], root=str(root),
            generated={n:digest(root / n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--prefill-exact-tokens', str(read), '--think', '--runs', str(RUNS), '--profile-request-gates',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')])
        plan['environment_sha256'] = comparison.fingerprint(environment(row, plan)); row['plans'].append(plan)
    save(row); print('Prepared12matched two-generation64K/117K processes;commit before host probes.', flush=True)


def current(row):
    require(build.committed(RESULT)[KEY] == compact(row), 'commit exact campaign state')
    candidate, baseline, providers, _ = proof(row['source_commit'])
    require(row['arms'] == {name:dict(binary=native['binaries']['strata'], native_source_commit=native['source_commit'])
            for name, native in (('candidate', candidate), ('reference', baseline))}, 'native generation changed')
    require(model.pinned_sources(row['source_commit']) == row['sources'], 'frozen model/controller source changed')
    inv = row['runtime_manifest']; require(digest(inv['path']) == inv['sha256'] and
        json.loads(Path(inv['path']).read_text()) == providers and len(providers) == inv['files'], 'provider inventory changed')
    require(digest(row['template']['path']) == row['template']['sha256'] and
            build.guards.driver_guard() == row['driver'], 'template/platform changed')
    require(build.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole output budget')
    shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    contract, manifest = model.python_manifest(); require(contract == row['python_contract'] and
        manifest['selected'] == row['python_selected'], 'frontend provider changed')
    return providers, manifest


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); build.guards.no_model_owners([])
        args = sys.argv[1:]
        if args == ['selfcheck']: print(json.dumps(selfcheck()))
        elif args == ['prepare']: prepare()
        elif args == ['host']: comparison.host(sys.modules[__name__])
        elif len(args) == 1 and args[0].isdigit() and int(args[0]) < len(CASES): comparison.run(int(args[0]), sys.modules[__name__])
        else: raise RuntimeError('compare_prefill_strategies.py selfcheck|prepare|host|INDEX0..11')
