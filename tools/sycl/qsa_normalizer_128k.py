#!/usr/bin/env python3
"""One-variable long-prefill QSA denominator comparison; repository LICENSE.

Uses the retained native14508 production engine. The workspace admission
reader is derived from this fork's closed 4d027b3e1 bank comparison.
"""
import copy
import fcntl
import json
from pathlib import Path
import re
import socket
import statistics
import sys

sys.dont_write_bytecode = True
import coherent_model_basis as capacity
import compare_official_prefill as comparison
import gu_small_bench as dispatch_reader
import qsa16_stream_bench as core
import ur_validation as validation

P, B, RESULT = capacity.P, capacity.B, capacity.RESULT
ROOT, KEY = B/'qsa-normalizer-128k-v1', 'qsa_normalizer_128k_v1'
COMPONENT = 'qsa_prefix_calibration_v2'
COMPONENT_ARCHIVE = '2da1e86236865ebcb3340b7d218cb9e51f027019'
WORKLOAD_ARCHIVE = '4d027b3e16fb546010e82e4f1d006fc4a9ceaac7'
PYTHON = capacity.reference.model.PYTHON
require, sha, signature = capacity.require, capacity.sha, capacity.signature
READ, RUNS, BANK, STRIDE, ROW = 131072, 3, 256, 2662400, 20480
READS = (32769,) + (READ,) * 8
ARMS = (1, 0, 1, 0, 1, 1, 0, 0, 1)
ORDER = [(read, 1) for read in READS]
SELECTORS = ('bf16-prefix-ordered-exp', 'bf16-prefix-cached-subgroup')
SOFTMAX = ('parallel-exp-ascending-sum', 'subgroup')


def component_gate(binary):
    row = validation.committed(RESULT, (COMPONENT,))
    archived = json.loads(validation.git('show', f'{COMPONENT_ARCHIVE}:{RESULT.relative_to(P)}'))[COMPONENT]
    assessment = row['decision']
    same_build = row['original_native_proof']['same_build_executables']
    require(row['stage'] == 'closed' and not row['active']
        and all(row[k] for k in ('passed', 'complete', 'ownership_closed'))
        and same_build['strata'] == binary
        and row['binary'] == same_build['sycl_qsa_bf16_prefix_cached_subgroup_bench']
        and row['native_source_commit'] == capacity.original.SOURCE
        and assessment == archived['decision'] and assessment['eligible_for_new_128K_plan']
        and [s['repeated_gain'] for s in assessment['shapes']] == [False, True, True],
        'closed exact same-build denominator component evidence required')
    return dict(commit=COMPONENT_ARCHIVE, key=COMPONENT, decision_sha256=signature(assessment))


def save(row):
    core.write(ROOT/'record.json', row)
    values = json.loads(RESULT.read_text())
    values[KEY] = core.compact(row)
    core.write(RESULT, values)


def environment(row, plan):
    env = comparison.environment(row, plan)
    env.update(STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS=str(BANK),
        STRATA_SYCL_PREFILL_MOE_WINDOW='10', STRATA_SYCL_PREFILL_MOE_CENSUS='0',
        STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',
        STRATA_SYCL_PREFILL_QSA_MATH=SELECTORS[plan['arm']])
    if plan['trace_progress']:
        env['STRATA_TRACE']='1'
    required = dict(STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING='0',
        STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED='1', STRATA_SYCL_CACHE_SCRATCH_LEASE='1',
        STRATA_SYCL_PREFILL_MIXED_SCRATCH='1', STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR='1',
        STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR='1', STRATA_SYCL_PREFILL_EXPERT_SOURCE_VIEW='1',
        STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL='original', NEOReadDebugKeys='1')
    require(all(env[k] == v for k, v in required.items()), 'bank, queue, upload or safety drift')
    return env


def command(plan):
    root = Path(plan['root'])
    return ['/bin/sh', str(P/'bench.sh'), '--config', str(root/'config.json'),
        '--port', str(plan['port']), '--prefill-exact-tokens', str(plan['read']), '--think',
        '--runs', str(RUNS), '--profile-request-gates', '--workload-id', 'native_gdn_20261018',
        '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root/'bench')]


def normalized_environment(row, plan):
    env = environment(row, plan)
    env.pop('STRATA_TRACE', None)
    env['STRATA_SYCL_PREFILL_QSA_MATH'] = '<denominator>'
    return comparison.fingerprint({k: v.replace(plan['root'], '<process>') for k, v in env.items()})


BASIS_FIELDS = ('configuration', 'environment_template', 'model_assets', 'template',
                'boot', 'driver', 'native_validation_probe', 'python_contract',
                'python_selected', 'runtime_manifest')


def pins(revision):
    sources = core.pins(revision)
    sources.update(comparison.build.controls(revision, controller=Path(__file__).resolve()))
    test = 'tools/test_qsa_normalizer_128k.py'
    require((P/test).read_bytes() == validation.git('show', f'{revision}:{test}'),
        'unit checks changed')
    sources[test] = sha(P/test)
    return sources


def request_hashes():
    raw = validation.git('show', f'{WORKLOAD_ARCHIVE}:{RESULT.relative_to(P)}')
    prior = json.loads(raw)['prefill_bank_minimum_v2']
    hashes = prior['request_hashes']
    require(prior['stage'] == 'closed' and prior['complete']
        and set(hashes) == {str(read) for read in READS}
        and all(len(set(group)) == len(group) == RUNS for group in hashes.values()),
        'recover the exact previously generated request bodies')
    return hashes


def current(row):
    require(validation.committed(RESULT, (KEY,)) == core.compact(row)
        and pins(row['source_commit']) == row['sources']
        and row['request_hashes'] == request_hashes()
        and all(sha(P/p) == h for p, h in row['policies'].items()),
        'commit and freeze exact measurement inputs')
    native, _, prior = capacity.basis(row['original_native_proof'])
    require(row['binary'] == native['binaries']['strata']
        and row['native_source_commit'] == capacity.original.SOURCE
        and row['component_gate'] == component_gate(row['binary'])
        and all(row[k] == prior[k] for k in BASIS_FIELDS),
        'retained native image, predecessor, decode settings or runtime changed')
    contract, python = core.frontend.python_manifest()
    require(contract == row['python_contract'] and python['selected'] == row['python_selected']
        and sha(row['template']['path']) == row['template']['sha256']
        and comparison.build.supervision.artifact_bytes(ROOT) <= row['whole_bytes']
        and len(row['plans']) == len(ORDER), 'frontend/template/output budget drift')
    for i, plan in enumerate(row['plans']):
        root = ROOT/str(i)
        require((plan['index'], plan['read'], plan['heads'], plan['arm'], plan['trace_progress'], plan['root'])
            == (i, READS[i], 1, ARMS[i], i < 3, str(root))
            and plan['command'] == command(plan)
            and normalized_environment(row, plan) == row['invariant_sha256']
            and json.loads((root/'config.json').read_text()) == dict(row['configuration'], log=str(root/'engine.stderr'))
            and json.loads((root/'plan.json').read_text()) == dict(binary=row['binary'], native_validation_probe=row['native_validation_probe'])
            and set(plan['generated']) == {'config.json', 'plan.json'}
            and all(sha(root/name) == pin for name, pin in plan['generated'].items())
            and comparison.fingerprint(environment(row, plan)) == plan['environment_sha256'],
            'unplanned arm or generated input')
    return json.loads(Path(row['runtime_manifest']['path']).read_text()), python


def records(text,prefix):
    found=[]
    for line in text.splitlines():
        if line.startswith(prefix):
            pairs=re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)',line.split(';',1)[0])
            require(len({k for k,v in pairs})==len(pairs),'duplicate '+prefix)
            found.append(dict(pairs))
    return found


def one(text,prefix):
    values=records(text,prefix)
    require(len(values)==1,'missing/duplicate '+prefix)
    return values[0]


def placement_route(text,plan):
    n=plan['read']-1
    require(plan['read'] in (32769,READ), 'undeclared placement case')
    main=one(text,'strata prefill: layer-major candidate, ')
    bank=one(text,'strata MoE expert bank: ')
    admission=one(text,'strata MoE accumulator admission: ')
    split=one(text,'strata MoE split admission: ')
    window=one(text,'strata MoE window configuration: ')
    work=one(text,'strata MoE accumulator window: ')
    source=one(text,'strata prefill expert source: ')
    mixed=one(text,'strata MoE mixed banks: ')
    gpu = int(main['residual_gpu_rows'])
    require(gpu==n and main['tokens']==str(n)
        and main['chunk']=='8192' and main['residual_dtype']=='bf16' and main['residual_element_bytes']=='2'
        and int(main['residual_total'])==n*ROW and int(main['residual_gpu'])==gpu*ROW
        and int(main['residual_host'])==(n-gpu)*ROW and main['residual_staging']=='0',
        'actual residual placement/byte conservation differs;no automatic recalibration')
    cache = BANK * STRIDE
    limit = int(bank['allocation_limit_bytes'])
    require(bank['requested_slots']==str(BANK) and int(bank['effective_slots'])==BANK
        and bank['staged']=='1' and int(bank['host_slab_bytes'])==512*STRIDE
        and int(bank['device_bytes'])==int(main['layer_cache'])==cache
        and int(bank['residual_gpu_rows'])==gpu and bank['general_map']=='private_scratch_never_published'
        and max(cache,gpu*ROW)<=limit,'bank/allocation/private-map contract differs')
    require(split['active']==split['primary_rows']==split['tail_payload_bytes']==split['extra_allocation_bytes']=='0'
        and split['rows']==str(n) and split['prefill_slots']==str(BANK) and split['decode_changed']=='0'
        and int(split['max_single_allocation_bytes'])==limit and not records(text,'strata MoE split complete: '),
        'unchanged startup split request must actually remain inactive')
    maximum = min(81920, gpu)
    windows = 48 * ((gpu + 81919) // 81920)
    require(window['requested_mode']==window['effective_mode']=='10' and window['mixed_fp16_bytes']=='0'
        and int(window['maximum_rows'])==int(work['maximum_rows'])==maximum
        and int(work['windows'])==windows and work['operator_max']=='8192'
        and work['GPU_resident_windows_only']=='1' and int(work['jobs'])>0 and int(work['dequants'])>0,
        'actual window extent/work differs')
    require(mixed['storage']=='scratch' and mixed['extra_allocation_bytes']=='0'
        and int(mixed['primary_bytes'])==min(65536,maximum)*5120
        and int(mixed['tail_bytes'])==max(0,maximum-65536)*5120
        and int(mixed['tail_offset'])>=0 and mixed['phase']=='FFN_only_after_attention_drain',
        'checked borrowed mixed storage differs')
    # Recompute native14508's automatic bank/R admission from its observed
    # free budget. Prefix cells round to16; two BF16 tensors plus INT32 flags.
    prefix = ((n + 15) // 16) * 16 * 2056 + 256
    headroom = 536870912
    free = int(admission['free_gpu_bytes'])
    fixed = cache + prefix + headroom + gpu * ROW
    require(fixed <= free and max(prefix, cache, gpu * ROW) <= limit
        and admission['requested_mode']=='10' and int(admission['rows'])==maximum
        and int(admission['gpu_prefix_rows'])==gpu and int(admission['shared_arena_bytes'])>0
        and int(admission['mixed_fp16_bytes'])==maximum*5120
        and int(admission['fixed_including_headroom_bytes'])==fixed,
        'observed native budget does not reproduce planned placement')
    require(source['requested_view']==source['effective_view']=='1'
        and source['layers']=='48' and source['owned_slab_bytes']=='0'
        and int(source['borrowed_layer_span_bytes'])==50292326400
        and source['host_copy']=='eliminated'
        and source['source_owner']=='caller_through_checked_retirement',
        'immutable expert source placement differs')
    end=one(text,'strata MoE expert bank complete: ')
    copied,hits,uploads=(int(end[k]) for k in ('requested_H2D_bytes','permanent_hits','uploads'))
    require(end['final_drain']=='checked' and int(end['total_slots'])>=BANK
        and 0<=int(end['permanent_slots'])<=int(end['total_slots'])<=512
        and min(copied,hits,uploads)>=0 and uploads+hits==int(work['dequants'])
        and int(work['source_H2D_bytes'])==48*n*40, 'staged upload/drain/map conservation differs')
    require(text.index('strata prefill heads: target detached ')<text.index('strata prefill heads: draft detached ')
        <text.index('strata MoE expert bank: ')<text.index('strata prefill: layer-major candidate, '),
        'head retirement/admission ordering differs')
    if plan['trace_progress']:
        geometry=records(text,'strata packed expert bank: ')
        require(len(geometry)==48,'packed view count differs')
        for layer,g in enumerate(geometry):
            stride = int(g['stride'])
            require(stride > 0 and stride % 256 == 0, 'invalid packed stride')
            count = min(512, cache // stride)
            require(int(g['layer'])==layer
                and int(g['slots'])==count and int(g['used_bytes'])==count*stride<=cache
                and int(g['backing_bytes'])==cache and g['full_fit']==str(int(count==512)),
                'actual per-layer packed geometry differs')
    return dict(jobs=int(work['jobs']),dequants=int(work['dequants']),rows=n,residual_gpu_rows=gpu,residual_host_rows=n-gpu,prefill_slots=BANK,staged=True,
        layer_cache_bytes=cache,windows=windows,maximum_window_rows=maximum,
        requested_bank_H2D_bytes=copied,permanent_hits=hits,uploads=uploads,
        immutable_all_expert_layer_bytes=int(source['borrowed_layer_span_bytes']),
        requested_source_map_H2D_bytes=int(work['source_H2D_bytes']),free_gpu_bytes=free,
        fixed_including_headroom_bytes=int(admission['fixed_including_headroom_bytes']),
        allocation_limit_bytes=limit,budget_reproduced=True,actual_split=False,
        actual_expert_source_view=True,actual_packed_layers=48 if plan['trace_progress'] else None,
        physical_bus_or_saturation_qualified=False)


def request_route(text, plan):
    route = core.request_route(text, plan)
    values = re.findall(
        r'^strata prefill: QSA BF16 operands, FP32 online state/accumulation; '
        r'KV=active-logical-prefix transient_bytes=([0-9]+); source=resident-Mode0 '
        r'score_pitch=33 operand_layout=row-major softmax=(\S+)$', text, re.M)
    require(len(values) == 1 and int(values[0][0]) > 0 and values[0][1] == SOFTMAX[plan['arm']],
        'actual attention arithmetic, storage or layout differs')
    phases = re.findall(r'^strata mtp: prompt expert phase=(\S+) ', text, re.M)
    require(phases == ['before-detach', 'after-detach', 'after-prompt-before-restore', 'after-restore']
        and re.findall(r'^strata mtp: experts restored bytes=(\d+),', text, re.M) == ['707788800'],
        'actual MTP handoff differs')
    return dict(route, bank=placement_route(text, plan), actual_attention_math=SELECTORS[plan['arm']],
        actual_GU_dispatch=dispatch_reader.dispatch(text, 'large'))


def check_sample(sample, count, read, minimum_ms=10000):
    comparison.check_sample(sample, count, read, minimum_ms)
    require(sample['generated_tokens'] == 24, '24-output prefill diagnostic required')


def decision(processes):
    require(len(processes) == len(ORDER) and all(p['passed'] and p['complete']
        and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
        and p['index'] == i and p['read'] == READS[i] and p['clean_speed'] == (i >= 3)
        and len(p['samples']) == len(p['routes']) == RUNS for i, p in enumerate(processes)),
        'all ordered closed processes required')
    identity = {(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}
    require(len(identity) == len(ORDER), 'independent processes required')
    comparisons = []
    for i in range(3, len(ORDER), 2):
        base, candidate = sorted(processes[i:i+2], key=lambda p: ARMS[p['index']])
        work = []
        for a, b in zip(base['routes'], candidate['routes']):
            keys = ('jobs', 'dequants', 'uploads', 'permanent_hits',
                    'requested_source_map_H2D_bytes', 'requested_bank_H2D_bytes')
            differences = {key: b['bank'][key] - a['bank'][key] for key in keys}
            work.append(dict(delta=differences, GU_equal=a['actual_GU_dispatch'] == b['actual_GU_dispatch']))
        comparisons.append(dict(indices=[base['index'], candidate['index']],
            warmed_ratio=candidate['subsequent_median_ms']/base['subsequent_median_ms'],
            first_ratio=candidate['samples'][0]['prompt_ms']/base['samples'][0]['prompt_ms'],
            paired_work=work))
    arms = {}
    for arm, selector in enumerate(SELECTORS):
        group = [p for p in processes[3:] if ARMS[p['index']] == arm]
        warmed = [p['subsequent_median_ms'] for p in group]
        first = [p['samples'][0]['prompt_ms'] for p in group]
        arms[selector] = dict(warmed_PP=[READ*1000/v for v in warmed],
            warmed_median_PP=READ*1000/statistics.median(warmed), first_PP=[READ*1000/v for v in first],
            first_TTFT_s=[p['samples'][0]['ttft_s'] for p in group],
            warmed_TTFT_s=[statistics.median(s['ttft_s'] for s in p['samples'][1:]) for p in group])
    ratios = [pair['warmed_ratio'] for pair in comparisons]
    repeated = all(ratio < 1 for ratio in ratios)
    same_work = all(v['GU_equal'] and not any(v['delta'].values())
        for pair in comparisons for v in pair['paired_work'])
    return dict(read=READ, arms=arms, clean_pairs=comparisons,
        geometric_mean_warmed_ratio=statistics.geometric_mean(ratios),
        repeated_warmed_gain=repeated, downstream_work_unchanged=same_work,
        eligible_for_next_model_gates=repeated,
        isolated_reduction_attribution_eligible=repeated and same_work,
        candidate_all_warmed_at_least_1000=all(v >= 1000 for v in arms[SELECTORS[1]]['warmed_PP']),
        inference_adopted=False, sustained_decode_and_task_qualified=False,
        physical262144_and_restore_qualified=False, resource_saturation_known=False)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh campaign only')
    native, proof, prior = capacity.basis()
    revision = validation.git('rev-parse', 'HEAD').decode().strip()
    binary = native['binaries']['strata']
    gate = component_gate(binary)
    source = pins(revision)
    hashes = request_hashes()
    for path in ('bench.sh', 'tools/strata_bench.py'):
        require((P/path).read_bytes() == validation.git('show', f'{WORKLOAD_ARCHIVE}:{path}'),
            'prior request generator bytes changed: '+path)
    ROOT.mkdir(mode=0o700)
    row = dict(stage='prepared', source_commit=revision, sources=source,
        policies={p: sha(P/p) for p in capacity.POLICIES}, original_native_proof=proof,
        **{k: copy.deepcopy(prior[k]) for k in BASIS_FIELDS}, binary=binary,
        native_source_commit=capacity.original.SOURCE, component_gate=gate,
        plans=[], processes=[], request_hashes=hashes, host_passed=False,
        passed=False, complete=False, active=False, ownership_closed=True,
        budgets=dict(wall=2400, cpu=20000, rss_cap=104<<30, text_cap=16<<20,
            file_cap=32<<20, total_cap=128<<20), whole_bytes=1536<<20,
        plan=dict(unit='Same native14508. Ordered FP32 denominator versus subgroup FP32 denominator. Same max, exp2, BF16 boundaries, packed operands and dispatch. Fixed bank256, window10, JIT upload, chunk8192, decode/KV/MTP placement and safety.',
            order='Three progress-traced processes: 32769 subgroup, then 131072 ordered and subgroup. Six clean 131072 processes AB BA AB; three identical exact thinking-on request bodies per process, 24 generations each, with no reused inputs.',
            admission='Verify host frontend probes, original source/recipe/ELFs, loaded providers, workload bodies, exact counts, clocks, actual all-GPU BF16 route, packed bank, window range, allocation/retirement, actual softmax and original decode handoff. All normal exits, empty sessions, quiet validation, unchanged boot and no new GPU fault.',
            performance='First use and warm remain separate. Report all three adjacent warmed ratios and spread. All three must favor subgroup to advance. A difference in downstream work remains part of the end-to-end fixed-input result; it prevents attributing all saved time to the reduction. Report requested bytes and actual job deltas.',
            stopping='Stop at the first input, owner, clock, route, validation, fault or budget refusal. No criterion changes, speed-based exclusions or retry. Commit each closed process before advancing and rehash all 15 model assets after the final process.',
            limits='No model adoption, physical context/RESTORE, sustained TG, agent quality, default strategy switch or measured resource saturation. Existing 32K/64K model failures stand.',
            retention='Keep 27 requests, paired decisions and original source/recipe identities. Retire ended raw captures and generated comparison inputs after recovering the exact original inputs. No new engine build.'))
    for i, (read, arm) in enumerate(zip(READS, ARMS)):
        root = ROOT/str(i)
        root.mkdir()
        (root/'tmp').mkdir()
        (root/'cache').mkdir()
        core.write(root/'config.json', dict(row['configuration'], log=str(root/'engine.stderr')))
        core.write(root/'plan.json', dict(binary=binary, native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        plan = dict(index=i, read=read, arm=arm, heads=1, trace_progress=i<3, root=str(root), port=port,
            generated={name: sha(root/name) for name in ('config.json', 'plan.json')})
        plan['command'] = command(plan)
        plan['environment_sha256'] = comparison.fingerprint(environment(row, plan))
        row['plans'].append(plan)
    row['invariant_sha256'] = normalized_environment(row, row['plans'][0])
    require(all(normalized_environment(row, p) == row['invariant_sha256'] for p in row['plans']),
        'more than one native setting changed')
    save(row)
    print('Prepared the denominator-only 128K comparison. Commit the plan before running probes.')


if __name__ == '__main__':
    require(sys.executable == PYTHON, 'use pinned benchmark Python: '+PYTHON)
    require(len(sys.argv) == 2 and (sys.argv[1] in ('prepare', 'host')
        or sys.argv[1].isdigit() and int(sys.argv[1]) < len(ORDER)), 'prepare|host|0..8')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        comparison.build.guards.no_model_owners([])
        action = sys.argv[1]
        if action == 'prepare':
            prepare()
        elif action == 'host':
            comparison.host(sys.modules[__name__])
        else:
            comparison.run(int(action), sys.modules[__name__])
