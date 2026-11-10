#!/usr/bin/env python3
"""Clean bench.sh input-length characterization of the admitted streamed route.

Main-authored from this fork's owned benchmark controls; repository LICENSE.
Source, original native build, workload and individual samples remain distinct.
This curve is not a matched speedup, sustained decode or model-adoption test.
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
import qsa16_stream_profile as source
import compare_official_prefill as comparison
import check_prefill_strategy_model as frontend
from prefill_scaling_bench import approximation_models, linear_fit

P, B, RESULT = source.P, source.B, source.RESULT
THIS = 'tools/sycl/qsa16_stream_bench.py'
ROOT, KEY = B / 'qsa16-stream-prefill-curve-v1', 'qsa16_stream_prefill_curve_v1'
READS, RUNS = (32769, 49153, 65537, 81921, 98305), 4
ORDER = [(n, 1) for n in READS]
REFERENCE = B / 'decode-iq3-audit-speed-v2/record.json'
require, digest, write = source.require, source.sha, source.device.write
shared, build = comparison.shared, comparison.build
FAST_ROOT, FAST_KEY = B / 'qsa16-stream-fast-entry-v1', 'qsa16_stream_fast_entry_v1'
CLEAN_FAST = False
FAST = dict(STRATA_PREFILL_RELEASE_MTP_EXPERTS='1', STRATA_SYCL_PREFILL_MOE_WINDOW='10',
    STRATA_SYCL_PREFILL_SCORE_GEMM='1', STRATA_SYCL_PREFILL_MIXED_SCRATCH='1',
    STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS='384', STRATA_SYCL_CACHE_SCRATCH_LEASE='1',
    STRATA_MTP_BATCH_F16='1', STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED='1',
    STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR='1')


def configure_baseline():
    global ROOT, KEY, READS, ORDER
    ROOT, KEY = B / 'qsa16-stream-prefill-baseline-v2', 'qsa16_stream_prefill_baseline_v2'
    READS = (32769,)
    ORDER = [(32769, 1)]


def configure_fast_clean():
    global ROOT, KEY, READS, ORDER, CLEAN_FAST
    ROOT, KEY = B / 'qsa16-stream-fast-prefill-v1', 'qsa16_stream_fast_prefill_v1'
    READS, ORDER, CLEAN_FAST = (32769,), [(32769, 1)], True


def configure_fast_curve():
    global ROOT, KEY, READS, ORDER, CLEAN_FAST
    ROOT, KEY = B / 'qsa16-stream-fast-curve-v1', 'qsa16_stream_fast_curve_v1'
    READS = (32769, 49153, 65537, 81921, 98305)
    ORDER, CLEAN_FAST = [(n, 1) for n in READS], True


def pins(revision):
    values = shared.pins(revision)
    values.update(build.controls(revision, controller=P / THIS))
    # The imported Niko server now uses Control outside serve/tools. Pin its
    # complete packages, including deferred imports; keep the failed old
    # serve/tools-only host admission in V1 rather than refreshing its hashes.
    values.update(source.device.common.source.pin(P,
        ['strata_control', 'strata_cli', 'packaging', 'setup.py'])['source_sha256'])
    return values


def compact(row):
    value = {k: v for k, v in row.items()
             if k not in ('sources', 'model_assets', 'environment_template')}
    value['source_inputs'] = dict(files=len(row['sources']),
        sha256=source.signature(row['sources']))
    value['environment_template_sha256'] = source.signature(row['environment_template'])
    value['model_assets_sha256'] = source.signature(row['model_assets'])
    return value


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def basis():
    # The clean benchmark is independent of collector admission. Check the
    # native reader and closed original model directly; do not call an earlier
    # profiler's frozen controller or inherit its successful/failed result.
    model, device = source.model, source.device
    native = device.native(observers={device.THIS, model.THIS, source.THIS, THIS})
    prior = json.loads((source.MODEL_ROOT / 'record.json').read_text())
    require(model.validation.committed(RESULT, (source.MODEL_KEY,)) == model.previous.compact(prior)
        and prior['stage'] == 'closed' and all(prior[k] for k in
            ('passed', 'complete', 'ownership_closed', 'model_executed'))
        and not prior['active'] and prior['native_source_commit'] == native['native_source_commit']
        and prior['native_build']['binaries'] == native['binaries']
        and digest(prior['native_record']['path']) == prior['native_record']['sha256'],
        'closed exact same-native model admission required')
    result, = prior['results']
    require(result['native_exit_code'] == 0 and result['ownership_closed'] and
        not result['fresh_faults'] and len(result['requests']) == 3
        and result['capacity_observation']['actual_chunk'] == 8192
        and result['routes']['actual_mode'] == 1 and result['routes']['typed_QSA_entry'],
        'actual8192 streamed typed-attention/handoff proof required')
    pin = prior['controller']; before = pin['source_sha256']
    recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in
        device.common.source.blobs(P, pin['revision'], before)}
    after = device.common.source.pin(P,
        ['tools/sycl', 'tools/strata_tokenizer.py', 'data/expert-profile.bin'])['source_sha256']
    require(recovered == before and source.signature(before) == pin['aggregate_sha256']
        and set(after) - set(before) == {source.THIS, THIS} and not set(before) - set(after)
        and all(after[p] == h for p, h in before.items()),
        'only the two committed performance observers may extend original model controls')
    require(source.signature(model.validation.committed(RESULT, (device.KEY,))) == prior['device_evidence_sha256']
        and digest(prior['runtime_manifest']['path']) == prior['runtime_manifest']['sha256']
        and all(digest(p) == h for p, h in prior['runtime_files'].items())
        and model.guards.driver_guard() == prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot'],
        'original device/runtime/boot/driver changed')
    model.previous.assets(prior)
    proof = dict(native=native['observer_admission'], original_model_source=prior['source_commit'],
        original_model_evidence_sha256=source.signature(model.previous.compact(prior)),
        original_model_receipt_sha256=digest(source.MODEL_ROOT / 'record.json'),
        original_model_controls_recovered=len(before),
        added_observers={p: after[p] for p in (source.THIS, THIS)},
        collector_qualification_inherited=False, native_recompiled=False)
    if CLEAN_FAST:
        proof['fast_configuration_diagnostic'] = fast_diagnostic_proof(native, prior)
    return native, dict(prior, observer_admission=proof), prior['runtime_files'], None


def environment(row, plan):
    env = comparison.environment(row, plan)
    require(not env.get('LD_PRELOAD') and not env.get('STRATA_TRACE') and
        not env.get('STRATA_UR_PROBE_EXPECTED_LOADER') and
        env['STRATA_KV_STAGE_OWN'] == env['STRATA_SYCL_PREFILL_QSA_STREAM'] == '1'
        and env['STRATA_MTP_BATCH_F16'] == ('1' if CLEAN_FAST else '0')
        and env['STRATA_SYCL_DECODE_Q6_DPAS_HEAD'] == '2'
        and env['STRATA_SYCL_DECODE_Q6_DPAS_DENSE'] == '1'
        and env['STRATA_PREFILL_HEAD_HOST_BACKING'] == '0',
        'preserve actual admitted placement/math/decode and private preload')
    if CLEAN_FAST:
        require(all(env.get(k) == v for k, v in FAST.items()), 'exact qualified fast settings required')
    return env


def native_context(row, plan):
    require(type(plan['index']) is int and 0 <= plan['index'] < len(ORDER)
        and (plan['read'], plan['heads']) == ORDER[plan['index']], 'unplanned curve point')
    return {k: row[k] for k in ('binary', 'native_source_commit', 'native_validation_probe', 'boot')}


def current(row):
    require(build.committed(RESULT)[KEY] == compact(row) and
        pins(row['source_commit']) == row['sources'], 'commit exact curve state and inputs')
    native, prior, providers, _ = basis()
    require(row['binary'] == native['binaries']['strata'] and
        row['native_source_commit'] == native['native_source_commit'] and
        row['observer_admission'] == prior['observer_admission'] and
        row['native_validation_probe'] == prior['probe'] and row['boot'] == prior['boot']
        and row['driver'] == prior['driver'] and row['model_assets'] == prior['models'],
        'curve native/runtime/asset basis changed')
    manifest = row['runtime_manifest']
    require(digest(manifest['path']) == manifest['sha256'] and
        json.loads(Path(manifest['path']).read_text()) == providers,
        'runtime manifest changed')
    require(digest(row['template']['path']) == row['template']['sha256'] and
        build.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'template/output budget differs')
    contract, python = frontend.python_manifest()
    require(contract == row['python_contract'] and python['selected'] == row['python_selected'],
        'benchmark/server Python providers changed')
    return providers, python


def request_route(text, plan):
    headers = [line for line in text.splitlines()
               if line.startswith('strata prefill: layer-major candidate, ')]
    require(len(headers) == 1, 'one actual layer-major entry required')
    pairs = re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)', headers[0].split(';', 1)[0])
    require(len({k for k, _ in pairs}) == len(pairs), 'duplicate placement field')
    f = dict(pairs); rows = plan['read'] - 1
    require(f['tokens'] == str(rows) and f['chunk'] == '8192' and
        f['residual_dtype'] == 'bf16' and f['residual_element_bytes'] == '2',
        'actual native16 shape/chunk differs')
    gpu = int(f['residual_gpu_rows'])
    require(0 < gpu <= rows and int(f['residual_total']) == rows * 10240 * 2 and
        int(f['residual_gpu']) == gpu * 10240 * 2 and
        int(f['residual_host']) == (rows - gpu) * 10240 * 2,
        'GPU/Host residual row/byte conservation differs')
    require(text.count('QSA tensor IO=BF16, indexer/GDN internals FP32;') == 1,
        'actual typed-QSA entry required')
    target = re.findall(r'^strata prefill heads: target detached bytes=(\d+) source=owned-RAM D2H=(\d+) H2D-on-restore=(\d+)$', text, re.M)
    draft = re.findall(r'^strata prefill heads: draft detached bytes=(\d+) retained-map-bytes=(\d+)$', text, re.M)
    require(target == [('521472000',) * 3] and draft == [('0', '425196')],
        'admitted target/sole-packed-draft handoff differs')
    result = dict(rows=rows, actual_chunk=8192, residual_dtype='bf16',
        residual_gpu_rows=gpu, residual_gpu_bytes=int(f['residual_gpu']),
        residual_host_bytes=int(f['residual_host']), layer_cache_bytes=int(f['layer_cache']),
        actual_typed_QSA=True, configured_context=262144, configured_resident=20480,
        owned_stage=True, stage_payload_bytes=276824064, target_requested_D2H=521472000,
        target_requested_H2D_restore=521472000, draft_requested_D2H=0,
        scope='Actual per-request placement headers and handoff extents,with pinned production state policy. No per-launch telemetry,physical bus traffic or saturation claim.')
    if CLEAN_FAST:
        # A private policy copy keeps the comparator's old joint-head contract
        # intact. This build's sole packed draft head retires zero device bytes.
        policy = source.device.load_module('qsa16_stream_clean_window_policy', comparison.__file__)
        policy.DRAFT_BYTES = 0
        result['fast_window'] = policy.route(text, plan['read'], plan['heads'], expected_rows=rows)
        phases = re.findall(r'^strata mtp: prompt expert phase=(\S+) ', text, re.M)
        require(phases == ['before-detach', 'after-detach', 'after-prompt-before-restore', 'after-restore']
            and re.findall(r'^strata mtp: experts restored bytes=(\d+),', text, re.M) == ['707788800'],
            'actual fast-prefill MTP expert handoff required')
        result['MTP_expert_restore_bytes'] = 707788800
    return result


def decision(processes):
    require(len(processes) == len(ORDER) and all(p['passed'] and p['complete'] and
        p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
        and (p['read'], p['heads']) == ORDER[i] and len(p['samples']) == RUNS
        for i, p in enumerate(processes)), 'complete ordered curve required')
    points = []
    def placement(value):
        # Work/copy counts legitimately change with nonce-token routing. Keep
        # them as evidence, but compare actual placement rather than all fields.
        keys = ('rows', 'actual_chunk', 'residual_dtype', 'residual_gpu_rows',
            'residual_gpu_bytes', 'residual_host_bytes', 'layer_cache_bytes',
            'configured_context', 'configured_resident', 'owned_stage')
        result = {k: value[k] for k in keys}
        if 'fast_window' in value:
            result['window'] = {k: value['fast_window'][k] for k in
                ('prefill_slots', 'staged', 'split', 'maximum_window_rows')}
        return result
    for p in processes:
        samples, routes = p['samples'], p['routes']
        points.append(dict(read=p['read'], first_use=samples[0],
            subsequent_prompt_ms=[s['prompt_ms'] for s in samples[1:]],
            subsequent_TTFT_s=[s['ttft_s'] for s in samples[1:]],
            median_prompt_ms=p['subsequent_median_ms'],
            median_PP=p['read'] * 1000 / p['subsequent_median_ms'],
            median_TTFT_s=statistics.median(s['ttft_s'] for s in samples[1:]),
            residual_GPU_rows=routes[0]['residual_gpu_rows'],
            placement=placement(routes[0]),
            placement_unchanged_within_process=all(placement(r) == placement(routes[0]) for r in routes)))
    groups = {}
    for point in points:
        key = 'all_GPU' if point['residual_GPU_rows'] == point['read'] - 1 else (
            'partial_GPU_' + str(point['residual_GPU_rows']))
        window = point['placement'].get('window')
        if window:
            key += f'/slots{window["prefill_slots"]}/staged{int(window["staged"])}/split{int(window["split"])}'
        point['route_group'] = key
        groups.setdefault(key, []).append(dict(actual_read_tokens=point['read'],
            prompt_ms=point['median_prompt_ms']))
    fits = {key: dict(affine=linear_fit(values), alternatives=approximation_models(values),
        adjacent_ms_per_token=[(b['prompt_ms']-a['prompt_ms']) /
            (b['actual_read_tokens']-a['actual_read_tokens']) for a, b in zip(values, values[1:])])
        for key, values in groups.items()}
    return dict(points=points, route_group_approximations=fits,
        single_process_per_length=True, clean_speed=True,
        input_length_curve_complete=len(READS) > 1,
        next='Fit only compatible route groups using actual n and retained individual times;compare affine,nlogn and quadratic residuals/held-out errors with sample spread. Gather independent-process repeats for promising points before claiming a gain or target.',
        matched_speedup=False, resource_saturation_qualified=False,
        sustained_decode_qualified=False, physical262144_qualified=False, inference_adopted=False)


def prepare():
    require(not ROOT.exists() and KEY not in build.committed(RESULT), 'fresh curve only;no retry')
    native, prior, providers, _ = basis()
    reference = json.loads(REFERENCE.read_text())
    contract, python = frontend.python_manifest()
    previous_attempt = None
    if CLEAN_FAST:
        # The preceding diagnostic fully hashed these exact assets after its
        # normal closed native process; identities are rechecked by basis().
        previous_attempt = prior['observer_admission']['fast_configuration_diagnostic']
    elif len(READS) == 1:
        old_root = B / 'qsa16-stream-prefill-baseline-v1'
        old = json.loads((old_root / 'record.json').read_text())
        require(build.committed(RESULT)['qsa16_stream_prefill_baseline_v1'] == compact(old)
            and old['stage'] == 'failed' and not old['passed'] and old['ownership_closed']
            and not old['active'] and not old['processes']
            and old['error'] == "RuntimeError('unfrozen project import')"
            and old['model_assets'] == prior['models'],
            'preserve closed pre-GPU host-source failure and exact model assets')
        recovered = source.device.common.source.git(P, 'show', old['source_commit'] + ':' + THIS)
        require(hashlib.sha256(recovered).hexdigest() == old['sources'][THIS]
            and b'source.model.previous.assets(prior, full=True)' in recovered,
            'recover the exact V1 preparation that fully hashed all assets')
        source.model.previous.assets(prior)
        previous_attempt = dict(key='qsa16_stream_prefill_baseline_v1',
            source_commit=old['source_commit'], receipt_sha256=digest(old_root / 'record.json'),
            passed=False, native_execution_attempted=False,
            model_hash_reuse='The exact recovered V1 prepare fully hashed all15assets before its CPU/mock host refusal. Every full file identity and manifest still matches;no native ran. Reuse these before-cohort hashes,not a fresh full-hash claim. Hash all15again after the V2 cohort.',
            correction='Pin complete current Control/CLI/packaging/setup source packages in addition to serve/tools. Keep native,config,workloads,environment and all acceptance criteria unchanged.')
    else:
        source.model.previous.assets(prior, full=True)
    config = copy.deepcopy(reference['configuration'])
    args = list(prior['argv'][1:]); require(args.count('--serve') == 1, 'one native serve selector')
    args.remove('--serve')
    config.update(exe=str(P / 'tools/sycl/engine_once.py'), args=args, family='qwen',
        backend='sycl', env={}, cwd=str(P), tokenizer=prior['tokenizer'])
    old = prior['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(ROOT)) for k, v in prior['environment'].items()}
    for key in ('STRATA_TRACE', 'LD_PRELOAD', 'STRATA_UR_PROBE_EXPECTED_LOADER',
                'STRATA_DECODE_TIMING', 'STRATA_DRAFT_TIMING', 'STRATA_SNAPSHOT_VERIFY',
                'STRATA_IQ_TIMING_PROGRESS', 'UR_LOG_TRACING'):
        env.pop(key, None)
    for key in env:
        if '_AUDIT' in key or '_PROFILE' in key: env[key] = '0'
    env.update(PATH=str(Path(python['selected']['path']).parent) + ':/usr/bin:/bin')
    if CLEAN_FAST:
        env.update(FAST)
    ROOT.mkdir(mode=0o700); write(ROOT / 'runtime-inputs.json', providers)
    revision = source.device.common.source.git(P, 'rev-parse', 'HEAD').decode().strip()
    row = dict(stage='prepared', source_commit=revision, sources=pins(revision),
        binary=native['binaries']['strata'], native_source_commit=native['native_source_commit'],
        native_validation_probe=prior['probe'], observer_admission=prior['observer_admission'],
        configuration=config, environment_template=env, model_assets=prior['models'],
        template=reference['template'], python_contract=contract, python_selected=python['selected'],
        boot=prior['boot'], driver=prior['driver'],
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'),
            sha256=digest(ROOT / 'runtime-inputs.json'), files=len(providers)),
        plans=[], processes=[], request_hashes={}, host_passed=False,
        passed=False, complete=False, active=False, ownership_closed=True,
        budgets=dict(wall=2400, cpu=28800, rss_cap=104 << 30, text_cap=16 << 20,
            file_cap=32 << 20, total_cap=128 << 20), whole_bytes=640 << 20,
        plan=dict(question=('How do actual input length and memory placement affect clean PP and client TTFT on the current owned-stage streamed-QSA16 route?' if len(READS) > 1 else
            'Establish the current clean32K PP/clientTTFT baseline before a separately qualified model-profiler overhead comparison.'),
            order=f'{len(READS)}fresh processes at{list(READS)}actual input tokens. Four distinct-nonce thinking-on bench.sh exact-count requests each,24outputs. First request separate;all subsequent3samples retained. Commit each closed process before advancing. The separately named single-length baseline supplies the clean32K point before profiler work;it cannot qualify an input-length curve.',
            settings='Original native b9aa/1e598,logical262144/resident20480INT8KV,actual8192chunks,own264MiBstage,1256decode expert slots,5workers,spec4,throughput,packedhead2,dense1,Hostbacking0,MTP_BATCH_F16=0. Independent decode algorithm/placement unchanged. Immutable qualified runtime safety retained;remove trace/timing/audit observers and keep the exact private pre-main validation.',
            acceptance='Current-source host probes before native. Require source/recipe/fiveELFs/runtime/model identities,owned single launch,normal SSE completion,exact prompt/read count,zero reuse,thinking,24outputs,valid clocks/clientTTFT,journal,actual8192BF16QSA route and residual byte conservation/head restoration,quiet validation,normal0,closed owner,unchanged boot/no fresh fault. A capacity-dependent GPU residual fraction is retained as a separate route,not discarded or pooled silently.',
            stopping='Stop first source,provider,owner,clock,count,route,validation,fault or finite budget failure. No retry,hash refresh,performance-based early stop or retrospective exclusion. Full15asset hashes before and after the cohort.',
            limits='Characterization only:one process per length,not a paired engine speedup,independent-process reproducibility,1000PP target qualification,sustainedTG,agent/task quality,full262144fill/RESTORE,ordinary-provider teardown or adoption. Fit by actual route only;no assumed fixed-plus-linear law.',
            retention=f'Keep all{len(READS)*RUNS}compact samples/TTFT,placement and source/native/environment/workload identity. Retire verified successful full logs/journals/cache/ownership output after extraction. Existing current native plus a347/552b imminent phase references unchanged.'))
    if previous_attempt is not None: row['previous_attempt'] = previous_attempt
    if CLEAN_FAST:
        row['plan'].update(question=('Measure clean32K PP/clientTTFT for the admitted fastest-setting composition on streamed QSA16 before longer shapes and independent-process repeats.' if len(READS) == 1 else
                'Characterize clean32/48/64/80/96K PP/clientTTFT and time scaling for the admitted fast-setting composition;repeat32K in a fresh process and separate actual bank/split routes.'),
            order=f'{len(READS)}fresh processes in fixed input order{list(READS)}. Four distinct-nonce thinking-on exact-count bench.sh requests per process;first separate,retain all subsequent3. Commit each closed process before advancing. No speed-based stop or replacement. One process per length in this cohort;the earlier32Kpoint is a separately scoped independent process.',
            settings=row['plan']['settings'].replace('MTP_BATCH_F16=0', 'MTP_BATCH_F16=1') +
                ' Restore exactly the nine FAST values from the owned552b reference before exec;qualified by the closed3d2a diagnostic. No native rebuild or decode-selector change.',
            acceptance=row['plan']['acceptance'] +
                ' Require all actual prefill rows on GPU,window10,checked bank drains and complete675MiB MTP expert restoration. At32/48/64/80K require staged384bank and ordinary accumulation;96K requires the actual staged128/split route with checked tail guard/drain. Keep observed jobs/copy counts;compare placement separately from these counters.',
            stopping=row['plan']['stopping'].replace('Full15asset hashes before and after the cohort.',
                'Reuse the exact closed fast-diagnostic final full hashes after identity/source recovery;hash all15assets again after this cohort.'),
            limits=row['plan']['limits'] +
                ' The old baseline is one earlier process,not a prospective repeated pair. Retain that distinction;no gain/target/adoption claim from one process per length. Group fits by actual GPU residency AND bank/split route;do not fit the96Ksplit point with the ordinary384-bank points. Compare affine,nlogn,quadratic models using residuals/held-out errors and adjacent slopes,without assigning a mixed-route intercept to physical fixed work.')
    for index, (read, heads) in enumerate(ORDER):
        root = ROOT / str(index); root.mkdir(mode=0o700)
        for part in ('tmp', 'cache'): (root / part).mkdir()
        cfg = copy.deepcopy(config); cfg['log'] = str(root / 'engine.stderr')
        write(root / 'config.json', cfg)
        write(root / 'plan.json', dict(binary=row['binary'], native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        plan = dict(index=index, read=read, heads=heads, root=str(root),
            generated={n: digest(root / n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--prefill-exact-tokens', str(read), '--think', '--runs', str(RUNS), '--profile-request-gates',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600',
                '--out', str(root / 'bench')])
        plan['environment_sha256'] = comparison.fingerprint(environment(row, plan))
        row['plans'].append(plan)
    save(row); print(f'Prepared{len(READS)}clean streamed-route processes;commit before host/native.', flush=True)


def fast_diagnostic_proof(native, prior):
    """Reuse closed composition evidence without relabeling its native/source."""
    path = FAST_ROOT / 'record.json'; row = json.loads(path.read_text())
    require(build.committed(RESULT)[FAST_KEY] == source.model.previous.compact(row)
        and row['stage'] == 'closed' and all(row[k] for k in
            ('passed', 'complete', 'ownership_closed', 'model_executed')) and not row['active']
        and row['native_source_commit'] == native['native_source_commit']
        and row['native_build']['binaries'] == native['binaries'] and row['models'] == prior['models']
        and row['fast_configuration']['settings'] == FAST, 'closed exact fast-composition proof required')
    result, = row['results']; route = result['routes']
    require(result['native_exit_code'] == 0 and result['ownership_closed'] and not result['fresh_faults']
        and len(result['requests']) == 3 and route['actual_mode'] == 1 and route['typed_QSA_entry']
        and route['packed_layer_entries'] == 96 and route['staged_bank_headers'] == 2
        and route['actual_batched_ring_callbacks'] == 8 and route['cache_binding_and_final_drains']
        and route['MTP_expert_restore_bytes'] == [707788800] * 2
        and route['MTP_round_recaptures'] >= 2 and route['decode_routes']['head']['actual_decode_C4_replays'] > 0,
        'actual composed prefill and restored decode proof required')
    pin = row['controller']; before = pin['source_sha256']
    recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in
        source.device.common.source.blobs(P, pin['revision'], before)}
    after = source.device.common.source.pin(P,
        ['tools/sycl', 'tools/strata_tokenizer.py', 'data/expert-profile.bin'])['source_sha256']
    require(recovered == before and source.signature(before) == pin['aggregate_sha256']
        and before.keys() == after.keys() and {p for p in before if before[p] != after[p]} == {THIS},
        'only this prospective clean controller may change after the fast diagnostic')
    source.model.previous.assets(row)
    return dict(key=FAST_KEY, evidence_commit='7851bc6de0c1d0398cc333d27b861a2efca3fa51',
        original_controller_source=row['source_commit'], original_native_source=row['native_source_commit'],
        receipt_sha256=digest(path), recovered_controller_inputs=len(recovered), changed_controller=[THIS],
        model_assets_sha256=source.signature(row['models']),
        meaning='Closed original fast diagnostic source/ELFs/recipe/runtime/models recovered. Reuse its final full hashes with exact asset identities;rehash all15after this clean cohort. No inherited clean speed,sustainedTG or adoption.')


def baseline_proof(native, prior):
    """Recover the exact closed benchmark; allow only this new controller delta."""
    root = B / 'qsa16-stream-prefill-baseline-v2'
    old = json.loads((root / 'record.json').read_text())
    require(build.committed(RESULT)['qsa16_stream_prefill_baseline_v2'] == compact(old)
        and old['stage'] == 'closed' and all(old[k] for k in ('passed', 'complete', 'ownership_closed'))
        and not old['active'] and len(old['processes']) == 1
        and old['processes'][0]['native_exit_code'] == 0 and not old['processes'][0]['fresh_faults']
        and old['binary'] == native['binaries']['strata']
        and old['native_source_commit'] == native['native_source_commit']
        and old['model_assets'] == prior['models'], 'closed same-engine clean32K baseline required')
    recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in
        source.device.common.source.blobs(P, old['source_commit'], old['sources'])}
    revision = source.device.common.source.git(P, 'rev-parse', 'HEAD').decode().strip()
    current = pins(revision)
    require(recovered == old['sources'] and current.keys() == recovered.keys()
        and {p for p in current if current[p] != recovered[p]} == {THIS},
        'only this explicitly committed prospective controller may change')
    source.model.previous.assets(prior)
    return old, dict(key='qsa16_stream_prefill_baseline_v2',
        evidence_commit='55f4a341ffcb4a8849c1056f3bd4bb4e5dd4985a',
        original_source_commit=old['source_commit'], receipt_sha256=digest(root / 'record.json'),
        source_files_recovered=len(recovered), changed_controller=[THIS],
        model_assets_sha256=source.signature(old['model_assets']),
        meaning='Original closed source recovered;all15full final model hashes precede this new plan and every identity remains exact. Reuse before-run hashes,rehash all15after the new process. No native rebuild or inherited fast-setting GPU result.')


def fast_native(expected=None):
    native, prior, _, _ = basis()
    proof = prior['observer_admission']
    require(expected is None or proof == expected, 'prospective fast native admission changed')
    return dict(native, model_observer_admission=proof)


def fast_route(text):
    result = source.model.route(text)
    def records(prefix, fields):
        values = []
        for line in text.splitlines():
            if not line.startswith(prefix): continue
            pairs = [p.split('=', 1) for p in line[len(prefix):].split()]
            require(line.isascii() and all(len(p) == 2 and p[0] and p[1] for p in pairs)
                and len(pairs) == len(fields) == len({p[0] for p in pairs})
                and {p[0] for p in pairs} == fields, 'exact fast-route producer schema required')
            values.append(dict(pairs))
        return values
    packed = records('strata packed expert bank: ',
        {'layer', 'slots', 'stride', 'used_bytes', 'backing_bytes', 'full_fit'})
    require(len(packed) == 96 and [int(v['layer']) for v in packed] == list(range(48)) * 2
        and all(384 <= int(v['slots']) <= 512 and int(v['stride']) > 0
            and int(v['used_bytes']) == int(v['slots']) * int(v['stride'])
            and int(v['used_bytes']) <= int(v['backing_bytes'])
            and int(v['full_fit']) == int(v['slots'] == '512') for v in packed),
        'actual packed bank geometry/48layer entries required in each long prefill')
    banks = records('strata MoE expert bank: ', {'requested_slots', 'effective_slots', 'staged',
        'host_slab_bytes', 'device_bytes', 'residual_gpu_rows', 'allocation_limit_bytes', 'general_map'})
    require(len(banks) == 2 and all(v['requested_slots'] == v['effective_slots'] == '384'
        and v['staged'] == '1' and v['residual_gpu_rows'] == '32768'
        and v['general_map'] == 'private_scratch_never_published' for v in banks),
        'actual staged384/private-map route required')
    phases = re.findall(r'^strata mtp: prompt expert phase=(\S+) ', text, re.M)
    require(phases == ['before-detach', 'after-detach', 'after-prompt-before-restore', 'after-restore'] * 2
        and re.findall(r'^strata mtp: experts restored bytes=(\d+),', text, re.M) == ['707788800'] * 2,
        'both complete actual675MiB MTP expert retirement/restores required')
    require(len(re.findall(r'^strata prefill cache binding: generation=\d+ parent_match=1 execution_distinct=1$', text, re.M)) == 2
        and len(re.findall(r'^strata MoE expert bank complete: .* final_drain=checked$', text, re.M)) == 2,
        'bound lease/private execution and final bank drains required')
    recaptures = re.findall(r'^strata mtp: expert decode recapture kind=round coupled=(\d+) ok=(\d+) ', text, re.M)
    require(len(recaptures) >= 2 and all(v == ('0', '1') for v in recaptures),
        'successful independent MTP round recapture after restoration required')
    callbacks = re.findall(r'^strata serve: DRAFT_PREFILL path=(\w+) mode=(\d+) cells=(\d+)$', text, re.M)
    require(callbacks == [('batched', '2', '8192')] * 8,
        'actual32Kring callbacks must reach the immutable FP16batch selector')
    result.update(packed_layer_entries=96, bank_slots_min=min(int(v['slots']) for v in packed),
        bank_slots_max=max(int(v['slots']) for v in packed), staged_bank_headers=2,
        MTP_expert_restore_bytes=[707788800, 707788800], MTP_round_recaptures=len(recaptures),
        actual_batched_ring_callbacks=8, cache_binding_and_final_drains=True,
        precision='Immutable MTP_BATCH_F16=1 plus actual batched mode2 callbacks selects the pinned FP16projection branch. No per-launch projector telemetry or task-quality claim.')
    return result


def fast_context(before_hashes=None):
    ctx = source.device.load_module('qsa16_stream_fast_model_context', source.model.__file__)
    ctx.ROOT, ctx.KEY, ctx.ATTEMPT = FAST_ROOT, FAST_KEY, 'v2'
    ctx.native, ctx.route = fast_native, fast_route
    original_context = ctx.context

    def private_context(admission=None):
        value = original_context(admission)
        if before_hashes is not None:
            original_assets = value.assets

            def assets(row, full=False):
                require(row['models'] == before_hashes['model_assets'], 'reused before-run assets changed')
                original_assets(row, full=False)

            value.assets = assets
        return value

    ctx.context = private_context
    return ctx


def fast_prepare():
    native, prior, _, _ = basis(); old, proof = baseline_proof(native, prior)
    best = build.committed(RESULT)['cache_map_prefill_speed117489_v1']
    require(all(best['environment_template'].get(k) == v for k, v in FAST.items()),
        'recover exact owned fastest-prefill settings,not an invented configuration')
    ctx = fast_context(old); ctx.prepare()
    row = json.loads((FAST_ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and not row['results'], 'fresh unexecuted generated plan only')
    row['environment'].update(FAST)
    row['fast_configuration'] = dict(settings=FAST, baseline=proof,
        fastest_reference_source=best['source_commit'], fastest_native_source=best['native_source_commit'],
        fastest_binary=best['binary']['sha256'], original_native_and_decode_implementation_unchanged=True)
    row['plan'] = ('New configuration diagnostic,not a clean speed comparison. Same b9aa/1e598 native/five fixtures,'
        'logical262144/resident20480INT8,own264MiBstage,8192chunks,1256decode slots,spec4,'
        'head2/dense1/Hostbacking0 and immutable safety. Restore exactly nine owned552b prefill settings'
        'before exec:window10,scoreGEMM,mixed scratch,packed384bank,lease,splitacc,MTPFP16batch,'
        'release675MiBMTPexperts. Thinking32769/82/32769input,32outputs each. Require existing exact'
        'READY/fullINFO/source/runtime/protocol/handoff/quiet validation/normal0/closure/no-fault gates,'
        '96actual packed-layer entries,staged384private bank,checked final drains,bound current leases,'
        'two full MTP expert restores,successful uncoupled round recaptures and8actual8192mode2batch callbacks.'
        'Original FP16branch is selected from immutable environment and actual callback,not extra telemetry.'
        'Reuse the closed clean baseline final full model hashes with exact identities;rehash all15after'
        'this process. Stop first failure;no retry,hash refresh or output-equality requirement.')
    row['limits'] = ('Prospective fast-setting composition/ownership/route check only. No clean speed,gain,'
        'sustainedTG,task quality,physical262144fill/RESTORE or adoption. The following clean bench.sh'
        'comparison must preserve actual bodies/configuration,repeat promising points and characterize length.')
    ctx.context().save(row)
    print('Prepared owned fast-prefill configuration diagnostic;commit before host/native.', flush=True)


def fast_execute(action):
    ctx = fast_context(); row = json.loads((FAST_ROOT / 'record.json').read_text())
    native, prior, _, _ = basis(); _, proof = baseline_proof(native, prior)
    require(row['fast_configuration']['baseline'] == proof and
        all(row['environment'].get(k) == v for k, v in FAST.items()), 'frozen fast configuration changed')
    if action == 'host':
        require(row['stage'] == 'prepared', 'one fresh configuration check only')
        ctx.context(row['native_observer_admission']).verify(row)
        args = row['argv']
        require(all(args.count(k) == 1 and args[args.index(k)+1] == v for k, v in
            {'--max-context':'262144', '--kv-resident':'20480', '--prefill':'8192', '--expert-cache':'1256',
             '--kv':'int8', '--spec':'4'}.items()) and '--no-token-graph' in args
            and '--no-coupled-draft' in args and '--mtp' in args and '--draft' not in args,
            'single-GPU native MTP4/lease configuration required')
        row.update(stage='host_passed', host_controls=dict(configuration_checked=True,
            original_INFO_controls_reused_from='qsa16_stream_model_entry_v2',
            INFO_source_commit='48e4a5e02f15b48b5f75a91e341bca4dc3dc53f9',
            new_numerical_tests_executed=False, GPU_executed=False))
        ctx.context().save(row); print('Fast configuration/native input checks pass;commit before GPU.', flush=True)
    else:
        require(action == 'run', 'use fast prepare|host|run')
        ctx.execute('run')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); source.model.guards.no_model_owners([])
        action = sys.argv[1:]
        if action[:1] == ['fast']:
            require(len(action) == 2 and action[1] in ('prepare', 'host', 'run'), 'use fast prepare|host|run')
            fast_prepare() if action[1] == 'prepare' else fast_execute(action[1])
            raise SystemExit(0)
        if action[:1] == ['baseline']:
            configure_baseline(); action = action[1:]
        elif action[:1] == ['fast-clean']:
            configure_fast_clean(); action = action[1:]
        elif action[:1] == ['fast-curve']:
            configure_fast_curve(); action = action[1:]
        if action == ['prepare']: prepare()
        elif action == ['host']: comparison.host(sys.modules[__name__])
        elif len(action) == 1 and action[0].isdigit() and int(action[0]) < len(ORDER):
            comparison.run(int(action[0]), sys.modules[__name__])
        else: raise RuntimeError('use prepare|host|INDEX0..4')
