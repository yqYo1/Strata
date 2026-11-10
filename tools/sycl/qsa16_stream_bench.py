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


def configure_baseline():
    global ROOT, KEY, READS, ORDER
    ROOT, KEY = B / 'qsa16-stream-prefill-baseline-v2', 'qsa16_stream_prefill_baseline_v2'
    READS = (32769,)
    ORDER = [(32769, 1)]


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
    return native, dict(prior, observer_admission=proof), prior['runtime_files'], None


def environment(row, plan):
    env = comparison.environment(row, plan)
    require(not env.get('LD_PRELOAD') and not env.get('STRATA_TRACE') and
        not env.get('STRATA_UR_PROBE_EXPECTED_LOADER') and
        env['STRATA_KV_STAGE_OWN'] == env['STRATA_SYCL_PREFILL_QSA_STREAM'] == '1'
        and env['STRATA_MTP_BATCH_F16'] == '0'
        and env['STRATA_SYCL_DECODE_Q6_DPAS_HEAD'] == '2'
        and env['STRATA_SYCL_DECODE_Q6_DPAS_DENSE'] == '1'
        and env['STRATA_PREFILL_HEAD_HOST_BACKING'] == '0',
        'preserve actual admitted placement/math/decode and private preload')
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
    return dict(rows=rows, actual_chunk=8192, residual_dtype='bf16',
        residual_gpu_rows=gpu, residual_gpu_bytes=int(f['residual_gpu']),
        residual_host_bytes=int(f['residual_host']), layer_cache_bytes=int(f['layer_cache']),
        actual_typed_QSA=True, configured_context=262144, configured_resident=20480,
        owned_stage=True, stage_payload_bytes=276824064, target_requested_D2H=521472000,
        target_requested_H2D_restore=521472000, draft_requested_D2H=0,
        scope='Actual per-request placement headers and handoff extents,with pinned production state policy. No per-launch telemetry,physical bus traffic or saturation claim.')


def decision(processes):
    require(len(processes) == len(ORDER) and all(p['passed'] and p['complete'] and
        p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
        and (p['read'], p['heads']) == ORDER[i] and len(p['samples']) == RUNS
        for i, p in enumerate(processes)), 'complete ordered curve required')
    points = []
    for p in processes:
        samples, routes = p['samples'], p['routes']
        points.append(dict(read=p['read'], first_use=samples[0],
            subsequent_prompt_ms=[s['prompt_ms'] for s in samples[1:]],
            subsequent_TTFT_s=[s['ttft_s'] for s in samples[1:]],
            median_prompt_ms=p['subsequent_median_ms'],
            median_PP=p['read'] * 1000 / p['subsequent_median_ms'],
            median_TTFT_s=statistics.median(s['ttft_s'] for s in samples[1:]),
            residual_GPU_rows=routes[0]['residual_gpu_rows'],
            placement_unchanged_within_process=all(r == routes[0] for r in routes)))
    groups = {}
    for point in points:
        key = 'all_GPU' if point['residual_GPU_rows'] == point['read'] - 1 else (
            'partial_GPU_' + str(point['residual_GPU_rows']))
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
    if len(READS) == 1:
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


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); source.model.guards.no_model_owners([])
        action = sys.argv[1:]
        if action[:1] == ['baseline']:
            configure_baseline(); action = action[1:]
        if action == ['prepare']: prepare()
        elif action == ['host']: comparison.host(sys.modules[__name__])
        elif len(action) == 1 and action[0].isdigit() and int(action[0]) < len(ORDER):
            comparison.run(int(action[0]), sys.modules[__name__])
        else: raise RuntimeError('use prepare|host|INDEX0..4')
