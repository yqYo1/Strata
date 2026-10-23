#!/usr/bin/env python3
"""Capture real QSA selections on the separately checked diagnostic engine.

One official thinking-enabled32K-minimum prefill request. This diagnostic adds
host readback/file writes; its clocks are not clean speed. Repository LICENSE.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import select
import shutil
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import generation_engine as gen
import resource_engine as resource
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation
import qsa_selection_capture as reader
import check_selection_capture as checked
import packed_bank as bank
from measure_down_scatter import postchecks

P, B, A = gen.P, gen.B, gen.A
ROOT = B / 'qsa-real-selection-v2'
RESULT = A / 'qsa-selection-current.json'
STREAM_SCOPE = 'model_selection_stream_contract_20261023'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def pins(revision):
    result = gen.benchmark_pins(revision)
    for name in ('tools/test_qsa_selection_capture.py', 'src/prefill/prefill.cpp',
                 'include/strata/kernels/qsa.hpp', 'src/kernels/sycl/qsa_select.cpp',
                 'src/program/generate.cpp', 'src/core/session.cpp', 'src/core/mtp.cpp',
                 'src/core/verify.cpp', 'src/core/expert_source.cpp', 'src/sycl/compat.cpp',
                 'src/prefill/layer_major.inc', 'data/expert-profile.bin'):
        raw = gen.git('show', revision + ':' + name)
        guards.require(raw == (P / name).read_bytes(), 'uncommitted capture input: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def environment(setup):
    _, restored, _ = resource.modules()
    env = restored.environment(setup, ROOT, '0')
    env.update(STRATA_CPU_WORK_CENSUS='0', STRATA_IQ2_PAIR_AUDIT='0',
        STRATA_DECODE_TIMING='0', STRATA_SNAPSHOT_VERIFY='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1',
        STRATA_QSA_DUMP=str(ROOT / 'selection.bin'), STRATA_QSA_DUMP_LAST='1')
    guards.require('LD_PRELOAD' not in env, 'validation belongs to native child')
    return env


def diagnostic_qualification():
    built, _, runtime = checked.admission()
    proof = gen.committed(checked.RESULT)
    guards.require(proof['passed'] and proof['complete'] and proof['ownership_closed'] and not proof['active'] and
        proof['fresh_faults'] == [] and proof['boot_unchanged'] and proof['postchecks_complete'] and
        proof['build_sha256'] == gen.common.digest(checked.candidate.RESULT) and
        proof['fixture'] == built['artifacts']['readback'] and proof['runtime_lineage'] == runtime and
        [r['mode'] for r in proof['cases']] == ['small', 'full'] and
        all(r['exit_code'] == 0 and r['exact_payload_and_guards'] and r['invalid_geometry_no_submit']
            and r['activation']['pid'] == r['native_identity']['pid'] for r in proof['cases']),
        'checked small/full native readback qualification required')
    review = validation.committed(validation.INDEX, (STREAM_SCOPE, 'review'))
    guards.require(review.get('status') == 'complete' and review.get('source_gate_passed') is True,
                   'model caller-stream source review required')
    return built, dict(build_sha256=gen.common.digest(checked.candidate.RESULT),
        readback_sha256=gen.common.digest(checked.RESULT), stream_review=review, runtime_lineage=runtime)


def current(record, full_models=False):
    full, proof, setup = resource.qualification()
    built, diagnostic = diagnostic_qualification()
    assets, restored, _ = resource.modules()
    guards.require(record['binary'] == built['artifacts']['engine'] and
        record['build_source_commit'] == built['artifact_source_commits']['engine'] and
        record['selected_binary'] == full['binary'] and record['qualification'] == proof and
        record['diagnostic_qualification'] == diagnostic and
        record['native_validation_probe'] == validation.preload_qualification(), 'native qualification changed')
    guards.require(record['source_fingerprint'] == assets.fingerprint(pins(record['source_commit'])) and
        record['environment'] == environment(setup) and
        record['config'] == restored.recipe.config(setup, ROOT), 'capture source/config/environment changed')
    for path, pin in record['generated_inputs'].items():
        guards.require(gen.common.digest(path) == pin, 'generated capture input changed')
    guards.require(assets.machine_identity() == record['machine_identity'] and
        assets.model_pins(record['config'], record['model_assets'], full=full_models) == record['model_assets'],
        'capture model/platform changed')
    # checked.admission binds all31 assets plus component/model normal-exit
    # lineage explicitly; this is a new diagnostic mode,not an engine-compare alias.
    return setup


def prepare():
    guards.require(not ROOT.exists(), 'fresh single-use capture only; inspect existing owner')
    previous = gen.committed(RESULT)
    guards.require(previous['stage'] == 'closed_prelaunch' and previous['passed'] is False and
        previous['GPU_executed'] is False and previous['ownership_closed'], 'preserve original refused capture')
    full, proof, setup = resource.qualification()
    built, diagnostic = diagnostic_qualification()
    assets, restored, _ = resource.modules()
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    source = pins(revision)
    probe = validation.preload_qualification()
    config = restored.recipe.config(setup, ROOT)
    args = config['args']
    for key, value in (('--expert-cache', '1256'), ('--prefill', '8192'),
                       ('--max-context', '131072'), ('--kv', 'int8')):
        guards.require(args[args.index(key) + 1] == value, 'selected configuration changed: ' + key)
    models = assets.model_pins(config, setup['model_assets'], full=True)
    ROOT.mkdir(mode=0o700)
    for name in ('cache', 'tmp'): (ROOT / name).mkdir()
    shutil.copyfile(P / 'data/expert-profile.bin', ROOT / 'expert-profile.bin')
    guards.require(gen.common.digest(ROOT / 'expert-profile.bin') == setup['expert_profile_sha256'], 'profile changed')
    write(ROOT / 'config.json', config)
    binary = built['artifacts']['engine']
    write(ROOT / 'plan.json', dict(binary=binary, native_validation_probe=probe))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    record = dict(stage='prepared', mode='checked-model-QSA-selection-capture', source_commit=revision,
        build_source_commit=built['artifact_source_commits']['engine'], binary=binary,
        selected_binary=full['binary'], qualification=proof, diagnostic_qualification=diagnostic,
        original_refusal=dict(commit=revision,path=str(RESULT.relative_to(P)),sha256=gen.common.digest(RESULT)),
        runtime_review_followup=previous['runtime_review_followup'],
        native_validation_probe=probe, source_count=len(source), source_fingerprint=assets.fingerprint(source),
        config=config, environment=environment(setup), model_assets=models,
        machine_identity=assets.machine_identity(), runtime_asset_delta=setup['runtime_asset_delta'],
        generated_inputs={str(ROOT / n): gen.common.digest(ROOT / n)
            for n in ('config.json', 'plan.json', 'expert-profile.bin')},
        command=['/bin/sh', str(P / 'bench.sh'), '--config', str(ROOT / 'config.json'), '--port', str(port),
            '--prefill-only', '--prefill-tokens', '32768', '--think', '--runs', '1',
            '--profile-request-gates', '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error',
            '--ready-timeout', '600', '--out', str(ROOT / 'bench')],
        prospective_gates='One native process,one short official warmup then one fresh >=32768 input/1..24 output thinking-on request. '
            'Require exact source/runtime/diagnostic3assets/selected13assets/15models,private same-PID validation before main, '
            'actual immutable environment and warmup journal before long dispatch,official stream/journal/native exit, '
            'Actual long-request GPU BF16/mode10/private48-layer route and1256-slot payload/map/dense restoration, '
            'all12QSA final-chunk records,all selectedIDs '
            'unique/ascending/causal,normal owned closure and no fresh faults. First failure stops;no retry.',
        retention='Raw selection.bin <=1GiB only through current validated12-layer geometry/logical-union analysis. '
            'No persistent real-ID replay payload:there is no implemented GPU replay consumer yet. '
            'Retire all IDs and normal traces after compact evidence is committed/verified. '
            'One official benchmark prompt/process cannot establish general workload locality. '
            'Logical unions do not measure traffic,cache hit rate or saturation.',
        budgets=dict(wall=1800, cpu=54000, rss_cap=96 << 30, text_cap=32 << 20,
            file_cap=1 << 30, total_cap=1536 << 20),
        complete=False, passed=False, active=False, ownership_closed=True,
        clean_speed_qualified=False, hardware_saturation_qualified=False, model_adoption_qualified=False)
    current(record)
    write(ROOT / 'capture-plan.json', record); write(RESULT, record)
    print('Prepared checked diagnostic-engine selection capture; commit exact plan before execution.', flush=True)


def route_frame(text, cells):
    """Validate the long frame only; warmup/startup markers cannot satisfy it."""
    headers = list(re.finditer(r'^strata serve: prompt .*$', text, re.M))
    guards.require(len(headers) == 2, 'exact warmup/long native frames required')
    segment = text[headers[0].end():headers[1].start()]
    route = re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 '
        r'residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) '
        r'residual_gpu=(\d+) residual_host=0 residual_staging=0 ', segment, re.M)
    guards.require(len(route) == 1 and tuple(map(int, route[0])) == (cells,cells*20480,cells,cells*20480) and
        len(re.findall(r'^strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=' +
            str(min(cells,81920)) + r' mixed_fp16_bytes=0$', segment,re.M)) == 1,
        'actual long-request GPU BF16/mode10 route differs')
    cycle, pub = bank.admission.loan_cycle(segment)
    dense = bank.admission.records(segment,'strata prefill dense',3)
    guards.require([(r['phase'],r['residency'],r['activations']) for _,r in dense] ==
        [('retired','Full','0'),('bank','Bank','0'),('restored','Full','48')] and
        pub[0][0] < dense[0][0] < dense[1][0] < dense[2][0] < pub[1][0],
        'dense/cache handoff not restored in the long-request loan')
    return dict(actual_GPU_BF16_rows=cells,actual_MoE_mode=10,
        private_bank=bank.geometry(segment,1),restored_dense_layers=48,
        cache_payload_and_map_match=True,loan_generation=cycle['generation'])


def official(record, boot, entry):
    result_path, = (ROOT / 'bench').glob('*/results.json')
    result = json.loads(result_path.read_text()); cohort = result['results']['prefill_only']
    guards.require(result['complete'] and not result['results']['failures'] and
        result['params']['mode'] == 'prefill-only' and result['params']['think'] is True and
        result['params']['prefill_tokens'] == 32768 and cohort['complete'] and cohort['eligible_runs'] == 1,
        'incomplete official prefill diagnostic')
    sample, = cohort['requests']
    guards.require(all(sample[k] is True for k in ('ok', 'eligible', 'stream_complete',
        'thinking_requested', 'reasoning_observed')) and
        32768 <= sample['actual_prompt_tokens'] == sample['actual_read_tokens'] <= 131072 and
        sample['actual_reused_tokens'] == 0 and 1 <= sample['generated_tokens'] <= 24 and
        math.isfinite(sample['prompt_ms']) and sample['prompt_ms'] > 0, 'ineligible diagnostic request')
    journal, _ = gen.journal_evidence(ROOT)
    path, = (ROOT / 'bench').glob('*/requests.jsonl')
    events = [json.loads(x) for x in path.read_text().splitlines()]
    guards.require(journal['valid'] and journal['complete'] and len(events) == 5 and
        [e['kind'] for e in events] == ['identity', 'start', 'result', 'start', 'result'] and
        events[-1]['sample'] == {k: v for k, v in sample.items() if k != 'run'} and
        events[-1]['request_index'] == events[-2]['index'] and
        events[-1]['sample']['request_sha256'] == events[-2]['request_sha256'], 'journal/request binding')
    native = gen.native_end(ROOT, record, dict(config=record['config']), boot, entry['pid'])
    log = (ROOT / 'engine.stderr').read_text(errors='replace')
    activation = validation.preload_evidence(log, native['native']['pid'])
    guards.require(activation['loader'] == record['native_validation_probe']['loader']['path'], 'validation provider differs')
    # The qualified pre-main canary intentionally exercises invalid-null-handle.
    body = log.split(f'strata UR validation: pid={native["native"]["pid"]} phase=before-main', 1)[1]
    guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', body), 'runtime error')
    footers = re.findall(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$', log, re.M)
    guards.require(len(footers) == 2 and tuple(map(int, footers[-1])) ==
        (sample['actual_read_tokens'], 0, sample['actual_read_tokens'], sample['generated_tokens']), 'native request work differs')
    warm = tuple(map(int, footers[0]))
    guards.require(1 < warm[0] < 1024 and warm[1] == 0 and warm[0] == warm[2] and 1 <= warm[3] <= 24,
        'unexpected warmup extent')
    resource.modules()[0].verify_pool(log, 5)
    route = route_frame(log,sample['actual_read_tokens']-1)
    geometry = reader.analyze(ROOT / 'selection.bin', sample['actual_read_tokens'], warm[0])
    return dict(request=sample, geometry=geometry, native_identity=native['native'],
        validation_activation=activation, raw_sha256=gen.common.digest(ROOT / 'selection.bin'),
        journal_sha256=gen.common.digest(path), actual_long_request_route=route,
        real_ID_payload_retained=False)


def run():
    record = gen.committed(RESULT)
    guards.require(record == json.loads((ROOT / 'capture-plan.json').read_text()) and record['stage'] == 'prepared'
        and not (ROOT / 'record.json').exists() and not (ROOT / 'launch.json').exists(), 'fresh committed capture required')
    current(record, full_models=True)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(x.split(': ', 1)[1] for x in before.splitlines() if x.startswith('-- cursor: '))
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(record, owned_commands=owner.commands if owner else []))
    def interact(io, leader):
        for stage in ('begin', 'end'):
            while True:
                line = io.line(1500)
                if line.startswith('STRATA_PROFILE_GATE '): break
            guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                dict(stage=stage, phase='prefill-only', run=0), 'request gate differs')
            if stage == 'begin':
                native = json.loads((ROOT / 'engine-start.json').read_text())['native']
                actual = supervision.identity(native['pid'])
                guards.require(actual and actual['start_ticks'] == native['start_ticks'] and actual['sid'] == leader['pid'] and
                    Path('/proc', str(native['pid']), 'exe').resolve() == Path(record['binary']['path']).resolve(), 'native identity differs')
                env = dict(x.split(b'=', 1) for x in Path('/proc', str(native['pid']), 'environ').read_bytes().split(b'\0') if b'=' in x)
                guards.require(all(env.get(k.encode()) == v.encode() for k, v in record['environment'].items()) and
                    env.get(b'LD_PRELOAD') == record['native_validation_probe']['library']['path'].encode(), 'native environment differs')
                path, = (ROOT / 'bench').glob('*/requests.jsonl')
                events = [json.loads(x) for x in path.read_text().splitlines()]
                guards.require(len(events) == 4 and events[2]['kind'] == 'result' and events[2]['sample']['ok'] is True
                    and events[3]['phase'] == 'prefill-only', 'warmup before long dispatch failed')
                validation.preload_evidence((ROOT / 'engine.stderr').read_text(errors='replace'), native['pid'])
                record.update(native_identity=native, effective_environment_verified=True); persist()
            io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); guards.require(time.monotonic() < deadline, 'benchmark teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    owner = supervision.Owner(ROOT / 'owned', persist); persist()
    try:
        entry, _, _ = owner.run('bench', record['command'], record['environment'], P, interaction=interact, **record['budgets'])
        guards.require(supervision.closed(entry) and entry['exit_code'] == 0, 'capture benchmark did not close normally')
        record.update(official(record, boot, entry)); current(record, full_models=True)
        record.update(passed=True, complete=True)
    except BaseException as error: record['error'] = repr(error)
    finally:
        record['ownership_closed'] = owner.active is None and all(supervision.closed(x) for x in owner.commands)
        postchecks(record, boot, cursor); record['stage'] = 'closed'; persist(); write(RESULT, record)
    print(json.dumps({k: record.get(k) for k in ('passed', 'complete', 'ownership_closed', 'fresh_faults', 'error')}), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    guards.require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'use prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([]); guards.driver_guard()
        sys.exit(prepare() if sys.argv[1] == 'prepare' else run())
