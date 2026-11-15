#!/usr/bin/env python3
"""Admit actual bulk-mirror prefill and decode handoff before clean benchmarks.

Main-authored from the maintained owned streamed-model controls; repo LICENSE.
This diagnostic uses the compiled candidate, not an inherited model result.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_qsa_bulk_model as device
import qsa16_stream_model_entry as original
import qsa16_stream_bench as benchmark
import qsa_bulk_mirror as component

P, B = device.P, device.B
ROOT, KEY = B / 'qsa16-bulk-mirror-model-entry-v3', 'qsa16_bulk_mirror_model_entry_v3'
THIS = 'tools/sycl/qsa_bulk_model_entry.py'
PYTHON = '/home/yayoi/.local/share/strata-sycl/venv/bin/python3'
require, sha = original.require, original.sha


def native(expected=None):
    """Admit the fresh five-stage result, without a legacy repair receipt."""
    dev = device.context()
    n = dev.native(observers={dev.THIS, THIS})
    prior = dev.validation.committed(dev.RESULT, (device.KEY,))
    results = prior['results']
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete']
        and prior['ownership_closed'] and not prior['active']
        and prior['binaries'] == n['binaries']
        and prior['native_source_commit'] == n['native_source_commit']
        and [v['mode'] for v in results] == list(device.STAGES)
        and all(v['passed'] and v['ownership_closed'] and v['native_exit_code'] == 0
                for v in results), 'fresh closed five-stage same-native result required')
    host = results[0]
    require(not host['GPU_executed'] and host['reader_cases'] == 50
        and host['native_cases'] == 11 and host['policy_check']['passed']
        and host['policy_check']['native_exit_code'] == 0
        and host['selected_head_IR']['same_as_current_reference']
        and host['selected_head_IR']['engine_fixture_identical'],
        'fresh reader/refusal/policy/head-module controls required')
    require(all(v['GPU_executed'] and v['native_execution_attempted']
        and v['host_probe_accepted'] and v['fresh_faults'] == []
        and v['boot_unchanged'] and v['driver_after'] == prior['driver']
        for v in results[1:]), 'fresh normal device closures without faults required')
    require(prior['native_observer_admission']['exact_seed_recipe_sha256']
        == n['observer_admission']['exact_seed_recipe_sha256'],
        'same exact qualified compiler recipe required')
    admission = prior['source_commit']
    source = dev.common.source.git(P, 'show', admission + ':' + dev.THIS)
    source_sha = hashlib.sha256(source).hexdigest()
    require(source_sha == prior['native_observer_admission']['added_observers'][dev.THIS]
        == n['observer_admission']['added_observers'][dev.THIS],
        'original qualified device-reader source must remain exact and recoverable')
    proof = dict(n['observer_admission'], device_key=device.KEY,
        device_evidence_sha256=original.signature(prior),
        qualified_device_observer_source=admission,
        qualified_device_observer_sha256=source_sha,
        fresh_five_stage_controls=True,
        changed_controller_scope='This new model reader only;fresh five-stage device '
            'evidence and exact original native/helper/fixture/recipe/SDK/ELF inputs. '
            'No historical controller-repair field or earlier model result inherited.',
        earlier_model_result_inherited=False)
    require(expected is None or proof == expected, 'bulk model admission changed')
    n['model_observer_admission'] = proof
    return n


def route(text):
    result = benchmark.fast_route(text)
    values = re.findall(r'^strata QSA mirror bulk: tokens=(\d+) pos0=(\d+) '
        r'segments=(\d+) copies=(\d+) requested_D2H_bytes=(\d+) '
        r'queue=same accepted=1 host_quantization=0$', text, re.M)
    expected = [(8192, pos, 1, 4, 8192 * 1056)
                for _ in range(2) for _ in range(12) for pos in (0, 8192, 16384, 24576)]
    require([tuple(map(int, v)) for v in values] == expected,
        'both actual 32K long-prefills must publish exact bulk mirror ranges')
    result.update(actual_bulk_calls=len(values), accepted_copy_commands=384,
        requested_D2H_bytes=2 * 12 * 32768 * 1056,
        bulk_route_observed=True, physical_bus_traffic_qualified=False)
    return result


def context():
    spec = importlib.util.spec_from_file_location('qsa_bulk_actual_model', original.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.device = device.context()
    ctx.native = native
    # Reuse the existing owned-stage host/execution checks. Preparation below
    # is fresh; it does not enter the old attempt's historical V2 branch.
    ctx.ATTEMPT = 'v2'
    ctx.route = route
    return ctx


def prepare():
    ctx = context()
    require(not ROOT.exists() and KEY not in ctx.device.common.public(), 'fresh model plan only')
    native = ctx.native()
    prior = ctx.validation.committed(ctx.RESULT, (device.KEY,))
    reference = json.loads(ctx.REFERENCE.read_text())
    require(reference['passed'] and reference['complete'] and reference['ownership_closed'],
        'closed original model/configuration assets required')
    argv = list(reference['argv'])
    argv[0] = native['binaries']['strata']['path']
    for flag, value in (('--max-context', '262144'), ('--kv-resident', '20480'), ('--prefill', '8192')):
        require(argv.count(flag) == 1, 'one argument required: ' + flag)
        argv[argv.index(flag) + 1] = value
    old = reference['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(ROOT)) for k, v in reference['environment'].items()}
    require('STRATA_KV_STAGE_OWN' not in env, 'owned stage must be explicit')
    env.update(benchmark.FAST, TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        STRATA_MEASUREMENT_ROOT=str(ROOT), STRATA_KV_STAGE_OWN='1',
        STRATA_SYCL_PREFILL_QSA_STREAM='1', STRATA_SYCL_DECODE_Q6_DPAS_HEAD='2',
        STRATA_SYCL_DECODE_Q6_DPAS_DENSE='1', STRATA_PREFILL_HEAD_HOST_BACKING='0',
        STRATA_TRACE='1', UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'),
        **{component.FLAG: '1'})
    env.pop('UR_LOG_TRACING', None)
    require(env['STRATA_SYCL_PREFILL_NATIVE16'] == 'bf16'
        and env['STRATA_SYCL_PREFILL_NATIVE16_QSA'] == '1'
        and env['STRATA_SYCL_PREFILL_QSA_MATH'] == 'bf16-prefix-ordered-exp'
        and all(env.get(k) == v for k, v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',
            NEOReadDebugKeys='1', EnableDirectSubmission='0', STRATA_PREFILL_RELEASE_HEADS='1',
            STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0').items()), 'preserve typed route and safety')
    owned = ctx.context(native['model_observer_admission'])
    row = dict(stage='prepared',
        source_commit=ctx.device.common.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
        controller=owned.controller_pin(), native_source_commit=native['native_source_commit'],
        native_build=dict(binaries=native['binaries']),
        native_record=dict(path=str(ctx.device.NATIVE / 'record.json'),
            sha256=sha(ctx.device.NATIVE / 'record.json')),
        native_observer_admission=native['model_observer_admission'],
        device_evidence_sha256=ctx.signature(prior), models=reference['models'],
        models_sha256=reference['models_sha256'], runtime_manifest=prior['runtime_manifest'],
        runtime_files=reference['runtime_files'], probe=prior['native_validation_probe'],
        generator_runtime=ctx.previous.prompts.generator_runtime(), tokenizer=reference['tokenizer'],
        requests=ctx.requests(Path(reference['tokenizer'])), environment=env, argv=argv,
        boot=prior['boot'], driver=prior['driver'],
        capacity=dict(full_stage_payload_bytes=262144 * 1056,
            logical_host_KV_payload_bytes=12 * 262144 * 1056,
            stage_allocation='Separately owned264MiB;actual8192chunk still required.'),
        budgets=dict(wall=1200, cpu=14400, rss_cap=104 << 30,
            text_cap=32 << 20, file_cap=32 << 20, total_cap=128 << 20),
        plan='Host INFO/stage checks,then one fresh thinking-enabled32769/82/32769input '
            'process with32outputs each. Candidate56cad9/native611a0fd6 and five same-build '
            'fixtures;logical262144/resident20480INT8,own264MiBstage,8192chunks,1256decode '
            'slots,spec4,head2/dense1/Hostbacking0,nine original FAST settings and bulk mirror1. '
            'Require actual96bulk calls/384copies/830472192requested bytes,full fast-route '
            'bank/MTP restoration and independent decode recapture,READY/INFO/source/provider/'
            'protocol/quiet-validation/normal0/closed-owner/unchanged-boot/no-fault checks. '
            'Full15model hashes before/after. No retries,pin refresh or environment mutation.',
        limits='Actual candidate route/handoff diagnostic only;validation/logging charged. '
            'No clean PP/TG/TTFT,gain,task quality,sustained decode,physical262144model/tails/'
            'RESTORE,ordinary-provider teardown or adoption. Follow with matched repeated '
            'same-binary off/on32K/64Kbench.sh and separate independent decode evidence.',
        results=[], commands=[], passed=False, complete=False, active=False,
        ownership_closed=True, model_executed=False, speed_eligible=False,
        model_adoption_qualified=False,
        retention='Keep compact actual dispatch/handoff and original identities. Retire '
            'successful full stage/UR logs and generated IDs after reduction. Four named '
            'temporary engine generations are unchanged;select candidate or b9aa at '
            'matched phase closure and retire superseded build after source recovery.')
    owned.assets(row, full=True)
    ROOT.mkdir(mode=0o700)
    for part in ('tmp', 'cache'):
        (ROOT / part).mkdir(mode=0o700)
    owned.save(row)
    print('Prepared actual bulk mirror long/short/long diagnostic;commit before host/model.', flush=True)


if __name__ == '__main__':
    require(sys.executable == PYTHON,
        'Use the existing pinned prompt-generator interpreter: ' + PYTHON)
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host', 'run'),
        'qsa_bulk_model_entry.py prepare|host|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            context().execute(sys.argv[1])
