#!/usr/bin/env python3
"""Owned current-build collector admission for streamed-QSA16 tuning.

Main-authored composition of this fork's checked readers; repository LICENSE.
Keep the original native build and model-result identities. A collector control
does not establish throughput, model coverage or a physical resource ceiling.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import qsa16_stream_model_entry as model
import profile_selected_compat as compatibility

device = model.device
P, B, RESULT = model.P, model.B, model.RESULT
THIS = 'tools/sycl/qsa16_stream_profile.py'
ROOT = B / 'qsa16-stream-collector-active-v1'
KEY = 'qsa16_stream_collector_active_v1'
MODEL_ROOT = B / 'qsa16-stream-model-entry-v2'
MODEL_KEY = 'qsa16_stream_model_entry_v2'
require, sha, signature = model.require, model.sha, model.signature


def basis():
    """Admit only this added observer, with all five native ELFs unchanged."""
    n = device.native(observers={device.THIS, model.THIS, THIS})
    prior = json.loads((MODEL_ROOT / 'record.json').read_text())
    require(model.validation.committed(RESULT, (MODEL_KEY,)) == model.previous.compact(prior)
        and prior['stage'] == 'closed' and all(prior[k] for k in
            ('passed', 'complete', 'ownership_closed', 'model_executed'))
        and not prior['active'] and len(prior['results']) == 1
        and prior['native_source_commit'] == n['native_source_commit']
        and prior['native_build']['binaries'] == n['binaries']
        and sha(prior['native_record']['path']) == prior['native_record']['sha256'],
        'closed same-native actual8192 streamed model proof required')
    result = prior['results'][0]
    require(result['native_exit_code'] == 0 and result['ownership_closed']
        and result['fresh_faults'] == [] and len(result['requests']) == 3
        and result['capacity_observation']['actual_chunk'] == 8192
        and result['routes']['actual_mode'] == 1 and result['routes']['typed_QSA_entry'],
        'actual model route and normal fault-free closure required')
    original = prior['controller']
    recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in
        device.common.source.blobs(P, original['revision'], original['source_sha256'])}
    current = device.common.source.pin(P,
        ['tools/sycl', 'tools/strata_tokenizer.py', 'data/expert-profile.bin'])
    before, after = original['source_sha256'], current['source_sha256']
    require(recovered == before and signature(before) == original['aggregate_sha256']
        and set(after) - set(before) == {THIS} and not set(before) - set(after)
        and all(after[p] == h for p, h in before.items()),
        'only this committed observer may extend the original model controls')
    closed_device = model.validation.committed(RESULT, (device.KEY,))
    require(signature(closed_device) == prior['device_evidence_sha256']
        and closed_device['passed'] and closed_device['complete']
        and closed_device['ownership_closed'] and not closed_device['active']
        and closed_device['binaries'] == n['binaries'], 'same-build device evidence changed')
    require(sha(prior['runtime_manifest']['path']) == prior['runtime_manifest']['sha256']
        and all(sha(p) == h for p, h in prior['runtime_files'].items())
        and model.guards.driver_guard() == prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot'],
        'qualified runtime/boot/driver changed')
    model.previous.assets(prior)
    proof = dict(native=n['observer_admission'], original_model_source=prior['source_commit'],
        original_model_evidence_sha256=signature(model.previous.compact(prior)),
        original_model_receipt_sha256=sha(MODEL_ROOT / 'record.json'),
        original_model_controls_recovered=len(before),
        added_observer={THIS: after[THIS]},
        meaning='Controller-only addition;original native/recipe/five fixtures/runtime and model evidence exact. No native rebuild or inherited collector qualification.')
    assets = compatibility.profile.collector_assets(json.loads(RESULT.read_text()))
    return (dict(n, source_commit=n['native_source_commit']),
        dict(prior, native_validation_probe=prior['probe'], observer_admission=proof),
        prior['runtime_files'], assets)


def context():
    ctx = device.load_module('qsa16_stream_active_collector', compatibility.__file__)
    ctx.ROOT = ctx.CONTROL_ROOT = ROOT
    ctx.KEY = ctx.CONTROL_KEY = KEY
    ctx.SOURCE = P / THIS
    ctx.ACTIVE_FROM_EXEC = True
    ctx.basis = basis

    def current(row):
        require(ctx.base.committed(RESULT)[KEY] == ctx.compact(row)
            and ctx.base.controls(row['source_commit'], controller=P / THIS) == row['sources'],
            'commit exact collector inputs and prospective plan before execution')
        n, prior, providers, assets = basis()
        require(row['native_source_commit'] == n['native_source_commit']
            and row['engine'] == n['binaries']['strata']
            and row['binary'] == n['binaries']['sycl_submission_test']
            and row['observer_admission'] == prior['observer_admission']
            and row['native_validation_probe'] == prior['probe']
            and row['environment_template'] == prior['environment']
            and row['boot'] == prior['boot'] and row['driver'] == prior['driver']
            and row['assets'] == assets and row['budgets'] == ctx.BUDGET
            and all(row['profile'][k] == v for k, v in assets.items())
            and row['profile']['environment']['UNITRACE_TraceOutputDir'] == str(ROOT / 'trace')
            and row['profile']['mode'] == 'aggregate' and row['whole_bytes'] == 128 << 20,
            'native/fixture/controller/runtime/collector envelope differs')
        return providers

    ctx.current = current
    return ctx


def prepare():
    ctx = context()
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    _, prior, _, _ = basis()
    row['observer_admission'] = prior['observer_admission']
    row['plan'].update(
        native='Current b9aa9c4/1e598e8 native build and all five exact fixture ELFs. Reuse the existing MIT Intel/PTI indexed collector and owned Sysman0 launcher;no rebuild or other-engine collector inheritance.',
        fixture='No-argument tests/sycl/compat.cpp:1direct+12first+24host-handoff+24cross-queue+1late=62 kernels. Both startup gates,ordinary graph recording,replay,event/copy and lifetime assertions are required.',
        order='After closed8192 mode1 long/short/long model admission,one fresh active-from-exec compatibility fixture. Existing aggregate mode,ChromeKernelLogging1,128-event buffer;Sysman0 and all qualified runtime safety remain immutable before exec.',
        acceptance='Exact committed source/recipe/fiveELFs/providers,pre-main parameter validation,numerical/event/copy/lifetime assertions,normal0,closed owner,unchanged boot/no fresh fault. Require62named kernels,unique native host-flow joins,flushed finite-clock trace and removed session. Stop first failure;no retry or pin refresh.',
        next='If this current-build control passes,commit a representative32K+ prefilling window and matched overhead controls before collecting model attribution. Obtain clean bench.sh multi-length PP/TTFT separately. Model classification must distinguish service-time coverage from saturation and requested bytes from measured bus traffic.',
        limits='Small collector compatibility only;not representative model trace,observer overhead,clean PP/TG,physical traffic/saturation,numerical model quality,262144fill/RESTORE or inference adoption.')
    ctx.save(row)
    print('Prepared current streamed-QSA16 collector control;commit before GPU.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['check']), 'use prepare|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else context().check()
