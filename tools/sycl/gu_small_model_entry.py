#!/usr/bin/env python3
"""Actual-model handoff check for the prefill-only compact-GU candidate.

Main-authored from the maintained owned Strata model controls; repo LICENSE.
This is an observed-route diagnostic, not a clean-speed comparison.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import build_gu_small_model as build
import check_gu_small_model as device
import moe_host_census_model as reference
import qsa16_stream_model_entry as original

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B/'gu-small-model-entry-v1', 'gu_small_model_entry_v1'
THIS = 'tools/sycl/gu_small_model_entry.py'
PYTHON = '/home/yayoi/.local/share/strata-sycl/venv/bin/python3'
require, sha = build.require, build.sha


def native(expected=None):
    n = build.native(observers={THIS})
    prior = original.validation.committed(RESULT, (device.KEY,))
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete']
        and prior['ownership_closed'] and not prior['active']
        and prior['binaries'] == n['binaries']
        and prior['native_source_commit'] == n['native_source_commit']
        and [r['mode'] for r in prior['results']] == list(device.STAGES)
        and all(r['passed'] and r['ownership_closed'] and r['native_exit_code'] == 0 for r in prior['results']),
        'all nine fresh native prerequisites required')
    proof = copy.deepcopy(n['observer_admission'])
    declared = proof['declared_inputs']
    require(THIS in declared['added_observers'], 'new model observer must be explicit')
    del declared['added_observers'][THIS]; declared['observer_source_files'] -= 1
    require(proof == prior['native_observer_admission'],
        'only this committed model observer may extend native qualification')
    host = prior['results'][0]
    require(host['host_census_controls']['passed'] and host['reader_cases'] == 54
        and host['native_cases'] == 11 and host['policy_check']['passed']
        and host['selected_head_IR']['same_as_current_reference']
        and host['selected_head_IR']['engine_fixture_identical'], 'fresh host and selected-decode contract required')
    require(all(r['GPU_executed'] and r['native_execution_attempted']
        and r['host_probe_accepted'] and r['fresh_faults'] == [] and r['boot_unchanged']
        and r['driver_after'] == prior['driver'] for r in prior['results'][1:]),
        'fresh validated fault-free device closures required')
    qsa = prior['results'][6]['checks']
    require(len(qsa) == 1 and qsa[0]['name'] == 'numerical'
        and sum(s.startswith('case cells=') for s in qsa[0]['cases']) == 11
        and any(s.startswith('case cells=262144 live_cells=262144 ') for s in qsa[0]['cases']),
        'physical262144 attention component boundary required')
    producer = prior['results'][8]['checks']
    require([r['flag'] for r in producer] == ['0', '1']
        and all(r['physical_cells'] == 262144 for r in producer)
        and producer[1]['bulk_calls'] == 33 and producer[1]['accepted_copy_commands'] == 136,
        'physical full producer and final-cell overwrite required')
    proof = dict(n['observer_admission'], device_key=device.KEY,
        device_evidence_sha256=original.signature(prior), model_observer=THIS,
        component_result_key='gu_compact_device_v4', original_model_result_inherited=False)
    require(expected is None or expected == proof, 'model native admission changed')
    return dict(n, model_observer_admission=proof)


def route(text):
    result = reference.route(text)
    require(text.splitlines().count('[native16-gu] fusion=auto compact_rows=1..96 wide_min_rows=2334') == 1,
        'actual model backend must initialize the exact compact-GU candidate')
    result['GU_startup_policy'] = 'auto:compact1..96;wide>=2334;separate97..2333'
    result['actual_GU_per_job_outcomes_observed'] = False
    result['GU_scope'] = ('Actual production binding in component evidence;model startup policy and routed '
        'job/upload census here. Per-job fused decisions are not separately observed in the model;'
        'no clean timed-dispatch or model speed/adoption claim.')
    return result


def context():
    spec = importlib.util.spec_from_file_location('compact_gu_actual_model', original.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.device = device.context().context()
    ctx.native, ctx.route, ctx.ATTEMPT = native, route, 'v2'
    return ctx


def prepare():
    ctx = context()
    require(not ROOT.exists() and KEY not in ctx.device.common.public(), 'fresh model plan only')
    n = native()
    prior = original.validation.committed(RESULT, (device.KEY,))
    old = json.loads((B/'moe-host-census-model-v1/record.json').read_text())
    require(original.validation.committed(RESULT, (reference.KEY,)) == original.previous.compact(old)
        and old['stage'] == 'closed' and old['passed'] and old['complete']
        and old['ownership_closed'] and not old['active'], 'exact closed reference configuration required')
    env = {k: v.replace(old['environment']['STRATA_MEASUREMENT_ROOT'], str(ROOT))
        for k, v in old['environment'].items()}
    require(env['STRATA_SYCL_PREFILL_GU_SWIGLU'] == 'large'
        and env['STRATA_SYCL_PREFILL_MOE_CENSUS'] == '1'
        and env['STRATA_PREFILL_LAYER_MAJOR_PROFILE'] == '0', 'preserve original diagnostic except GU policy')
    env['STRATA_SYCL_PREFILL_GU_SWIGLU'] = 'auto'
    argv = list(old['argv']); argv[0] = n['binaries']['strata']['path']
    require(ctx.requests(Path(old['tokenizer'])) == old['requests'], 'same thinking-enabled workload required')
    owned = ctx.context(n['model_observer_admission'])
    row = dict(stage='prepared', source_commit=ctx.device.common.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
        controller=owned.controller_pin(), native_source_commit=n['native_source_commit'],
        native_build=dict(binaries=n['binaries']),
        native_record=dict(path=str(build.ROOT/'record.json'), sha256=sha(build.ROOT/'record.json')),
        native_observer_admission=n['model_observer_admission'],
        device_evidence_sha256=original.signature(prior), models=old['models'], models_sha256=old['models_sha256'],
        runtime_manifest=prior['runtime_manifest'], runtime_files=old['runtime_files'], probe=prior['native_validation_probe'],
        generator_runtime=ctx.previous.prompts.generator_runtime(), tokenizer=old['tokenizer'],
        requests=copy.deepcopy(old['requests']), environment=env, argv=argv, boot=prior['boot'], driver=prior['driver'],
        budgets=dict(old['budgets']),
        plan='First literal host INFO/stage checks,then one fresh thinking-enabled32769/82/32769input process with32outputs each. Preserve original model/runtime/placement/MTP/head2/dense1/FAST9/bulk1/source-view/staging0/census1/profile0 inputs;change only the one-archive model and immutable GUlarge->auto. Require actual policy marker,full original upload/job/QSA/bulk/MTP restore/recapture route,normal0,quiet validation,closed owner,unchanged boot/driver,no fresh fault,15full model hashes before/after. No retries or pin refresh.',
        limits='Observed model route/handoff only;no per-job fused model telemetry,clean PP/TG/TTFT or engine speedup. Full-model physical262144/RESTORE,task/agent,numerical/task acceptance,independent sustained-decode preservation and adoption remain open. Eight unchanged native fixtures do not exercise the changed GU archive;the actual GU component has independent44-case evidence.',
        retention='Retain compact extents,dispatch/handoff and original identities for imminent32K/64Kcomparison. Retire normal UR/stage output after reduction;select candidate or fallback after matched evidence. Keep56cad9 and a347 for their named phase comparisons.',
        results=[], commands=[], passed=False, complete=False, active=False, ownership_closed=True,
        model_executed=False, speed_eligible=False, model_adoption_qualified=False)
    owned.assets(row, full=True)
    ROOT.mkdir(mode=0o700)
    for part in ('tmp', 'cache'): (ROOT/part).mkdir(mode=0o700)
    owned.save(row)
    print('Prepared actual compact-GU long/short/long model diagnostic;commit before host/model.', flush=True)


if __name__ == '__main__':
    require(sys.executable == PYTHON, 'Use pinned prompt-generator interpreter: '+PYTHON)
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host', 'run'), 'prepare|host|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        else: context().execute(sys.argv[1])
