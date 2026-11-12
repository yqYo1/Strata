#!/usr/bin/env python3
"""Owned actual-model handoff for the coherent full build.

Main-authored from this fork's maintained model controls; repository LICENSE.
This checks routes/state,not clean speed,task quality or full model context.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_coherent_prefill as device
import moe_host_census_model as reference
import qsa16_stream_model_entry as original
import gu_small_bench as dispatch_reader

P, B, RESULT = device.P, device.B, device.RESULT
ROOT, KEY = B/'coherent-prefill-model-v2', 'coherent_prefill_model_v2'
THIS = 'tools/sycl/coherent_prefill_model.py'
PYTHON = '/home/yayoi/.local/share/strata-sycl/venv/bin/python3'
require, sha = device.require, device.sha


def native(expected=None, *, observers=()):
    extras = {THIS, *observers}
    require(all(p.startswith('tools/sycl/') and p.endswith('.py') for p in extras),
        'only named committed model observers may extend the native admission')
    n = device.native(observers={device.THIS, *extras})
    prior = original.validation.committed(RESULT, (device.KEY,))
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete']
        and prior['ownership_closed'] and not prior['active']
        and prior['binaries'] == n['binaries'] and prior['native_source_commit'] == n['native_source_commit']
        and [v['mode'] for v in prior['results']] == list(device.STAGES)
        and all(v['passed'] and v['ownership_closed'] and v['native_exit_code'] == 0
            for v in prior['results']), 'all eleven fresh coherent native stages required')
    proof = copy.deepcopy(n['observer_admission'])
    after = original.device.common.source.pin(P, original.device.common.PREFIXES)['source_sha256']
    require(original.device.common.signature(after) == proof['observer_source_sha256'],
        'actual native observer source census differs')
    for name in extras:
        require(name in proof['added_observers'] and name in after, 'explicit model observer missing')
        del proof['added_observers'][name]; del after[name]; proof['observer_source_files'] -= 1
    proof['observer_source_sha256'] = original.device.common.signature(after)
    require(proof == prior['native_observer_admission'],
        'recover the exact qualified native/controller inputs before adding model observers')
    host = prior['results'][0]
    require(host['reader_cases'] == 54 and host['native_cases'] == 11 and host['policy_check']['passed']
        and host['host_census_controls']['passed'] and host['physical_context_host']['passed']
        and host['selected_head_IR']['same_as_current_reference']
        and host['selected_head_IR']['engine_fixture_identical'], 'fresh host and decode-module gates required')
    require(all(v['GPU_executed'] and v['native_execution_attempted'] and v['host_probe_accepted']
        and not v['fresh_faults'] and v['boot_unchanged'] and v['driver_after'] == prior['driver']
        for v in prior['results'][1:]), 'normal validated device closures without faults required')
    qsa = prior['results'][6]['checks']
    require(len(qsa) == 1 and qsa[0]['name'] == 'numerical'
        and sum(s.startswith('case cells=') for s in qsa[0]['cases']) == 11
        and any(s.startswith('case cells=262144 live_cells=262144 ') for s in qsa[0]['cases']),
        'fresh physical attention-component boundary required')
    producer = prior['results'][8]['checks']
    require([v['flag'] for v in producer] == ['0', '1']
        and all(v['physical_cells'] == 262144 for v in producer)
        and producer[1]['bulk_calls'] == 33 and producer[1]['accepted_copy_commands'] == 136,
        'fresh full producer and final-cell overwrite required')
    numeric = prior['results'][-1]['numerics']
    require(numeric['correctness_cases'] == 44 and numeric['max_half_ulp'] <= 1
        and numeric['BF16_capture_exact'] and numeric['guards'] and numeric['inputs_immutable'],
        'fresh same-build GU numerical/dispatch boundary required')
    proof = dict(n['observer_admission'], device_key=device.KEY,
        device_evidence_sha256=original.signature(prior), fresh_eleven_stage_controls=True,
        model_observer=THIS, additional_observers=sorted(set(observers)),
        original_partial_relink_or_model_result_inherited=False)
    require(expected is None or proof == expected, 'coherent model native admission changed')
    return dict(n, model_observer_admission=proof)


def route(text):
    result = reference.route(text)
    require(text.splitlines().count('[native16-gu] fusion=large min_rows=2334') == 1,
        'actual coherent model must initialize the selected large-GU policy')
    groups = [re.findall(pattern, text, re.M) for pattern in
        (r'^strata MoE GU fusion:.*$', r'^strata MoE Down fusion:.*$',
         r'^strata MoE accumulator window:.*$')]
    require(all(len(v) == 2 for v in groups), 'two actual long-entry dispatch aggregates required')
    result['actual_GU_dispatch'] = [dispatch_reader.dispatch('\n'.join(lines)+'\n', 'large')
        for lines in zip(*groups)]
    result['GU_startup_policy'] = 'large:wide>=2334;separate1..2333'
    result['actual_GU_returned_outcome_aggregates_observed'] = True
    result['GU_scope'] = 'Actual returned-boolean aggregates after checked drains;not per-job traces or speed.'
    return result


def context():
    spec = importlib.util.spec_from_file_location('coherent_actual_model', original.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.device = device.context().context()
    ctx.native, ctx.route, ctx.ATTEMPT = native, route, 'v2'
    return ctx


def prepare():
    ctx = context()
    require(not ROOT.exists() and KEY not in ctx.device.common.public(), 'fresh coherent model plan required')
    n = native(); prior = original.validation.committed(RESULT, (device.KEY,))
    old = json.loads((reference.ROOT/'record.json').read_text())
    require(original.validation.committed(RESULT, (reference.KEY,)) == original.previous.compact(old)
        and old['stage'] == 'closed' and old['passed'] and old['complete']
        and old['ownership_closed'] and not old['active'], 'exact owned reference configuration required')
    env = {k: v.replace(old['environment']['STRATA_MEASUREMENT_ROOT'], str(ROOT))
        for k, v in old['environment'].items()}
    require(env['STRATA_SYCL_PREFILL_GU_SWIGLU'] == 'large'
        and env['STRATA_SYCL_PREFILL_MOE_CENSUS'] == '1'
        and env['STRATA_PREFILL_LAYER_MAJOR_PROFILE'] == '0', 'preserve qualified large-GU diagnostic policy')
    argv = list(old['argv']); argv[0] = n['binaries']['strata']['path']
    require(ctx.requests(Path(old['tokenizer'])) == old['requests'], 'preserve thinking-enabled workloads')
    owned = ctx.context(n['model_observer_admission'])
    row = dict(stage='prepared', source_commit=ctx.device.common.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
        controller=owned.controller_pin(), native_source_commit=n['native_source_commit'],
        native_build=dict(binaries=n['binaries']), native_record=dict(path=str(device.build.ROOT/'record.json'),
            sha256=sha(device.build.ROOT/'record.json')), native_observer_admission=n['model_observer_admission'],
        device_evidence_sha256=original.signature(prior), models=old['models'], models_sha256=old['models_sha256'],
        controller_python=dict(path=sys.executable, prefix=sys.prefix, sha256=sha(sys.executable)),
        previous_preparation=dict(evidence_commit='3a977d13fe9c3d602b6ca2d81589506445d66749',
            key='coherent_prefill_model_v1', stage='preparation_failed', GPU_executed=False,
            correction='Use the qualified model venv,not system Python;native/workload/route criteria unchanged.'),
        runtime_manifest=prior['runtime_manifest'], runtime_files=old['runtime_files'],
        probe=prior['native_validation_probe'], generator_runtime=ctx.previous.prompts.generator_runtime(),
        tokenizer=old['tokenizer'], requests=copy.deepcopy(old['requests']), environment=env, argv=argv,
        boot=prior['boot'], driver=prior['driver'], budgets=dict(old['budgets']),
        plan='Literal host INFO/stage controls,then a fresh thinking-enabled32769/82/32769-input '
             'process with32outputs each. Only the engine/fixture build changes to coherent14508;'
             'retain placement,MTP,head2/dense1,FAST9,bulk1,source-view/private-map,staging0,census1 '
             'and profile0. Require exact actual wide-GU/Down/job conservation,typed QSA/bulk ranges,'
             'cache bindings,MTP expert restore and decode recapture;normal0,quiet validation,'
             'closed ownership,no fresh fault and15full model hashes before/after. Stop first '
             'failure;no retry or pin refresh. Then commit a separate repeated32K/64K speed plan.',
        limits='Actual route/state handoff only. Native/component success is not model task quality,'
             'sustainedTG,clean PP/TTFT,physical262144/tails/RESTORE or inference adoption. '
             'Historical partial-relink observations stay historical even though final ELF bytes match.',
        retention='Keep compact route/count/numerical/fault evidence for the imminent coherent '
             'prefill comparison;retire successful traces and temporary caches after verified closure. '
             'No new engine generation. Migrate/retire obsolete partial-build proof readers separately.',
        results=[], commands=[], passed=False, complete=False, active=False, ownership_closed=True,
        model_executed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT/name).mkdir(mode=0o700)
    owned.save(row)
    print('Prepared coherent long/short/long model handoff;commit before execution.', flush=True)


if __name__ == '__main__':
    require(Path(sys.prefix) == Path(PYTHON).parent.parent,
        'Run this model controller with '+PYTHON+';its tokenizer requires the qualified regex environment')
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host', 'device'), 'prepare|host|device')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            context().execute(sys.argv[1])
