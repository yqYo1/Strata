#!/usr/bin/env python3
"""Actual production handoff for the naming-only KV unit; Main owns execution.

Uses this fork's unchanged model protocol, routes and lifecycle readers.
Repository LICENSE. No external implementation or earlier GPU result copied.
"""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import check_named_kv_oracle as oracle
import coherent_prefill_model as coherent
import decode_kv_model as previous

native_device, common = oracle.native, oracle.common
P, B, RESULT = oracle.P, oracle.B, oracle.RESULT
ROOT, KEY = B / 'phase-kv-naming-model-v1', 'phase_kv_naming_model_v1'
THIS = 'tools/sycl/check_named_kv_model.py'
require, sha, signature = oracle.require, oracle.sha, oracle.signature


def native(expected=None):
    n = native_device.native(observers={*oracle.OBSERVERS, THIS})
    gate = common.validation.committed(RESULT, (oracle.KEY,))
    require(gate['stage'] == 'closed' and not gate['active']
            and all(gate[k] for k in ('passed', 'complete', 'ownership_closed'))
            and [c['mode'] for c in gate['cases']] == ['smoke', 'full']
            and all(c['passed'] and c['complete'] and c['ownership_closed']
                    and c['native_exit_code'] == 0 and c['oracle_exact']
                    and c['fresh_faults'] == [] for c in gate['cases'])
            and gate['binary'] == n['binaries']['sycl_kv_copy_groups_bench']
            and gate['native_source_commit'] == n['native_source_commit'],
            'closed same-build traced/full KV oracle required')
    prior = common.validation.committed(RESULT, (native_device.KEY,))
    require(signature(prior) == gate['native_gate_sha256']
            and prior['stage'] == 'closed' and prior['passed'] and prior['complete']
            and prior['ownership_closed'] and not prior['active']
            and prior['binaries'] == n['binaries']
            and [r['mode'] for r in prior['results']] == list(native_device.STAGES),
            'original eleven-stage numerical qualification changed')
    proof = copy.deepcopy(n['observer_admission'])
    current = common.source.pin(P, common.PREFIXES)['source_sha256']
    require(THIS in proof['added_observers'] and THIS in current,
            'declare this uncompiled model observer')
    del proof['added_observers'][THIS]; del current[THIS]
    proof['observer_source_files'] -= 1
    proof['observer_source_sha256'] = signature(current)
    require(proof == gate['native_observer_admission'],
            'only this committed uncompiled observer may extend the oracle admission')
    proof = dict(n['observer_admission'], oracle_key=oracle.KEY,
        oracle_evidence_sha256=signature(gate), device_key=native_device.KEY,
        device_evidence_sha256=signature(prior), model_observer=THIS,
        native_or_model_result_inherited=False)
    require(expected is None or proof == expected, 'frozen model admission changed')
    return dict(n, model_observer_admission=proof)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def controller():
    module = load('named_kv_model_protocol', coherent.original.__file__)
    module.ROOT, module.KEY, module.THIS, module.ATTEMPT = ROOT, KEY, THIS, 'v2'
    def context(admission=None):
        ctx = load('named_kv_model_owned', module.previous.__file__)
        ctx.ROOT, ctx.KEY = ROOT, KEY
        ctx.build = SimpleNamespace(**{**vars(common), 'ROOT': native_device.build.ROOT,
            'KEY': native_device.build.KEY})
        ctx.device = SimpleNamespace(KEY=oracle.KEY)
        ctx.native = lambda: native(admission)
        ctx.controller_pin = lambda: common.source.pin(P,
            ['tools/sycl', 'tools/strata_tokenizer.py', 'data/expert-profile.bin'])
        ctx.requests = module.requests
        return ctx
    def route(text):
        result = coherent.route(text)
        result['decode_KV'] = previous.kv_evidence(text)
        return result
    module.context, module.native, module.route = context, native, route
    return module


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),
            'one fresh naming-only model plan')
    n = native(); module = controller(); ctx = module.context(n['model_observer_admission'])
    old = json.loads((previous.ROOT / 'record.json').read_text())
    require(common.validation.committed(RESULT, (previous.KEY,)) == ctx.compact(old)
            and old['stage'] == 'closed' and old['passed'] and old['complete']
            and old['ownership_closed'] and not old['active'],
            'exact closed owned model configuration required')
    env = {k: v.replace(str(previous.ROOT), str(ROOT)) for k, v in old['environment'].items()}
    require(env['STRATA_SYCL_DECODE_KV_COALESCED'] == '1'
            and env['STRATA_SYCL_KV_COPY_GROUPS'] == '0'
            and env['STRATA_SYCL_PREFILL_GU_SWIGLU'] == 'large'
            and env['STRATA_SYCL_PREFILL_MOE_CENSUS'] == '1'
            and env['STRATA_PREFILL_LAYER_MAJOR_PROFILE'] == '0',
            'preserve the actual qualified production handoff configuration')
    require(ctx.requests(Path(old['tokenizer'])) == old['requests'],
            'preserve thinking-enabled generated inputs')
    row = {k: copy.deepcopy(old[k]) for k in ('models', 'models_sha256', 'runtime_manifest',
        'runtime_files', 'probe', 'generator_runtime', 'tokenizer', 'requests', 'boot', 'driver', 'budgets')}
    gate = common.validation.committed(RESULT, (oracle.KEY,))
    row.update(stage='prepared', source_commit=common.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
        controller=ctx.controller_pin(), native_source_commit=n['native_source_commit'],
        native_build=dict(binaries=n['binaries']), native_record=gate['native_record'],
        native_observer_admission=n['model_observer_admission'],
        device_evidence_sha256=signature(gate), environment=env,
        argv=[n['binaries']['strata']['path'], *old['argv'][1:]],
        controller_python=dict(path=sys.executable, prefix=sys.prefix, sha256=sha(sys.executable)),
        controller_commands=[[coherent.PYTHON, '-B', THIS, action] for action in coherent.ACTIONS],
        plan='Commit exact controller/configuration/assets;host INFO/stage and literal KV-route controls;'
             'then one fresh naming-only6d2b4fengine process with thinking-on32769/82/32769inputs and32outputs.'
             'Preserve all old placement,8192chunk,MTPspec4,head2/dense1,large-GU and immutable safety settings.'
             'Require two actual scalar-prefill/cooperative-decode4 handoffs,typed QSA/GU/Down/job conservation,'
             'MTPrestore/cache/recapture,normal0,quiet validation,no fresh fault,closed ownership and15asset'
             'hashes before/after. Stop first failure,no retry or changed pins.',
        limits='Production route/state handoff only. Static ELF comparison and this check do not establish'
               'sustained TG,clean PP/TTFT,task/agent quality,physical262144fill/tails/RESTORE,ordinary'
               'teardown or default adoption. The original naming-unit performance obligations remain unchanged.',
        retention='No new engine generation. Keep compact routing/source/closure outcomes for the naming'
                  'unit decision;retire successful stage logs/cache after verification. Preserve both'
                  'phase references until the pending per-unit comparison closes.',
        results=[], commands=[], passed=False, complete=False, active=False,
        ownership_closed=True, model_executed=False)
    ctx.assets(row, full=True)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir(mode=0o700)
    ctx.save(row)
    print('Prepared same-build naming-only production handoff;commit before host/model.', flush=True)


if __name__ == '__main__':
    require(Path(sys.prefix) == Path(coherent.PYTHON).parent.parent, 'use qualified model venv')
    require(len(sys.argv) == 2 and sys.argv[1] in coherent.ACTIONS, 'prepare|host|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); common.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] == 'host':
            controls = previous.controls(); module = controller(); module.execute('host')
            row = json.loads((ROOT / 'record.json').read_text()); row['KV_host_controls'] = controls
            module.context(row['native_observer_admission']).save(row)
        else:
            row = json.loads((ROOT / 'record.json').read_text())
            require(row['KV_host_controls']['passed'], 'literal KV controls required')
            controller().execute('run')
