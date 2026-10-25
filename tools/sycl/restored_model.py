#!/usr/bin/env python3
"""Read the closed restored-model state gate for independent phase comparisons.

Repository LICENSE. Main-authored from this fork's qualification interfaces.
The single-use launcher and exact diagnostic plan recover from7d065a48f/deea7d31b;
do not relabel their native or controller source with this later reader revision.
State qualification does not qualify performance,quality or physical262144.
"""
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import prefill_restore as native
import full_context_model as models

gen, guards = native.gen, native.guards
P, B = native.P, native.B
ROOT = B / 'restored-model-state-v2'
RESULT = gen.A / 'prefill-restore-model-current.json'
COMPONENT = gen.A / 'prefill-restore-device-current.json'
digest, fingerprint = native.build.digest, native.native.fingerprint


def qualification(revision):
    """Check original evidence and unchanged native bytes under a new controller."""
    row = json.loads((ROOT / 'record.json').read_text())
    public = {k: v for k, v in row.items() if k not in ('model_assets', 'environment', 'commands')}
    guards.require(gen.committed(RESULT) == public and row['stage'] == 'model-closed' and
        row['passed'] and row['complete'] and row['ownership_closed'] and not row['active'],
        'closed exact real-model state qualification required')
    for stage, gpu in (('protocol', False), ('observation', True)):
        outcome = row[stage]
        guards.require(outcome['passed'] and outcome['ownership_closed'] and outcome['exit_code'] == 0 and
            outcome['GPU_executed'] is gpu and outcome['boot_unchanged'] and outcome['fresh_faults'] == [] and
            outcome['driver_after'] == row['driver'], 'model/protocol closure differs')
    guards.require(row['verification'] == dict(state_and_restore_replay_matches=True,
        parameter_validation_bytes=0, final_model_hashes_matched=15,
        actual_source_runtime_and17_assets_rechecked=True, mapped_runtime_boundaries=5),
        'required model/source/runtime replay evidence differs')
    pins = gen.benchmark_pins(row['controller_source_commit'], match_worktree=False)
    guards.require(len(pins) == row['controller_files'] and fingerprint(pins) == row['controller_fingerprint'],
        'original diagnostic controller inputs cannot recover')
    launch = row['launch_plan']
    raw = gen.git('show', launch['commit'] + ':' + launch['path'])
    guards.require(hashlib.sha256(raw).hexdigest() == launch['sha256'], 'original model launch plan cannot recover')
    plan = json.loads(raw)
    guards.require(plan['stage'] == 'protocol-closed' and plan['observation'] is None and
        all(plan[k] == row[k] for k in ('controller_source_commit', 'controller_fingerprint',
            'native_source_commit', 'binary', 'component_sha256', 'args', 'environment_fingerprint',
            'model_fingerprint', 'profile_sha256', 'requests', 'boot', 'driver', 'budgets', 'criteria')),
        'model state result changed its frozen launch inputs')
    outcomes = row['observation']['requests']
    guards.require(len(outcomes) == 4 and
        [(s['prompt'], s['reused'], s['read'], s['prefill_path'], s['dense_activations']) for s in outcomes] ==
        [(33,0,33,'chunk-major',0), (16385,0,16385,'layer-major',48),
         (33,0,33,'chunk-major',0), (16401,16385,16,'chunk-major',0)] and
        all(s['cache_slots_restored'] == 1256 and s['cache_payload_bytes'] == 2575206400 and
            s['state_L'] == s['prompt'] - 1 + s['windows'] + s['accepted'] for s in outcomes),
        'independent short/long handoff or cache/state conservation differs')
    restore = row['observation']['restore']
    guards.require(restore['result'] == 'PASS' and restore['upto'] == '16385' and
        restore['main_layers'] == '12' and all(int(restore[k]) > 0 for k in
            ('main_bytes', 'draft_bytes', 'running_bytes')), 'complete live RESTORE proof missing')
    components = gen.committed(COMPONENT)
    guards.require(digest(COMPONENT) == row['component_sha256'] and
        components['passed'] and components['complete'] and components['ownership_closed'] and
        not components['active'] and components['verification']['restore_smoke_checks'] == 113 and
        components['final_checks']['fresh_faults'] == [], 'same-build component gate differs')
    build = json.loads((native.ROOT / 'record.json').read_text())
    native.current(build, controller_revision=revision)
    guards.require(build['source_commit'] == components['build_source_commit'] == row['native_source_commit'] and
        build['assets']['strata'] == components['binary'] == row['binary'] and
        fingerprint(build['assets']) == components['native_build_reference']['assets_fingerprint'] and
        digest(native.ROOT / 'record.json') == components['native_build_reference']['private_record_sha256'] and
        digest(native.runtime.RESULT) == components['runtime_reference']['sha256'],
        'qualified17 native artifacts or runtime differ')
    base = models.setup()
    guards.require(base['reference'] == row['base_reference'] and base['models'] == row['model_assets'] and
        len(row['model_assets']) == row['model_files'] == 15 and
        fingerprint(row['model_assets']) == row['model_fingerprint'] and
        fingerprint(row['environment']) == row['environment_fingerprint'] and
        digest(P / 'data/expert-profile.bin') == row['profile_sha256'], 'model/profile/environment identity differs')
    models.model_pins(base['config'], row['model_assets'])
    guards.require(not any(row[k] for k in ('production_selected', 'speed_qualified',
        'decode_preservation_qualified', 'physical262144_qualified', 'ordinary_runtime_teardown_qualified')),
        'state-only qualification must not silently become inference adoption')
    return row, components
