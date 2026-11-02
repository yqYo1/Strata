#!/usr/bin/env python3
"""Fresh bench.sh model checks for the worker/map correction; repository LICENSE.

Use the unchanged maintained entry implementation in a private module context.
Only the generation proof, source pinning and scope/root selection are replaced;
no imported shared context, native source or process environment is mutated.
These short-output requests qualify routes/lifecycle,not clean or sustained TG.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_cache_map_model as model
import checked_worker_device as device
import check_prefill_strategy_model as template

base, P, B, RESULT = model.base, model.P, model.B, model.RESULT
SCOPE = 'checked_worker_retirement_contract_20261102'
INDEX = P / 'bench/results/2026-10-15-pr6-recovery-audit/research-index.json'
require, digest, write = base.require, base.digest, base.write


def proof():
    row = json.loads((device.ROOT / 'record.json').read_text())
    native = device.current(row)
    require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
            row['ownership_closed'] and not row['active'] and
            [case['native_exit_code'] for case in row['cases']] == [0, 0, 1] and
            all(case['passed'] and case['complete'] and case['ownership_closed'] and
                case['pending_copy'] and case['payload_equal'] and case['fresh_faults'] == []
                for case in row['cases']) and not row['cases'][-1]['owner_release'],
            'same-source real-copy worker/cache error qualification required')
    review = template.shared.runtime.validation.committed(INDEX, (SCOPE,))
    require(review['status'] == 'reviewed' and review['ownership_closed'] and
            review['worker_done_accepted'] and review['source_correction_required'] is False and
            review['component_test_result_key'] == device.KEY and
            review['source_identity']['native_recipe_commit'] == native['source_commit'] and
            review['source_identity']['engine_sha256'] == native['binaries']['strata']['sha256'] and
            all(digest(path) == sha for path, sha in review['source_identity']['files'].items()),
            'reviewed actual composed source and scoped correction required')
    old = json.loads((model.prior.ROOT / 'record.json').read_text())
    require(base.compact(old) == base.committed(RESULT)[model.prior.KEY] and old['passed'] and
            old['ownership_closed'] and not old['active'] and all(digest(asset['path']) == asset['sha256']
                for asset in old['binaries'].values()), 'preserve the actual adaptive fallback')
    require(len(native['native_sources']) == len(old['native_sources']) + 1 and
            {name for name in native['native_sources']
                if native['native_sources'][name] != old['native_sources'].get(name)} ==
                {*model.CHANGED, model.WORKER_HEADER}, 'unplanned native difference')
    def normalized(value, root):
        return json.dumps(value, sort_keys=True).replace(str(root), '<generation>')
    old_db = json.loads(Path(old['compile_database']['path']).read_text())
    new_db = json.loads(Path(native['compile_database']['path']).read_text())
    require(digest(old['compile_database']['path']) == old['compile_database']['sha256'] and
            normalized(old_db, model.prior.ROOT) == normalized(new_db, model.ROOT) and
            all(normalized(old[name], model.prior.ROOT) == normalized(native[name], model.ROOT)
                for name in ('configure_argv', 'environment')) and
            old['actual_link_inputs']['frozen_inputs'] == native['actual_link_inputs']['frozen_inputs'],
            'unchanged mathematical fixtures require identical normalized compiler/link recipe')
    names = set(native['binaries']) - {'strata'}
    require(len(names) == 8 and all(native['binaries'][name]['sha256'] == old['binaries'][name]['sha256']
        and native['binaries'][name]['bytes'] == old['binaries'][name]['bytes'] for name in names),
        'mathematical fixture bytes changed')
    evidence = base.committed(RESULT)
    for key in ('gdn16_model_component_device_v1', 'gdn16_model_component_speed_v1'):
        prior = evidence[key]
        require(prior['passed'] and prior['complete'] and prior['ownership_closed'] and
                not prior['active'] and not prior['fresh_faults'] and
                prior['binary']['sha256'] == native['binaries'][model.prior.fallback.COMPONENT]['sha256'],
                'original unchanged GDN component numerical/performance evidence required')
    providers = model.prior.fallback.retained.runtime.inventory(row['runtime_manifest'])
    env = dict(row['environment_template'], LD_PRELOAD=native['native_validation_probe']['library']['path'],
               STRATA_UR_PROBE_EXPECTED_LOADER=native['native_validation_probe']['loader']['path'])
    # The template only reads this gate's source and effective provider fields.
    gate = dict(source_commit=row['source_commit'], environment=env)
    return native, gate, providers


def pinned_sources(revision):
    pins = template.pinned_sources(revision)
    pins.update(base.controls(revision, controller=Path(__file__).resolve()))
    return pins


def context(arm):
    require(arm in ('zero', 'positive'), 'choose zero|positive')
    spec = importlib.util.spec_from_file_location('strata_cache_map_entry_context', template.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.configure('adaptive-throughput' if arm == 'zero' else 'adaptive-positive-throughput')
    ctx.ROOT = model.ROOT / ('model-entry-' + arm + '-v1')
    ctx.KEY = 'cache_map_model_entry_' + arm + '_v1'
    ctx.proof, ctx.pinned_sources = proof, pinned_sources
    return ctx


def prepare(ctx, arm):
    ctx.prepare()
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    require(row['configuration']['args'][row['configuration']['args'].index('--adapt-swaps') + 1] ==
            ('0' if arm == 'zero' else '8'), 'declared cache arm differs')
    row['environment']['STRATA_SYCL_ADAPT_SYNC'] = '1'
    row['qualification_basis'] = dict(engine_result_key=model.KEY,
        worker_device_result_key=device.KEY, source_review_scope=SCOPE,
        unchanged_math_fixture_count=8,
        reuse='Identical normalized production compile/link recipe and eight fixture ELFs;source changes are the declared cache-map/worker correction only. Fresh real-copy/cache-close error evidence replaces the retired-generation entry proof. No new host-canary/submission or model qualification is claimed.')
    row['criteria'] = row['criteria'].replace('Fresh host-canary/submission passes;',
        'Fresh pending-copy/worker/cache-close cases pass0/0/1;')
    row['criteria'] += ' New composed worker/map engine only;explicit immutable ADAPT_SYNC1. Two fresh32K requests exercise prefill/decode/next-GEN lifecycle. This is not a long-generation task or speed result;full payload arithmetic and physical-context gates remain separate.'
    row['retention'] = 'Keep compact route/token/lifecycle/latency and fault outcomes. Retire successful API/journal/JIT output at closure. Preserve the adaptive fallback and this candidate for imminent clean speed comparison;no new engine generation is created.'
    ctx.save(row)
    print('Prepared new-engine ' + arm + '-swap model entry;commit before host probes.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 3 and sys.argv[2] in ('prepare', 'host', 'run'),
            'use zero|positive prepare|host|run')
    ctx = context(sys.argv[1])
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if sys.argv[2] == 'prepare': prepare(ctx, sys.argv[1])
        elif sys.argv[2] == 'host': ctx.host()
        else: ctx.run()
