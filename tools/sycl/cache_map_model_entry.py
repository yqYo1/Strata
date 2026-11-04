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
from functools import lru_cache

sys.dont_write_bytecode = True
import build_cache_map_model as model
import checked_worker_device as device
import check_prefill_strategy_model as template

base, P, B, RESULT = model.base, model.P, model.B, model.RESULT
SCOPE = 'checked_worker_retirement_contract_20261102'
INDEX = P / 'bench/results/2026-10-15-pr6-recovery-audit/research-index.json'
require, digest, write = base.require, base.digest, base.write
HISTORY_COMMIT = 'c23a2f7ca5d177102f347eeabae605903cb2ff9d'
NATIVE_CONTROLS = ('tools/sycl/build_cache_map_model.py', 'tools/sycl/compare_official_prefill.py',
                   'tools/sycl/official_model_entry.py', 'tools/sycl/gdn_model_sources.py',
                   'tools/sycl/host_link_inventory.py')


@lru_cache(maxsize=1)
def history():
    return json.loads(base.basis.build.git(P, 'show', HISTORY_COMMIT + ':' + str(RESULT.relative_to(P))))


def mathematical_fixture_basis(native):
    """Reuse closed byte-identical fixtures without reopening retired build products."""
    original = history()
    old = original[model.prior.KEY]
    require(json.dumps(base.committed(RESULT)[model.prior.KEY], sort_keys=True) ==
            json.dumps(old, sort_keys=True) and old['passed'] is True
            and old['ownership_closed'] is True and old['active'] is False,
            'immutable original adaptive build qualification changed')
    bridge = original['cache_map_model_entry_zero_v1']
    require(all(bridge[k] is True for k in ('passed', 'complete', 'ownership_closed'))
            and bridge['active'] is False
            and bridge['binary'] == native['binaries']['strata']
            and bridge['native_source_commit'] == native['source_commit']
            and bridge['qualification_basis']['unchanged_math_fixture_count'] == 8,
            'closed same-engine fixture bridge required')
    _, projected = model.prior.projection()
    before = {name: hashlib.sha256(raw).hexdigest() for name, raw in projected.items()}
    require(old['native_sources'] == dict(files=len(before), aggregate_sha256=hashlib.sha256(
                json.dumps(before, sort_keys=True).encode()).hexdigest())
            and len(native['native_sources']) == len(before) + 1
            and {name for name in native['native_sources']
                 if native['native_sources'][name] != before.get(name)} ==
                {*model.CHANGED, model.WORKER_HEADER}, 'unplanned native difference')
    def normalized(value, root):
        return json.dumps(value, sort_keys=True).replace(str(root), '<generation>')
    database = Path(native['compile_database']['path']).read_bytes()
    require(hashlib.sha256(database).hexdigest() == native['compile_database']['sha256']
            and hashlib.sha256(database.replace(str(model.ROOT).encode(),
                str(model.prior.ROOT).encode())).hexdigest() == old['compile_database']['sha256']
            and all(normalized(old[name], model.prior.ROOT) == normalized(native[name], model.ROOT)
                    for name in ('configure_argv', 'environment'))
            and old['actual_link_inputs']['frozen_inputs'] == native['actual_link_inputs']['frozen_inputs'],
            'unchanged fixtures require the exact original normalized compiler/link recipe')
    names = set(native['binaries']) - {'strata'}
    require(len(names) == 8 and all(native['binaries'][name]['sha256'] == old['binaries'][name]['sha256']
            and native['binaries'][name]['bytes'] == old['binaries'][name]['bytes']
            and digest(native['binaries'][name]['path']) == native['binaries'][name]['sha256']
            and Path(native['binaries'][name]['path']).stat().st_size == native['binaries'][name]['bytes']
            for name in names), 'actual selected mathematical fixture bytes changed')
    return dict(history_commit=HISTORY_COMMIT, original_build_key=model.prior.KEY,
        original_source_commit=old['source_commit'], closed_bridge='cache_map_model_entry_zero_v1',
        current_native_source_commit=native['source_commit'], mathematical_fixtures=8,
        original_native_sources_recovered=len(before), original_compile_database_reconstructed=True,
        old_build_products_reopened=False, GPU_executed=False)


def proof():
    row = json.loads((device.ROOT / 'record.json').read_text())
    require(json.dumps(device.compact(row), sort_keys=True) ==
            json.dumps(history()[device.KEY], sort_keys=True), 'immutable worker-device proof changed')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    native = device.current(row, reuse_revision=revision,
        controller_changes=(*NATIVE_CONTROLS, 'tools/sycl/checked_worker_device.py'),
        native_controller_changes=NATIVE_CONTROLS)
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
    fixture_basis = mathematical_fixture_basis(native)
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
    gate = dict(source_commit=row['source_commit'], environment=env,
                mathematical_fixture_basis=fixture_basis)
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
    row['retention'] = 'Keep compact route/token/lifecycle/latency and fault outcomes. Retire successful API/journal/JIT output at closure. Keep the selected current generation;the original adaptive fixture basis is recovered from Git and the selected compile database/eight actual fixture ELFs,without reopening old build products.'
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
