#!/usr/bin/env python3
"""Fresh corrected-engine v5 generation comparison; repository LICENSE.

Reuse the maintained benchmark in a private policy context. The failed older
engine cohort remains failed. Native mathematics and workload gates are unchanged;
the new worker/map build has its own real-copy and two-request qualification.
Main owns all serialized host/model execution.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import cache_map_model_entry as entry
import pool_spin as maintained

base, P, B, RESULT = entry.base, entry.P, entry.B, entry.RESULT
require, digest = base.require, base.digest
SOURCE = 'tools/sycl/cache_map_generation.py'
ROOT = entry.model.ROOT / 'generation-speed-v1'
KEY = 'cache_map_generation_speed_v1'
OLD_KEY = 'adaptive_cache_generation_speed_v2'


def pinned_sources(revision):
    pins = entry.pinned_sources(revision)
    pins.update(base.controls(revision, controller=Path(__file__).resolve()))
    return pins


def basis(revision):
    prior_ctx = entry.context('positive')
    prior = json.loads((prior_ctx.ROOT / 'record.json').read_text())
    providers, manifest = prior_ctx.current(prior)
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete'] and
            prior['ownership_closed'] and not prior['active'] and prior['host_passed'] and
            prior['outcome']['native_exit_code'] == 0 and not prior['outcome']['fresh_faults'] and
            len(prior['outcome']['samples']) == len(prior['outcome']['routes']) == 2 and
            prior['outcome']['adaptive_work']['deferred_next_long_prefill_boundaries'] > 0 and
            prior['environment']['STRATA_SYCL_ADAPT_SYNC'] == '1',
            'closed current-engine positive-swap next-GEN model qualification required')
    zero_ctx = entry.context('zero')
    zero = json.loads((zero_ctx.ROOT / 'record.json').read_text())
    zero_ctx.current(zero)
    require(zero['stage'] == 'closed' and zero['passed'] and zero['complete'] and
            zero['ownership_closed'] and not zero['active'] and
            zero['binary'] == prior['binary'] and zero['outcome']['native_exit_code'] == 0 and
            not zero['outcome']['fresh_faults'], 'same-engine zero-swap model qualification required')
    original = {name: hashlib.sha256(raw).hexdigest()
                for name, raw in prior_ctx.shared.gen.blobs(P, prior['source_commit'], prior['sources'])}
    current = pinned_sources(revision)
    require(original == prior['sources'] and
            all(current.get(name) == sha for name, sha in original.items()) and
            set(current) - set(original) == {SOURCE},
            'only the new adapter may extend unchanged model/controller/workload inputs')
    failed = base.committed(RESULT)[OLD_KEY]
    require(failed['stage'] == 'failed' and not failed['passed'] and not failed['active'] and
            failed['ownership_closed'] and len(failed['processes']) == 2 and
            failed['binary'] != prior['binary'],
            'preserve the older engine failed cohort;never resume or relabel it')
    native = entry.model.verify()
    names = ('src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/pool.hpp')
    pool = {name: native['native_sources'][name] for name in names}
    require(all(digest(entry.model.ROOT / 'source' / name) == sha for name, sha in pool.items()),
            'compiled pool inputs differ')
    require(prior['binary'] == native['binaries']['strata'] and
            prior['native_source_commit'] == native['source_commit'], 'current engine differs')
    return prior, providers, manifest, pool


def context():
    spec = importlib.util.spec_from_file_location('strata_cache_map_generation_context', maintained.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.select_mode('cohort')
    ctx.ROOT, ctx.KEY = ROOT, KEY
    ctx.model = entry.context('positive')
    ctx.model.pinned_sources = pinned_sources
    ctx.basis = basis
    return ctx


def prepare(ctx):
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    require(len(row['plans']) == 6 and all(ctx.environment(row, plan)['STRATA_SYCL_ADAPT_SYNC'] == '1'
            for plan in row['plans']), 'six immutable same-engine cache arms required')
    row['qualification_reuse'] = dict(
        zero_model_result_key=entry.context('zero').KEY,
        positive_model_result_key=entry.context('positive').KEY,
        worker_device_result_key=entry.device.KEY,
        source_review_scope=entry.SCOPE,
        unchanged_math_fixture_count=8,
        added_control_paths=[SOURCE],
        preserved_failed_cohort_key=OLD_KEY,
        scope='Exact corrected worker/map engine,unchanged mathematics/fixtures,qualified zero/positive two-request lifecycle. No old generation diagnostic or failed-cohort qualification is inherited.')
    row['plan']['question'] = 'Does8swaps/every1 reproducibly improve sustained v5 code and prose generation over0swaps in the corrected worker/map engine?'
    row['plan']['motivation'] = 'The older engine V2cohort remains failed after two processes,including candidate thinking-only prose. The new exact engine passed fresh real-copy/cache-close error and zero/positive next-GEN lifecycle checks. Measure long answers and observed agreement before considering adoption.'
    row['plan']['constants'] += ' Corrected worker/map native recipe479a6fdac,explicit immutable ADAPT_SYNC1;no other control or workload changes.'
    row['plan']['stopping'] += ' The first new-engine process must close and be committed before launching its paired arm. All earlier failed source/task outcomes remain unchanged.'
    ctx.save(row)


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        maintained.build.guards.no_model_owners([])
        ctx = context()
        args = sys.argv[1:]
        if args == ['prepare']:
            prepare(ctx)
        elif args == ['host']:
            ctx.comparison.host(ctx)
        elif args == ['selfcheck']:
            print(json.dumps(ctx.selfcheck()))
        elif len(args) == 1 and args[0].isdigit() and int(args[0]) < len(ctx.ARMS):
            ctx.run(int(args[0]))
        else:
            raise RuntimeError('cache_map_generation.py prepare|host|selfcheck|INDEX0..5')
