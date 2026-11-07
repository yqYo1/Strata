#!/usr/bin/env python3
"""Build only the host-qualified audit-limit change onto the owned429b engine.

Main-authored; repository LICENSE. Keep the resource cohort frozen. Preparation
requires all six processes closed. A temporary third comparison generation is
reserved only for native32 positive/refusal qualification;retire429b before
clean8/32speed trials,or retire this candidate if qualification fails.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

sys.dont_write_bytecode = True
import adaptive_payload_host as host
import adaptive_payload_sources as source
import build_adaptive_audit_model as original

P, B, RESULT, base = original.P, original.B, original.RESULT, original.base
ROOT, KEY = B / 'adaptive-payload-audit-model-v1', 'adaptive_payload_native_build_v1'
SELF = Path(__file__).resolve()
require, digest, write, signature = original.require, original.digest, original.write, original.signature


def host_proof():
    row = json.loads((host.ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[host.KEY] == host.compact(row) and row['passed'] and
        row['complete'] and row['ownership_closed'] and not row['active'] and
        row['GPU_executed'] is False and row['model_executed'] is False and
        [c['receipt'] for c in row['outcome']['cases']] ==
        ['adaptive_residency_audit cases=41 status=PASS', 'adaptive_payload_budget cases=16 status=PASS'] and
        len(row['commands']) == 6 and all(base.supervision.closed(c) and c['exit_code'] == 0
            for c in row['commands']), 'closed exact57-case host qualification required')
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
        base.basis.build.blobs(P, row['source_commit'], row['sources'])}
    require(recovered == row['sources'], 'recover all original117host inputs')
    return row


def basis():
    data = base.committed(RESULT)
    cohort = data['decode_resource_regular_code_v1']
    require(cohort['stage'] == 'closed' and cohort['passed'] and cohort['complete'] and
        cohort['ownership_closed'] and not cohort['active'] and len(cohort['processes']) == 6 and
        all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0 and
            not p['fresh_faults'] and p['resources']['passed'] for p in cohort['processes']),
        'close all six frozen resource processes before source preparation')
    # This closure gate is independent of incremental-observer overhead: keep
    # its verdict,including an inconclusive or failed cost comparison,intact.
    old = json.loads((original.ROOT / 'record.json').read_text())
    require(data[original.KEY] == original.compact(old) and old['passed'] and old['complete'] and
        old['ownership_closed'] and not old['active'], 'closed original429b build required')
    seed = json.loads((original.OLD / 'record.json').read_text())
    parent, _, before = original.selected.projection()
    require(original.selected.pins(before) == seed['native_sources'] and
        original.selected.recipe(parent) == seed['recipe'], 'recover original567-source seed and recipe')
    # Historical recovery uses Git,not a public header that may later advance.
    raw = original.selected.blob(source.HEADER_SOURCE, source.HEADER)
    prior = original.source.projection(before, raw)
    require(len(prior) == 568 and original.selected.pins(prior) == old['native_sources'] and
        original.recipe(seed) == old['recipe'], 'recover exact final429b sources/recipe')
    proof = host_proof(); values = source.projection(prior)
    require(hashlib.sha256(values[source.HEADER]).hexdigest() == proof['generated'][source.HEADER] and
        proof['expert_limit_bytes'] == source.EXPERT_BYTES and
        proof['cycle_limit_bytes'] == source.CYCLE_BYTES,
        'native projection differs from the host-qualified64/256MiB header')
    return old, values


def recipe(old):
    value = copy.deepcopy(old['recipe'])
    value['configure'] = [v.replace(str(original.ROOT), str(ROOT)) for v in value['configure']]
    value['environment'] = {n: v.replace(str(original.ROOT), str(ROOT))
        for n, v in value['environment'].items()}
    value['retention'] = ('Temporarily keep552b fallback,429b fixture/reference and this one successor '
        'through native32positive and second-record-refusal qualification. The old15outputs are '
        'protected. Retire429b after prospective consumer migration and source/recipe recovery '
        'before clean8/32speed trials;retire this candidate if qualification fails. No open-ended '
        'third-generation retention;older configured-deployment provenance holds remain separate.')
    return value


def compact(row):
    value = original.compact(row)
    value['recipe'].update(helper='tools/sycl/build_adaptive_payload_model.py:recipe',
        scope='Exact429b recipe with private paths;only the host-qualified audit header changes.')
    return value


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def retention_gate():
    row = json.loads((ROOT / 'record.json').read_text())
    old, _ = basis()
    require(original.retirement_basis() == row['retirement_basis'] and
        all(digest(v['path']) == v['sha256'] for v in old['binaries'].values()) and
        all(digest(p) == h for p, h in row['fallback_binaries'].items()),
        'preserve552b and429b before replacing the diagnostic generation')


def factory():
    spec = importlib.util.spec_from_file_location('strata_payload_native_build', original.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    proof = host_proof()
    ctx.OLD, ctx.ROOT, ctx.KEY, ctx.SELF = original.ROOT, ROOT, KEY, SELF
    ctx.source = SimpleNamespace(GENERATOR=original.source.GENERATOR, HEADER=source.HEADER,
        HEADER_SOURCE=proof['source_commit'], HEADER_SHA=proof['generated'][source.HEADER])
    ctx.basis, ctx.recipe, ctx.compact, ctx.save = basis, recipe, compact, save
    ctx.retirement_basis, ctx.retention_gate = original.retirement_basis, retention_gate
    return ctx


def prepare():
    ctx = factory(); ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    old, _ = basis(); proof = host_proof()
    row['selected_basis'].update(recovered_files=568, changed=[source.HEADER],
        unchanged_native_kernel_CPU_runtime_prefill_files=567)
    row['host_header_qualification'].update(cases=57, legacy_cases=41, added_budget_cases=16,
        source_commit=proof['source_commit'],
        scope='Exact64MiB expert/256MiB cycle host state machine;native32 remains unqualified.')
    row['comparison_predecessor'] = dict(source_commit=old['source_commit'], binaries=old['binaries'],
        record_sha256=digest(original.ROOT / 'record.json'))
    row['temporary_generation_budget'] = dict(max_temporary_comparison_generations=3,
        protected_existing_generations=2, expected_successor_outputs=6,
        preserve_outputs_before_compile=15,
        consumer='Exact old/new five-fixture byte comparison and bounded native32 positive/refusal qualification.',
        retirement_point='Before clean8/32performance:select qualified successor,update consumers,retire429b;on failure retire successor.')
    row['plan'] = ('Commit this exact source/recipe/compiler/dependency plan before compilation. '
        'Only the diagnostic header differs from429b;all567other native inputs and all kernels,'
        'prefill and runtime safety remain unchanged. Keep552b and429b through successor '
        'qualification. Require normal closed compilation,actual header/link/flag checks and '
        'five byte-identical fixtures before reusing any old component evidence. Commit build '
        'before fresh same-build CPU checks and bounded validated native32positive plus '
        'second-record refusal;full publication/dispatch/occupied-loan checks remain required. '
        'No speed,quality,ordinary-provider-teardown or physical262144qualification from build. '
        'Prospectively migrate consumers and retire429b before same-successor8/32clean comparison.')
    save(row)
    print('Prepared one-header successor;commit before compilation. Native32 remains unqualified.', flush=True)


def build():
    factory().build()
    row = json.loads((ROOT / 'record.json').read_text())
    try:
        old, _ = basis()
        names = set(old['binaries']) - {'strata'}
        require(len(names) == 5 and all(row['binaries'][n]['sha256'] == old['binaries'][n]['sha256']
            for n in names), 'changed fixture requires fresh component qualification')
        row['native_recipe_checks'].update(unchanged_native_kernel_CPU_runtime_prefill_files=567,
            only_audit_header_changed=True, same_five_fixture_bytes=True,
            fixture_sha256={n: row['binaries'][n]['sha256'] for n in sorted(names)},
            host_audit_cases=57)
    except BaseException as error:
        row.update(stage='failed-successor-reuse-check', passed=False, complete=False, error=repr(error)); raise
    finally: save(row)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['build'], ['host']), 'use prepare|build|host')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] == 'build': build()
        else: factory().context().host()
