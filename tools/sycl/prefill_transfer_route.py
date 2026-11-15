#!/usr/bin/env python3
"""Observe the qualified original engine's actual long-prefill transfer route.

Main-authored using this fork's owned bench.sh controller; repository LICENSE.
No inference edits or native rebuild. This is route evidence, not a speed trial.
"""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import coherent_original_reuse as original
import qsa16_stream_bench as core

P, B, RESULT = core.P, core.B, core.RESULT
ROOT, KEY = B/'prefill-transfer-route-v1', 'prefill_transfer_route_v1'
THIS = 'tools/sycl/prefill_transfer_route.py'
SEED = B/'coherent-generations-prefill-v1/record.json'
require, sha = core.require, core.digest
PYTHON = '/home/yayoi/.local/share/strata-sycl/venv/bin/python3'


def basis():
    native, proof = original.verify_original()
    seed = json.loads(SEED.read_text())
    public = original.coherent.build.validation.committed(RESULT, ('coherent_generations_prefill_v1',))
    require(public == core.compact(seed) and seed['stage'] == 'closed' and not seed['active']
        and seed['ownership_closed'], 'exact closed configuration/asset seed required')
    require(seed['native_builds']['reference']['binary'] == native['binaries']['strata']
        and seed['native_builds']['reference']['native_source_commit'] == original.SOURCE,
        'original configuration/native binding changed')
    core.shared.models.model_pins(seed['configuration'], seed['model_assets'], full=False)
    require(core.build.guards.driver_guard() == seed['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == seed['boot'], 'boot/driver changed')
    manifest = seed['runtime_manifest']
    require(sha(manifest['path']) == manifest['sha256'], 'seed provider manifest changed')
    providers = json.loads(Path(manifest['path']).read_text())
    require(all(sha(p) == h for p, h in providers.items()), 'provider bytes changed')
    cfg = seed['configuration']
    require(cfg['exe'] == str(P/'tools/sycl/engine_once.py') and '--serve' not in cfg['args'],
        'preserve supervised production entrypoint and native argv')
    prior = dict(models=seed['model_assets'], environment=seed['environment_template'],
        tokenizer=cfg['tokenizer'], argv=[native['binaries']['strata']['path'], *cfg['args'], '--serve'],
        probe=seed['native_validation_probe'], boot=seed['boot'], driver=seed['driver'],
        observer_admission=dict(original_native=proof, fast_configuration_diagnostic=dict(
            seed_path=str(SEED), seed_sha256=sha(SEED),
            original_native_source=original.SOURCE, original_full_asset_identities_rechecked=True,
            previous_candidate_performance_inherited=False)))
    require(prior['probe'] == native['native_validation_probe'], 'same qualified pre-main validation required')
    return native, prior, providers, None


def transfer_route(text):
    def records(prefix):
        return [dict(re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)', line))
            for line in text.splitlines() if line.startswith(prefix)]
    source = records('strata prefill expert source: ')
    bank = records('strata MoE expert bank: ')
    layers = records('strata packed expert bank: ')
    require(len(source) == len(bank) == 1 and len(layers) == 48,
        'one source/bank marker and48actual packed-layer markers required')
    src, selected = source[0], bank[0]
    require(src['requested_view'] == '1' and src['effective_view'] in ('0','1') and
        selected['staged'] == '1' and selected['requested_slots'] == selected['effective_slots'] == '384',
        'unexpected source selection or bank route')
    effective = src['effective_view'] == '1'
    require(src['host_copy'] == ('eliminated' if effective else 'synchronous')
        and int(src['layers']) == (48 if effective else 0)
        and (int(src['owned_slab_bytes']) == 0) == effective,
        'inconsistent actual source-view accounting')
    require([int(l['layer']) for l in layers] == list(range(48)), 'incomplete or reordered layer trace')
    grouped = {}
    for l in layers:
        slots, stride, used, backing, full = (int(l[k]) for k in
            ['slots','stride','used_bytes','backing_bytes','full_fit'])
        require(1 <= slots <= 512 and used == slots*stride and used <= backing
            and full == int(slots == 512), 'private packed-bank geometry mismatch')
        key = f'slots{slots}/stride{stride}/full_fit{full}'
        grouped[key] = grouped.get(key, 0) + 1
    require('strata prefill expert upload staging:' not in text and
        'strata prefill expert upload staging complete:' not in text,
        'unexpected host-USM upload-staging route')
    return dict(actual_source_view=effective, source_marker=src, staged_bank=True,
        actual_private_layer_geometries=grouped, actual_layers=48,
        upload_staging_enabled=False,
        source_bound_staging_evidence='Immutable STAGING=0 plus exact called source and absence of its mandatory construction/footer markers',
        source_bound_queue_evidence='Successful layer-major and per-expert admission in14508require m.copy==m.cs;all copy/dequant/GEMM submissions use that same in-order stream',
        limits='Actual emitted source/bank/layer markers plus pinned branch contracts;no per-command timestamps,engine ordinals,bus traffic,overlap or speed claim.')


def decision(processes):
    require(len(processes) == 1 and processes[0]['passed'] and processes[0]['ownership_closed']
        and len(processes[0]['routes']) == 2, 'one closed process/two requests required')
    routes = [v['transfer_route'] for v in processes[0]['routes']]
    require(routes[0] == routes[1], 'first/subsequent transfer selections differ')
    return dict(actual_transfer_route=routes[0], observation_complete=True,
        inference_changed=False, clean_speed_or_adoption_qualified=False,
        installed_UR_engine_selection_still_required=True)


def context():
    spec = importlib.util.spec_from_file_location('prefill_transfer_route_context', core.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (32769,), [(32769,1)], 2, True
    ctx.basis, ctx.decision = basis, decision
    old_environment, old_route = ctx.environment, ctx.request_route
    def environment(row, plan):
        env = old_environment(row, plan)
        env.update(STRATA_TRACE='1', STRATA_SYCL_PREFILL_QSA_MATH='bf16-prefix-ordered-exp',
            STRATA_SYCL_PREFILL_GU_SWIGLU='large', STRATA_SYCL_PREFILL_EXPERT_SOURCE_VIEW='1',
            STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING='0',
            STRATA_PREFILL_LAYER_MAJOR_PROFILE='0', STRATA_SYCL_PREFILL_MOE_CENSUS='0')
        return env
    def route(text, plan):
        result = old_route(text, plan)
        result['transfer_route'] = transfer_route(text)
        return result
    ctx.environment, ctx.request_route = environment, route
    return ctx


def prepare():
    ctx = context(); ctx.prepare()
    row = json.loads((ROOT/'record.json').read_text())
    row['plan'] = dict(question='Which actual source,bank,staging and queue-ordering route runs on the fastest owned32Kprefill?',
        order='One original14508/dcb8process;two distinct32769input thinking-on24output pinned bench.sh requests. Retain first/subsequent separately. No engine/configuration arm or speed-adoption decision.',
        settings='Original coherent build12ELFs and84providers,logical262144/resident20480,throughput/chunk8192,BF16ordered-exp,largeGU,sourceview1/staging0/384packedbank,qualified safety/private pre-main validation. Add only existing STRATA_TRACE progress.',
        acceptance='Exact original Git/source/recipe/12ELFs,committed current frontend/controller/generated config,assets,providers,pre-main validation,normal SSE/fresh counts/quiet validation,normal0/closed owners/no new fault. Every request must report one effective source marker,all48packed-layer geometries,completed staged-bank drain and unchanged transfer selection.',
        stopping='Stop first source,identity,provider,owner,route,token,clock,validation,fault or budget failure. No retry,pin refresh or replacement. Rehash all15model assets after the second request.',
        limits='Route observation only. Profiled24output PP/TG/TTFT is not clean speed,decode qualification,model-task or physical262144qualification. Queue ordinal and hardware overlap remain unknown.',
        retention='Keep compact actual route/geometry counts,two samples,source/config/workload/owner/fault identities. Retire successful raw logs and this observer once the source-audit consumer has its pinned evidence. Both retained engine generations remain unchanged.')
    row['budgets'].update(wall=1200, cpu=7200, text_cap=16<<20, file_cap=32<<20, total_cap=128<<20)
    row['whole_bytes'] = 192<<20
    for plan in row['plans']:
        plan['trace_progress'] = True
        plan['environment_sha256'] = core.comparison.fingerprint(ctx.environment(row, plan))
    ctx.save(row)
    print('Prepared one bounded actual-route observation;commit before host/native.', flush=True)


if __name__ == '__main__':
    require(sys.executable == PYTHON, 'use pinned benchmark Python')
    require(sys.argv[1:] in [['prepare'],['host'],['run']], 'prepare|host|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        core.build.guards.no_model_owners([])
        action = sys.argv[1]
        if action == 'prepare': prepare()
        elif action == 'host': core.comparison.host(context())
        else: core.comparison.run(0, context())
