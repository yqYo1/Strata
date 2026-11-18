#!/usr/bin/env python3
"""Check and compare production prefix expert issue timing; repository LICENSE."""

import fcntl
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_prefix_expert_build as build
import prefill_window_device as device
import prefill_workspace_admission as workspace

basis, builder = build.basis, build.builder
P, B, RESULT = build.P, build.B, build.RESULT
require, sha, signature, write = build.require, build.sha, build.signature, build.write
THIS = 'tools/sycl/prefill_prefix_expert_measure.py'
SMOKE = 'prefill_prefix_expert_smoke_v1'
MODE = 'STRATA_SYCL_PREFILL_PREFIX_EXPERT_UPLOAD'
PREFIX_BYTES = 8192 * (2560 * 2 + 4 * 4 + 10 * 4 + 10 * 4 + 4 + 512 * 4)
COHORTS = {
    'numerical': ('off', 'ffn', 'prefix'),
    'ffn': ('off', 'ffn', 'ffn', 'off', 'off', 'ffn'),
    'prefix': ('ffn', 'prefix', 'prefix', 'ffn', 'ffn', 'prefix'),
    'combined': ('off', 'prefix', 'prefix', 'off', 'off', 'prefix'),
}


def native():
    row = json.loads((build.ROOT / 'record.json').read_text())
    original = build.check(row)
    require(row['stage'] == 'built' and row['passed'] and row['complete'] and not row['active']
            and row['ownership_closed'] and row['host_control']['passed'],
            'fresh native and CPU gates required')
    return row, original


def bank_smoke():
    sources = build.source.pin(P, [THIS, 'docs/SYCL_NATIVE16_PREFILL.md'])
    binary, original = native()
    require(SMOKE not in json.loads(RESULT.read_text()), 'one planned fixture run')
    root = build.ROOT / 'bank-smoke'
    root.mkdir(mode=0o700)
    for part in ('tmp', 'cache'):
        (root / part).mkdir()
    probe = original['native_validation_probe']
    env = dict(binary['recipe']['environment'], **device.FIXTURE_ENV)
    env.update(TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
        STRATA_SYCL_PREFILL_FULLFIT_UPLOAD='eager', STRATA_TRACE='1', LD_DEBUG='libs',
        LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    owner = builder.ownership.Owner(root / 'owned', lambda: write(root / 'ownership.json', owner.commands))
    cursor = next(line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).splitlines() if line.startswith('-- cursor: '))
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    candidate = binary['binaries']['sycl_expert_bank_test']
    args = [candidate['path'], '--fullfit-eager-smoke']
    outcome = dict(source=sources['revision'], sources=sources['aggregate_sha256'],
                   binary=candidate, command=args, environment_sha256=signature(env),
                   native_build=signature(builder.compact(binary)))
    try:
        entry, out, err = owner.run('small', args, env, P, **device.SMALL)
        errors = err.read_text()
        require(builder.ownership.closed(entry) and entry['exit_code'] == 0
                and 'urEnqueue' in errors, 'native fixture closure or traced submissions absent')
        require(not (root / 'validation.log').exists() or (root / 'validation.log').stat().st_size == 0,
                'parameter validation warning')
        validation = basis.validation.preload_evidence(errors, entry['pid'])
        require(validation['loader'] == probe['loader']['path'], 'fixture loader differs')
        device.diagnostics.diagnostics(errors, entry['pid'])
        providers = device.loaded_providers(errors, dict(
            environment=env, native_validation_probe=probe,
            runtime_manifest=binary['foundation']['runtime_manifest']))
        outcome.update(**device.gpu_cases('bank-smoke', out.read_text()), providers=providers,
                       native_exit_code=entry['exit_code'], host_probe_accepted=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        basis.validation.finish(outcome, owner, cursor, boot)
        outcome['passed'] &= outcome.get('driver_after') == binary['driver']
        data = json.loads(RESULT.read_text())
        data[SMOKE] = outcome
        write(RESULT, data)
    require(outcome['passed'], outcome.get('error', 'small expert bank fixture failed'))


def request_route(log, plan):
    route = workspace.request_route(log, plan)
    producer = workspace.one(log, 'strata MoE prefix router: ')
    retirement = workspace.one(log, 'strata MoE prefix router complete: ')
    require(producer == dict(requested='early', effective='early', workspace_bytes=str(PREFIX_BYTES),
                            decode_changed='0')
            and retirement == dict(effective='early', layers='48', rows='8192', final_drain='checked'),
            'fixed prefix router not exercised')
    require(route['permanent_hits'] == 0, 'expected one dequant and upload per demanded expert')
    mode = plan['variant']
    admissions = workspace.records(log, 'strata MoE prefix expert upload: ')
    layers = workspace.records(log, 'strata MoE prefix expert upload layer: ')
    ends = workspace.records(log, 'strata MoE prefix expert upload complete: ')
    if mode == 'off':
        require(not admissions and not layers and not ends, 'JIT reference uploaded a prefix subset')
    else:
        require(admissions == [dict(requested=mode, effective=mode, queue='compute')]
                and len(layers) == 48, 'prefix expert upload did not admit all layers')
        experts = payload = 0
        for index, layer in enumerate(layers):
            count, size = int(layer['experts']), int(layer['requested_H2D_bytes'])
            require(int(layer['layer']) == index and 0 < count <= 512 and size > 0
                and size % count == 0 and size // count in (1510400, 1971200, 1715200, 1868800)
                and re.fullmatch('[0-9a-f]{16}', layer['ids_hash']) and layer['readers'] == 'checked',
                'actual prefix-issued copies differ from expected geometry')
            experts += count
            payload += size
        require(ends == [dict(effective=mode, layers='48', experts=str(experts),
                             requested_H2D_bytes=str(payload), final_drain='checked')]
                and experts <= route['uploads'] and payload <= route['requested_bank_H2D_bytes'],
                'prefix upload receipt differs from checked bank totals')
        route['prefix_upload'] = dict(experts=experts, bytes=payload, selected=signature(layers))
    if plan['trace_progress']:
        frames = re.findall(r'^strata MoE router census: layer=(\d+) first=(\d+) rows=(\d+) '
            r'ids=([0-9a-f]{16}) prefix_weights=([0-9a-f]{16})$', log, re.M)
        require(len(frames) == 48 and all(tuple(map(int, v[:3])) == (i, 0, plan['read'] - 1)
                    for i, v in enumerate(frames)), 'exact router comparison evidence absent')
        route['route_digest'] = signature(frames)
    return route


def work(process):
    keys = ('actual_read_tokens', 'actual_reused_tokens', 'generated_tokens', 'request_sha256')
    fields = ('uploads', 'requested_bank_H2D_bytes', 'requested_source_map_H2D_bytes',
              'actual_GU_dispatch', 'route_digest')
    return [({k: sample[k] for k in keys}, {k: route.get(k) for k in fields})
        for route, sample in zip(process['routes'], process['samples'], strict=True)]


def selected(process):
    return [route.get('prefix_upload') for route in process['routes']]


def timing(process):
    first, *warmed = process['samples']
    return dict(first_prompt_ms=first['prompt_ms'], first_ttft_s=first['ttft_s'],
                warm_prompt_ms=statistics.median(s['prompt_ms'] for s in warmed),
                warm_ttft_s=statistics.median(s['ttft_s'] for s in warmed))


def model_context(cohort):
    order = COHORTS[cohort]
    traced = cohort == 'numerical'
    read, runs = (32769, 2) if traced else (65537, 3)
    spec = importlib.util.spec_from_file_location('prefix_expert_policy', basis.core.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT = B / ('prefill-prefix-expert-' + cohort + '-v1')
    ctx.KEY, ctx.THIS = 'prefill_prefix_expert_' + cohort + '_v1', THIS
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (read,) * len(order), [(read, 1)] * len(order), runs, True
    base_environment, base_current = ctx.environment, ctx.current

    def reference():
        value, _ = native()
        smoke = basis.validation.committed(RESULT, (SMOKE,))
        require(smoke['passed'] and smoke['ownership_closed'] and smoke['native_exit_code'] == 0
                and smoke['native_build'] == signature(builder.compact(value)), 'same-build fixture required')
        gates = {SMOKE: signature(smoke)}
        if not traced:
            gate = basis.validation.committed(RESULT, ('prefill_prefix_expert_numerical_v1',))
            require(gate['stage'] == 'closed' and gate['ownership_closed'] and
                    gate['decision']['matched_work'] and gate['binary'] == value['binaries']['strata'],
                    'matched numerical cohort required')
            gates['numerical'] = signature(gate)
        _, _, speed = basis.basis()
        old = json.loads((basis.reference.model.ROOT / 'record.json').read_text())
        require(old['models'] == speed['model_assets'] and old['probe'] == speed['native_validation_probe'],
                'exact coherent model assets and probe required')
        old['observer_admission'] = dict(gates=gates, native_build=signature(builder.compact(value)),
            fast_configuration_diagnostic=speed['observer_admission']['fast_configuration_diagnostic'])
        return dict(binaries=value['binaries'], native_source_commit=value['source_commit']), old, \
            json.loads(Path(speed['runtime_manifest']['path']).read_text()), None

    def environment(row, plan):
        env = base_environment(row, plan)
        env.update({MODE: order[plan['index']], 'STRATA_SYCL_PREFILL_MOE_PREFIX_ROUTER': 'early',
                    'STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS': '512',
                    'STRATA_SYCL_PREFILL_QSA_MATH': 'bf16-prefix-ordered-exp',
                    'STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR': '1',
                    'STRATA_SYCL_PREFILL_GU_SWIGLU': 'large',
                    'STRATA_SYCL_PREFILL_MOE_CENSUS': str(int(traced)),
                    'STRATA_PREFILL_LAYER_MAJOR_PROFILE': '0',
                    'STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING': '0', 'STRATA_SYCL_PREFILL_FULLFIT_UPLOAD': 'jit'})
        if traced:
            env['STRATA_TRACE'] = '1'
        return env

    def current(row):
        require([p['variant'] for p in row['plans']] == list(order)
                and all((p['bank'], p['window'], p['relocate'], p['trace_progress'], p['upload'], p['read'])
                    == (512, 10, 0, traced, 'jit', read) for p in row['plans']), 'frozen comparison differs')
        return base_current(row)

    def after_execution(outcome, row, plan):
        for index, process in enumerate(row['processes']):
            require(work(outcome) == work(process), 'logical work differs between matched requests')
            if order[index] != 'off' and plan['variant'] != 'off':
                require(selected(outcome) == selected(process), 'selected experts or issue bytes differ')

    def decision(processes):
        result = dict(matched_work=True, independent_processes=len(processes), measured_modes=list(order),
                      overlap_measured=False, model_adopted=False)
        if not traced:
            pairs = []
            for index in range(0, len(order), 2):
                times = {order[j]: timing(processes[j]) for j in (index, index + 1)}
                pairs.append(dict(arm_indices=[index, index + 1], times=times,
                    ratios={key: times[order[1]][key] / times[order[0]][key] for key in times[order[0]]}))
            ratios = [pair['ratios']['warm_prompt_ms'] for pair in pairs]
            result.update(pairs=pairs, all_warmed_pairs_faster=all(value < 1 for value in ratios),
                          warmed_prompt_time_geomean_ratio=statistics.geometric_mean(ratios))
        return result

    def sample(sample, count, actual_read, minimum_ms=10000):
        basis.comparison.check_sample(sample, count, actual_read, minimum_ms)
        require(sample['generated_tokens'] == 24, 'paired requests must generate exactly 24 tokens')

    ctx.basis, ctx.environment, ctx.current = reference, environment, current
    ctx.request_route, ctx.decision, ctx.check_sample = request_route, decision, sample
    ctx.after_execution = after_execution
    return ctx


def prepare(ctx, cohort):
    ctx.prepare()
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    envs = []
    for plan, mode in zip(row['plans'], COHORTS[cohort], strict=True):
        plan.update(variant=mode, bank=512, window=10, relocate=0,
                    trace_progress=cohort == 'numerical', upload='jit')
        env = ctx.environment(row, plan)
        require(plan['environment_sha256'] == basis.comparison.fingerprint(env), 'prepared environment differs')
        envs.append({k: v.replace(plan['root'], '<case>') for k, v in env.items() if k != MODE})
    require(all(env == envs[0] for env in envs), 'another environment variable changed between arms')
    row['plan'] = dict(cohort=cohort, modes=COHORTS[cohort], actual_input=ctx.READS[0], requests=ctx.RUNS,
        question='Compare named issue policies with fixed early router, main compute queue, resident host '
                 'sources, packed bank, layer math and decode handoff. Each pair starts new processes.',
        gates='Match logical work and observed expert bytes. For both issue arms match selected IDs. '
              'Retain every sample and separate first use. Stop on validation, fault or ownership failure.',
        retention='Keep compact matched work, PP/TTFT and checked closure. Review extra build at closure.')
    ctx.save(row)


if __name__ == '__main__':
    require(sys.executable == basis.reference.model.PYTHON, 'use pinned Python')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        if sys.argv[1:] == ['bank-smoke']:
            bank_smoke()
        else:
            cohort, action = sys.argv[1:]
            ctx = model_context(cohort)
            if action == 'prepare':
                prepare(ctx, cohort)
            elif action == 'host':
                basis.comparison.host(ctx)
            else:
                basis.comparison.run(int(action), ctx)
