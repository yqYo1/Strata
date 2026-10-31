#!/usr/bin/env python3
"""Current-build original/row64 model comparison; repository LICENSE.

Main-authored from this fork's owned controls. Same ELF, explicit pre-exec
selection, no native build, XeStrata input or inference adoption.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_resources as reference
import qualify_prefill_candidates as device

comparison, shared, build = reference.comparison, reference.shared, reference.build
P, B, RESULT = reference.P, reference.B, reference.RESULT
ROOT, KEY = B / 'official-row64-model-v2', 'official_row64_model_v2'
READ, RUNS = 117489, 4
KERNELS = ('row64', 'original', 'original', 'row64', 'row64', 'original')
ORDER = ((READ, 1),) * len(KERNELS)
require, digest, write = reference.require, reference.digest, reference.write
CHANGED = {'tools/sycl/compare_official_prefill.py'}
ADDED = {'tools/sycl/qualify_prefill_candidates.py', 'tools/sycl/compare_prefill_candidates.py'}


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = reference.compact(row)
    write(RESULT, data)


def qualification(revision):
    built, providers = reference.native_reuse(revision)
    data = build.committed(RESULT)
    native = json.loads((device.ROOT / 'record.json').read_text())
    gate = data[device.KEY]
    require(device.compact(native) == gate and gate['passed'] and gate['complete']
            and gate['ownership_closed'] and not gate['active'] and gate['binaries'] == built['binaries']
            and gate['native_source_commit'] == built['source_commit']
            and len(gate['processes']) == 3, 'closed same-build row64 qualification required')
    require(all(p['passed'] and p['complete'] and p['ownership_closed'] and not p['fresh_faults']
            and p['driver_after'] == gate['driver'] and p['native_exit_code'] == 0
            and p['actual_split_kernel_launches'] == dict(original=0, row128=0, row64=n)
            for p, n in zip(gate['processes'], (10, 24, 6))), 'row64 component/dispatch proof differs')
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
        shared.gen.blobs(P, native['source_commit'], native['sources'])}
    require(recovered == native['sources'], 'original component controller recovery failed')
    require({n for n, sha in native['sources'].items() if digest(P / n) != sha} == CHANGED,
            'only the newly parameterized model controller may differ after device closure')
    prior = json.loads((reference.ROOT / 'record.json').read_text())
    require(reference.compact(prior) == data[reference.KEY] and prior['passed'] and prior['complete']
            and prior['ownership_closed'] and not prior['active'] and len(prior['processes']) == 2
            and prior['binary'] == built['binaries']['strata'] and prior['driver'] == build.guards.driver_guard()
            and prior['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'closed unchanged current-generation model/resource reference required')
    require(json.loads(Path(gate['runtime_manifest']['path']).read_text()) == providers,
            'same84 providers required for row64 and original')
    return prior, providers


def environment(row, plan):
    require(plan['kernel'] == KERNELS[plan['index']] and plan['read'] == READ
            and plan['heads'] == 1, 'unplanned candidate/order/placement')
    env = reference.environment(row, dict(plan, observed=False))
    env['STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL'] = plan['kernel']
    return env


def route(text, read, heads):
    result = comparison.route(text, read, heads)
    require(result['residual_gpu_rows'] == READ - 1 and result['residual_host_bytes'] == 0
            and result['prefill_slots'] == 128 and result['split_guard_checked'],
            'all-GPU128-slot split placement required in both arms')
    return result


def validate_model_text(text, plan):
    choices = re.findall(r'^\[native16-down\] fusion=large min_rows=2334 split_kernel=(\w+)$', text, re.M)
    require(choices == [plan['kernel']], 'compiled immutable Down choice differs')


def decision(processes):
    require(len(processes) == len(KERNELS) and all(p['passed'] and p['complete']
            and p['ownership_closed'] and not p['fresh_faults'] and p['read'] == READ
            and p['heads'] == 1 and p['kernel'] == KERNELS[i]
            for i, p in enumerate(processes)), 'incomplete/mismatched comparison')
    require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
            for p in processes}) == len(KERNELS), 'native process reused')
    ratios, ttft_ratios, candidates, controls = [], [], [], []
    for start in (0, 2, 4):
        pair = {p['kernel']: p for p in processes[start:start + 2]}
        a, b = pair['row64'], pair['original']
        require(all(math.isfinite(p['subsequent_median_ms']) and p['subsequent_median_ms'] > 0
                for p in (a, b)), 'invalid process clock')
        ratios.append(a['subsequent_median_ms'] / b['subsequent_median_ms'])
        times = [statistics.median(s['ttft_s'] for s in p['samples'][1:]) for p in (a, b)]
        require(all(math.isfinite(v) and v > 0 for v in times), 'invalid TTFT')
        ttft_ratios.append(times[0] / times[1])
        candidates.append(READ * 1000 / a['subsequent_median_ms'])
        controls.append(READ * 1000 / b['subsequent_median_ms'])
    geometric = math.exp(statistics.mean(math.log(v) for v in ratios))
    return dict(row64_over_original_prompt_time=ratios, geometric_mean_time_ratio=geometric,
        reproducible_prompt_gain=all(v < 1 for v in ratios) and geometric < 1,
        minimum_percentage_gain_required=False, row64_over_original_client_TTFT=ttft_ratios,
        row64_subsequent_process_pp=candidates, original_subsequent_process_pp=controls,
        all_row64_processes_at_least1000=all(v >= 1000 for v in candidates),
        TTFT_and_PP_choices_independent=True, inference_adopted=False,
        limits='Three paired process medians;no sample exclusions or statistical-confidence claim. '
            '24-token outputs do not qualify sustained decode,task/agent quality or physical262144/RESTORE.')


def selfcheck():
    result = dict(route=comparison.selfcheck(), finalization=reference.finalization_controls())
    # Reproduce the reviewed gaps: a warning appended after the last gate and
    # an invalid first/warmed request hidden by reducing three later values.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='row64-log-guard-', dir=B) as temporary:
        root = Path(temporary)
        comparison.require_quiet_validation(root)
        (root / 'validation.log').write_text('')
        comparison.require_quiet_validation(root)
        (root / 'validation.log').write_text('late validation warning\n')
        try:
            comparison.require_quiet_validation(root)
        except RuntimeError:
            result['final_validation'] = dict(absent_and_empty_accepted=True, late_warning_rejected=True)
        else:
            raise RuntimeError('post-gate validation warning accepted')
    comparison.require_valid_client_ttft(dict(ttft_s=1.0, total_s=1.0))
    bad_clocks = (None, True, 0, -1, float('nan'), float('inf'), 126.0)
    rejected = 0
    for position in range(RUNS):
        for value in bad_clocks:
            requests = [dict(ttft_s=121.0, total_s=125.0) for _ in range(RUNS)]
            requests[position]['ttft_s'] = value
            try:
                for request in requests: comparison.require_valid_client_ttft(request)
            except RuntimeError:
                rejected += 1
            else:
                raise RuntimeError('invalid individual TTFT hidden by other requests')
    result['individual_TTFT'] = dict(invalid_values_rejected_at_all_four_positions=rejected)
    def sample(i):
        return dict(passed=True, complete=True, ownership_closed=True, fresh_faults=[],
            read=READ, heads=1, kernel=KERNELS[i], native_identity=dict(pid=i + 1, start_ticks=i + 10),
            subsequent_median_ms=120000 if KERNELS[i] == 'original' else 119880,
            samples=[dict(ttft_s=121.0 if KERNELS[i] == 'original' else 121.1) for _ in range(RUNS)])
    cases = [sample(i) for i in range(len(KERNELS))]
    accepted = decision(cases)
    require(accepted['reproducible_prompt_gain'] and all(v > 1 for v in
            accepted['row64_over_original_client_TTFT']), 'small gain or separate TTFT rejected')
    cases[0]['subsequent_median_ms'] = 120001
    require(not decision(cases)['reproducible_prompt_gain'], 'one slower pair accepted')
    cases[0] = sample(0); cases[1]['native_identity'] = cases[0]['native_identity']
    try:
        decision(cases)
    except RuntimeError:
        result['selection'] = dict(small_gain_kept=True, slower_TTFT_reported=True,
            mixed_pair_rejected=True, duplicate_native_rejected=True)
        return result
    raise RuntimeError('duplicate native accepted')


def prepare():
    controls = selfcheck()
    revision = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    prior, providers = qualification(revision)
    require(not ROOT.exists() and KEY not in build.committed(RESULT), 'fresh campaign only')
    sources = shared.pins(revision)
    require(set(sources) - set(prior['sources']) == ADDED
            and not (set(prior['sources']) - set(sources))
            and {n for n in set(sources) & set(prior['sources'])
                 if sources[n] != prior['sources'][n]} == CHANGED, 'undeclared frontend/controller change')
    contract, manifest = comparison.entry.python_manifest()
    require(contract == prior['python_contract'] and manifest['selected'] == prior['python_selected'],
            'same frontend interpreter required')
    assets = shared.models.model_pins(prior['configuration'], prior['model_assets'], full=True)
    ROOT.mkdir(mode=0o700)
    write(ROOT / 'runtime-inputs.json', providers)
    row = dict(stage='prepared', source_commit=revision, sources=sources,
        native_source_commit=prior['native_source_commit'], binary=prior['binary'],
        native_validation_probe=prior['native_validation_probe'], model_assets=assets,
        configuration=prior['configuration'], environment_template=prior['environment_template'],
        python_contract=contract, python_selected=manifest['selected'], template=prior['template'],
        boot=prior['boot'], driver=prior['driver'],
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'),
            sha256=digest(ROOT / 'runtime-inputs.json'), files=len(providers)),
        qualification_reuse=dict(row64_device_key=device.KEY, unchanged_native_generation=True,
            original_component_controller_inputs_recovered=True, changed_component_controls=sorted(CHANGED),
            changed_model_controls=sorted(CHANGED), added_model_controls=sorted(ADDED),
            new_native_build=False, inherited_model_speed=False),
        request_hashes={str(READ): prior['request_hashes']}, plans=[], processes=[], controls=controls,
        host_passed=False, passed=False, complete=False, active=False, ownership_closed=True,
        inference_adopted=False, whole_bytes=400 << 20, budgets=prior['budgets'],
        plan=dict(question='Does previously faster row64 shorten current-generation117K prefill/TTFT?',
            prior='V1closed after one diagnostic process because source review found missing final '
                'validation-log and per-request TTFT guards. Preserve its automatic pass and individual '
                'samples;do not continue or reuse it as matched evidence. V2uses six new processes.',
            order='Six fresh processes,row64/original,original/row64,row64/original;four matched '
                'thinking-on requests each. First long use separate;median of subsequent3 is the process unit.',
            constants='Same seven-ELF native generation,official2026.2 providers,safety controls,128staged '
                'slots,117488GPU BF16 rows,head retirement1,Top-k0,8192chunks,131072INT8 context,'
                '1256decode slots,5workers,spec4. Only Down split kernel changes before exec.',
            acceptance='All24requests must match bodies/actual117489read,zero reuse,complete thinking '
                'streams,engine prompt clock,full-GPU split placement,checked restoration,quiet validation,'
                'normal owner/native closure and no fresh fault. Every TTFT must be a finite positive '
                'number no greater than its finite total duration,including first use;validate before '
                'any median. Check validation.log at all request gates AND after the owned native '
                'session closes,including runtime teardown. Source-bound fused-job routes plus '
                'immutable compiled selector identify the production choice;no model UR launch count is claimed. '
                'All3paired warmed prompt-time ratios<1 and geometric mean<1 permit further row64 '
                'qualification;no minimum percentage. Report first-use and clientTTFT independently. '
                'The1000target separately requires all3row64 process medians>=1000.',
            stopping='Stop first source/provider/owner/clock/route/validation/fault/budget/sample failure. '
                'No speed-based stop,retry,replacement,exclusion or hash refresh. Commit every closed '
                'process before the next. Full model hashes before and after cohort;identities throughout.',
            limits='No current-generation speed is inherited from historical1007.3. No automatic length '
                'dispatch,TTFT-profile adoption,sustained decode preservation,agent/task,physical262144or '
                'RESTORE qualification. Hardware saturation/ceilings remain unknown.',
            retention='Keep individual process/request timing/route samples and decisions. Retire successful '
                'full logs,journals,JIT/owner output after each committed closure. Retain current official '
                'generation and selected fallback for this comparison and remaining phase-specific gates.'))
    for index, kernel in enumerate(KERNELS):
        root = ROOT / str(index); root.mkdir(mode=0o700)
        for part in ('tmp', 'cache'): (root / part).mkdir()
        config = copy.deepcopy(row['configuration']); config['log'] = str(root / 'engine.stderr')
        write(root / 'config.json', config)
        write(root / 'plan.json', dict(binary=row['binary'], native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        plan = dict(index=index, kernel=kernel, read=READ, heads=1, root=str(root),
            generated={n: digest(root / n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--prefill-exact-tokens', str(READ), '--think', '--runs', str(RUNS), '--profile-request-gates',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600',
                '--out', str(root / 'bench')])
        plan['environment_sha256'] = comparison.fingerprint(environment(row, plan)); row['plans'].append(plan)
    save(row)
    print('Prepared6same-ELF original/row64 processes;commit before host checks.', flush=True)


def current(row):
    require(build.committed(RESULT)[KEY] == reference.compact(row), 'commit exact campaign state')
    prior, providers = qualification(row['source_commit'])
    require(shared.pins(row['source_commit']) == row['sources'], 'frozen model inputs changed')
    inventory = row['runtime_manifest']
    require(digest(inventory['path']) == inventory['sha256'] and
            json.loads(Path(inventory['path']).read_text()) == providers and len(providers) == inventory['files'],
            'provider inventory changed')
    require(row['binary'] == prior['binary'] and row['driver'] == build.guards.driver_guard()
            and digest(row['template']['path']) == row['template']['sha256'], 'binary/platform/template differs')
    require(build.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole campaign output budget')
    shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    contract, manifest = comparison.entry.python_manifest()
    require(contract == row['python_contract'] and manifest['selected'] == row['python_selected'],
            'frontend provider changed')
    return providers, manifest


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); build.guards.no_model_owners([])
        argument = sys.argv[1:]
        if argument == ['selfcheck']: print(json.dumps(selfcheck()))
        elif argument == ['prepare']: prepare()
        elif argument == ['host']: comparison.host(sys.modules[__name__])
        elif len(argument) == 1 and argument[0] in tuple(str(i) for i in range(len(KERNELS))):
            comparison.run(int(argument[0]), sys.modules[__name__])
        else: raise RuntimeError('compare_prefill_candidates.py selfcheck|prepare|host|INDEX0..5')
