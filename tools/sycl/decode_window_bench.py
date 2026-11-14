#!/usr/bin/env python3
"""Compare existing MTP windows without rebuilding the coherent native engine.

Main-authored from this fork's measurement controls; repository LICENSE applies.
The original native Git snapshot remains the executable's source. The current
checkout supplies only the separately pinned Python frontend and controller.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode = True
import coherent_decode_bench as previous

P, B, RESULT = previous.P, previous.B, previous.RESULT
ROOT, KEY = B / 'decode-window-speed-v1', 'decode_window_speed_v1'
THIS = 'tools/sycl/decode_window_bench.py'
ARMS, WORKLOADS = previous.ARMS, previous.WORKLOADS
WORKLOAD_ID = 'decode_window_20261114'
PYTHON = previous.PYTHON
require, sha, signature = previous.require, previous.sha, previous.signature
reuse, core, runner, shared, build = (previous.reuse, previous.core,
    previous.runner, previous.shared, previous.build)
POLICIES = ('AGENTS.md', 'docs/ARTIFACT_RETENTION.md')
INDEX = None


def compact(row):
    return core.compact(row)


def save(row):
    core.write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); core.write(RESULT, data)


def retained_native():
    """Recover original inputs; do not substitute today's native source bytes.

    Project inputs used at compilation may have since changed in the checkout.
    Their exact hashes must instead match both reachable Git and the original
    compiler census. Dependencies, generated inputs, recipe, SDK and all twelve
    executable/fixture files still have to match their actual retained bytes.
    """
    native = json.loads((reuse.ROOT / 'record.json').read_text())
    require(reuse.build.validation.committed(RESULT, (reuse.KEY,)) == reuse.build.compact(native)
        and native['stage'] == 'built' and not native['active']
        and all(native[k] for k in ('passed', 'complete', 'ownership_closed'))
        and native['native_source_commit'] == native['project']['revision'] == reuse.SOURCE,
        'closed original coherent build and committed receipt required')
    project = reuse.recover(native['project'])
    recovered = dict(project)
    for pin in native['dependencies']:
        values = reuse.recover(pin)
        require(all(sha(p) == h for p, h in values.items()), 'retained dependency source changed')
        recovered.update(values)
    require(sha(native['recipe_seed']['path']) == native['recipe_seed']['sha256']
        and sha(native['external_inputs']['path']) == native['external_inputs']['sha256']
        and all(sha(p) == h for p, h in native['toolchain'].items()),
        'original native recipe/compiler/SDK manifest changed')
    external = json.loads(Path(native['external_inputs']['path']).read_text())
    require(all(sha(p) == h for p, h in external.items()), 'original SDK/link bytes changed')
    require(all(sha(p) == h for p, h in
        (native['configured_files'] | native['generated_inputs']).items()),
        'retained configured/generated native inputs changed')
    census = native['actual_inputs']
    require(sha(census['path']) == census['sha256'] and census['uncovered'] == 0
        and census['source_commit'] == reuse.SOURCE and census['translation_units'] == 134
        and len(native['planned_compilation']) == 137, 'original coherent compiler census required')
    actual = json.loads(Path(census['path']).read_text())
    allowed = recovered | external | native['generated_inputs']
    require(len(actual['compiled']) == 1470 and len(actual['links']) == 29
        and all(allowed.get(p) == h and (p in project or sha(p) == h)
            for family in ('compiled', 'generated', 'links') for p, h in actual[family].items()),
        'original compiled/header/link inputs do not match recoverable source/recipe')
    coherence = census['class_definition_coherence']
    require(coherence['header_sha256'] == project[str(P / 'include/strata/prefill/gemm.hpp')]
        and coherence['includer_count'] == 5 and coherence['same_SYCL_class_condition']
        and coherence['all_objects_from_fresh_build'] and not coherence['partial_archive_reuse'],
        'original fresh class-definition coherence required')
    require(len(native['binaries']) == 12 and all(Path(v['path']).stat().st_size == v['bytes']
        and sha(v['path']) == v['sha256'] for v in native['binaries'].values()),
        'retained engine or a required same-build fixture changed')
    require(reuse.build.probe_qualification() ==
        (native['native_validation_probe'], native['validation_qualification']),
        'qualified pre-main validation probe changed')
    evidence = {}
    for key in ('coherent_prefill_device_v1', 'coherent_prefill_model_v3', 'coherent_prefill_speed_v1'):
        value = reuse.build.validation.committed(RESULT, (key,))
        require(value['stage'] == 'closed' and not value['active']
            and all(value[k] for k in ('passed', 'complete', 'ownership_closed'))
            and value['native_source_commit'] == reuse.SOURCE, 'closed coherent qualification required')
        evidence[key] = signature(value)
    device = reuse.build.validation.committed(RESULT, ('coherent_prefill_device_v1',))
    require(device['binaries'] == native['binaries'] and len(device['results']) == 11
        and all(v['passed'] and v['ownership_closed'] and v['native_exit_code'] == 0
            for v in device['results']) and not device['results'][0]['GPU_executed']
        and all(v['GPU_executed'] and not v['fresh_faults'] and v['boot_unchanged']
            and v['driver_after'] == device['driver'] for v in device['results'][1:]),
        'original same-build native gates differ')
    proof = dict(original_native_source=reuse.SOURCE,
        original_record_sha256=sha(reuse.ROOT / 'record.json'),
        original_source_dependency_files=len(recovered), original_recipe_sha256=signature(native['recipe']),
        actual_inputs_sha256=census['sha256'], actual_compiler_inputs=1470, external_link_inputs=29,
        actual_objects=137, coherent_translation_units=134,
        same_build_executables=native['binaries'], closed_evidence_sha256=evidence,
        current_native_checkout_is_execution_input=False, native_recompiled=False,
        new_GPU_or_model_qualification=False)
    return native, proof


def basis(expected=None):
    native, proof = retained_native()
    require(expected is None or expected == proof, 'frozen original native proof changed')
    prior = json.loads((previous.prefill.ROOT / 'record.json').read_text())
    require(reuse.build.validation.committed(RESULT, (previous.prefill.KEY,)) == core.compact(prior)
        and prior['stage'] == 'closed' and prior['passed'] and prior['complete']
        and prior['ownership_closed'] and not prior['active']
        and prior['binary'] == native['binaries']['strata']
        and prior['native_source_commit'] == reuse.SOURCE, 'closed clean prefill basis required')
    providers = json.loads(Path(prior['runtime_manifest']['path']).read_text())
    require(sha(prior['runtime_manifest']['path']) == prior['runtime_manifest']['sha256']
        and all(sha(p) == h for p, h in providers.items())
        and build.guards.driver_guard() == prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot'],
        'runtime, boot or driver changed')
    return native, proof, prior, providers


def configuration(row, plan):
    value = copy.deepcopy(row['configuration']); value['log'] = str(Path(plan['root']) / 'engine.stderr')
    args = value['args']
    require(args.count('--spec') == 1 and args[args.index('--spec') + 1] == '4'
        and args[args.index('--pcie-frac') + 1] == '0'
        and args[args.index('--adapt-swaps') + 1] == '0', 'preserve qualified static placement')
    args[args.index('--spec') + 1] = '2' if plan['kernel'] == 'candidate' else '4'
    return value


def invariant(row, plan):
    value = dict(config=configuration(row, plan), environment=previous.environment(row, plan))
    args = value['config']['args']; args[args.index('--spec') + 1] = '<window>'
    return signature(json.dumps(value, sort_keys=True).replace(plan['root'], '<process>'))


def current(row):
    require(reuse.build.validation.committed(RESULT, (KEY,)) == compact(row)
        and core.pins(row['source_commit']) == row['sources']
        and all(sha(P / p) == h for p, h in row['policies'].items()),
        'commit and preserve exact controller/frontend/policy inputs')
    native, proof, prior, providers = basis(row['native_observer_admission'])
    require(row['binary'] == native['binaries']['strata']
        and row['native_source_commit'] == reuse.SOURCE
        and all(row[k] == prior[k] for k in ('native_validation_probe', 'configuration',
            'environment_template', 'model_assets', 'runtime_manifest', 'template')),
        'native/configuration/assets differ')
    contract, manifest = runner.model.python_manifest()
    require(contract == row['python_contract'] and manifest['selected'] == row['python_selected']
        and sha(row['template']['path']) == row['template']['sha256']
        and build.supervision.artifact_bytes(ROOT) <= row['whole_bytes']
        and len(row['plans']) == len(ARMS), 'frontend/template/campaign budget differs')
    for i, plan in enumerate(row['plans']):
        root = Path(plan['root'])
        require(plan['index'] == i and plan['kernel'] == ARMS[i] and plan['heads'] == 1
            and plan['adapt_swaps'] == 0 and invariant(row, plan) == row['invariant_signature']
            and json.loads((root / 'config.json').read_text()) == configuration(row, plan)
            and json.loads((root / 'plan.json').read_text()) == dict(binary=row['binary'],
                native_validation_probe=row['native_validation_probe'])
            and all(sha(root / n) == h for n, h in plan['generated'].items())
            and runner.comparison.fingerprint(previous.environment(row, plan)) == plan['environment_sha256'],
            'actual generated arm/configuration/environment differs')
    shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    return providers, manifest


def decision(processes):
    result = previous.runner_context().original_decision(processes)
    for name, group in result['workloads'].items():
        for arm, values in group['arms'].items():
            rates = [s['acceptance'] for s in values['samples']]
            values['observed_MTP_class'] = ('strong' if all(x >= .60 for x in rates)
                else 'weak' if all(x <= .40 for x in rates) else 'intermediate_or_variable')
        group['reference_workload_class_confirmed'] = group['arms']['reference']['observed_MTP_class'] == (
            'strong' if name == 'regular-code' else 'weak')
        group['candidate_screen_eligible_for_followup'] = (group['reproducible_TG_gain']
            and group['reference_workload_class_confirmed'])
    result.update(variable='--spec4 reference versus --spec2 candidate;all other settings matched',
        inference_adopted=False, physical262144_or_RESTORE_qualified=False,
        next='Evaluate each workload separately. Candidate acceptance can change with the window; '
            'classify both arms from actual offers. Preserve prefill/decode separation and collect '
            'remaining dedicated-prefill,agent,task,context/length and ordinary-teardown evidence before adoption.')
    return result


def context():
    spec = importlib.util.spec_from_file_location('decode_window_private_context', previous.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.WORKLOAD_ID, ctx.INDEX = ROOT, KEY, THIS, WORKLOAD_ID, INDEX
    ctx.configuration, ctx.invariant, ctx.current = configuration, invariant, current
    ctx.save, ctx.compact, ctx.decision = save, compact, decision
    original_copy = previous.copy_records
    ctx.copy_records = lambda text, samples=None, *, arm: original_copy(text, samples, arm='reference')
    return ctx


def controls():
    previous.copy_controls(); previous.draft_controls()
    row = dict(configuration=dict(log='old', args=['--spec', '4', '--pcie-frac', '0', '--adapt-swaps', '0']))
    for arm, window in (('reference', '4'), ('candidate', '2')):
        plan = dict(root='/tmp/decode-window-control-' + arm, kernel=arm)
        config = configuration(row, plan)
        require(config['args'][1] == window and row['configuration']['args'][1] == '4',
            'window arm/source mutation control')
    for args in (['--spec', '2', '--pcie-frac', '0', '--adapt-swaps', '0'],
        ['--spec', '4', '--pcie-frac', '0.0625', '--adapt-swaps', '0'],
        ['--spec', '4', '--pcie-frac', '0', '--adapt-swaps', '8']):
        try: configuration(dict(configuration=dict(args=args)), dict(root='/tmp/x', kernel='candidate'))
        except (RuntimeError, KeyError): pass
        else: raise RuntimeError('changed base configuration accepted')
    return dict(passed=True, windows=[4, 2], base_unchanged=True, rejected_changed_bases=3,
        copy_and_draft_controls=True, GPU_executed=False)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh comparison only')
    native, proof, prior, _ = basis()
    contract, manifest = runner.model.python_manifest()
    assets = shared.models.model_pins(prior['configuration'], prior['model_assets'], full=True)
    row = {k: copy.deepcopy(prior[k]) for k in ('binary', 'native_source_commit',
        'native_validation_probe', 'configuration', 'environment_template', 'template', 'boot', 'driver')}
    revision = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    row.update(stage='prepared', source_commit=revision, sources=core.pins(revision),
        native_observer_admission=proof, policies={p: sha(P / p) for p in POLICIES},
        python_contract=contract, python_selected=manifest['selected'], model_assets=assets,
        runtime_manifest=prior['runtime_manifest'], plans=[], processes=[], host_passed=False,
        passed=False, complete=False, active=False, ownership_closed=True, whole_bytes=448 << 20,
        budgets=dict(wall=2700, cpu=16200, rss_cap=104 << 30, text_cap=16 << 20,
            file_cap=32 << 20, total_cap=64 << 20),
        plan=dict(question='Does a two-token MTP target window reduce wasted verification enough to improve sustained decode?',
            evidence='Clean spec4 code/prose acceptance87.009%/33.935%,TG15.037/9.073. '
                'Current measured decode uses T4;only T2 mapped-row copy passed its isolated component screen. '
                'No physical CPU,PCIe,VRAM or GPU saturation follows from service shares. '
                'This whole-entrypoint configuration test includes all resulting compute,copy,wait and dispatch costs.',
            order='Six independent processes R,C,C,R,R,C. Each runs four official short warmups, '
                'then regular-code and varied-prose once with the same fixed workload id. '
                'Commit each closed process before the next. No speed-based early stopping.',
            constants='Original coherent14508/dcb8 executable and12fixtures;262144context,20480INT8residentKV, '
                '1256static expert slots,5workers,PCIE0/adapt0,suffix0,min-p0;8192BF16/vLLM ordered-exp, '
                'largeGU,head2/dense1,throughput strategy. Only --spec4/2 differs. '
                'No native build or process-environment mutation;preserve qualified runtime safety.',
            observers='Same clean nominal Arena/copy/cache/DRAFT_PREFILL footers and quiet parameter validation '
                'as closed coherent_decode_pcie_speed_v3. No per-job,census,trace,query,caller or stage-timer observer.',
            acceptance='All6processes/12long requests:>=32768fresh input,>=1024generated including thinking, '
                '>=512answer chars,>=256drafts,normal SSE/journal/clocks/TTFT and identical request bodies. '
                'Exact native argv,source/build/SDK/12ELF/provider/frontend/model identities;all15full asset '
                'hashes before/after cohort;normal0,closed owners,quiet validation,no fresh fault. '
                'Actual largeGU/Down/typed long-prefill route and full read-1 batched8192MTP callbacks. '
                'All warmup/long native prompt/copy/API counts,drafts,rounded clocks bind;copies0in both arms. '
                'Short warmups retain their separate bounded callback contract.',
            decision='Retain every sample and3adjacent paired TG,prompt-time andTTFT ratios per workload. '
                'Reproducible TG gain requires all3candidate/reference ratios>1,no minimum gain floor. '
                'Reference code must actually be strong>=.60 and reference prose weak<=.40. '
                'Report candidate class independently:changing window may change acceptance and is not an eligibility failure. '
                'Do not adopt a global window from one workload;prefill preservation,agent/task,fullphysical262144/'
                'RESTORE,ordinary teardown and length dependence remain separate. Output equality is not required.',
            stopping='First identity,route,journal,clock,validation,fault,budget or sample-eligibility failure stops. '
                'Exclusive attempt precedes preflight;no retry,hash refresh,exclusion or retrospective criteria changes.',
            retention='No new engine generation. Retain compact per-process samples and original-source/recipe proof. '
                'Retire successful raw output after verified closure. The unused mapped-row prototype remains '
                'only until this named configuration decision;faster spec2would require its own integration plan.'))
    ROOT.mkdir(mode=0o700)
    for index, arm in enumerate(ARMS):
        root = ROOT / str(index); root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root / name).mkdir()
        plan = dict(index=index, kernel=arm, adapt_swaps=0, heads=1, root=str(root))
        core.write(root / 'config.json', configuration(row, plan))
        core.write(root / 'plan.json', dict(binary=row['binary'], native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        plan.update(generated={n: sha(root / n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--long-generation-only', '--think', '--runs', '1', '--profile-request-gates',
                '--workload-id', WORKLOAD_ID, '--fail-fast-engine-error', '--ready-timeout', '600',
                '--out', str(root / 'bench')])
        plan['environment_sha256'] = runner.comparison.fingerprint(previous.environment(row, plan))
        row['plans'].append(plan)
    row['invariant_signature'] = invariant(row, row['plans'][0])
    require(all(invariant(row, p) == row['invariant_signature'] for p in row['plans']), 'undeclared arm difference')
    save(row)
    print('Prepared6same-engine window4/2processes;commit plan before host/model.', flush=True)


def execute(action):
    global INDEX
    INDEX = int(action) if action.isdigit() else None
    attempt = ROOT / ('host-attempt.json' if INDEX is None else f'{INDEX}/attempt.json')
    with attempt.open('x') as stream: json.dump(dict(action=action, result_key=KEY), stream)
    row = json.loads((ROOT / 'record.json').read_text())
    try:
        if INDEX is None:
            current(row); result = controls()
            runner.comparison.host(context().runner_context())
            row = json.loads((ROOT / 'record.json').read_text()); row['host_controls'] = result; save(row)
        else:
            context().runner_context().run(INDEX)
    except BaseException as error:
        row = json.loads((ROOT / 'record.json').read_text())
        if row['stage'] != 'failed':
            row.update(stage='failed', passed=False, complete=False, error=repr(error), failed_action=action)
            save(row)
        raise


if __name__ == '__main__':
    require(sys.executable == PYTHON, 'Use the qualified interpreter: ' + PYTHON)
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host', *map(str, range(len(ARMS)))), 'prepare|host|0..5')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); build.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
