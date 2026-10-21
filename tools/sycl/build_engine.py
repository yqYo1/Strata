#!/usr/bin/env python3
"""Build the current comparison engine from verified Git inputs; Main only.

Uses the known-good engine's committed configure recipe, changing only the
output directory. The explicit copy-safety build includes its bounded fixtures.
No engine/GPU execution. The CPU-census mode runs its bounded host-only contract
before compiling the engine. An explicit rebuild reuses a closed comparison's cache
only after verifying its original inputs from Git.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'main-native-gdn-engine'
BASELINE = P / 'bench/results/2026-10-15-pr6-recovery-audit/qsa-score-gemm-entry-engine-build-v2.json'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/native-gdn-engine-build.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binary_origin_source(previous, binary_sha256, current_revision):
    """Preserve known byte identity; this alone does not reuse qualification.

    main() recovers the previous build's exact source/recipe before this call.
    Fixture-only edits can change native inputs without changing the engine.
    """
    if previous and previous['passed'] and previous['binary']['sha256'] == binary_sha256:
        return previous.get('binary_origin_source_commit', previous['source_commit'])
    return current_revision


def blobs(root, revision, names):
    """Recover exact Git blobs in bounded batches, without one process per file."""
    names = list(names)
    for start in range(0, len(names), 128):
        batch = names[start:start + 128]
        guards.require(all('\n' not in n and '\r' not in n for n in batch), 'unsupported source pathname')
        request = ''.join(revision + ':' + n + '\n' for n in batch).encode()
        data = subprocess.check_output(['git', '-C', str(root), 'cat-file', '--batch'], input=request)
        offset = 0
        for name in batch:
            end = data.index(b'\n', offset)
            fields = data[offset:end].split()
            guards.require(len(fields) == 3 and fields[1] == b'blob', 'unrecoverable source: ' + name)
            length = int(fields[2]); begin = end + 1; end = begin + length
            guards.require(length <= 256 << 20 and data[end:end + 1] == b'\n', 'Git blob extent')
            yield name, data[begin:end]
            offset = end + 1
        guards.require(offset == len(data), 'unexpected Git batch output')


def pin(root, prefixes):
    revision = git(root, 'rev-parse', 'HEAD').decode().strip()
    paths = git(root, 'ls-files', '-z', '--', *prefixes).decode().strip('\0').split('\0')
    guards.require(paths and all(paths), 'empty source selection')
    pins = {}
    for name, data in blobs(root, revision, sorted(paths)):
        path = root / name
        guards.require(path.read_bytes() == data, 'uncommitted build input: ' + str(path))
        pins[name] = hashlib.sha256(data).hexdigest()
    extras = git(root, 'ls-files', '--others', '--exclude-standard', '-z', '--', *prefixes).decode().split('\0')
    guards.require(not any(x and Path(x).suffix in ('.h', '.hpp', '.cpp', '.c', '.cmake') for x in extras),
                   'untracked source in build tree: ' + str(root))
    aggregate = hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(root=str(root), revision=revision, count=len(pins), source_sha256=pins, aggregate_sha256=aggregate)


def fresh_retained_reference(compact):
    """Carry the validated fallback across deletion of a closed candidate cache."""
    guards.require(compact.get('retirement', {}).get('complete') is True,
        'missing cache requires an explicit completed retirement decision')
    reference=compact['retained_validation_reference']
    receipt=reference['source_build_record']
    guards.require(digest(receipt['path'])==receipt['sha256'], 'retained source receipt changed')
    full=json.loads(Path(receipt['path']).read_text())
    guards.require(full['source_commit']==reference['source_commit'], 'retained source identity differs')
    guards.require(reference['fixtures'].keys()==full['fixtures'].keys(),
        'retained fixture selection differs from the original build')
    for name,item in dict(strata=reference['binary'],**reference['fixtures']).items():
        original=full['binary'] if name=='strata' else full['fixtures'][name]
        # Original fixture receipts pin hashes without sizes. The relocation
        # receipt pins both; check its actual size without rewriting history.
        guards.require(item['sha256']==original['sha256'] and
            ('bytes' not in original or item['bytes']==original['bytes']) and
            not Path(item['path']).resolve().is_relative_to(ROOT.resolve()) and
            Path(item['path']).stat().st_size==item['bytes'] and
            digest(item['path'])==item['sha256'], 'retained artifact identity or protected location differs')
    for source in [full['project'],*full['dependencies']]:
        recovered={n:hashlib.sha256(data).hexdigest() for n,data in
            blobs(Path(source['root']),source['revision'],source['source_sha256'])}
        guards.require(recovered==source['source_sha256'],'retained original inputs cannot be recovered')
    recipe=full['recipe_reference']
    guards.require(hashlib.sha256(git(Path(full['project']['root']),'show',
        full['project']['revision']+':'+recipe['path'])).hexdigest()==recipe['sha256'],
        'retained original recipe cannot be recovered')
    return reference


def retained_model_evidence(build, prefill, generation):
    """Same-build execution evidence; this does not qualify speed or adoption."""
    for value in (prefill, generation):
        guards.require(value.get('build_source_commit') == build['source_commit'] and
                       value.get('binary') == build['binary'] and
                       value.get('passed') is True and value.get('complete') is True and
                       not value.get('active', False),
                       'fallback model evidence belongs to another build or is incomplete')
    runs = prefill.get('runs', [])
    guards.require(len(runs) == 2 and prefill.get('full_model_hashes_matched') == 15,
                   'two complete same-build prefill observations required')
    for row in runs:
        sample = row.get('sample', {})
        guards.require(row.get('passed') is True and row.get('complete') is True and
                       row.get('ownership_closed') is True and row.get('exit_code') == 0 and
                       row.get('boot_unchanged') is True and
                       row.get('fault_probe_successful') is True and row.get('fresh_faults') == [] and
                       sample.get('ok') is True and sample.get('eligible') is True and
                       sample.get('stream_complete') is True and
                       type(sample.get('actual_read_tokens')) is int and
                       sample['actual_read_tokens'] >= 117489 and
                       sample.get('actual_prompt_tokens') == sample['actual_read_tokens'] and
                       sample.get('actual_reused_tokens') == 0,
                       'same-build117K execution/ownership/fault evidence is incomplete')
    rows = generation.get('processes', [])
    guards.require(generation.get('stage') == 'closed' and
                   generation.get('ownership_closed') is True and len(rows) == 6,
                   'closed six-process same-build generation evidence required')
    for row in rows:
        sample, fault = row.get('request', {}), row.get('fault_probe', {})
        guards.require(row.get('passed') is True and row.get('ownership_closed') is True and
                       row.get('native_exit_code') == 0 and fault.get('fresh_faults') == [] and
                       fault.get('boot_unchanged') is True and
                       row.get('journal_valid') is True and
                       sample.get('ok') is True and sample.get('stream_complete') is True and
                       sample.get('truncated') is False and
                       type(sample.get('read_tokens')) is int and sample['read_tokens'] >= 32768 and
                       sample.get('prompt_tokens') == sample['read_tokens'] and
                       sample.get('reused_tokens') == 0 and
                       type(sample.get('generated_tokens')) is int and sample['generated_tokens'] >= 1024 and
                       sample.get('generated_tokens_source') == 'usage' and
                       sample.get('completion_includes_thinking') is True and
                       sample.get('output_chars', 0) >= 512 and sample.get('reasoning_chars', 0) > 0,
                       'same-build sustained-generation/ownership/fault evidence is incomplete')
    return dict(prefill_observations=len(runs), long_generation_processes=len(rows),
                prefill_kind='instrumented117K execution, not clean speed',
                prefill_read_tokens=sorted({row['sample']['actual_read_tokens'] for row in runs}),
                generation_kind='minimal-marker long generation',
                generation_read_tokens=sorted({row['request']['read_tokens'] for row in rows}),
                clean117K_speed_qualified=False, model_adoption_qualified=False)


def packed_model_evidence(build, component, prefill, generation):
    """Bind the retained packed generation, never the retired 3b model reports.

    Original receipts and executable hashes supply qualification continuity;
    the source/recipe recovery and all fixture bytes are checked by the caller.
    This is retention admission, not promotion of pilot speed or model quality.
    """
    for value in (component, prefill, generation):
        guards.require(value.get('passed') is True and value.get('complete') is True and
            value.get('ownership_closed') is True and not value.get('active', False),
            'closed component/prefill/generation evidence required')
    guards.require(component['source_commit'] == build['source_commit'] and
        component['binary'] == build['binary'] and component['fixtures'] == build['fixtures'] and
        component['fresh_faults'] == [] and
        prefill['build_source_commit'] == build['source_commit'] and prefill['binary'] == build['binary'] and
        generation['stage'] == prefill['stage'] == 'closed', 'same-build packed qualification required')
    rows = prefill['processes']
    guards.require(len(rows) == 6 and [r['arm'] for r in rows] == [0, 1, 1, 0, 0, 1],
        'closed six-process prefill cohort required')
    for row in rows:
        guards.require(row['passed'] and row['complete'] and row['ownership_closed'] and not row['active'] and
            row['native_exit_code'] == 0 and row['fresh_faults'] == [] and row['fault_probe_successful'] and
            row['boot_unchanged'] and row['actual_route_checked'] and row['observer_records_absent'] and
            row['build_source_commit'] == build['source_commit'] and
            row['binary_sha256'] == build['binary']['sha256'] and len(row['requests']) == 4,
            'prefill native/route/fault identity differs')
        for sample in row['requests']:
            guards.require(sample['ok'] and sample['eligible'] and sample['stream_complete'] and
                sample['actual_read_tokens'] >= 117489 and
                sample['actual_prompt_tokens'] == sample['actual_read_tokens'] and sample['actual_reused_tokens'] == 0,
                'complete fresh117K prefill required')
    candidates = [r for r in generation['processes'] if
        r['build_source_commit'] == build['source_commit'] and r['binary_sha256'] == build['binary']['sha256']]
    guards.require(len(generation['processes']) == 2 and len(candidates) == 1,
        'unambiguous same-build generation pilot required')
    row, = candidates
    guards.require(row['passed'] and row['complete'] and row['ownership_closed'] and not row['active'] and
        row['native_exit_code'] == 0 and row['fresh_faults'] == [] and row['fault_probe_successful'] and
        row['boot_unchanged'] and row['effective_environment_verified'] and row['journal']['valid'] and
        len(row['requests']) == len(row['routes']) == 2 and
        [r['workload'] for r in row['requests']] == ['regular-code', 'varied-prose'],
        'same-build generation/route/owner/fault evidence required')
    for sample, route in zip(row['requests'], row['routes']):
        guards.require(sample['ok'] and sample['eligible'] and sample['stream_complete'] and
            sample['read_tokens'] >= 32768 and sample['prompt_tokens'] == sample['read_tokens'] and
            sample['reused_tokens'] == 0 and sample['generated_tokens'] >= 1024 and
            sample['generated_tokens_source'] == 'usage' and sample['completion_includes_thinking'] and
            sample['reasoning_chars'] > 0 and sample['output_chars'] >= 512 and
            route['actual_MoE_mode'] == 10 and route['restored_dense_layers'] == 48 and
            route['cache_payload_and_map_match'] and route['private_bank']['actual_private_layers'] == 48,
            'complete thinking/answer/restored-route pilot required')
    return dict(prefill_processes=6, prefill_requests=24, current_long_generation_requests=2,
        prefill_read_tokens=sorted({s['actual_read_tokens'] for r in rows for s in r['requests']}),
        generation_read_tokens=[s['read_tokens'] for s in row['requests']],
        clean117K_speed_observed=True, sustained_decode_speed_qualified=False,
        model_adoption_qualified=False)


def census_model_evidence(build, component, census, build_record_sha256, component_sha256):
    """Retain the census-qualified build, without borrowing earlier speed claims."""
    from query_profile import component_evidence
    from generation_engine import sample_eligible
    groups = component_evidence(build, component)
    guards.require(component.get('boot_unchanged') is True and
        all(r.get('passed') is True and r.get('exit_code') == r.get('expected_exit')
            for r in component['checks']), 'closed same-build fixture outcomes required')
    host = build['cpu_census_host_contract']
    guards.require(host['passed'] is True and host['ownership_closed'] is True and
        host['exit_code'] == 0 and host['cases'] == 15 and
        host['fixture_sha256'] == build['fixtures']['cpu_work_census_host_test']['sha256'],
        'same-build census host contract required')
    guards.require(census.get('passed') is True and census.get('complete') is True and
        census.get('ownership_closed') is True and not census.get('active', False) and
        census.get('stage') == 'closed' and census.get('build_source_commit') == build['source_commit'] and
        census.get('binary') == build['binary'] and census.get('native_exit_code') == 0 and
        census.get('boot_unchanged') is True and census.get('fault_probe_successful') is True and
        census.get('fresh_faults') == [] and census.get('effective_environment_verified') is True,
        'closed same-build census native/owner/fault evidence required')
    proof = census['qualification']
    guards.require(proof['build_receipt_sha256'] == build_record_sha256 and
        proof['shared_component_sha256'] == component_sha256 and
        type(proof['original_inputs_recovered']) is int and proof['original_inputs_recovered'] > 0,
        'census qualification receipt identity differs')
    bank = census['bank']
    guards.require(bank['passed'] is True and bank['complete'] is True and bank['ownership_closed'] is True and
        not bank['active'] and bank['boot_unchanged'] is True and bank['fault_probe_successful'] is True and
        bank['fresh_faults'] == [] and [len(r['cases'])for r in bank['checks']] == [2,3] and
        [r['mode']for r in bank['checks']] == ['packed-smoke','packed-validate'] and
        all(r['passed'] is True and r['exit_code'] == 0 and r['cases'] and
            r['fixture'] == build['fixtures']['sycl_expert_bank_test'] for r in bank['checks']),
        'closed same-build packed-bank qualification required')
    warmups = census['warmups']
    samples, routes = census['requests'], census['routes']
    guards.require(warmups['valid'] is True and warmups['successful_warmups'] == 4 and
        warmups['successful_long_requests'] == 2 and warmups['phase_order'] ==
        ['warm-up/chat','calibration','calibration','warm-up/prompt',
         'long-generation/regular-code','long-generation/varied-prose'] and
        len(warmups['request_sha256']) == 6 and len(samples) == len(routes) == 2 and
        [r['workload']for r in samples] == ['regular-code','varied-prose'] and
        [r['request_sha256']for r in samples] == warmups['request_sha256'][4:],
        'complete official warmup/request binding required')
    for sample, route in zip(samples, routes):
        guards.require(sample_eligible(sample) and sample.get('truncated') is False and
            route['actual_GPU_BF16_rows'] == sample['read_tokens']-1 and
            route['actual_MoE_mode'] == 10 and route['restored_dense_layers'] == 48 and
            route['cache_payload_and_map_match'] is True and
            route['private_bank']['actual_private_layers'] == 48,
            'complete thinking/answer and actual restored route required')
    return dict(component_groups=groups,official_warmups=4,current_long_generation_requests=2,
        generation_read_tokens=[r['read_tokens']for r in samples],
        original_inputs_recovered=proof['original_inputs_recovered'],
        clean117K_speed_qualified=False,sustained_decode_speed_qualified=False,
        hardware_saturation_qualified=False,model_adoption_qualified=False)


def recovery_model_evidence(build, component, recovery, build_sha256, component_sha256):
    """Retain exact phase-scoped117K evidence, without inventing long decode."""
    from query_profile import component_evidence
    groups = component_evidence(build, component)
    guards.require(component.get('boot_unchanged') is True and
        all(r.get('passed') is True and r.get('exit_code') == r.get('expected_exit')
            for r in component['checks']), 'closed same-build fixture outcomes required')
    phases = build['cpu_phase_host_contract']
    guards.require(len(phases) == 2 and [r['arm'] for r in phases] == [0,1] and
        all(r['passed'] and r['ownership_closed'] and r['cases'] == 12 and
            r['fixture_sha256'] == build['fixtures']['iq2_phase_host_test']['sha256'] for r in phases),
        'same-build24 phase cases required')
    proof = recovery['qualification']
    guards.require(recovery['stage'] == 'closed' and recovery['binary'] == build['binary'] and
        recovery['plan']['native_source'] == build['source_commit'] and
        proof['build_sha256'] == build_sha256 and proof['component_sha256'] == component_sha256 and
        type(proof['original_inputs_recovered']) is int and proof['original_inputs_recovered'] > 0 and
        len(recovery['models']) == 15, 'closed same-build recovery provenance required')
    rows = recovery['processes']; bank = recovery['bank']
    guards.require([r['mode'] for r in rows] == ['checked','clean-0','clean-1','clean-2'] and
        bank['passed'] and bank['complete'] and bank['ownership_closed'] and not bank['active'] and
        bank['boot_unchanged'] and bank['fault_probe_successful'] and bank['fresh_faults'] == [] and
        [(r['mode'],len(r['cases']),r['passed']) for r in bank['checks']] ==
            [('packed-smoke',2,True),('packed-validate',3,True)] and
        all(c['status'] == '0' and all(c[k] == 'PASS' for k in
            ('immutable_slab','guards','private_map','checked_drain'))
            for r in bank['checks'] for c in r['cases']), 'closed packed/recovery cohort required')
    for i,row in enumerate(rows):
        samples = row['requests']; route = row['route']
        guards.require(row['passed'] and row['complete'] and row['ownership_closed'] and not row['active'] and
            row['native_exit_code'] == 0 and row['boot_unchanged'] and row['fault_probe_successful'] and
            row['fresh_faults'] == [] and len(samples) == (1 if i == 0 else 4) and
            route['expert_cache_slots' if i else 'slots'] == 1256 and
            row['journal']['successful_warmups'] == 1 and
            row['journal']['successful_long_requests'] == len(samples) and
            row['journal']['request_sha256'][1:] == [r['request_sha256'] for r in samples],
            'same-placement recovery execution/owner/fault/request evidence required')
        for sample in samples:
            guards.require(sample['ok'] and sample['eligible'] and sample['stream_complete'] and
                sample['actual_read_tokens'] == sample['actual_prompt_tokens'] == 117489 and
                sample['actual_reused_tokens'] == 0 and sample['thinking_requested'] and
                sample['reasoning_observed'] and sample['generated_tokens'] == 24,
                'fresh117K short-thinking request evidence required')
        if i:
            guards.require(route['actual_route_checked'] and route['residual_gpu_rows'] == 117488 and
                route['mode'] == 10, 'actual recovery GPU route required')
        else:
            guards.require(route['exact_payload_and_map_cycles'] == route['completed_requests'] == 2 and
                route['dense_layers'] == 48 and route['prefill_rows'] == 117488 and
                route['checked_retirements'] == route['borrowed_ffn_entries'] == 96 and
                route['execution_cache_parent_checked'], 'checked cache/dense restoration required')
    return dict(component_groups=groups,phase_host_cases=24,prefill_requests=13,
        clean_prefill_processes=3,prefill_read_tokens=[117489],generated_tokens_per_request=24,
        original_inputs_recovered=proof['original_inputs_recovered'],
        clean117K_speed_observed=True,sustained_decode_speed_qualified=False,
        hardware_saturation_qualified=False,model_adoption_qualified=False)


def archived_model_observation(reference):
    """Recover an explicitly selected original model receipt, without fallback."""
    guards.require(type(reference) is dict and set(reference) == {'commit', 'path', 'sha256'} and
        all(type(reference[k]) is str for k in reference), 'explicit model receipt reference required')
    revision, path, sha = (reference[k] for k in ('commit', 'path', 'sha256'))
    guards.require(len(revision) == 40 and len(sha) == 64 and
        all(c in '0123456789abcdef' for c in revision + sha) and
        Path(path).parent == RESULT.relative_to(P).parent and Path(path).suffix == '.json',
        'model receipt must pin an immutable commit, result path and SHA256')
    raw = git(P, 'show', revision + ':' + path)
    guards.require(hashlib.sha256(raw).hexdigest() == sha, 'original model receipt hash differs')
    return json.loads(raw)


def retain_current():
    """Preserve the validated generation before a native cache is overwritten."""
    raw = RESULT.read_bytes()
    guards.require(raw == git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))), 'commit current build identity first')
    compact = json.loads(raw)
    guards.require(not compact.get('retirement'), 'closed candidate is retired; use an explicit fresh build plan')
    full_path = ROOT / 'record.json'
    old = json.loads(full_path.read_bytes())
    guards.require(old['passed'] and old['complete'] and old['ownership_closed'] and not old['active'] and
                   old['source_commit'] == compact['source_commit'] and old['binary'] == compact['binary'],
                   'current validated build identity/ownership differs')
    evidence, observations = {}, {}
    census_build = 'cpu_work_census_host_test' in old['fixtures']
    recovery_build = compact.get('retention_model_kind') == 'prefill-recovery'
    guards.require(compact.get('retention_model_kind') in (None,'prefill-recovery'), 'unknown retention model kind')
    model_reference = compact.get('retention_model_reference')
    model_name = None
    if model_reference is not None:
        guards.require(census_build and type(compact.get('retention_consumer')) is str and
            bool(compact['retention_consumer'].strip()), 'named census-build retention consumer required')
        # Validation rejects a missing/wrong archive rather than silently using
        # the older census result at a reused filename/build directory.
        selected = archived_model_observation(model_reference)
        model_name = Path(model_reference['path']).name
        observations[model_name] = selected
        evidence[model_name] = model_reference['commit']
    names = ('decode-query-profile-component-current.json',) + (
        (() if model_reference is not None else ('cpu-work-census-current.json',)) if census_build else
        ('prefill-packed-bank-speed-current.json', 'generation-reference-current.json'))
    for name in names:
        path = RESULT.with_name(name)
        value = json.loads(path.read_bytes())
        guards.require(path.read_bytes() == git(P, 'show', 'HEAD:' + str(path.relative_to(P))),
                       'commit exact current model evidence: ' + name)
        observations[name] = value
        evidence[name] = git(P, 'log', '-1', '--format=%H', '--', str(path.relative_to(P))).decode().strip()
    if recovery_build:
        guards.require(model_reference is not None and model_name == 'prefill-recovery-current.json',
            'explicit committed recovery model reference required')
        scope = recovery_model_evidence(old, observations['decode-query-profile-component-current.json'],
            observations[model_name], digest(full_path), digest(RESULT.with_name(names[0])))
        qualification = 'Same-build18 fixture groups,24 CPU phase cases and checked plus three clean117K processes passed with24 generated tokens/request. This is retention/execution evidence,not sustained decode,task quality,agent sessions,physical262144/RESTORE or model adoption.'
    elif census_build:
        scope = census_model_evidence(old, observations['decode-query-profile-component-current.json'],
            observations[model_name or 'cpu-work-census-current.json'], digest(full_path), digest(RESULT.with_name(names[0])))
        qualification = 'Same-build18 fixture groups and two33K thinking-generation census requests closed without fresh faults. This is execution/retention qualification,not clean117K or sustained decode speed,task quality,agent sessions,physical262144/RESTORE or model adoption.'
    else:
        scope = packed_model_evidence(old, observations['decode-query-profile-component-current.json'],
            observations['prefill-packed-bank-speed-current.json'], observations['generation-reference-current.json'])
        qualification = 'Same-build component/fixture/source checks, six clean117K processes and the current arm of the two-engine33K thinking-generation pilot closed without fresh faults. A pilot does not qualify sustained decode speed,task quality,agent sessions,physical262144/RESTORE or model adoption.'
    recovered = 0
    for source in [old['project'], *old['dependencies']]:
        for name, data in blobs(Path(source['root']), source['revision'], source['source_sha256']):
            guards.require(hashlib.sha256(data).hexdigest() == source['source_sha256'][name], 'original source not recovered')
            recovered += 1
    if census_build:
        guards.require(recovered == scope['original_inputs_recovered'], 'census original-input count differs')
    recipe = old['recipe_reference']
    guards.require(hashlib.sha256(git(Path(old['project']['root']), 'show',
                   old['project']['revision'] + ':' + recipe['path'])).hexdigest() == recipe['sha256'],
                   'original recipe not recovered')
    prior = compact.get('retained_validation_reference', {})
    guards.require(not prior or not Path(prior['binary']['path']).exists(),
                   'retire the previous completed fallback before retaining another generation')
    target = B / ('known-good-' + old['source_commit'][:12])
    guards.require(not target.exists(), 'never overwrite a retained generation')
    original = dict(strata=old['binary'], **old['fixtures'])
    for item in original.values():
        guards.require(digest(item['path']) == item['sha256'], 'current build artifact changed')
    target.mkdir(mode=0o700)
    copies = {}
    for name, item in original.items():
        path = target / name
        shutil.copy2(item['path'], path)
        guards.require(digest(path) == item['sha256'], 'fallback copy differs')
        copies[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=item['sha256'])
    receipt = target / 'source-build.json'
    shutil.copy2(full_path, receipt)
    guards.require(receipt.read_bytes() == full_path.read_bytes(), 'fallback source receipt differs')
    compact['retained_validation_reference'] = dict(source_commit=old['source_commit'],
        binary_origin_source_commit=old.get('binary_origin_source_commit', old['source_commit']),
        binary=copies.pop('strata'), fixtures=copies,
        source_build_record=dict(path=str(receipt), sha256=digest(receipt)), recipe_reference=recipe,
        original_source_blobs_recovered_and_verified=recovered, model_evidence_commits=evidence,
        model_evidence_kind='prefill-recovery' if recovery_build else ('census' if census_build else 'packed'),
        model_evidence_scope=scope,
        qualification=qualification,
        consumer=compact.get('retention_consumer') or 'Fallback for the imminent IQ2 pair inference build and same-build component/model qualification; select the validated generation and retire the extra at comparison closure.')
    if model_reference is not None:
        compact['retained_validation_reference']['model_evidence_reference'] = model_reference
    RESULT.write_text(json.dumps(compact, indent=2) + '\n')
    print(json.dumps(dict(retained_source=old['source_commit'], artifacts=len(original), recovered_inputs=recovered,
                         fallback=str(target), GPU_executed=False)))
    return 0


def main():
    guards.require(len(sys.argv) == 1 or sys.argv[1:] in
                   (['retain-current'], ['build-copy-safety'], ['build-status-profile'], ['rebuild-copy-safety'], ['rebuild-arena-copy'], ['rebuild-event-gate'], ['rebuild'], ['rebuild-attention'], ['rebuild-gemm'], ['rebuild-gdn'], ['rebuild-down'], ['rebuild-ordered'], ['rebuild-submission'], ['rebuild-gu'], ['rebuild-mixed'], ['rebuild-expert-bank'], ['rebuild-packed-bank'], ['rebuild-cpu-census'], ['rebuild-mtp-observer']),
                   'use build_engine.py retain-current or a named rebuild mode')
    rebuild = len(sys.argv) == 2 and sys.argv[1] not in ('build-copy-safety', 'build-status-profile')
    attention = sys.argv[1:] == ['rebuild-attention']
    targets = ['strata']
    event_gate=sys.argv[1:]==['rebuild-event-gate']
    observer = sys.argv[1:] == ['rebuild-mtp-observer']
    census = observer or sys.argv[1:] == ['rebuild-cpu-census']
    if census or sys.argv[1:] in (['build-copy-safety'], ['build-status-profile'], ['rebuild-copy-safety'], ['rebuild-arena-copy'], ['rebuild-event-gate'], ['rebuild-packed-bank']):
        targets += ['sycl_verifier_retirement_test', 'sycl_submission_test']
    if census or sys.argv[1:] in (['build-status-profile'], ['rebuild-arena-copy'],['rebuild-event-gate'], ['rebuild-packed-bank']):
        targets += ['file_expert_source_test', 'sycl_mmvq_test', 'residency_publication_host_test', 'cache_tail_host_test', 'prefill_scratch_host_test', 'sycl_prefill_scratch_test']
    if event_gate:
        targets += ['sycl_host_event_gate_test']
    if attention:
        targets += ['sycl_qsa_bf16_late_flags_test', 'sycl_qsa_bf16_prefix_optimized_test',
                    'sycl_qsa_bf16_prefix_qcache_test', 'sycl_qsa_bf16_prefix_kvec4_test',
                    'sycl_qsa_bf16_prefix_kvvec4_test',
                    'sycl_qsa_bf16_prefix_qcache_bench']
    if sys.argv[1:] == ['rebuild-gemm']:
        targets += ['sycl_native16_paired_gemm_bench']
    if sys.argv[1:] == ['rebuild-gdn']:
        targets += ['sycl_gdn_columns_bench']
    if sys.argv[1:] == ['rebuild-down']:
        targets += ['sycl_moe_down_scatter_bench', 'sycl_moe_gu_swiglu_bench',
                    'sycl_moe_accum_test', 'sycl_expert_bank_test', 'file_expert_source_test']
    if sys.argv[1:] == ['rebuild-ordered']:
        targets += ['sycl_qsa_bf16_prefix_kvvec4_test',
                    'sycl_qsa_bf16_prefix_ordered_exp_test',
                    'sycl_qsa_bf16_prefix_ordered_exp_bench']
    if sys.argv[1:] == ['rebuild-submission']:
        targets += ['sycl_submission_test']
    if sys.argv[1:] == ['rebuild-gu']:
        targets += ['sycl_moe_gu_swiglu_bench']
    if sys.argv[1:] in (['rebuild-mixed'], ['rebuild-expert-bank']):
        targets += ['sycl_moe_accum_test', 'sycl_moe_gu_swiglu_bench', 'sycl_moe_down_scatter_bench']
    if sys.argv[1:] == ['rebuild-expert-bank']:
        targets += ['sycl_expert_bank_test', 'file_expert_source_test']
    if census or sys.argv[1:] == ['rebuild-packed-bank']:
        targets += ['sycl_expert_bank_test']
    if census:
        targets += ['cpu_work_census_host_test']
    if observer:
        targets += ['sycl_graph_envelope_test', 'iq2_phase_host_test']
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1:] == ['retain-current']:
            guards.require(Path(__file__).read_bytes() == git(P, 'show', 'HEAD:tools/sycl/build_engine.py'),
                           'commit exact fallback controller first')
            return retain_current()
        device = guards.driver_guard()
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        guards.require(ROOT.exists() == rebuild, 'use rebuild only for the closed comparison cache')
        previous = None
        old = None
        retained_reference = None
        legacy_exception = None
        if not rebuild and RESULT.exists():
            committed=git(P,'show','HEAD:'+str(RESULT.relative_to(P)))
            guards.require(committed==RESULT.read_bytes(),'commit the retired build decision first')
            compact=json.loads(committed)
            retained_reference=fresh_retained_reference(compact)
            legacy_exception=compact.get('legacy_provenance_exception')
        if rebuild:
            committed = git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P)))
            guards.require(committed == RESULT.read_bytes(), 'commit the previous build identity first')
            old = json.loads((ROOT / 'record.json').read_text())
            retained_reference = json.loads(committed).get('retained_validation_reference')
            legacy_exception = json.loads(committed).get('legacy_provenance_exception')
            guards.require(retained_reference is not None,
                           'preserve and commit the qualified fallback before reusing this build directory')
            for item in [retained_reference['binary'], *retained_reference.get('fixtures', {}).values()]:
                guards.require(not Path(item['path']).resolve().is_relative_to(ROOT.resolve()),
                               'fallback would be overwritten by the rebuild')
                guards.require(digest(item['path']) == item['sha256'], 'retained known-good binary changed')
            guards.require(old['ownership_closed'] and not old['active'] and not old['GPU_executed'],
                           'previous build ownership not closed')
            # A failed compile is terminal, not a qualified binary. Preserve its
            # receipt and recover the separately pinned retained executable.
            old_binary = old.get('binary') if old['passed'] else (old.get('replaces') or {}).get('binary')
            retained_source = old['source_commit'] if old['passed'] else (
                old['replaces'].get('retained_binary_source_commit', old['replaces']['source_commit']))
            if not old['passed'] and retained_reference:
                old_binary = retained_reference['binary']
                retained_source = retained_reference['source_commit']
            guards.require(old_binary is not None, 'no pinned retained executable after failed build')
            guards.require(digest(old_binary['path']) == old_binary['sha256'], 'previous executable changed')
            for source in [old['project'], *old['dependencies']]:
                for name, data in blobs(Path(source['root']), source['revision'], source['source_sha256']):
                    guards.require(hashlib.sha256(data).hexdigest() == source['source_sha256'][name],
                                   'previous inputs not recoverable')
            recipe = old['recipe_reference']
            guards.require(hashlib.sha256(git(Path(old['project']['root']), 'show',
                old['project']['revision'] + ':' + recipe['path'])).hexdigest() == recipe['sha256'],
                'previous build recipe not recoverable')
            previous = dict(source_commit=old['source_commit'], binary=old_binary,
                            source_build_passed=old['passed'],
                            retained_binary_source_commit=retained_source,
                            reason='Closed unused comparison engine; original source/recipe recovered from Git. '
                                   'Reuse its cache for the immediate corrected build, not a retained old generation.')
        project = pin(P, ['CMakeLists.txt', 'cmake', 'include', 'src', 'tests',
                          'third_party', 'bench/sycl',
                          'tools/sycl/build_engine.py', 'tools/sycl/measurement_owner.py',
                          'tools/sycl/check_attention.py',
                          'tools/sycl/check_submission.py',
                          'tools/sycl/check_verifier_retirement.py',
                          'tools/sycl/check_event_gate.py',
                          'tools/sycl/check_host_profile.py',
                          'tools/sycl/check_mtp_observer.py',
                          'tools/sycl/ur_validation.py',
                          'tools/sycl/measure_gemm.py',
                          'tools/sycl/measure_gdn_columns.py',
                          'tools/sycl/measure_down_scatter.py',
                          'tools/sycl/measure_gu_swiglu.py',
                          'tools/sycl/check_mixed_scratch.py',
                          'tools/sycl/check_expert_bank.py',
                          'tools/sycl/measurement_guards.py', str(SETTINGS.relative_to(P)),
                          str(BASELINE.relative_to(P))])
        baseline = json.loads(BASELINE.read_text())
        argv = list(baseline['configure_argv'])
        argv[argv.index('-B') + 1] = str(ROOT / 'build')
        dependencies = []
        for prefix in ('-DSTRATA_GGML_DIR:PATH=', '-DSYCL_TLA_SOURCE_DIR='):
            root = Path(next(a[len(prefix):] for a in argv if a.startswith(prefix)))
            dependencies.append(pin(root, ['.']))
        settings = json.loads(SETTINGS.read_text())
        compiler = Path(next(a.split('=', 1)[1] for a in argv if a.startswith('-DCMAKE_CXX_COMPILER=')))
        guards.require(digest(compiler) == settings['compiler_sha256'], 'compiler changed')
        external_headers={}
        if event_gate:
            external_headers={str(path):digest(path) for path in (
                Path('/usr/include/level_zero/ze_api.h'),
                compiler.parent.parent/'include/sycl/ext/oneapi/backend/level_zero.hpp',
                compiler.parent.parent/'include/sycl/ext/oneapi/backend/level_zero_ownership.hpp')}
        if observer:
            headers = compiler.parent.parent/'include/sycl/ext/oneapi/experimental'
            external_headers = {str(path):digest(path) for path in
                [headers/'graph.hpp', *sorted((headers/'graph').glob('*.hpp'))]}
            guards.require(len(external_headers) > 1, 'installed graph headers required')
        ROOT.mkdir(mode=0o700, exist_ok=rebuild)
        for name in ('tmp', 'cache'):
            (ROOT / name).mkdir(exist_ok=rebuild)
        env = dict(baseline['environment'])
        env.update(XDG_CACHE_HOME=str(ROOT / 'cache'), TMPDIR=str(ROOT / 'tmp'))
        # Configuration/build never executes the model. The future first model
        # invocation must apply the qualified tracing/validation environment.
        record = dict(source_commit=project['revision'], project=project, dependencies=dependencies,
                      compiler=dict(path=str(compiler), sha256=digest(compiler)),
                      configure_argv=argv, environment=env, device=device,
                      recipe_reference=dict(path=str(BASELINE.relative_to(P)), sha256=digest(BASELINE)),
                      complete=False, passed=False, active=False, GPU_executed=False,
                      production_selected=False, model_quality_qualified=False, replaces=previous,
                      targets=targets,
                      retention_consumer='Current comparison engine and requested fixtures: ' + ', '.join(targets) +
                          '; select the validated generation and retire the extra after qualification/comparison closure.')
        if external_headers:
            record['fixture_external_headers']=external_headers
            record['fixture_external_header_origin']=dict(level_zero='libze-dev installed v1.17-r1.17.24 header',
                sycl='Installed original Intel LLVM compiler headers; no XeStrata application source',
                compiler_version=subprocess.check_output([str(compiler),'--version'],text=True,timeout=10).strip(),
                packages=subprocess.check_output(['dpkg-query','-W','-f=${Package} ${Version}\n','libze-dev','libze1'],text=True).splitlines())
            if observer:
                record['fixture_external_header_origin'].pop('level_zero')
        if retained_reference:
            record['retained_validation_reference'] = retained_reference
        if legacy_exception:
            record['legacy_provenance_exception'] = legacy_exception
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            if owner:
                record['commands'] = owner.commands
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = supervision.Owner(ROOT / ('owned-' + project['revision'][:12]), persist)
        persist()
        try:
            commands = [('configure', argv)]
            if census:
                commands += [('compile-host-contract', ['/usr/bin/cmake', '--build', str(ROOT/'build'),
                    '--target', 'cpu_work_census_host_test', *(['iq2_phase_host_test'] if observer else []), '-j', '4']),
                    ('host-contract', [str(ROOT/'build/cpu_work_census_host_test')])]
            if observer:
                commands += [('phase-host-'+str(arm), [str(ROOT/'build/iq2_phase_host_test'),'--pair='+str(arm)])
                             for arm in (0,1)]
            commands.append(('compile-link', ['/usr/bin/cmake', '--build', str(ROOT / 'build'),
                '--target', *targets, '-j', '4']))
            for label, command in commands:
                host_contract = label == 'host-contract'
                phase_contract = label.startswith('phase-host-')
                child_env = dict(env, STRATA_IQ2_PAIR=label[-1], STRATA_IQ2_PAIR_AUDIT='1', STRATA_IQ_SINGLE_EXACT='0') if phase_contract else env
                entry, out, err = owner.run(label, command, child_env, P,
                    wall=30 if host_contract or phase_contract else 1800, cpu=60 if host_contract or phase_contract else 7200,
                    rss_cap=12 << 30, file_cap=256 << 20, text_cap=64 << 20, total_cap=4 << 30)
                guards.require(supervision.closed(entry) and entry.get('exit_code') == 0,
                               label + ' failed: ' + (err.read_text(errors='replace') +
                                                      out.read_text(errors='replace'))[-8192:])
                if host_contract:
                    lines = out.read_text().splitlines()
                    guards.require(len(lines) == 16 and
                        all(x.startswith('cpu_work_census case=') and x.endswith(' status=PASS') for x in lines[:-1]) and
                        lines[-1] == 'cpu_work_census_fixture cases=15 status=PASS', 'incomplete CPU census host contract')
                    record['cpu_census_host_contract'] = dict(cases=15, passed=True,
                        exit_code=0, ownership_closed=True, fixture_sha256=digest(command[0]),
                        stdout_sha256=digest(out), stderr_sha256=digest(err), GPU_executed=False)
                if phase_contract:
                    guards.require(out.read_text().strip() == 'iq2_phase_scope cases=12 pair='+label[-1]+
                        ' prefill_pair_rows=0 decode_dispatch=PASS NT2_reference=PASS bits=PASS guards=PASS scope_restore=PASS',
                        'incomplete per-phase CPU dispatch contract')
                    record.setdefault('cpu_phase_host_contract', []).append(dict(arm=int(label[-1]),cases=12,
                        passed=True,ownership_closed=True,GPU_executed=False,fixture_sha256=digest(command[0]),
                        stdout_sha256=digest(out),stderr_sha256=digest(err)))
            # Detect input changes made while compilation was in progress.
            for source in [project, *dependencies]:
                for name, expected in source['source_sha256'].items():
                    guards.require(digest(Path(source['root']) / name) == expected, 'build input changed: ' + name)
            for name,expected in external_headers.items():
                guards.require(digest(name)==expected,'external fixture header changed: '+name)
            database = ROOT / 'build/compile_commands.json'
            entries = json.loads(database.read_text())
            roots = [P, ROOT, *(Path(d['root']) for d in dependencies)]
            guards.require(all(any(Path(e['file']).resolve().is_relative_to(root) for root in roots)
                               for e in entries), 'unqualified configured source root')
            for name in ('src/prefill/prefill.cpp', 'src/prefill/sycl/gdn_chunk64.cpp'):
                guards.require(any(Path(e['file']).resolve() == P / name for e in entries), 'missing configured source: ' + name)
            binary = ROOT / 'build/strata'
            guards.require(binary.is_file(), 'no engine executable')
            guards.driver_guard()
            guards.require(boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'boot changed')
            record.update(binary=dict(path=str(binary), bytes=binary.stat().st_size, sha256=digest(binary)),
                          compile_database_sha256=digest(database), configured_compile_entries=len(entries),
                          complete=True, passed=True)
            record['fixtures'] = {
                name: dict(path=str(ROOT / 'build' / name), sha256=digest(ROOT / 'build' / name))
                for name in targets if name != 'strata'}
            record['binary_origin_source_commit'] = binary_origin_source(
                old, record['binary']['sha256'], project['revision'])
            if old and old['passed']:
                old_pins = old['project']['source_sha256']
                new_pins = project['source_sha256']
                changed = sorted(name for name in set(old_pins) | set(new_pins)
                                 if old_pins.get(name) != new_pins.get(name))
                native_match = (all(name.startswith('tools/sycl/') and name.endswith('.py')
                                    for name in changed) and
                    len(dependencies) == len(old['dependencies']) and all(
                        a['root'] == b['root'] and a['source_sha256'] == b['source_sha256']
                        for a, b in zip(dependencies, old['dependencies'])) and
                    all(record[k] == old[k] for k in
                        ('compiler', 'configure_argv', 'environment', 'recipe_reference',
                         'compile_database_sha256', 'targets')))
                binary_match = record['binary']['sha256'] == old['binary']['sha256']
                fixture_matches = {name: info['sha256'] == old.get('fixtures', {}).get(name, {}).get('sha256')
                                   for name, info in record['fixtures'].items()}
                record['previous_output_comparison'] = dict(changed_project_inputs=changed,
                    native_and_recipe_inputs_match=native_match, engine_bytes_match=binary_match,
                    fixture_bytes_match=fixture_matches,
                    controller_only_admission=bool(native_match and binary_match and all(fixture_matches.values())),
                    interpretation='A real configure/build invocation admits the recorded inputs after original source/recipe recovery. Identical engine bytes preserve their known source identity even after fixture edits; qualification reuse separately requires the relevant native/recipe/fixture proof. No new speed result is implied.')
        except BaseException as error:
            record['error'] = str(error)
            record['passed'] = False
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            if not record['ownership_closed']:
                record['passed'] = False
            persist()
            compact = {k: v for k, v in record.items() if k not in ('project', 'dependencies', 'commands')}
            compact['inputs'] = [{k: v for k, v in source.items() if k != 'source_sha256'}
                                 for source in [project, *dependencies]]
            RESULT.write_text(json.dumps(compact, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('source_commit', 'complete', 'passed', 'ownership_closed', 'error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
