#!/usr/bin/env python3
"""One checked117K resource observation on the unchanged borrowed-cache engine.

This is attribution, not a clean speed comparison or a saturation measurement.
Source and the generated plan must be committed separately before execution.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
import cache_long_compare as c

gen, assets, guards, supervision = c.gen, c.assets, c.guards, c.supervision
P, B, A = c.P, c.B, c.A
ROOT = B / 'prefill-resources-117k-v1'
RESULT = A / 'prefill-resources-current.json'
READER = c.resources.READER


def source_pins(revision):
    pins = gen.benchmark_pins(revision)
    name = str(READER.relative_to(P))
    raw = gen.git('show', revision + ':' + name)
    guards.require(raw == READER.read_bytes(), 'commit exact resource reader first')
    pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def prerequisite():
    old, current, proof, original = c.qualification()
    c.reviewed()
    comparison = gen.committed(c.RESULT)
    guards.require(comparison.get('comparison_closed') is True and
        c.summarize(comparison['processes']) == comparison['comparison_summary'],
        'close and verify the frozen six-process comparison first')
    return current, proof, original, c.validation_probe(old)


def require_observer_environment(env):
    guards.require('LD_PRELOAD' not in env and
        env.get('STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS', '0') == '0',
        'private native preload and planned-window census required')


def long_interval(log, sample):
    """Exclude the short warmup's intentionally incomplete NativeDense frame."""
    footers = list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*', log, re.M))
    guards.require(len(footers) == 2 and 0 < int(footers[0][1]) < 32768 and
        int(footers[0][2]) == 0 and int(footers[0][1]) == int(footers[0][3]) and
        1 <= int(footers[0][4]) <= 24 and tuple(map(int, footers[1].groups())) ==
        (117489, 0, 117489, sample['generated_tokens']), 'native request extent differs')
    return log[footers[0].end():footers[1].start()]


def restore_intervals(log, lent):
    frames = list(re.finditer(r'^strata native dense host profile: (.+)$', log, re.M))
    refills = list(re.finditer(r'^strata trace: refilled (\d+) slots on (\d+) stage\(s\) in ([\d.eE+\-]+) ms$', log, re.M))
    guards.require(len(frames) == len(refills) == 1 and frames[0].end() < refills[0].start(),
        'missing, duplicate or reordered dense/cache restoration')
    tokens = frames[0][1].split()
    guards.require(all(t.count('=') == 1 for t in tokens), 'malformed dense profile')
    fields = dict(t.split('=', 1) for t in tokens)
    guards.require(len(fields) == len(tokens) and all(fields.get(k) == v for k, v in
        dict(v='2', complete='1', outcome='success', entry_done='1', run_returned='1',
             restored='1', intervals='NONEXCLUSIVE', outer_startup_snapshot_unmeasured='1').items()),
        'incomplete or incompatible long-request dense profile')
    times = {k: float(fields[k]) for k in ('entry_ms', 'run_ms', 'restore_ms')}
    refill = float(refills[0][3])
    guards.require(all(math.isfinite(v) and v >= 0 for v in [*times.values(), refill]) and
        int(refills[0][1]) == lent and int(refills[0][2]) == 1, 'invalid restore extent/time')
    return dict(native_dense=times, expert_cache_refill_wall_ms=refill, refilled_slots=lent,
        intervals_nonexclusive=True, expert_refill_outside_native_dense=True,
        limits='Dense entry/run/restore are nonexclusive wrapper clocks. Expert refill includes RAM source lookup, enqueue, completion and residency publication, not pure PCIe service. Do not add these to nested queue intervals.')


def analyze(log, sample, loan):
    part = long_interval(log, sample)
    c.cache_placement(log, 1, loan)
    c.admission.cap.matches(r'strata prefill: layer-major candidate, tokens=117488 chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=2406154240 residual_gpu_rows=117488 residual_gpu=2406154240 residual_host=0 residual_staging=0 layer_cache=1022361600; extra allocations, GPU headroom=536870912, RAM headroom=8589934592', part, 1)
    c.admission.cap.matches(r'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=81920 mixed_fp16_bytes=0', part, 1)
    guards.require('effective_mode=2' not in log and 'strata loan audit:' not in log,
        'fallback or payload audit changes diagnostic work')
    owners = borrowed_owners(part, loan)
    profile = gen.common.extract_resource_profile(part, True, False, '', 4, c.admission.WINDOWS)
    result = gen.common.compact_host_resource_profile(profile)
    result.update(restore_intervals(part, loan['lent']), borrowed_owner_evidence=owners)
    return result


def borrowed_owners(part, loan):
    a = c.admission
    masked, ready = a.records(part, 'strata cache loan', 2)
    phase = masked[1]
    generation = a.integer(phase, 'generation')
    for (_, fields), expected, state in ((masked, 'masked', '0'), (ready, 'ready', '1')):
        guards.require(fields['phase'] == expected and fields['decode_ready'] == state and
            a.integer(fields, 'generation') == generation > 0 and
            all(a.integer(fields, key) == loan[key] for key in ('chunk', 'first', 'lent')),
            'borrowed lease publication differs')
    dense = a.records(part, 'strata prefill dense', 3)
    guards.require([(r['phase'], r['residency'], a.integer(r, 'activations')) for _, r in dense] ==
        [('retired', 'Full', 0), ('bank', 'Bank', 0), ('restored', 'Full', 48)], 'dense owner restoration differs')
    binding, = a.records(part, 'strata prefill cache binding', 1)
    placement, = a.cap.matches(r'strata prefill: layer-major candidate, tokens=117488 [^\n]+', part, 1)
    admitted, = a.records(part, 'strata MoE accumulator admission', 1)
    guards.require(binding[1] == dict(generation=str(generation), parent_match='1', execution_distinct='1') and
        masked[0] < dense[0][0] < dense[1][0] < admitted[0] < binding[0] < placement.start() <
        dense[2][0] < ready[0], 'borrowed execution escaped checked parent lease')
    return a.borrowed_window_owners(part, dict(loan, generation=generation), admitted[1],
        placement.end(), dense[2][0])


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(), 'fresh diagnostic only; never retry')
    current, proof, original, probe = prerequisite()
    source = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = source_pins(source)
    loan = c.admitted_loan()
    env = {k: v.replace(original['root'], str(ROOT)) for k, v in original['environment'].items()
           if k != 'STRATA_TRACE' and k != 'UR_ENABLE_LAYERS' and not k.startswith('UR_LOG_')}
    settings = gen.committed(gen.SETTINGS)
    env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', STRATA_TRACE='1',
        CUDA_MODULE_LOADING='EAGER', STRATA_SYCL_CACHE_SCRATCH_LEASE='1',
        STRATA_SYCL_CACHE_LOAN_AUDIT='0', STRATA_PREFILL_LAYER_MAJOR_PROFILE='1',
        STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='1', STRATA_PREFILL_LAYER_MAJOR_GDN_PROFILE='1',
        STRATA_PREFILL_CPU_WAIT_PROFILE='1', STRATA_SYCL_PREFILL_COMMAND_PROFILE='0')
    env.update({k: v for k, v in settings['environment'].items()
                if k.startswith('UR_LOG_') and k != 'UR_LOG_TRACING'})
    for k in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys',
              'EnableDirectSubmission', 'STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[k] == settings['environment'][k], 'qualified safety setting changed')
    require_observer_environment(env)
    config = copy.deepcopy(original['config'])
    flags = config['args']
    guards.require(flags[flags.index('--expert-cache') + 1] == '1256' and
        '--no-prefill-borrow' not in flags and config['exe'] == str(P / 'tools/sycl/engine_once.py'),
        'unchanged admitted borrowed-cache route required')
    c.admission.require_batched_warmup(config)
    flags[flags.index('--expert-profile') + 1] = str(ROOT / 'expert-profile.bin')
    config.update(log=str(ROOT / 'engine.stderr'), env={})
    models = assets.model_pins(config, original['model_assets'], full=True)
    ROOT.mkdir(mode=0o700)
    for name in ('cache', 'tmp'): (ROOT / name).mkdir()
    shutil.copyfile(P / 'data/expert-profile.bin', ROOT / 'expert-profile.bin')
    guards.require(gen.common.digest(ROOT / 'expert-profile.bin') == original['profile_sha256'],
        'expert profile changed')
    (ROOT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    command = ['/bin/sh', str(P / 'bench.sh'), '--config', str(ROOT / 'config.json'), '--port', str(port),
        '--prefill-only', '--prefill-tokens', '98304', '--think', '--runs', '1',
        '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600',
        '--out', str(ROOT / 'bench')]
    record = dict(mode='engine-compare', diagnostic='prefill-resources-117k-v1', stage='diagnostic',
        source_commit=source, build_source_commit=current['source_commit'], binary=current['binary'],
        retained_build_proof_sha256=assets.fingerprint(proof),
        comparison_sha256=gen.common.digest(c.RESULT), capacity_result_sha256=gen.common.digest(c.admission.RESULT),
        admitted_long_loan=loan, native_validation_probe=probe, root=str(ROOT), environment=env,
        config=config, command=command, config_sha256=gen.common.digest(ROOT / 'config.json'),
        profile_sha256=original['profile_sha256'], settings_sha256=gen.common.digest(gen.SETTINGS),
        runtime_asset_delta=original['runtime_asset_delta'], tokenizer_assets=original['tokenizer_assets'],
        model_assets=models, machine_identity=assets.machine_identity(), source_count=len(pins),
        benchmark_fingerprint=assets.fingerprint(pins),
        resource_collection=dict(source=dict(path=str(READER), sha256=gen.common.digest(READER)), seconds=900, interval=.2),
        complete=False, passed=False, active=False, clean_speed_qualified=False,
        hardware_saturation_qualified=False, model_adoption_qualified=False,
        prospective_gates='One native process; one successful short warmup then one fresh117489-input/24-output thinking request through bench.sh. Require same-PID/pre-model validation activation, exact1256/first132/lent1124 placement,96 checked borrowed-owner retirements, all planned jobs reconciled to completed issuer totals, complete v3 coarse/QSA and v4 disjoint host/wait accounting, complete dense restoration and separate outer expert refill, bound client collector coverage, unchanged source/runtime/assets and normal owned exit with no fresh fault. Stop on first failed gate; no retries.',
        limits='Attribution only, not clean speed, observer overhead, sustained decode or model qualification. GPU queue intervals are nonexclusive; scheduled CPU includes polling. fdinfo reports engine cycles and memory residency, not physical PCIe/GDDR bytes. Physical bandwidth and matched BF16/FP16 GEMM ceilings remain unknown; no saturation claim.',
        retention='Keep compact work-conservation, aggregate host/queue/restore and client-cycle evidence. Raw96-window histograms/normal logs are temporary only through attribution validation and selection of a concrete matched-GEMM ceiling workload; retire at that boundary. Keep original outcomes and useful request sample; do not retain successful traces.',
        budgets=dict(wall=1800, cpu=54000, rss_cap=96 << 30, text_cap=64 << 20, file_cap=64 << 20, total_cap=64 << 20))
    gen.runtime_assets_unchanged(record)
    (ROOT / 'plan.json').write_text(json.dumps(record, indent=2) + '\n')
    RESULT.write_text(json.dumps(record, indent=2) + '\n')
    print('Prepared resource diagnostic; commit exact generated plan before run.')


def finish(record, owner, cursor, boot, persist):
    record.update(ownership_closed=owner.active is None and all(supervision.closed(x) for x in owner.commands),
        fresh_faults=None, fault_probe_successful=False, boot_unchanged=False)
    try:
        journal = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True, timeout=15)
        record['fresh_faults'] = [x for x in journal.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', x)]
        record['fault_probe_successful'] = True
    except Exception as error:
        record['kernel_journal_probe_error'] = str(error)
    try:
        record['boot_unchanged'] = boot == c.read_boot()
        record['driver_after'] = guards.driver_guard()
    except Exception as error:
        record['driver_or_boot_probe_error'] = str(error)
    record['passed'] = bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and
        record['fault_probe_successful'] and record['fresh_faults'] == [] and 'driver_or_boot_probe_error' not in record)
    try:
        persist()
    except BaseException as error:
        record.update(passed=False, persistence_error=str(error))
        raise
    finally:
        RESULT.write_text(json.dumps(record, indent=2) + '\n')


def run():
    record = gen.committed(RESULT)
    guards.require(record == json.loads((ROOT / 'plan.json').read_text()) and not record['complete'] and
        not (ROOT / 'record.json').exists(), 'fresh frozen plan required; never retry')
    current, proof, _, probe = prerequisite()
    guards.require(current['binary'] == record['binary'] and assets.fingerprint(proof) == record['retained_build_proof_sha256'] and
        probe == record['native_validation_probe'] and c.admitted_loan() == record['admitted_long_loan'] and
        gen.common.digest(c.RESULT) == record['comparison_sha256'] and
        gen.common.digest(c.admission.RESULT) == record['capacity_result_sha256'], 'native/admission evidence changed')
    pins = source_pins(record['source_commit'])
    guards.require(len(pins) == record['source_count'] and assets.fingerprint(pins) == record['benchmark_fingerprint'],
        'diagnostic source changed')
    for path, expected in [(gen.SETTINGS, record['settings_sha256']), (ROOT / 'config.json', record['config_sha256']),
        (ROOT / 'expert-profile.bin', record['profile_sha256']), *record['tokenizer_assets'].items()]:
        guards.require(gen.common.digest(path) == expected, 'frozen input changed: ' + str(path))
    gen.runtime_assets_unchanged(record)
    guards.require(assets.model_pins(record['config'], record['model_assets']) == record['model_assets'] and
        assets.machine_identity() == record['machine_identity'], 'model/platform changed')
    boot = c.read_boot()
    before = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(x.split(': ', 1)[1] for x in before.splitlines() if x.startswith('-- cursor: '))
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (ROOT / 'record.json').write_text(json.dumps(dict(record, owned_commands=owner.commands if owner else []), indent=2) + '\n')
    owner = supervision.Owner(ROOT / 'owned', persist)
    try:
        entry, _, _ = owner.run('bench', record['command'], record['environment'], P, **record['budgets'])
        record.update(complete=bool(supervision.closed(entry)), exit_code=entry.get('exit_code'))
        guards.require(record['complete'] and entry['exit_code'] == 0, 'official benchmark failed')
        path, = (ROOT / 'bench').glob('*/results.json')
        samples, journal = c.official(ROOT, json.loads(path.read_text()), 1)
        record.update(requests=samples, request_journal=journal)
        native = json.loads((ROOT / 'engine-exit.json').read_text())
        guards.require(native['exit_code'] == native['resource_collector_exit_code'] == 0 and
            native['boot_id'] == boot and native['binary'] == record['binary'] and
            native['args'] == ['--serve', *record['config']['args']] and
            native['native_validation_probe'] == probe, 'native/collector identity or closure differs')
        log = (ROOT / 'engine.stderr').read_text(errors='replace')
        assets.verify_pool(log, 5)
        guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', log), 'runtime failure')
        record['validation_activation'] = c.activation(log, native['native']['pid'], probe)
        record['attribution'] = analyze(log, samples[0], record['admitted_long_loan'])
        captures = [json.loads(line) for line in (ROOT / 'client-resources.jsonl').read_text().splitlines()]
        record['client_resources'] = c.resources.client_resources.summarize_capture(captures, native['native'], boot, samples[0]['client_interval'])
        _, after_proof, _, after_probe = prerequisite()
        guards.require(assets.fingerprint(after_proof) == record['retained_build_proof_sha256'] and after_probe == probe and
            source_pins(record['source_commit']) == pins and assets.machine_identity() == record['machine_identity'], 'inputs changed during diagnostic')
        gen.runtime_assets_unchanged(record)
        assets.model_pins(record['config'], record['model_assets'], full=True)
        record.update(passed=True, native_identity=native['native'])
    except BaseException as error:
        record['error'] = str(error)
        if (ROOT / 'engine.stderr').exists(): record['error_excerpt'] = (ROOT / 'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        finish(record, owner, cursor, boot, persist)
    print(json.dumps({k: record.get(k) for k in ('complete', 'passed', 'ownership_closed', 'error')}))
    return 0 if record['passed'] else 1


def analyze_closed():
    record = gen.committed(RESULT)
    guards.require(record['complete'] and record['ownership_closed'] and not record['active'] and
        record['passed'] is False and record['error'] == 'incomplete mixed owner/phase census' and
        record['fresh_faults'] == [] and record['fault_probe_successful'] and not record.get('closed_analysis'),
        'only the original closed parser failure may be reanalyzed; no native launch')
    plan = record['reanalysis_plan']
    guards.require(plan['native_execution_permitted'] is False and plan['original_outcome_unchanged'],
        'record the separate analysis contract first')
    for name, expected in plan['raw_sha256'].items():
        guards.require(gen.common.digest(ROOT / name) == expected, 'closed capture changed: ' + name)
    original = gen.benchmark_pins(record['source_commit'], match_worktree=False)
    name = str(READER.relative_to(P))
    original[name] = hashlib.sha256(gen.git('show', record['source_commit'] + ':' + name)).hexdigest()
    guards.require(assets.fingerprint(original) == record['benchmark_fingerprint'], 'original source cannot be recovered')
    changed = sorted(name for name, sha in original.items() if gen.common.digest(P / name) != sha)
    guards.require(set(changed) <= {'tools/sycl/cache_long.py', 'tools/sycl/prefill_resources.py',
        'tools/test_sycl_prefill_resources.py'}, 'unrelated measurement input changed')
    current, proof, _, probe = prerequisite()
    guards.require(current['binary'] == record['binary'] and assets.fingerprint(proof) == record['retained_build_proof_sha256'] and
        probe == record['native_validation_probe'], 'native qualification changed')
    check = record['post_closure_input_check']
    guards.require(check == json.loads((ROOT / 'post-closure-input-check.json').read_text()) and
        check['original_collection_passed'] is False and check['source_and_fixture_proof'] is True and
        check['benchmark_inputs_unchanged'] == len(original) and check['full_model_hashes_matched'] == len(record['model_assets']) and
        check['binary_sha256'] == record['binary']['sha256'] and check['source_commit'] == record['source_commit'],
        'exact post-closure/pre-correction full input proof required')
    gen.runtime_assets_unchanged(record)
    guards.require(assets.machine_identity() == record['machine_identity'] and
        assets.model_pins(record['config'], record['model_assets']) == record['model_assets'], 'platform/assets changed')
    official, = (ROOT / 'bench').glob('*/results.json')
    samples, _ = c.official(ROOT, json.loads(official.read_text()), 1)
    guards.require(samples == record['requests'], 'original request differs')
    native = json.loads((ROOT / 'engine-exit.json').read_text())
    guards.require(native['exit_code'] == native['resource_collector_exit_code'] == 0 and
        native['boot_id'] == c.read_boot() and native['binary'] == record['binary'], 'original native exit differs')
    log = (ROOT / 'engine.stderr').read_text(errors='replace')
    attribution = analyze(log, samples[0], record['admitted_long_loan'])
    captures = [json.loads(line) for line in (ROOT / 'client-resources.jsonl').read_text().splitlines()]
    clients = c.resources.client_resources.summarize_capture(captures, native['native'], native['boot_id'], samples[0]['client_interval'])
    record['closed_analysis'] = dict(complete=True, attribution_checks_passed=True,
        analyzer_source_commit=gen.git('rev-parse', 'HEAD').decode().strip(), changed_analysis_inputs=changed,
        original_source_inputs_recovered=len(original), native_execution=False,
        original_collection_passed=False, original_failure_preserved=True,
        attribution=attribution, client_resources=clients,
        validation_activation=c.activation(log, native['native']['pid'], probe),
        limits='Post-closure correction of an incompatible ownership parser. Original execution receipt remains failed. These data attribute this observed request only; no clean speed, observer overhead, physical bandwidth, arithmetic saturation or model adoption qualification.')
    RESULT.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(original_collection_passed=record['passed'],
        closed_analysis_passed=True, expert_cache_refill_ms=attribution['expert_cache_refill_wall_ms'],
        jobs=attribution['window_census']['jobs'], windows=attribution['window_census']['window_count'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'analyze-closed'))
    args = parser.parse_args()
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([]); guards.driver_guard()
        return run() if args.action == 'run' else analyze_closed() if args.action == 'analyze-closed' else prepare()


if __name__ == '__main__': sys.exit(main())
