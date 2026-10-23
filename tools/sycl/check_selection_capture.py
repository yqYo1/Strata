#!/usr/bin/env python3
"""Main-only bounded checked-readback qualification, never a model/speed test.

Repository LICENSE applies. Freeze and commit prepare before run. Stop at the
first failed case; no retries or settings changes. Keeps the selected engine.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import build_selection_capture as candidate
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation
from measure_down_scatter import postchecks

P, B, A = gen.P, gen.B, gen.A
ROOT = B / 'qsa-selection-qual-v1'
RESULT = A / 'qsa-selection-qualification-current.json'
SCOPE = 'checked_selection_native_contract_20261023'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def runtime_lineage():
    """Bind the prior component, failed model record and separate analysis.

    The model's original parser failure and preload teardown limitation remain
    unchanged. This is runtime lineage, not qualification of this new binary.
    """
    reference = validation.committed(A / 'qsa-selection-current.json', ('runtime_review_followup',))
    path = P / reference['component_reference']['path']
    guards.require(validation.digest(path) == reference['component_reference']['sha256'], 'runtime component changed')
    component = gen.committed(path)
    profile = gen.committed(A / 'cpu-profiler-current.json')
    original, analysis = profile['native_model_aggregate_v2'], profile['native_model_aggregate_v2_reanalysis']
    canonical = json.dumps(analysis, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    guards.require(hashlib.sha256(canonical).hexdigest() == reference['separate_closed_analysis']['canonical_sha256'],
                   'closed runtime analysis changed')
    old = reference['model_profile_original_failure']
    guards.require(hashlib.sha256(gen.git('show', old['commit'] + ':' + old['path'])).hexdigest() == old['sha256'],
                   'original failed model receipt not recoverable')
    guards.require(component['passed'] and component['complete'] and component['ownership_closed'] and
        not component['active'] and component['fresh_faults'] == [] and
        not original['passed'] and original['ownership_closed'] and not original['active'] and
        original['fresh_faults'] == [] and original['boot_unchanged'] and
        analysis['passed'] and analysis['complete'] and analysis['original_failure_preserved'] and
        analysis['evidence']['native_exit_code'] == 0, 'runtime/normal native-exit evidence incomplete')
    probe = validation.preload_qualification()
    guards.require(original['native_validation_probe'] == probe and
        component['runtime_asset_delta'] == original['runtime_asset_delta'], 'runtime lineage differs')
    settings = gen.committed(gen.SETTINGS)
    delta = component['runtime_asset_delta']
    guards.require(set(delta) <= settings['runtime_elfs'].keys(), 'unknown runtime asset')
    pins = {}
    for name, item in settings['runtime_elfs'].items():
        changed = delta.get(name)
        guards.require(not changed or changed['qualified_sha256'] == item['sha256'], 'runtime baseline differs')
        pins[name] = changed['candidate_sha256'] if changed else item['sha256']
        guards.require(validation.digest(name) == pins[name], 'runtime bytes changed: ' + name)
    return dict(reference=reference, native_validation_probe=probe, runtime_pins=pins)


def admission():
    built = gen.committed(candidate.RESULT)
    guards.require(built['stage'] == 'closed' and built['passed'] and built['complete'] and
        built['ownership_closed'] and not built['active'] and not built['GPU_executed'], 'closed diagnostic build required')
    guards.require(validation.digest(candidate.ROOT / 'plan.json') == built['plan_sha256'], 'build plan changed')
    plan = json.loads((candidate.ROOT / 'plan.json').read_text()); candidate.current(plan)
    for item in built['artifacts'].values():
        guards.require(validation.digest(item['path']) == item['sha256'], 'diagnostic artifact changed')
    review = validation.committed(validation.INDEX, (SCOPE, 'review'))
    guards.require(review.get('source_gate_passed') is True and review.get('status') == 'complete',
                   'independent native Luna-max source review must pass before GPU execution')
    return built, review, runtime_lineage()


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(), 'fresh bounded qualification only')
    built, review, runtime = admission()
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    sources = gen.benchmark_pins(revision)
    env = dict(json.loads((candidate.ROOT / 'plan.json').read_text())['environment'])
    probe = runtime['native_validation_probe']
    env.update(LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    ROOT.mkdir(mode=0o700)
    for name in ('cache', 'tmp'): (ROOT / name).mkdir()
    record = dict(stage='prepared', source_commit=revision, source_fingerprint=candidate.fingerprint(sources),
        build_source_commit=built['source_commit'], build_sha256=validation.digest(candidate.RESULT),
        fixture=built['artifacts']['readback'], review=review, runtime_lineage=runtime, environment=env,
        cases_order=['small', 'full'], cases=[],
        prospective_gates='Two fresh processes,small17x33 with full UR trace then8192x2051 with parameter validation. '
            'Exact payload/header and source guards,no invalid-geometry submission,checked normal teardown; '
            'same-PID pre-main validation,unchanged inputs/runtime/selected fallback,no fresh fault. '
            'First failure stops without retry. No model,physical-context,decode or speed qualification.',
        budgets=dict(wall=90, cpu=180, rss_cap=2 << 30, text_cap=32 << 20, file_cap=96 << 20, total_cap=256 << 20),
        retention='Retire both synthetic payloads and successful API traces after recorded exact checks/hash/closure. '
            'Keep compact case evidence for one real-selection capture admission,then review diagnostic build retirement.',
        passed=False, complete=False, active=False, ownership_closed=True, GPU_executed=False,
        clean_speed_qualified=False, model_adoption_qualified=False)
    write(ROOT / 'plan.json', record); write(RESULT, record)
    print('Prepared bounded checked-readback qualification; commit exact plan before run.', flush=True)


def current(record):
    built, review, runtime = admission()
    guards.require(validation.digest(candidate.RESULT) == record['build_sha256'] and
        built['artifacts']['readback'] == record['fixture'] and review == record['review'] and
        runtime == record['runtime_lineage'] and candidate.fingerprint(gen.benchmark_pins(record['source_commit'])) ==
        record['source_fingerprint'], 'qualification inputs changed')


def run():
    record = gen.committed(RESULT)
    guards.require(record['stage'] == 'prepared' and record == json.loads((ROOT / 'plan.json').read_text()) and
        not (ROOT / 'record.json').exists(), 'single-use committed plan required')
    current(record)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(line.split(': ', 1)[1] for line in text.splitlines() if line.startswith('-- cursor: '))
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(record, commands=owner.commands if owner else []))
    owner = supervision.Owner(ROOT / 'owned', persist); persist()
    try:
        for mode in record['cases_order']:
            current(record)
            env = dict(record['environment'], UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr')
            if mode == 'small':
                env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                           UR_LOG_TRACING='level:info;flush:info;output:stderr')
            else:
                env.pop('UR_LOG_TRACING', None)
            output = ROOT / (mode + '.bin')
            record['native_launch_attempted'] = True
            if not record['cases']:
                record['GPU_executed'] = None  # an early process failure may precede GPU work
            entry, out, err = owner.run(mode, [record['fixture']['path'], mode, str(output)], env, P, **record['budgets'])
            guards.require(supervision.closed(entry) and entry['exit_code'] == 0, 'readback fixture failed: ' + mode)
            stderr = err.read_text(errors='replace')
            activation = validation.preload_evidence(stderr, entry['direct_identity']['pid'])
            body = stderr.split('phase=before-main', 1)[1].split('\n', 1)[1]
            guards.require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|Strata SYCL cleanup:', body),
                           'native runtime error after validation canary')
            if mode == 'small':
                guards.require('urEnqueueUSMMemcpy' in body and 'urEnqueueKernelLaunch' in body, 'missing actual GPU trace')
            rows, cap = (17, 33) if mode == 'small' else (8192, 2051)
            expected = f'selection_dump mode={mode} rows={rows} cap={cap} values={rows*cap} exact=PASS guards=PASS invalid_geometry=PASS'
            guards.require(out.read_text().strip() == expected and output.stat().st_size == 16 + rows * cap * 4,
                           'exact readback/guard/header evidence missing')
            record['cases'].append(dict(mode=mode, native_identity=entry['direct_identity'], exit_code=0,
                payload_bytes=output.stat().st_size, payload_sha256=validation.digest(output),
                activation=activation, exact_payload_and_guards=True, invalid_geometry_no_submit=True))
            record['GPU_executed'] = True
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            record['passed'] = True; postchecks(record, boot, cursor); persist()
            guards.require(record['passed'], 'fault/ownership observation failed; stop before next case')
        current(record); record.update(passed=True, complete=True)
    except BaseException as error:
        record.update(passed=False, error=repr(error))
    finally:
        record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        postchecks(record, boot, cursor); record['stage'] = 'closed'; persist(); write(RESULT, record)
    print(json.dumps({k: record.get(k) for k in ('passed', 'complete', 'ownership_closed', 'fresh_faults', 'error')}), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    guards.require(sys.argv[1:] in (['prepare'], ['run']), 'use prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([]); guards.driver_guard()
        sys.exit(prepare() if sys.argv[1] == 'prepare' else run())
