"""Qualify direct BF16 attention output in the admitted DFlash2 one-window layout."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_window_admission as window
import dflash2_prefill_window_speed as timing

bank, basis, require = window.bank, window.basis, window.require
KEY = 'dflash2_prefill_window_direct_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
REFERENCE = '4b25af35d4ff9f7f080b0ac562d0f39347a2656c'
ARM, CONTROL = 'direct', 'window16'
COPY_BYTES = 116769423360


def inputs(revision):
    trace = basis.validation.committed(bank.RESULT, (window.KEY,))
    quiet = basis.validation.committed(bank.RESULT, (timing.KEY,))
    control = trace['outcomes'][CONTROL]
    require(trace == bank.previous.historic(REFERENCE, window.KEY)
            and trace['stage'] == 'closed' and trace['complete'] and trace['passed']
            and trace['ownership_closed'] and not trace['active'] and control['passed']
            and control['native_exit_code'] == 0 and not control['fresh_faults']
            and control['route']['windows'] == 48 and control['route']['logical_copy_bytes'] == COPY_BYTES,
            'recover the original traced mode16 control')
    require(quiet['stage'] == 'closed' and quiet['passed'] and quiet['complete']
            and quiet['ownership_closed'] and not quiet['active']
            and quiet['decision']['repeated_warm_gain']
            and all(p['passed'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in quiet['processes']),
            'require the completed three-pair window timing screen')
    source, envs, requests, reference = window.inputs(revision)
    identities = ('binary', 'native_source_commit', 'native_build_sha256', 'boot', 'driver', 'runtime_manifest')
    require(all(source[k] == trace['plan'][k] == quiet[k] for k in identities)
            and source['validation'] == trace['plan']['validation'] == quiet['native_validation_probe']
            and source['drafter_assets'] == trace['plan']['drafter_assets'] == quiet['dflash2_assets']
            and source['cases'][CONTROL] == trace['plan']['cases'][CONTROL],
            'same-build copy control or environment changed')
    env = {k: v.replace(str(window.ROOT / CONTROL), str(ROOT / ARM)) for k, v in envs[CONTROL].items()}
    writer = window.store.SELECTOR
    require(env[writer] == '0' and env[window.MODE] == '16' and env[window.UPLOAD] == 'demand',
            'require the admitted demand-upload mode16 layout')
    env[writer] = '1'
    path = bank.P / THIS
    require(path.read_bytes() == basis.validation.git('show', revision + ':' + THIS),
            'commit the direct-writer admission controller')
    plan = dict(source, cases={ARM: dict(source['cases'][CONTROL], environment_sha256=bank.signature(env))},
                order=[ARM], copy_reference=REFERENCE, copy_reference_sha256=bank.signature(trace),
                quiet_window_sha256=bank.signature(quiet), controller_sha256=bank.sha(path),
                scope='Use one fresh traced process for the BF16 direct-writer candidate. Compare with the '
                      'sealed demand-upload mode16 copy process at identical 82 / 118785 / 82 inputs. '
                      'Require 48 windows, both checked drains, actual direct stores, matching expert work, '
                      'initial short state, restored-cache evidence, full tap coverage, active state, KV, '
                      'draft counters and output. Collect quiet timing separately.')
    return plan, env, requests, reference, control


def save(row):
    window.store.gate.save(row, ROOT, KEY)


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT),
            'fresh admission plan required')
    revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
    plan, _, _, _, _ = inputs(revision)
    root = ROOT / ARM
    for directory in ('cache', 'tmp'):
        (root / directory).mkdir(mode=0o700, parents=True)
    save(dict(plan=plan, stage='prepared', passed=False, complete=False, active=False, ownership_closed=True))


def run():
    row = bank.previous.read(ROOT / 'record.json')
    root = ROOT / ARM
    require(row == basis.validation.committed(bank.RESULT, (KEY,)) and row['stage'] == 'prepared'
            and row['ownership_closed'] and not row['active'] and 'outcome' not in row
            and not (root / 'execution.json').exists() and not (root / 'owned').exists(),
            'require the committed plan and an unused native process')
    plan, env, requests, reference, control = inputs(row['plan']['source_commit'])
    require(plan == row['plan'], 'frozen writer inputs changed')

    def inspect(log, pid, answers):
        result = window.inspect(log, plan, pid, CONTROL, answers, direct=True)
        window.store.speed.verify_model_assets(reference, {'args': plan['cases'][ARM]['command'][2:]})
        return result

    tested = window.device.run_case(plan, root, ARM, env, requests, inspect)
    if tested['passed']:
        work = ('requested_bank_H2D_bytes', 'uploads', 'jobs', 'dequants', 'windows', 'maximum_rows')
        short_seed = tested['info'] == control['info'] and all(
            tested[k][0] == control[k][0] for k in ('requests', 'states'))
        matched_state = all(tested[k] == control[k] for k in ('requests', 'states', 'kv_stage', 'loan'))
        matched_taps = tested['tap_rows'] == control['tap_rows']
        matched_work = all(tested['route'][k] == control['route'][k] for k in work)
        independent = tested['native_identity'] != control['native_identity']
        removed = control['route']['logical_copy_bytes'] - tested['route']['logical_copy_bytes']
        row['comparison'] = dict(matched_initial_short_state=short_seed,
                                 matched_state_and_generated_work=matched_state, matched_tap_coverage=matched_taps,
                                 matched_expert_work=matched_work, independent_processes=independent,
                                 requested_D2D_bytes_removed=removed)
        if not (short_seed and matched_state and matched_taps and matched_work
                and independent and removed == COPY_BYTES):
            tested.update(passed=False,
                          error='writer state, tap coverage, expert work, ownership or requested bytes differ')
    passed = bool(tested['passed'])
    row.update(outcome=tested, stage='closed' if passed else 'failed', passed=passed, complete=passed,
               active=not tested['ownership_closed'], ownership_closed=tested['ownership_closed'])
    save(row)
    print({k: tested.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(passed, tested.get('error', 'direct writer did not qualify'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'run'),
            'use the pinned interpreter with prepare or run')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        {'prepare': prepare, 'run': run}[action]()
