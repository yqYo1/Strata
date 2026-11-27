"""Check model placement with the qualified fixtures and loader diagnostics."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_phase_placement_check as original

bank, validation, require = original.bank, original.validation, original.require
FIXTURES = 'dflash2_decode_phase_placement_check_v2'
FIXTURE_ARMS = ('host', 'arc')
MODEL_ARMS = ('stock', 'blend')
KEY = 'dflash2_decode_phase_placement_check_v3'
ROOT = bank.B / KEY.replace('_', '-')
THIS = Path(__file__).resolve().relative_to(bank.P).as_posix()
first_inputs = original.inputs


def previous():
    row = validation.committed(bank.RESULT, (FIXTURES,))
    outcomes = row['outcomes']
    stock = outcomes['stock']
    require(row['stage'] == 'failed' and row['ownership_closed'] and not row['active']
            and tuple(outcomes) == (*FIXTURE_ARMS, 'stock')
            and all(outcomes[arm]['passed'] and outcomes[arm]['ownership_closed']
                    and outcomes[arm]['native_exit_code'] == 0 and not outcomes[arm]['fresh_faults']
                    for arm in FIXTURE_ARMS)
            and outcomes['arc']['providers']['initialized_ELFs'] > 0
            and stock['native_exit_code'] == 0 and stock['ownership_closed'] and not stock['fresh_faults']
            and stock['error'] == "RuntimeError('unexpected initialized ELF/provider')",
            'require the qualified fixtures and preserve the failed model launch')
    return row


def inputs(revision):
    prior = previous()
    plan, base, environments = first_inputs(revision)
    require(all(plan[k] == prior[k] for k in ('build_sha256', 'native_source_commit', 'requests', 'profiles', 'boot', 'driver'))
            and all(plan['cases'][name]['command'] == prior['cases'][name]['command']
                    and plan['cases'][name]['binary'] == prior['cases'][name]['binary'] for name in original.ORDER),
            'require the same qualified fixtures and identical native model commands')
    file = bank.P / THIS
    require(file.read_bytes() == validation.git('show', revision + ':' + THIS), 'commit the corrected model controller')
    plan['sources'][THIS] = bank.sha(file)
    for arm in FIXTURE_ARMS:
        plan['cases'][arm] = prior['cases'][arm]
    for arm in MODEL_ARMS:
        require('LD_DEBUG' not in environments[arm], 'unexpected existing loader observer')
        environments[arm]['LD_DEBUG'] = 'libs'
        plan['cases'][arm]['environment_sha256'] = bank.signature(environments[arm])
    plan.update(reused_fixture_proof=bank.signature(prior), model_loader_observer='libs')
    return plan, base, environments


def main(action):
    original.KEY, original.ROOT, original.inputs = KEY, ROOT, inputs
    require(action in ('prepare', *MODEL_ARMS), 'choose prepare, stock, or blend')
    if action != 'prepare':
        original.main(action)
        return
    require(sys.executable == bank.basis.reference.model.PYTHON, 'use the pinned Python')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original.builder.guards.no_model_owners([])
        require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'review the existing model check first')
        plan, _, _ = inputs(validation.git('rev-parse', 'HEAD').decode().strip())
        for arm in MODEL_ARMS:
            for child in ('tmp', 'cache'):
                (ROOT / arm / child).mkdir(mode=0o700, parents=True)
        outcomes = {arm: previous()['outcomes'][arm] for arm in FIXTURE_ARMS}
        original.save(dict(plan, stage='running', outcomes=outcomes, complete=False, passed=False,
                           ownership_closed=True, active=False))


if __name__ == '__main__':
    main(*sys.argv[1:])
