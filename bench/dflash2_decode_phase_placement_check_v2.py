"""Repeat the placement safety checks with the Arc fixture's launch settings."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_phase_placement_check as original

bank, validation, require = original.bank, original.validation, original.require
FIRST_KEY = original.KEY
KEY = 'dflash2_decode_phase_placement_check_v2'
ROOT = bank.B / KEY.replace('_', '-')
THIS = Path(__file__).resolve().relative_to(bank.P).as_posix()
FIXTURE_LAUNCH = 'tools/sycl/check_verifier_retirement.py'
FIXTURE_ENVIRONMENT = dict(STRATA_SYCL_PREFILL_SCRATCH_ALIAS='1', STRATA_PREFILL_MMQ='0',
    STRATA_PREFILL_MOE_COMPACT_FP16='1', STRATA_SYCL_PREFILL_NATIVE16='bf16',
    STRATA_SYCL_PREFILL_NATIVE16_QSA='1', STRATA_SYCL_PREFILL_GDN_CHUNK64='0',
    STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm')
first_inputs = original.inputs


def inputs(revision):
    prior = validation.committed(bank.RESULT, (FIRST_KEY,))
    plan, base, environments = first_inputs(revision)
    require(prior['stage'] == 'failed' and prior['ownership_closed'] and not prior['active']
            and prior['outcomes']['host']['passed'] and prior['outcomes']['arc']['native_exit_code'] == 1
            and not prior['outcomes']['arc']['fresh_faults']
            and all(plan[k] == prior[k] for k in ('build_sha256', 'native_source_commit', 'requests', 'profiles'))
            and all(plan['cases'][name]['command'] == prior['cases'][name]['command']
                    and plan['cases'][name]['binary'] == prior['cases'][name]['binary'] for name in original.ORDER),
            'keep the failed launch and use its exact build, commands and model inputs')
    for name in (THIS, FIXTURE_LAUNCH):
        file = bank.P / name
        require(file.read_bytes() == validation.git('show', revision + ':' + name), 'commit exact fixture source: ' + name)
        plan['sources'][name] = bank.sha(file)
    environments['arc'].update(FIXTURE_ENVIRONMENT)
    plan['cases']['arc']['environment_sha256'] = bank.signature(environments['arc'])
    plan.update(original_failed_launch=bank.signature(prior), arc_fixture_environment=FIXTURE_ENVIRONMENT)
    return plan, base, environments


if __name__ == '__main__':
    original.KEY, original.ROOT, original.inputs = KEY, ROOT, inputs
    original.main(*sys.argv[1:])
