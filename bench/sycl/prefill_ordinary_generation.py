"""Check long, agent and sustained decode with the selected 65K prefill path."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_split_generation as split_suite

clock, suite, require = split_suite.clock, split_suite.suite, split_suite.require
speed = split_suite.curve.speed
bank = speed.bank
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'prefill_ordinary_generation_v1'
ROOT = bank.B / 'prefill-ordinary-generation-v1'
READ = 65537
TIMING = f'prefill_kv_gather_speed_{READ}_v1'
ARM = 'both'


def check_timing(native, prior, timing):
    require(timing['stage'] == 'closed' and timing['complete'] and timing['passed']
            and timing['ownership_closed'] and not timing['active']
            and timing['binary'] == native['binaries']['strata']
            and timing['native_source_commit'] == native['native_source_commit']
            and timing['observer_admission'] == prior['observer_admission'],
            'finish the balanced 65K comparison on this exact build and dispatch')
    decision = speed.decision(READ, timing['processes'])
    require(decision == timing['decision'] and split_suite.curve.select_arm(decision) == ARM,
            'the combined route must beat the baseline and each constituent at 65K')
    return decision


def baseline():
    native, prior, _, _ = speed.admission(READ)
    timing = split_suite.state.bank.basis.validation.committed(bank.RESULT, (TIMING,))
    decision = check_timing(native, prior, timing)
    reference = bank.previous.read(clock.model.basis.reference.ROOT / 'record.json')
    fixed = {name: reference[name] for name in (*clock.model.FIELDS, 'template', 'python_contract', 'python_selected')}
    runtime, measured = fixed['runtime_manifest'], timing['runtime_manifest']
    seed = fixed['environment_template']
    timing_root = str(Path(timing['plans'][0]['root']).parent)
    timing_env = {k: v.replace(seed['STRATA_MEASUREMENT_ROOT'], timing_root) for k, v in seed.items()}
    require(all(fixed[k] == timing[k] for k in
                ('configuration', 'native_validation_probe', 'boot', 'driver', 'template', 'python_contract', 'python_selected'))
            and runtime['files'] == measured['files']
            and bank.sha(runtime['path']) == runtime['sha256'] == measured['sha256']
            and bank.signature(timing_env) == timing['environment_template_sha256']
            and bank.signature(fixed['model_assets']) == timing['model_assets_sha256']
            and bank.sha(native['binaries']['strata']['path']) == native['binaries']['strata']['sha256'],
            'long generation must use the same engine, assets and ordinary route as the 65K comparison')
    return dict(fixed, binary=native['binaries']['strata'],
                native_source_commit=native['native_source_commit'], selected_arm=ARM,
                timing_sha256=bank.signature(decision), ordinary_control=clock.model.ordinary_control())


def experiment():
    # Give each campaign its own context while keeping the queued split-route
    # controller and all its frozen inputs intact.
    template = clock.model.basis.core.source.device.load_module('ordinary_65k_suite', split_suite.__file__)
    template.KEY, template.ROOT, template.baseline = KEY, ROOT, baseline
    ctx = template.experiment()
    ctx.THIS = THIS
    ctx.protocol = dict(order=list(ctx.ORDER), compare='Three independent AB, BA, AB pairs on the stock 1256-slot '
        'decode placement. A uses reload and scalar gather. B uses the KV reuse and vec8 combination selected '
        'at 65537 fresh inputs. The same executable runs official long, six-turn agent and both v5 thinking '
        'workloads in each process. The 65K prefill-only screen is separate.',
        decision='Use the maintained full-suite preservation gates. Require actual matched long-prompt work '
        'and dispatch, observed MTP classes, matching agent request bodies and work counts for the decode '
        'preservation decision. Report all paired rates, PP, client TTFT and MTP acceptance. Different output '
        'work remains workload throughput evidence. Model adoption and physical final-slot coverage are separate.',
        stop='Stop on wrong inputs or dispatch, incomplete execution, ownership, budget or runtime fault. '
        'Keep every completed pair with no speed-based stopping, replacement or automatic adoption.')
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
