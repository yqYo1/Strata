"""Check the standalone final slot on the measured early-read build."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_early_speed as timing
import dflash2_primary_context as physical

bank, basis, require = physical.bank, physical.basis, physical.require
build, SELECTOR = timing.build, timing.check.SELECTOR
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_early_capacity_v2'
ROOT = bank.B / KEY.replace('_', '-')
PLAN = dict(order=physical.PLAN['order'],
            scope='The measured early-read engine with its selector enabled on the original standalone inputs.',
            gates='Verify host bounds, the 16385-input handoff and the checked final slot at 262144 inputs. '
                  'Require the original copy-writer geometry, per-layer KV work and outputs. '
                  'Both physical fallback modes must leave the early reader idle.',
            limits='The 118785-token timing and mode-16 STOP/consumption results have separate receipts. '
                   'Live RESTORE and sustained decode remain separate.')


def prerequisites(native_context):
    measured = bank.previous.read(timing.ROOT / 'record.json')
    require(timing.wait.retained.closed(measured, 'closed')
            and basis.compact(measured) == basis.validation.committed(bank.RESULT, (timing.KEY,))
            and measured['decision'] == timing.decision(measured['processes'])
            and measured['decision']['advance'], 'require the sealed early-read timing decision')
    baseline = timing.baseline(native_context)
    require(all(measured[k] == baseline[k] for k in (
        'binary', 'native_source_commit', 'native_build_sha256', 'ple_check_sha256',
        'boot', 'driver', 'runtime_manifest', 'native_validation_probe', 'dflash2_assets')),
        'the physical check must use the timed build and runtime')
    return bank.previous.read(build.ROOT / 'record.json'), baseline, bank.signature(basis.compact(measured))


def experiment():
    native_context = build.context()
    ctx = physical.experiment()
    ctx.ROOT, ctx.KEY, ctx.ORDER = ROOT, KEY, physical.ORDER
    original_environment, original_evidence = ctx.environment, ctx.evidence
    timing_pins = timing.experiment().pins

    def environment(row, case):
        return dict(original_environment(row, case), **{SELECTOR: '1'})

    def inputs(revision):
        native, baseline, measured_sha = prerequisites(native_context)
        prior, seed = physical.reference()
        require(all(seed[k] == baseline[k] for k in (
            'boot', 'driver', 'runtime_manifest', 'native_validation_probe', 'model_assets', 'dflash2_assets')),
            'physical token reference uses another model or runtime')
        profile = bank.previous.historic(physical.checks.primary.PREFILL, 'dflash2_snapshot_direct_prefill_v1')[
            'ranking_profile']
        path = bank.P / 'data/expert-profile.bin'
        args = list(seed['draft_arguments'])
        index = args.index('--expert-profile') + 1
        require(args[index] == profile['path'] and bank.sha(path) == profile['sha256'],
                'retain the checked physical-fixture expert ranking')
        args[index] = str(path)
        pins = ctx.source_pins(revision) | timing_pins(revision)
        for name in (THIS, physical.THIS, physical.upload.THIS, str(path.relative_to(bank.P))):
            require((bank.P / name).read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact physical controller input: ' + name)
            pins[name] = bank.sha(bank.P / name)
        keys = ('configuration', 'environment_template', 'model_assets', 'native_validation_probe', 'driver',
                'boot', 'runtime_manifest', 'dflash2_assets', 'vocab_size')
        row = {k: seed[k] for k in keys}
        row.update(native_source_commit=baseline['native_source_commit'],
                   native_build_sha256=baseline['native_build_sha256'],
                   binary=native['binaries']['strata_full_context_diagnostic'],
                   host_binary=native['binaries']['full_context_diagnostic_host_test'],
                   source_commit=revision, sources=pins, budgets=ctx.BUDGET, plan=PLAN,
                   prior_evidence=physical.REFERENCE, timing_sha256=measured_sha, draft_arguments=args,
                   ranking_paths=dict(reference=profile['path'], primary=str(path)))
        return row, prior

    def cases(row, prior):
        launch = dict(row, configuration=dict(row['configuration'], args=row['draft_arguments']))
        result = []
        for old, name in zip(prior['cases'], physical.ORDER, strict=True):
            case = ctx.make_case(launch, name)
            path = Path(case['fixture']['path'])
            relocated = {old['fixture']['path']: str(path),
                         row['ranking_paths']['reference']: row['ranking_paths']['primary']}
            require(old['name'] == name and case['fixture']['sha256'] == old['fixture']['sha256']
                    and path.read_bytes() == ctx.token_bytes(case['tokens'])
                    and case['command'][1:] == [relocated.get(a, a) for a in old['command'][1:]]
                    and bank.signature(original_environment(row, dict(case, root=old['root'])))
                    == old['environment_sha256'], 'physical fixture or original launch changed')
            result.append(case)
        return result

    def evidence(stdout, stderr, row, case, pid):
        outcome = original_evidence(stdout, stderr, row, case, pid)
        require(outcome['same_fixture_output_ids'] and timing.MARKER not in stderr
                and timing.check.ISSUED not in stderr, 'the standalone fallback changed output or started an early read')
        return dict(outcome, early_read_count=0)

    def prepare():
        require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT),
                'inspect any previous capacity attempt')
        row, prior = inputs(basis.validation.git('rev-parse', 'HEAD').decode().strip())
        ROOT.mkdir(mode=0o700)
        (ROOT / 'host').mkdir(mode=0o700)
        for size, length in (('short', 16385), ('full', ctx.CAPACITY)):
            (ROOT / ('tokens-' + size + '.txt')).write_bytes(ctx.token_bytes(length))
        row['cases'] = cases(row, prior)
        for case in row['cases']:
            for folder in ('tmp', 'cache'):
                (Path(case['root']) / folder).mkdir(mode=0o700, parents=True)
        row.update(stage='prepared', outcomes={}, complete=False, passed=False, active=False, ownership_closed=True)
        ctx.save(row)

    def current(row):
        require(basis.validation.committed(bank.RESULT, (KEY,)) == basis.compact(row),
                'commit the preceding early-read capacity stage')
        expected, prior = inputs(row['source_commit'])
        require(all(row[k] == v for k, v in expected.items()) and row['cases'] == cases(row, prior),
                'the early-read capacity plan changed')

    ctx.environment, ctx.evidence, ctx.prepare, ctx.current = environment, evidence, prepare, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'host', *physical.ORDER),
            'select one planned standalone capacity stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        control = experiment()
        control.prepare() if action == 'prepare' else control.run(action)
