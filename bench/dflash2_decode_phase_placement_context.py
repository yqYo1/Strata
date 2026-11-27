"""Check the timed decode-placement build's physical context and live RESTORE."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_phase_placement_speed as timing
import dflash2_primary_context as physical

bank, basis, require = physical.bank, physical.basis, physical.require
checked = timing.checked
stock, resume = checked.stock, checked.stock.primary
THIS = Path(__file__).resolve().relative_to(bank.P).as_posix()
KEY = 'dflash2_decode_phase_placement_context_v1'
ROOT = bank.B / KEY.replace('_', '-')
RESTORE_KEY = 'dflash2_decode_phase_placement_restore_v1'
RESTORE_ROOT = bank.B / RESTORE_KEY.replace('_', '-')
RESTORE_ORDER = ('host', 'near-stock', 'near-blend')
PLAN = dict(physical.PLAN,
    scope='Repeat the standalone short and physical full-fill token bodies on the timed build. '
          'Use the same global stock expert map as the 96K comparison.',
    stop='Check actual handoff, all 262144 input positions, final commit and refill drain, '
         'zero full-context outputs, normal retirement and no fresh Xe fault. '
         'Live serving RESTORE and the decode-only selector require their own check.')
RESTORE_PLAN = dict(resume.PLAN,
    order=list(RESTORE_ORDER),
    scope='Restore the original global stock workload on the timed serving executable, '
          'first with the decode selector omitted, then with it enabled.',
    cases='Park 253919 tokens, switch to 33 tokens, restore and read 8209 more tokens. '
          'Reach position 262128 with one output and an idle placement selector.',
    limits='Live target and cache ownership at the context boundary. The earlier short checks '
           'cover switching and STOP with sufficient generation room.')


def qualified_speed():
    measured = bank.previous.read(timing.ROOT / 'record.json')
    compact = basis.compact(measured)
    require(timing.build.retained.closed(measured, 'closed')
            and compact == basis.validation.committed(bank.RESULT, (timing.KEY,))
            and measured['decision'] == timing.decision(measured['processes'])
            and measured['decision']['eligible_for_qualification'],
            'complete the paired timing and prefill preservation gates first')
    selected = timing.baseline()
    require(all(measured[key] == selected[key] for key in (
        'binary', 'native_source_commit', 'native_build_sha256', 'boot', 'driver',
        'runtime_manifest', 'native_validation_probe', 'dflash2_assets', 'model_assets',
        'qualified_profiles', 'placement_validation_sha256')),
        'the timed build, selected profiles or runtime changed')
    return measured, bank.signature(compact)


def experiment():
    ctx = physical.experiment()
    ctx.ROOT, ctx.KEY = ROOT, KEY
    original_environment, original_evidence = ctx.environment, ctx.evidence
    timing_pins = timing.experiment().pins

    def environment(row, case):
        env = original_environment(row, case)
        require(not any(key in env for key in (checked.SELECTOR, checked.AUDIT, checked.CLOCK)),
                'the standalone fixture uses only the global stock map')
        return env

    def inputs(revision):
        measured, measured_sha = qualified_speed()
        native = bank.previous.read(timing.build.ROOT / 'record.json')
        prior, seed = physical.reference()
        keys = ('configuration', 'environment_template', 'model_assets', 'native_validation_probe',
                'driver', 'boot', 'runtime_manifest', 'dflash2_assets', 'vocab_size')
        require(all(seed[key] == measured[key] for key in (
            'driver', 'boot', 'runtime_manifest', 'native_validation_probe', 'model_assets', 'dflash2_assets')),
            'the physical-context reference uses another model or runtime')
        pins = ctx.source_pins(revision) | timing_pins(revision)
        for name in (THIS, physical.THIS, physical.upload.THIS):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit exact physical-context input: ' + name)
            pins[name] = bank.sha(path)
        return dict({key: seed[key] for key in keys}, source_commit=revision, sources=pins,
                    native_source_commit=measured['native_source_commit'],
                    native_build_sha256=measured['native_build_sha256'],
                    binary=native['binaries']['strata_full_context_diagnostic'],
                    host_binary=native['binaries']['full_context_diagnostic_host_test'],
                    plan=PLAN, budgets=ctx.BUDGET, draft_arguments=seed['draft_arguments'],
                    stock_profile=measured['qualified_profiles']['stock'], timing_sha256=measured_sha,
                    prior_evidence=physical.REFERENCE), prior

    def cases(row, prior):
        launch = dict(row, configuration=dict(row['configuration'], args=row['draft_arguments']))
        result = []
        for old, name in zip(prior['cases'], physical.ORDER, strict=True):
            case = ctx.make_case(launch, name)
            path = Path(case['fixture']['path'])
            original_args = [str(path) if arg == old['fixture']['path'] else arg for arg in old['command'][1:]]
            require(old['name'] == name and case['fixture']['sha256'] == old['fixture']['sha256']
                    and path.read_bytes() == ctx.token_bytes(case['tokens'])
                    and case['command'][1:] == original_args
                    and bank.signature(ctx.environment(row, dict(case, root=old['root'])))
                    == old['environment_sha256'], 'physical fixture or original launch changed')
            command = case['command']
            require(command.count('--expert-profile') == command.count('--expert-cache-per-layer') == 1
                    and command[command.index('--expert-cache') + 1] == '1256',
                    'the original standalone expert layout changed')
            command[command.index('--expert-profile') + 1] = row['stock_profile']['path']
            command[command.index('--expert-cache') + 1] = '968'
            command.remove('--expert-cache-per-layer')
            result.append(case)
        return result

    def evidence(stdout, stderr, row, case, pid):
        result = original_evidence(stdout, stderr, row, case, pid)
        stock.check_loader(stderr, row['stock_profile']['path'], per_layer=False)
        require('strata decode placement:' not in stderr, 'standalone context cannot switch decode placement')
        return result

    def prepare():
        require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT),
                'review any earlier physical context attempt')
        row, prior = inputs(basis.validation.git('rev-parse', 'HEAD').decode().strip())
        ROOT.mkdir(mode=0o700)
        (ROOT / 'host').mkdir(mode=0o700)
        for label, length in (('short', 16385), ('full', ctx.CAPACITY)):
            (ROOT / ('tokens-' + label + '.txt')).write_bytes(ctx.token_bytes(length))
        row['cases'] = cases(row, prior)
        for case in row['cases']:
            for folder in ('tmp', 'cache'):
                (Path(case['root']) / folder).mkdir(mode=0o700, parents=True)
        row.update(stage='prepared', outcomes={}, complete=False, passed=False, active=False, ownership_closed=True)
        ctx.save(row)

    def current(row):
        require(basis.validation.committed(bank.RESULT, (KEY,)) == basis.compact(row),
                'commit the preceding capacity stage')
        expected, prior = inputs(row['source_commit'])
        require(all(row[key] == value for key, value in expected.items()) and row['cases'] == cases(row, prior),
                'the frozen standalone context inputs changed')

    ctx.environment, ctx.evidence, ctx.prepare, ctx.current = environment, evidence, prepare, current
    return ctx


def restore_experiment():
    ctx = resume.experiment()
    ctx.ROOT, ctx.KEY, ctx.ORDER = RESTORE_ROOT, RESTORE_KEY, RESTORE_ORDER
    original_environment, original_inspect = ctx.environment, ctx.inspect
    timing_pins = timing.experiment().pins

    def diagnostic_environment(row, case):
        return dict(original_environment(row, case), STRATA_CPU_WORK_CENSUS='1', STRATA_DECODE_TIMING='1')

    def environment(row, case):
        env = diagnostic_environment(row, case)
        require(not any(key in env for key in (checked.SELECTOR, checked.AUDIT, checked.CLOCK)),
                'the reference already contains a decode placement override')
        env[checked.CLOCK] = '1'
        if case['name'] == 'near-blend':
            env[checked.SELECTOR] = row['profiles']['blend']['path']
        return env

    def inputs(revision):
        measured, measured_sha = qualified_speed()
        capacity = basis.validation.committed(bank.RESULT, (KEY,))
        native = bank.previous.read(timing.build.ROOT / 'record.json')
        require(timing.build.retained.closed(capacity, 'closed')
                and capacity['outcomes']['full-copy']['final_slot_consumed']
                and capacity['outcomes']['full-copy']['batched_rows'] == 262143
                and capacity['outcomes']['full-copy']['generated_ids'] == 0
                and capacity['binary'] == native['binaries']['strata_full_context_diagnostic']
                and capacity['native_build_sha256'] == measured['native_build_sha256']
                and capacity['timing_sha256'] == measured_sha,
                'complete the checked standalone final slot on this build')
        reference = basis.validation.committed(bank.RESULT, (stock.KEY,))
        old = bank.previous.read(stock.ROOT / 'record.json')
        require(timing.build.retained.closed(old, 'closed')
                and basis.compact(old) == reference
                and old['outcomes']['near-global']['restored_tokens'] == 253919
                and old['outcomes']['near-global']['requests'][-1]['prompt'] == 262128
                and old['profile_sha256'] == measured['qualified_profiles']['stock']['sha256']
                and old['drafter_assets'] == measured['dflash2_assets']
                and all(old[key] == measured[key] for key in (
                    'boot', 'driver', 'runtime_manifest', 'native_validation_probe', 'model_assets')),
                'recover the exact global restore workload and runtime')
        pins = ctx.sources(revision) | timing_pins(revision)
        for name in (THIS, stock.THIS, resume.THIS, str(stock.STOCK.relative_to(bank.P))):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit exact live-restore input: ' + name)
            pins[name] = bank.sha(path)
        keys = ('configuration', 'environment_template', 'model_assets', 'native_validation_probe',
                'driver', 'boot', 'runtime_manifest', 'vocab_size', 'draft_arguments')
        row = dict({key: old[key] for key in keys}, source_commit=revision, sources=pins,
                   native_source_commit=measured['native_source_commit'],
                   native_build_sha256=measured['native_build_sha256'], binary=measured['binary'],
                   profiles=measured['qualified_profiles'], timing_sha256=measured_sha,
                   physical_context_sha256=bank.signature(capacity), plan=RESTORE_PLAN,
                   reference_sha256=bank.signature(reference))
        original, = [case for case in old['plans'] if case['name'] == 'near-global']
        require(original['requests'] == ctx.input_plan(True) and original['near'] and not original['direct']
                and original['command'][original['command'].index('--expert-profile') + 1]
                    == row['profiles']['stock']['path']
                and bank.signature(diagnostic_environment(row, original)) == original['environment_sha256'],
                'the original context, wire bodies or environment changed')
        plans = []
        for name in RESTORE_ORDER[1:]:
            case = dict(original, name=name, root=str(RESTORE_ROOT / name),
                        command=[measured['binary']['path'], *original['command'][1:]])
            case['environment_sha256'] = bank.signature(environment(row, case))
            plans.append(case)
        row['plans'] = plans
        return row

    def inspect(log, row, case, requests, pid):
        result = original_inspect(log, row, dict(case, name='near-copy'), requests, pid)
        stock.check_loader(log, row['profiles']['stock']['path'], per_layer=False)
        _, members, formats = stock.geometry(row)
        result.update(stock.check_placement(log, requests, members, formats))
        require('decode phase profile inactive' not in log and 'strata decode placement:' not in log
                and result['restored_tokens'] == 253919 and requests[-1]['prompt'] == 262128,
                'the enabled selector must stay idle with one available output')
        if case['name'] == 'near-blend':
            control = row['outcomes']['near-stock']
            require(all(result[key] == control[key] for key in ('restore_payload', 'loans', 'decode_placement')),
                    'the enabled selector changed the near-capacity restore or expert layout')
        return result

    def prepare():
        require(not RESTORE_ROOT.exists() and RESTORE_KEY not in basis.validation.committed(bank.RESULT),
                'review any earlier live restore attempt')
        row = inputs(basis.validation.git('rev-parse', 'HEAD').decode().strip())
        RESTORE_ROOT.mkdir(mode=0o700)
        (RESTORE_ROOT / 'host').mkdir(mode=0o700)
        for case in row['plans']:
            for folder in ('tmp', 'cache'):
                (Path(case['root']) / folder).mkdir(mode=0o700, parents=True)
        row.update(stage='prepared', outcomes={}, complete=False, passed=False, active=False,
                   ownership_closed=True, speed_eligible=False)
        ctx.save(row)

    def current(row):
        require(basis.validation.committed(bank.RESULT, (RESTORE_KEY,)) == ctx.compact(row),
                'commit the preceding live restore stage')
        expected = inputs(row['source_commit'])
        require(all(row[key] == value for key, value in expected.items()), 'the frozen live restore inputs changed')

    ctx.environment, ctx.inspect, ctx.prepare, ctx.current = environment, inspect, prepare, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    live_restore = action.startswith('restore-')
    step = action.removeprefix('restore-')
    order = RESTORE_ORDER if live_restore else ('host', *physical.ORDER)
    require(sys.executable == basis.reference.model.PYTHON and step in ('prepare', *order),
            'select one physical context or live restore stage with the qualified interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        control = restore_experiment() if live_restore else experiment()
        control.prepare() if step == 'prepare' else control.run(step)
