"""Verify the ranked-refill build's standalone physical context."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_ranked_refill_restore as refill

physical, bank, basis, require = refill.primary.physical, refill.bank, refill.basis, refill.require
validation = basis.validation
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_ranked_refill_context_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('host', *physical.ORDER)
PLAN = dict(physical.PLAN,
            scope='Repeat the same standalone short output and full physical input fill on the ranked-refill build. '
                  'Keep the standalone stock profile and no serving cache loan. Live handoff is checked separately.')


def experiment():
    ctx = physical.experiment()
    ctx.ROOT, ctx.KEY = ROOT, KEY

    def inputs(revision):
        native, envelope, qualification = refill.qualified_build()
        seed = bank.previous.read(physical.ROOT / 'record.json')
        original = bank.previous.historic(refill.build.retained.CONTEXT, physical.KEY)
        require(basis.compact(seed) == original == validation.committed(bank.RESULT, (physical.KEY,))
                and refill.build.retained.closed(seed, 'closed')
                and seed['native_build_sha256'] == native['base_build_sha256']
                and all(seed[k] == envelope[k] for k in ('boot', 'driver', 'native_validation_probe', 'runtime_manifest'))
                and all(seed['outcomes'][name]['passed'] for name in ORDER)
                and seed['outcomes']['full-copy']['final_slot_consumed'], 'require the exact passed primary full-fill case')
        basis.comparison.shared.models.model_pins(seed['configuration'], seed['model_assets'], full=False)
        require(all(bank.sha(p) == h for p, h in seed['dflash2_assets'].items()), 'DFlash2 model or tokenizer changed')
        path = Path(seed['ranking_paths']['primary'])
        profile = path.relative_to(bank.P).as_posix()
        require(bank.sha(path) == refill.stock.quota.PROFILE_SHA256, 'original standalone expert ranking changed')
        pins = ctx.source_pins(revision)
        for name in (THIS, refill.THIS, refill.fixtures.THIS, refill.build.THIS, physical.THIS,
                     physical.checks.THIS, profile):
            source = bank.P / name
            require(source.read_bytes() == validation.git('show', revision + ':' + name),
                    'commit the exact standalone control input: ' + name)
            pins[name] = bank.sha(source)
        fields = ('configuration', 'environment_template', 'model_assets', 'native_validation_probe', 'driver',
                  'boot', 'runtime_manifest', 'dflash2_assets', 'draft_arguments', 'vocab_size', 'budgets')
        row = {k: seed[k] for k in fields}
        row.update(native_source_commit=native['native_source_commit'], native_build_sha256=envelope['build_sha256'],
                   binary=native['binaries']['strata_full_context_diagnostic'],
                   host_binary=native['binaries']['full_context_diagnostic_host_test'],
                   source_commit=revision, sources=pins, plan=PLAN, previous_sha256=bank.signature(original),
                   fixtures_sha256=bank.signature(qualification))
        return row, seed

    def cases(row, seed):
        launch = dict(row, configuration=dict(row['configuration'], args=row['draft_arguments']))
        result = []
        for name, old in zip(physical.ORDER, seed['cases'], strict=True):
            case = ctx.make_case(launch, name)
            path = Path(case['fixture']['path'])
            require(name == old['name'] and case['fixture']['sha256'] == old['fixture']['sha256']
                    and path.read_bytes() == ctx.token_bytes(case['tokens'])
                    and case['command'][1:] == [str(path) if arg == old['fixture']['path'] else arg
                                                for arg in old['command'][1:]]
                    and bank.signature(ctx.environment(row, old)) == old['environment_sha256'],
                    'standalone request, runtime or placement differs from the primary case')
            result.append(case)
        return result

    def prepare():
        require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'require a fresh capacity cohort')
        row, seed = inputs(validation.git('rev-parse', 'HEAD').decode().strip())
        ROOT.mkdir(mode=0o700)
        (ROOT / 'host').mkdir(mode=0o700)
        for label, length in (('short', 16385), ('full', ctx.CAPACITY)):
            (ROOT / ('tokens-' + label + '.txt')).write_bytes(ctx.token_bytes(length))
        row['cases'] = cases(row, seed)
        for case in row['cases']:
            for folder in ('tmp', 'cache'):
                (Path(case['root']) / folder).mkdir(mode=0o700, parents=True)
        row.update(stage='prepared', outcomes={}, complete=False, passed=False, active=False, ownership_closed=True)
        ctx.save(row)

    def current(row):
        require(basis.compact(row) == validation.committed(bank.RESULT, (KEY,)), 'commit the preceding capacity stage')
        expected, seed = inputs(row['source_commit'])
        require(all(row[k] == v for k, v in expected.items()) and row['cases'] == cases(row, seed),
                'standalone source or capacity recipe changed')

    ctx.prepare, ctx.current = prepare, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'select a physical context stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        runner = experiment()
        if action == 'prepare':
            runner.prepare()
        else:
            runner.run(action)
