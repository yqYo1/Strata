"""Exercise the DFlash2 standalone final slot with the checked BF16 fallback."""

import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_kv_stage_speed as speed
import prefill_kv_upload_capacity as upload

gate, bank, basis, require = speed.gate, speed.bank, speed.basis, speed.require
KEY = 'dflash2_prefill_physical_context_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PREVIOUS = '6d5debd9'
ORDER = ('short-copy', 'full-copy')
PLAN = dict(order=['host', *ORDER],
            scope='Standalone selected DFlash2, BF16 residuals, aligned KV retention and host-backed fallback.',
            gates='Same-build host bounds, 16385 inputs with four output IDs, then all 262144 inputs '
                  'with zero output IDs and a checked final slot. Match the captured prompt tail and KV work.',
            stop='Stop on first input, extent, route, ownership, runtime or fault failure. '
                 'Keep the standalone result separate from server RESTORE and speed qualification.')


def experiment():
    ctx = upload.experiment()
    previous_environment, make_case = ctx.environment, ctx.make_case
    ctx.KEY, ctx.ROOT, ctx.ORDER = KEY, ROOT, ORDER
    speed_pins = speed.experiment(gate.SPLIT_READ).pins
    models = basis.comparison.shared.models

    def environment(row, case):
        return dict(previous_environment(row, case), STRATA_COMMIT_SYNC='1', STRATA_WATCHDOG_S='240',
                    STRATA_PREFILL_RELEASE_MTP_EXPERTS='0', STRATA_SYCL_DECODE_Q6_DPAS_HEAD='1')

    def inputs(revision):
        baseline = speed.baseline(gate.SPLIT_READ)
        native = bank.previous.read(speed.build.DFLASH2_KV_ROOT / 'record.json')
        require(bank.signature(bank.builder.compact(native)) == baseline['native_build_sha256'],
                'standalone diagnostic must belong to the same verified native build')
        prior = bank.previous.historic(PREVIOUS, upload.KEY)
        require(prior['stage'] == 'closed' and prior['passed'] and prior['ownership_closed']
                and prior['outcomes']['full-copy']['final_slot_consumed']
                and prior['model_assets_sha256'] == basis.core.source.signature(baseline['model_assets']),
                'recover the checked physical fixture for this model')
        fields = ('configuration', 'environment_template', 'model_assets', 'native_validation_probe',
                  'driver', 'boot', 'runtime_manifest', 'native_source_commit', 'native_build_sha256',
                  'dflash2_assets', 'split_model_sha256')
        row = {field: baseline[field] for field in fields}
        for field, target in (('binary', 'strata_full_context_diagnostic'),
                              ('host_binary', 'full_context_diagnostic_host_test')):
            row[field] = native['binaries'][target]
        args = list(baseline['validated_dflash2_args'])
        require('--mtp' not in args and args[args.index('--draft') + 1] == 'dflash2'
                and args[args.index('--draft-fallback') + 1] == 'none', 'require the named no-fallback route')
        profile = baseline['ranking_profile']
        index = args.index('--expert-profile') + 1
        require(args[index] == profile['old_path'], 'unexpected ranking profile in the qualified command')
        args[index] = profile['path']
        sources = speed_pins(revision) | ctx.source_pins(revision)
        for path in (THIS, upload.THIS):
            source = bank.P / path
            require(source.read_bytes() == basis.validation.git('show', revision + ':' + path),
                    'changed standalone controller: ' + path)
            sources[path] = bank.sha(source)
        manifest = row['runtime_manifest']
        providers = bank.previous.read(Path(manifest['path']))
        require(bank.sha(manifest['path']) == manifest['sha256'] and len(providers) == manifest['files']
                and all(bank.sha(path) == digest for path, digest in providers.items())
                and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'],
                'provider manifest or boot changed')
        models.model_pins(row['configuration'], row['model_assets'], full=False)
        row.update(source_commit=revision, sources=sources, draft_arguments=args, vocab_size=prior['vocab_size'],
                   prior_physical_fixture=PREVIOUS, budgets=ctx.BUDGET, plan=PLAN)
        return row, {case['tokens']: case['fixture']['sha256'] for case in prior['cases']}

    def cases(row, expected_hashes):
        # The reference configuration still identifies the verified model inventory.
        launch = dict(row, configuration=dict(row['configuration'], args=row['draft_arguments']))
        result = [make_case(launch, name) for name in ORDER]
        for case in result:
            path = Path(case['fixture']['path'])
            require(path.read_bytes() == ctx.token_bytes(case['tokens'])
                    and case['fixture']['sha256'] == expected_hashes[case['tokens']], 'fixture changed')
        return result

    def evidence(stdout, stderr, row, case, pid):
        value = upload.capacity.evidence(stdout, stderr, row, case, pid)
        rows, cold, size = map(int, ctx.one(stderr,
            r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=([01]) vram_bytes=(\d+)$',
            'standalone DFlash2 handoff'))
        require((rows, cold) == (2046, 0) and size > 0 and 'strata mtp: draft layer loaded' not in stderr,
                'DFlash2 prompt tail, KV lineage or draft owner differs')
        value['dflash2_tail_rows'] = rows
        value['dflash2_vram_bytes'] = size
        value['kv_stage'] = speed.KV.check_request(stderr, 0, value['batched_rows'], ctx.CHUNK, reuse=True,
                                                  tap_layers=gate.TAPPED_QSA_LAYERS)
        return value

    def prepare():
        require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT),
                'inspect any existing standalone context attempt')
        revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
        row, fixtures = inputs(revision)
        vocab = json.loads((Path(row['configuration']['tokenizer']) / 'vocab.json').read_bytes())
        require(set(range(4096)) <= set(vocab.values()) and max(vocab.values()) + 1 == row['vocab_size'],
                'fixture vocabulary differs from the checked model')
        ROOT.mkdir(mode=0o700)
        (ROOT / 'host').mkdir(mode=0o700)
        for label, length in (('short', 16385), ('full', ctx.CAPACITY)):
            (ROOT / ('tokens-' + label + '.txt')).write_bytes(ctx.token_bytes(length))
        row['cases'] = cases(row, fixtures)
        for case in row['cases']:
            for folder in ('tmp', 'cache'):
                (Path(case['root']) / folder).mkdir(mode=0o700, parents=True)
        row.update(stage='prepared', outcomes={}, complete=False, passed=False, active=False, ownership_closed=True)
        ctx.save(row)

    def current(row):
        require(basis.validation.committed(bank.RESULT, (KEY,)) == basis.compact(row),
                'commit the exact standalone context receipt')
        expected, fixtures = inputs(row['source_commit'])
        require(all(row[key] == value for key, value in expected.items()) and row['cases'] == cases(row, fixtures),
                'frozen standalone context inputs changed')

    ctx.current, ctx.prepare, ctx.environment, ctx.evidence = current, prepare, environment, evidence
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'host', *ORDER),
            'select a standalone stage with the pinned benchmark interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        control = experiment()
        if action == 'prepare':
            control.prepare()
        else:
            control.run(action)
