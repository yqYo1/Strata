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
SNAPSHOT_KEY = 'dflash2_snapshot_physical_context_v1'
INCUMBENT_FULL = '0795b5ada220f49afc4e7b456340b9bc019c9462'
SNAPSHOT_RESTORE = '962acee66e7e3ff4969370f8fadb28e673fba006'
ORDER = ('short-copy', 'full-copy')
PLAN = dict(order=['host', *ORDER],
            scope='Standalone selected DFlash2, BF16 residuals, aligned KV retention and host-backed fallback.',
            gates='Same-build host bounds, 16385 inputs with four output IDs, then all 262144 inputs '
                  'with zero output IDs and a checked final slot. Match the captured prompt tail and KV work.',
            stop='Stop on first input, extent, route, ownership, runtime or fault failure. '
                 'Keep the standalone result separate from server RESTORE and speed qualification.')


def snapshot_reference(baseline):
    import dflash2_target_snapshot_build as snapshot

    candidate = bank.previous.read(snapshot.ROOT / 'record.json')
    snapshot.context().check(candidate)
    build_sha256 = bank.signature(bank.builder.compact(candidate))
    restore_key = 'dflash2_target_snapshot_restore_v2'
    tested = bank.previous.historic(SNAPSHOT_RESTORE, restore_key)
    require(candidate['stage'] == 'built' and candidate['complete'] and candidate['passed']
            and candidate['host_control']['passed'] and candidate['ownership_closed']
            and not candidate['active'] and candidate['driver'] == baseline['driver']
            and candidate['projection']['target_snapshot_parent_hash'] == baseline['binary']['sha256']
            and tested['binary'] == candidate['binaries']['strata']
            and tested['native_build_sha256'] == build_sha256
            and tested['stage'] == 'closed' and tested['complete'] and tested['passed']
            and tested['ownership_closed'] and not tested['active']
            and all(tested[k] == baseline[k] for k in ('boot', 'driver', 'runtime_manifest',
                                                        'native_validation_probe'))
            and tested['model_assets_sha256'] == bank.signature(baseline['model_assets'])
            and tested['drafter_assets'] == baseline['dflash2_assets'],
            'recover the tested coherent target-only engine and same-model restore')
    near = tested['outcomes']['near-copy']
    require(near['passed'] and near['ownership_closed'] and near['native_exit_code'] == 0
            and not near['fresh_faults'] and near['restored_tokens'] == 253919
            and near['restore_payload']['result'] == 'PASS'
            and near['requests'][-1]['prompt'] == 262128,
            'recover the verified near-capacity target-only restore')
    return candidate, build_sha256, snapshot.THIS


def experiment(*, snapshot=False):
    ctx = upload.experiment()
    previous_environment, make_case = ctx.environment, ctx.make_case
    key = SNAPSHOT_KEY if snapshot else KEY
    root = bank.B / SNAPSHOT_KEY.replace('_', '-') if snapshot else ROOT
    source_commit = INCUMBENT_FULL if snapshot else PREVIOUS
    plan = dict(PLAN, scope='Standalone target-only snapshot build. Reuse the same short and full DFlash2 '
                            'BF16 fallback fixtures and require the original short output IDs.') if snapshot else PLAN
    ctx.KEY, ctx.ROOT, ctx.ORDER = key, root, ORDER
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
        prior_key = KEY if snapshot else upload.KEY
        prior = bank.previous.historic(source_commit, prior_key)
        require(prior['stage'] == 'closed' and prior['passed'] and prior['ownership_closed']
                and prior['outcomes']['full-copy']['final_slot_consumed']
                and prior['model_assets_sha256'] == basis.core.source.signature(baseline['model_assets']),
                'recover the checked physical fixture for this model')
        fields = ('configuration', 'environment_template', 'model_assets', 'native_validation_probe',
                  'driver', 'boot', 'runtime_manifest', 'native_source_commit', 'native_build_sha256',
                  'dflash2_assets', 'split_model_sha256')
        row = {field: baseline[field] for field in fields}
        paths = [THIS, upload.THIS]
        if snapshot:
            require(prior['native_source_commit'] == baseline['native_source_commit']
                    and prior['native_build_sha256'] == baseline['native_build_sha256']
                    and all(prior[k] == baseline[k] for k in ('boot', 'driver', 'runtime_manifest',
                                                             'native_validation_probe', 'dflash2_assets'))
                    and prior['outcomes']['full-copy']['batched_rows'] == ctx.CAPACITY - 1
                    and prior['outcomes']['full-copy']['generated_ids'] == 0
                    and prior['outcomes']['short-copy']['passed']
                    and prior['outcomes']['short-copy']['generated_ids'] == 4,
                    'recover the completed incumbent standalone comparison and output IDs')
            native, build_hash, snapshot_source = snapshot_reference(baseline)
            paths.append(snapshot_source)
            row.update(native_source_commit=native['native_source_commit'], native_build_sha256=build_hash,
                       verified_restore=SNAPSHOT_RESTORE,
                       incumbent_short_ids=prior['outcomes']['short-copy']['output_ids_sha256'])
        for field, target in (('binary', 'strata_full_context_diagnostic'),
                              ('host_binary', 'full_context_diagnostic_host_test')):
            row[field] = native['binaries'][target]
            image = Path(row[field]['path'])
            require(image.stat().st_size == row[field]['bytes'] and bank.sha(image) == row[field]['sha256'],
                    'checked standalone fixture changed: ' + target)
        args = list(baseline['validated_dflash2_args'])
        require('--mtp' not in args and args[args.index('--draft') + 1] == 'dflash2'
                and args[args.index('--draft-fallback') + 1] == 'none', 'require the named no-fallback route')
        profile = baseline['ranking_profile']
        index = args.index('--expert-profile') + 1
        require(args[index] == profile['old_path'], 'unexpected ranking profile in the qualified command')
        args[index] = profile['path']
        sources = speed_pins(revision) | ctx.source_pins(revision)
        for path in paths:
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
                   prior_physical_fixture=source_commit, budgets=ctx.BUDGET, plan=plan)
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
        if snapshot and not case['full']:
            require(value['output_ids_sha256'] == row['incumbent_short_ids'],
                    'target-only snapshot build changed deterministic standalone IDs')
        value['dflash2_tail_rows'] = rows
        value['dflash2_vram_bytes'] = size
        value['kv_stage'] = speed.KV.check_request(stderr, 0, value['batched_rows'], ctx.CHUNK, reuse=True,
                                                  tap_layers=gate.TAPPED_QSA_LAYERS)
        return value

    def prepare():
        require(not root.exists() and key not in basis.validation.committed(bank.RESULT),
                'inspect any existing standalone context attempt')
        revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
        row, fixtures = inputs(revision)
        vocab = json.loads((Path(row['configuration']['tokenizer']) / 'vocab.json').read_bytes())
        require(set(range(4096)) <= set(vocab.values()) and max(vocab.values()) + 1 == row['vocab_size'],
                'fixture vocabulary differs from the checked model')
        root.mkdir(mode=0o700)
        (root / 'host').mkdir(mode=0o700)
        for label, length in (('short', 16385), ('full', ctx.CAPACITY)):
            (root / ('tokens-' + label + '.txt')).write_bytes(ctx.token_bytes(length))
        row['cases'] = cases(row, fixtures)
        for case in row['cases']:
            for folder in ('tmp', 'cache'):
                (Path(case['root']) / folder).mkdir(mode=0o700, parents=True)
        row.update(stage='prepared', outcomes={}, complete=False, passed=False, active=False, ownership_closed=True)
        ctx.save(row)

    def current(row):
        require(basis.validation.committed(bank.RESULT, (key,)) == basis.compact(row),
                'commit the exact standalone context receipt')
        expected, fixtures = inputs(row['source_commit'])
        require(all(row[key] == value for key, value in expected.items()) and row['cases'] == cases(row, fixtures),
                'frozen standalone context inputs changed')

    ctx.current, ctx.prepare, ctx.environment, ctx.evidence = current, prepare, environment, evidence
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    snapshot_mode = action.startswith('snapshot-')
    if snapshot_mode:
        action = action.removeprefix('snapshot-')
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'host', *ORDER),
            'select a standalone stage with the pinned benchmark interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        control = experiment(snapshot=snapshot_mode)
        if action == 'prepare':
            control.prepare()
        else:
            control.run(action)
