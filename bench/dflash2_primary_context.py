"""Recheck the matched DFlash2 physical-context input on the primary build."""

import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_primary_checks as checks
import prefill_kv_upload_capacity as upload
import prefill_kv_stage_trace as kv

bank, basis, require = checks.bank, checks.bank.basis, checks.require
KEY = 'dflash2_primary_physical_context_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
REFERENCE = '321c5776ab4c73d502529d4a4f23f7175f39bfd6'
REFERENCE_KEY = 'dflash2_snapshot_physical_context_v1'
RECEIPT = bank.B / REFERENCE_KEY.replace('_', '-') / 'record.json'
ORDER = ('short-copy', 'full-copy')
TAPPED_QSA_LAYERS = (15, 35)
PLAN = dict(order=['host', *ORDER],
            scope='Same standalone DFlash2 copy writer and original token IDs. Change only the native build.',
            gates='Same-build host bounds, 16385 inputs and four output IDs, then all 262144 inputs '
                  'and zero output IDs. Require actual fallback allocation, tap tail and cache coverage.',
            stop='Stop on the first input, extent, route, validation, ownership or Xe fault. '
                 'Keep observed token IDs for diagnostics. Live RESTORE and throughput are separate.')


def reference():
    prior = bank.previous.historic(REFERENCE, REFERENCE_KEY)
    seed = bank.previous.read(RECEIPT)
    require(basis.compact(seed) == prior and prior['stage'] == 'closed' and prior['passed']
            and prior['complete'] and prior['ownership_closed'] and not prior['active'],
            'recover the original closed physical-context fixture')
    short, full = (prior['outcomes'][label] for label in ORDER)
    require(short['passed'] and full['passed'] and short['generated_ids'] == 4
            and full['generated_ids'] == 0 and full['batched_rows'] == upload.capacity.CAPACITY - 1
            and full['final_slot_consumed'], 'require the original short handoff and checked final slot')
    return prior, seed


def verify_kv(text, rows, expected):
    positions = range(0, rows, upload.capacity.CHUNK)
    planned = []
    for layer in range(3, 48, 4):
        loaded, kept, _ = kv.expected_prefix(positions, layer, reuse=layer not in TAPPED_QSA_LAYERS)
        planned.append((layer, loaded, kept))
    actual, mirrors = [], []
    for line in text.splitlines():
        if line.startswith('strata prefill KV stage:'):
            match = kv.STAGE.fullmatch(line)
            require(match, 'malformed KV staging record')
            actual.append(tuple(map(int, match.groups())))
        elif line.startswith('strata QSA mirror bulk:'):
            match = kv.MIRROR.fullmatch(line)
            require(match, 'malformed KV host publication record')
            mirrors.append(tuple(map(int, match.groups())))
    require(actual == planned and len(mirrors) == 12 * len(positions),
            'per-layer KV staging or mirror coverage changed')
    required_chunks = [(min(upload.capacity.CHUNK, rows - pos), pos) for pos in positions]
    require([(m[0], m[1]) for m in mirrors] == required_chunks * 12,
            'KV host mirror skipped or reordered an input chunk')
    result = dict(reloaded_pages=sum(p[1] for p in actual), reused_pages=sum(p[2] for p in actual),
                  mirror_D2H_calls=sum(m[3] for m in mirrors), mirror_D2H_bytes=sum(m[4] for m in mirrors))
    require(all(expected[k] == v for k, v in result.items()), 'KV staging or host publication counts changed')
    return result


def experiment():
    ctx = upload.experiment()
    ctx.ROOT, ctx.KEY, ctx.ORDER = ROOT, KEY, ORDER
    old_environment, make_case = ctx.environment, ctx.make_case

    def environment(row, case):
        return dict(old_environment(row, case), STRATA_COMMIT_SYNC='1', STRATA_WATCHDOG_S='240',
                    STRATA_PREFILL_RELEASE_MTP_EXPERTS='0', STRATA_SYCL_DECODE_Q6_DPAS_HEAD='1')

    def inputs(revision):
        prior, seed = reference()
        native, probe = checks.build()
        fixture = basis.validation.committed(bank.RESULT, (checks.KEY,))
        require(fixture['stage'] == 'restore_passed' and fixture['complete'] and fixture['passed']
                and fixture['ownership_closed'] and not fixture['active']
                and fixture['build_sha256'] == bank.signature(bank.builder.compact(native)),
                'finish the same-build host and device fixture sequence')
        envelope = checks.parameters(native, probe)
        require(all(seed[k] == envelope[k] for k in ('boot', 'driver', 'runtime_manifest', 'native_validation_probe')),
                'physical context reference uses different runtime inputs')
        require(all(bank.sha(p) == h for p, h in seed['dflash2_assets'].items()), 'draft weights or vocabulary changed')
        basis.comparison.shared.models.model_pins(seed['configuration'], seed['model_assets'], full=False)
        pins = ctx.source_pins(revision)
        for name in (THIS, checks.THIS, upload.THIS):
            raw = basis.validation.git('show', revision + ':' + name)
            require(raw == (bank.P / name).read_bytes(), 'changed model controller: ' + name)
            pins[name] = bank.sha(bank.P / name)
        keys = ('configuration', 'environment_template', 'model_assets', 'native_validation_probe', 'driver',
                'boot', 'runtime_manifest', 'dflash2_assets', 'draft_arguments', 'vocab_size')
        row = {k: seed[k] for k in keys}
        row.update(native_source_commit=native['native_source_commit'], native_build_sha256=envelope['build_sha256'],
                   binary=native['binaries']['strata_full_context_diagnostic'],
                   host_binary=native['binaries']['full_context_diagnostic_host_test'],
                   source_commit=revision, sources=pins, budgets=ctx.BUDGET, plan=PLAN,
                   prior_evidence=REFERENCE, fixture_sha256=bank.signature(fixture))
        return row, prior

    def cases(row, prior):
        launch = dict(row, configuration=dict(row['configuration'], args=row['draft_arguments']))
        result = []
        for old, name in zip(prior['cases'], ORDER, strict=True):
            case = make_case(launch, name)
            path = Path(case['fixture']['path'])
            require(old['name'] == name and case['fixture']['sha256'] == old['fixture']['sha256']
                    and path.read_bytes() == ctx.token_bytes(case['tokens'])
                    and case['command'][1:] == [str(path) if a == old['fixture']['path'] else a
                                                for a in old['command'][1:]]
                    and bank.signature(environment(row, dict(case, root=old['root']))) == old['environment_sha256'],
                    'frozen standalone argument, token or runtime recipe changed')
            result.append(case)
        return result

    def evidence(stdout, stderr, row, case, pid):
        result = upload.capacity.evidence(stdout, stderr, row, case, pid)
        rows, cold, size = map(int, ctx.one(stderr,
            r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=([01]) vram_bytes=(\d+)$',
            'standalone DFlash2 handoff'))
        require((rows, cold) == (2046, 0) and size > 0 and 'strata mtp: draft layer loaded' not in stderr
                and result['window']['mode'] == (2 if case['full'] else 10), 'DFlash2 tail or fallback route changed')
        prior, _ = reference()
        original = prior['outcomes'][case['name']]
        return dict(result, dflash2_tail_rows=rows, dflash2_vram_bytes=size,
                    kv_stage=verify_kv(stderr, result['batched_rows'], original['kv_stage']),
                    same_fixture_output_ids=result['output_ids_sha256'] == original['output_ids_sha256'])

    def prepare():
        require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT),
                'inspect the existing primary physical-context attempt first')
        row, prior = inputs(basis.validation.git('rev-parse', 'HEAD').decode().strip())
        vocab = json.loads((Path(row['configuration']['tokenizer']) / 'vocab.json').read_bytes())
        require(set(range(4096)) <= set(vocab.values()) and max(vocab.values()) + 1 == row['vocab_size'],
                'physical fixture IDs outside the checked model vocabulary')
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
                'commit the preceding primary context stage')
        expected, prior = inputs(row['source_commit'])
        require(all(row[k] == v for k, v in expected.items()) and row['cases'] == cases(row, prior),
                'primary context plan or inputs changed')

    ctx.environment, ctx.evidence, ctx.prepare, ctx.current = environment, evidence, prepare, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'host', *ORDER),
            'select one planned primary physical-context stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        else:
            ctx.run(action)
