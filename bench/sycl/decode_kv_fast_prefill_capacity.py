"""Exercise the physical last slot on the fast-prefill decode KV build."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import decode_kv_fast_prefill_restore as handoff
import prefill_kv_capacity as parent

bank, basis, require = parent.bank, parent.basis, parent.require
build = handoff.build
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'decode_kv_fast_prefill_capacity_v1'
ROOT = bank.B / 'decode-kv-fast-prefill-capacity-v1'
ORIGINAL = '5fb96bc69b26f1cc01241c0685b60a52ff5c1834'
ORDER = ('short-off', 'short-on', 'full-off', 'full-on')
TARGET = 'strata_full_context_diagnostic'
HOST = 'full_context_diagnostic_host_test'
SOURCES = [THIS, handoff.THIS, build.THIS, *parent.SOURCES]
PLAN = dict(order=list(ORDER),
    scope='Reuse the original standalone 16385-token smoke and exact 262144-token body. '
          'Keep aligned prefill KV reuse in every arm. Toggle only the decode KV selector. '
          'Check actual resident/host rows, KV staging and mirror writes, terminal verifier and drain. '
          'The full final verifier uses scalar width one in both arms. The separate server RESTORE gate tests width four.',
    stop='Stop on wrong source/build/provider/route, bad terminal marker, changed output or state extent, '
         'budget, ownership or runtime failure. Diagnostic safety evidence only.')


def inputs():
    candidate, speed, build_sha, device_sha = handoff.inputs()
    retained = bank.previous.read(build.stock.MODEL_ROOT / 'record.json')
    original = bank.previous.historic(ORIGINAL, parent.KEY)
    require(original['stage'] == 'closed' and original['complete'] and original['passed']
            and original['ownership_closed'] and not original['active']
            and original['binary'] == retained['binaries'][TARGET]
            and original['native_source_commit'] == retained['native_source_commit']
            and candidate['projection']['parent_binary'] == retained['binaries']['strata']
            and all(original[k] == speed[k] for k in ('configuration', 'boot', 'driver', 'native_validation_probe'))
            and original['model_assets_sha256'] == bank.signature(speed['model_assets'])
            and original['environment_template_sha256'] == bank.signature(speed['environment_template'])
            and original['runtime_manifest']['sha256'] == speed['runtime_manifest']['sha256'],
            'require the measured standalone controls on the exact retained image and runtime')
    cases = {c['name'].removesuffix('-reuse'): c for c in original['cases'] if c['name'].endswith('-reuse')}
    expected = {}
    for size in ('short', 'full'):
        row = original['outcomes'][size + '-reuse']
        require(row['passed'] and row['complete'] and row['GPU_executed'] and row['ownership_closed']
                and row['native_exit_code'] == 0 and not row['fresh_faults']
                and row['final_slot_consumed'] == (size == 'full'), 'previous work or final slot is unqualified')
        expected[size] = {k: row[k] for k in (*parent.MATCH, 'kv')}
    proof = dict(build_sha256=build_sha, device_sha256=device_sha, standalone_origin=bank.signature(original))
    return candidate, speed, cases, expected, proof, original['vocab_size']


def experiment():
    ctx = basis.core.source.device.load_module('decode_kv_capacity', parent.capacity.__file__)
    ctx.ROOT, ctx.KEY, ctx.ORDER = ROOT, KEY, ORDER
    parent_env = parent.experiment().environment
    standalone_evidence = ctx.evidence

    def environment(row, case):
        ordinary = parent_env(row, dict(case, name='full-reuse' if case['full'] else 'short-reuse'))
        require(ordinary['STRATA_SYCL_KV_COPY_GROUPS'] == '0' and handoff.SELECTOR not in ordinary,
                'require the original scalar copy settings')
        return dict(ordinary, **{handoff.SELECTOR: str(int(case['name'].endswith('-on')))})

    def evidence(stdout, stderr, row, case, pid):
        value = standalone_evidence(stdout, stderr, row, case, pid)
        size = 'full' if case['full'] else 'short'
        counts = value['store']
        geometry = dict(pieces=counts['pieces'], gpu_rows=value['gpu_rows'], maximum_rows=value['window']['maximum_rows'])
        value['attention_input'] = parent.model.input_receipt(stderr, value['batched_rows'], geometry, False)
        value['kv'] = parent.state.kv.check_request(stderr, 0, value['batched_rows'], ctx.CHUNK, reuse=True)
        require(all(value[k] == v for k, v in row['reference'][size].items())
                and 'strata MoE prefill gather:' not in stderr and 'strata MoE mixed banks: storage=scratch ' not in stderr
                and 'strata MoE prefix expert upload' not in stderr, 'standalone workload, residency or publication changed')
        enabled = case['name'].endswith('-on')
        captures = handoff.previous.kv_captures(stderr, int(enabled))
        require(any(c['width'] == 1 for c in captures) and (not case['full'] or all(c['width'] == 1 for c in captures)),
                'terminal width-one verifier was not observed')
        return dict(value, KV_captures=captures)

    def cases(row, old_cases):
        planned = []
        for name in ORDER:
            case = ctx.make_case(row, name)
            old = old_cases[name.split('-')[0]]
            args = [case['fixture']['path'] if arg == old['fixture']['path'] else arg for arg in old['command'][1:]]
            env = environment(row, dict(case, root=old['root']))
            del env[handoff.SELECTOR]
            require(case['fixture']['sha256'] == old['fixture']['sha256'] and case['command'][1:] == args
                    and bank.signature(env) == old['environment_sha256'], 'original input or standalone settings changed')
            planned.append(case)
        return planned

    def fixed():
        candidate, speed, original, expected, proof, vocab = inputs()
        pins = bank.stage.source.pin(bank.P, SOURCES)
        values = {k: speed[k] for k in parent.model.FIELDS}
        return dict(values, sources=pins['source_sha256'], native_source_commit=candidate['native_source_commit'],
                    qualification=proof, reference=expected, vocab_size=vocab, binary=candidate['binaries'][TARGET],
                    host_binary=candidate['binaries'][HOST], plan=PLAN, budgets=ctx.BUDGET), original, pins['revision']

    def prepare():
        require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'fresh capacity case required')
        values, original, revision = fixed()
        row = dict(values, source_commit=revision, stage='prepared', outcomes={}, complete=False, passed=False,
                   active=False, ownership_closed=True)
        ROOT.mkdir(mode=0o700)
        (ROOT / 'host').mkdir(mode=0o700)
        for size, case in original.items():
            (ROOT / f'tokens-{size}.txt').write_bytes(ctx.token_bytes(case['tokens']))
        row['cases'] = cases(row, original)
        for case in row['cases']:
            for folder in ('tmp', 'cache'):
                (Path(case['root']) / folder).mkdir(mode=0o700, parents=True)
        ctx.save(row)

    def current(row):
        values, original, _ = fixed()
        require(basis.validation.committed(bank.RESULT, (KEY,)) == basis.compact(row)
                and all(row[k] == v for k, v in values.items()) and row['cases'] == cases(row, original),
                'commit the unchanged build, case order and workload')
        recovered = bank.stage.source.blobs(bank.P, row['source_commit'], row['sources'])
        require(bank.stage.hashed(dict(recovered)) == row['sources'], 'the original capacity controller differs')
        for case in row['cases']:
            require(Path(case['fixture']['path']).read_bytes() == ctx.token_bytes(case['tokens']), 'standalone token body changed')
        basis.comparison.shared.models.model_pins(row['configuration'], row['model_assets'])

    ctx.environment, ctx.evidence, ctx.prepare, ctx.current = environment, evidence, prepare, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'host', *ORDER),
            'use the pinned interpreter and one physical test stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        ctx.prepare() if action == 'prepare' else ctx.run(action)
