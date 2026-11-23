"""Check physical context on the coherent KV-upload engine with both attention stores."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_kv_upload_compare as upload

bank, basis, require = upload.bank, upload.basis, upload.require
model, state = upload.gather.model, upload.gather.state
capacity = state.restore.capacity
TIMING = 'be8caf6138b633d23df45cf0ee9ca10480589bcb'
KEY = 'prefill_kv_upload_capacity_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PLAN = dict(order=list(capacity.ORDER),
            unit='Select copy or direct attention output on the same standalone native build. '
                 'Use the standalone bank and host tail, with no server cache loan or split scratch.',
            gates='Host bounds, matched short output, then both checked physical final slots. '
                  'Check actual placement, BF16 store destination and aligned KV reuse. Stop on first failure.',
            scope='Standalone physical fallback and short handoff. Server validation follows separately.')


def prerequisites(comparison):
    native = bank.previous.read(upload.build.ROOT / 'record.json')
    upload.build.context().check(native)
    timing = bank.previous.historic(TIMING, comparison.KEY)
    live = bank.previous.read(comparison.ROOT / 'record.json')
    require(upload.gather.closed(timing) and timing == basis.compact(live)
            and timing['decision'] == comparison.decision(timing['processes'])
            and timing['decision']['repeated_warm_gain'] and timing['binary'] == native['binaries']['strata']
            and timing['native_source_commit'] == native['native_source_commit']
            and timing['driver'] == native['driver']
            and timing['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'recover the complete same-build direct-store timing and runtime')
    manifest = live['runtime_manifest']
    providers = bank.previous.read(Path(manifest['path']))
    require(bank.sha(manifest['path']) == manifest['sha256'] and len(providers) == manifest['files']
            and all(bank.sha(p) == h for p, h in providers.items()), 'runtime providers changed')
    basis.comparison.shared.models.model_pins(live['configuration'], live['model_assets'], full=False)

    earlier = basis.validation.committed(bank.RESULT, ('prefill_owned_kv_capacity_v1',))
    full = earlier['outcomes']['full-reuse']
    require(upload.gather.closed(earlier) and full['final_slot_consumed']
            and full['batched_rows'] == capacity.CAPACITY - 1 and full['generated_ids'] == 0
            and earlier['native_source_commit'] == native['projection']['parent_source']
            and all(earlier[k] == timing[k] for k in ('configuration', 'driver', 'boot',
                    'native_validation_probe', 'model_assets_sha256')), 'recover the matched physical fixture')
    proof = dict(timing_commit=TIMING, timing_sha256=bank.signature(timing), previous_sha256=bank.signature(earlier),
                 build_sha256=bank.signature(bank.builder.compact(native)))
    return native, live, earlier, proof


def experiment():
    ctx = basis.core.source.device.load_module('kv_upload_capacity', capacity.__file__)
    ctx.ROOT, ctx.KEY = ROOT, KEY
    comparison = upload.experiment('store-time')
    old_environment, old_evidence = ctx.environment, ctx.evidence

    def environment(row, case):
        env = old_environment(row, case)
        env.update({model.INPUT: '0', state.GATHER: '0', state.KV: '1', upload.previous.UPLOAD: 'batched',
                    'STRATA_SYCL_PREFILL_PREFIX_EXPERT_UPLOAD': 'off', 'STRATA_SYCL_PREFILL_PREFIX_EXPERT_COPY': 'sycl',
                    'STRATA_SYCL_PREFILL_PREFIX_EXPERT_DEFER_COMPLETION': '0',
                    'STRATA_SYCL_KV_COPY_GROUPS': '0', 'STRATA_SYCL_PREFILL_SPLIT_SCRATCH_CHECK_ONCE': '0'})
        return env

    def evidence(stdout, stderr, row, case, pid):
        value = old_evidence(stdout, stderr, row, case, pid)
        require(value['window']['mode'] == (2 if case['full'] else 10)
                and not any(label in stderr for label in ('strata MoE prefill gather:',
                    'strata MoE mixed banks: storage=scratch ', 'strata MoE prefix expert upload',
                    'strata MoE split expert upload')),
                'unexpected standalone scratch or expert upload route')
        geometry = dict(pieces=value['store']['pieces'], gpu_rows=value['gpu_rows'],
                        maximum_rows=value['window']['maximum_rows'])
        value['attention_input'] = model.input_receipt(stderr, value['batched_rows'], geometry, False)
        value['kv'] = state.kv.check_request(stderr, 0, value['batched_rows'], ctx.CHUNK, reuse=True)
        if case['direct']:
            control = row['outcomes'][case['name'].replace('-direct', '-copy')]
            require(all(value[k] == control[k] for k in ('attention_input', 'kv')),
                    'attention input or KV path differs between store modes')
        return value

    def pins(revision):
        return (comparison.pins(revision) | ctx.source_pins(revision)
                | bank.stage.source.pin(bank.P, [THIS])['source_sha256'])

    def cases(row, earlier):
        grid = [ctx.make_case(row, name) for name in ctx.ORDER]
        fixtures = {c['tokens']: c['fixture']['sha256'] for c in earlier['cases']}
        for c in grid:
            require(c['fixture']['sha256'] == fixtures[c['tokens']]
                    and Path(c['fixture']['path']).read_bytes() == ctx.token_bytes(c['tokens']),
                    'physical token body changed')
        for copy, direct in zip(grid[::2], grid[1::2], strict=True):
            common = [{k: v.replace(c['root'], '<case>') for k, v in environment(row, c).items()
                       if k != ctx.SELECTOR} for c in (copy, direct)]
            require(common[0] == common[1], 'another setting changed between store modes')
        return grid

    def prepare():
        require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT),
                'inspect any previous capacity attempt first')
        native, speed, earlier, proof = prerequisites(comparison)
        revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
        row = {k: speed[k] for k in model.FIELDS}
        row.update(stage='prepared', source_commit=revision, sources=pins(revision), qualification=proof,
                   native_source_commit=native['native_source_commit'], vocab_size=earlier['vocab_size'],
                   binary=native['binaries']['strata_full_context_diagnostic'],
                   host_binary=native['binaries']['full_context_diagnostic_host_test'],
                   outcomes={}, complete=False, passed=False, active=False, ownership_closed=True,
                   budgets=ctx.BUDGET, plan=PLAN)
        ROOT.mkdir(mode=0o700)
        (ROOT / 'host').mkdir(mode=0o700)
        for size, length in (('short', 16385), ('full', ctx.CAPACITY)):
            (ROOT / ('tokens-' + size + '.txt')).write_bytes(ctx.token_bytes(length))
        row['cases'] = cases(row, earlier)
        for c in row['cases']:
            for folder in ('tmp', 'cache'):
                (Path(c['root']) / folder).mkdir(mode=0o700, parents=True)
        ctx.save(row)

    def current(row):
        native, speed, earlier, proof = prerequisites(comparison)
        require(basis.validation.committed(bank.RESULT, (KEY,)) == basis.compact(row)
                and row['sources'] == pins(row['source_commit'])
                and row['qualification'] == proof and row['native_source_commit'] == native['native_source_commit']
                and row['binary'] == native['binaries']['strata_full_context_diagnostic']
                and row['host_binary'] == native['binaries']['full_context_diagnostic_host_test']
                and row['vocab_size'] == earlier['vocab_size'] and row['plan'] == PLAN and row['budgets'] == ctx.BUDGET
                and all(row[k] == speed[k] for k in model.FIELDS) and row['cases'] == cases(row, earlier),
                'frozen physical-context inputs changed')

    ctx.environment, ctx.evidence, ctx.prepare, ctx.current = environment, evidence, prepare, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'host', *capacity.ORDER),
            'select a capacity stage with the pinned benchmark interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        control = experiment()
        if action == 'prepare':
            control.prepare()
        else:
            control.run(action)
