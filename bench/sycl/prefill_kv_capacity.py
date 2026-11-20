"""Check owned prefill KV reuse through the engine's physical final slot."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_kv_gather_dispatch as dispatch

state, bank, model, require = dispatch.state, dispatch.bank, dispatch.model, dispatch.require
basis = bank.basis
capacity = state.restore.capacity
PARENT = '0f5e11521819e332c3ad34a87e4eb59b36b9dec1'
KEY = 'prefill_owned_kv_capacity_v1'
ROOT = bank.B / 'prefill-owned-kv-capacity-v1'
ORDER = ('short-reload', 'short-reuse', 'full-reload', 'full-reuse')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
SOURCES = [THIS, dispatch.THIS, dispatch.split.THIS, dispatch.split.bank.THIS,
           *state.SCRIPTS, 'tools/sycl', *basis.POLICIES]
MATCH = ('generated_ids', 'output_ids_sha256', 'final_slot_consumed', 'batched_rows',
         'gpu_rows', 'host_rows', 'bank_bytes', 'qsa_bytes', 'window', 'store', 'attention_input')


def prerequisites():
    native, speed, build_hash, device_hash = state.inputs()
    traced = basis.validation.committed(bank.RESULT, (dispatch.KEY,))
    require(traced['stage'] == 'closed' and traced['complete'] and traced['passed']
            and traced['ownership_closed'] and not traced['active']
            and traced['binary'] == native['binaries']['strata']
            and traced['native_source_commit'] == native['native_source_commit']
            and len(traced['processes']) == len(dispatch.ORDER)
            and all(p['complete'] and p['passed'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in traced['processes'])
            and traced['decision'] == dict(dispatch_verified=True, requests=16, speed_qualified=False),
            'finish the same-build 64K and 117K server dispatch checks first')
    parent = bank.previous.historic(PARENT, 'prefill_resident_input_capacity_v1')
    original = bank.previous.historic(PARENT, bank.KEY)
    require(parent['stage'] == 'closed' and parent['passed'] and parent['complete']
            and parent['ownership_closed'] and not parent['active']
            and original['binaries']['strata'] == native['projection']['parent_binary']
            and original['binaries']['strata_full_context_diagnostic'] == parent['binary']
            and parent['native_source_commit'] == original['native_source_commit'] == bank.INPUT_SOURCE
            and parent['configuration'] == speed['configuration']
            and parent['model_assets_sha256'] == bank.signature(speed['model_assets'])
            and parent['environment_template_sha256'] == bank.signature(speed['environment_template']),
            'original standalone executable and matched physical workloads required')
    proof = dict(build=build_hash, device=device_hash, dispatch=bank.signature(traced), parent=bank.signature(parent))
    return native, speed, parent, proof


def experiment():
    ctx = basis.core.source.device.load_module('owned_kv_capacity', capacity.__file__)
    ctx.ROOT, ctx.KEY, ctx.ORDER = ROOT, KEY, ORDER
    old_environment, old_evidence = ctx.environment, ctx.evidence

    def environment(row, case):
        env = old_environment(row, case)
        env.update({model.INPUT: '0', state.GATHER: '1', state.KV: str(int(case['name'].endswith('-reuse'))),
                    'STRATA_SYCL_PREFILL_PREFIX_EXPERT_UPLOAD': 'off', 'STRATA_SYCL_PREFILL_PREFIX_EXPERT_COPY': 'sycl',
                    'STRATA_SYCL_PREFILL_PREFIX_EXPERT_DEFER_COMPLETION': '0'})
        require(env['STRATA_SYCL_PREFILL_MIXED_SCRATCH'] == '0' and env[model.OUTPUT] == '0'
                and env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1' and 'STRATA_KV_STAGE_OWN' in env,
                'capacity case requires the standalone fallback and private KV stage')
        return env

    def evidence(stdout, stderr, row, case, pid):
        value = old_evidence(stdout, stderr, row, case, pid)
        geometry = dict(pieces=value['store']['pieces'], gpu_rows=value['gpu_rows'],
                        maximum_rows=value['window']['maximum_rows'])
        value['attention_input'] = model.input_receipt(stderr, value['batched_rows'], geometry, False)
        require(value['window']['mode'] == (2 if case['full'] else 10)
                and 'strata MoE prefill gather:' not in stderr and 'strata MoE mixed banks: storage=scratch ' not in stderr
                and 'strata MoE prefix expert upload' not in stderr,
                'wrong standalone fallback or expert upload route')
        size, arm = case['name'].split('-')
        kv = state.kv.check_request(stderr, 0, value['batched_rows'], ctx.CHUNK, reuse=(arm == 'reuse'))
        require(all(value[k] == row['parent_controls'][size][k] for k in MATCH),
                'physical extent, placement, output or scratch path differs from original')
        if arm == 'reuse':
            control = row['outcomes'][size + '-reload']
            require(all(value[k] == control[k] for k in MATCH) and kv['reused_pages'] > 0
                    and kv['reused_pages'] + kv['reloaded_pages'] == control['kv']['reloaded_pages']
                    and all(kv[k] == control['kv'][k] for k in ('mirror_D2H_bytes', 'mirror_D2H_calls')),
                    'same-build prefix extent, host mirror or standalone outcome differs')
        return dict(value, kv=kv)

    def controls(parent):
        return {size: {k: parent['outcomes'][size + '-scratch-copy'][k] for k in MATCH}
                for size in ('short', 'full')}

    def prepare():
        require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'fresh capacity cohort required')
        native, speed, parent, proof = prerequisites()
        pins = bank.stage.source.pin(bank.P, SOURCES)
        row = {k: speed[k] for k in model.FIELDS}
        row.update(stage='prepared', source_commit=pins['revision'], sources=pins['source_sha256'],
                   native_source_commit=native['native_source_commit'], qualification=proof, parent_controls=controls(parent),
                   vocab_size=parent['vocab_size'], binary=native['binaries']['strata_full_context_diagnostic'],
                   host_binary=native['binaries']['full_context_diagnostic_host_test'], cases=[], outcomes={},
                   complete=False, passed=False, active=False, ownership_closed=True, budgets=ctx.BUDGET,
                   plan=dict(order=ORDER, unit='Change only aligned KV reuse. Request vec8 in both arms; the standalone '
                             'entrypoint has no mixed scratch. Its fallback is a control, not vec8 final-slot evidence.',
                             gates='Compare per-layer KV pages and mirror publication, final-slot consumption, output and '
                             'bank placement with the original and same-build control. Stop on first failure. No speed claim.'))
        ROOT.mkdir(mode=0o700)
        (ROOT / 'host').mkdir(mode=0o700)
        for name in ORDER:
            size = name.split('-')[0]
            old = next(c for c in parent['cases'] if c['name'] == size + '-scratch-copy')
            path = ROOT / ('tokens-' + size + '.txt')
            if not path.exists():
                path.write_bytes(ctx.token_bytes(old['tokens']))
            case = ctx.make_case(row, name)
            old_env = environment(row, dict(case, root=old['root']))
            del old_env[state.KV], old_env[state.GATHER]
            require(case['fixture']['sha256'] == old['fixture']['sha256'] and case['command'][1:] ==
                    [str(path) if arg == old['fixture']['path'] else arg for arg in old['command'][1:]],
                    'standalone input or command differs from the original')
            require(bank.signature(old_env) == old['environment_sha256'],
                    'standalone runtime settings differ beyond the two prefill selectors')
            for folder in ('tmp', 'cache'):
                (Path(case['root']) / folder).mkdir(mode=0o700, parents=True)
            row['cases'].append(case)
        ctx.save(row)

    def current(row):
        native, speed, parent, proof = prerequisites()
        require(basis.validation.committed(bank.RESULT, (KEY,)) == basis.compact(row)
                and row['sources'] == bank.stage.source.pin(bank.P, SOURCES)['source_sha256']
                and row['qualification'] == proof and row['native_source_commit'] == native['native_source_commit']
                and all(row[k] == speed[k] for k in model.FIELDS) and row['parent_controls'] == controls(parent)
                and row['vocab_size'] == parent['vocab_size'] and row['plan']['order'] == list(ORDER)
                and row['binary'] == native['binaries']['strata_full_context_diagnostic']
                and row['host_binary'] == native['binaries']['full_context_diagnostic_host_test']
                and row['cases'] == [ctx.make_case(row, name) for name in ORDER], 'frozen capacity inputs differ')
        for case in row['cases']:
            require(Path(case['fixture']['path']).read_bytes() == ctx.token_bytes(case['tokens']), 'token body changed')
        basis.comparison.shared.models.model_pins(row['configuration'], row['model_assets'])

    ctx.environment, ctx.evidence, ctx.prepare, ctx.current = environment, evidence, prepare, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'host', *ORDER),
            'use the pinned interpreter and a physical-context stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        else:
            ctx.run(action)
