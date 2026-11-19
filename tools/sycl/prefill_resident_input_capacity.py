"""Exercise the physical final prompt slot with all four BF16 bank choices."""

import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_ordinary_bank_compare as ordinary
import prefill_resident_input_model as model

build, basis, require = model.build, model.basis, model.require
original = model.restore.capacity
KEY = 'prefill_resident_input_capacity_v1'
ROOT = build.B / 'prefill-resident-input-capacity-v1'
ORDER = tuple(f'{length}-{source}-{store}' for length in ('short','full')
              for source in ('scratch','borrow') for store in ('copy','direct'))


def prerequisites():
    native, speed, identity, device = model.inputs()
    state = basis.validation.committed(build.RESULT, (model.KEY,))
    require(state['passed'] and state['complete'] and state['ownership_closed'] and not state['active']
            and state['native_source_commit'] == build.INPUT_SOURCE and state['build_sha256'] == identity
            and state['device_sha256'] == device and state['binary'] == native['binaries']['strata']
            and list(state['outcomes']) == list(model.ORDER)
            and all(s['passed'] and s['complete'] and s['ownership_closed'] and not s['fresh_faults']
                    and s['native_exit_code'] == 0 for s in state['outcomes'].values()),
            'require eight checked server cases including mixed-residency RESTORE')
    prior = basis.validation.committed(build.RESULT, ('prefill_attention_bank_capacity_v2',))
    require(prior['stage'] == 'closed' and prior['configuration'] == speed['configuration']
            and all(prior['outcomes'][arm]['passed'] and prior['outcomes'][arm]['final_slot_consumed']
                    for arm in ('full-copy','full-direct')), 'require the original standalone final-slot evidence')
    return native, speed, dict(build=identity, state=build.signature(state), capacity=build.signature(prior))


def context():
    # The original runner owns the standalone native process, token fixture,
    # terminal marker and fault checks. Keep its old cohort globals isolated.
    ctx = ordinary.original.core.source.device.load_module('resident_input_capacity', original.__file__)
    ctx.ROOT, ctx.KEY, ctx.ORDER = ROOT, KEY, ORDER
    old_env, old_evidence, host_check = ctx.environment, ctx.evidence, ctx.check_parser

    def environment(row, case):
        env = old_env(row, case)
        env[model.INPUT] = str(int('-borrow-' in case['name']))
        for name, disabled in (('STRATA_SYCL_PREFILL_PREFIX_EXPERT_UPLOAD','off'),
                               ('STRATA_SYCL_PREFILL_PREFIX_EXPERT_COPY','sycl'),
                               ('STRATA_SYCL_PREFILL_PREFIX_EXPERT_DEFER_COMPLETION','0')):
            require(env.get(name, disabled) == disabled, 'unsafe standalone expert copy setting: ' + name)
            env[name] = disabled
        return env

    def evidence(stdout, stderr, row, case, pid):
        value = old_evidence(stdout, stderr, row, case, pid)
        require('strata MoE prefix expert upload' not in stderr, 'unexpected early expert upload')
        geometry = dict(pieces=value['store']['pieces'], gpu_rows=value['gpu_rows'],
                        maximum_rows=value['window']['maximum_rows'])
        value['attention_input'] = model.input_receipt(stderr, value['batched_rows'], geometry,
                                                      '-borrow-' in case['name'])
        control = case['name'].split('-')[0] + '-scratch-copy'
        if case['name'] != control:
            previous = row['outcomes'][control]
            for key in ('batched_rows','generated_ids','output_ids_sha256','final_slot_consumed',
                        'gpu_rows','host_rows','bank_bytes','window','qsa_bytes'):
                require(value[key] == previous[key], 'physical comparison changed ' + key)
        return value

    def check_parser():
        host_check()
        model.check_reader()

    def prepare():
        require(not ROOT.exists() and KEY not in basis.validation.committed(build.RESULT), 'fresh physical cohort required')
        native, speed, proof = prerequisites()
        pin = build.stage.source.pin(build.P, ['tools/sycl', *basis.POLICIES])
        vocab = json.loads((Path(speed['configuration']['tokenizer']) / 'vocab.json').read_bytes())
        require(set(range(4096)) <= set(vocab.values()), 'fixture IDs missing')
        row = {k:speed[k] for k in model.FIELDS}
        row.update(stage='prepared', source_commit=pin['revision'], sources=pin['source_sha256'],
                   native_source_commit=build.INPUT_SOURCE, qualification=proof, vocab_size=max(vocab.values())+1,
                   binary=native['binaries']['strata_full_context_diagnostic'],
                   host_binary=native['binaries']['full_context_diagnostic_host_test'], cases=[], outcomes={},
                   complete=False, passed=False, active=False, ownership_closed=True, budgets=ctx.BUDGET,
                   plan=dict(order=list(ORDER), unit='Compare all four input/output choices. Use ordinary SYCL '
                             'expert copies, including on the standalone 512-slot allocation for a host tail.',
                             extent='First 16385 input/4 output in each arm. Then 262144 input, 262143 batched '
                                    'rows, checked refill of the final slot, zero output in each arm.',
                             acceptance='Verify actual host/device split, layer receipts, QSA scratch, identical '
                                        'work, post-release marker, normal exit and no new driver fault. '
                                        'Stop at first refusal or error. This is standalone capacity evidence.'))
        ROOT.mkdir(mode=0o700)
        (ROOT / 'host').mkdir(mode=0o700)
        for size, extent in (('short',16385), ('full',ctx.CAPACITY)):
            (ROOT / f'tokens-{size}.txt').write_bytes(ctx.token_bytes(extent))
        for name in ORDER:
            case = ctx.make_case(row, name)
            for folder in ('tmp','cache'):
                (Path(case['root']) / folder).mkdir(parents=True, mode=0o700)
            row['cases'].append(case)
        ctx.save(row)

    def current(row):
        native, speed, proof = prerequisites()
        require(basis.validation.committed(build.RESULT, (KEY,)) == basis.compact(row)
                and row['sources'] == build.stage.source.pin(build.P, list(row['sources']))['source_sha256']
                and row['qualification'] == proof and row['native_source_commit'] == build.INPUT_SOURCE
                and all(row[k] == speed[k] for k in model.FIELDS)
                and row['binary'] == native['binaries']['strata_full_context_diagnostic']
                and row['host_binary'] == native['binaries']['full_context_diagnostic_host_test']
                and row['cases'] == [ctx.make_case(row, name) for name in ORDER], 'frozen capacity inputs differ')
        for case in row['cases']:
            require(Path(case['fixture']['path']).read_bytes() == ctx.token_bytes(case['tokens']), 'token body differs')

    ctx.environment, ctx.evidence, ctx.check_parser = environment, evidence, check_parser
    ctx.prepare, ctx.current = prepare, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'host', *ORDER),
            'use the pinned interpreter and a planned physical-context stage')
    with (build.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        ctx = context()
        if action == 'prepare':
            ctx.prepare()
        else:
            ctx.run(action)
