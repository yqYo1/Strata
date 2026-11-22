"""Check exact-fit generation and physical fill on the coherently fixed MTP CLI."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import decode_mtp_context_reference as reference
import decode_mtp_prefix_build as build

bank, basis, require = reference.bank, reference.basis, reference.require
previous, capacity = reference.previous, reference.capacity
ROOT = bank.B / 'decode-mtp-context-fixed-v2'
KEY = 'decode_mtp_context_fixed_v2'
FIRST_TRIAL = '0a6d935887fc05a0a0f34600884fd7165f364053'
FIRST_KEY = 'decode_mtp_context_fixed_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('near-four', 'full')
GEOMETRY = {
    'near-four': (capacity.CAPACITY, capacity.CAPACITY - 2, 2, 4),
    'full': (capacity.CAPACITY, capacity.CAPACITY, 0, 4),
}
PLAN = dict(order=list(ORDER), geometry={k: list(v) for k, v in GEOMETRY.items()},
    prerequisite='Reuse the closed same-build width-four smoke and its original long-case inputs. '
                 'The first trial stopped at the packed head width-eight initialization refusal.',
    scope='The fixed production CLI handles the width-four 262142-token exact-fit request. The linked '
          'diagnostic consumes the physical final input with zero output. Keep the independent '
          'MTP prefix selector off.',
    acceptance='Two completed verifier windows with exactly one offered draft establish the short '
               'tail width. Check the observed first catch-up, emitted count, admitted extent, '
               'provider, ownership and faults. On the exact fill require the same physical work '
               'and final marker as the original. Stop on the first mismatch.')
SELECTOR = 'STRATA_SYCL_MTP_DECODE_COMMITTED_PREFIX'


def first_trial():
    return bank.previous.historic(FIRST_TRIAL, FIRST_KEY)


def short_control(values):
    trial = first_trial()
    host, small, rejected = (trial['outcomes'][key] for key in ('host', 'small-four', 'small-eight'))
    require(trial['stage'] == 'failed' and trial['ownership_closed'] and not trial['active']
            and not trial['complete'] and not trial['passed'] and trial['budgets'] == capacity.BUDGET
            and host['passed'] and not host['GPU_executed']
            and small['passed'] and small['complete'] and small['GPU_executed']
            and small['ownership_closed'] and small['native_exit_code'] == 0 and not small['fresh_faults']
            and (small['batched_rows'], small['outputs'], small['last_width']) == (reference.TOKENS - 1, 2, 2)
            and not rejected['passed'] and rejected['native_exit_code'] == 1
            and rejected['ownership_closed'] and not rejected['fresh_faults'],
            'retain the original closed width-four pass and width-eight refusal')
    require(all(trial[k] == v for k, v in values.items() if k not in ('environment_template', 'model_assets'))
            and all(trial[k + '_sha256'] == bank.signature(values[k])
                    for k in ('environment_template', 'model_assets')),
            'the candidate, providers or payloads differ from the width-four smoke')
    return dict(revision=FIRST_TRIAL, evidence_key=FIRST_KEY, case='small-four',
                outcome_sha256=bank.signature(small))


def inputs():
    candidate = bank.previous.read(build.ROOT / 'record.json')
    build.check(candidate)
    require(candidate['stage'] == 'built' and candidate['passed'] and candidate['complete']
            and not candidate['active'] and candidate['ownership_closed'], 'complete the exact build first')
    old = bank.basis.validation.committed(bank.RESULT, (previous.KEY,))
    prior = old['outcomes']['full-reuse']
    require(old['stage'] == 'closed' and old['native_source_commit'] == reference.SOURCE
            and prior['passed'] and prior['ownership_closed'] and prior['native_exit_code'] == 0
            and prior['final_slot_consumed'] and not prior['fresh_faults'], 'require original exact-fill evidence')
    values = reference.inputs()
    require(values['binary'] == candidate['projection']['parent_binary'], 'original reference changed')
    values.update(binary=candidate['binaries']['strata'],
        diagnostic=candidate['binaries']['strata_full_context_diagnostic'],
        host_binary=candidate['binaries']['full_context_diagnostic_host_test'],
        native_source_commit=candidate['native_source_commit'], vocab_size=old['vocab_size'],
        build_sha256=bank.signature(build.builder.compact(candidate)),
        owner_reference_sha256=build.reference_gate(),
        full_reference={key: prior[key] for key in (*previous.MATCH, 'kv')})
    values['short_reference'] = short_control(values)
    return values


def environment(row, case):
    name = 'full-reuse' if case['full'] else 'short-reuse'
    env = previous.experiment().environment(row, dict(case, name=name))
    require(SELECTOR not in env and env['STRATA_SYCL_DECODE_Q6_DPAS_HEAD'] == '2',
            'inherit the original packed-head width-four environment')
    return dict(env, **{SELECTOR: '0'})


def make_case(row, name):
    context, tokens, outputs, spec = GEOMETRY[name]
    full = name == 'full'
    fixture = ROOT / f'tokens-{tokens}.txt'
    argv = reference.command(dict(row, binary=row['diagnostic'] if full else row['binary']))
    for flag, value in (('--max-context', context), ('--spec', spec), ('--tokens-file', fixture),
                        ('--max-new', outputs)):
        argv[argv.index(flag)+1] = str(value)
    if full:
        argv.append('--consume-prompt')
    case = dict(name=name, full=full, direct=False, tokens=tokens, outputs=outputs, spec=spec,
                root=str(ROOT / name), fixture=dict(path=str(fixture), sha256=bank.sha(fixture)),
                wall=2400 if context == capacity.CAPACITY else 480, command=argv)
    case['environment_sha256'] = bank.signature(environment(row, case))
    return case


def cases(row):
    original = {case['name']: case for case in first_trial()['cases']}
    result = [make_case(row, name) for name in ORDER]
    for case in result:
        prior = original[case['name']]
        require(all(case[k] == prior[k] for k in ('full', 'direct', 'tokens', 'outputs', 'spec', 'wall'))
                and case['fixture']['sha256'] == prior['fixture']['sha256']
                and case['command'] == [case['fixture']['path'] if arg == prior['fixture']['path'] else arg
                                        for arg in prior['command']]
                and bank.signature(environment(row, dict(case, root=prior['root']))) == prior['environment_sha256'],
                'width-four long case changed beyond its private output directory')
    return result


def tail_evidence(stdout, stderr, row, case, pid):
    proof = basis.validation.preload_evidence(stderr, pid)
    count, timing = capacity.one(stdout, r'^prefill\s+(\d+) tokens in ([0-9.]+) ms', 'actual batched extent')
    ids = [int(t) for t in capacity.one(stdout, r'^output\s*:(.*)$', 'output IDs').split()]
    rounds, spec, accepted, offered = map(int, capacity.one(stdout,
        r'^speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+) ', 'actual verifier rounds'))
    require(proof['loader'] == row['native_validation_probe']['loader']['path']
            and int(count) == case['tokens']-1 and float(timing) > 0 and len(ids) == case['outputs']
            and all(0 <= t < row['vocab_size'] for t in ids) and 0 <= accepted <= offered
            and (rounds, spec, offered) == (2, case['spec'], 1)
            and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed|ran out of context', stderr),
            'actual two-output tail or provider evidence differs')
    return dict(validation=proof, batched_rows=int(count), outputs=len(ids), output_ids_sha256=bank.signature(ids),
                verifier_rounds=rounds, last_width=offered+1, accepted=accepted)


def evidence(stdout, stderr, row, case, pid):
    if case['full']:
        value = capacity.evidence(stdout, stderr, row, case, pid)
        counts = value['store']
        geometry = dict(pieces=counts['pieces'], gpu_rows=value['gpu_rows'], maximum_rows=value['window']['maximum_rows'])
        value['attention_input'] = previous.model.input_receipt(stderr, value['batched_rows'], geometry, False)
        value['kv'] = previous.state.kv.check_request(stderr, 0, value['batched_rows'], capacity.CHUNK, reuse=True)
        require(all(value[k] == v for k, v in row['full_reference'].items()), 'the measured exact fill changed')
        return value

    value = tail_evidence(stdout, stderr, row, case, pid)
    p, verified, accepted_first, rows, coupled = map(int, capacity.one(stderr,
        r'^strata mtp: decode catchup p=(\d+) verified=(\d+) accepted=(\d+) rows=(\d+) coupled=(\d+)$',
        'one completed MTP catch-up'))
    require((p, verified, accepted_first, rows, coupled) == (case['tokens']-1, 1, 0, 1, 0),
            'actual first MTP catch-up differs')
    return dict(value, first_catchup_position=p)


def experiment():
    ctx = basis.core.source.device.load_module('mtp_fixed_capacity', capacity.__file__)
    ctx.ROOT, ctx.KEY, ctx.ORDER = ROOT, KEY, ORDER
    ctx.environment, ctx.evidence = environment, evidence

    def prepare():
        require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'fresh fixed-context plan')
        row = inputs()
        controllers = [THIS, build.THIS, reference.THIS, *previous.SOURCES]
        pins = bank.stage.source.pin(bank.P, controllers)
        ROOT.mkdir(mode=0o700)
        (ROOT / 'host').mkdir()
        for n in {item[1] for item in GEOMETRY.values()}:
            (ROOT / f'tokens-{n}.txt').write_bytes(capacity.token_bytes(n))
        planned = cases(row)
        for case in planned:
            for name in ('cache', 'tmp'):
                (Path(case['root']) / name).mkdir(parents=True, mode=0o700)
        row.update(source_commit=pins['revision'], sources=pins['source_sha256'], cases=planned, outcomes={},
                   plan=PLAN, budgets=ctx.BUDGET, stage='prepared', active=False, ownership_closed=True,
                   passed=False, complete=False)
        ctx.save(row)

    def current(row):
        require(basis.validation.committed(bank.RESULT, (KEY,)) == basis.compact(row)
                and row['sources'] == bank.stage.source.pin(bank.P, list(row['sources']))['source_sha256']
                and all(row[k] == v for k, v in inputs().items()) and row['plan'] == PLAN
                and row['budgets'] == ctx.BUDGET and row['cases'] == cases(row),
                'committed build, runtime or case differs')
        for case in row['cases']:
            require(Path(case['fixture']['path']).read_bytes() == capacity.token_bytes(case['tokens']),
                    'fixed context body differs')

    ctx.prepare, ctx.current = prepare, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', 'host', *ORDER) and sys.executable == basis.reference.model.PYTHON,
            'use the pinned interpreter and one fixed-context stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        ctx.prepare() if action == 'prepare' else ctx.run(action)
