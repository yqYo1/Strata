"""Validate the native width-five dense route with its head held fixed."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_c5_dense_build as build
import dflash2_c5_head_probe as head

bank, basis, require = head.bank, head.basis, head.require
KEY = 'dflash2_c5_dense_admission_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
FLAG = 'STRATA_SYCL_DECODE_Q6_DPAS_C5_10240'
ORDER = ('reference', 'dense')
SHAPE = '2560x10240'
CALLS = 22
DENSE = re.compile(r'strata Q6_DPAS_DENSE request=1 device=0 phase=decode tokens=5 capture_id=(\d+) '
                   r'ni=2560 no=10240 nc=5 layout=canonical extra_weight_bytes=0 '
                   r'canonical_table_excludes_dpas=1 calls=(\d+) complete=1')


def inputs(revision):
    upstream, environment = head.width_probe.inputs(revision)
    compiled = bank.previous.read(build.ROOT / 'record.json')
    build.context().check(compiled)
    policy = compiled['dense_policy_control']
    require(compiled['host_control']['passed'] and policy['passed'] and policy['ownership_closed']
            and policy['exit_code'] == 0 and not policy['GPU_executed']
            and policy['binary'] == compiled['binaries'][build.head.POLICY_TARGET],
            'require both same-build host gates')
    binary = compiled['binaries']['strata']
    sources = {THIS, build.THIS, head.THIS, str(Path(head.width_probe.__file__).resolve().relative_to(bank.P)),
               str(Path(head.audit.__file__).resolve().relative_to(bank.P))}
    pins = bank.stage.hashed(dict(bank.stage.source.blobs(bank.P, revision, sorted(sources))))
    require(all(bank.sha(bank.P / name) == digest for name, digest in pins.items()),
            'the committed dense probe and readers changed')
    previous = upstream['cases']['5']
    require(previous['command'][0] == upstream['binary']['path'] and
            not {FLAG, head.FLAG}.intersection(environment['5']), 'unexpected width-five predecessor')
    cases, launches = {}, {}
    original = str(head.width_probe.ROOT / '5')
    for arm in ORDER:
        root = str(ROOT / arm)
        env = {k: v.replace(original, root) for k, v in environment['5'].items()}
        env.update({head.FLAG: '1', FLAG: str(int(arm == 'dense'))})
        cases[arm] = dict(command=[binary['path'], *previous['command'][1:]], info=previous['info'],
                          environment_sha256=bank.signature(env))
        launches[arm] = env
    normalized = [{k: v.replace(str(ROOT / arm), '<case>') for k, v in launches[arm].items() if k != FLAG}
                  for arm in ORDER]
    require(normalized[0] == normalized[1] and all(
        env[head.FLAG] == env['STRATA_SYCL_Q6_ROUTE_AUDIT'] == '1' for env in launches.values()),
        'the dense selector must be the only independent setting')
    plan = {k: upstream[k] for k in ('boot', 'driver', 'runtime_manifest', 'model_assets_sha256',
                                     'validation', 'drafter_assets', 'request', 'native_limits')}
    plan.update(source_commit=revision, sources=pins, predecessor_inputs=upstream['inputs'],
                binary=binary, native_source_commit=compiled['native_source_commit'],
                native_build_sha256=bank.signature(build.builder.compact(compiled)),
                cases=cases, order=list(ORDER),
                scope='Independent 82-input, 32-output, width-five requests with identical head settings. '
                      'Move exactly 22 canonical Q6 2560x10240 calls to DPAS. Require unchanged verifier '
                      'work, token hashes and placement, checked closure and zero fresh Xe faults.')
    return plan, launches


def inspect(text, plan, pid, requests, responses, enabled):
    result = head.inspect_case(text, plan, pid, requests, responses, True)
    for name in ('width_counts', 'accepted_prefix_counts'):
        work = result['window_work']
        work[name] = {str(width): count for width, count in work[name].items()}
    actual = []
    for line in text.splitlines():
        if line.startswith('strata Q6_DPAS_DENSE ') and ' nc=5 ' in line:
            match = DENSE.fullmatch(line)
            require(match is not None, 'dense projection escaped its exact shape, phase or width')
            actual.append(tuple(map(int, match.groups())))
    capture = result['head_captures'][-1]['capture_id']
    require(actual == ([(capture, CALLS)] if enabled else []), 'exact dense capture count differs')
    require(result['dense_reference_calls'].get(SHAPE, 0) == (0 if enabled else CALLS),
            'ordinary reference calls did not move exactly once')
    result['dense_dpas_calls'] = sum(calls for _, calls in actual)
    return result


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def main(action):
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use pinned Python with prepare, reference or dense')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT),
                    'fresh dense model probe required')
            revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
            plan, _ = inputs(revision)
            for arm in ORDER:
                for name in ('tmp', 'cache'):
                    (ROOT / arm / name).mkdir(parents=True, mode=0o700)
            save(dict(stage='prepared', plan=plan, outcomes={}, complete=False, passed=False, ownership_closed=True))
            return
        row = bank.previous.read(ROOT / 'record.json')
        require(row == basis.validation.committed(bank.RESULT, (KEY,)), 'commit the preceding dense probe stage')
        plan, launches = inputs(row['plan']['source_commit'])
        path = ROOT / action
        require(row['plan'] == plan and row['ownership_closed']
                and tuple(row['outcomes']) == ORDER[:ORDER.index(action)]
                and all(r['passed'] and r['ownership_closed'] for r in row['outcomes'].values())
                and not (path / 'owned').exists() and not (path / 'execution.json').exists(),
                'preserve the dense probe order and inputs')
        requests = [(plan['request']['ids'], plan['request']['output'])]
        outcome = head.native.run_case(plan, path, action, launches[action], requests,
            lambda text, pid, responses: inspect(text, plan, pid, requests, responses, action == 'dense'),
            limits=plan['native_limits'], expected_info=plan['cases'][action]['info'])
        row['outcomes'][action] = outcome
        passed = outcome['passed']
        if passed and action == 'dense':
            ref = row['outcomes']['reference']
            passed = (all(outcome[k] == ref[k] for k in
                ('requests', 'window_work', 'head_captures', 'draft_vram_bytes', 'verifier_buffer_mib'))
                and outcome['capture_counts'] == dict(ref['capture_counts'],
                    reference=ref['capture_counts']['reference'] - CALLS)
                and {k: v for k, v in outcome['dense_reference_calls'].items() if k != SHAPE} ==
                    {k: v for k, v in ref['dense_reference_calls'].items() if k != SHAPE})
        complete = passed and action == 'dense'
        row.update(stage=('closed' if complete else 'partial') if passed else 'failed',
                   passed=complete, complete=complete, ownership_closed=outcome['ownership_closed'])
        save(row)
        print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'dense_dpas_calls', 'fresh_faults', 'error')})
        require(passed, outcome.get('error') or 'dense route, output or placement changed')


if __name__ == '__main__':
    main(*sys.argv[1:])
