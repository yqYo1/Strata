"""Check the real width-five Q6 head dispatch before model speed runs."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_c5_head_build as build
import dflash2_width_probe as width_probe

bank, basis, model = width_probe.bank, width_probe.basis, width_probe.model
native, audit = width_probe.native, width_probe.audit
require = width_probe.require
KEY = 'dflash2_c5_head_admission_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
FLAG = 'STRATA_SYCL_DECODE_Q6_DPAS_C5_HEAD'
ORDER = ('reference', 'c5')
DECODE_C5 = re.compile(r'(?m)^strata verify: phase=decode tokens=5 q6_reference_calls=(\d+) q6_columns_calls=(\d+)$')


def inputs(revision):
    admitted, environment = width_probe.inputs(revision)
    compiled = bank.previous.read(build.ROOT / 'record.json')
    build.context().check(compiled)
    require(compiled['stage'] == 'built' and compiled['passed'] and compiled['ownership_closed']
            and compiled['host_control']['passed'] and compiled['head_policy_control']['passed']
            and compiled['head_policy_control']['ownership_closed'], 'verify the complete C5 build and host controls')
    binary = compiled['binaries']['strata']
    sources = {THIS, build.THIS, str(Path(width_probe.__file__).resolve().relative_to(bank.P)),
               str(Path(audit.__file__).resolve().relative_to(bank.P))}
    pins = bank.stage.hashed(dict(bank.stage.source.blobs(bank.P, revision, sorted(sources))))
    require(all(bank.sha(bank.P / name) == digest for name, digest in pins.items()),
            'the committed C5 probe and readers changed')

    base_command = admitted['cases']['5']['command']
    require(base_command[0] == admitted['binary']['path'] and FLAG not in environment['5'],
            'the width-five predecessor must use the ordinary head')
    cases, launches = {}, {}
    old_root = str(width_probe.ROOT / '5')
    for arm in ORDER:
        path = str(ROOT / arm)
        env = {k: v.replace(old_root, path) for k, v in environment['5'].items()}
        env[FLAG] = str(int(arm == 'c5'))
        require(env['STRATA_SYCL_DECODE_Q6_DPAS_HEAD'] == '1'
                and env['STRATA_SYCL_Q6_ROUTE_AUDIT'] == '1'
                and env['STRATA_SYCL_CACHE_SCRATCH_LEASE'] == '0'
                and env['STRATA_COMMIT_SYNC'] == '1' and old_root not in str(env),
                'the width-five certified runtime settings changed')
        cases[arm] = dict(command=[binary['path'], *base_command[1:]], info=admitted['cases']['5']['info'],
                          environment_sha256=bank.signature(env))
        launches[arm] = env
    normalized = [dict((k, v.replace(str(ROOT / arm), '<case>')) for k, v in launches[arm].items() if k != FLAG)
                  for arm in ORDER]
    require(normalized[0] == normalized[1], 'another DFlash2 probe setting differs')
    plan = {k: admitted[k] for k in ('boot', 'driver', 'runtime_manifest', 'model_assets_sha256',
                                     'validation', 'drafter_assets', 'request', 'native_limits')}
    plan.update(source_commit=revision, sources=pins, predecessor_inputs=admitted['inputs'],
                binary=binary, native_source_commit=compiled['native_source_commit'],
                native_build_sha256=bank.signature(build.builder.compact(compiled)),
                cases=cases, order=list(ORDER),
                scope='Two independent, parameter-validated 82-input and 32-output DFlash2 processes. '
                      'Use identical width-five inputs and placement. The Q6 head must use canonical DPAS '
                      'only on decode width five in the candidate. Check the actual replay counts and all '
                      'fallback routes. Compare output hashes as a diagnostic and keep timing out of the decision.')
    return plan, launches


def inspect_case(text, plan, pid, requests, responses, enabled):
    proof = basis.validation.preload_evidence(text, pid)
    require(proof['loader'] == plan['validation']['loader']['path']
            and text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
            and not re.search(r'(?m)^strata mtp: .*loaded|^strata cache loan: phase=|'
                              r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain', text),
            'unexpected provider, head setup, borrowing or MTP weights')
    taps = re.findall(r'(?m)^strata DFlash2: eager, prompt_tail_rows=81 cold_prefix=0 vram_bytes=([1-9]\d*)$', text)
    ready = re.findall(r'(?m)^strata verify: window up to 5 tokens, ([\d.]+) MiB of device buffers$', text)
    require(len(taps) == len(ready) == 1, 'missing the real width-five verifier or DFlash2 taps')
    sample, = audit.audit_requests(text, requests, responses, width=5)
    require(sample['width_counts'].keys() == {1, 5}, 'the native request must execute both width-one and width-five graphs')
    (reference, columns), = DECODE_C5.findall(text)
    return dict(validation=proof, window_work=sample,
                head_captures=audit.audit_head(text, sample['width_counts'], c5=enabled),
                capture_counts=dict(reference=int(reference), columns=int(columns)),
                draft_vram_bytes=int(taps[0]), verifier_buffer_mib=float(ready[0]))


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def main(action):
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned interpreter with prepare, reference or c5')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'fresh C5 probe required')
            revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
            plan, _ = inputs(revision)
            for case in ORDER:
                for name in ('tmp', 'cache'):
                    (ROOT / case / name).mkdir(parents=True, mode=0o700)
            save(dict(stage='prepared', plan=plan, outcomes={}, complete=False, passed=False, ownership_closed=True))
            return

        row = bank.previous.read(ROOT / 'record.json')
        require(row == basis.validation.committed(bank.RESULT, (KEY,)), 'commit the C5 probe plan and preceding outcome')
        plan, launches = inputs(row['plan']['source_commit'])
        path = ROOT / action
        require(row['plan'] == plan and row['ownership_closed']
                and tuple(row['outcomes']) == ORDER[:ORDER.index(action)]
                and all(r['passed'] and r['ownership_closed'] for r in row['outcomes'].values())
                and not (path / 'owned').exists() and not (path / 'execution.json').exists(),
                'C5 probe settings or process order changed')
        requests = [(plan['request']['ids'], plan['request']['output'])]
        outcome = native.run_case(plan, path, action, launches[action], requests,
            lambda text, pid, responses: inspect_case(text, plan, pid, requests, responses, action == 'c5'),
            limits=plan['native_limits'], expected_info=plan['cases'][action]['info'])
        row['outcomes'][action] = outcome
        passed = outcome['passed'] and (action != 'c5' or all(
            outcome[k] == row['outcomes']['reference'][k] for k in ('draft_vram_bytes', 'verifier_buffer_mib')))
        if passed and action == 'c5':
            old = row['outcomes']['reference']['capture_counts']
            passed = outcome['capture_counts'] == dict(old, reference=old['reference'] - 1)
        complete = passed and action == 'c5'
        row.update(stage=('closed' if complete else 'partial') if passed else 'failed', passed=complete,
                   complete=complete, ownership_closed=outcome['ownership_closed'])
        save(row)
        print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'head_captures', 'error')}, flush=True)
        require(passed, outcome.get('error') or 'width-five model or placement check failed')


if __name__ == '__main__':
    main(*sys.argv[1:])
