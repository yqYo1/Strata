"""Qualify the DFlash2 tap observer, then compare the two long-prefill writers."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_window_bank as extent
import dflash2_tap_payload_build as build
import dflash2_tap_payloads as taps

bank, basis, require = extent.bank, extent.basis, extent.require
window = extent.window
KEY = 'dflash2_tap_payload_model_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
READER = str(Path(taps.__file__).resolve().relative_to(bank.P))
ORDER = ('smoke', 'copy', 'direct')
AUDIT = 'STRATA_DFLASH2_TAP_PAYLOAD_AUDIT'
WRITER = window.store.SELECTOR
WORK = ('requested_bank_H2D_bytes', 'uploads', 'jobs', 'dequants', 'windows', 'maximum_rows')


def inputs(revision):
    original, env, requests, reference, _ = extent.inputs(revision, extent.LONG_READ, ROOT)
    native = bank.previous.read(build.ROOT / 'record.json')
    build.context().check(native)
    require(native['stage'] == 'built' and native['complete'] and native['passed']
            and native['ownership_closed'] and not native['active'] and native['host_control']['passed']
            and native['projection']['tap_parent_prefill_evidence'] == build.SPEED
            and original['driver'] == native['driver'], 'complete the same-build tap observer host gate')
    binary = native['binaries']['strata']
    require(taps.PROMPTS == tuple(len(ids) for ids, _ in requests) and AUDIT not in env
            and env[WRITER] == env[window.store.ATTENTION_READER] == '0',
            'unexpected original fixture or attention policy')
    controller_paths = {THIS, READER, build.THIS}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and (path := Path(filename).resolve()).is_relative_to(bank.P / 'bench' / 'sycl'):
            controller_paths.add(path.relative_to(bank.P).as_posix())
    pins = bank.stage.source.pin(bank.P, [*sorted(controller_paths), 'tools/sycl', *bank.basis.POLICIES])
    require(pins['source_sha256'] == bank.stage.hashed(dict(bank.stage.source.blobs(
                bank.P, revision, pins['source_sha256'])))
            and (bank.P / READER).read_bytes() == basis.validation.git('show', build.OBSERVER + ':' + READER),
            'the observer or its committed protocol readers changed')

    old_case = original['cases'][extent.CASE]
    require(old_case['command'][0] == original['binary']['path'], 'unexpected executable in the traced fixture')
    cases, launches = {}, {}
    for name in ORDER:
        settings = {k: v.replace(str(ROOT / extent.CASE), str(ROOT / name)) for k, v in env.items()}
        settings.update({AUDIT: '1', WRITER: str(int(name == 'direct'))})
        wire = old_case['wire'][:1] if name == 'smoke' else old_case['wire']
        limits = (dict(original['limits'], wall=300, cpu=1200, text_cap=4 << 20)
                  if name == 'smoke' else original['limits'])
        cases[name] = dict(old_case, command=[binary['path'], *old_case['command'][1:]], wire=wire,
                           environment_sha256=bank.signature(settings), limits=limits)
        launches[name] = settings
    normalized = [{k: v.replace(str(ROOT / name), '<case>') for k, v in launches[name].items() if k != WRITER}
                  for name in ORDER]
    require(normalized[0] == normalized[1] == normalized[2], 'another native input changes between cases')
    plan = dict(original, cases=cases, order=list(ORDER), binary=binary,
                native_source_commit=native['native_source_commit'],
                native_build_sha256=bank.signature(build.builder.compact(native)),
                sources_count=len(pins['source_sha256']), sources_sha256=pins['aggregate_sha256'],
                scope='First verify the observer for one fresh 82-input request. Then compare the copy and direct '
                      'writers in fresh processes on the same 82/122881/82 sequence. Require equal tap bytes by layer '
                      'digest, equal request and target-state evidence, and an intact cache refill. The observer is charged.')
    return plan, launches, requests, reference


def inspect(log, pid, responses, plan, name, reference):
    case = plan['cases'][name]
    records = taps.read_fresh_tap_payloads(log, tuple(item['read'] for item in case['wire']))
    for record in records:
        record['layer_digests'] = {str(layer): digest for layer, digest in record['layer_digests'].items()}
    if name == 'smoke':
        proof = basis.validation.preload_evidence(log, pid)
        request, = responses
        require(proof['loader'] == plan['validation']['loader']['path']
                and not re.search(r'(?m)^strata mtp: .*loaded|UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain', log)
                and request['prompt'] == 82 and request['offered'] > 0, 'invalid 82-input tap smoke')
        evidence = dict(validation=proof, states=[window.split.committed_state(log, request)])
        window.store.speed.verify_model_assets(reference, {'args': case['command'][2:]})
    else:
        require(records[0] == records[-1], 'the repeated fresh short prompt produced different tap contents')
        geometry = dict(plan, cases={extent.CASE: case})
        evidence = extent.inspect(log, geometry, pid, responses, extent.LONG_READ, reference, direct=name == 'direct')
    return dict(evidence, tap_payloads=records)


def comparison(outcomes):
    copy, direct = (outcomes[name] for name in ORDER[1:])
    differences = [dict(prompt=ref['prompt'], index=i, layers=[layer for layer in ref['layer_digests']
                    if ref['layer_digests'][layer] != seen['layer_digests'][layer]])
                   for i, (ref, seen) in enumerate(zip(copy['tap_payloads'], direct['tap_payloads'], strict=True))
                   if ref != seen]
    identities = {(result['native_identity']['pid'], result['native_identity']['start_ticks'])
                  for result in outcomes.values()}
    return dict(independent_native_processes=len(identities) == len(ORDER), tap_differences=differences,
        matched_state=all(copy[key] == direct[key] for key in ('requests', 'states', 'info', 'loan', 'kv_stage',
            'tap_rows', 'packed_layers_sha256', 'usable_slots_by_layer', 'first_verifier_window')),
        matched_work=all(copy['route'][key] == direct['route'][key] for key in WORK),
        correct_writer=copy['route']['logical_copy_bytes'] == 48 * (extent.LONG_READ - 1) * window.workspace.ROW
                       and direct['route']['logical_copy_bytes'] == 0)


def save(row):
    window.store.gate.save(row, ROOT, KEY)


def main(action):
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned interpreter with prepare, smoke, copy or direct')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'fresh tap plan required')
            revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
            plan, _, _, _ = inputs(revision)
            for name in ORDER:
                for directory in ('cache', 'tmp'):
                    (ROOT / name / directory).mkdir(mode=0o700, parents=True)
            save(dict(plan=plan, stage='prepared', outcomes={}, active=False, complete=False, passed=False,
                      ownership_closed=True, speed_eligible=False))
            return

        row = bank.previous.read(ROOT / 'record.json')
        root, earlier = ROOT / action, ORDER[:ORDER.index(action)]
        require(row == basis.validation.committed(bank.RESULT, (KEY,)) and row['ownership_closed']
                and not row['active'] and tuple(row['outcomes']) == earlier
                and row['stage'] == ('partial' if earlier else 'prepared')
                and all(old['passed'] for old in row['outcomes'].values())
                and not (root / 'owned').exists() and not (root / 'execution.json').exists(),
                'require the committed plan and unused process in the declared order')
        plan, launches, requests, reference = inputs(row['plan']['source_commit'])
        require(plan == row['plan'], 'frozen tap observer inputs changed')
        outcome = window.device.run_case(plan, root, action, launches[action],
            requests[:1] if action == 'smoke' else requests,
            lambda log, pid, responses: inspect(log, pid, responses, plan, action, reference),
            limits=plan['cases'][action]['limits'])
        row['outcomes'][action] = outcome
        if action == 'copy' and outcome['passed']:
            smoke = row['outcomes']['smoke']
            if any(outcome[key][:1] != smoke[key] for key in ('requests', 'states', 'tap_payloads')):
                outcome.update(passed=False, error='copy process did not reproduce the fresh smoke request')
        if action == 'direct' and outcome['passed']:
            result = row['comparison'] = comparison(row['outcomes'])
            if result['tap_differences'] or not all(value for key, value in result.items() if key != 'tap_differences'):
                outcome.update(passed=False, error='observed tap payload, target state, work, writer or owner differs')
        complete = outcome['passed'] and action == 'direct'
        stage = ('closed' if complete else 'partial') if outcome['passed'] else 'failed'
        row.update(stage=stage, passed=complete, complete=complete, active=not outcome['ownership_closed'],
                   ownership_closed=outcome['ownership_closed'])
        save(row)
        require(outcome['passed'], outcome.get('error') or 'tap observer case failed')


if __name__ == '__main__':
    main(*sys.argv[1:])
