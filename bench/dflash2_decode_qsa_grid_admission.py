"""Check the same-binary DFlash2 scorer dispatch and short-prompt parity."""

import fcntl
from hashlib import sha256
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_decode_qsa_grid_native as build
import dflash2_decode_ranked_generation as ranked
from check_dflash2_model import request_ids, wire
from engine_once import native_environment

bank, basis, require = ranked.bank, ranked.basis, ranked.require
primary, validation = ranked.primary, basis.validation
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_qsa_grid_admission_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('host', 'reference', 'grid')
FLAG = 'STRATA_SYCL_DECODE_QSA_SCORE_GRID'
HOST_TARGET = 'sycl_verify_window_policy_test'
PREFIX = 'strata verify QSA_SCORE: '
SCORE = re.compile(r'phase=(prompt|decode) tokens=(\d+) reference_calls=(\d+) grid_calls=(\d+)')
CAPTURE = re.compile(r'strata verify: captured the (\d+)-token window in (\d+) event-completed segments')
PLAN = dict(order=list(ORDER), scope='One new executable and the qualified stock global expert placement.',
            checks='Run the same-build host phase policy test, then one native short thinking prompt in each arm. '
                   'Require exact generated IDs and drafts, the actual short prompt route, only the selected '
                   'scorer in each decoder capture and the stock scorer in every verifier prompt capture.',
            limits='Two fresh bounded native processes with parameter validation and stage trace. '
                   'Keep clean model timing and the full-context and RESTORE checks separate.')


def baseline():
    reference = build.retained.baseline()
    native = bank.previous.read(build.ROOT / 'record.json')
    require(build.retained.closed(native, 'built') and native['base_binary'] == reference['binary']
            and native['base_build_sha256'] == reference['native_build_sha256'],
            'finish the coherent grid build with the qualified fallback')
    build.check(native, historical_controls=True)
    stock = ranked.restore_proof(reference, ranked.restore.KEY, ranked.restore.ORDER)
    profile = ranked.qualified_profile('stock', stock, None)
    return reference, native, profile


def score_routes(text, *, grid):
    rows = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith(PREFIX):
            continue
        qsa = SCORE.fullmatch(line[len(PREFIX):])
        previous = CAPTURE.fullmatch(lines[i - 1]) if i else None
        require(qsa and previous and previous[1] == qsa[2], 'QSA calls are not bound to a finalized verifier capture')
        phase, width, ref, candidate = qsa[1], *map(int, qsa.groups()[1:])
        segments = int(previous[2])
        groups, remainder = divmod(segments - 1, 48)
        selected = grid and phase == 'decode' and width in (1, 2, 3, 4)
        require(1 <= groups <= 2 and not remainder and 1 <= width <= 8
                and (ref, candidate) == ((0, 12 * groups) if selected else (12 * groups, 0)),
                'wrong QSA scorer or missing layer coverage')
        rows.append(dict(phase=phase, width=width, reference_calls=ref, grid_calls=candidate))
    require(any(x['phase'] == 'decode' and x['width'] <= 4 for x in rows),
            'no actual supported DFlash2 decode capture')
    return rows


def inputs(revision):
    base, native, profile = baseline()
    pins = ranked.experiment('blend').pins(revision)
    for name in (THIS, build.THIS):
        require((bank.P / name).read_bytes() == validation.git('show', revision + ':' + name),
                'commit the exact grid admission input: ' + name)
        pins[name] = bank.sha(bank.P / name)
    cfg = ranked.configuration(dict(base, qualified_profiles={'stock': profile}), dict(arm='stock', root=str(ROOT)))
    pack = request_ids(Path(cfg['tokenizer']))
    ids, output = pack['ids'], 32
    require(1 <= len(ids) <= 128 and pack['wire_sha256'] == sha256(wire(ids, output)).hexdigest(),
            'the short thinking prompt or its tokenization changed')
    cases, live = [], {}
    for index, arm in enumerate(ORDER):
        root = ROOT / arm
        if arm == 'host':
            binary, args = native['binaries'][HOST_TARGET], []
            env = dict(native['recipe']['environment'], TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
            limits = bank.device.HOST
        else:
            binary, args = native['binaries']['strata'], ['--serve', *cfg['args']]
            env = primary.curve.environment(base, dict(root=str(root), index=index))
            env = native_environment(base, env)
            env.update({FLAG: str(int(arm == 'grid')), 'STRATA_TRACE': '1', 'LD_DEBUG': 'libs',
                        'UR_LOG_VALIDATION': 'level:warning;flush:warning;output:file,' + str(root / 'validation.log')})
            env.pop('UR_LOG_TRACING', None)
            env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
            limits = primary.speed.generation.model.LIMITS
        cases.append(dict(arm=arm, root=str(root), command=[binary['path'], *args], binary=binary,
                          environment_sha256=bank.signature(env), limits=limits))
        live[arm] = env
    result = dict(source_commit=revision, source_inputs=dict(files=len(pins), sha256=bank.signature(pins)),
                  native_source_commit=native['native_source_commit'], build_sha256=bank.signature(build.builder.compact(native)),
                  boot=base['boot'], driver=base['driver'], runtime_manifest=base['runtime_manifest'], stock=profile,
                  plan=PLAN, cases=cases, read_tokens=len(ids), output_cap=output, wire_sha256=pack['wire_sha256'])
    return result, live, (ids, output), base


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    data = bank.previous.read(bank.RESULT)
    data[KEY] = row
    bank.stage.write(bank.RESULT, data)


def run(row, environments, request, base, arm):
    case = next(c for c in row['cases'] if c['arm'] == arm)
    root = Path(case['root'])
    require(not (root / 'execution.json').exists() and bank.sha(case['binary']['path']) == case['binary']['sha256'],
            'new owned case and exact executable required')
    device = primary.speed.generation.model
    owner_kind = device.owner_module
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    owner = owner_kind.Owner(root / 'owned', lambda: bank.stage.write(
        root / 'execution.json', dict(active=bool(owner.active), commands=owner.commands)))
    outcome = dict(arm=arm, host_probe_accepted=False)

    def interact(io, _entry):
        info = {}
        while (line := io.line(600)).startswith('INFO '):
            info.update(field.split('=', 1) for field in line.split()[1:])
        require(line == 'READY 262144 stop' and all(info.get(k) == v for k, v in device.INFO.items()),
                'unexpected native context, cache or drafter')
        outcome['request'] = device.request(io, *request)
        io.send(b'QUIT\n')
        io.exit(90)

    try:
        entry, out, err = owner.run(arm, case['command'], environments[arm], bank.P,
                                    interaction=None if arm == 'host' else interact, **case['limits'])
        outcome['native_exit_code'] = entry['exit_code']
        require(owner_kind.closed(entry) and entry['exit_code'] == 0, 'incorrect exit or owned teardown')
        if arm == 'host':
            require(out.read_text().startswith('verify-window-policy: passed '), 'same-build verifier policy test failed')
        else:
            text = err.read_text()
            proof = validation.preload_evidence(text, entry['pid'])
            warnings = root / 'validation.log'
            require(proof['loader'] == base['native_validation_probe']['loader']['path']
                    and (not warnings.exists() or warnings.stat().st_size == 0)
                    and not re.search(r'(?m)^strata mtp: .*loaded|UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain', text),
                    'unexpected MTP or runtime validation')
            outcome['providers'] = bank.device.loaded_providers(text, dict(
                environment=environments[arm], native_validation_probe=base['native_validation_probe'],
                runtime_manifest=base['runtime_manifest']))
            ranked.restore.check_loader(text, row['stock']['path'], per_layer=False)
            require(text.count('strata prefill policy: requested=throughput route=cache-window') == 1
                    and text.count('strata DFlash2: eager, ') == 1
                    and outcome['request']['generated'] == row['output_cap']
                    and outcome['request']['accepted'] > 0, 'the short prefill or speculative generation changed')
            outcome.update(validation=proof, score_routes=score_routes(text, grid=arm == 'grid'))
            if arm == 'grid':
                require(outcome['request'] == row['outcomes']['reference']['request'],
                        'the two scorers generated different tokens or draft counts')
        outcome['host_probe_accepted'] = True
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        validation.finish(outcome, owner, cursor, row['boot'])
        outcome['passed'] &= outcome.get('driver_after') == row['driver']
        row['outcomes'][arm] = outcome
        complete = outcome['passed'] and arm == ORDER[-1]
        row.update(stage='closed' if complete else 'running' if outcome['passed'] else 'failed',
                   complete=complete, passed=complete, ownership_closed=outcome['ownership_closed'])
        save(row)
    require(outcome['passed'], outcome.get('error', 'DFlash2 grid admission failed'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned Python and one admission stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one new admission cohort')
            revision = validation.git('rev-parse', 'HEAD').decode().strip()
            row, _, _, _ = inputs(revision)
            for case in row['cases']:
                for name in ('tmp', 'cache'):
                    (Path(case['root']) / name).mkdir(mode=0o700, parents=True)
            save(dict(row, stage='prepared', outcomes={}, complete=False, passed=False, ownership_closed=True, active=False))
        else:
            row = bank.previous.read(ROOT / 'record.json')
            pinned, env, request, base = inputs(row['source_commit'])
            require(row == validation.committed(bank.RESULT, (KEY,)) and all(row[k] == v for k, v in pinned.items())
                    and row['ownership_closed'] and not row['active'] and row['stage'] in ('prepared', 'running')
                    and list(row['outcomes']) == list(ORDER[:ORDER.index(action)]), 'commit the exact next admission case')
            run(row, env, request, base, action)
