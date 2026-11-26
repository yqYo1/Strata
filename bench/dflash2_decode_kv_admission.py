"""Check cooperative KV dispatch and exact short-request parity on DFlash2."""

import fcntl
from hashlib import sha256
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_decode_qsa_grid_admission as scorer
from check_dflash2_model import request_ids, wire
from engine_once import native_environment

bank, basis, ranked, primary, build = scorer.bank, scorer.basis, scorer.ranked, scorer.primary, scorer.build
require, validation = scorer.require, scorer.validation
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_kv_admission_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('scalar', 'cooperative')
FLAG = 'STRATA_SYCL_DECODE_KV_COALESCED'
PREFIX = 'strata verify KV_PLAN: '
KV_PLAN = re.compile(r'phase=(prompt|decode) tokens=(\d+) enabled=([01]) scalar_calls=(\d+) coalesced_calls=(\d+)')
PLAN = dict(order=list(ORDER), scope='One retained executable, stock global placement and scalar QSA scorer.',
            checks='Use the same exact 82-input/32-output thinking request in both processes. '
                   'Require identical generated IDs and draft counters, scalar KV in prefill and decode width one, '
                   'and the selected KV path in every eligible decode capture. Validate the actual runtime and ownership.',
            limits='Parameter validation and short stage tracing. This checks route and output parity at short extent. '
                   'Clean long-generation timing, streamed KV at physical capacity and RESTORE are separate.')


def kv_routes(text, *, cooperative, widths=(2, 4)):
    """Reconcile each KV dispatch with its finalized, scalar-QSA capture."""
    qsa_rows = scorer.score_routes(text, grid=False)
    calls = [KV_PLAN.fullmatch(line[len(PREFIX):]) for line in text.splitlines() if line.startswith(PREFIX)]
    require(len(qsa_rows) == len(calls), 'missing or duplicate KV capture record')
    routes = []
    for qsa, kv in zip(qsa_rows, calls, strict=True):
        require(kv is not None, 'malformed KV capture record')
        width, use_cooperative, scalar, coalesced = map(int, kv.groups()[1:])
        phase = qsa['phase']
        eligible = phase == 'decode' and width in widths
        layers = qsa['reference_calls']
        selected = cooperative and eligible
        require(kv[1] == phase and width == qsa['width'] and use_cooperative == int(cooperative)
                and (scalar, coalesced) == ((0, layers) if selected else (layers, 0)),
                'unexpected prompt, scalar fallback or cooperative KV dispatch')
        routes.append(dict(phase=phase, width=width, scalar_calls=scalar, coalesced_calls=coalesced))
    require(any(r['phase'] == 'decode' and r['width'] in widths for r in routes),
            'no eligible decode KV capture')
    return routes


def inputs(revision):
    base, native, profile = scorer.baseline()
    checked = validation.committed(bank.RESULT, (scorer.KEY,))
    identity = bank.signature(build.builder.compact(native))
    require(build.retained.closed(checked, 'closed') and checked['build_sha256'] == identity
            and checked['stock'] == profile and checked['outcomes']['reference']['passed'],
            'require the closed scalar-QSA control on the same executable and profile')
    sources = ranked.experiment('blend').pins(revision)
    for name in (THIS, scorer.THIS, build.THIS):
        path = bank.P / name
        require(path.read_bytes() == validation.git('show', revision + ':' + name),
                'commit the exact cooperative KV input: ' + name)
        sources[name] = bank.sha(path)
    cfg = ranked.configuration(dict(base, qualified_profiles={'stock': profile}), dict(arm='stock', root=str(ROOT)))
    args = cfg['args']
    for flag, value in (('--kv', 'int8'), ('--kv-resident', '20480'), ('--max-context', '262144'), ('--spec', '4')):
        require(args.count(flag) == 1 and args[args.index(flag) + 1] == value, 'unexpected KV geometry: ' + flag)
    pack = request_ids(Path(cfg['tokenizer']))
    ids, output = pack['ids'], checked['output_cap']
    require(len(ids) == checked['read_tokens'] == 82 and output == 32
            and pack['wire_sha256'] == checked['wire_sha256'] == sha256(wire(ids, output)).hexdigest(),
            'probe request differs from the known scalar control')

    device = primary.speed.generation.model
    binary = native['binaries']['strata']
    cases, environments = [], {}
    for index, arm in enumerate(ORDER):
        root = ROOT / arm
        env = native_environment(base, primary.curve.environment(base, dict(root=str(root), index=index)))
        require(env.get('STRATA_SYCL_KV_COPY_GROUPS') == '0' and FLAG not in env and scorer.FLAG not in env,
                'unexpected inherited KV or QSA selector')
        env.update({FLAG: str(int(arm == 'cooperative')), scorer.FLAG: '0', 'STRATA_TRACE': '1', 'LD_DEBUG': 'libs',
                    'UR_ENABLE_LAYERS': 'UR_LAYER_PARAMETER_VALIDATION',
                    'UR_LOG_VALIDATION': f'level:warning;flush:warning;output:file,{root / "validation.log"}'})
        env.pop('UR_LOG_TRACING', None)
        environments[arm] = env
        cases.append(dict(arm=arm, root=str(root), command=[binary['path'], '--serve', *args],
                          environment_sha256=bank.signature(env), limits=device.LIMITS))
    record = dict(source_commit=revision, source_inputs=dict(files=len(sources), sha256=bank.signature(sources)),
                  native_source_commit=native['native_source_commit'], build_sha256=identity, binary=binary,
                  boot=base['boot'], driver=base['driver'], runtime_manifest=base['runtime_manifest'], stock=profile,
                  scalar_qsa_admission_sha256=bank.signature(checked), plan=PLAN, cases=cases,
                  read_tokens=len(ids), output_cap=output, wire_sha256=pack['wire_sha256'])
    return record, environments, (ids, output), base


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    data = bank.previous.read(bank.RESULT)
    data[KEY] = row
    bank.stage.write(bank.RESULT, data)


def short_routes(text, *, cooperative):
    require(text.count('strata prefill policy: requested=throughput route=cache-window') == 1,
            'the actual short prefill route changed')
    routes = kv_routes(text, cooperative=cooperative)
    require({('prompt', 1), ('decode', 1), ('decode', 4)} <= {(r['phase'], r['width']) for r in routes},
            'the short route must validate prompt, width-one fallback and width-four decode')
    return routes


def execute(row, env, request, base, arm, *, read_routes=short_routes, publish=save):
    case = next(c for c in row['cases'] if c['arm'] == arm)
    root = Path(case['root'])
    binary = row['binary']
    require(not (root / 'execution.json').exists() and bank.sha(binary['path']) == binary['sha256'],
            'require the fresh case and exact executable')
    device = primary.speed.generation.model
    owner_type = device.owner_module
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    owner = owner_type.Owner(root / 'owned', lambda: bank.stage.write(
        root / 'execution.json', dict(active=bool(owner.active), commands=owner.commands)))
    outcome = dict(arm=arm, host_probe_accepted=False)

    def interact(io, _entry):
        info = {}
        while (line := io.line(600)).startswith('INFO '):
            info.update(field.split('=', 1) for field in line.split()[1:])
        require(line == 'READY 262144 stop' and all(info.get(k) == v for k, v in device.INFO.items())
                and info.get('kv_resident') == '20480', 'unexpected native KV extent or drafter')
        outcome['request'] = device.request(io, *request)
        io.send(b'QUIT\n')
        io.exit(90)

    try:
        entry, _, err = owner.run(arm, case['command'], env, bank.P, interaction=interact, **case['limits'])
        outcome['native_exit_code'] = entry['exit_code']
        require(owner_type.closed(entry) and entry['exit_code'] == 0, 'incorrect exit or owned teardown')
        text = err.read_text()
        proof = validation.preload_evidence(text, entry['pid'])
        warnings = root / 'validation.log'
        require(proof['loader'] == base['native_validation_probe']['loader']['path']
                and (not warnings.exists() or warnings.stat().st_size == 0)
                and not re.search(r'(?m)^strata mtp: .*loaded|UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain', text),
                'unexpected MTP or runtime validation')
        outcome['providers'] = bank.device.loaded_providers(text, dict(
            environment=env, native_validation_probe=base['native_validation_probe'], runtime_manifest=base['runtime_manifest']))
        ranked.restore.check_loader(text, row['stock']['path'], per_layer=False)
        require(text.count('strata DFlash2: eager, ') == 1 and outcome['request']['generated'] == row['output_cap']
                and outcome['request']['accepted'] > 0, 'the actual DFlash2 speculative work changed')
        routes = read_routes(text, cooperative=arm == 'cooperative')
        outcome.update(validation=proof, kv_routes=routes)
        if arm == 'cooperative':
            require(outcome['request'] == row['outcomes']['scalar']['request'], 'output IDs or draft counters differ')
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
        publish(row)
    require(outcome['passed'], outcome.get('error', 'DFlash2 cooperative KV admission failed'))


def main(action):
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned Python and one cooperative KV stage')
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
            return

        row = bank.previous.read(ROOT / 'record.json')
        pinned, environments, request, base = inputs(row['source_commit'])
        require(row == validation.committed(bank.RESULT, (KEY,)) and all(row[k] == v for k, v in pinned.items())
                and row['ownership_closed'] and not row['active'] and row['stage'] in ('prepared', 'running')
                and list(row['outcomes']) == list(ORDER[:ORDER.index(action)]), 'commit the exact next admission case')
        execute(row, environments[action], request, base, action)


if __name__ == '__main__':
    action, = sys.argv[1:]
    main(action)
