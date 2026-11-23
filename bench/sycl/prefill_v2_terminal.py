"""Qualify the terminal server route on the retained V2 prefill and decode code."""

import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import build_prefill_mixed_gather as fixture
import prefill_kv_upload_compare as upload
import prefill_terminal_server as terminal
from engine_once import native_environment
from official_model_sources import merge_delta

bank, basis, require = upload.bank, upload.basis, upload.require
restore = terminal.restore
reference = upload.build
KEY = 'prefill_v2_terminal_v1'
ROOT = bank.B / KEY.replace('_', '-')
RELINK = ROOT / 'relink'
SOURCE, BUILD = reference.ROOT / 'source', reference.ROOT / 'build'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
HEADER = 'include/strata/program/full_context_diagnostic.hpp'
GENERATOR = 'src/program/generate.cpp'
HOST_SOURCE = 'tests/core/full_context_diagnostic_test.cpp'
DIAG, HOST = 'strata_full_context_diagnostic', 'full_context_diagnostic_host_test'
OVERLAYS = ('1919720d900ede6c9b55d25d09bb82b503777fd2', 'fa4b9855a3f1706b4689a18473c20e69bd84dd1e')
PROJECTED = {
    GENERATOR: 'c36777200cec703c5de9d8e5ea4820b9eb2c2cc7fd1640d315ce927c743b64b9',
    HEADER: '0429a265793de2c825ebf803c93fab5712ef0b87bb2e01c9ede303920ecaeb4f',
    HOST_SOURCE: '5465a38a009a93e7b689137aacd94a69cf8cf6bf640c3c232d5fa14ee189bb68',
}
PHYSICAL = '6d5debd9cbcb9011d0fc7aa7374ec1700d9e6ccd'
SERVER_RESTORE = '1697efe2f16cd35927c3ab320304633fa395f5b4'
ORDER = ('build', 'host', 'fresh-a', 'fresh-b', 'smoke', 'full')
PROTOCOLS = {'fresh-a': 'fresh-a', 'fresh-b': 'fresh-b', 'smoke': 'fresh-b', 'full': 'base'}


def reference_inputs():
    native = bank.previous.read(reference.ROOT / 'record.json')
    reference.context().check(native)
    key = upload.KEYS['time']
    measured = bank.previous.historic(upload.UPLOAD_TIMING, key)
    speed = bank.previous.read(upload.experiment('time').ROOT / 'record.json')
    physical = bank.previous.historic(PHYSICAL, 'prefill_kv_upload_capacity_v1')
    restored = bank.previous.historic(SERVER_RESTORE, 'prefill_kv_upload_restore_v2')
    ordinary = bank.previous.historic(SERVER_RESTORE, 'prefill_kv_upload_restore_v1')
    require(upload.gather.closed(measured) and basis.compact(speed) == measured
            and measured['binary'] == native['binaries']['strata']
            and measured['decision']['repeated_1000']['batched']
            and upload.gather.closed(physical) and physical['binary'] == native['binaries'][DIAG]
            and physical['outcomes']['full-copy']['final_slot_consumed']
            and upload.gather.closed(restored)
            and restored['plan']['binary'] == measured['binary']
            and restored['outcomes']['split-batched']['passed']
            and ordinary['outcomes']['near-fallback']['passed'],
            'require the measured V2 engine, physical diagnostic and native RESTORE evidence')
    manifest = speed['runtime_manifest']
    providers = bank.previous.read(Path(manifest['path']))
    require(bank.sha(manifest['path']) == manifest['sha256'] and len(providers) == manifest['files']
            and all(bank.sha(path) == h for path, h in providers.items())
            and speed['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and speed['driver'] == native['driver'] == bank.builder.guards.driver_guard(),
            'V2 runtime or boot changed')
    basis.comparison.shared.models.model_pins(speed['configuration'], speed['model_assets'])
    return native, speed, ordinary['outcomes']['near-fallback']['requests'][0]


def projected_sources(native):
    source = {}
    for name in PROJECTED:
        original = (SOURCE / name).read_bytes()
        require(bank.sha(SOURCE / name) == native['source_files'][name], 'original source changed: ' + name)
        raw = original
        for revision in OVERLAYS:
            modified = merge_delta(raw, name, revision + '^', revision)
            require(merge_delta(modified, name, revision, revision + '^') == raw, 'unrelated source changed: ' + name)
            raw = modified
        source[name] = raw
    require(bank.stage.hashed(source) == PROJECTED, 'terminal source projection changed')
    return source


def relink_recipe(native):
    database = BUILD / 'compile_commands.json'
    ninja = '/usr/bin/ninja'
    require(bank.sha(database) == native['configured_files'][str(database)]
            and bank.sha(ninja) == native['build_tools'][ninja], 'retained build configuration changed')
    units = bank.previous.read(database)
    commands, affected, link_inputs = [], [], {}
    for target, path in ((HOST, HOST_SOURCE), (DIAG, GENERATOR)):
        item, = [u for u in units if u['file'] == str(SOURCE / path)
                 and u['output'].startswith(f'CMakeFiles/{target}.dir/')]
        argv = shlex.split(item['command'])
        old = item['output']
        require(item['directory'] == str(BUILD) and argv[-4:] == ['-o', old, '-c', item['file']]
                and ('-DSTRATA_SYCL_FULL_CONTEXT_DIAGNOSTIC=1' in argv) == (target == DIAG),
                'unexpected retained compile recipe')
        obj = str(RELINK / Path(old).name)
        commands.append([argv[0], '-I' + str(RELINK / 'include'), *argv[1:-4], '-MD', '-MF',
                         obj + '.d', '-o', obj, '-c', str(RELINK / path)])
        words = shlex.split(subprocess.check_output([ninja, '-t', 'commands', target], cwd=BUILD,
                                                    text=True, timeout=20).splitlines()[-1])
        require(words[:2] == [':', '&&'] and words[-2:] == ['&&', ':'] and '-Wl,-t' in words
                and words.count(old) == 1 and words[words.index('-o') + 1] == target,
                'unexpected retained link recipe')
        reused = [(BUILD / arg).resolve() for arg in words if arg != old and arg.endswith(('.a', '.o', '.so'))]
        known = native['actual_inputs']['produced'] | native['actual_inputs']['links']
        for p in reused:
            require(known.get(str(p)) == bank.sha(p), 'unverified retained link input: ' + str(p))
            link_inputs[str(p)] = known[str(p)]
        replacements = {old: obj, target: str(RELINK / target)}
        commands.append([words[2], '-v', *(replacements.get(w, w) for w in words[3:-2])])
        affected.append(old)

    deps = subprocess.check_output([ninja, '-t', 'deps'], cwd=BUILD, text=True, timeout=30)
    consumers, recorded = [], set()
    objects = native['actual_inputs']['objects']
    for block in deps.strip().split('\n\n'):
        head, *inputs = block.splitlines()
        name = head.split(': #deps ', 1)[0]
        if str(BUILD / name) not in objects:
            continue
        require(head.endswith('(VALID)') and name not in recorded, 'missing original dependency trace')
        recorded.add(name)
        if str(SOURCE / HEADER) in [p.strip() for p in inputs]:
            consumers.append(name)
    require(len(recorded) == len(objects) and sorted(consumers) == sorted(affected),
            'the changed header has other affected translation units')
    return commands, dict(affected=affected, retained_objects=len(recorded), closure_sha256=bank.signature(deps),
                          retained_link_inputs=link_inputs)


def environment(speed, root):
    ctx = upload.experiment('time')
    env = native_environment(speed, ctx.environment(speed, dict(root=str(root), index=upload.MODES.index('batched'))))
    env.update(STRATA_TRACE='1', STRATA_STATE_HASH='1', STRATA_SNAPSHOT_VERIFY='1',
               STRATA_SYCL_CACHE_LOAN_AUDIT='1', TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
    require(env['STRATA_SYCL_CACHE_SCRATCH_LEASE'] == env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1'
            and env['UR_ENABLE_LAYERS'] == 'UR_LAYER_PARAMETER_VALIDATION'
            and 'UR_LOG_TRACING' not in env, 'retain qualified runtime settings')
    return env


def make_plan(revision, native, speed):
    pins = upload.experiment('time').pins(revision)
    helper_path = str(Path(fixture.__file__).resolve().relative_to(bank.P))
    pins.update(bank.stage.source.pin(bank.P, terminal.SCRIPTS + [THIS, helper_path])['source_sha256'])
    commands, closure = relink_recipe(native)
    base = dict(speed, binary=native['binaries']['strata'], diagnostic=dict(path=str(RELINK / DIAG)))
    cases = {}
    for action, arm in PROTOCOLS.items():
        argv = terminal.command(base, arm)
        if action == 'smoke':
            argv[0] = base['diagnostic']['path']
        cases[action] = dict(command=argv, requests=terminal.plan(arm), wall=1800 if action == 'full' else 900,
                             environment_sha256=bank.signature(environment(speed, ROOT / action)))
    env = dict(native['recipe']['environment'], TMPDIR=str(RELINK / 'tmp'), XDG_CACHE_HOME=str(RELINK / 'cache'))
    return dict(controller_commit=revision, source_inputs=dict(files=len(pins), sha256=bank.signature(pins)),
                original_native_source=native['native_source_commit'], parent=bank.signature(bank.builder.compact(native)),
                overlay_commits=list(OVERLAYS), projected_sources=PROJECTED, commands=commands, closure=closure,
                build_environment_sha256=bank.signature(env), runtime_manifest=speed['runtime_manifest'],
                model_assets_sha256=bank.signature(speed['model_assets']), binary=base['binary'],
                speed_reference=upload.UPLOAD_TIMING, capacity_reference=PHYSICAL, restore_reference=SERVER_RESTORE,
                boot=speed['boot'], driver=speed['driver'], order=list(ORDER), cases=cases, budgets=restore.capacity.BUDGET,
                scope='Dedicated diagnostic sibling. Exercise the real leased server prefill, final verifier and restore.',
                acceptance='Check compiled bounds, both independent normal-server short controls and a short request '
                           'on the diagnostic first. Then consume exactly 262144 tokens with zero output, check '
                           'the loan generation and preserve the full snapshot across refusal and restore. '
                           'Compare both short MTP requests against independent fresh controls.',
                retention='Keep the qualified prefill and decode binaries through their imminent paired '
                          'phase validation. Retire this temporary diagnostic and its captures when the terminal '
                          'comparison closes after source and command recovery.')


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def helper_for_relink():
    helper = basis.core.source.device.load_module('v2_terminal_relink', fixture.__file__)
    helper.SOURCE, helper.BUILD = SOURCE, BUILD
    return helper


def build(plan, native):
    helper = helper_for_relink()
    sources = projected_sources(native)
    commands = plan['commands']
    allowed = helper.build_inputs(native, RELINK, sources, commands)
    helper.build_fixture(RELINK, DIAG, native, sources, commands, allowed,
                         dict(controller_commit=plan['controller_commit'], plan_sha256=bank.signature(plan)))
    receipt = bank.previous.read(RELINK / 'record.json')
    require(receipt['complete'] and receipt['ownership_closed'], 'diagnostic relink failed to close')
    host_object = Path(commands[0][commands[0].index('-o') + 1])
    observed = {}
    for line in (RELINK / 'owned' / 'command-1.stdout').read_text().splitlines():
        path = (BUILD / line.strip()).resolve()
        require(path.is_file(), 'invalid host linker input')
        h = bank.sha(path)
        require(path == host_object or allowed.get(str(path)) == h, 'unverified host linker input')
        observed[str(path)] = h
    require(str(host_object) in observed and len(observed) >= 8, 'incomplete host linker trace')
    require(all(receipt['compiled'].get(str(RELINK / p)) == h for p, h in PROJECTED.items()),
            'missing projected compiler input')
    host = RELINK / HOST
    return dict(complete=True, passed=True, ownership_closed=True, build_sha256=bank.sha(RELINK / 'record.json'),
                diagnostic=receipt['binary'], host=dict(path=str(host), sha256=bank.sha(host), bytes=host.stat().st_size),
                compiled_inputs=len(receipt['compiled']), compiled_sha256=bank.signature(receipt['compiled']),
                linked_inputs=len(receipt['linked']), linked_sha256=bank.signature(receipt['linked']),
                host_link_inputs=len(observed), host_link_sha256=bank.signature(observed),
                host_object_sha256=observed[str(host_object)], fixture_host_slices=receipt['fixture_host_slices'])


def verify_build(plan, outcome):
    receipt = bank.previous.read(RELINK / 'record.json')
    require(bank.sha(RELINK / 'record.json') == outcome['build_sha256']
            and receipt['plan_sha256'] == bank.signature(plan) and receipt['ownership_closed'] and receipt['complete']
            and receipt['binary'] == outcome['diagnostic']
            and all(bank.sha(p) == h for key in ('compiled', 'linked') for p, h in receipt[key].items())
            and all(bank.sha(RELINK / p) == h for p, h in receipt['driver_inputs'].items())
            and bank.sha(RELINK / 'full_context_diagnostic_test.cpp.o') == outcome['host_object_sha256']
            and all(bank.sha(outcome[k]['path']) == outcome[k]['sha256'] for k in ('diagnostic', 'host')),
            'diagnostic or exact compile/link inputs changed')


def run(action, plan, speed, prefix, outcomes):
    verify_build(plan, outcomes['build'])
    root = ROOT / action
    require(not root.exists(), 'inspect the existing case before retrying')
    for part in ('tmp', 'cache'):
        (root / part).mkdir(mode=0o700, parents=True)
    owner = restore.owner_module.Owner(root / 'owned', lambda: bank.stage.write(
        root / 'execution.json', dict(active=bool(owner.active), commands=owner.commands)))
    cursor, = (s.removeprefix('-- cursor: ') for s in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if s.startswith('-- cursor: '))
    outcome = dict(complete=False, host_probe_accepted=False)
    try:
        if action == 'host':
            argv = [outcomes['build']['host']['path']]
            entry, stdout, _ = owner.run('host', argv, dict(PATH='/usr/bin:/bin', LANG='C.UTF-8',
                LD_LIBRARY_PATH=speed['environment_template']['LD_LIBRARY_PATH']), bank.P,
                wall=45, cpu=40, rss_cap=512 << 20, text_cap=128 << 10, file_cap=1 << 20, total_cap=4 << 20)
            require(restore.owner_module.closed(entry) and entry['exit_code'] == 0, 'compiled bounds test failed')
            text = stdout.read_text()
            require('full_context_request_and_terminal_bounds=PASS' in text
                    and 'byte_order_and_bounds=PASS full_extent_spans=1 GPU=UNRUN' in text,
                    'host bounds evidence differs')
            outcome['host'] = text.strip().splitlines()
        else:
            case, arm = plan['cases'][action], PROTOCOLS[action]
            env = environment(speed, root)
            require(bank.signature(env) == case['environment_sha256'], 'native child environment changed')
            expected = outcomes['build']['diagnostic'] if action in ('smoke', 'full') else plan['binary']
            require(case['command'][0] == expected['path'] and bank.sha(expected['path']) == expected['sha256'],
                    'native case binary changed')
            ids = set(json.loads((Path(speed['configuration']['tokenizer']) / 'vocab.json').read_bytes()).values())
            require(set(range(4096)) <= ids, 'diagnostic token IDs are outside the pinned vocabulary')
            responses = {}

            def interact(io, _):
                responses.update(terminal.exchange(io, arm, case['requests'], max(ids) + 1))

            print(f'Native terminal stage {action} starts', flush=True)
            entry, _, stderr = owner.run(action, case['command'], env, bank.P, interaction=interact,
                                         wall=case['wall'], **plan['budgets'])
            outcome.update(native_exit_code=entry['exit_code'], requests=responses)
            require(restore.owner_module.closed(entry) and entry['exit_code'] == 0, 'server or teardown failed')
            outcome.update(terminal.inspect(stderr.read_text(), arm, responses, outcomes, entry['pid'],
                                            speed['native_validation_probe'], v2_prefix=prefix))
            if action == 'smoke':
                require(responses == outcomes['fresh-b']['requests'], 'diagnostic smoke changed ordinary decode')
            if action == 'full':
                basis.comparison.shared.models.model_pins(speed['configuration'], speed['model_assets'], full=True)
        verify_build(plan, outcomes['build'])
        outcome.update(complete=True, host_probe_accepted=True)
    except Exception as error:
        outcome['error'] = repr(error)
    finally:
        basis.validation.finish(outcome, owner, cursor, plan['boot'])
        outcome['passed'] = bool(outcome['passed'] and outcome['driver_after'] == plan['driver'])
    return outcome


def execute(action):
    native, speed, prefix = reference_inputs()
    if action == 'prepare':
        require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'inspect existing campaign')
        revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
        plan = make_plan(revision, native, speed)
        projected_sources(native)
        ROOT.mkdir(mode=0o700)
        save(dict(plan=plan, outcomes={}, stage='prepared', active=False, ownership_closed=True))
        return

    row = bank.previous.read(ROOT / 'record.json')
    plan = row['plan']
    require(row == basis.validation.committed(bank.RESULT, (KEY,)) and row['ownership_closed'] and not row['active']
            and row['stage'] in ('prepared', 'running') and list(row['outcomes']) == list(ORDER[:ORDER.index(action)])
            and plan == make_plan(plan['controller_commit'], native, speed), 'commit the exact plan and preceding stages')
    if action == 'build':
        try:
            outcome = build(plan, native)
        except Exception as error:
            receipt = RELINK / 'record.json'
            closed = receipt.is_file() and bank.previous.read(receipt)['ownership_closed']
            outcome = dict(passed=False, complete=False, ownership_closed=closed, error=repr(error))
    else:
        outcome = run(action, plan, speed, prefix, row['outcomes'])
    done = outcome['passed'] and action == ORDER[-1]
    row['outcomes'][action] = outcome
    row.update(stage='closed' if done else 'running' if outcome['passed'] else 'failed', complete=done, passed=done,
               ownership_closed=outcome['ownership_closed'], active=False)
    save(row)
    print(json.dumps({k: outcome.get(k) for k in ('passed', 'complete', 'native_exit_code', 'fresh_faults', 'error')}),
          flush=True)
    require(outcome['passed'], outcome.get('error', 'terminal experiment gate stopped'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned interpreter and one planned terminal stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        execute(action)
