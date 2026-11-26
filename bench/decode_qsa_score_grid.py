"""Qualify the bounded decode QSA score grid against retained Arc graphs."""

import fcntl
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
import decode_qsa_score_grid_source as source
import dflash2_primary_retained as retained

bank, basis, require = retained.bank, retained.basis, retained.require
sha, write, validation = bank.sha, bank.stage.write, basis.validation
PRIMARY = retained.PRIMARY.ROOT
KEY = 'decode_qsa_score_grid_v1'
ROOT = bank.B / KEY.replace('_', '-')
BUILD = ROOT / 'fixture-build'
TARGET = 'decode_qsa_score_grid'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
FIXTURE = 'bench/sycl/decode_qsa_score_grid.cpp'
GENERATOR = 'bench/sycl/decode_qsa_score_grid_source.py'
GENERATED = 'decode_qsa_score_grid_kernel.cpp'
LINK_DRIVER = 'bench/sycl/build_prefill_mixed_gather.py'
TEMPLATE_TARGET = 'sycl_decode_moe_combine_test'
TEMPLATE_FIXTURE = 'tests/sycl/decode_moe_combine.cpp'
LIBRARIES = ('libstrata_kernels.a', 'libstrata_core.a', 'libstrata_sycl_runtime.a')
STAGES = ('smoke', 'validate', '0', '1', '2')
MODES = ('smoke', 'validate', 'forward', 'reverse', 'forward')
CONTEXTS, WIDTHS, REPLAYS, BATCHES = (96204, 262144), (1, 2, 3, 4), 512, 3
PROFILE_REPLAYS = 5
LIMITS = dict(wall=300, cpu=600, rss_cap=4 << 30, text_cap=24 << 20,
              file_cap=32 << 20, total_cap=64 << 20)
PLAN = dict(stages=list(STAGES), modes=list(MODES), contexts=list(CONTEXTS), widths=list(WIDTHS),
            grid=source.GROUPS, replays=REPLAYS, batches=BATCHES, budget=LIMITS,
            diagnostic=dict(stage='profile', after=list(STAGES), replays=PROFILE_REPLAYS,
                            clock='Outer SYCL graph end minus start. Report device ticks and wall completion in '
                                  'the profiled process. Preserve unsupported clocks without changing clean timing.'),
            scope='Same retained scorer and FP32 arithmetic. Change only its decode grid. Use the production '
                  'SYCL compatibility graph and the unchanged top-k. Compare scorer-only and scorer-plus-top-k replay.',
            gates='Trace the short smoke first, then check exact score bits, selection IDs and guards at every '
                  'tail remainder, invalid steps, ties and small→large→small replay on the same graph. '
                  'Each timing batch is at least 10 ms and the empty clock is at most 1% of its completion interval.',
            decision='Require an improvement in the median scorer and combined graph in all three independent '
                     'processes for a shape. Report graph setup separately and the number of replays required to '
                     'recover an observed extra setup cost. Model throughput and adoption require separate checks.',
            stop='Stop on the first invalid input, validation, timing, owner or Xe fault. Preserve every captured result.')


def inputs(revision):
    live = retained.baseline()
    parent = bank.previous.read(PRIMARY / 'record.json')
    paths = (THIS, FIXTURE, GENERATOR, LINK_DRIVER, retained.THIS)
    committed = {p: validation.git('show', revision + ':' + p) for p in paths}
    require(all((bank.P / p).read_bytes() == data for p, data in committed.items()), 'commit the exact fixture and recipe')
    stock = validation.git('show', live['native_source_commit'] + ':' + source.SOURCE)
    require(sha(PRIMARY / 'source' / source.SOURCE) == parent['source_files'][source.SOURCE] == source.SOURCE_SHA256,
            'the exact retained stock scorer is required')
    overlay = {FIXTURE: committed[FIXTURE], GENERATED: source.generate(stock)}

    parent_dir = PRIMARY / 'build'
    db = parent_dir / 'compile_commands.json'
    units = json.loads(db.read_bytes())
    commands, local_objects = [], []
    for name, original in ((FIXTURE, TEMPLATE_FIXTURE), (GENERATED, source.SOURCE)):
        src = str(PRIMARY / 'source' / original)
        unit, = [u for u in units if u['file'] == src]
        stock_object = unit['output']
        argv = shlex.split(unit['command'])
        require(Path(unit['directory']) == parent_dir and argv[-4:] == ['-o', stock_object, '-c', src]
                and '-fsycl' in argv and '-ffp-contract=off' in argv
                and not any(f in argv for f in ('-ffast-math', '-Ofast')), 'unexpected strict SYCL compile recipe')
        obj = str(BUILD / (Path(name).stem + '.o'))
        local_objects.append(obj)
        commands.append([*argv[:-4], '-MD', '-MF', obj + '.d', '-o', obj, '-c', str(BUILD / name)])
        if name == FIXTURE:
            fixture_object = stock_object

    ninja = '/usr/bin/ninja'
    require(sha(ninja) == parent['build_tools'][ninja], 'selected Ninja tool changed')
    raw = subprocess.check_output([ninja, '-t', 'commands', TEMPLATE_TARGET], cwd=parent_dir, text=True, timeout=20)
    link = shlex.split(raw.splitlines()[-1])
    require(link[:2] == [':', '&&'] and link[-2:] == ['&&', ':'] and '-Wl,-t' in link
            and link.count(fixture_object) == 1 and link[link.index('-o') + 1] == TEMPLATE_TARGET
            and link[2] == commands[0][0] and tuple(x for x in link if x.endswith('.a')) == LIBRARIES,
            'selected component link recipe changed')
    argv = link[2:-2]
    argv[argv.index('-o') + 1] = str(BUILD / TARGET)
    i = argv.index(fixture_object)
    argv[i:i + 1] = local_objects
    commands.append([argv[0], '-v', *argv[1:]])
    archives = {str(parent_dir / p): parent['actual_inputs']['produced'][str(parent_dir / p)] for p in LIBRARIES}
    fields = {k: live[k] for k in ('boot', 'driver', 'native_source_commit', 'native_build_sha256',
                                    'native_validation_probe', 'runtime_manifest')}
    fields.update(source_commit=revision, input_sha256=bank.stage.hashed(committed),
                  overlay_sha256=bank.stage.hashed(overlay), compile_database_sha256=sha(db),
                  parent_environment_sha256=bank.signature(parent['recipe']['environment']),
                  commands=commands, archives=archives, plan=PLAN)
    return fields, parent, overlay


def save(row):
    write(ROOT / 'record.json', row)
    entries = bank.previous.read(bank.RESULT)
    entries[KEY] = row
    write(bank.RESULT, entries)


def build(row, parent, overlay):
    require(row['stage'] == 'prepared' and not BUILD.exists(), 'one fresh component build')
    driver = basis.core.source.device.load_module('decode_qsa_grid_build', bank.P / LINK_DRIVER)
    driver.BUILD, driver.SOURCE = PRIMARY / 'build', PRIMARY / 'source'
    try:
        allowed = driver.build_inputs(parent, BUILD, overlay, row['commands'])
        driver.build_fixture(BUILD, TARGET, parent, overlay, row['commands'], allowed,
                             dict(source_commit=row['source_commit'], overlay_sha256=row['overlay_sha256']))
        proof = bank.previous.read(BUILD / 'record.json')
        require(proof['complete'] and proof['ownership_closed'] and proof['commands'] == row['commands']
                and all(proof['linked'].get(p) == h for p, h in row['archives'].items()), 'component link proof differs')
        row.update(stage='built', binary=proof['binary'], build_receipt_sha256=sha(BUILD / 'record.json'),
                   compiled_inputs=len(proof['compiled']), linked_inputs=len(proof['linked']))
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        receipt = BUILD / 'record.json'
        if receipt.exists():
            row['ownership_closed'] = bank.previous.read(receipt)['ownership_closed']
        raise
    finally:
        save(row)


def observations(text, mode):
    widths = (1,) if mode == 'smoke' else WIDTHS
    lines = text.strip().splitlines()
    require(lines and lines[-1] == 'decode QSA scorer completed', 'missing closed scorer fixture')
    timings, device, setup, checked = [], [], {}, []
    number = r'(\d+\.\d+)'
    for line in lines[:-1]:
        if line.startswith('setup '):
            value = re.fullmatch(r'setup width=(\d+)' + ''.join(f' ms{i}={number}' for i in range(4)), line)
            require(value is not None and int(value[1]) not in setup, 'wrong graph setup row')
            setup[int(value[1])] = [float(x) for x in value.groups()[1:]]
        elif line.startswith('checked '):
            value = re.fullmatch(r'checked width=(\d+) scores=PASS ids=PASS guards=PASS replay=PASS', line)
            require(value is not None, 'missing numerical and replay proof')
            checked.append(int(value[1]))
        elif line.startswith('device '):
            value = re.fullmatch(r'device width=(\d+) context=(\d+) stage=(\d+) replay=(\d+) arm=(\d+) '
                                 rf'ns=(\d+) tick_ns=(\d+) wall_ms={number}', line)
            require(value is not None, 'unrecognized profiled graph row')
            width, context, stage, replay, arm, ns, tick_ns = map(int, value.groups()[:7])
            wall_ms = float(value[8])
            require(ns >= tick_ns > 0 and math.isfinite(wall_ms) and wall_ms > 0,
                    'invalid profiled graph interval')
            device.append(dict(width=width, context=context, stage=stage, replay=replay, arm=arm,
                               ns=ns, tick_ns=tick_ns, wall_ms=wall_ms))
        else:
            value = re.fullmatch(r'sample width=(\d+) context=(\d+) stage=(\d+) batch=(\d+) arm=(\d+) '
                                 rf'replays={REPLAYS} ms={number} clock_ms={number}', line)
            require(value is not None, 'unrecognized timing row')
            width, context, stage, batch, arm = map(int, value.groups()[:5])
            ms, empty = map(float, value.groups()[5:])
            require(math.isfinite(ms) and ms >= 10 and 0 < empty <= ms * .01, 'invalid completion clock or observer overhead')
            timings.append(dict(width=width, context=context, stage=stage, batch=batch, arm=arm, ms=ms, clock_ms=empty))
    require(tuple(setup) == tuple(checked) == widths and all(all(math.isfinite(x) and x > 0 for x in s) for s in setup.values()),
            'incomplete graph setup or correctness coverage')
    offset = int(mode == 'reverse')
    expected = ([(w, c, s, b, (i + b + offset) % 2) for w in WIDTHS for c in CONTEXTS for s in range(2)
                 for b in range(BATCHES) for i in range(2)] if mode in ('forward', 'reverse') else [])
    require([(s['width'], s['context'], s['stage'], s['batch'], s['arm']) for s in timings] == expected, 'incomplete timing order')
    expected_device = ([(w, c, s, i, (j + i) % 2) for w in WIDTHS for c in CONTEXTS for s in range(2)
                       for i in range(PROFILE_REPLAYS) for j in range(2)] if mode == 'profile' else [])
    require([(s['width'], s['context'], s['stage'], s['replay'], s['arm']) for s in device] == expected_device,
            'incomplete profiled graph order')
    return {str(k): v for k, v in setup.items()}, timings, device


def device_case(row, parent, stage, mode):
    build_record = BUILD / 'record.json'
    proof = bank.previous.read(build_record)
    require(sha(build_record) == row['build_receipt_sha256'] and proof['binary'] == row['binary']
            and sha(row['binary']['path']) == row['binary']['sha256']
            and all(sha(p) == h for g in ('compiled', 'linked') for p, h in proof[g].items()), 'component build changed')
    case = ROOT / stage
    case.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (case / name).mkdir(mode=0o700)
    probe = row['native_validation_probe']
    env = dict(parent['recipe']['environment'], TMPDIR=str(case / 'tmp'), XDG_CACHE_HOME=str(case / 'cache'),
               LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
               LD_DEBUG='libs', UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(case / 'validation.log'))
    env.pop('UR_LOG_TRACING', None)
    env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
    if stage == 'smoke':
        env['UR_ENABLE_LAYERS'] = 'UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION'
        env['UR_LOG_TRACING'] = 'level:info;flush:info;output:stderr'
    require(all(env[k] == v for k, v in (('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', '1'),
                 ('NEOReadDebugKeys', '1'), ('EnableDirectSubmission', '0'))), 'runtime safety settings changed')
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    owner = bank.builder.ownership.Owner(case / 'owned', lambda: write(case / 'ownership.json', owner.commands))
    command = [row['binary']['path'], mode]
    result = dict(stage=stage, mode=mode, command=command, environment_sha256=bank.signature(env), ownership_closed=False)
    try:
        entry, out, err = owner.run(stage, command, env, case, **LIMITS)
        errors = err.read_text()
        require(bank.builder.ownership.closed(entry) and entry['exit_code'] == 0,
                'fixture execution failed: ' + errors[-1200:])
        runtime_proof = validation.preload_evidence(errors, entry['pid'])
        warning = case / 'validation.log'
        require(runtime_proof['loader'] == probe['loader']['path']
                and (not warning.exists() or warning.stat().st_size == 0)
                and (stage != 'smoke' or 'urEnqueue' in errors), 'runtime or API validation differs')
        result['providers'] = bank.device.loaded_providers(errors, dict(row, environment=env))
        result['setup_ms'], samples, device = observations(out.read_text(), mode)
        result['device_samples' if mode == 'profile' else 'samples'] = device if mode == 'profile' else samples
        leader, = [item for item in entry['owners'] if item['pid'] == entry['pid']]
        result.update(host_probe_accepted=True, validation=runtime_proof, native_identity=leader)
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        validation.finish(result, owner, cursor, row['boot'])
        result['passed'] &= result.get('driver_after') == row['driver']
        write(case / 'outcome.json', result)
    return result


def decision(processes):
    identities = {(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}
    require(len(identities) == len(STAGES), 'independent process identities required')
    rows = []
    for width in WIDTHS:
        for context in CONTEXTS:
            ratios = [[], []]
            break_even = []
            for p in processes[2:]:
                for stage in range(2):
                    durations = [statistics.median(s['ms'] for s in p['samples']
                        if (s['width'], s['context'], s['stage'], s['arm']) == (width, context, stage, arm))
                        for arm in range(2)]
                    ratios[stage].append(durations[1] / durations[0])
                    if stage == 1:
                        setup = p['setup_ms'][str(width)]
                        saved = (durations[0] - durations[1]) / REPLAYS
                        break_even.append(math.ceil(max(0, setup[3] - setup[2]) / saved) if saved > 0 else None)
            rows.append(dict(width=width, context=context, scorer_ratios=ratios[0], combined_ratios=ratios[1],
                             extra_setup_replays=break_even, eligible=all(r < 1 for r in (*ratios[0], *ratios[1]))))
    return rows


def main(action):
    if action == 'prepare':
        require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'fresh decode scorer experiment required')
        revision = validation.git('rev-parse', 'HEAD').decode().strip()
        fields, _, _ = inputs(revision)
        ROOT.mkdir(mode=0o700)
        save(dict(fields, stage='prepared', processes=[], active=False, ownership_closed=True,
                  complete=False, passed=False, model_speed_qualified=False))
        return
    row = bank.previous.read(ROOT / 'record.json')
    require(validation.committed(bank.RESULT, (KEY,)) == row and row['ownership_closed'] and not row['active'],
            'commit the exact preceding component stage')
    fields, parent, overlay = inputs(row['source_commit'])
    require(all(row[k] == v for k, v in fields.items()), 'frozen component inputs changed')
    if action == 'build':
        build(row, parent, overlay)
        return
    if action == 'profile':
        require(row['stage'] == 'closed' and 'profile' not in row, 'one separate profile after clean closure')
        result = device_case(row, parent, action, action)
        row.update(profile=result, ownership_closed=result['ownership_closed'])
        save(row)
        require(result['passed'], result.get('error', 'profile validation or ownership gate failed'))
        return
    index = STAGES.index(action)
    require(row['stage'] == ('built' if index == 0 else STAGES[index - 1]) and len(row['processes']) == index,
            'run the next committed stage only')
    try:
        result = device_case(row, parent, action, MODES[index])
        row['processes'].append(result)
        require(result['passed'], result.get('error', 'component validation or ownership gate failed'))
        row['stage'] = action
        if index == len(STAGES) - 1:
            row.update(stage='closed', complete=True, passed=True, decision=decision(row['processes']))
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        raise
    finally:
        row['ownership_closed'] = all(p['ownership_closed'] for p in row['processes'])
        save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'build', 'profile', *STAGES),
            'use the pinned interpreter and one planned scorer stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        main(action)
