"""Coherently rebuild the verified attention-store image with the input-borrow candidate."""

import fcntl
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_attention_bank_build as previous
import prefill_window_device as device
import qsa_prefill_probability_source as probability

basis, stage, builder = previous.basis, previous.stage, previous.builder
require, sha, signature = previous.require, previous.sha, previous.signature
P, B, RESULT = previous.P, previous.B, previous.RESULT
ROOT = B / 'prefill-resident-input-native-v1'
KEY = 'prefill_resident_input_native_v1'
BASE_EVIDENCE = 'a3ad7713654cf9dbc61a786d02912d7b7ee56e57'
PATCH_BASE = '3d263da77f548c5252cd2864fd3e3588519bab1c'
INPUT_SOURCE = '1bf54ec3277f60c77417dac68642966f3782652d'
FIXTURE = 'tests/sycl/prefill_attention_bank.cpp'
CMAKE = 'cmake/sycl_vllm_xe2_native16.cmake'
TARGET = 'sycl_prefill_attention_bank_test'
TARGETS = [*stage.TARGETS, TARGET]
COMPARISON = 'prefill_ordinary_bank_comparison_v1'
PV_SOURCE = '45a1b7b773d41e0de532d779ce7e48428ca76154'
PV_GENERATOR = 'tools/sycl/qsa_prefill_probability_source.py'
PV_KERNEL = 'bench/sycl/qsa_prefill_probability_generated.cpp'
PV_FIXTURES = ('tests/sycl/qsa_bf16_prefix.cpp', 'tests/sycl/qsa_bf16_prefix_layout_bench.cpp',
               'bench/sycl/qsa_prefill_probability.hpp')


def image(native):
    """Recover every source byte from Git, including the superseded store image."""
    old = previous.historic(BASE_EVIDENCE, previous.KEY)
    component = previous.historic(previous.COMPONENT_EVIDENCE, previous.COMPONENT)
    _, payload = previous.stage_image(native, component)
    outputs = dict(stage.source.blobs(P, component['source_commit'], previous.OVERLAYS))
    require(stage.hashed(outputs) == old['history']['overlays'], 'output-store source differs')
    payload.update(outputs)
    require(old['source_files'] == dict(files=len(payload), sha256=signature(stage.hashed(payload))),
            'recovered parent is not the measured coherent store image')
    paths = (*previous.OVERLAYS, CMAKE)
    base = dict(stage.source.blobs(P, PATCH_BASE, paths))
    require(all(base[k] == payload[k] for k in paths) and FIXTURE not in payload,
            'resident-input patch base differs from the original compiled inputs')
    changes = dict(stage.source.blobs(P, INPUT_SOURCE, (*paths, FIXTURE)))
    payload.update(changes)
    fixtures = dict(stage.source.blobs(P, PV_SOURCE, (CMAKE, PV_GENERATOR, *PV_FIXTURES)))
    require(fixtures.pop(PV_GENERATOR) == (P / PV_GENERATOR).read_bytes() and PV_KERNEL not in payload,
            'QSA component generator or destination differs')
    payload.update(fixtures)
    payload[PV_KERNEL] = probability.candidate(payload['src/kernels/sycl/qsa_bf16_prefix.cpp'])
    origin, target = str(previous.ROOT), str(ROOT)
    original = old['recipe']
    require(original['targets'] == stage.TARGETS and original['configure'][2] == origin + '/source'
            and original['configure'][4] == origin + '/build', 'parent configure paths or targets differ')
    recipe = dict(configure=[v.replace(origin, target) for v in original['configure']],
                  environment={k: v.replace(origin, target) for k, v in original['environment'].items()},
                  targets=TARGETS)
    recipe['configure'].append('-DSTRATA_QSA_PREFILL_PV_SOURCE:FILEPATH=' + str(ROOT / 'source' / PV_KERNEL))
    return payload, recipe


def comparison_closed():
    value = basis.validation.committed(RESULT, (COMPARISON,))
    require(value['stage'] == 'closed' and value['passed'] and value['complete']
            and not value['active'] and value['ownership_closed'] and len(value['processes']) == 9
            and all(p['passed'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in value['processes']), 'complete the ordinary store comparison')
    return signature({name: value[name] for name in ('source_commit', 'binary', 'plan', 'processes', 'decision')})


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(RESULT), 'fresh native build required')
    cohort = comparison_closed()
    old = previous.historic(BASE_EVIDENCE, previous.KEY)
    retained = basis.validation.committed(RESULT, (previous.KEY,))
    retirement = retained.get('artifact_retirement', {})
    require({k: v for k, v in retained.items() if k != 'artifact_retirement'} == old
            and retirement.get('complete') and retirement.get('all_actual_inputs_checked')
            and retirement.get('checked_ELF') == old['binaries']['strata']['sha256']
            and retirement.get('source_sha256') == old['source_files']['sha256']
            and not previous.ROOT.exists(), 'close the prior build consumers and retire that generation first')
    native, foundation, _ = basis.basis()
    payload, recipe = image(native)
    pins = stage.source.pin(P, ['tools/sycl', *basis.POLICIES])
    decode = builder.decode_reference(recover=True)
    row = dict(stage='prepared', source_commit=pins['revision'], native_source_commit=INPUT_SOURCE,
               qsa_fixture_commit=PV_SOURCE,
               native_base_commit=basis.original.SOURCE, parent_evidence=BASE_EVIDENCE,
               comparison_sha256=cohort, foundation=foundation, controller_sources=pins['source_sha256'],
               source_files=stage.hashed(payload), recipe=recipe, external_inputs=native['external_inputs'],
               driver=builder.guards.driver_guard(), retained_decode_reference=decode, commands=[],
               active=False, ownership_closed=True, passed=False, complete=False,
               plan=dict(unit='Only the input-borrow engine overlays enter the model. The separate QSA '
                              'component targets reuse its reference libraries and are built on demand.',
                         build='New directory, no object reuse, dependencies and all actual link inputs checked.',
                         validation='Host control, traced input-correctness device fixture, model numerical and '
                                    'route cases before timing the four independent input/output combinations.',
                         retention='Third generation alongside the independently retained prefill and decode '
                                   'references. Review at first failed gate or the completed four-arm decision.'))
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'tmp', 'cache', 'owned'):
        (ROOT / name).mkdir(mode=0o700)
    for name, content in payload.items():
        target = ROOT / 'source' / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(content)
    check(row, False)
    builder.save(row)


def check(row, committed=True):
    if committed:
        require(basis.validation.committed(RESULT, (KEY,)) == builder.compact(row), 'commit the exact build stage')
    native, foundation, _ = basis.basis(expected=row['foundation'])
    payload, recipe = image(native)
    require(row['native_source_commit'] == INPUT_SOURCE and row['parent_evidence'] == BASE_EVIDENCE
            and row['qsa_fixture_commit'] == PV_SOURCE
            and row['comparison_sha256'] == comparison_closed() and row['foundation'] == foundation
            and row['controller_sources'] == stage.source.pin(P, list(row['controller_sources']))['source_sha256']
            and row['source_files'] == stage.hashed(payload) and row['recipe'] == recipe
            and row['external_inputs'] == native['external_inputs'] and row['driver'] == builder.guards.driver_guard()
            and row['retained_decode_reference'] == builder.decode_reference(), 'pinned build inputs differ')
    require(all(sha(ROOT / 'source' / n) == h for n, h in row['source_files'].items())
            and all(sha(Path(dep['root']) / n) == h for dep in native['dependencies']
                    for n, h in dep['source_sha256'].items())
            and all(sha(p) == h for p, h in row.get('configured_files', {}).items()), 'source or dependency changed')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        require(actual['uncovered'] == 0 and not actual['partial_archive_reuse']
                and str(ROOT / 'source' / FIXTURE) in actual['compiled']
                and set(row['binaries']) == set(TARGETS)
                and all(sha(v['path']) == v['sha256'] for v in row['binaries'].values())
                and all(sha(p) == h for group in previous.ACTUAL for p, h in actual[group].items()),
                'fresh objects, fixture, link inputs or ELF differ')
    return native


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        files = [build / name for name in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        files.extend((build / 'CMakeFiles').glob('*.ninja'))
        row['configured_files'] = {str(path): sha(path) for path in files}
    stage.save_build(row)


def device_fixture():
    row = previous.read(ROOT / 'record.json')
    native = check(row)
    require(row['stage'] == 'built' and row['passed'] and row['complete'] and row['ownership_closed']
            and not row['active'] and row['host_control']['passed'], 'closed coherent build and host control required')
    key = 'prefill_resident_input_device_v1'
    require(key not in basis.validation.committed(RESULT), 'inspect the previous fixture outcome')
    root = ROOT / 'input-device'
    root.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (root / name).mkdir(mode=0o700)
    probe = native['native_validation_probe']
    env = dict(row['recipe']['environment'], **device.FIXTURE_ENV)
    env.update(TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'), LD_DEBUG='libs',
               LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
               UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_TRACING='level:info;flush:info;output:stderr',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    binary = row['binaries'][TARGET]
    args = [binary['path'], 'input-correctness']
    owner = builder.ownership.Owner(root / 'owned', lambda: stage.write(root / 'ownership.json', owner.commands))
    cursor, = (line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: '))
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    outcome = dict(source=INPUT_SOURCE, build_source=row['source_commit'], binary=binary,
                   native_build_sha256=signature(builder.compact(row)), command=args,
                   environment_sha256=signature(env), budgets=device.SMALL)
    try:
        command, out, err = owner.run('input-correctness', args, env, P, **device.SMALL)
        require(builder.ownership.closed(command) and command['exit_code'] == 0, 'input fixture failed')
        log = err.read_text()
        require('urEnqueue' in log and (not (root / 'validation.log').exists()
                or (root / 'validation.log').stat().st_size == 0), 'missing trace or validation warning')
        validation = basis.validation.preload_evidence(log, command['pid'])
        require(validation['loader'] == probe['loader']['path'], 'fixture loader changed')
        device.diagnostics.diagnostics(log, command['pid'])
        providers = device.loaded_providers(log, dict(environment=env, native_validation_probe=probe,
                                                       runtime_manifest=row['foundation']['runtime_manifest']))
        sizes = [1, 63, 7905, 8191, 8192]
        required = [f'attention input rows={n} copy_and_direct=1 bits={n * 10240} guards=1' for n in sizes]
        required.extend(('attention window rows=11 pieces=8+3 arms=4 checked_final_callback=1',
                         'attention_bank_fixture=PASS'))
        observed = [line for line in out.read_text().splitlines() if line.startswith(('attention ', 'attention_bank_fixture='))]
        require(observed == required, 'incomplete BF16, guard or owner-transition cases')
        outcome.update(native_exit_code=0, rows=sizes, resident_window=[8, 3], arms=4,
                       validation=validation, providers=providers, host_probe_accepted=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        basis.validation.finish(outcome, owner, cursor, boot)
        outcome['passed'] &= outcome.get('driver_after') == row['driver']
        data = previous.read(RESULT)
        data[key] = outcome
        stage.write(RESULT, data)
    require(outcome['passed'], outcome.get('error', 'same-build device checks did not pass'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', 'configure', 'compile', 'host', 'device'), 'prepare|configure|compile|host|device')
    builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, TARGETS
    builder.check, builder.save = check, save
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        elif action == 'host':
            stage.host(root_path=ROOT, check_build=check, save_record=save)
        elif action == 'device':
            device_fixture()
        else:
            builder.execute(action)
