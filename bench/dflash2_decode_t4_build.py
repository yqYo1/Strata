"""Build the same-binary DFlash2 decode T4 schedule candidate."""

import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_result_copy as fixture
import dflash2_primary_retained as retained

primary = retained.PRIMARY
bank, builder, require = primary.bank, primary.builder, primary.require
source, validation = bank.stage.source, bank.basis.validation
KEY = 'dflash2_decode_t4_native_v3'
ROOT = bank.B / KEY.replace('_', '-')
TREE = ROOT / 'source'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
TARGETS = tuple(dict.fromkeys((*primary.TARGETS, fixture.TARGET, 'sycl_verify_window_policy_test')))
REQUIRED = ('src/program/generate.cpp', 'src/core/verify.cpp', 'src/core/expert_source.cpp',
            'src/kernels/cpu/pool.cpp', 'src/kernels/sycl/shared_expert.cpp', fixture.FIXTURE)
CPU_EVIDENCE = '7d38b06800cf6eb344e5a9215ec2bf3efd03eb8f'
ARC_EVIDENCE = 'd7c710d12'
Q6_EVIDENCE, Q6_KEY = '697559569', 'dflash2_decode_t4_q6_control_v1'
C2_KEY = 'q6_t4_c2_component_build_v2'
C2_NATIVE = 'dflash2_decode_t4_c2_native_v1'
C2_VERIFY_SHAPES = ['6144x2560', '2560x6144', '2560x10240', '2560x12288']
C2_KERNEL = 'src/kernels/sycl/q6_dpas_component'
C2_LAUNCHER = 'bench/dflash2_decode_t4_model.py'


def select_c2_generation():
    global KEY, ROOT, TREE
    KEY = C2_NATIVE
    ROOT = bank.B / KEY.replace('_', '-')
    TREE = ROOT / 'source'


def c2_component():
    proof = validation.committed(bank.RESULT, (C2_KEY,))
    require(retained.closed(proof, 'closed') and proof['numerical']['passed']
            and proof['decision']['eligible_shapes'] == C2_VERIFY_SHAPES
            and len(proof['timing']) == 6 and all(
                t['passed'] and t['ownership_closed'] and t['native_exit_code'] == 0 and t['fresh_faults'] == []
                for t in proof['timing']), 'close the four-shape C2 component comparison first')
    revision = proof['project']['revision']
    require(proof['binary']['source_commit'] == revision and all(
        source.git(bank.P, 'show', revision + ':' + path) == (bank.P / path).read_bytes()
        for path in (C2_KERNEL + '.cpp', C2_KERNEL + '.hpp')), 'use the component-tested C2 kernel source')
    return dict(result_sha256=bank.signature(proof), source_commit=revision, shapes=C2_VERIFY_SHAPES)


def inputs(revision, *, allow_c2_launcher_update=False):
    fallback = retained.baseline()
    parent = bank.previous.read(primary.ROOT / 'record.json')
    cpu_path = bank.RESULT.with_name('cpu-fixed-expert-handoff-current.json').relative_to(bank.P)
    cpu_raw = source.git(bank.P, 'show', f'{CPU_EVIDENCE}:{cpu_path}')
    cpu = json.loads(cpu_raw)
    arc = bank.previous.historic(ARC_EVIDENCE, 'dflash2_decode_split_publication_v1')
    require(cpu['stage'] == 'closed' and cpu['ownership_closed'] and not cpu['active']
            and [(r['gu'], r['width'], r['eligible']) for r in cpu['handoff_result']]
                == [(18, 4, True), (21, 4, True)]
            and retained.closed(arc, 'closed') and arc['split_publication_correctness_qualified'],
            'recover the closed CPU margins and Arc publication proof')

    original_path = builder.original.ROOT / 'record.json'
    require(bank.sha(original_path) == parent['foundation']['original_record_sha256'],
            'build foundation changed')
    original = bank.previous.read(original_path)
    require(original['external_inputs'] == parent['external_inputs']
            and all(bank.sha(p) == h for p, h in (original['toolchain'] | parent['build_tools']).items()),
            'compiler or SDK identity changed')
    for pin in original['dependencies']:
        root = Path(pin['root'])
        require(bank.stage.hashed(dict(source.blobs(root, pin['revision'], pin['source_sha256'])))
                == pin['source_sha256']
                and all(bank.sha(root / p) == h for p, h in pin['source_sha256'].items()),
                'pinned dependency changed')
    controls = [THIS, fixture.THIS, retained.THIS]
    candidate = {}
    if KEY == C2_NATIVE:
        candidate['c2_component'] = c2_component()
        controls.extend(('bench/dflash2_decode_t4_check.py', C2_LAUNCHER,
                         'data/expert-profile.bin'))
    pins = source.pin(bank.P, [*primary.PREFIXES, *controls])['source_sha256']
    payload = dict(source.blobs(bank.P, revision, pins))
    frozen_pins = bank.stage.hashed(payload)
    if allow_c2_launcher_update:
        # The model recipe can fix a launch-only controller while the engine
        # build, its materialized sources and actual compiler inputs stay pinned.
        changed = {path for path, digest in frozen_pins.items() if digest != pins[path]}
        require(KEY == C2_NATIVE and frozen_pins.keys() == pins.keys()
                and changed <= {THIS, C2_LAUNCHER}, 'unexpected change to the compiled C2 candidate')
        pins = frozen_pins
    else:
        require(frozen_pins == pins, 'commit the exact T4 candidate and controllers')
    old, new = str(primary.ROOT), str(ROOT)
    args = [a.replace(old, new) for a in parent['recipe']['configure']]
    env = {k: v.replace(old, new) for k, v in parent['recipe']['environment'].items()}
    require(args[2:5] == [str(TREE), '-B', str(ROOT / 'build')], 'unexpected primary build recipe')
    return dict(source_commit=revision, native_source_commit=revision, source_files=pins,
                controller_sources={p: pins[p] for p in controls},
                foundation=parent['foundation'], external_inputs=parent['external_inputs'],
                build_tools=parent['build_tools'], driver=parent['driver'],
                retained_binary=fallback['binary'], runtime_manifest=fallback['runtime_manifest'],
                native_validation_probe=fallback['native_validation_probe'],
                cpu_evidence_sha256=hashlib.sha256(cpu_raw).hexdigest(),
                arc_evidence_sha256=bank.signature(arc),
                recipe=dict(configure=args, environment=env, targets=list(TARGETS)), **candidate), payload, original


def check(row, committed=True, *, allow_c2_launcher_update=False):
    if allow_c2_launcher_update:
        require(retained.closed(row, 'built'), 'use the closed C2 build for a launcher correction')
    if committed:
        require(validation.committed(bank.RESULT, (KEY,)) == builder.compact(row),
                'commit the previous build stage')
    fields, _, original = inputs(row['source_commit'], allow_c2_launcher_update=allow_c2_launcher_update)
    require(all(row[k] == v for k, v in fields.items())
            and builder.guards.driver_guard() == row['driver']
            and {p.relative_to(TREE).as_posix() for p in TREE.rglob('*') if p.is_file()} == fields['source_files'].keys()
            and all(bank.sha(TREE / p) == h for p, h in fields['source_files'].items())
            and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'T4 source, recipe or runtime changed')
    if row['stage'] == 'built':
        proof = row['actual_inputs']
        required = (*REQUIRED, C2_KERNEL + '.cpp') if KEY == C2_NATIVE else REQUIRED
        if allow_c2_launcher_update:
            require(all(str(TREE / p) not in proof[kind] for p in (THIS, C2_LAUNCHER)
                        for kind in bank.previous.ACTUAL),
                    'launcher correction must preserve all compiled and linked inputs')
        require(not proof['partial_archive_reuse'] and proof['uncovered'] == 0
                and all(proof['compiled'].get(str(TREE / p)) == row['source_files'][p] for p in required)
                and all(bank.sha(p) == h for kind in bank.previous.ACTUAL for p, h in proof[kind].items())
                and set(row['binaries']) == set(TARGETS)
                and all(bank.sha(b['path']) == b['sha256'] and Path(b['path']).stat().st_size == b['bytes']
                        for b in row['binaries'].values()), 'require one coherent engine and same-build fixtures')
    return original


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        files = [build / p for p in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        files.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in files}
    primary.publish(row)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT),
            'one fresh T4 build generation')
    previous = bank.previous.historic(Q6_EVIDENCE, Q6_KEY)
    require(retained.closed(previous, 'closed') and not previous['changed_state_keys']['stock/split with DPAS off']
            and all(previous['changed_state_keys'][k] for k in ('stock on/off', 'split on/off')),
            'close the original dense DPAS comparison first')
    revision = source.git(bank.P, 'rev-parse', 'HEAD').decode().strip()
    fields, payload, _ = inputs(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('owned', 'tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, raw in payload.items():
        target = TREE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    row = dict(fields, stage='prepared', commands=[], active=False, passed=False,
               complete=False, ownership_closed=True,
               plan=dict(schedules=['stock', 'after-cpu', 'early'],
                         selection='Split only layers supported by the decode CPU pool format policy.',
                         numerical_reference=Q6_EVIDENCE,
                         first_device='Shared-plan half-combine check with validation before any model run.',
                         model='Commit a bounded same-binary parity plan before model execution.',
                         retention='Keep the existing speed reference and this candidate through parity.'))
    if KEY == C2_NATIVE:
        row['plan'].update(selection='C2 only in the four qualified Q6 shapes within split T4 decode.',
                           numerical_reference=C2_KEY,
                           retention='Keep the existing long-prefill and T4 references through the C2 trial.')
    check(row, committed=False)
    save(row)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--c2']:
        sys.argv.pop(1)
        select_c2_generation()
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON
            and action in ('prepare', 'configure', 'compile'), 'one pinned T4 build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, TARGETS
        builder.check, builder.save = check, save
        prepare() if action == 'prepare' else builder.execute(action)
