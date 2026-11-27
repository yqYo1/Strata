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
PREFIX_HELP_NATIVE = 'dflash2_decode_t4_prefix_help_native_v1'
PREFIX_HELP_CPU = '2c6c66ec82185b90faaa06684c57e13159717bc4'
PREFIX_HELP_SOURCE = '896f328535093d00c6b398fd22316577b2f45ceb'
PREFIX_HELP_REFERENCE = '4273c6292eeda5b2407c57e2cf15aed80e7f40a5'
PREFIX_HELP_FILES = ('include/strata/core/expert_source.hpp', 'include/strata/core/verify.hpp',
                     'include/strata/kernels/cpu/pool.hpp', 'src/core/expert_source.cpp',
                     'src/core/verify.cpp', 'src/kernels/cpu/pool.cpp')


def select_prefix_help_generation():
    global KEY, ROOT, TREE
    KEY = PREFIX_HELP_NATIVE
    ROOT = bank.B / KEY.replace('_', '-')
    TREE = ROOT / 'source'


def prefix_help_component(revision):
    path = bank.RESULT.with_name('cpu-fixed-expert-prefix-help-current.json').relative_to(bank.P)
    raw = source.git(bank.P, 'show', f'{PREFIX_HELP_CPU}:{path}')
    proof = json.loads(raw)
    timed = proof['processes'][2:]
    require(proof['stage'] == 'closed' and proof['ownership_closed'] and not proof['active']
            and not proof['gpu_executed'] and len(timed) == 6
            and all(p['owner']['exit_code'] == 0 and len(p['samples']) == 10 for p in timed)
            and [(f['gu'], f['eligible']) for f in proof['prefix_help_result']] == [(18, True), (21, True)],
            'close the measured first-pair CPU replay in both formats')
    require(all(source.git(bank.P, 'show', f'{revision}:{path}')
                == source.git(bank.P, 'show', f'{PREFIX_HELP_SOURCE}:{path}')
                for path in PREFIX_HELP_FILES), 'commit the reviewed T4 prefix-help model projection')
    changed = source.git(bank.P, 'diff', '--name-only', PREFIX_HELP_REFERENCE, revision, '--',
                         'CMakeLists.txt', 'cmake', 'include', 'src', 'third_party', 'tests').decode().splitlines()
    require(set(changed) == set(PREFIX_HELP_FILES), 'preserve the original early T4 native source outside the helper')
    return dict(result_commit=PREFIX_HELP_CPU, result_sha256=hashlib.sha256(raw).hexdigest(),
                source_revision=proof['source_commit'], reviewed_model_projection=PREFIX_HELP_SOURCE,
                reference_native_source=PREFIX_HELP_REFERENCE)


def inputs(revision):
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
    if KEY == PREFIX_HELP_NATIVE:
        candidate['prefix_help_component'] = prefix_help_component(revision)
        controls.extend(('bench/dflash2_decode_t4_check.py', 'bench/dflash2_decode_t4_model.py',
                         'data/expert-profile.bin'))
    pins = source.pin(bank.P, [*primary.PREFIXES, *controls])['source_sha256']
    payload = dict(source.blobs(bank.P, revision, pins))
    require(bank.stage.hashed(payload) == pins, 'commit the exact T4 candidate and controllers')
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


def check(row, committed=True):
    if committed:
        require(validation.committed(bank.RESULT, (KEY,)) == builder.compact(row),
                'commit the previous build stage')
    fields, _, original = inputs(row['source_commit'])
    require(all(row[k] == v for k, v in fields.items())
            and builder.guards.driver_guard() == row['driver']
            and {p.relative_to(TREE).as_posix() for p in TREE.rglob('*') if p.is_file()} == fields['source_files'].keys()
            and all(bank.sha(TREE / p) == h for p, h in fields['source_files'].items())
            and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'T4 source, recipe or runtime changed')
    if row['stage'] == 'built':
        proof = row['actual_inputs']
        require(not proof['partial_archive_reuse'] and proof['uncovered'] == 0
                and all(proof['compiled'].get(str(TREE / p)) == row['source_files'][p] for p in REQUIRED)
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
    if KEY == PREFIX_HELP_NATIVE:
        row['plan'].update(schedules=['early'],
                           selection='Compare a reserved caller with first-pair help on the same split graphs.',
                           numerical_reference=PREFIX_HELP_CPU)
    check(row, committed=False)
    save(row)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--prefix-help']:
        sys.argv.pop(1)
        select_prefix_help_generation()
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON
            and action in ('prepare', 'configure', 'compile'), 'one pinned T4 build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, TARGETS
        builder.check, builder.save = check, save
        prepare() if action == 'prepare' else builder.execute(action)
