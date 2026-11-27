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
KEY = 'dflash2_decode_t4_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
TREE = ROOT / 'source'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
TARGETS = ('strata', fixture.TARGET, 'sycl_verify_window_policy_test')
REQUIRED = ('src/program/generate.cpp', 'src/core/verify.cpp', 'src/core/expert_source.cpp',
            'src/kernels/cpu/pool.cpp', 'src/kernels/sycl/shared_expert.cpp', fixture.FIXTURE)
CPU_EVIDENCE = '7d38b06800cf6eb344e5a9215ec2bf3efd03eb8f'
ARC_EVIDENCE = 'd7c710d12'


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
    pins = source.pin(bank.P, [*primary.PREFIXES, THIS, fixture.THIS, retained.THIS])['source_sha256']
    payload = dict(source.blobs(bank.P, revision, pins))
    require(bank.stage.hashed(payload) == pins, 'commit the exact T4 candidate and controllers')
    old, new = str(primary.ROOT), str(ROOT)
    args = [a.replace(old, new) for a in parent['recipe']['configure']]
    env = {k: v.replace(old, new) for k, v in parent['recipe']['environment'].items()}
    require(args[2:5] == [str(TREE), '-B', str(ROOT / 'build')], 'unexpected primary build recipe')
    return dict(source_commit=revision, native_source_commit=revision, source_files=pins,
                controller_sources={p: pins[p] for p in (THIS, fixture.THIS, retained.THIS)},
                foundation=parent['foundation'], external_inputs=parent['external_inputs'],
                build_tools=parent['build_tools'], driver=parent['driver'],
                retained_binary=fallback['binary'], runtime_manifest=fallback['runtime_manifest'],
                native_validation_probe=fallback['native_validation_probe'],
                cpu_evidence_sha256=hashlib.sha256(cpu_raw).hexdigest(),
                arc_evidence_sha256=bank.signature(arc),
                recipe=dict(configure=args, environment=env, targets=list(TARGETS))), payload, original


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
                         first_device='Shared-plan half-combine check with validation before any model run.',
                         model='Commit a bounded same-binary parity plan before model execution.',
                         retention='Keep the existing speed reference and this candidate through parity.'))
    check(row, committed=False)
    save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON
            and action in ('prepare', 'configure', 'compile'), 'one pinned T4 build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, TARGETS
        builder.check, builder.save = check, save
        prepare() if action == 'prepare' else builder.execute(action)
