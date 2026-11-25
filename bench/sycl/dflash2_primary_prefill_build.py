"""Build the actual primary SYCL source for the matched DFlash2 checks."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_kv_upload_build as comparison

bank, builder, require = comparison.bank, comparison.builder, comparison.require
publish = builder.save
foundation_reader = builder.original.verify_original
KEY = 'dflash2_primary_prefill_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PREFILL = 'ad8652973c8f025e73bd07bc791a9976a55e20a9'
DECODE = 'd6f2b82947018f85e8e6169bd2072fb076c03f19'
SNAPSHOT_KEY = 'dflash2_target_snapshot_native_v2'
SNAPSHOT_ROOT = bank.B / SNAPSHOT_KEY.replace('_', '-')
OBSERVER_ROOT = bank.B / 'dflash2-tap-payload-native-v1'
PREFIXES = ('CMakeLists.txt', 'cmake', 'include', 'src', 'third_party', 'tests',
            'bench/sycl', 'tools/sycl', *bank.basis.POLICIES)
TARGETS = [*builder.TARGETS, 'sycl_conversation_transfer_test',
           'sycl_conversation_snapshot_test', 'sycl_dflash2_test']
REQUIRED_SOURCES = ('src/program/generate.cpp', 'src/prefill/prefill.cpp',
                    'src/prefill/sycl/dflash2_taps.cpp', 'src/core/conversation_state.cpp',
                    'src/core/conversation_validation_test.cpp', 'tests/sycl/dflash2.cpp')


def references():
    prefill = bank.previous.historic(PREFILL, 'dflash2_snapshot_direct_prefill_v1')
    decode = bank.previous.historic(DECODE, 'dflash2_snapshot_regular_code_v1')
    require(all(r['stage'] == 'closed' and r['passed'] and r['complete']
                and r['ownership_closed'] and not r['active'] for r in (prefill, decode))
            and prefill['decision']['all_warm_requests_1000']['snapshot']
            and decode['decision']['matched_work'], 'recover the two closed phase references')
    snapshot = bank.previous.read(SNAPSHOT_ROOT / 'record.json')
    retained = prefill['snapshot_reference']
    require(snapshot['binaries']['strata'] == retained['binary']
            and snapshot['native_source_commit'] == retained['source']
            and bank.signature(builder.compact(snapshot)) == retained['native_build_sha256'],
            'recover the exact measured snapshot recipe')
    controls = dict(prefill=retained['binary'], decode=decode['binary'])
    require(all(Path(b['path']).stat().st_size == b['bytes'] and bank.sha(b['path']) == b['sha256']
                for b in controls.values()), 'preserve the measured phase reference executables')
    return snapshot, controls


def recipe(snapshot):
    old, new = str(SNAPSHOT_ROOT), str(ROOT)
    original = snapshot['recipe']
    command = [v.replace(old, new) for v in original['configure']
               if not v.startswith('-DSTRATA_QSA_PREFILL_PV_SOURCE:')]
    switch = '-DSTRATA_BUILD_CONVERSATION_TESTS:BOOL='
    command[command.index(switch + 'OFF')] = switch + 'ON'
    require(command[2:5] == [new + '/source', '-B', new + '/build'],
            'relocate the exact SYCL configure recipe')
    return dict(configure=command, environment={k: v.replace(old, new)
                for k, v in original['environment'].items()}, targets=TARGETS)


def source_pins():
    return bank.stage.source.pin(bank.P, PREFIXES)


def check(row, committed=True):
    if committed:
        require(bank.basis.validation.committed(bank.RESULT, (KEY,)) == builder.compact(row),
                'commit the preceding primary build stage')
    snapshot, controls = references()
    native, foundation = foundation_reader(expected=row['foundation'])
    current = source_pins()['source_sha256']
    recovered = bank.stage.hashed(dict(bank.stage.source.blobs(bank.P, row['native_source_commit'], current)))
    require(current == recovered == row['source_files'] and row['controller_sources'] == {THIS: current[THIS]}
            and row['native_source_commit'] == row['source_commit'] and row['foundation'] == foundation
            and row['retained_references'] == controls and row['recipe'] == recipe(snapshot)
            and row['external_inputs'] == native['external_inputs'] == snapshot['external_inputs']
            and row['driver'] == builder.guards.driver_guard()
            and all(bank.sha(p) == h for p, h in row['build_tools'].items()),
            'primary tree, reference, build recipe or platform changed')
    tree = ROOT / 'source'
    require({p.relative_to(tree).as_posix() for p in tree.rglob('*') if p.is_file()} == current.keys()
            and all(bank.sha(tree / p) == h for p, h in current.items())
            and all(bank.sha(Path(pin['root']) / p) == h for pin in native['dependencies']
                    for p, h in pin['source_sha256'].items())
            and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'materialized primary source or compiler inputs changed')
    if row['stage'] == 'built':
        census = row['actual_inputs']
        require(census['uncovered'] == 0 and not census['partial_archive_reuse']
                and all(census['compiled'].get(str(tree / p)) == current[p] for p in REQUIRED_SOURCES)
                and all(bank.sha(p) == h for kind in bank.previous.ACTUAL for p, h in census[kind].items())
                and set(row['binaries']) == set(TARGETS)
                and all(Path(b['path']).stat().st_size == b['bytes'] and bank.sha(b['path']) == b['sha256']
                        for b in row['binaries'].values()), 'require a coherent primary build and same-build fixtures')
    return native


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        files = [build / name for name in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        files.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in files}
    publish(row)


def prepare():
    require(not ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT),
            'inspect the existing primary build before preparing')
    require(not (OBSERVER_ROOT / 'build' / 'strata').exists(),
            'retire the closed observer generation after exact source recovery')
    snapshot, controls = references()
    native, foundation = foundation_reader()
    pins = source_pins()
    files, revision = pins['source_sha256'], pins['revision']
    row = dict(stage='prepared', source_commit=revision, native_source_commit=revision,
               controller_sources={THIS: files[THIS]}, source_files=files, foundation=foundation,
               retained_references=controls, prefill_evidence=PREFILL, decode_evidence=DECODE,
               recipe=recipe(snapshot), external_inputs=native['external_inputs'], build_tools=snapshot['build_tools'],
               driver=builder.guards.driver_guard(), active=False, passed=False, complete=False,
               ownership_closed=True, commands=[],
               scope='Build the primary tree without historical overlays. Check aligned KV retention, '
                     'tap payloads, target snapshots and the unchanged decode path using this source.',
               retention='Keep the measured prefill and decode references through the matched checks. '
                         'Review this third generation at the first failed gate or comparison closure.')
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'owned', 'tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, data in bank.stage.source.blobs(bank.P, revision, files):
        target = ROOT / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    check(row, committed=False)
    save(row)


def main(action):
    require(action in ('prepare', 'configure', 'compile'), 'select one primary build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, TARGETS
        builder.check, builder.save = check, save
        if action == 'prepare':
            prepare()
        else:
            builder.execute(action)


if __name__ == '__main__':
    main(*sys.argv[1:])
