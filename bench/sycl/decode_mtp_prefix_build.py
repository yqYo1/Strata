"""Rebuild the measured prefill image with the bounded decode MTP candidate."""

import fcntl
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
import decode_mtp_context_reference as reference
import decode_mtp_prefix_source as source
import prefill_large_gather_model_source as stock

bank = stock.bank
builder, require = bank.builder, bank.require
ROOT = bank.B / 'decode-mtp-prefix-native-v1'
KEY = 'decode_mtp_prefix_native_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PLAN = dict(unit='Project only the two bounded MTP translation units onto the measured prefill image.',
    build='Fresh objects for the original target list and recipes in an isolated build root.',
    qualification='Keep committed-prefix selection off for the CLI boundary comparison. Then compare the '
                  'two prefix selections on the same fixed binary.',
    retention='The retained fast prefill supplies the reference CLI and first-use/steady-state controls. '
              'Keep the independent decode fallback through the decision. Retire the temporary third '
              'generation at a failed gate or after matched safety, phase and speed decisions.')


def reference_gate():
    row = bank.basis.validation.committed(bank.RESULT, (reference.KEY,))
    outcome = row['outcome']
    expected = reference.inputs()
    require(row['stage'] == 'closed' and row['passed'] and row['complete'] and row['ownership_closed']
            and not row['active'] and row['plan'] == reference.PLAN
            and all(row[k] == v for k, v in expected.items() if k not in ('model_assets', 'environment_template'))
            and row['model_assets_sha256'] == bank.signature(expected['model_assets'])
            and row['environment_template_sha256'] == bank.signature(expected['environment_template'])
            and outcome['native_exit_code'] == reference.PLAN['expected_exit']
            and outcome['refusal_position'] == reference.TOKENS and outcome['prefill_tokens'] == reference.TOKENS - 1
            and outcome['ownership_closed'] and not outcome['fresh_faults'], 'close the original exact-fit CLI refusal')
    return bank.signature(row)


def projection(expected=None):
    native, foundation, files, recipe, proof = stock.build_inputs(expected=expected)
    original = bank.stage.hashed(files)
    with tempfile.TemporaryDirectory(prefix='strata-mtp-build-source-') as work:
        directory = Path(work)
        parent, _ = source.recipe(directory)
        require(parent['source_files'] == original and parent['recipe'] == recipe
                and parent['native_source_commit'] == stock.DISPATCH_COUNTS,
                'recover the entire measured prefill composition')
        require(all(files[name] == (stock.MODEL_ROOT / 'source' / name).read_bytes()
                    for name in source.PROJECTED), 'the two edited sources require the exact compiled base')
        files.update({name: (directory / name).read_bytes() for name in source.PROJECTED})
    hashes = bank.stage.hashed(files)
    require(set(hashes) == set(original) and {name: h for name, h in hashes.items() if h != original[name]}
            == source.PROJECTED, 'unexpected engine or build inputs in the MTP projection')
    old, new = str(stock.MODEL_ROOT), str(ROOT)
    child = dict(configure=[value.replace(old, new) for value in recipe['configure']],
                 environment={name: value.replace(old, new) for name, value in recipe['environment'].items()},
                 targets=recipe['targets'])
    return native, foundation, files, child, dict(parent_source=stock.DISPATCH_COUNTS,
        parent_source_sha256=proof['source_sha256'], parent_binary=parent['binaries']['strata'],
        boundary_source=source.BOUNDARY, width_source=source.CHANGE, overlays=source.PROJECTED)


def check(row, committed=True):
    if committed:
        require(bank.basis.validation.committed(bank.RESULT, (KEY,)) == builder.compact(row),
                'commit the exact preceding build stage')
    native, foundation, files, recipe, proof = projection(row['foundation'])
    require(row['reference_gate_sha256'] == reference_gate()
            and row['controller_sources'] == bank.stage.source.pin(bank.P, list(row['controller_sources']))['source_sha256']
            and row['foundation'] == foundation and row['projection'] == proof and row['plan'] == PLAN
            and row['native_source_commit'] == source.BOUNDARY and row['recipe'] == recipe
            and row['source_files'] == bank.stage.hashed(files) and row['external_inputs'] == native['external_inputs']
            and row['retained_decode_reference'] == builder.decode_reference()
            and row['driver'] == builder.guards.driver_guard(), 'build sources, recipe or retained reference differs')
    materialized = ROOT / 'source'
    actual_files = {p.relative_to(materialized).as_posix() for p in materialized.rglob('*') if p.is_file()}
    require(actual_files == files.keys() and all(bank.sha(materialized / p) == h for p, h in row['source_files'].items())
            and all(bank.sha(Path(dep['root']) / p) == h for dep in native['dependencies']
                    for p, h in dep['source_sha256'].items())
            and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'materialized source, dependency or configure input differs')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        require(actual['uncovered'] == 0 and not actual['partial_archive_reuse']
                and all(actual['compiled'].get(str(materialized / name)) == h for name, h in source.PROJECTED.items())
                and set(row['binaries']) == set(recipe['targets'])
                and all(bank.sha(v['path']) == v['sha256'] for v in row['binaries'].values())
                and all(bank.sha(p) == h for group in bank.previous.ACTUAL for p, h in actual[group].items()),
                'fresh MTP/compiler/link/ELF proof differs')
    return native


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        files = [build / name for name in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        files.extend((build / 'CMakeFiles').glob('*.ninja'))
        row['configured_files'] = {str(path): bank.sha(path) for path in files}
    bank.stage.save_build(row)


def prepare():
    require(not ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT), 'one fresh build')
    gate = reference_gate()
    controllers = [THIS, reference.THIS, *reference.previous.SOURCES,
                   *(str(Path(m.__file__).resolve().relative_to(bank.P)) for m in (source, stock, stock.fixture, stock.kv))]
    pins = bank.stage.source.pin(bank.P, controllers)
    native, foundation, files, recipe, proof = projection()
    row = dict(stage='prepared', source_commit=pins['revision'], native_source_commit=source.BOUNDARY,
               controller_sources=pins['source_sha256'], source_files=bank.stage.hashed(files), projection=proof,
               foundation=foundation, recipe=recipe, external_inputs=native['external_inputs'],
               driver=builder.guards.driver_guard(), retained_decode_reference=builder.decode_reference(recover=True),
               reference_gate_sha256=gate, plan=PLAN, commands=[], active=False, ownership_closed=True,
               complete=False, passed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'tmp', 'cache', 'owned'):
        (ROOT / name).mkdir(mode=0o700)
    for name, content in files.items():
        dest = ROOT / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    check(row, committed=False)
    save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', 'configure', 'compile') and sys.executable == bank.basis.reference.model.PYTHON,
            'use the pinned interpreter and one build stage')
    builder.ROOT, builder.KEY, builder.check, builder.save = ROOT, KEY, check, save
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        else:
            builder.TARGETS = bank.previous.read(ROOT / 'record.json')['recipe']['targets']
            builder.execute(action)
