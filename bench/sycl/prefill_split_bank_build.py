"""Build the independent split-prefill bank-capacity candidate on the measured image."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_large_gather_model_source as stock

bank = stock.bank
builder, require = bank.builder, bank.require
ROOT = bank.B / 'prefill-split-bank-native-v1'
KEY = 'prefill_split_bank_native_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
HISTORY = '568d015bae99e9aa0455ffedabe9a3f61a70f1f3'
SOURCE = 'd8b327c06d88d5347caa0ee594f94a4798a7c767'
LAYER_MAJOR = 'src/prefill/layer_major.inc'
READ = 117489
SLOTS = (128, 1, 1, 128, 128, 1)


def project(expected=None):
    native, foundation, source, recipe, _ = stock.build_inputs(expected=expected)
    parent = bank.previous.historic(HISTORY, stock.KEY)
    hashes = bank.stage.hashed(source)
    require(parent['native_source_commit'] == stock.DISPATCH_COUNTS
            and parent['stage'] == 'built' and parent['passed'] and parent['complete']
            and parent['host_control']['passed'] and parent['ownership_closed'] and not parent['active']
            and parent['source_files'] == dict(files=len(hashes), sha256=bank.signature(hashes))
            and parent['recipe'] == recipe and parent['external_inputs'] == native['external_inputs'],
            'recover the exact coherent native image used in the 117K measurement')
    before = stock.fixture.git_file(SOURCE + '^', LAYER_MAJOR)
    after = stock.fixture.git_file(SOURCE, LAYER_MAJOR)
    require(source[LAYER_MAJOR] == before and before != after,
            'the slot change must apply to the actual 117K source without replacing another change')
    source[LAYER_MAJOR] = after
    old, new = str(stock.MODEL_ROOT), str(ROOT)
    relocated_recipe = dict(configure=[arg.replace(old, new) for arg in recipe['configure']],
                            environment={k: v.replace(old, new) for k, v in recipe['environment'].items()},
                            targets=recipe['targets'])
    proof = dict(parent_evidence=HISTORY, parent_source=stock.DISPATCH_COUNTS,
                 parent_binary=parent['binaries']['strata'], parent_source_files=parent['source_files'],
                 changed_path=LAYER_MAJOR, old_sha256=hashes[LAYER_MAJOR],
                 new_sha256=bank.stage.hashed({LAYER_MAJOR: after})[LAYER_MAJOR])
    return native, foundation, source, relocated_recipe, proof


def check(row, committed=True):
    if committed:
        require(bank.basis.validation.committed(bank.RESULT, (KEY,)) == builder.compact(row),
                'commit the preceding build stage')
    native, foundation, source, recipe, proof = project(row['foundation'])
    require(row['controller_sources'] == bank.stage.source.pin(bank.P, list(row['controller_sources']))['source_sha256']
            and row['native_source_commit'] == SOURCE and row['source_files'] == bank.stage.hashed(source)
            and row['foundation'] == foundation and row['projection'] == proof and row['recipe'] == recipe
            and row['external_inputs'] == native['external_inputs']
            and bank.sha(row['external_inputs']['path']) == row['external_inputs']['sha256']
            and all(bank.sha(p) == h for p, h in row['build_tools'].items())
            and row['driver'] == builder.guards.driver_guard()
            and row['retained_decode_reference'] == builder.decode_reference(),
            'frozen build inputs or independent reference changed')
    materialized = ROOT / 'source'
    require({p.relative_to(materialized).as_posix() for p in materialized.rglob('*') if p.is_file()}
            == row['source_files'].keys()
            and all(bank.sha(materialized / n) == h for n, h in row['source_files'].items())
            and all(bank.sha(Path(dep['root']) / n) == h
                    for dep in native['dependencies'] for n, h in dep['source_sha256'].items())
            and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'materialized engine, dependency or configured input changed')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        require(not actual['partial_archive_reuse'] and actual['uncovered'] == 0
                and actual['compiled'].get(str(materialized / LAYER_MAJOR)) == proof['new_sha256']
                and set(row['binaries']) == set(recipe['targets'])
                and all(bank.sha(p) == h for group in bank.previous.ACTUAL for p, h in actual[group].items())
                and all(bank.sha(b['path']) == b['sha256'] for b in row['binaries'].values()),
                'the fresh prefill translation unit and final executables must be verified')
    return native


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        paths = [build / n for n in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        paths.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in paths}
    bank.stage.write(ROOT / 'record.json', row)
    results = bank.previous.read(bank.RESULT)
    results[KEY] = builder.compact(row)
    bank.stage.write(bank.RESULT, results)


def prepare():
    require(not ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT),
            'inspect the preceding candidate before creating a build')
    native, foundation, source, recipe, proof = project()
    stock_controls = [str(Path(m.__file__).resolve().relative_to(bank.P))
                      for m in (stock, stock.fixture, stock.kv)]
    controllers = [THIS, LAYER_MAJOR, *stock_controls]
    pins = bank.stage.source.pin(bank.P, [*controllers, 'tools/sycl', *bank.basis.POLICIES])
    require(pins['source_sha256'][LAYER_MAJOR] == proof['new_sha256'], 'working source differs from the candidate commit')
    row = dict(stage='prepared', source_commit=pins['revision'], controller_sources=pins['source_sha256'],
               native_source_commit=SOURCE, source_files=bank.stage.hashed(source),
               foundation=foundation, projection=proof, recipe=recipe, external_inputs=native['external_inputs'],
               build_tools={p: bank.sha(p) for p in ('/usr/bin/cmake', '/usr/bin/ninja')},
               driver=builder.guards.driver_guard(), retained_decode_reference=builder.decode_reference(recover=True),
               commands=[], active=False, passed=False, complete=False, ownership_closed=True,
               plan=dict(read=READ, clean_order=SLOTS,
                         scope='Same ELF, aligned KV reuse, scalar expert gather and fixed decode settings. '
                               'Compare actual 128-slot and one-slot admission with guards and decode loans checked.',
                         retention='Keep the validated phase references. Retire the candidate generation '
                                   'after the bank-capacity decision or its first failed gate.'))
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'tmp', 'cache', 'owned'):
        (ROOT / name).mkdir(mode=0o700)
    for name, content in source.items():
        path = ROOT / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    check(row, False)
    save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', 'configure', 'compile', 'host'), 'select one build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        elif action == 'host':
            bank.stage.host(root_path=ROOT, check_build=check, save_record=save)
        else:
            row = bank.previous.read(ROOT / 'record.json')
            builder.ROOT, builder.TARGETS = ROOT, row['recipe']['targets']
            builder.check, builder.save = check, save
            builder.execute(action)
