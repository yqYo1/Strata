"""Build the isolated cooperative decode KV resolver on the measured prefill image."""

import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import prefill_large_gather_model_source as stock

bank = stock.bank
builder, require = bank.builder, bank.require
save_build = builder.save
ROOT = bank.B / 'decode-kv-fast-prefill-native-v1'
KEY = 'decode_kv_fast_prefill_native_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PHASE = '644412c8edc625b1c634a8ecfa2fe6585aa12d67'
ISOLATED = '93b964caeb829f909624bf6dc625fe9ad88c5718'
PARENT_ELF = '85e3f202ef99661d52c03a17a8b609b8a8a44e9f584bfa8fc1a5c1c3d51aea7c'
VERIFY = 'src/core/verify.cpp'
POLICY = 'include/strata/core/verify_window_policy.hpp'
PHASE_PATHS = (VERIFY, POLICY, 'include/strata/core/verify.hpp')
SCALAR = 'src/kernels/sycl/kv_stream.cpp'
COOPERATIVE = 'src/kernels/sycl/decode_kv_stream.cpp'
HEADER = 'include/strata/kernels/kv_stream.hpp'
CMAKE = 'cmake/sycl_backend.cmake'
PROJECTED = {
    VERIFY: '1c2dfebe0a81e63255a2f1f5d9b4bd6a3839f875d0e148efe97f6fe3a69b17f2',
    POLICY: 'be517d179213990d646d8d40f1843298d2e9c8df43f014a5eb41d87531dd87a4',
    'include/strata/core/verify.hpp': 'd0e6b46ac93d2451de3696aaa7ccdf788db71e02195b2aa867794570e3a7cf25',
    HEADER: 'b5a8ec1431ae747587a6a65d3657201c5d94184c03e5a70cef3bf08db371d525',
    CMAKE: 'd850eb227e5936b5d8e6815d484f5ac0f714b33e6fd0171dde5a181f660b7f5a',
    COOPERATIVE: '40b4962e1d39db8e1a779d03f3a0150189c38589562203e106662c791ba238d1',
}


def replace_once(source, text, replacement):
    require(source.count(text) == 1, 'unexpected source context for ' + text.decode().strip())
    return source.replace(text, replacement, 1)


def project(expected=None):
    native, foundation, source, recipe, _ = stock.build_inputs(expected=expected)
    record_path = stock.MODEL_ROOT / 'record.json'
    parent = bank.previous.read(record_path)
    basis = bank.stage.hashed(source)
    reference = parent['binaries']['strata']
    require(parent['native_source_commit'] == stock.DISPATCH_COUNTS and parent['stage'] == 'built'
            and parent['passed'] and parent['complete'] and parent['ownership_closed'] and not parent['active']
            and parent['host_control']['passed']
            and parent['source_files'] == basis and parent['recipe'] == recipe
            and parent['external_inputs'] == native['external_inputs']
            and reference['sha256'] == PARENT_ELF and bank.sha(reference['path']) == PARENT_ELF,
            'retain and recover the exact fast prefill reference')
    require(COOPERATIVE not in source and all(path in source for path in PROJECTED if path != COOPERATIVE),
            'unexpected cooperative KV projection base')
    patch = subprocess.check_output(['git', 'diff', PHASE + '^', PHASE, '--', *PHASE_PATHS],
                                    cwd=bank.P, timeout=30)
    with tempfile.TemporaryDirectory(prefix='strata-decode-kv-projection-') as tmp:
        directory = Path(tmp)
        for name in PHASE_PATHS:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source[name])
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', '-'],
                       cwd=directory, input=patch, check=True, timeout=30)
        changed = {name: (directory / name).read_bytes() for name in PHASE_PATHS}
    for name in (VERIFY, POLICY):
        changed[name] = replace_once(changed[name], b'verify_coalesced_kv(', b'decode_verify_coalesced_kv(')
    changed[VERIFY] = replace_once(changed[VERIFY], b'kv_stream_resolve_with_coalesced_plan(',
                                   b'decode_kv_stream_resolve_coalesced(')
    marker = b'/// The static ring table `block -> block % n_slots`.'
    original = stock.fixture.git_file(ISOLATED, HEADER)
    declaration = original[original.index(b'#ifdef STRATA_ENABLE_SYCL\n/// Opt-in SYCL decode candidate:'):
                           original.index(marker)]
    changed[HEADER] = replace_once(source[HEADER], marker, declaration + marker)
    after = b'target_sources(strata_kernels PRIVATE src/kernels/sycl/dflash2.cpp)\n'
    link = b'target_sources(strata_kernels PRIVATE src/kernels/sycl/decode_kv_stream.cpp)\n'
    changed[CMAKE] = replace_once(source[CMAKE], after, after + link)
    changed[COOPERATIVE] = stock.fixture.git_file(ISOLATED, COOPERATIVE)
    require(bank.stage.hashed(changed) == PROJECTED, 'projected cooperative KV sources differ')
    source.update(changed)
    require({name for name, digest in bank.stage.hashed(source).items() if basis.get(name) != digest}
            == set(PROJECTED) and basis[SCALAR] == bank.stage.hashed({SCALAR: source[SCALAR]})[SCALAR],
            'shared scalar KV source or unrelated source changed')
    old, new = str(stock.MODEL_ROOT), str(ROOT)
    recipe = dict(configure=[s.replace(old, new) for s in recipe['configure']],
                  environment={k: v.replace(old, new) for k, v in recipe['environment'].items()},
                  targets=recipe['targets'])
    parent_objects = parent['actual_inputs']['objects']
    expected_objects = {Path(p).relative_to(stock.MODEL_ROOT / 'build').as_posix() for p in parent_objects}
    expected_objects.add(f'CMakeFiles/strata_kernels.dir/{COOPERATIVE}.o')
    proof = dict(parent_source=stock.DISPATCH_COUNTS, parent_record_sha256=bank.sha(record_path),
                 parent_binary=reference, phase=PHASE, isolated=ISOLATED, projected=PROJECTED,
                 object_count=len(expected_objects), object_set_sha256=bank.signature(sorted(expected_objects)),
                 unchanged_scalar_sha256=basis[SCALAR])
    return native, foundation, source, recipe, proof


def check(row, committed=True):
    if committed:
        require(bank.basis.validation.committed(bank.RESULT, (KEY,)) == builder.compact(row),
                'commit the exact preceding build stage')
    native, foundation, source, recipe, proof = project(row['foundation'])
    require(row['source_commit'] == row['native_source_commit']
            and row['controller_sources'] == bank.stage.source.pin(bank.P, list(row['controller_sources']))['source_sha256']
            and row['recipe'] == recipe and row['foundation'] == foundation and row['projection'] == proof
            and row['source_files'] == bank.stage.hashed(source) and row['external_inputs'] == native['external_inputs']
            and row['driver'] == builder.guards.driver_guard()
            and row['retained_decode_reference'] == builder.decode_reference()
            and all(bank.sha(p) == h for p, h in row['build_tools'].items()), 'build inputs changed')
    tree = ROOT / 'source'
    require({p.relative_to(tree).as_posix() for p in tree.rglob('*') if p.is_file()} == row['source_files'].keys()
            and all(bank.sha(tree / p) == h for p, h in row['source_files'].items())
            and all(bank.sha(Path(dep['root']) / p) == h for dep in native['dependencies']
                    for p, h in dep['source_sha256'].items())
            and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'materialized source, dependency or configured input changed')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        objects = sorted(Path(p).relative_to(ROOT / 'build').as_posix() for p in actual['objects'])
        require(not actual['partial_archive_reuse'] and actual['uncovered'] == 0
                and len(objects) == proof['object_count'] == actual['fresh_objects']
                and bank.signature(objects) == proof['object_set_sha256']
                and all(actual['compiled'].get(str(tree / p)) == row['source_files'][p]
                        for p in (VERIFY, POLICY, SCALAR, COOPERATIVE))
                and set(row['binaries']) == set(recipe['targets'])
                and all(bank.sha(b['path']) == b['sha256'] for b in row['binaries'].values())
                and all(bank.sha(p) == h for group in bank.previous.ACTUAL for p, h in actual[group].items()),
                'missing a coherent verifier, resolver, dependency or final executable')
    return native


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        files = [build / n for n in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        files.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in files}
    save_build(row)


def prepare():
    require(not ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT),
            'inspect the existing build before preparing a candidate')
    controllers = [THIS, *(str(Path(m.__file__).resolve().relative_to(bank.P))
                          for m in (stock, stock.fixture, stock.kv))]
    pins = bank.stage.source.pin(bank.P, [*controllers, 'tools/sycl', *bank.basis.POLICIES])
    require(all(name in pins['source_sha256'] for name in controllers), 'commit the projection controllers')
    native, foundation, source, recipe, proof = project()
    row = dict(stage='prepared', source_commit=pins['revision'], native_source_commit=pins['revision'],
               controller_sources=pins['source_sha256'], source_files=bank.stage.hashed(source),
               foundation=foundation, recipe=recipe, projection=proof, external_inputs=native['external_inputs'],
               build_tools={p: bank.sha(p) for p in ('/usr/bin/cmake', '/usr/bin/ninja')},
               driver=builder.guards.driver_guard(), retained_decode_reference=builder.decode_reference(recover=True),
               commands=[], active=False, passed=False, complete=False, ownership_closed=True,
               scope='One cooperative decode resolver and its capture-time selector. Preserve the scalar source, '
                     'fast long-prefill route, KV placement and all unrelated configuration.',
               retention='Keep both measured references. Prepare only after the previous candidate build retires. '
                         'Retire this build at the first failed gate or after its qualified comparisons.')
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'owned', 'tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, content in source.items():
        path = ROOT / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    check(row, False)
    save(row)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['configure'], ['compile']), 'select one build stage')
    action, = sys.argv[1:]
    builder.ROOT, builder.KEY = ROOT, KEY
    builder.check = check
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        else:
            builder.TARGETS = bank.previous.read(ROOT / 'record.json')['recipe']['targets']
            builder.save = save
            builder.execute(action)
