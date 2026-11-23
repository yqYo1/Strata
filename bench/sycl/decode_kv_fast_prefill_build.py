"""Build the context-boundary control and isolated decode KV candidate."""

import fcntl
from functools import partial
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
import prefill_large_gather_model_source as stock
import kv_planner_experiment as planner

bank = stock.bank
builder, require = bank.builder, bank.require
save_build = builder.save
ROOT = bank.B / 'decode-kv-fast-prefill-native-v1'
KEY = 'decode_kv_fast_prefill_native_v1'
BOUNDARY_ROOT = bank.B / 'decode-kv-boundary-native-v1'
BOUNDARY_KEY = 'decode_kv_boundary_native_v1'
BOUNDARY_CAPACITY_KEY = 'decode_kv_boundary_capacity_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
BOUNDARY_FIX = '178d53212581cc5d11a0e80d8991e9fda3aaa339'
PHASE = '644412c8edc625b1c634a8ecfa2fe6585aa12d67'
ISOLATED = '93b964caeb829f909624bf6dc625fe9ad88c5718'
FIXTURE_SOURCE = 'ae74403d163208bec938e0af92c5d10eb3d92263'
FIXTURE = 'tools/sycl/kv_copy_groups_bench.cpp'
ORACLE = 'tools/sycl/kv_copy_groups_oracle.hpp'
FIXTURE_TARGET = planner.TARGET
DEVICE_ROOT = ROOT / 'kv-fixture'
DEVICE_KEY = 'decode_kv_fast_prefill_device_v1'
PARENT_ELF = '85e3f202ef99661d52c03a17a8b609b8a8a44e9f584bfa8fc1a5c1c3d51aea7c'
VERIFY = 'src/core/verify.cpp'
POLICY = 'include/strata/core/verify_window_policy.hpp'
PHASE_PATHS = (VERIFY, POLICY, 'include/strata/core/verify.hpp')
SCALAR = 'src/kernels/sycl/kv_stream.cpp'
COOPERATIVE = 'src/kernels/sycl/decode_kv_stream.cpp'
HEADER = 'include/strata/kernels/kv_stream.hpp'
CMAKE = 'cmake/sycl_backend.cmake'
BOUNDARY_PROJECTED = {
    'src/core/mtp.cpp': '243078ea8c2ca2ae7cbac3345fc33872ca579f3e1671810a871ac1eaf67fb397',
    'src/program/generate.cpp': '81fe4b074aa94c1dc86730ec930b004d246dc8328e59d33a13d2640f91344dc9',
}
PROJECTED = {
    VERIFY: '1c2dfebe0a81e63255a2f1f5d9b4bd6a3839f875d0e148efe97f6fe3a69b17f2',
    POLICY: 'be517d179213990d646d8d40f1843298d2e9c8df43f014a5eb41d87531dd87a4',
    'include/strata/core/verify.hpp': 'd0e6b46ac93d2451de3696aaa7ccdf788db71e02195b2aa867794570e3a7cf25',
    HEADER: 'b5a8ec1431ae747587a6a65d3657201c5d94184c03e5a70cef3bf08db371d525',
    CMAKE: 'd850eb227e5936b5d8e6815d484f5ac0f714b33e6fd0171dde5a181f660b7f5a',
    COOPERATIVE: '40b4962e1d39db8e1a779d03f3a0150189c38589562203e106662c791ba238d1',
}
DEVICE_PLAN = dict(order=['smoke', 'full'],
    scope='The existing KV planner fixture links the same strata_kernels archive as the new server. '
          'Run traced bounded correctness before full oracle, graph, and physical-tail component checks.')


def replace_once(source, text, replacement):
    require(source.count(text) == 1, 'unexpected source context for ' + text.decode().strip())
    return source.replace(text, replacement, 1)


def source_patch(source, revision, paths, *, context=3):
    patch = subprocess.check_output(['git', 'diff', f'--unified={context}',
                                     revision + '^', revision, '--', *paths],
                                    cwd=bank.P, timeout=30)
    with tempfile.TemporaryDirectory(prefix='strata-decode-kv-projection-') as tmp:
        directory = Path(tmp)
        for name in paths:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source[name])
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', '-'],
                       cwd=directory, input=patch, check=True, timeout=30)
        return {name: (directory / name).read_bytes() for name in paths}


def cooperative_sources(source):
    require(COOPERATIVE not in source and all(path in source for path in PROJECTED if path != COOPERATIVE),
            'unexpected cooperative KV projection base')
    changed = source_patch(source, PHASE, PHASE_PATHS)
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
    return changed


def parent_reference():
    """Recover the committed build identity for source and workload references."""
    parent = bank.previous.read(stock.MODEL_ROOT / 'record.json')
    require(bank.basis.validation.committed(bank.RESULT, (stock.KEY,)) == builder.compact(parent)
            and parent['native_source_commit'] == stock.DISPATCH_COUNTS and parent['stage'] == 'built'
            and parent['passed'] and parent['complete'] and parent['ownership_closed'] and not parent['active']
            and parent['host_control']['passed']
            and parent['binaries']['strata']['sha256'] == PARENT_ELF,
            'require the exact committed fast-prefill build record')
    return parent


def project(expected=None, *, boundary_only=False):
    parent = parent_reference()

    # The foundation supplies historical source and compiler inputs. The
    # comparison runs the composed fast-prefill engine and its projections.
    foundation = parent['foundation']
    native_record = bank.basis.original.ROOT / 'record.json'
    native = bank.previous.read(native_record)
    external = native['external_inputs']
    require((expected is None or foundation == expected)
            and bank.sha(native_record) == foundation['original_record_sha256']
            and native['external_inputs'] == parent['external_inputs']
            and bank.sha(external['path']) == external['sha256']
            and all(bank.sha(path) == digest for path, digest in native['toolchain'].items()),
            'the original recipe, compiler and SDK inputs changed')
    source, recipe, base_projection = stock.project(native)
    stock.verify_projection(base_projection)
    basis = bank.stage.hashed(source)
    require(parent['projection'] == base_projection and parent['source_files'] == basis
            and parent['recipe'] == recipe, 'recover the exact fast-prefill composition from Git')

    boundary = source_patch(source, BOUNDARY_FIX, BOUNDARY_PROJECTED, context=1)
    require(bank.stage.hashed(boundary) == BOUNDARY_PROJECTED, 'projected context fix differs')
    source.update(boundary)
    projected = dict(BOUNDARY_PROJECTED)
    targets = list(recipe['targets'])
    old, new = str(stock.MODEL_ROOT), str(BOUNDARY_ROOT if boundary_only else ROOT)
    expected_objects = {Path(p).relative_to(stock.MODEL_ROOT / 'build').as_posix()
                        for p in parent['actual_inputs']['objects']}
    proof = dict(parent_source=stock.DISPATCH_COUNTS,
                 parent_record_sha256=bank.sha(stock.MODEL_ROOT / 'record.json'),
                 parent_binary=parent['binaries']['strata'], boundary_fix=BOUNDARY_FIX,
                 unchanged_scalar_sha256=basis[SCALAR])

    if not boundary_only:
        source.update(cooperative_sources(source))
        projected.update(PROJECTED)
        fixture = stock.fixture.git_file(FIXTURE_SOURCE, FIXTURE)
        fixture_hashes = bank.stage.hashed({FIXTURE: fixture, ORACLE: source[ORACLE]})
        require(fixture_hashes[FIXTURE] ==
                'f5a993d7ac8ed25f782bc87350a728b8214a837a13b253c3d8e3810b29924d07'
                and source[ORACLE] == stock.fixture.git_file(FIXTURE_SOURCE, ORACLE),
                'cooperative fixture or independent host oracle changed')
        source[FIXTURE] = fixture
        targets.append(FIXTURE_TARGET)
        expected_objects.add(f'CMakeFiles/strata_kernels.dir/{COOPERATIVE}.o')
        expected_objects.add(f'CMakeFiles/{FIXTURE_TARGET}.dir/{FIXTURE}.o')
        proof.update(phase=PHASE, isolated=ISOLATED, fixture_source=FIXTURE_SOURCE, fixture_files=fixture_hashes)

    allow = set(projected) | ({FIXTURE} if not boundary_only else set())
    require({name for name, digest in bank.stage.hashed(source).items() if basis.get(name) != digest}
            == allow, 'shared scalar KV source or unrelated source changed')
    recipe = dict(configure=[s.replace(old, new) for s in recipe['configure']],
                  environment={k: v.replace(old, new) for k, v in recipe['environment'].items()},
                  targets=targets)
    proof.update(native_projected=projected, object_count=len(expected_objects),
                 object_set_sha256=bank.signature(sorted(expected_objects)))
    return native, foundation, source, recipe, proof


def build_location(boundary_only):
    return (BOUNDARY_ROOT, BOUNDARY_KEY) if boundary_only else (ROOT, KEY)


def check(row, committed=True, *, boundary_only=False):
    root, key = build_location(boundary_only)
    if committed:
        require(bank.basis.validation.committed(bank.RESULT, (key,)) == builder.compact(row),
                'commit the exact preceding build stage')
    controls = row['controller_sources']
    if committed and row['stage'] == 'built':
        original = bank.stage.source.blobs(bank.P, row['source_commit'], controls)
        checked = bank.stage.hashed(dict(original))
    else:
        checked = bank.stage.source.pin(bank.P, list(controls))['source_sha256']
    require(controls == checked, 'the recorded build controller inputs changed')
    native, foundation, source, recipe, proof = project(row['foundation'], boundary_only=boundary_only)
    require(row['source_commit'] == row['native_source_commit']
            and row['recipe'] == recipe and row['foundation'] == foundation and row['projection'] == proof
            and row['source_files'] == bank.stage.hashed(source) and row['external_inputs'] == native['external_inputs']
            and row['driver'] == builder.guards.driver_guard()
            and all(bank.sha(p) == h for p, h in row['build_tools'].items()), 'build inputs changed')
    tree = root / 'source'
    require({p.relative_to(tree).as_posix() for p in tree.rglob('*') if p.is_file()} == row['source_files'].keys()
            and all(bank.sha(tree / p) == h for p, h in row['source_files'].items())
            and all(bank.sha(Path(dep['root']) / p) == h for dep in native['dependencies']
                    for p, h in dep['source_sha256'].items())
            and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'materialized source, dependency or configured input changed')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        objects = sorted(Path(p).relative_to(root / 'build').as_posix() for p in actual['objects'])
        compiled_sources = [*BOUNDARY_PROJECTED, SCALAR]
        if not boundary_only:
            compiled_sources.extend((VERIFY, POLICY, HEADER, COOPERATIVE, FIXTURE, ORACLE))
        require(not actual['partial_archive_reuse'] and actual['uncovered'] == 0
                and len(objects) == proof['object_count'] == actual['fresh_objects']
                and bank.signature(objects) == proof['object_set_sha256']
                and all(actual['compiled'].get(str(tree / p)) == row['source_files'][p]
                        for p in compiled_sources)
                and set(row['binaries']) == set(recipe['targets'])
                and all(bank.sha(b['path']) == b['sha256'] for b in row['binaries'].values())
                and all(bank.sha(p) == h for group in bank.previous.ACTUAL for p, h in actual[group].items()),
                'missing coherent compiler, dependency or final executable evidence')
    return native


def boundary_build():
    row = bank.previous.read(BOUNDARY_ROOT / 'record.json')
    check(row, boundary_only=True)
    require(row['stage'] == 'built' and row['passed'] and row['complete']
            and row['ownership_closed'] and not row['active'], 'complete the context-boundary control build first')
    return row


def qualified_boundary():
    row = boundary_build()
    capacity = bank.basis.validation.committed(bank.RESULT, (BOUNDARY_CAPACITY_KEY,))
    build_sha = bank.signature(builder.compact(row))
    outcomes = capacity['outcomes']
    require(capacity['stage'] == 'closed' and capacity['passed'] and capacity['complete']
            and capacity['ownership_closed'] and not capacity['active']
            and capacity['native_source_commit'] == row['native_source_commit']
            and capacity['build_sha256'] == build_sha and capacity['driver'] == row['driver']
            and capacity['binary'] == row['binaries']['strata']
            and capacity['diagnostic'] == row['binaries']['strata_full_context_diagnostic']
            and capacity['host_binary'] == row['binaries']['full_context_diagnostic_host_test']
            and tuple(outcomes) == ('host', 'small-four', 'near-four', 'full')
            and all(v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults']
                    for v in outcomes.values())
            and not outcomes['host']['GPU_executed']
            and all(outcomes[k]['GPU_executed'] and outcomes[k]['native_exit_code'] == 0
                    for k in ('small-four', 'near-four', 'full'))
            and all((outcomes[k]['batched_rows'], outcomes[k]['outputs'], outcomes[k]['last_width'])
                    == (tokens - 1, 2, 2) for k, tokens in (('small-four', 16385), ('near-four', 262142)))
            and outcomes['full']['final_slot_consumed'] and outcomes['full']['batched_rows'] == 262143
            and outcomes['full']['generated_ids'] == 0, 'complete the boundary-only host and physical-context checks')
    proof = dict(binary=row['binaries']['strata'], native_source_commit=row['native_source_commit'],
                 build_sha256=build_sha, capacity_sha256=bank.signature(capacity))
    return row, proof


def save(row, *, boundary_only=False):
    if row['stage'] == 'configured':
        root, _ = build_location(boundary_only)
        build = root / 'build'
        files = [build / n for n in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        files.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in files}
    save_build(row)


def prepare(*, boundary_only=False):
    root, key = build_location(boundary_only)
    require(not root.exists() and key not in bank.basis.validation.committed(bank.RESULT),
            'inspect the existing build before preparing a candidate')
    controllers = [THIS, *(str(Path(m.__file__).resolve().relative_to(bank.P))
                          for m in (stock, stock.fixture, stock.kv))]
    pins = bank.stage.source.pin(bank.P, [*controllers, 'tools/sycl', *bank.basis.POLICIES])
    require(all(name in pins['source_sha256'] for name in controllers), 'commit the projection controllers')
    native, foundation, source, recipe, proof = project(boundary_only=boundary_only)
    row = dict(stage='prepared', source_commit=pins['revision'], native_source_commit=pins['revision'],
               controller_sources=pins['source_sha256'], source_files=bank.stage.hashed(source),
               foundation=foundation, recipe=recipe, projection=proof, external_inputs=native['external_inputs'],
               build_tools={p: bank.sha(p) for p in ('/usr/bin/cmake', '/usr/bin/ninja')},
               driver=builder.guards.driver_guard(),
               commands=[], active=False, passed=False, complete=False, ownership_closed=True,
               scope=('Two context-boundary translation units on the exact fast-prefill image.' if boundary_only else
                      'Keep the context fix and add the isolated decode-only KV resolver and selector.'),
               retention='Review after the boundary and paired phase comparisons or the first failed gate.')
    if not boundary_only:
        _, row['boundary_control'] = qualified_boundary()
    root.mkdir(mode=0o700)
    for name in ('source', 'owned', 'tmp', 'cache'):
        (root / name).mkdir(mode=0o700)
    for name, content in source.items():
        path = root / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    check(row, committed=False, boundary_only=boundary_only)
    save(row, boundary_only=boundary_only)


def device_inputs():
    row = bank.previous.read(ROOT / 'record.json')
    native = check(row)
    require(row['stage'] == 'built' and row['passed'] and row['complete']
            and row['ownership_closed'] and not row['active'], 'complete the exact native/fixture build first')
    runtime = row['foundation']['runtime_manifest']
    files = bank.previous.read(Path(runtime['path']))
    require(bank.sha(runtime['path']) == runtime['sha256'] and len(files) == runtime['files']
            and all(bank.sha(path) == h for path, h in files.items()), 'frozen runtime differs')
    driver, = (path for path in files if Path(path).name.startswith('libze_intel_gpu.so.'))
    return dict(source_commit=row['source_commit'], native_source_commit=row['native_source_commit'],
                binary=row['binaries'][FIXTURE_TARGET], full_engine=row['binaries']['strata'],
                build_sha256=bank.signature(builder.compact(row)), plan=DEVICE_PLAN,
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=row['driver'],
                runtime_manifest=runtime, runtime_files=files, recipe=row['recipe'], driver_library=driver,
                native_validation_probe=native['native_validation_probe'])


def compact_device(row):
    value = {k: v for k, v in row.items() if k not in ('runtime_files', 'recipe')}
    return dict(value, recipe_sha256=bank.signature(row['recipe']))


def save_device(row):
    bank.stage.write(DEVICE_ROOT / 'record.json', row)
    results = bank.previous.read(bank.RESULT)
    results[DEVICE_KEY] = compact_device(row)
    bank.stage.write(bank.RESULT, results)


def device_gate(action):
    if action == 'prepare-device':
        require(not DEVICE_ROOT.exists() and DEVICE_KEY not in bank.basis.validation.committed(bank.RESULT),
                'fresh fixture plan only')
        row = device_inputs()
        DEVICE_ROOT.mkdir(mode=0o700)
        save_device(dict(row, stage='built', cases=[], passed=False, complete=False,
                         active=False, ownership_closed=True, GPU_executed=False))
        return

    loader = bank.basis.core.source.device.load_module
    runner = loader('fast_prefill_kv_planner_fixture', planner.__file__)
    fixture = loader('fast_prefill_kv_fixture_environment', planner.common.__file__)
    runner.ROOT, fixture.ROOT = DEVICE_ROOT, DEVICE_ROOT

    def check_device(row):
        basis = device_inputs()
        require(bank.basis.validation.committed(bank.RESULT, (DEVICE_KEY,)) == compact_device(row)
                and all(row.get(k) == v for k, v in basis.items()),
                'commit the exact device plan, binary, firmware/runtime and prior result')

    runner.context = lambda: SimpleNamespace(check=check_device, environment=fixture.environment, save=save_device)
    runner.run(action)


if __name__ == '__main__':
    action, = sys.argv[1:]
    build_stages = ('prepare', 'configure', 'compile')
    boundary_only = action in tuple(stage + '-boundary' for stage in build_stages)
    if boundary_only:
        action = action.removesuffix('-boundary')
    require(action in (*build_stages, 'prepare-device', 'smoke', 'full'), 'select one stage')
    builder.ROOT, builder.KEY = build_location(boundary_only)
    builder.check = partial(check, boundary_only=boundary_only)
    builder.save = partial(save, boundary_only=boundary_only)
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare(boundary_only=boundary_only)
        elif action in ('configure', 'compile'):
            builder.TARGETS = bank.previous.read(builder.ROOT / 'record.json')['recipe']['targets']
            builder.execute(action)
        else:
            device_gate(action)
