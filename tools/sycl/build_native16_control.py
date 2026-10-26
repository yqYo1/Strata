#!/usr/bin/env python3
"""Build public-GU calibration with the independent official SDK; Main only.

Repository LICENSE. Uses our pinned CMake recipe and SDK admission, without
the former compiler or any XeStrata source. This never builds/runs the engine.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import build_fused_profile as shared

r, P, B = shared.r, shared.P, shared.B
RESULT = shared.RESULT
ROOT = B / 'native16-public-control-v2'
KEY = 'native16_public_calibration_build'
OFFICIAL = Path('/opt/intel/oneapi/compiler/2026.2')
UMF = Path('/opt/intel/oneapi/umf/1.2')
SDK = B / 'prefill-expert-staging-v3/sdk-inputs.json'
SDK_SHA = 'c8879f0b8d772979bf2096dc64a3bba49853bd15dad221b46427a922030647d8'
RECIPE = B / 'prefill-fused-command-profile-v2/record.json'
RECIPE_SHA = '7a9d4f81952571c8d4e06aa579891e11e072b7c2d7fb5395272b232afd45337b'
PACKAGES = ['intel-oneapi-dpcpp-cpp-2026.2', 'intel-oneapi-compiler-dpcpp-cpp-runtime-2026.2',
    'intel-oneapi-umf-1.2']
PREFIXES = ['CMakeLists.txt', 'cmake', 'include', 'src', 'tests', 'third_party', 'bench/sycl', 'tools/sycl']
TARGETS = ['sycl_submission_test', 'sycl_vllm_native16_test',
    'sycl_moe_gu_swiglu_bench', 'sycl_moe_down_scatter_bench']
require, digest, write = r.require, r.digest, r.write


def scope(container):
    return container['event_reconciliation']['fused_command_profile']


def compact(row):
    result = {k: v for k, v in row.items() if k not in ('project', 'dependencies', 'external_inputs', 'commands')}
    result['source_groups'] = [{k: g[k] for k in ('root', 'revision', 'count', 'aggregate_sha256')}
        for g in [row['project'], *row['dependencies']]]
    return result


def save(row):
    container = json.loads(RESULT.read_text()); scope(container)[KEY] = compact(row)
    write(ROOT / 'record.json', row); write(RESULT, container)


def sdk_admit(meta):
    require(digest(SDK) == SDK_SHA and meta['path'] == str(SDK) and meta['sha256'] == SDK_SHA,
        'official SDK snapshot changed')
    files = json.loads(SDK.read_text())
    require(len(files) == 10945 and all(digest(p) == sha for p, sha in files.items()), 'official SDK input changed')
    require(subprocess.check_output(['dpkg-query', '-W', '-f=${Package} ${Version}\n', *PACKAGES],
        text=True, timeout=15).splitlines() == meta['packages'], 'official package versions changed')
    return files


def current(row):
    require(r.validation.committed(RESULT)['event_reconciliation']['fused_command_profile'][KEY] == compact(row),
        'commit exact component plan/outcome')
    require(shared.build.pin(P, PREFIXES)['source_sha256'] == row['project']['source_sha256'],
        'frozen component source changed')
    for group in row['dependencies']:
        require(shared.build.pin(Path(group['root']), ['.']) == group, 'independent dependency changed')
    sdk_admit(row['sdk'])
    for path, sha in row['external_inputs'].items(): require(digest(path) == sha, 'external input changed: ' + path)
    require(digest(RECIPE) == RECIPE_SHA and digest(r.RESULT) == row['runtime_sha256'], 'recipe/runtime proof changed')
    rt = r.validation.committed(r.RESULT)
    require(r.current(rt)['binary'] == row['selected_binary'], 'selected fallback changed')
    for tool in row['tools'].values(): require(digest(tool['path']) == tool['sha256'], 'build tool changed')


def prepare():
    container = r.validation.committed(RESULT); source = scope(container)
    require(not ROOT.exists() and KEY not in source, 'single-use component build;inspect existing attempt')
    require(source['new_build_toolchain']['independent_build_admitted'], 'independent toolchain origin required')
    rt = r.validation.committed(r.RESULT); full = r.current(rt)
    sdk = copy.deepcopy(source['prefill_upload_staging_v5']['sdk']); sdk_admit(sdk)
    require(digest(RECIPE) == RECIPE_SHA, 'original CMake recipe changed')
    previous = json.loads(RECIPE.read_text())
    require(previous['source_commit'] == '26fdb5dd56070da7e336b44f91e8c04b7a7acf1b', 'original recipe revision differs')
    # Recover original recipe, rather than relying only on the retained receipt hash.
    raw = shared.build.git(P, 'show', previous['source_commit'] + ':tools/sycl/build_fused_profile.py')
    require(hashlib.sha256(raw).hexdigest() ==
        previous['project']['source_sha256']['tools/sycl/build_fused_profile.py'],
        'original configure recipe is not recoverable')
    inherited = list(full['configure_argv'])
    inherited[inherited.index('-B') + 1] = str(shared.ROOT / 'build')
    require(inherited == previous['configure_argv'], 'original inherited configure recipe differs')
    compiler = source['new_build_toolchain']['official_candidate']
    require(compiler['compiler'] == str(OFFICIAL / 'bin/icpx') and
        digest(compiler['compiler']) == compiler['compiler_sha256'], 'official compiler changed')
    project = shared.build.pin(P, PREFIXES)
    deps = [shared.build.pin(Path(g['root']), ['.']) for g in full['dependencies']]
    require(all(a['source_sha256'] == b['source_sha256'] for a, b in zip(deps, full['dependencies'])),
        'independent GGML/TLA dependency differs')
    cfg = [v.replace(str(shared.ROOT / 'build'), str(ROOT / 'build')) for v in previous['configure_argv']]
    cfg = [('-DCMAKE_CXX_COMPILER=' + compiler['compiler']) if v.startswith('-DCMAKE_CXX_COMPILER=') else v for v in cfg]
    cfg.append('-DCMAKE_CXX_FLAGS=-fsycl-id-queries-range=size_t')
    require(not any('/MistVVK/' in v for v in cfg), 'former SDK path remains in new configure command')
    env = dict(r.validation.committed(r.SETTINGS)['environment'])
    old_lib = str(Path(next(p for p in rt['runtime_files'] if '/libsycl.so.' in p)).parent)
    env['LD_LIBRARY_PATH'] = ':'.join([str(OFFICIAL / 'lib'), str(UMF / 'lib'),
        *(p for p in env['LD_LIBRARY_PATH'].split(':') if p != old_lib)])
    env.update(UR_ADAPTERS_FORCE_LOAD=str(OFFICIAL / 'lib/libur_adapter_level_zero_v2.so.0.12.0'),
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    tools = copy.deepcopy(previous['tools']); tools['sycl'] = dict(path=compiler['compiler'], sha256=compiler['compiler_sha256'])
    row = dict(stage='prepared', source_commit=project['revision'], project=project, dependencies=deps,
        sdk=sdk, tools=tools, external_inputs=shared.external_inputs(rt),
        runtime_sha256=digest(r.RESULT), selected_binary=full['binary'], boot=rt['boot'], driver=rt['driver'],
        original_recipe=dict(path=str(RECIPE), sha256=RECIPE_SHA, source_commit=previous['source_commit']),
        configure_argv=cfg, environment=env, targets=TARGETS,
        build_command=['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--parallel', '2', '--target', *TARGETS],
        dependency_command=['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'],
        generated_expected={str(ROOT / 'build/ggml/src/ggml-version.h'): hashlib.sha256(shared.HEADER.encode()).hexdigest()},
        budgets=dict(wall=1200, cpu=3600, rss_cap=12 << 30, text_cap=8 << 20, file_cap=2 << 30, total_cap=4 << 30),
        active=False, ownership_closed=True, passed=False, complete=False, GPU_executed=False,
        engine_built=False, production_selected=False, model_adoption_qualified=False,
        question='Build actual public GU dispatcher calibration and independently check the legacy GEMM ABI with official SDK.',
        criteria='Source/dependency/SDK/tools pinned before configure;actual ninja compile closure fully covered;'
            'normal finite owned configure/build/dependency exits;four exact fixture hashes;no engine/model execution. '
            'Compilation alone qualifies neither runtime,GEMM math,performance nor physical256K context.',
        next_gate='Official-runtime host canary,traced same-build submission and native16/GU/Down correctness;'
            'then public GU fixed batches M160/2333/2334 in startup0/large independent processes. '
            'Selected engine and existing two older GU/Down comparison fixtures stay intact until calibration closure.',
        retention='One official component candidate only,no model engine build. Keep source/recipe and useful closure/hash proof;'
            'retire normal build output/objects after qualification selection. Prior GU/Down fixtures retire after their '
            'named same-shape/legacy-ABI comparison;retain original13 selected assets.')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    save(row); print('Prepared;commit the frozen build plan before configure.', flush=True)


def dependencies(row, text):
    expected = {str((Path(g['root']) / name).resolve()): sha
        for g in [row['project'], *row['dependencies']] for name, sha in g['source_sha256'].items()}
    expected.update(sdk_admit(row['sdk'])); expected.update(row['external_inputs']); expected.update(row['generated_expected'])
    found = {str((Path(p) if Path(p).is_absolute() else ROOT / 'build' / p).resolve())
        for p in (line.strip() for line in text.splitlines() if line.startswith('    ')) if p}
    require(len(found) >= 100 and {str(P / 'bench/sycl/moe_gu_swiglu.cpp'),
        str(P / 'src/prefill/sycl/gemm16.cpp'), str(P / 'src/prefill/sycl/vllm_xe2/native16.cpp')} <= found,
        'actual dependency closure incomplete')
    require(all(p in expected and digest(p) == expected[p] for p in found), 'unplanned or changed compiled input')
    write(ROOT / 'compiler-inputs.json', {p: expected[p] for p in sorted(found)})
    return dict(files=len(found), path=str(ROOT / 'compiler-inputs.json'), sha256=digest(ROOT / 'compiler-inputs.json'), uncovered=0)


def build(row):
    current(row); require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use owned build')
    start = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(v.split(': ', 1)[1] for v in start.splitlines() if v.startswith('-- cursor: '))
    initial = copy.deepcopy(row); owner = None; row.update(stage='building', active=True)
    def persist():
        write(ROOT / 'record.json', dict(row, commands=owner.commands if owner else [], active=bool(owner and owner.active)))
    try:
        owner = r.supervision.Owner(ROOT / 'owned', persist)
        for label, command in [('configure', row['configure_argv']), ('compile', row['build_command'])]:
            entry, _, _ = owner.run(label, command, row['environment'], P, **row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code'] == 0, label + ' failed')
        entry, out, _ = owner.run('dependencies', row['dependency_command'], row['environment'], P,
            wall=30, cpu=30, rss_cap=1 << 30, text_cap=8 << 20, file_cap=8 << 20, total_cap=4 << 30)
        require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'dependency command failed')
        row['compiled_inputs'] = dependencies(row, out.read_text())
        row['fixtures'] = {name: dict(path=str(ROOT / 'build' / name), sha256=digest(ROOT / 'build' / name),
            bytes=(ROOT / 'build' / name).stat().st_size) for name in TARGETS}
        current(initial); row.update(complete=True, host_probe_accepted=True)
    except BaseException as error: row['error'] = repr(error)
    finally:
        if owner: r.validation.finish(row, owner, cursor, row['boot'])
        row.update(stage='built' if row['passed'] and row['complete'] else 'failed', active=False)
        row['commands'] = owner.commands if owner else []; save(row)
    print(json.dumps({k: row.get(k) for k in ('stage', 'complete', 'passed', 'ownership_closed', 'error')}), flush=True)
    return int(not row['passed'])


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); r.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare(); return 0
        if sys.argv[1:] == ['build']: return build(json.loads((ROOT / 'record.json').read_text()))
        raise RuntimeError('use prepare or build')


if __name__ == '__main__': sys.exit(main())
