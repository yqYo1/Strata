#!/usr/bin/env python3
"""Own isolated direct-K model build; no GPU execution or production selection.

Recover the selected engine's exact Git inputs, overlay one prefill source
block, and build production-entrypoint fixtures alongside the engine. The
current worktree's other native changes must not enter this comparison.
"""
import ast
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import runtime_probe as runtime
import build_fused_profile as closure
import build_qsa_k_loop as component
import qsa_k_model_sources as model

P, B, A = component.P, component.B, component.A
ROOT = B / 'qsa-direct-k-model-v2'
RESULT, KEY = component.RESULT, 'direct_K_loop_model'
ORIGIN = B / 'main-native-gdn-engine/record.json'
ORIGIN_SHA = 'a38f477670fafcd1a931be2e20d90d6d9718df75b61954b421f361af9f6881b2'
TARGETS = ['strata', 'sycl_submission_test', 'sycl_graph_envelope_test']
ADAPTER = 'bench/sycl/qsa_k_model_oracle_adapter.cpp'
CONFIGURATION_SOURCES = ['tools/sycl/' + name + '.cpp' for name in (
    'host_transfer_probe', 'kv_copy_groups_host', 'kv_copy_groups_bench',
    'gemm_orientation_bench', 'projection_bench', 'moe_bench', 'gr_bench',
    'expert_contract', 'gemm_device_probe')]
require, digest, write = component.require, component.digest, component.write


def origin():
    require(digest(ORIGIN) == ORIGIN_SHA, 'original selected build receipt changed')
    row = json.loads(ORIGIN.read_text())
    require(row['source_commit'] == model.BASE and row['project']['count'] == 542,
            'unexpected original engine source')
    return row


def sources(revision):
    """Freeze recursively imported local controllers and current fixture inputs."""
    names = {component.FIXTURE, component.ADMISSION, ADAPTER, component.ORACLE,
             'tools/sycl/measurement_environment.json'}
    pending, seen = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        name = str(path.relative_to(P))
        raw = component.gen.git('show', revision + ':' + name)
        require(path.read_bytes() == raw, 'commit controller first: ' + name)
        names.add(name)
        for node in ast.walk(ast.parse(raw)):
            modules = [v.name for v in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                local = path.parent / ((module or '').split('.')[0] + '.py')
                if local.is_file():
                    pending.append(local)
    pins = {}
    for name in sorted(names):
        raw = component.gen.git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'source changed: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def native_sources(full):
    expected = full['project']['source_sha256']
    for name, raw in runtime.build.blobs(P, model.BASE, [*expected, *CONFIGURATION_SOURCES]):
        require(name not in expected or hashlib.sha256(raw).hexdigest() == expected[name],
                'original source receipt differs: ' + name)
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'source path escape')
        yield name, model.production_prefix(raw) if name == model.component.PREFIX else raw


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('input_sha256', 'commands')}
    out['controller_inputs'] = dict(files=len(row['input_sha256']), aggregate_sha256=hashlib.sha256(
        json.dumps(row['input_sha256'], sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    out['recipe_sha256'] = hashlib.sha256(json.dumps(row['commands'], sort_keys=True).encode()).hexdigest()
    return out


def admit(row):
    full = origin()
    require(sources(row['source_commit']) == row['input_sha256'], 'controller inputs changed')
    require(closure.sdk_inputs() == row['sdk'], 'SDK changed')
    for group in full['dependencies']:
        require(runtime.build.pin(Path(group['root']), ['.']) == group, 'selected dependency changed')
    rt = runtime.validation.committed(runtime.RESULT)
    require(digest(runtime.RESULT) == row['runtime_sha256'], 'runtime qualification changed')
    require(runtime.current(rt)['binary'] == row['selected_binary'], 'validated fallback changed')
    require(digest(row['external_manifest']['path']) == row['external_manifest']['sha256'],
            'external input manifest changed')
    allowed = json.loads(Path(row['external_manifest']['path']).read_text())
    require(len(allowed) == row['external_manifest']['files'], 'external manifest count changed')
    for path, sha in allowed.items():
        require(digest(path) == sha, 'external compiler/configure/link input changed: ' + path)
    for v in json.loads(closure.lineage.SDK.read_text()):
        allowed[str(Path(v['path']).resolve())] = closure.SYSTEM_HEADER_UPDATES.get(
            v['path'], (None, v['sha256']))[1]
    for name, raw in native_sources(full):
        path = ROOT / 'source' / name
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                'isolated source changed: ' + name)
        allowed[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    for group in full['dependencies']:
        allowed.update({str((Path(group['root']) / name).resolve()): sha
                        for name, sha in group['source_sha256'].items()})
    for name, raw in component.generated().items():
        path = ROOT / name
        require(path.read_bytes() == raw and digest(path) == row['generated'][name], 'private fixture changed')
        allowed[str(path)] = row['generated'][name]
    allowed.update({str((P / name).resolve()): sha for name, sha in row['input_sha256'].items()})
    allowed.update(row['generated_expected'])
    return allowed


def prepare():
    require(not ROOT.exists(), 'fresh model generation only; inspect existing owner before resuming')
    record = component.gen.committed(RESULT)
    scope, qualified = record[KEY], record[component.KEY]
    require('build' not in scope, 'close and retain the prior build outcome before a new plan')
    require(scope['stage'] == 'source_constructor_prepared_not_built' and
            qualified['component_performance_summary']['all_six_groups_pass'] and
            qualified['component_generation_retirement']['complete'], 'component gate/retirement required')
    full = origin()
    rt = runtime.validation.committed(runtime.RESULT)
    require(runtime.current(rt)['binary'] == full['binary'], 'selected engine identity differs')
    revision = component.gen.git('rev-parse', 'HEAD').decode().strip()
    pins = sources(revision)
    # All original project inputs were ordinary files; preserve this invariant.
    modes = component.gen.git('ls-tree', '-r', model.BASE).decode().splitlines()
    mode = {line.split('\t', 1)[1]: line.split(None, 1)[0] for line in modes}
    require(all(mode[name] == '100644' for name in [*full['project']['source_sha256'], *CONFIGURATION_SOURCES]),
            'unexpected source mode')
    external = closure.external_inputs(rt)
    for root in (Path('/usr/share/cmake-3.28/Modules'), Path('/usr/share/cmake-3.28/Templates')):
        external.update({str(p.resolve()): digest(p) for p in root.rglob('*') if p.is_file()})
    tools = {n: str(Path('/usr/bin/' + n).resolve()) for n in ('cmake','ninja','cc','ar','ranlib','as','ld','git')}
    tools['cc1'] = subprocess.check_output(['/usr/bin/cc','-print-prog-name=cc1'], text=True, timeout=10).strip()
    tools['sycl'] = full['compiler']['path']
    external.update({str(Path(p).resolve()): digest(p) for p in tools.values()})
    old = json.loads(component.gen.git('show', '8849d71e5:' + str(RESULT.relative_to(P))))
    external.update(old[component.KEY]['native_build']['link_inputs'])
    require(all(digest(path) == sha for path, sha in external.items()), 'original link input changed')
    sdk = closure.sdk_inputs()
    ROOT.mkdir(mode=0o700)
    for name in ('source','tmp','cache'):
        (ROOT / name).mkdir()
    for name, raw in native_sources(full):
        path = ROOT / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    blobs = component.generated()
    for name, raw in blobs.items():
        (ROOT / name).write_bytes(raw)
    write(ROOT / 'external-inputs.json', external)
    argv = list(full['configure_argv'])
    argv[argv.index('-S') + 1] = str(ROOT / 'source')
    argv[argv.index('-B') + 1] = str(ROOT / 'build')
    env = dict(full['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    flags = ['-std=c++20','-O3','-DNDEBUG','-g0','-fsycl','-fsycl-id-queries-range=size_t',
             '-fno-fast-math','-ffp-contract=off','-DSTRATA_ENABLE_SYCL=1','-DSTRATA_QSA_PACKED_GUARD=17',
             '-I' + str(ROOT / 'source/include'), '-I' + str(ROOT / 'source/include/strata/sycl/compat'),
             '-I' + str(ROOT)]
    files = [ROOT / 'source' / n for n in component.NATIVE] + [ROOT / 'direct_k_control.cpp',
        ROOT / 'direct_k_invariant.cpp', P / component.FIXTURE, P / ADAPTER, P / component.ORACLE]
    objects = [ROOT / ('input-' + str(i) + '.o') for i in range(len(files))]
    commands = [('configure', argv), ('engine', ['/usr/bin/cmake','--build',str(ROOT / 'build'),
                '--target',*TARGETS,'-j','4'])]
    for i, (path, obj) in enumerate(zip(files, objects)):
        extra = ['-Dqsa_bf16_prefix_batch=qsa_k_model_test_batch','-DSTRATA_QSA_COMPARE_PREFIX33=1'] \
            if path == P / component.ORACLE else []
        commands.append(('input-' + str(i), [full['compiler']['path'],*flags,*extra,'-MD','-MF',
                         str(obj.with_suffix('.d')),'-c',str(path),'-o',str(obj)]))
    for name, chosen in [('fixture', objects[:8]), ('numerical', objects[:7] + objects[8:])]:
        commands.append((name, [full['compiler']['path'],*flags,*map(str, chosen),'-pthread','-ldl',
                               '-o',str(ROOT / name)]))
    for name, path in [('original', files[4]), ('control', files[5]), ('candidate', files[6])]:
        commands.append((name + '-ir', [full['compiler']['path'],*flags,'-fsycl-device-only','-MD','-MF',
            str(ROOT / (name + '-ir.d')),'-S','-emit-llvm',str(path),'-o',str(ROOT / (name + '.ll'))]))
    commands.append(('dependencies', ['/usr/bin/ninja','-C',str(ROOT / 'build'),'-t','deps']))
    row = dict(stage='prepared', source_commit=revision, input_sha256=pins,
        original_build=dict(path=str(ORIGIN), sha256=ORIGIN_SHA, source_commit=model.BASE,
            source_groups=[{k:g[k] for k in ('root','revision','count','aggregate_sha256')}
                           for g in [full['project'], *full['dependencies']]]),
        source_change=dict(path=model.component.PREFIX, before=model.component.PREFIX_SHA256,
                           after=scope['candidate_prefix_sha256']),
        configuration_sources={n:hashlib.sha256(raw).hexdigest() for n,raw in
            runtime.build.blobs(P, model.BASE, CONFIGURATION_SOURCES)},
        selected_binary=full['binary'], compiler=full['compiler'], sdk=sdk,
        external_manifest=dict(path=str(ROOT / 'external-inputs.json'), sha256=digest(ROOT / 'external-inputs.json'), files=len(external)),
        runtime_sha256=digest(runtime.RESULT), boot=rt['boot'], driver=rt['driver'],
        environment=env, targets=TARGETS, commands=commands,
        generated={n:hashlib.sha256(v).hexdigest() for n,v in blobs.items()},
        generated_expected={str(ROOT / 'build/ggml/src/ggml-version.h'):hashlib.sha256(closure.HEADER.encode()).hexdigest()},
        budgets=dict(wall=1800,cpu=7200,rss_cap=12 << 30,text_cap=32 << 20,file_cap=512 << 20,total_cap=4 << 30),
        total_wall_seconds=3600, active=False, ownership_closed=True, passed=False, GPU_executed=False,
        model_built=False, model_selected=False,
        criteria='Build only. Exact selected9853Git inputs plus one independently reviewed prefill block;9additional6e40auxiliary target source files required solely for CMake generation,never requested as build targets. Current native differences excluded. Same-production-source numerical/actual-ID fixtures,3unique18argSG16 kernels;zero uncovered actual dependencies/imported link inputs. No GPU/model/performance qualification inherited.',
        retention='One additional model/fixture generation for imminent same-build property,actual-ID,oracle and bounded long-model comparison. Preserve original6e40/12fixtures. Retire objects,normal logs and rejected generation after source/recipe recovery;legacy holds remain separate.')
    admit(row)
    write(ROOT / 'record.json', row)
    scope['build'] = compact(row)
    write(RESULT, record)
    print('Prepared isolated production engine and same-source fixtures; commit before build.', flush=True)


def run():
    record = component.gen.committed(RESULT)
    scope = record[KEY]
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and compact(row) == scope['build'] and
            not (ROOT / 'owned-build').exists(), 'fresh committed build only')
    allowed = admit(row)
    original, owner = copy.deepcopy(row), None
    deadline = time.monotonic() + row['total_wall_seconds']
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(row, commands_executed=owner.commands if owner else []))
    def monitor(io, entry):
        next_size = 0
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if time.monotonic() >= next_size:
                require(runtime.supervision.artifact_bytes(ROOT) <= row['budgets']['total_cap'], 'whole build size budget')
                next_size = time.monotonic() + 1
            if select.select([io.proc.stdout], [], [], .02)[0]:
                io.read(); io.pending.clear()
        io.exit(1)
    try:
        row['stage'] = 'building'
        owner = runtime.supervision.Owner(ROOT / 'owned-build', persist)
        for label, command in row['commands']:
            remaining = deadline - time.monotonic()
            require(remaining > 1, 'whole build deadline')
            print('CPU build:', label, flush=True)
            entry, out, err = owner.run(label, command, row['environment'], ROOT,
                interaction=monitor, **dict(row['budgets'], wall=min(remaining, row['budgets']['wall'])))
            require(runtime.supervision.closed(entry) and entry['exit_code'] == 0,
                    label + ' failed: ' + (out.read_text(errors='replace') + err.read_text(errors='replace'))[-5000:])
        admit(original)
        paths = {Path(line.strip()) for line in out.read_text().splitlines() if line.startswith('    ') and line.strip()}
        paths = {(p if p.is_absolute() else ROOT / 'build' / p).resolve() for p in paths}
        for dep in ROOT.glob('*.d'):
            paths.update(Path(p).resolve() for p in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]))
        require(len(paths) > 700 and {ROOT / 'source' / n for n in component.NATIVE} <= paths,
                'incomplete actual dependency closure')
        observed = {}
        for path in sorted(paths):
            sha = digest(path)
            require(allowed.get(str(path)) == sha, 'unfrozen actual compiler input: ' + str(path))
            observed[str(path)] = sha
        write(ROOT / 'compiler-inputs.json', observed)
        ninja = (ROOT / 'build/build.ninja').read_text()
        imported = {}
        for name in TARGETS:
            stanza = ninja[ninja.index('build ' + name + ':'):].split('\n\n',1)[0]
            links = shlex.split(next(line.split(' = ',1)[1] for line in stanza.splitlines()
                                    if line.startswith('  LINK_LIBRARIES = ')))
            for path in (Path(p).resolve() for p in links if p.startswith('/')):
                require(allowed.get(str(path)) == digest(path), 'unfrozen imported link provider: ' + str(path))
                imported[str(path)] = digest(path)
        def asset(path):
            return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))
        images = {name:asset(ROOT / (name + '.ll')) for name in ('original','control','candidate')}
        ids = component.kernel_ids(images)
        row.update(stage='built-not-executed', passed=True, model_built=True,
            binaries={name:asset(ROOT / 'build' / name) for name in TARGETS},
            fixtures={name:asset(ROOT / name) for name in ('fixture','numerical')}, device_IR=images,
            attention_kernel_ids=ids, imported_link_inputs=imported,
            actual_compiler_inputs=dict(files=len(observed), path=str(ROOT / 'compiler-inputs.json'),
                sha256=digest(ROOT / 'compiler-inputs.json'), uncovered_files=0),
            compile_database_sha256=digest(ROOT / 'build/compile_commands.json'))
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row['ownership_closed'] = owner is None or owner.active is None and all(
            runtime.supervision.closed(c) for c in owner.commands)
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        persist(); scope['build'] = compact(row); write(RESULT, record)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','model_built','ownership_closed','error')}), flush=True)
    require(row['passed'], 'model build did not qualify')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare':prepare, 'build':run}[sys.argv[1]]()
