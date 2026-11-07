#!/usr/bin/env python3
"""Build the reviewed split-MoE overlay on exact selected6e40; repository LICENSE.

No XeStrata implementation input. Keep the validated engine/12 fixtures intact.
This controller compiles only; device and model qualification are separate gates.
"""
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
import matrix_shapes as q
import moe_split_sources as model
import moe_down_sources as down

r, P, B = q.r, q.P, q.B
ROOT = B / 'moe-split-accumulator-v1/native-v2'
RESULT, KEY = q.RESULT, 'moe_split_accumulator'
ORIGIN = B / 'main-native-gdn-engine/record.json'
ORIGIN_SHA = 'a38f477670fafcd1a931be2e20d90d6d9718df75b61954b421f361af9f6881b2'
TARGETS = ['strata', 'sycl_moe_down_scatter_bench', 'sycl_moe_accum_test']
CONFIGURATION_SOURCES = ['tools/sycl/' + name + '.cpp' for name in (
    'host_transfer_probe', 'kv_copy_groups_host', 'kv_copy_groups_bench',
    'gemm_orientation_bench', 'projection_bench', 'moe_bench', 'gr_bench',
    'expert_contract', 'gemm_device_probe')]
require, digest, write = q.require, q.digest, q.write


def context(variant):
    require(variant in ('split', 'down'), 'unknown model build variant')
    return (ROOT, KEY) if variant == 'split' else (B / 'moe-down-model-v1', 'moe_down_model')


def origin():
    require(digest(ORIGIN) == ORIGIN_SHA, 'selected original build receipt changed')
    row = json.loads(ORIGIN.read_text())
    require(row['source_commit'] == model.BASE and row['project']['count'] == 542,
            'unexpected selected build source')
    return row


def sources(revision):
    return q.sources(revision, Path(__file__).resolve(),
                     [*model.DIRECT, *model.ORCHESTRATION, model.GEMM, model.FIXTURE])


def native_sources(full, variant='split', revision=None):
    overlay = down.overlays(revision) if variant == 'down' else model.overlays()
    expected = full['project']['source_sha256']
    require(set(overlay) <= set(expected), 'overlay input absent from original build')
    for name, raw in r.build.blobs(P, model.BASE, [*expected, *CONFIGURATION_SOURCES]):
        require(name not in expected or hashlib.sha256(raw).hexdigest() == expected[name],
                'original source receipt mismatch:' + name)
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'source path escape')
        yield name, overlay.get(name, raw)


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('input_sha256', 'commands_executed')}
    out['controller_inputs'] = dict(files=len(row['input_sha256']), aggregate_sha256=hashlib.sha256(
        json.dumps(row['input_sha256'], sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    return out


def admit(row, root=ROOT):
    full = origin()
    require(sources(row['source_commit']) == row['input_sha256'], 'controller/native proposal changed')
    variant = row.get('variant', 'split')
    proof = down.proof(row['source_commit']) if variant == 'down' else model.proof()
    require(proof == row['overlay'], 'overlay constructor output changed')
    require(q.closure.sdk_inputs() == row['sdk'], 'compiler SDK changed')
    for group in full['dependencies']:
        require(r.build.pin(Path(group['root']), ['.']) == group, 'independent dependency changed')
    rt = r.validation.committed(r.RESULT)
    require(digest(r.RESULT) == row['runtime_sha256'], 'runtime qualification changed')
    require(r.current(rt)['binary'] == row['selected_binary'], 'validated fallback changed')
    require(digest(row['external_manifest']['path']) == row['external_manifest']['sha256'],
            'external input inventory changed')
    allowed = json.loads(Path(row['external_manifest']['path']).read_text())
    require(len(allowed) == row['external_manifest']['files'], 'external inventory count changed')
    for path, sha in allowed.items():
        require(digest(path) == sha, 'external compiler/configure/link input changed:' + path)
    for item in json.loads(q.closure.lineage.SDK.read_text()):
        allowed[str(Path(item['path']).resolve())] = q.closure.SYSTEM_HEADER_UPDATES.get(
            item['path'], (None, item['sha256']))[1]
    for name, raw in native_sources(full, variant, row['source_commit']):
        path = root / 'source' / name
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                'isolated native source changed:' + name)
        allowed[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    for group in full['dependencies']:
        allowed.update({str((Path(group['root']) / name).resolve()): sha
                        for name, sha in group['source_sha256'].items()})
    allowed.update({str((P / name).resolve()): sha for name, sha in row['input_sha256'].items()})
    allowed.update(row['generated_expected'])
    for item in row.get('binaries', {}).values():
        require(digest(item['path']) == item['sha256'], 'built executable changed')
    return allowed


def prepare(variant='split'):
    root, key = context(variant)
    require(not root.exists(), 'fresh native generation only; inspect existing owner/record')
    record = r.validation.committed(RESULT)
    if variant == 'split':
        scope = record[key]
        require(scope['host']['passed'] and scope['host']['ownership_closed'] and
                scope['source_review']['accepted_for_bounded_component_validation_only'] and
                not scope.get('build'), 'closed host/source gates and new native plan required')
    else:
        require(key not in record, 'fresh Down model candidate only')
        for component in ('moe_down_row_address', 'moe_down_row64'):
            gate = record[component]['device']
            require(gate['stage'] == 'closed' and gate['decision']['component_speed_gate_passed'] and
                    all(c['passed'] and c['complete'] and c['ownership_closed'] and not c['active'] and
                        not c['fresh_faults'] for c in gate['results']), 'closed component gates required')
        require(record['moe_down_row64']['build']['source_commit'] == down.PROTOTYPE and
                record['moe_down_row64']['raw_retirement']['post_retirement_admission_passed'],
                'exact qualified Down prototype required')
        scope = record[key] = dict(status='source-prepared', inference_adopted=False, decode_changed=False,
            mechanism='Immutable prefill-only original/row128/row64 split-Down selection;same BF16 rounding '
                'and FP32 expert-order FMA. One executable supplies matched process arms. No GU,non-split '
                'or decode math/placement change.',
            component_selection=dict(source_commit=down.PROTOTYPE,
                row128_service_ratio=record['moe_down_row_address']['device']['decision']['service_geomean'],
                row64_vs_row128_service_ratio=record['moe_down_row64']['device']['decision']['service_geomean'],
                model_qualification_inherited=False, executable_reused=False),
            next_gate='Fresh same-build small traced production-route checks for all three selections;'
                'then matched long-prefill PP/TTFT,independent decode/handoff,task quality and physical262144.')
    full = origin()
    rt = r.validation.committed(r.RESULT)
    require(r.current(rt)['binary'] == full['binary'], 'selected original executable differs')
    revision = r.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = sources(revision)
    overlay = down.proof(revision) if variant == 'down' else model.proof()
    external = q.closure.external_inputs(rt)
    for cmake_root in (Path('/usr/share/cmake-3.28/Modules'), Path('/usr/share/cmake-3.28/Templates')):
        external.update({str(p.resolve()): digest(p) for p in cmake_root.rglob('*') if p.is_file()})
    tools = {n: str(Path('/usr/bin/' + n).resolve()) for n in ('cmake','ninja','cc','ar','ranlib','as','ld','git')}
    tools['cc1'] = subprocess.check_output(['/usr/bin/cc','-print-prog-name=cc1'], text=True, timeout=10).strip()
    tools['sycl'] = full['compiler']['path']
    external.update({str(Path(p).resolve()): digest(p) for p in tools.values()})
    old = json.loads(r.build.git(P, 'show', '8849d71e5:' + str(
        RESULT.with_name('qsa-codegen-current.json').relative_to(P))))
    # Original imported linker providers are independently pinned, not inferred
    # from whatever happens to resolve during this build.
    external.update(old['direct_K_loop_invariant']['native_build']['link_inputs'])
    require(all(digest(path) == sha for path, sha in external.items()), 'prospective external input changed')
    sdk = q.closure.sdk_inputs()
    native = dict(native_sources(full, variant, revision))
    root.mkdir(mode=0o700)
    for name in ('source','tmp','cache'):
        (root / name).mkdir()
    for name, raw in native.items():
        path = root / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    write(root / 'external-inputs.json', external)
    argv = list(full['configure_argv'])
    argv[argv.index('-S') + 1] = str(root / 'source')
    argv[argv.index('-B') + 1] = str(root / 'build')
    row = dict(stage='prepared', source_commit=revision, input_sha256=pins, overlay=overlay,
        original_build=dict(path=str(ORIGIN), sha256=ORIGIN_SHA, source_commit=model.BASE,
            source_groups=[{k:g[k] for k in ('root','revision','count','aggregate_sha256')}
                           for g in [full['project'], *full['dependencies']]]),
        native_inputs=dict(files=len(native), aggregate_sha256=hashlib.sha256(json.dumps(
            {n:hashlib.sha256(v).hexdigest() for n,v in native.items()}, sort_keys=True).encode()).hexdigest()),
        configuration_sources=CONFIGURATION_SOURCES, selected_binary=full['binary'],
        compiler=full['compiler'], sdk=sdk,
        external_manifest=dict(path=str(root / 'external-inputs.json'),
            sha256=digest(root / 'external-inputs.json'), files=len(external)),
        runtime_sha256=digest(r.RESULT), boot=rt['boot'], driver=rt['driver'],
        environment=dict(full['environment'], TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache')),
        targets=TARGETS, commands=[('configure',argv), ('engine', ['/usr/bin/cmake','--build',
            str(root / 'build'),'--target',*TARGETS,'-j','4']),
            ('dependencies', ['/usr/bin/ninja','-C',str(root / 'build'),'-t','deps'])],
        generated_expected={str(root / 'build/ggml/src/ggml-version.h'):hashlib.sha256(q.closure.HEADER.encode()).hexdigest()},
        budgets=dict(wall=1800,cpu=7200,rss_cap=12 << 30,text_cap=32 << 20,file_cap=512 << 20,total_cap=4 << 30),
        total_wall_seconds=3600, active=False, ownership_closed=True, passed=False,
        GPU_executed=False, model_selected=False,
        criteria='Build only: exact original9853Git inputs plus12explicit overlay paths and9CMake-only sources. '
            'Unselected diagnostics/staging excluded;independent max-allocation guard preserved. '
            'Actual compiler/imported linker closure must have zero uncovered inputs. All3 executables '
            'from one source/build. No device,numerical,performance or model qualification inherited.',
        retention='One additional candidate engine/component generation for imminent bounded device and '
            'whole-model comparison. Keep selected6e40/12fixtures untouched. Retire disposable host/build '
            'objects/logs and rejected generation after original source/recipe recovery.')
    if variant == 'down':
        row.update(variant='down',
            retire_before_build=record['moe_split_accumulator']['build']['binaries'],
            criteria='Build only:selected6e40 plus the twelve split-overlay paths,with only four additional '
                'Down changes. Immutable original/row128/row64 selection defaults to original;selected '
                'observer refusal guards and max-allocation fix stay intact. Zero uncovered compiler '
                'or imported linker inputs;three executables from one build. No model qualification inherited.',
            retention='Keep selected6e40+12fixtures untouched. Retire the completed split engine after '
                'exact source/recipe recovery before compiling this additional generation;the new executable '
                'contains its own original-policy control. Retain the current Down component only through '
                'same-build route validation;then retire superseded component artifacts.')
    admit(row, root)
    write(root / 'record.json', row)
    scope['build'] = compact(row)
    scope['source_preparation'] = dict(source_commit=revision, passed=True, paths=len(overlay['paths']),
                                     original_failure_preserved=True, GPU_executed=False)
    write(RESULT, record)
    print('Prepared exact selected-source ' + variant + ' engine/components;commit plan before build.', flush=True)


def run(variant='split'):
    root, key = context(variant)
    record = r.validation.committed(RESULT)
    scope = record[key]
    row = json.loads((root / 'record.json').read_text())
    require(row['stage'] == 'prepared' and compact(row) == scope['build'] and
            not (root / 'owned-build').exists(), 'fresh committed native build required')
    if variant == 'down':
        retirement = scope.get('prior_generation_retirement', {})
        require(retirement.get('stage') == 'retired' and retirement.get('original_inputs_recovered') and
                all(not Path(asset['path']).exists() for asset in row['retire_before_build'].values()),
                'retire the superseded split engine after original input recovery before building')
    allowed = admit(row, root)
    original, owner = copy.deepcopy(row), None
    deadline = time.monotonic() + row['total_wall_seconds']
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(root / 'record.json', dict(row, commands_executed=owner.commands if owner else []))
    def monitor(io, entry):
        next_size = 0
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if time.monotonic() >= next_size:
                require(r.supervision.artifact_bytes(root) <= row['budgets']['total_cap'], 'whole build size budget')
                next_size = time.monotonic() + 1
            if select.select([io.proc.stdout], [], [], .02)[0]:
                io.read(); io.pending.clear()
        io.exit(1)
    try:
        row['stage'] = 'building'
        owner = r.supervision.Owner(root / 'owned-build', persist)
        for label, command in row['commands']:
            remaining = deadline - time.monotonic()
            require(remaining > 1, 'whole build deadline')
            print('CPU build:', label, flush=True)
            entry, out, err = owner.run(label,command,row['environment'],root,interaction=monitor,
                **dict(row['budgets'], wall=min(remaining,row['budgets']['wall'])))
            require(r.supervision.closed(entry) and entry['exit_code'] == 0,
                    label + ' failed:' + (out.read_text(errors='replace') + err.read_text(errors='replace'))[-5000:])
        admit(original, root)
        paths = {Path(line.strip()) for line in out.read_text().splitlines() if line.startswith('    ') and line.strip()}
        paths = {(p if p.is_absolute() else root / 'build' / p).resolve() for p in paths}
        require(len(paths) > 700 and {root/'source'/n for n in [model.GEMM,model.FIXTURE,*model.DIRECT]}
                <= paths, 'incomplete actual compiler closure')
        observed = {}
        for path in sorted(paths):
            sha = digest(path)
            require(allowed.get(str(path)) == sha, 'unfrozen actual compiler input:' + str(path))
            observed[str(path)] = sha
        write(root / 'compiler-inputs.json', observed)
        ninja = (root / 'build/build.ninja').read_text()
        imported = {}
        for name in TARGETS:
            stanza = ninja[ninja.index('build ' + name + ':'):].split('\n\n',1)[0]
            links = shlex.split(next(line.split(' = ',1)[1] for line in stanza.splitlines()
                                    if line.startswith('  LINK_LIBRARIES = ')))
            for path in (Path(p).resolve() for p in links if p.startswith('/')):
                require(allowed.get(str(path)) == digest(path), 'unfrozen imported linker input:' + str(path))
                imported[str(path)] = digest(path)
        row.update(stage='built-not-executed',passed=True,
            binaries={name:dict(path=str(root/'build'/name),sha256=digest(root/'build'/name),
                bytes=(root/'build'/name).stat().st_size) for name in TARGETS},
            actual_compiler_inputs=dict(files=len(observed), path=str(root/'compiler-inputs.json'),
                sha256=digest(root/'compiler-inputs.json'), uncovered=0),
            imported_link_inputs=imported, compile_database_sha256=digest(root/'build/compile_commands.json'))
    except BaseException as error:
        row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row['ownership_closed'] = owner is None or owner.active is None and all(
            r.supervision.closed(c) for c in owner.commands)
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        persist(); scope['build'] = compact(row); write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','ownership_closed','error')}),flush=True)
    require(row['passed'], 'split native build failed')


if __name__ == '__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(sys.argv[1:] in (['prepare'], ['build'], ['prepare-down'], ['build-down']),
                'build_moe_split.py prepare|build|prepare-down|build-down')
        variant = 'down' if sys.argv[1].endswith('-down') else 'split'
        (prepare if sys.argv[1].startswith('prepare') else run)(variant)
