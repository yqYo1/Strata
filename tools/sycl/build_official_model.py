#!/usr/bin/env python3
"""Main-owned official-SDK joint dense/head model build; repository LICENSE.

Freeze one composed source/recipe before compilation. Keep selected6e40 and
its twelve fixtures; retire the superseded second engine before building.
Compilation supplies no GPU, model, performance or teardown qualification.
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
import official_joint_sources as source
import measurement_owner as supervision
import measurement_guards as guards
import host_link_inventory

P, B = source.P, source.B
basis = source.original
OLD_ROOT = B / 'official-prefill-model-v1'
ROOT = B / 'official-prefill-joint-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'official_prefill_joint_build_v1'
SDK = B / 'prefill-expert-staging-v3/sdk-inputs.json'
SDK_SHA = 'c8879f0b8d772979bf2096dc64a3bba49853bd15dad221b46427a922030647d8'
COMPILER = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
COMPILER_SHA = '59eaaf55744c7290fb870cd12049ec11877b5224e333cba33270b433bdb73213'
TARGETS = ['strata', 'sycl_qsa_topk_prefill_entry_test', 'sycl_submission_test',
           'sycl_vllm_native16_test', 'sycl_moe_gu_swiglu_bench',
           'sycl_moe_down_scatter_bench', 'sycl_moe_accum_test']
HEADER = b'#pragma once\n\n#define GGML_VERSION "0.24.0"\n#define GGML_COMMIT  "3cf03257f"\n'
require, digest = source.require, basis.build.digest


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def committed(path):
    raw = basis.build.git(P, 'show', 'HEAD:' + str(path.relative_to(P)))
    require(path.read_bytes() == raw, 'commit exact input: ' + str(path))
    return json.loads(raw)


def controls(revision, *, reuse_revision=None, controller_changes=()):
    """Recover original controls; explicit reuse still verifies current Git bytes.

    Strict build admission remains the default. A diagnostic controller may
    reuse a closed native generation only with an exact, prospective set of
    changed Python controls. Native/compiler/fixture checks remain mandatory.
    """
    require(reuse_revision is not None or not controller_changes,
            'controller changes require an explicit committed reuse revision')
    require(len(set(controller_changes)) == len(controller_changes) and
            all(type(n) is str and n.startswith('tools/sycl/') and n.endswith('.py')
                and '..' not in Path(n).parts for n in controller_changes),
            'invalid controller reuse paths')
    pending, pins, changed = [Path(__file__).resolve()], {}, set()
    while pending:
        path = pending.pop()
        name = str(path.relative_to(P))
        if name in pins:
            continue
        raw = basis.build.git(P, 'show', revision + ':' + name)
        actual = path.read_bytes()
        if reuse_revision is None:
            require(actual == raw, 'controller input changed: ' + name)
        else:
            current = basis.build.git(P, 'show', reuse_revision + ':' + name)
            require(actual == current, 'uncommitted reuse controller: ' + name)
            if actual != raw:
                require(name in controller_changes, 'undeclared reuse controller: ' + name)
                changed.add(name)
        pins[name] = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            names = [v.name for v in node.names] if isinstance(node, ast.Import) else \
                    [node.module] if isinstance(node, ast.ImportFrom) else []
            for module in names:
                local = path.parent / ((module or '').split('.')[0] + '.py')
                if local.is_file():
                    pending.append(local)
    require(reuse_revision is None or changed == set(controller_changes),
            'declared controller reuse changes do not match actual changes')
    return pins


def sdk():
    require(digest(SDK) == SDK_SHA, 'original official SDK inventory changed')
    files = json.loads(SDK.read_text())
    require(len(files) == 10945 and all(digest(p) == sha for p, sha in files.items()),
            'official SDK input changed; do not refresh its original hashes')
    require(digest(COMPILER) == COMPILER_SHA, 'official compiler changed')
    return files


def traced_link_inputs(allowed):
    """Account for GNU ld -t paths, apart from Ninja/warning text.

    Temporary device objects are produced by this owned compiler below its
    private TMPDIR. External linker inputs must match the prospective map.
    No missing external path is accepted as a compiler temporary.
    """
    frozen, generated = {}, set()
    stdout = (ROOT / 'owned-build/compile.stdout').read_text(errors='replace')
    for line in stdout.splitlines():
        value = line.strip()
        path = Path(value)
        if not value or any(c.isspace() for c in value) or not (path.is_absolute() or
                path.suffix in ('.o', '.a') or '.so' in path.name):
            continue
        path = (path if path.is_absolute() else ROOT / 'build' / path).resolve()
        if path.is_relative_to(ROOT / 'build') or path.is_relative_to(ROOT / 'tmp'):
            generated.add(str(path))
            continue
        require(path.is_file() and allowed.get(str(path)) == digest(path),
                'unfrozen actual external linker input: ' + str(path))
        frozen[str(path)] = digest(path)
    require(len(frozen) > 8, 'incomplete actual host linker trace')
    return dict(frozen_inputs=frozen, owned_build_or_compiler_products=sorted(generated),
                uncovered_external_inputs=0,
                scope='Observed GNU ld -t input paths;private generated objects are owned outputs,not external source evidence.')


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('controller_inputs', 'native_sources', 'commands_executed')}
    out['controller_inputs'] = dict(files=len(row['controller_inputs']), aggregate_sha256=hashlib.sha256(
        json.dumps(row['controller_inputs'], sort_keys=True).encode()).hexdigest())
    out['native_sources'] = dict(files=len(row['native_sources']), aggregate_sha256=hashlib.sha256(
        json.dumps(row['native_sources'], sort_keys=True).encode()).hexdigest())
    return out


def save(row):
    write(ROOT / 'record.json', row)
    result = json.loads(RESULT.read_text())
    result[KEY] = compact(row)
    write(RESULT, result)


def previous_build(sdk_files, full):
    """Recover the closed seed before retiring its executable generation.

    Old controller bytes come from their original Git revision. They need not
    equal this deliberately new controller; no old qualification is refreshed.
    """
    row = json.loads((OLD_ROOT / 'record.json').read_text())
    require(committed(RESULT)['official_prefill_model_v1'] == compact(row) and
            row['source_commit'] == 'e4617c85c2f0009eced4484d43c2f53661b76d1e' and
            row['stage'] == 'built-not-executed' and row['passed'] and not row['active'] and
            row['ownership_closed'] and all(supervision.closed(v) for v in row['commands_executed']),
            'closed original official seed required')
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in basis.build.blobs(
        P, row['source_commit'], row['controller_inputs'])}
    require(recovered == row['controller_inputs'], 'original seed controller recovery failed')
    _, native = basis.sources()
    require({n: hashlib.sha256(raw).hexdigest() for n, raw in native.items()} == row['native_sources'],
            'original seed source recovery failed')
    allowed = dict(sdk_files)
    inventory = row['external_manifest']
    require(digest(inventory['path']) == inventory['sha256'], 'original external inventory changed')
    external = json.loads(Path(inventory['path']).read_text())
    require(len(external) == inventory['files'] and all('/MistVVK/' not in p and digest(p) == sha
            for p, sha in external.items()), 'original external input changed')
    allowed.update(external)
    for name, raw in native.items():
        path = OLD_ROOT / 'source' / name
        require(path.read_bytes() == raw, 'original native input changed: ' + name)
        allowed[str(path)] = row['native_sources'][name]
    for group in full['dependencies']:
        expected = group['source_sha256']
        recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in basis.build.blobs(
            Path(group['root']), group['revision'], expected)}
        require(recovered == expected, 'original dependency Git recovery failed')
        for name, sha in expected.items():
            path = (Path(group['root']) / name).resolve()
            require(digest(path) == sha, 'original dependency changed: ' + name)
            allowed[str(path)] = sha
    allowed.update(row['generated_expected'])
    proof = row['compiled_inputs']
    require(digest(proof['path']) == proof['sha256'] and proof['uncovered'] == 0,
            'original compiler inventory changed')
    used = json.loads(Path(proof['path']).read_text())
    require(len(used) == proof['files'] and all(allowed.get(p) == sha == digest(p)
            for p, sha in used.items()), 'original actual compiler input is not recovered')
    require(digest(OLD_ROOT / 'build/compile_commands.json') == row['compile_database_sha256'],
            'original compile recipe changed')
    for item in row['binaries'].values():
        require(digest(item['path']) == item['sha256'], 'original executable changed')
    gate = committed(RESULT)['official_prefill_device_v3']
    require(gate['passed'] and gate['complete'] and gate['ownership_closed'] and
            gate['binaries'] == row['binaries'] and gate['fresh_faults'] == [],
            'original component result changed')
    rt = gate['runtime_manifest']
    require(digest(rt['path']) == rt['sha256'], 'original runtime inventory changed')
    runtime_files = json.loads(Path(rt['path']).read_text())
    require(len(runtime_files) == rt['files'] and all(digest(p) == sha for p, sha in runtime_files.items()),
            'original runtime provider changed')
    return row, external, gate, runtime_files


def prepare():
    result = committed(RESULT)
    require(KEY not in result and not ROOT.exists(), 'fresh single model transition only')
    full, native = source.sources()
    revision = basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = controls(revision)
    sdk_files = sdk()
    original, external, gate, runtime_files = previous_build(sdk_files, full)
    correction = result['joint_prefill_heads_v1']['correction']
    require(not correction['source_review_pending'] and
            not correction['source_review']['source_blockers_remaining_from_this_review'] and
            correction['host_test']['result'] == dict(kind='prefill_head_scope', cases=15, checks=315, status='PASS') and
            source.original.aggregate(native) == correction['source_construction']['aggregate_sha256'],
            'exact corrected source and host/source gates required')
    env = dict(original['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    require(all('/MistVVK/' not in v for v in env.values()), 'legacy SDK in official environment')
    links = host_link_inventory.inventory(COMPILER, env)
    external.update(links['files'])
    for path in ('/usr/bin/cmake', '/usr/bin/ninja', '/usr/bin/cc', '/usr/bin/ar', '/usr/bin/ranlib',
                 '/usr/bin/as', '/usr/bin/ld', '/usr/bin/git'):
        external[str(Path(path).resolve())] = digest(path)
    require(all(digest(p) == sha for p, sha in external.items()), 'prospective external input changed')
    require(links == original['host_link_inputs'], 'seed compiler host inputs changed')
    cfg = [v.replace(str(OLD_ROOT), str(ROOT)) for v in original['configure_argv']]
    require(not any('/MistVVK/' in v for v in cfg), 'legacy compiler in new configure recipe')
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'tmp', 'cache'):
        (ROOT / name).mkdir()
    for name, raw in native.items():
        path = ROOT / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    write(ROOT / 'external-inputs.json', external)
    write(ROOT / 'runtime-basis.json', runtime_files)
    row = dict(stage='prepared', source_commit=revision, controller_inputs=pins,
        native_sources={n: hashlib.sha256(raw).hexdigest() for n, raw in native.items()},
        selected_binary=full['binary'], source_basis=dict(selected=basis.split.BASE, down=basis.DOWN,
            topk=basis.TOPK, fixture=basis.FIXTURE, gu_progress_fixture=basis.GU_PROGRESS,
            joint_source=source.JOINT, seed_native_source=original['source_commit'],
            seed_recipe_git_commit=revision, fully_submitted_MoE_jobs_per_checkpoint=8,
            outside_prefill_phase_section_unchanged=True),
        sdk=dict(path=str(SDK), sha256=SDK_SHA, files=len(sdk_files)),
        compiler=dict(path=COMPILER, sha256=COMPILER_SHA), host_link_inputs=links,
        packages=subprocess.check_output(['dpkg-query', '-W', '-f=${Package} ${Version}\n',
            'intel-oneapi-dpcpp-cpp-2026.2', 'intel-oneapi-compiler-dpcpp-cpp-runtime-2026.2',
            'intel-oneapi-umf-1.2'], text=True, timeout=15).splitlines(),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=guards.driver_guard(),
        dependencies=[{k: g[k] for k in ('root', 'revision', 'count', 'aggregate_sha256')}
                      for g in full['dependencies']],
        external_manifest=dict(path=str(ROOT / 'external-inputs.json'),
            sha256=digest(ROOT / 'external-inputs.json'), files=len(external)),
        runtime_basis=dict(path=str(ROOT / 'runtime-basis.json'), sha256=digest(ROOT / 'runtime-basis.json'),
            files=len(runtime_files), original_component_source=gate['source_commit'],
            environment=gate['environment'], canary=gate['canary']),
        configure_argv=cfg, environment=env, targets=TARGETS,
        commands=[('configure', cfg), ('compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'),
            '--target', *TARGETS, '-j', '4']), ('dependencies', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'])],
        generated_expected={str(ROOT / 'build/ggml/src/ggml-version.h'): hashlib.sha256(HEADER).hexdigest()},
        budgets=dict(wall=1800, cpu=7200, rss_cap=12 << 30, text_cap=16 << 20,
                     file_cap=512 << 20, total_cap=4 << 30), whole_wall_seconds=3600,
        active=False, ownership_closed=True, passed=False, GPU_executed=False,
        inference_adopted=False, ordinary_runtime_teardown_qualified=False,
        retention='One joint replacement candidate for imminent real-owner release/restore and matched model comparison. '
            'Preserve selected6e40+12fixtures;recover/retire officiale461before compile and this candidate after selection.',
        next_gate='Commit plan;retire officiale461;build only. Fresh same-build bounded production-GU/provider '
            'progression before one small model,then32K+ long/agent/thinking PP/TTFT and independent repeated '
            'decode,handoff,physical262144fill/tails/RESTORE. No inherited model qualification.')
    save(row)
    print('Prepared one official model source/recipe; commit plan before compiling.', flush=True)


def admit(row, *, reuse_revision=None, controller_changes=()):
    require(committed(RESULT)[KEY] == compact(row), 'commit exact model build plan')
    require(controls(row['source_commit'], reuse_revision=reuse_revision,
            controller_changes=controller_changes) == row['controller_inputs'], 'build controller changed')
    full, native = source.sources()
    require({n: hashlib.sha256(raw).hexdigest() for n, raw in native.items()} == row['native_sources'],
            'composed model source changed')
    allowed = sdk()
    require(subprocess.check_output(['dpkg-query', '-W', '-f=${Package} ${Version}\n',
        'intel-oneapi-dpcpp-cpp-2026.2', 'intel-oneapi-compiler-dpcpp-cpp-runtime-2026.2',
        'intel-oneapi-umf-1.2'], text=True, timeout=15).splitlines() == row['packages'],
        'official package versions changed')
    require(host_link_inventory.inventory(COMPILER, row['environment']) == row['host_link_inputs'],
            'compiler-resolved host linker inputs changed')
    require(digest(row['external_manifest']['path']) == row['external_manifest']['sha256'], 'external inventory changed')
    external = json.loads(Path(row['external_manifest']['path']).read_text())
    require(len(external) == row['external_manifest']['files'] and
            all(digest(p) == sha for p, sha in external.items()), 'external input changed')
    allowed.update(external)
    for name, raw in native.items():
        path = ROOT / 'source' / name
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                'frozen source changed: ' + name)
        allowed[str(path.resolve())] = row['native_sources'][name]
    for group in full['dependencies']:
        for name, sha in group['source_sha256'].items():
            path = Path(group['root']) / name
            require(digest(path) == sha, 'dependency input changed: ' + name)
            allowed[str(path.resolve())] = sha
    require(digest(full['binary']['path']) == full['binary']['sha256'], 'selected fallback changed')
    runtime = row['runtime_basis']
    require(digest(runtime['path']) == runtime['sha256'], 'runtime provider basis changed')
    files = json.loads(Path(runtime['path']).read_text())
    require(len(files) == runtime['files'] and all(digest(p) == sha for p, sha in files.items()),
            'runtime provider bytes changed')
    allowed.update(row['generated_expected'])
    return allowed


def run():
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and not (ROOT / 'owned-build').exists(), 'fresh committed build only')
    retirement = committed(RESULT)['official_prefill_model_v1'].get('joint_transition_retirement', {})
    require(retirement.get('complete') and not OLD_ROOT.exists(),
            'recover and retire superseded second generation before compile')
    allowed = admit(row)
    initial, owner = copy.deepcopy(row), None
    deadline = time.monotonic() + row['whole_wall_seconds']
    def persist():
        write(ROOT / 'record.json', dict(row, commands_executed=owner.commands if owner else [],
              active=bool(owner and owner.active)))
    def monitor(io, entry):
        next_size = 0
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if time.monotonic() >= next_size:
                require(supervision.artifact_bytes(ROOT) <= row['budgets']['total_cap'], 'whole model build size budget')
                next_size = time.monotonic() + 1
            if select.select([io.proc.stdout], [], [], .02)[0]:
                io.read()
                io.pending.clear()
        io.exit(1)
    try:
        row['stage'] = 'building'
        owner = supervision.Owner(ROOT / 'owned-build', persist)
        for label, argv in row['commands']:
            print('CPU build:', label, flush=True)
            entry, out, err = owner.run(label, argv, row['environment'], ROOT, interaction=monitor,
                **dict(row['budgets'], wall=min(deadline - time.monotonic(), row['budgets']['wall'])))
            require(supervision.closed(entry) and entry['exit_code'] == 0,
                    label + ' failed: ' + (out.read_text(errors='replace') + err.read_text(errors='replace'))[-4000:])
        admit(initial)
        found = {str((Path(line.strip()) if Path(line.strip()).is_absolute()
                      else ROOT / 'build' / line.strip()).resolve())
                 for line in out.read_text().splitlines() if line.startswith('    ') and line.strip()}
        require(len(found) >= 1000 and {str(ROOT / 'source' / n) for n in
                ('src/prefill/prefill.cpp', 'src/prefill/sycl/gemm16.cpp', *basis.TOPK_FILES[:1])} <= found,
                'incomplete actual compiler closure')
        require(all('/MistVVK/' not in p and allowed.get(p) == digest(p) for p in found),
                'unplanned or changed actual compiler input')
        write(ROOT / 'compiler-inputs.json', {p: allowed[p] for p in sorted(found)})
        ninja = (ROOT / 'build/build.ninja').read_text()
        imported = {}
        for name in TARGETS:
            stanza = ninja[ninja.index('build ' + name + ':'):].split('\n\n', 1)[0]
            links = shlex.split(next(v.split(' = ', 1)[1] for v in stanza.splitlines()
                                    if v.startswith('  LINK_LIBRARIES = ')))
            for path in (Path(p).resolve() for p in links if p.startswith('/')):
                if path.is_relative_to(ROOT / 'build'):
                    continue
                require(allowed.get(str(path)) == digest(path), 'unfrozen imported link input: ' + str(path))
                imported[str(path)] = digest(path)
        require(supervision.artifact_bytes(ROOT) <= row['budgets']['total_cap'], 'final whole model size budget')
        row.update(stage='built-not-executed', passed=True,
            binaries={name: dict(path=str(ROOT / 'build' / name), sha256=digest(ROOT / 'build' / name),
                bytes=(ROOT / 'build' / name).stat().st_size) for name in TARGETS},
            compiled_inputs=dict(files=len(found), path=str(ROOT / 'compiler-inputs.json'),
                sha256=digest(ROOT / 'compiler-inputs.json'), uncovered=0),
            imported_link_inputs=imported, actual_link_inputs=traced_link_inputs(allowed),
            compile_database_sha256=digest(ROOT / 'build/compile_commands.json'))
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row.update(active=False, ownership_closed=bool(owner and all(supervision.closed(v) for v in owner.commands)))
        require(owner is None or owner.active is None, 'build ownership remains unresolved')
        row['commands_executed'] = owner.commands if owner else []
        save(row)
    require(row['passed'] and row['ownership_closed'], row.get('error', 'build did not close'))
    print('Official model and same-build fixtures compiled; GPU execution remains unqualified.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build'), 'use prepare|build')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        (prepare if sys.argv[1] == 'prepare' else run)()
