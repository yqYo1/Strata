#!/usr/bin/env python3
"""Main-owned whole-engine extension of our retained fused-profile generation.

Repository LICENSE. Reuses our component builder and original source proofs;
does not refresh its old qualification or overwrite the selected6e40 fallback.
This builds a diagnostic candidate, not a restored or adopted production route.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import shlex
import sys
import time

sys.dont_write_bytecode = True
import build_fused_profile as n

r, P, B, ROOT, RESULT = n.r, n.P, n.B, n.ROOT, n.RESULT
require, digest, write = n.require, n.digest, n.write
PREFIXES = n.PREFIXES + ['tools/sycl/build_profile_engine.py']
RECORD = ROOT / 'engine-build.json'
DRIFT = {'bench/sycl/client_resources.py', 'bench/sycl/hardware_conditions.py',
         'tests/core/test_client_resources.py'}


def original():
    """Recover original inputs; admit only three explicitly uncompiled changes."""
    container = r.validation.committed(RESULT)
    scope = container['event_reconciliation']['fused_command_profile']
    row = json.loads((ROOT / 'record.json').read_text())
    require(scope['build'] == n.compact(row) and row['passed'] and row['ownership_closed']
        and not row['active'] and not row['engine_built'], 'closed component generation required')
    device = scope['device']
    require(device['passed'] and device['ownership_closed'] and not device['active']
        and device['native_source_commit'] == row['source_commit'], 'original component execution required')
    for group in [row['project'], *row['dependencies']]:
        recovered = {name: hashlib.sha256(raw).hexdigest() for name, raw in
                     n.build.blobs(Path(group['root']), group['revision'], group['source_sha256'])}
        require(recovered == group['source_sha256'], 'original source is unrecoverable')
    current = n.build.pin(P, n.PREFIXES)['source_sha256']
    old = row['project']['source_sha256']
    changed = {k for k in set(old) | set(current) if old.get(k) != current.get(k)}
    require(changed == DRIFT, 'unexpected component snapshot drift: ' + str(sorted(changed)))
    manifest = row['compiled_inputs']['manifest']
    require(digest(manifest['path']) == manifest['sha256'], 'original compiled manifest changed')
    compiled = json.loads(Path(manifest['path']).read_text())
    require(len(compiled) == 1385 and all(digest(p) == h for p, h in compiled.items()),
            'actual component compiled input changed')
    require(not {str(P / name) for name in DRIFT} & set(compiled), 'drift includes a compiled input')
    for fixture in row['fixtures'].values():
        require(digest(fixture['path']) == fixture['sha256'] and
                Path(fixture['path']).stat().st_size == fixture['bytes'], 'component fixture changed')
    return row, dict(original_source_commit=row['source_commit'], recovered_source_groups=3,
        compiled_inputs=1385, fixtures=3, original_record_sha256=digest(ROOT / 'record.json'),
        original_compiler_manifest=manifest,
        uncompiled_controller_changes={k: dict(before=old.get(k), after=current.get(k)) for k in sorted(changed)},
        limits='Original component qualification only. Three Python changes are outside actual compiled closure. '
            'Old source hashes and original failed broad-snapshot admission are not rewritten. '
            'New whole engine needs its own source closure and model/observer-cost gates.')


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('project', 'dependencies', 'external_inputs', 'commands')}
    out['source_groups'] = [{k: g[k] for k in ('root', 'revision', 'count', 'aggregate_sha256')}
                            for g in [row['project'], *row['dependencies']]]
    return out


def admit(row):
    require(n.build.pin(P, PREFIXES)['source_sha256'] == row['project']['source_sha256'],
            'whole-engine committed inputs changed')
    _, proof = original()
    require(proof == row['component_reuse'], 'component reuse proof changed')
    for dep in row['dependencies']:
        require(n.build.pin(Path(dep['root']), ['.'])['source_sha256'] == dep['source_sha256'],
                'independent dependency changed')
    require(n.sdk_inputs() == row['sdk'], 'SDK inputs changed')
    for path, sha in row['external_inputs'].items():
        require(digest(path) == sha, 'external build input changed: ' + path)
    for tool in row['tools'].values():
        require(digest(tool['path']) == tool['sha256'], 'build tool changed')
    rt = r.validation.committed(r.RESULT)
    require(digest(r.RESULT) == row['runtime_sha256'] and r.current(rt)['binary'] == row['selected_binary'],
            'qualified runtime or original selected engine changed')


def dependencies(row, text):
    expected = {}
    for group in [row['project'], *row['dependencies']]:
        expected.update({str((Path(group['root']) / k).resolve()): h for k, h in group['source_sha256'].items()})
    expected.update({str(Path(v['path']).resolve()): n.SYSTEM_HEADER_UPDATES.get(v['path'], (None, v['sha256']))[1]
                     for v in json.loads(n.lineage.SDK.read_text())})
    expected.update(row['external_inputs'])
    paths = {Path(line.strip()) for line in text.splitlines() if line.startswith('    ') and line.strip()}
    paths = {(p if p.is_absolute() else ROOT / 'build' / p).resolve() for p in paths}
    require({P / name for name in ('src/program/generate.cpp', 'src/program/dense_prefill_nll.cpp',
        'src/sycl/command_profile.cpp', 'src/prefill/sycl/gemm16.cpp')} <= paths,
        'whole-engine dependency closure misses required translation units')
    require(len(paths) >= 1385, 'incomplete whole-engine dependency closure')
    observed, generated = {}, {}
    for path in sorted(paths):
        sha = digest(path)
        if str(path) in expected:
            require(sha == expected[str(path)], 'compiled input differs: ' + str(path))
        else:
            require(row['generated_expected'].get(str(path)) == sha,
                    'unplanned generated/compiler input: ' + str(path))
            generated[str(path)] = sha
        observed[str(path)] = sha
    manifest = ROOT / 'engine-compiler-inputs.json'
    write(manifest, observed)
    return dict(files=len(observed), manifest=dict(path=str(manifest), sha256=digest(manifest)),
                generated_sha256=generated, uncovered_files=0)


def prepare():
    require(not RECORD.exists() and not (ROOT / 'build/strata').exists(), 'fresh engine extension only')
    old, proof = original()
    rt = r.validation.committed(r.RESULT)
    full = r.current(rt)
    row = dict(stage='prepared', source_commit=n.build.git(P, 'rev-parse', 'HEAD').decode().strip(),
        project=n.build.pin(P, PREFIXES), dependencies=old['dependencies'], component_reuse=proof,
        sdk=n.sdk_inputs(), external_inputs=n.external_inputs(rt), tools=old['tools'],
        configure_argv=old['configure_argv'], environment=old['environment'],
        generated_expected=old['generated_expected'], selected_binary=full['binary'],
        runtime_sha256=digest(r.RESULT), boot=rt['boot'], driver=rt['driver'],
        build_command=['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', 'strata', '-j', '4'],
        dependency_command=old['dependency_command'],
        active=False, ownership_closed=True, passed=False, GPU_executed=False,
        model_qualified=False, observer_cost_qualified=False, production_selected=False,
        budgets=dict(wall=1800, cpu=7200, rss_cap=12 << 30, text_cap=32 << 20,
                     file_cap=256 << 20, total_cap=4 << 30),
        question='Extend retained diagnostic component generation to the whole engine. Build only;no model execution. '
            'Current sources include unadopted restoration and independent diagnostics relative to6e40;'
            'they are not an isolated historical-regression counterfactual.',
        criteria='Original1385 actual inputs and3qualified component ELF hashes unchanged;all original sources '
            'recoverable. Current whole-engine source/SDK/MKL/tools/environment committed before build. '
            'Owned configure/build/dependency scan exit0;zero uncovered actual compiler inputs;'
            'engine hash and imported links verified;original6e40+12fixtures remain unchanged. '
            'Compilation does not permit speed adoption or long attribution. Small-model safety/dispatch/stream '
            'qualification and a prospective observer-cost plan are required before long model profiling.',
        retention='Reuse existing fused-profile generation and objects;no third generation. '
            'Original selected6e40 and12fixtures stay untouched. Preserve original component record and1385manifest '
            'with3ELFs;write distinct engine manifest. Review at build closure/model qualification or abandonment;'
            'retire normal build logs after verified extraction.')
    write(RECORD, row)
    container = r.validation.committed(RESULT)
    container['event_reconciliation']['fused_command_profile']['engine_build'] = compact(row)
    write(RESULT, container)
    print('Prepared diagnostic whole-engine build;commit generated plan before build.', flush=True)


def run():
    container = r.validation.committed(RESULT)
    scope = container['event_reconciliation']['fused_command_profile']
    row = json.loads(RECORD.read_text())
    require(scope['engine_build'] == compact(row) and row['stage'] == 'prepared'
        and not (ROOT / 'owned-engine-build').exists(), 'single-use committed engine plan required')
    admit(row)
    initial = copy.deepcopy(row)
    owner = None
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(RECORD, dict(row, commands=owner.commands if owner else []))
    def monitor(io, entry):
        next_size = 0
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if time.monotonic() >= next_size:
                require(r.supervision.artifact_bytes(ROOT) <= row['budgets']['total_cap'], 'build size cap')
                next_size = time.monotonic() + 1
            if select.select([io.proc.stdout], [], [], .02)[0]:
                io.read(); io.pending.clear()
        io.exit(1)
        require(r.supervision.artifact_bytes(ROOT) <= row['budgets']['total_cap'], 'final build size cap')
    try:
        row['stage'] = 'building'
        owner = r.supervision.Owner(ROOT / 'owned-engine-build', persist); persist()
        for label, command in [('configure', row['configure_argv']), ('compile', row['build_command']),
                               ('dependencies', row['dependency_command'])]:
            entry, out, err = owner.run(label, command, row['environment'], P, interaction=monitor, **row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code'] == 0,
                    label + ' failed: ' + (out.read_text(errors='replace') + err.read_text(errors='replace'))[-6000:])
            if label == 'configure':
                admit(initial)
            if label == 'dependencies':
                row['compiled_inputs'] = dependencies(row, out.read_text())
        binary = ROOT / 'build/strata'
        ninja = (ROOT / 'build/build.ninja').read_text()
        stanza = ninja[ninja.index('build strata:'):].split('\n\n', 1)[0]
        links = shlex.split(next(x.split(' = ', 1)[1] for x in stanza.splitlines()
                                 if x.startswith('  LINK_LIBRARIES = ')))
        origins = dict(row['external_inputs'])
        for path in (Path(p).resolve() for p in links if p.startswith('/')):
            require(str(path) in origins and digest(path) == origins[str(path)], 'unplanned imported link: ' + str(path))
        admit(initial)
        row.update(stage='built-not-executed', passed=True,
                   binary=dict(path=str(binary), sha256=digest(binary), bytes=binary.stat().st_size))
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row['ownership_closed'] = owner is None or owner.active is None and all(r.supervision.closed(c) for c in owner.commands)
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        persist(); scope['engine_build'] = compact(row); write(RESULT, container)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'binary', 'error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['build']), 'build_profile_engine.py prepare|build')
    with (B / 'owned-v0141-measurement.lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r.guards.no_model_owners([])
        sys.exit(prepare() if sys.argv[1] == 'prepare' else run())
