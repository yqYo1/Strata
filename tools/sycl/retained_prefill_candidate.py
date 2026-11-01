#!/usr/bin/env python3
"""Verify the selected eight-ELF generation without its retired fallback.

Main-authored; repository LICENSE. Recover the original source and recipe,
not current worktree implementations. All same-build fixtures and actual
compiler/provider inputs remain mandatory; no new GPU qualification is claimed.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import build_official_model as base
import measure_gdn_scheduling as runtime
import prefill_strategy_sources as original

P, B, RESULT = base.P, base.B, base.RESULT
ROOT = B / 'official-prefill-strategies-v1'
KEY = 'official_prefill_strategies_build_v2'
NATIVE = 'f6cea2f63315e2598d63cb04d5d0b6443dc21a24'
require, digest = base.require, base.digest


def verify():
    data = base.committed(RESULT)
    row = json.loads((ROOT / 'record-v2.json').read_text())
    require(base.compact(row) == data[KEY] and row['passed'] and
            row['source_commit'] == NATIVE and not row['active'] and row['ownership_closed'] and
            all(base.supervision.closed(v) for v in row['commands_executed']),
            'closed original candidate build required')
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                 base.basis.build.blobs(P, NATIVE, row['controller_inputs'])}
    require(recovered == row['controller_inputs'], 'original build controls not recoverable')
    full, native = original.sources()
    require(len(native) == 555 and {n: hashlib.sha256(raw).hexdigest() for n, raw in native.items()}
            == row['native_sources'], 'original candidate native source not recoverable')
    gate = data['official_prefill_strategies_device_v2']
    require(gate['passed'] and gate['complete'] and not gate['active'] and gate['ownership_closed']
            and not gate['fresh_faults'] and len(gate['processes']) == 21 and
            all(v['passed'] for v in gate['processes']) and gate['binaries'] == row['binaries'] and
            gate['native_source_commit'] == NATIVE, 'same-build 21-case qualification changed')
    providers = runtime.inventory(gate['runtime_manifest'])
    probe = runtime.qualified_probe(data, providers)
    require(probe == row['native_validation_probe'], 'candidate native probe changed')
    allowed = base.sdk()
    allowed.update(runtime.inventory(row['external_manifest']))
    allowed.update(providers)
    for name, raw in native.items():
        path = ROOT / 'source' / name
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                'retained native source changed: ' + name)
        allowed[str(path.resolve())] = row['native_sources'][name]
    # The recoverable source receipt holds every dependency hash; the build
    # recipe stores its compact descriptor. Check both representations.
    descriptors = [{k: v for k, v in group.items() if k != 'source_sha256'}
                   for group in full['dependencies']]
    require(descriptors == row['dependencies'], 'dependency recipe differs')
    for group in full['dependencies']:
        found = {n: hashlib.sha256(raw).hexdigest() for n, raw in base.basis.build.blobs(
            Path(group['root']), group['revision'], group['source_sha256'])}
        require(found == group['source_sha256'] and len(found) == group['count'] and
                hashlib.sha256(json.dumps(found, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest() ==
                group['aggregate_sha256'], 'dependency Git recovery differs')
        for name, sha in found.items():
            path = (Path(group['root']) / name).resolve()
            require(digest(path) == sha, 'installed dependency changed: ' + str(path))
            allowed[str(path)] = sha
    allowed.update(row['generated_expected'])
    actual = row['compiled_inputs']
    require(digest(actual['path']) == actual['sha256'], 'compiler-input manifest changed')
    inputs = json.loads(Path(actual['path']).read_text())
    require(len(inputs) == actual['files'] == 1449 and all('/MistVVK/' not in p and
            allowed.get(p) == sha == digest(p) for p, sha in inputs.items()),
            'actual candidate compiler inputs not recovered')
    require(digest(ROOT / 'build/compile_commands.json') ==
            row['compile_command_attestation']['database_sha256'], 'actual compile recipe changed')
    require(row['actual_link_inputs']['uncovered_external_inputs'] == 0 and all(
            allowed.get(p) == sha == digest(p) for p, sha in
            row['actual_link_inputs']['frozen_inputs'].items()), 'actual external linker inputs changed')
    require(set(row['binaries']) == set(row['targets']) and len(row['binaries']) == 8 and all(
            digest(v['path']) == v['sha256'] and Path(v['path']).stat().st_size == v['bytes']
            for v in row['binaries'].values()), 'candidate engine or fixture bytes changed')
    packages = subprocess.check_output(['dpkg-query', '-W', '-f=${Package} ${Version}\n',
        'intel-oneapi-dpcpp-cpp-2026.2', 'intel-oneapi-compiler-dpcpp-cpp-runtime-2026.2',
        'intel-oneapi-umf-1.2'], text=True, timeout=15).splitlines()
    require(packages == row['packages'], 'qualified package versions changed')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'] and
            base.guards.driver_guard() == row['driver'], 'qualified boot/driver changed')
    return row, providers, probe


if __name__ == '__main__':
    import fcntl
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        row, providers, _ = verify()
        print(json.dumps(dict(source_commit=row['source_commit'], native_files=555,
            compiler_inputs=1449, binaries=len(row['binaries']), providers=len(providers),
            original_build_recovered=True, GPU_executed=False)))
