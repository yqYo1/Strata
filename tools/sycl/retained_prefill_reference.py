#!/usr/bin/env python3
"""Verify the retained official generation without a retired ancestor ELF.

Main-authored; repository LICENSE. A seed executable is not a compiler input
or a runtime dependency of the completed reference. Recover its still-consumed
source receipt through the existing constructor, while checking the reference's
actual native/compiler/link/provider inputs and all seven executable bytes.
"""
import hashlib
import json
from pathlib import Path

import build_official_model as build

require, digest = build.require, build.digest
CONTROL_CHANGES = ('tools/sycl/build_official_model.py', 'tools/sycl/bench_engine.py')


def verify(candidate, providers, revision):
    data = build.committed(build.RESULT)
    row = json.loads((build.ROOT / 'record.json').read_text())
    require(build.compact(row) == data[build.KEY] and row['passed'] and
            row['stage'] == 'built-not-executed' and row['ownership_closed'] and not row['active'] and
            all(build.supervision.closed(v) for v in row['commands_executed']), 'closed reference build required')
    require(build.controls(row['source_commit'], reuse_revision=revision,
            controller_changes=CONTROL_CHANGES) == row['controller_inputs'], 'reference controls changed')
    # Candidate admission already hashes these installed inputs. Equality is
    # mandatory before reusing that check, not permission to update old hashes.
    for key in ('compiler', 'sdk', 'external_manifest', 'host_link_inputs', 'packages', 'dependencies'):
        require(row[key] == candidate[key], 'reference/candidate shared input differs: ' + key)
    full, native = build.source.sources()
    require(len(native) == 554 and {n:hashlib.sha256(raw).hexdigest() for n,raw in native.items()}
            == row['native_sources'], 'reference native Git reconstruction differs')
    allowed = dict(providers)
    for key in ('sdk', 'external_manifest'):
        allowed.update(json.loads(Path(row[key]['path']).read_text()))
    for name, raw in native.items():
        path = build.ROOT / 'source' / name
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                'reference frozen native source differs: ' + name)
        allowed[str(path.resolve())] = row['native_sources'][name]
    for group in full['dependencies']:
        recovered = {n:hashlib.sha256(raw).hexdigest() for n,raw in
                     build.basis.build.blobs(Path(group['root']), group['revision'], group['source_sha256'])}
        require(recovered == group['source_sha256'], 'reference dependency Git recovery differs')
        for name, sha in recovered.items():
            path = (Path(group['root']) / name).resolve()
            require(digest(path) == sha, 'reference dependency bytes changed')
            allowed[str(path)] = sha
    allowed.update(row['generated_expected'])
    proof = row['compiled_inputs']
    require(digest(proof['path']) == proof['sha256'] and proof['uncovered'] == 0,
            'reference actual compiler-input manifest changed')
    used = json.loads(Path(proof['path']).read_text())
    require(len(used) == proof['files'] == 1447 and all('/MistVVK/' not in p and
            allowed.get(p) == sha == digest(p) for p,sha in used.items()), 'reference compiler input not recovered')
    require(str(Path(full['binary']['path']).resolve()) not in used, 'ancestor ELF is a live compiler input')
    require(digest(build.ROOT / 'build/compile_commands.json') == row['compile_database_sha256'],
            'reference actual compiler recipe changed')
    require(set(row['binaries']) == set(build.TARGETS) and
            all(digest(v['path']) == v['sha256'] for v in row['binaries'].values()), 'reference executable changed')
    gate = data['official_prefill_joint_device_v1']
    require(gate['passed'] and gate['complete'] and gate['ownership_closed'] and not gate['active']
            and len(gate['processes']) == 13 and all(v['passed'] for v in gate['processes'])
            and not gate['fresh_faults'] and gate['binaries'] == row['binaries']
            and gate['native_source_commit'] == row['source_commit'], 'same-generation reference device proof differs')
    inv = gate['runtime_manifest']; require(digest(inv['path']) == inv['sha256'] and
        json.loads(Path(inv['path']).read_text()) == providers and len(providers) == inv['files'],
        'reference device provider qualification differs')
    return row
