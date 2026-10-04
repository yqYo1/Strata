#!/usr/bin/env python3
"""Inventory fixed source trees; file identity is NOT semantic or runtime parity."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

UPSTREAM = '99f3dbd0b21d1401b3769e0c0d963913607f380b'
LEGACY = '22e61b9'
GGML = '3cf03257f219afbe7334045ff7c6a06ac68c627d'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def tree(root, ref, prefixes):
    names = git(root, 'ls-tree', '-r', '--name-only', ref, '--', *prefixes).decode().splitlines()
    # Read all blobs in one git process (no working-tree or timestamp dependency).
    batch = subprocess.run(['git', '-C', str(root), 'cat-file', '--batch'],
        input=''.join(f'{ref}:{p}\n' for p in names).encode(), check=True, stdout=subprocess.PIPE).stdout
    result, pos = {}, 0
    for path in names:
        end = batch.index(b'\n', pos)
        size = int(batch[pos:end].split()[2])
        result[path] = batch[end + 1:end + 1 + size]
        pos = end + 2 + size
    return result


def digest(blob):
    return hashlib.sha256(blob).hexdigest() if blob is not None else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ggml', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    legacy = git(root, 'rev-parse', LEGACY).decode().strip()
    scope = ['src', 'include', 'cmake', 'CMakeLists.txt', 'third_party/ggml']
    upstream, old = tree(root, UPSTREAM, scope), tree(root, legacy, scope)
    # Conservative superset: ALL pinned CUDA and CPU backend sources, plus common
    # layouts and headers. This is not a resolved CMake build or include graph.
    deps = tree(a.ggml, GGML, ['ggml/src/ggml-cuda', 'ggml/src/ggml-cpu',
        'ggml/src/ggml-common.h', 'ggml/src/ggml-quants.c', 'ggml/src/ggml-quants.h',
        'ggml/include', 'ggml/src/CMakeLists.txt', 'ggml/CMakeLists.txt'])
    records = []
    for path in sorted(upstream.keys() | old.keys()):
        u, o = upstream.get(path), old.get(path)
        kind = ('legacy_only' if u is None else 'upstream_only' if o is None else
                'identical' if u == o else 'modified')
        records.append(dict(path=path, upstream_sha256=digest(u), legacy_sha256=digest(o),
            source_relation=kind, semantic_parity='unverified', runtime_parity='unverified'))
    cmake = upstream['CMakeLists.txt'].decode('utf-8-sig')
    literal_cu = sorted(set(re.findall(r'\b(?:src|tests)/[\w/.-]+\.cu\b', cmake)))
    # Include all headers, with full hashes; name scanning would miss macros,
    # overloads, methods, and inline entry points. None are marked verified here.
    result = dict(schema=1, upstream=UPSTREAM, legacy=legacy, ggml=GGML,
        scope=scope, limitations=[
            'Inventory only; no parity verdict follows from source identity.',
            'CMake literal .cu references are not a resolved build graph.',
            'GGML directory inventory is a conservative superset, not all transitive dependencies.',
            'Header files are complete in scope; per-entry-point semantic audit remains pending.',
            'Working-tree changes and runtime environment are outside these fixed snapshots.'],
        files=records, cmake_literal_cu=literal_cu,
        upstream_cu_not_literal=sorted(p for p in upstream if p.endswith('.cu') and p not in literal_cu),
        external_sources=[dict(path=p, sha256=digest(b)) for p, b in sorted(deps.items())])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    counts = {k: sum(r['source_relation'] == k for r in records)
              for k in ('identical', 'modified', 'legacy_only', 'upstream_only')}
    print(json.dumps(dict(files=len(records), relations=counts, literal_cu=len(literal_cu),
                          external_files=len(deps)), sort_keys=True))


if __name__ == '__main__':
    main()
