#!/usr/bin/env python3
"""Verify pinned bytes and materialize a proposal overlay; never apply to checkout."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True, help='new proposal directory, outside source checkout')
args = parser.parse_args()
a = Path(__file__).resolve().parent
root = args.source_root.resolve()
out = args.output.resolve()
if out == root or root in out.parents or out.exists():
    parser.error('output must be nonexistent and outside source checkout')
m = json.loads((a / 'manifest.json').read_text())
patch = []
for f in m['files']:
    rel = f['path']
    old = (a / 'base' / rel).read_bytes() if f['base_sha256'] else b''
    new = (a / 'proposed' / rel).read_bytes()
    sha = lambda b: hashlib.sha256(b).hexdigest()
    if f['base_sha256']:
        if sha(old) != f['base_sha256'] or sha((root / rel).read_bytes()) != f['base_sha256']:
            raise SystemExit('base mismatch: ' + rel)
    elif (root / rel).exists():
        raise SystemExit('new path exists: ' + rel)
    if sha(new) != f['new_sha256']:
        raise SystemExit('proposal mismatch: ' + rel)
    patch.extend(difflib.unified_diff(old.decode().splitlines(True), new.decode().splitlines(True),
                 fromfile='a/' + rel if old else '/dev/null', tofile='b/' + rel))
patch = ''.join(patch).encode()
if patch != (a / 'source.diff').read_bytes():
    raise SystemExit('stored diff differs from pinned sources')
out.mkdir(parents=True)
for f in m['files']:
    dest = out / 'proposed' / f['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes((a / 'proposed' / f['path']).read_bytes())
(out / 'source.diff').write_bytes(patch)
(out / 'manifest.json').write_bytes((a / 'manifest.json').read_bytes())
print('Proposal only; SHA256 ' + hashlib.sha256(patch).hexdigest())
