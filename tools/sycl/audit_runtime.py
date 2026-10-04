#!/usr/bin/env python3
"""Inventory CUDA API candidates in the pinned source, independent of port code.

This is a lexical inventory, not an AST/call graph or proof that a path executes.
It includes programs, tests, inactive preprocessor branches and compatibility code.
"""
import argparse
import bisect
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess

BASELINE = '99f3dbd0b21d1401b3769e0c0d963913607f380b'

def git(*args):
    return subprocess.check_output(['git', *args])

def family(name):
    for prefix, group in [('cudaStream', 'streams'), ('cudaEvent', 'events'), ('cudaGraph', 'graphs'),
                          ('cudaMemcpy', 'copies'), ('cudaMemset', 'copies'), ('cudaLaunchHost', 'host callbacks'),
                          ('cudaMalloc', 'allocation'), ('cudaFree', 'allocation'), ('cudaHost', 'host memory')]:
        if name.startswith(prefix):
            return group
    return 'device, launch and errors'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = git('ls-tree', '-r', '--name-only', BASELINE, '--', 'src', 'include').decode().splitlines()
    refs = collections.defaultdict(list)
    hashes = {}
    mask = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
    for path in files:
        if Path(path).suffix not in {'.cpp', '.cu', '.hpp', '.h', '.cuh', '.c'}:
            continue
        raw = git('show', f'{BASELINE}:{path}')
        text = raw.decode()
        source = mask.sub(lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]), text)
        lines = [-1] + [m.start() for m in re.finditer('\n', source)]
        matches = list(re.finditer(r'\b(cuda[A-Z][A-Za-z0-9_]*)\s*\(', source))
        if matches:
            hashes[path] = hashlib.sha256(raw).hexdigest()
        for m in matches:
            refs[m[1]].append({'path': path, 'line': bisect.bisect_left(lines, m.start())})
    result = {
        'baseline': BASELINE,
        'method': 'Lexical function-like cuda[A-Z] identifiers after masking comments and quoted literals; includes declarations and macro definitions, not just executed calls.',
        'limitations': ['No preprocessor resolution or reachability analysis', 'Kernel <<<...>>> launches are not counted',
                        'Includes production, harness and compatibility files', 'Not a CUDA/SYCL parity verdict'],
        'files_with_references': len(hashes), 'api_identifiers': len(refs),
        'candidate_occurrences': sum(map(len, refs.values())),
        'source_sha256': hashes,
        'apis': {name: {'family': family(name), 'count': len(places), 'locations': places}
                 for name, places in sorted(refs.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['files_with_references', 'api_identifiers', 'candidate_occurrences']}))

if __name__ == '__main__':
    main()
