#!/usr/bin/env python3
"""Offline full-buffer comparator. Source proposal; NOT_RUN."""
import array
import hashlib
import json
import math
import pathlib
import sys

a, b = map(pathlib.Path, sys.argv[1:3])
def read(p, suffix):
    return pathlib.Path(str(p) + suffix).read_bytes()
def meta(p):
    return json.loads(read(p, '.json'))
def sha(v):
    return hashlib.sha256(v).hexdigest()
r, c = meta(a), meta(b)
assert r['arm'] == 'ref' and c['arm'] == 'packed'
for key in ['type', 'gu', 'm', 'n', 'k', 'max_w', 'max_x']:
    assert r[key] == c[key], key
for m in [r, c]:
    assert m['queue_retired']
    assert m['guards'] and m['unchanged'] and m['finite']
hashes = {}
for suffix in ['.w.bin', '.x.bin']:
    v, w = read(a, suffix), read(b, suffix)
    assert v == w, suffix
    hashes[suffix] = sha(v)
x, y = array.array('f'), array.array('f')
x.frombytes(read(a, '.y.bin')); y.frombytes(read(b, '.y.bin'))
assert len(x) == len(y) == r['m'] * r['n']
assert all(math.isfinite(v) for v in [*x, *y])
errors = [abs(float(u)-float(v)) for u,v in zip(x,y)]
# Two conventional FP32 summation error bounds, using max operands instead of
# per-output sum(abs(x*w)). FP16 operands multiply exactly into FP32 here.
# A deliberately conservative screening bound, NOT a model-quality threshold.
u = 2**-24
k = r['k']
gamma = k*u/(1-k*u)
bound = 2*gamma*k*r['max_w']*r['max_x'] + 1e-6
report = dict(status='component_screen_pass' if max(errors) <= bound else 'FAIL',
    max_abs=max(errors), rms=math.sqrt(sum(v*v for v in errors)/len(errors)),
    max_rel_floor_1e_6=max(e/max(abs(float(v)),1e-6) for e,v in zip(errors,x)),
    reference_range=[min(x),max(x)], candidate_range=[min(y),max(y)],
    conservative_abs_bound=bound, input_sha256=hashes,
    reference_output_sha256=sha(read(a,'.y.bin')),candidate_output_sha256=sha(read(b,'.y.bin')),
    contract='FP16 X/W, FP32 accumulate; changed sum order. No exact IDs or model adoption.')
print(json.dumps(report, indent=2))
sys.exit(0 if report['status'] == 'component_screen_pass' else 1)
