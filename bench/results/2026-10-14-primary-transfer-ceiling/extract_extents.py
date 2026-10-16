#!/usr/bin/env python3
"""Read only native_experts.txt metadata; never open GGUF/expert payloads."""
import argparse, hashlib, json, pathlib, re, statistics
p = argparse.ArgumentParser()
p.add_argument('metadata', type=pathlib.Path)
p.add_argument('--extents-output', required=True, type=pathlib.Path)
a = p.parse_args()
# Bounded metadata read, rejecting symlinks is unnecessary: caller pins path/hash.
with a.metadata.open('rb') as f:
    raw = f.read(1024 * 1024 + 1)
if len(raw) > 1024 * 1024:
    raise SystemExit('metadata exceeds 1 MiB bound')
text = raw.decode('utf-8')
header = text.splitlines()[0]
v = re.search(r'native experts v([234])\b', header)
n = re.search(r'n_expert (\d+), total (\d+)', header)
if not v or not n:
    raise SystemExit('unsupported/missing native metadata header')
experts, total = map(int, n.groups())
if not 1 <= experts <= 4096:
    raise SystemExit('invalid expert count')
rows = []
offset = 0
for line in text.splitlines():
    if not line.strip() or line.lstrip().startswith('#'):
        continue
    t = line.split()
    if len(t) not in (8, 9):
        raise SystemExit('unexpected metadata column count')
    layer, gu, down, off, size, gate, up, d = map(int, t[:8])
    if layer != len(rows) or off != offset or size <= 0 or min(gu, down, gate, up, d) < 0:
        raise SystemExit('invalid layer/offset/extent metadata')
    offset += experts * size
    rows.append(dict(layer=layer, blob_bytes=size, gu_type=gu, d_type=down))
if not rows or len(rows) > 4096 or offset != total:
    raise SystemExit('invalid layer count or total')
sizes = sorted(r['blob_bytes'] for r in rows)
# Lower order statistic is an actual layer extent, never a fabricated averaged size.
representatives = sorted({sizes[0], sizes[(len(sizes)-1)//2], sizes[-1]})
with a.extents_output.open('x', encoding='ascii') as f:
    f.write(f'{experts} {len(rows)}\n')
    for r in rows:
        f.write(f"{r['layer']} {r['blob_bytes']}\n")
print(json.dumps(dict(schema='strata-transfer-extents-v1', metadata=str(a.metadata),
    metadata_sha256=hashlib.sha256(raw).hexdigest(), version=int(v[1]),
    n_expert=experts, n_layers=len(rows), total=total, layers=rows,
    min=sizes[0], median_actual=sizes[(len(sizes)-1)//2], max=sizes[-1],
    mathematical_median=statistics.median(sizes), representatives=representatives,
    payload_files_opened=0, payload_bytes_read=0), sort_keys=True))
