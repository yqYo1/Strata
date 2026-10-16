#!/usr/bin/env python3
"""Exact finite raw comparison; completion/coverage is independent of equality/adoption."""
import hashlib
import json
import sys
from pathlib import Path

NAMES = ['gdn_input_mixed_h', 'gdn_out_bo', 'attn_write_R', 'ffn_mixed_h', 'ffn_mixed_bf',
         'ffn_mixed_bf_low', 'router_logits', 'router_ids', 'router_weights', 'host_grouping',
         'moe_combine_bo', 'final_R']
SPECS = dict(zip(NAMES, [('fp16',2560), ('fp32',2560), ('fp32',10240), ('fp16',2560),
                       ('bf16',2560), ('bf16',2560), ('fp32',512), ('int32',10),
                       ('fp32',10), ('int32',70), ('fp32',2560), ('fp32',10240)]))


def kv(line):
    words = [w.split('=', 1) for w in line.split() if '=' in w]
    if len({k for k,v in words}) != len(words):
        raise ValueError('duplicate metadata key')
    return dict(words)


def read(path):
    path = Path(path)
    rawpath = Path(str(path)+'.raw')
    if path.stat().st_size > 1048576 or rawpath.stat().st_size > 8388608:
        raise ValueError('budget exceeded')
    text = path.read_bytes()
    raw = rawpath.read_bytes()
    lines = text.decode('ascii').splitlines()
    if len(lines)<3 or not lines[0].startswith('phase_audit_v1 '):
        raise ValueError('missing version/header')
    h, end = kv(lines[0]), kv(lines[-1])
    if h['status']!='incomplete' or end.get('status')!='complete' or end.get('coverage')!='complete':
        raise ValueError('incomplete capture')
    if end.get('memcmp')!='not_compared' or end.get('adoption')!='unqualified':
        raise ValueError('incorrect completion interpretation')
    n, pos, chunk, selected, low = (int(h[k]) for k in ['tokens','pos0','chunk','selected','low'])
    if h['layer']!='26' or not (0<n<=262144 and 0<=pos and pos+n<=262144 and
            0<chunk<=8192 and 0<selected<=6 and low in (0,1)):
        raise ValueError('invalid capture geometry')
    if h.get('byteorder') not in ('little','big'):
        raise ValueError('missing byte order')
    if any(h[k]!=str(v) for k,v in [('raw_limit',8388608), ('text_limit',1048576), ('host_reservation',1048576)]):
        raise ValueError('unexpected budget')
    if h.get('grouping_fields')!='expert,slot,src,cnt,off,order_index,group_local':
        raise ValueError('grouping schema mismatch')
    rows = []
    for s in range(selected):
        item = kv(lines[1+s])
        if item.keys()!={'selection','row'} or int(item['selection'])!=s:
            raise ValueError('missing row selection')
        row = int(item['row'])
        if not 0<=row<n or row in rows:
            raise ValueError('invalid selected row')
        rows.append(row)
    names = [name for name in NAMES if low or name!='ffn_mixed_bf_low']
    records, next_tensor, offset = {}, dict.fromkeys(rows,0), 0
    for line in lines[1+selected:-1]:
        r = kv(line)
        name, row = r['tensor'], int(r['row'])
        if row not in next_tensor or next_tensor[row]>=len(names) or name!=names[next_tensor[row]]:
            raise ValueError('duplicate/missing/out-of-order tensor')
        dt, elems = SPECS[name]
        length = elems*(2 if dt in ('bf16','fp16') else 4)
        start = row//chunk*chunk
        expected = {'record': len(records), 'layer':26, 'absolute':pos+row, 'chunk':row//chunk,
                    'start':start, 'count':min(chunk,n-start), 'elements':elems,
                    'bytes':length, 'offset':offset,
                    'source_offset':(row-start)*(40 if name=='host_grouping' else length)}
        if any(int(r[k])!=v for k,v in expected.items()) or r['dtype']!=dt or r['memcmp']!='NA':
            raise ValueError('record metadata/extent mismatch')
        if r['source']!=('host_selected' if name=='host_grouping' else 'GPU'):
            raise ValueError('record source mismatch')
        if offset+length>len(raw):
            raise ValueError('truncated raw')
        records[(row,name)] = (r, raw[offset:offset+length])
        offset += length
        next_tensor[row] += 1
    if any(v!=len(names) for v in next_tensor.values()) or offset!=len(raw) or \
            int(end['records'])!=len(records) or int(end['raw_bytes'])!=len(raw):
        raise ValueError('incomplete coverage/raw extents')
    return h, records, {'text_bytes':len(text), 'raw_bytes':len(raw),
        'text_sha256':hashlib.sha256(text).hexdigest(), 'raw_sha256':hashlib.sha256(raw).hexdigest()}


def main():
    if len(sys.argv)!=3:
        raise ValueError('usage: compare.py FIXED_PHASE_LOG AUTO_PHASE_LOG')
    left, right = read(sys.argv[1]), read(sys.argv[2])
    if left[0]!=right[0] or left[1].keys()!=right[1].keys():
        raise ValueError('pair geometry/schema/selection mismatch')
    different = []
    for key in sorted(left[1], key=lambda key:(key[0],NAMES.index(key[1]))):
        a,b = left[1][key],right[1][key]
        if a[0]!=b[0]:
            raise ValueError('pair metadata mismatch')
        if a[1]!=b[1]: #Actual exact bytes. Hashes above identify fixtures only.
            first = next(i for i,(x,y) in enumerate(zip(a[1],b[1])) if x!=y)
            different.append({'row':key[0],'tensor':key[1],'first_different_byte':first,'bytes':len(a[1])})
    print(json.dumps({'coverage_complete':True, 'raw_memcmp':'different' if different else 'equal',
        'different':different,'first_selected_difference':different[0] if different else None,
        'fixed':left[2], 'auto':right[2], 'adoption':'unqualified', 'GPU_executed':False,
        'scope':'six-or-fewer selected layer26 rows only; not complete recurrent-input equality'},indent=2))
    return 1 if different else 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (OSError,ValueError,KeyError,IndexError,UnicodeError) as exc:
        print(json.dumps({'coverage_complete':False,'raw_memcmp':'not_compared','error':str(exc),
                          'adoption':'unqualified','GPU_executed':False}), file=sys.stderr)
        sys.exit(2)
