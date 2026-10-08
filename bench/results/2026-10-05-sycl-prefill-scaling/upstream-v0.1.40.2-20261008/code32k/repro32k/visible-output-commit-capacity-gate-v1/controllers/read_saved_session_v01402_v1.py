"""Bounded, read-only evidence for the existing v1 session codec.

Hash every state/KV byte, including the last page. Only checkpoint LRU stamps
are omitted from the semantic comparison. Engine RESTORE validates checksums;
this reader does not substitute for that validation.
"""
from pathlib import Path
import hashlib
import json
import struct


def read_session(path):
    path = Path(path)
    evidence = {'file': str(path), 'bytes': path.stat().st_size,
                'checksum_validation': 'performed by engine RESTORE, not this reader'}
    with path.open('rb') as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:8] != b'STRSESS\x01':
            raise ValueError('invalid session header')
        version, size, model, config, payload, r0, r1, header_hash = struct.unpack('<II6Q', header[8:])
        if version != 1 or size != 64 or r0 or r1 or evidence['bytes'] != 64 + payload + 16:
            raise ValueError('invalid version, reserved fields or extent')
        left = payload

        def raw(n):
            nonlocal left
            if not 0 <= n <= left:
                raise ValueError('payload extent exceeded')
            data = stream.read(n)
            if len(data) != n:
                raise ValueError('truncated payload')
            left -= n
            return data

        def u64():
            return struct.unpack('<Q', raw(8))[0]

        def part(element_bytes=1):
            count = u64()
            n = count * element_bytes
            if n > left:
                raise ValueError('part count exceeds payload')
            offset = stream.tell()
            digest = hashlib.sha256()
            remaining = n
            while remaining:
                data = raw(min(1048576, remaining))
                digest.update(data)
                remaining -= len(data)
            return {'count': count, 'bytes': n, 'offset': offset, 'sha256': digest.hexdigest()}

        def checkpoint():
            ids = part(4)
            images = part(16)
            state = {name: part() for name in ['gdn', 'ple', 'tails', 'dead', 'block_pos']}
            used = u64()
            return {'ids': ids, 'images': images, 'state': state, 'used': used}

        geometry = list(struct.unpack('<18q', raw(144)))
        layer_lo, layer_hi, cvec = struct.unpack('<qqQ', raw(24))
        if layer_lo < 0 or layer_hi < layer_lo or cvec not in [0, 1]:
            raise ValueError('invalid layer range or steering flag')
        live = checkpoint()
        n = u64()
        if n > 4096 or n > left // 64:
            raise ValueError('checkpoint count exceeds bounds')
        checkpoints = [checkpoint() for _ in range(n)]
        n = u64()
        if n > 1024 or n > left // 96:
            raise ValueError('KV layer count exceeds bounds')
        kv = []
        for _ in range(n):
            shape = list(struct.unpack('<7q', raw(56)))
            if any(v < 0 for v in shape) or any(shape[i] <= 0 for i in [2, 3, 4, 6]):
                raise ValueError('invalid KV geometry')
            parts = {name: part() for name in ['k', 'v', 'k_scale', 'v_scale', 'pooled']}
            kv.append({'shape': shape, 'parts': parts})
        if left:
            raise ValueError('unparsed payload bytes')
        trailer = stream.read(16)
        if len(trailer) != 16 or trailer[:8] != b'STRSEND\x01' or stream.read(1):
            raise ValueError('invalid trailer')
        semantic = {'model': model, 'config': config, 'geometry': geometry,
                    'layer_range': [layer_lo, layer_hi], 'cvec': cvec,
                    'live': live, 'checkpoints': checkpoints, 'kv': kv}
        # File offsets and LRU are representation/retention metadata, not tensor state.
        def canonical(value):
            if isinstance(value, dict):
                return {k: canonical(v) for k, v in value.items() if k not in ['offset', 'used']}
            if isinstance(value, list):
                return [canonical(v) for v in value]
            return value
        normalized = canonical(semantic)
        evidence.update(semantic=semantic, semantic_sha256=hashlib.sha256(
            json.dumps(normalized, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            payload_bytes=payload, header_checksum=header_hash,
            payload_checksum=struct.unpack('<Q', trailer[8:])[0],
            excluded_fields=['checkpoint.used (LRU only)', 'file offsets', 'derived checksums'])
    with path.open('rb') as stream:
        evidence['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return evidence


def full_kv_gate(evidence):
    semantic = evidence['semantic']
    if semantic['live']['ids']['count'] != 262143 or len(semantic['kv']) != 13:
        raise ValueError('expected 262143 consumed tokens and all12 main plus MTP KV layers')
    for i, layer in enumerate(semantic['kv']):
        fmt, cells, heads, dim, page, pooled, idx_dim = layer['shape']
        if cells != 262144 or heads != 2 or dim != 256 or page != 4 or idx_dim != 128:
            raise ValueError('unexpected full-capacity KV geometry')
        if fmt not in [1, 17]:
            raise ValueError('expected int8 KV')
        expected_pooled = 65536 if i < 12 else 0
        if pooled != expected_pooled:
            raise ValueError('unexpected pooled or spare row extent')
        expected = [cells * heads * dim, cells * heads * dim,
                    cells * heads * (dim // 64) * 2, cells * heads * (dim // 64) * 2,
                    pooled * idx_dim * 4]
        if [layer['parts'][name]['bytes'] for name in ['k', 'v', 'k_scale', 'v_scale', 'pooled']] != expected:
            raise ValueError('unexpected full-capacity KV part size')
    return {'passed': True, 'layers': 13, 'physical_cells_per_layer': 262144,
            'last_physical_cell': 262143, 'ignored_tensor_bytes': 0}
