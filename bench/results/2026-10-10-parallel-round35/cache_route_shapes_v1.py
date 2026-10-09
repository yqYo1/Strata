"""Frozen spec4/split/static-cache exposure join; counts are not timing.

Source proof R42: unique top10 per token, at most two rows per callback,
one CPU job per refused expert callback pair. No math/policy decisions.
"""
import hashlib

U64 = (1 << 64) - 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def formats_from_metadata(data, expected_sha256):
    require(type(data) is bytes and len(data) <= 131072, 'metadata input budget/type')
    require(hashlib.sha256(data).hexdigest() == expected_sha256, 'metadata hash mismatch')
    text = data.decode('ascii')
    require(text.startswith('# strata native experts v3:') and 'n_expert 512,' in text.splitlines()[0],
            'metadata version/geometry')
    formats = []
    offset = 0
    gu_blocks = {18: 98, 21: 110, 22: 82, 23: 136}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        words = line.split()
        require(len(words) in (8, 9), 'metadata field count')
        require(all(word.isascii() and word.isdecimal() for word in words[:8]), 'metadata integers')
        layer, gu, down, logical, blob, gate, up, d = map(int, words[:8])
        require(layer == len(formats) and layer < 48, 'metadata layer order/count')
        require(gu in gu_blocks and down in (20, 42), 'metadata format outside frozen pack')
        # Expert H=2560, FF=640, distinct from the four-stream residual R=10240.
        expected_blob = 2 * (2560 // 256 * gu_blocks[gu]) * 640
        expected_blob += (640 // (32 if down == 20 else 64) * 18) * 2560
        require(logical == offset and blob == expected_blob and min(gate, up, d) > 0,
                'metadata logical layout/shape')
        offset += 512 * blob
        formats.append((gu, down))
    require(len(formats) == 48, 'metadata layer order/count')
    return formats


def reconcile_shapes(parsed, residents, formats, histogram):
    require(parsed.get('passed') is True and histogram.get('passed') is True,
            'unqualified parser/histogram')
    require(len(formats) == 48, 'format layer count')
    require(parsed['begin']['complete'] == parsed['end']['complete'] == 1 and
            parsed['end']['offloaded'] == 0, 'incomplete/adaptive route scope')
    expected = {}
    pair_shapes = []
    for pair in parsed['pairs']:
        layer, expert = pair['layer'], pair['expert']
        require(type(layer) is int and type(expert) is int and 0 <= layer < 48 and 0 <= expert < 512,
                'pair coordinates')
        n, j = pair['entries'], pair['callback_pairs']
        require(type(n) is int and type(j) is int and 0 < j <= n <= min(2*j, U64),
                'pair entry/job shape outside source-proved NT1/NT2')
        j1, j2 = 2*j - n, n - j
        pair_shapes.append((layer, expert, n, j, j1, j2))
        if (layer, expert) in residents:
            require(pair['hits'] == n and pair['refused'] == pair['offloaded'] == 0,
                    'resident pair classification')
            continue
        require(pair['refused'] == n and pair['hits'] == pair['offloaded'] == 0,
                'nonresident pair classification')
        for phase, typ, rows in (('GU', formats[layer][0], 640), ('Down', formats[layer][1], 2560)):
            for nt, jobs in ((1, j1), (2, j2)):
                if jobs:
                    cell = expected.setdefault((phase, typ, nt), [0, 0, 0])
                    for index, increment in enumerate((jobs, nt*jobs, rows*jobs)):
                        cell[index] += increment
                        require(cell[index] <= U64, 'derived histogram overflow')
    actual = {}
    for cell in histogram['cells']:
        key = (cell['phase'], cell['type'], cell['nt'])
        require(key not in actual, 'duplicate actual histogram cell')
        actual[key] = [cell['experts'], cell['tokens'], cell['output_rows']]
    require(actual == expected, 'per-format/NT histogram differs from pair-derived jobs')
    return {'passed': True, 'pairs': len(pair_shapes),
            'cells': [dict(phase=key[0], type=key[1], nt=key[2], experts=values[0],
                           tokens=values[1], output_rows=values[2])
                      for key, values in sorted(expected.items())],
            'pair_shapes': pair_shapes,
            'semantics': 'fixed successful target evaluated work incl speculative rejects; static NT1/NT2 exposure, not service time/traffic/latency'}
