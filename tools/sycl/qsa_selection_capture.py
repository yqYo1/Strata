"""Read the native QSA dump; validate every selected ID and optional replay input.

Main-authored under the repository LICENSE. Logical unions are reuse opportunity,
not cache hits, physical traffic, bandwidth or additive avoidable time.
"""
from array import array
import hashlib
from pathlib import Path
import struct
import sys


def require(value, message):
    if not value:
        raise ValueError(message)


def analyze(path, prompt_tokens, warmup_tokens, replay_path=None):
    require(sys.byteorder == 'little' and array('i').itemsize == 4, 'native dump host ABI')
    require(32768 <= prompt_tokens <= 131072 and 1 < warmup_tokens < 32768, 'request extent')
    cells = prompt_tokens - 1
    pos = ((cells - 1) // 8192) * 8192
    tail = cells - pos
    records, long_rows, warm_rows = [], [], []
    replay = None
    with Path(path).open('rb') as stream:
        while header := stream.read(16):
            require(len(header) == 16 and len(records) < 24, 'truncated or extra dump header')
            layer, first, count, cap = struct.unpack('<4i', header)
            require(0 <= layer < 12 and 0 <= first < 131072 and
                    1 <= count <= 8192 and cap == 2051, 'dump geometry')
            body = stream.read(count * cap * 4)
            require(len(body) == count * cap * 4, 'truncated dump body')
            ids = array('i'); ids.frombytes(body)
            selected = 0
            for row in range(count):
                width = min(first + row + 1, cap)
                values = ids[row * cap:row * cap + width]
                require(values[0] >= 0 and values[-1] <= first + row and
                        all(a < b for a, b in zip(values, values[1:])),
                        'selected IDs must be unique, ascending and causal')
                selected += width
            descriptor = dict(layer=layer, first=first, queries=count, cap=cap,
                selected_cells=selected, record_sha256=hashlib.sha256(header + body).hexdigest())
            records.append(descriptor)
            if (first, count) == (pos, tail):
                starts = sorted({i * (count - 16) // 31 for i in range(32)}) if count >= 16 else []
                separate = union = 0
                for start in starts:
                    unique = set()
                    for row in range(start, start + 16):
                        width = min(first + row + 1, cap)
                        unique.update(ids[row * cap:row * cap + width])
                        separate += width
                    union += len(unique)
                descriptor['logical_16query_union'] = dict(sampled_tiles=len(starts),
                    separate_cells=separate, union_cells=union,
                    union_fraction=union / separate if separate else None)
                long_rows.append(descriptor)
                if layer == 0:
                    replay = header + body
            else:
                require((first, count) == (0, warmup_tokens - 1), 'unbound dump record')
                warm_rows.append(descriptor)
    require([r['layer'] for r in long_rows] == list(range(12)), 'all twelve long-request layers required')
    require(not warm_rows or [r['layer'] for r in warm_rows] == list(range(12)), 'warmup layer coverage')
    require(records == warm_rows + long_rows and replay is not None, 'request ordering')
    if replay_path is not None:
        with Path(replay_path).open('xb') as stream:
            stream.write(replay)
    return dict(prompt_tokens=prompt_tokens, processed_prefill_cells=cells,
        captured_first=pos, captured_queries=tail, cap=2051, long_layers=long_rows,
        warmup_records=len(warm_rows), all_selected_IDs_validated=True,
        replay=dict(bytes=len(replay), sha256=hashlib.sha256(replay).hexdigest(),
            layer=0, retained=replay_path is not None,
            consumer='Explicit caller replay input.' if replay_path is not None else None),
        limits='Final chunk only, one prompt/process. Deterministic sampled logical16query unions '
               'are reuse opportunities, not physical traffic,cache hits or model speed. Query/KV values not captured.')
