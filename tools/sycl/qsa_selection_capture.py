"""Bounded reader for the selected engine's existing STRATA_QSA_DUMP format.

Repository LICENSE. IDs describe logical reads, not physical cache traffic.
Only the per-position selection width is initialized; ignore the cap suffix.
"""
from array import array
import bisect
import hashlib
import math
import os
from pathlib import Path
import stat
import statistics
import struct
import sys

HEADER = struct.Struct('<4i')
MAX_ROWS, MAX_CAP, MAX_POSITION = 8192, 2051, 262144


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def identity(info):
    return tuple(getattr(info, field) for field in
                 ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def headers(rows, position=0):
    return [(layer, position, rows, MAX_CAP) for layer in range(12)]


def describe(values):
    ordered = sorted(values)
    require(bool(ordered), 'empty statistic')
    return dict(count=len(ordered), minimum=ordered[0], median=statistics.median(ordered),
                p95=ordered[math.ceil(len(ordered)*.95)-1], maximum=ordered[-1])


class Locality:
    """Nominal packed BF16 K+V footprint; no cache-line/transaction model."""
    def __init__(self):
        self.cells, self.pages, self.window = set(), set(), set()
        self.previous = None
        self.shared, self.window_cells, self.window_rows = [], [], []
        self.rows, self.entries = 0, 0
        self.ages = [0, 32, 256, 2048, 8192, 32768, 131072, 262144]
        self.age_counts = [0] * (len(self.ages)-1)

    def add(self, ids, position):
        cells = set(ids)
        self.cells.update(cells)
        self.pages.update(cell//4 for cell in cells)
        if self.previous is not None:
            self.shared.append(len(cells & self.previous) / len(cells))
        self.previous = cells
        self.window.update(cells)
        self.rows += 1
        self.entries += len(ids)
        for cell in ids:
            age = position - cell
            self.age_counts[bisect.bisect_right(self.ages, age)-1] += 1
        if self.rows % 16 == 0:
            self.flush(16)

    def flush(self, rows):
        self.window_cells.append(len(self.window))
        self.window_rows.append(rows)
        self.window.clear()

    def result(self):
        if self.rows % 16:
            self.flush(self.rows % 16)
        require(sum(self.age_counts) == self.entries, 'age counts do not conserve selections')
        return dict(query_rows=self.rows, selected_entries=self.entries,
            unique_logical_cells=len(self.cells), unique_logical_pages4=len(self.pages),
            adjacent_query_shared_fraction=describe(self.shared),
            windows16_unique_cells=describe(self.window_cells),
            windows16_nominal_packed_KV_bytes=describe([v*2048 for v in self.window_cells]),
            windows16_reuse_factor=describe([r*MAX_CAP/c for r,c in zip(self.window_rows,self.window_cells)]),
            age_bin_edges=self.ages, age_bin_counts=self.age_counts,
            interpretation='Logical selected cells; 2048 packed BF16 K+V bytes/cell '
                '(2KV heads,256dimensions,2bytes,2tensors). Windows are consecutive '
                '16queries including a final partial window. Reuse is selected entries '
                '/unique cells,not measured cache hits. No physical traffic or saturation claim.')


def inspect(path, expected, selected=None, statistics_enabled=False):
    """Reject malformed extents before allocation; never interpret unused IDs.

    Expected headers bind record count/order/geometry to the source and workload.
    Reads at most one 8192*2051*4 record at a time. Requires an immutable, closed
    writer (or a benchmark request gate) and detects file mutation during scan.
    """
    path = Path(path)
    require(sys.byteorder == 'little' and array('i').itemsize == 4, 'host integer ABI differs')
    require(0 < len(expected) <= 24 and len(set(map(tuple, expected))) == len(expected),
            'invalid expected record domain')
    payload_total = 0
    for layer, pos, rows, cap in expected:
        require(all(type(v) is int for v in (layer,pos,rows,cap)) and 0 <= layer < 12 and
            0 <= pos < MAX_POSITION and 0 < rows <= MAX_ROWS and pos+rows <= MAX_POSITION and
            cap == MAX_CAP, 'invalid expected header')
        payload_total += HEADER.size + rows*cap*4
    require(payload_total <= 1 << 30, 'capture budget exceeded')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_size == payload_total, 'capture extent differs')
    require(selected is None or tuple(selected) in set(map(tuple, expected)), 'selected record absent')
    require(not statistics_enabled or selected is not None, 'statistics need one selected record')
    records, offset, selected_record = [], 0, None
    local = Locality() if statistics_enabled else None
    whole = hashlib.sha256()
    with path.open('rb') as stream:
        require(identity(os.fstat(stream.fileno())) == identity(before),
                'capture replaced before open')
        for wanted in expected:
            raw = stream.read(HEADER.size)
            require(len(raw) == HEADER.size and HEADER.unpack(raw) == tuple(wanted), 'record header differs')
            whole.update(raw)
            layer, pos, rows, cap = wanted
            count = rows*cap
            raw_ids = stream.read(count*4)
            require(len(raw_ids) == count*4, 'truncated selection payload')
            whole.update(raw_ids)
            ids = array('i'); ids.frombytes(raw_ids)
            payload_sha = hashlib.sha256(raw_ids).hexdigest()
            record_sha = hashlib.sha256(raw + raw_ids).hexdigest()
            del raw_ids
            entries = 0
            for row in range(rows):
                width = min(pos+row+1, cap)
                values = ids[row*cap:row*cap+width]
                previous = -1
                for cell in values:
                    require(previous < cell <= pos+row, 'invalid/duplicate/unordered logical cell')
                    previous = cell
                # Before selection becomes sparse this must be the identity.
                if width == pos+row+1:
                    require(values[0] == 0 and values[-1] == pos+row, 'dense selection is not identity')
                entries += width
                if local is not None and tuple(wanted) == tuple(selected):
                    local.add(values, pos+row)
            row = dict(header=list(wanted), offset=offset, bytes=HEADER.size+count*4,
                payload_sha256=payload_sha, sha256=record_sha, valid_entries=entries,
                unused_cap_entries=count-entries)
            records.append(row)
            if selected is not None and tuple(wanted) == tuple(selected):
                require(entries == count, 'fixture may not retain uninitialized cap suffix')
                selected_record = row
            offset += row['bytes']
        require(not stream.read(1), 'trailing capture bytes')
        require(identity(os.fstat(stream.fileno())) == identity(before), 'capture changed during scan')
    require(identity(path.stat()) == identity(before), 'capture path changed during scan')
    result = dict(bytes=payload_total, sha256=whole.hexdigest(), records=records,
                  selected=selected_record, file_identity=list(identity(before)))
    if local is not None:
        result['locality'] = local.result()
    return result


def extract(path, result, destination):
    """Retain one fully validated initialized record; x-create, never overwrite."""
    path, destination = Path(path), Path(destination)
    selected = result['selected']
    require(selected is not None and identity(path.stat()) == tuple(result['file_identity']),
            'validated selection changed')
    created = False
    try:
        with path.open('rb') as source, destination.open('xb') as target:
            created = True
            require(identity(os.fstat(source.fileno())) == tuple(result['file_identity']),
                    'capture replaced before extraction')
            source.seek(selected['offset'])
            remaining = selected['bytes']; digest = hashlib.sha256()
            while remaining:
                data = source.read(min(remaining, 1 << 20))
                require(bool(data), 'capture truncated during extraction')
                target.write(data); digest.update(data); remaining -= len(data)
            require(digest.hexdigest() == selected['sha256'] and
                identity(path.stat()) == tuple(result['file_identity']), 'fixture extraction mismatch')
    except FileExistsError:
        raise
    except BaseException:
        # This function owns only a newly-created destination, never the source.
        if created and destination.exists(): destination.unlink()
        raise
    return dict(path=str(destination), bytes=selected['bytes'], sha256=selected['sha256'],
                header=selected['header'], consumer='next selected-QSA working-set attribution fixture',
                review='Retire when that comparison closes or is superseded; maximum64MiB.')
