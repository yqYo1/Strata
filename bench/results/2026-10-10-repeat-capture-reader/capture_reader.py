"""Bounded native RepeatCapture reader. SOURCE ONLY / UNTESTED.

Import performs no I/O. Parsing never treats numerical findings as framing
errors. Producer freshness must be established by its O_EXCL creation and root's
ledger/provenance, not inferred from this read-only consumer.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys

MAGIC = 0x5354524152505431
HEADER_BYTES = 128
MAX_BYTES = 128 * 1024**2
CHUNK_BYTES = 64 * 1024
MAX_LEDGER_BYTES = 64 * 1024
MAX_RECORDS = 690  # two actual full GEN sets; bound manifest memory/JSON too
REQUESTS = ('control32k-before', 'resume32k-reference', 'full256k-first',
            'control32k-between-full-reads', 'full256k-repeat')
FULL_REQUESTS = ('full256k-first', 'full256k-repeat')
R_PHASES = ('R_input', 'R_post_attention_gdn', 'R_post_moe')
BATCH_PHASES = {'K_quant_input': 512, 'V_quant_input': 512, 'indexer_raw': 128,
                'query': 6144, 'q_indexer': 512, 'attention_output': 6144}
PHASES = set(R_PHASES) | set(BATCH_PHASES) | {'scores', 'steps', 'selected_ids'}
GENERIC_PHASES = PHASES | {'test', 'float_rows'}
FIELDS = ('magic', 'version', 'layer', 'p0', 'T', 'first', 'rows', 'elements',
          'type', 'width', 'ordinal', 'header_bytes')
SEMANTIC_FIELDS = ('layer', 'phase', 'p0', 'T', 'first', 'rows', 'elements', 'type', 'width')


class CaptureError(ValueError):
    """A framing/security error with previously collected evidence preserved."""
    def __init__(self, message, manifest=None):
        super().__init__(message)
        self.manifest = manifest


def endian(byteorder):
    if byteorder not in ('little', 'big'):
        raise CaptureError('byteorder must be little or big')
    return '<' if byteorder == 'little' else '>'


def file_identity(st):
    return {k: getattr(st, 'st_' + k) for k in
            ('dev', 'ino', 'uid', 'nlink', 'size', 'mtime_ns', 'ctime_ns')}


def private_path(path):
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise CaptureError('file path must be text')
    if not raw:
        raise CaptureError('empty file path')
    absolute = os.path.abspath(raw)
    if not Path(absolute).parts[1:]:
        raise CaptureError('root is not a file path')
    return absolute


def open_private(path, expected_identity=None, max_bytes=MAX_BYTES):
    """No symlink in any path component; final parent private and file exact 0600."""
    absolute = private_path(path)
    parts = Path(absolute).parts[1:]
    parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    fd = None
    try:
        for part in parts[:-1]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                          dir_fd=parent)
            os.close(parent)
            parent = new
        pst = os.fstat(parent)
        if pst.st_uid != os.geteuid() or stat.S_IMODE(pst.st_mode) & 0o077:
            raise CaptureError('capture parent must be owned and private')
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=parent)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or \
                stat.S_IMODE(st.st_mode) != 0o600 or st.st_nlink != 1:
            raise CaptureError('capture must be owned 0600 regular file with one link')
        if st.st_size > max_bytes:
            raise CaptureError('private file exceeds byte budget')
        identity = file_identity(st)
        if expected_identity is not None and any(identity.get(k) != v
                                                  for k, v in expected_identity.items()):
            raise CaptureError('capture identity differs from pinned file')
        return fd, identity
    except BaseException:
        if fd is not None:
            os.close(fd)
        raise
    finally:
        os.close(parent)


def read_ledger(path):
    """Private descriptor-pinned JSON; total read budget is 64 KiB plus one."""
    fd = None
    try:
        fd, pinned = open_private(path, max_bytes=MAX_LEDGER_BYTES)
        data = bytearray()
        while len(data) <= MAX_LEDGER_BYTES:
            chunk = os.read(fd, min(CHUNK_BYTES, MAX_LEDGER_BYTES + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > MAX_LEDGER_BYTES:
            raise CaptureError('ledger exceeds 64 KiB')
        if file_identity(os.fstat(fd)) != pinned or len(data) != pinned['size']:
            raise CaptureError('ledger changed during reading')
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise CaptureError('duplicate JSON ledger key')
                result[key] = value
            return result
        ledger = json.loads(data.decode('utf-8'), object_pairs_hook=unique_object)
        # A ledger is the five-GEN schema, rather than arbitrary JSON. Check its
        # extent against the same capture budget before capture-file validation.
        if not isinstance(ledger, list) or len(ledger) != 5 or not isinstance(ledger[-1], dict):
            raise CaptureError('ledger JSON must contain five GEN intervals')
        validate_ledger(ledger, ledger[-1].get('end'))
        return ledger
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise CaptureError(str(error)) from error
    finally:
        if fd is not None:
            os.close(fd)


def expected_full_records():
    """Actual prefill hook order, not the fake producer's different phase order."""
    result = []
    def add(layer, phase, first=98304, rows=32, width=10240, kind=1):
        result.append({'layer': layer, 'phase': phase, 'p0': 98304, 'T': 8192,
                       'first': first, 'rows': rows, 'elements': rows * width,
                       'type': kind, 'width': width})
    for layer in range(13):
        add(layer, 'R_input')
        if layer in (3, 7, 11):
            for phase in ('K_quant_input', 'V_quant_input', 'indexer_raw', 'query', 'q_indexer'):
                add(layer, phase, width=BATCH_PHASES[phase])
            for pos in range(98304, 98336):
                add(layer, 'scores', pos, 1, (pos + 1) // 4 + 1)
            for pos in range(98304, 98336):
                add(layer, 'steps', pos, 1, 4, 2)
                add(layer, 'selected_ids', pos, 1, 2051, 2)
            add(layer, 'attention_output', width=6144)
        add(layer, 'R_post_attention_gdn')
        add(layer, 'R_post_moe')
    return result


def semantic(record):
    return tuple(record[k] for k in SEMANTIC_FIELDS)


def validate_full_segment(records):
    expected = expected_full_records()
    if len(records) != len(expected):
        raise CaptureError('full GEN must contain exactly 345 records')
    seen = set()
    for index, (record, wanted) in enumerate(zip(records, expected)):
        key = semantic(record)
        if key in seen:
            raise CaptureError('duplicate full-GEN semantic record at ' + str(index))
        seen.add(key)
        if key != semantic(wanted):
            raise CaptureError('full-GEN coverage/order mismatch at ' + str(index))


def validate_ledger(ledger, size):
    if type(size) is not int or not 0 <= size <= MAX_BYTES:
        raise CaptureError('GEN ledger file size outside capture budget')
    if not isinstance(ledger, list) or len(ledger) != 5:
        raise CaptureError('owned_full requires exactly five GEN ledger intervals')
    normalized = []
    previous = 0
    for expected, row in zip(REQUESTS, ledger):
        if not isinstance(row, dict) or row.get('name') != expected:
            raise CaptureError('GEN ledger request identity/order mismatch')
        begin, end = row.get('begin'), row.get('end')
        if type(begin) is not int or type(end) is not int or \
                begin != previous or not 0 <= begin <= end <= size <= MAX_BYTES:
            raise CaptureError('GEN ledger gap/overlap/invalid offset')
        if (expected in FULL_REQUESTS) != (end > begin):
            raise CaptureError('only the two full GEN intervals may be nonempty')
        normalized.append({'name': expected, 'begin': begin, 'end': end})
        previous = end
    if previous != size:
        raise CaptureError('GEN ledger does not cover file exactly')
    return normalized


def decode_header(data, byteorder, ordinal):
    values = struct.unpack(endian(byteorder) + '12Q', data[:96])
    row = dict(zip(FIELDS, values))
    if row['magic'] != MAGIC or row['version'] != 1 or row['header_bytes'] != HEADER_BYTES:
        raise CaptureError('wrong magic/version/header size')
    if row['ordinal'] != ordinal:
        raise CaptureError('global ordinal must start at zero and increase by one')
    name = data[96:]
    zero = name.find(b'\0')
    if zero <= 0 or any(name[zero:]):
        raise CaptureError('phase must have nonempty ASCII name and zero padding')
    try:
        row['phase'] = name[:zero].decode('ascii')
    except UnicodeDecodeError as error:
        raise CaptureError('phase is not ASCII') from error
    if row['phase'] not in GENERIC_PHASES or row['layer'] > 12 or row['type'] not in (1, 2):
        raise CaptureError('unknown phase, layer or payload type')
    if row['T'] == 0 or row['rows'] == 0 or row['width'] == 0 or \
            max(row['p0'], row['T'], row['first'], row['rows'], row['width']) > (1 << 63) - 1:
        raise CaptureError('zero shape or signed producer field overflow')
    if row['p0'] + row['T'] > (1 << 63) - 1 or \
            not row['p0'] <= row['first'] < row['first'] + row['rows'] <= row['p0'] + row['T']:
        raise CaptureError('row extent outside chunk or extent overflow')
    # Divide before multiplying, even though Python integers cannot overflow.
    if row['width'] > MAX_BYTES // 4 // row['rows'] or \
            row['elements'] != row['rows'] * row['width']:
        raise CaptureError('forged/overflowing element count or shape')
    if row['phase'] in ('steps', 'selected_ids') and row['type'] != 2:
        raise CaptureError('integer phase has float payload type')
    if row['phase'] not in ('steps', 'selected_ids') and row['type'] != 1:
        raise CaptureError('float phase has integer payload type')
    return row


def numeric_summary(fd, row, prefix, stream_hash):
    remaining = row['elements'] * 4
    payload_hash = hashlib.sha256()
    result = {'finite_count': 0, 'nan_count': 0, 'positive_inf_count': 0,
              'negative_inf_count': 0, 'invalid_integer_count': 0,
              'minimum': None, 'maximum': None, 'first_finding': None}
    word_index = 0
    step_values = []
    fmt = prefix + ('f' if row['type'] == 1 else 'i')
    while remaining:
        chunk = os.read(fd, min(remaining, CHUNK_BYTES))
        if not chunk or len(chunk) % 4:
            raise CaptureError('truncated or unaligned payload during read')
        remaining -= len(chunk)
        payload_hash.update(chunk)
        stream_hash.update(chunk)
        for chunk_word, (value,) in enumerate(struct.iter_unpack(fmt, chunk)):
            finding = None
            if row['type'] == 1:
                if math.isnan(value):
                    result['nan_count'] += 1
                    finding = 'nan'
                elif math.isinf(value):
                    result['positive_inf_count' if value > 0 else 'negative_inf_count'] += 1
                    finding = 'positive_inf' if value > 0 else 'negative_inf'
                else:
                    result['finite_count'] += 1
            else:
                result['finite_count'] += 1
                if row['phase'] == 'steps' and row['elements'] == 4:
                    step_values.append(value)
                    pos = row['first']
                    wanted = (pos, pos + 1, (pos + 1) // 4, min(pos + 1, 2051))[word_index]
                    if value != wanted:
                        finding = 'invalid_step'
                elif row['phase'] == 'selected_ids':
                    pos = row['first'] + word_index // row['width']
                    if not 0 <= value <= pos:
                        finding = 'selected_id_outside_cached_cells'
                if finding:
                    result['invalid_integer_count'] += 1
            if row['type'] == 2 or math.isfinite(value):
                result['minimum'] = value if result['minimum'] is None else min(result['minimum'], value)
                result['maximum'] = value if result['maximum'] is None else max(result['maximum'], value)
            if finding and result['first_finding'] is None:
                local = chunk_word * 4
                raw = chunk[local:local + 4]
                result['first_finding'] = {'kind': finding, 'word': word_index,
                    'physical_row': row['first'] + word_index // row['width'],
                    'column': word_index % row['width'], 'raw_bytes': raw.hex(),
                    'raw_bits': int.from_bytes(raw, 'little' if prefix == '<' else 'big'),
                    'value': value if row['type'] == 2 or math.isfinite(value) else finding}
            word_index += 1
    if step_values:
        result['step_values'] = step_values
    result['has_findings'] = bool(result['nan_count'] or result['positive_inf_count'] or
                                  result['negative_inf_count'] or result['invalid_integer_count'])
    return payload_hash.hexdigest(), result


def read_capture(path, ledger=None, mode='framing', byteorder=sys.byteorder,
                 expected_identity=None):
    """Return compact manifest; CaptureError.manifest preserves completed records.

    owned_full requires root's five before/after GEN intervals and exact actual
    model coverage/order. framing supports small real-producer test/float_rows
    fixtures without claiming model coverage. Native byte order must come from
    producer architecture provenance when reading on another architecture.
    """
    prefix = endian(byteorder)
    manifest = {'schema': 'strata-repeat-capture-reader-v1', 'mode': mode,
                'byteorder': byteorder, 'path': None,
                'structural_complete': False, 'records': [], 'numeric_findings': [],
                'original_C_rejection_cleared': False, 'adopted': False,
                'performance_eligible': False, 'full_lifecycle_passed': False}
    fd = None
    try:
        manifest['path'] = private_path(path)
        if mode not in ('framing', 'owned_full'):
            raise CaptureError('unknown reader mode')
        fd, pinned = open_private(path, expected_identity)
        manifest['file_identity'] = pinned
        size = pinned['size']
        ranges = validate_ledger(ledger, size) if mode == 'owned_full' else []
        manifest['request_ledger'] = ranges
        offset, ordinal = 0, 0
        stream_hash = hashlib.sha256()
        seen = set()
        while offset < size:
            if ordinal >= MAX_RECORDS or size - offset < HEADER_BYTES:
                raise CaptureError('record limit or truncated/trailing header')
            header = os.read(fd, HEADER_BYTES)
            if len(header) != HEADER_BYTES:
                raise CaptureError('truncated header during read')
            row = decode_header(header, byteorder, ordinal)
            length = row['elements'] * 4
            end = offset + HEADER_BYTES + length
            if end > size or end > MAX_BYTES:
                raise CaptureError('truncated payload or payload exceeds file bounds')
            row.update({'offset': offset, 'payload_offset': offset + HEADER_BYTES,
                        'end_offset': end, 'payload_bytes': length, 'request_id': None})
            if ranges:
                owners = [r for r in ranges if r['begin'] <= offset < end <= r['end']]
                if len(owners) != 1 or owners[0]['name'] not in FULL_REQUESTS:
                    raise CaptureError('record crosses or escapes full-GEN ledger interval')
                row['request_id'] = owners[0]['name']
                key = (row['request_id'], semantic(row))
                if key in seen:
                    raise CaptureError('duplicate request semantic record')
                seen.add(key)
            stream_hash.update(header)
            digest, summary = numeric_summary(fd, row, prefix, stream_hash)
            row['payload_sha256'], row['numeric_summary'] = digest, summary
            manifest['records'].append(row)
            if summary['has_findings']:
                manifest['numeric_findings'].append({'ordinal': ordinal, 'offset': offset,
                    'request_id': row['request_id'], 'layer': row['layer'], 'phase': row['phase'],
                    'summary': summary})
            offset, ordinal = end, ordinal + 1
        if ranges:
            for name in FULL_REQUESTS:
                validate_full_segment([r for r in manifest['records'] if r['request_id'] == name])
        if file_identity(os.fstat(fd)) != pinned:
            raise CaptureError('capture changed during parsing; require orderly closed producer')
        manifest.update({'structural_complete': True, 'record_count': ordinal,
                         'final_offset': offset, 'file_sha256': stream_hash.hexdigest(),
                         'numerical_findings_present': bool(manifest['numeric_findings'])})
        return manifest
    except (CaptureError, OSError) as error:
        manifest['structural_error'] = str(error)
        manifest['record_count'] = len(manifest['records'])
        manifest['numerical_findings_present'] = bool(manifest['numeric_findings'])
        raise CaptureError(str(error), manifest) from error
    finally:
        if fd is not None:
            os.close(fd)


def word_summary(raw, kind, byteorder):
    value = struct.unpack(endian(byteorder) + ('f' if kind == 1 else 'i'), raw)[0]
    return {'raw_bytes': raw.hex(), 'raw_bits': int.from_bytes(raw, byteorder),
            'value': value if kind == 2 or math.isfinite(value) else
                     ('nan' if math.isnan(value) else ('positive_inf' if value > 0 else 'negative_inf')),
            'finite': kind == 2 or math.isfinite(value)}


def verify_manifest_file(fd, pinned, manifest):
    """Rehash the entire pinned file, including headers, before hash shortcuts.

    Closed producer provenance is still required. Metadata alone cannot prove
    content equality when a same-size write has indistinguishable timestamps.
    """
    size = pinned['size']
    if type(size) is not int or not 0 <= size <= MAX_BYTES:
        raise CaptureError('comparison file size outside capture budget')
    os.lseek(fd, 0, os.SEEK_SET)
    remaining = size
    digest = hashlib.sha256()
    while remaining:
        chunk = os.read(fd, min(remaining, CHUNK_BYTES))
        if not chunk:
            raise CaptureError('comparison file shortened during verification')
        digest.update(chunk)
        remaining -= len(chunk)
    if file_identity(os.fstat(fd)) != pinned:
        raise CaptureError('capture changed during comparison file verification')
    if digest.hexdigest() != manifest.get('file_sha256'):
        raise CaptureError('comparison manifest whole-file hash became stale')


def compare_requests(left_path, left_manifest, left_request,
                     right_path, right_manifest, right_request):
    """Pair by semantic identity, ignoring process-global ordinals. Bounded I/O.

    Earliest means actual hook order in the left manifest; only the fixed window
    is covered. Equal captures do not clear the uninstrumented C rejection.
    """
    if not left_manifest.get('structural_complete') or not right_manifest.get('structural_complete'):
        raise CaptureError('comparison requires structurally complete manifests')
    if left_manifest['byteorder'] != right_manifest['byteorder']:
        raise CaptureError('bitwise comparison requires matching producer byte order')
    left = [r for r in left_manifest['records'] if r['request_id'] == left_request]
    right = [r for r in right_manifest['records'] if r['request_id'] == right_request]
    if not left or not right:
        raise CaptureError('comparison request contains no records')
    akeys, bkeys = [semantic(r) for r in left], [semantic(r) for r in right]
    if len(set(akeys)) != len(akeys) or len(set(bkeys)) != len(bkeys) or set(akeys) != set(bkeys):
        raise CaptureError('comparison has duplicate/missing semantic identities')
    bmap = {semantic(r): r for r in right}
    result = {'bitwise_equal': True, 'paired_records': len(left), 'different_records': 0,
              'first_difference': None, 'original_C_rejection_cleared': False,
              'adopted': False, 'performance_eligible': False, 'full_lifecycle_passed': False}
    afd = bfd = None
    try:
        afd, apin = open_private(left_path, left_manifest['file_identity'])
        bfd, bpin = open_private(right_path, right_manifest['file_identity'])
        verify_manifest_file(afd, apin, left_manifest)
        verify_manifest_file(bfd, bpin, right_manifest)
        for a in left:
            b = bmap[semantic(a)]
            if a['payload_sha256'] == b['payload_sha256']:
                continue
            result['bitwise_equal'] = False
            result['different_records'] += 1
            os.lseek(afd, a['payload_offset'], os.SEEK_SET)
            os.lseek(bfd, b['payload_offset'], os.SEEK_SET)
            remaining, word = a['payload_bytes'], 0
            ah, bh = hashlib.sha256(), hashlib.sha256()
            while remaining:
                count = min(remaining, CHUNK_BYTES)
                achunk, bchunk = os.read(afd, count), os.read(bfd, count)
                if len(achunk) != count or len(bchunk) != count:
                    raise CaptureError('comparison payload shortened during read')
                ah.update(achunk); bh.update(bchunk)
                if achunk != bchunk and result['first_difference'] is None:
                    local = next(i for i in range(0, count, 4) if achunk[i:i+4] != bchunk[i:i+4])
                    index = word + local // 4
                    result['first_difference'] = {
                        'semantic_identity': dict(zip(SEMANTIC_FIELDS, semantic(a))),
                        'left_ordinal': a['ordinal'], 'right_ordinal': b['ordinal'],
                        'word': index, 'physical_row': a['first'] + index // a['width'],
                        'column': index % a['width'],
                        'left_byte_offset': a['payload_offset'] + index * 4,
                        'right_byte_offset': b['payload_offset'] + index * 4,
                        'left': word_summary(achunk[local:local+4], a['type'], left_manifest['byteorder']),
                        'right': word_summary(bchunk[local:local+4], b['type'], right_manifest['byteorder'])}
                remaining -= count
                word += count // 4
            if ah.hexdigest() != a['payload_sha256'] or bh.hexdigest() != b['payload_sha256']:
                raise CaptureError('comparison manifest payload hash became stale')
        if file_identity(os.fstat(afd)) != apin or file_identity(os.fstat(bfd)) != bpin:
            raise CaptureError('capture changed during comparison')
        return result
    finally:
        for fd in (afd, bfd):
            if fd is not None:
                os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--mode', choices=('framing', 'owned_full'), default='framing')
    parser.add_argument('--byteorder', choices=('little', 'big'), default=sys.byteorder)
    args = parser.parse_args()
    ledger = None
    try:
        if args.ledger:
            ledger = read_ledger(args.ledger)
        manifest = read_capture(args.capture, ledger, args.mode, args.byteorder)
    except CaptureError as error:
        print(json.dumps(error.manifest or {'structural_error': str(error)}, allow_nan=False))
        return 1
    print(json.dumps(manifest, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
