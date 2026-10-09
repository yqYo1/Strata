"""Root-run CPU validation source. SOURCE ONLY / UNTESTED; no GPU or model.

Fresh private --out required. Fixtures/receipt stay for root's serial retention
review; this test does not remove files. The optional real fixtures are read-only.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
from unittest import mock

import capture_reader as reader

PREFIX = '<' if sys.byteorder == 'little' else '>'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(reader.CHUNK_BYTES), b''):
            result.update(chunk)
    return result.hexdigest()


def packed(ordinal=0, phase='float_rows', values=(1.0, 2.0), **changes):
    row = {'magic': reader.MAGIC, 'version': 1, 'layer': 3, 'p0': 98304,
           'T': 8192, 'first': 98304, 'rows': 1, 'elements': len(values),
           'type': 1, 'width': len(values), 'ordinal': ordinal, 'header_bytes': 128}
    row.update(changes)
    name = phase.encode('ascii')
    header = struct.pack(PREFIX + '12Q', *(row[k] for k in reader.FIELDS)) + name.ljust(32, b'\0')
    payload = struct.pack(PREFIX + ('f' if row['type'] == 1 else 'i') * len(values), *values)
    return header + payload


def write_private(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as out:
        out.write(data)


def ledger_for(first_end, final_end):
    return [{'name': name, 'begin': begin, 'end': end} for name, begin, end in (
        (reader.REQUESTS[0], 0, 0), (reader.REQUESTS[1], 0, 0),
        (reader.REQUESTS[2], 0, first_end), (reader.REQUESTS[3], first_end, first_end),
        (reader.REQUESTS[4], first_end, final_end))]


PRODUCER_HEAD = '9c2ebde89e5157c81a9c9ae135719452a0244ca3'
PRODUCER_SHA256 = 'acd062d1a4f4ab29230630e066fbc29080f92c270cb1fc3e900be12502a17b56'
PRODUCER_FIELDS = ('magic', 'version', 'layer', 'p0', 'T', 'first', 'rows',
                   'elements', 'type', 'width', 'ordinal', 'header_bytes')


def producer_reference_records():
    """Independent static transcription of pinned prefill.cpp hooks/geometry.

    No reader descriptor, phase-width table, field order or constants are used.
    The root runner must supply and hash-check that producer source.
    """
    result = []
    for layer in range(13):
        phases = [('R_input', 98304, 32, 10240, 1)]
        if layer in (3, 7, 11):
            phases.extend((phase, 98304, 32, width, 1) for phase, width in (
                ('K_quant_input', 512), ('V_quant_input', 512), ('indexer_raw', 128),
                ('query', 6144), ('q_indexer', 512)))
            phases.extend(('scores', pos, 1, (pos + 1) // 4 + 1, 1)
                          for pos in range(98304, 98336))
            for pos in range(98304, 98336):
                phases.extend((('steps', pos, 1, 4, 2), ('selected_ids', pos, 1, 2051, 2)))
            phases.append(('attention_output', 98304, 32, 6144, 1))
        phases.extend((phase, 98304, 32, 10240, 1)
                      for phase in ('R_post_attention_gdn', 'R_post_moe'))
        for phase, first, rows, width, kind in phases:
            result.append(dict(layer=layer, phase=phase, p0=98304, T=8192,
                               first=first, rows=rows, elements=rows * width,
                               type=kind, width=width))
    if len(result) != 345 or sum(128 + r['elements'] * 4 for r in result) != 66747936:
        raise AssertionError('independent producer grammar/count/budget changed')
    return result


def generate_full(path):
    """Actual hook-order fixture, deliberately unlike the earlier fake harness.

    Only one <=1.25-MiB record payload is ever resident. Repeated selected IDs
    are legal test data: the reader must not invent uniqueness/sorting rules.
    """
    descriptors = producer_reference_records()
    if len(descriptors) != 345:
        raise AssertionError('source-derived count changed')
    if [r['phase'] for r in descriptors[9:15]] != [
            'R_input', 'K_quant_input', 'V_quant_input', 'indexer_raw', 'query', 'q_indexer']:
        raise AssertionError('actual first QSA hook order changed')
    if [r['phase'] for r in descriptors[15:47]] != ['scores'] * 32 or \
            [r['phase'] for r in descriptors[47:51]] != ['steps', 'selected_ids'] * 2:
        raise AssertionError('actual scores-before-selection hook order changed')
    if descriptors[47]['first'] != 98304 or descriptors[15]['width'] != 24577 or \
            descriptors[18]['width'] != 24578:
        raise AssertionError('pos+1 block-boundary arithmetic changed')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    ends = []
    with os.fdopen(fd, 'wb') as out:
        for full in range(2):
            for index, descriptor in enumerate(descriptors):
                row = dict(descriptor, magic=0x5354524152505431, version=1,
                           ordinal=full * 345 + index, header_bytes=128)
                header = struct.pack(PREFIX + '12Q', *(row[k] for k in PRODUCER_FIELDS))
                out.write(header + row['phase'].encode().ljust(32, b'\0'))
                if row['phase'] == 'steps':
                    pos = row['first']
                    out.write(struct.pack(PREFIX + '4i', pos, pos + 1, (pos + 1) // 4, 2051))
                else:
                    out.write(bytes(row['elements'] * 4))
            ends.append(out.tell())
    if ends != [66747936, 133495872]:
        raise AssertionError('source-derived per-full byte count changed: ' + repr(ends))
    return ledger_for(*ends)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--producer-source', type=Path, required=True,
                        help='pinned producer sycl/src/prefill/prefill.cpp')
    parser.add_argument('--real-normal', type=Path)
    parser.add_argument('--real-partial', type=Path)
    args = parser.parse_args()
    args.out.mkdir(mode=0o700)  # no overwrite, deletion, or stale-fixture reuse
    start = time.monotonic()
    receipt = {'source_status': 'SOURCE ONLY / UNTESTED until root executes',
               'active': True, 'passed': False, 'gpu_executed': False,
               'model_opened': False, 'adopted': False, 'cases': [],
               'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'reader_sha256': digest(Path(reader.__file__)), 'test_sha256': digest(__file__),
               'python': sys.version, 'uname': list(os.uname()),
               'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               'deadline_owner': 'root serial supervisor',
               'fixture_budget_bytes': 140 * 1024**2,
               'retention': {'owner': 'root reader CPU validation', 'raw_deleted': False,
                   'consumer': 'framing/parser/pair validation',
                   'next_review': 'root closes receipt, extracts evidence and retires fixtures'}}
    file_index = 0

    def save():
        receipt['elapsed_seconds'] = time.monotonic() - start
        temp = args.out / 'record.json.tmp'
        temp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        temp.replace(args.out / 'record.json')

    def check(name, action):
        entry = {'case': name, 'passed': False}
        receipt['cases'].append(entry)
        try:
            details = action()
            entry['details'] = details
            entry['passed'] = True
        except BaseException as error:
            entry['error'] = repr(error)
            if isinstance(error, reader.CaptureError) and error.manifest:
                entry['parse_error'] = error.manifest.get('structural_error')
        save()

    def fixture(data):
        nonlocal file_index
        file_index += 1
        path = args.out / ('small-%03d.bin' % file_index)
        write_private(path, data)
        return path

    def expect_rejected(data):
        path = fixture(data)
        try:
            reader.read_capture(path)
        except reader.CaptureError as error:
            return {'rejection': str(error), 'fixture': str(path), 'sha256': digest(path)}
        raise AssertionError('malformed stream accepted')

    def expect_bad_metadata(name, **changes):
        check(name, lambda: expect_rejected(packed(**changes)))

    def good_small():
        path = fixture(packed())
        manifest = reader.read_capture(path)
        if manifest['record_count'] != 1 or manifest['records'][0]['numeric_summary']['minimum'] != 1.0:
            raise AssertionError('finite payload summary mismatch')
        json.dumps(manifest, allow_nan=False)
        return {'fixture': str(path), 'sha256': manifest['file_sha256']}

    def producer_pin():
        actual = digest(args.producer_source)
        if actual != PRODUCER_SHA256:
            raise AssertionError('producer source differs from independently reviewed reference')
        if reader.expected_full_records() != producer_reference_records():
            raise AssertionError('reader coverage differs from independent producer reference')
        return {'producer_head': PRODUCER_HEAD, 'source': str(args.producer_source),
                'sha256': actual, 'records_per_full': 345, 'bytes_per_full': 66747936}
    check('independent-producer-source-pin-and-all-record-descriptors', producer_pin)
    if not receipt['cases'][-1]['passed']:
        receipt['active'] = False
        save()
        return 1

    check('finite-framing', good_small)
    for name, changes in (
        ('wrong-magic', {'magic': 0}), ('wrong-version', {'version': 2}),
        ('wrong-header-size', {'header_bytes': 96}), ('wrong-first-ordinal', {'ordinal': 1}),
        ('forged-huge-elements', {'elements': (1 << 64) - 1}),
        ('overflow-shape', {'rows': (1 << 63) - 1, 'width': (1 << 63) - 1}),
        ('extent-overflow', {'p0': (1 << 63) - 2, 'T': 8192}),
        ('invalid-type', {'type': 3}), ('invalid-layer', {'layer': 13}),
        ('invalid-phase', {'phase': 'unknown'}), ('zero-rows', {'rows': 0}),
        ('row-outside-chunk', {'first': 1}), ('element-shape-mismatch', {'elements': 1}),
        ('float-phase-int-type', {'type': 2, 'values': (1, 2)}),
        ('integer-phase-float-type', {'phase': 'steps'})):
        # invalid-type packing itself must remain valid test bytes; alter header
        # after a normally packed frame so test preparation cannot be the failure.
        if name == 'invalid-type':
            base = bytearray(packed()); base[64:72] = struct.pack(PREFIX + 'Q', 3)
            check(name, lambda data=bytes(base): expect_rejected(data))
        else:
            check(name, lambda changes=changes: expect_rejected(packed(**changes)))
    check('truncated-header', lambda: expect_rejected(packed()[:100]))
    check('truncated-payload', lambda: expect_rejected(packed()[:-1]))
    check('trailing-byte', lambda: expect_rejected(packed() + b'x'))
    check('ordinal-duplicate', lambda: expect_rejected(packed() + packed(ordinal=0)))
    check('ordinal-gap', lambda: expect_rejected(packed() + packed(ordinal=2)))
    check('bounded-record-manifest', lambda: expect_rejected(
        b''.join(packed(ordinal=i) for i in range(691))))
    padding = bytearray(packed()); padding[127] = 1
    check('phase-nonzero-padding', lambda: expect_rejected(bytes(padding)))

    def nan_and_partial():
        path = fixture(packed(values=(float('nan'), 1.0)))
        manifest = reader.read_capture(path)
        if not manifest['structural_complete'] or not manifest['numerical_findings_present'] or \
                manifest['records'][0]['numeric_summary']['nan_count'] != 1:
            raise AssertionError('NaN discarded or mistaken for framing error')
        json.dumps(manifest, allow_nan=False)
        incomplete = fixture(path.read_bytes() + b'x')
        try:
            reader.read_capture(incomplete)
        except reader.CaptureError as error:
            if not error.manifest['numeric_findings'] or error.manifest['record_count'] != 1:
                raise AssertionError('earlier bad numerical payload lost on later framing error')
        else:
            raise AssertionError('trailing bytes accepted')
        return {'finding': manifest['numeric_findings'][0]}
    check('NaN-is-data-and-preserved-on-later-framing-error', nan_and_partial)

    def invalid_numeric():
        path = fixture(packed(phase='steps', type=2, values=(98304, 98305, 24577, 2051)) +
                       packed(ordinal=1, phase='selected_ids', type=2, values=(-1, 98305)))
        manifest = reader.read_capture(path)
        kinds = [r['numeric_summary']['first_finding']['kind'] for r in manifest['records']]
        if kinds != ['invalid_step', 'selected_id_outside_cached_cells']:
            raise AssertionError('invalid step/ID missing')
        return {'findings': manifest['numeric_findings']}
    check('invalid-step-and-ID-remain-structurally-valid', invalid_numeric)

    def pair_difference():
        left = fixture(packed(values=(1.0, 2.0, 3.0, 4.0), rows=2, width=2))
        right = fixture(packed(values=(1.0, 2.0, 9.0, 4.0), rows=2, width=2))
        a, b = reader.read_capture(left), reader.read_capture(right)
        result = reader.compare_requests(left, a, None, right, b, None)
        first = result['first_difference']
        if result['bitwise_equal'] or first['word'] != 2 or first['physical_row'] != 98305 or \
                first['column'] != 0 or first['left']['value'] != 3.0 or first['right']['value'] != 9.0:
            raise AssertionError('first difference row/word/value incorrect')
        same = reader.compare_requests(left, a, None, left, a, None)
        if not same['bitwise_equal'] or same['original_C_rejection_cleared']:
            raise AssertionError('equal pair changed original rejection')
        return result
    check('pair-first-difference-and-equal-caution', pair_difference)

    def pair_integer():
        left = fixture(packed(phase='selected_ids', type=2, values=(1, 2)))
        right = fixture(packed(phase='selected_ids', type=2, values=(1, 3)))
        result = reader.compare_requests(left, reader.read_capture(left), None,
                                        right, reader.read_capture(right), None)
        first = result['first_difference']
        if first['word'] != 1 or first['left']['value'] != 2 or first['right']['value'] != 3 or \
                not first['left']['finite']:
            raise AssertionError('integer difference summary mismatch')
        return result
    check('pair-integer-raw-word-summary', pair_integer)

    def chunk_boundary_difference():
        values = (0.0,) * (reader.CHUNK_BYTES // 4 + 1)
        left = fixture(packed(values=values))
        right_data = bytearray(packed(values=values))
        right_data[-4:] = struct.pack(PREFIX + 'I', 0x7fc01234)
        right = fixture(bytes(right_data))
        a, b = reader.read_capture(left), reader.read_capture(right)
        result = reader.compare_requests(left, a, None, right, b, None)
        first = result['first_difference']
        finding = b['records'][0]['numeric_summary']['first_finding']
        if first['word'] != reader.CHUNK_BYTES // 4 or first['right']['raw_bits'] != 0x7fc01234 or \
                finding['raw_bits'] != 0x7fc01234:
            raise AssertionError('chunk-boundary raw NaN bits/word lost')
        return result
    check('pair-chunk-boundary-NaN-raw-bits', chunk_boundary_difference)

    def stale_identity():
        path = fixture(packed())
        manifest = reader.read_capture(path)
        with path.open('r+b') as out:
            out.seek(128); out.write(struct.pack(PREFIX + 'f', 7.0))
        try:
            reader.compare_requests(path, manifest, None, path, manifest, None)
        except reader.CaptureError as error:
            return {'rejection': str(error)}
        raise AssertionError('stale manifest accepted')
    check('stale-manifest-identity-rejected', stale_identity)

    def stale_content_with_unchanged_identity(header, same_path):
        left = fixture(packed())
        right = left if same_path else fixture(packed())
        a = reader.read_capture(left)
        b = a if same_path else reader.read_capture(right)
        pinned = b['file_identity']
        with right.open('r+b') as out:
            out.seek(16 if header else 128)
            out.write(struct.pack(PREFIX + 'Q', 4) if header else struct.pack(PREFIX + 'f', 7.0))
        actual_identity = reader.file_identity
        def unchanged_identity(st):
            identity = actual_identity(st)
            return dict(pinned) if identity['ino'] == pinned['ino'] and \
                identity['dev'] == pinned['dev'] else identity
        with mock.patch.object(reader, 'file_identity', unchanged_identity):
            try:
                reader.compare_requests(left, a, None, right, b, None)
            except reader.CaptureError as error:
                if 'whole-file hash became stale' not in str(error):
                    raise AssertionError('stale content was not rejected by whole-file digest')
                if a['original_C_rejection_cleared'] or b['original_C_rejection_cleared']:
                    raise AssertionError('stale comparison cleared original C rejection')
                return {'rejection': str(error), 'mutation': 'header' if header else 'payload',
                        'same_path': same_path, 'metadata_identity_simulated_unchanged': True}
        raise AssertionError('equal manifest hashes hid stale closed-file bytes')
    for header in (False, True):
        for same_path in (False, True):
            check('stale-%s-unchanged-identity-%s' %
                  ('header' if header else 'payload', 'same-path' if same_path else 'right-file'),
                  lambda header=header, same_path=same_path:
                      stale_content_with_unchanged_identity(header, same_path))

    def unsafe_file():
        original = fixture(packed())
        link = args.out / 'symlink.bin'; link.symlink_to(original)
        for path in (link,):
            try:
                reader.read_capture(path)
            except reader.CaptureError:
                pass
            else:
                raise AssertionError('symlink accepted')
        hard = args.out / 'hardlink.bin'; os.link(original, hard)
        try:
            reader.read_capture(original)
        except reader.CaptureError:
            pass
        else:
            raise AssertionError('two-link file accepted')
        mode = fixture(packed()); mode.chmod(0o644)
        try:
            reader.read_capture(mode)
        except reader.CaptureError:
            return {'symlink': 'rejected', 'hardlink': 'rejected', '0644': 'rejected'}
        raise AssertionError('nonprivate file accepted')
    check('unsafe-file-rejected', unsafe_file)

    def wrong_parent():
        directory = args.out / 'public'; directory.mkdir(mode=0o755)
        path = directory / 'file.bin'; write_private(path, packed())
        try:
            reader.read_capture(path)
        except reader.CaptureError as error:
            return {'rejection': str(error)}
        raise AssertionError('nonprivate parent accepted')
    check('nonprivate-parent-rejected', wrong_parent)

    def path_edges():
        for path in ('', '/', '////'):
            try:
                reader.read_capture(path)
            except reader.CaptureError as error:
                if not error.manifest or error.manifest['structural_complete']:
                    raise AssertionError('path rejection lost incomplete manifest')
            else:
                raise AssertionError('empty/root path accepted')
            try:
                reader.open_private(path)
            except reader.CaptureError:
                pass
            else:
                raise AssertionError('private opener accepted empty/root path')
        return {'rejected_paths': ['', '/', '////']}
    check('empty-and-root-path-CaptureError', path_edges)

    def ledger_ingestion():
        valid = json.dumps(ledger_for(100, 200)).encode()
        path = fixture(valid)
        if reader.read_ledger(path) != ledger_for(100, 200):
            raise AssertionError('private ledger JSON changed')
        boundary = fixture(valid + b' ' * (65536 - len(valid)))
        reader.read_ledger(boundary)
        link = args.out / 'ledger-symlink'; link.symlink_to(path)
        parent_link = args.out / 'ledger-parent-symlink'; parent_link.symlink_to(args.out)
        hard = fixture(valid); os.link(hard, args.out / 'ledger-hardlink')
        mode = fixture(valid); mode.chmod(0o644)
        directory = args.out / 'ledger-directory'; directory.mkdir(mode=0o700)
        fifo = args.out / 'ledger-fifo'; os.mkfifo(fifo, 0o600)
        public = args.out / 'ledger-public'; public.mkdir(mode=0o755)
        public_file = public / 'ledger.json'; write_private(public_file, valid)
        cases = (link, parent_link / path.name, hard, mode, directory, fifo, public_file,
                 fixture(b'x' * 65537), fixture(b'{'), fixture(b'\xff'),
                 fixture(b'[' * 2000 + b']' * 2000), Path('/'))
        for bad in cases:
            try:
                reader.read_ledger(bad)
            except reader.CaptureError:
                pass
            else:
                raise AssertionError('unsafe/malformed ledger accepted: ' + str(bad))
        return {'rejected_variants': len(cases), 'exact_64KiB_accepted': True}
    check('bounded-private-ledger-JSON-and-filesystem-policy', ledger_ingestion)

    def ledger_read_mutation(grow):
        path = fixture(b'[]')
        actual_read = os.read
        requests = []
        def changed_read(fd, count):
            requests.append(count)
            if len(requests) == 1:
                with path.open('ab' if grow else 'wb') as out:
                    out.write(b' ' * 65536 if grow else b'{}')
            return actual_read(fd, count)
        with mock.patch.object(reader.os, 'read', changed_read):
            try:
                reader.read_ledger(path)
            except reader.CaptureError as error:
                if grow and sum(requests) > 65537:
                    raise AssertionError('ledger read requested more than bounded budget')
                if not grow and 'changed' not in str(error):
                    raise AssertionError('same-size mutation not rejected by stable fstat')
                return {'rejection': str(error), 'read_requests': requests}
        raise AssertionError('concurrently changed ledger accepted')
    check('ledger-growth-read-budget-plus-one', lambda: ledger_read_mutation(True))
    check('ledger-same-size-mutation-stable-fstat', lambda: ledger_read_mutation(False))

    def bad_ledger():
        good = ledger_for(100, 200)
        variants = []
        short = copy.deepcopy(good); short.pop(); variants.append(short)
        gap = copy.deepcopy(good); gap[3]['begin'] = 99; variants.append(gap)
        cross = copy.deepcopy(good); cross[2]['name'] = 'full256k-repeat'; variants.append(cross)
        extra = copy.deepcopy(good); extra[0]['end'] = 1; variants.append(extra)
        missing = copy.deepcopy(good); missing[4]['end'] = 199; variants.append(missing)
        boolean = copy.deepcopy(good); boolean[0]['begin'] = False; variants.append(boolean)
        for index in range(5):
            for field in ('begin', 'end'):
                for value in (-1, 1 << 200, True, 1.5, None):
                    bad = copy.deepcopy(good); bad[index][field] = value; variants.append(bad)
        for size in (-1, reader.MAX_BYTES + 1, 1 << 200, True, 200.0, None):
            try:
                reader.validate_ledger(good, size)
            except reader.CaptureError:
                pass
            else:
                raise AssertionError('invalid ledger file size accepted')
        for variant in variants:
            try:
                reader.validate_ledger(variant, 200)
            except reader.CaptureError:
                continue
            raise AssertionError('bad ledger accepted')
        return {'rejected_variants': len(variants)}
    check('request-ledger-missing-gap-order-extra-trailing-bool', bad_ledger)

    def bad_coverage():
        good = producer_reference_records()
        reader.validate_full_segment(good)
        variants = [good[:-1]]
        dup = copy.deepcopy(good); dup[1] = copy.deepcopy(dup[0]); variants.append(dup)
        order = copy.deepcopy(good); order[47], order[48] = order[48], order[47]; variants.append(order)
        scores = copy.deepcopy(good); scores[18]['width'] -= 1; scores[18]['elements'] -= 1; variants.append(scores)
        row = copy.deepcopy(good); row[15]['first'] += 1; variants.append(row)
        phase = copy.deepcopy(good); phase[1]['phase'] = 'test'; variants.append(phase)
        layer = copy.deepcopy(good); layer[1]['layer'] = 13; variants.append(layer)
        for variant in variants:
            try:
                reader.validate_full_segment(variant)
            except reader.CaptureError:
                continue
            raise AssertionError('missing/duplicate/wrong-order/shape coverage accepted')
        return {'rejected_variants': len(variants)}
    check('owned-model-missing-duplicate-order-score-row-phase-layer', bad_coverage)

    def malformed_owned_stream(duplicate=False):
        frames = [packed(ordinal=i) for i in range(3 if duplicate else 2)]
        first_end = sum(len(frame) for frame in frames[:-1])
        data = b''.join(frames)
        path = fixture(data)
        try:
            reader.read_capture(path, ledger=ledger_for(first_end, len(data)), mode='owned_full')
        except reader.CaptureError as error:
            wanted = 'duplicate' if duplicate else '345'
            if wanted not in str(error):
                raise
            return {'rejection': str(error), 'fixture_sha256': digest(path)}
        raise AssertionError('malformed owned coverage stream accepted')
    check('owned-stream-duplicate-record', lambda: malformed_owned_stream(True))
    check('owned-stream-missing-model-coverage', malformed_owned_stream)

    full_path = args.out / 'two-full.bin'
    full_manifest = None
    def full_parse():
        nonlocal full_manifest
        ledger = generate_full(full_path)
        full_manifest = reader.read_capture(full_path, ledger=ledger, mode='owned_full')
        if full_manifest['record_count'] != 690 or full_manifest['numerical_findings_present']:
            raise AssertionError('complete finite owned fixture failed')
        pair = reader.compare_requests(full_path, full_manifest, reader.FULL_REQUESTS[0],
                                       full_path, full_manifest, reader.FULL_REQUESTS[1])
        if not pair['bitwise_equal'] or pair['paired_records'] != 345:
            raise AssertionError('global ordinal should not affect payload comparison')
        return {'record_count': 690, 'size': full_manifest['file_identity']['size'],
                'file_sha256': full_manifest['file_sha256'], 'pair': pair}
    check('actual-order-two-full-parser-and-ordinal-independent-pair', full_parse)

    def crossed_interval():
        # Valid total ledger size but boundary divides a real record.
        ledger = ledger_for(66747935, 133495872)
        try:
            reader.read_capture(full_path, ledger=ledger, mode='owned_full')
        except reader.CaptureError as error:
            if 'crosses' not in str(error):
                raise
            return {'rejection': str(error), 'preserved_records': error.manifest['record_count']}
        raise AssertionError('split frame request boundary accepted')
    if full_manifest is not None:
        check('ledger-boundary-splits-record', crossed_interval)

    if args.real_normal:
        def real_normal():
            manifest = reader.read_capture(args.real_normal)
            if not manifest['structural_complete'] or manifest['record_count'] != 2:
                raise AssertionError('real producer normal framing mismatch')
            return {'file_sha256': manifest['file_sha256'], 'records': [
                {'phase': r['phase'], 'type': r['type'], 'rows': r['rows'], 'width': r['width']}
                for r in manifest['records']], 'numeric_findings': manifest['numeric_findings']}
        check('real-production-small-normal-framing-only', real_normal)
    if args.real_partial:
        def real_partial():
            try:
                reader.read_capture(args.real_partial)
            except reader.CaptureError as error:
                return {'rejection': str(error), 'file_sha256': digest(args.real_partial)}
            raise AssertionError('real partial producer write accepted')
        check('real-production-partial-write-rejected', real_partial)
    receipt['fixtures'] = [{'path': str(p), 'size': p.stat().st_size,
                            'sha256': digest(p)} for p in args.out.rglob('*.bin')
                           if p.is_file() and not p.is_symlink()]
    receipt['passed'] = all(case['passed'] for case in receipt['cases'])
    receipt['active'] = False
    receipt['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print(json.dumps({'passed': receipt['passed'], 'cases': len(receipt['cases']),
                      'receipt': str(args.out / 'record.json')}))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
