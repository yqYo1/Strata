#!/usr/bin/env python3
"""Prepare an unadopted CPU-history-first profile from closed, bounded counts.

Pure data preparation: no model, device, native loader, test or benchmark runs.
The input omits old GPU-owned frequencies; absent counts are not measured zeros.
"""
import base64
import importlib.util
import json
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A))
from analyze_profile_cache_coverage_v1 import INPUT_SHA, PROFILE_SHA, read, require, sha


def main():
    training_path = A / 'cpu-gu-reuse-long-v1.json'
    training_data = read(training_path, 1 << 20)
    prior_path = P / 'data/expert-profile.bin'
    prior_data = read(prior_path, 256 << 10)
    require(sha(training_data) == INPUT_SHA and sha(prior_data) == PROFILE_SHA, 'training/prior identity')
    source = json.loads(training_data)
    require(source['passed'] and source['complete'] and not source['active'] and not source['fresh_faults'],
            'closed successful training census required')
    request = source['requests'][0]
    counts_info = request['recurrence']
    packed = base64.b64decode(counts_info['expert_counts_base64'], validate=True)
    require(len(packed) <= 1 << 20 and sha(packed) == counts_info['encoded_counts_sha256'], 'exact bounded counts')
    values, value, shift = [], 0, 0
    for byte in packed:
        require(shift < 64, 'counter overflow')
        value |= (byte & 127) << shift
        if byte & 128:
            shift += 7
        else:
            values.append(value)
            value, shift = 0, 0
    require(not shift and len(values) == 98304 and sum(values) == counts_info['header']['jobs'],
            'complete job histogram')
    frequencies = {(l, e): sum((t + 1) * values[(l * 512 + e) * 4 + t] for t in range(4))
                   for l in range(48) for e in range(512)}
    frequencies = {pair: n for pair, n in frequencies.items() if n}
    require(sum(frequencies.values()) == counts_info['header']['tokens'] and
            len(frequencies) == counts_info['header']['first_touches_unique'], 'training conservation')
    writer_path = P / 'tools/make_profile.py'
    writer_data = read(writer_path, 64 << 10)
    spec = importlib.util.spec_from_file_location('existing_profile_writer', writer_path)
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    prior = writer.read_profile(prior_path)
    require(len(prior) == len(set(prior)) == 24576, 'complete prior ranking')
    ranked, origin = writer.rank_profile(prior, frequencies, reorder=True)
    require(len(ranked) == len(set(ranked)) == 24576 and
            all(0 <= l < 48 and 0 <= e < 512 for l, e in ranked), 'complete candidate ranking')
    root = B / 'main-cpu-count-profile-preparation-root-v1'
    receipt = A / 'cpu-count-profile-preparation-v1.json'
    require(not root.exists() and not receipt.exists(), 'fresh preparation destination required')
    candidate = root / 'cpu-history-first-profile.bin'
    record = dict(schema='cpu-history-first-profile-preparation-v1', preparation_complete=False,
        GPU_executed=False, new_inference_executed=False, native_loader_qualified=False,
        performance_qualified=False, quality_qualified=False, adopted=False,
        recipe_sha256=sha(read(Path(__file__), 64 << 10)),
        count_reader_sha256=sha(read(A / 'analyze_profile_cache_coverage_v1.py', 64 << 10)),
        profile_writer_sha256=sha(writer_data), python=sys.version,
        training_receipt_sha256=sha(training_data), training_input_sha256=source['input_sha256'],
        prior_profile_sha256=sha(prior_data), candidate=str(candidate),
        known_CPU_jobs=sum(values), known_CPU_rows=sum(frequencies.values()), known_CPU_pairs=len(frequencies),
        ranking_origin=dict(origin), ranking='Observed CPU expert-token rows descending, pair ID tie break; '
            'the unmodified prior fills every remaining pair in original order.',
        missing_GPU_ID_frequencies='Unknown, not measured zeros. This deliberately CPU-history-first candidate '
            'does not estimate a complete routing distribution.',
        next_consumer='Held-out official bench.sh profile pilot at fixed legacy128 quota, actual native loader, '
            'cache payload, owner/numerical/task-quality and separate clean repeated throughput validation.',
        retention=dict(byte_budget=196632, review_point='After the held-out pilot decision or supersession; '
            'retire the unadopted candidate if no comparison consumes it.'),
        scope='Prepared configuration only. Historical profile content identity was not pinned by the training '
            'producer. No held-out benefit, native loading, live VRAM, physical256K or RESTORE qualification.')
    root_created = False
    try:
        root.mkdir(mode=0o700)
        root_created = True
        writer.write_profile(candidate, ranked)
        data = read(candidate, 196632)
        require(len(data) == 196632 and data[:4] == b'STRP' and
                struct.unpack_from('<5I', data, 4) == (1, 48, 512, 24576, 24576), 'candidate header/length')
        decoded = [struct.unpack_from('<HH', data, 24 + 4 * i) for i in range(24576)]
        require(decoded == ranked, 'candidate ordered pairs')
        inverse = struct.unpack_from('<24576i', data, 24 + 24576 * 4)
        require(all(inverse[l * 512 + e] == i for i, (l, e) in enumerate(decoded)), 'candidate full inverse table')
        quota, used, selected = [2] * 47 + [34], [0] * 48, []
        for l, e in decoded:
            if used[l] < quota[l]:
                selected.append((l, e))
                used[l] += 1
        require(len(selected) == 128 and used == quota, 'current legacy128 quota')
        training_rows = sum(frequencies.get(pair, 0) for pair in selected)
        require(training_rows == 10159, 'expected known-CPU training coverage; not held-out inference')
        require(read(training_path, 1 << 20) == training_data and read(prior_path, 256 << 10) == prior_data and
                read(writer_path, 64 << 10) == writer_data, 'inputs changed during preparation')
        record.update(preparation_complete=True, candidate_bytes=len(data), candidate_sha256=sha(data),
                      binary_schema_and_inverse_validated=True, known_training_selected_rows=training_rows,
                      training_coverage_is_hindsight=True)
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        if root_created:
            with receipt.open('x') as stream:
                json.dump(record, stream, indent=2)
                stream.write('\n')
    print(json.dumps(dict(preparation_complete=record['preparation_complete'], candidate=str(candidate),
                          candidate_sha256=record['candidate_sha256'], GPU_executed=False)))


if __name__ == '__main__':
    main()
