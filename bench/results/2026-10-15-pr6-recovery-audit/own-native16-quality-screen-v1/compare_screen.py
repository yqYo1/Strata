#!/usr/bin/env python3
"""Independent host FP64 analysis of completed, bound sparse logit captures.

Original project code under the repository LICENSE. No inference backend imports.
"""
import argparse
import array
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

sys.dont_write_bytecode = True
ARMS = ('fp32-a', 'fp32-b', 'bf16')
POSITIONS = [255 + 512 * i for i in range(64)]
VOCAB = 248320
RAW_BYTES = 8 + 64 * VOCAB * 4
SCHEMA = 'strata-native16-natural-sparse-screen-v1'
LIMITATION = ('One ordered project-document corpus and 64 correlated positions: '
              'exploratory numerical screen only. No general/task-quality, statistical '
              'confidence interval, clean-speed, full physical-context or RESTORE '
              'qualification; no automatic adoption or numerical acceptance threshold.')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_sha(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)

    require(Path(path).stat().st_size <= 16 << 20, 'bounded JSON input')
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad_constant)


def artifact(path):
    path = Path(path).resolve()
    require(path.is_file(), 'missing artifact: ' + str(path))
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def check_artifact(value, *, cap):
    path = Path(value['path'])
    require(path.is_absolute() and path.is_file(), 'absolute regular artifact path required')
    require(path.stat().st_size == value['bytes'] <= cap and sha(path) == value['sha256'],
            'artifact size/hash mismatch: ' + str(path))
    return path


def load_input(receipt_path, expected_sha, *, check_sources=True):
    require(sha(receipt_path) == expected_sha, 'input receipt hash')
    value = load_json(receipt_path)
    require(value['complete'] is True and value['GPU_executed'] is False and
            value['thinking_requested'] is True and value['roundtrip_exact'] is True and
            value['repeated_padding'] is False, 'natural/template/thinking provenance')
    require(value['sampled_positions'] == POSITIONS, 'fixed 64 sampling positions')
    require(32768 < value['count'] < 35000 and value['batched_rows'] == value['count'] - 1,
            'actual natural input must exceed 32768 tokens')
    path = Path(value['token_file'])
    require(path.stat().st_size <= 1 << 20 and sha(path) == value['file_sha256'], 'token file pin/cap')
    text = path.read_text(encoding='ascii')
    require(re.fullmatch(r'\s*[0-9]+(?:[\s,]+[0-9]+)*\s*', text), 'strict unsigned token file')
    tokens = [int(part) for part in re.split(r'[\s,]+', text.strip())]
    require(len(tokens) == value['count'] and all(0 <= t < VOCAB for t in tokens), 'token count/range')
    require(hashlib.sha256(struct.pack('<' + str(len(tokens)) + 'I', *tokens)).hexdigest() ==
            value['ids_le_u32_sha256'], 'exact little-endian conditioning IDs')
    require(POSITIONS[-1] + 1 < len(tokens) - 1, 'all targets precede last prompt token')
    require(value['source_pins'], 'input source bindings absent')
    if check_sources:
        for name, digest in value['source_pins'].items():
            require(sha(name) == digest, 'input source/template pin: ' + name)
    return value, tokens


def parse_metadata(path, tokens, positions, vocab, *, expected_head_type):
    require(Path(path).stat().st_size <= 8192, 'metadata budget')
    data = Path(path).read_bytes()
    require(data.endswith(b'\n') and b'\r' not in data, 'exact metadata line endings')
    lines = data.decode('ascii').splitlines()
    expected = [
        'strata-prefill-logits-v1',
        'format=native-endian-int32-vocab-count-float32-rows',
        'vocab=' + str(vocab), 'count=' + str(len(positions)),
        'batched=' + str(len(tokens) - 1), 'head=native',
        'head_ggml_type=' + str(expected_head_type), 'row\tposition\tnext_token_id',
    ]
    expected += [f'{i}\t{p}\t{tokens[p + 1]}' for i, p in enumerate(positions)]
    expected += [f'complete\t{len(positions)}\t{8 + len(positions) * vocab * 4}']
    require(lines == expected, 'exact metadata header/order/positions/targets/completion')
    return dict(vocab=vocab, count=len(positions), batched=len(tokens) - 1,
                head='native', head_ggml_type=expected_head_type,
                positions=list(positions), targets=[tokens[p + 1] for p in positions])


def open_capture(stack, raw_path, meta_path, tokens, positions, vocab, head_type):
    require(sys.byteorder == 'little' and array.array('f').itemsize == 4,
            'native capture platform must be little-endian IEEE FP32')
    require(positions and len(positions) <= 64 and positions == sorted(set(positions)) and
            all(0 <= p < len(tokens) - 1 for p in positions), 'valid unique ascending positions')
    expected_size = 8 + len(positions) * vocab * 4
    require(expected_size < 64 << 20 and Path(raw_path).stat().st_size == expected_size,
            'complete full-vocabulary payload size, no trailing bytes')
    meta = parse_metadata(meta_path, tokens, positions, vocab, expected_head_type=head_type)
    stream = stack.enter_context(Path(raw_path).open('rb'))
    require(struct.unpack('<ii', stream.read(8)) == (vocab, len(positions)),
            'raw vocabulary/count header (zero count is incomplete)')
    return stream, meta


def read_row(stream, vocab, target):
    data = stream.read(vocab * 4)
    require(len(data) == vocab * 4, 'truncated row')
    row = array.array('f')
    row.frombytes(data)
    require(0 <= target < vocab and all(math.isfinite(x) for x in row),
            'target range and every vocabulary logit finite')
    maximum = float(max(row))
    log_shifted_sum = math.log(math.fsum(math.exp(float(x) - maximum) for x in row))
    nll = (maximum - float(row[target])) + log_shifted_sum
    require(math.isfinite(nll) and nll >= 0, 'finite nonnegative independent NLL')
    # max(range(...)) returns the first index on ties; make that policy explicit.
    top1 = max(range(vocab), key=row.__getitem__)
    return row, dict(NLL=nll, top1=top1, top1_ties=sum(x == maximum for x in row),
                     maximum=maximum, log_shifted_sum=log_shifted_sum,
                     row_sha256=hashlib.sha256(data).hexdigest())


def pair_metrics(control, candidate, control_stats, candidate_stats, distributions):
    delta = candidate_stats['NLL'] - control_stats['NLL']
    result = dict(NLL_delta=delta, absolute_NLL_delta=abs(delta),
                  logit_max_absolute_delta=max(abs(float(a) - float(b)) for a, b in zip(control, candidate)),
                  top1_agrees=control_stats['top1'] == candidate_stats['top1'])
    if distributions:
        am, az = control_stats['maximum'], control_stats['log_shifted_sum']
        bm, bz = candidate_stats['maximum'], candidate_stats['log_shifted_sum']
        # Keep the shifted representation: adding a large max back to log(Z)
        # would lose bits before subtracting logits again. Zero underflowed
        # probabilities contribute zero; no artificial probability floor.
        kl = math.fsum(math.exp((float(a) - am) - az) *
                       (((float(a) - am) - az) - ((float(b) - bm) - bz))
                       for a, b in zip(control, candidate))
        l1 = math.fsum(abs(math.exp((float(a) - am) - az) - math.exp((float(b) - bm) - bz))
                       for a, b in zip(control, candidate))
        result.update(KL_control_to_candidate=kl, L1=l1)
    require(all(math.isfinite(v) for v in result.values()), 'finite pair metrics')
    return result


def exp_or_overflow(value):
    try:
        return dict(value=math.exp(value), overflow=False)
    except OverflowError:
        return dict(value=None, overflow=True, log_value=value)


def analyze_triplet(paths, tokens, positions, *, vocab=VOCAB, head_type=14, distributions=True):
    """Three row streams, bounded host memory; usable by small host fixtures."""
    require(set(paths) == set(ARMS), 'exactly three named arms')
    pairs = (('fp32-a', 'fp32-b'), ('fp32-a', 'bf16'), ('fp32-b', 'bf16'))
    individual = []
    with ExitStack() as stack:
        streams = {arm: open_capture(stack, *paths[arm], tokens, positions, vocab, head_type)
                   for arm in ARMS}
        for index, position in enumerate(positions):
            rows, stats = {}, {}
            target = tokens[position + 1]
            for arm in ARMS:
                rows[arm], stats[arm] = read_row(streams[arm][0], vocab, target)
            metrics = {a + '-vs-' + b: pair_metrics(rows[a], rows[b], stats[a], stats[b], distributions)
                       for a, b in pairs}
            individual.append(dict(index=index, position=position, target=target, arms=stats, pairs=metrics))
        require(all(not stream.read(1) for stream, _ in streams.values()), 'exact EOF on all arms')
    mean_nll = {arm: math.fsum(v['arms'][arm]['NLL'] for v in individual) / len(individual) for arm in ARMS}
    summaries = {}
    for a, b in pairs:
        key = a + '-vs-' + b
        values = [v['pairs'][key] for v in individual]
        mean_delta = math.fsum(v['NLL_delta'] for v in values) / len(values)
        item = dict(control=a, candidate=b, mean_NLL_delta=mean_delta,
                    exp_mean_delta_sampled_PPL_ratio=exp_or_overflow(mean_delta),
                    mean_absolute_NLL_delta=math.fsum(v['absolute_NLL_delta'] for v in values) / len(values),
                    max_absolute_NLL_delta=max(v['absolute_NLL_delta'] for v in values),
                    logit_max_absolute_delta=max(v['logit_max_absolute_delta'] for v in values),
                    top1_agreements=sum(v['top1_agrees'] for v in values), positions=len(values))
        if distributions:
            item.update(mean_KL_control_to_candidate=math.fsum(v['KL_control_to_candidate'] for v in values) / len(values),
                        mean_L1=math.fsum(v['L1'] for v in values) / len(values))
        summaries[key] = item
    return dict(valid=True, positions=len(positions), full_vocabulary_finite=True,
                FP64_oracle='Python float/math.fsum, max-shifted logsumexp; FP32 payload decoded exactly',
                top1_tie_rule='lowest token ID; tie count retained',
                distribution_metrics=distributions, mean_NLL=mean_nll,
                sampled_PPL={a: exp_or_overflow(n) for a, n in mean_nll.items()},
                pairs=summaries, individual_positions=individual,
                quality_accepted=False, acceptance_threshold=None, exact_text_equality_required=False,
                interpretation=LIMITATION)


def command_closed(command):
    return (command.get('observation_complete') and command.get('direct_child_reaped') and
            command.get('session_empty') and not command.get('error') and not command.get('errors') and
            not command.get('cleanup') and not command.get('survivors') and command.get('exit_code') == 0)


def compare_record(record, *, distributions=True):
    require(record['schema'] == SCHEMA and record['capture_complete'] is True and
            record['closure_validated'] is True and record['active'] is False, 'closed capture cohort')
    require(record['boot_unchanged'] is True and record['fresh_faults'] == [] and
            record['driver_dump_exists'] is False, 'fresh kernel/boot/device guards')
    require(record['declared_order'] == list(ARMS) and set(record['cases']) == set(ARMS), 'three ordered arms')
    require(all(command_closed(c) for c in record['commands']), 'every command closed normally')
    require(record['native_byteorder'] == 'little', 'capture byte order')
    require(record['quality_accepted'] is False and record['model_quality_qualified'] is False and
            record['performance_qualified'] is False and record['physical_262144_qualified'] is False,
            'screen must retain qualification limits')
    value, tokens = load_input(record['input_receipt']['path'], record['input_receipt']['sha256'])
    require(record['input_receipt']['bytes'] == Path(record['input_receipt']['path']).stat().st_size,
            'input receipt size')
    binding = record['binding']
    require(binding['input_receipt_sha256'] == record['input_receipt']['sha256'] and
            binding['input_ids_sha256'] == value['ids_le_u32_sha256'] and
            binding['input_file_sha256'] == value['file_sha256'] and
            binding['input_source_pins'] == value['source_pins'], 'conditioning/source binding')
    require(record['binding_sha256'] == canonical_sha(binding), 'common binding hash')
    require(record['source_pins'] == binding['source_pins'] and record['source_commit'] == binding['source_commit'],
            'record/source binding')
    require(record['controller'] == binding['controller'], 'true controller binding')
    require(record['owner']['sha256'] == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f',
            'qualified Owner source identity')
    for name in ('manifest', 'comparator'):
        check_artifact(record[name], cap=150 << 10)
    check_artifact(record['controller']['source'], cap=150 << 10)
    require(binding['binary']['sha256'] and binding['compiler']['sha256'] and binding['runtime_files'] and
            binding['model_identity'] and binding['pack_metadata_sha256'] and binding['MTP_metadata_sha256'],
            'binary/toolchain/model/pack/runtime identity evidence')
    version = record['build_version']
    build_artifact = record['references'][f'own-native16-prefill-build-{version}.json']
    build = load_json(check_artifact(build_artifact, cap=16 << 20))
    smoke_artifact = record['references'][f'own-native16-model-smoke-{version}.json']
    smoke = load_json(check_artifact(smoke_artifact, cap=16 << 20))
    require(build_artifact['sha256'] == binding['build_receipt_sha256'] and
            smoke_artifact['sha256'] == binding['diagnostic_environment_origin'] and
            build['source_pins'] == smoke['source_pins'] == binding['source_pins'] and
            build['source_commit'] == smoke['source_commit'] == binding['source_commit'] and
            build['binary_sha256'] == smoke['binary_sha256'] == binding['binary']['sha256'] and
            build['compiler_sha256'] == binding['compiler']['sha256'], 'qualified build/model receipt source chain')
    require(all(binding[k] == smoke[k] for k in ('runtime_files', 'model_identity', 'pack_metadata_sha256', 'MTP_metadata_sha256')),
            'runtime/model/pack metadata receipt chain')
    paths, raw_total = {}, 0
    environments = []
    command_indices = []
    for arm in ARMS:
        case = record['cases'][arm]
        recipe_path = check_artifact(case['recipe'], cap=2 << 20)
        recipe = load_json(recipe_path)
        command = record['commands'][case['command_index']]
        command_indices.append(case['command_index'])
        require(command['label'] == 'screen-' + arm and command['argv'] == recipe['argv'] and
                command['environment'] == recipe['environment'], 'actual owned command/recipe identity')
        require(recipe['arm'] == arm and recipe['binding'] == binding and
                recipe['binding_sha256'] == record['binding_sha256'], 'arm source/binary/model binding')
        require(case['complete'] is True and recipe['argv'] == [binding['binary']['path'], *record['args']] and
                command['direct_identity']['ppid'] == record['controller']['identity']['pid'] and
                command['direct_identity']['sid'] == command['pid'], 'same-argument direct native command')
        require(case['local_profile_initial_sha256'] == recipe['local_profile_initial_sha256'] ==
                binding['external_profile']['sha256'], 'identically reset expert profile')
        args = recipe['argv']
        for flag, expected in (('--max-context', '262144'), ('--kv-resident', '32768'), ('--prefill', '8192'), ('--max-new', '1')):
            require(args.count(flag) == 1 and args[args.index(flag) + 1] == expected, 'fixed CLI setting: ' + flag)
        require(args.count('--tokens-file') == 1 and sha(args[args.index('--tokens-file') + 1]) == value['file_sha256'],
                'private exact conditioning-file binding')
        require(recipe['environment']['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1' and
                recipe['environment']['STRATA_SYCL_PREFILL_NATIVE16'] == ('bf16' if arm == 'bf16' else '0') and
                recipe['environment']['STRATA_PREFILL_LAYER_MAJOR_R_DTYPE'] == ('bf16' if arm == 'bf16' else 'fp32') and
                not any(k.startswith('STRATA_PREFILL_NLL') for k in recipe['environment']), 'sparse-only actual arm')
        common_env = {k: v for k, v in recipe['environment'].items() if k not in (
            'STRATA_SYCL_PREFILL_NATIVE16', 'STRATA_PREFILL_LAYER_MAJOR_R_DTYPE',
            'STRATA_PREFILL_LOGITS', 'XDG_CACHE_HOME')}
        environments.append(common_env)
        raw = check_artifact(case['raw'], cap=64 << 20)
        meta = check_artifact(case['metadata'], cap=8192)
        require(str(raw) == recipe['environment']['STRATA_PREFILL_LOGITS'] and
                str(meta) == str(raw) + '.meta' and case['raw']['bytes'] == RAW_BYTES,
                'exclusive native observer output binding')
        require(recipe['environment']['STRATA_PREFILL_LOGITS_POSITIONS'] == ','.join(map(str, POSITIONS)),
                'actual sampled positions')
        require(recipe['argv'][0] == binding['binary']['path'] and '--no-token-graph' in recipe['argv'] and
                '--serve' not in recipe['argv'], 'direct standalone native CLI')
        for kind in ('stdout', 'stderr'):
            path = check_artifact(case[kind], cap=64 << 20)
            require(command['logs'][path.name] == {k: case[kind][k] for k in ('bytes', 'sha256')},
                    'owner log fingerprint')
        raw_total += case['raw']['bytes']
        paths[arm] = (raw, meta)
    require(command_indices == sorted(set(command_indices)), 'actual FP32-a/FP32-b/BF16 launch order')
    require(environments[0] == environments[1] == environments[2], 'same effective non-arm environment')
    require(raw_total <= 256 << 20, 'whole-cohort tensor budget')
    result = analyze_triplet(paths, tokens, POSITIONS, head_type=record['head_ggml_type'], distributions=distributions)
    # Detect changed captures during the streaming comparison as well.
    for case in record['cases'].values():
        check_artifact(case['raw'], cap=64 << 20)
        check_artifact(case['metadata'], cap=8192)
    result.update(binding_sha256=record['binding_sha256'], raw_tensor_bytes=raw_total)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--skip-distributions', action='store_true')
    options = parser.parse_args()
    require(not options.output.exists(), 'fresh comparison output; failed receipts stay immutable')
    result = dict(valid=False, complete=False, GPU_executed=False, recipe_sha256=sha(__file__),
                  input_receipt_sha256=options.receipt_sha256, quality_accepted=False, interpretation=LIMITATION)
    try:
        require(sha(options.receipt) == options.receipt_sha256, 'explicit source receipt pin')
        record = load_json(options.receipt)
        require(record['passed'] is True and record['complete'] is True,
                'original failed/incomplete receipt stays ineligible; inspect its evidence separately')
        result['comparison'] = compare_record(record, distributions=not options.skip_distributions)
        result.update(valid=True, complete=True)
    except Exception as error:
        result['error'] = repr(error)
    with options.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(valid=result['valid'], complete=result['complete'], quality_accepted=False)))
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
