#!/usr/bin/env python3
"""Main-only four-arm analysis; reuse the frozen native16 FP64 oracle unchanged.

Repository LICENSE applies. No inference/runtime imports, launches or cleanup.
The old three-arm record validator cannot accept the integrated vLLM receipt.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import time

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
A = D.parent
P = A.parents[2]
ORACLE = A / 'own-native16-quality-screen-v1/compare_screen.py'
ORACLE_SHA = 'dc4e56b916c014acc60a716bdc67bc1bd1256f53b9f1ef119093b16b5feef69d'
with ORACLE.open('rb') as _stream:
    if hashlib.file_digest(_stream, 'sha256').hexdigest() != ORACLE_SHA:
        raise RuntimeError('unchanged legacy FP64 oracle required')
_spec = importlib.util.spec_from_file_location('vllm_screen_legacy_oracle', ORACLE)
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
require, sha, canonical_sha = L.require, L.sha, L.canonical_sha
load_json, artifact, check_artifact = L.load_json, L.artifact, L.check_artifact
open_capture, read_row, pair_metrics = L.open_capture, L.read_row, L.pair_metrics
exp_or_overflow = L.exp_or_overflow
POSITIONS, VOCAB, RAW_BYTES = L.POSITIONS, L.VOCAB, L.RAW_BYTES
ARMS = ('fp32-a', 'fp32-b', 'bf16-vllm', 'bf16-vllm-qsa')
PAIRS = (('fp32-a', 'fp32-b'),) + tuple((a, b) for a in ARMS[:2] for b in ARMS[2:])
SCHEMA = 'strata-vllm-natural-quality-screen-v1'
LIMITATION = L.LIMITATION
BUILD_SHA = '990635731be1198706abeb682817fdab247f8f51d31bab971d3d4f4db53cb639'
SMOKE_SHA = '03b2bd388a897873bb2dc8c310ec7c0386e6b484828d1e633fb872cd059a14d3'
SOURCE_COMMIT = '6354f0b4ff3731466795410e11ca9ef785c9d326'
NLL_TOLERANCE = 1.e-10  # Same independent-oracle tolerance as integrated model smoke.
RAW_CAP = 384 << 20


def deadline_check(deadline):
    if deadline is not None:
        require(time.monotonic() < deadline, '4200-second whole-screen deadline')


def load_input(path, digest):
    value, tokens = L.load_input(path, digest)
    require(value['count'] == 32863 and value['batched_rows'] == 32862,
            'reuse the exact existing natural32863 input, without retokenization')
    return value, tokens


def arm_settings(arm):
    require(arm in ARMS, 'known arm')
    native = arm not in ARMS[:2]
    return dict(STRATA_SYCL_PREFILL_NATIVE16='bf16' if native else '0',
                STRATA_PREFILL_LAYER_MAJOR_R_DTYPE='bf16' if native else 'fp32',
                STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm' if native else 'own',
                STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',
                STRATA_SYCL_PREFILL_NATIVE16_QSA='1' if arm == 'bf16-vllm-qsa' else '0')


def parse_dense(raw, meta, tokens, recipe_sha, *, head_type=14):
    """Validate every dense target/loss and the exact native v2 completion schema."""
    rows = len(tokens) - 1
    expected_size = 64 + 12 * rows
    require(Path(raw).stat().st_size == expected_size < 4 << 20, 'dense NLL exact bytes')
    require(Path(meta).stat().st_size <= 4096, 'dense metadata cap')
    lines = Path(meta).read_bytes()
    require(lines.endswith(b'\n') and b'\r' not in lines, 'dense metadata line endings')
    lines = lines.decode('ascii').splitlines()
    expected = ['strata-dense-prefill-nll-v2', 'recipe_sha256=' + recipe_sha,
                'coverage=initial-prefill-all-n_input-minus-one',
                'format=little-endian-header64-target-u32-loss-f64',
                'projection=existing-single-row-lm-head', 'performance_cohort=excluded',
                'n_input=' + str(len(tokens)), 'vocab=' + str(VOCAB), 'rows=' + str(rows),
                'raw_bytes=' + str(expected_size)]
    require(len(lines) == 12 and lines[:10] == expected and
            lines[10].startswith('callback_ms=') and lines[11] == 'complete=1',
            'exact dense schema/recipe/coverage/completion tie')
    cost = float(lines[10].split('=', 1)[1])
    require(math.isfinite(cost) and cost >= 0, 'dense callback diagnostic cost finite')
    losses = []
    with Path(raw).open('rb') as stream:
        header = stream.read(64)
        require(header[:8] == b'STNLLV2\n' and header[40:] == bytes(24) and
                struct.unpack_from('<8I', header, 8) ==
                (2, 64, len(tokens), VOCAB, rows, tokens[0], head_type, 1),
                'exact dense LE header, native head and completion count')
        for position in range(rows):
            payload = stream.read(12)
            require(len(payload) == 12, 'dense row truncation')
            target, loss = struct.unpack('<Id', payload)
            require(target == tokens[position + 1] and math.isfinite(loss) and loss >= 0,
                    'dense aligned finite nonnegative NLL at ' + str(position))
            losses.append(loss)
        require(not stream.read(1), 'dense exact EOF')
    return dict(rows=rows, first_input_token=tokens[0], targets_exact=True, finite=True,
                position_rule='index i targets frozen conditioning ID i+1',
                mean_NLL=math.fsum(losses) / rows, all_target_NLLs=losses,
                sampled_PPL=exp_or_overflow(math.fsum(losses) / rows),
                callback_ms_diagnostic_only=cost, tolerance=NLL_TOLERANCE,
                native_recipe_sha256=recipe_sha)


def summarize_pairs(individual):
    summaries = {}
    for a, b in PAIRS:
        key = a + '-vs-' + b
        values = [row['pairs'][key] for row in individual]
        mean_delta = math.fsum(v['NLL_delta'] for v in values) / len(values)
        summaries[key] = dict(control=a, candidate=b, mean_NLL_delta=mean_delta,
            exp_mean_delta_sampled_PPL_ratio=exp_or_overflow(mean_delta),
            mean_absolute_NLL_delta=math.fsum(v['absolute_NLL_delta'] for v in values) / len(values),
            max_absolute_NLL_delta=max(v['absolute_NLL_delta'] for v in values),
            logit_max_absolute_delta=max(v['logit_max_absolute_delta'] for v in values),
            top1_agreements=sum(v['top1_agrees'] for v in values), positions=len(values),
            mean_KL_control_to_candidate=math.fsum(v['KL_control_to_candidate'] for v in values) / len(values),
            mean_L1=math.fsum(v['L1'] for v in values) / len(values))
    return summaries


def analyze_four(paths, dense, tokens, *, head_type=14, deadline=None):
    require(set(paths) == set(ARMS) == set(dense), 'four independent actual arms')
    individual = []
    with ExitStack() as stack:
        streams = {arm: open_capture(stack, *paths[arm], tokens, POSITIONS, VOCAB, head_type)
                   for arm in ARMS}
        for index, position in enumerate(POSITIONS):
            deadline_check(deadline)
            rows, stats = {}, {}
            target = tokens[position + 1]
            for arm in ARMS:
                rows[arm], stats[arm] = read_row(streams[arm][0], VOCAB, target)
                difference = stats[arm]['NLL'] - dense[arm]['all_target_NLLs'][position]
                require(abs(difference) <= NLL_TOLERANCE, 'independent FP64/dense NLL oracle: ' + arm)
                stats[arm].update(dense_NLL=dense[arm]['all_target_NLLs'][position],
                                  independent_minus_dense=difference, full_vocabulary_finite=True)
            require(stats['fp32-a']['row_sha256'] == stats['fp32-b']['row_sha256'],
                    'FP32 full-vocabulary control arrays must be byte-identical')
            metrics = {a + '-vs-' + b: pair_metrics(rows[a], rows[b], stats[a], stats[b], True)
                       for a, b in PAIRS}
            individual.append(dict(index=index, position=position, target=target, arms=stats, pairs=metrics))
        require(all(not stream.read(1) for stream, _ in streams.values()), 'exact EOF on four arms')
    mean = {arm: math.fsum(v['arms'][arm]['NLL'] for v in individual) / len(individual) for arm in ARMS}
    dense_differences = {}
    for a, b in PAIRS:
        delta = [y - x for x, y in zip(dense[a]['all_target_NLLs'], dense[b]['all_target_NLLs'])]
        require(all(math.isfinite(x) for x in delta), 'finite dense differences')
        average = math.fsum(delta) / len(delta)
        dense_differences[a + '-vs-' + b] = dict(rows=len(delta), mean_NLL_delta=average,
            mean_absolute_NLL_delta=math.fsum(map(abs, delta)) / len(delta),
            max_absolute_NLL_delta=max(map(abs, delta)), NLL_delta_by_position=delta,
            exp_mean_delta_PPL_ratio=exp_or_overflow(average))
    return dict(valid=True, positions=64, full_vocabulary_finite=True, FP32_controls_identical=True,
        FP64_oracle='Unchanged legacy Python float/math.fsum, max-shifted logsumexp; exact FP32 decoding',
        oracle_sha256=ORACLE_SHA, top1_tie_rule='lowest token ID; tie count retained',
        distribution_metrics=True, mean_NLL=mean, sampled_PPL={a: exp_or_overflow(n) for a, n in mean.items()},
        pairs=summarize_pairs(individual), individual_positions=individual, dense_NLL=dense,
        dense_differences=dense_differences, quality_accepted=False, model_quality_accepted=False,
        acceptance_threshold=None, exact_text_equality_required=False, interpretation=LIMITATION)


def compare_record(record, *, deadline=None):
    require(record['schema'] == SCHEMA and record['capture_complete'] is True and
            record['closure_validated'] is True and record['active'] is False, 'closed capture cohort')
    require(record['boot_unchanged'] is True and record['fresh_faults'] == [] and
            record['driver_dump_exists'] is False, 'fresh fault/boot/dump guards')
    require(record['declared_order'] == list(ARMS) and set(record['cases']) == set(ARMS), 'four ordered arms')
    require(all(L.command_closed(c) for c in record['commands']), 'every original command closed normally')
    require(record['native_byteorder'] == 'little' and record['FP32_controls_identical'] is True,
            'capture platform and prior control-array gate')
    for key in ('quality_accepted', 'model_quality_accepted', 'model_quality_qualified',
                'performance_qualified', 'physical_262144_qualified', 'restore_qualified'):
        require(record[key] is False, 'qualification remains false: ' + key)
    binding = record['binding']
    require(record['binding_sha256'] == canonical_sha(binding), 'common source/recipe binding')
    require(record['source_commit'] == binding['source_commit'] == SOURCE_COMMIT and
            record['source_pins'] == binding['source_pins'], 'current compiled source binding')
    require(record['controller'] == binding['controller'], 'true controller binding')
    for name in ('manifest', 'comparator', 'legacy_oracle'):
        check_artifact(record[name], cap=150 << 10)
    require(record['legacy_oracle']['sha256'] == ORACLE_SHA, 'old oracle pin')
    check_artifact(record['controller']['source'], cap=150 << 10)
    build = load_json(check_artifact(record['references']['build'], cap=16 << 20))
    smoke = load_json(check_artifact(record['references']['model_smoke'], cap=16 << 20))
    require(record['references']['build']['sha256'] == BUILD_SHA == binding['build_receipt_sha256'] and
            record['references']['model_smoke']['sha256'] == SMOKE_SHA == binding['diagnostic_environment_origin'],
            'integrated immutable build/model receipts')
    require(build['passed'] is True and smoke['passed'] is True and build['complete'] is True and
            smoke['complete'] is True and build['active'] is False and smoke['active'] is False,
            'failed/incomplete build or correctness can never qualify')
    require(build['source_pins'] == smoke['source_pins'] == binding['source_pins'] and
            build['source_commit'] == smoke['source_commit'] == SOURCE_COMMIT and
            build['binary_sha256'] == smoke['binary_sha256'] == binding['binary']['sha256'] and
            build['pins'] == binding['build_file_pins'], 'all compiled/build-source/binary pins')
    require(all(smoke[k] == binding[k] for k in ('runtime_files', 'model_identity',
                'pack_metadata_sha256', 'MTP_metadata_sha256')), 'model/runtime metadata origin')
    compilers = [v.split('=', 1)[1] for v in build['configure_argv'] if v.startswith('-DCMAKE_CXX_COMPILER=')]
    require(len(compilers) == 1 and compilers[0] in build['pins'] and
            binding['compiler']['path'] == build['pins'][compilers[0]]['realpath'] and
            binding['compiler']['sha256'] == build['pins'][compilers[0]]['sha256'], 'compiler/build receipt chain')
    check_artifact(record['references']['prior'], cap=16 << 20)
    prior = load_json(record['references']['prior']['path'])
    require(prior.get('passed') is True and prior.get('complete') is True and prior.get('active') is False and
            prior.get('fresh_faults') == [] and prior.get('driver_dump_exists') is False and
            prior.get('boot_id') == record['boot_id'] and prior.get('commands') and
            all(L.command_closed(c) for c in prior['commands']), 'explicit normally closed PASS prior')
    value, tokens = load_input(record['input_receipt']['path'], record['input_receipt']['sha256'])
    check_artifact(record['input_receipt'], cap=1 << 20)
    require(binding['input_receipt_sha256'] == record['input_receipt']['sha256'] and
            binding['input_ids_sha256'] == value['ids_le_u32_sha256'] and
            binding['input_file_sha256'] == value['file_sha256'] and
            binding['input_source_pins'] == value['source_pins'], 'exact thinking-on input/source binding')
    paths, dense, common_envs, indices = {}, {}, [], []
    raw_total = 0
    for arm in ARMS:
        deadline_check(deadline)
        case = record['cases'][arm]
        recipe_path = check_artifact(case['recipe'], cap=2 << 20)
        recipe = load_json(recipe_path)
        request_path = check_artifact(case['request'], cap=16 << 10)
        request = load_json(request_path)
        command = record['commands'][case['command_index']]
        indices.append(case['command_index'])
        env = dict(recipe['environment'], STRATA_PREFILL_NLL_RECIPE_SHA256=sha(recipe_path))
        require(command['label'] == 'screen-' + arm and command['argv'] == recipe['argv'] and
                command['environment'] == env == case['effective_environment'], 'actual owned recipe/environment')
        require(recipe['schema'] == SCHEMA and recipe['arm'] == arm and recipe['binding'] == binding and
                recipe['binding_sha256'] == record['binding_sha256'] and
                recipe['request'] == case['request'], 'arm source/request/recipe binding')
        require(case['complete'] is True and command['direct_identity']['ppid'] == record['controller']['identity']['pid'] and
                command['direct_identity']['sid'] == command['pid'] and
                command['direct_identity']['start_ticks'] > 0 and command.get('session_anchored'),
                'native direct PID/start/session owner')
        require(case['local_profile_initial_sha256'] == recipe['local_profile_initial_sha256'] ==
                binding['external_profile']['sha256'], 'identically reset local profile')
        require(command['argv'] == [binding['binary']['path'], *record['args']], 'identical four-arm arguments')
        require(command['argv'].count('--tokens-file') == 1 and
                sha(command['argv'][command['argv'].index('--tokens-file') + 1]) == value['file_sha256'],
                'actual private conditioning-file pin')
        for flag, expected in (('--max-context', '65536'), ('--kv-resident', '65536'),
                               ('--prefill', '8192'), ('--max-new', '1')):
            require(command['argv'].count(flag) == 1 and
                    command['argv'][command['argv'].index(flag) + 1] == expected, 'fixed CLI ' + flag)
        require('--no-token-graph' in command['argv'] and '--serve' not in command['argv'], 'standalone native')
        require(all(env.get(k) == v for k, v in arm_settings(arm).items()) and
                env['STRATA_PREFILL_LAYER_MAJOR_R_GPU'] == '32768' and env['STRATA_ROPE_TABLE'] == '1' and
                env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1' and env['EnableDirectSubmission'] == '0' and
                env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1', 'frozen four-arm native settings')
        common_envs.append({k: v for k, v in env.items() if k not in (*arm_settings(arm),
            'STRATA_PREFILL_LOGITS', 'STRATA_PREFILL_NLL', 'STRATA_PREFILL_NLL_RECIPE_SHA256', 'XDG_CACHE_HOME')})
        raw = check_artifact(case['raw'], cap=64 << 20)
        meta = check_artifact(case['metadata'], cap=8192)
        require(request == dict(schema='strata-selected-logits-request-v1', arm=arm,
            input_ids_sha256=value['ids_le_u32_sha256'], input_receipt_sha256=record['input_receipt']['sha256'],
            positions=POSITIONS, targets=[tokens[p + 1] for p in POSITIONS], vocab=VOCAB,
            head='native', head_ggml_type=14, batched=len(tokens) - 1, raw_path=str(raw),
            metadata_path=str(meta), expected_raw_bytes=RAW_BYTES,
            native_schema='strata-prefill-logits-v1'), 'selected schema/request/positions/target tie')
        require(str(raw) == env['STRATA_PREFILL_LOGITS'] and str(meta) == str(raw) + '.meta' and
                case['raw']['bytes'] == RAW_BYTES and
                env['STRATA_PREFILL_LOGITS_POSITIONS'] == ','.join(map(str, POSITIONS)), 'selected observer binding')
        nll = check_artifact(case['dense_raw'], cap=4 << 20)
        nll_meta = check_artifact(case['dense_metadata'], cap=4096)
        require(str(nll) == env['STRATA_PREFILL_NLL'] and str(nll_meta) == str(nll) + '.meta', 'dense output tie')
        dense[arm] = parse_dense(nll, nll_meta, tokens, sha(recipe_path))
        for kind in ('stdout', 'stderr'):
            path = check_artifact(case[kind], cap=64 << 20)
            require(command['logs'][path.name] == {k: case[kind][k] for k in ('bytes', 'sha256')}, 'owner log hash')
        raw_total += case['raw']['bytes'] + case['dense_raw']['bytes']
        paths[arm] = (raw, meta)
    require(indices == sorted(set(indices)) and all(v == common_envs[0] for v in common_envs),
            'FP32-a, FP32-b then candidates, identical non-arm environment')
    require(record['cases']['fp32-a']['raw']['sha256'] == record['cases']['fp32-b']['raw']['sha256'] and
            record['cases']['fp32-a']['dense_raw']['sha256'] == record['cases']['fp32-b']['dense_raw']['sha256'],
            'both control arrays byte-identical before candidates')
    require(raw_total < RAW_CAP, 'strict whole-cohort raw capture budget')
    result = analyze_four(paths, dense, tokens, head_type=record['head_ggml_type'], deadline=deadline)
    for case in record['cases'].values():
        for key, cap in (('raw', 64 << 20), ('metadata', 8192), ('dense_raw', 4 << 20), ('dense_metadata', 4096)):
            check_artifact(case[key], cap=cap)
    result.update(binding_sha256=record['binding_sha256'], raw_tensor_bytes=raw_total,
                  controls_gated_before_candidates=record['control_gate'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    require(not options.output.exists(), 'fresh comparison output; original receipts stay immutable')
    result = dict(valid=False, complete=False, GPU_executed=False, recipe_sha256=sha(__file__),
                  input_receipt_sha256=options.receipt_sha256, model_quality_accepted=False, quality_accepted=False)
    try:
        require(sha(options.receipt) == options.receipt_sha256, 'explicit immutable receipt pin')
        value = load_json(options.receipt)
        require(value.get('passed') is True and value.get('complete') is True, 'failed correctness never qualifies')
        result['comparison'] = compare_record(value, deadline=time.monotonic() + 4200)
        result.update(valid=True, complete=True)
    except Exception as error:
        result['error'] = repr(error)
    with options.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(valid=result['valid'], complete=result['complete'], model_quality_accepted=False)))
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
