#!/usr/bin/env python3
"""Selected-only FP64 numerical analysis. Repository LICENSE applies; Main runs it.

The SHA-pinned project oracle is unchanged. No backend/runtime imports, native
launches, dense/all-position observer, tolerance gate, retry or cleanup.
"""
import argparse
from contextlib import ExitStack
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys
import time

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
A = D.parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ORACLE = A / 'own-native16-quality-screen-v1/compare_screen.py'
ORACLE_SHA = 'dc4e56b916c014acc60a716bdc67bc1bd1256f53b9f1ef119093b16b5feef69d'
with ORACLE.open('rb') as stream:
    if hashlib.file_digest(stream, 'sha256').hexdigest() != ORACLE_SHA:
        raise RuntimeError('unchanged project FP64 oracle required')
spec = importlib.util.spec_from_file_location('selected_quality_legacy_oracle', ORACLE)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
require, sha, canonical_sha = L.require, L.sha, L.canonical_sha
load_json, artifact, check_artifact = L.load_json, L.artifact, L.check_artifact
open_capture, read_row, pair_metrics, exp_or_overflow = L.open_capture, L.read_row, L.pair_metrics, L.exp_or_overflow
VOCAB = 248320
SCHEMA = 'strata-vllm-selected-quality-v2'
SOURCE_COMMIT = '6354f0b4ff3731466795410e11ca9ef785c9d326'
BUILD_SHA = '990635731be1198706abeb682817fdab247f8f51d31bab971d3d4f4db53cb639'
BINARY_SHA = '85643f3926bb9d212945659cb7ddb264e101ca9b2d5fa8f19430e945140ad191'
HISTORICAL_SMOKE_SHA = '03b2bd388a897873bb2dc8c310ec7c0386e6b484828d1e633fb872cd059a14d3'
FAILURE_SHA = 'bf9d6f0036bff3d6c8f49a384ecc03d849453a41e3ab93b2d90fd4c9833a4f18'
INPUT_SHA = 'c0e89883e69aad815adc165ad199091c990dbcb8ce47275e155a2793056af0b1'
TOKEN_SHA = '6e5341d8e4898b4932b2e77d5c2fbc6f0ba304d1cf19ef258cb6a60bba279e95'
IDS_SHA = 'e4754da0c72e7495319ff1f0483a39032f0cffffac532a6d902b3aeb5d434f16'
PREPARER_SHA = 'b67e1b7b29c0d00cf895eac61f6e8aa21133191d9cfdb4b7303752b938e00fbf'
QUALIFICATION_KEYS = ('quality_accepted', 'model_quality_accepted', 'model_quality_qualified',
                      'performance_qualified', 'physical_262144_qualified', 'restore_qualified')
LIMITATION = ('Selected-only numerical mechanics: synthetic129/three positions or one natural32863 corpus/64 '
              'correlated positions. No all-position NLL, quality acceptance, general/task-quality, clean speed, '
              'physical262144 or RESTORE qualification. FP32 drift and exact equality are descriptive; '
              'there is no output equality or numerical adoption threshold.')


def deadline_check(deadline):
    if deadline is not None:
        require(time.monotonic() < deadline, 'finite selected-only work deadline')


def contract(mode):
    require(mode in ('smoke', 'screen'), 'explicit finite mode')
    smoke = mode == 'smoke'
    return dict(arms=['fp32', 'bf16-vllm', 'bf16-vllm-qsa'] if smoke else
                ['fp32-a', 'fp32-b', 'bf16-vllm', 'bf16-vllm-qsa'],
                positions=[0, 63, 127] if smoke else list(L.POSITIONS),
                count=129 if smoke else 32863, chunk=64 if smoke else 8192,
                context=4096 if smoke else 65536, resident=4096 if smoke else 65536,
                R_GPU=64 if smoke else 32768,
                raw_bytes=8 + (3 if smoke else 64) * VOCAB * 4)


def pairs_for(mode):
    controls = ['fp32'] if mode == 'smoke' else ['fp32-a', 'fp32-b']
    return ([] if mode == 'smoke' else [('fp32-a', 'fp32-b')]) + [
        (a, b) for a in controls for b in ('bf16-vllm', 'bf16-vllm-qsa')]


def arm_settings(arm):
    require(arm in ('fp32', 'fp32-a', 'fp32-b', 'bf16-vllm', 'bf16-vllm-qsa'), 'known arm')
    native = arm.startswith('bf16')
    return dict(STRATA_SYCL_PREFILL_NATIVE16='bf16' if native else '0',
                STRATA_PREFILL_LAYER_MAJOR_R_DTYPE='bf16' if native else 'fp32',
                STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm' if native else 'own',
                STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',
                STRATA_SYCL_PREFILL_NATIVE16_QSA='1' if arm == 'bf16-vllm-qsa' else '0')


def no_nll(env):
    require(not any(k.startswith('STRATA_PREFILL_NLL') for k in env),
            'all dense-NLL environment keys absent, including recipe digest')


def parse_tokens(path, count, file_sha, ids_sha):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 1 << 20 and sha(path) == file_sha, 'exact bounded token file')
    text = path.read_text(encoding='ascii')
    require(re.fullmatch(r'\s*[0-9]+(?:[\s,]+[0-9]+)*\s*', text), 'strict unsigned token file')
    tokens = [int(v) for v in re.split(r'[\s,]+', text.strip())]
    require(len(tokens) == count and all(0 <= v < VOCAB for v in tokens), 'exact count and vocabulary range')
    require(hashlib.sha256(struct.pack('<' + str(count) + 'I', *tokens)).hexdigest() == ids_sha, 'exact LE-u32 IDs')
    return tokens


def load_natural(receipt_path, token_path):
    require(sha(receipt_path) == INPUT_SHA, 'original R837 input receipt unchanged')
    value = load_json(receipt_path)
    require(value['complete'] is True and value['GPU_executed'] is False and
            value['thinking_requested'] is True and value['roundtrip_exact'] is True and
            value['repeated_padding'] is False and value['count'] == 32863 and value['batched_rows'] == 32862 and
            value['sampled_positions'] == L.POSITIONS and value['recipe_sha256'] == PREPARER_SHA and
            value['file_sha256'] == TOKEN_SHA and value['ids_le_u32_sha256'] == IDS_SHA, 'unchanged natural recipe/conditioning')
    require(value['source_pins'], 'tokenizer/template/document pins present')
    for path, digest in value['source_pins'].items():
        require(sha(path) == digest, 'original tokenizer/template/input source: ' + path)
    # Only the retained R837 token copy is consumed; its original receipt is not
    # rewritten to point elsewhere and an absent input.json is never fabricated.
    return value, parse_tokens(token_path, 32863, TOKEN_SHA, IDS_SHA)


def closed_receipt(value, *, boot=None, gpu=False):
    require(value.get('passed') is True and value.get('complete') is True and value.get('active') is False and
            value.get('ownership_closed') is True and value.get('commands') and
            all(L.command_closed(c) for c in value['commands']), 'original normal PASS/ownership required')
    require(value.get('fresh_faults') == [] and value.get('driver_dump_exists') is False and
            not value.get('error') and not value.get('closure_errors') and not value.get('final_fault_check_error'),
            'no failed/faulted/unknown closure evidence')
    if boot is not None:
        require(value.get('boot_id') == boot and value.get('boot_unchanged') is True, 'normally closed sameboot health')
    if gpu:
        require(value.get('GPU_executed') is True, 'actual GPU health work required')
    for c in value['commands']:
        require(c.get('session_anchored') and c.get('direct_identity') and c.get('owners'), 'owned PID/start/session evidence')


def health_contract(value, build, current_boot, failure):
    closed_receipt(value, boot=current_boot, gpu=True)
    require(value.get('source_commit') == SOURCE_COMMIT and value.get('source_pins') == build['source_pins'],
            'health uses exact compiled6354 source pins')
    require(datetime.datetime.fromisoformat(value['finished_utc']) > datetime.datetime.fromisoformat(failure['finished_utc']),
            'explicit Main-selected health must postdate latest preserved failure')
    if value.get('schema') == 'strata-postfault-allocation-health-v1':
        product = build['products']['sycl_allocation_teardown_test']
        require(value.get('original_failure_sha256') == FAILURE_SHA and value.get('original_failure_status_preserved') is True and
                value.get('build_receipt_sha256') == BUILD_SHA and value.get('binary_pin') == product and
                value.get('binary') == product['path'] and Path(value['binary']).name == 'sycl_allocation_teardown_test' and
                value.get('binary_sha256') == product['sha256'], 'actual tiny allocation binary; never pretend this is engine quality')
        native = [c for c in value['commands'] if c.get('label') == 'allocation-early']
        require(len(native) == 1 and native[0]['argv'] == [product['path'], '--early'] and
                value.get('operation_proof', {}).get('bytes') == 64 and
                value['operation_proof'].get('normal_process_exit') is True and
                value['operation_proof'].get('operations') == ['urUSMDeviceAlloc', 'urEnqueueUSMFill', 'urUSMFree'],
                'one64byte allocation/fill/drain/free operation and normal exit')
        require(value.get('original_owners_absence') and all(v.get('original_absent') is True for v in value['original_owners_absence']),
                'postfault original PID/start absence recorded')
        no_nll(native[0]['environment'])
        require(all(value.get(k) is False for k in ('quality_accepted', 'model_quality_qualified', 'performance_qualified',
                    'physical_262144_qualified', 'restore_qualified')), 'allocation health retains narrow scope')
        return 'tiny-adapter-allocation/teardown-health-only; no engine/model/numerical qualification'
    require(value.get('schema') == SCHEMA and value.get('mode') == 'smoke' and value.get('selected_only') is True and
            value.get('capture_complete') is True and value.get('closure_validated') is True and
            value.get('binary_sha256') == BINARY_SHA and value.get('references', {}).get('build', {}).get('sha256') == BUILD_SHA and
            value.get('comparison', {}).get('valid') is True and value.get('declared_order') == contract('smoke')['arms'] and
            set(value.get('cases', {})) == set(contract('smoke')['arms']), 'only subsequent normally closed same-source selected smoke')
    require(all(value.get(k) is False for k in QUALIFICATION_KEYS), 'selected smoke retains narrow scope')
    return 'same-source selected129/three-position model-mechanics-only; no model-quality qualification'


def request_for(arm, binding, spec, raw):
    return dict(schema='strata-selected-logits-request-v1', arm=arm, input_ids_sha256=binding['input_ids_sha256'],
                input_receipt_sha256=binding['input_receipt_sha256'], positions=spec['positions'],
                targets=[binding['tokens'][p + 1] for p in spec['positions']], vocab=VOCAB, head='native', head_ggml_type=14,
                batched=spec['count'] - 1, raw_path=str(raw), metadata_path=str(raw) + '.meta',
                expected_raw_bytes=spec['raw_bytes'], native_schema='strata-prefill-logits-v1')


def analyze(paths, tokens, mode, *, deadline=None):
    cfg = contract(mode)
    require(set(paths) == set(cfg['arms']), 'all separate original native arms')
    individual = []
    with ExitStack() as stack:
        streams = {a: open_capture(stack, *paths[a], tokens, cfg['positions'], VOCAB, 14) for a in cfg['arms']}
        for index, position in enumerate(cfg['positions']):
            deadline_check(deadline)
            rows, stats = {}, {}
            for arm in cfg['arms']:
                rows[arm], stats[arm] = read_row(streams[arm][0], VOCAB, tokens[position + 1])
                stats[arm]['full_vocabulary_finite'] = True
            metrics = {a + '-vs-' + b: pair_metrics(rows[a], rows[b], stats[a], stats[b], True) for a, b in pairs_for(mode)}
            individual.append(dict(index=index, position=position, target=tokens[position + 1], arms=stats, pairs=metrics))
        require(all(not s.read(1) for s, _ in streams.values()), 'exact EOF on every full-vocabulary capture')
    summaries = {}
    for a, b in pairs_for(mode):
        key = a + '-vs-' + b
        vals = [row['pairs'][key] for row in individual]
        delta = math.fsum(v['NLL_delta'] for v in vals) / len(vals)
        summaries[key] = dict(control=a, candidate=b, positions=len(vals), mean_NLL_delta=delta,
            exp_mean_delta_sampled_PPL_ratio=exp_or_overflow(delta),
            mean_absolute_NLL_delta=math.fsum(v['absolute_NLL_delta'] for v in vals) / len(vals),
            max_absolute_NLL_delta=max(v['absolute_NLL_delta'] for v in vals),
            logit_max_absolute_delta=max(v['logit_max_absolute_delta'] for v in vals),
            top1_agreements=sum(v['top1_agrees'] for v in vals),
            mean_KL_control_to_candidate=math.fsum(v['KL_control_to_candidate'] for v in vals) / len(vals),
            mean_L1=math.fsum(v['L1'] for v in vals) / len(vals))
    means = {a: math.fsum(r['arms'][a]['NLL'] for r in individual) / len(individual) for a in cfg['arms']}
    return dict(valid=True, positions=len(individual), full_vocabulary_finite=True,
        FP64_oracle='Unchanged project Python float/math.fsum max-shifted logsumexp; exact FP32 payload decoding',
        oracle_sha256=ORACLE_SHA, top1_tie_rule='lowest token ID; tie count retained',
        mean_NLL=means, sampled_PPL={a: exp_or_overflow(n) for a, n in means.items()},
        pairs=summaries, individual_positions=individual, distribution_metrics=True,
        FP32_controls_identical=(all(r['arms']['fp32-a']['row_sha256'] == r['arms']['fp32-b']['row_sha256'] for r in individual)
                                 if mode == 'screen' else None),
        control_equality_is_descriptive=True, acceptance_threshold=None, exact_text_equality_required=False,
        dense_allposition_NLL_collected=False, interpretation=LIMITATION, **{k: False for k in QUALIFICATION_KEYS})


def compare_record(record, *, deadline=None):
    cfg = contract(record['mode'])
    require(record['schema'] == SCHEMA and record['selected_only'] is True and record['capture_complete'] is True and
            record['closure_validated'] is True and record['active'] is False and record['ownership_closed'] is True,
            'normally closed selected-only capture cohort')
    require(record['fresh_faults'] == [] and record['boot_unchanged'] is True and record['driver_dump_exists'] is False and
            all(L.command_closed(c) for c in record['commands']), 'all original command/fault/boot outcomes')
    require(record['declared_order'] == cfg['arms'] and set(record['cases']) == set(cfg['arms']) and
            record['native_byteorder'] == 'little' and all(record[k] is False for k in QUALIFICATION_KEYS), 'mode/qualification scope')
    binding = record['binding']
    require(record['binding_sha256'] == canonical_sha(binding) and binding['controller'] == record['controller'] and
            binding['source_commit'] == record['source_commit'] == SOURCE_COMMIT and
            binding['source_pins'] == record['source_pins'], 'controller/source/common recipe binding')
    for name in ('manifest', 'comparator', 'legacy_oracle'):
        check_artifact(record[name], cap=256 << 10)
    check_artifact(record['controller']['source'], cap=256 << 10)
    require(record['legacy_oracle']['sha256'] == ORACLE_SHA, 'frozen FP64 oracle')
    refs = record['references']
    for value in refs.values():
        check_artifact(value, cap=16 << 20)
    build, historical, failure, prior = (load_json(refs[k]['path']) for k in ('build', 'historical_recipe', 'original_failure', 'health'))
    require(refs['build']['sha256'] == BUILD_SHA and refs['historical_recipe']['sha256'] == HISTORICAL_SMOKE_SHA and
            refs['original_failure']['sha256'] == FAILURE_SHA and failure['passed'] is False and failure['complete'] is False,
            'historical failure is preserved; historical PASS is recipe identity only')
    closed_receipt(build)
    closed_receipt(historical)
    require(build['source_commit'] == historical['source_commit'] == SOURCE_COMMIT and
            build['source_pins'] == historical['source_pins'] == binding['source_pins'] and len(build['source_pins']) == 432 and
            build['pins'] == binding['build_file_pins'] and len(build['pins']) == 314 and
            build['binary_sha256'] == historical['binary_sha256'] == binding['binary']['sha256'] == BINARY_SHA and
            binding['build_receipt_sha256'] == BUILD_SHA, 'complete source432/build314/engine6354 identities')
    require(all(binding[k] == historical[k] for k in ('runtime_files', 'model_identity', 'pack_metadata_sha256', 'MTP_metadata_sha256')),
            'unchanged runtime/model metadata/stat origin')
    compilers = [v.split('=', 1)[1] for v in build['configure_argv'] if v.startswith('-DCMAKE_CXX_COMPILER=')]
    require(len(compilers) == 1 and binding['compiler']['path'] == build['pins'][compilers[0]]['realpath'] and
            binding['compiler']['sha256'] == build['pins'][compilers[0]]['sha256'], 'actual compiled toolchain pin')
    require(record['health_scope'] == health_contract(prior, build, record['boot_id'], failure), 'explicit narrow health scope')
    tokens = parse_tokens(check_artifact(record['input_artifact'], cap=1 << 20), cfg['count'],
                          binding['input_file_sha256'], binding['input_ids_sha256'])
    require(tokens == binding['tokens'] and record['input_receipt']['sha256'] == binding['input_receipt_sha256'], 'exact conditioning binding')
    check_artifact(record['input_receipt'], cap=1 << 20)
    if record['mode'] == 'screen':
        natural, natural_tokens = load_natural(record['input_receipt']['path'], record['input_artifact']['path'])
        require(tokens == natural_tokens and binding['input_source_pins'] == natural['source_pins'], 'R837 tokenizer/template/source pins')
        require(refs['canonical_tokens']['sha256'] == TOKEN_SHA, 'original R837 canonical capture')
    else:
        require(tokens == list(range(129)) and binding['input_source_pins'] == {}, 'original synthetic129 mechanics')
    require(re.fullmatch(r'v[1-9][0-9]{0,3}', record['run_id']), 'finite canonical run ID')
    root = B / ('main-vllm-selected-' + record['mode'] + '-' + record['run_id'])
    require(record['input_artifact']['path'] == str(root / 'tokens.txt'), 'canonical private conditioning path')
    expected_args = list(historical['args'])
    for flag, value in (('--max-context', cfg['context']), ('--kv-resident', cfg['resident']), ('--prefill', cfg['chunk']),
                        ('--max-new', 1), ('--tokens-file', root / 'tokens.txt'), ('--expert-profile', root / 'expert-profile.bin')):
        require(expected_args.count(flag) == 1, 'original argv flag: ' + flag)
        expected_args[expected_args.index(flag) + 1] = str(value)
    require(record['args'] == expected_args, 'entire original argv plus only documented placement/input changes')
    paths, common, indices = {}, [], []
    for arm in cfg['arms']:
        deadline_check(deadline)
        case = record['cases'][arm]
        recipe_path = check_artifact(case['recipe'], cap=2 << 20)
        recipe = load_json(recipe_path)
        request = load_json(check_artifact(case['request'], cap=16 << 10))
        command = record['commands'][case['command_index']]
        env = recipe['environment']
        no_nll(env)
        require(case['status'] == 'complete' and case['complete'] is True and command['label'] == record['mode'] + '-' + arm and
                command['argv'] == recipe['argv'] == [binding['binary']['path'], *record['args']] and
                command['environment'] == env == case['effective_environment'] and
                recipe['schema'] == SCHEMA and recipe['arm'] == arm and recipe['mode'] == record['mode'] and
                recipe['binding'] == binding and recipe['binding_sha256'] == record['binding_sha256'] and
                recipe['request'] == case['request'], 'actual owned command/request/recipe/environment binding')
        require(command['direct_identity']['ppid'] == record['controller']['identity']['pid'] and
                command['direct_identity']['sid'] == command['pid'] and command['direct_identity']['start_ticks'] > 0 and
                command['session_anchored'] is True and case['post_exit_PID_start_absence'], 'actual direct native ownership')
        require(all(v['original_absent'] is True for v in case['post_exit_PID_start_absence']), 'post-exit PID/start absence')
        require(case['local_profile_initial_sha256'] == recipe['local_profile_initial_sha256'] == binding['external_profile']['sha256'],
                'identical private expert profile input per process')
        require(command['argv'].count('--tokens-file') == 1 and
                command['argv'][command['argv'].index('--tokens-file') + 1] == record['input_artifact']['path'], 'actual private conditioning path')
        for flag, expected in (('--max-context', cfg['context']), ('--kv-resident', cfg['resident']),
                               ('--prefill', cfg['chunk']), ('--max-new', 1), ('--spec', 4)):
            require(command['argv'].count(flag) == 1 and command['argv'][command['argv'].index(flag) + 1] == str(expected), 'fixed CLI ' + flag)
        require('--no-token-graph' in command['argv'] and '--serve' not in command['argv'] and
                all(env.get(k) == v for k, v in arm_settings(arm).items()), 'explicit native16/backend/QSA per mode')
        for key, expected in (('STRATA_PREFILL_LAYER_MAJOR_R_GPU', str(cfg['R_GPU'])), ('STRATA_ROPE_TABLE', '1'),
                ('STRATA_SYCL_PREFILL_SAME_QUEUE', '1'), ('EnableDirectSubmission', '0'),
                ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', '1'), ('SYCL_CACHE_PERSISTENT', '0'),
                ('STRATA_PREFILL_NATIVE_DENSE_BANK', '1'), ('STRATA_PREFILL_CANONICAL_DENSE_BANK', '1'),
                ('STRATA_SYCL_PREFILL_SCRATCH_ALIAS', '1'), ('STRATA_TRACE', '1')):
            require(env.get(key) == expected, 'fixed safe diagnostic environment: ' + key)
        require(Path(env['UR_ADAPTERS_FORCE_LOAD']).name == 'libur_adapter_level_zero_v2.so.0', 'forced URv2 adapter')
        tracing = bool(record['trace_first_smoke'] and arm == 'fp32' and record['mode'] == 'smoke')
        require(env.get('UR_ENABLE_LAYERS') == ('UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if tracing else 'UR_LAYER_PARAMETER_VALIDATION'),
                'bounded optional first synthetic trace only')
        require(env.get('UR_LOG_TRACING') == ('level:info;flush:info;output:stderr' if tracing else None), 'no long full API trace')
        common.append({k: v for k, v in env.items() if k not in (*arm_settings(arm), 'STRATA_PREFILL_LOGITS',
            'XDG_CACHE_HOME', 'UR_ENABLE_LAYERS', 'UR_LOG_TRACING')})
        raw, meta = check_artifact(case['raw'], cap=64 << 20), check_artifact(case['metadata'], cap=8192)
        require(raw == root / arm / 'selected.bin', 'fresh canonical per-arm raw output')
        expected_env = {k: v for k, v in historical['environment'].items() if not k.startswith('STRATA_PREFILL_NLL')}
        expected_env.update(arm_settings(arm), STRATA_PREFILL_LAYER_MAJOR_R_GPU=str(cfg['R_GPU']),
            XDG_CACHE_HOME=str(root / arm / 'cache'), STRATA_PREFILL_LOGITS=str(raw),
            STRATA_PREFILL_LOGITS_POSITIONS=','.join(map(str, cfg['positions'])))
        if tracing:
            expected_env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:stderr')
        require(env == expected_env, 'entire frozen original environment plus documented arm/observer/placement/cache/trace changes')
        require(case['raw']['bytes'] == cfg['raw_bytes'] and request == request_for(arm, binding, cfg, raw) and
                str(raw) == env['STRATA_PREFILL_LOGITS'] and str(meta) == str(raw) + '.meta' and
                env['STRATA_PREFILL_LOGITS_POSITIONS'] == ','.join(map(str, cfg['positions'])), 'exact selected request/header/shape binding')
        for key in ('stdout', 'stderr'):
            path = check_artifact(case[key], cap=64 << 20)
            require(command['logs'][path.name] == {k: case[key][k] for k in ('bytes', 'sha256')}, 'original owner log fingerprints')
        paths[arm] = (raw, meta)
        indices.append(case['command_index'])
    require(indices == sorted(set(indices)) and all(v == common[0] for v in common), 'fresh ordered arms with matched common environment')
    result = analyze(paths, tokens, record['mode'], deadline=deadline)
    result.update(binding_sha256=record['binding_sha256'], raw_tensor_bytes=sum(c['raw']['bytes'] for c in record['cases'].values()))
    require(result['raw_tensor_bytes'] <= 256 << 20, 'tensor budget')
    for case in record['cases'].values():
        check_artifact(case['raw'], cap=64 << 20)
        check_artifact(case['metadata'], cap=8192)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), 'fresh host comparison output')
    result = dict(valid=False, complete=False, GPU_executed=False, comparator_sha256=sha(__file__),
                  input_receipt_sha256=args.receipt_sha256, **{k: False for k in QUALIFICATION_KEYS})
    try:
        require(sha(args.receipt) == args.receipt_sha256, 'explicit original receipt hash')
        record = load_json(args.receipt)
        require(record.get('passed') is True and record.get('complete') is True, 'failed receipt remains failed')
        result['comparison'] = compare_record(record, deadline=time.monotonic() + 1800)
        result.update(valid=True, complete=True)
    except Exception as error:
        result['error'] = repr(error)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(valid=result['valid'], complete=result['complete'], quality_accepted=False)))
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
