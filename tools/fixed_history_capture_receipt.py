# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Read-only, fail-closed adapter for closed standalone capture evidence.

No launch, model tensor read, timing, or producer-authenticity assertion.
The current producer registry is deliberately empty: schema fixtures cannot
qualify a production capture. See docs/FIXED_HISTORY_CAPTURE_RECEIPT.md.
"""
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

from fixed_history_logits import (ContractError, digest, ids_hash, integer,
                                  load_json, pinned, require)

OWNER_SHA256 = 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
PROVIDER = 'standalone-native-segmented-spec-follow-dump-logits'
SEGMENTED = ('strata verify: queue-completed host/device handoff: each window '
             'runs as segments the host launches')
# Reviewed runtime producers must be registered by a separate reviewed change.
# A controller name or passed:true in a sidecar does not authorize qualification.
REVIEWED_PRODUCERS: frozenset[str] = frozenset()
MAX_JSON_BYTES = 16 << 20
MAX_LOG_BYTES = 64 << 20
MAX_CAPTURE_BYTES = 1 << 30
MAX_VOCAB = 2 << 20


def fields(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected.split()), label + ' fields')


def sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'SHA256 syntax')
    return value


def bounded_json(path):
    require(path.stat().st_size <= MAX_JSON_BYTES, 'JSON byte budget')
    return load_json(path)


def timestamp(value):
    require(isinstance(value, str), 'timestamp type')
    parsed = datetime.datetime.fromisoformat(value)
    require(parsed.tzinfo is not None and parsed.utcoffset() == datetime.timedelta(0), 'UTC timestamp')
    return parsed


def read_text(path):
    require(path.stat().st_size <= MAX_LOG_BYTES, 'log byte budget')
    return path.read_text(encoding='utf-8')


def ids(text):
    require(isinstance(text, str) and re.fullmatch(r'\s*\d+(?:[\s,]+\d+)*\s*', text), 'token text')
    return [int(item) for item in re.split(r'[\s,]+', text.strip())]


def options(argv):
    """Conservative supported route: every option has explicit arity/value."""
    require(isinstance(argv, list) and all(isinstance(x, str) and x for x in argv), 'argv')
    require(len(argv) >= 3 and argv[1] == 'generate', 'standalone command')
    switches = {'--greedy', '--native-qsa', '--native-qsa-indexer', '--native-rope',
                '--native-ple-postops'}
    values = {'--pack', '--native', '--tokens-file', '--spec-follow', '--dump-logits',
              '--logits-stride', '--max-new', '--max-context', '--spec', '--prefill',
              '--suffix-draft', '--lookup-chain', '--adapt-every', '--expert-profile',
              '--expert-cache', '--threads', '--seed', '--temperature', '--top-k',
              '--top-p', '--min-p'}
    result = {}
    index = 2
    while index < len(argv):
        key = argv[index]
        require(key in switches | values and key not in result, 'unsupported/duplicate option: ' + key)
        if key in switches:
            result[key] = True
            index += 1
        else:
            require(index + 1 < len(argv) and not argv[index + 1].startswith('--'), 'option value')
            result[key] = argv[index + 1]
            index += 2
    return result


def validate_common(common, budget):
    fields(common, 'P N vocab max_context endian stride position_convention policy prompt_ids '
           'follow_ids prompt_ids_sha256 follow_ids_sha256 tokenizer model', 'common')
    p, n, vocab = [common[key] for key in ('P', 'N', 'vocab')]
    require(all(integer(x, 1) for x in (p, n, vocab, common['max_context'])), 'dimensions')
    require(vocab >= 2 and p + n - 1 <= common['max_context'], 'context')
    require(common['endian'] in ('little', 'big') and type(common['stride']) is int
            and common['stride'] == 1, 'layout')
    require(common['position_convention'] == 'P-1+i predicts follow[i]', 'position convention')
    require(common['policy'] == {'bos': 'prompt-explicit', 'eos': 'score-all-stop-off',
                                 'mask': 'all-follow'}, 'policy')
    for key, count in (('prompt_ids', p), ('follow_ids', n)):
        sequence = common[key]
        require(isinstance(sequence, list) and len(sequence) == count and all(
            integer(x) and x < vocab for x in sequence), 'IDs')
        require(ids_hash(sequence) == common[key + '_sha256'], 'history SHA')
    fields(budget, 'max_vocab max_rows max_capture_bytes', 'budget')
    require(all(integer(budget[k], 1) for k in budget), 'budget integers')
    require(vocab <= budget['max_vocab'] and n <= budget['max_rows']
            and 8 + 4 * vocab * n <= budget['max_capture_bytes'], 'capture budget')
    require(budget['max_capture_bytes'] <= MAX_CAPTURE_BYTES and vocab <= MAX_VOCAB,
            'adapter hard memory/byte limit')


def adapt(sidecar_path):
    sidecar_path = Path(sidecar_path).resolve()
    root = sidecar_path.parent
    initial_sidecar_sha = digest(sidecar_path)
    sidecar = bounded_json(sidecar_path)
    fields(sidecar, 'schema producer evidence_kind common budget consumer retire_after owner_source '
           'owner_receipt command_index artifacts identities environment fault close', 'sidecar')
    require(sidecar['schema'] == 'strata.fixed-history-production-sidecar.v1', 'sidecar schema')
    require(sidecar['evidence_kind'] in ('production', 'synthetic-fixture'), 'evidence kind')
    require(isinstance(sidecar['producer'], str) and bool(sidecar['producer']), 'producer')
    common = sidecar['common']
    validate_common(common, sidecar['budget'])
    for key in ('consumer', 'retire_after'):
        require(isinstance(sidecar[key], str) and bool(sidecar[key]), 'retention ' + key)
    tracked = {sidecar_path: initial_sidecar_sha}

    def pin(entry):
        path = pinned(root, entry)
        tracked[path] = entry['sha256']
        return path

    owner_source = pin(sidecar['owner_source'])
    require(digest(owner_source) == OWNER_SHA256, 'unreviewed owner source')
    owner_path = pin(sidecar['owner_receipt'])
    owner = bounded_json(owner_path)
    require(isinstance(owner, dict) and isinstance(owner.get('commands'), list), 'owner commands')
    index = sidecar['command_index']
    require(integer(index) and index < len(owner['commands']), 'command index')
    command = owner['commands'][index]
    require(isinstance(command, dict), 'command object')
    for key in ('observation_complete', 'direct_child_reaped', 'session_empty', 'session_anchored'):
        require(command.get(key) is True, 'owner closure: ' + key)
    require(command.get('ownership_status') == 'observed', 'owner observation')
    require(command.get('errors') == [] and command.get('cleanup') == []
            and not command.get('error') and command.get('survivors') == [], 'owner error/cleanup/survivors')
    require(type(command.get('exit_code')) is int and command['exit_code'] == 0, 'exit code')
    begin, end = timestamp(command['started_utc']), timestamp(command['finished_utc'])
    require(begin < end, 'command interval')
    identity = command['direct_identity']
    require(integer(identity['pid'], 1) and integer(identity['start_ticks'], 1)
            and command['pid'] == identity['pid'] and identity['sid'] == identity['pid'], 'owner identity')
    require(any(all(entry.get(k) == identity[k] for k in ('pid', 'start_ticks', 'sid', 'pgid'))
                for entry in command['owners']), 'owner identity registration')
    argv = command['argv']
    opts = options(argv)
    artifacts = sidecar['artifacts']
    fields(artifacts, 'capture binary source runtime generator verifier prompt follow input_manifest '
           'stdout stderr provider_evidence', 'artifacts')
    require((root / artifacts['capture']['path']).stat().st_size
            == 8 + 4 * common['vocab'] * common['N'], 'capture size before hashing')
    paths = {key: pin(value) for key, value in artifacts.items()}
    require(Path(command['cwd']).is_absolute(), 'absolute command cwd')
    cwd = Path(command['cwd'])

    def actual(value):
        return (cwd / value).resolve()

    require(actual(argv[0]) == paths['binary'].resolve(), 'binary argv identity')
    for option, key in (('--tokens-file', 'prompt'), ('--spec-follow', 'follow'),
                        ('--dump-logits', 'capture')):
        require(option in opts and actual(opts[option]) == paths[key].resolve(), option + ' identity')
    expected = {'--max-new': common['N'], '--max-context': common['max_context'],
                '--logits-stride': 1, '--suffix-draft': 0, '--lookup-chain': 0, '--adapt-every': 0}
    for option, value in expected.items():
        require(opts.get(option) == str(value), 'route value: ' + option)
    require('--spec' in opts and re.fullmatch('[0-9]+', opts['--spec'])
            and int(opts['--spec']) >= 2 and common['N'] >= 2, 'verifier coverage')
    require('--native' in opts and '--pack' in opts and '--expert-profile' in opts
            and '--expert-cache' in opts and '--prefill' in opts, 'native provider options')
    require(ids(read_text(paths['prompt'])) == common['prompt_ids']
            and ids(read_text(paths['follow'])) == common['follow_ids'], 'actual token files')
    require(bounded_json(paths['input_manifest']) == common, 'input manifest identity')
    # This model pin is a small identity manifest, never the model tensor binary.
    require((root / common['model']['path']).stat().st_size <= MAX_JSON_BYTES,
            'model identity manifest size; tensors are not accepted')
    model_path = pin(common['model'])
    pin(common['tokenizer'])
    effective = bounded_json(pin(sidecar['environment']))
    require(isinstance(effective, dict) and
            all(isinstance(k, str) and isinstance(v, str) for k, v in effective.items()),
            'effective environment')
    require(command['environment'] == effective, 'effective environment mismatch')
    require(effective.get('STRATA_VERIFY_SEGMENTED') == '1'
            and effective.get('STRATA_VERIFY_DEVICE_PLAN') == '0', 'segmented environment')
    forbidden_env = ('LD_PRELOAD', 'STRATA_FULL_CONTEXT_PROBE', 'STRATA_HW_BENCH')
    require(not any(effective.get(k) for k in forbidden_env), 'unsupported environment route')
    identities = bounded_json(pin(sidecar['identities']))
    fields(identities, 'schema command_sha256 environment_sha256 artifacts model_files pack_path '
           'native_path profile_path boot_before boot_after started_utc finished_utc', 'identities')
    require(identities['schema'] == 'strata.capture-identities.v1', 'identities schema')
    canonical = json.dumps(argv, separators=(',', ':')).encode()
    require(identities['command_sha256'] == hashlib.sha256(canonical).hexdigest()
            and identities['environment_sha256'] == sidecar['environment']['sha256'], 'command/env hash')
    require(identities['started_utc'] == command['started_utc']
            and identities['finished_utc'] == command['finished_utc'], 'identity interval')
    require(isinstance(identities['boot_before'], str) and bool(identities['boot_before'])
            and identities['boot_before'] == identities['boot_after'], 'boot identity')
    require(set(identities['artifacts']) == set(artifacts), 'artifact coverage')
    for key, entry in identities['artifacts'].items():
        fields(entry, 'pre_sha256 post_sha256', 'pre/post artifact')
        require(entry['post_sha256'] == artifacts[key]['sha256'], 'post artifact: ' + key)
        expected_pre = None if key in ('capture', 'stdout', 'stderr') else artifacts[key]['sha256']
        require(entry['pre_sha256'] == expected_pre, 'changed/preexisting artifact: ' + key)
    require(isinstance(identities['model_files'], list) and bool(identities['model_files']), 'model identities')
    model_manifest = bounded_json(model_path)
    fields(model_manifest, 'schema files', 'model manifest')
    require(model_manifest['schema'] == 'strata.model-identity.v1', 'model manifest schema')
    require(model_manifest['files'] == identities['model_files'], 'model pre/post binding')
    model_paths = set()
    for entry in identities['model_files']:
        fields(entry, 'path bytes pre_sha256 post_sha256', 'model file')
        require(isinstance(entry['path'], str) and Path(entry['path']).is_absolute()
                and entry['path'] not in model_paths and integer(entry['bytes'], 1), 'model file identity')
        model_paths.add(entry['path'])
        require(sha(entry['pre_sha256']) == sha(entry['post_sha256']), 'changed model')
    for option, key in (('--pack', 'pack_path'), ('--native', 'native_path'),
                        ('--expert-profile', 'profile_path')):
        require(str(actual(opts[option])) == identities[key], 'native path binding')
    require(identities['native_path'] in model_paths and identities['profile_path'] in model_paths,
            'native/profile pre/post coverage')
    provider = bounded_json(paths['provider_evidence'])
    fields(provider, 'schema provider generator_sha256 verifier_sha256 binary_sha256 runtime_sha256 '
           'source_sha256 native_pack device_plan segmented clip_scope', 'provider')
    require(provider['schema'] == 'strata.capture-provider.v1' and provider['provider'] == PROVIDER,
            'provider schema/route')
    for key in ('generator', 'verifier', 'binary', 'runtime', 'source'):
        require(provider[key + '_sha256'] == artifacts[key]['sha256'], 'provider binding')
    require(provider['native_pack'] is True and provider['segmented'] is True
            and type(provider['device_plan']) is int and provider['device_plan'] == 0,
            'provider route')
    require(provider['clip_scope'] == 'full-vocabulary-all-follow-no-position-clipping', 'clip scope')
    verifier_text = read_text(paths['verifier'])
    require('strata verify: queue-completed host/device handoff: each window runs as '
            in verifier_text and 'segments the host launches' in verifier_text,
            'verifier source progress string')
    stdout, stderr = read_text(paths['stdout']), read_text(paths['stderr'])
    require(stderr.splitlines().count(SEGMENTED) == 1, 'actual segmented init')
    for key in ('stdout', 'stderr'):
        log = command['logs'].get(paths[key].name)
        require(log == {'bytes': paths[key].stat().st_size, 'sha256': artifacts[key]['sha256']}, 'owner log pin')
    outputs = re.findall(r'^output  :(.*)$', stdout, re.MULTILINE)
    prompts = re.findall(r'^prompt  :(.*)$', stdout, re.MULTILINE)
    require(len(outputs) == len(prompts) == 1 and ids(outputs[0]) == common['follow_ids']
            and ids(prompts[0]) == common['prompt_ids'], 'completed prompt/follow progress')
    require(len(re.findall(r'^decode\s+' + str(common['N']) + r' tokens in ', stdout, re.MULTILINE)) == 1,
            'normal decode count')
    require(re.findall(r'^logits dumped\s+(.+)$', stdout, re.MULTILINE) == [opts['--dump-logits']], 'dump completion')
    require(not re.search(r'(?i)\b(error|failed|cannot|nonfinite|nan|inf|fault|cancelled|timeout)\b', stderr),
            'runtime diagnostic error')
    fault_path = pin(sidecar['fault'])
    fault = bounded_json(fault_path)
    fields(fault, 'schema boot_before boot_after before_utc after_utc command_sha256 '
           'kernel_complete health_complete records health_records kernel_evidence health_evidence', 'fault')
    require(fault['schema'] == 'strata.capture-fault-health.v1', 'fault schema')
    require(fault['boot_before'] == fault['boot_after'] == identities['boot_before'], 'fault boot')
    fault_begin, fault_end = timestamp(fault['before_utc']), timestamp(fault['after_utc'])
    require(fault_begin <= begin < end <= fault_end
            and (begin - fault_begin).total_seconds() <= 60
            and (fault_end - end).total_seconds() <= 60, 'fresh fault interval')
    require(fault['command_sha256'] == identities['command_sha256'], 'fault command binding')
    require(fault['kernel_complete'] is True and fault['health_complete'] is True, 'fault coverage')
    # Producer must normalize the entire bounded interval to JSON lists. Empty
    # health/fault sets alone cannot substitute for pinned complete collectors.
    for key, evidence in (('records', 'kernel_evidence'), ('health_records', 'health_evidence')):
        require(fault[key] == bounded_json(pin(fault[evidence])) == [],
                'fault/health records or collector mismatch')
    close = bounded_json(pin(sidecar['close']))
    fields(close, 'schema command_sha256 capture_sha256 bytes closed_after_utc close_error '
           'forced canceled deadline_hit budget_hit', 'capture close')
    require(close['schema'] == 'strata.capture-close.v1' and close['command_sha256'] == identities['command_sha256']
            and close['capture_sha256'] == artifacts['capture']['sha256'], 'capture close identity')
    require(timestamp(close['closed_after_utc']) >= end and close['close_error'] is None, 'capture close observation')
    require(all(close[key] is False for key in ('forced', 'canceled', 'deadline_hit', 'budget_hit')),
            'abnormal closure')
    capture = paths['capture']
    size = 8 + 4 * common['vocab'] * common['N']
    require(type(close['bytes']) is int and close['bytes'] == capture.stat().st_size == size, 'exact capture bytes')
    endian = '<' if common['endian'] == 'little' else '>'
    with capture.open('rb') as stream:
        require(struct.unpack(endian + 'ii', stream.read(8)) == (common['vocab'], common['N']), 'capture header')
        for _ in range(common['N']):
            row = struct.unpack(endian + str(common['vocab']) + 'f', stream.read(4 * common['vocab']))
            require(all(math.isfinite(value) for value in row), 'nonfinite full vocabulary')
        require(not stream.read(1), 'capture trailing bytes')
    # Rehash small evidence and capture after validation; never read tensor files.
    for path, expected_hash in tracked.items():
        require(digest(path) == expected_hash, 'evidence changed during validation')
    qualified = (sidecar['evidence_kind'] == 'production'
                 and sidecar['producer'] in REVIEWED_PRODUCERS)
    return {'schema': 'strata.fixed-history-capture-receipt.v1', 'common': common,
            'bindings': {key: artifacts[key]['sha256'] for key in ('capture', 'binary', 'source', 'runtime')},
            'positions': list(range(common['P'] - 1, common['P'] + common['N'] - 1)),
            'targets': common['follow_ids'], 'observed_rows': common['N'], 'source_provider': PROVIDER,
            'supervisor_evidence': sidecar['owner_receipt'], 'fault_evidence': sidecar['fault'],
            'provider_evidence': artifacts['provider_evidence'],
            'lifecycle': {'observed': True, 'reaped': True, 'owned_processes_empty': True,
                          'closed': True, 'capture_close_ok': True, 'normal_exit': True,
                          'fault_check_complete': True, 'forced': False, 'canceled': False,
                          'deadline_hit': False, 'budget_hit': False, 'exit_code': 0, 'fault_count': 0,
                          'boot_before': identities['boot_before'], 'boot_after': identities['boot_after'],
                          'owned_pid': identity['pid'], 'owned_start': str(identity['start_ticks'])},
            'sidecar_sha256': digest(sidecar_path), 'adapter_schema_validated': True,
            'production_capture_qualified': qualified, 'quality_qualified': False,
            'qualification_blocker': None if qualified else 'reviewed real producer is not registered',
            'persistent_state_restore': 'not implemented', 'heldout_task_quality': 'not evaluated',
            'adoption': 'no automatic decision'}


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: fixed_history_capture_receipt.py capture-sidecar.json')
    try:
        print(json.dumps(adapt(sys.argv[1]), sort_keys=True, allow_nan=False))
    except (ContractError, KeyError, TypeError, OSError, ValueError, struct.error) as error:
        raise SystemExit('capture rejected: ' + str(error)) from error


if __name__ == '__main__':
    main()
