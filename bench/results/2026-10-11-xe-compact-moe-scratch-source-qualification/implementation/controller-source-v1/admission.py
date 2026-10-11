"""Exported controller gates. Pure predicates; Main owns all execution."""
import re
from pathlib import Path

BASE_REVISION = '0740b43488d812117f021ccbe4bd0e15d223f955'
OWNER_SHA256 = 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
SOURCE_REVIEW_SHA256 = '3040ba6c87ed575ec8ea5f2452646c6cef77a74ea331f56b60a38b2d4e054eac'
SHARED = {
    'batch_census.py': 'ce794be37f2aa670ba446cd8894ca9394a1be6a8b3a77f467cf77076d0d6f7e3',
    'fixtures.py': '706274e56ae8dbc524142688336f9053bf786cd46c2436a4434f49b588fd6a08',
    'hash_inputs.py': '25d21cfb5d9977fcabe7e1b036a7959bf8ca99d3ce797c44cacc6e56ef387df2',
    'protocol.py': 'f323da52a9f688c5d08aa45c9b0ab037f77e99d40ef577ec5ec2cd42974a6aa1',
    'route_census.py': '95c511a6ebb2feac7f5a302ede193d6b9ef34c77055d2f959e76bb5e6bb1819e',
    'runtime.py': 'e3827641ab38ffa0d60db92153c614e7e4911c48b67f5bd841617272dd7d2fd0',
    'timer_overhead.py': '13b62f3cc85c3b9b49e72c3e1838f7248d5c2530d3357c7a45a81dbdf439ac3e',
    'transfer_ledger.py': '7e5e37467208c229eeda3c7367fa16f506c11fe502ad93b28bbcc70fa4e9ab6d',
}
CODE_NAMES = set(SHARED) | {'run_root.py', 'admission.py', 'controller_cpu_fixtures.py'}
SEQUENCE = ((64, 'off'), (64, 'on'), (2049, 'off'), (2049, 'on'), (32768, 'off'), (32768, 'on'))
HISTORICAL = {
    64: 'b3041c64d43bcbf5d558e3faeac56b6102c8706a161b5a1742d05b1065364744',
    2049: 'fca4e11f4fdc284b2fe871ead1be534c89da6ac5bcf0b9ffe3f6769f56ce6717',
    32768: '01992f0ba3b15a135636e97ff4dd37fc03f1137486df948ff7631edc96a1c9d1',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def admitted_variant(n, group, compact, reference):
    require(type(n) is int and n in (64, 2049, 32768), 'explicit diagnostic N')
    require(type(group) is int and group == 16, 'fixed G16 recipe')
    require(compact in ('off', 'on'), 'explicit compact OFF/ON')
    require((compact == 'on') == (reference is not None), 'ON requires fresh OFF; OFF forbids fresh reference')
    return SEQUENCE.index((n, compact))


def root_path(base, n, compact):
    admitted_variant(n, 16, compact, Path('reference') if compact == 'on' else None)
    return Path(base) / 'xe-compact-moe-model-root-v1' / f'n{n}-g16-{compact}'


def predecessor_path(base, n, compact, first):
    index = SEQUENCE.index((n, compact))
    if index == 0:
        return Path(first)
    previous_n, previous_compact = SEQUENCE[index - 1]
    return root_path(base, previous_n, previous_compact) / 'record.json'


def require_sequence_prefix(present, n, compact):
    index = SEQUENCE.index((n, compact))
    require(set(present) == set(SEQUENCE[:index]), 'exact finite sequence prefix; no skipped, existing or future roots')


def safe_record_path(path):
    # This credential-bearing receipt is outside every diagnostic consumer, including a malicious manifest.
    require('prefill-gemm-only-contract-root-v1' not in str(path), 'excluded credential receipt')
    require(Path(path).is_absolute(), 'absolute immutable receipt path')
    require('prefill-gemm-only-contract-root-v1' not in str(Path(path).resolve()), 'excluded credential target')


def require_source_pins(pins, expected):
    require(type(pins) is dict and len(pins) == 42 and pins == expected, 'exact reviewed42 source bytes')


def require_published_state(current_pins, expected, head, published, status, diff, binary, expected_binary):
    require_source_pins(current_pins, expected)
    require(head.strip() == published and not status.strip() and diff == '', 'clean exact published source state')
    require(binary == expected_binary, 'exact actually built binary identity')


def require_qualification(q, expected):
    require(q.get('status') == 'passed' and q.get('main_reviewed') is True, 'actual Main qualification; no pending launch')
    require(q.get('base_revision') == BASE_REVISION and q.get('source_review_sha256') == SOURCE_REVIEW_SHA256,
            'exact baseline and applied Main review')
    require_source_pins(q.get('production_source_pins'), expected)
    require(q.get('owner_sha256') == OWNER_SHA256, 'global helper ownership identity')
    require(set(q.get('build_receipts', {})) == {'free', 'contrib-llvm', 'free-asan', 'contrib-llvm-asan'},
            'all four build qualifications mandatory')
    require(set(q.get('controller_source_pins', {})) == CODE_NAMES, 'entire controller/helper closure')
    require(all(q['controller_source_pins'][name] == digest for name, digest in SHARED.items()), 'eight immutable helper bytes')
    require(q.get('full262144_qualified') is False and q.get('broad_quality_qualified') is False and
            q.get('restore_qualified') is False and q.get('performance_eligible') is False, 'bounded diagnostic claims only')
    for path, digest in q.get('receipt_sha256', {}).items():
        safe_record_path(path)
        require(re.fullmatch('[0-9a-f]{64}', digest) is not None, 'actual exact receipt hash')


def require_publication(bridge, q, expected, source_tree):
    require(bridge.get('status') == 'passed' and bridge.get('main_reviewed') is True, 'actual reviewed clean publication bridge')
    require(bridge.get('base_revision') == BASE_REVISION and bridge.get('source_review_sha256') == SOURCE_REVIEW_SHA256,
            'publication baseline/review identity')
    require(bridge.get('source_tree') == str(source_tree) and bridge.get('clean') is True,
            'reviewed publication is the candidate worktree and clean')
    require(re.fullmatch('[0-9a-f]{40}', bridge.get('published_revision', '')) is not None and
            bridge['published_revision'] != BASE_REVISION, 'actual published candidate revision')
    require_source_pins(bridge.get('source_pins'), expected)
    require(bridge.get('built_source_pins') == expected == q['production_source_pins'], 'built uncommitted bytes equal published42 bytes')
    require(bridge.get('build_receipt_sha256') == q['build_receipt_sha256'], 'same exact four build receipts in publication bridge')
    require(bridge.get('controller_source_pins') == q['controller_source_pins'] and
            bridge.get('lint_receipt_sha256') == q['lint_receipt_sha256'], 'reviewed controller and lint bridge')


def strong_closed_command(command):
    return bool(command.get('observation_complete') and command.get('direct_child_reaped') and
                command.get('session_empty') and command.get('session_anchored') and
                not command.get('survivors') and not command.get('error') and
                not command.get('errors') and not command.get('cleanup'))


def require_normal_model(v, boot):
    require(v.get('passed') is True and v.get('complete') is True and v.get('active') is False,
            'closed PASS predecessor/model receipt')
    require(v.get('boot_id') == boot and v.get('boot_unchanged') is True and v.get('kernel_gate_passed') is True,
            'same-boot fault-free predecessor/model')
    require(v.get('reset_executed') is False and v.get('retry_count') == 0, 'no recovery or retry predecessor')
    commands = v.get('commands', []) + v.get('guard_commands', [])
    require(commands and all(strong_closed_command(c) and c.get('exit_code') == 0 for c in commands),
            'every predecessor command strongly closed normally')
    model = [c for c in commands if c['label'] == 'model']
    require(len(model) == 1 and v.get('model_normal_close') is True, 'one normally closed predecessor model')
    close = v.get('model_close', {})
    require(close.get('quit_requested') is True and close.get('quit_send_outcome') == 'sent' and
            close.get('quit_exit_outcome') == 'exit observed' and close.get('exit_code') == 0 and
            close.get('normal') is True and close.get('forced') is False, 'predecessor exact QUIT exit0')


def compact_environment(base, compact):
    require(compact in ('off', 'on'), 'explicit compact environment variant')
    env = dict(base, STRATA_PREFILL_COMPACT_MOE_SCRATCH='1' if compact == 'on' else '0')
    require(env.get('STRATA_PREFILL_NATIVE_DEQUANT_BATCH') == '0' and
            env.get('STRATA_PREFILL_FUSED') == '0' and env.get('STRATA_PRELOAD_KERNELS') == '0',
            'batch0 FUSED0 preload0 in both arms')
    require('STRATA_DBG_NAN' not in env and 'STRATA_PREFILL_TIMING' not in env and
            'UR_ENABLE_LAYERS' not in env and 'UR_LOG_TRACING' not in env, 'bounded nonquiet diagnostic allowlist')
    require(env.get('STRATA_TRACE') == '1' and env.get('ZE_ENABLE_VALIDATION_LAYER') == '1' and
            env.get('ZE_ENABLE_PARAMETER_VALIDATION') == '1', 'progress and parameter validation retained')
    return env


def require_exact_variant_env(off, on):
    require(off.get('STRATA_PREFILL_COMPACT_MOE_SCRATCH') == '0' and
            on.get('STRATA_PREFILL_COMPACT_MOE_SCRATCH') == '1', 'explicit compact flag0/1')
    require(on == dict(off, STRATA_PREFILL_COMPACT_MOE_SCRATCH='1'), 'only compact flag differs between arms')


def require_additional_mappings(snapshot, expected):
    require(snapshot.get('complete') is True and snapshot.get('selected_rows_stable') is True, 'complete stable mapping phase')
    actual = {p: x for p, x in snapshot['files'].items() if not x['original_runtime']}
    require(set(actual) == set(expected), 'exact compiler/driver/libc mapping set')
    for path, wanted in expected.items():
        require(all(actual[path].get(k) == wanted.get(k) for k in
                    ('path', 'canonical_path', 'sha256', 'stat', 'elf_build_id')), 'exact compiler/driver mapping identity')
    return dict(all_additional_exactly_qualified=True, paths=sorted(actual))


def compare_strict_outputs(left, right):
    for value in (left, right):
        require(value['validation']['generated_tokens'] == 16, 'strict16 comparisons only')
    def selected(value):
        return [x for x in value['request_protocol'] if x.startswith(('T ', 'LP '))]
    result = dict(output_ids_equal=left['validation']['output_ids'] == right['validation']['output_ids'],
                  selected_logprob_top5_equal=selected(left) == selected(right),
                  scope='exact16 TIDs and selected LP5 only; no source/sampler/global quality equivalence')
    require(result['output_ids_equal'] and result['selected_logprob_top5_equal'], 'strict diagnostic output/LP5 equality')
    return result


def normal_quit(io, record, *, reason='completed GEN', send_seconds=5, exit_seconds=30):
    close = record.setdefault('model_close', {})
    close.update(quit_requested=True, quit_reason=reason, quit_send_outcome='pending', quit_exit_outcome='pending')
    io.send(b'QUIT\n', seconds=send_seconds)
    close['quit_send_outcome'] = 'sent'
    io.exit(exit_seconds)
    close['quit_exit_outcome'] = 'exit observed'


def close_after_gate_failure(io, record, owner_poll, save):
    """Finite normal close only at READY-before-GEN or after DONE; preserve the primary failure."""
    require(record.get('primary_model_failure'), 'primary failure already recorded')
    if not record.get('ready_seen') or (record.get('gen_send_attempted') and not record.get('request_done_seen')):
        return False
    close = record.setdefault('model_close', {})
    close.update(quit_requested=True,
        quit_reason='completed GEN post-completion gate failure' if record.get('request_done_seen') else 'READY pre-GEN gate failure',
        quit_send_outcome='pending', quit_exit_outcome='pending')
    save()
    io.poll = owner_poll
    try:
        io.send(b'QUIT\n', seconds=5)
        close['quit_send_outcome'] = 'sent'
    except BaseException as error:
        close['quit_send_outcome'] = repr(error)
    try:
        io.exit(10)
        close['quit_exit_outcome'] = 'exit observed'
    except BaseException as error:
        close['quit_exit_outcome'] = repr(error)
    save()
    return close['quit_exit_outcome'] == 'exit observed'
