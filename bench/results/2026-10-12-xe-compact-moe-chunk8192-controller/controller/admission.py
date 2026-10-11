"""Exported controller gates. Pure predicates; Main owns all execution."""
import re
from collections import Counter
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
ORIGINAL_CODE_NAMES = set(SHARED) | {'run_root.py', 'admission.py', 'controller_cpu_fixtures.py'}
CODE_NAMES = ORIGINAL_CODE_NAMES | {'window_protocol.py'}
PUBLISHED_REVISION = '39e395a103af36323262cdc0dbaea27cba81f3e4'
SEQUENCE = ((32768, 8192), (65536, 6144), (65536, 8192))
MATRIX = ((64, 'off'), (64, 'on'), (2049, 'off'), (2049, 'on'), (32768, 'off'), (32768, 'on'))
HISTORICAL = {
    64: 'b3041c64d43bcbf5d558e3faeac56b6102c8706a161b5a1742d05b1065364744',
    2049: 'fca4e11f4fdc284b2fe871ead1be534c89da6ac5bcf0b9ffe3f6769f56ce6717',
    32768: '01992f0ba3b15a135636e97ff4dd37fc03f1137486df948ff7631edc96a1c9d1',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def admitted_variant(n, chunk, compact, reference, group=16):
    require(type(n) is int and type(chunk) is int and (n, chunk) in SEQUENCE, 'explicit three-case long diagnostic')
    require(type(group) is int and group == 16, 'fixed G16 recipe')
    require(compact == 'on', 'compact ON only; 8192 OFF has no fit proof')
    require((chunk == 8192) == (reference is not None), '8192 requires same-N6144 reference; fresh64K6144 control forbids reference')
    return SEQUENCE.index((n, chunk))


def root_path(base, n, chunk):
    admitted_variant(n, chunk, 'on', Path('reference') if chunk == 8192 else None)
    return Path(base) / 'xe-compact-moe-chunk8192-root-v1' / f'n{n}-t{chunk}-g16-on'


def predecessor_path(base, n, chunk, first):
    index = SEQUENCE.index((n, chunk))
    if index == 0:
        return Path(first)
    return root_path(base, *SEQUENCE[index - 1]) / 'record.json'


def require_sequence_prefix(present, n, chunk):
    index = SEQUENCE.index((n, chunk))
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


def require_qualification(q, expected, code_names=None):
    require(q.get('status') == 'passed' and q.get('main_reviewed') is True, 'actual Main qualification; no pending launch')
    require(q.get('base_revision') == BASE_REVISION and q.get('source_review_sha256') == SOURCE_REVIEW_SHA256,
            'exact baseline and applied Main review')
    require_source_pins(q.get('production_source_pins'), expected)
    require(q.get('owner_sha256') == OWNER_SHA256, 'global helper ownership identity')
    require(set(q.get('build_receipts', {})) == {'free', 'contrib-llvm', 'free-asan', 'contrib-llvm-asan'},
            'all four build qualifications mandatory')
    require(set(q.get('controller_source_pins', {})) == (CODE_NAMES if code_names is None else code_names), 'entire controller/helper closure')
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


def matrix_path(base, n, compact):
    require((n, compact) in MATRIX, 'exact original six-case matrix root')
    return Path(base)/'xe-compact-moe-model-root-v1'/f'n{n}-g16-{compact}'/'record.json'


def require_chunk_qualification(q, expected, original):
    require_qualification(q, expected)
    require_qualification(original, expected, ORIGINAL_CODE_NAMES)
    require(q.get('published_revision') == PUBLISHED_REVISION and q.get('compact') == 'on' and
            q.get('sequence') == [list(x) for x in SEQUENCE], 'exact committed source and three-case ON campaign')
    for key in ('production_source_pins', 'owner_sha256', 'source_review_sha256', 'base_revision', 'build_receipts',
                'build_receipt_sha256', 'qualified_build_binaries', 'free_release_binary', 'host_fixture_receipt',
                'compiler_receipt', 'compiler_receipt_sha256', 'lint_receipt', 'lint_receipt_sha256', 'qualified_additional_mappings'):
        require(q.get(key) == original.get(key), 'unchanged actually qualified production proof ' + key)
    require(all(q['receipt_sha256'].get(path) == digest for path, digest in original['receipt_sha256'].items()),
            'every original Main prerequisite identity preserved')
    require(set(q.get('six_case_receipts', {})) == {f'n{n}-{v}' for n, v in MATRIX}, 'actual six-case prerequisites mandatory')


def require_same_recipe(reference, current, before_chunk, after_chunk):
    require(reference.count('--prefill') == current.count('--prefill') == 1, 'one fixed prefill request')
    before, after = reference.index('--prefill') + 1, current.index('--prefill') + 1
    require(before == after and reference[before] == str(before_chunk) and current[after] == str(after_chunk),
            'actual explicit before/after fixed chunk')
    changed = list(reference); changed[before] = str(after_chunk)
    require(current == changed, 'only fixed --prefill width differs in actual argv')


def require_original_controller_proofs(cpu, review, original):
    require_closed_prerequisite(cpu)
    require(cpu['source_pins'] == original['controller_source_pins'] and cpu['source_stable'] is True and
            cpu['owner_sha256'] == OWNER_SHA256 and cpu['main_review_sha256'] == original['controller_review_sha256'],
            'actual original controller CPU source/review closure')
    require(review['main_reviewed'] is True and review['source_pins'] == original['controller_source_pins'] and
            review['source_publication_revision'] == PUBLISHED_REVISION, 'actual original Main controller review')
    require(cpu['gpu_executed'] is False and cpu['model_executed'] is False and cpu['full262144_qualified'] is False and
            cpu['broad_quality_qualified'] is False and cpu['restore_qualified'] is False and cpu['performance_eligible'] is False,
            'original CPU receipt cannot assert runtime quality')


def require_completed_matrix(values, pins, original, original_sha, boot, base):
    from protocol import validate
    require(set(values) == set(pins) == {f'n{n}-{v}' for n, v in MATRIX}, 'complete exact six-case matrix')
    anchor = values['n32768-on']
    for n, variant in MATRIX:
        key = f'n{n}-{variant}'
        value, pin = values[key], pins[key]
        require(pin['path'] == str(matrix_path(base, n, variant)) and re.fullmatch('[0-9a-f]{64}', pin['sha256']) is not None,
                'actual exact original root/hash; no PENDING pin')
        require_normal_model(value, boot)
        require(value['input_tokens'] == n and value['group'] == 16 and value['compact'] == variant and value['batch'] == 'off',
                'exact prerequisite variant')
        require(value['source_pins'] == original['production_source_pins'] and value['binary'] == original['free_release_binary'] and
                value['published_revision'] == PUBLISHED_REVISION and value['qualification_sha256'] == original_sha and
                value['controller_sha256'] == original['controller_source_pins']['run_root.py'] and
                value['diagnostic_helper_pins'] == original['controller_source_pins'] and
                value['controller_main_review_sha256'] == original['controller_review_sha256'], 'original qualified source/controller/proof closure')
        require(value['source_guard_phases'] == ['admission', 'ready', 'done', 'final'] and
                value['actual_environment_matches'] is True and value['requested_chunk'] == value['expected_actual_chunk'] == 6144 and
                value['actual_chunk_validation']['actual'] == 6144 and value['actual_chunk_validation']['no_borrow'] is True,
                'original current source and actual6144 no-borrow guards')
        proof = value['cpu_exclusion_proof']
        require(proof['cpu_experts'] == 0 and proof['gpu_header_sha256'] == original['production_source_pins']['include/strata/core/gpu.hpp'] and
                proof['prefill_sha256'] == original['production_source_pins']['src/prefill/prefill.cpp'], 'actual original hardCPU0 pinned source proof')
        require(validate(value['request_protocol'], n=n) == value['validation'] and value['validation']['generated_tokens'] == 16,
                'actual six-case strict16 protocol')
        settings = value['effective_settings']
        require(all(settings.get(k) == v for k, v in dict(context='262144', kv='int8', kv_resident='32768',
            spec='4', spec_min_p='0.00', expert_slots='144', pool_workers='5', pcie_frac='0.00').items()) and
            int(settings['vram_free_mib']) >= 512, 'original actual effective settings and READY reserve')
        require(value['argv'] == anchor['argv'], 'six-case identical fixed recipe argv')
        require(compact_environment({k:v for k,v in anchor['environment'].items() if k != 'STRATA_PREFILL_COMPACT_MOE_SCRATCH'}, variant)
                == value['environment'], 'six-case only compact flag differs')
        offsets = list(range(0, n-1, 6144))
        rows = value['route_census']['rows']
        require([(x['offset'], x['layer']) for x in rows] == [(o,layer) for o in offsets for layer in range(48)] and
                all(x['mmq_enabled'] == 0 and x['plan_fallback'] == 1 and x['tokens'] == min(6144,n-1-x['offset']) for x in rows),
                'actual original48-layer pure fallback full/tail coverage')
        for phase in ('ready_before_gen', 'after_gen'):
            require_additional_mappings(value['mapping_snapshots'][phase], original['qualified_additional_mappings'][phase])
        require(value['ready_runtime_mapping_validation']['all_observed_exactly_qualified'] is True and
                value['after_gen_runtime_mapping_validation']['all_observed_exactly_qualified'] is True and
                value['runtime_mapping_validation']['all_observed_exactly_qualified'] is True, 'original core runtime gates')
        require(value['health-before_protocol'][:6] == value['health-after_protocol'][:6] == anchor['health-before_protocol'][:6],
                'same exact fresh integer health protocol')
        require(value['performance_eligible'] is False and value['full262144_qualified'] is False and
                value['restore_qualified'] is False and value['broad_quality_qualified'] is False, 'diagnostic-only prerequisite scope')
        if variant == 'on':
            off = values[f'n{n}-off']
            require(off['input'] == value['input'] and off['model_inputs'] == value['model_inputs'], 'original paired exact input/model identities')
            require(value['reference']['path'] == pins[f'n{n}-off']['path'] and
                    value['reference']['sha256'] == pins[f'n{n}-off']['sha256'], 'exact actual paired OFF receipt')
            compare_strict_outputs(off, value)
    return dict(all_six_closed_pass=True, original_qualification_sha256=original_sha,
                original_controller_pins=original['controller_source_pins'], receipt_pins=pins,
                scope='closed existing6144 OFF/ON matrix only; no8192/fullcontext qualification')


def strong_closed_command(command):
    return bool(command.get('observation_complete') and command.get('direct_child_reaped') and
                command.get('session_empty') and command.get('session_anchored') and
                not command.get('survivors') and not command.get('error') and
                not command.get('errors') and not command.get('cleanup'))


BUILD_MODES = ('free', 'contrib-llvm', 'free-asan', 'contrib-llvm-asan')
RELEASE_REJECTIONS = {
    'invalid-max-new-nan': ['--max-new', 'nan'],
    'invalid-max-new-9223372036854775808': ['--max-new', '9223372036854775808'],
    'invalid-max-context-262144x': ['--max-context', '262144x'],
}


def require_closed_prerequisite(value, nonzero=None):
    """Actual immutable receipt closure, with explicit argv-bound nonzero cases only."""
    require(value.get('passed') is True and value.get('complete') is True and value.get('active') is False and
            not value.get('error'), 'actual completed PASS prerequisite')
    commands = value.get('commands', []) + value.get('reused_commands', [])
    require(commands and len({c['label'] for c in commands}) == len(commands), 'unique actual prerequisite commands')
    require(all(strong_closed_command(c) for c in commands), 'all prerequisite commands strongly closed')
    exceptions = nonzero or {}
    for command in commands:
        if command['label'] in exceptions:
            code, argv = exceptions[command['label']]
            require(command['exit_code'] == code and command['argv'] == argv, 'exact prerequisite nonzero command recipe')
        else:
            require(command['exit_code'] == 0, 'normal prerequisite command exit0')
    require(set(exceptions) <= {c['label'] for c in commands}, 'every exact exception command is present')


def require_source_prerequisite(value, expected, source_tree, phase):
    require_source_pins(value.get('source_pins'), expected)
    require(value.get('phase') == phase and value.get('source_base') == BASE_REVISION and
            value.get('source_tree') == str(source_tree) and value.get('main_review_sha256') == SOURCE_REVIEW_SHA256 and
            value.get('owner_sha256') == OWNER_SHA256 and value.get('source_stable') is True,
            'actual applied source/build prerequisite identity')
    require(value.get('gpu_executed') is False and value.get('model_executed') is False and
            value.get('full262144_runtime_qualified') is False and value.get('adopted') is False,
            'source-only prerequisite does not assert runtime adoption')


def require_host_prerequisite(value, expected, source_tree, compiler_sha):
    require_closed_prerequisite(value)
    require_source_prerequisite(value, expected, source_tree, 'host')
    require(value.get('compiler_receipt_sha256') == compiler_sha, 'host exact compiler proof')
    require(set(value.get('binaries', {})) == {'release', 'asan-ubsan'}, 'actual release and sanitizer host fixtures')
    require({'compile-release', 'test-release', 'compile-asan-ubsan', 'test-asan-ubsan'} <=
            {c['label'] for c in value['commands']}, 'actual host fixture compilation and execution')


def require_build_prerequisite(value, mode, expected, source_tree, host_sha, compiler_sha, binary, cache):
    require(mode in BUILD_MODES, 'explicit qualified build mode')
    exceptions = {} if mode.endswith('-asan') else {
        label: (2, [binary['path'], *args]) for label, args in RELEASE_REJECTIONS.items()}
    require_closed_prerequisite(value, exceptions)
    require_source_prerequisite(value, expected, source_tree, mode)
    require(value.get('binary') == binary and value.get('host_fixture_receipt_sha256') == host_sha and
            value.get('compiler_receipt_sha256') == compiler_sha, 'same actual binary/host/compiler build provenance')
    require('100% tests passed, 0 tests failed out of 14' in value.get('cpu_test_output', '') and
            'expert_multi_test (Skipped)' in value['cpu_test_output'], 'actual selected CPU tests and recorded AVX512 skip')
    require({'configure', 'compile-full', 'ninja-recipe', 'cpu-ctest'} <= {c['label'] for c in value['commands']},
            'actual full build and CPU test commands')
    require(set(value.get('build_recipe', {})) == {'CMakeCache.txt', 'compile_commands.json', 'BUILD.json'},
            'complete retained build recipe files')
    license_mode = 'contrib' if mode.startswith('contrib-llvm') else 'free'
    require('STRATA_LICENSE:STRING=' + license_mode in cache and 'STRATA_WERROR:BOOL=ON' in cache and
            'STRATA_BUILD_TESTS:BOOL=ON' in cache and 'std=c++20' in value.get('ninja_compile_link_recipe', ''),
            'actual license/test/warning/C++20 recipe')
    for label in exceptions:
        require('needs an int64 value' in value['command_evidence'][label]['stderr'], 'exact CLI refusal before device setup')


def require_compiler_prerequisite(value):
    require_closed_prerequisite(value)
    require(value.get('owner_sha256') == OWNER_SHA256 and value.get('gpu_executed') is False and
            value.get('model_executed') is False and value.get('queue_created') is False,
            'actual original source compiler CPU qualification')
    require({'bin/clang', 'bin/clang++', 'lib/libsycl.so'} <= set(value.get('installed_files', {})),
            'actual installed compiler and SYCL identities')


def require_lint_tools(stdout, stderr):
    require('skipped (' not in stdout and 'possible secret' not in stdout, 'all mandatory lint tools ran')
    for tool in ('clang-tidy', 'cppcheck', 'cmake-lint', 'codespell', 'reuse'):
        require('== ' + tool in stdout, 'actual mandatory lint tool ' + tool)
    require(not stdout.split('== codespell\n', 1)[1].split('== reuse', 1)[0].strip() and
            not stdout.split('== reuse\n', 1)[1].strip(), 'actual codespell/reuse clean')
    require(not any('error:' in line and '[unknownMacro]' not in line for line in stderr.splitlines()),
            'no unclassified lint tool errors')


def require_lint_prerequisite(value, expected, changed, build_hashes, original_failed=None, original_sha=None):
    """Actual lint v2 schema; matched baseline exit1 is a narrow immutable diagnostic exception."""
    require(value.get('main_review_sha256') == SOURCE_REVIEW_SHA256 and value.get('source_base') == BASE_REVISION and
            value.get('owner_sha256') == OWNER_SHA256, 'actual Main lint identity')
    require_source_pins(value.get('source_pins'), expected)
    require(value.get('gpu_executed') is False and value.get('model_executed') is False and
            value.get('adopted') is False, 'lint CPU-only scope')
    require(set(value.get('current_files', [])) == set(changed) and
            set(value['baseline_files']).isdisjoint(value['new_files']) and
            set(value['baseline_files']) | set(value['new_files']) == set(changed), 'lint exact six-file coverage')
    require(set(value.get('comparisons', {})) == {'free', 'contrib-llvm'}, 'actual two-mode lint comparison')
    exceptions = {}
    for mode, comparison in value['comparisons'].items():
        be, ce = comparison['baseline_exit_code'], comparison['candidate_exit_code']
        require(be in (0, 1) and ce in (0, 1) and (ce == 0 or be == ce == 1), 'only matched lint exit1 exception')
        require(not comparison['new_findings'] and not
                (Counter(map(tuple, comparison['candidate_diagnostics'])) - Counter(map(tuple, comparison['baseline_diagnostics']))),
                'no new diagnostic occurrences, independently counted')
        require(comparison['build_receipt_sha256'] == build_hashes[mode], 'lint exact actual mode build proof')
        for side, files, code in (('baseline', value['baseline_files'], be), ('candidate', value['current_files'], ce)):
            require_lint_tools(comparison[side + '_stdout'], comparison[side + '_stderr'])
            label = side + '-' + mode
            argv = ['/bin/bash', 'tools/lint/run.sh', *files]
            commands = value['commands'] + value.get('reused_commands', [])
            selected = [c for c in commands if c['label'] == label]
            require(len(selected) == 1 and selected[0]['argv'] == argv and selected[0]['exit_code'] == code,
                    'actual exact full mandatory lint command')
            if code:
                exceptions[label] = (code, argv)
    require_closed_prerequisite(value, exceptions)
    if value.get('reused_commands'):
        origin = value['original_failed_comparison']
        safe_record_path(origin['path'])
        require(origin.get('original_status') == 'FAILED' and origin.get('free_lints_reexecuted') is False and
                origin.get('sha256') == original_sha and original_failed is not None,
                'actual immutable original comparison provenance')
        require(original_failed.get('active') is False and original_failed.get('passed') is False and
                original_failed.get('complete') is False and
                original_failed.get('error') == "RuntimeError('no new diagnostic occurrences free')",
                'original failed lint status preserved')
        require(original_failed['source_pins'] == expected and original_failed['main_review_sha256'] == SOURCE_REVIEW_SHA256 and
                original_failed['owner_sha256'] == OWNER_SHA256 and
                all(strong_closed_command(c) for c in original_failed['commands']), 'original lint source and anchored closure')
        free = [c for c in original_failed['commands'] if c['label'] in ('baseline-free', 'candidate-free')]
        require(value['reused_commands'] == free and len(free) == 2, 'byte-content exact original free command structures')
        original = original_failed['comparisons']['free']
        require(original['build_receipt_sha256'] == build_hashes['free'] and all(
            value['comparisons']['free'][key] == original[key] for key in
            ('baseline_stdout', 'baseline_stderr', 'candidate_stdout', 'candidate_stderr')),
            'unchanged actual complete free lint outputs; corrected comparison only')


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
