# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Read-only adapter for explicitly reviewed, closed actual SERVE GEN1 rows.

Registration is a source-reviewed immutable evidence tuple, never a CLI switch
or a caller's producer/provenance claims. No inference, tensor reads or launch.
"""
import hashlib
import json
import math
import re
from pathlib import Path
import struct
import subprocess

from fixed_history_logits import digest, ids_hash, integer, require
from fixed_history_capture_receipt import bounded_json, fields, ids, timestamp

PROVIDER = 'serve-phase-gated-fresh-gen1-first-verifier-row'
REGISTRY_SHA256 = '412b1ceaba077859e1b7511c5c6256a20fdcc183d3aa9c64fba207a36ebf52cc'
MAX_ROW_BYTES = 2 << 20
MAX_FIXTURE_BYTES = 2 << 20


def reviewed_producers():
    path = Path(__file__).with_name('serve_first_row_reviewed_producers.json')
    require(path.stat().st_size <= 16 << 20, 'registry byte budget')
    require(digest(path) == REGISTRY_SHA256, 'reviewed registry source pin')
    value = bounded_json(path)
    require(value['schema'] == 'strata.reviewed-serve-first-row-producers.v1', 'registry schema')
    return value['producers']


def read_row(path, vocab):
    """Native format has no magic: exact LE int32 (vocab, 1) is its header."""
    require(integer(vocab, 2) and 8 + 4 * vocab <= MAX_ROW_BYTES, 'row budget')
    require(path.stat().st_size == 8 + 4 * vocab, 'row exact bytes')
    with path.open('rb') as stream:
        require(struct.unpack('<ii', stream.read(8)) == (vocab, 1), 'row header/endian/shape')
        row = struct.unpack('<' + str(vocab) + 'f', stream.read(4 * vocab))
        require(not stream.read(1), 'row trailing bytes')
    require(all(math.isfinite(x) for x in row), 'row nonfinite')
    return row


def closed_command(command):
    require(all(command.get(k) is True for k in ('observation_complete', 'direct_child_reaped',
            'session_empty', 'session_anchored')), 'actual owner closure')
    require(command.get('ownership_status') == 'observed' and command.get('errors') == []
            and command.get('cleanup') == [] and command.get('survivors') == []
            and not command.get('error'), 'actual owner errors')
    require(type(command.get('exit_code')) is int and command['exit_code'] == 0, 'actual owner exit')
    identity = command['direct_identity']
    require(integer(identity['pid'], 1) and integer(identity['start_ticks'], 1)
            and command['pid'] == identity['pid'] == identity['sid'], 'actual owner identity')
    require(any(all(x.get(k) == identity[k] for k in ('pid', 'start_ticks', 'sid', 'pgid'))
                for x in command['owners']), 'actual registered owner')
    require(timestamp(command['started_utc']) < timestamp(command['finished_utc']), 'owner interval')


def validate_record(record, registration, fixture_ids, request_index):
    """Semantic review of the authenticated receipt, also used by CPU fixtures."""
    require(record['active'] is False and all(record[k] is True for k in
            ('passed', 'complete', 'main_reviewed', 'actual_first_row_capture_qualified',
             'full_vocabulary_capture', 'phase_engine_lifetime_qualified')), 'actual receipt status')
    require(record['source_revision'] == registration['source_revision']
            and record['source_pins'] == registration['source_pins'], 'actual source provenance')
    require(record['binary_sha256'] == registration['binary_sha256']
            and record['provider_hashes'] == registration['provider_hashes'], 'actual binary/provider identity')
    before, after = record['model-hashes-before'], record['model-hashes-after']
    require(before['complete'] is True and after['complete'] is True
            and before['files'] == after['files'] == registration['model_files'], 'actual model identity')
    require(record['fixture'] == registration['fixture'], 'entire conditioning fixture identity')
    require(len(fixture_ids) == record['fixture']['total_tokens']
            and all(integer(x) and x < 248320 for x in fixture_ids), 'fixture IDs')
    require(type(request_index) is int and request_index == registration['request_index'], 'reviewed request index')
    request, capture = record['requests'][request_index], record['captures'][request_index]
    p = request['prompt_tokens']
    require(integer(p, 2) and p < len(fixture_ids) and request['max_new'] == 1
            and type(request['max_new']) is int and request['complete'] is True
            and request['checkpoint_key_omitted'] is True, 'fresh GEN1 request')
    packet = ('GEN 1 temperature=0 ' + ' '.join(map(str, fixture_ids[:p])) + '\n').encode()
    require(hashlib.sha256(packet).hexdigest() == request['packet_sha256'], 'actual request packet identity')
    require(capture['prompt_tokens'] == p and capture['predicted_position'] == p - 1
            and capture['shape'] == [1, 248320] and capture['dtype'] == 'IEEE754FP32LE'
            and capture['bytes'] == 993288 and capture['finite'] is True, 'actual request row position/layout')
    require({'path': capture['path'], 'sha256': capture['sha256']} == registration['retained_capture'],
            'retained reviewed row identity')
    require(capture['consumer'] and capture['next_review'] and capture['byte_budget'] >= capture['bytes'],
            'actual retained consumer/review')
    copied = [str(request_index + 1), str(p), str(p - 1), '1', '0']
    completed = copied[:3] + ['248320', '993288', capture['path']]
    require(record['capture_copied_rows'][request_index] == copied
            and record['capture_completed_rows'][request_index] == completed, 'actual copied/completed mapping')
    env = record['environment']
    expected = {'STRATA_SERVE_FIRST_ROW_CAPTURE': '1', 'STRATA_PHASE_DECODE_CACHE': '1',
                'STRATA_PHASE_HEAD_LEASE': '1', 'STRATA_PREFILL_SCRATCH_ALIAS': '1',
                'STRATA_PREFILL_GATED_ALIAS': '1', 'STRATA_PHASE_PREFILL_12K': '0',
                'STRATA_VERIFY_SEGMENTED': '1', 'STRATA_VERIFY_DEVICE_PLAN': '0'}
    require(all(env.get(k) == v for k, v in expected.items()), 'qualified effective route')
    require(record['configured_prefill_capacity'] == 8192 and record['effective_pipeline_windows'] == 0,
            'qualified 8K route')
    argv = record['engine_argv']
    require('--serve' in argv and argv[argv.index('--spec') + 1] == '4'
            and argv[argv.index('--prefill') + 1] == '8192'
            and argv[argv.index('--max-context') + 1] == '262144', 'actual raw spec/prefill/context capacity')
    commands = {c['label']: c for c in record['commands']}
    for label in ('engine-phase-lifetime', 'health-before', 'health-after', 'kernel-before',
                  'kernel-after-prehealth', 'kernel-after-model', 'kernel-posthealth',
                  'model-hashes-before', 'model-hashes-after'):
        closed_command(commands[label])
        # Exact text was extracted into immutable owner evidence before log retirement.
        for channel in ('stdout', 'stderr'):
            text = record['evidence'][label][channel].encode()
            require(commands[label]['logs'][label + '.' + channel] ==
                    {'bytes': len(text), 'sha256': hashlib.sha256(text).hexdigest()}, 'actual embedded log pin')
    engine = commands['engine-phase-lifetime']
    require(engine['argv'] == argv and engine['environment'] == env
            and record['ready']['pid'] == engine['pid']
            and record['ready']['start_ticks'] == engine['direct_identity']['start_ticks'], 'actual ready owner binding')
    quit_receipt = record['normal_quit']
    require(quit_receipt == {'original_command': 'QUIT', 'written': True,
            'pipe_closed_after_write': True, 'fallback_signals': [], 'engine_exit': 0,
            'owner_closed': True, 'reaped': True, 'session_empty': True}, 'actual normal QUIT')
    evidence = record['evidence']['engine-phase-lifetime']
    stderr, stdout = evidence['stderr'], evidence['stdout']
    copied_line = (f'strata serve-first-row: COPIED request={request_index + 1} prompt_P={p} '
                   f'predicted_position={p - 1} source_row=0 effective_T=1 actual_reuse=0 complete=0')
    complete_line = (f'strata serve-first-row: COMPLETE request={request_index + 1} prompt_P={p} '
                    f'predicted_position={p - 1} source_row=0 actual_reuse=0 vocab=248320 bytes=993288 '
                    f'file={capture["path"]} diagnostic_only=1')
    require(stderr.splitlines().count(copied_line) == stderr.splitlines().count(complete_line) == 1
            and stderr.index(copied_line) < stderr.index(complete_line)
            and 'strata phase-cache: normal_QUIT owner_closed requests=2' in stderr,
            'actual checked copy/publication/normal close progress')
    require(request['done_line'] in stdout.splitlines() and len(request['output_ids']) == 1
            and capture['argmax'] == request['output_ids'][0], 'actual self argmax consistency')
    done = request['done_line'].split()
    require(done[0:3] == ['DONE', '1', str(p)] and done[6:9] == ['0', '0', '0']
            and done[14] == str(p), 'actual fresh no-reuse/no-drafts DONE')
    require('spec=4 mtp_max=0 lookup=0' in stdout and 'batch_slots=0' in stdout,
            'actual no MTP/lookup/batch')
    require(timestamp(commands['health-before']['finished_utc']) <= timestamp(engine['started_utc'])
            and timestamp(engine['finished_utc']) <= timestamp(commands['health-after']['started_utc'])
            and timestamp(commands['health-after']['finished_utc']) <=
            timestamp(commands['kernel-posthealth']['started_utc']), 'fresh health/kernel coverage interval')
    for label in ('health-before', 'health-after'):
        require('PASS 0000:05:00.0: 1 round, 16384 exact words' in record['evidence'][label]['stdout'],
                'actual GPU health')
    validate_faults(record)
    return request, capture, engine


def validate_faults(record):
    """Complete pinned kernel observations and same boot; no absent-fault shortcut."""
    require(set(record['kernel_checks']) == {'kernel-before', 'kernel-after-prehealth',
            'kernel-after-model', 'kernel-posthealth'}, 'actual fault coverage')
    for label, check in record['kernel_checks'].items():
        require(check['fault_entries'] == [] and check['relevant_entries'] == []
                and check['dump_present'] is False and bool(check['filter']), 'actual fault result')
        journal = record['evidence'][label]['stdout']
        entries = [json.loads(line) for line in journal.splitlines() if line.strip()]
        require(not any(re.search(check['filter'], x.get('MESSAGE', ''), re.IGNORECASE)
                        for x in entries), 'actual kernel filter found fault')
        require(len(entries) == check['total_rows'] and all(
                x['_BOOT_ID'] == record['boot_id'].replace('-', '') for x in entries), 'actual kernel boot/count')


def validate_retained_consumer(retention, original_path, sidecar):
    require(not any(x['path'] == original_path for x in retention['deletions']), 'selected row retired')
    for key in ('consumer', 'retire_after'):
        require(isinstance(sidecar[key], str) and bool(sidecar[key].strip()), 'current retained ' + key)


def adapt_serve(sidecar_path):
    sidecar_path = Path(sidecar_path).resolve()
    initial = digest(sidecar_path)
    sidecar = bounded_json(sidecar_path)
    fields(sidecar, 'schema producer receipt publication retention capture fixture source_root source_commit '
           'request_index model_files consumer retire_after', 'serve sidecar')
    require(sidecar['schema'] == 'strata.serve-first-row-sidecar.v1', 'serve sidecar schema')
    registrations = reviewed_producers()
    require(sidecar['producer'] in registrations, 'unreviewed actual SERVE producer')
    reg = registrations[sidecar['producer']]
    tracked = {sidecar_path: initial}

    def pin(entry, expected):
        fields(entry, 'path sha256', 'serve pin')
        require(entry['sha256'] == expected['sha256'], 'authoritative reviewed evidence hash')
        path = (sidecar_path.parent / entry['path']).resolve()
        require(path.is_file() and path.stat().st_size <= 16 << 20, 'serve evidence byte budget')
        if expected['sha256'] == reg['retained_capture']['sha256']:
            require(path.stat().st_size == 993288, 'selected retained row byte count')
        if expected['sha256'] == reg['fixture']['sha256']:
            require(path.stat().st_size <= MAX_FIXTURE_BYTES, 'conditioning fixture byte budget')
        require(digest(path) == expected['sha256'], 'serve evidence pin mismatch')
        tracked[path] = expected['sha256']
        return path

    receipt = bounded_json(pin(sidecar['receipt'], reg['receipt']))
    publication = bounded_json(pin(sidecar['publication'], reg['publication']))
    require(publication['active'] is False and publication['complete'] is True
            and publication['passed'] is True and publication['main_reviewed'] is True
            and publication['commit'] == reg['source_commit'], 'reviewed committed publication')
    retention = bounded_json(pin(sidecar['retention'], reg['retention']))
    require(retention['deleted'] is True and retention['owner'] == 'Main', 'reviewed retirement manifest')
    require(sidecar['source_commit'] == reg['source_commit'], 'reviewed producer commit')
    source = Path(sidecar['source_root']).resolve()
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(source), *args], stderr=subprocess.DEVNULL,
                                           timeout=10).decode().strip()
        except (subprocess.SubprocessError, OSError):
            require(False, 'reviewed producer Git identity unavailable')
    require(git('rev-parse', 'HEAD') == reg['source_commit'] and
            git('rev-parse', 'HEAD^{tree}') == reg['source_tree'], 'full committed source tree identity')
    for relative, expected_hash in reg['source_files'].items():
        pin({'path': str(source / relative), 'sha256': expected_hash}, {'sha256': expected_hash})
    for relative, expected_hash in reg['source_pins'].items():
        pin({'path': str(source / relative), 'sha256': expected_hash}, {'sha256': expected_hash})
    fixture = pin(sidecar['fixture'], reg['fixture'])
    require(fixture.stat().st_size <= MAX_FIXTURE_BYTES, 'conditioning fixture byte budget')
    fixture_ids = ids(fixture.read_text(encoding='utf-8'))
    require(sidecar['model_files'] == reg['model_files'], 'sidecar model/provider/tokenizer identity')
    for key in ('consumer', 'retire_after'):
        require(isinstance(sidecar[key], str) and bool(sidecar[key].strip()), 'current retained ' + key)
    request, capture, engine = validate_record(receipt, reg, fixture_ids, sidecar['request_index'])
    capture_path = pin(sidecar['capture'], reg['retained_capture'])
    # Historical sibling and .partial aliases are deliberately not opened or required.
    validate_retained_consumer(retention, capture['path'], sidecar)
    row = read_row(capture_path, 248320)
    require(max(range(len(row)), key=lambda i: (row[i], -i)) == capture['argmax'], 'actual row self argmax')
    p = request['prompt_tokens']
    conditioning = {'fixture_sha256': reg['fixture']['sha256'], 'fixture_total_tokens': len(fixture_ids),
                    'prompt_ids_sha256': ids_hash(fixture_ids[:p]), 'P': p,
                    'target': fixture_ids[p], 'target_source': 'frozen fixture[P]',
                    'position': p - 1, 'vocab': 248320, 'endian': 'little', 'N': 1,
                    'model_files': reg['model_files'], 'max_context': 262144,
                    'packet_sha256': request['packet_sha256'],
                    'numeric_argv': receipt['engine_argv'][1:],
                    'effective_environment': {k: v for k, v in receipt['environment'].items()
                                              if k != 'STRATA_SERVE_FIRST_ROW_DIR'}}
    for path, expected_hash in tracked.items():
        require(digest(path) == expected_hash, 'serve evidence changed during validation')
    return {'schema': 'strata.serve-first-row-capture-receipt.v1', 'producer': sidecar['producer'],
            'source_provider': PROVIDER, 'conditioning': conditioning,
            'capture': {'path': str(capture_path), 'sha256': capture['sha256']},
            'source_commit': reg['source_commit'], 'source_tree': reg['source_tree'],
            'binary_sha256': reg['binary_sha256'], 'provider_hashes': reg['provider_hashes'],
            'source_files_sha256': hashlib.sha256(json.dumps(reg['source_files'], sort_keys=True,
                                                            separators=(',', ':')).encode()).hexdigest(),
            'receipt_sha256': reg['receipt']['sha256'], 'publication_sha256': reg['publication']['sha256'],
            'sidecar_sha256': initial, 'observed_rows': 1, 'position': p - 1,
            'owned_pid': engine['pid'], 'owned_start': engine['direct_identity']['start_ticks'],
            'normal_quit_reaped_fault_checked': True, 'production_capture_qualified': True,
            'consumer': sidecar['consumer'], 'retire_after': sidecar['retire_after'],
            'quality_qualified': False, 'full_physical262144_qualified': False,
            'performance_eligible': False, 'restore_qualified': False,
            'scope': 'one actual first verifier row; self argmax is producer consistency only'}


def compare_serve(manifest_path):
    """Bounded one-row numerical comparison; each arm is adapted again from pins."""
    from fixed_history_logits import distribution
    path = Path(manifest_path).resolve()
    manifest_sha = digest(path)
    manifest = bounded_json(path)
    fields(manifest, 'schema baseline candidate consumer retire_after', 'serve comparison')
    require(manifest['schema'] == 'strata.serve-first-row-comparison.v1', 'serve comparison schema')
    for key in ('consumer', 'retire_after'):
        require(isinstance(manifest[key], str) and bool(manifest[key].strip()), 'comparison retention ' + key)
    receipts, rows = [], []
    for name in ('baseline', 'candidate'):
        entry = manifest[name]
        fields(entry, 'path sha256', 'comparison sidecar pin')
        sidecar = (path.parent / entry['path']).resolve()
        require(digest(sidecar) == entry['sha256'], 'comparison sidecar hash')
        receipt = adapt_serve(sidecar)
        receipts.append(receipt)
        rows.append(read_row(Path(receipt['capture']['path']), receipt['conditioning']['vocab']))
        require(digest(Path(receipt['capture']['path'])) == receipt['capture']['sha256'], 'comparison row changed')
    require(receipts[0]['conditioning'] == receipts[1]['conditioning'], 'paired conditioning/model mismatch')
    target = receipts[0]['conditioning']['target']
    a, am = distribution(rows[0], target)
    b, bm = distribution(rows[1], target)
    require(digest(path) == manifest_sha, 'comparison manifest changed')
    return {'schema': 'strata.serve-first-row-numerical-evidence.v1', 'manifest_sha256': manifest_sha,
            'conditioning': receipts[0]['conditioning'], 'receipts': receipts,
            'baseline': am, 'candidate': bm,
            'target_logprob_delta_candidate_minus_baseline': b[target] - a[target],
            'target_nll_delta_candidate_minus_baseline': a[target] - b[target],
            'KL_baseline_to_candidate': math.fsum(math.exp(x) * (x-y) for x,y in zip(a,b)),
            'KL_candidate_to_baseline': math.fsum(math.exp(y) * (y-x) for x,y in zip(a,b)),
            'top1_agrees': am['top1'] == bm['top1'], 'observed_rows_per_arm': 1,
            'quality_qualified': False, 'heldout_task_quality': 'not evaluated',
            'zero_ability_loss': 'not established', 'full_physical262144_qualified': False,
            'performance_eligible': False, 'restore_qualified': False,
            'adoption': 'no automatic decision'}
