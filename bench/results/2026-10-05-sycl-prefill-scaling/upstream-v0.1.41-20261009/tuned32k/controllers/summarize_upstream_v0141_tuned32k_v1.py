"""Summarize only the finished quiet actual8192 update comparison."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import statistics

b = Path(__file__).parent
path = b / 'upstream-v0141-tuned-matched32k-comparison-sequence-v1/record.json'
sequence = json.loads(path.read_text())
assert not sequence['active'] and sequence['passed'], 'comparison incomplete'
assert len(sequence['steps']) == 4
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected_binaries = {
    'control': 'dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34',
    'integrated': '86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8',
}
processes = []
reads = []
argvs, environments = set(), set()
for step in sequence['steps']:
    receipt = Path(step['receipt'])
    assert sha(receipt) == step['receipt_sha256']
    data = json.loads(receipt.read_text())
    assert not data['active'] and data['completed'] and data['healthy'] and data['math_gate_passed']
    assert data['performance_eligible'] and data['phase'] == 'clean'
    assert data['exit_code'] == 0 and not data['exit_signal']
    assert not any(data['cleanup'].values()) and not data['new_fault_messages']
    assert data['binary_sha256'] == data['actual_executable_identity']['sha256'] == expected_binaries[step['mode']]
    assert len(data['requests']) == 4
    argvs.add(tuple(data['argv'][1:]))
    environments.add(json.dumps(data['actual_target_environment'], sort_keys=True))
    info = next(x for x in data['startup'] if x.startswith('INFO '))
    assert 'expert_slots=128 ' in info and 'kv_resident=32768 ' in info
    processes.append({
        'mode': step['mode'], 'repetition': step['repetition'],
        'receipt': str(receipt), 'receipt_sha256': sha(receipt),
        'startup': data['startup'], 'binary_sha256': data['binary_sha256'],
        'exit_code': data['exit_code'], 'cleanup': data['cleanup'],
        'new_fault_messages': data['new_fault_messages'],
    })
    for request in data['requests']:
        assert request['math_gate_passed']
        progress = [int(line.split()[1]) for line in request['protocol'] if line.startswith('PP ')]
        assert progress == [8192, 16384, 24576, 32767]
        reused = [int(line.split()[1]) for line in request['protocol'] if line.startswith('REUSED ')]
        assert request['resume_tokens'] == [0, 0] and reused == [0]
        measurement = request['measurement']
        assert measurement['purpose'] == 'quiet timing'
        assert measurement['prompt_tokens'] == 32768 and measurement['generated_tokens'] == 64
        reads.append({
            'mode': step['mode'], 'repetition': step['repetition'], 'name': request['name'],
            'first_process_read': request['first_process_read'],
            'measurement': measurement, 'actual_prefill_progress': progress,
            'output_ids_sha256': hashlib.sha256(json.dumps(request['ids'], separators=(',', ':')).encode()).hexdigest(),
            'logprobs_sha256': hashlib.sha256(json.dumps(request['logprobs'], separators=(',', ':')).encode()).hexdigest(),
            'mtp_counts': request['mtp_counts'],
        })
assert len(argvs) == len(environments) == 1
assert len(reads) == 16
groups = {}
for mode in ['control', 'integrated']:
    groups[mode] = {}
    for label, first in [('first_process_read', True), ('later_fresh_reads', False)]:
        selected = [r for r in reads if r['mode'] == mode and r['first_process_read'] == first]
        assert len(selected) == (2 if first else 6)
        pp = [r['measurement']['prefill_tok_s'] for r in selected]
        tg = [r['measurement']['decode_tok_s'] for r in selected]
        pooled_pp = 1000 * sum(r['measurement']['prompt_tokens'] for r in selected) / sum(r['measurement']['prompt_ms'] for r in selected)
        pooled_tg = 1000 * sum(r['measurement']['generated_tokens'] for r in selected) / sum(r['measurement']['decode_ms'] for r in selected)
        prior = sequence['summary'][mode][label]
        assert pooled_pp == prior['prefill_tok_s'] and pooled_tg == prior['decode_tok_s']
        groups[mode][label] = {
            'count': len(selected), 'pooled_prefill_tok_s': pooled_pp, 'pooled_decode_tok_s': pooled_tg,
            'prefill_median': statistics.median(pp), 'prefill_min': min(pp), 'prefill_max': max(pp),
            'decode_median': statistics.median(tg), 'decode_min': min(tg), 'decode_max': max(tg),
        }
cross = sequence['cross_version_output_comparison']
assert len(cross) == 4
for row in cross:
    assert row['ids_equal_control'] and row['logprobs_equal_control'] and row['mtp_counts_equal_control']
differences = {}
for label in ['first_process_read', 'later_fresh_reads']:
    differences[label] = {
        metric: 100 * (groups['integrated'][label][metric] / groups['control'][label][metric] - 1)
        for metric in ['pooled_prefill_tok_s', 'pooled_decode_tok_s']
    }
summary = {
    'active': False, 'passed': True,
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Finished actual8192 quiet comparison, context262144, sixteen fresh32768-token inputs with64 visible outputs. First/later separated; two processes per mode. No isolated kernel attribution.',
    'sequence_receipt': str(path), 'sequence_receipt_sha256': sha(path),
    'physical_gate_receipt_sha256': sequence['new_full_receipt_sha256'],
    'default6144_comparison_receipt_sha256': sequence['default_comparison_receipt_sha256'],
    'engine_commits': {'integrated': '1eb89482a4afd20277ae0405780ed4f8eb98eb20', 'control_observed_worktree': '452044ab2546185bebad51c2cb16b9375b6683e3'},
    'control_source_note': 'Observed old worktree commit is not a source-identical rebuild of the privately qualified DD5 corrections; compiled inputs remain bound by the prior build record.',
    'binary_sha256': expected_binaries, 'same_requested_configuration_and_environment': True,
    'actual_chunk': 8192, 'actual_expert_slots': 128,
    'configuration_argv_without_executable': list(next(iter(argvs))),
    'actual_target_environment': json.loads(next(iter(environments))),
    'processes': processes, 'reads': reads, 'summary': groups,
    'integrated_percent_change': differences,
    'all_four_cross_version_outputs_logprobs_mtp_counts_equal': True,
    'cross_version_output_comparison': cross,
    'limitations': ['Two processes per mode; six later reads are not six independently launched processes.', 'TTFT, answer quality, peak memory and isolated CPU/SYCL kernel contributions were not measured.', 'Raw upstream was measured separately at its unchanged default actual6144 geometry and144 slots.'],
}
out = b / 'upstream-v0141-tuned32k-measured-summary-20261009-v1.json'
assert not out.exists()
out.write_text(json.dumps(summary, indent=2) + '\n')
csvpath = out.with_suffix('.csv')
assert not csvpath.exists()
fields = ['mode', 'read_group', *next(iter(groups['control'].values())).keys()]
with csvpath.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for mode, values in groups.items():
        for label, metrics in values.items():
            writer.writerow({'mode': mode, 'read_group': label, **metrics})
print(json.dumps({'summary': groups, 'integrated_percent_change': differences, 'summary_sha256': sha(out)}, indent=2))
