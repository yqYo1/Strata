"""Summarize only a terminal six-process,24-read quiet comparison."""
from pathlib import Path
import datetime
import hashlib
import json
import statistics

B = Path(__file__).parent
SEQ = B / 'host-prefill-accounting-v0141-quiet32k-comparison-sequence-v1/record.json'
OUT = B / 'host-prefill-accounting-v0141-quiet32k-summary-v1.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def alive(ident):
    try:
        text = (Path('/proc') / str(ident['pid']) / 'stat').read_text()
    except FileNotFoundError:
        return False
    return int(text.rsplit(')', 1)[1].split()[19]) == ident['start_ticks']


assert not OUT.exists()
s = json.loads(SEQ.read_text())
assert not s['active'] and s['passed'] and len(s['steps']) == 6
rows, host = [], []
configuration = None
for step in s['steps']:
    p = Path(step['receipt'])
    assert digest(p) == step['receipt_sha256'] and step['exit_code'] == 0
    d = json.loads(p.read_text())
    assert not d['active'] and d['completed'] and d['healthy'] and d['math_gate_passed'] and d['host_accounting_gate_passed']
    assert d['exit_code'] == 0 and not d['exit_signal'] and not d['new_fault_messages'] and not any(d['cleanup'].values())
    assert not any(alive(d[k]) for k in ['inferior', 'debugger']) and not alive(step['controller_identity'])
    assert d['phase'] == 'clean' and len(d['requests']) == 4 and d['performance_eligible']
    env = dict(d['environment'])
    host_flag = env.pop('STRATA_PREFILL_HOST_TIMING', None)
    assert host_flag == {'baseline': None, 'hostoff': '0', 'hoston': '1'}[step['mode']]
    args = list(d['argv'])
    args[0] = '<measured-binary>'
    normalized = {'argv': args, 'environment_except_host_flag': env}
    if configuration is None:
        configuration = normalized
    else:
        assert normalized == configuration
    assert d['actual_target_environment'] == d['environment']
    info = [line for line in d['startup'] if line.startswith('INFO ')]
    assert len(info) == 1
    fields = dict(part.split('=', 1) for part in info[0].split()[1:] if '=' in part)
    assert fields['context'] == '262144' and fields['expert_slots'] == '128'
    assert fields['expert_cache_mib'] == '325' and fields['kv_resident'] == '32768'
    if step['mode'] == 'baseline':
        assert d['binary_sha256'] == '86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8'
    else:
        assert d['binary_sha256'] == '494cf4be288595f7abf4d56412fb519155cfccf602a2a060093a1f8fcd988123'
    reports = d['host_accounting_reports']
    assert len(reports) == (4 if step['mode'] == 'hoston' else 0)
    for i, request in enumerate(d['requests']):
        progress = [int(line.split()[1]) for line in request['protocol'] if line.startswith('PP ')]
        assert progress == [8192, 16384, 24576, 32767]
        assert request['math_gate_passed'] and all(request['validation'].values())
        assert all(request['qualified_physical_reference_output_comparison'].values())
        if i >= 2:
            assert all(request['repeat_comparison'].values())
        m = request['measurement']
        assert m['prompt_tokens'] == 32768 and m['generated_tokens'] == 64 and m['purpose'] == 'quiet timing'
        rows.append({'mode': step['mode'], 'repetition': step['repetition'], 'request': request['name'],
                     'first_process_read': request['first_process_read'], **m})
        if reports:
            report = reports[i]
            assert report['tokens'] == 32767 and report['chunks'] == 4
            assert len(report['layers']) == 48
            assert sum(x['bytes'] for x in report['layers']) == report['expert_bytes']
            assert sum(x['copies'] for x in report['layers']) == report['expert_copies']
            assert report['expert_bytes'] == {'A': 190240998400, 'B': 190199219200}[request['fixture']]
            host.append({'mode': step['mode'], 'repetition': step['repetition'], 'request': request['name'],
                         'first_process_read': request['first_process_read'], **report})
assert len(rows) == 24 and len(host) == 8


def stats(selected):
    pp = [x['prefill_tok_s'] for x in selected]
    tg = [x['decode_tok_s'] for x in selected]
    return {'reads': len(selected),
            'pooled_prefill_tok_s': sum(x['prompt_tokens'] for x in selected) / (sum(x['prompt_ms'] for x in selected) / 1000),
            'pooled_decode_tok_s': sum(x['generated_tokens'] for x in selected) / (sum(x['decode_ms'] for x in selected) / 1000),
            'prefill_median': statistics.median(pp), 'prefill_min': min(pp), 'prefill_max': max(pp),
            'decode_median': statistics.median(tg), 'decode_min': min(tg), 'decode_max': max(tg)}


groups = {}
for mode in ['baseline', 'hostoff', 'hoston']:
    first = [x for x in rows if x['mode'] == mode and x['first_process_read']]
    later = [x for x in rows if x['mode'] == mode and not x['first_process_read']]
    assert len(first) == 2 and len(later) == 6
    groups[mode] = {'first': stats(first), 'later': stats(later)}
effects = {}
for mode in ['hostoff', 'hoston']:
    effects[mode] = {name + '_relative_to_baseline_percent': 100 * (groups[mode]['later'][name] / groups['baseline']['later'][name] - 1)
                     for name in ['pooled_prefill_tok_s', 'pooled_decode_tok_s']}
later_host = [x for x in host if not x['first_process_read']]
host_summary = {key: {'mean': statistics.mean(x[key] for x in later_host),
                      'min': min(x[key] for x in later_host), 'max': max(x[key] for x in later_host)}
                for key in ['expert_submit_host_ms', 'group_gpu_wait_host_ms', 'group_cpu_ms', 'group_profiler_host_ms',
                            'issuer_publication_wait_host_ms', 'host_copy_worker_ms', 'wall_ms']}
report = {'active': False, 'passed': True, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'Quiet same-day baseline/off/on/on/off/baseline;24 fresh32K reads,64 outputs each, context262144. First/later separate; actual8192 chunks and128 slots; no API logs, profiler or dumps.',
          'sequence_sha256': digest(SEQ), 'summarizer_sha256': digest(__file__), 'configuration': configuration,
          'groups': groups, 'effects': effects, 'later_host_summary': host_summary,
          'rows': rows, 'host_reports': host,
          'candidate_full_lifecycle_passed': False, 'candidate_adopted': False,
          'interpretation_limits': ['Host submission/group wait is not DMA active time or exclusive PCIe waiting.',
                                    'RAM-worker durations are a sum over workers and can overlap with GPU/other CPU work.',
                                    'Decode spread and this small sample do not establish a small gain/regression.',
                                    'The candidate has four-fresh32K full live-state numerical proof, but has not passed full262144 lifecycle.']}
OUT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': True, 'receipt_sha256': digest(OUT), 'groups': groups, 'effects': effects,
                  'later_host_summary': host_summary}, indent=2))
