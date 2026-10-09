"""Summarize only a normal terminal24-read native-copy comparison."""
from pathlib import Path
import datetime
import hashlib
import json
import statistics

B = Path(__file__).parent
SEQ = B / 'native-expert-copy-v0141-quiet32k-comparison-sequence-v1/record.json'
OUT = B / 'native-expert-copy-v0141-quiet32k-summary-v1.json'
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def alive(ident):
    try:
        text = (Path('/proc') / str(ident['pid']) / 'stat').read_text()
    except FileNotFoundError:
        return False
    return int(text.rsplit(')', 1)[1].split()[19]) == ident['start_ticks']

assert not OUT.exists()
s = json.loads(SEQ.read_text())
assert not s['active'] and s['passed'] and len(s['steps']) == 6
assert s['configuration_equal_except_explicit_native_copy_flag']
rows, jobs = [], []
configuration = None
for step in s['steps']:
    p = Path(step['receipt'])
    assert digest(p) == step['receipt_sha256'] and step['exit_code'] == 0
    d = json.loads(p.read_text())
    assert not d['active'] and d['completed'] and d['healthy'] and d['math_gate_passed']
    assert d['native_copy_queue_startup_gate_passed']
    assert bool(d['native_copy_queue_ordinals']) == (step['mode'] == 'nativeon')
    assert all(x == 1 for x in d['native_copy_queue_ordinals'])
    assert d['exit_code'] == 0 and not d['exit_signal'] and not d['new_fault_messages'] and not any(d['cleanup'].values())
    assert not any(alive(d[k]) for k in ('inferior', 'debugger')) and not alive(step['controller_identity'])
    assert d['phase'] == 'clean' and len(d['requests']) == 4 and d['performance_eligible']
    env = dict(d['environment'])
    flag = env.pop('STRATA_PREFILL_COPY_ENGINE', None)
    assert flag == {'baseline': None, 'nativeoff': '0', 'nativeon': '1'}[step['mode']]
    assert 'STRATA_PREFILL_HOST_TIMING' not in env
    assert not any(k.startswith(('UR_LOG_', 'ZE_ENABLE_', 'ZEL_')) or k in
                   ('UR_ENABLE_LAYERS', 'STRATA_TRACE', 'STRATA_DUMP_FIRST_LOGITS', 'STRATA_PREFILL_DUMP_STATE', 'STRATA_PREFILL_TRANSFER_TIMING') for k in env)
    args = list(d['argv']); args[0] = '<measured-binary>'
    normalized = {'argv': args, 'environment_except_native_copy_flag': env}
    if configuration is None:
        configuration = normalized
    else:
        assert normalized == configuration
    assert d['actual_target_environment'] == d['environment']
    assert d['actual_executable_identity']['sha256'] == d['binary_sha256']
    infos = [line for line in d['startup'] if line.startswith('INFO ')]
    assert len(infos) == 1
    fields = dict(part.split('=', 1) for part in infos[0].split()[1:] if '=' in part)
    assert fields['context'] == '262144' and fields['expert_slots'] == '128'
    assert fields['expert_cache_mib'] == '325' and fields['kv_resident'] == '32768'
    assert d['binary_sha256'] == {
        'baseline': '86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8',
        'nativeoff': '2721f8ef417456c8a549cf438292ce34955a078d30974ed0200e1115ab8a3f5f',
        'nativeon': '2721f8ef417456c8a549cf438292ce34955a078d30974ed0200e1115ab8a3f5f'}[step['mode']]
    for i, request in enumerate(d['requests']):
        assert [int(line.split()[1]) for line in request['protocol'] if line.startswith('PP ')] == [8192, 16384, 24576, 32767]
        assert request['math_gate_passed'] and all(request['validation'].values())
        assert all(request['qualified_physical_reference_output_comparison'].values())
        if i >= 2:
            assert all(request['repeat_comparison'].values())
        m = request['measurement']
        assert m['prompt_tokens'] == 32768 and m['generated_tokens'] == 64 and m['purpose'] == 'quiet timing'
        rows.append({'mode': step['mode'], 'repetition': step['repetition'], 'request': request['name'],
                     'first_process_read': request['first_process_read'], **m})
    jobs.append({'mode': step['mode'], 'repetition': step['repetition'],
                 'first_prefill_tok_s': d['requests'][0]['measurement']['prefill_tok_s'],
                 'later_prefill_tok_s': 32768 * 3 * 1000 / sum(x['measurement']['prompt_ms'] for x in d['requests'][1:]),
                 'later_decode_tok_s': 64 * 3 * 1000 / sum(x['measurement']['decode_ms'] for x in d['requests'][1:]),
                 'receipt_sha256': digest(p)})
assert len(rows) == 24

def stats(selected):
    pp = [x['prefill_tok_s'] for x in selected]
    tg = [x['decode_tok_s'] for x in selected]
    return {'reads': len(selected),
            'pooled_prefill_tok_s': sum(x['prompt_tokens'] for x in selected) * 1000 / sum(x['prompt_ms'] for x in selected),
            'pooled_decode_tok_s': sum(x['generated_tokens'] for x in selected) * 1000 / sum(x['decode_ms'] for x in selected),
            'prefill_median': statistics.median(pp), 'prefill_min': min(pp), 'prefill_max': max(pp),
            'decode_median': statistics.median(tg), 'decode_min': min(tg), 'decode_max': max(tg)}

groups = {}
for mode in ('baseline', 'nativeoff', 'nativeon'):
    first = [x for x in rows if x['mode'] == mode and x['first_process_read']]
    later = [x for x in rows if x['mode'] == mode and not x['first_process_read']]
    assert len(first) == 2 and len(later) == 6
    groups[mode] = {'first': stats(first), 'later': stats(later)}
effects = {}
for mode in ('nativeoff', 'nativeon'):
    effects[mode] = {name + '_relative_to_baseline_percent': 100 * (groups[mode]['later'][name] / groups['baseline']['later'][name] - 1)
                     for name in ('pooled_prefill_tok_s', 'pooled_decode_tok_s')}
effects['nativeon_vs_nativeoff'] = {
    name + '_relative_percent': 100 * (groups['nativeon']['later'][name] / groups['nativeoff']['later'][name] - 1)
    for name in ('pooled_prefill_tok_s', 'pooled_decode_tok_s')}
for mode in groups:
    for label, sequence_label in (('first', 'first_process_read'), ('later', 'later_fresh_reads')):
        for metric in ('prefill_tok_s', 'decode_tok_s'):
            assert abs(groups[mode][label]['pooled_' + metric] - s['summary'][mode][sequence_label][metric]) < 1e-10
report = {'active': False, 'passed': True, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'Quiet same-day baseline/off/on/on/off/baseline;24 fresh32K reads,64 outputs with5 logprob alternatives each, context262144. First/later separate; actual8192 chunks and128 slots; no API logs, profiler, dumps or extra phase waits.',
          'sequence_sha256': digest(SEQ), 'summarizer_sha256': digest(__file__),
          'configuration': configuration, 'groups': groups, 'effects': effects, 'rows': rows, 'jobs': jobs,
          'candidate_full_lifecycle_passed': False, 'candidate_adopted': False,
          'interpretation_limits': ['Queue selection and its measured throughput do not measure DMA duration or an exclusive PCIe wait fraction.',
                                    'Decode includes requested5-alternative logprob readback/scoring; no-logprob generation is a separate unmeasured condition.',
                                    'Decode variation and this sample do not establish a small gain/regression.',
                                    'The candidate has four-fresh32K full live-state/head proof; its full262144 lifecycle remains pending.']}
OUT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': True, 'receipt_sha256': digest(OUT), 'groups': groups, 'effects': effects}, indent=2))
