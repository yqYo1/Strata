"""Bind the quiet comparison to three successful CPU preflights and first use."""
from pathlib import Path
import ast
import hashlib
import json

B = Path(__file__).parent
old = B / 'run_host_prefill_accounting_v0141_quiet_sequence_v1.py'
out = B / 'run_native_expert_copy_v0141_quiet_sequence_v1.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old) == '466f4c943c6861f2cc812dc72c6938abda978e656f9a6e0d048b33a3ec1257c1'
preflight = B / 'native-expert-copy-v0141-quiet-cpu-preflight-v1.json'
assert sha(preflight) == '7b21a22fb06bef5a831aa1afc1c823c4916e6d2259c07bb8959aa8c2ae9a9483'
cpu = json.loads(preflight.read_text())
assert cpu['passed'] and not cpu['active'] and not cpu['gpu_executed']
assert [x['mode'] for x in cpu['cases']] == ['baseline', 'nativeoff', 'nativeon']
assert all(x['exit_code'] == 0 and not x['stderr'] for x in cpu['cases'])
text = old.read_text()

def replace(before, after):
    global text
    assert text.count(before) == 1, before
    text = text.replace(before, after)

for before, after in {
    'run_owned_host_accounting_v0141_quiet32k_v1.py': 'run_owned_native_expert_copy_v0141_quiet32k_v1.py',
    'a23c23b8a55488dce3be96ec5a3daf54772739ac5d72fd69970374b32f47deb0': '0904960ef87f84eee795fb5b5105fd20e5468bcb8bc3c2c9c432627dc95700c8',
    'owned-host-accounting-v0141-code32k-diagnostic-r1/record.json': 'owned-native-expert-copy-v0141-code32k-diagnostic-r1/record.json',
    '6605f3c24444b6069ff10ab594e429ee66c6498a8a9f0cc21d6fcd33e388b2ed': '83670ee10b75d702c621e0ce029c5be1a4701ac6684d08b4f22de905704ae908',
    'host-prefill-accounting-v0141-quiet-controller-preparation-v1.json': 'native-expert-copy-v0141-quiet-cpu-preflight-v1.json',
    '442f209105f26402acd510fde195adf3cd2f7f8e0c8c212c95996623325c8f54': '7b21a22fb06bef5a831aa1afc1c823c4916e6d2259c07bb8959aa8c2ae9a9483',
    'host-prefill-accounting-v0141-quiet32k-comparison-sequence-v1': 'native-expert-copy-v0141-quiet32k-comparison-sequence-v1',
    'owned-host-accounting-v0141-code32k-{mode}-clean-r{rep}/record.json': 'owned-native-expert-copy-v0141-code32k-{mode}-clean-r{rep}/record.json',
}.items():
    replace(before, after)
text = text.replace('host_accounting_gate_passed', 'native_copy_queue_startup_gate_passed')
text = text.replace('hostoff', 'nativeoff').replace('hoston', 'nativeon')
text = text.replace('STRATA_PREFILL_HOST_TIMING', 'STRATA_PREFILL_COPY_ENGINE')
replace("Same-day quiet baseline/host-counter-off/host-counter-on/on/off/baseline. Four fresh32768 A/B/A/B and64 outputs per process, context262144, actual8192 chunks and128 expert slots. First/later separated. No native event queries, profiler, extra waits, API logging or dumps. Instrumented code is private and not adopted.",
        "Same-day quiet baseline/native-copy-off/native-copy-on/on/off/baseline. Four fresh32768 A/B/A/B and64 outputs per process, context262144, actual8192 chunks and128 expert slots. First/later separated. No native timestamp queries, profiler, extra phase waits, API logging or dumps. Candidate remains private and not adopted.")
replace("    record['host_reports'] = [r for d in groups['nativeon'] for r in d['host_accounting_reports']]\n    assert len(record['host_reports']) == 8\n", '')
replace('configuration_equal_except_explicit_host_accounting_flag', 'configuration_equal_except_explicit_native_copy_flag')
replace("    assert data['performance_eligible'] and data['native_copy_queue_startup_gate_passed']",
        "    assert data['performance_eligible'] and data['native_copy_queue_startup_gate_passed']\n    assert data['boot_id'] == first['boot_id']\n    assert all(x == 1 for x in data['native_copy_queue_ordinals'])\n    assert bool(data['native_copy_queue_ordinals']) == (data['mode'] == 'nativeon')")
replace("            flag = env.pop('STRATA_PREFILL_COPY_ENGINE', None)",
        "            assert 'STRATA_PREFILL_HOST_TIMING' not in env\n            flag = env.pop('STRATA_PREFILL_COPY_ENGINE', None)")
assert not out.exists()
ast.parse(text)
compile(text, str(out), 'exec')
out.write_text(text)
record = {'active': False, 'prepared': True, 'gpu_executed': False,
          'original_sequence_sha256': sha(old), 'sequence_sha256': sha(out),
          'cpu_preflight_receipt_sha256': sha(preflight),
          'controller_sha256': cpu['controller_sha256'],
          'scope': 'Six fresh processes in reversed order,24 quiet32K reads, all numerical and normal-exit gates retained. Preparation itself uses no GPU.'}
p = B / 'native-expert-copy-v0141-quiet-sequence-preparation-v1.json'
assert not p.exists()
p.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
