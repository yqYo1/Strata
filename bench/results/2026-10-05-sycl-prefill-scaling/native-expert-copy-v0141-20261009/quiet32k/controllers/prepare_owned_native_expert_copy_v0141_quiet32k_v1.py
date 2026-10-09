"""Prepare quiet baseline/off/on only after terminal native numerical qualification."""
from pathlib import Path
import ast
import datetime
import hashlib
import json

B = Path(__file__).parent
OLD = B / 'run_owned_host_accounting_v0141_quiet32k_v1.py'
FIRST_CONTROLLER = B / 'run_owned_native_expert_copy_v0141_code32k_v1.py'
FIRST_RECEIPT = B / 'owned-native-expert-copy-v0141-code32k-diagnostic-r1/record.json'
OUT = B / 'run_owned_native_expert_copy_v0141_quiet32k_v1.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert digest(OLD) == 'a23c23b8a55488dce3be96ec5a3daf54772739ac5d72fd69970374b32f47deb0'
assert digest(FIRST_CONTROLLER) == '3f3df622937ee0f2f6d3e201b3db40ec095b4d68691cfbe02a2f88ee49abef1c'
first = json.loads(FIRST_RECEIPT.read_text())
assert not first['active'] and first['completed'] and first['healthy'] and first['math_gate_passed']
assert first['exit_code'] == 0 and not first['exit_signal'] and not first['new_fault_messages'] and not any(first['cleanup'].values())
assert first['native_copy_queue_startup_gate_passed'] and first['native_copy_queue_ordinals'] and all(x == 1 for x in first['native_copy_queue_ordinals'])
assert len(first['requests']) == 4 and all(x['math_gate_passed'] for x in first['requests'])
assert not OUT.exists()
original = OLD.read_text()
text = original
native = FIRST_CONTROLLER.read_text()


def replace(old, new):
    global text
    assert text.count(old) == 1, old[:120]
    text = text.replace(old, new)


replace("mode in ['baseline','hostoff','hoston']", "mode in ['baseline','nativeoff','nativeon']")
replace("review_head='16b1b04ffd77101c4b49543507c06fc71fbf2368'\nengine_commit='1eb89482a4afd20277ae0405780ed4f8eb98eb20'\nroot=observer.parent/'sync-upstream-v0.1.41-20261009'",
        "candidate_head='6caa1421f9212750a425fe1729139ffdde6e9f9a'\ncandidate_root=observer.parent/'perf-sycl-prefill-native-copy-v0141-20261009'\nreview_head='fe96101e2c3c112d1b7a36caaff66ceda80334ff' if mode=='baseline' else candidate_head\nengine_commit='1eb89482a4afd20277ae0405780ed4f8eb98eb20' if mode=='baseline' else candidate_head\nroot=observer.parent/'sync-upstream-v0.1.41-20261009' if mode=='baseline' else candidate_root")
start = text.index("build_path=base/'host-prefill-accounting-v0141-private-build-v1/record.json'")
stop = text.index("state_comparator=base/'compare_live_prefill_state_v01402_v2.py'", start)
native_start = native.index("build_path=base/'native-expert-copy-v0141-private-build-v1/record.json'")
native_stop = native.index("state_comparator=base/'compare_live_prefill_state_v01402_v2.py'", native_start)
bindings = native[native_start:native_stop].replace("build['source_head']==review_head", "build['source_head']==candidate_head")
bindings = bindings.replace("candidate_source=root/", "candidate_source=candidate_root/").replace("candidate_header=root/", "candidate_header=candidate_root/")
bindings += f'''first_path=base/'owned-native-expert-copy-v0141-code32k-diagnostic-r1/record.json'
assert digest(first_path)=='{digest(FIRST_RECEIPT)}'
first=json.loads(first_path.read_text());terminal_owned(first)
assert first['native_copy_queue_startup_gate_passed'] and len(first['requests'])==4
assert first['native_copy_queue_ordinals'] and all(x==1 for x in first['native_copy_queue_ordinals'])
assert first['binary_sha256']==build['binary_sha256']
if mode=='baseline':
    baseline_path=base/'integrated-upstream-v0.1.41-20261009-v2/build-record.json'
    baseline=json.loads(baseline_path.read_text());assert baseline['passed'] and not baseline['active']
    binary=Path(baseline['binary']);assert digest(binary)==baseline['binary_sha256']==diagnostic['binary_sha256']
    build_path=baseline_path
for previous_path in base.glob('owned-native-expert-copy-v0141-code32k-*-clean-r*/record.json'):
    terminal_owned(json.loads(previous_path.read_text()))
'''
text = text[:start] + bindings + text[stop:]
replace("out=base/f'owned-host-accounting-v0141-code32k-{mode}-clean-r{repetition}'", "out=base/f'owned-native-expert-copy-v0141-code32k-{mode}-clean-r{repetition}'")
replace("Quiet host accounting comparison: qualified integrated baseline versus private host-counter code disabled0/enabled1. Four fresh32768 A/B/A/B with64 outputs and own qualified numerical references. No API logs, dumps, native event profiling or additional waits. First process read separate from later full reads; no adoption/full-capacity claim.",
        "Quiet native expert-copy comparison: qualified baseline869 versus uniformly rebuilt candidate272 with native queue disabled0/enabled1. Four fresh32768 A/B/A/B and64 outputs with qualified numerical references. No API logs, profiler, dumps, timestamp queries or extra phase waits. First/later separated. No adoption/full-capacity claim.")
replace("'host_cpu_receipt_sha256':digest(host_cpu_path)", "'candidate_header_sha256':digest(candidate_header) if mode!='baseline' else None,'source_preparation_receipt_sha256':digest(host_cpu_path),'uniform_flags_receipt_sha256':digest(flags_path)")
replace("'first_host_accounting_diagnostic_receipt_sha256':digest(first_path)", "'first_native_copy_diagnostic_receipt_sha256':digest(first_path)")
replace("state_comparator,candidate_source]", "state_comparator,candidate_source,candidate_header,flags_path]")
replace("if mode!='baseline': env['STRATA_PREFILL_HOST_TIMING']='1' if mode=='hoston' else '0'",
        "env.pop('STRATA_PREFILL_HOST_TIMING',None)\n    env.pop('STRATA_PREFILL_COPY_ENGINE',None)\n    if mode!='baseline':\n        env['STRATA_PREFILL_COPY_ENGINE']={'nativeoff':'0','nativeon':'1'}[mode]")
start = text.index('    reports=[]\n')
stop = text.index("    record['engine_log_bytes']", start)
native_start = native.index('    queue_lines=[')
native_stop = native.index("    record['engine_log_bytes']", native_start)
parser = native[native_start:native_stop]
parser_start = parser.index("    assert queue_lines,")
prefix, checks = parser[:parser_start], parser[parser_start:]
text = text[:start] + prefix + "    if mode=='nativeon':\n" + ''.join('    '+line+'\n' for line in checks.splitlines()) + "    else:\n        assert not queue_lines, 'native queue must be absent for baseline/off'\n        record['native_copy_queue_ordinals']=[]\n        record['native_copy_queue_startup_gate_passed']=True\n" + text[stop:]
old_functions = {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
new_functions = {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
assert old_functions == new_functions
compile(text, str(OUT), 'exec')
OUT.write_text(text)
receipt = B / 'native-expert-copy-v0141-quiet-controller-preparation-v1.json'
record = {'active':False,'prepared':True,'gpu_executed':False,
          'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'original_quiet_controller_sha256':digest(OLD),'controller_sha256':digest(OUT),
          'first_diagnostic_receipt_sha256':digest(FIRST_RECEIPT),
          'all_observation_function_ASTs_unchanged':list(old_functions),
          'modes':['baseline','nativeoff','nativeon'],
          'scope':'Three quiet modes,4 fresh32K inputs per process, actual8192/128 enforced; first/later separate. No GPU use in preparation. CPU-preflight for each mode is required before sequence launch.'}
receipt.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'prepared':True,'controller_sha256':digest(OUT),'receipt_sha256':digest(receipt)},indent=2))
