"""Prepare a quiet baseline/accounting-off/accounting-on comparison, no GPU launch."""
from pathlib import Path
import hashlib,json
b=Path(__file__).parent
p=b/'run_owned_host_accounting_v0141_code32k_v1.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='ac8938dedeafb8d449e8185d92875e9db2d0ac453a2a9b6f5ac5c514d66bc418'
s=p.read_text()
first=b/'owned-host-accounting-v0141-code32k-diagnostic-r1/record.json';first_sha=hashlib.sha256(first.read_bytes()).hexdigest();d=json.loads(first.read_text())
assert not d['active'] and d['completed'] and d['healthy'] and d['math_gate_passed'] and d['host_accounting_gate_passed']
assert d['exit_code']==0 and not d['exit_signal'] and not any(d['cleanup'].values()) and not d['new_fault_messages']
assert len(d['requests'])==len(d['host_accounting_reports'])==4
s=s.replace("assert mode == 'accounting' and phase == 'diagnostic' and repetition == 1","assert mode in ['baseline','hostoff','hoston'] and phase == 'clean' and repetition in [1,2]",1)
needle="candidate_source=base/'host-prefill-accounting-v0141-source-v1/prefill.cpp'"
assert s.count(needle)==1
add=f'''first_path=base/'owned-host-accounting-v0141-code32k-diagnostic-r1/record.json'
assert digest(first_path)=='{first_sha}'
first=json.loads(first_path.read_text());terminal_owned(first)
assert first['host_accounting_gate_passed'] and len(first['requests'])==len(first['host_accounting_reports'])==4
assert first['binary_sha256']==build['binary_sha256']
if mode=='baseline':
    baseline_path=base/'integrated-upstream-v0.1.41-20261009-v2/build-record.json'
    baseline=json.loads(baseline_path.read_text());assert baseline['passed'] and not baseline['active']
    binary=Path(baseline['binary']);assert digest(binary)==baseline['binary_sha256']==diagnostic['binary_sha256']
    build_path=baseline_path
for previous_path in base.glob('owned-host-accounting-v0141-code32k-*-clean-r*/record.json'):
    terminal_owned(json.loads(previous_path.read_text()))
'''
s=s.replace(needle,add+needle,1)
s=s.replace("out=base/'owned-host-accounting-v0141-code32k-diagnostic-r1'","out=base/f'owned-host-accounting-v0141-code32k-{mode}-clean-r{repetition}'",1)
s=s.replace("'deadline_seconds':3600,'protocol_timeout_seconds':700,'log_limit_bytes':64*1024**3","'deadline_seconds':1500,'protocol_timeout_seconds':300,'log_limit_bytes':128*1024**2",1)
s=s.replace("'performance_eligible':False,'full_lifecycle_passed':False,'adopted':False","'performance_eligible':True,'full_lifecycle_passed':False,'adopted':False,\n    'first_host_accounting_diagnostic_receipt_sha256':digest(first_path)",1)
start=s.index("    'scope':'First host-accounting binary qualification:")
end=s.index("\n    'started_utc':",start)
s=s[:start]+"    'scope':'Quiet host accounting comparison: qualified integrated baseline versus private host-counter code disabled0/enabled1. Four fresh32768 A/B/A/B with64 outputs and own qualified numerical references. No API logs, dumps, native event profiling or additional waits. First process read separate from later full reads; no adoption/full-capacity claim.',"+s[end:]
needle="    env.update(STRATA_PREFILL_HOST_TIMING='1',STRATA_DUMP_FIRST_LOGITS=str(out/'first-head.bin'),STRATA_PREFILL_DUMP_STATE=str(out/'prefill-state.bin'))"
assert s.count(needle)==1
s=s.replace(needle,"    if mode!='baseline': env['STRATA_PREFILL_HOST_TIMING']='1' if mode=='hoston' else '0'",1)
start=s.index("        result.update(captures(current,True))")
end=s.index("        if key in seen:",start)
s=s[:start]+"        checks = list(result['validation'].values())\n        assert [int(x.split()[1]) for x in result['protocol'] if x.startswith('PP ')]==[8192,16384,24576,32767]\n"+s[end:]
s=s.replace("    assert len(reports)==4, 'all4 fresh prefill host reports required'","    assert len(reports)==(4 if mode=='hoston' else 0), 'host reports must exactly match selected mode'",1)
s=s.replace("CPU submit and group-wait times include logging, queue dependencies and compute; not DMA active time. RAM-worker sums may overlap. Exact expert bytes/copies are useful; logged timing is not throughput evidence.","CPU submit/group-wait times include queue dependencies and compute, not DMA active time. RAM-worker sums may overlap. This quiet run provides host durations; diagnostic runs stay excluded.",1)
s=s.replace("'candidate_source_sha256':digest(candidate_source),'host_cpu_receipt_sha256':digest(host_cpu_path),","'candidate_source_sha256':digest(candidate_source) if mode!='baseline' else None,\n    'measured_prefill_source_sha256':digest(candidate_source) if mode!='baseline' else digest(root/'sycl/src/prefill/prefill.cpp'),\n    'host_cpu_receipt_sha256':digest(host_cpu_path),",1)
compile(s,'new_quiet_controller','exec')
o=b/'run_owned_host_accounting_v0141_quiet32k_v1.py';assert not o.exists();o.write_text(s)
print('quiet_controller_sha256',hashlib.sha256(o.read_bytes()).hexdigest());print('first_diagnostic_receipt_sha256',first_sha);print('GPU executed',False)
