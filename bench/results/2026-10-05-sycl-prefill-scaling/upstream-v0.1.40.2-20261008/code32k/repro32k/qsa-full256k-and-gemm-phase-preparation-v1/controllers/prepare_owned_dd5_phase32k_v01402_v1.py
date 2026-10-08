"""Prepare logged existing compute-phase timing, keeping copy timing disabled."""
from pathlib import Path
import ast, datetime, hashlib, json

base = Path(__file__).parent
parent = base/'run_owned_qsa_reduce12_code32k_v01402_v2.py'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(parent) == 'f16ec6dfc63c749e9812ece21a4183a475a955eea43d338be09926ca854fe825'
review_path = base/'existing-prefill-profiler-feasibility-v01402-source-review-v1.json'
review = json.loads(review_path.read_text())
assert review['passed'] and review['source_only'] and not review['measured_GPU_or_DMA_durations']
s = parent.read_text()
s = s.replace(s.splitlines()[0], '"""Logged qualified DD5 compute-phase32K numerical/instrumentation gate."""', 1)
begin = s.index("qsa_path=base/'qsa-reduce12-sg32-v01402-build-v1/record.json'")
end = s.index("probes=out/'probes';probes.mkdir()",begin)
top = '''preceding_path=base/'owned-qsa-reduce12-sg32-v01402-full256k-diagnostic-r1/record.json'
preceding=json.loads(preceding_path.read_text())
assert not preceding['active'], 'owned full256K job remains active; no profile process'
assert preceding['completed'] and preceding['healthy'] and preceding['math_gate_passed']
assert preceding['physical256k_sequence_completed'] and preceding['capacity_sequence_completed']
assert preceding['exit_code']==0 and not preceding['exit_signal'] and not preceding.get('error')
assert not preceding['new_fault_messages'] and not any(preceding['cleanup'].values())
for role in ['inferior','debugger']:
    owned=preceding[role];now=process_identity(owned['pid'])
    assert not now or now['start_ticks']!=owned['start_ticks']
assert preceding['boot_id']==qualified['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
out = base/f'owned-dd5-phase-v01402-code32k-{phase}-r{repetition}';out.mkdir(mode=0o700)
'''
s = s[:begin]+top+s[end:]
begin = s.index("    source_review_path=base/'qsa-reduce12-sg32-v01402-source-v3/record.json'")
end = s.index("    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()",begin)
s = s[:begin]+'''    phase_review_path=base/'existing-prefill-profiler-feasibility-v01402-source-review-v1.json'
    phase_review=json.loads(phase_review_path.read_text())
    assert phase_review['passed'] and phase_review['source_only']
    for path,expected in phase_review['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    assert binary.resolve()==Path(qualified['argv'][0]).resolve()
    record['phase_source_review_sha256']=hashlib.sha256(phase_review_path.read_bytes()).hexdigest()
    record['qualified_dd5_full256k_receipt_sha256']=hashlib.sha256(qualified_path.read_bytes()).hexdigest()
    record['preceding_qsa_full256k_receipt_sha256']=hashlib.sha256(preceding_path.read_bytes()).hexdigest()
    record['physical256k_sequence_completed']=False
''' + s[end:]
marker = "    assert 'STRATA_PREFILL_TRANSFER_TIMING' not in env\n"
assert s.count(marker) == 1
s = s.replace(marker,marker+"    env['STRATA_PREFILL_TIMING']='1'\n")
old = "    record['candidate_build_receipt_sha256']=record['qsa_build_receipt_sha256']"
assert s.count(old) == 1
s = s.replace(old,"    record['candidate_build_receipt_sha256']=record['uniform_header_build_receipt_sha256']")
old = 'Only QSA decode-attention archive member replaced on fully-qualified DD5. New SG32/transposed12-head reduction uses explicit subgroup lane/group IDs; same dot expressions, chunk64, scales, softmax/PV/merge, queues/events and host lifetimes.'
new = 'Same fully-qualified DD5 binary; existing compute-phase timer enabled, with marker events and two extra compute waits per chunk. Copy timing/queue profiling stays disabled; no interposer or source/binary change. Diagnostic timings excluded.'
assert s.count(old) == 1
s = s.replace(old,new)
old = 'Logged QSA reduce12/actual-subgroup-routing numerical gate on qualified DD5: four fresh32768-token A/B reads, baseline head/used-state/IDs/logprobs/MTP comparison, actual32K SAVE/RESTORE continuation. Diagnostic durations excluded. No full-capacity/speed/stall-fix/adoption claim; source-derived native counter capture only at owned stopped failure.'
new = 'Logged DD5 existing compute-phase instrumentation gate: four fresh32768 A/B reads with qualified head/used-state/IDs/logprobs/MTP and actual32K disk continuation. Phase intervals include host/copy gaps and instrumentation waits; all times excluded from speed comparisons. No pure GPU-busy/PCIe dominance or stall-cause claim. Profiling errors and missing/nonfinite/negative/inconsistent/wall-exceeding reports reject instrumentation.'
assert s.count(old) == 1
s = s.replace(old,new)
old = "    record['previous_goal_turn']='Progress: sixteen matched quiet32768 reads passed full output/logprob/MTP/freshness/normal-exit gates. Polling speed gain rejected; evidence committed/pushed0f103c4c. DD5 full physical gate remains required.'"
assert s.count(old) == 1
s = s.replace(old,"    record['previous_goal_turn']='Compute-phase feasibility reviewed on DD5; main stream is created separately and passed to Prefill. This instrumentation remains unexecuted at preparation and waits for the current owned QSA full lifecycle to finish. Rejected unitrace GPU timestamps are not retried.'")
marker = "        windows=[]\n"
assert s.count(marker) == 1
s = s.replace(marker,"        windows=[];phase_reports=[];phase_tokens=[];profiling_errors=[]\n")
marker = "            for item in stream:\n                if item.startswith(b'strata trace: window '):"
new = '''            for item in stream:
                if item.startswith(b'strata prefill phases: '):
                    try:
                        phase_reports.append(json.loads(item.split(b': ',1)[1]))
                    except (ValueError,UnicodeError) as error:
                        if len(profiling_errors)<64:profiling_errors.append('invalid phase JSON: '+repr(error))
                if item.startswith(b'strata prefill timing: ') and b' tokens,' in item:
                    match=re.match(rb'strata prefill timing: (\\d+) tokens,',item)
                    if match:phase_tokens.append(int(match.group(1)))
                if ((b'urEventGetProfilingInfo' in item and b'UR_RESULT_ERROR_' in item) or
                    (b'zeEventQueryKernelTimestamp' in item and b'ZE_RESULT_ERROR_' in item) or
                    b'Exception caught at file:' in item):
                    if len(profiling_errors)<64:profiling_errors.append(item.decode(errors='replace').strip())
                if item.startswith(b'strata trace: window '):'''
assert s.count(marker) == 1
s = s.replace(marker,new)
marker = "        math_ok=math_ok and result['math_gate_passed'];save()\n        return result\n"
new = '''        math_ok=math_ok and result['math_gate_passed']
        if fresh:
            profile_checks={'reported_complete_prefill_prefix':len(tokens)-1 in phase_tokens,
                            'phase_reports_present':bool(phase_reports),'no_profiling_errors':not profiling_errors,
                            'finite_nonnegative_consistent_timeline':True,'timeline_not_exceed_request_wall':True}
            for report in phase_reports:
                values=report.get('phase_ms',{})
                timeline=report.get('gpu_timeline_ms',float('nan'))
                profile_checks['finite_nonnegative_consistent_timeline'] &= bool(values) and all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in values.values()) and isinstance(timeline,(int,float)) and math.isfinite(timeline) and timeline>0 and abs(sum(values.values())-timeline)<0.001
                profile_checks['timeline_not_exceed_request_wall'] &= isinstance(timeline,(int,float)) and math.isfinite(timeline) and timeline<=result['diagnostic_wall_seconds']*1000+1000
            result['phase_reports']=phase_reports;result['phase_prefix_tokens']=phase_tokens
            result['profiling_errors']=profiling_errors;result['profiling_checks']=profile_checks
            result['profiling_gate_passed']=all(profile_checks.values());save()
            if not result['profiling_gate_passed']:raise PhaseProfileRejected('compute-phase instrumentation invalid; normal QUIT, no speed conclusion')
        else:save()
        return result
'''
assert s.count(marker) == 1
s = s.replace(marker,new)
marker = "    def reject():\n"
assert s.count(marker) == 1
s = s.replace(marker,"    class PhaseProfileRejected(ValueError):\n        pass\n\n"+marker)
old = "    except ValueError as rejected:\n        if not math_ok:\n            record['mathematical_rejection']=str(rejected)\n        else:raise\n"
new = "    except ValueError as rejected:\n        if isinstance(rejected,PhaseProfileRejected):\n            record['profiling_rejection']=str(rejected)\n        elif not math_ok:\n            record['mathematical_rejection']=str(rejected)\n        else:raise\n"
assert s.count(old) == 1
s = s.replace(old,new)
marker = "    record['math_gate_passed']=math_ok and bool(record.get('code32k_sequence_completed')) and all(req['math_gate_passed'] for req in record['requests'])\n"
assert s.count(marker) == 1
s = s.replace(marker,marker+"    fresh_requests=[req for req in record['requests'] if req.get('name') in ['control32k-before','alternate32k-first','control32k-repeat','alternate32k-repeat']]\n    record['profiling_gate_passed']=len(fresh_requests)==4 and all(req.get('profiling_gate_passed') for req in fresh_requests)\n")
ast.parse(s)
assert 'qsa_build' not in s and 'qsa_path' not in s and 'cpu_semantic_routing_cases' not in s
assert "repetition==1" in s and 'repetition==7' not in s
assert s.index("assert not preceding['active']") < s.index("out = base/f'owned-dd5-phase")
assert s.index("env['STRATA_PREFILL_TIMING']='1'") < s.index("record['environment']")
assert s.count('capture_native_counter(g,snap,out') == 1
assert "request('alternate32k-first',alternate,64,dd5_alternate,fresh=True)" in s
target = base/'run_owned_dd5_phase32k_v01402_v1.py'
assert not target.exists()
target.write_text(s)
record = {'active':False,'prepared':True,'engine_built_by_preparer':False,'gpu_launched':False,
          'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preparer_sha256':sha(__file__),
          'parent_numerical_controller_sha256':sha(parent),'source_review_sha256':sha(review_path),
          'controller':str(target),'controller_sha256':sha(target),
          'checks':{'AST':True,'same_qualified_DD5_binary':True,'current_full_terminal_owned_absence_before_output_directory':True,
                    'four_fresh32768_A_B_numerical_and_actual_disk_gates':True,'existing_compute_phase_flag_only':True,
                    'copy_timing_sync_extra_interposer_disabled':True,'profiling_error_and_timeline_rejection':True,
                    'normal_QUIT_on_instrumentation_rejection':True,'initial_diagnostic_logs_and_owned_failure_capture':True},
          'minimum_performance_input_tokens':32768,
          'scope':'Prepared only. No phase profile executed, no clean speed or PCIe dominance conclusion, no source/binary change or production adoption.'}
p = base/'prepare-owned-dd5-phase32k-v01402-v1.json'
assert not p.exists()
p.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'prepared':True,'controller_sha256':sha(target),'gpu_launched':False}))
