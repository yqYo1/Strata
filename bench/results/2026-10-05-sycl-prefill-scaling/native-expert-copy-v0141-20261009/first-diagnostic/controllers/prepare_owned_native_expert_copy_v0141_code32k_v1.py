"""Derive an owned four-fresh32K diagnostic without changing its observation code."""
from pathlib import Path
import ast
import datetime
import hashlib
import json

B = Path(__file__).parent
SOURCE = B / 'run_owned_host_accounting_v0141_code32k_v1.py'
OUT = B / 'run_owned_native_expert_copy_v0141_code32k_v1.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert digest(SOURCE) == 'ac8938dedeafb8d449e8185d92875e9db2d0ac453a2a9b6f5ac5c514d66bc418'
assert not OUT.exists()
original = SOURCE.read_text()
text = original


def replace(old, new):
    global text
    assert text.count(old) == 1, old[:120]
    text = text.replace(old, new)


replace("mode == 'accounting'", "mode == 'nativecopy'")
replace("review_head='16b1b04ffd77101c4b49543507c06fc71fbf2368'", "review_head='6caa1421f9212750a425fe1729139ffdde6e9f9a'")
replace("engine_commit='1eb89482a4afd20277ae0405780ed4f8eb98eb20'", "engine_commit=review_head")
replace("root=observer.parent/'sync-upstream-v0.1.41-20261009'", "root=observer.parent/'perf-sycl-prefill-native-copy-v0141-20261009'")
begin = text.index("build_path=base/'host-prefill-accounting-v0141-private-build-v1/record.json'")
end = text.index("state_comparator=base/'compare_live_prefill_state_v01402_v2.py'", begin)
text = text[:begin] + '''build_path=base/'native-expert-copy-v0141-private-build-v1/record.json'
assert digest(build_path)=='74b48ad90996c942fd74fa826d22dfecf76145d8dec82a40cfa0d35dc9b65fad'
build=json.loads(build_path.read_text());assert not build['active'] and build['passed'] and build['compiled_engine'] and not build['gpu_tested'] and not build['adopted']
assert build['source_head']==review_head
binary=Path(build['binary']);assert digest(binary)==build['binary_sha256']=='2721f8ef417456c8a549cf438292ce34955a078d30974ed0200e1115ab8a3f5f'
candidate_source=root/'sycl/src/prefill/prefill.cpp'
candidate_header=root/'sycl/include/dpct/device.hpp'
assert digest(candidate_source)==build['source_sha256']['sycl/src/prefill/prefill.cpp']=='6399d65feea884f932a5a1079a91c80228293bfc3be371732b48413977229f84'
assert digest(candidate_header)==build['source_sha256']['sycl/include/dpct/device.hpp']=='8805874ecb1d58d7acea8ca931154a38d0ad3d623f7c5ee8d599fa2ba56a841b'
host_cpu_path=base/'native-expert-copy-v0141-source-preparation-v1.json'
assert digest(host_cpu_path)=='9ee58f5ebfa8f2de86c36b40b7edd0a519a123c18ec1426eb16e8930d5dcf027'
host_cpu=json.loads(host_cpu_path.read_text());assert host_cpu['prepared'] and not host_cpu['active'] and not host_cpu['adopted']
flags_path=base/'native-expert-copy-v0141-uniform-build-flags-v1.json'
assert digest(flags_path)=='54a1a491a7f00874a90af1b7776d0654b8b001e9aea3da99f6e0b1dfe0a83e53'
flags=json.loads(flags_path.read_text());assert flags['passed'] and not flags['active'] and flags['build_receipt_sha256']==digest(build_path)
assert flags['old_compile_count']==flags['candidate_compile_count']==114 and not flags['missing_sources'] and not flags['different_flags']
quiet_path=base/'host-prefill-accounting-v0141-quiet32k-comparison-sequence-v1/record.json'
assert digest(quiet_path)=='d57fba6b0826ad8402505571b8cdc92eee4e0fa5bbd1c1a9b779531bd2f4bc83'
quiet=json.loads(quiet_path.read_text());assert not quiet['active'] and quiet['passed'] and len(quiet['steps'])==6
for step in quiet['steps']:
    p=Path(step['receipt']);assert digest(p)==step['receipt_sha256']
    terminal_owned(json.loads(p.read_text()))
    identity=step['controller_identity'];actual=process_identity(identity['pid'])
    assert not actual or actual['start_ticks']!=identity['start_ticks']
assert digest('/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0.12.0')=='bfdc0f26bf88bd0bccd66fc25302a4b22559f8fe30e499be18529be3ed610e60'
''' + text[end:]
replace("out=base/'owned-host-accounting-v0141-code32k-diagnostic-r1'", "out=base/'owned-native-expert-copy-v0141-code32k-diagnostic-r1'")
replace("'deadline_seconds':3600,'protocol_timeout_seconds':700,'log_limit_bytes':64*1024**3", "'deadline_seconds':1500,'protocol_timeout_seconds':300,'log_limit_bytes':24*1024**3")
replace("First host-accounting binary qualification: four fresh32768 A/B/A/B with64 visible outputs, first logits, all66 live state parts and MTP/logprob references against qualified integrated869. Flushed UR/ZE/ZEL diagnostics; host counts only, no native event queries/extra phase waits/profiler. Logged times are excluded; no full-capacity or adoption claim.",
        "First native copy-only queue binary qualification: four fresh32768 A/B/A/B with64 visible outputs, first logits, all66 live state parts and MTP/logprob references against qualified integrated869. Flushed UR/ZE/ZEL diagnostics; no transfer profiler, extra waits, healthy-process attachment or event-timestamp queries. Existing completion checks and ring barriers are retained. Logged times are excluded; no full-capacity or adoption claim.")
replace("'candidate_source_sha256':digest(candidate_source),'host_cpu_receipt_sha256':digest(host_cpu_path)",
        "'candidate_source_sha256':digest(candidate_source),'candidate_header_sha256':digest(candidate_header),'source_preparation_receipt_sha256':digest(host_cpu_path),'uniform_flags_receipt_sha256':digest(flags_path),'completed_host_quiet_sequence_sha256':digest(quiet_path)")
replace("state_comparator,candidate_source]", "state_comparator,candidate_source,candidate_header,flags_path]")
replace("env.update(STRATA_PREFILL_HOST_TIMING='1',STRATA_DUMP_FIRST_LOGITS=", "env.update(STRATA_PREFILL_COPY_ENGINE='1',STRATA_DUMP_FIRST_LOGITS=")
replace("'STRATA_VERIFY_NO_HOST', 'STRATA_VERIFY_DEVICE_PLAN']", "'STRATA_VERIFY_NO_HOST', 'STRATA_VERIFY_DEVICE_PLAN', 'STRATA_PREFILL_HOST_TIMING']")
start = text.index('    reports=[]\n')
stop = text.index("    record['engine_log_bytes']", start)
text = text[:start] + '''    queue_lines=[value for value in (out/'project-messages.txt').read_text().splitlines() if value.startswith('strata prefill copy queue: ')]
    assert queue_lines, 'native copy-only queue creation required'
    prefix='strata prefill copy queue: native copy-only, ordinal '
    suffix=', index0, in-order, profiling0'
    ordinals=[]
    for value in queue_lines:
        assert value.startswith(prefix) and value.endswith(suffix), 'native copy-only queue metadata required'
        ordinal=value[len(prefix):-len(suffix)]
        assert ordinal.isdecimal() and int(ordinal)==1, 'recorded B570 copy-only queue ordinal required'
        ordinals.append(int(ordinal))
    record['native_copy_queue_ordinals']=ordinals
    record['native_copy_queue_startup_gate_passed']=True
    record['native_copy_queue_note']='Factory validates COPY without COMPUTE, immediate/order/native identity and existing context/device, then registers the imported queue. This metadata and four-fresh numerical gate do not establish full-capacity correctness or general stall prevention.'
''' + text[stop:]

# Keep all observation, comparisons, capture and cleanup functions unchanged.
old_nodes = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
new_nodes = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
assert old_nodes == new_nodes
compile(text, str(OUT), 'exec')
OUT.write_text(text)
record = {'active': False, 'prepared': True, 'gpu_executed': False,
          'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'original_controller_sha256': digest(SOURCE), 'candidate_controller_sha256': digest(OUT),
          'all_top_level_observation_function_ASTs_unchanged': list(old_nodes),
          'binary_sha256': '2721f8ef417456c8a549cf438292ce34955a078d30974ed0200e1115ab8a3f5f',
          'deadlines': {'total_seconds': 1500, 'request_seconds': 300, 'log_bytes': 24*1024**3},
          'notes': ['Only provenance, admitted candidate/flag, native metadata parser and bounded deadlines changed.',
                    'CPU-preflight must pass and must be inspected before launch; no GPU use from preparation.',
                    'Candidate full262144 lifecycle and quiet performance remain pending.']}
receipt = B / 'native-expert-copy-v0141-first-controller-preparation-v1.json'
receipt.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'prepared': True, 'controller_sha256': digest(OUT), 'receipt_sha256': digest(receipt)}, indent=2))
