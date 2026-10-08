"""Freeze matched quiet >=32K controls after both logged correctness gates."""
from pathlib import Path
import ast,hashlib,json,datetime,difflib
base=Path(__file__).parent
parent=base/'run_owned_kv_layer_major32k_v01402_v1.py';text=parent.read_text();s=text
def replace(a,b):
    global s
    assert s.count(a)==1,a[:100];s=s.replace(a,b)
replace("assert mode=='kv-layer-major32k' and phase in ['diagnostic','state'] and repetition in [1,2,3]", "assert mode in ['kv-stream32k-quiet','kv-layer-major32k-quiet'] and phase=='quiet' and repetition in [1,2]")
replace("    binary=root/'build-sycl-kv-layer-major-v1-20261008/strata'", """    layer_gate_path=base/'owned-kv-layer-major32k-v01402-code32k-diagnostic-r1/record.json'
    completed_layer=json.loads(layer_gate_path.read_text())
    assert completed_layer['healthy'] and completed_layer['completed'] and completed_layer['math_gate_passed'] and not completed_layer['active']
    assert len(completed_layer['requests'])==3 and all(x['math_gate_passed'] for x in completed_layer['requests'])
    assert completed_layer['exit_code']==0 and not completed_layer['new_fault_messages']
    assert not any(completed_layer['cleanup'][x] for x in ['inferior_survived','gdb_survived'])
    for key in ['inferior','debugger']:
        old=completed_layer[key];now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks']
    record['layer_major_stream_gate_sha256']=hashlib.sha256(layer_gate_path.read_bytes()).hexdigest()
    is_layer=mode=='kv-layer-major32k-quiet'
    binary=root/('build-sycl-kv-layer-major-v1-20261008/strata' if is_layer else 'build-sycl-kv-stream-safe-v3-20261008/strata')""")
replace("kv_receipt=base/'kv-layer-major-v01402-build-v1/record.json'", "kv_receipt=base/('kv-layer-major-v01402-build-v1/record.json' if is_layer else 'kv-stream-safe-v01402-build-v3/record.json')")
replace("kv_source=binary.parent/'source/prefill.cpp'", "kv_source=binary.parent/('source/prefill.cpp' if is_layer else 'source/kv_stream.dp.cpp')")
replace("review=base/'kv-layer-major32k-v01402-source-review-v1/record.json'", "review=base/('kv-layer-major32k-v01402-source-review-v1/record.json' if is_layer else 'kv-stream32k-v01402-source-review-v2/record.json')")
replace("list(base.glob('owned-kv-layer-major32k-v01402-code32k-*/record.json')):", "list(base.glob('owned-kv-layer-major32k-v01402-code32k-*/record.json'))+list(base.glob('owned-kv-*-quiet-v01402-code32k-quiet-*/record.json')):")
before="""    env.update(STRATA_PREFILL_LAYER_MAJOR='2',STRATA_KV_STAGE_OWN='1',STRATA_KV_PREFETCH='0',
               STRATA_PREFILL_RELEASE_DRAFT='1',STRATA_PREFILL_DRAFT_VERIFY='1',
               STRATA_PREFILL_LAYER_MAJOR_R_GPU='0',STRATA_PREFILL_LAYER_MAJOR_R_INPLACE='1',
               STRATA_PREFILL_RELEASE_CACHE='1',STRATA_PREFILL_CACHE_ALLOC='vmm',
               STRATA_PREFILL_CACHE_RESTORE='ram',STRATA_PREFILL_CACHE_RELEASE_FRAC='1',
               STRATA_PREFILL_CACHE_VERIFY='1')
    assert env['STRATA_PREFILL_LAYER_MAJOR_R_GPU']=='0'
    assert env['STRATA_KV_STAGE_OWN']=='1' and env['STRATA_KV_PREFETCH']=='0'"""
after="""    env.update(STRATA_PREFILL_LAYER_MAJOR='2' if is_layer else '0',STRATA_KV_STAGE_OWN='1',STRATA_KV_PREFETCH='0')
    if is_layer:
        env.update(STRATA_PREFILL_RELEASE_DRAFT='1',
                   STRATA_PREFILL_LAYER_MAJOR_R_GPU='0',STRATA_PREFILL_LAYER_MAJOR_R_INPLACE='1',
                   STRATA_PREFILL_RELEASE_CACHE='1',STRATA_PREFILL_CACHE_ALLOC='vmm',
                   STRATA_PREFILL_CACHE_RESTORE='ram',STRATA_PREFILL_CACHE_RELEASE_FRAC='1')
    assert env['STRATA_KV_STAGE_OWN']=='1' and env['STRATA_KV_PREFETCH']=='0'"""
replace(before,after)
replace("    env['STRATA_TRACE']='1'","    assert 'STRATA_TRACE' not in env")
replace("    env['STRATA_DUMP_FIRST_LOGITS']=str(out/'first-head.bin')\n    env['STRATA_PREFILL_DUMP_STATE']=str(out/'prefill-state.bin')", """    assert not any(k in env for k in [
        'STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE','STRATA_PREFILL_CACHE_VERIFY',
        'STRATA_PREFILL_DRAFT_VERIFY','STRATA_PREFILL_TIMING','STRATA_PROFILE','STRATA_VERIFY_PROFILE'])""")
begin=s.index("        head=out/'first-head.bin';assert head.exists()")
end=s.index("\n    current=None;send(b'QUIT",begin)
s=s[:begin]+"""        baseline=json.loads((base/'owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]
        result['comparison_to_default_counter_control']={
            'ids_equal':result['ids']==baseline['ids'],
            'logprobs_equal':result['logprobs']==baseline['logprobs']}
        result['mtp_counts']=list(map(int,result['protocol'][-1].split()[6:8]))
        result['no_prompt_reuse']=any(line in ['RESUME 0','REUSED 0'] for line in result['protocol']) and not any(line.startswith(('RESUME ','REUSED ')) and line.split()[1]!='0' for line in result['protocol'])
        result['math_gate_passed']=all(result['comparison_to_default_counter_control'].values()) and result['mtp_counts']==[43,66] and result['no_prompt_reuse']
        record['requests'].append(result);save()
        if not result['math_gate_passed']:break
""" +s[end:]
replace("    cursor=m.journal_cursor(r,'kernel-before');save()", """    record['scope']='Matched quiet controls: context262144, kv-resident32768, same8192/int8/cache128/workers5/pcie0/MTP4/GEN64/32768 fixture. Compare existing chunk-major versus streamed layer-major with RAM residual/main+MTP leases. Three completely fresh reads per process; first startup/JIT/capture separate from reads2/3. No model API logs, state/head/payload validation, trace/profiler, extra waits or prompt reuse. All64IDs/logprobs/MTP43/66 still exact. Prior logged full-state/head correctness required. Not full262144 occupancy or production adoption.'
    record['configuration_note']=record['scope']
    record['previous_goal_turn']='Progress: defined KV representation/page-claim candidate and both streamed processing-order configurations must have terminal3x32K math/noXE/no-survivor receipts before quiet comparison.'
    cursor=m.journal_cursor(r,'kernel-before');save()""")
ast.parse(s)
target=base/'run_owned_kv_matched_quiet32k_v01402_v1.py';assert not target.exists();target.write_text(s)
out=base/'kv-matched-quiet32k-v01402-controller-v1';out.mkdir(mode=0o700)
r=dict(active=False,prepared=True,gpu_tested=False,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),scope='Prepared matched quiet comparator only; runtime requires both logged3x32K terminal gates. Planned ABBA fresh processes, three full32768 reads each. Initial and repeated timings reported separately.')
(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
(out/'controller.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),s.splitlines(True),fromfile=str(parent),tofile=str(target))))
print(json.dumps(r,indent=2))
