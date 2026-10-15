#!/usr/bin/env python3
"""Main-only bounded receipt consistency checker; not a GPU supervisor or test pass."""
import argparse, hashlib, json, pathlib, re
ap=argparse.ArgumentParser()
ap.add_argument('--stdout',type=pathlib.Path,required=True)
ap.add_argument('--tokens-file',type=pathlib.Path,required=True)
ap.add_argument('--expected-count',type=int,required=True)
ap.add_argument('--context',type=int,required=True)
ap.add_argument('--exit-code',type=int,required=True)
ap.add_argument('--kv-resident',type=int,required=True)
ap.add_argument('--expect-kv-mode',type=int,choices=(0,1),required=True)
ap.add_argument('--expect-resident-cells',type=int,required=True)
a=ap.parse_args()
def need(ok,reason):
    if not ok: raise SystemExit('REJECT: '+reason)
need(a.exit_code==0,'owned child normal exit required')
need(a.stdout.stat().st_size<=64*1024*1024,'stdout budget exceeded')
need(a.tokens_file.stat().st_size<=32*1024*1024,'token input budget exceeded')
raw=a.tokens_file.read_bytes()
ids=[int(x) for x in re.split(r'[,\s]+',raw.decode().strip()) if x]
need(len(ids)==a.expected_count==a.context and 2<=a.context<=262144,'exact requested physical input/limit')
lines=[x[len('STRATA_CONSUME_PROMPT '):] for x in a.stdout.read_text().splitlines() if x.startswith('STRATA_CONSUME_PROMPT ')]
need(len(lines)==1,'exactly one terminal receipt')
r=json.loads(lines[0]);n=len(ids)
need(r['schema']==2 and r['input_count']==n and r['configured_limit']==a.context and r['exact_capacity'] is True,'input/limit receipt')
h=14695981039346656037
for token in ids:
    for byte in (token & ((1<<64)-1)).to_bytes(8,'little'):
        h=((h^byte)*1099511628211)&((1<<64)-1)
need(r['input_fnv1a64_i64le']==f'{h:016x}','ordered token fingerprint')
need(r['batched_range']==[0,n-1] and r['batched_scope']=='successful_final_layer_residual_callbacks','N-1 batched range and observation scope')
chunk=r['batched_chunk_capacity'];need(isinstance(chunk,int) and chunk>0,'actual chunk capacity')
callbacks=(n-2)//chunk+1
need(r['batched_callbacks']==callbacks and r['batched_last']==[(callbacks-1)*chunk,n-1-(callbacks-1)*chunk],'contiguous chunk completion counts')
need(r['terminal_pos']==n-1 and r['terminal_token']==ids[-1] and r['verifier_T']==1,'terminal prompt input identity')
need(r['run_ok'] is True and r['commit_rows']==1 and r['commit_ok'] is True and r['wait_ok'] is True,'terminal run/commit/drain')
need(r['completion_tokens']==0 and r['target_windows']==1 and r['draft_rounds']==0 and r['position_advanced_after_terminal'] is False,'zero output/no subsequent work')
need(r['geometry_scope']=='host_owner_geometry' and (r['layers'],r['qsa_layers'],r['gdn_layers'])==(48,12,36),'host geometry scope')
need(r['gdn_scope']=='verifier_run_and_commit_no_per_layer_position_measurement','no invented GDN position counters')
need(r['qsa_scope']=='device_indexer_last_completed_block_origin_after_commit' and r['qsa_block_origins']==[(n//4-1)*4 if n>=4 else 0]*12,'twelve bounded observed QSA block origins')
need(r['kv_lifecycle_scope']=='normal_host_authoritative_append_resolve_then_attention_commit' and r['kv_requested_resident_cells']==a.kv_resident,'actual KV request/lifecycle scope')
need(len(r['kv_layers'])==12,'twelve actual KV observations')
for q,kv in enumerate(r['kv_layers']):
    need(kv['qsa']==q and kv['mode']==a.expect_kv_mode and kv['context']==a.context and kv['resident_cells']==a.expect_resident_cells,'actual mode/resident/context differs from requested gate')
    need(kv['overflow_before']==0 and kv['overflow_after']==0,'stream resolve overflow')
    if kv['mode']==1:
        need(kv['host_authoritative'] is True and 0<a.kv_resident<=kv['resident_cells']<a.context,'host-authoritative partial slots')
        need(kv['resolve_calls_before']==0 and kv['resolve_calls_after']==1 and kv['resolve_misses']>0 and kv['resolve_lookups']>=kv['resolve_misses'],'actual fresh terminal resolve counters')
        need(kv['probe_scope']=='one_selected_prior_block_host_vs_resolved_slot_not_all_context','prior probe scope')
        need(0<=kv['probe_block']<(n-1)//4 and 0<=kv['probe_slot']<kv['resident_cells']//4,'actual selected prior block mapping')
        need(kv['format'] in (0,1,2) and kv['probe_bytes']=={0:8192,1:4224,2:2304}[kv['format']] and re.fullmatch('[0-9a-f]{16}',kv['probe_fnv1a64']),'bounded exact packed prior payload observation')
    else:
        need(kv['host_authoritative'] is False and kv['resident_cells']==((a.context+3)//4)*4 and kv['probe_bytes']==0 and kv['probe_block']==-1 and kv['resolve_calls_after']==0,'resident arm is not streamed evidence')
need(r['ple_scope']=='host_token_predecessors_after_commit' and r['ple_prev']==ids[-2:],'PLE token predecessor scope')
need(r['mtp_callback_cells']==[0,n-1] and r['mtp_next_token_ids']==[1,n] and r['mtp_callbacks']==callbacks,'distinct MTP callback input-pair coverage')
need(r['head_and_mtp_owners_attached'] is True and r['native_publications']>0,'precleanup owner publication observations')
need(r['cleanup_scope']=='existing_manual_cleanup_reached_process_exit_verified_externally','cleanup observation scope')
need(r['task_quality_tested'] is False and r['restore_tested'] is False,'quality/RESTORE claim boundary')
print(json.dumps({'receipt_consistent':True,'input_count':n,'token_file_sha256':hashlib.sha256(raw).hexdigest(),
    'stdout_sha256':hashlib.sha256(a.stdout.read_bytes()).hexdigest(),
    'scope':'receipt consistency only; supervisor must independently verify source/binary/environment, owner exit and fresh GPU faults'},sort_keys=True))
