"""Archive terminal evidence and immutable reproduction inputs, never live logs."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
dest=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k/visible-output-commit-capacity-gate-v1'
dest.mkdir();(dest/'controllers').mkdir();(dest/'source').mkdir();(dest/'runs').mkdir();(dest/'checks').mkdir()
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(p,q):
    q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists();shutil.copy2(p,q);assert digest(p)==digest(q)
controllers=['build_kv_access_safe_v01402_v1.py','build_kv_access_safe_v01402_v2.py',
 'build_visible_commit_v01402_v1.py','read_saved_session_v01402_v1.py','read_saved_session_v01402_v2.py',
 'prepare_saved_session_reader_v01402_v2.py','check_saved_session_reader_v01402_v1.py','check_saved_session_reader_v01402_v2.py',
 'prepare_owned_full_kv_access_v01402_v1.py','prepare_owned_full_kv_access_v01402_v2.py',
 'prepare_owned_full_visible_commit_v01402_v3.py','prepare_owned_full_visible_commit_v01402_v4.py',
 'run_owned_full_kv_access_v01402_v1.py','run_owned_full_kv_access_v01402_v2.py',
 'run_owned_full_visible_commit_v01402_v3.py','run_owned_full_visible_commit_v01402_v4.py',
 'check_health_after_save_assert_v01402_v1.py','check_health_after_controller_assert_v01402_v2.py',Path(__file__).name]
for name in controllers:copy(base/name,dest/'controllers'/name)
checks=['kv-access-safe-v01402-build-v1','kv-access-safe-v01402-build-v2','visible-commit-v01402-build-v1',
 'saved-session-reader-v01402-host-check-v1','saved-session-reader-v01402-host-check-v2',
 'saved-session-reader-v01402-real-file-v2','full-kv-access-v01402-source-review-v1',
 'full-kv-access-v01402-source-review-v2','full-visible-commit-v01402-source-review-v3',
 'full-visible-commit-v01402-source-review-v4','post-save-assert-v01402-health-v1','post-controller-assert-v01402-health-v2']
for name in checks:
    receipt=base/name/'record.json';record=json.loads(receipt.read_text());assert not record['active']
    for p in sorted((base/name).iterdir()):
        if p.is_file() and p.name!='prefix-check' and p.stat().st_size<8*1024**2:
            copy(p,dest/'checks'/name/p.name)
terminal=[];private=[]
live_path=base/'owned-full-kv-access-v01402-full256k-diagnostic-r4/record.json'
live=json.loads(live_path.read_text())
assert live['active'] and live['active_request']=='full256k-first'
assert live['requests'][0]['math_gate_passed'] and live['requests'][0]['name']=='control32k-before'
assert live['requests'][1]['name']=='resume32k-reference' and max(live['requests'][1]['resume_tokens'])==32831
assert live['sessions'][0]['passed'] and live['sessions'][0]['tokens']==32831
assert live['sessions'][0]['saved_ids_match_exact_consumed_visible_prefix']
initial={k:live[k] for k in ['started_utc','boot_id','environment','argv','binary_sha256','inferior','debugger']}
initial.update(active=False,completed=True,scope='Immutable initial32K correctness and saved-prefix evidence from the still-running full-capacity process. Not whole-process completion, GPU cleanup, full256K or speed proof.',
               requests=live['requests'][:2],sessions=live['sessions'][:1],initial32k_math_passed=True,
               parent_live_receipt=str(live_path),captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(dest/'initial32k-v4-snapshot.json').write_text(json.dumps(initial,indent=2)+'\n')
for rep in [1,2,3]:
    name=f'owned-full-kv-access-v01402-full256k-diagnostic-r{rep}'
    r=json.loads((base/name/'record.json').read_text());assert not r['active'] and not r['new_fault_messages']
    assert not r['cleanup']['inferior_survived'] and not r['cleanup']['gdb_survived']
    for p in sorted((base/name).rglob('*')):
        if not p.is_file():continue
        relative=p.relative_to(base/name)
        if p.suffix=='.bin' or str(relative)=='debugger/inferior.stderr':
            private.append({'file':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
        elif p.stat().st_size<8*1024**2:copy(p,dest/'runs'/name/relative)
    terminal.append({'name':name,'receipt_sha256':digest(base/name/'record.json'),'healthy':r['healthy'],
                     'completed':r['completed'],'error':r.get('error'),'math_gate_passed':r.get('math_gate_passed')})
sources={
 'layer.cpp':root/'build-sycl-kv-access-safe-v2-20261008/source/layer.cpp',
 'generate.cpp':root/'build-sycl-visible-commit-v1-20261008/source/generate.cpp',
 'kv_stream.dp.cpp':root/'build-sycl-kv-stream-safe-v3-20261008/source/kv_stream.dp.cpp',
 'prefill.cpp':root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/prefill/prefill.cpp'}
for name,p in sources.items():copy(p,dest/'source'/name)
old_rates=json.loads((dest.parent/'kv-streaming-32k-and-layer-order-v3/summary.json').read_text())['quiet']
summary={'active':False,'completed':True,'capacity_passed':False,'adopted':False,
 'scope':'Terminal preflight evidence, visible-output commit bug and CPU-tested private fix. No new performance measurement. Active full256K job is separate and not archived as a completed test.',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'feature_head_at_start':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'terminal_runs':terminal,'minimum_comparison_input_tokens':32768,
 'prior_clean_repeated32k_rates':{
     'chunk_major_PP':old_rates['kv-stream32k-quiet']['repeated_full_reads']['prefill_tok_s'],
     'chunk_major_TG':old_rates['kv-stream32k-quiet']['repeated_full_reads']['decode_tok_s'],
     'layer_major_PP':old_rates['kv-layer-major32k-quiet']['repeated_full_reads']['prefill_tok_s'],
     'layer_major_TG':old_rates['kv-layer-major32k-quiet']['repeated_full_reads']['decode_tok_s']},
 'prior_rates_source':'../kv-streaming-32k-and-layer-order-v3/summary.json; historical quiet ABBA, not these diagnostic runs',
 'reproduced_bug':{'input_tokens':32768,'emitted_tokens':64,'saved_consumed_tokens':32833,
                   'expected_consumed_tokens':32831,'extra_committed_tokens':2,
                   'head_ids_logprobs_and_live_prefill_state_equal_to_control':True,'legacy_MTP_counts':[43,66]},
 'fix':{'sites':['serial serve','pipeline serve','CLI speculative loop'],'cpu_sanitizer_cases':12,
        'verifier_window_T_unchanged':True,'clip_before_commit':True,'limits':['max_new remaining','first emitted EOS'],
        'expected_32k_visible_MTP_counts':[41,66],
        'initial32k_gpu_gate_passed':True,'saved_consumed_tokens':32831,
        'saved_token_ids_equal_prompt_plus_outputs_except_last':True,
        'actual_live_continuation_resume_tokens':32831,
        'binary':str(root/'build-sycl-visible-commit-v1-20261008/strata'),
        'binary_sha256':'fdca351f73bd2433d45d9db7962fde7ad1a4b83f5e254751d27782fe9f5f75c6'},
 'capacity_controller':{'file':'controllers/run_owned_full_visible_commit_v01402_v4.py',
      'receipt_private':str(base/'owned-full-kv-access-v01402-full256k-diagnostic-r4/record.json'),
      'requirements':['full physical cell262143','fresh full reread with RESUME0','all13 saved KV layers including final cell',
                      'restored32K live continuation','restored256K-prefix clipped tail','capacity refusal','later full32K control'],
      'all_inputs_at_least_32768':True,'timings_excluded_from_speed':True},
 'source_authority':'actual pinned compiled overlay plus explicit private object replacements, not current worktree source',
 'production_binary_sha256':digest(root/'build-sycl-e8ca-refresh-20261007/strata'),
 'accepted_binary_sha256':digest(root/'build-sycl-event-ack-no-root-prefill-20261008/strata'),
 'no_reset_rebind_reboot_service_package_global_change':True,
 'limitations':['CPU helper/synthetic codec tests are not GPU or full-capacity proof.',
                'Unpatched upstream comparison is in the earlier archived32K baseline; this private fix is not applied to it.',
                'Pipeline/CLI branches are compiled and CPU prefix-tested, not new GPU parity-tested here.']}
(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(dest/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
files=[p for p in sorted(dest.rglob('*')) if p.is_file()]
(dest/'SHA256SUMS').write_text(''.join(digest(p)+'  '+str(p.relative_to(dest))+'\n' for p in files))
print(json.dumps({'destination':str(dest),'files':len(files)+1,'summary_sha256':digest(dest/'summary.json')},indent=2))
