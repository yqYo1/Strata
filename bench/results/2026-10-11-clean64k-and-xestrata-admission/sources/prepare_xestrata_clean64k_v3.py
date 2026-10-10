"""Root-only private correction/fixtures; never modifies original receipts or fork."""
from pathlib import Path
import ast, fcntl, hashlib, json, shutil
B=Path(__file__).parent
Q=B/'xestrata-clean-64k-comparison-v3'
def sha(p):
    with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
with (B/'owned-v0141-measurement.lock').open('a')as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not Q.exists();Q.mkdir(mode=0o700)
    source=B/'xestrata-clean-64k-comparison-v2/run_clean.py';text=source.read_text()
    assert sha(source)=='0c42b27b88a2118382726fedd25e9b1312e88cc22f3cbf8e81cd041229fe2900'
    text=text.replace("        load_gate(options.fork_diagnostic,options.fork_diagnostic_sha,'xe')","        if options.arm=='xe': load_gate(options.fork_diagnostic,options.fork_diagnostic_sha,'xe')")
    text=text.replace("    if arm=='baseline':","    if arm=='xe': target['STRATA_ARENA_HOST_USM']='1'\n    if arm=='baseline':")
    text=text.replace("'expert_slots':'128','pool_workers':'5'","'expert_slots':('144' if options.arm=='xe' else '128'),'pool_workers':'5'")
    text=text.replace("effective_ring_runtime_measured=False,","effective_ring_runtime_measured=False,requested_expert_cache_budget=128,\n                    expected_physical_expert_slots=144 if options.arm=='xe' else 128,\n                    fork_allocator_mode='source-supported per-layer host USM; default mmap-import failed startup',")
    text=text.replace("for p in (options.fork_diagnostic,options.baseline_diagnostic,options.owner_qualification)","for p in (options.fork_diagnostic,options.baseline_diagnostic,options.owner_qualification) if p is not None")
    compile(text,str(Q/'run_clean.py'),'exec');(Q/'run_clean.py').write_text(text)
    shutil.copyfile(source.with_name('direct_owner.py'),Q/'direct_owner.py')
    fixture_source=B/'owned-upstream-v0141-integrated-full256k-diagnostic-r2/full256k-repeat.input-tokens.txt'
    words=fixture_source.read_text().split();tokens=list(map(int,words))
    assert len(tokens)>=65536 and all(0<=x<248320 for x in tokens)
    fixture=Q/'coding-context-65536.tokens.txt';fixture.write_text(' '.join(map(str,tokens[:65536]))+'\n')
    old=B/'owned-prefill-gemm-only-code32k-diagnostic-r1/record.json'
    assert sha(old)=='dd929ca6c954163f0116ce0492dbfbdb316d727761e1cbc31f27f39eda3826da'
    r=json.loads(old.read_text());assert not r['active'] and r['exit_code']==0 and r['exit_signal']is None and r['boot_unchanged'] and not r['new_fault_messages'] and not any(r['cleanup'].values())
    assert r['completed']is False and r['healthy']is False and r['error']=='AssertionError()' and r['math_gate_passed']
    assert all(all(v is True for v in request[k].values())for request in r['requests']for k in ['validation','comparison'])
    stderr=B/'owned-prefill-gemm-only-code32k-diagnostic-r1/debugger/inferior.stderr'
    eventline=next(v.split('strata prefill service validity: ',1)[1]for v in stderr.read_text().splitlines()if 'strata prefill service validity: 'in v)
    ledger=json.loads(eventline);assert ledger['status']=='invalid' and ledger['reason']=='backward_event_timestamps' and ledger['backward']==101
    admission=Q/'baseline-math-execution-admission.json'
    admission.write_text(json.dumps(dict(schema='baseline-math-execution-admission-v1',original_receipt=pin(old),binary_sha256=r['binary_sha256'],engine_normal_exit=True,owned_closed=True,math_gate_passed=True,qualification_scope='32K execution/math only; original event-timing qualification remains invalid',event_timing_qualified=False,requests=r['requests'],original_event_ledger=ledger),indent=2)+'\n')
    manifest=dict(gpu_executed=False,model_executed=False,controller=pin(Q/'run_clean.py'),owner=pin(Q/'direct_owner.py'),parent=pin(source),fixture=pin(fixture),fixture_source=pin(fixture_source),source_input_tokens=len(tokens),selection='explicit immutable first65536 token IDs from historical fullcontext coding fixture; no runtime concatenation/truncation/reuse',original_baseline_receipt=pin(old),admission=pin(admission),scope='independent baseline admission; Xe only after healthy hostUSM32K; original default cache mismatch and invalid ledger remain FAILED')
    (Q/'preparation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(prepared=True,fixture_sha256=sha(fixture),admission_sha256=sha(admission),directory=str(Q))))
