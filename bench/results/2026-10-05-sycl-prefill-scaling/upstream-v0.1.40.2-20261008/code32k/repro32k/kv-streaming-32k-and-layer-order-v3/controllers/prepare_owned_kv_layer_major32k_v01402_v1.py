"""Prepare the second streamed configuration; require terminal first gate."""
from pathlib import Path
import datetime,hashlib,json,ast,difflib
base=Path(__file__).parent
root=Path("/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05")
parent=base/'run_owned_kv_stream32k_v01402_v2.py'
target=base/'run_owned_kv_layer_major32k_v01402_v1.py';assert not target.exists()
text=parent.read_text();s=text
def replace(a,b):
    global s
    assert s.count(a)==1,a[:100];s=s.replace(a,b)
replace("assert mode=='kv-stream32k'", "assert mode=='kv-layer-major32k'")
replace("binary=root/'build-sycl-kv-stream-safe-v3-20261008/strata'", """stream_gate=base/'owned-kv-stream32k-v01402-code32k-diagnostic-r1/record.json'
    first=json.loads(stream_gate.read_text())
    assert first['healthy'] and first['completed'] and first['math_gate_passed'] and not first['active']
    assert len(first['requests'])==3 and all(x['math_gate_passed'] for x in first['requests'])
    assert first['exit_code']==0 and not first['new_fault_messages']
    assert not any(first['cleanup'][x] for x in ['inferior_survived','gdb_survived'])
    for k in ['inferior','debugger']:
        old=first[k];now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks']
    record['chunk_major_stream_gate_sha256']=hashlib.sha256(stream_gate.read_bytes()).hexdigest()
    binary=root/'build-sycl-kv-layer-major-v1-20261008/strata'""")
replace("kv_receipt=base/'kv-stream-safe-v01402-build-v3/record.json'", "kv_receipt=base/'kv-layer-major-v01402-build-v1/record.json'")
replace("kv_source=binary.parent/'source/kv_stream.dp.cpp'", "kv_source=binary.parent/'source/prefill.cpp'")
replace("review=base/'kv-stream32k-v01402-source-review-v2/record.json'", "review=base/'kv-layer-major32k-v01402-source-review-v1/record.json'")
replace("list(base.glob('owned-kv-stream32k-v01402-code32k-*/record.json')):", "list(base.glob('owned-kv-stream32k-v01402-code32k-*/record.json'))+list(base.glob('owned-kv-layer-major32k-v01402-code32k-*/record.json')):")
replace("env.update(STRATA_PREFILL_LAYER_MAJOR='0',STRATA_KV_STAGE_OWN='1',STRATA_KV_PREFETCH='0')", """env.update(STRATA_PREFILL_LAYER_MAJOR='2',STRATA_KV_STAGE_OWN='1',STRATA_KV_PREFETCH='0',
               STRATA_PREFILL_RELEASE_DRAFT='1',STRATA_PREFILL_DRAFT_VERIFY='1',
               STRATA_PREFILL_LAYER_MAJOR_R_GPU='0',STRATA_PREFILL_LAYER_MAJOR_R_INPLACE='1',
               STRATA_PREFILL_RELEASE_CACHE='1',STRATA_PREFILL_CACHE_ALLOC='vmm',
               STRATA_PREFILL_CACHE_RESTORE='ram',STRATA_PREFILL_CACHE_RELEASE_FRAC='1',
               STRATA_PREFILL_CACHE_VERIFY='1')""")
replace("""    # Streaming uses its existing chunk-major path; no dormant layer-major leases.
    assert not any(k in env for k in [
        'STRATA_PREFILL_RELEASE_CACHE','STRATA_PREFILL_RELEASE_DRAFT',
        'STRATA_PREFILL_CACHE_ALLOC','STRATA_PREFILL_LAYER_MAJOR_R_GPU'])""", """    assert env['STRATA_PREFILL_LAYER_MAJOR_R_GPU']=='0'
    assert env['STRATA_KV_STAGE_OWN']=='1' and env['STRATA_KV_PREFETCH']=='0'""")
replace("KV streaming candidate, chunk-major, context262144, resident32768, own staging, no prefetch or layer-major cache leases.", "KV streaming plus layer-major candidate, context262144, resident32768, own staging, no prefetch, RAM residuals, full RAM main-cache/MTP leases.")
replace("in one logged/validated normal-MTP process on private v3 KV binary.", "in one logged/validated normal-MTP process on private v1 streamed layer-major +v3 KV binary.")
replace("Context262144/kv-resident32768 (main mode1, MTP ring), own stage, chunk-major0, no cache release or prefetch;", "Context262144/kv-resident32768 (main mode1, MTP ring), own stage, layer-major2, full RAM main/MTP release, R_GPU0, no prefetch;")
ast.parse(s);target.write_text(s)
out=base/'kv-layer-major32k-v01402-source-review-v1';out.mkdir(mode=0o700)
prepare=json.loads((base/'kv-layer-major-v01402-source-v1/record.json').read_text())
audit=json.loads((base/'kv-stream32k-v01402-source-review-v2/record.json').read_text())
r=dict(active=False,passed=True,gpu_tested=False,adopted=False,
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    candidate_sha256=prepare['candidate_sha256'],
    controller_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    prior_audit_sha256=hashlib.sha256((base/'kv-stream32k-v01402-source-review-v2/record.json').read_bytes()).hexdigest(),
    source_scope=prepare['scope'],source_basis=prepare['source_basis'],
    unchanged_reviewed_paths=audit['reviewed'],
    limits=['GPU starts only after chunk-major3x32K terminal math/noXE/no-survivor gate.',
            'Owned full-context staging is264MiB/layer reused on in-order queue. Prefix upload retained unchanged; no persistent prefix cache optimization.',
            'R_GPU0 avoids full-context/repeated temporary VRAM exhaustion; full256K physical occupancy and restore/tail/refusal are pending.',
            'Cache and MTP leases retire prior graphs and restore before recapture with source-verified callbacks; repeated retirement is validated by first3x32K candidate gate.'])
(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
(out/'controller.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),s.splitlines(True),fromfile=str(parent),tofile=str(target))))
print(json.dumps({'passed':True,'controller':str(target),'sha256':r['controller_sha256']},indent=2))
