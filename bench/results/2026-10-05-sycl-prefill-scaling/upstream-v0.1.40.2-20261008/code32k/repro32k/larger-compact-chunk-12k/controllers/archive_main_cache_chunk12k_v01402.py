"""Archive the normally exited mathematical rejection and offline evidence."""
from pathlib import Path
import datetime, hashlib, json, shutil, sys

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro=parent/'code32k/repro32k'
out=repro/'larger-compact-chunk-12k'
assert not out.exists()
out.mkdir()
sys.path.insert(0,str(root/'sycl/tools'))
from owned_gdb import process_identity
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def copy(p,relative):
    assert p.is_file() and p.stat().st_size<20*1024**2,p
    q=out/relative;q.parent.mkdir(parents=True,exist_ok=True)
    assert not q.exists();shutil.copy2(p,q)
job=base/'owned-main-vmm-full-ram-chunk12k-v01402-code32k-diagnostic-r1'
r=json.loads((job/'record.json').read_text())
assert r['completed'] and not r['active'] and r['exit_code']==0 and r['exit_signal'] is None
assert not r['new_fault_messages'] and not r['math_gate_passed']
assert not any(r['cleanup'][key] for key in ['forced','inferior_survived','gdb_survived'])
for key in ['inferior','debugger']:
    old=r[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
health=base/'post-chunk12k-v01402-health'
h=json.loads((health/'record.json').read_text());assert h['healthy'] and not h['active']
assert h['boot_id']==r['boot_id']
assert digest(root/'build-sycl-e8ca-refresh-20261007/strata')=='c88f94d81bfb22227310ea00d09ecc8ab21a6e670aee9556307746530f5af714'
for source,relative in [
    (job,'run'),(health,'health'),
    (base/'compact-chunk-v01402-host-accounting','accounting/v1'),
    (base/'compact-chunk-v01402-host-accounting-v2','accounting/v2'),
    (base/'main-cache-chunk12k-v01402-source-review','review/chunk'),
    (base/'native-head-prefill-v01402-lifetime-review','review/output-head'),
    (base/'main-cache-chunk12k-v01402-diagnostic-analysis-v2','analysis'),
    (base/'main-cache-chunk12k-v01402-native-resources','analysis/native-resources')]:
    for p in sorted(source.rglob('*')):
        if not p.is_file() or p.suffix=='.bin' or p.name in ['inferior.stderr','count']:continue
        assert p.stat().st_size<20*1024**2,p
        copy(p,relative+'/'+str(p.relative_to(source)))
controllers=[
    'count_compact_chunk_v01402_host.py','count_compact_chunk_v01402_host_v2.py',
    'prepare_main_cache_chunk12k_v01402_code32k.py','run_owned_main_cache_chunk12k_v01402_code32k.py',
    'verify_chunk12k_v01402_health.py','analyze_chunk12k_v01402_diagnostic.py',
    'analyze_chunk12k_v01402_diagnostic_v2.py','count_chunk12k_v01402_native_resources.py',Path(__file__).name]
for name in controllers:copy(base/name,'controllers/'+name)
private=[]
for rel in ['debugger/inferior.stderr','first-head.bin','prefill-state.bin']:
    p=job/rel;private.append({'file':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')

compiled=root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/program/generate.cpp'
generate=compiled.read_text()
assert generate.count('ver.warm(err)')==1
serve_start=generate.index('if (o.serve)')
serve_end=generate.index('const int shutdown_status = strata::finish_sycl_serve',serve_start)
assert '.warm(' not in generate[serve_start:serve_end]
api=json.loads((base/'main-cache-chunk12k-v01402-native-resources/record.json').read_text())
assert api['counts']['temporary buffers released']['zeCommandListCreate']==680
assert api['counts']['temporary buffers released']['zeModuleCreate']==61
assert api['requested_sizes']['temporary buffers released']['zePhysicalMemCreate']==402653184
analysis=json.loads((base/'main-cache-chunk12k-v01402-diagnostic-analysis-v2/record.json').read_text())
assert not analysis['logged_device_allocations_at_end']
memory={'scope':'One logged32K fresh server. Source and native API evidence distinguish first verifier capture from repeated-recapture/lifetime claims. Not clean timing or isolated driver allocation attribution.',
        'compiled_generate_sha256':digest(compiled),
        'serve_has_startup_verifier_warm':False,
        'only_generate_warm_call_is_in_nonserve_path':True,
        'before_prefill_free_bytes':1381425152,'after_restore_free_bytes':415526912,
        'reported_free_reduction_bytes':965898240,
        'main_restore_and_first_capture_reported_free_reduction_bytes':1280651264,
        'main_physical_restore_bytes':402653184,
        'callback_reported_free_reduction_beyond_main_mapping_bytes':877998080,
        'new_device_usm_requested_in_callback_bytes':1114112,
        'callback_new_command_lists':680,'callback_new_modules':61,
        'all_logged_device_usm_freed_at_exit':True,
        'all_native_modules_kernels_command_lists_destroyed_at_exit':True,
        'limitations':['First capture/JIT/command resources coexist in the callback. Do not attribute the unexplained reported-free difference solely to recapture, byte-copying or a proven leak.',
                       'Physical mapping/API requested sizes are different from resident VRAM, runtime-private heaps, pooling and allocation rounding.',
                       'The older second-full-prefill capacity failure is not proven solved or explained by this32K observation.',
                       'Fresh kept-fraction-zero arms defer capture to first verifier use; released arms warm all legal sizes before that use. Quiet means include these differing capture schedules.']}
(out/'review/serve-first-capture-memory.json').write_text(json.dumps(memory,indent=2)+'\n')
(out/'analysis/revisions.json').write_text(json.dumps({'scope':'Executed CPU controller versions are preserved. These are offline accounting/parser corrections, not GPU faults or production edits.',
    'accounting_v1':'Guard rejects the borrowed per-slot ring padding count against the owned single-ring log. V2 removes seven256B per-slot guards and matches the measured8192 accounted/shared bytes.',
    'allocation_parser_v1':'Stops at the second reused loader-probe address because it did not handle zeMemFreeExt. V2 tracks successful zeMemFree and zeMemFreeExt; all logged device allocations are freed at exit.'},indent=2)+'\n')
(out/'README.md').write_text('''# Larger compact chunk: rejected 32K control

This uses the unchanged private e82fc5 executable on Arc B570, 10 GiB VRAM,
Ryzen 5600X and 128 GiB RAM. Input stays at 32,768 tokens; only the compact-hc
chunk grows from 8,192 to 12,288. Main cache and MTP decode-only payloads
restore from immutable RAM, with complete payload checks. Context is 33,024,
normal MTP is 4, all 32,767 prefill residual rows stay on GPU, and 128 main
expert slots use the matched 64 MiB segmented allocation.

The [offline accounting](accounting/v2/record.json) extracts actual sizing
helpers. Its 8K accounted/shared sizes match the executed log after the
owned single-ring padding correction. At 12K the accounted workspace is
2,738,729,216 B, and reused residual scratch reduces the new GPU-row storage
to 838,819,840 B. The engine's page-budget estimator gives a net increment
of 708,837,376 B over 8K. Those page estimates are not a measurement of SYCL
physical granularity. The 16K increment is 1,417,674,752 B; it is not run.

The logged/validated 32K check exits normally, receives 64 outputs, performs
owned cleanup and records no new xe fault. Main and MTP payload checks pass,
but the mathematical gate rejects 12K: 63 of 66 main-state parts differ;
all 248,320 first-head floats differ (max absolute 0.873567, RMS 0.180161).
The first generated ID differs at zero-based index 2, and logprobs differ
at index 0. MTP counts are 38/75, versus the accepted 8K reference's 43/66.
This is a normal engine exit with a rejected correctness gate, not a GPU
hang. The later [exact-word health check](health/record.json) passes.
No clean timing follows and diagnostic durations are not speed evidence.
The changed chunk/GEMM/callback boundaries have not been isolated as the
cause; neither a rounding-only explanation nor a code bug is established.

The [allocation analysis](analysis/record.json) records a 921.152 MiB drop
in reported free VRAM between prefill entry and restored weights. During
main restoration plus graph warm, the main mapping grows by 384 MiB, while
reported free drops by 1,221.324 MiB. Only 1,114,112 B of new device USM is
requested in that interval. Native logs show 680 new command lists and 61
new modules; all device USM and native modules/kernels/command lists are
destroyed at process exit. This does not isolate runtime-private heap,
pooling, allocation rounding, or prove a leak.

The [source review](review/serve-first-capture-memory.json) verifies that
this serve path does not call verifier warm at startup. Released arms warm
all legal window sizes at cache restoration; a kept-fraction-zero arm
captures on first use. Consequently the restoration interval includes
initial graph capture and kernel loading in these fresh processes. The
matched cache-release means include those different capture schedules;
do not attribute decode variation solely to transferring or releasing
cache bytes. Repeated recapture needs a separate warm-process comparison.

The [output-head lifetime review](review/output-head/record.json) retains
source facts for a possible later phase lease; no head lease is implemented.
Any future lease must retain immutable GGUF bytes and restore output backing
before verifier capture, including partial-failure cleanup.

Full 262,144-cell occupancy/repeat/restore/clipped-tail/refusal/later-valid
gates remain mandatory and open. The older repeated-prefill capacity failure
is not resolved by this 32K observation. Production binaries/defaults remain
unchanged. Raw API/state/head payloads stay private with exact hashes; no
reset/rebind/reboot/service/package/global change occurs.
''')
old=repro/'main-cache-release-and-restore/README.md'
t=old.read_text();t=t.replace('Graph recapture overhead is included in\nclean prefill time.','The graph callback is included in clean prefill time. These fresh serve\nprocesses do not warm verifier graphs at startup: release arms capture all\nlegal window sizes at restoration, while kept backing captures on first use.\nInitial capture/kernel loading is part of this comparison, so it does not\nisolate repeated recapture. See the [later source/API audit](../larger-compact-chunk-12k/README.md).')
t=t.replace('These measurements hold fixed the8192-token chunk and residual geometry;\nrelease alone does not test the benefit of enlarging chunks with freed VRAM.','These measurements hold fixed the 8192-token chunk and residual geometry;\nrelease alone does not test the benefit of enlarging chunks with freed VRAM.\nA later logged 32K/12288-chunk attempt has enough measured capacity but fails\nstate/head/output equality and is rejected before clean timing.')
old.write_text(t)
for directory in [out,repro/'main-cache-release-and-restore',repro,parent/'code32k',parent]:
    p=directory/'manifest.json'
    m=json.loads(p.read_text()) if p.exists() else {}
    m.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_reason='Retain rejected32K larger chunk and first-capture/native-resource evidence; clarify fresh-process cache-release scope.',
             files={str(f.relative_to(directory)):{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(directory.rglob('*')) if f.is_file() and f!=p})
    p.write_text(json.dumps(m,indent=2)+'\n')
    for rel,v in m['files'].items():
        f=directory/rel;assert f.stat().st_size==v['bytes'] and digest(f)==v['sha256']
    print(directory.name,len(m['files']))
