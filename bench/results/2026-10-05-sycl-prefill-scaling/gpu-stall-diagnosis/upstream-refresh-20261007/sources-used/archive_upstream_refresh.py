"""Freeze upstream merge/build/host and exact-head GPU integration checks."""
from pathlib import Path
import json,hashlib,datetime,subprocess
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis'
helpers=(base/'archive_lazy_full_and_dequant.py').read_text().split('\nfull = json.loads(',1)[0]
exec(compile(helpers,str(base/'archive_lazy_full_and_dequant.py'),'exec'))
source=base/'upstream-refresh-20261007'
record=json.loads((source/'record.json').read_text())
for name in ['build','serve','setup']:
 test=json.loads((source/(name+'-record.json')).read_text());assert test['passed'] and not test['active']
assert json.loads((source/'cpu-tests/record.json').read_text())['passed']
runs=[]
for name in ['owned-upstream-refresh-short','owned-upstream-refresh-2048']:
 r=json.loads((base/name/'record.json').read_text())
 assert r['healthy'] and r['completed'] and not r['active'] and not r['new_fault_messages']
 assert r['exit_code']==0 and not any(r['cleanup'].values())
 for key in ['inferior','debugger']:
  old=r[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
 assert all(all(req['equality'].values()) for req in r['requests'])
 runs.append(dict(name=name,requests=len(r['requests']),release_pairs=r['release_pairs'],binary_sha256=r['binary_sha256']))
assert runs[0]['binary_sha256']==runs[1]['binary_sha256']
assert subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=root)==b''
record.update(tested=True,gpu_jobs_active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              binary_sha256=runs[0]['binary_sha256'],gpu_runs=runs,host_setup_scripts_passed=24,
              serve_tests_run=452,serve_tests_skipped=8,standalone_cpu_tests_passed=4,
              scope='Upstream source/server/shared headers incorporated, existing SYCL math and phase waits retained; CPU build/host tests and short/multichunk GPU exact-head regression only. New CUDA-only fused/padded/VMM modes are not enabled in SYCL. Full262144-cell gates remain unresolved and incomplete.')
(source/'record.json').write_text(json.dumps(record,indent=2)+'\n')
dest=parent/'upstream-refresh-20261007';assert not dest.exists();dest.mkdir()
for name in ['archive_upstream_refresh.py','test_upstream_refresh_v3.py','test_upstream_cpu_standalone.py',
             'run_owned_upstream_refresh_short.py','run_owned_upstream_refresh_2048.py']:
 copy(base/name,dest,Path('sources-used')/name)
# Only chosen logs/receipts: do not archive the dependency environment or compiled test executables.
chosen=[]
for p in source.rglob('*'):
 if not p.is_file() or 'test-venv' in p.relative_to(source).parts:continue
 if p.suffix not in ['.json','.stdout','.stderr','.cpp','.md'] and p.name not in ['README.md']:continue
 chosen.append(p)
for p in chosen:
 rel=p.relative_to(source);d=dest/'build-host'/rel;d.parent.mkdir(parents=True,exist_ok=True)
 if p.stat().st_size<=300000:copy(p,dest,Path('build-host')/rel)
 else:
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-65536));d.with_name(d.name+'.tail').write_bytes(f.read())
  d.with_name(d.name+'.metadata.json').write_text(json.dumps(dict(private_path=str(p),private_bytes=p.stat().st_size,private_sha256=digest(p),saved_tail=str((Path('build-host')/rel).with_name(p.name+'.tail'))),indent=2)+'\n')
for run in runs:save_folder(base/run['name'],dest,run['name'])
stat=subprocess.check_output(['git','diff',record['equivalent_rewritten_upstream'],record['latest_upstream'],'--stat'],cwd=root,text=True)
(dest/'upstream-changes.stat').write_text(stat)
(dest/'README.md').write_text("""# Upstream refresh checked on 2026-10-07

Upstream Niko1221/Strata at 82f46a8c8f475f001ad76d92f58f4a4f8ffb0253 is
integrated on the existing SYCL work branch. Upstream rewrote its history:
old base 6f32ec07 and new-history a1641e9f have the exact same whole-tree
SHA 27b0e86f. An unchanged-tree bridge records this equivalence before a
normal three-way merge. Five conflicts were resolved with full file-tail
checks, retaining local graph retirement, atomic host doorbells and source
lifetime protection. No branch history was force pushed.

An isolated Release build with oneAPI 2026.1.1 and the existing GCC IQ2_S
object option passes. Shared API compatibility adds default contiguous
strides, optional host-registration waits and upstream CPU file-source
methods. Unsupported new CUDA fused/padded modes reject explicit requests.
Upstream CUDA elastic K/V remains unavailable; the existing Level Zero
expert-cache mapping is retained. SYCL device arithmetic, launch geometry
and per-phase waits are unchanged. The SYCL engine retains its upstream
0.1.39-sycl version; this receipt does not assert every CUDA 0.1.40 feature
has a SYCL implementation.

Host validation: 452 serve tests pass, with eight skips; all 24 setup scripts
pass (the unsloth test mocks were updated for the new minimum engine);
message-boundary, exchange-storage, draft-policy and suffix-drafter existing
standalone CPU tests pass. Initial build and test failures are retained
alongside their passing replacement receipts.

On B570 10 GiB / Ryzen 5600X / 128 GiB RAM, the updated executable
ceff2d8a…680c0bd0 runs through owned GDB with flushed Level Zero/UR logs and
validation. Four context-128 normal-MTP requests match the frozen pre-update
IDs, every printed logprob and every byte of all 248,320 first-head floats.
Six MTP release/restore pairs complete. A separate context-4096, int8 KV,
2048-token/two-1024-chunk normal-MTP request matches the earlier exact-head
control and completes one release/restore pair. Both exit normally; no
owned processes survive and no new xe fault is recorded. Diagnostic times
are not clean throughput figures.

The repeated 262144-cell CLI/serve, clipped-tail, overflow-refusal and
later-valid gates remain incomplete. Previous full-capacity failures and
eager/lazy whole-head differences are not resolved by this update. No
reset, service change or package change was performed for these GPU checks.
Private whole heads and large diagnostic logs are hashed; bounded tails,
controllers, build/source hashes and terminal receipts are preserved here.
""")
report=manifest(dest);(base/'upstream-refresh-archive.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
