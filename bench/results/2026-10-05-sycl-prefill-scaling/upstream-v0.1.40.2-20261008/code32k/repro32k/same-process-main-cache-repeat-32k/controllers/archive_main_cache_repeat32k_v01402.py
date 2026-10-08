"""Retain the nine full32K captures and offline lifetime evidence."""
from pathlib import Path
import datetime, hashlib, json, shutil, sys
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro=parent/'code32k/repro32k'
out=repro/'same-process-main-cache-repeat-32k';assert not out.exists();out.mkdir()
sys.path.insert(0,str(root/'sycl/tools'))
from owned_gdb import process_identity
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def copy(p,relative):
    assert p.is_file() and p.stat().st_size<20*1024**2,p
    q=out/relative;q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists();shutil.copy2(p,q)
sequence=json.loads((base/'main-cache-repeat32k-v01402-state-sequence/record.json').read_text())
native=json.loads((base/'main-cache-repeat32k-v01402-resource-analysis/record.json').read_text())
graphs=json.loads((base/'main-cache-repeat32k-v01402-graph-analysis/record.json').read_text())
assert all(r['passed'] and not r['active'] for r in [sequence,native,graphs])
assert all(x['objects']==0 for x in native['live_native_at_exit'].values())
assert not graphs['command_buffer_references_at_exit']
for n in [2,3]:
    assert graphs['counts'][f'read-{n}/before decode cache release']['urCommandBufferReleaseExp']==680
    assert graphs['counts'][f'read-{n}/main decode cache released']['urCommandBufferReleaseExp']==6
    assert graphs['counts'][f'read-{n}/temporary buffers released']['urCommandBufferCreateExp']==680
assert digest(root/'build-sycl-e8ca-refresh-20261007/strata')=='c88f94d81bfb22227310ea00d09ecc8ab21a6e670aee9556307746530f5af714'
private=[]
for phase,repetition in [('diagnostic',1),('state',1),('state',2)]:
    p=base/f'owned-main-vmm-full-ram-repeat32k-v01402-code32k-{phase}-r{repetition}'
    r=json.loads((p/'record.json').read_text())
    assert r['healthy'] and r['completed'] and r['math_gate_passed'] and not r['active']
    assert r['exit_code']==0 and r['exit_signal'] is None and not r['new_fault_messages']
    assert not any(r['cleanup'][k] for k in ['forced','inferior_survived','gdb_survived'])
    for key in ['inferior','debugger']:
        old=r[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
    assert len(r['requests'])==3
    for q in r['requests']:
        c=q['comparison_to_default_counter_control']
        assert q['math_gate_passed'] and q['no_prompt_reuse'] and q['mtp_counts']==[43,66]
        assert q['measurement']['prompt_tokens']==32768 and q['measurement']['generated_tokens']==64
        assert not c['different_prefill_state_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal'])
    for f in sorted(p.rglob('*')):
        if not f.is_file():continue
        rel=f.relative_to(p)
        if f.suffix=='.bin' or f.name=='inferior.stderr' or f.stat().st_size>=20*1024**2:
            private.append({'file':str(f),'bytes':f.stat().st_size,'sha256':digest(f)})
        else:copy(f,f'runs/{phase}-r{repetition}/{rel}')
for source,relative in [
    ('main-cache-repeat32k-v01402-state-sequence','sequence'),
    ('main-cache-repeat32k-v01402-source-review','source-review'),
    ('live-prefill-state-v01402-host-check','host-check/v1-negative'),
    ('live-prefill-state-v01402-host-check-v2','host-check/v2'),
    ('main-cache-repeat32k-v01402-resource-analysis','analysis/native'),
    ('main-cache-repeat32k-v01402-graph-analysis','analysis/graphs')]:
    for p in sorted((base/source).rglob('*')):
        if p.is_file():copy(p,relative+'/'+str(p.relative_to(base/source)))
controllers=['compare_live_prefill_state_v01402.py','test_live_prefill_state_v01402.py',
             'compare_live_prefill_state_v01402_v2.py','test_live_prefill_state_v01402_v2.py',
             'prepare_main_cache_repeat32k_v01402.py','run_owned_main_cache_repeat32k_v01402.py',
             'run_main_cache_repeat32k_v01402_state_sequence.py',
             'analyze_main_cache_repeat32k_v01402_resources.py','analyze_main_cache_repeat32k_v01402_graphs.py',Path(__file__).name]
for name in controllers:copy(base/name,'controllers/'+name)
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
(out/'README.md').write_text('''# Same-process main-cache release: nine full 32K captures

Arc B570 10 GiB, Ryzen 5600X, 128 GiB RAM, unchanged private e82fc5 executable.
Each of three fresh processes reads the same 32,768-token code-review input
three times and generates 64 tokens with normal MTP4. Context is 33,024;
chunk 8,192; int8 KV; all 32,767 residual rows stay on GPU. Complete main
cache and MTP decode-only payloads are released/restored from immutable RAM.
Main expert slots use the matched 64 MiB segmented allocation. Prompt and
conversation reuse are disabled: every request reports resume0 and rereads
the entire input. The fixed dump paths must be absent before each request;
each new head/state is renamed and preserved after the reply.

The [sequence](sequence/record.json) passes all nine requests. Every raw
66-part main state and all 248,320 head floats match the hard control, as do
all 64 IDs/logprobs and MTP acceptance/offered counts43/66. All three engine
processes exit0 normally without forced cleanup, survivors or new xe faults.
One process has flushed UR/LevelZero validation/API logs; two have those
logs off but still have state/head capture, complete payload checks and phase
tracing. None of these durations is clean performance evidence. Every
performance comparison must use at least32,768 input tokens, and must keep
first-use loading/capture separate from later full-input rereads.

The [34 CPU comparator checks](host-check/v2/record.json) cover actual four-cell
snapshot pages, both heads' live/tail cells, all pooled rows including the
moving spare, wrong metadata and full262,144 occupancy. V1's page-offset unit
error is caught before GPU execution and preserved as a CPU negative. V2 may
exclude only rounded future KV cells, retaining raw hashes and exact ranges;
no exclusion is needed in these nine GPU captures. At full262,144 occupancy
there are no future cells to exclude. These host checks are not a proof of
SYCL mapping or absence of runtime UB.

The [UR reference analysis](analysis/graphs/record.json) verifies that the
second and third main releases each release680 old command buffers before
unmapping the main cache. MTP suspension releases6 more. Main restoration
creates/finalizes680 buffers each time. One non-cache commit buffer remains
through prefill. All UR buffer references reach zero at normal process exit.
The [native handle analysis](analysis/native/record.json) separately tracks
successful LevelZero create/destroy calls, handle reuse, device-USM free and
freeExt, physical mapping/unmapping and requested sizes. All tracked native
objects and mappings reach zero at exit. Native command-list counts remain
704 across later cache releases even while UR buffers are released, so
counting LevelZero objects alone does not demonstrate retained live graphs.
The source review also checks queue drain before unmap and recapture after
restoring stable addresses and complete payloads.

API logging is not required for the reported-free reduction: the two
non-API capture processes have identical markers, within64 KiB of the logged
process at matching later phases. Their MiB values are:

| Phase | First read | Second read | Third read |
| --- | ---: | ---: | ---: |
| Before main release | 2153.496 | 1131.254 | 1120.297 |
| After main/MTP release | 3433.500 | 2411.258 | 2400.301 |
| Prefill complete, temporary cache retained | 1089.609 | 151.059 | 140.289 |
| Temporary buffers released | 3349.613 | 2411.070 | 2400.301 |
| Main restore and verifier warm | 2128.289 | 2016.363 | 2010.852 |
| MTP restored | 1232.281 | 1120.359 | 1114.848 |

At matching prefill entry, read2 has1,022.242 MiB less free than read1;
read3 has10.957 MiB less than read2. Initial main warm creates61 modules,
61 kernels,680 regular command lists and requests1,114,112 B of device USM.
Later main warm creates no new native module/kernel/list/device-USM objects;
its384 MiB remapping accompanies reported-free drops of394.707 and389.438
MiB. Requested sizes and native counts are not resident heap sizes: these
observations do not isolate pooling, runtime-private allocations, residency
or granules, prove a leak, or establish steady state after more requests.

This closes only the captured full-RAM32K already-used-graph repeat gate.
Matched quiet kept-versus-released repeats and full262,144-cell occupancy,
repeat, restore, clipped tail, refusal and later-valid gates remain open.
The fixed32K GPU-row allocation is not a full-context default. Production
binaries/defaults remain unchanged; no reset, rebind, reboot, service, package
or global change occurs. Large API/state/head files remain private with exact
hashes. This candidate remains unadopted and the PP1000/TG70 goal stays active.
''')
for directory in [out,repro,parent/'code32k',parent]:
    p=directory/'manifest.json';m=json.loads(p.read_text()) if p.exists() else {}
    m.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             revision_reason='Nine same-process full32K raw-state/head/output controls and UR/native resource lifetimes; no speed/fullcontext/adoption claim.',
             files={str(f.relative_to(directory)):{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(directory.rglob('*')) if f.is_file() and f!=p})
    p.write_text(json.dumps(m,indent=2)+'\n')
    for rel,v in m['files'].items():
        f=directory/rel;assert f.stat().st_size==v['bytes'] and digest(f)==v['sha256']
    print(directory.name,len(m['files']),flush=True)
