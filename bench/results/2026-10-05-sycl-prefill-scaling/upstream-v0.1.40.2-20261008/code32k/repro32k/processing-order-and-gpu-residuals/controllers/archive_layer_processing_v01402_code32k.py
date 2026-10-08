from pathlib import Path
import datetime, hashlib, json, shutil, sys

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro=parent/'code32k/repro32k'
out=repro/'processing-order-and-gpu-residuals'
sys.path.insert(0,str(root/'sycl/tools'))
from owned_gdb import process_identity
def digest(p):
    with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def load(name):return json.loads((base/name/'record.json').read_text())
sequences=['compact2-layout1-v01402-state-sequence','layer2-ram-layout1-v01402-state-sequence','compact-layer-v01402-state-sequence','layer-gpu-release-v01402-state-sequence','layer-processing-v01402-clean-sequence']
for name in sequences:
    r=load(name);assert r['passed'] and not r['active'],name
assert digest(root/'build-sycl-e8ca-refresh-20261007/strata')=='c88f94d81bfb22227310ea00d09ecc8ab21a6e670aee9556307746530f5af714'
assert not out.exists();out.mkdir()
def copy(p,dst):
    assert p.is_file() and p.stat().st_size<20*1024**2,p
    q=out/dst;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
private=[];runs=[]
baseline=load('owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1')['requests'][0]
ref_done=baseline['protocol'][-1].split()
def archive(name,comparison):
    d=base/name;r=load(name)
    assert r['healthy'] and r['completed'] and not r['active'] and r['exit_code']==0
    assert not r['new_fault_messages'] and not any(r['cleanup'][k] for k in ['inferior_survived','gdb_survived'])
    for k in ['inferior','debugger']:
        old=r[k];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
    req=r['requests'][0];mm=req['measurement'];assert mm['prompt_tokens']==32768 and mm['generated_tokens']==64
    c=req[comparison];assert c['ids_equal'] and c['logprobs_equal']
    if 'first_head_equal' in c:assert c['first_head_equal'] and not c['different_prefill_state_parts']
    done=req['protocol'][-1].split();assert done[0]=='DONE' and done[6:8]==ref_done[6:8]
    runs.append({'name':name,'drafts_accepted':int(done[6]),'drafts_offered':int(done[7]),'baseline_draft_counts_equal':True})
    assert 'MKL_CBWR' not in r['environment'] and 'EnableImplicitConvertionToCounterBasedEvents' not in r['environment']
    for f in ['record.json','protocol.stdout.raw','events.jsonl','project-messages.txt','input-tokens.txt']:
        p=d/f
        if p.exists():copy(p,'runs/'+name+'/'+f)
    for p in (d/'probes').glob('*'):
        if p.is_file():copy(p,'runs/'+name+'/probes/'+p.name)
    for f in ['inferior-argv.json','inferior-environment.json']:
        p=d/'debugger'/f
        if p.exists():copy(p,'runs/'+name+'/debugger/'+f)
    for f in ['debugger/inferior.stderr','first-head.bin','prefill-state.bin']:
        p=d/f
        if p.exists():private.append({'file':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
for mode in ['compact2-layout1','layer2-ram-layout1','layer2-gpu-release-layout1']:
    for phase,rep in [('diagnostic',1),('state',1),('state',2)]:
        archive(f'owned-{mode}-v01402-code32k-{phase}-r{rep}','comparison_to_default_counter_control')
for mode in ['processing-default','layer2-ram-layout1','layer2-gpu-release-layout1']:
    for rep in [1,2]:archive(f'owned-{mode}-v01402-code32k-clean-r{rep}','comparison_to_logged_control')
for name in sequences+['compact-layer-v01402-source-review','layer-gpu-release-v01402-source-review']:
    copy(base/name/'record.json','analysis/'+name+'.json')
controllers=['prepare_compact_layer_v01402_code32k.py','run_owned_compact_layer_v01402_code32k.py','run_compact_layer_v01402_state_sequence.py','prepare_layer_gpu_release_v01402_code32k.py','run_owned_layer_gpu_release_v01402_code32k.py','run_layer_gpu_release_v01402_state_sequence.py','prepare_layer_processing_v01402_clean.py','run_owned_layer_processing_v01402_code32k_clean.py','run_layer_processing_v01402_clean_sequence.py',Path(__file__).name]
for name in controllers:copy(base/name,'controllers/'+name)
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
(out/'mtp-count-parity.json').write_text(json.dumps({'scope':'Protocol acceptance/offered counts only; this is not a comparison of every internal MTP tensor. All64 IDs/logprobs and captured main prefill-state/head match separately.','baseline_counts':ref_done[6:8],'runs':runs},indent=2)+'\n')
seq=load('layer-processing-v01402-clean-sequence');means=seq['clean_mean']
table='| Setting | Run | Prefill token/s | Decode token/s |\n| --- | --- | ---: | ---: |\n'
for mode in means:
    for s in seq['steps']:
        if s['argv'][2]==mode:table+=f"| {mode} | {s['argv'][-1]} | {s['measurement']['prefill_tok_s']:.3f} | {s['measurement']['decode_tok_s']:.3f} |\n"
    table+=f"| {mode} | mean | {means[mode]['prefill_tok_s']:.3f} | {means[mode]['decode_tok_s']:.3f} |\n"
leases=[]
for rep in [1,2]:
    p=base/f'owned-layer2-gpu-release-layout1-v01402-code32k-clean-r{rep}/project-messages.txt'
    leases.extend({'run':rep,'message':s} for s in p.read_text().splitlines() if s.startswith(('strata mtp decode release:','strata mtp decode restore:','strata prefill residual in-place:','strata prefill: layer-major,')))
(out/'clean-release-messages.json').write_text(json.dumps(leases,indent=2)+'\n')
(out/'README.md').write_text('''# Processing order and GPU residuals on 32K input

Arc B570 10 GiB, Ryzen 5 5600X and 128 GiB RAM; kernel 7.0.0-38,
NEO 26.31.39395.14, oneAPI 2026.1.1, default implicit counter conversion and
no MKL CNR. All conditions use the unchanged private scheduling executable
e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323,
the same 32,768-token code-review fixture, context 33024, 8192-token chunks,
int8 KV, normal MTP4, 64 greedy output tokens, expert cache 128, five CPU
workers, pcie 0, FIRST 0/RING 8 and no prompt/conversation caching.

COMPACT2 reuses phase storage after its last consumer on the in-order queue.
QSA layout 1 retains subgroup 32, workgroup 256 and ordered arithmetic, while
reconstructing query operands through scalar loads from transposed local
storage. Batch remains 32. Three fresh 32K controls first check that combination.
Startup INFO freeVRAM increases from 830 MiB to 2220 MiB; this is a startup snapshot,
not a peak-usage measurement. The
[source review](analysis/compact-layer-v01402-source-review.json) records
allocation/carving agreement, scratch bounds, uniform barriers and the
inactive key-head alternative. These reviews do not prove all runtime UB absent.

Layer-major 2 then retains a full layer's 512 experts in a separate temporary
cache while processing all four chunks. Three more exact 32K controls check
this order with residual rows in RAM. Its logged row image is 1,342,136,320 B,
temporary layer cache 1,363,148,800 B. Main decode cache and MTP weights remain
backed for this arm. Expert copies repeat once per layer instead of once per
layer/chunk, but RAM residual upload/download adds transfers. Source-derived
copy avoidance is not a measured PCIe bandwidth or DMA duration.

A third configuration keeps all 32,767 batched residual rows on GPU, reuses
the original 8192-row scratch prefix in place, and allocates the remaining
24,575 rows separately. MTP decode-only expert/head weights are temporarily
unmapped after queues drain and graphs are retired; persistent dense weights,
state and KV remain backed. Restoration remaps the same virtual addresses,
copies the immutable RAM payload, waits before readiness and recaptures graphs
when needed. The main cache remains backed. The
[additional review](analysis/layer-gpu-release-v01402-source-review.json)
checks row/tail partitions, stable MTP graph inputs and reclamation ordering.
Three fresh 32K controls enable full byte verification before/after each lease.

All nine captured controls match all 66 main-prefill state parts, all 248,320
first-head float bytes, 64 IDs and all logprobs with the completed default
control. Every first use has flushed UR/Level Zero logs and parameter validation.
These timings are excluded. The [MTP protocol counts](mtp-count-parity.json)
also match 43 accepted/66 offered in every captured/clean request; this compares
counts, not every internal MTP tensor.

Only after all gates, six fresh clean jobs run default/RAM/GPU/GPU/RAM/default.
No diagnostics, validation, byte verification, state/head dumps, profiler,
transfer timing or extra waits are enabled. The uninterrupted owned GDB/PTY
observer is common. Every clean job matches all 64 IDs/logprobs. All 15 jobs
exit normally, complete owned cleanup and record no new xe fault.

'''+table+'''
The [clean sequence](analysis/layer-processing-v01402-clean-sequence.json)
contains each duration, ordering and relative change. The
[release/restore messages](clean-release-messages.json) preserve measured
mapping/copy/graph-drop timings in both clean GPU-row repetitions. These
measurements apply to this model, hardware and configuration, not an
unmodified-upstream equivalence claim. Two repetitions per condition do not
establish small changes beyond observed variation or attribute decode variation
to a prefill-only flag.
The GPU-row pair differs by about 11% in prompt throughput. Its two-run mean
therefore does not establish a reproducible improvement over the default.

The candidate is private and unadopted. Full 262,144-cell normal-MTP
occupancy/repeat/restore/clipped-tail/refusal/later-valid gates remain open;
the older second-prefill memory failure is not solved by this 32K check.
Snapshot-versus-RAM and partial main-cache release are separate pending
comparisons. PP 1000/TG 70 is not declared achieved. No production executable,
reset/rebind/reboot, service, package or global setting changes.

Public files retain exact argv/environment, fixture, protocol, source hashes,
executed controllers, ownership, journal and state/head hashes. Large API,
state and head payloads remain private with byte counts and SHA256 hashes.
''')
with (repro/'README.md').open('a') as s:
    s.write('''

The [processing-order/GPU-residual follow-up](processing-order-and-gpu-residuals/README.md)
adds nine exact 32K main-state/head/output controls and a clean three-condition
default/RAM/GPU/GPU/RAM/default comparison. GPU residuals use MTP decode-only
release and immutable-RAM restoration, with full payload checks before timing.
All comparisons read 32,768 tokens; full262,144-cell serving remains unvalidated.
''')
for directory in [out,repro,parent/'code32k',parent]:
    p=directory/'manifest.json';r=json.loads(p.read_text()) if p.exists() else {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    r.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_reason='Append gated32K processing-order and GPU-residual/restore comparison.',files={str(f.relative_to(directory)):{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(directory.rglob('*')) if f.is_file() and f!=p})
    p.write_text(json.dumps(r,indent=2)+'\n')
    for name,v in r['files'].items():
        f=directory/name;assert f.stat().st_size==v['bytes'] and digest(f)==v['sha256']
    print(directory.name,len(r['files']))
