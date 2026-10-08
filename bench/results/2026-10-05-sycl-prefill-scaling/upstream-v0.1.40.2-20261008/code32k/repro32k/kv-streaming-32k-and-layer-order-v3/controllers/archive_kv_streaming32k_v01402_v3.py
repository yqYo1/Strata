"""Archive terminal32K source/correctness and matched quiet processing order."""
from pathlib import Path
import datetime,hashlib,json,shutil,statistics
base=Path(__file__).parent
root=Path("/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05")
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro=parent/'code32k/repro32k'
out=repro/'kv-streaming-32k-and-layer-order-v3';assert not out.exists()
sequence_path=base/'kv-matched-quiet32k-v01402-sequence-v1/record.json'
seq=json.loads(sequence_path.read_text());assert seq['passed'] and not seq['active']
diagnostics=[]
for mode in ['kv-stream32k','kv-layer-major32k']:
 p=base/f'owned-{mode}-v01402-code32k-diagnostic-r1'
 r=json.loads((p/'record.json').read_text())
 assert r['healthy'] and not r['active'] and r['math_gate_passed'] and r['exit_code']==0
 assert not r['new_fault_messages'] and not any(r['cleanup'].values()) and len(r['requests'])==3
 for q in r['requests']:
  assert q['math_gate_passed'] and q['no_prompt_reuse'] and q['mtp_counts']==[43,66]
  assert not q['live_prefill_comparison']['different_live_parts']
  assert all(q['comparison_to_default_counter_control'][k] for k in ['first_head_equal','ids_equal','logprobs_equal'])
 diagnostics.append((mode,p,r))
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out.mkdir()
def copy(p,rel):
 assert p.is_file() and p.stat().st_size<20*1024**2
 q=out/rel;assert not q.exists();q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
private=[]
def private_file(p):
 private.append(dict(file=str(p),bytes=p.stat().st_size,sha256=digest(p)))
for directory,rel in [
 ('kv-stream-safe-v01402-source-candidate-v3','kv-source-review/v3'),
 ('kv-stream-safe-v01402-build-v3','kv-compile-link-receipt'),
 ('kv-stream32k-v01402-source-review','ownership-review/v1-negative'),
 ('kv-stream32k-v01402-source-review-v2','ownership-review/v2'),
 ('kv-layer-major-v01402-source-v1','layer-source-review'),
 ('kv-layer-major-v01402-build-v1','layer-compile-link-receipt'),
 ('kv-layer-major32k-v01402-source-review-v1','layer-ownership-review'),
 ('kv-matched-quiet32k-v01402-controller-v1','quiet-controller-review'),
 ('kv-matched-quiet32k-v01402-sequence-v1','quiet-sequence')]:
 for p in sorted((base/directory).rglob('*')):
  if p.is_file():copy(p,rel+'/'+str(p.relative_to(base/directory)))
names=['prepare_kv_stream_safe_v01402_source_v3.py','build_kv_stream_safe_v01402_v3.py',
 'prepare_owned_kv_stream32k_v01402.py','run_owned_kv_stream32k_v01402.py',
 'prepare_owned_kv_stream32k_v01402_v2.py','run_owned_kv_stream32k_v01402_v2.py',
 'prepare_kv_layer_major_v01402_v1.py','build_kv_layer_major_v01402_v1.py',
 'prepare_owned_kv_layer_major32k_v01402_v1.py','run_owned_kv_layer_major32k_v01402_v1.py',
 'prepare_owned_kv_matched_quiet32k_v01402_v1.py','run_owned_kv_matched_quiet32k_v01402_v1.py',
 'run_kv_matched_quiet32k_v01402_sequence_v1.py',Path(__file__).name]
for n in names:copy(base/n,'controllers/'+n)
for name,path in [
 ('kv_stream.original.dp.cpp',root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/kernels/cuda/kv_stream.dp.cpp'),
 ('kv_stream.v3.dp.cpp',root/'build-sycl-kv-stream-safe-v3-20261008/source/kv_stream.dp.cpp'),
 ('prefill.original.cpp',root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/prefill/prefill.cpp'),
 ('prefill.stream-layer-v1.cpp',root/'build-sycl-kv-layer-major-v1-20261008/source/prefill.cpp')]:copy(path,'source/'+name)
for dirname in ['build-sycl-kv-stream-safe-v3-20261008','build-sycl-kv-layer-major-v1-20261008']:
 for p in sorted((root/dirname).iterdir()):
  if p.is_file():private_file(p)
jobs=[(m,p,r,'diagnostic') for m,p,r in diagnostics]
for step in seq['steps']:
 p=Path(step['record_file']).parent;r=json.loads((p/'record.json').read_text())
 assert digest(p/'record.json')==step['record_sha256']
 jobs.append((step['mode'],p,r,'quiet'))
for mode,p,r,phase in jobs:
 for f in sorted(p.rglob('*')):
  if not f.is_file():continue
  if f.name.endswith(('.state.bin','.head.bin')) or f.stat().st_size>=20*1024**2:
   private_file(f)
  else:copy(f,'jobs/'+p.name+'/'+str(f.relative_to(p)))
summary=dict(active=False,passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 hardware='Arc B57010GiB, Ryzen5600X,128GiB RAM; same0a908c16 boot/oneAPI2026.1.1/UR0.12/NEO26.31.39395.14',
 minimum_performance_input_tokens=32768,actual_performance_input_tokens=32768,
 configuration='Qwen3.8-Flash-Next IQ3_S pack/native, context262144, resident32768, int8KV,8192 chunks, cache128, workers5, pcie0, normalMTP4, greedyGEN64, ckpt0/prompt-cache0, own stage and prefetch0.',
 correctness={},quiet={},adopted=False,full_262144_occupancy_tested=False)
for mode,p,r in diagnostics:
 summary['correctness'][mode]=dict(requests=3,full_head_ids_logprobs_mtp_exact=True,
    used_state_all66_parts_exact=True,
    different_raw_parts=[q['comparison_to_default_counter_control']['different_prefill_state_parts'] for q in r['requests']],
    only_unwritten_rounded_future_cells_may_differ=True,orderly_exit=True,new_xe_faults=False,
    record_sha256=digest(p/'record.json'),diagnostic_seconds_excluded_from_speed=True)
for mode in ['kv-stream32k-quiet','kv-layer-major32k-quiet']:
 steps=[s for s in seq['steps'] if s['mode']==mode];assert len(steps)==2
 group={}
 for label,rows in [('first_request',[s['measurements'][0] for s in steps]),
                    ('repeated_full_reads',[m for s in steps for m in s['measurements'][1:]])]:
  mean_prompt=statistics.mean(m['prompt_ms'] for m in rows)
  mean_decode=statistics.mean(m['decode_ms'] for m in rows)
  group[label]=dict(reads=len(rows),prefill_tok_s=32768000/mean_prompt,decode_tok_s=64000/mean_decode,
     prompt_ms=mean_prompt,decode_ms=mean_decode,individual_prefill_tok_s=[m['prefill_tok_s'] for m in rows],
     individual_decode_tok_s=[m['decode_tok_s'] for m in rows])
 summary['quiet'][mode]=group
a=summary['quiet']['kv-stream32k-quiet']['repeated_full_reads']
b=summary['quiet']['kv-layer-major32k-quiet']['repeated_full_reads']
summary['layer_vs_chunk_repeated_prefill_percent']=100*(b['prefill_tok_s']/a['prefill_tok_s']-1)
summary['layer_vs_chunk_repeated_decode_percent']=100*(b['decode_tok_s']/a['decode_tok_s']-1)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
table='| Configuration | First request PP/TG (tok/s) | Repeated full reads PP/TG (tok/s) |\n| --- | ---: | ---: |\n'
for mode,label in [('kv-stream32k-quiet','Chunk-major, main/MTP kept'),('kv-layer-major32k-quiet','Layer-major, RAM residuals, main/MTP RAM leases')]:
 q=summary['quiet'][mode];f=q['first_request'];w=q['repeated_full_reads']
 table+=f"| {label} | {f['prefill_tok_s']:.2f} / {f['decode_tok_s']:.2f} | {w['prefill_tok_s']:.2f} / {w['decode_tok_s']:.2f} |\n"
(out/'README.md').write_text("""# RAM-authoritative KV and processing order: repeated32K checks

Measured on Arc B57010GiB, Ryzen5600X and128GiB RAM with the pinned
oneAPI2026.1.1/UR0.12/NEO26.31.39395.14 stack. Production c88f94d and
accepted e82fc5 binaries remain unchanged; both candidates are private.

The v3 KV source adds an atomic page-table read paired with the existing
atomic claim, and inherits atomic shared-hit metadata. Host/device counter
representations and16-byte copies use memcpy; resolve requires host-USM,
WG1024 and SG32 support before submission. Four unused root properties stay
removed. Earlier accepted mode0 runs did not activate this route, so these
possible conflicting accesses are not attributed as their hang cause.

The layer-major candidate replaces its mode0-only rejection with mode0/1
support requiring owned identity staging and disabled KV prefetch. Existing
in-order prefix uploads, appends, attention and host mirrors remain unchanged.
It runs one layer over all chunks, using the already allocated full-context
stage. It keeps RAM residuals (R_GPU0), full main-cache RAM restoration and
the previously validated MTP decode lease. No staging-prefix reuse optimization
is introduced here.

Each configuration passes three fresh32768-token/64-output requests in one
logged/validated process: all66 used main-state parts, all248320 first-head
floats,64 IDs/logprobs and MTP43/66 match the accepted control. First raw state
matches too; later differences are confined to unwritten rounded-page future
cells. The pooled spare row is compared. Both exit0 without forced cleanup,
survivors or new xe faults. This does not prove full262144-cell occupancy.

The first source-authority guard rejects comparison of pinned compiled
prefill with production source before GPU execution. Its preserved negative
and corrected review explicitly use the accepted actual-event-completion
overlay. The private receipts record the compile/link inputs and object
replacement rather than inferring an executable from current worktree code.

Quiet comparison uses ABBA fresh processes and three complete32768-token
reads per process. Context262144, resident32768,8192 chunks, int8 KV,
cache128 requested, workers5, pcie0, normalMTP4, greedyGEN64, FIRST0/RING8,
own stage and prefetch0 are common. Layer-major includes its RAM-residual
and main/MTP lease choices; this is a configuration comparison, not an
isolated transfer or graph metric. Every reply still matches64IDs/logprobs,
MTP43/66 and resume0. State/head/payload verification, traces, profilers,
API logs and extra phase waits are absent. Health probes run before requests.

First-request numbers average two reads after READY; startup model loading
is not part of the engine's prompt timer. Repeated numbers use four full
rereads, converting their mean prompt/decode durations to rates.

""" +table+"""
Diagnostics and captures are excluded from performance. Individual readings,
source/build/owned-process receipts and protocol logs are retained; huge API
logs, state images and binaries remain private with hashes.

Full262144 occupancy through cell262143, repeated full prefill, restored
state after another session, clipped speculative tails, capacity refusals
and a later valid>=32768 request remain required. No production adoption,
reset/rebind/reboot, service, package, runtime or global change is made.
""")
p=repro/'README.md';p.write_text(p.read_text()+"""
\nThe [RAM-KV32K processing-order comparison](kv-streaming-32k-and-layer-order-v3/README.md)
records defined KV representations/page claims, three exact used-state/head
reads per configuration, and matched quiet ABBA runs with initial versus
repeated32768-token measurements. Full262144 occupancy and adoption remain
pending.
""")
for d in [out,repro,parent/'code32k',parent]:
 p=d/'manifest.json';m=json.loads(p.read_text()) if p.exists() else {}
 m.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 revision_reason='Retain private RAM-KV32K correctness and matched processing-order gates.',
 files={str(f.relative_to(d)):{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(d.rglob('*')) if f.is_file() and f!=p})
 p.write_text(json.dumps(m,indent=2)+'\n')
 for k,v in m['files'].items():assert (d/k).stat().st_size==v['bytes'] and digest(d/k)==v['sha256']
 print(d.name,len(m['files']),flush=True)
print(json.dumps(summary,indent=2))
