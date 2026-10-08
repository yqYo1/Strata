"""Retain an explicitly untested private streaming source/build candidate."""
from pathlib import Path
import datetime,hashlib,json,shutil
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro=parent/'code32k/repro32k';out=repro/'kv-streaming-source-candidate';assert not out.exists();out.mkdir()
def digest(p):
    with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def copy(p,rel):
    assert p.is_file() and p.stat().st_size<20*1024**2
    q=out/rel;assert not q.exists();q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
r=json.loads((base/'kv-stream-safe-v01402-build/record.json').read_text())
s=json.loads((base/'kv-stream-safe-v01402-source-candidate-v2/record.json').read_text())
assert r['passed'] and not r['active'] and not r['gpu_tested'] and not r['adopted'] and r['baseline_inputs_unchanged']
assert s['source_prepared'] and not s['gpu_tested'] and not s['adopted']
binary=Path(r['candidate_binary']);assert digest(binary)==r['candidate_binary_sha256']
candidate=Path(s['candidate_source']);original=Path(s['original_source'])
assert digest(candidate)==s['candidate_sha256']==r['candidate_source_sha256']
assert digest(original)==s['original_sha256']==r['original_source_sha256']
assert digest(root/'build-sycl-event-ack-no-root-prefill-20261008/strata')==r['baseline_binary_sha256']=='e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323'
assert digest(root/'build-sycl-e8ca-refresh-20261007/strata')=='c88f94d81bfb22227310ea00d09ecc8ab21a6e670aee9556307746530f5af714'
for directory,relative in [('kv-stream-safe-v01402-source-candidate','source-check/v1-negative'),('kv-stream-safe-v01402-source-candidate-v2','source-check/v2'),('kv-stream-safe-v01402-build','build')]:
    for p in sorted((base/directory).rglob('*')):
        if p.is_file():copy(p,relative+'/'+str(p.relative_to(base/directory)))
copy(original,'source/kv_stream.original.dp.cpp');copy(candidate,'source/kv_stream.candidate.dp.cpp')
copy(binary.parent/'kv_stream.dp.cpp.o.d','build/compiler-dependencies.d')
for n in ['prepare_kv_stream_safe_v01402_source.py','prepare_kv_stream_safe_v01402_source_v2.py','build_kv_stream_safe_v01402.py',Path(__file__).name]:copy(base/n,'controllers/'+n)
private=[]
for n in ['strata','kv_stream.dp.cpp.o','libstrata_kernels.a']:
    p=binary.parent/n;private.append({'file':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
(out/'README.md').write_text('''# Private KV-streaming source candidate: compiled, not GPU-tested

Existing upstream KV streaming (`--kv-resident 32768`) keeps authoritative
context K/V in host memory and a bounded subset in VRAM. It is an avenue to
investigate full262,144-cell capacity, not a new measured result. The existing
SYCL path is audited before activation; earlier accepted32K controls use
KV mode0 and do not validate this path.

The [source candidate](source/kv_stream.candidate.dp.cpp) changes only one
KV translation unit. A legal pair of sorted cell selections `[16,32]` and
`[4,16]` hits page4 from query0/lane0 and query1/lane1. The initial hit loop
contains no inter-query barrier and writes the same slot's stamp/reference
bits non-atomically. The candidate uses relaxed device-scope global atomic
stores for these shared hit writes. Its later work-group barrier and unique
victim writes remain unchanged. This is a source-level possible conflicting
access; it is not a claim this caused earlier non-streaming hangs.

Packed64-bit host counters are read from the int32_t copy through `memcpy`,
avoiding the previous type-punning/alignment assumptions. Four unused
root-sync properties are removed from reset/resolve/copy/ring launches.
Kernel arithmetic, ranges and required SG32/WG1024 resolve geometry remain
unchanged. This is not device capability or runtime correctness proof.

The first preparer expects three root properties, but its guard finds four
and stops before writing a candidate. Its CPU negative and immutable
controller are retained. The corrected preparer accounts for reset too and
produces the isolated source; no original source/binary changes.

The [private build](build/record.json) passes compilation/linking with the
accepted e82 compiler flags, changing only the KV object in a copied kernel
archive and reusing the accepted no-root prefill archive. Every other archive
member/link input and both baseline/production binaries remain unchanged.
Compilation takes5.974 s and linking19.698 s on Ryzen5600X; these are build
durations, not inference measurements. The candidate executable hash is
`4af18d831559e68215090d55ddba878986a07af61d834485351af81d793d6e77`.

No GPU run, performance result or production adoption is claimed. Host-USM
ownership, failure cleanup, ring restoration, staging/clipping and snapshot
identity reconstruction remain source gates before a first bounded logged
normal-MTP32K request at max-context262144/kv-resident32768. Its state/head,
64IDs/logprobs and MTP counts must match the accepted control. Any clean
comparison uses at least32,768 tokens with matching settings and separates
initial load/capture from later full rereads. Full262,144-cell occupancy,
repeat, restore, clipped-tail, refusal and later-valid gates remain mandatory.
No reset, rebind, reboot, service, package, runtime or global change occurs.
''')
p=repro/'README.md';p.write_text(p.read_text()+'''

The [private KV-streaming source candidate](kv-streaming-source-candidate/README.md)
fixes possible shared hit writes and host-counter type-punning before
exercising the existing RAM-KV route. It compiles/links but is not GPU-tested;
no speed, full-context or adoption result is claimed.
''')
for d in [out,repro,parent/'code32k',parent]:
    p=d/'manifest.json';m=json.loads(p.read_text()) if p.exists() else {};m.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_reason='Retain isolated compiled KV-stream source candidate with explicit pending source/GPU/full-context gates.',files={str(f.relative_to(d)):{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(d.rglob('*')) if f.is_file() and f!=p});p.write_text(json.dumps(m,indent=2)+'\n')
    for k,v in m['files'].items():
        f=d/k;assert f.stat().st_size==v['bytes'] and digest(f)==v['sha256']
    print(d.name,len(m['files']),flush=True)
