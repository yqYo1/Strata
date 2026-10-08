from pathlib import Path
import datetime, hashlib, json, shutil, sys

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro=parent/'code32k/repro32k';out=repro/'main-cache-release-and-restore'
sys.path.insert(0,str(root/'sycl/tools'));from owned_gdb import process_identity
def digest(p):
    with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def load(name):return json.loads((base/name/'record.json').read_text())
cpu=load('main-cache-lease-v01402-host-check');patterned=load('main-cache-lease-v01402-host-patterned-v2')
assert cpu['passed'] and patterned['passed'] and not cpu['active'] and not patterned['active']
assert len(cpu['cases'])==len(patterned['cases'])==19 and patterned['wrong_offset_control']['expected_failure_detected']
modes=[m for m in ['main-vmm-kept-ram','main-vmm-half-ram','main-vmm-full-ram','main-vmm-full-snapshot'] if (base/(m+'-v01402-state-sequence')/'record.json').exists()]
assert modes[:2]==['main-vmm-kept-ram','main-vmm-half-ram']
for mode in modes:
    r=load(mode+'-v01402-state-sequence');assert r['passed'] and not r['active'] and len(r['steps'])==3
baseline=load('owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1')['requests'][0]
counts=baseline['protocol'][-1].split()[6:8]
for mode in modes:
    for phase,rep in [('diagnostic',1),('state',1),('state',2)]:
        name=f'owned-{mode}-v01402-code32k-{phase}-r{rep}';r=load(name)
        assert r['healthy'] and r['completed'] and not r['active'] and r['exit_code']==0 and not r['new_fault_messages']
        assert not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
        for key in ['inferior','debugger']:
            old=r[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
        req=r['requests'][0];c=req['comparison_to_default_counter_control']
        assert req['measurement']['prompt_tokens']==32768 and req['measurement']['generated_tokens']==64
        assert c['first_head_equal'] and c['ids_equal'] and c['logprobs_equal'] and not c['different_prefill_state_parts']
        assert req['protocol'][-1].split()[6:8]==counts
assert digest(root/'build-sycl-e8ca-refresh-20261007/strata')=='c88f94d81bfb22227310ea00d09ecc8ab21a6e670aee9556307746530f5af714'
assert not out.exists();out.mkdir()
def copy(p,dst):
    assert p.is_file() and p.stat().st_size<20*1024**2,p
    q=out/dst;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for name in ['main-cache-lease-v01402-host-check','main-cache-lease-v01402-host-patterned-v2']:
    for p in (base/name).rglob('*'):
        if p.is_file() and p.name not in ['probe','core'] and p.suffix in ['.json','.inc','.cpp','.stdout','.stderr']:
            copy(p,'host/'+name+'/'+str(p.relative_to(base/name)))
copy(base/'main-cache-release-v01402-source-review/record.json','analysis/source-review.json')
controllers=['test_main_cache_lease_v01402_host.py','test_main_cache_lease_v01402_host_patterned_v2.py','prepare_main_cache_release_v01402_code32k.py','run_owned_main_cache_release_v01402_code32k.py','run_main_cache_release_v01402_state_sequence.py','prepare_main_cache_release_v01402_clean.py','run_owned_main_cache_release_v01402_code32k_clean.py','run_main_cache_release_v01402_clean_sequence.py',Path(__file__).name]
for name in controllers:copy(base/name,'controllers/'+name)
private=[];jobs=[]
for mode in modes:
    gate=load(mode+'-v01402-state-sequence');copy(base/(mode+'-v01402-state-sequence')/'record.json','analysis/'+mode+'-state-sequence.json')
    for phase,rep in [('diagnostic',1),('state',1),('state',2)]:
        name=f'owned-{mode}-v01402-code32k-{phase}-r{rep}';d=base/name;r=load(name)
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
        jobs.append({'name':name,'prompt_tokens':32768,'ids_and_logprobs_equal':True,'all66_main_state_parts_equal':True,'all248320_first_head_floats_equal':True,'mtp_counts':counts,'exit_code':0,'no_new_xe_fault':True})
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
(out/'completed-state-controls.json').write_text(json.dumps({'scope':'Only completed three-run arms at this checkpoint; captured timings are excluded from speed evidence. MTP counts are not all internal MTP tensor bytes.','modes':modes,'jobs':jobs},indent=2)+'\n')
messages=[]
for mode in modes:
    for s in load(mode+'-v01402-state-sequence')['steps']:
        messages.append({'mode':mode,'phase':s['argv'][3],'repetition':s['argv'][4],**{k:s[k] for k in ['main_cache','main_verify','mtp_release','mtp_restore']}})
(out/'captured-release-messages.json').write_text(json.dumps(messages,indent=2)+'\n')
(out/'README.md').write_text('''# Main-cache release and restoration controls

Arc B570 10 GiB / Ryzen 5 5600X / 128 GiB RAM, kernel 7.0.0-38,
NEO 26.31.39395.14 and oneAPI 2026.1.1. This checkpoint retains completed
correctness controls; clean timing remains pending. No speed claim or
production adoption follows from diagnostic/state-capture durations.

The same private scheduling binary
e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323
reads the same 32,768-token code-review fixture, with 8192-token chunks,
int8 KV and normal MTP4 generating 64 greedy tokens. Compact storage 2,
attention layout 1 / batch 32 / subgroup 32, layer-major 2, 32,767 GPU
residual rows, in-place scratch reuse and MTP decode-only release are common.
All arms request 128 main-cache experts and use 64 MiB physical segments.
The release fractions are 0, 0.5 and 1; full release additionally compares
immutable-RAM restoration and a GPU-to-RAM snapshot. No implicit-counter
override or MKL CNR is set. Exact argv and environment remain with each run.

The [source review](analysis/source-review.json) checks ordered queue drains,
current residency, tail bounds, complete immutable-RAM sources, partial
failure rollback, stable virtual addresses and graph retirement before
unmapping. The main decoder/verifier cache pointers are refreshed only after
bytes are restored and checked. Verifier recapture records nodes rather than
executing a model window. Temporary layer-cache/residual buffers are reclaimed
before restoring decode weights. The explicit release experiment selects
segmented allocation without the prohibited interactive --vram-elastic flag.
Startup INFO's vram_elastic field reports whether the cache is segmented;
it is not evidence that the interactive resize command was enabled.

CPU tests extract the actual compiled private CacheLease struct. The initial
19 ASan/UBSan cases cover mixed profile order, a segment boundary inside an
expert, partial/full/control RAM or snapshot, adaptive residency, invalid
fractions/sources/spans, payload mismatch, partial-shrink rollback and
grow/refresh retries. A second set uses a distinct pattern at every byte;
all 19 pass. Its pinned negative control removes the intra-expert RAM offset
and is rejected by the actual pre-unmap payload check. These CPU resource
stand-ins do not prove SYCL mapping, GPU arithmetic or absence of runtime UB.

Completed three-run arms at this checkpoint: '''+', '.join(modes)+'''.
Each has a first logged/validated 32K request and two fresh state/head captures.
All completed requests match every one of the 66 main-prefill state parts,
all 248,320 first-head floats (993,280 bytes), all 64 IDs and every logprob
with the logged default. MTP acceptance/offered counts match 43/66; this is
not a comparison of every internal MTP tensor. Owned processes exit normally,
complete cleanup and record no new xe fault.

The [captured messages](captured-release-messages.json) distinguish physical,
logical-tail and occupied payload bytes. Half release unmaps 201,326,592 B,
with 139,460,608 logical tail bytes and 130,731,008 occupied expert bytes;
the retained/released boundary may split a slot. Both sides of restoration
check the entire occupied payload. Full-RAM/snapshot and the clean matched
kept/half-RAM/full-RAM/full-snapshot/reverse comparison are pending unless
their completed controls appear above. No captured duration enters a speed
comparison. Graph recapture overhead will be included in clean prefill time.

Full 262,144-cell occupancy/repeat/restore/clipped-tail/refusal/later-valid
gates remain open. A full 262,143-row residual image alone would require
10,737,377,280 B at the observed 40,960 B/row, before KV/dense/cache/scratch.
The fixed 32,767-row setting is not a full-context default. Dynamic prefix
budgeting and the older second-prefill allocation/rollback failure remain
unresolved. PP 1000 / TG 70 has not been achieved.

Source hashes, executed controllers, protocol, environment, journal and
ownership receipts are public. Large API/state/head payloads remain private
with exact bytes and SHA256. Production code/executable, main branch,
services, packages and global settings are unchanged; no reset/rebind/reboot.
''')
with (repro/'README.md').open('a') as s:s.write('''

The [main-cache release/restoration controls](main-cache-release-and-restore/README.md)
add actual-code ASan/UBSan offset/rollback tests with a patterned negative
control and completed 32K kept/partial/full lease checks. Clean timing and
full-context serving remain separate gates; see the checkpoint's completed list.
''')
for directory in [out,repro,parent/'code32k',parent]:
    p=directory/'manifest.json';r=json.loads(p.read_text()) if p.exists() else {}
    r.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_reason='Append CPU lease checks and completed32K main-cache release controls; no speed/full-context claim.',files={str(f.relative_to(directory)):{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(directory.rglob('*')) if f.is_file() and f!=p});p.write_text(json.dumps(r,indent=2)+'\n')
    for name,v in r['files'].items():
        f=directory/name;assert f.stat().st_size==v['bytes'] and digest(f)==v['sha256']
    print(directory.name,len(r['files']))
