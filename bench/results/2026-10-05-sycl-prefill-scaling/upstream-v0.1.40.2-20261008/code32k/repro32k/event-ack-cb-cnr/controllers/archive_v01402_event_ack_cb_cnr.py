from pathlib import Path
import datetime,hashlib,json,shutil
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
out=parent/'code32k/repro32k/event-ack-cb-cnr';assert not out.exists();out.mkdir()
def copy(source,destination):
 target=out/destination;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
private=[]
for phase in ['diagnostic','state']:
 name=f'owned-event-ack-cb-cnr-v01402-code32k-{phase}-r1';directory=base/name
 r=json.loads((directory/'record.json').read_text());assert not r['active'] and not r['new_fault_messages']
 assert not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
 if phase=='diagnostic':
  assert r['healthy'] and r['exit_code']==0
  c=r['requests'][0]['comparison_to_production_cnr_control'];assert not c['different_prefill_state_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal'])
 else:assert not r['healthy'] and not r['completed']
 for n in ['record.json','protocol.stdout.raw','events.jsonl','project-messages.txt']:
  p=directory/n
  if p.exists():copy(p,'runs/'+name+'/'+n)
 for n in ['inferior-argv.json','inferior-environment.json','failure.mi.txt','failure.console.txt']:
  p=directory/'debugger'/n
  if p.exists():copy(p,'runs/'+name+'/debugger/'+n)
 for p in (directory/'probes').glob('*'):
  if p.is_file():copy(p,'runs/'+name+'/probes/'+p.name)
 for p in [directory/'debugger/inferior.stderr',directory/'first-head.bin',directory/'prefill-state.bin']:
  if not p.exists():continue
  with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  private.append({'file':str(p),'bytes':p.stat().st_size,'sha256':digest})
 if phase=='state':
  p=directory/'debugger/inferior.stderr'
  with p.open('rb') as stream:stream.seek(max(0,p.stat().st_size-32768));tail=stream.read()
  (out/'runs'/name/'debugger/inferior.stderr.tail').write_bytes(tail)
health='post-event-ack-cb-cnr-v01402-stall-health';r=json.loads((base/health/'record.json').read_text());assert r['healthy'] and not r['active']
copy(base/health/'record.json','health/record.json')
for p in (base/health/'probes').glob('*'):
 if p.is_file():copy(p,'health/probes/'+p.name)
sequence='event-ack-cb-cnr-v01402-state-sequence';r=json.loads((base/sequence/'record.json').read_text());assert not r['active'] and not r['passed'];copy(base/sequence/'record.json','sequence.json')
for n in ['run_owned_event_ack_cb_cnr_v01402_code32k.py','run_event_ack_cb_cnr_v01402_state_sequence.py','check_post_event_ack_cb_cnr_v01402_stall_health.py','archive_v01402_event_ack_cb_cnr.py']:copy(base/n,'controllers/'+n)
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
(out/'README.md').write_text('''# Actual-DMA completion candidate with the32K CNR settings

This uses the same private binary4b805390b3f732d21607f846826ee54c2f36eb650b2ddc611d087626938e12bf
as the earlier event-status-query candidate. Only the process-local implicit
counter conversion setting and documented MKL_CBWR=AUTO are added, as in the
five completed production controls. Model input/configuration remain32,768
tokens,8192-token chunks, normal MTP4, int8 KV and explicit128cache slots.

The logged first process completes64finite-logprob outputs and normal exit,
and matches the production CNR logged control in all66prefill state parts,
all248,320 first-head float bytes, all64IDs and every protocol logprob. No new
xe fault is recorded. This is functional equality evidence, not a timing job.

The next unlogged state/head process stops in its first chunk, before a full
state or head is captured. The main thread waits in Stager::wait. Three workers
query the actual DMA event in queryCounterBasedEventStatus ->
synchronizeTimestampCompletionWithTimeout -> assignKernelEventCompletionData.
Thus disabling implicit conversion does not remove every observed counter-event
completion path. Neither this setting nor CNR is proved to prevent the stall.
The kernel journal records no new xe fault.

The owned inferior/debugger are both terminated and verified absent before a
fresh logged H2D/kernel/D2H health check passes on the same boot. No reset,
rebind, reboot, package or service change is performed. The sequence stops
before its second unlogged repeat. The candidate remains unadopted, and no
timing is accepted. The first logged equality pass cannot establish reliability
of the unlogged completion path or full262,144-cell capacity.

The observed stack motivates a separate private candidate that omits copy-queue
profiling unless transfer profiling is explicitly requested, while retaining
compute-queue properties, arithmetic, in-order dependencies and explicit drains.
That candidate is not GPU-validated by this record.
''')
with (parent/'code32k/repro32k/README.md').open('a') as stream:stream.write('''

The [actual-DMA candidate with CNR](event-ack-cb-cnr/README.md) matches the
production control's complete state/head in its logged32K run, then stops in
native timestamp status queries in the next unlogged run. It remains unadopted;
owned cleanup and same-boot GPU health pass without a manual recovery action.
''')
for directory in [out,parent/'code32k/repro32k',parent/'code32k',parent]:
 path=directory/'manifest.json';meta=json.loads(path.read_text()) if path.exists() else {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 meta.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_reason='Append rejected32K actual-DMA candidate with CNR and same-boot health; preserve earlier raw records.',files={str(p.relative_to(directory)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(directory.rglob('*')) if p.is_file() and p!=path})
 path.write_text(json.dumps(meta,indent=2)+'\n')
 for name,item in meta['files'].items():
  p=directory/name;assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
 print(directory.name,len(meta['files']))
