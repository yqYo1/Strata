from pathlib import Path
import datetime,hashlib,json,shutil,sys

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro=parent/'code32k/repro32k'
out=repro/'registered-copy-default-counter-conversion'
seq=json.loads((base/'event-ack-no-root-prefill-default-v01402-state-sequence/record.json').read_text())
gate=json.loads((base/'event-ack-no-root-default-v01402-gated-checks/record.json').read_text())
assert seq['passed'] and gate['passed'] and not seq['active'] and not gate['active']
assert not out.exists();out.mkdir()
sys.path.insert(0,str(root/'sycl/tools'))
from owned_gdb import process_identity
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def copy(p,dst):
 assert p.is_file() and p.stat().st_size<20*1024**2
 q=out/dst;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
private=[]
for phase,rep in [('diagnostic',1),('state',1),('state',2)]:
 name=f'owned-event-ack-no-root-prefill-default-v01402-code32k-{phase}-r{rep}'
 d=base/name;r=json.loads((d/'record.json').read_text())
 assert r['healthy'] and r['completed'] and not r['active'] and r['exit_code']==0
 assert not r['new_fault_messages'] and not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
 for key in ['inferior','debugger']:
  old=r[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
 assert 'EnableImplicitConvertionToCounterBasedEvents' not in r['environment'] and 'MKL_CBWR' not in r['environment']
 assert len(r['requests'])==1 and r['requests'][0]['measurement']['prompt_tokens']==32768
 c=r['requests'][0]['comparison_to_counter_off_control']
 assert not c['different_prefill_state_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal'])
 for f in ['record.json','protocol.stdout.raw','events.jsonl','project-messages.txt','input-tokens.txt']:
  p=d/f
  if p.exists():copy(p,'runs/'+name+'/'+f)
 for p in (d/'probes').glob('*'):
  if p.is_file():copy(p,'runs/'+name+'/probes/'+p.name)
 for f in ['inferior-argv.json','inferior-environment.json']:
  p=d/'debugger'/f
  if p.exists():copy(p,'runs/'+name+'/debugger/'+f)
 for f in ['debugger/inferior.stderr','first-head.bin','prefill-state.bin']:
  p=d/f;private.append({'file':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
for name in ['event-ack-no-root-prefill-default-v01402-state-sequence','event-ack-no-root-default-v01402-gated-checks']:
 copy(base/name/'record.json','sequences/'+name+'.json')
for name in ['prepare_event_ack_no_root_default_v01402.py','run_owned_event_ack_no_root_default_v01402_code32k.py','run_event_ack_no_root_default_v01402_state_sequence.py','run_event_ack_no_root_default_v01402_gated_checks.py',Path(__file__).name]:
 copy(base/name,'controllers/'+name)
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
(out/'README.md').write_text('''# Registered copy: default counter conversion, three exact32K checks

On the same Arc B57010GiB / Ryzen5 5600X /128GiB boot, kernel7.0.0-38,
NEO26.31.39395.14 and oneAPI2026.1.1, the private scheduling candidate
e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323
is unchanged from the [completed no-CNR controls and clean comparison](../registered-copy-no-cnr-and-prefill-scheduling/README.md).
Only EnableImplicitConvertionToCounterBasedEvents=0 is removed.
MKL_CBWR remains absent; queue properties, arithmetic, subgroups, ranges,
barriers, drains and graphs remain unchanged. Other common process-local
settings, including disabled direct submission/persistent cache/copy offload,
remain recorded. This tests default counter conversion, not every environment
variable at its default.

The first logged/validated32K request and two fresh unlogged state/head
repetitions all match all66prefill state parts, all248,320first-head float
bytes, all64generated IDs and every protocol logprob with the prior counter-off
control and one another. Every process reads32,768 input tokens, with context
33024,8192-token chunks, int8 KV, normal MTP4, cache128, five CPU workers,
pcie0, FIRST0/RING8 and no prompt/conversation caching. All three exit normally,
complete owned cleanup and record no new xe fault. The [state sequence](sequences/event-ack-no-root-prefill-default-v01402-state-sequence.json)
passes both fresh comparisons.

These executions show that forced-off implicit conversion and MKL CNR are
unnecessary for the tested32K candidate controls. They do not prove general
hang prevention, full262,144-cell serving/repeats/restore or the PP1000/TG70
target. State/head dumping is enabled in all three; the first also has UR/
Level Zero API logs and parameter validation. No timing from these jobs is
used for a speed claim or substituted into the earlier clean comparison.
The candidate remains private and unadopted while full-context gates are open.

Exact environment/argv, fixture hashes, raw protocol, project messages, journal
and process ownership are public. Large API/state/head payloads remain private
with byte counts and SHA256 hashes in private-artifacts.json. No reset, rebind,
reboot, package, service or global environment setting is changed.
''')
with (repro/'README.md').open('a') as stream:
 stream.write('''

The [default counter-conversion check](registered-copy-default-counter-conversion/README.md)
then passes another three exact32K full state/head/output controls on the same
private scheduling binary, with both implicit-conversion override and MKL CNR
absent. All exit normally without new xe faults. These captured runs are not
clean timing evidence; full262,144-cell serving remains unvalidated.
''')
for directory in [out,repro,parent/'code32k',parent]:
 p=directory/'manifest.json';r=json.loads(p.read_text()) if p.exists() else {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 r.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_reason='Append three exact32K default-counter controls without CNR.',files={str(f.relative_to(directory)):{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(directory.rglob('*')) if f.is_file() and f!=p})
 p.write_text(json.dumps(r,indent=2)+'\n')
 for name,item in r['files'].items():
  f=directory/name;assert f.stat().st_size==item['bytes'] and digest(f)==item['sha256']
 print(directory.name,len(r['files']))
