"""Archive validated 32K state gates and clean ABBA timings without large payloads."""
from pathlib import Path
import datetime,hashlib,json,shutil,sys

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro=parent/'code32k/repro32k'
out=repro/'registered-copy-no-cnr-and-prefill-scheduling'
clean_path=base/'registered-vs-no-root-v01402-clean-sequence/record.json'
clean=json.loads(clean_path.read_text())
assert not clean['active'] and clean['passed'] and len(clean['steps'])==4
assert not out.exists()
out.mkdir()
sys.path.insert(0,str(root/'sycl/tools'))
from owned_gdb import process_identity

def digest(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def copy(source,destination):
 target=out/destination;target.parent.mkdir(parents=True,exist_ok=True)
 assert source.is_file() and source.stat().st_size<20*1024**2
 shutil.copy2(source,target)
def payload(path):
 return {'file':str(path),'bytes':path.stat().st_size,'sha256':digest(path)}

private=[]
modes=['event-ack-registered-copy-cb','event-ack-no-root-prefill-cb']
for mode in modes:
 sequence=base/f'{mode}-v01402-state-sequence/record.json'
 seq=json.loads(sequence.read_text());assert not seq['active'] and seq['passed']
 copy(sequence,f'sequences/{mode}.json')
 for phase,rep in [('diagnostic',1),('state',1),('state',2),('clean',1),('clean',2)]:
  name=f'owned-{mode}-v01402-code32k-{phase}-r{rep}'
  directory=base/name;r=json.loads((directory/'record.json').read_text())
  assert not r['active'] and r['healthy'] and r['completed'] and r['exit_code']==0
  assert not r['new_fault_messages'] and not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
  for key in ['inferior','debugger']:
   old=r[key];now=process_identity(old['pid'])
   assert not now or now['start_ticks']!=old['start_ticks']
  assert len(r['requests'])==1 and r['requests'][0]['measurement']['prompt_tokens']==32768
  request=r['requests'][0]
  if phase=='clean':
   assert all(request['comparison_to_logged_control'].values())
  else:
   key='comparison_to_production_cb_off_control' if mode==modes[0] else 'comparison_to_registered_no_cnr_control'
   c=request[key];assert not c['different_prefill_state_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal'])
  for filename in ['record.json','protocol.stdout.raw','events.jsonl','project-messages.txt','input-tokens.txt']:
   p=directory/filename
   if p.exists():copy(p,'runs/'+name+'/'+filename)
  for filename in ['inferior-argv.json','inferior-environment.json']:
   p=directory/'debugger'/filename
   if p.exists():copy(p,'runs/'+name+'/debugger/'+filename)
  for p in (directory/'probes').glob('*'):
   if p.is_file():copy(p,'runs/'+name+'/probes/'+p.name)
  for filename in ['debugger/inferior.stderr','first-head.bin','prefill-state.bin']:
   p=directory/filename
   if p.exists():private.append(payload(p))
copy(clean_path,'sequences/clean-abba.json')
build=base/'event-ack-no-root-prefill-v01402-build'
r=json.loads((build/'record.json').read_text());assert r['passed'] and not r['active'] and r['baseline_inputs_unchanged']
for p in build.iterdir():
 if not p.is_file():continue
 if p.stat().st_size>1048576:private.append(payload(p))
 else:copy(p,'build/'+p.name)
candidate=Path(r['candidate_binary'])
assert digest(candidate)==r['candidate_binary_sha256']
private.append(payload(candidate))
source=candidate.parent/'source/kernels.dp.cpp'
assert digest(source)==r['candidate_source_sha256']
copy(source,'private-source/kernels.dp.cpp')
for filename in ['prepare_event_ack_registered_copy_cb_v01402.py','run_owned_event_ack_registered_copy_cb_v01402_code32k.py','run_event_ack_registered_copy_cb_v01402_state_sequence.py','build_event_ack_no_root_prefill_v01402.py','prepare_event_ack_no_root_prefill_cb_v01402.py','prepare_event_ack_no_root_prefill_cb_v01402_v2.py','run_owned_event_ack_no_root_prefill_cb_v01402_code32k.py','run_event_ack_no_root_prefill_cb_v01402_state_sequence.py','prepare_registered_vs_no_root_v01402_clean.py','run_owned_event_ack_registered_copy_cb_v01402_code32k_clean.py','run_owned_event_ack_no_root_prefill_cb_v01402_code32k_clean.py','run_registered_vs_no_root_v01402_clean_sequence.py',Path(__file__).name]:
 copy(base/filename,'controllers/'+filename)
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
means=clean['clean_mean'];a=means[modes[0]];b=means[modes[1]]
table='\n'.join(f"| {label} | 2 | {means[mode]['prefill_tok_s']:.2f} | {means[mode]['decode_tok_s']:.2f} |" for label,mode in [('Registered copy, original prefill launch properties',modes[0]),('Registered copy, without14 prefill root-sync properties',modes[1])])
(out/'README.md').write_text(f'''# Registered nonprofiling copy: no CNR and32K prefill scheduling comparison

Hardware: Arc B57010GiB, Ryzen5 5600X,128GiB RAM, kernel7.0.0-38,
NEO26.31.39395.14 and oneAPI2026.1.1, on the same boot as the earlier controls.
The [registered-copy candidate](../event-ack-registered-copy-cb-cnr/README.md)
is unchanged at SHA256 f02213fc440a1f11b64857fceaa028ab43eeee7fdfb40e5cec33abca411a9705.
Removing only MKL_CBWR=AUTO permits three fresh32K executions to match all66
prefill state parts, all248,320first-head float bytes, all64IDs and every
protocol logprob with the production counter-conversion-off control and each
other. This demonstrates that CNR is unnecessary for these tested controls.
EnableImplicitConvertionToCounterBasedEvents=0 remains set in every run here.

A scheduling-only candidate at SHA256 {r['candidate_binary_sha256']} removes
exactly14 use_root_sync properties in the prefill kernel translation unit.
Every kernel body, arithmetic operation, ND-range, subgroup attribute, queue,
barrier, explicit release drain and graph operation remains unchanged.
The [build receipt](build/record.json) verifies all other archive members and
link inputs remain identical and the matched registered-copy build is unchanged.
All compilation uses the same matched header definition in every translation unit.
The first logged32K check and two fresh unlogged state/head repeats also match
all66state parts, full head and complete64-token output with the registered-copy
no-CNR control. All six state/head jobs finish normally without new xe faults.

The subsequent [clean ABBA sequence](sequences/clean-abba.json) uses registered /
no-root / no-root / registered, each as a fresh process. Every request has32,768
input tokens and64greedy output tokens, normal MTP4, context33024,8192-token
chunks, int8 KV, cache128, five CPU workers, pcie0, FIRST0 and RING8.
Prompt/conversation caching is off; no short-input timing is used. Timing comes
from engine DONE intervals, excludes model startup and includes cold request
JIT/graph preparation. No API tracing, validation, state/head dumps, transfer
profiling, profiler or additional waits are enabled. An uninterrupted owned
GDB/PTY observer is common to both arms. Each clean run requires all64IDs and
every protocol logprob equal the logged control, normal exit, complete owned
cleanup and no new xe fault; all four pass.

| Private configuration | Clean repetitions | Prompt tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
{table}

These are means of two runs per arm. Relative no-root change is
{100*(b['prefill_tok_s']/a['prefill_tok_s']-1):+.2f}% for prompt and
{100*(b['decode_tok_s']/a['decode_tok_s']-1):+.2f}% for decode; two runs do not
establish statistical significance. The no-root decode runs are17.11 and15.01
tok/s, versus17.03 and16.90 in the registered control. The second no-root run
has slower decode despite identical output and a changed prefill-only launch
property; these data do not justify a decode regression or gain attribution.
The prompt difference is below the registered control's within-arm spread,
so this experiment does not establish a useful prompt speed improvement.
Only the scheduling property differs in
this comparison. The earlier unmodified-upstream393.65/17.14 rates are a
separate completed sequence with different resident cache packing and timing;
they are not contemporaneous paired controls for these two private candidates.

The [root-group extension specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_root_group.asciidoc)
requires use_root_sync for cross-work-group root-group functions and limits
the permitted work-group count. No root-group calls or cross-work-group atomic
coordination occur in this prefill translation unit. Its independent groups
use local barriers and partition recurrence state by head/column. The installed
2026.1 kernel_launch_helper.hpp also marshals the property. This supports removal
as a scheduling experiment; it does not prove the property caused an earlier
stall or that an installed backend takes a particular cooperative launch path.
Core persistent/global coordination kernel properties are retained.

Both candidates remain private and unadopted. These completed32K jobs do not
prove general prevention of GPU/native-event waits, default counter-event
conversion, full262,144-cell serving/repeats/restore, or the PP1000/TG70 goal.
Diagnostic and state/head job timings are excluded from the speed table.

The first scheduling-controller preparer fails before writing a controller or
launching a GPU process: its string replacement also changed a CNR control
prefix. A separate v2 preparer targets exact mode and configuration names and
retains the earlier failure source. Executed controllers are never overwritten.
Full multi-GiB API logs, state/head captures and the private binary remain outside
Git, with hashes and byte counts in private-artifacts.json. Public records retain
the exact environment/argv, fixture, protocol, project progress, journal probes,
build provenance and process ownership. No reset, rebind, reboot, service,
package or system setting change is performed.
''')
with (repro/'README.md').open('a') as stream:
 stream.write('''

The [no-CNR and prefill scheduling follow-up](registered-copy-no-cnr-and-prefill-scheduling/README.md)
adds six complete exact32K state/head checks and a clean32K ABBA comparison.
The registered-copy candidate is reproducible in these checks without MKL CNR;
the prefill scheduling candidate changes only14 root-sync properties. Both
remain private while default-backend and full262,144-cell serving gates are open.
''')
for directory in [out,repro,parent/'code32k',parent]:
 path=directory/'manifest.json'
 meta=json.loads(path.read_text()) if path.exists() else {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 meta.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_reason='Append six exact32K no-CNR/scheduling state controls and four clean32K ABBA jobs.',files={str(p.relative_to(directory)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(directory.rglob('*')) if p.is_file() and p!=path})
 path.write_text(json.dumps(meta,indent=2)+'\n')
 for name,item in meta['files'].items():
  p=directory/name;assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
 print(directory.name,len(meta['files']))
