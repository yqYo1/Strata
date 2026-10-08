from pathlib import Path
import datetime,hashlib,json,shutil
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
out=parent/'code32k/repro32k/cnr32k';assert out.is_dir()
expected_partial={'runs/owned-cnr-v01402-code32k-diagnostic-r1/'+n for n in ['record.json','protocol.stdout.raw','events.jsonl','project-messages.txt','debugger/inferior-argv.json']}
assert {str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}==expected_partial
def copy(source,destination):
 target=out/destination;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
private=[]
for rep in ['diagnostic-r1','state-r1','state-r2']:
 name='owned-cnr-v01402-code32k-'+rep;directory=base/name
 r=json.loads((directory/'record.json').read_text());assert r['healthy'] and not r['active'] and not r['new_fault_messages']
 assert r['exit_code']==0 and not any(r['cleanup'][k] for k in ['inferior_survived','gdb_survived'])
 for n in ['record.json','protocol.stdout.raw','events.jsonl','project-messages.txt']:copy(directory/n,'runs/'+name+'/'+n)
 for n in ['inferior-argv.json','inferior-environment.json']:
  p=directory/'debugger'/n
  if p.exists():copy(p,'runs/'+name+'/debugger/'+n)
 for p in (directory/'probes').glob('*'):
  if p.is_file():copy(p,'runs/'+name+'/probes/'+p.name)
 for p in [directory/'debugger/inferior.stderr',directory/'first-head.bin',directory/'prefill-state.bin']:
  with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  private.append({'file':str(p),'bytes':p.stat().st_size,'sha256':digest})
for name in ['cnr-v01402-state-sequence','cnr-v01402-clean-sequence']:
 r=json.loads((base/name/'record.json').read_text());assert not r['passed'] and not r['active'];copy(base/name/'record.json','sequences/'+name+'/record.json')
assert json.loads((base/'cnr-v01402-clean-sequence/record.json').read_text())['steps']==[]
for name in ['run_owned_cnr_v01402_code32k.py','run_cnr_v01402_state_sequence.py','run_owned_cnr_v01402_code32k_clean.py','run_cnr_v01402_clean_sequence.py','archive_v01402_cnr32k.py','archive_v01402_cnr32k_v2.py']:copy(base/name,'controllers/'+name)
copy(base/'cb-off-v01402-state-analysis/cnr-research.json','research.json')
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
(out/'README.md').write_text('''# Documented oneMKL GPU CNR experiment at32K

The same production binary/input/configuration and process-local counter conversion
setting as the parent are retained. MKL_CBWR=AUTO is the only additional setting.
Intel documents GPU CNR for SYCL BLAS level3/GEMM and recommends AUTO for Intel
GPUs; it is also supported on the Ryzen host. This experiment tests that numerical
mode, without assuming BLAS variation causes the earlier discrepancy.

The logged first process and two unlogged state/head processes all complete32,768
input and64 finite-logprob output tokens, normal exit, complete owned cleanup and
no new xe fault. The first unlogged process matches all66prefill state parts,
all248,320 first-head float bytes, all64IDs and every protocol logprob of the
logged control. Their whole-head SHA256 is
053ccfe54cabe2528f8f44570684a8dd0258a2b0630006b2a86dfb61f777c586.

The second unlogged process differs in state parts1,3 and56..65, first-head bytes,
IDs and logprobs. The first differing saved QSA ordinal is10 (model layer43).
That is a later observed state discrepancy than in the preceding experiment,
without identifying its producing operation. CNR alone does not resolve output
reproducibility. CNR can change reduction choices, so equality with the older
non-CNR output is not used as a correctness requirement.

The failed state/head gate prevents either clean speed job from launching.
These jobs are functional/equality evidence only: state/head copies and diagnostic
logs exclude every rate here from performance comparisons. Three successful exits
and no xe fault do not establish model correctness or hang prevention. No production
setting, source/binary, package or service is changed. Full262,144-cell gates and
PP1000/TG70 remain open. The actual-DMA-event candidate is evaluated separately
under these same process-local settings; no candidate is adopted in this record.

Primary references: [GPU reproducibility conditions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/reproducibility-conditions.html),
[GPU CNR configuration](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/getting-started-with-conditional-numerical.html).
''')
note='''

The [oneMKL GPU CNR follow-up](cnr32k/README.md) also leaves the equality gate
open: two fresh32K captures agree in every state/head byte, but a third differs
in later state parts and output. All three exit normally without xe faults;
neither clean speed job is launched. CNR is a tested diagnostic condition, not
an adopted fix or explanation of all failures.
'''
with (parent/'code32k/repro32k/README.md').open('a') as stream:stream.write(note)
for directory in [out,parent/'code32k/repro32k',parent/'code32k',parent]:
 path=directory/'manifest.json';meta=json.loads(path.read_text()) if path.exists() else {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 meta.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_reason='Append completed32K CNR experiment and rejected equality gate; preserve earlier raw records.',files={str(p.relative_to(directory)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(directory.rglob('*')) if p.is_file() and p!=path})
 path.write_text(json.dumps(meta,indent=2)+'\n')
 for name,item in meta['files'].items():
  p=directory/name;assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
 print(directory.name,len(meta['files']))
