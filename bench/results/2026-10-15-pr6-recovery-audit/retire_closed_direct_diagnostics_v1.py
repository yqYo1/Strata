#!/usr/bin/env python3
"""Main serial retention for closed Direct/diagnostic-removal comparisons."""
import fcntl,json,re,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import B,P,load_owner,require,sha

def main():
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  names=['native-dense-bank-gpu32768-v1.json','native-dense-bank-direct1-v1.json','license-clean-prefill-model-v1.json']
  pairnames=['native-dense-bank-direct-comparison-v1.json','license-clean-prefill-model-comparison-v1.json']
  module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
  for name in pairnames:
   d=json.loads((A/name).read_text());require(d['passed'] and d['complete'] and d['sampled_FP32_values_equal'] and d['sampled_FP32_values_finite'],'completed exact comparison')
  target=A/'closed-direct-diagnostics-retention-v1.json';compact=A/'closed-direct-diagnostics-compact-v1.json'
  require(not target.exists() and not compact.exists(),'fresh retention')
  evidence=dict(passed=True,complete=True,GPU_executed=False,recipe_sha256=sha(Path(__file__)),references={n:sha(A/n) for n in names+pairnames},models=[])
  files={}
  def add(path,h):
   require(path.is_file() and not path.is_symlink() and sha(path)==h,'closed artifact identity '+str(path));files[str(path)]=dict(path=str(path),bytes=path.stat().st_size,sha256=h,reason='Closed exact768-row comparison; useful model/layer/fault evidence compacted. No remaining full text/old raw consumer; original results unchanged.')
  for i,name in enumerate(names):
   r=json.loads((A/name).read_text());require(r['passed'] and r['complete'] and not r['active'] and not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists'],'healthy closed model')
   for c in r['commands']:
    require(module.closed(c),'owned command closed')
    for old in c['owners']:
     now=module.identity(old['pid']);require(now is None or now['start_ticks']!=old['start_ticks'],'owner retired')
   native=Path(r['config']['log']);root=native.parents[1];text=native.read_text()
   completed=[list(map(int,m)) for m in re.findall(r'layer-major layer (\d+) of 48: completed (\d+) chunks',text)]
   require(completed==[[n,10] for n in range(48)],'all48layer/10chunk completions')
   evidence['models'].append(dict(receipt=name,completed_layer_chunks=completed,
    retained_native_lines=[s for s in text.splitlines() if s.startswith(('strata prefill dense:','strata mtp: prompt expert phase=','strata prefill: layer-major candidate,','strata serve: prompt '))],
    boundary_text_sha256=r['boundary']['text_sha256'],boundary_raw_sha256=r['boundary']['raw_sha256'],
    dense_lifecycle=r['dense_lifecycle'],fresh_faults=r['fresh_faults']))
   add(native,r['native_log_sha256']);add(Path(r['boundary']['text_path']),r['boundary']['text_sha256'])
   for c in r['commands']:
    for n,v in c['logs'].items():add(root/'owned'/n,v['sha256'])
   for v in r['request_journals']:add(root/v['path'],v['sha256'])
   duplicate=root/'record.json';require(json.loads(duplicate.read_text())==r,'duplicate final external record');add(duplicate,sha(duplicate))
   if i==1:add(Path(r['boundary']['raw_path']),r['boundary']['raw_sha256'])
   if i==2:
    retained=Path(r['boundary']['raw_path']);require(retained.stat().st_size==31457280 and sha(retained)==r['boundary']['raw_sha256'],'current FP32 original capture')
    evidence['retained_current_raw']=dict(path=str(retained),bytes=retained.stat().st_size,sha256=sha(retained),consumer='Main same-workload canonical dense bank v3 selected-row comparison; current clean-source FP32 Direct1/GPU32768 reference. Review after comparison closes or settings/input change.',byte_budget=32<<20)
  with compact.open('x') as f:json.dump(evidence,f,indent=2);f.write('\n')
  require(json.loads(compact.read_text())==evidence,'verified compact roundtrip')
  manifest=dict(passed=False,complete=False,GPU_executed=False,recipe_sha256=sha(Path(__file__)),replacement=compact.name,replacement_sha256=sha(compact),deleted=list(files.values()),original_result_status_changed=False)
  with target.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
  for item in manifest['deleted']:
   p=Path(item['path']);require(sha(p)==item['sha256'],'identity before unlink');p.unlink()
  manifest.update(passed=True,complete=True,removed_bytes=sum(x['bytes'] for x in manifest['deleted']),finished_utc=module.utc());target.write_text(json.dumps(manifest,indent=2)+'\n')
  print(json.dumps(dict(passed=True,removed_files=len(files),removed_bytes=manifest['removed_bytes'])) ,flush=True)
if __name__=='__main__':main()
