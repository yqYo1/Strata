import json,os,re,time
from pathlib import Path
parent=Path('/proc/507615');exe=b'/tmp/strata-upstream-arc-compact-diag-jit'
out=Path('/tmp/strata-upstream-arc-compact-confirm/memory-samples.jsonl')
summary_path=out.with_name('memory-summary.json');peaks={}
def memory(d):
 result={}
 status=(d/'status').read_text()
 for field in ['VmRSS','VmHWM','VmSize','VmSwap']:
  m=re.search(r'^'+field+r':\s*(\d+) kB$',status,re.M)
  if m:result[field+'_bytes']=int(m[1])*1024
 clients={}
 for fd in (d/'fd').iterdir():
  try:
   if '/dev/dri/' not in os.readlink(fd):continue
   info=(d/'fdinfo'/fd.name).read_text()
   identity=re.search(r'^drm-client-id:\s*(\d+)$',info,re.M)
   if not identity:continue
   if identity[1] in clients:continue
   clients[identity[1]]={}
   for key,value,unit in re.findall(r'^(drm-(?:total|resident|shared|active|purgeable)-[\w]+):\s*(\d+)(?:\s*(KiB|MiB|GiB))?$',info,re.M):
    clients[identity[1]][key+'_bytes']=int(value)*{'':1,'KiB':1024,'MiB':1024**2,'GiB':1024**3}[unit]
  except OSError:pass
 for values in clients.values():
  for key,value in values.items():result[key]=result.get(key,0)+value
 return result
with out.open('w') as handle:
 while parent.exists():
  for d in Path('/proc').iterdir():
   if not d.name.isdigit():continue
   try:
    cmd=(d/'cmdline').read_bytes().split(b'\0')
    if not cmd or cmd[0]!=exe or b'--tokens-file' not in cmd:continue
    name=Path(os.fsdecode(cmd[cmd.index(b'--tokens-file')+1])).stem
    values=memory(d);item={'time_ns':time.time_ns(),'pid':int(d.name),'run':name,'bytes':values}
    handle.write(json.dumps(item)+'\n');handle.flush()
    record=peaks.setdefault(name,{'samples':0,'sampled_peak_bytes':{}});record['samples']+=1
    for key,value in values.items():record['sampled_peak_bytes'][key]=max(value,record['sampled_peak_bytes'].get(key,0))
   except (OSError,ValueError):pass
  summary_path.write_text(json.dumps({'interval_seconds':1,'note':'Sampled /proc status and DRM client fdinfo, deduplicated by client ID. GPU bytes refer to this process, not total card usage; a peak can occur between samples. Monitoring started during baseline 64K: earlier baseline lengths and MTP checks are not covered.','runs':peaks},indent=2)+'\n')
  time.sleep(1)
print('memory watch complete',len(peaks),'runs',flush=True)
