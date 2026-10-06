from pathlib import Path
import json,re,sys
p=Path('/proc')/sys.argv[1]
s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split()
assert int(f[19])==int(sys.argv[2])
io={k:int(v.strip()) for k,v in (line.split(':',1) for line in (p/'io').read_text().splitlines())}
gpu=[]
for q in (p/'fdinfo').iterdir():
 t=q.read_text()
 if re.search(r'^drm-pdev:\s*0000:05:00.0\s*$',t,re.M):
  gpu.append({'fd':q.name,'busy_cycles':{k:int(v) for k,v in re.findall(r'^drm-cycles-(\w+):\s*(\d+)',t,re.M)},'raw':t})
print(json.dumps({'pid':int(sys.argv[1]),'start_ticks':int(f[19]),'state':f[0],'minor_faults':int(f[7]),'major_faults':int(f[9]),'user_ticks':int(f[11]),'system_ticks':int(f[12]),'io':io,'gpu':gpu}))
